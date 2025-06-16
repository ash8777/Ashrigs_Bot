import os, json, re, logging, asyncio, torch, discord
from discord.ext import commands
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

# ── ENV ──────────────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MEMORY_FILE      = "memory.json"

# ── LOGGING ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_log
hf_log.set_verbosity_error()

# ── DISCORD BOT ─────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.presences       = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ── MODEL (CPU) ─────────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ── MEMORY WITH CACHED EMBEDDINGS ──────────────────────────────────
def _embed(text: str):
    return model.encode(text, convert_to_tensor=True).cpu().tolist()

def _ensure_mem_file():
    if not os.path.exists(MEMORY_FILE):
        json.dump([], open(MEMORY_FILE, "w"))

def load_memory() -> List[Dict]:
    _ensure_mem_file()
    data = json.load(open(MEMORY_FILE))
    changed = False
    for item in data:
        if "emb" not in item:
            text = item.get("question") if item["type"] == "qa" else item["content"]
            item["emb"] = _embed(text)
            changed = True
    if changed:
        json.dump(data, open(MEMORY_FILE, "w"), indent=2)
    return data

memory = load_memory()

def add_entry(entry: Dict):
    text = entry.get("question") if entry["type"] == "qa" else entry["content"]
    entry["emb"] = _embed(text)
    memory.append(entry)
    json.dump(memory, open(MEMORY_FILE, "w"), indent=2)

async def best(query: str, k: int = 3):
    q = await asyncio.to_thread(model.encode, query, convert_to_tensor=True)
    scored = [(util.cos_sim(q, torch.tensor(m["emb"])).item(), m) for m in memory]
    scored.sort(reverse=True)
    return scored[:k]

# ── EVENTS ──────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    await bot.change_presence(status=discord.Status.online)
    logging.info("Logged in as %s", bot.user)

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    # ─ Training channel handling ─
    if msg.channel.name == TRAINING_CHANNEL:
        txt = msg.content.strip()

        # FACT block → auto-chunk
        if txt.lower().startswith("fact:"):
            body   = txt[5:].strip()
            parts  = re.split(r"[\\n\\r•\\-]+|\\.\\s+", body)
            chunks = [p.strip() for p in parts if len(p.strip()) > 15]
            for c in chunks:
                add_entry({"type": "fact", "content": c})
            await msg.add_reaction("✅")
            await msg.reply(f"Saved {len(chunks)} chunk(s).", delete_after=3)
            return

        # Q/A pair
        if txt.startswith("Q:") and "\\n" in txt:
            q, a = txt.split("\\n", 1)
            if a.startswith("A:"):
                add_entry({"type": "qa",
                           "question": q[2:].strip(),
                           "answer":   a[2:].strip()})
                await msg.add_reaction("✅")
            return

    # ─ Retrieval & reply ─
    matches = await best(msg.content, 3)
    if not matches:
        return

    top_sim, t_
