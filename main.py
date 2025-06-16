import os, json, re, logging, asyncio, torch
import discord
from discord.ext import commands
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict

# ── ENV ─────────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MEMORY_FILE      = "memory.json"

# ── LOGGING ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_log
hf_log.set_verbosity_error()

# ── DISCORD BOT ────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.presences       = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ── MODEL (CPU) ────────────────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# ── MEMORY WITH CACHED EMBEDDINGS ─────────────────────────────
def _embed(text: str):
    """Return list[float] cpu embedding."""
    tensor = model.encode(text, convert_to_tensor=True)
    return tensor.cpu().tolist()

def _ensure_memory_file():
    if not os.path.exists(MEMORY_FILE):
        json.dump([], open(MEMORY_FILE, "w"))

def load_memory() -> List[Dict]:
    _ensure_memory_file()
    data = json.load(open(MEMORY_FILE))
    changed = False
    for item in data:
        if "emb" not in item:           # first time run on old entry
            txt = item.get("question") if item["type"] == "qa" else item["content"]
            item["emb"] = _embed(txt)
            changed = True
    if changed:
        json.dump(data, open(MEMORY_FILE, "w"), indent=2)
    return data

memory = load_memory()

def add_entry(entry: Dict):
    txt = entry.get("question") if entry["type"] == "qa" else entry["content"]
    entry["emb"] = _embed(txt)
    memory.append(entry)
    json.dump(memory, open(MEMORY_FILE, "w"), indent=2)

# ── SIMILARITY USING CACHED EMBEDDINGS ────────────────────────
async def best_matches(query: str, k: int = 3):
    q_emb = await asyncio.to_thread(model.encode, query, convert_to_tensor=True)
    sims  = [(util.cos_sim(q_emb, torch.tensor(it["emb"])).item(), it) for it in memory]
    sims.sort(reverse=True)
    return sims[:k]

# ── EVENTS ────────────────────────────────────────────────────
@bot.event
async def on_ready():
    await bot.change_presence(status=discord.Status.online)
    logging.info("Logged in as %s", bot.user)

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    # ─ Training channel ─
    if msg.channel.name == TRAINING_CHANNEL:
        txt = msg.content.strip()
        # FACT paste (auto-chunk)
        if txt.lower().startswith("fact:"):
            body  = txt[5:].strip()
            parts = re.split(r"[\\n\\r•\\-]+|\\.\\s+", body)
            chunks = [p.strip() for p in parts if len(p.strip()) > 15]
            for c in chunks:
                add_entry({"type": "fact", "content": c})
            await msg.add_reaction("✅")
            await msg.reply(f"Saved {len(chunks)} FACT chunk(s).", delete_after=3)
            return
        # Classic Q/A pair
        if txt.startswith("Q:") and "\\n" in txt:
            q, a = txt.split("\\n", 1)
            if a.startswith("A:"):
                add_entry({"type": "qa",
                           "question": q[2:].strip(),
                           "answer":   a[2:].strip()})
                await msg.add_reaction("✅")
            return

    # ─ Retrieval / auto-reply ─
    top = await best_matches(msg.content, 3)
    if not top:
        return

    top_sim, top_item = top[0]
    log_ch = discord.utils.get(msg.guild.text_channels, name=LOG_CHANNEL_NAME)
    if log_ch:
        await log_ch.send(f"⚖️ Similarity {top_sim:.2f} for “{msg.content[:120]}…”")

    if top_sim < 0.30:           # threshold
        return

    # direct answer
    if top_item["type"] == "qa":
        await msg.channel.send(top_item["answer"])
        return

    # build context from FACTs
    context = "\\n".join(f"• {m[1]['content']}" for m in top if m[1]["type"] == "fact")
    reply   = f"📌 Based on what I know:\\n{context}"
    if len(reply) > 2000:
        reply = reply[:1990] + "\\n... (truncated)"
    await msg.channel.send(reply)

    await bot.process_commands(msg)

# ─ Commands ───────────────────────────────────────────────────
@bot.command()
@commands.has_permissions(administrator=True)
async def check(ctx, *, query):
    """DM the best match & similarity (debug)"""
    m = await best_matches(query, 1)
    if not m:
        return await ctx.reply("No match.")
    await ctx.author.send(f"{m[0][0]:.2f} – {m[0][1]}")

# ─ Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
