import discord, os, json, logging, re, asyncio
from discord.ext import commands
from dotenv import load_dotenv
from typing import List, Dict

# ── ENV ─────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MEMORY_FILE      = "memory.json"

# ── LOGGING ─────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_log
hf_log.set_verbosity_error()

# ── BOT ─────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.presences       = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ── MEMORY ──────────────────────────────────────────
if not os.path.exists(MEMORY_FILE):
    json.dump([], open(MEMORY_FILE, "w"))

def load_mem() -> List[Dict]:
    return json.load(open(MEMORY_FILE))

def save_mem(m: List[Dict]):
    json.dump(m, open(MEMORY_FILE, "w"), indent=2)

# ── EMBEDDINGS (run in worker thread) ───────────────
from sentence_transformers import SentenceTransformer, util
_model = SentenceTransformer("all-MiniLM-L6-v2")

async def embed(text: str):
    """Run model.encode off-loop to avoid blocking heartbeats."""
    return await asyncio.to_thread(_model.encode, text, convert_to_tensor=True)

async def best(query: str, mem: List[Dict], k: int = 3):
    q_emb = await embed(query)
    scored = []
    for item in mem:
        text = item["question"] if item["type"] == "qa" else item["content"]
        emb  = await embed(text)
        scored.append((util.cos_sim(q_emb, emb).item(), item))
    scored.sort(reverse=True)
    return scored[:k]

# ── EVENTS ──────────────────────────────────────────
@bot.event
async def on_ready():
    await bot.change_presence(status=discord.Status.online)
    logging.info("Logged in as %s", bot.user)

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot:
        return

    mem    = load_mem()
    log_ch = discord.utils.get(msg.guild.text_channels, name=LOG_CHANNEL_NAME)

    # ─ training channel ─
    if msg.channel.name == TRAINING_CHANNEL:
        txt = msg.content.strip()
        if txt.lower().startswith("fact:"):
            body   = txt[5:].strip()
            parts  = re.split(r"[\\n\\r•\\-]+|\\.\\s+", body)
            chunks = [p.strip() for p in parts if len(p.strip()) > 15]
            for c in chunks:
                mem.append({"type": "fact", "content": c})
            save_mem(mem)
            await msg.add_reaction("✅")
            await msg.reply(f"Saved {len(chunks)} FACT chunk(s).", delete_after=3)
            return

        if txt.startswith("Q:") and "\\n" in txt:
            q, a = txt.split("\\n", 1)
            if a.startswith("A:"):
                mem.append({"type": "qa",
                            "question": q[2:].strip(),
                            "answer":   a[2:].strip()})
                save_mem(mem)
                await msg.add_reaction("✅")
            return

    # ─ retrieval ─
    matches = await best(msg.content, mem)
    if not matches:
        return

    top_sim, top_item = matches[0]
    if log_ch:
        await log_ch.send(f"⚖️ Similarity {top_sim:.2f} for “{msg.content[:120]}…”")

    if top_sim < 0.30:            # threshold
        return

    if top_item["type"] == "qa":
        await msg.channel.send(top_item["answer"])
        return

    context = "\\n".join(f"• {m[1]['content']}" for m in matches if m[1]["type"] == "fact")
    reply   = f"📌 Based on what I know:\\n{context}"
    if len(reply) > 2000:
        reply = reply[:1990] + "\\n... (truncated)"
    await msg.channel.send(reply)

    await bot.process_commands(msg)

# ─ commands ─────────────────────────────────────────
@bot.command()
@commands.has_permissions(administrator=True)
async def check(ctx, *, query):
    mem = load_mem()
    m   = await best(query, mem, 1)
    if not m:
        return await ctx.reply("No match.")
    await ctx.reply(f"{m[0][0]:.2f} – {m[0][1]}")

# ─ run ──────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
