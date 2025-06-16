import discord, os, json, logging, re
from discord.ext import commands
from dotenv import load_dotenv
from typing import List, Dict

# ── ENV ────────────────────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MEMORY_FILE      = "memory.json"

# ── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_log
hf_log.set_verbosity_error()

# ── BOT SETUP ──────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
intents.presences = True          # shows a green status-dot
bot = commands.Bot(command_prefix="/", intents=intents)

# ── MEMORY HELPERS ────────────────────────────────────────────────────────
if not os.path.exists(MEMORY_FILE):
    json.dump([], open(MEMORY_FILE, "w"))

def load_mem() -> List[Dict]:
    return json.load(open(MEMORY_FILE))

def save_mem(data: List[Dict]):
    json.dump(data, open(MEMORY_FILE, "w"), indent=2)

# ── SIMILARITY  (Sentence-Transformers) ───────────────────────────────────
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

def best(query: str, mem: List[Dict], k: int = 3):
    q_emb = model.encode(query, convert_to_tensor=True)
    scored = []
    for item in mem:
        text = item["question"] if item["type"] == "qa" else item["content"]
        emb  = model.encode(text, convert_to_tensor=True)
        scored.append((util.cos_sim(q_emb, emb).item(), item))
    scored.sort(reverse=True)
    return scored[:k]

# ── EVENTS ────────────────────────────────────────────────────────────────
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

    # ─ Training channel ───────────────────────────────────────────────────
    if msg.channel.name == TRAINING_CHANNEL:
        txt = msg.content.strip()
        if txt.lower().startswith("fact:"):
            body   = txt[5:].strip()
            parts  = re.split(r"[\\n\\r•\\-]+|\\.\\s+", body)
            added  = 0
            for p in parts:
                s = p.strip()
                if len(s) > 15:
                    mem.append({"type": "fact", "content": s})
                    added += 1
            save_mem(mem)
            await msg.add_reaction("✅")
            await msg.reply(f"Saved {added} FACT chunk(s).", delete_after=3)
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

    # ─ Retrieval / auto-reply ─────────────────────────────────────────────
    MIN_SIM = 0.30                                 # lowered threshold
    matches = best(msg.content, mem)

    if not matches:
        return

    top_sim, top_item = matches[0]
    if log_ch:
        await log_ch.send(f"⚖️ Similarity {top_sim:.2f} for “{msg.content[:120]}…”")

    if top_sim < MIN_SIM:
        return

    # direct Q-A
    if top_item["type"] == "qa":
        await msg.channel.send(top_item["answer"])
        return

    # build reply from FACTs
    context = "\\n".join(f"• {m[1]['content']}" for m in matches if m[1]["type"] == "fact")
    reply   = f"📌 Based on what I know:\\n{context}"
    if len(reply) > 2000:
        reply = reply[:1990] + "\\n... (truncated)"
    await msg.channel.send(reply)

    await bot.process_commands(msg)

# ─ Commands ───────────────────────────────────────────────────────────────
@bot.command()
@commands.has_permissions(administrator=True)
async def check(ctx, *, query):
    """DM the best match & score (debug)."""
    mem = load_mem()
    m   = best(query, mem, 1)
    if not m:
        return await ctx.reply("No match.")
    await ctx.author.send(f"{m[0][0]:.2f} – {m[0][1]}")

@bot.command()
@commands.has_permissions(administrator=True)
async def train(ctx, *, text):
    """/train question || answer"""
    if "||" not in text:
        return await ctx.send("Usage: /train question || answer")
    q, a = [p.strip() for p in text.split("||", 1)]
    mem  = load_mem()
    mem.append({"type": "qa", "question": q, "answer": a})
    save_mem(mem)
    await ctx.send("Training added ✅")

# ─ Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
