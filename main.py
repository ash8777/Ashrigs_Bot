
import discord
from discord.ext import commands
import os, json, logging
from dotenv import load_dotenv
from typing import List, Dict

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------
load_dotenv()
DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MEMORY_FILE      = "memory.json"

# ---------------------------------------------------------------------------
# Logging – silence noisy libs
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.presences = True   # show online status
bot = commands.Bot(command_prefix="/", intents=intents)

# memory helpers ------------------------------------------------------------
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

def load_memory() -> List[Dict]:
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error("memory.json corrupted – resetting (%s)", e)
        return []

def save_memory(data: List[Dict]):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# similarity search ---------------------------------------------------------
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def best_facts(query: str, memory: List[Dict], top_k: int = 3):
    q_emb = model.encode(query, convert_to_tensor=True)
    scored = []
    for item in memory:
        text = item.get("question") if item.get("type") == "qa" else item.get("content")
        if not text: continue
        item_emb = model.encode(text, convert_to_tensor=True)
        scored.append((util.cos_sim(q_emb, item_emb).item(), item))
    scored.sort(reverse=True)
    return scored[:top_k]

# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    await bot.change_presence(status=discord.Status.online, activity=None)
    logging.info("Logged in as %s (%s)", bot.user, bot.user.id)

# ---------------------------------------------------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    memory = load_memory()
    log_ch = discord.utils.get(message.guild.text_channels, name=LOG_CHANNEL_NAME)

    # TRAINING CHANNEL
    if message.channel.name == TRAINING_CHANNEL:
        content = message.content.strip()
        if content.lower().startswith("fact:"):
            fact = content[5:].strip()
            memory.append({"type": "fact", "content": fact})
            save_memory(memory)
            await message.add_reaction("✅")
        elif content.startswith("Q:") and "\n" in content:
            q_line, a_line = content.split("\n", 1)
            if a_line.startswith("A:"):
                memory.append({"type": "qa", "question": q_line[2:].strip(), "answer": a_line[2:].strip()})
                save_memory(memory)
                await message.add_reaction("✅")
        return

    # RETRIEVE
    MIN_SIM = 0.45
    matches = best_facts(message.content, memory)
    if not matches or matches[0][0] < MIN_SIM:
        if log_ch:
            await log_ch.send(f"🔍 No match for: {message.content[:150]}...")
        return

    top_sim, top_item = matches[0]
    if top_item["type"] == "qa":
        await message.channel.send(top_item["answer"])
        return

    # build context from facts
    context = "\n".join(f"• {m[1]['content']}" for m in matches if m[1]["type"] == "fact")
    reply_text = f"📌 Based on what I know:\n{context}"
    if len(reply_text) > 2000:
        reply_text = reply_text[:1990] + "\n... (truncated)"
    await message.channel.send(reply_text)

    await bot.process_commands(message)

# ---------------------------------------------------------------------------
@bot.command()
@commands.has_permissions(administrator=True)
async def train(ctx, *, text):
    """/train question || answer"""
    if "||" not in text:
        return await ctx.send("Usage: /train question || answer")
    q, a = [p.strip() for p in text.split("||", 1)]
    memory = load_memory()
    memory.append({"type": "qa", "question": q, "answer": a})
    save_memory(memory)
    await ctx.send("Training added ✅")

@bot.command()
@commands.has_permissions(administrator=True)
async def check(ctx, *, query):
    """DMs the best match and similarity for debugging"""
    memory = load_memory()
    match = best_facts(query, memory, top_k=1)
    if match:
        sim, item = match[0]
        await ctx.author.send(f"Top sim: {sim:.2f}\nItem: {item}")
    else:
        await ctx.author.send("No match at all.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
