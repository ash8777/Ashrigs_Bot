# main.py – QA + FACT knowledge support bot

import discord
from discord.ext import commands
import os
import json
import asyncio
import openai
from dotenv import load_dotenv
from typing import List, Dict

# ---------------------------------------------------------------------------
# ENV SETTINGS (Railway / .env)
# ---------------------------------------------------------------------------
load_dotenv()

DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL      = os.getenv("LOG_CHANNEL", "bot-logs")
MODEL_NAME       = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

MEMORY_FILE = "memory.json"

# ---------------------------------------------------------------------------
# DISCORD SETUP
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# ---------------------------------------------------------------------------
# MEMORY HELPERS
# ---------------------------------------------------------------------------
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def load_memory() -> List[Dict]:
    with open(MEMORY_FILE, encoding="utf-8") as f:
        return json.load(f)

def save_memory(data: List[Dict]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# OPENAI HELPER (using 0.28.x client)
# ---------------------------------------------------------------------------
openai.api_key = OPENAI_API_KEY
async def fetch_openai_response(prompt: str) -> str:
    """Async wrapper around OpenAI ChatCompletion (0.28.x)"""
    response = await openai.ChatCompletion.acreate(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful FiveM support assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ---------------------------------------------------------------------------
# SIMPLE RETRIEVAL (exact QA + keyword FACT)
# ---------------------------------------------------------------------------

def exact_qa_match(query: str, memory: List[Dict]):
    for entry in memory:
        if entry.get("type", "qa") == "qa" and query.lower() == entry["question"].lower():
            return entry["answer"]
    return None

def gather_relevant_facts(query: str, memory: List[Dict], limit: int = 5):
    words = set(w.lower() for w in query.split())
    hits = []
    for entry in memory:
        if entry.get("type") == "fact":
            text = entry["content"].lower()
            if any(w in text for w in words):
                hits.append(entry["content"])
    return hits[:limit]

# ---------------------------------------------------------------------------
# EVENTS
# ---------------------------------------------------------------------------
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    memory = load_memory()

    # ---------------- Training channel ----------------
    if message.channel.name == TRAINING_CHANNEL:
        content = message.content.strip()

        # FACT: ...
        if content.lower().startswith("fact:"):
            fact = content[5:].strip()
            memory.append({"type": "fact", "content": fact})
            save_memory(memory)
            await message.add_reaction("✅")
            return

        # Q: ...\nA: ...
        if content.startswith("Q:") and "\n" in content:
            q_line, a_line = content.split("\n", 1)
            if a_line.startswith("A:"):
                memory.append({
                    "type": "qa",
                    "question": q_line[2:].strip(),
                    "answer": a_line[2:].strip()
                })
                save_memory(memory)
                await message.add_reaction("✅")
        return  # never auto‑reply in training channel

    # ---------------- Normal channels ----------------
    # 1) exact QA answer
    match = exact_qa_match(message.content, memory)
    if match:
        await message.channel.send(match)
        return

    # 2) build context from facts + optionally GPT
    facts = gather_relevant_facts(message.content, memory)
    context = "\n".join(f"• {f}" for f in facts) if facts else "(no saved facts matched)"

    prompt = (
        "Known facts about troubleshooting FiveM assets/maps:\n" + context + "\n\n" +
        f"Customer says: {message.content}\n" +
        "Provide a concise, helpful reply that references the facts when relevant."
    )

    try:
        reply = await fetch_openai_response(prompt)
        await message.channel.send(reply)
    except Exception as e:
        print("OpenAI error", e)

    await bot.process_commands(message)

# ---------------------------------------------------------------------------
# COMMAND: /train  (legacy)
# ---------------------------------------------------------------------------
@bot.command()
async def train(ctx, *, text):
    """/train question || answer  – legacy quick add"""
    if "||" not in text:
        await ctx.send("Format: /train question || answer")
        return
    q, a = [part.strip() for part in text.split("||", 1)]
    memory = load_memory()
    memory.append({"type": "qa", "question": q, "answer": a})
    save_memory(memory)
    await ctx.send("Training example added ✅")

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("DEBUG ▸ DISCORD_TOKEN length:", len(DISCORD_TOKEN) if DISCORD_TOKEN else "None")
    bot.run(DISCORD_TOKEN)
