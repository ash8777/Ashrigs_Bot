import discord
from discord.ext import commands
import os
import json
import asyncio
import openai
from dotenv import load_dotenv

# Load secrets from environment variables (Railway uses these)
load_dotenv()

DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
TRAINING_CHANNEL= os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL     = os.getenv("LOG_CHANNEL", "bot-logs")
MODEL_NAME      = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

MEMORY_FILE = "memory.json"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Ensure memory file exists
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

def load_memory():
    with open(MEMORY_FILE) as f:
        return json.load(f)

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

async def fetch_openai_response(prompt: str) -> str:
    openai.api_key = OPENAI_API_KEY
    response = await openai.ChatCompletion.acreate(
        model=MODEL_NAME,
        messages=[{"role":"system","content":"You are a helpful FiveM support assistant."},
                  {"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()

def best_match(question: str, memory):
    # Very naive similarity: exact match; improve later
    for pair in memory:
        if question.lower() == pair["question"].lower():
            return pair["answer"]
    return None

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # Training channel logic
    if message.channel.name == TRAINING_CHANNEL:
        if message.content.startswith("Q:"):
            lines = message.content.splitlines()
            if len(lines) >= 2 and lines[1].startswith("A:"):
                q = lines[0][2:].strip()
                a = lines[1][2:].strip()
                memory = load_memory()
                memory.append({"question": q, "answer": a})
                save_memory(memory)
                await message.add_reaction("✅")
        return  # do not auto‑reply in training channel

    # Auto‑reply in every other text channel
    memory = load_memory()
    match = best_match(message.content, memory)

    if match:
        await message.channel.send(match)
    else:
        # fallback to OpenAI
        prompt = f"Customer says: {message.content}\nProvide a concise helpful reply for FiveM asset support."
        try:
            reply = await fetch_openai_response(prompt)
            await message.channel.send(reply)
        except Exception as e:
            print("OpenAI error", e)

    await bot.process_commands(message)

# Manual /train command (alternative to training channel)
@bot.command()
async def train(ctx, *, text):
    """/train Question || Answer"""
    if "||" not in text:
        await ctx.send("Format should be: /train question || answer")
        return
    q,a = [part.strip() for part in text.split("||",1)]
    memory = load_memory()
    memory.append({"question":q,"answer":a})
    save_memory(memory)
    await ctx.send("Training example added ✅")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
