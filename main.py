import discord
from discord.ext import commands
import os
import json
import asyncio
import openai
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MODEL_NAME       = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MEMORY_FILE      = "memory.json"

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

def load_memory():
    with open(MEMORY_FILE) as f:
        return json.load(f)

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def best_facts(query: str, memory: List[Dict], top_k: int = 3):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(query, convert_to_tensor=True)

    results = []
    for item in memory:
        content = item.get("question") if item.get("type") == "qa" else item.get("content")
        if not content: continue
        item_emb = model.encode(content, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, item_emb).item()
        results.append((sim, item))
    results.sort(reverse=True)
    return [item for _, item in results[:top_k]]

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    memory = load_memory()
    log_channel = discord.utils.get(message.guild.text_channels, name=LOG_CHANNEL_NAME)

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
                memory.append({
                    "type": "qa",
                    "question": q_line[2:].strip(),
                    "answer": a_line[2:].strip()
                })
                save_memory(memory)
                await message.add_reaction("✅")
        return

    facts = best_facts(message.content, memory)
    if facts and facts[0].get("type") == "qa":
        await message.channel.send(facts[0]["answer"])
        return
    elif facts and any(f["type"] == "fact" for f in facts):
        context = "\n".join(
            f"• {f['content'] if f['type']=='fact' else f['answer']}"
            for f in facts
        )
        reply_text = f"📌 I found this based on what I know:\n{context}"
        if len(reply_text) > 2000:
            reply_text = reply_text[:1990] + "\n... (truncated)"
        await message.channel.send(reply_text)
    else:
        print("No match found — skipping reply.")

    await bot.process_commands(message)

@bot.command()
async def train(ctx, *, text):
    if "||" not in text:
        await ctx.send("Format should be: /train question || answer")
        return
    q, a = [part.strip() for part in text.split("||", 1)]
    memory = load_memory()
    memory.append({"type": "qa", "question": q, "answer": a})
    save_memory(memory)
    await ctx.send("Training example added ✅")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
