
import discord, os, json, logging, re
from discord.ext import commands
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
DISCORD_TOKEN    = os.getenv("DISCORD_TOKEN")
TRAINING_CHANNEL = os.getenv("TRAINING_CHANNEL", "bot-training")
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL", "bot-log")
MEMORY_FILE      = "memory.json"

logging.basicConfig(level=logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from transformers.utils import logging as hf_log
hf_log.set_verbosity_error()

intents = discord.Intents.default()
intents.message_content = True
intents.presences = True
bot = commands.Bot(command_prefix="/", intents=intents)

if not os.path.exists(MEMORY_FILE):
    json.dump([], open(MEMORY_FILE, "w"))

def load_mem(): return json.load(open(MEMORY_FILE))
def save_mem(m): json.dump(m, open(MEMORY_FILE, "w"), indent=2)

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def best(query:str, mem:List[Dict], top_k:int=3):
    q = model.encode(query, convert_to_tensor=True)
    scored=[]
    for it in mem:
        txt = it.get("question") if it["type"]=="qa" else it["content"]
        emb = model.encode(txt, convert_to_tensor=True)
        scored.append((util.cos_sim(q, emb).item(), it))
    scored.sort(reverse=True)
    return scored[:top_k]

@bot.event
async def on_ready():
    await bot.change_presence(status=discord.Status.online)
    logging.info("Ready as %s", bot.user)

@bot.event
async def on_message(msg: discord.Message):
    if msg.author.bot: return
    mem=load_mem()
    if msg.channel.name==TRAINING_CHANNEL:
        txt=msg.content.strip()
        if txt.lower().startswith("fact:"):
            body=txt[5:].strip()
            # split on bullets/newlines/period-space
            raw_parts = re.split(r"[\n\r•\-]+|\.\s+", body)
            added=0
            for part in raw_parts:
                s=part.strip()
                if len(s)>15:
                    mem.append({"type":"fact","content":s})
                    added+=1
            save_mem(mem)
            await msg.add_reaction("✅")
            await msg.reply(f"Saved {added} FACT chunk(s).", delete_after=3)
            return
        if txt.startswith("Q:") and "\n" in txt:
            q,a=txt.split("\n",1)
            if a.startswith("A:"):
                mem.append({"type":"qa","question":q[2:].strip(),"answer":a[2:].strip()})
                save_mem(mem); await msg.add_reaction("✅")
            return

    # normal channels
    MIN_SIM=0.45
    matches=best(msg.content, mem)
    if not matches or matches[0][0]<MIN_SIM: return
    top_sim, top=matches[0]
    if top["type"]=="qa":
        await msg.channel.send(top["answer"]); return
    context="\n".join(f"• {m[1]['content']}" for m in matches if m[1]["type"]=="fact")
    reply=f"📌 Based on what I know:\n{context}"
    if len(reply)>2000: reply=reply[:1990]+"\n... (truncated)"
    await msg.channel.send(reply)

@bot.command()
async def check(ctx,*,q): 
    mem=load_mem(); m=best(q,mem,1)
    if not m: await ctx.reply("no match"); return
    await ctx.reply(f"{m[0][0]:.2f} – {m[0][1]}")

if __name__=="__main__":
    bot.run(DISCORD_TOKEN)
