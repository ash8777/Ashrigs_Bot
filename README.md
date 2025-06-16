# Discord AI Support Bot

A simple Discord bot that auto‑replies to FiveM MLO support tickets using OpenAI **and** learns from examples you post in a dedicated training channel.

## Quick Deploy (Railway)

1. Fork / download this repo  
2. Click the **Deploy on Railway** button or create a new project from GitHub  
3. Add the following variables in Railway → `Variables`:
   - `DISCORD_TOKEN`
   - `OPENAI_API_KEY`
   - (optional) `TRAINING_CHANNEL`, `LOG_CHANNEL`, `OPENAI_MODEL`

## Training

Create a text channel (default `#bot-training`) and post:

```
Q: my map is not loading
A: Please clear cache and restart FiveM.
```

The bot reacts ✅ and stores it in `memory.json`.  
Future matching questions will get the stored answer instantly.

## Commands

- `/train question || answer` – Add training example via command.

Enjoy! 🎉
