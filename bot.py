import discord
from discord.ext import commands
import os

intents = discord.Intents.default()
intents.message_content = True  # This is CRITICAL

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'✅ {bot.user} is online and ready!')

@bot.event
async def on_message(message):
    # Log every message received to Render logs (so we can see if intents work)
    print(f"📩 Received: {message.content} from {message.author}")
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('pong')

# Run the bot using the token from environment variable
bot.run(os.getenv('DISCORD_TOKEN'))