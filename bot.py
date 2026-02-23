import discord
from discord.ext import commands
import os

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Bot is ready! Logged in as {bot.user}')

@bot.event
async def on_message(message):
    print(f'Received message: {message.content} from {message.author}')
    if message.content.startswith('!'):
        await message.channel.send('Echo: ' + message.content)
    await bot.process_commands(message)

@bot.command()
async def ping(ctx):
    await ctx.send('pong')

bot.run(os.getenv('DISCORD_TOKEN'))