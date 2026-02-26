import discord
from discord.ext import commands
import aiohttp
import pandas as pd
import numpy as np
import ta
import asyncio
import json
import os
from datetime import datetime, timedelta

# === CONFIGURATION ===
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Watchlist file
WATCHLIST_FILE = 'watchlist.json'

# Simple cooldown to prevent duplicate command execution
last_command_time = {}

# ====================
# HELPER FUNCTIONS
# ====================

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    default = {
        "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG"],
        "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD"]
    }
    save_watchlist(default)
    return default

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=2)

def normalize_symbol(symbol):
    """Convert common crypto symbols to BASE/USD format."""
    symbol = symbol.upper()
    crypto_map = {
        'BTC': 'BTC/USD', 'ETH': 'ETH/USD', 'SOL': 'SOL/USD',
        'XRP': 'XRP/USD', 'DOGE': 'DOGE/USD', 'PEPE': 'PEPE/USD',
        'ADA': 'ADA/USD', 'DOT': 'DOT/USD', 'LINK': 'LINK/USD'
    }
    # If it's a known crypto shorthand, convert
    if symbol in crypto_map:
        return crypto_map[symbol]
    # If it already contains '/', assume it's correct
    if '/' in symbol:
        return symbol
    # Otherwise, treat as stock
    return symbol

async def fetch_ohlcv(symbol, timeframe):
    interval_map = {
        'daily': '1day',
        'weekly': '1week',
        '4h': '4h'
    }
    interval = interval_map.get(timeframe)
    if not interval:
        return None

    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': interval,
        'apikey': TWELVEDATA_API_KEY,
        'outputsize': 200,
        'format': 'JSON'
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if 'values' not in data:
                    print(f"Twelve Data error for {symbol}: {data}")
                    return None
                df = pd.DataFrame(data['values'])
                df = df.rename(columns={'datetime': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                df = df.sort_index()
                return df
    except Exception as e:
        print(f"Error fetching {symbol} from Twelve Data: {e}")
        return None

def calculate_indicators(df):
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema13'] = ta.trend.ema_indicator(df['close'], window=13)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=3)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_basis'] = bb.bollinger_mavg()

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    return df

def get_signals(df):
    if len(df) < 2:
        return {}
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {}
    signals['price'] = latest['close']
    signals['ema5'] = latest['ema5']
    signals['ema13'] = latest['ema13']
    signals['ema50'] = latest['ema50']
    signals['ema200'] = latest['ema200']

    signals['ema5_above_13'] = latest['ema5'] > latest['ema13']
    signals['ema13_above_50'] = latest['ema13'] > latest['ema50']
    signals['ema5_cross_above_13'] = prev['ema5'] <= prev['ema13'] and latest['ema5'] > latest['ema13']
    signals['ema5_cross_below_13'] = prev['ema5'] >= prev['ema13'] and latest['ema5'] < latest['ema13']
    signals['ema13_cross_above_50'] = prev['ema13'] <= prev['ema50'] and latest['ema13'] > latest['ema50']
    signals['ema13_cross_below_50'] = prev['ema13'] >= prev['ema50'] and latest['ema13'] < latest['ema50']

    signals['touch_upper_bb'] = latest['high'] >= latest['bb_upper']
    signals['touch_lower_bb'] = latest['low'] <= latest['bb_lower']

    signals['rsi'] = latest['rsi']
    signals['rsi_overbought'] = latest['rsi'] >= 75
    signals['rsi_oversold'] = latest['rsi'] <= 25

    signals['buy_signal'] = signals['ema5_cross_above_13'] and latest['rsi'] >= 50
    signals['sell_signal'] = signals['ema5_cross_below_13'] and latest['rsi'] <= 50

    bullish = [
        signals['ema5_cross_above_13'],
        signals['ema13_cross_above_50'],
        signals['buy_signal'],
        signals['touch_lower_bb'] and signals['rsi_oversold'],
        signals['rsi_oversold']
    ]
    bearish = [
        signals['ema5_cross_below_13'],
        signals['ema13_cross_below_50'],
        signals['sell_signal'],
        signals['touch_upper_bb'] and signals['rsi_overbought'],
        signals['rsi_overbought']
    ]
    signals['bullish_count'] = sum(bullish)
    signals['bearish_count'] = sum(bearish)
    signals['net_score'] = signals['bullish_count'] - signals['bearish_count']

    return signals

def format_symbol_result(symbol, signals, timeframe):
    if not signals:
        return f"`{symbol}`: No data or error."

    sym_type = "Crypto" if '/' in symbol else "Stock"

    if signals['net_score'] >= 2:
        emoji = "🟢🟢"
    elif signals['net_score'] == 1:
        emoji = "🟢"
    elif signals['net_score'] == 0:
        emoji = "🟡"
    elif signals['net_score'] == -1:
        emoji = "🔴"
    else:
        emoji = "🔴🔴"

    lines = [
        f"{emoji} **{symbol}** ({sym_type}) – {timeframe}",
        f"Price: ${signals['price']:.2f}  | RSI: {signals['rsi']:.1f}",
        f"EMAs: 5={signals['ema5']:.2f}, 13={signals['ema13']:.2f}, 50={signals['ema50']:.2f}, 200={signals['ema200']:.2f}"
    ]

    active = []
    if signals['ema5_cross_above_13']: active.append("5↑13")
    if signals['ema13_cross_above_50']: active.append("13↑50")
    if signals['buy_signal']: active.append("📈 Buy")
    if signals['touch_lower_bb'] and signals['rsi_oversold']: active.append("🔻 Oversold Triangle")
    if signals['rsi_oversold']: active.append("OS X")

    if signals['ema5_cross_below_13']: active.append("5↓13")
    if signals['ema13_cross_below_50']: active.append("13↓50")
    if signals['sell_signal']: active.append("📉 Sell")
    if signals['touch_upper_bb'] and signals['rsi_overbought']: active.append("🔺 Overbought Triangle")
    if signals['rsi_overbought']: active.append("OB X")

    if active:
        lines.append("Signals: " + ", ".join(active))
    else:
        lines.append("No active signals.")

    return "\n".join(lines)

# ====================
# DISCORD EVENTS
# ====================

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == bot.user:
        return
    print(f"Received message: {message.content} from {message.author}")
    await bot.process_commands(message)

# ====================
# DISCORD COMMANDS
# ====================

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('pong')

@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    # Simple cooldown to prevent duplicate execution (5 seconds)
    now = datetime.now()
    last = last_command_time.get(ctx.author.id)
    if last and (now - last) < timedelta(seconds=5):
        return  # Ignore if user spams
    last_command_time[ctx.author.id] = now

    timeframe = timeframe.lower()
    if timeframe not in ['daily', 'weekly', '4h']:
        await ctx.send("Invalid timeframe. Use daily, weekly, or 4h.")
        return

    watchlist = load_watchlist()
    symbols = watchlist['stocks'] + watchlist['crypto']

    if target.lower() != 'all':
        # Normalize symbol (convert crypto shorthand)
        symbol = normalize_symbol(target)
        await ctx.send(f"Scanning **{symbol}** ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}.")
            return
        df = calculate_indicators(df)
        signals = get_signals(df)
        result = format_symbol_result(symbol, signals, timeframe)
        await ctx.send(result)
        return

    # Scan all symbols
    await ctx.send(f"Scanning all symbols ({len(symbols)}) on {timeframe} timeframe. This may take a few minutes. Results will appear as they come.")

    results = []
    for symbol in symbols:
        df = await fetch_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            signals = get_signals(df)
            # Always show result (remove filter)
            result = format_symbol_result(symbol, signals, timeframe)
            results.append(result)
            if len("\n\n".join(results)) > 1800:
                await ctx.send("\n\n".join(results))
                results = []
        await asyncio.sleep(1)

    if results:
        await ctx.send("\n\n".join(results))
    await ctx.send("Scan complete.")

@bot.command(name='add')
async def add_symbol(ctx, symbol):
    symbol = normalize_symbol(symbol.upper())
    watchlist = load_watchlist()
    if '/' in symbol:
        if symbol not in watchlist['crypto']:
            watchlist['crypto'].append(symbol)
            save_watchlist(watchlist)
            await ctx.send(f"Added {symbol} to crypto watchlist.")
        else:
            await ctx.send(f"{symbol} already in crypto watchlist.")
    else:
        if symbol not in watchlist['stocks']:
            watchlist['stocks'].append(symbol)
            save_watchlist(watchlist)
            await ctx.send(f"Added {symbol} to stocks watchlist.")
        else:
            await ctx.send(f"{symbol} already in stocks watchlist.")

@bot.command(name='remove')
async def remove_symbol(ctx, symbol):
    symbol = normalize_symbol(symbol.upper())
    watchlist = load_watchlist()
    removed = False
    if symbol in watchlist['stocks']:
        watchlist['stocks'].remove(symbol)
        removed = True
    if symbol in watchlist['crypto']:
        watchlist['crypto'].remove(symbol)
        removed = True
    if removed:
        save_watchlist(watchlist)
        await ctx.send(f"Removed {symbol} from watchlist.")
    else:
        await ctx.send(f"{symbol} not found in watchlist.")

@bot.command(name='list')
async def list_watchlist(ctx):
    watchlist = load_watchlist()
    stocks = ", ".join(watchlist['stocks']) if watchlist['stocks'] else "None"
    cryptos = ", ".join(watchlist['crypto']) if watchlist['crypto'] else "None"
    await ctx.send(f"**Stocks:** {stocks}\n**Crypto:** {cryptos}")

@bot.command(name='help')
async def help_command(ctx):
    help_text = """
**5-13-50 Trading Bot Commands**
`!scan all [timeframe]` – Scan all watchlist symbols (shows all symbols now). Timeframes: daily, weekly, 4h (default daily).
`!scan SYMBOL [timeframe]` – Scan a single symbol (e.g., `!scan AAPL`, `!scan XRP/USD`).
`!add SYMBOL` – Add a symbol (use `BTC/USD` for crypto).
`!remove SYMBOL` – Remove a symbol.
`!list` – Show current watchlist.
`!ping` – Test if bot is responsive.
`!help` – This message.
    """
    await ctx.send(help_text)

# Run the bot
if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)