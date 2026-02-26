import discord
from discord.ext import commands
import finnhub
import pandas as pd
import numpy as np
import ta
import asyncio
import json
import os
from datetime import datetime

# === CONFIGURATION ===
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Discord bot setup – with help_command=None to allow custom !help
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

# Watchlist file
WATCHLIST_FILE = 'watchlist.json'

# ====================
# HELPER FUNCTIONS
# ====================

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE) as f:
            return json.load(f)
    default = {
        "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG"],
        "crypto": ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", "BINANCE:SOLUSDT", "BINANCE:XRPUSDT", "BINANCE:DOGEUSDT", "BINANCE:PEPEUSDT"]
    }
    save_watchlist(default)
    return default

def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(watchlist, f, indent=2)

async def fetch_ohlcv(symbol, timeframe):
    if timeframe == 'daily':
        res = 'D'
        limit = 200
    elif timeframe == 'weekly':
        res = 'W'
        limit = 200
    elif timeframe == '4h':
        res = '60'
        limit = 800
    else:
        return None

    to_time = int(datetime.now().timestamp())
    from_time = to_time - limit * 24 * 60 * 60

    try:
        data = finnhub_client.stock_candles(symbol, res, from_time, to_time)
        if data['s'] != 'ok':
            return None
        df = pd.DataFrame({
            'timestamp': data['t'],
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)

        if timeframe == '4h':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
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

    sym_type = "Crypto" if ':' in symbol else "Stock"

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
    print(f"Received message: {message.content} from {message.author}")
    await bot.process_commands(message)

# ====================
# DISCORD COMMANDS (Start with ping)
# ====================

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('pong')

# Run the bot
if __name__ == '__main__':
    bot.run(DISCORD_TOKEN)