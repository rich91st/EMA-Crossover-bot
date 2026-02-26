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
# CoinGecko does not require an API key for free tier

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
    if symbol in crypto_map:
        return crypto_map[symbol]
    if '/' in symbol:
        return symbol
    return symbol

async def fetch_twelvedata(symbol, timeframe):
    """Fetch OHLCV from Twelve Data (used for stocks and fallback)."""
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

async def fetch_coingecko(symbol, timeframe):
    """Fetch OHLCV from CoinGecko for crypto symbols (format: BTC/USD)."""
    # CoinGecko expects IDs like 'bitcoin', not symbols. We'll use the /simple/price endpoint for current price,
    # but for historical OHLC we need the coin ID. Since we need full OHLC for indicators, we'll use the /coins/{id}/ohlc endpoint.
    # We'll map common symbols to CoinGecko IDs. For simplicity, we'll extract the base (e.g., 'BTC' from 'BTC/USD').
    base = symbol.split('/')[0].lower()
    coin_map = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'sol': 'solana',
        'xrp': 'ripple',
        'doge': 'dogecoin',
        'pepe': 'pepe',
        'ada': 'cardano',
        'dot': 'polkadot',
        'link': 'chainlink'
    }
    coin_id = coin_map.get(base)
    if not coin_id:
        print(f"Unsupported crypto symbol for CoinGecko: {symbol}")
        return None

    # Map timeframe to CoinGecko days parameter
    days_map = {
        'daily': 30,      # last 30 days of daily data
        'weekly': 90,     # last 90 days (approx 13 weeks)
        '4h': 7           # last 7 days of 4-hour data (CoinGecko provides 4h data for up to 7 days)
    }
    days = days_map.get(timeframe)
    if not days:
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        'vs_currency': 'usd',
        'days': days
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"CoinGecko error for {symbol}: status {resp.status}")
                    return None
                data = await resp.json()
                # data is list of [timestamp, open, high, low, close]
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                # CoinGecko does not provide volume in this endpoint; we'll set volume to NaN
                df['volume'] = np.nan
                return df
    except Exception as e:
        print(f"Error fetching {symbol} from CoinGecko: {e}")
        return None

async def fetch_ohlcv(symbol, timeframe):
    """Main fetch function: uses CoinGecko for crypto, Twelve Data for stocks, with fallback."""
    if '/' in symbol:
        # Try CoinGecko first
        df = await fetch_coingecko(symbol, timeframe)
        if df is not None and not df.empty:
            return df
        # Fallback to Twelve Data
        print(f"Falling back to Twelve Data for {symbol}")
        return await fetch_twelvedata(symbol, timeframe)
    else:
        # Stocks: use Twelve Data
        return await fetch_twelvedata(symbol, timeframe)

def calculate_indicators(df):
    """Calculate all TA indicators, safely handling missing volume."""
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema13'] = ta.trend.ema_indicator(df['close'], window=13)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)

    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=3)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_basis'] = bb.bollinger_mavg()

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Volume average – only if volume column exists and has data
    if 'volume' in df.columns and df['volume'].notna().any():
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
    else:
        df['volume_avg'] = None

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

    # Safely handle volume
    if 'volume' in latest and 'volume_avg' in latest and pd.notna(latest['volume_avg']):
        signals['volume'] = latest['volume']
        signals['volume_avg'] = latest['volume_avg']
    else:
        signals['volume'] = None
        signals['volume_avg'] = None

    # Trend direction
    signals['trend'] = 'UPTREND' if latest['close'] > latest['ema50'] and latest['ema5'] > latest['ema13'] else 'DOWNTREND'

    # Support & Resistance (simple: 20-day low/high)
    signals['support_20'] = df['low'].tail(20).min()
    signals['resistance_20'] = df['high'].tail(20).max()

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

    signals['overbought_triangle'] = signals['touch_upper_bb'] and signals['rsi_overbought']
    signals['oversold_triangle'] = signals['touch_lower_bb'] and signals['rsi_oversold']

    bullish = [
        signals['ema5_cross_above_13'],
        signals['ema13_cross_above_50'],
        signals['buy_signal'],
        signals['oversold_triangle'],
        signals['rsi_oversold']
    ]
    bearish = [
        signals['ema5_cross_below_13'],
        signals['ema13_cross_below_50'],
        signals['sell_signal'],
        signals['overbought_triangle'],
        signals['rsi_overbought']
    ]
    signals['bullish_count'] = sum(bullish)
    signals['bearish_count'] = sum(bearish)
    signals['net_score'] = signals['bullish_count'] - signals['bearish_count']

    return signals

def get_rating(signals):
    """Return (rating_text, color) based on signals."""
    net = signals['net_score']
    rsi = signals['rsi']
    buy_signal = signals['buy_signal']
    sell_signal = signals['sell_signal']
    above_200 = signals['price'] > signals['ema200'] if not pd.isna(signals['ema200']) else False
    ob_triangle = signals['overbought_triangle']
    os_triangle = signals['oversold_triangle']

    if net >= 2 or (buy_signal and rsi >= 60) or os_triangle:
        return "STRONG BUY", 0x00ff00
    elif net == 1 or (buy_signal and rsi >= 50):
        return "BUY", 0x00cc00
    elif net == 0 and (above_200 or signals['rsi_oversold'] or os_triangle):
        return "WEAK BUY", 0x88ff88
    elif net <= -2 or (sell_signal and rsi <= 40) or ob_triangle:
        return "STRONG SELL", 0xff0000
    elif net == -1 or (sell_signal and rsi <= 50):
        return "SELL", 0xcc0000
    elif net == 0 and (not above_200 or signals['rsi_overbought'] or ob_triangle):
        return "WEAK SELL", 0xff8888
    else:
        return "NEUTRAL", 0xffff00

def format_embed(symbol, signals, timeframe):
    """Return a discord.Embed with all the fancy formatting."""
    if not signals:
        return discord.Embed(title=f"Error", description=f"No data for {symbol}", color=0xff0000)

    sym_type = "Crypto" if '/' in symbol else "Stock"
    rating, color = get_rating(signals)

    # Volume status – safely handle missing volume
    if signals.get('volume') and signals.get('volume_avg') and signals['volume_avg'] > 0:
        vol_ratio = signals['volume'] / signals['volume_avg']
        if vol_ratio > 1.5:
            vol_status = "High"
        elif vol_ratio < 0.5:
            vol_status = "Low"
        else:
            vol_status = "Normal"
        vol_display = f"{vol_status} ({vol_ratio:.1f}x)"
    else:
        vol_display = "N/A (no volume data)"

    # Build reason string
    reasons = []
    if signals['ema5_cross_above_13']:
        reasons.append("EMA5 ↑ EMA13")
    if signals['ema13_cross_above_50']:
        reasons.append("EMA13 ↑ EMA50")
    if signals['ema5_cross_below_13']:
        reasons.append("EMA5 ↓ EMA13")
    if signals['ema13_cross_below_50']:
        reasons.append("EMA13 ↓ EMA50")
    if signals['oversold_triangle']:
        reasons.append("🔻 Oversold BB touch")
    if signals['overbought_triangle']:
        reasons.append("🔺 Overbought BB touch")
    if signals['rsi_oversold']:
        reasons.append("RSI Oversold")
    if signals['rsi_overbought']:
        reasons.append("RSI Overbought")
    if signals['price'] > signals['ema200'] and not pd.isna(signals['ema200']):
        reasons.append("Above 200 EMA")
    elif signals['price'] < signals['ema200'] and not pd.isna(signals['ema200']):
        reasons.append("Below 200 EMA")
    if not reasons:
        reasons.append("No significant signals")

    reason_str = " | ".join(reasons)

    # Bollinger Bands status field
    if signals['overbought_triangle']:
        bb_status = "🔴 Overbought (touch)"
    elif signals['oversold_triangle']:
        bb_status = "🟢 Oversold (touch)"
    else:
        bb_status = "⚪ Normal"

    # Support, Resistance, Stop Loss, Target
    support = signals['support_20']
    resistance = signals['resistance_20']
    stop_loss = support
    target = resistance + (resistance - support)  # simple projection

    # Create embed
    embed = discord.Embed(
        title=f"{rating}",
        description=f"**{symbol}** · ${signals['price']:.2f}",
        color=color
    )
    embed.add_field(name="RSI", value=f"{signals['rsi']:.1f}", inline=True)
    embed.add_field(name="Trend", value=signals['trend'], inline=True)
    embed.add_field(name="Volume", value=vol_display, inline=True)

    ema_text = f"5: ${signals['ema5']:.2f}\n13: ${signals['ema13']:.2f}\n50: ${signals['ema50']:.2f}\n200: ${signals['ema200']:.2f}"
    embed.add_field(name="EMAs", value=ema_text, inline=True)

    embed.add_field(name="Bollinger Bands", value=bb_status, inline=True)
    embed.add_field(name="Reason", value=reason_str, inline=False)

    embed.add_field(name="Support", value=f"${support:.2f}", inline=True)
    embed.add_field(name="Resistance", value=f"${resistance:.2f}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop_loss:.2f}", inline=True)
    embed.add_field(name="Target", value=f"${target:.2f}", inline=True)

    embed.set_footer(text=f"{sym_type} · {timeframe} timeframe")
    return embed

# ====================
# DISCORD EVENTS
# ====================

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
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
    # Cooldown
    now = datetime.now()
    last = last_command_time.get(ctx.author.id)
    if last and (now - last) < timedelta(seconds=5):
        return
    last_command_time[ctx.author.id] = now

    timeframe = timeframe.lower()
    if timeframe not in ['daily', 'weekly', '4h']:
        await ctx.send("Invalid timeframe. Use daily, weekly, or 4h.")
        return

    watchlist = load_watchlist()
    symbols = watchlist['stocks'] + watchlist['crypto']

    if target.lower() != 'all':
        symbol = normalize_symbol(target)
        await ctx.send(f"Scanning **{symbol}** ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}.")
            return
        df = calculate_indicators(df)
        signals = get_signals(df)
        embed = format_embed(symbol, signals, timeframe)
        await ctx.send(embed=embed)
        return

    # Scan all
    await ctx.send(f"Scanning all symbols ({len(symbols)}) on {timeframe} timeframe. This may take a few minutes. Results will appear as they come.")

    for symbol in symbols:
        df = await fetch_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            signals = get_signals(df)
            embed = format_embed(symbol, signals, timeframe)
            await ctx.send(embed=embed)
        await asyncio.sleep(8)  # Respect rate limits

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
`!scan all [timeframe]` – Scan all watchlist symbols with enhanced output.
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