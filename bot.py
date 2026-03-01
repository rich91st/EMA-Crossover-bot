import discord
from discord.ext import commands
import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import ta
import asyncio
import json
import os
import warnings
from datetime import datetime, timedelta
import motor.motor_asyncio

# Charting libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import io

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI')
PORT = int(os.getenv('PORT', 10000))

# MongoDB
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client['trading_bot']
watchlist_collection = db['watchlist']

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)
bot._skip_check = lambda x, y: False

last_command_time = {}
cancellation_flags = {}
user_locks = {}

# ====================
# WEB SERVER
# ====================
async def handle_health(request):
    return web.Response(text="OK")

async def start_web_server():
    app = web.Application()
    app.router.add_get('/', handle_health)
    app.router.add_get('/health', handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', PORT)
    await site.start()
    print(f"✅ Web server running on port {PORT}")

# ====================
# WATCHLIST (MongoDB)
# ====================
async def load_watchlist():
    try:
        doc = await watchlist_collection.find_one({'_id': 'main'})
        if doc:
            return {"stocks": doc.get('stocks', []), "crypto": doc.get('crypto', [])}
        else:
            default = {
                "_id": "main",
                "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG"],
                "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD"]
            }
            await watchlist_collection.insert_one(default)
            return default
    except:
        return {"stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG"], "crypto": []}

async def save_watchlist(watchlist):
    try:
        await watchlist_collection.replace_one({'_id': 'main'}, {'_id': 'main', 'stocks': watchlist['stocks'], 'crypto': watchlist['crypto']}, upsert=True)
        return True
    except:
        return False

def normalize_symbol(symbol):
    s = symbol.upper()
    crypto_map = {
        'BTC': 'BTC/USD', 'ETH': 'ETH/USD', 'SOL': 'SOL/USD',
        'XRP': 'XRP/USD', 'DOGE': 'DOGE/USD', 'PEPE': 'PEPE/USD'
    }
    if s in crypto_map:
        return crypto_map[s]
    if '/' in s:
        return s
    return s

# ====================
# DATA FETCHING – CLEAN VERSION
# ====================
async def fetch_twelvedata(symbol, timeframe):
    interval_map = {'daily': '1day', 'weekly': '1week', '4h': '4h'}
    interval = interval_map.get(timeframe)
    if not interval:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': symbol, 'interval': interval, 'apikey': TWELVEDATA_API_KEY, 'outputsize': 200}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if 'values' not in data:
                    return None
                df = pd.DataFrame(data['values'])
                df = df.rename(columns={'datetime': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df.astype(float).sort_index()
    except:
        return None

async def fetch_finnhub_stock(symbol, timeframe):
    res_map = {'daily': 'D', 'weekly': 'W', '4h': '60'}
    resolution = res_map.get(timeframe)
    if not resolution:
        return None
    url = f"https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': int((datetime.now() - timedelta(days=200)).timestamp()),
        'to': int(datetime.now().timestamp()),
        'token': FINNHUB_API_KEY
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get('s') != 'ok':
                    return None
                df = pd.DataFrame({'timestamp': data['t'], 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['v']})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                if timeframe == '4h' and resolution == '60':
                    df = df.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                return df.sort_index()
    except:
        return None

async def fetch_finnhub_crypto(symbol, timeframe):
    # Convert BTC/USD to BINANCE:BTCUSDT
    if '/' not in symbol:
        return None
    base, quote = symbol.split('/')
    finnhub_sym = f"BINANCE:{base}{quote}"
    res_map = {'daily': 'D', 'weekly': 'W', '4h': '60'}
    resolution = res_map.get(timeframe)
    if not resolution:
        return None
    url = f"https://finnhub.io/api/v1/crypto/candle"
    params = {
        'symbol': finnhub_sym,
        'resolution': resolution,
        'from': int((datetime.now() - timedelta(days=200)).timestamp()),
        'to': int(datetime.now().timestamp()),
        'token': FINNHUB_API_KEY
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get('s') != 'ok':
                    return None
                df = pd.DataFrame({'timestamp': data['t'], 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['v']})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                if timeframe == '4h' and resolution == '60':
                    df = df.resample('4H').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                return df.sort_index()
    except:
        return None

async def fetch_ohlcv(symbol, timeframe):
    if '/' in symbol:
        # Crypto: try Twelve Data first, then Finnhub crypto
        print(f"🔍 Crypto {symbol}: trying Twelve Data")
        df = await fetch_twelvedata(symbol, timeframe)
        if df is not None:
            return df
        print(f"⚠️ Twelve Data failed, trying Finnhub crypto")
        df = await fetch_finnhub_crypto(symbol, timeframe)
        if df is not None:
            return df
        print(f"❌ All crypto sources failed for {symbol}")
        return None
    else:
        # Stock: try Twelve Data first, then Finnhub stock
        print(f"🔍 Stock {symbol}: trying Twelve Data")
        df = await fetch_twelvedata(symbol, timeframe)
        if df is not None:
            return df
        print(f"⚠️ Twelve Data failed, trying Finnhub stock")
        df = await fetch_finnhub_stock(symbol, timeframe)
        if df is not None:
            return df
        print(f"❌ All stock sources failed for {symbol}")
        return None

# ====================
# INDICATORS (unchanged)
# ====================
def calculate_indicators(df):
    df['ema5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['ema13'] = ta.trend.ema_indicator(df['close'], window=13)
    df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema200'] = ta.trend.ema_indicator(df['close'], window=200)
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=3)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
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
    if 'volume' in latest and 'volume_avg' in latest and pd.notna(latest['volume_avg']):
        signals['volume'] = latest['volume']
        signals['volume_avg'] = latest['volume_avg']
    else:
        signals['volume'] = None
    signals['trend'] = 'UPTREND' if latest['close'] > latest['ema50'] and latest['ema5'] > latest['ema13'] else 'DOWNTREND'
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
    bullish = [signals['ema5_cross_above_13'], signals['ema13_cross_above_50'], signals['buy_signal'], signals['oversold_triangle'], signals['rsi_oversold']]
    bearish = [signals['ema5_cross_below_13'], signals['ema13_cross_below_50'], signals['sell_signal'], signals['overbought_triangle'], signals['rsi_overbought']]
    signals['bullish_count'] = sum(bullish)
    signals['bearish_count'] = sum(bearish)
    signals['net_score'] = signals['bullish_count'] - signals['bearish_count']
    return signals

def get_rating(signals):
    net = signals['net_score']
    rsi = signals['rsi']
    buy = signals['buy_signal']
    sell = signals['sell_signal']
    above_200 = signals['price'] > signals['ema200'] if not pd.isna(signals['ema200']) else False
    ob = signals['overbought_triangle']
    os = signals['oversold_triangle']
    if net >= 2 or (buy and rsi >= 60) or os:
        return "STRONG BUY", 0x00ff00
    if net == 1 or (buy and rsi >= 50):
        return "BUY", 0x00cc00
    if net == 0 and (above_200 or signals['rsi_oversold'] or os):
        return "WEAK BUY", 0x88ff88
    if net <= -2 or (sell and rsi <= 40) or ob:
        return "STRONG SELL", 0xff0000
    if net == -1 or (sell and rsi <= 50):
        return "SELL", 0xcc0000
    if net == 0 and (not above_200 or signals['rsi_overbought'] or ob):
        return "WEAK SELL", 0xff8888
    return "NEUTRAL", 0xffff00

def generate_chart_image(df, symbol, timeframe):
    if len(df) < 20:
        return None
    chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(30).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    apds = []
    if not df['ema5'].tail(30).isna().all():
        apds.append(mpf.make_addplot(df['ema5'].tail(30), color='#00ff00', width=2.5, label='EMA5'))
    if not df['ema13'].tail(30).isna().all():
        apds.append(mpf.make_addplot(df['ema13'].tail(30), color='#ffd700', width=2.5, label='EMA13'))
    if not df['ema50'].tail(30).isna().all():
        apds.append(mpf.make_addplot(df['ema50'].tail(30), color='#ff4444', width=2.5, label='EMA50'))
    if not df['ema200'].tail(30).isna().all():
        apds.append(mpf.make_addplot(df['ema200'].tail(30), color='#ff00ff', width=3.5, label='EMA200'))
    mc = mpf.make_marketcolors(up='#00ff88', down='#ff4d4d', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)
    fig, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds, volume=True, figsize=(10,6), returnfig=True, title=f'{symbol} – {timeframe}', tight_layout=True)
    if apds:
        axes[0].legend(loc='upper left')
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def format_embed(symbol, signals, timeframe):
    sym_type = "Crypto" if '/' in symbol else "Stock"
    rating, color = get_rating(signals)
    vol_display = "N/A"
    if signals.get('volume') and signals.get('volume_avg') and signals['volume_avg'] > 0:
        vol_ratio = signals['volume'] / signals['volume_avg']
        vol_status = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
        vol_display = f"{vol_status} ({vol_ratio:.1f}x)"
    reasons = []
    if signals['ema5_cross_above_13']: reasons.append("EMA5 ↑ EMA13")
    if signals['ema13_cross_above_50']: reasons.append("EMA13 ↑ EMA50")
    if signals['ema5_cross_below_13']: reasons.append("EMA5 ↓ EMA13")
    if signals['ema13_cross_below_50']: reasons.append("EMA13 ↓ EMA50")
    if signals['oversold_triangle']: reasons.append("🔻 Oversold BB touch")
    if signals['overbought_triangle']: reasons.append("🔺 Overbought BB touch")
    if signals['rsi_oversold']: reasons.append("RSI Oversold")
    if signals['rsi_overbought']: reasons.append("RSI Overbought")
    if signals['price'] > signals['ema200'] and not pd.isna(signals['ema200']): reasons.append("Above 200 EMA")
    elif signals['price'] < signals['ema200'] and not pd.isna(signals['ema200']): reasons.append("Below 200 EMA")
    if not reasons: reasons.append("No significant signals")
    reason_str = " | ".join(reasons)
    bb_status = "🔴 Overbought (touch)" if signals['overbought_triangle'] else "🟢 Oversold (touch)" if signals['oversold_triangle'] else "⚪ Normal"
    support = signals['support_20']
    resistance = signals['resistance_20']
    stop_loss = support
    target = resistance + (resistance - support)
    ema_items = [
        (signals['ema5'], '5', '🟢'),
        (signals['ema13'], '13', '🟡'),
        (signals['ema50'], '50', '🔴'),
        (signals['ema200'], '200', '🟣')
    ]
    valid_items = [(val, lbl, emoji) for val, lbl, emoji in ema_items if not pd.isna(val)]
    valid_items.sort(reverse=True)
    ema_text = "\n".join([f"{emoji} {lbl}: ${val:.2f}" for val, lbl, emoji in valid_items]) if valid_items else "N/A"
    embed = discord.Embed(title=f"{rating}", description=f"**{symbol}** · ${signals['price']:.2f}", color=color)
    embed.add_field(name="RSI", value=f"{signals['rsi']:.1f}", inline=True)
    embed.add_field(name="Trend", value=signals['trend'], inline=True)
    embed.add_field(name="Volume", value=vol_display, inline=True)
    embed.add_field(name="Bollinger Bands", value=bb_status, inline=True)
    embed.add_field(name="Reason", value=reason_str, inline=False)
    embed.add_field(name="Support", value=f"${support:.2f}", inline=True)
    embed.add_field(name="Resistance", value=f"${resistance:.2f}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop_loss:.2f}", inline=True)
    embed.add_field(name="Target", value=f"${target:.2f}", inline=True)
    embed.add_field(name="EMAs (sorted)", value=ema_text, inline=False)
    embed.set_footer(text=f"{sym_type} · {timeframe}")
    return embed

# ====================
# COMMANDS
# ====================
@bot.event
async def on_ready():
    print(f'{bot.user} online')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('pong')

async def send_symbol(ctx, symbol, df, timeframe):
    df = calculate_indicators(df)
    signals = get_signals(df)
    embed = format_embed(symbol, signals, timeframe)
    chart = generate_chart_image(df, symbol, timeframe)
    if chart:
        file = discord.File(chart, filename='chart.png')
        embed.set_image(url='attachment://chart.png')
        await ctx.send(embed=embed, file=file)
    else:
        await ctx.send(embed=embed)

@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    # Simple cooldown to prevent double execution
    user_id = ctx.author.id
    now = datetime.now()
    last = last_command_time.get(user_id)
    if last and (now - last) < timedelta(seconds=5):
        return
    last_command_time[user_id] = now

    timeframe = timeframe.lower()
    if timeframe not in ['daily', 'weekly', '4h']:
        await ctx.send("Invalid timeframe")
        return

    watchlist = await load_watchlist()
    symbols = watchlist['stocks'] + watchlist['crypto']

    if target.lower() != 'all':
        symbol = normalize_symbol(target)
        await ctx.send(f"Scanning **{symbol}**...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}")
            return
        await send_symbol(ctx, symbol, df, timeframe)
        return

    await ctx.send(f"Scanning all {len(symbols)} symbols...")
    for symbol in symbols:
        df = await fetch_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            await send_symbol(ctx, symbol, df, timeframe)
        await asyncio.sleep(8)
    await ctx.send("Scan complete.")

@bot.command(name='add')
async def add(ctx, symbol):
    s = normalize_symbol(symbol.upper())
    wl = await load_watchlist()
    if '/' in s:
        if s not in wl['crypto']:
            wl['crypto'].append(s)
            if await save_watchlist(wl):
                await ctx.send(f"✅ Added {s}")
            else:
                await ctx.send("❌ Save failed")
        else:
            await ctx.send(f"{s} already exists")
    else:
        if s not in wl['stocks']:
            wl['stocks'].append(s)
            if await save_watchlist(wl):
                await ctx.send(f"✅ Added {s}")
            else:
                await ctx.send("❌ Save failed")
        else:
            await ctx.send(f"{s} already exists")

@bot.command(name='list')
async def list_(ctx):
    wl = await load_watchlist()
    stocks = ", ".join(wl['stocks']) if wl['stocks'] else "None"
    cryptos = ", ".join(wl['crypto']) if wl['crypto'] else "None"
    await ctx.send(f"**Stocks:** {stocks}\n**Crypto:** {cryptos}")

@bot.command(name='help')
async def help_(ctx):
    await ctx.send("""
**Commands**
`!scan all [daily|weekly|4h]` – scan all
`!scan SYMBOL` – scan single
`!add SYMBOL` – add to watchlist
`!list` – show watchlist
`!ping` – test
`!help` – this
""")

# ====================
# MAIN
# ====================
async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())