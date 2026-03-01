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

# Suppress harmless mplfinance warnings about all‑NaN volumes
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

# === CONFIGURATION ===
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI')
PORT = int(os.getenv('PORT', 10000))

# MongoDB setup
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client['trading_bot']
watchlist_collection = db['watchlist']

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)
bot._skip_check = lambda x, y: False

last_command_time = {}
cancellation_flags = {}
user_locks = {}  # Per‑user lock to prevent simultaneous commands

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
# WATCHLIST FUNCTIONS (MongoDB)
# ====================

async def load_watchlist():
    try:
        doc = await watchlist_collection.find_one({'_id': 'main'})
        if doc:
            return {
                "stocks": doc.get('stocks', []),
                "crypto": doc.get('crypto', [])
            }
        else:
            default = {
                "_id": "main",
                "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG"],
                "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD"]
            }
            await watchlist_collection.insert_one(default)
            return default
    except Exception as e:
        print(f"❌ Error loading watchlist from MongoDB: {e}")
        return {
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG"],
            "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD"]
        }

async def save_watchlist(watchlist):
    try:
        await watchlist_collection.replace_one(
            {'_id': 'main'},
            {'_id': 'main', 'stocks': watchlist['stocks'], 'crypto': watchlist['crypto']},
            upsert=True
        )
        print(f"✅ Watchlist saved to MongoDB")
        return True
    except Exception as e:
        print(f"❌ Error saving watchlist to MongoDB: {e}")
        return False

def normalize_symbol(symbol):
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

# ----- Helper for crypto symbol conversion (for Finnhub) -----
def finnhub_crypto_symbol(symbol):
    """Convert 'BTC/USD' to 'BINANCE:BTCUSDT' (or similar). Adjust exchange as needed."""
    if '/' in symbol:
        base, quote = symbol.split('/')
        return f"BINANCE:{base}{quote}"
    return symbol

# ----- Twelve Data fetch (for stocks and crypto) -----
async def fetch_twelvedata(symbol, timeframe):
    interval_map = {'daily': '1day', 'weekly': '1week', '4h': '4h'}
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
    print(f"📡 Twelve Data request for {symbol}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"❌ Twelve Data error for {symbol}: status {resp.status}")
                    return None
                data = await resp.json()
                if 'values' not in data:
                    print(f"❌ Twelve Data error for {symbol}: {data}")
                    return None
                df = pd.DataFrame(data['values'])
                df = df.rename(columns={'datetime': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                df = df.sort_index()
                print(f"✅ Twelve Data success for {symbol}")
                return df
    except asyncio.TimeoutError:
        print(f"⏰ Twelve Data timeout for {symbol}")
        return None
    except Exception as e:
        print(f"❌ Twelve Data exception for {symbol}: {e}")
        return None

# ----- Finnhub fetch (for stocks and crypto) -----
async def fetch_finnhub(symbol, timeframe):
    is_crypto = '/' in symbol
    if is_crypto:
        finnhub_sym = finnhub_crypto_symbol(symbol)
        endpoint = "crypto/candle"
    else:
        finnhub_sym = symbol
        endpoint = "stock/candle"

    res_map = {'daily': 'D', 'weekly': 'W', '4h': '60'}
    resolution = res_map.get(timeframe)
    if not resolution:
        return None

    url = f"https://finnhub.io/api/v1/{endpoint}"
    params = {
        'symbol': finnhub_sym,
        'resolution': resolution,
        'from': int((datetime.now() - timedelta(days=200)).timestamp()),
        'to': int(datetime.now().timestamp()),
        'token': FINNHUB_API_KEY
    }
    print(f"📡 Finnhub request for {symbol} (as {finnhub_sym})")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"❌ Finnhub error for {symbol}: status {resp.status}")
                    return None
                data = await resp.json()
                if data.get('s') != 'ok':
                    print(f"❌ Finnhub error for {symbol}: {data}")
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
                df = df.sort_index()
                if timeframe == '4h' and resolution == '60':
                    df = df.resample('4H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                print(f"✅ Finnhub success for {symbol}")
                return df
    except asyncio.TimeoutError:
        print(f"⏰ Finnhub timeout for {symbol}")
        return None
    except Exception as e:
        print(f"❌ Finnhub exception for {symbol}: {e}")
        return None

# ----- Main fetch with fallback -----
async def fetch_ohlcv(symbol, timeframe):
    """Try Twelve Data first, then Finnhub."""
    print(f"🔍 Fetching {symbol} via Twelve Data (primary)")
    df = await fetch_twelvedata(symbol, timeframe)
    if df is not None:
        return df
    print(f"⚠️ Twelve Data failed, trying Finnhub as fallback for {symbol}")
    df = await fetch_finnhub(symbol, timeframe)
    if df is not None:
        return df
    print(f"❌ Both Twelve Data and Finnhub failed for {symbol}")
    return None

# ----- Finnhub News Fetching -----
async def fetch_stock_news(symbol):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': symbol,
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'to': datetime.now().strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    print(f"Finnhub news error for {symbol}: status {resp.status}")
                    return None
                data = await resp.json()
                return data if isinstance(data, list) else None
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return None

# ----- Indicator Calculations -----
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
        signals['volume_avg'] = None

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

    fig, axes = mpf.plot(
        chart_data,
        type='candle',
        style=s,
        addplot=apds,
        volume=True,
        figsize=(10, 6),
        returnfig=True,
        title=f'{symbol} – {timeframe} (last 30)',
        tight_layout=True,
        ylabel='Price (USD)'
    )

    if apds:
        axes[0].legend(loc='upper left', fontsize='small')

    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=120, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def format_embed(symbol, signals, timeframe):
    if not signals:
        return discord.Embed(title=f"Error", description=f"No data for {symbol}", color=0xff0000)

    sym_type = "Crypto" if '/' in symbol else "Stock"
    rating, color = get_rating(signals)

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

    if signals['overbought_triangle']:
        bb_status = "🔴 Overbought (touch)"
    elif signals['oversold_triangle']:
        bb_status = "🟢 Oversold (touch)"
    else:
        bb_status = "⚪ Normal"

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
    ema_lines = [f"{emoji} {lbl}: ${val:.2f}" for val, lbl, emoji in valid_items]
    ema_text = "\n".join(ema_lines) if valid_items else "N/A"

    embed = discord.Embed(
        title=f"{rating}",
        description=f"**{symbol}** · ${signals['price']:.2f}",
        color=color
    )
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
    embed.set_footer(text=f"{sym_type} · {timeframe} timeframe")
    return embed

# ====================
# DEDUPLICATION & USER LOCKS
# ====================

async def get_user_lock(user_id):
    """Get or create a lock for a specific user."""
    if user_id not in user_locks:
        user_locks[user_id] = asyncio.Lock()
    return user_locks[user_id]

# ====================
# SCAN CANCELLATION
# ====================

async def check_cancel(ctx):
    user_id = ctx.author.id
    if cancellation_flags.get(user_id, False):
        cancellation_flags[user_id] = False
        await ctx.send("🛑 Scan cancelled.")
        return True
    return False

@bot.command(name='stopscan')
async def stop_scan(ctx):
    cancellation_flags[ctx.author.id] = True
    await ctx.send("⏹️ Cancelling scan... (will stop after current symbol)")

# ====================
# DISCORD EVENTS & COMMANDS
# ====================

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    print("🔥 on_message triggered!")
    if message.author == bot.user:
        return
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('pong')

async def send_symbol_with_chart(ctx, symbol, df, timeframe):
    if await check_cancel(ctx):
        return False
    df = calculate_indicators(df)
    signals = get_signals(df)
    embed = format_embed(symbol, signals, timeframe)
    chart_buffer = generate_chart_image(df, symbol, timeframe)
    if chart_buffer:
        file = discord.File(chart_buffer, filename='chart.png')
        embed.set_image(url='attachment://chart.png')
        await ctx.send(embed=embed, file=file)
    else:
        embed.add_field(name="📊 Chart", value="*Insufficient data for chart*", inline=False)
        await ctx.send(embed=embed)
    return True

@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    user_id = ctx.author.id
    lock = await get_user_lock(user_id)

    # Acquire lock – this prevents the same user from running two commands simultaneously
    async with lock:
        # Debug: log every scan command
        print(f"👀 scan command received for user {user_id}, target={target}, timeframe={timeframe}")

        # Simple cooldown (5 seconds)
        now = datetime.now()
        last = last_command_time.get(user_id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[user_id] = now

        timeframe = timeframe.lower()
        if timeframe not in ['daily', 'weekly', '4h']:
            await ctx.send("Invalid timeframe. Use daily, weekly, or 4h.")
            return

        watchlist = await load_watchlist()
        symbols = watchlist['stocks'] + watchlist['crypto']

        if target.lower() != 'all':
            symbol = normalize_symbol(target)
            await ctx.send(f"Scanning **{symbol}** ({timeframe})...")
            df = await fetch_ohlcv(symbol, timeframe)
            if df is None or df.empty:
                await ctx.send(f"Could not fetch data for {symbol}.")
                return
            await send_symbol_with_chart(ctx, symbol, df, timeframe)
            return

        await ctx.send(f"Scanning all symbols ({len(symbols)}) on {timeframe} timeframe. This may take a few minutes. Results will appear as they come.")

        for symbol in symbols:
            if await check_cancel(ctx):
                break
            df = await fetch_ohlcv(symbol, timeframe)
            if df is not None and not df.empty:
                await send_symbol_with_chart(ctx, symbol, df, timeframe)
            await asyncio.sleep(8)

        cancellation_flags[user_id] = False
        await ctx.send("Scan complete.")

@bot.command(name='signals')
async def signals(ctx, timeframe='daily'):
    user_id = ctx.author.id
    lock = await get_user_lock(user_id)

    async with lock:
        print(f"👀 signals command received for user {user_id}, timeframe={timeframe}")

        now = datetime.now()
        last = last_command_time.get(user_id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[user_id] = now

        timeframe = timeframe.lower()
        if timeframe not in ['daily', 'weekly', '4h']:
            await ctx.send("Invalid timeframe. Use daily, weekly, or 4h.")
            return

        watchlist = await load_watchlist()
        symbols = watchlist['stocks'] + watchlist['crypto']

        await ctx.send(f"Scanning for signals on {timeframe} timeframe. This may take a few minutes. Results will appear as they come.")

        found_any = False
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            df = await fetch_ohlcv(symbol, timeframe)
            if df is not None and not df.empty:
                df_calc = calculate_indicators(df)
                sig = get_signals(df_calc)
                if sig and sig['net_score'] != 0:
                    found_any = True
                    await send_symbol_with_chart(ctx, symbol, df, timeframe)
            await asyncio.sleep(8)

        if not found_any and not cancellation_flags.get(user_id, False):
            await ctx.send("No symbols with active signals found.")
        cancellation_flags[user_id] = False
        await ctx.send("Signal scan complete.")

@bot.command(name='news')
async def stock_news(ctx, ticker: str, limit: int = 5):
    user_id = ctx.author.id
    lock = await get_user_lock(user_id)

    async with lock:
        print(f"👀 news command received for user {user_id}, ticker={ticker}, limit={limit}")

        now = datetime.now()
        last = last_command_time.get(user_id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[user_id] = now

        await ctx.send(f"📰 Fetching the latest **{limit}** news headlines for **{ticker.upper()}**...")

        news_data = await fetch_stock_news(ticker.upper())

        if not news_data or len(news_data) == 0:
            await ctx.send(f"Could not fetch news for {ticker.upper()}.")
            return

        embed = discord.Embed(
            title=f"Latest News for {ticker.upper()}",
            color=0x3498db
        )

        for article in news_data[:limit]:
            headline = article.get('headline', 'No Headline')
            source = article.get('source', 'Unknown')
            date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
            url = article.get('url', '')

            if len(headline) > 256:
                headline = headline[:253] + "..."

            embed.add_field(
                name=f"{source} - {date}",
                value=f"[{headline}]({url})",
                inline=False
            )

        embed.set_footer(text=f"Requested by {ctx.author.display_name}")
        await ctx.send(embed=embed)

@bot.command(name='add')
async def add_symbol(ctx, symbol):
    user_id = ctx.author.id
    lock = await get_user_lock(user_id)

    async with lock:
        print(f"👀 add command received for user {user_id}, symbol={symbol}")

        symbol = normalize_symbol(symbol.upper())
        watchlist = await load_watchlist()
        if '/' in symbol:
            if symbol not in watchlist['crypto']:
                watchlist['crypto'].append(symbol)
                if await save_watchlist(watchlist):
                    await ctx.send(f"✅ Added {symbol} to crypto watchlist.")
                else:
                    await ctx.send(f"❌ Could not save watchlist. Please check logs.")
            else:
                await ctx.send(f"{symbol} already in crypto watchlist.")
        else:
            if symbol not in watchlist['stocks']:
                watchlist['stocks'].append(symbol)
                if await save_watchlist(watchlist):
                    await ctx.send(f"✅ Added {symbol} to stocks watchlist.")
                else:
                    await ctx.send(f"❌ Could not save watchlist. Please check logs.")
            else:
                await ctx.send(f"{symbol} already in stocks watchlist.")

@bot.command(name='remove')
async def remove_symbol(ctx, symbol):
    user_id = ctx.author.id
    lock = await get_user_lock(user_id)

    async with lock:
        print(f"👀 remove command received for user {user_id}, symbol={symbol}")

        symbol = normalize_symbol(symbol.upper())
        watchlist = await load_watchlist()
        removed = False
        if symbol in watchlist['stocks']:
            watchlist['stocks'].remove(symbol)
            removed = True
        if symbol in watchlist['crypto']:
            watchlist['crypto'].remove(symbol)
            removed = True
        if removed:
            if await save_watchlist(watchlist):
                await ctx.send(f"✅ Removed {symbol} from watchlist.")
            else:
                await ctx.send(f"❌ Could not save watchlist. Please check logs.")
        else:
            await ctx.send(f"{symbol} not found in watchlist.")

@bot.command(name='list')
async def list_watchlist(ctx):
    user_id = ctx.author.id
    lock = await get_user_lock(user_id)

    async with lock:
        print(f"👀 list command received for user {user_id}")
        watchlist = await load_watchlist()
        stocks = ", ".join(watchlist['stocks']) if watchlist['stocks'] else "None"
        cryptos = ", ".join(watchlist['crypto']) if watchlist['crypto'] else "None"
        await ctx.send(f"**Stocks:** {stocks}\n**Crypto:** {cryptos}")

@bot.command(name='help')
async def help_command(ctx):
    help_text = """
**5-13-50 Trading Bot Commands**
`!scan all [timeframe]` – Scan all watchlist symbols (full overview with charts).
`!scan SYMBOL [timeframe]` – Scan a single symbol (with chart).
`!signals [timeframe]` – Scan only symbols with active signals (with charts).
`!news TICKET [limit]` – Fetch latest news headlines (e.g., `!news AAPL 5`).
`!add SYMBOL` – Add a symbol (use `BTC/USD` for crypto).
`!remove SYMBOL` – Remove a symbol.
`!list` – Show current watchlist.
`!ping` – Test if bot is responsive.
`!stopscan` – Stops any ongoing scan.
`!help` – This message.
    """
    await ctx.send(help_text)

# ====================
# MAIN ENTRY POINT
# ====================

async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())