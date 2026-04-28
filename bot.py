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
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta
import motor.motor_asyncio
import yfinance as yf

# Alpaca (only for crypto fallback)
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# Charting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import mplfinance as mpf
import io

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

# === CONFIGURATION ===
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI')
PORT = int(os.getenv('PORT', 10000))

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# MongoDB – watchlist only
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client['trading_bot']
watchlist_collection = db['watchlist']

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

last_command_time = {}
user_busy = {}
cancellation_flags = {}

# Cache
data_cache = {}
CACHE_DURATION = timedelta(minutes=10)

# Rate limiters for fallback APIs
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = asyncio.Lock()
    async def wait_if_needed(self):
        async with self.lock:
            now = datetime.now()
            self.calls = [t for t in self.calls if now - t < timedelta(seconds=self.period)]
            if len(self.calls) >= self.max_calls:
                oldest = self.calls[0]
                sleep_time = (oldest + timedelta(seconds=self.period) - now).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            self.calls.append(now)

twelvedata_limiter = RateLimiter(8, 60)
finnhub_limiter = RateLimiter(60, 60)
coingecko_limiter = RateLimiter(30, 60)

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
    print(f"✅ Web server on port {PORT}")

# ====================
# WATCHLIST HELPERS
# ====================
async def load_watchlist():
    try:
        doc = await watchlist_collection.find_one({'_id': 'main'})
        if doc:
            return {"stocks": doc.get('stocks', []), "crypto": doc.get('crypto', [])}
        default = {
            "_id": "main",
            "stocks": ["AAPL","MSFT","GOOGL","AMZN","NVDA","VUG","QUBT","TSLA","LYFT","NFLX","ORCL","UBER","HOOD","SOFI","SPY","NIO","PLTR","GRAB","LMT","MARA","SOUN","APLD","CLSK","OPEN","ASML","RIOT","AAL","F","FCEL"],
            "crypto": ["BTC/USD","ETH/USD","SOL/USD","XRP/USD","DOGE/USD","PEPE/USD","LINK/USD"]
        }
        await watchlist_collection.insert_one(default)
        return default
    except:
        return {"stocks": ["AAPL","MSFT","GOOGL","AMZN","NVDA","SPY"], "crypto": []}

async def save_watchlist(watchlist):
    try:
        await watchlist_collection.replace_one({'_id': 'main'}, {'_id': 'main', 'stocks': watchlist['stocks'], 'crypto': watchlist['crypto']}, upsert=True)
        return True
    except:
        return False

def normalize_symbol(symbol):
    symbol = symbol.upper()
    crypto_map = {'BTC':'BTC/USD','ETH':'ETH/USD','SOL':'SOL/USD','XRP':'XRP/USD','DOGE':'DOGE/USD','PEPE':'PEPE/USD','ADA':'ADA/USD','DOT':'DOT/USD','LINK':'LINK/USD'}
    if symbol in crypto_map:
        return crypto_map[symbol]
    if '/' in symbol:
        return symbol
    return symbol

def get_tradingview_web_link(symbol):
    if '/' in symbol:
        return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.split('/')[0]}USDT"
    return f"https://www.tradingview.com/chart/?symbol=NASDAQ:{symbol}"

# ====================
# DATA FETCHING – RELIABLE VOLUME
# ====================
async def fetch_yfinance_4h(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="30d", interval="1h")
        if data.empty:
            return None
        df = data.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        if df.empty:
            return None
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        print(f"✅ yfinance 4h {symbol}: price={df['close'].iloc[-1]:.2f}, vol={df['volume'].iloc[-1]:.0f}")
        return df
    except Exception as e:
        print(f"yfinance 4h error {symbol}: {e}")
        return None

async def fetch_twelvedata(symbol, timeframe):
    interval_map = {'5min':'5min','15min':'15min','30min':'30min','1h':'1h','4h':'4h','daily':'1day','weekly':'1week'}
    interval = interval_map.get(timeframe)
    if not interval:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': symbol, 'interval': interval, 'apikey': TWELVEDATA_API_KEY, 'outputsize': 100}
    await twelvedata_limiter.wait_if_needed()
    try:
        async with aiohttp.ClientSession(timeout=15) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if 'values' not in data:
                    return None
                df = pd.DataFrame(data['values'])
                df = df.rename(columns={'datetime': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                for col in ['open','high','low','close','volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'volume' not in df.columns:
                    df['volume'] = 0
                df['volume'] = df['volume'].fillna(0)
                df = df.sort_index().tail(100)
                return df
    except:
        return None

async def fetch_ohlcv(symbol, timeframe):
    cache_key = f"{symbol}_{timeframe}"
    now = datetime.now()
    if cache_key in data_cache and data_cache[cache_key][1] > now:
        return data_cache[cache_key][0]

    df = None
    is_crypto = '/' in symbol

    if not is_crypto:
        if timeframe == 'daily':
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="120d", interval="1d")
                if not df.empty:
                    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            except:
                pass
        elif timeframe == 'weekly':
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1y", interval="1wk")
                if not df.empty:
                    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            except:
                pass
        elif timeframe == '4h':
            df = await fetch_yfinance_4h(symbol)
            if df is None:
                df = await fetch_twelvedata(symbol, '4h')
        elif timeframe == '1h':
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1h")
                if not data.empty:
                    df = data.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
            except:
                pass
            if df is None:
                df = await fetch_twelvedata(symbol, '1h')
        else:  # lower timeframes
            df = await fetch_twelvedata(symbol, timeframe)

    if df is not None and not df.empty:
        if 'volume' not in df.columns:
            df['volume'] = 0
        df['volume'] = df['volume'].fillna(0).astype(float)
        data_cache[cache_key] = (df, now + CACHE_DURATION)
    return df

# ====================
# INDICATORS & SIGNALS
# ====================
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
    bullish = [signals['ema5_cross_above_13'], signals['ema13_cross_above_50'], signals['buy_signal'],
               signals['oversold_triangle'], signals['rsi_oversold']]
    bearish = [signals['ema5_cross_below_13'], signals['ema13_cross_below_50'], signals['sell_signal'],
               signals['overbought_triangle'], signals['rsi_overbought']]
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

# ====================
# CHART GENERATION
# ====================
def generate_chart_image(df, symbol, timeframe):
    if len(df) < 20:
        return None
    if timeframe in ['5min','15min','1h']:
        chart_data = df[['open','high','low','close','volume']].tail(50).copy()
    else:
        chart_data = df[['open','high','low','close','volume']].tail(30).copy()
    chart_data.columns = ['Open','High','Low','Close','Volume']
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='white', wick='white', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False, facecolor='#1e1e1e', figcolor='#1e1e1e', gridcolor='#444444')
    apds = []
    chart_len = len(chart_data)
    if not df['ema5'].tail(chart_len).isna().all():
        apds.append(mpf.make_addplot(df['ema5'].tail(chart_len), color='#00ff00', width=2.5, label='EMA5'))
    if not df['ema13'].tail(chart_len).isna().all():
        apds.append(mpf.make_addplot(df['ema13'].tail(chart_len), color='#ffd700', width=2.5, label='EMA13'))
    if not df['ema50'].tail(chart_len).isna().all():
        apds.append(mpf.make_addplot(df['ema50'].tail(chart_len), color='#ff4444', width=2.5, label='EMA50'))
    if not df['ema200'].tail(chart_len).isna().all():
        apds.append(mpf.make_addplot(df['ema200'].tail(chart_len), color='#ff00ff', width=3.5, label='EMA200'))
    try:
        vol_nan = chart_data['Volume'].isna().all()
        fig, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds, volume=not vol_nan, figsize=(10,6), returnfig=True, title=f'{symbol} – {timeframe}', tight_layout=True)
        if apds:
            axes[0].legend(loc='upper left', fontsize=9, facecolor='#333333', edgecolor='white', labelcolor='white', framealpha=0.8)
        axes[0].tick_params(colors='white')
        axes[0].set_facecolor('#1e1e1e')
        if not vol_nan:
            axes[2].set_ylabel('Volume', color='white')
            axes[2].tick_params(colors='white')
            axes[2].set_facecolor('#1e1e1e')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name, format='PNG', dpi=120, bbox_inches='tight', facecolor=s['facecolor'])
            tmp.flush()
            with open(tmp.name, 'rb') as f:
                img = f.read()
        os.unlink(tmp.name)
        plt.close(fig)
        return io.BytesIO(img)
    except:
        return None

def generate_zone_chart(df, symbol, zones):
    if len(df) < 20:
        return None
    chart_data = df[['open','high','low','close','volume']].tail(100).copy()
    chart_data.columns = ['Open','High','Low','Close','Volume']
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='white', wick='white', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False, facecolor='#1e1e1e', figcolor='#1e1e1e', gridcolor='#444444')
    apds = []
    if zones:
        strengths = [z['strength'] for z in zones]
        min_s, max_s = min(strengths), max(strengths)
        norm = [(s-min_s)/(max_s-min_s) if max_s>min_s else 0.5 for s in strengths]
        cmap = matplotlib.colormaps['RdYlGn_r']
        for i, z in enumerate(zones):
            color = mcolors.to_hex(cmap(norm[i]))
            apds.append(mpf.make_addplot([z['level']]*len(chart_data), color=color, width=2.0, linestyle='-', label=f"Demand ${z['level']:.2f} (touches:{z['strength']})"))
    try:
        fig, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds, volume=True, figsize=(12,7), returnfig=True, title=f'{symbol} Demand Zones (30min)', tight_layout=True)
        if apds:
            axes[0].legend(loc='upper left', fontsize=10, facecolor='#333', edgecolor='white', labelcolor='white')
        axes[2].set_ylabel('Volume', color='white')
        axes[2].tick_params(colors='white')
        axes[0].tick_params(colors='white')
        axes[0].set_facecolor('#1e1e1e')
        axes[2].set_facecolor('#1e1e1e')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig.savefig(tmp.name, format='PNG', dpi=150, bbox_inches='tight', facecolor=s['facecolor'])
            tmp.flush()
            with open(tmp.name, 'rb') as f:
                img = f.read()
        os.unlink(tmp.name)
        plt.close(fig)
        return io.BytesIO(img)
    except:
        return None

# ====================
# STRUCTURE ANALYSIS (FIXED)
# ====================
def find_swings(df, window=5):
    if len(df) < window*2+1:
        return [], []
    highs, lows = df['high'].values, df['low'].values
    idx = df.index
    sh, sl = [], []
    for i in range(window, len(df)-window):
        if highs[i] == max(highs[i-window:i+window+1]):
            sh.append((idx[i], highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            sl.append((idx[i], lows[i]))
    return sh, sl

def analyze_structure(df, window=5):
    if len(df) < 50:
        return {'trend':'neutral','last_event':None,'last_event_direction':None,'bos_events':[],'choch_events':[],'description':'Insufficient data'}
    sh, sl = find_swings(df, window)
    ema50 = df['ema50'].iloc[-1]
    price = df['close'].iloc[-1]
    primary = 'bullish' if price > ema50 else 'bearish' if price < ema50 else 'neutral'
    bos = []
    for i in range(1, min(len(sh),10)):
        if sh[i][1] > sh[i-1][1]:
            bos.append({'type':'BOS','direction':'up','price':sh[i][1],'date':sh[i][0]})
    for i in range(1, min(len(sl),10)):
        if sl[i][1] < sl[i-1][1]:
            bos.append({'type':'BOS','direction':'down','price':sl[i][1],'date':sl[i][0]})
    choch = []
    if len(sh)>=3:
        for i in range(2, len(sh)):
            if sh[i][1] > sh[i-1][1] and sh[i-1][1] < sh[i-2][1]:
                choch.append({'type':'CHoCH','direction':'up','price':sh[i][1],'date':sh[i][0]})
    if len(sl)>=3:
        for i in range(2, len(sl)):
            if sl[i][1] < sl[i-1][1] and sl[i-1][1] > sl[i-2][1]:
                choch.append({'type':'CHoCH','direction':'down','price':sl[i][1],'date':sl[i][0]})
    bos = bos[-3:]
    choch = choch[-3:]
    all_events = sorted(bos+choch, key=lambda x: x['date'], reverse=True)
    last = all_events[0] if all_events else None
    desc = f"Trend: {primary}. Last: {last['type']} {last['direction'] if last else ''}"
    return {'trend':primary,'last_event':last['type'] if last else None,'last_event_direction':last['direction'] if last else None,'bos_events':bos,'choch_events':choch,'description':desc}

def generate_structure_chart(df, symbol, structure):
    if len(df) < 50:
        return None
    chart_data = df[['open','high','low','close']].tail(100).copy()
    chart_data.columns = ['Open','High','Low','Close']
    sh, sl = find_swings(df)
    fig, ax = plt.subplots(figsize=(14,8), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.grid(True, color='#444', linestyle='--', alpha=0.5)
    dates = chart_data.index
    width = 0.6 * (dates[1]-dates[0]).total_seconds()/(24*3600) if len(dates)>1 else 0.5
    for idx, row in chart_data.iterrows():
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        ax.bar(idx, row['High']-row['Low'], bottom=row['Low'], width=width, color=color, alpha=0.5)
        ax.bar(idx, row['Close']-row['Open'], bottom=row['Open'], width=width, color=color, alpha=1.0)
    for p in sh:
        if p[0] in chart_data.index:
            ax.plot(p[0], p[1], '^', color='lime', markersize=10)
    for p in sl:
        if p[0] in chart_data.index:
            ax.plot(p[0], p[1], 'v', color='red', markersize=10)
    for e in structure.get('bos_events',[]):
        if e['date'] in chart_data.index:
            col = '#00aaff' if e['direction']=='up' else '#ff8800'
            ax.axhline(y=e['price'], color=col, linestyle='--', lw=1.5, alpha=0.7)
            ax.text(e['date'], e['price'], f"BOS {e['direction'].upper()}", fontsize=8, color=col, ha='left', va='bottom')
    for e in structure.get('choch_events',[]):
        if e['date'] in chart_data.index:
            col = '#ff00ff' if e['direction']=='up' else '#ff4444'
            ax.axhline(y=e['price'], color=col, linestyle='--', lw=2, alpha=0.8)
            ax.text(e['date'], e['price'], f"CHoCH {e['direction'].upper()}", fontsize=9, color=col, ha='left', va='top')
    all_events = structure.get('bos_events',[]) + structure.get('choch_events',[])
    if all_events:
        last = sorted(all_events, key=lambda x: x['date'], reverse=True)[0]
        if last['date'] in chart_data.index:
            ax.axhline(y=last['price'], color='white', lw=3, alpha=0.9)
            ax.text(last['date'], last['price'], f"★ MOST RECENT: {last['type']} {last['direction'].upper()}", fontsize=10, color='yellow', ha='left', va='bottom')
    ax.legend(loc='upper left', fontsize=9, facecolor='#333', edgecolor='white', labelcolor='white')
    ax.set_title(f'{symbol} Market Structure', color='white')
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Price', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, format='png', dpi=120, facecolor='#1e1e1e', bbox_inches='tight')
        tmp.flush()
        with open(tmp.name, 'rb') as f:
            img = f.read()
    os.unlink(tmp.name)
    plt.close(fig)
    return io.BytesIO(img)

# ====================
# COMMAND: !structure
# ====================
@bot.command(name='structure')
async def market_structure(ctx, ticker: str, timeframe: str = 'daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        if last_command_time.get(ctx.author.id) and (now - last_command_time[ctx.author.id]) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        if ticker.lower() == 'all':
            watchlist = await load_watchlist()
            symbols = watchlist['stocks']
            if not symbols:
                await ctx.send("No stocks in watchlist.")
                return
            if timeframe not in ['1h','4h','daily','weekly']:
                await ctx.send("Use 1h,4h,daily,weekly")
                return
            await ctx.send(f"Scanning {len(symbols)} symbols for structure on {timeframe}...")
            bos_up, choch_up, bos_down, choch_down = [], [], [], []
            for sym in symbols:
                if await check_cancel(ctx):
                    break
                df = await fetch_ohlcv(sym, timeframe)
                if df is None or len(df) < 50:
                    continue
                df = calculate_indicators(df)
                struct = analyze_structure(df)
                price = df['close'].iloc[-1]
                if struct['last_event'] == 'BOS':
                    if struct['last_event_direction'] == 'up':
                        bos_up.append(f"{sym} (${price:.2f}) – {struct['trend']}")
                    else:
                        bos_down.append(f"{sym} (${price:.2f}) – {struct['trend']}")
                elif struct['last_event'] == 'CHoCH':
                    if struct['last_event_direction'] == 'up':
                        choch_up.append(f"{sym} (${price:.2f}) – {struct['trend']}")
                    else:
                        choch_down.append(f"{sym} (${price:.2f}) – {struct['trend']}")
                await asyncio.sleep(0.5)
            embed = discord.Embed(title=f"Structure Scan – {timeframe}", color=0x3498db)
            if bos_up: embed.add_field(name="🟢 BOS UP (uptrend continuing)", value="\n".join(bos_up[:5]), inline=False)
            if choch_up: embed.add_field(name="🟠 CHoCH UP (reversal up)", value="\n".join(choch_up[:5]), inline=False)
            if bos_down: embed.add_field(name="🔴 BOS DOWN (downtrend continuing)", value="\n".join(bos_down[:5]), inline=False)
            if choch_down: embed.add_field(name="🟣 CHoCH DOWN (reversal down)", value="\n".join(choch_down[:5]), inline=False)
            await ctx.send(embed=embed)
            return
        symbol = normalize_symbol(ticker)
        if '/' in symbol:
            await ctx.send("Only stocks.")
            return
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send("No data.")
            return
        df = calculate_indicators(df)
        struct = analyze_structure(df)
        price = df['close'].iloc[-1]
        trend = struct['trend']
        last_event = struct['last_event']
        last_dir = struct['last_event_direction']
        if trend == 'bullish' and last_event == 'BOS' and last_dir == 'up':
            action, color = "✅ HOLD/ADD CALLS – Uptrend continuing", 0x00cc00
        elif trend == 'bullish' and last_event == 'CHoCH' and last_dir == 'up':
            action, color = "✅ BUY CALLS – Reversal to uptrend", 0x00ff00
        elif trend == 'bearish' and last_event == 'BOS' and last_dir == 'down':
            action, color = "🔴 BUY PUTS – Downtrend continuing", 0xcc0000
        elif trend == 'bearish' and last_event == 'CHoCH' and last_dir == 'down':
            action, color = "🔴 BUY PUTS – Reversal to downtrend", 0xff0000
        else:
            action, color = "⏸️ WAIT – No clear signal", 0xffff00
        embed = discord.Embed(title=f"Market Structure: {symbol} ({timeframe})", description=f"Price: ${price:.2f}\n\n{action}", color=color)
        embed.add_field(name="Overall Trend", value=trend.capitalize(), inline=True)
        if struct['last_event']:
            embed.add_field(name="Last Event", value=f"{struct['last_event']} {struct['last_event_direction']}", inline=True)
        if struct['bos_events']:
            txt = "\n".join([f"{e['direction'].upper()} at ${e['price']:.2f} ({e['date'].strftime('%m/%d')})" for e in struct['bos_events'][-3:]])
            embed.add_field(name="Recent BOS", value=txt, inline=False)
        if struct['choch_events']:
            txt = "\n".join([f"{e['direction'].upper()} at ${e['price']:.2f} ({e['date'].strftime('%m/%d')})" for e in struct['choch_events'][-3:]])
            embed.add_field(name="Recent CHoCH", value=txt, inline=False)
        embed.add_field(name="Analysis", value=struct['description'], inline=False)
        embed.add_field(name="TradingView", value=f"[Chart]({get_tradingview_web_link(symbol)})", inline=False)
        chart = generate_structure_chart(df, symbol, struct)
        if chart:
            file = discord.File(chart, filename='struct.png')
            embed.set_image(url='attachment://struct.png')
            await ctx.send(embed=embed, file=file)
        else:
            await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !confirm (REWRITTEN)
# ====================
def get_unified_trend(df):
    if len(df) < 200:
        return "neutral"
    price = df['close'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    ema200 = df['ema200'].iloc[-1]
    if price > ema50 and ema50 > ema200:
        return "bullish"
    elif price < ema50 and ema50 < ema200:
        return "bearish"
    return "neutral"

async def fetch_earnings_upcoming(symbol, days=7):
    try:
        stock = yf.Ticker(symbol)
        earnings = stock.earnings_dates
        if earnings is None or earnings.empty:
            return []
        today = datetime.now().date()
        upcoming = []
        for date, row in earnings.iterrows():
            edate = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
            if edate >= today and (edate - today).days <= days:
                upcoming.append({'date': edate.strftime('%Y-%m-%d')})
        return upcoming
    except:
        return []

def get_key_expirations(ticker):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return []
        today = datetime.now().date()
        exp_dates = [datetime.strptime(e, '%Y-%m-%d').date() for e in expirations]
        key = []
        for ed, es in zip(exp_dates, expirations):
            dte = (ed - today).days
            if 0 <= dte <= 7:
                key.append((es, dte, "WEEKLY"))
                break
        for ed, es in zip(exp_dates, expirations):
            dte = (ed - today).days
            if 8 <= dte <= 21:
                key.append((es, dte, "MONTHLY"))
                break
        best_exp, best_dte, min_diff = None, None, float('inf')
        for ed, es in zip(exp_dates, expirations):
            dte = (ed - today).days
            diff = abs(dte - 38)
            if diff < min_diff:
                min_diff, best_exp, best_dte = diff, es, dte
        if best_exp:
            key.append((best_exp, best_dte, "PRIMARY (30-45 DTE)"))
        return key
    except:
        return []

def find_demand_zones(df, lookback=200, threshold_percentile=90, touch_tolerance=0.005):
    if len(df) < 50:
        return []
    df = df.iloc[-lookback:].copy()
    df['range'] = df['high'] - df['low']
    thresh = np.percentile(df['range'].dropna(), threshold_percentile)
    large = df[df['range'] > thresh]
    zones = []
    for idx, row in large.iterrows():
        level = row['low']
        after = df.loc[idx:]
        if len(after) < 2:
            continue
        if (after['close'] < level * (1 - touch_tolerance)).any():
            continue
        touches = after['low'] <= level * (1 + touch_tolerance)
        strength = int(touches.sum())
        if strength >= 1:
            zones.append({'level': level, 'date': idx, 'strength': strength})
    zones.sort(key=lambda x: x['level'])
    return zones

@bot.command(name='confirm')
async def trade_confirmation(ctx, ticker: str, timeframe: str = 'daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        if last_command_time.get(ctx.author.id) and (now - last_command_time[ctx.author.id]) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        symbol = normalize_symbol(ticker)
        if '/' in symbol:
            await ctx.send("Only stocks.")
            return
        if timeframe not in ['1h','4h','daily','weekly']:
            await ctx.send("Use 1h,4h,daily,weekly")
            return
        await ctx.send(f"Analyzing {symbol} ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send("No data.")
            return
        df = calculate_indicators(df)
        price = df['close'].iloc[-1]
        struct = analyze_structure(df)
        rsi = df['rsi'].iloc[-1]
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1
        ema13, ema50, ema200 = df['ema13'].iloc[-1], df['ema50'].iloc[-1], df['ema200'].iloc[-1]
        above_ema13 = price > ema13
        above_ema50 = price > ema50
        above_ema200 = price > ema200

        earnings = await fetch_earnings_upcoming(symbol, days=7)
        earnings_soon = len(earnings) > 0

        spy_df = await fetch_ohlcv('SPY', 'daily')
        spy_trend = "bullish" if spy_df is not None and spy_df['close'].iloc[-1] > spy_df['ema50'].iloc[-1] else "bearish"

        # Scoring
        score = 0
        reasons_bull = []
        reasons_bear = []

        if struct['last_event'] == 'CHoCH' and struct['last_event_direction'] == 'up':
            score += 30; reasons_bull.append("CHoCH UP (bullish reversal)")
        elif struct['last_event'] == 'BOS' and struct['last_event_direction'] == 'up':
            score += 25; reasons_bull.append("BOS UP (uptrend continuing)")
        elif struct['last_event'] == 'CHoCH' and struct['last_event_direction'] == 'down':
            score -= 25; reasons_bear.append("CHoCH DOWN (bearish reversal)")
        elif struct['last_event'] == 'BOS' and struct['last_event_direction'] == 'down':
            score -= 20; reasons_bear.append("BOS DOWN (downtrend continuing)")

        if adx > 25:
            score += 15; reasons_bull.append(f"Strong trend ADX={adx:.1f}")
        elif adx > 20:
            score += 8; reasons_bull.append(f"Moderate trend ADX={adx:.1f}")
        else:
            score -= 5; reasons_bear.append(f"Weak trend ADX={adx:.1f}")

        if 30 <= rsi <= 70:
            score += 10; reasons_bull.append(f"RSI neutral {rsi:.1f}")
        elif rsi < 30:
            score += 8; reasons_bull.append(f"RSI oversold {rsi:.1f}")
        else:
            score -= 8; reasons_bear.append(f"RSI overbought {rsi:.1f}")

        ema_pts = 0
        if above_ema13: ema_pts += 3
        if above_ema50: ema_pts += 3
        if above_ema200: ema_pts += 4
        if ema13 > ema50 > ema200:
            ema_pts += 5; reasons_bull.append("Perfect EMA stack")
        elif ema13 < ema50 < ema200:
            ema_pts -= 5; reasons_bear.append("Bearish EMA stack")
        score += ema_pts

        if vol_ratio > 1.5:
            score += 10; reasons_bull.append(f"High volume {vol_ratio:.1f}x")
        elif vol_ratio > 1.2:
            score += 6; reasons_bull.append(f"Above avg volume {vol_ratio:.1f}x")
        elif vol_ratio < 0.7:
            score -= 5; reasons_bear.append(f"Low volume {vol_ratio:.1f}x")

        df_30 = await fetch_ohlcv(symbol, '30min')
        zones = find_demand_zones(df_30) if df_30 is not None else []
        near_zone = any(abs((price - z['level'])/price) < 0.02 for z in zones)
        if near_zone:
            score += 5; reasons_bull.append("Near demand zone")

        # Options flow check
        unusual = False
        try:
            exps = get_key_expirations(symbol)
            if exps:
                stock = yf.Ticker(symbol)
                chain = stock.option_chain(exps[0][0])
                calls = chain.calls
                if not calls.empty:
                    for _, opt in calls.iterrows():
                        vol = opt.get('volume',0)
                        oi = opt.get('openInterest',0)
                        if oi>0 and vol/oi > 1.5:
                            unusual = True
                            break
        except:
            pass
        if unusual:
            score += 10; reasons_bull.append("Unusual call volume")

        if earnings_soon:
            score -= 15; reasons_bear.append("Earnings in <7 days")

        if spy_trend == 'bullish':
            score += 5; reasons_bull.append("SPY uptrend")
        else:
            score -= 5; reasons_bear.append("SPY downtrend")

        score = max(0, min(100, score))
        trend = get_unified_trend(df)

        if score >= 65 and trend == 'bullish':
            rec, col, action = "🟢 HIGH PROBABILITY BUY", 0x00ff00, "BUY CALLS (or stock)"
            sl = price * 0.95
            tp = price * 1.10
        elif score >= 50 and trend == 'bullish':
            rec, col, action = "🟡 MODERATE BUY", 0xffff00, "Consider CALLS with smaller size"
            sl = price * 0.93
            tp = price * 1.07
        elif score >= 65 and trend == 'bearish':
            rec, col, action = "🔴 HIGH PROBABILITY SELL / PUTS", 0xff0000, "BUY PUTS"
            sl = price * 1.05
            tp = price * 0.90
        elif score >= 50 and trend == 'bearish':
            rec, col, action = "🟠 MODERATE SELL / PUTS", 0xff8800, "Consider PUTS with smaller size"
            sl = price * 1.03
            tp = price * 0.93
        else:
            rec, col, action = "⚪ NEUTRAL – WAIT", 0x888888, "No clear edge"
            sl = tp = None

        embed = discord.Embed(title=f"Trade Confirmation: {symbol} ({timeframe})", description=f"Price: ${price:.2f}\nScore: {score}/100\n{rec}", color=col)
        if reasons_bull:
            embed.add_field(name="🟢 Bullish Factors", value="\n".join(reasons_bull[:8]), inline=False)
        if reasons_bear:
            embed.add_field(name="🔴 Bearish Factors", value="\n".join(reasons_bear[:5]), inline=False)
        if sl and tp:
            embed.add_field(name="Suggested Parameters", value=f"Action: {action}\nStop: ${sl:.2f} ({((sl-price)/price*100):+.1f}%)\nTarget: ${tp:.2f} ({((tp-price)/price*100):+.1f}%)\nR/R: {abs((tp-price)/(sl-price)):.2f}", inline=False)
        else:
            embed.add_field(name="Suggested Action", value=action, inline=False)
        embed.add_field(name="Summary", value=f"Trend: {trend}\nADX: {adx:.1f} | RSI: {rsi:.1f}\nVolume: {vol_ratio:.1f}x avg\nEMAs: {'Bullish' if above_ema13 and above_ema50 else 'Mixed/Bearish'}", inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !strength (kept simple)
# ====================
@bot.command(name='strength')
async def trend_strength(ctx, ticker: str, timeframe: str = 'daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        symbol = normalize_symbol(ticker)
        if '/' in symbol:
            await ctx.send("Crypto not supported.")
            return
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send("No data.")
            return
        df = calculate_indicators(df)
        price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
        avg_vol = df['volume'].tail(20).mean()
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1
        embed = discord.Embed(title=f"Trend Strength: {symbol} ({timeframe})", description=f"Price: ${price:.2f}", color=0x3498db)
        embed.add_field(name="ADX", value=f"{'Strong' if adx>25 else 'Moderate' if adx>20 else 'Weak'} ({adx:.1f})", inline=True)
        embed.add_field(name="RSI", value=f"{'Oversold' if rsi<30 else 'Overbought' if rsi>70 else 'Neutral'} ({rsi:.1f})", inline=True)
        embed.add_field(name="Volume", value=f"{'High' if vol_ratio>1.5 else 'Low' if vol_ratio<0.5 else 'Normal'} ({vol_ratio:.1f}x)", inline=True)
        embed.add_field(name="Price vs EMA13", value="Above" if price>df['ema13'].iloc[-1] else "Below", inline=True)
        embed.add_field(name="Price vs EMA50", value="Above" if price>df['ema50'].iloc[-1] else "Below", inline=True)
        embed.add_field(name="Price vs EMA200", value="Above" if price>df['ema200'].iloc[-1] else "Below", inline=True)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !zone (unchanged but uses fixed fetch)
# ====================
@bot.command(name='zone')
async def zone(ctx, ticker: str, timeframe: str = '30min'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        if last_command_time.get(ctx.author.id) and (now - last_command_time[ctx.author.id]) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        timeframe = timeframe.lower()
        if timeframe not in ['5min','15min','30min','1h','4h','daily','weekly']:
            await ctx.send("Invalid timeframe.")
            return
        symbol = normalize_symbol(ticker)
        await ctx.send(f"Fetching zones for {symbol} ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send("No data.")
            return
        df = calculate_indicators(df)
        signals = get_signals(df)
        price = signals['price']
        support = signals['support_20']
        resistance = signals['resistance_20']
        ema5, ema13, ema50, ema200 = signals['ema5'], signals['ema13'], signals['ema50'], signals['ema200']
        supports = [support]
        resistances = [resistance]
        if not pd.isna(ema200):
            (supports if ema200 < price else resistances).append(ema200)
        if not pd.isna(ema50):
            (supports if ema50 < price else resistances).append(ema50)
        if not pd.isna(ema13):
            (supports if ema13 < price else resistances).append(ema13)
        if not pd.isna(ema5):
            (supports if ema5 < price else resistances).append(ema5)
        supports.sort(reverse=True)
        resistances.sort()
        embed = discord.Embed(title=f"Zones: {symbol} ({timeframe})", description=f"Price: ${price:.2f}", color=0x00ff00 if signals['net_score']>0 else 0xff0000 if signals['net_score']<0 else 0xffff00)
        sup_txt = "\n".join([f"**Primary Support:** ${l:.2f}" if i==0 else f"Secondary Support: ${l:.2f}" for i,l in enumerate(supports[:3])])
        if sup_txt:
            embed.add_field(name="📉 Support (Buy Zone)", value=sup_txt, inline=False)
        res_txt = "\n".join([f"**Primary Resistance:** ${l:.2f}" if i==0 else f"Secondary Resistance: ${l:.2f}" for i,l in enumerate(resistances[:3])])
        if res_txt:
            embed.add_field(name="📈 Resistance (Sell Zone)", value=res_txt, inline=False)
        target = resistance + (resistance - support)
        embed.add_field(name="🎯 Target", value=f"${target:.2f}", inline=False)
        embed.add_field(name="TradingView", value=f"[Chart]({get_tradingview_web_link(symbol)})", inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !flow and !scanflow (unchanged but with yfinance price)
# ====================
async def get_stock_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else None
    except:
        return None

def format_premium(volume, last_price):
    try:
        prem = volume * 100 * last_price
        if prem >= 1_000_000:
            return f"${prem/1_000_000:.1f}M"
        elif prem >= 1_000:
            return f"${prem/1_000:.0f}K"
        else:
            return f"${prem:.0f}"
    except:
        return "N/A"

def get_whale_emoji(premium):
    if premium >= 1_000_000:
        return "🐋🐋"
    elif premium >= 100_000:
        return "🐋"
    elif premium >= 10_000:
        return "🐬"
    else:
        return "🐟"

def analyze_expiration(opt_chain, current_price, dte, min_volume=5):
    if opt_chain.calls.empty and opt_chain.puts.empty:
        return []
    calls, puts = opt_chain.calls.copy(), opt_chain.puts.copy()
    if not calls.empty:
        calls['type'] = 'CALL'
    if not puts.empty:
        puts['type'] = 'PUT'
    all_opt = pd.concat([calls, puts], ignore_index=True)
    analyzed = []
    for _, opt in all_opt.iterrows():
        try:
            vol = opt.get('volume', 0)
            oi = opt.get('openInterest', 0)
            if pd.isna(vol) or pd.isna(oi) or vol < min_volume or oi == 0:
                continue
            strike = opt.get('strike', 0)
            last = opt.get('lastPrice', 0)
            opt_type = opt.get('type', 'CALL')
            ratio = vol / oi
            prem = vol * 100 * last
            dist = abs(strike - current_price) / current_price * 100
            analyzed.append({'strike': strike, 'type': opt_type, 'volume': int(vol), 'oi': int(oi), 'vol_oi_ratio': ratio, 'last': last, 'premium': format_premium(vol, last), 'raw_premium': prem, 'distance_pct': dist, 'dte': dte})
        except:
            continue
    return analyzed

def add_field_safe(embed, name, value, inline=False):
    if len(value) <= 1024:
        embed.add_field(name=name, value=value, inline=inline)
    else:
        for i, chunk in enumerate([value[i:i+1024] for i in range(0, len(value), 1024)]):
            embed.add_field(name=f"{name} (cont.)" if i else name, value=chunk, inline=inline)

@bot.command(name='flow')
async def options_flow(ctx, ticker: str):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send(f"Analyzing options flow for {ticker.upper()}...")
        price = await get_stock_price(ticker.upper())
        if not price:
            await ctx.send("Could not fetch price.")
            return
        exps = get_key_expirations(ticker.upper())
        if not exps:
            await ctx.send("No options.")
            return
        stock = yf.Ticker(ticker.upper())
        embed = discord.Embed(title=f"Options Flow: {ticker.upper()}", description=f"Price: ${price:.2f}", color=0x00ff00)
        high_prob, lottery = [], []
        for exp_str, dte, label in exps:
            chain = stock.option_chain(exp_str)
            analyzed = analyze_expiration(chain, price, dte)
            analyzed.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
            for opt in analyzed[:10]:
                if 30 <= dte <= 45 and opt['distance_pct'] <= 20:
                    high_prob.append((label, exp_str, dte, opt))
                else:
                    lottery.append((label, exp_str, dte, opt))
        hp_txt = ""
        for label, exp_str, dte, opt in high_prob[:6]:
            whale = get_whale_emoji(opt['raw_premium'])
            hp_txt += f"**${opt['strike']:.2f} {opt['type']}** ({label})\n  • Vol: {opt['volume']} ({opt['vol_oi_ratio']:.1f}x) Premium: {opt['premium']} {whale}\n  • DTE: {dte} Distance: {opt['distance_pct']:.1f}%\n\n"
        if hp_txt:
            add_field_safe(embed, "High Probability Setups (30-45 DTE, near money)", hp_txt, inline=False)
        lt_txt = ""
        for label, exp_str, dte, opt in lottery[:10]:
            whale = get_whale_emoji(opt['raw_premium'])
            lt_txt += f"**${opt['strike']:.2f} {opt['type']}** ({label})\n  • Vol: {opt['volume']} ({opt['vol_oi_ratio']:.1f}x) Premium: {opt['premium']} {whale}\n  • DTE: {dte}\n\n"
        if lt_txt:
            add_field_safe(embed, "Other Activity", lt_txt, inline=False)
        if not high_prob and not lottery:
            await ctx.send("No significant activity.")
            return
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='scanflow')
async def scan_options_flow(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        watchlist = await load_watchlist()
        symbols = watchlist['stocks']
        if not symbols:
            await ctx.send("No stocks.")
            return
        await ctx.send(f"Scanning {len(symbols)} symbols for unusual options flow...")
        all_flow = []
        for sym in symbols:
            if await check_cancel(ctx):
                break
            try:
                price = await get_stock_price(sym)
                if not price:
                    continue
                exps = get_key_expirations(sym)
                stock = yf.Ticker(sym)
                for exp_str, dte, label in exps:
                    if "WEEKLY" in label or "PRIMARY" in label:
                        chain = stock.option_chain(exp_str)
                        calls = chain.calls
                        if calls.empty:
                            continue
                        for _, opt in calls.iterrows():
                            vol = opt.get('volume',0)
                            oi = opt.get('openInterest',0)
                            if pd.isna(vol) or pd.isna(oi) or oi==0 or vol<10:
                                continue
                            ratio = vol/oi
                            if ratio >= 1.5:
                                strike = opt.get('strike',0)
                                last = opt.get('lastPrice',0)
                                dist = abs(strike - price)/price*100
                                if dist <= 20:
                                    prem = vol * 100 * last
                                    all_flow.append({'symbol':sym, 'strike':strike, 'dte':dte, 'volume':vol, 'oi':oi, 'ratio':ratio, 'premium':format_premium(vol, last), 'raw_premium':prem, 'distance':dist, 'label':label})
                await asyncio.sleep(1)
            except:
                continue
        high = [f for f in all_flow if 30 <= f['dte'] <= 45 and f['distance'] <= 20]
        low = [f for f in all_flow if f not in high]
        high.sort(key=lambda x: x['raw_premium'], reverse=True)
        low.sort(key=lambda x: x['raw_premium'], reverse=True)
        if not all_flow:
            await ctx.send("No unusual flow found.")
            return
        embed = discord.Embed(title="Unusual Options Flow Scan", color=0x00ff00)
        txt = ""
        for f in high[:8]:
            whale = get_whale_emoji(f['raw_premium'])
            txt += f"**{f['symbol']} ${f['strike']:.2f} CALL** ({f['label']})\n  • Vol: {f['volume']} ({f['ratio']:.1f}x) Premium: {f['premium']} {whale}\n  • DTE: {f['dte']} Distance: {f['distance']:.1f}%\n\n"
        if txt:
            add_field_safe(embed, "High Probability Setups (30-45 DTE)", txt, inline=False)
        txt2 = ""
        for f in low[:12]:
            whale = get_whale_emoji(f['raw_premium'])
            txt2 += f"**{f['symbol']} ${f['strike']:.2f} CALL** ({f['label']})\n  • Vol: {f['volume']} ({f['ratio']:.1f}x) Premium: {f['premium']} {whale}\n  • DTE: {f['dte']}\n\n"
        if txt2:
            add_field_safe(embed, "Other Activity", txt2, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !upcoming (unchanged)
# ====================
@bot.command(name='upcoming')
async def upcoming_events(ctx, ticker: str):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        symbol = ticker.upper()
        await ctx.send(f"Fetching upcoming events for {symbol}...")
        stock = yf.Ticker(symbol)
        earnings = []
        try:
            edates = stock.earnings_dates
            if edates is not None and not edates.empty:
                today = datetime.now().date()
                for date, row in edates.iterrows():
                    e_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    if e_date >= today:
                        eps = row.get('epsEstimated') if 'epsEstimated' in row else row.get('epsEstimate')
                        eps = 'N/A' if eps is None or pd.isna(eps) else f"{eps:.2f}"
                        earnings.append({'date': e_date.strftime('%Y-%m-%d'), 'eps': eps})
        except:
            pass
        dividends = []
        try:
            divs = stock.dividends
            if not divs.empty:
                today = datetime.now().date()
                for date, amt in divs.items():
                    d_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    if d_date >= today:
                        dividends.append({'date': d_date.strftime('%Y-%m-%d'), 'amount': f"${amt:.2f}"})
        except:
            pass
        splits = []
        try:
            sp = stock.splits
            if not sp.empty:
                today = datetime.now().date()
                for date, ratio in sp.items():
                    s_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    if s_date >= today:
                        ratio_str = f"{ratio:.0f}:1" if ratio >= 1 else f"1:{int(1/ratio)}"
                        splits.append({'date': s_date.strftime('%Y-%m-%d'), 'ratio': ratio_str})
        except:
            pass
        if not earnings and not dividends and not splits:
            await ctx.send("No upcoming events.")
            return
        embed = discord.Embed(title=f"Upcoming: {symbol}", color=0x00ff00)
        if earnings:
            txt = "\n".join([f"**{e['date']}** – EPS Est: {e['eps']}" for e in earnings[:3]])
            embed.add_field(name="Earnings", value=txt, inline=False)
        if dividends:
            txt = "\n".join([f"**{d['date']}** – {d['amount']}" for d in dividends[:2]])
            embed.add_field(name="Dividends", value=txt, inline=False)
        if splits:
            txt = "\n".join([f"**{s['date']}** – {s['ratio']}" for s in splits[:2]])
            embed.add_field(name="Stock Splits", value=txt, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !backtest (unchanged)
# ====================
@bot.command(name='backtest')
async def backtest(ctx, symbol: str, days: int = 365, cost: float = 0.001):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send(f"Backtesting {symbol.upper()} over {days} days...")
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=f"{days}d", interval="1d")
        if df.empty:
            await ctx.send("No data.")
            return
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
        df = calculate_indicators(df)
        signals = [get_signals(df.iloc[:i+1]) for i in range(len(df)) if get_signals(df.iloc[:i+1])]
        equity, pos, entry_price, trades = [1.0], False, 0, []
        for i in range(1, len(signals)):
            sig, prev_sig = signals[i], signals[i-1]
            open_today = df['open'].iloc[i] if i < len(df) else df['close'].iloc[-1]
            if not pos:
                if prev_sig['net_score'] > 0:
                    pos, entry_price = True, open_today
                    equity[-1] *= (1 - cost)
                elif prev_sig['net_score'] < 0:
                    pos, entry_price = True, open_today
                    equity[-1] *= (1 - cost)
            else:
                if (prev_sig['net_score'] < 0) or (i == len(signals)-1):
                    exit_price = open_today if i < len(df) else df['close'].iloc[-1]
                    ret = (exit_price - entry_price) / entry_price
                    equity.append(equity[-1] * (1+ret) * (1-cost))
                    pos = False
        if len(equity) < 2:
            await ctx.send("No trades.")
            return
        ret = (equity[-1]-1)*100
        embed = discord.Embed(title=f"Backtest {symbol.upper()}", description=f"Return: {ret:.2f}%", color=0x00ff00 if ret>0 else 0xff0000)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !verify (fixed with lowercase '4h')
# ====================
@bot.command(name='verify')
async def verify_data(ctx, symbol: str, timeframe: str = 'daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        symbol = symbol.upper()
        await ctx.send(f"Verifying {symbol} on {timeframe} against live market...")
        df_bot = await fetch_ohlcv(symbol, timeframe)
        if df_bot is None or df_bot.empty:
            await ctx.send("Bot has no data.")
            return
        ticker = yf.Ticker(symbol)
        if timeframe == '4h':
            data = ticker.history(period="10d", interval="1h")
            if not data.empty:
                df_live = data.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            else:
                df_live = None
        elif timeframe == '1h':
            df_live = ticker.history(period="5d", interval="1h")
        elif timeframe == 'daily':
            df_live = ticker.history(period="10d")
        else:
            await ctx.send("Only 1h,4h,daily.")
            return
        if df_live is None or df_live.empty:
            await ctx.send("Live data empty.")
            return
        bot_price = df_bot['close'].iloc[-1]
        live_price = df_live['Close'].iloc[-1]
        diff = abs(bot_price - live_price)/live_price*100
        bot_vol = df_bot['volume'].iloc[-1] if 'volume' in df_bot.columns else 0
        live_vol = df_live['Volume'].iloc[-1] if 'Volume' in df_live.columns else 0
        embed = discord.Embed(title=f"Verify {symbol} ({timeframe})", color=0x3498db)
        if diff < 0.5:
            embed.add_field(name="✅ Price Match", value=f"Bot: ${bot_price:.2f} | Live: ${live_price:.2f} | Diff: {diff:.2f}%", inline=False)
        else:
            embed.add_field(name="⚠️ Price Mismatch", value=f"Bot: ${bot_price:.2f} | Live: ${live_price:.2f} | Diff: {diff:.2f}%", inline=False)
        if live_vol > 0:
            match = "✅ Volume OK" if abs(bot_vol - live_vol)/max(live_vol,1) < 0.2 else "⚠️ Volume Differs"
            embed.add_field(name=match, value=f"Bot Vol: {bot_vol:,.0f} | Live Vol: {live_vol:,.0f}", inline=False)
        else:
            embed.add_field(name="❌ Volume Missing", value="Live vol zero", inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !score (quick score, uses unified scoring)
# ====================
@bot.command(name='score')
async def quick_score(ctx, target: str = None, timeframe: str = 'daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        if target and target.lower() != 'all':
            symbol = normalize_symbol(target)
            if '/' in symbol:
                await ctx.send("Only stocks.")
                return
            df = await fetch_ohlcv(symbol, timeframe)
            if df is None or df.empty:
                await ctx.send("No data.")
                return
            df = calculate_indicators(df)
            score, details = await compute_quick_score(df)
            embed = discord.Embed(title=f"Score: {symbol} ({timeframe})", description=f"Score: {score}/100 – {get_score_rating(score)}", color=get_score_color(score))
            embed.add_field(name="Breakdown", value=details, inline=False)
            await ctx.send(embed=embed)
            return
        watchlist = await load_watchlist()
        symbols = watchlist['stocks']
        if not symbols:
            await ctx.send("No stocks.")
            return
        await ctx.send(f"Scanning {len(symbols)} symbols on {timeframe}...")
        results = []
        for sym in symbols:
            if await check_cancel(ctx):
                break
            df = await fetch_ohlcv(sym, timeframe)
            if df is None or len(df) < 30:
                continue
            df = calculate_indicators(df)
            score, _ = await compute_quick_score(df)
            price = df['close'].iloc[-1]
            results.append({'symbol': sym, 'price': price, 'score': score, 'rating': get_score_rating(score)})
            await asyncio.sleep(0.3)
        if not results:
            await ctx.send("No results.")
            return
        results.sort(key=lambda x: x['score'], reverse=True)
        embed = discord.Embed(title=f"Quick Score – {timeframe}", color=0x3498db)
        txt = ""
        for r in results[:15]:
            emoji = "🟢" if r['score']>=70 else "🟡" if r['score']>=50 else "🔴"
            txt += f"{emoji} **{r['symbol']}** ${r['price']:.2f} – {r['score']}/100 ({r['rating']})\n"
        embed.add_field(name="Top Results", value=txt, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

async def compute_quick_score(df):
    price = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
    ema13, ema50, ema200 = df['ema13'].iloc[-1], df['ema50'].iloc[-1], df['ema200'].iloc[-1]
    above_ema13 = price > ema13
    above_ema50 = price > ema50
    above_ema200 = price > ema200
    vol_ratio = df['volume'].iloc[-1] / df['volume'].tail(20).mean() if df['volume'].tail(20).mean() > 0 else 1
    struct = analyze_structure(df)
    score = 0
    details = []
    # Structure
    if struct['last_event'] == 'CHoCH' and struct['last_event_direction'] == 'up':
        score += 25; details.append("CHoCH UP +25")
    elif struct['last_event'] == 'BOS' and struct['last_event_direction'] == 'up':
        score += 20; details.append("BOS UP +20")
    elif struct['last_event'] == 'CHoCH' and struct['last_event_direction'] == 'down':
        score -= 25; details.append("CHoCH DOWN -25")
    elif struct['last_event'] == 'BOS' and struct['last_event_direction'] == 'down':
        score -= 20; details.append("BOS DOWN -20")
    else:
        details.append("No structure signal")
    # ADX
    if adx > 25:
        score += 15; details.append(f"ADX {adx:.1f} +15")
    elif adx > 20:
        score += 8; details.append(f"ADX {adx:.1f} +8")
    else:
        score -= 5; details.append(f"ADX {adx:.1f} -5")
    # RSI
    if 30 <= rsi <= 70:
        score += 10; details.append(f"RSI {rsi:.1f} +10")
    elif rsi < 30:
        score += 8; details.append(f"RSI oversold +8")
    else:
        score -= 8; details.append(f"RSI overbought -8")
    # EMA
    ema_pts = 0
    if above_ema13: ema_pts += 3
    if above_ema50: ema_pts += 3
    if above_ema200: ema_pts += 4
    if ema13 > ema50 > ema200: ema_pts += 5
    score += ema_pts
    details.append(f"EMA position +{ema_pts}")
    # Volume
    if vol_ratio > 1.5:
        score += 10; details.append(f"High volume {vol_ratio:.1f}x +10")
    elif vol_ratio > 1.2:
        score += 6; details.append(f"Above avg volume {vol_ratio:.1f}x +6")
    elif vol_ratio > 0.8:
        score += 3; details.append(f"Normal volume +3")
    elif vol_ratio > 0.5:
        score -= 5; details.append(f"Low volume -5")
    else:
        score -= 10; details.append(f"Very low volume -10")
    score = max(0, min(100, score))
    return score, "\n".join(details)

def get_score_rating(score):
    if score >= 80: return "STRONG BUY"
    if score >= 65: return "BUY"
    if score >= 50: return "NEUTRAL (WAIT)"
    if score >= 35: return "WEAK / AVOID"
    return "STRONG AVOID"

def get_score_color(score):
    if score >= 65: return 0x00ff00
    if score >= 50: return 0xffff00
    return 0xff0000

# ====================
# COMMAND: !scan (original, uses fetch_ohlcv)
# ====================
@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        if last_command_time.get(ctx.author.id) and (now - last_command_time[ctx.author.id]) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        if timeframe not in ['5min','15min','30min','1h','4h','daily','weekly']:
            await ctx.send("Invalid timeframe.")
            return
        watchlist = await load_watchlist()
        symbols = watchlist['stocks'] + watchlist['crypto']
        if target.lower() != 'all':
            symbol = normalize_symbol(target)
            df = await fetch_ohlcv(symbol, timeframe)
            if df is None or df.empty:
                await ctx.send("No data.")
                return
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
            return
        await ctx.send(f"Scanning {len(symbols)} symbols on {timeframe}...")
        for sym in symbols:
            if await check_cancel(ctx):
                break
            df = await fetch_ohlcv(sym, timeframe)
            if df is not None and not df.empty:
                df = calculate_indicators(df)
                signals = get_signals(df)
                embed = format_embed(sym, signals, timeframe)
                chart = generate_chart_image(df, sym, timeframe)
                if chart:
                    file = discord.File(chart, filename='chart.png')
                    embed.set_image(url='attachment://chart.png')
                    await ctx.send(embed=embed, file=file)
                else:
                    await ctx.send(embed=embed)
            await asyncio.sleep(1)
        await ctx.send("Scan complete.")
    finally:
        user_busy[ctx.author.id] = False

def format_embed(symbol, signals, timeframe):
    if not signals:
        return discord.Embed(title="Error", description=f"No data for {symbol}", color=0xff0000)
    rating, color = get_rating(signals)
    vol_display = "N/A"
    if signals.get('volume') and signals.get('volume_avg') and signals['volume_avg'] > 0:
        ratio = signals['volume'] / signals['volume_avg']
        vol_display = f"{'High' if ratio>1.5 else 'Low' if ratio<0.5 else 'Normal'} ({ratio:.1f}x)"
    reasons = []
    if signals['ema5_cross_above_13']: reasons.append("EMA5 ↑ EMA13")
    if signals['ema13_cross_above_50']: reasons.append("EMA13 ↑ EMA50")
    if signals['ema5_cross_below_13']: reasons.append("EMA5 ↓ EMA13")
    if signals['ema13_cross_below_50']: reasons.append("EMA13 ↓ EMA50")
    if signals['oversold_triangle']: reasons.append("Oversold BB")
    if signals['overbought_triangle']: reasons.append("Overbought BB")
    if signals['price'] > signals['ema200'] and not pd.isna(signals['ema200']): reasons.append("Above 200 EMA")
    if not reasons: reasons.append("No signals")
    embed = discord.Embed(title=f"{rating}", description=f"{symbol} · ${signals['price']:.2f}", color=color)
    embed.add_field(name="RSI", value=f"{signals['rsi']:.1f}", inline=True)
    embed.add_field(name="Trend", value=signals['trend'], inline=True)
    embed.add_field(name="Volume", value=vol_display, inline=True)
    embed.add_field(name="Reason", value=" | ".join(reasons), inline=False)
    embed.add_field(name="Support", value=f"${signals['support_20']:.2f}", inline=True)
    embed.add_field(name="Resistance", value=f"${signals['resistance_20']:.2f}", inline=True)
    embed.add_field(name="TradingView", value=f"[Chart]({get_tradingview_web_link(symbol)})", inline=False)
    return embed

# ====================
# COMMAND: !signals and !signal (multi‑timeframe)
# ====================
@bot.command(name='signals')
async def signals(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        watchlist = await load_watchlist()
        symbols = watchlist['stocks'] + watchlist['crypto']
        timeframes = ['5min','15min','30min','1h','4h','daily','weekly']
        await ctx.send(f"Scanning {len(symbols)} symbols across all timeframes...")
        for sym in symbols:
            if await check_cancel(ctx):
                break
            found = False
            for tf in timeframes:
                df = await fetch_ohlcv(sym, tf)
                if df is not None and not df.empty:
                    df = calculate_indicators(df)
                    sig = get_signals(df)
                    if sig and sig['net_score'] != 0:
                        if not found:
                            await ctx.send(f"**{sym}**")
                            found = True
                        embed = format_embed(sym, sig, tf)
                        await ctx.send(embed=embed)
                await asyncio.sleep(0.3)
        await ctx.send("Scan complete.")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='signal')
async def signal_single(ctx, ticker: str):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        symbol = normalize_symbol(ticker)
        await ctx.send(f"Multi‑timeframe analysis for {symbol}...")
        timeframes = ['5min','15min','30min','1h','4h','daily','weekly']
        found = False
        for tf in timeframes:
            df = await fetch_ohlcv(symbol, tf)
            if df is not None and not df.empty:
                df = calculate_indicators(df)
                sig = get_signals(df)
                if sig and sig['net_score'] != 0:
                    if not found:
                        await ctx.send(f"**{symbol}**")
                        found = True
                    embed = format_embed(symbol, sig, tf)
                    await ctx.send(embed=embed)
            await asyncio.sleep(0.3)
        if not found:
            await ctx.send(f"No active signals for {symbol}.")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# WATCHLIST COMMANDS
# ====================
@bot.command(name='add')
async def add_symbol(ctx, symbol):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        sym = normalize_symbol(symbol.upper())
        wl = await load_watchlist()
        if '/' in sym:
            if sym not in wl['crypto']:
                wl['crypto'].append(sym)
                if await save_watchlist(wl):
                    await ctx.send(f"✅ Added {sym} to crypto watchlist.")
                else:
                    await ctx.send("Failed to save.")
            else:
                await ctx.send(f"{sym} already in crypto list.")
        else:
            if sym not in wl['stocks']:
                wl['stocks'].append(sym)
                if await save_watchlist(wl):
                    await ctx.send(f"✅ Added {sym} to stocks watchlist.")
                else:
                    await ctx.send("Failed to save.")
            else:
                await ctx.send(f"{sym} already in stocks list.")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='remove')
async def remove_symbol(ctx, symbol):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        sym = normalize_symbol(symbol.upper())
        wl = await load_watchlist()
        if sym in wl['stocks']:
            wl['stocks'].remove(sym)
            if await save_watchlist(wl):
                await ctx.send(f"✅ Removed {sym} from stocks.")
            else:
                await ctx.send("Failed.")
        elif sym in wl['crypto']:
            wl['crypto'].remove(sym)
            if await save_watchlist(wl):
                await ctx.send(f"✅ Removed {sym} from crypto.")
            else:
                await ctx.send("Failed.")
        else:
            await ctx.send(f"{sym} not found.")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='list')
async def list_watchlist(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        wl = await load_watchlist()
        stocks = ", ".join(wl['stocks']) if wl['stocks'] else "None"
        crypto = ", ".join(wl['crypto']) if wl['crypto'] else "None"
        await ctx.send(f"**Stocks:** {stocks}\n**Crypto:** {crypto}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# UTILITY
# ====================
@bot.command(name='help')
async def help_command(ctx):
    embed = discord.Embed(title="Bot Commands", description="Prefix `!`", color=0x3498db)
    embed.add_field(name="Scanning", value="`!scan all [tf]` `!score all [tf]` `!signals` `!signal SYMBOL`", inline=False)
    embed.add_field(name="Analysis", value="`!confirm SYMBOL [tf]` `!structure SYMBOL [tf]` `!strength SYMBOL [tf]` `!zone SYMBOL [tf]`", inline=False)
    embed.add_field(name="Options", value="`!flow SYMBOL` `!scanflow`", inline=False)
    embed.add_field(name="Other", value="`!upcoming SYMBOL` `!backtest SYMBOL` `!verify SYMBOL [tf]`", inline=False)
    embed.add_field(name="Watchlist", value="`!add SYMBOL` `!remove SYMBOL` `!list`", inline=False)
    embed.add_field(name="Utility", value="`!ping` `!stopscan` `!cancel` `!help`", inline=False)
    embed.set_footer(text="Timeframes: 5min,15min,30min,1h,4h,daily,weekly")
    await ctx.send(embed=embed)

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('pong')

async def check_cancel(ctx):
    if cancellation_flags.get(ctx.author.id, False):
        cancellation_flags[ctx.author.id] = False
        await ctx.send("Scan cancelled.")
        return True
    return False

@bot.command(name='stopscan')
async def stop_scan(ctx):
    cancellation_flags[ctx.author.id] = True
    await ctx.send("Stopping scan...")

@bot.command(name='cancel')
async def cancel_scan(ctx):
    await stop_scan(ctx)

# ====================
# MAIN
# ====================
async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())