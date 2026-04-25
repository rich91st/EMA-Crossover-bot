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

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# Charting libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mplfinance as mpf
import io

# Finviz integration
from finvizfinance.screener.overview import Overview

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

# === CONFIGURATION ===
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI')
PORT = int(os.getenv('PORT', 10000))
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')

# Alpaca keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# MongoDB setup
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client['trading_bot']
watchlist_collection = db['watchlist']
trades_collection = db['trades']

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)
bot._skip_check = lambda x, y: False

last_command_time = {}
user_busy = {}
cancellation_flags = {}

# ====================
# CACHE SETUP
# ====================
data_cache = {}
CACHE_DURATION = timedelta(minutes=10)

world_news_cache = {"data": None, "expiry": datetime.min}

# ====================
# RATE LIMITERS
# ====================
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

twelvedata_limiter = RateLimiter(max_calls=8, period=60)
finnhub_limiter = RateLimiter(max_calls=60, period=60)
coingecko_limiter = RateLimiter(max_calls=30, period=60)

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
                "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG", "QUBT", "TSLA", "LYFT", "NFLX", "ORCL", "UBER", "HOOD", "SOFI", "SPY", "NIO", "PLTR", "GRAB", "LMT", "MARA", "SOUN", "APLD", "CLSK", "OPEN", "ASML", "RIOT", "AAL", "F", "FCEL"],
                "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD", "LINK/USD"]
            }
            await watchlist_collection.insert_one(default)
            return default
    except Exception as e:
        print(f"❌ Error loading watchlist: {e}")
        return {
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG", "QUBT", "TSLA", "LYFT", "NFLX", "ORCL", "UBER", "HOOD", "SOFI", "SPY", "NIO", "PLTR", "GRAB", "LMT", "MARA", "SOUN", "APLD", "CLSK", "OPEN", "ASML", "RIOT", "AAL", "F", "FCEL"],
            "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD", "LINK/USD"]
        }

async def save_watchlist(watchlist):
    try:
        await watchlist_collection.replace_one(
            {'_id': 'main'},
            {'_id': 'main', 'stocks': watchlist['stocks'], 'crypto': watchlist['crypto']},
            upsert=True
        )
        print(f"✅ Watchlist saved")
        return True
    except Exception as e:
        print(f"❌ Error saving watchlist: {e}")
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

# ====================
# TRADINGVIEW WEB LINK
# ====================
def get_tradingview_web_link(symbol):
    if '/' in symbol:
        base = symbol.split('/')[0]
        exchange = "BINANCE"
        tv_symbol = f"{exchange}:{base}USDT"
    else:
        exchange = "NASDAQ"
        tv_symbol = f"{exchange}:{symbol}"
    web_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
    return web_url

# ====================
# DATA FETCHING FUNCTIONS (condensed for length)
# ====================
async def fetch_finnhub(symbol, timeframe):
    resolution_map = {
        '5min': '5', '15min': '15', '30min': '30',
        '1h': '60', '4h': '240', 'daily': 'D', 'weekly': 'W'
    }
    resolution = resolution_map.get(timeframe)
    if not resolution:
        return None

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol, 'resolution': resolution,
        'from': int((datetime.now() - timedelta(days=60)).timestamp()),
        'to': int(datetime.now().timestamp()), 'token': FINNHUB_API_KEY
    }
    await finnhub_limiter.wait_if_needed()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                if data.get('s') != 'ok':
                    return None
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'], 'high': data['h'],
                    'low': data['l'], 'close': data['c'], 'volume': data['v']
                }).set_index('timestamp')
                return df
    except Exception:
        return None

async def fetch_twelvedata(symbol, timeframe):
    interval_map = {
        '5min': '5min', '15min': '15min', '30min': '30min',
        '1h': '1h', '4h': '4h', 'daily': '1day', 'weekly': '1week'
    }
    interval = interval_map.get(timeframe)
    if not interval:
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {'symbol': symbol, 'interval': interval, 'apikey': TWELVEDATA_API_KEY, 'outputsize': 500}
    await twelvedata_limiter.wait_if_needed()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(60)
                    return None
                data = await resp.json()
                if 'values' not in data:
                    return None
                df = pd.DataFrame(data['values'])
                df = df.rename(columns={'datetime': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.astype(float).sort_index()
                return df
    except Exception:
        return None

async def fetch_coingecko_ohlc(symbol, timeframe):
    base = symbol.split('/')[0].lower()
    coin_map = {'btc': 'bitcoin', 'eth': 'ethereum', 'sol': 'solana', 'xrp': 'ripple', 'doge': 'dogecoin',
                'pepe': 'pepecoin', 'ada': 'cardano', 'dot': 'polkadot', 'link': 'chainlink'}
    coin_id = coin_map.get(base)
    if not coin_id:
        return None
    days_map = {'5min': 1, '15min': 2, '30min': 2, '1h': 7, '4h': 7, 'daily': 30, 'weekly': 90}
    days = days_map.get(timeframe)
    if not days:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {'vs_currency': 'usd', 'days': days}
    await coingecko_limiter.wait_if_needed()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['volume'] = np.nan
                return df
    except Exception:
        return None

async def fetch_coingecko_price(symbol):
    base = symbol.split('/')[0].lower()
    coin_map = {'btc': 'bitcoin', 'eth': 'ethereum', 'sol': 'solana', 'xrp': 'ripple', 'doge': 'dogecoin',
                'pepe': 'pepecoin', 'ada': 'cardano', 'dot': 'polkadot', 'link': 'chainlink'}
    coin_id = coin_map.get(base)
    if not coin_id:
        return None
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {'ids': coin_id, 'vs_currencies': 'usd'}
    await coingecko_limiter.wait_if_needed()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                price = data.get(coin_id, {}).get('usd')
                if price is None:
                    return None
                np.random.seed(42)
                dates = pd.date_range(end=datetime.now(), periods=200, freq='H')
                close_prices = price * (1 + np.random.normal(0, 0.01, 200).cumsum() * 0.01)
                open_prices = close_prices * 0.99
                high_prices = close_prices * 1.02
                low_prices = close_prices * 0.98
                volumes = np.abs(np.random.normal(1e6, 2e5, 200))
                df = pd.DataFrame({'timestamp': dates, 'open': open_prices, 'high': high_prices,
                                   'low': low_prices, 'close': close_prices, 'volume': volumes})
                df.set_index('timestamp', inplace=True)
                return df
    except Exception:
        return None

async def fetch_ohlcv(symbol, timeframe):
    cache_key = f"{symbol}_{timeframe}"
    now = datetime.now()
    if cache_key in data_cache and data_cache[cache_key][1] > now:
        return data_cache[cache_key][0]
    df = None
    is_crypto = '/' in symbol
    if timeframe == '30min' and not is_crypto and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                                       timeframe_multiplier=15, start=now - timedelta(days=60), end=now)
            bars = await asyncio.to_thread(client.get_stock_bars, request)
            if bars.data:
                df_15 = bars.df.reset_index(level=0, drop=True)
                df_15.index = pd.to_datetime(df_15.index)
                df_30 = df_15.resample('30T').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                                   'close': 'last', 'volume': 'sum'}).dropna()
                if not df_30.empty:
                    df = df_30
        except Exception as e:
            print(f"Alpaca 15min fetch failed: {e}")
    if df is None and not is_crypto and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        tf_map = {'5min': (TimeFrame.Minute, 5), '15min': (TimeFrame.Minute, 15), '1h': (TimeFrame.Hour, 1),
                  '4h': (TimeFrame.Hour, 4), 'daily': (TimeFrame.Day, 1), 'weekly': (TimeFrame.Week, 1)}
        if timeframe in tf_map:
            tf, mult = tf_map[timeframe]
            try:
                client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
                request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, timeframe_multiplier=mult,
                                           start=now - timedelta(days=60), end=now)
                bars = await asyncio.to_thread(client.get_stock_bars, request)
                if bars.data:
                    df = bars.df.reset_index(level=0, drop=True)[['open', 'high', 'low', 'close', 'volume']]
            except Exception as e:
                print(f"Alpaca fetch failed: {e}")
    if df is None and not is_crypto:
        df = await fetch_finnhub(symbol, timeframe)
    if df is None and not is_crypto:
        df = await fetch_twelvedata(symbol, timeframe)
    if is_crypto:
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            tf_map = {'5min': (TimeFrame.Minute, 5), '15min': (TimeFrame.Minute, 15), '1h': (TimeFrame.Hour, 1),
                      '4h': (TimeFrame.Hour, 4), 'daily': (TimeFrame.Day, 1), 'weekly': (TimeFrame.Week, 1)}
            if timeframe in tf_map:
                tf, mult = tf_map[timeframe]
                try:
                    client = CryptoHistoricalDataClient()
                    alpaca_symbol = symbol.replace('/', '')
                    request = CryptoBarsRequest(symbol_or_symbols=alpaca_symbol, timeframe=tf,
                                                timeframe_multiplier=mult, start=now - timedelta(days=60), end=now)
                    bars = await asyncio.to_thread(client.get_crypto_bars, request)
                    if bars.data:
                        df = bars.df.reset_index(level=0, drop=True)[['open', 'high', 'low', 'close', 'volume']]
                except Exception as e:
                    print(f"Alpaca crypto fetch failed: {e}")
        if df is None:
            df = await fetch_coingecko_ohlc(symbol, timeframe)
        if df is None:
            df = await fetch_coingecko_price(symbol)
    if df is not None and not df.empty:
        data_cache[cache_key] = (df, now + CACHE_DURATION)
    return df

# ====================
# NEWS FETCHING FUNCTIONS (condensed)
# ====================
async def fetch_finnhub_news(symbol):
    url = "https://finnhub.io/api/v1/company-news"
    params = {'symbol': symbol,
              'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
              'to': datetime.now().strftime('%Y-%m-%d'), 'token': FINNHUB_API_KEY}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, list) else None
    except Exception:
        return None

async def fetch_finnhub_general_news():
    url = "https://finnhub.io/api/v1/news"
    params = {'category': 'general', 'token': FINNHUB_API_KEY}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, list) else None
    except Exception:
        return None

async def fetch_newsapi_top_headlines(country='us'):
    if not NEWSAPI_KEY:
        return None
    url = "https://newsapi.org/v2/top-headlines"
    params = {'apiKey': NEWSAPI_KEY, 'language': 'en', 'country': country, 'pageSize': 10}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get('articles', []) if data.get('status') == 'ok' else None
    except Exception:
        return None

async def fetch_analyst_ratings(symbol, limit=3):
    url = "https://finnhub.io/api/v1/stock/recommendation"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data[:limit] if data else []
    except Exception:
        return []

async def fetch_earnings_upcoming(symbol, days=14):
    try:
        stock = yf.Ticker(symbol)
        earnings_dates = stock.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            return []
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        upcoming = []
        for date, row in earnings_dates.iterrows():
            e_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
            if today <= e_date <= cutoff:
                eps_est = row.get('epsEstimated') if 'epsEstimated' in row else row.get('epsEstimate')
                eps_est = 'N/A' if eps_est is None or pd.isna(eps_est) else f"{eps_est:.2f}"
                upcoming.append({'date': e_date.strftime('%Y-%m-%d'), 'epsEstimate': eps_est, 'hour': 'AMC'})
        return upcoming
    except Exception:
        return []

async def fetch_dividends_upcoming(symbol, days=14):
    try:
        stock = yf.Ticker(symbol)
        dividends = stock.dividends
        if dividends.empty:
            return []
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        upcoming = []
        for date, amount in dividends.items():
            d_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
            if today <= d_date <= cutoff:
                upcoming.append({'exDate': d_date.strftime('%Y-%m-%d'), 'amount': f"{amount:.2f}", 'payDate': ''})
        return upcoming
    except Exception:
        return []

async def fetch_splits_upcoming(symbol, days=14):
    try:
        stock = yf.Ticker(symbol)
        splits = stock.splits
        if splits.empty:
            return []
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        upcoming = []
        for date, ratio in splits.items():
            s_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
            if today <= s_date <= cutoff:
                ratio_str = f"{ratio:.0f}:1" if ratio >= 1 else f"1:{int(1/ratio)}"
                upcoming.append({'date': s_date.strftime('%Y-%m-%d'), 'splitRatio': ratio_str})
        return upcoming
    except Exception:
        return []

async def fetch_economic_events(days=14):
    if not FINNHUB_API_KEY:
        return []
    url = "https://finnhub.io/api/v1/calendar/economic"
    params = {'from': datetime.now().strftime('%Y-%m-%d'),
              'to': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'), 'token': FINNHUB_API_KEY}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get('economicCalendar', [])
    except Exception:
        return []

# ====================
# INDICATOR CALCULATIONS
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
# CHART GENERATION (condensed)
# ====================
def generate_chart_image(df, symbol, timeframe):
    if len(df) < 20:
        return None
    if timeframe in ['5min', '15min', '1h']:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(50).copy()
    else:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(30).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='white', wick='white', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False, facecolor='#1e1e1e',
                           figcolor='#1e1e1e', gridcolor='#444444')
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
        volume_all_nan = chart_data['Volume'].isna().all()
        fig, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds, volume=not volume_all_nan,
                             figsize=(10,6), returnfig=True, title=f'{symbol} – {timeframe}', tight_layout=True)
        if apds:
            axes[0].legend(loc='upper left', fontsize=9, facecolor='#333333', edgecolor='white',
                           labelcolor='white', framealpha=0.8)
        axes[0].tick_params(colors='white')
        axes[0].yaxis.label.set_color('white')
        axes[0].xaxis.label.set_color('white')
        axes[0].set_facecolor('#1e1e1e')
        if not volume_all_nan:
            axes[2].set_ylabel('Volume', color='white')
            axes[2].tick_params(colors='white')
            axes[2].yaxis.label.set_color('white')
            axes[2].set_facecolor('#1e1e1e')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name, format='PNG', dpi=120, bbox_inches='tight', facecolor=s['facecolor'])
            tmpfile.flush()
            with open(tmpfile.name, 'rb') as f:
                img_data = f.read()
        os.unlink(tmpfile.name)
        plt.close(fig)
        return io.BytesIO(img_data)
    except Exception as e:
        print(f"Chart generation failed: {e}")
        return None

def generate_zone_chart(df, symbol, zones):
    if len(df) < 20:
        return None
    chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(100).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', edge='white', wick='white', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False, facecolor='#1e1e1e',
                           figcolor='#1e1e1e', gridcolor='#444444')
    if zones:
        strengths = [z['strength'] for z in zones]
        min_s, max_s = min(strengths), max(strengths)
        norm_strengths = [(s - min_s) / (max_s - min_s) for s in strengths] if max_s > min_s else [0.5] * len(strengths)
        colormap = matplotlib.colormaps['RdYlGn_r']
        line_colors = [colormap(norm) for norm in norm_strengths]
    else:
        line_colors = []
    apds = []
    for i, zone in enumerate(zones):
        level = zone['level']
        color = mcolors.to_hex(line_colors[i]) if line_colors else '#ffffff'
        label = f"Demand ${level:.2f} (touches: {zone['strength']})"
        apds.append(mpf.make_addplot([level] * len(chart_data), color=color, width=2.0, linestyle='-', label=label))
    try:
        fig, axes = mpf.plot(chart_data, type='candle', style=s, addplot=apds, volume=True, figsize=(12, 7),
                             returnfig=True, title=f'\n{symbol} Demand Zones (30min)', tight_layout=True,
                             scale_padding={'left': 0.5, 'right': 0.5, 'top': 0.5, 'bottom': 0.5})
        if apds:
            axes[0].legend(loc='upper left', fontsize=10, facecolor='#333333', edgecolor='white',
                           labelcolor='white', framealpha=0.8)
        axes[2].set_ylabel('Volume', color='white')
        axes[2].tick_params(colors='white')
        axes[0].tick_params(colors='white')
        axes[0].set_facecolor('#1e1e1e')
        axes[2].set_facecolor('#1e1e1e')
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name, format='PNG', dpi=150, bbox_inches='tight', facecolor=s['facecolor'])
            tmpfile.flush()
            with open(tmpfile.name, 'rb') as f:
                img_data = f.read()
        os.unlink(tmpfile.name)
        plt.close(fig)
        return io.BytesIO(img_data)
    except Exception as e:
        print(f"Zone chart failed: {e}")
        return None

# ====================
# WORLD NEWS COMMAND
# ====================
@bot.command(name='worldnews')
async def world_news(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send("🌍 Fetching global market news with AI sentiment analysis...")
        api_key = MARKETAUX_API_KEY
        if not api_key:
            await ctx.send("❌ Marketaux API key not configured.")
            return
        url = "https://api.marketaux.com/v1/news/all"
        params = {'api_token': api_key, 'language': 'en', 'limit': 10, 'must_have_entities': 'true'}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    await ctx.send(f"❌ News API error: {resp.status}")
                    return
                data = await resp.json()
        articles = data.get('data', [])
        if not articles:
            await ctx.send("No news articles found.")
            return
        embed = discord.Embed(title="🌍 World News – AI Market Sentiment",
                              description="Top financial headlines from Marketaux", color=0x3498db, timestamp=datetime.now())
        trump_post = await fetch_latest_trump_post()
        if trump_post:
            analysis = analyze_trump_post(trump_post['text'])
            trump_value = (f"**Post:** {trump_post['text'][:300]}{'…' if len(trump_post['text'])>300 else ''}\n"
                           f"**Time:** {trump_post['timestamp']}\n**Sentiment:** {analysis['sentiment']}\n"
                           f"**TACO Probability:** {analysis['taco_probability']}\n"
                           f"**Affected Sectors:** {analysis['affected_sectors']}\n"
                           f"**Suggested Stocks:** {analysis['suggested_stocks']}\n"
                           f"**Advice:** {analysis['advice']}\n[Link to post]({trump_post['url']})")
            embed.add_field(name="🔔 @realDonaldTrump (Latest Truth)", value=trump_value, inline=False)
        for article in articles[:8]:
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            published_at = article.get('published_at', '')
            url_link = article.get('url', '')
            entities = article.get('entities', [])
            sentiment_score = entities[0].get('sentiment_score', 0.0) if entities else 0.0
            sentiment_emoji = "🟢 Bullish" if sentiment_score > 0.2 else "🔴 Bearish" if sentiment_score < -0.2 else "⚪ Neutral"
            time_ago = "recently"
            if published_at:
                try:
                    pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    delta = datetime.now() - pub_time
                    if delta.total_seconds() < 3600:
                        mins = int(delta.total_seconds() / 60)
                        time_ago = f"{mins} min ago" if mins > 0 else "just now"
                    elif delta.total_seconds() < 86400:
                        hours = int(delta.total_seconds() / 3600)
                        time_ago = f"{hours} hour ago" if hours == 1 else f"{hours} hours ago"
                    else:
                        days = delta.days
                        time_ago = f"{days} day ago" if days == 1 else f"{days} days ago"
                except:
                    pass
            field_value = f"**Source:** {source} | {time_ago}\n**Sentiment:** {sentiment_emoji} (score: {sentiment_score:.2f})"
            if url_link:
                field_value += f"\n[Read more]({url_link})"
            embed.add_field(name=f"📰 {title[:100]}{'…' if len(title)>100 else ''}", value=field_value, inline=False)
        embed.set_footer(text="🟢 Bullish | 🔴 Bearish | ⚪ Neutral • AI-powered sentiment analysis")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# ENHANCED NEWS COMMAND
# ====================
@bot.command(name='news')
async def stock_news_enhanced(ctx, ticker: str):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        symbol = ticker.upper()
        await ctx.send(f"🔍 Gathering market intelligence for **{symbol}**...")
        current_price = await get_stock_price(symbol)
        prev_close = None
        if current_price:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="2d")
                if len(hist) >= 2:
                    prev_close = hist['Close'].iloc[-2]
            except:
                pass
        ratings_task = fetch_analyst_ratings(symbol, limit=3)
        news_task = fetch_finnhub_news(symbol)
        ratings_data, finnhub_data = await asyncio.gather(ratings_task, news_task, return_exceptions=True)
        if isinstance(ratings_data, Exception):
            ratings_data = None
        if isinstance(finnhub_data, Exception):
            finnhub_data = None
        if not finnhub_data:
            await ctx.send(f"❌ Could not fetch news data for {symbol}.")
            return
        web_url = get_tradingview_web_link(symbol)
        tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"
        embed = discord.Embed(title=f"Latest News for {symbol}", color=0x3498db, timestamp=datetime.now())
        if current_price:
            embed.description = f"Current Price: **${current_price:.2f}**"
            if prev_close:
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                arrow = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                embed.description += f" | {arrow} {change:+.2f} ({change_pct:+.2f}%)"
        if ratings_data:
            ratings_text = ""
            for r in ratings_data[:3]:
                period = r.get('period', '')
                sb = r.get('strongBuy', 0)
                b = r.get('buy', 0)
                h = r.get('hold', 0)
                s = r.get('sell', 0)
                ss = r.get('strongSell', 0)
                total = sb + b + h + s + ss
                buys = sb + b
                sells = s + ss
                sentiment = "🟢" if buys > sells else "🔴" if sells > buys else "⚪"
                if total > 0:
                    ratings_text += f"**{period}** – {buys} Buy / {h} Hold / {sells} Sell {sentiment}\n"
            if ratings_text:
                embed.add_field(name="📈 Analyst Ratings (last 3)", value=ratings_text, inline=False)
        for article in finnhub_data[:5]:
            headline = article.get('headline', 'No Headline')
            source = article.get('source', 'Unknown')
            dt = article.get('datetime')
            date = datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M') if isinstance(dt, (int, float)) else str(dt)[:16] if dt else 'Unknown'
            url = article.get('url', '')
            if len(headline) > 256:
                headline = headline[:253] + "..."
            embed.add_field(name=f"{source} - {date}", value=f"[{headline}]({url})", inline=False)
        embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
        embed.set_footer(text=f"Requested by {ctx.author.display_name}")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# PEG RATIO HELPER
# ====================
async def get_peg_ratio(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        peg = None
        for field in ['pegRatio', 'pegRatio5yr', 'pegRatioTTM', 'trailingPEG']:
            val = info.get(field)
            if val and isinstance(val, (int, float)) and val > 0:
                peg = float(val)
                break
        if peg is None:
            pe = info.get('trailingPE')
            earnings_growth = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth') or info.get('earningsGrowth5y')
            if pe and earnings_growth and isinstance(earnings_growth, (int, float)) and earnings_growth != 0:
                peg = pe / (earnings_growth * 100)
                if peg <= 0:
                    peg = None
        if peg is None or peg <= 0:
            return None, None
        if peg < 1.0:
            emoji = "🟢"
        elif peg < 2.0:
            emoji = "🟡"
        else:
            emoji = "🔴"
        return peg, f"{emoji} {peg:.2f}"
    except Exception as e:
        print(f"Error fetching PEG for {symbol}: {e}")
        return None, None

# ====================
# EMBED FORMATTING
# ====================
def format_embed(symbol, signals, timeframe, peg_str=None):
    if not signals:
        return discord.Embed(title=f"Error", description=f"No data for {symbol}", color=0xff0000)
    sym_type = "Crypto" if '/' in symbol else "Stock"
    rating, color = get_rating(signals)
    if signals.get('volume') and signals.get('volume_avg') and signals['volume_avg'] > 0:
        vol_ratio = signals['volume'] / signals['volume_avg']
        vol_status = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
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
    ema_items = [(signals['ema5'], '5', '🟢'), (signals['ema13'], '13', '🟡'),
                 (signals['ema50'], '50', '🔴'), (signals['ema200'], '200', '🟣')]
    valid_items = [(val, lbl, emoji) for val, lbl, emoji in ema_items if not pd.isna(val)]
    valid_items.sort(reverse=True)
    ema_lines = [f"{emoji} {lbl}: ${val:.2f}" for val, lbl, emoji in valid_items]
    ema_text = "\n".join(ema_lines) if valid_items else "N/A"
    web_url = get_tradingview_web_link(symbol)
    tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"
    embed = discord.Embed(title=f"{rating}", description=f"**{symbol}** · ${signals['price']:.2f}", color=color)
    embed.add_field(name="RSI", value=f"{signals['rsi']:.1f}", inline=True)
    embed.add_field(name="Trend", value=signals['trend'], inline=True)
    embed.add_field(name="Volume", value=vol_display, inline=True)
    if peg_str and '/' not in symbol:
        embed.add_field(name="PEG Ratio", value=peg_str, inline=True)
    embed.add_field(name="EMAs (sorted)", value=ema_text, inline=False)
    embed.add_field(name="Bollinger Bands", value=bb_status, inline=True)
    embed.add_field(name="Reason", value=reason_str, inline=False)
    embed.add_field(name="Support", value=f"${support:.2f}", inline=True)
    embed.add_field(name="Resistance", value=f"${resistance:.2f}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop_loss:.2f}", inline=True)
    embed.add_field(name="Target", value=f"${target:.2f}", inline=True)
    embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
    embed.set_footer(text=f"{sym_type} · {timeframe}")
    return embed

def format_zone_embed(symbol, signals, timeframe):
    sym_type = "Crypto" if '/' in symbol else "Stock"
    price = signals['price']
    support = signals['support_20']
    resistance = signals['resistance_20']
    ema5, ema13, ema50, ema200 = signals['ema5'], signals['ema13'], signals['ema50'], signals['ema200']
    support_levels = [support]
    resistance_levels = [resistance]
    if not pd.isna(ema200):
        support_levels.append(ema200) if ema200 < price else resistance_levels.append(ema200)
    if not pd.isna(ema50):
        support_levels.append(ema50) if ema50 < price else resistance_levels.append(ema50)
    if not pd.isna(ema13):
        support_levels.append(ema13) if ema13 < price else resistance_levels.append(ema13)
    if not pd.isna(ema5):
        support_levels.append(ema5) if ema5 < price else resistance_levels.append(ema5)
    support_levels.sort(reverse=True)
    resistance_levels.sort()
    web_url = get_tradingview_web_link(symbol)
    tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"
    embed = discord.Embed(title=f"📊 {symbol} – {timeframe.capitalize()} Zones",
                          description=f"Current Price: **${price:.2f}**",
                          color=0x00ff00 if signals['net_score'] > 0 else 0xff0000 if signals['net_score'] < 0 else 0xffff00)
    sup_text = ""
    for i, level in enumerate(support_levels):
        sup_text += f"**Primary Support:** ${level:.2f}\n" if i == 0 else f"Secondary Support: ${level:.2f}\n"
    if sup_text:
        embed.add_field(name="📉 Support (Buy Zone)", value=sup_text, inline=False)
    res_text = ""
    for i, level in enumerate(resistance_levels):
        res_text += f"**Primary Resistance:** ${level:.2f}\n" if i == 0 else f"Secondary Resistance: ${level:.2f}\n"
    if res_text:
        embed.add_field(name="📈 Resistance (Sell Zone)", value=res_text, inline=False)
    target = resistance + (resistance - support)
    embed.add_field(name="🎯 Projected Target", value=f"${target:.2f}", inline=False)
    embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
    embed.set_footer(text=f"{sym_type} · Based on 20-day high/low and EMAs")
    return embed

# ====================
# COMBINED SYMBOL REPORT FUNCTIONS
# ====================
async def send_combined_symbol_report(ctx, symbol, symbol_signals):
    timeframe_priority = {'5min': 1, '15min': 2, '30min': 3, '1h': 4, '4h': 5, 'daily': 6, 'weekly': 7}
    best_tf, best_score = None, -float('inf')
    for tf, data in symbol_signals.items():
        strength = abs(data['signals']['net_score'])
        if strength > best_score or (strength == best_score and timeframe_priority.get(tf, 99) < timeframe_priority.get(best_tf, 99)):
            best_score, best_tf = strength, tf
    if not best_tf:
        return
    best_data = symbol_signals[best_tf]
    df, signals = best_data['df'], best_data['signals']
    peg_str = None
    if '/' not in symbol:
        _, peg_str = await get_peg_ratio(symbol)
    main_embed = format_embed(symbol, signals, best_tf, peg_str=peg_str)
    try:
        chart_buffer = generate_chart_image(df, symbol, best_tf)
        if chart_buffer:
            file = discord.File(chart_buffer, filename='chart.png')
            main_embed.set_image(url='attachment://chart.png')
            main_embed.description = f"**{symbol}** · ${signals['price']:.2f} · ⭐ **Primary: {best_tf}**"
            await ctx.send(embed=main_embed, file=file)
        else:
            await ctx.send(embed=main_embed)
    except Exception as e:
        print(f"Chart generation failed: {e}")
        await ctx.send(embed=main_embed)
    await send_symbol_timeframe_summary(ctx, symbol, symbol_signals)

async def send_symbol_timeframe_summary(ctx, symbol, symbol_signals):
    timeframe_order = {'5min': 1, '15min': 2, '30min': 3, '1h': 4, '4h': 5, 'daily': 6, 'weekly': 7}
    sorted_timeframes = sorted(symbol_signals.keys(), key=lambda x: timeframe_order.get(x, 99))
    bullish_count = sum(1 for data in symbol_signals.values() if data['signals']['net_score'] > 0)
    bearish_count = sum(1 for data in symbol_signals.values() if data['signals']['net_score'] < 0)
    total = len(symbol_signals)
    summary_lines = []
    for tf in sorted_timeframes:
        net = symbol_signals[tf]['signals']['net_score']
        if net >= 2:
            emoji, signal_text = "🟢", "STRONG BUY"
        elif net == 1:
            emoji, signal_text = "🟢", "BUY"
        elif net == 0:
            emoji, signal_text = "⚪", "NEUTRAL"
        elif net == -1:
            emoji, signal_text = "🔴", "SELL"
        else:
            emoji, signal_text = "🔴", "STRONG SELL"
        summary_lines.append(f"{emoji} {tf}: {signal_text} (Score: {net})")
    embed = discord.Embed(title=f"📊 MULTI-TIMEFRAME SUMMARY: {symbol}", description="\n".join(summary_lines), color=0x3498db)
    if bullish_count == total:
        recommendation = "🎯 **RECOMMENDATION: STRONG BUY - All timeframes aligned!**"
    elif bearish_count == total:
        recommendation = "🎯 **RECOMMENDATION: STRONG SELL - All timeframes aligned!**"
    elif bullish_count >= total * 0.6:
        recommendation = "🎯 **RECOMMENDATION: CAUTIOUS BUY - Most timeframes bullish**"
    elif bearish_count >= total * 0.6:
        recommendation = "🎯 **RECOMMENDATION: CAUTIOUS SELL - Most timeframes bearish**"
    else:
        recommendation = "🎯 **RECOMMENDATION: NEUTRAL - Mixed signals**"
    embed.add_field(name="", value=recommendation, inline=False)
    await ctx.send(embed=embed)

async def send_final_summary(ctx, signal_summary):
    if not signal_summary:
        return
    embed = discord.Embed(title="📊 MULTI-TIMEFRAME SCAN COMPLETE",
                          description=f"Found signals for **{len(signal_summary)}** symbols", color=0x3498db)
    strong_buy, buy, neutral, sell, strong_sell = [], [], [], [], []
    for symbol, timeframes in signal_summary.items():
        avg_score = sum(sig['net_score'] for sig in timeframes.values()) / len(timeframes)
        bullish_count = sum(1 for sig in timeframes.values() if sig['net_score'] > 0)
        if avg_score >= 1.5:
            strong_buy.append(f"{symbol} ({bullish_count}/{len(timeframes)})")
        elif avg_score > 0:
            buy.append(f"{symbol} ({bullish_count}/{len(timeframes)})")
        elif avg_score == 0:
            neutral.append(f"{symbol}")
        elif avg_score > -1.5:
            sell.append(f"{symbol}")
        else:
            strong_sell.append(f"{symbol}")
    if strong_buy:
        embed.add_field(name="🟢🟢 STRONG BUY", value="\n".join(strong_buy[:10]), inline=False)
    if buy:
        embed.add_field(name="🟢 BUY", value="\n".join(buy[:10]), inline=False)
    if neutral:
        embed.add_field(name="⚪ NEUTRAL", value="\n".join(neutral[:10]), inline=False)
    if sell:
        embed.add_field(name="🔴 SELL", value="\n".join(sell[:10]), inline=False)
    if strong_sell:
        embed.add_field(name="🔴🔴 STRONG SELL", value="\n".join(strong_sell[:10]), inline=False)
    embed.set_footer(text="Use !signal SYMBOL for detailed analysis")
    await ctx.send(embed=embed)

# ====================
# OPTIONS FLOW SCANNER
# ====================
async def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        return float(data['Close'].iloc[-1]) if not data.empty else None
    except Exception:
        return None

def get_key_expirations(ticker):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return []
        today = datetime.now().date()
        exp_dates = [datetime.strptime(e, '%Y-%m-%d').date() for e in expirations]
        key_exps = []
        for exp_date, exp_str in zip(exp_dates, expirations):
            dte = (exp_date - today).days
            if 0 <= dte <= 7:
                key_exps.append((exp_str, dte, "🔥 WEEKLY (0-7 DTE)"))
                break
        for exp_date, exp_str in zip(exp_dates, expirations):
            dte = (exp_date - today).days
            if 8 <= dte <= 21:
                key_exps.append((exp_str, dte, "💎 MONTHLY (8-21 DTE)"))
                break
        best_exp, best_dte, min_diff = None, None, float('inf')
        for exp_date, exp_str in zip(exp_dates, expirations):
            dte = (exp_date - today).days
            diff = abs(dte - 38)
            if diff < min_diff:
                min_diff, best_exp, best_dte = diff, exp_str, dte
        if best_exp and best_dte:
            key_exps.append((best_exp, best_dte, "🛡️ PRIMARY (30-45 DTE)"))
        return key_exps
    except Exception as e:
        print(f"Error getting key expirations: {e}")
        return []

def format_premium(volume, last_price):
    try:
        premium = volume * 100 * last_price
        if premium >= 1000000:
            return f"${premium/1000000:.1f}M"
        elif premium >= 1000:
            return f"${premium/1000:.0f}K"
        else:
            return f"${premium:.0f}"
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
    all_options = pd.concat([calls, puts], ignore_index=True)
    analyzed = []
    for _, opt in all_options.iterrows():
        try:
            volume = opt.get('volume', 0)
            oi = opt.get('openInterest', 0)
            strike = opt.get('strike', 0)
            last = opt.get('lastPrice', 0)
            opt_type = opt.get('type', 'CALL')
            if pd.isna(volume) or pd.isna(oi) or volume < min_volume or oi == 0:
                continue
            vol_oi_ratio = volume / oi
            premium = volume * 100 * last
            distance_pct = abs(strike - current_price) / current_price * 100
            analyzed.append({'strike': strike, 'type': opt_type, 'volume': int(volume), 'oi': int(oi),
                             'vol_oi_ratio': vol_oi_ratio, 'last': last, 'premium': format_premium(volume, last),
                             'raw_premium': premium, 'distance_pct': distance_pct, 'dte': dte})
        except:
            continue
    return analyzed

def add_field_safe(embed, name, value, inline=False):
    if len(value) <= 1024:
        embed.add_field(name=name, value=value, inline=inline)
    else:
        chunks = [value[i:i+1024] for i in range(0, len(value), 1024)]
        for i, chunk in enumerate(chunks):
            suffix = f" (cont.)" if i > 0 else ""
            embed.add_field(name=f"{name}{suffix}", value=chunk, inline=inline)

@bot.command(name='flow')
async def options_flow(ctx, ticker: str):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send(f"🔍 Analyzing options flow for **{ticker.upper()}**...")
        current_price = await get_stock_price(ticker.upper())
        if not current_price:
            await ctx.send(f"❌ Could not fetch current price for {ticker.upper()}")
            return
        key_exps = get_key_expirations(ticker.upper())
        if not key_exps:
            await ctx.send(f"❌ No options expirations found for {ticker.upper()}")
            return
        stock = yf.Ticker(ticker.upper())
        embed = discord.Embed(title=f"🔍 OPTIONS FLOW: {ticker.upper()}", description=f"Current Price: **${current_price:.2f}**", color=0x00ff00)
        high_prob_options, lottery_options = [], []
        for exp_str, dte, label in key_exps:
            opt_chain = stock.option_chain(exp_str)
            analyzed = analyze_expiration(opt_chain, current_price, dte)
            analyzed.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
            significant = [opt for opt in analyzed if opt['volume'] >= 5 and opt['oi'] > 0][:10]
            for opt in significant:
                if 30 <= dte <= 45 and opt['distance_pct'] <= 20:
                    high_prob_options.append((label, exp_str, dte, opt))
                else:
                    lottery_options.append((label, exp_str, dte, opt))
        hp_text = ""
        for label, exp_str, dte, opt in high_prob_options[:6]:
            strike, whale = f"${opt['strike']:.2f}", get_whale_emoji(opt['raw_premium'])
            distance, prob_est = f"{opt['distance_pct']:.1f}% away", "High" if opt['distance_pct'] < 10 else "Moderate"
            hp_text += f"**{strike} {opt['type']}** ({label})\n   • Vol: {opt['volume']} ({opt['vol_oi_ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n   • DTE: {dte}  Distance: {distance} – Prob: {prob_est}\n\n"
        if hp_text:
            add_field_safe(embed, "📈 HIGH PROBABILITY SETUPS (30-45 DTE, Near Money)", hp_text, inline=False)
        lt_text = ""
        for label, exp_str, dte, opt in lottery_options[:10]:
            strike, whale = f"${opt['strike']:.2f}", get_whale_emoji(opt['raw_premium'])
            lt_text += f"**{strike} {opt['type']}** ({label})\n   • Vol: {opt['volume']} ({opt['vol_oi_ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n   • DTE: {dte}\n\n"
        if lt_text:
            add_field_safe(embed, "🎰 LOTTERY / OTHER ACTIVITY", lt_text, inline=False)
        if not high_prob_options and not lottery_options:
            await ctx.send("ℹ️ No significant options activity found for this ticker.")
            return
        explanation = """
📊 **WHALE RATINGS:**
• 🐋🐋 = >$1M premium (massive institutional)
• 🐋 = $100K–$1M (strong interest)
• 🐬 = $10K–$100K (notable)
• 🐟 = <$10K (small)

💡 **TIP:** Focus on **High Probability Setups** (30-45 DTE, within 20% of money) for consistent wins.
        """
        embed.add_field(name="", value=explanation, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error analyzing options: {str(e)}")
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
            await ctx.send("No stocks in watchlist to scan.")
            return
        await ctx.send(f"🔍 **SCANNING {len(symbols)} SYMBOLS FOR UNUSUAL OPTIONS ACTIVITY**")
        await ctx.send(f"⏱️ Checking weekly (0‑7 DTE) and primary (30‑45 DTE) expirations...\n")
        all_unusual = []
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            try:
                current_price = await get_stock_price(symbol)
                if not current_price:
                    continue
                stock = yf.Ticker(symbol)
                key_exps = get_key_expirations(symbol)
                for exp_str, dte, label in key_exps:
                    if "WEEKLY" in label or "PRIMARY" in label:
                        opt_chain = stock.option_chain(exp_str)
                        calls = opt_chain.calls
                        if calls.empty:
                            continue
                        for _, opt in calls.iterrows():
                            try:
                                volume = opt.get('volume', 0)
                                oi = opt.get('openInterest', 0)
                                strike = opt.get('strike', 0)
                                last = opt.get('lastPrice', 0)
                                if pd.isna(volume) or pd.isna(oi):
                                    continue
                                if oi > 0 and volume > 0:
                                    vol_oi_ratio = volume / oi
                                    if vol_oi_ratio >= 1.5 and volume >= 10:
                                        distance_pct = abs(strike - current_price) / current_price * 100
                                        if distance_pct <= 20:
                                            premium = volume * 100 * last
                                            all_unusual.append({'symbol': symbol, 'strike': strike, 'expiration': exp_str,
                                                                'dte': dte, 'volume': int(volume), 'oi': int(oi),
                                                                'ratio': vol_oi_ratio, 'price': current_price,
                                                                'premium': format_premium(volume, last), 'raw_premium': premium,
                                                                'distance': distance_pct, 'label': label})
                            except:
                                continue
                await asyncio.sleep(2)
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue
        high_prob = [opt for opt in all_unusual if 30 <= opt['dte'] <= 45 and opt['distance'] <= 20]
        lottery = [opt for opt in all_unusual if opt not in high_prob]
        high_prob.sort(key=lambda x: x['raw_premium'], reverse=True)
        lottery.sort(key=lambda x: x['raw_premium'], reverse=True)
        if not all_unusual:
            await ctx.send("📭 No unusual options activity detected in your watchlist.")
            return
        embed = discord.Embed(title="🔥 UNUSUAL OPTIONS ACTIVITY SUMMARY",
                              description=f"Found {len(all_unusual)} unusual setups across your watchlist", color=0x00ff00)
        hp_text = ""
        for opt in high_prob[:8]:
            whale = get_whale_emoji(opt['raw_premium'])
            hp_text += f"**{opt['symbol']} ${opt['strike']:.2f} CALL** ({opt['label']})\n   • Vol: {opt['volume']} ({opt['ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n   • DTE: {opt['dte']}  Distance: {opt['distance']:.1f}%\n\n"
        if hp_text:
            add_field_safe(embed, "📈 HIGH PROBABILITY SETUPS (30-45 DTE, Near Money)", hp_text, inline=False)
        lt_text = ""
        for opt in lottery[:12]:
            whale = get_whale_emoji(opt['raw_premium'])
            lt_text += f"**{opt['symbol']} ${opt['strike']:.2f} CALL** ({opt['label']})\n   • Vol: {opt['volume']} ({opt['ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n   • DTE: {opt['dte']}\n\n"
        if lt_text:
            add_field_safe(embed, "🎰 LOTTERY / OTHER ACTIVITY", lt_text, inline=False)
        explanation = """
📊 **WHALE RATINGS:**
• 🐋🐋 = >$1M premium (massive institutional)
• 🐋 = $100K–$1M (strong interest)
• 🐬 = $10K–$100K (notable)
• 🐟 = <$10K (small)

💡 **TIP:** Focus on **High Probability Setups** (30-45 DTE, within 20% of money) for consistent wins.
        """
        embed.add_field(name="", value=explanation, inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error scanning options flow: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# MARKET STRUCTURE ANALYSIS
# ====================
def find_swings(df, window=5):
    if len(df) < window * 2 + 1:
        return [], []
    highs, lows = df['high'].values, df['low'].values
    idx = df.index
    swing_highs, swing_lows = [], []
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append((idx[i], highs[i]))
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append((idx[i], lows[i]))
    return swing_highs, swing_lows

def analyze_structure(df, window=5):
    if len(df) < 50:
        return {'trend': 'insufficient data', 'last_event': None, 'last_event_direction': None,
                'bos_events': [], 'choch_events': [], 'description': 'Not enough data.', 'event_points': None}
    highs, lows = find_swings(df, window)
    current_price = df['close'].iloc[-1]
    price_40_ago = df['close'].iloc[-40] if len(df) >= 40 else df['close'].iloc[0]
    pct_change = (current_price - price_40_ago) / price_40_ago * 100
    if pct_change > 3:
        trend = 'uptrend'
    elif pct_change < -3:
        trend = 'downtrend'
    else:
        if len(highs) >= 2 and highs[-1][1] > highs[-2][1]:
            trend = 'uptrend'
        elif len(lows) >= 2 and lows[-1][1] < lows[-2][1]:
            trend = 'downtrend'
        else:
            trend = 'sideways'
    bos_events, choch_events = [], []
    for i in range(1, min(len(highs), 10)):
        if highs[i][1] > highs[i-1][1]:
            bos_events.append({'type': 'BOS', 'direction': 'up', 'price': highs[i][1], 'date': highs[i][0], 'index': i})
    for i in range(1, min(len(lows), 10)):
        if lows[i][1] < lows[i-1][1]:
            bos_events.append({'type': 'BOS', 'direction': 'down', 'price': lows[i][1], 'date': lows[i][0], 'index': i})
    if len(highs) >= 3 and len(lows) >= 3:
        for i in range(2, min(len(lows), 10)):
            if lows[i][1] < lows[i-1][1]:
                choch_events.append({'type': 'CHoCH', 'direction': 'down', 'price': lows[i][1], 'date': lows[i][0], 'index': i})
        for i in range(2, min(len(highs), 10)):
            if highs[i][1] > highs[i-1][1]:
                choch_events.append({'type': 'CHoCH', 'direction': 'up', 'price': highs[i][1], 'date': highs[i][0], 'index': i})
    bos_events = bos_events[-3:] if len(bos_events) > 3 else bos_events
    choch_events = choch_events[-3:] if len(choch_events) > 3 else choch_events
    all_events = bos_events + choch_events
    all_events.sort(key=lambda x: x['date'], reverse=True)
    last_event = all_events[0] if all_events else None
    last_event_type = last_event['type'] if last_event else None
    last_event_direction = last_event['direction'] if last_event else None
    description = f"Trend: {trend}. "
    if last_event_type == 'BOS':
        description += f"Last event: BOS {'↑' if last_event_direction == 'up' else '↓'} (trend continuing)."
    elif last_event_type == 'CHoCH':
        description += f"Last event: CHoCH {'↑' if last_event_direction == 'up' else '↓'} (trend reversing)."
    else:
        description += "No clear BOS or CHoCH events detected."
    return {'trend': trend, 'last_event': last_event_type, 'last_event_direction': last_event_direction,
            'bos_events': bos_events, 'choch_events': choch_events, 'description': description, 'event_points': None}

def generate_structure_chart(df, symbol, structure):
    if len(df) < 50:
        return None
    chart_data = df[['open', 'high', 'low', 'close']].tail(100).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close']
    swing_highs, swing_lows = find_swings(df)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.grid(True, color='#444444', linestyle='--', alpha=0.5)
    dates = chart_data.index
    width = 0.6 * (dates[1] - dates[0]).total_seconds() / (24*3600)
    for idx, row in chart_data.iterrows():
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        ax.bar(idx, row['High'] - row['Low'], bottom=row['Low'], width=width, color=color, alpha=0.5)
        ax.bar(idx, row['Close'] - row['Open'], bottom=row['Open'], width=width, color=color, alpha=1.0)
    for idx, price in swing_highs:
        if idx in chart_data.index:
            ax.plot(idx, price, '^', color='lime', markersize=10, zorder=5, linewidth=2)
    for idx, price in swing_lows:
        if idx in chart_data.index:
            ax.plot(idx, price, 'v', color='red', markersize=10, zorder=5, linewidth=2)
    if structure.get('bos_events'):
        for bos in structure['bos_events'][-3:]:
            if bos['date'] in chart_data.index:
                line_color = '#00aaff' if bos['direction'] == 'up' else '#ff8800'
                ax.axhline(y=bos['price'], color=line_color, linestyle='--', linewidth=1.5, alpha=0.7)
                ax.text(bos['date'], bos['price'], f"BOS {bos['direction'].upper()}", fontsize=8, color=line_color,
                        ha='left', va='bottom', bbox=dict(facecolor='#1e1e1e', alpha=0.7, pad=1))
    if structure.get('choch_events'):
        for choch in structure['choch_events'][-3:]:
            if choch['date'] in chart_data.index:
                line_color = '#ff00ff' if choch['direction'] == 'up' else '#ff4444'
                ax.axhline(y=choch['price'], color=line_color, linestyle='--', linewidth=2, alpha=0.8)
                ax.text(choch['date'], choch['price'], f"CHoCH {choch['direction'].upper()}", fontsize=9, color=line_color,
                        ha='left', va='top', weight='bold', bbox=dict(facecolor='#1e1e1e', alpha=0.8, pad=2))
    all_events = (structure.get('bos_events', []) + structure.get('choch_events', []))
    all_events.sort(key=lambda x: x['date'], reverse=True)
    if all_events:
        last = all_events[0]
        if last['date'] in chart_data.index:
            ax.axhline(y=last['price'], color='white', linestyle='-', linewidth=3, alpha=0.9)
            ax.text(last['date'], last['price'], f"★ MOST RECENT: {last['type']} {last['direction'].upper()}",
                    fontsize=10, color='yellow', ha='left', va='bottom', weight='bold',
                    bbox=dict(facecolor='#1e1e1e', alpha=0.9, pad=3))
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', label='Swing High', markerfacecolor='lime', markersize=8),
        plt.Line2D([0], [0], marker='v', color='w', label='Swing Low', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], color='#00aaff', linestyle='--', linewidth=2, label='BOS Up'),
        plt.Line2D([0], [0], color='#ff8800', linestyle='--', linewidth=2, label='BOS Down'),
        plt.Line2D([0], [0], color='#ff00ff', linestyle='--', linewidth=2, label='CHoCH Up'),
        plt.Line2D([0], [0], color='#ff4444', linestyle='--', linewidth=2, label='CHoCH Down'),
        plt.Line2D([0], [0], color='white', linestyle='-', linewidth=3, label='Most Recent Event'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, facecolor='#333333',
              edgecolor='white', labelcolor='white', framealpha=0.8)
    ax.set_title(f'{symbol} Market Structure - BOS (blue/orange) & CHoCH (purple/red)', color='white', fontsize=14)
    ax.set_xlabel('Date', color='white')
    ax.set_ylabel('Price', color='white')
    ax.tick_params(colors='white')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', dpi=120, facecolor='#1e1e1e', bbox_inches='tight')
        tmpfile.flush()
        with open(tmpfile.name, 'rb') as f:
            img_data = f.read()
    os.unlink(tmpfile.name)
    plt.close(fig)
    return io.BytesIO(img_data)

# ====================
# STRUCTURE COMMAND
# ====================
@bot.command(name='structure')
async def market_structure(ctx, ticker: str, timeframe: str = '4h'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        if ticker.lower() == 'all':
            watchlist = await load_watchlist()
            symbols = watchlist['stocks']
            if not symbols:
                await ctx.send("No stocks in watchlist to scan.")
                return
            if timeframe.lower() == '4hr':
                timeframe = '4h'
            valid_timeframes = ['1h', '4h', 'daily', 'weekly']
            if timeframe not in valid_timeframes:
                await ctx.send(f"Invalid timeframe. Use: {', '.join(valid_timeframes)}")
                return
            await ctx.send(f"🔍 Scanning {len(symbols)} stocks for structure changes on {timeframe}...")
            bos_up, choch_up, bos_down, choch_down = [], [], [], []
            for sym in symbols:
                if await check_cancel(ctx):
                    break
                try:
                    df = await fetch_ohlcv(sym, timeframe)
                    if df is None or len(df) < 50:
                        continue
                    structure = analyze_structure(df)
                    current_price = df['close'].iloc[-1]
                    desc = structure['description'][:100]
                    if structure['last_event'] == 'BOS':
                        if structure['last_event_direction'] == 'up':
                            bos_up.append(f"{sym} (${current_price:.2f}) – {desc}")
                        else:
                            bos_down.append(f"{sym} (${current_price:.2f}) – {desc}")
                    elif structure['last_event'] == 'CHoCH':
                        if structure['last_event_direction'] == 'up':
                            choch_up.append(f"{sym} (${current_price:.2f}) – {desc}")
                        else:
                            choch_down.append(f"{sym} (${current_price:.2f}) – {desc}")
                except Exception as e:
                    print(f"Error scanning {sym}: {e}")
                await asyncio.sleep(0.5)
            embed = discord.Embed(title=f"📊 Market Structure Scan – {timeframe.upper()}", color=0x3498db, timestamp=datetime.now())
            if bos_up:
                embed.add_field(name="🟢 BOS UP (uptrend continuing – HOLD/ADD)", value="\n".join(bos_up[:5]), inline=False)
            if choch_up:
                embed.add_field(name="🟠 CHoCH UP (reversal to uptrend – BUY CALLS)", value="\n".join(choch_up[:5]), inline=False)
            if bos_down:
                embed.add_field(name="🔴 BOS DOWN (downtrend continuing – BUY PUTS)", value="\n".join(bos_down[:5]), inline=False)
            if choch_down:
                embed.add_field(name="🟣 CHoCH DOWN (reversal to downtrend – SELL CALLS / BUY PUTS)", value="\n".join(choch_down[:5]), inline=False)
            if not (bos_up or choch_up or bos_down or choch_down):
                embed.description = "No clear BOS or CHoCH events found."
            await ctx.send(embed=embed)
            return
        symbol = normalize_symbol(ticker)
        if '/' in symbol:
            await ctx.send("Market structure analysis is currently only available for stocks.")
            return
        if timeframe.lower() == '4hr':
            timeframe = '4h'
        valid_timeframes = ['1h', '4h', 'daily', 'weekly']
        if timeframe not in valid_timeframes:
            await ctx.send(f"Invalid timeframe. Use: {', '.join(valid_timeframes)}")
            return
        await ctx.send(f"🔍 Analyzing market structure for **{symbol}** ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}.")
            return
        structure = analyze_structure(df)
        current_price = df['close'].iloc[-1]
        if structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'up':
            action, action_color = "✅ BUY CALLS – Reversal to uptrend detected", 0x00ff00
        elif structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'down':
            action, action_color = "🔴 SELL CALLS / BUY PUTS – Reversal to downtrend detected", 0xff0000
        elif structure['last_event'] == 'BOS' and structure['last_event_direction'] == 'up':
            action, action_color = "📈 HOLD/ADD CALLS – Uptrend continuing", 0x00cc00
        elif structure['last_event'] == 'BOS' and structure['last_event_direction'] == 'down':
            action, action_color = "📉 BUY PUTS – Downtrend continuing", 0xcc0000
        else:
            action, action_color = "⏸️ WAIT – No clear signal", 0xffff00
        embed = discord.Embed(title=f"📈 Market Structure: {symbol} ({timeframe.upper()})",
                              description=f"Current Price: **${current_price:.2f}**\n\n**{action}**", color=action_color)
        embed.add_field(name="Trend", value=structure['trend'].capitalize(), inline=True)
        if structure['last_event']:
            emoji = "🟢" if structure['last_event'] == 'BOS' else "🟠"
            direction = "🔼" if structure['last_event_direction'] == 'up' else "🔽"
            embed.add_field(name="📌 Most Recent Event", value=f"{emoji} {structure['last_event']} {direction}", inline=True)
        if structure['bos_events']:
            bos_text = ""
            for i, bos in enumerate(reversed(structure['bos_events'][-3:]), 1):
                date_str = bos['date'].strftime('%m/%d %H:%M') if hasattr(bos['date'], 'strftime') else str(bos['date'])[:16]
                arrow = "🔼" if bos['direction'] == 'up' else "🔽"
                bos_text += f"{i}. {arrow} BOS {bos['direction'].upper()} at ${bos['price']:.2f} ({date_str})\n"
            embed.add_field(name="📊 Recent BOS (Break of Structure)", value=bos_text, inline=False)
        if structure['choch_events']:
            choch_text = ""
            for i, choch in enumerate(reversed(structure['choch_events'][-3:]), 1):
                date_str = choch['date'].strftime('%m/%d %H:%M') if hasattr(choch['date'], 'strftime') else str(choch['date'])[:16]
                arrow = "🔼" if choch['direction'] == 'up' else "🔽"
                choch_text += f"{i}. {arrow} CHoCH {choch['direction'].upper()} at ${choch['price']:.2f} ({date_str})\n"
            embed.add_field(name="🔄 Recent CHoCH (Change of Character)", value=choch_text, inline=False)
        embed.add_field(name="📖 Analysis", value=structure['description'], inline=False)
        embed.add_field(name="💡 What this means",
                        value="**BOS (Break of Structure)**: Trend continues.\n**CHoCH (Change of Character)**: Trend reverses.\n• BOS up + uptrend = HOLD/ADD CALLS\n• BOS down + downtrend = BUY PUTS\n• CHoCH up + downtrend = BUY CALLS\n• CHoCH down + uptrend = SELL CALLS / BUY PUTS", inline=False)
        web_url = get_tradingview_web_link(symbol)
        embed.add_field(name="📊 TradingView", value=f"[Click here for charts]({web_url})", inline=False)
        chart_buffer = generate_structure_chart(df, symbol, structure)
        if chart_buffer:
            file = discord.File(chart_buffer, filename='structure_chart.png')
            embed.set_image(url='attachment://structure_chart.png')
            await ctx.send(embed=embed, file=file)
        else:
            await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# TREND STRENGTH COMMAND
# ====================
@bot.command(name='strength')
async def trend_strength(ctx, ticker: str, timeframe: str = '4h'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        symbol = normalize_symbol(ticker)
        if '/' in symbol:
            await ctx.send("Crypto not supported for strength analysis yet.")
            return
        valid_tfs = ['1h', '4h', 'daily']
        if timeframe not in valid_tfs:
            await ctx.send(f"Invalid timeframe. Use: 1h, 4h, daily")
            return
        await ctx.send(f"📊 Analyzing trend strength for **{symbol}** on {timeframe}...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}")
            return
        df = calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        ema13, ema50, ema200 = df['ema13'].iloc[-1], df['ema50'].iloc[-1], df['ema200'].iloc[-1]
        price_above_ema13, price_above_ema50, price_above_ema200 = current_price > ema13, current_price > ema50, current_price > ema200
        price_last_5, rsi_last_5 = df['close'].tail(5).values, df['rsi'].tail(5).values
        bearish_divergence = bullish_divergence = False
        if len(price_last_5) >= 2 and len(rsi_last_5) >= 2:
            if price_last_5[-1] > price_last_5[-2] and rsi_last_5[-1] < rsi_last_5[-2]:
                bearish_divergence = True
            if price_last_5[-1] < price_last_5[-2] and rsi_last_5[-1] > rsi_last_5[-2]:
                bullish_divergence = True
        trend_strength_msg = f"🟢 STRONG (ADX: {adx:.1f})" if adx > 25 else f"🟡 MODERATE (ADX: {adx:.1f})" if adx > 20 else f"🔴 WEAK / RANGING (ADX: {adx:.1f})"
        rsi_status = "🔴 OVERBOUGHT (sell signal possible)" if rsi > 70 else "🟢 OVERSOLD (buy signal possible)" if rsi < 30 else f"⚪ NEUTRAL ({rsi:.1f})"
        volume_status = f"🔊 HIGH volume ({volume_ratio:.1f}x average)" if volume_ratio > 1.5 else f"🔇 LOW volume ({volume_ratio:.1f}x average)" if volume_ratio < 0.5 else f"📊 NORMAL volume ({volume_ratio:.1f}x average)"
        embed = discord.Embed(title=f"📊 Trend Strength: {symbol} ({timeframe.upper()})", description=f"Current Price: **${current_price:.2f}**", color=0x3498db, timestamp=datetime.now())
        embed.add_field(name="📈 ADX (Trend Strength)", value=trend_strength_msg, inline=True)
        embed.add_field(name="📉 RSI (Momentum)", value=rsi_status, inline=True)
        embed.add_field(name="🔊 Volume", value=volume_status, inline=True)
        embed.add_field(name="📊 Price vs EMA13", value="🟢 Above" if price_above_ema13 else "🔴 Below", inline=True)
        embed.add_field(name="📊 Price vs EMA50", value="🟢 Above" if price_above_ema50 else "🔴 Below", inline=True)
        embed.add_field(name="📊 Price vs EMA200", value="🟢 Above" if price_above_ema200 else "🔴 Below", inline=True)
        if bearish_divergence:
            embed.add_field(name="⚠️ Bearish Divergence", value="Price making higher highs but RSI making lower highs – potential reversal DOWN", inline=False)
        if bullish_divergence:
            embed.add_field(name="✅ Bullish Divergence", value="Price making lower lows but RSI making higher lows – potential reversal UP", inline=False)
        embed.add_field(name="💡 Trading Advice",
                        value="• **ADX > 25** = Strong trend (trade with trend)\n• **ADX < 20** = Weak trend (avoid, wait for breakout)\n• **RSI > 70** = Overbought (consider taking profits)\n• **RSI < 30** = Oversold (look for buying opportunities)\n• **Divergence** = Early reversal warning\nUse `!structure` for exact BOS/CHoCH signals.", inline=False)
        embed.set_footer(text="ADX > 25 = strong trend • RSI extremes = caution")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# ZONE COMMAND
# ====================
@bot.command(name='zone')
async def zone(ctx, ticker: str, timeframe: str = '30min'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        timeframe = timeframe.lower()
        valid_timeframes = ['5min', '15min', '30min', '1h', '4h', 'daily', 'weekly']
        if timeframe not in valid_timeframes:
            await ctx.send("Invalid timeframe. Use 5min, 15min, 30min, 1h, 4h, daily, or weekly.")
            return
        symbol = normalize_symbol(ticker)
        if timeframe == '30min':
            await ctx.send(f"🔍 Scanning 30‑minute chart for **{symbol}** to find demand zones...")
            try:
                df = await fetch_ohlcv(symbol, '30min')
            except Exception as e:
                await ctx.send(f"❌ Error fetching data for {symbol}: {str(e)}")
                return
            if df is None or df.empty:
                await ctx.send(f"❌ Could not fetch 30min data for {symbol}.")
                return
            current_price = df['close'].iloc[-1]
            try:
                zones = find_demand_zones(df)
            except Exception as e:
                await ctx.send(f"❌ Error analyzing demand zones: {str(e)}")
                return
            struct_df = await fetch_ohlcv(symbol, '4h')
            structure = analyze_structure(struct_df) if struct_df is not None else None
            if not zones:
                await ctx.send(f"No clear demand zones found for {symbol} on 30min.")
                return
            embed = discord.Embed(title=f"📉 Demand Zones for {symbol} (30min)", description=f"Current Price: **${current_price:.2f}**", color=0x00ff00)
            if structure:
                struct_text = f"**{structure['trend'].capitalize()}** – {structure['description']}"
                embed.add_field(name="🏛️ Market Structure (4h)", value=struct_text, inline=False)
            for z in zones:
                distance = (current_price - z['level']) / current_price * 100
                status = "🔵 **NEAR**" if abs(distance) < 2 else ""
                date_str = z['date'].strftime('%m/%d') if hasattr(z['date'], 'strftime') else ''
                embed.add_field(name=f"Support at ${z['level']:.2f} ({date_str})", value=f"Distance: {distance:.1f}% {status}\nTouches: {z['strength']}", inline=False)
            near_zones = [z for z in zones if abs((current_price - z['level']) / current_price) < 0.02]
            if near_zones and '/' not in symbol:
                best_zone = min(near_zones, key=lambda z: abs(current_price - z['level']))
                try:
                    stock = yf.Ticker(symbol)
                    expirations = stock.options
                    if expirations:
                        today = datetime.now().date()
                        primary_exp = None
                        for exp in expirations:
                            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                            dte = (exp_date - today).days
                            if 30 <= dte <= 45:
                                primary_exp = exp
                                break
                        if not primary_exp and expirations:
                            primary_exp = expirations[0]
                        if primary_exp:
                            opt_chain = stock.option_chain(primary_exp)
                            price = current_price
                            offset = 5.0 if price > 100 else 2.0 if price > 50 else 1.0 if price > 10 else max(0.5, price * 0.15)
                            target_strike = price - offset
                            calls = opt_chain.calls
                            if not calls.empty:
                                calls['strike_diff'] = abs(calls['strike'] - target_strike)
                                best_call = calls.loc[calls['strike_diff'].idxmin()]
                                strike, last = best_call['strike'], best_call.get('lastPrice', 'N/A')
                                bid, ask = best_call.get('bid', 'N/A'), best_call.get('ask', 'N/A')
                                volume = best_call.get('volume', 'N/A')
                                premium = (bid + ask) / 2 if bid != 'N/A' and ask != 'N/A' and bid > 0 and ask > 0 else last if last != 'N/A' else None
                                breakeven = strike + premium if premium else 'N/A'
                                option_text = (f"**Strike:** ${strike:.2f}\n**Expiration:** {primary_exp}\n**Last:** {last}\n"
                                               f"**Bid/Ask:** {bid}/{ask}\n**Volume:** {volume}\n**Est. Premium:** ${premium:.2f}\n"
                                               f"**Breakeven:** ${breakeven:.2f}" if breakeven != 'N/A' else "Breakeven N/A")
                                embed.add_field(name="💡 Suggested Call Option (ITM)", value=option_text, inline=False)
                except Exception as e:
                    embed.add_field(name="Options suggestion", value=f"Could not fetch options: {str(e)}", inline=False)
            if structure and structure['last_event'] == 'BOS' and structure['trend'] == 'downtrend':
                embed.add_field(name="⚠️ Trading Advice", value="**Downtrend with BOS – trend continuing DOWN.** Consider PUTS or stay away.", inline=False)
            elif structure and structure['last_event'] == 'CHoCH' and structure['trend'] == 'downtrend' and structure['last_event_direction'] == 'up':
                embed.add_field(name="📈 Trading Advice", value="**Change of Character (CHoCH) detected – reversal to UPTREND.** Consider CALLS.", inline=False)
            elif structure and structure['trend'] == 'uptrend':
                embed.add_field(name="📈 Trading Advice", value="**Uptrend confirmed.** This demand zone is a potential bounce area. Consider CALLS.", inline=False)
            try:
                chart_buffer = generate_zone_chart(df, symbol, zones)
                if chart_buffer:
                    file = discord.File(chart_buffer, filename='zone_chart.png')
                    embed.set_image(url='attachment://zone_chart.png')
                    embed.set_footer(text="⚠️ Options are risky. This is not financial advice.")
                    await ctx.send(embed=embed, file=file)
                else:
                    embed.set_footer(text="⚠️ Options are risky. This is not financial advice.")
                    await ctx.send(embed=embed)
            except Exception as e:
                await ctx.send(f"❌ Error generating chart: {str(e)}")
            return
        await ctx.send(f"🔍 Fetching zones for **{symbol}** ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}.")
            return
        df = calculate_indicators(df)
        signals = get_signals(df)
        embed = format_zone_embed(symbol, signals, timeframe)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ An unexpected error occurred: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# LEAPS COMMAND
# ====================
@bot.command(name='leaps')
async def leaps_candidates(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        watchlist = await load_watchlist()
        symbols = watchlist['stocks']
        if not symbols:
            await ctx.send("No stocks in watchlist to scan.")
            return
        await ctx.send(f"🔍 **SCANNING {len(symbols)} SYMBOLS FOR LEAPS CANDIDATES**")
        await ctx.send("Looking for stocks with recent Change of Character (CHoCH) on the weekly chart...\n")
        candidates = []
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            try:
                df = await fetch_ohlcv(symbol, 'weekly')
                if df is None or len(df) < 30:
                    continue
                structure = analyze_structure(df)
                if structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'up' and structure['trend'] == 'downtrend':
                    fin_data = await fetch_analyst_ratings(symbol, limit=1)
                    rating_info = ""
                    if fin_data and len(fin_data) > 0:
                        r = fin_data[0]
                        total = r.get('strongBuy',0) + r.get('buy',0) + r.get('hold',0) + r.get('sell',0) + r.get('strongSell',0)
                        if total > 0:
                            buys = r.get('strongBuy',0) + r.get('buy',0)
                            rating_info = f"Analyst consensus: {buys} Buy / {r.get('hold',0)} Hold / {r.get('sell',0)+r.get('strongSell',0)} Sell"
                    candidates.append((symbol, structure, rating_info))
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue
        if not candidates:
            await ctx.send("No LEAPS candidates found with a clear Change of Character.")
            return
        embed = discord.Embed(title="📈 LEAPS CANDIDATES", description="Stocks showing Change of Character (CHoCH) on the weekly chart – potential long-term reversal setups.", color=0x00ff00)
        for symbol, structure, rating_info in candidates[:8]:
            price = await get_stock_price(symbol)
            if price is None:
                continue
            try:
                stock = yf.Ticker(symbol)
                expirations = stock.options
                if expirations:
                    exp_dates = [datetime.strptime(e, '%Y-%m-%d').date() for e in expirations]
                    today = datetime.now().date()
                    leaps_exp = None
                    for exp_date, exp_str in sorted(zip(exp_dates, expirations), key=lambda x: x[0]):
                        dte = (exp_date - today).days
                        if dte >= 365:
                            leaps_exp = exp_str
                            break
                    if leaps_exp:
                        opt_chain = stock.option_chain(leaps_exp)
                        calls = opt_chain.calls
                        if not calls.empty:
                            calls['diff'] = abs(calls['strike'] - price)
                            best_call = calls.loc[calls['diff'].idxmin()]
                            strike = best_call['strike']
                            last = best_call.get('lastPrice', 'N/A')
                            bid, ask = best_call.get('bid', 'N/A'), best_call.get('ask', 'N/A')
                            premium = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                            option_text = f"**{strike:.2f} Call**\nExp: {leaps_exp}\nPremium: ${premium:.2f} (if available)"
                        else:
                            option_text = "No calls available for this expiration."
                    else:
                        option_text = "No LEAPS expiration found (1+ year)."
                else:
                    option_text = "No options available."
            except Exception as e:
                option_text = f"Error fetching options: {str(e)}"
            embed.add_field(name=f"{symbol} – ${price:.2f}", value=f"{structure['description']}\n{rating_info}\n\n💡 Suggested LEAPS: {option_text}", inline=False)
        embed.set_footer(text="LEAPS are long-term options (1-3 years). Use after a confirmed CHoCH to catch major reversals.")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error scanning LEAPS: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# FINVIZ SCANNER
# ====================
@bot.command(name='finviz_scan')
async def finviz_scan(ctx, *, filters: str = None):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        if not filters:
            help_embed = discord.Embed(title="🔎 Finviz Scanner Help", description="Use `!finviz_scan` followed by a comma‑separated list of filters.\n\n**Example:**\n`!finviz_scan Price: $10 to $20, Option/Short: Optionable, Average Volume: Over 1M, Relative Volume: Over 1.5, Change: Up 3%`\n\n**Available filters (common ones):**\n- `Price: $10 to $20`\n- `Option/Short: Optionable`\n- `Average Volume: Over 1M`\n- `Relative Volume: Over 1.5`\n- `Change: Up 3%`\n- `Market Cap.: +Mid (over $2bln)`\n- `EPS growth this year: Positive`\n- `RSI (14): Over 50`\n- `20-Day Simple Moving Average: Price above SMA20`\n\n**Tip:** You can combine many filters. The scanner will return up to 50 results.", color=0x3498db)
            await ctx.send(embed=help_embed)
            return
        await ctx.send("🔎 Running Finviz scan... This may take a few seconds.")
        filter_list = [f.strip() for f in filters.split(',') if f.strip()]
        if not filter_list:
            await ctx.send("❌ No valid filters provided.")
            return
        filter_dict = {}
        for f in filter_list:
            if ':' in f:
                key, value = f.split(':', 1)
                filter_dict[key.strip()] = value.strip()
            else:
                filter_dict[f] = ""
        fov = Overview()
        fov.set_filter(filters_dict=filter_dict)
        df = fov.screener_view()
        if df.empty:
            await ctx.send("📭 No stocks found matching your filters.")
            return
        df = df.head(25)
        embed = discord.Embed(title="📊 Finviz Scan Results", description=f"Filters used: {filters[:200]}{'...' if len(filters) > 200 else ''}", color=0x00ff00, timestamp=datetime.now())
        results_text = ""
        for idx, row in df.iterrows():
            ticker = row.get('Ticker', 'N/A')
            price = row.get('Price', 'N/A')
            change = row.get('Change', 'N/A')
            volume = row.get('Volume', 'N/A')
            rel_vol = row.get('Rel Volume', 'N/A')
            market_cap = row.get('Market Cap', 'N/A')
            results_text += f"**{ticker}** – ${price} ({change}) | Vol: {volume} | RelVol: {rel_vol} | MCap: {market_cap}\n"
        if results_text:
            add_field_safe(embed, "📈 Top Matches", results_text, inline=False)
        else:
            embed.add_field(name="📈 Top Matches", value="No data returned", inline=False)
        embed.set_footer(text="Use !structure TICKER or !flow TICKER for deeper analysis")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Finviz scan error: {str(e)}")
        print(f"Finviz error: {e}")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='cheap_options')
async def cheap_options(ctx):
    filters = "Price: $10 to $20, Option/Short: Optionable, Average Volume: Over 1M, Relative Volume: Over 1.5, Change: Up 3%"
    await finviz_scan(ctx, filters=filters)

@bot.command(name='hype_stocks')
async def hype_stocks(ctx):
    filters = "Relative Volume: Over 2, Change: Up 5%, Option/Short: Optionable, Average Volume: Over 500K"
    await finviz_scan(ctx, filters=filters)

@bot.command(name='cheap_plays')
async def cheap_plays(ctx):
    await cheap_options(ctx)

# ====================
# UPCOMING COMMAND
# ====================
async def get_earnings_stats(symbol, earnings_date):
    try:
        stock = yf.Ticker(symbol)
        earn_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
        today = datetime.now()
        expirations = stock.options
        if not expirations:
            return "N/A", "N/A"
        target_exp = None
        for exp in expirations:
            exp_dt = datetime.strptime(exp, '%Y-%m-%d')
            if exp_dt >= earn_dt:
                target_exp = exp
                break
        if not target_exp:
            return "N/A", "N/A"
        opt_chain = stock.option_chain(target_exp)
        calls, puts = opt_chain.calls, opt_chain.puts
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        if calls.empty or puts.empty:
            return "N/A", "N/A"
        calls['diff'], puts['diff'] = abs(calls['strike'] - current_price), abs(puts['strike'] - current_price)
        atm_call, atm_put = calls.loc[calls['diff'].idxmin()], puts.loc[puts['diff'].idxmin()]
        call_price = (atm_call['bid'] + atm_call['ask']) / 2 if atm_call['bid'] > 0 and atm_call['ask'] > 0 else atm_call['lastPrice']
        put_price = (atm_put['bid'] + atm_put['ask']) / 2 if atm_put['bid'] > 0 and atm_put['ask'] > 0 else atm_put['lastPrice']
        straddle = call_price + put_price
        expected_pct = (straddle / current_price) * 100
        return f"{expected_pct:.1f}%", "N/A"
    except Exception as e:
        print(f"Error getting earnings stats: {e}")
        return "N/A", "N/A"

@bot.command(name='upcoming')
async def upcoming_events(ctx, ticker: str):
    """Show ALL upcoming events for a stock: earnings, dividends, splits, and analyst ratings."""
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now

        symbol = ticker.upper()
        await ctx.send(f"🔍 Fetching upcoming events for **{symbol}**...")

        # Fetch all data in parallel
        stock = yf.Ticker(symbol)
        
        # 1. Get earnings dates (all upcoming)
        earnings_list = []
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                today = datetime.now().date()
                for date, row in earnings_dates.iterrows():
                    e_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    if e_date >= today:  # Only future dates
                        eps_est = row.get('epsEstimated') if 'epsEstimated' in row else row.get('epsEstimate')
                        eps_est = 'N/A' if eps_est is None or pd.isna(eps_est) else f"{eps_est:.2f}"
                        # Calculate expected move if options available
                        exp_move = "N/A"
                        try:
                            expirations = stock.options
                            if expirations:
                                # Find closest expiration after earnings
                                target_exp = None
                                for exp in expirations:
                                    exp_dt = datetime.strptime(exp, '%Y-%m-%d').date()
                                    if exp_dt >= e_date:
                                        target_exp = exp
                                        break
                                if target_exp:
                                    opt_chain = stock.option_chain(target_exp)
                                    calls, puts = opt_chain.calls, opt_chain.puts
                                    current_price = stock.history(period="1d")['Close'].iloc[-1]
                                    if not calls.empty and not puts.empty:
                                        calls['diff'] = abs(calls['strike'] - current_price)
                                        puts['diff'] = abs(puts['strike'] - current_price)
                                        atm_call = calls.loc[calls['diff'].idxmin()]
                                        atm_put = puts.loc[puts['diff'].idxmin()]
                                        call_price = (atm_call['bid'] + atm_call['ask']) / 2 if atm_call['bid'] > 0 and atm_call['ask'] > 0 else atm_call['lastPrice']
                                        put_price = (atm_put['bid'] + atm_put['ask']) / 2 if atm_put['bid'] > 0 and atm_put['ask'] > 0 else atm_put['lastPrice']
                                        straddle = call_price + put_price
                                        exp_move = f"{(straddle / current_price * 100):.1f}%"
                        except:
                            pass
                        earnings_list.append({
                            'date': e_date.strftime('%Y-%m-%d'),
                            'eps_est': eps_est,
                            'expected_move': exp_move
                        })
        except Exception as e:
            print(f"Error fetching earnings: {e}")

        # 2. Get dividends (upcoming)
        dividends_list = []
        try:
            dividends = stock.dividends
            if not dividends.empty:
                today = datetime.now().date()
                for date, amount in dividends.items():
                    d_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    if d_date >= today:
                        dividends_list.append({
                            'date': d_date.strftime('%Y-%m-%d'),
                            'amount': f"${amount:.2f}"
                        })
        except Exception as e:
            print(f"Error fetching dividends: {e}")

        # 3. Get splits (upcoming)
        splits_list = []
        try:
            splits = stock.splits
            if not splits.empty:
                today = datetime.now().date()
                for date, ratio in splits.items():
                    s_date = date.date() if hasattr(date, 'date') else datetime.strptime(str(date), '%Y-%m-%d').date()
                    if s_date >= today:
                        ratio_str = f"{ratio:.0f}:1" if ratio >= 1 else f"1:{int(1/ratio)}"
                        splits_list.append({
                            'date': s_date.strftime('%Y-%m-%d'),
                            'ratio': ratio_str
                        })
        except Exception as e:
            print(f"Error fetching splits: {e}")

        # 4. Get analyst ratings
        ratings = await fetch_analyst_ratings(symbol, limit=3)

        # Check if we found anything
        if not earnings_list and not dividends_list and not splits_list and not ratings:
            await ctx.send(f"📭 No upcoming events found for {symbol}.")
            return

        embed = discord.Embed(
            title=f"📅 Upcoming Catalysts for {symbol}",
            color=0x00ff00,
            timestamp=datetime.now()
        )

        # Earnings section
        if earnings_list:
            earnings_text = ""
            for e in earnings_list[:3]:  # Show next 3 earnings
                earnings_text += f"**{e['date']}** – EPS Est: {e['eps_est']} | Expected Move: {e['expected_move']}\n"
            embed.add_field(name="📊 Earnings (Upcoming)", value=earnings_text, inline=False)

        # Dividends section
        if dividends_list:
            div_text = ""
            for d in dividends_list[:2]:
                div_text += f"**{d['date']}** – Amount: {d['amount']}\n"
            embed.add_field(name="💰 Dividends", value=div_text, inline=False)

        # Splits section
        if splits_list:
            split_text = ""
            for s in splits_list[:2]:
                split_text += f"**{s['date']}** – Ratio: {s['ratio']}\n"
            embed.add_field(name="🔄 Stock Splits", value=split_text, inline=False)

        # Analyst Ratings
        if ratings:
            ratings_text = ""
            for r in ratings[:3]:
                period = r.get('period', '')
                sb = r.get('strongBuy', 0)
                b = r.get('buy', 0)
                h = r.get('hold', 0)
                s = r.get('sell', 0)
                ss = r.get('strongSell', 0)
                total = sb + b + h + s + ss
                buys = sb + b
                sells = s + ss
                sentiment = "🟢" if buys > sells else "🔴" if sells > buys else "⚪"
                if total > 0:
                    ratings_text += f"**{period}** – {buys} Buy / {h} Hold / {sells} Sell {sentiment}\n"
                else:
                    ratings_text += f"**{period}** – No data\n"
            if ratings_text:
                embed.add_field(name="📈 Analyst Ratings (Last 3 Months)", value=ratings_text, inline=False)

        web_url = get_tradingview_web_link(symbol)
        embed.add_field(name="📊 TradingView", value=f"[Click here for charts]({web_url})", inline=False)
        embed.set_footer(text="Expected move = implied volatility from options")

        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# BACKTESTING COMMAND
# ====================
@bot.command(name='backtest')
async def backtest(ctx, symbol: str, days: int = 365, cost: float = 0.001):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send(f"⏳ Backtesting **{symbol.upper()}** over the last {days} days (cost: {cost*100:.1f}%)...")
        ticker = yf.Ticker(symbol.upper())
        df = ticker.history(period=f"{days}d", interval="1d")
        if df.empty:
            await ctx.send("❌ No historical data found.")
            return
        df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = calculate_indicators(df)
        signals_list = [get_signals(df.iloc[:i+1]) for i in range(len(df)) if get_signals(df.iloc[:i+1])]
        dates = df.index.tolist()
        equity, in_position, entry_price, entry_date, position_type, trades, equity_dates = [1.0], False, 0, None, None, [], [dates[0]]
        for i in range(1, len(signals_list)):
            sig, prev_sig = signals_list[i], signals_list[i-1]
            today_open = df['open'].iloc[i] if i < len(df) else None
            if not in_position:
                if prev_sig['net_score'] > 0 and today_open is not None:
                    in_position, position_type, entry_price, entry_date = True, 'long', today_open, df.index[i]
                    equity[-1] *= (1 - cost)
                elif prev_sig['net_score'] < 0 and today_open is not None:
                    in_position, position_type, entry_price, entry_date = True, 'short', today_open, df.index[i]
                    equity[-1] *= (1 - cost)
            elif in_position:
                exit_signal = (position_type == 'long' and prev_sig['net_score'] < 0) or (position_type == 'short' and prev_sig['net_score'] > 0)
                if exit_signal:
                    exit_price = today_open if today_open is not None else df['close'].iloc[-1]
                    exit_date = df.index[i] if today_open is not None else df.index[-1]
                    ret = (exit_price - entry_price) / entry_price if position_type == 'long' else (entry_price - exit_price) / entry_price
                    new_equity = equity[-1] * (1 + ret) * (1 - cost)
                    equity.append(new_equity)
                    equity_dates.append(exit_date)
                    trades.append({'entry_date': entry_date, 'exit_date': exit_date, 'type': position_type, 'ret': ret})
                    in_position = False
        if in_position:
            exit_price = df['close'].iloc[-1]
            exit_date = df.index[-1]
            ret = (exit_price - entry_price) / entry_price if position_type == 'long' else (entry_price - exit_price) / entry_price
            equity.append(equity[-1] * (1 + ret) * (1 - cost))
            equity_dates.append(exit_date)
            trades.append({'entry_date': entry_date, 'exit_date': exit_date, 'type': position_type, 'ret': ret})
        if not trades:
            await ctx.send("No trades generated during this period.")
            return
        final_equity, total_return = equity[-1], (equity[-1] - 1) * 100
        winning_trades = [t for t in trades if t['ret'] > 0]
        losing_trades = [t for t in trades if t['ret'] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = np.mean([t['ret']*100 for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['ret']*100 for t in losing_trades]) if losing_trades else 0
        gross_profit, gross_loss = sum(t['ret'] for t in winning_trades), abs(sum(t['ret'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        peak = np.maximum.accumulate(equity)
        max_drawdown = np.max((peak - equity) / peak * 100)
        embed = discord.Embed(title=f"📈 BACKTEST RESULTS: {symbol.upper()}", description=f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\nTransaction cost: {cost*100:.1f}% per trade", color=0x00ff00 if total_return > 0 else 0xff0000)
        embed.add_field(name="Total Return", value=f"{total_return:.2f}%", inline=True)
        embed.add_field(name="Win Rate", value=f"{win_rate:.1f}%", inline=True)
        embed.add_field(name="Profit Factor", value=f"{profit_factor:.2f}", inline=True)
        embed.add_field(name="Number of Trades", value=str(len(trades)), inline=True)
        embed.add_field(name="Avg Win", value=f"{avg_win:.2f}%", inline=True)
        embed.add_field(name="Avg Loss", value=f"{avg_loss:.2f}%", inline=True)
        embed.add_field(name="Max Drawdown", value=f"{max_drawdown:.2f}%", inline=True)
        sample = trades[:5]
        trade_list = "\n".join([f"{t['entry_date'].strftime('%m/%d')} {t['type']} {t['ret']*100:+.2f}%" for t in sample])
        embed.add_field(name="Sample Trades", value=trade_list or "None", inline=False)
        await ctx.send(embed=embed)
        plt.figure(figsize=(10, 5))
        plt.plot(equity_dates, equity, color='blue', linewidth=2)
        plt.title(f'{symbol.upper()} Equity Curve (Backtest)')
        plt.ylabel('Equity ($)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        plt.close()
        file = discord.File(buf, filename='equity.png')
        await ctx.send(file=file)
    except Exception as e:
        await ctx.send(f"❌ Backtest error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !signal
# ====================
@bot.command(name='signal')
async def signal_single(ctx, ticker: str):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        symbol = normalize_symbol(ticker)
        await ctx.send(f"🔍 Fetching multi-timeframe signals for **{symbol}**...")
        all_timeframes = ['5min', '15min', '30min', '1h', '4h', 'daily', 'weekly']
        symbol_signals = {}
        for tf in all_timeframes:
            df = await fetch_ohlcv(symbol, tf)
            if df is not None and not df.empty:
                df_calc = calculate_indicators(df)
                sig = get_signals(df_calc)
                if sig and sig['net_score'] != 0:
                    symbol_signals[tf] = {'signals': sig, 'df': df}
            await asyncio.sleep(0.5)
        if not symbol_signals:
            await ctx.send(f"📭 No active signals found for {symbol} on any timeframe.")
            return
        await send_combined_symbol_report(ctx, symbol, symbol_signals)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !signals
# ====================
@bot.command(name='signals')
async def signals(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        watchlist = await load_watchlist()
        symbols = watchlist['stocks'] + watchlist['crypto']
        all_timeframes = ['5min', '15min', '30min', '1h', '4h', 'daily', 'weekly']
        await ctx.send(f"🔍 **MULTI-TIMEFRAME SIGNAL SCAN**")
        await ctx.send(f"📊 Scanning **{len(symbols)}** symbols across **ALL {len(all_timeframes)} timeframes**")
        await ctx.send(f"⏱️ Timeframes: 5min, 15min, 30min, 1h, 4h, daily, weekly")
        await ctx.send(f"📈 Using Alpaca (high limits) for stocks – this will be fast!")
        await ctx.send("⏳ Results will appear as they come...\n")
        all_symbol_signals = defaultdict(dict)
        found_any = False
        for symbol in symbols:
            if await check_cancel(ctx):
                await ctx.send("🛑 Scan cancelled after processing the last symbol.")
                break
            for tf in all_timeframes:
                df = await fetch_ohlcv(symbol, tf)
                if df is not None and not df.empty:
                    df_calc = calculate_indicators(df)
                    sig = get_signals(df_calc)
                    if sig and sig['net_score'] != 0:
                        found_any = True
                        if symbol not in all_symbol_signals:
                            all_symbol_signals[symbol] = {}
                        all_symbol_signals[symbol][tf] = {'signals': sig, 'df': df}
                await asyncio.sleep(0.5)
            if symbol in all_symbol_signals:
                await send_combined_symbol_report(ctx, symbol, all_symbol_signals[symbol])
        if not found_any and not cancellation_flags.get(ctx.author.id, False):
            await ctx.send(f"📭 No symbols with active signals found.")
        cancellation_flags[ctx.author.id] = False
        if not cancellation_flags.get(ctx.author.id, False):
            await ctx.send(f"✅ Signal scan complete!")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !scan
# ====================
@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        timeframe = timeframe.lower()
        valid_timeframes = ['5min', '15min', '30min', '1h', '4h', 'daily', 'weekly']
        if timeframe not in valid_timeframes:
            await ctx.send("Invalid timeframe. Use 5min, 15min, 30min, 1h, 4h, daily, or weekly.")
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
            df_calc = calculate_indicators(df)
            signals = get_signals(df_calc)
            peg_str = None
            if '/' not in symbol:
                _, peg_str = await get_peg_ratio(symbol)
            embed = format_embed(symbol, signals, timeframe, peg_str=peg_str)
            try:
                chart_buffer = generate_chart_image(df, symbol, timeframe)
                if chart_buffer:
                    file = discord.File(chart_buffer, filename='chart.png')
                    embed.set_image(url='attachment://chart.png')
                    await ctx.send(embed=embed, file=file)
                else:
                    await ctx.send(embed=embed)
            except Exception as e:
                print(f"Chart generation failed: {e}")
                await ctx.send(embed=embed)
            return
        await ctx.send(f"Scanning all symbols ({len(symbols)}) on {timeframe} timeframe...")
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            df = await fetch_ohlcv(symbol, timeframe)
            if df is not None and not df.empty:
                df_calc = calculate_indicators(df)
                signals = get_signals(df_calc)
                peg_str = None
                if '/' not in symbol:
                    _, peg_str = await get_peg_ratio(symbol)
                embed = format_embed(symbol, signals, timeframe, peg_str=peg_str)
                try:
                    chart_buffer = generate_chart_image(df, symbol, timeframe)
                    if chart_buffer:
                        file = discord.File(chart_buffer, filename='chart.png')
                        embed.set_image(url='attachment://chart.png')
                        await ctx.send(embed=embed, file=file)
                    else:
                        await ctx.send(embed=embed)
                except Exception as e:
                    print(f"Chart generation failed: {e}")
                    await ctx.send(embed=embed)
            await asyncio.sleep(1)
        cancellation_flags[ctx.author.id] = False
        await ctx.send("Scan complete.")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# ZONE HELPER
# ====================
def find_demand_zones(df, lookback=200, threshold_percentile=90, touch_tolerance=0.005):
    if len(df) < 50:
        return []
    df = df.iloc[-lookback:].copy()
    df['range'] = df['high'] - df['low']
    threshold = np.percentile(df['range'].dropna(), threshold_percentile)
    large_candles = df[df['range'] > threshold]
    zones = []
    for idx, row in large_candles.iterrows():
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

# ====================
# TRADE TRACKING COMMANDS
# ====================
@bot.command(name='track')
async def track_trade(ctx, action: str, symbol: str, *, details: str = None):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        action, symbol = action.upper(), symbol.upper()
        if action in ['BUY', 'OPEN']:
            parts = details.split() if details else []
            option_type = parts[0].upper() if parts and parts[0] in ['CALL', 'PUT'] else None
            strike, premium = None, None
            for i, p in enumerate(parts):
                if p.lower() == 'strike' and i + 1 < len(parts):
                    strike = float(parts[i + 1])
                elif p.replace('.', '').isdigit() and not p.lower() == 'strike':
                    if premium is None:
                        premium = float(p)
            trade = {'user_id': str(ctx.author.id), 'symbol': symbol, 'action': 'BUY', 'option_type': option_type,
                     'strike': strike, 'premium': premium, 'entry_date': now, 'entry_price': premium, 'status': 'OPEN', 'strategy': None}
            await trades_collection.insert_one(trade)
            opt_text = f" {option_type} ${strike:.0f} strike" if option_type and strike else ""
            await ctx.send(f"✅ Tracked: **BUY {symbol}{opt_text} at ${premium:.2f}**" if premium else f"✅ Tracked: **BUY {symbol}**")
        elif action in ['SELL', 'CLOSE']:
            open_trade = await trades_collection.find_one({'user_id': str(ctx.author.id), 'symbol': symbol, 'status': 'OPEN'}, sort=[('entry_date', -1)])
            if not open_trade:
                await ctx.send(f"❌ No open trade found for {symbol}. Use `!track BUY {symbol}` first.")
                return
            if details:
                if details.endswith('%'):
                    profit_pct = float(details[:-1])
                    entry_premium = open_trade.get('premium', 0)
                    exit_premium = entry_premium * (1 + profit_pct / 100) if entry_premium > 0 else None
                else:
                    exit_premium = float(details)
                    profit_pct = ((exit_premium - open_trade.get('premium', 0)) / open_trade.get('premium', 1)) * 100 if open_trade.get('premium') else None
            else:
                exit_premium, profit_pct = None, None
            await trades_collection.update_one({'_id': open_trade['_id']}, {'$set': {'status': 'CLOSED', 'exit_date': now,
                'exit_premium': exit_premium, 'profit_pct': profit_pct,
                'profit_amount': (exit_premium - open_trade.get('premium', 0)) * 100 if exit_premium and open_trade.get('premium') else None}})
            if profit_pct is not None:
                emoji = "🟢" if profit_pct > 0 else "🔴" if profit_pct < 0 else "⚪"
                await ctx.send(f"✅ Closed **{symbol}** trade: {emoji} {profit_pct:+.1f}%")
            else:
                await ctx.send(f"✅ Closed **{symbol}** trade.")
        else:
            await ctx.send("❌ Invalid action. Use `!track BUY SYMBOL` or `!track CLOSE SYMBOL`")
    except Exception as e:
        await ctx.send(f"❌ Error tracking trade: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='trades')
async def list_trades(ctx, symbol: str = None, limit: int = 10):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        query = {'user_id': str(ctx.author.id)}
        if symbol:
            query['symbol'] = symbol.upper()
        trades = await trades_collection.find(query).sort('entry_date', -1).to_list(length=limit)
        if not trades:
            await ctx.send(f"📭 No trades found. Start tracking with `!track BUY SYMBOL`")
            return
        title = f"📋 Recent Trades - {symbol.upper()}" if symbol else "📋 Recent Trades"
        embed = discord.Embed(title=title, color=0x3498db, timestamp=datetime.now())
        trade_list = ""
        for t in trades[:limit]:
            status = "🟢 OPEN" if t.get('status') == 'OPEN' else "🔒 CLOSED"
            profit_str = f"{t.get('profit_pct', 0):+.1f}%" if t.get('profit_pct') is not None else "N/A"
            entry_date = t['entry_date'].strftime('%m/%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
            opt_info = f" {t['option_type']} ${t.get('strike', 0):.0f} strike" if t.get('option_type') else ""
            trade_list += f"**{t['symbol']}**{opt_info} {status} | Entry: ${t.get('premium', 0):.2f} | P&L: {profit_str} | {entry_date}\n"
        embed.add_field(name="Trades", value=trade_list or "None", inline=False)
        embed.set_footer(text="Use !performance for detailed stats | !performance NIO for symbol stats")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='performance', aliases=['stats', 'perf'])
async def performance(ctx, symbol: str = None, days: int = 30):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        if symbol and symbol.isdigit():
            days, symbol = int(symbol), None
        now, cutoff = datetime.now(), datetime.now() - timedelta(days=days)
        query = {'user_id': str(ctx.author.id), 'status': 'CLOSED', 'exit_date': {'$gte': cutoff}}
        if symbol:
            query['symbol'] = symbol.upper()
        trades = await trades_collection.find(query).to_list(length=1000)
        if not trades:
            await ctx.send(f"📭 No closed trades found. Start tracking with `!track BUY SYMBOL`")
            return
        total_trades, winning_trades = len(trades), [t for t in trades if t.get('profit_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_pct', 0) < 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        total_profit = sum(t.get('profit_pct', 0) for t in trades)
        avg_win = sum(t.get('profit_pct', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum(t.get('profit_pct', 0) for t in losing_trades) / len(losing_trades)) if losing_trades else 0
        gross_profit, gross_loss = sum(t.get('profit_pct', 0) for t in winning_trades), abs(sum(t.get('profit_pct', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        best_trade = max(trades, key=lambda x: x.get('profit_pct', -999)) if trades else None
        worst_trade = min(trades, key=lambda x: x.get('profit_pct', 999)) if trades else None
        title = f"📊 Trading Performance{f' - {symbol.upper()}' if symbol else ''} (Last {days} days)"
        embed = discord.Embed(title=title, color=0x00ff00 if total_profit > 0 else 0xff0000, timestamp=now)
        embed.add_field(name="Total Trades", value=str(total_trades), inline=True)
        embed.add_field(name="Win Rate", value=f"{win_rate:.1f}%", inline=True)
        embed.add_field(name="Total P&L", value=f"{total_profit:+.1f}%", inline=True)
        embed.add_field(name="Avg Win", value=f"{avg_win:+.1f}%", inline=True)
        embed.add_field(name="Avg Loss", value=f"{avg_loss:+.1f}%", inline=True)
        embed.add_field(name="Profit Factor", value=f"{profit_factor:.2f}", inline=True)
        if best_trade:
            embed.add_field(name="🏆 Best Trade", value=f"{best_trade['symbol']}: {best_trade.get('profit_pct', 0):+.1f}%", inline=True)
        if worst_trade:
            embed.add_field(name="📉 Worst Trade", value=f"{worst_trade['symbol']}: {worst_trade.get('profit_pct', 0):+.1f}%", inline=True)
        if not symbol:
            symbol_stats = {}
            for t in trades:
                sym = t['symbol']
                if sym not in symbol_stats:
                    symbol_stats[sym] = {'trades': 0, 'wins': 0, 'profit': 0}
                symbol_stats[sym]['trades'] += 1
                symbol_stats[sym]['profit'] += t.get('profit_pct', 0)
                if t.get('profit_pct', 0) > 0:
                    symbol_stats[sym]['wins'] += 1
            top_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['profit'], reverse=True)[:3]
            if top_symbols:
                symbol_text = ""
                for sym, stats in top_symbols:
                    win_pct = stats['wins'] / stats['trades'] * 100
                    symbol_text += f"**{sym}**: {stats['profit']:+.1f}% ({stats['wins']}/{stats['trades']} wins, {win_pct:.0f}%)\n"
                embed.add_field(name="📈 Best Symbols", value=symbol_text, inline=False)
        embed.set_footer(text="Track trades with !track BUY SYMBOL and !track CLOSE SYMBOL")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# CONFIRMATION COMMAND - COMBINES ALL SIGNALS
# ====================
@bot.command(name='confirm')
async def trade_confirmation(ctx, ticker: str, timeframe: str = '4h'):
    """
    Comprehensive trade confirmation combining all bot signals.
    Returns a score from 0-100 and actionable recommendation.
    Usage: !confirm NVDA 4h
    """
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now

        symbol = normalize_symbol(ticker)
        if '/' in symbol:
            await ctx.send("Confirmation analysis currently only available for stocks.")
            return

        await ctx.send(f"🔍 Running comprehensive trade confirmation for **{symbol}** ({timeframe})...")

        # Fetch all data
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}.")
            return

        df = calculate_indicators(df)
        current_price = df['close'].iloc[-1]
        
        # 1. Market Structure Analysis
        structure = analyze_structure(df)
        
        # 2. Trend Strength
        rsi = df['rsi'].iloc[-1]
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
        
        # 3. Volume Analysis
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # 4. Price vs EMAs
        ema13 = df['ema13'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]
        above_ema13 = current_price > ema13
        above_ema50 = current_price > ema50
        above_ema200 = current_price > ema200
        
        # 5. Demand Zones (30min)
        df_30 = await fetch_ohlcv(symbol, '30min')
        zones = find_demand_zones(df_30) if df_30 is not None else []
        near_zone = any(abs((current_price - z['level']) / current_price) < 0.02 for z in zones)
        
        # 6. Options Flow (check for unusual activity)
        unusual_flow = False
        try:
            expirations = get_key_expirations(symbol)
            for exp_str, dte, label in expirations[:1]:
                stock = yf.Ticker(symbol)
                opt_chain = stock.option_chain(exp_str)
                calls = opt_chain.calls
                if not calls.empty:
                    for _, opt in calls.iterrows():
                        volume = opt.get('volume', 0)
                        oi = opt.get('openInterest', 0)
                        if oi > 0 and volume / oi > 1.5:
                            unusual_flow = True
                            break
        except:
            pass
        
        # 7. Earnings Proximity
        earnings = await fetch_earnings_upcoming(symbol, days=7)
        earnings_soon = len(earnings) > 0
        
        # 8. Market Regime (SPY trend)
        spy_df = await fetch_ohlcv('SPY', 'daily')
        spy_trend = "bullish"
        if spy_df is not None and len(spy_df) > 20:
            spy_close = spy_df['close'].iloc[-1]
            spy_20_ago = spy_df['close'].iloc[-20]
            spy_trend = "bullish" if spy_close > spy_20_ago else "bearish"
        
        # 9. Historical Win Rate from tracked trades
        historical_win_rate = 0
        similar_trades = await trades_collection.find({
            'user_id': str(ctx.author.id),
            'status': 'CLOSED',
            'symbol': symbol
        }).to_list(length=100)
        if similar_trades:
            wins = sum(1 for t in similar_trades if t.get('profit_pct', 0) > 0)
            historical_win_rate = wins / len(similar_trades) * 100
        
        # Calculate confirmation score (0-100)
        score = 0
        reasons_bullish = []
        reasons_bearish = []
        
        # Structure signal (25 points max)
        if structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'up':
            score += 25
            reasons_bullish.append("✅ CHoCH UP detected – bullish reversal signal")
        elif structure['last_event'] == 'BOS' and structure['last_event_direction'] == 'up':
            score += 20
            reasons_bullish.append("✅ BOS UP – uptrend continuing")
        elif structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'down':
            score -= 25
            reasons_bearish.append("🔴 CHoCH DOWN – bearish reversal signal")
        elif structure['last_event'] == 'BOS' and structure['last_event_direction'] == 'down':
            score -= 20
            reasons_bearish.append("🔴 BOS DOWN – downtrend continuing")
        else:
            reasons_bullish.append("⚪ No clear BOS/CHoCH signal")
        
        # ADX (15 points)
        if adx > 25:
            score += 15
            reasons_bullish.append(f"✅ Strong trend (ADX: {adx:.1f})")
        elif adx > 20:
            score += 8
            reasons_bullish.append(f"🟡 Moderate trend (ADX: {adx:.1f})")
        else:
            score -= 5
            reasons_bearish.append(f"🔴 Weak trend (ADX: {adx:.1f}) – ranging market")
        
        # RSI (10 points)
        if 30 < rsi < 70:
            score += 10
            reasons_bullish.append(f"✅ RSI neutral ({rsi:.1f}) – room to run")
        elif rsi < 30:
            score += 5
            reasons_bullish.append(f"🟢 RSI oversold ({rsi:.1f}) – potential bounce")
        elif rsi > 70:
            score -= 5
            reasons_bearish.append(f"🔴 RSI overbought ({rsi:.1f}) – potential pullback")
        
        # EMA alignment (10 points)
        ema_score = 0
        if above_ema13:
            ema_score += 3
        if above_ema50:
            ema_score += 3
        if above_ema200:
            ema_score += 4
        if ema13 > ema50 > ema200:
            ema_score += 5
            reasons_bullish.append("✅ EMAs in perfect bullish alignment (13 > 50 > 200)")
        elif ema13 < ema50 < ema200:
            ema_score -= 5
            reasons_bearish.append("🔴 EMAs in bearish alignment (13 < 50 < 200)")
        score += ema_score
        if ema_score > 0 and "EMA" not in str(reasons_bullish):
            reasons_bullish.append(f"✅ Price above {sum([above_ema13, above_ema50, above_ema200])}/3 major EMAs")
        
        # Volume (10 points)
        if volume_ratio > 1.5:
            score += 10
            reasons_bullish.append(f"✅ High volume ({volume_ratio:.1f}x avg) – strong participation")
        elif volume_ratio > 1.2:
            score += 5
            reasons_bullish.append(f"🟡 Above average volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio < 0.7:
            score -= 5
            reasons_bearish.append(f"🔴 Low volume ({volume_ratio:.1f}x avg) – weak conviction")
        
        # Demand zone (5 points)
        if near_zone:
            score += 5
            reasons_bullish.append("✅ Price near demand zone support")
        
        # Options flow (10 points)
        if unusual_flow:
            score += 10
            reasons_bullish.append("✅ Unusual call options activity detected – whale watching")
        
        # Earnings warning (penalty)
        if earnings_soon:
            score -= 15
            reasons_bearish.append("⚠️ Earnings in next 7 days – consider waiting or hedging")
        
        # Market regime (5 points)
        if spy_trend == "bullish":
            score += 5
            reasons_bullish.append("✅ SPY in uptrend (bullish market regime)")
        else:
            score -= 5
            reasons_bearish.append("🔴 SPY in downtrend (bearish market regime)")
        
        # Historical win rate bonus (up to 10 points)
        if historical_win_rate > 0:
            hist_bonus = min(10, int(historical_win_rate / 10))
            score += hist_bonus
            reasons_bullish.append(f"📊 Historical win rate on {symbol}: {historical_win_rate:.0f}% ({hist_bonus} pt bonus)")
        
        # Cap score between 0 and 100
        score = max(0, min(100, score))
        
        # Determine recommendation
        if score >= 80:
            recommendation = "🟢 HIGH PROBABILITY BUY"
            rec_color = 0x00ff00
            rec_action = "BUY CALLS (or stock) with confidence"
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.10
        elif score >= 65:
            recommendation = "🟡 MODERATE PROBABILITY BUY"
            rec_color = 0xffff00
            rec_action = "Consider BUY with reduced size"
            stop_loss = current_price * 0.93
            take_profit = current_price * 1.07
        elif score >= 50:
            recommendation = "⚪ NEUTRAL – WAIT FOR CONFIRMATION"
            rec_color = 0x888888
            rec_action = "No clear edge. Wait for better setup."
            stop_loss = None
            take_profit = None
        elif score >= 35:
            recommendation = "🟠 MODERATE PROBABILITY SELL / PUTS"
            rec_color = 0xff8800
            rec_action = "Consider PUTS with reduced size"
            stop_loss = current_price * 1.07
            take_profit = current_price * 0.90
        else:
            recommendation = "🔴 HIGH PROBABILITY SELL / AVOID"
            rec_color = 0xff0000
            rec_action = "AVOID or buy PUTS"
            stop_loss = current_price * 1.10
            take_profit = current_price * 0.85
        
        # Build embed
        embed = discord.Embed(
            title=f"📊 TRADE CONFIRMATION: {symbol} ({timeframe.upper()})",
            description=f"Current Price: **${current_price:.2f}**\n\n**{recommendation}** (Score: {score}/100)",
            color=rec_color,
            timestamp=datetime.now()
        )
        
        # Bullish signals
        if reasons_bullish:
            embed.add_field(name="🟢 Bullish Signals", value="\n".join(reasons_bullish[:8]), inline=False)
        
        # Bearish signals
        if reasons_bearish:
            embed.add_field(name="🔴 Bearish Signals / Warnings", value="\n".join(reasons_bearish[:5]), inline=False)
        
        # Trade parameters
        if stop_loss and take_profit:
            embed.add_field(name="🎯 Suggested Trade Parameters", 
                           value=f"**Action:** {rec_action}\n**Stop Loss:** ${stop_loss:.2f} ({((stop_loss - current_price)/current_price*100):+.1f}%)\n**Take Profit:** ${take_profit:.2f} ({((take_profit - current_price)/current_price*100):+.1f}%)\n**Risk/Reward:** {abs((take_profit - current_price)/(stop_loss - current_price)):.2f}", 
                           inline=False)
        else:
            embed.add_field(name="🎯 Suggested Action", value=rec_action, inline=False)
        
        # Signal summary
        embed.add_field(name="📈 Signal Summary",
                       value=f"• Structure: {structure['last_event'] or 'None'} {structure['last_event_direction'] or ''}\n"
                             f"• ADX: {adx:.1f} | RSI: {rsi:.1f}\n"
                             f"• Volume: {volume_ratio:.1f}x avg\n"
                             f"• EMAs: {'Bullish' if above_ema13 and above_ema50 else 'Mixed/Bearish'}\n"
                             f"• Market: SPY {spy_trend.upper()}",
                       inline=False)
        
        embed.set_footer(text="Use !structure, !strength, !zone for deeper analysis | Not financial advice")
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
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
        symbol = normalize_symbol(symbol.upper())
        watchlist = await load_watchlist()
        if '/' in symbol:
            if symbol not in watchlist['crypto']:
                watchlist['crypto'].append(symbol)
                if await save_watchlist(watchlist):
                    await ctx.send(f"✅ Added {symbol} to crypto watchlist.")
                else:
                    await ctx.send("❌ Could not save watchlist.")
            else:
                await ctx.send(f"{symbol} already in crypto watchlist.")
        else:
            if symbol not in watchlist['stocks']:
                watchlist['stocks'].append(symbol)
                if await save_watchlist(watchlist):
                    await ctx.send(f"✅ Added {symbol} to stocks watchlist.")
                else:
                    await ctx.send("❌ Could not save watchlist.")
            else:
                await ctx.send(f"{symbol} already in stocks watchlist.")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='remove')
async def remove_symbol(ctx, symbol):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
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
                await ctx.send("❌ Could not save watchlist.")
        else:
            await ctx.send(f"{symbol} not found in watchlist.")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='list')
async def list_watchlist(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        watchlist = await load_watchlist()
        stocks = ", ".join(watchlist['stocks']) if watchlist['stocks'] else "None"
        cryptos = ", ".join(watchlist['crypto']) if watchlist['crypto'] else "None"
        await ctx.send(f"**Stocks:** {stocks}\n**Crypto:** {cryptos}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# HELP COMMAND
# ====================
@bot.command(name='help')
async def help_command(ctx):
    try:
        embed = discord.Embed(
            title="📚 5-13-50 Trading Bot Commands",
            description=(
                "All commands use the prefix `!`\n\n"
                "**SCAN & SIGNALS**\n"
                "`!scan all [timeframe]` – Scan all watchlist symbols on a single timeframe\n"
                "`!scan SYMBOL [timeframe]` – Scan a single symbol\n"
                "`!signals` – Scan ENTIRE watchlist across ALL 7 timeframes\n"
                "`!signal SYMBOL` – Multi‑timeframe report for a single symbol\n\n"
                "**CONFIRMATION (NEW!)**\n"
                "`!confirm SYMBOL [timeframe]` – Comprehensive trade confirmation score (0-100)\n"
                "  Combines structure, trend strength, volume, options flow, earnings, and market regime\n\n"
                "**NEWS & EVENTS**\n"
                "`!news TICKER` – Company news with analyst ratings\n"
                "`!worldnews` – Global headlines with AI sentiment analysis\n"
                "`!upcoming [TICKER]` – Earnings, dividends, splits, expected move\n\n"
                "**ZONES & STRUCTURE**\n"
                "`!zone SYMBOL [timeframe]` – Demand zones with ITM option suggestions\n"
                "`!structure SYMBOL [timeframe]` – BOS/CHoCH analysis with chart\n"
                "`!structure all [timeframe]` – Scan watchlist for BOS/CHoCH\n\n"
                "**TREND STRENGTH**\n"
                "`!strength SYMBOL [timeframe]` – ADX, RSI, volume, divergence\n\n"
                "**TACO TRADE**\n"
                "`!taco` – Trump trade signal\n"
                "`!pennant [timeframe]` – Bullish pennant (flag) scanner\n\n"
                "**OPTIONS FLOW**\n"
                "`!flow TICKER` – Unusual options activity\n"
                "`!scanflow` – Scan watchlist for unusual options\n\n"
                "**BACKTESTING & LEAPS**\n"
                "`!backtest SYMBOL [days=365]` – EMA crossover backtest\n"
                "`!leaps` – LEAPS candidates (weekly CHoCH)\n\n"
                "**FINVIZ SCANNER**\n"
                "`!finviz_scan [filters]` – Custom Finviz scan\n"
                "`!cheap_options` – Preset: $10-20, optionable, high volume\n"
                "`!hype_stocks` – Preset: high relative volume, up 5%\n\n"
                "**TRADE TRACKING**\n"
                "`!track BUY SYMBOL [CALL/PUT strike] premium` – Record entry\n"
                "`!track CLOSE SYMBOL price_or_percent` – Close trade\n"
                "`!trades [symbol] [limit]` – List recent trades\n"
                "`!performance [symbol] [days]` – Win rate and P&L stats\n\n"
                "**WATCHLIST**\n"
                "`!add SYMBOL` – Add to watchlist\n"
                "`!remove SYMBOL` – Remove from watchlist\n"
                "`!list` – Show watchlist\n\n"
                "**UTILITY**\n"
                "`!ping` – Test bot\n"
                "`!stopscan` / `!cancel` – Stop ongoing scan\n"
                "`!help` – This message\n\n"
                "**TIMEFRAMES:** 5min, 15min, 30min, 1h, 4h, daily, weekly\n\n"
                "💡 **Pro tip:** Use `!confirm` before any trade for a complete signal analysis!"
            ),
            color=0x3498db
        )
        embed.set_footer(text="Use !help for this menu")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send("📚 Commands: !scan, !signals, !signal, !confirm, !news, !worldnews, !upcoming, !zone, !structure, !strength, !taco, !pennant, !leaps, !finviz_scan, !cheap_options, !hype_stocks, !cheap_plays, !flow, !scanflow, !backtest, !track, !performance, !trades, !add, !remove, !list, !ping, !stopscan, !cancel")
        print(f"Help command error: {e}")

# ====================
# SCAN FOR PENNANT
# ====================
@bot.command(name='pennant')
async def scan_pennant(ctx, timeframe: str = '4h'):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        valid_tfs = ['1h', '4h', 'daily', 'weekly']
        if timeframe not in valid_tfs:
            await ctx.send(f"❌ Invalid timeframe. Use: {', '.join(valid_tfs)}")
            return
        watchlist = await load_watchlist()
        symbols = watchlist['stocks']
        if not symbols:
            await ctx.send("No stocks in watchlist.")
            return
        await ctx.send(f"🔍 Scanning {len(symbols)} stocks for bullish pennants on {timeframe}...")
        found = []
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            try:
                df = await fetch_ohlcv(symbol, timeframe)
                if df is None or len(df) < 30:
                    continue
                recent_low = df['low'].tail(20).min()
                highs_last10 = df['high'].tail(10)
                lows_last10 = df['low'].tail(10)
                descending = highs_last10.iloc[0] > highs_last10.iloc[-1] * 1.01
                flat_bottom = (lows_last10.max() - lows_last10.min()) / lows_last10.min() < 0.02
                if descending and flat_bottom:
                    current_price = df['close'].iloc[-1]
                    if current_price <= recent_low * 1.02:
                        found.append(f"{symbol} (${current_price:.2f}) – flat support near ${recent_low:.2f}")
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
            await asyncio.sleep(0.3)
        if not found:
            await ctx.send(f"📭 No bullish pennants found on {timeframe}.")
            return
        embed = discord.Embed(title=f"🚩 Bullish Pennant (Flag) Scan – {timeframe.upper()}", description=f"Found {len(found)} stocks with flat-bottom pennants", color=0x00ff00, timestamp=datetime.now())
        embed.add_field(name="📈 Potential Breakout Candidates", value="\n".join(found[:10]), inline=False)
        embed.add_field(name="💡 Trading Advice", value="• Wait for price to break ABOVE the descending resistance line.\n• Entry on breakout + volume confirmation.\n• Stop loss just below the flat support.\n• Use `!structure SYMBOL` to see the chart.", inline=False)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error scanning pennants: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# TACO TRADE COMMAND
# ====================
@bot.command(name='taco')
async def taco_trade(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send("🌮 Checking Trump's latest Truth for TACO signal...")
        trump_post = await fetch_latest_trump_post()
        if not trump_post:
            await ctx.send("❌ Could not fetch Trump's latest post. RSS feed may be down.")
            return
        analysis = analyze_trump_post(trump_post['text'])
        embed = discord.Embed(title="🌮 TACO TRADE SIGNAL", description="**Trump Always Chickens Out** – trade the reversal", color=0xff6600, timestamp=datetime.now())
        embed.add_field(name="📢 Latest Truth", value=f"> {trump_post['text'][:500]}{'…' if len(trump_post['text'])>500 else ''}", inline=False)
        embed.add_field(name="📅 Time", value=trump_post['timestamp'], inline=True)
        embed.add_field(name="📊 Sentiment", value=analysis['sentiment'], inline=True)
        embed.add_field(name="🎲 TACO Probability", value=analysis['taco_probability'], inline=False)
        embed.add_field(name="🎯 Affected Sectors", value=analysis['affected_sectors'], inline=True)
        embed.add_field(name="💎 Suggested Stocks", value=analysis['suggested_stocks'], inline=True)
        if "Bearish" in analysis['sentiment'] and "High" in analysis['taco_probability']:
            embed.add_field(name="✅ TRADE ACTION", value="**BUY CALLS** on the suggested stocks.\n• Wait for the market to dip on the threat.\n• Look for a **flat bottom (bullish pennant)** on the 4h chart.\n• Enter when price breaks above descending resistance.\n• Set stop loss just below the flat support.\n• Target: +10‑20% move when Trump backtracks.\nUse `!structure SYMBOL` to see the chart and `!pennant` to find flat bottoms.", inline=False)
        elif "Bullish" in analysis['sentiment']:
            embed.add_field(name="✅ TRADE ACTION", value="**SELL / TAKE PROFITS** on the suggested stocks.\n• Trump is talking peace – the rally may end soon.\n• Consider selling calls or buying puts after a bounce.\n• Wait for a CHoCH down on the 4h chart.\nUse `!structure SYMBOL` to confirm reversal.", inline=False)
        else:
            embed.add_field(name="⏳ TRADE ACTION", value="No clear TACO signal yet. **Monitor** for threats or peace talks.\nWhen you see a threat, prepare to buy dips. When you see peace talks, prepare to sell.\nUse `!worldnews` for broader context.", inline=False)
        embed.add_field(name="🔗 Link", value=f"[View on Truth Social]({trump_post['url']})", inline=False)
        embed.set_footer(text="TACO = Trump Always Chickens Out • Not financial advice")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# EVENT HANDLERS
# ====================
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)

@bot.command(name='ping')
async def ping(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send('pong')
    finally:
        user_busy[ctx.author.id] = False

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
    await ctx.send("⏹️ Cancelling scan... (will stop after the current symbol)")

@bot.command(name='cancel')
async def cancel_scan(ctx):
    await stop_scan(ctx)

async def send_symbol_with_chart(ctx, symbol, df, timeframe):
    df_calc = calculate_indicators(df)
    signals = get_signals(df_calc)
    peg_str = None
    if '/' not in symbol:
        _, peg_str = await get_peg_ratio(symbol)
    embed = format_embed(symbol, signals, timeframe, peg_str=peg_str)
    try:
        chart_buffer = generate_chart_image(df, symbol, timeframe)
        if chart_buffer:
            file = discord.File(chart_buffer, filename='chart.png')
            embed.set_image(url='attachment://chart.png')
            await ctx.send(embed=embed, file=file)
        else:
            await ctx.send(embed=embed)
    except Exception as e:
        print(f"Error: {e}")
        await ctx.send(embed=embed)

# ====================
# TRUMP / TACO FUNCTIONS
# ====================
async def fetch_latest_trump_post():
    feeds = ["https://trumpstruth.org/feed", "https://truthsocial.rss.simple-web.org/feed.xml"]
    for url in feeds:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    root = ET.fromstring(text)
                    item = root.find('.//item')
                    if item is not None:
                        return {'text': item.find('title').text, 'timestamp': item.find('pubDate').text, 'url': item.find('link').text}
                    entry = root.find('.//{http://www.w3.org/2005/Atom}entry')
                    if entry is not None:
                        return {'text': entry.find('{http://www.w3.org/2005/Atom}title').text,
                                'timestamp': entry.find('{http://www.w3.org/2005/Atom}updated').text,
                                'url': entry.find('{http://www.w3.org/2005/Atom}link').get('href')}
        except Exception as e:
            print(f"Error with feed {url}: {e}")
            continue
    return None

def analyze_trump_post(post_text):
    text_lower = post_text.lower()
    threat_keywords = ['tariff', 'sanction', 'war', 'military', 'bomb', 'strike', 'deadline', 'penalty', 'tax', 'duty', 'ban', 'restrict', 'withdraw']
    backtrack_keywords = ['exemption', 'delay', 'postpone', 'review', 'negotiate', 'deal', 'agreement', 'temporary', 'exception', 'reconsider']
    has_threat, has_backtrack = any(kw in text_lower for kw in threat_keywords), any(kw in text_lower for kw in backtrack_keywords)
    if has_threat and not has_backtrack:
        sentiment, taco_prob, advice = "🔴 Bearish", "High (expected market dip & reversal)", "TACO play likely. Watch for dip – DO NOT buy puts. Wait for dip to buy calls."
    elif has_threat and has_backtrack:
        sentiment, taco_prob, advice = "🟡 Mixed (threat with caveats)", "Medium – possible backtrack", "Monitor for clarification. Could swing either way."
    elif any(word in text_lower for word in ['good', 'great', 'deal', 'agreement']):
        sentiment, taco_prob, advice = "🟢 Bullish", "Low – positive news", "Potential rally. Check related sectors."
    else:
        sentiment, taco_prob, advice = "⚪ Neutral", "Low – no clear threat or positive", "No immediate TACO signal."
    sectors, stocks = [], []
    if 'tariff' in text_lower or 'trade' in text_lower:
        sectors.extend(['Industrials', 'Consumer Goods'])
        stocks.extend(['CAT', 'DE', 'F', 'GM', 'WMT', 'TGT'])
    if 'china' in text_lower:
        sectors.append('China exposure')
        stocks.extend(['BABA', 'JD', 'NIO', 'FXI', 'KWEB'])
    if 'oil' in text_lower or 'energy' in text_lower:
        sectors.append('Energy')
        stocks.extend(['XOM', 'CVX', 'OXY', 'USO'])
    if 'military' in text_lower or 'defense' in text_lower:
        sectors.append('Defense')
        stocks.extend(['LMT', 'NOC', 'GD', 'RTX'])
    if 'iran' in text_lower or 'middle east' in text_lower:
        sectors.append('Geopolitical risk')
        stocks.extend(['XOM', 'CVX', 'LMT', 'NOC'])
    return {'sentiment': sentiment, 'taco_probability': taco_prob, 'affected_sectors': ', '.join(list(dict.fromkeys(sectors))) if sectors else 'General market',
            'suggested_stocks': ', '.join([f"${s}" for s in list(dict.fromkeys(stocks))[:5]]) if stocks else '$SPY, $QQQ', 'advice': advice}

# ====================
# MAIN ENTRY POINT
# ====================
async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())