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
trades_collection = db['trades']  # New collection for trade tracking

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
    if '/' in symbol:  # Crypto
        base = symbol.split('/')[0]
        exchange = "BINANCE"
        tv_symbol = f"{exchange}:{base}USDT"
    else:  # Stock
        exchange = "NASDAQ"
        tv_symbol = f"{exchange}:{symbol}"
    web_url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
    return web_url

# ====================
# DATA FETCHING – Multi-source strategy
# ====================
async def fetch_finnhub(symbol, timeframe):
    resolution_map = {
        '5min': '5',
        '15min': '15',
        '30min': '30',
        '1h': '60',
        '4h': '240',
        'daily': 'D',
        'weekly': 'W'
    }
    resolution = resolution_map.get(timeframe)
    if not resolution:
        return None

    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': resolution,
        'from': int((datetime.now() - timedelta(days=60)).timestamp()),
        'to': int(datetime.now().timestamp()),
        'token': FINNHUB_API_KEY
    }

    await finnhub_limiter.wait_if_needed()

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"Finnhub error for {symbol}: {resp.status}")
                    return None
                data = await resp.json()
                if data.get('s') != 'ok':
                    print(f"Finnhub data error for {symbol}: {data}")
                    return None
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                }).set_index('timestamp')
                return df
    except asyncio.TimeoutError:
        print(f"Finnhub timeout for {symbol}")
        return None
    except Exception as e:
        print(f"Finnhub exception for {symbol}: {e}")
        return None

async def fetch_twelvedata(symbol, timeframe):
    interval_map = {
        '5min': '5min', '15min': '15min', '30min': '30min',
        '1h': '1h', '4h': '4h', 'daily': '1day', 'weekly': '1week'
    }
    interval = interval_map.get(timeframe)
    if not interval:
        return None

    outputsize = 200
    if timeframe in ['5min', '15min', '1h']:
        outputsize = 500

    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': interval,
        'apikey': TWELVEDATA_API_KEY,
        'outputsize': outputsize,
        'format': 'JSON'
    }

    await twelvedata_limiter.wait_if_needed()

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    print("Twelve Data rate limit hit, waiting 60s...")
                    await asyncio.sleep(60)
                    return None
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
    except asyncio.TimeoutError:
        print(f"Twelve Data timeout for {symbol}")
        return None
    except Exception as e:
        print(f"Twelve Data exception for {symbol}: {e}")
        return None

async def fetch_coingecko_ohlc(symbol, timeframe):
    base = symbol.split('/')[0].lower()
    coin_map = {
        'btc': 'bitcoin', 'eth': 'ethereum', 'sol': 'solana',
        'xrp': 'ripple', 'doge': 'dogecoin', 'pepe': 'pepecoin',
        'ada': 'cardano', 'dot': 'polkadot', 'link': 'chainlink'
    }
    coin_id = coin_map.get(base)
    if not coin_id:
        print(f"CoinGecko: no coin_id for {symbol}")
        return None

    days_map = {
        '5min': 1,
        '15min': 2,
        '30min': 2,
        '1h': 7,
        '4h': 7,
        'daily': 30,
        'weekly': 90
    }
    days = days_map.get(timeframe)
    if not days:
        return None

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {'vs_currency': 'usd', 'days': days}

    await coingecko_limiter.wait_if_needed()

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"CoinGecko OHLC error for {symbol}: status {resp.status}")
                    return None
                data = await resp.json()
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['volume'] = np.nan
                return df
    except asyncio.TimeoutError:
        print(f"CoinGecko timeout for {symbol}")
        return None
    except Exception as e:
        print(f"CoinGecko OHLC exception for {symbol}: {e}")
        return None

async def fetch_coingecko_price(symbol):
    base = symbol.split('/')[0].lower()
    coin_map = {
        'btc': 'bitcoin', 'eth': 'ethereum', 'sol': 'solana',
        'xrp': 'ripple', 'doge': 'dogecoin', 'pepe': 'pepecoin',
        'ada': 'cardano', 'dot': 'polkadot', 'link': 'chainlink'
    }
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
                # Create synthetic OHLC
                np.random.seed(42)
                dates = pd.date_range(end=datetime.now(), periods=200, freq='H')
                close_prices = price * (1 + np.random.normal(0, 0.01, 200).cumsum() * 0.01)
                open_prices = close_prices * 0.99
                high_prices = close_prices * 1.02
                low_prices = close_prices * 0.98
                volumes = np.abs(np.random.normal(1e6, 2e5, 200))

                df = pd.DataFrame({
                    'timestamp': dates,
                    'open': open_prices,
                    'high': high_prices,
                    'low': low_prices,
                    'close': close_prices,
                    'volume': volumes
                })
                df.set_index('timestamp', inplace=True)
                return df
    except Exception as e:
        print(f"CoinGecko price exception for {symbol}: {e}")
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
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                timeframe_multiplier=15,
                start=now - timedelta(days=60),
                end=now
            )
            bars = await asyncio.to_thread(client.get_stock_bars, request)
            if bars.data:
                df_15 = bars.df
                df_15 = df_15.reset_index(level=0, drop=True)
                df_15.index = pd.to_datetime(df_15.index)
                df_30 = df_15.resample('30T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                if not df_30.empty:
                    df = df_30
                    print(f"✅ Alpaca 15min resampled to 30min for {symbol}")
        except Exception as e:
            print(f"⚠️ Alpaca 15min fetch failed for {symbol}, trying fallbacks... {e}")

    if df is None and not is_crypto and ALPACA_API_KEY and ALPACA_SECRET_KEY:
        tf_map = {
            '5min': (TimeFrame.Minute, 5),
            '15min': (TimeFrame.Minute, 15),
            '1h': (TimeFrame.Hour, 1),
            '4h': (TimeFrame.Hour, 4),
            'daily': (TimeFrame.Day, 1),
            'weekly': (TimeFrame.Week, 1),
        }
        if timeframe in tf_map:
            tf, mult = tf_map[timeframe]
            try:
                client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    timeframe_multiplier=mult,
                    start=now - timedelta(days=60),
                    end=now
                )
                bars = await asyncio.to_thread(client.get_stock_bars, request)
                if bars.data:
                    df = bars.df
                    df = df.reset_index(level=0, drop=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    print(f"✅ Alpaca stock data for {symbol} ({timeframe})")
            except Exception as e:
                print(f"⚠️ Alpaca stock fetch failed for {symbol}, trying Finnhub... {e}")

    if df is None and not is_crypto:
        df = await fetch_finnhub(symbol, timeframe)

    if df is None and not is_crypto:
        df = await fetch_twelvedata(symbol, timeframe)

    if is_crypto:
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            tf_map_crypto = {
                '5min': (TimeFrame.Minute, 5),
                '15min': (TimeFrame.Minute, 15),
                '1h': (TimeFrame.Hour, 1),
                '4h': (TimeFrame.Hour, 4),
                'daily': (TimeFrame.Day, 1),
                'weekly': (TimeFrame.Week, 1),
            }
            if timeframe in tf_map_crypto:
                tf, mult = tf_map_crypto[timeframe]
                try:
                    client = CryptoHistoricalDataClient()
                    alpaca_symbol = symbol.replace('/', '')
                    request = CryptoBarsRequest(
                        symbol_or_symbols=alpaca_symbol,
                        timeframe=tf,
                        timeframe_multiplier=mult,
                        start=now - timedelta(days=60),
                        end=now
                    )
                    bars = await asyncio.to_thread(client.get_crypto_bars, request)
                    if bars.data:
                        df = bars.df
                        df = df.reset_index(level=0, drop=True)
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        print(f"✅ Alpaca crypto data for {symbol} ({timeframe})")
                except Exception as e:
                    print(f"⚠️ Alpaca crypto fetch failed for {symbol}, trying CoinGecko... {e}")

        if df is None:
            df = await fetch_coingecko_ohlc(symbol, timeframe)
        if df is None:
            df = await fetch_coingecko_price(symbol)

    if df is not None and not df.empty:
        data_cache[cache_key] = (df, now + CACHE_DURATION)

    return df

# ====================
# ENHANCED NEWS FETCHING
# ====================
async def fetch_finnhub_news(symbol):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': symbol,
        'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'to': datetime.now().strftime('%Y-%m-%d'),
        'token': FINNHUB_API_KEY
    }
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, list) else None
    except Exception as e:
        print(f"Error fetching Finnhub news for {symbol}: {e}")
        return None

async def fetch_finnhub_general_news():
    url = "https://finnhub.io/api/v1/news"
    params = {'category': 'general', 'token': FINNHUB_API_KEY}
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, list) else None
    except Exception as e:
        print(f"Error fetching Finnhub general news: {e}")
        return None

async def fetch_newsapi_top_headlines(country='us'):
    if not NEWSAPI_KEY:
        return None
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'country': country,
        'pageSize': 10
    }
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"NewsAPI error for {country}: {resp.status}")
                    return None
                data = await resp.json()
                if data.get('status') != 'ok':
                    return None
                return data.get('articles', [])
    except Exception as e:
        print(f"Error fetching NewsAPI for {country}: {e}")
        return None

async def fetch_analyst_ratings(symbol, limit=3):
    url = "https://finnhub.io/api/v1/stock/recommendation"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data[:limit] if data else []
    except Exception as e:
        print(f"Error fetching analyst ratings for {symbol}: {e}")
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
            if hasattr(date, 'date'):
                e_date = date.date()
            else:
                e_date = datetime.strptime(str(date), '%Y-%m-%d').date()
            if today <= e_date <= cutoff:
                eps_est = row.get('epsEstimated') if 'epsEstimated' in row else row.get('epsEstimate')
                if eps_est is None or pd.isna(eps_est):
                    eps_est = 'N/A'
                else:
                    eps_est = f"{eps_est:.2f}"
                upcoming.append({
                    'date': e_date.strftime('%Y-%m-%d'),
                    'epsEstimate': eps_est,
                    'hour': 'AMC'
                })
        return upcoming
    except Exception as e:
        print(f"Error fetching earnings for {symbol} via yfinance: {e}")
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
            if hasattr(date, 'date'):
                d_date = date.date()
            else:
                d_date = datetime.strptime(str(date), '%Y-%m-%d').date()
            if today <= d_date <= cutoff:
                upcoming.append({
                    'exDate': d_date.strftime('%Y-%m-%d'),
                    'amount': f"{amount:.2f}",
                    'payDate': ''
                })
        return upcoming
    except Exception as e:
        print(f"Error fetching dividends for {symbol} via yfinance: {e}")
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
            if hasattr(date, 'date'):
                s_date = date.date()
            else:
                s_date = datetime.strptime(str(date), '%Y-%m-%d').date()
            if today <= s_date <= cutoff:
                ratio_str = f"{ratio:.0f}:1" if ratio >= 1 else f"1:{int(1/ratio)}"
                upcoming.append({
                    'date': s_date.strftime('%Y-%m-%d'),
                    'splitRatio': ratio_str
                })
        return upcoming
    except Exception as e:
        print(f"Error fetching splits for {symbol} via yfinance: {e}")
        return []

async def fetch_economic_events(days=14):
    if not FINNHUB_API_KEY:
        return []
    url = "https://finnhub.io/api/v1/calendar/economic"
    start = datetime.now().strftime('%Y-%m-%d')
    end = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        'from': start,
        'to': end,
        'token': FINNHUB_API_KEY
    }
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get('economicCalendar', [])
    except Exception as e:
        print(f"Error fetching economic calendar: {e}")
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

# ====================
# CHART GENERATION
# ====================
def generate_chart_image(df, symbol, timeframe):
    if len(df) < 20:
        return None

    if timeframe in ['5min', '15min', '1h']:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(50).copy()
    else:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(30).copy()

    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    mc = mpf.make_marketcolors(
        up='#26a69a',
        down='#ef5350',
        edge='white',
        wick='white',
        volume='in',
        inherit=True
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        facecolor='#1e1e1e',
        figcolor='#1e1e1e',
        gridcolor='#444444'
    )

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
        if volume_all_nan:
            fig, axes = mpf.plot(
                chart_data,
                type='candle',
                style=s,
                addplot=apds,
                volume=False,
                figsize=(10,6),
                returnfig=True,
                title=f'{symbol} – {timeframe}',
                tight_layout=True
            )
        else:
            fig, axes = mpf.plot(
                chart_data,
                type='candle',
                style=s,
                addplot=apds,
                volume=True,
                figsize=(10,6),
                returnfig=True,
                title=f'{symbol} – {timeframe}',
                tight_layout=True
            )
        if apds:
            axes[0].legend(
                loc='upper left',
                fontsize=9,
                facecolor='#333333',
                edgecolor='white',
                labelcolor='white',
                framealpha=0.8
            )
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
        print(f"⚠️ Chart generation failed for {symbol}: {e}")
        return None

def generate_zone_chart(df, symbol, zones):
    if len(df) < 20:
        return None

    chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(100).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    mc = mpf.make_marketcolors(
        up='#26a69a',
        down='#ef5350',
        edge='white',
        wick='white',
        volume='in',
        inherit=True
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        facecolor='#1e1e1e',
        figcolor='#1e1e1e',
        gridcolor='#444444'
    )

    if zones:
        strengths = [z['strength'] for z in zones]
        min_s = min(strengths)
        max_s = max(strengths)
        if max_s > min_s:
            norm_strengths = [(s - min_s) / (max_s - min_s) for s in strengths]
        else:
            norm_strengths = [0.5] * len(strengths)

        colormap = matplotlib.colormaps['RdYlGn_r']
        line_colors = [colormap(norm) for norm in norm_strengths]
    else:
        line_colors = []

    apds = []
    for i, zone in enumerate(zones):
        level = zone['level']
        if line_colors:
            color = mcolors.to_hex(line_colors[i])
        else:
            color = '#ffffff'
        label = f"Demand ${level:.2f} (touches: {zone['strength']})"
        apds.append(mpf.make_addplot(
            [level] * len(chart_data),
            color=color,
            width=2.0,
            linestyle='-',
            label=label
        ))

    try:
        fig, axes = mpf.plot(
            chart_data,
            type='candle',
            style=s,
            addplot=apds,
            volume=True,
            figsize=(12, 7),
            returnfig=True,
            title=f'\n{symbol} Demand Zones (30min)',
            tight_layout=True,
            scale_padding={'left': 0.5, 'right': 0.5, 'top': 0.5, 'bottom': 0.5}
        )
        if apds:
            axes[0].legend(
                loc='upper left',
                fontsize=10,
                facecolor='#333333',
                edgecolor='white',
                labelcolor='white',
                framealpha=0.8
            )
        axes[2].set_ylabel('Volume', color='white')
        axes[2].tick_params(colors='white')
        axes[2].yaxis.label.set_color('white')
        axes[0].tick_params(colors='white')
        axes[0].yaxis.label.set_color('white')
        axes[0].xaxis.label.set_color('white')
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
        print(f"⚠️ Zone chart generation failed for {symbol}: {e}")
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
        params = {
            'api_token': api_key,
            'language': 'en',
            'limit': 10,
            'must_have_entities': 'true'
        }

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

        embed = discord.Embed(
            title="🌍 World News – AI Market Sentiment",
            description="Top financial headlines from Marketaux",
            color=0x3498db,
            timestamp=datetime.now()
        )

        trump_post = await fetch_latest_trump_post()
        if trump_post:
            analysis = analyze_trump_post(trump_post['text'])
            trump_value = (
                f"**Post:** {trump_post['text'][:300]}{'…' if len(trump_post['text'])>300 else ''}\n"
                f"**Time:** {trump_post['timestamp']}\n"
                f"**Sentiment:** {analysis['sentiment']}\n"
                f"**TACO Probability:** {analysis['taco_probability']}\n"
                f"**Affected Sectors:** {analysis['affected_sectors']}\n"
                f"**Suggested Stocks:** {analysis['suggested_stocks']}\n"
                f"**Advice:** {analysis['advice']}\n"
                f"[Link to post]({trump_post['url']})"
            )
            embed.add_field(name="🔔 @realDonaldTrump (Latest Truth)", value=trump_value, inline=False)

        for article in articles[:8]:
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown')
            published_at = article.get('published_at', '')
            url_link = article.get('url', '')
            entities = article.get('entities', [])
            sentiment_score = 0.0
            if entities and len(entities) > 0:
                sentiment_score = entities[0].get('sentiment_score', 0.0)
            if sentiment_score > 0.2:
                sentiment_emoji = "🟢 Bullish"
            elif sentiment_score < -0.2:
                sentiment_emoji = "🔴 Bearish"
            else:
                sentiment_emoji = "⚪ Neutral"

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

            embed.add_field(
                name=f"📰 {title[:100]}{'…' if len(title)>100 else ''}",
                value=field_value,
                inline=False
            )

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

        ratings_data, finnhub_data = await asyncio.gather(
            ratings_task, news_task, return_exceptions=True
        )

        if isinstance(ratings_data, Exception):
            ratings_data = None
        if isinstance(finnhub_data, Exception):
            finnhub_data = None

        if not finnhub_data:
            await ctx.send(f"❌ Could not fetch news data for {symbol}.")
            return

        web_url = get_tradingview_web_link(symbol)
        tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"

        embed = discord.Embed(
            title=f"Latest News for {symbol}",
            color=0x3498db,
            timestamp=datetime.now()
        )

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
            if dt:
                if isinstance(dt, (int, float)):
                    date = datetime.fromtimestamp(dt).strftime('%Y-%m-%d %H:%M')
                else:
                    date = str(dt)[:16]
            else:
                date = 'Unknown'
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

    web_url = get_tradingview_web_link(symbol)
    tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"

    embed = discord.Embed(
        title=f"{rating}",
        description=f"**{symbol}** · ${signals['price']:.2f}",
        color=color
    )
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
    ema5 = signals['ema5']
    ema13 = signals['ema13']
    ema50 = signals['ema50']
    ema200 = signals['ema200']

    support_levels = [support]
    resistance_levels = [resistance]

    if not pd.isna(ema200):
        if ema200 < price:
            support_levels.append(ema200)
        else:
            resistance_levels.append(ema200)
    if not pd.isna(ema50):
        if ema50 < price:
            support_levels.append(ema50)
        else:
            resistance_levels.append(ema50)
    if not pd.isna(ema13):
        if ema13 < price:
            support_levels.append(ema13)
        else:
            resistance_levels.append(ema13)
    if not pd.isna(ema5):
        if ema5 < price:
            support_levels.append(ema5)
        else:
            resistance_levels.append(ema5)

    support_levels.sort(reverse=True)
    resistance_levels.sort()

    web_url = get_tradingview_web_link(symbol)
    tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"

    embed = discord.Embed(
        title=f"📊 {symbol} – {timeframe.capitalize()} Zones",
        description=f"Current Price: **${price:.2f}**",
        color=0x00ff00 if signals['net_score'] > 0 else 0xff0000 if signals['net_score'] < 0 else 0xffff00
    )

    sup_text = ""
    for i, level in enumerate(support_levels):
        if i == 0:
            sup_text += f"**Primary Support:** ${level:.2f}\n"
        else:
            sup_text += f"Secondary Support: ${level:.2f}\n"
    if sup_text:
        embed.add_field(name="📉 Support (Buy Zone)", value=sup_text, inline=False)

    res_text = ""
    for i, level in enumerate(resistance_levels):
        if i == 0:
            res_text += f"**Primary Resistance:** ${level:.2f}\n"
        else:
            res_text += f"Secondary Resistance: ${level:.2f}\n"
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
    timeframe_priority = {
        '5min': 1, '15min': 2, '30min': 3, '1h': 4, '4h': 5, 'daily': 6, 'weekly': 7
    }

    best_tf = None
    best_score = -float('inf')

    for tf, data in symbol_signals.items():
        net_score = data['signals']['net_score']
        strength = abs(net_score)
        if strength > best_score or (strength == best_score and timeframe_priority.get(tf, 99) < timeframe_priority.get(best_tf, 99)):
            best_score = strength
            best_tf = tf

    if not best_tf:
        return

    best_data = symbol_signals[best_tf]
    df = best_data['df']
    signals = best_data['signals']

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
        print(f"⚠️ Chart generation failed for {symbol}: {e}")
        await ctx.send(embed=main_embed)

    await send_symbol_timeframe_summary(ctx, symbol, symbol_signals)

async def send_symbol_timeframe_summary(ctx, symbol, symbol_signals):
    timeframe_order = {'5min': 1, '15min': 2, '30min': 3, '1h': 4, '4h': 5, 'daily': 6, 'weekly': 7}
    sorted_timeframes = sorted(symbol_signals.keys(), key=lambda x: timeframe_order.get(x, 99))

    bullish_count = sum(1 for tf, data in symbol_signals.items() if data['signals']['net_score'] > 0)
    bearish_count = sum(1 for tf, data in symbol_signals.items() if data['signals']['net_score'] < 0)
    total = len(symbol_signals)

    summary_lines = []
    for tf in sorted_timeframes:
        signals = symbol_signals[tf]['signals']
        net = signals['net_score']

        if net >= 2:
            emoji = "🟢"
            signal_text = "STRONG BUY"
        elif net == 1:
            emoji = "🟢"
            signal_text = "BUY"
        elif net == 0:
            emoji = "⚪"
            signal_text = "NEUTRAL"
        elif net == -1:
            emoji = "🔴"
            signal_text = "SELL"
        elif net <= -2:
            emoji = "🔴"
            signal_text = "STRONG SELL"

        summary_lines.append(f"{emoji} {tf}: {signal_text} (Score: {net})")

    embed = discord.Embed(
        title=f"📊 MULTI-TIMEFRAME SUMMARY: {symbol}",
        description="\n".join(summary_lines),
        color=0x3498db
    )

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

    embed = discord.Embed(
        title="📊 MULTI-TIMEFRAME SCAN COMPLETE",
        description=f"Found signals for **{len(signal_summary)}** symbols",
        color=0x3498db
    )

    strong_buy = []
    buy = []
    neutral = []
    sell = []
    strong_sell = []

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
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
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
        best_exp = None
        best_dte = None
        min_diff = float('inf')
        for exp_date, exp_str in zip(exp_dates, expirations):
            dte = (exp_date - today).days
            diff = abs(dte - 38)
            if diff < min_diff:
                min_diff = diff
                best_exp = exp_str
                best_dte = dte
        if best_exp and best_dte:
            key_exps.append((best_exp, best_dte, "🛡️ PRIMARY (30-45 DTE)"))
        return key_exps
    except Exception as e:
        print(f"Error getting key expirations for {ticker}: {e}")
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
    calls = opt_chain.calls.copy()
    puts = opt_chain.puts.copy()
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

            analyzed.append({
                'strike': strike,
                'type': opt_type,
                'volume': int(volume),
                'oi': int(oi),
                'vol_oi_ratio': vol_oi_ratio,
                'last': last,
                'premium': format_premium(volume, last),
                'raw_premium': premium,
                'distance_pct': distance_pct,
                'dte': dte
            })
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
        embed = discord.Embed(
            title=f"🔍 OPTIONS FLOW: {ticker.upper()}",
            description=f"Current Price: **${current_price:.2f}**",
            color=0x00ff00
        )

        high_prob_options = []
        lottery_options = []

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
            strike = f"${opt['strike']:.2f}"
            whale = get_whale_emoji(opt['raw_premium'])
            distance = f"{opt['distance_pct']:.1f}% away"
            prob_est = "High" if opt['distance_pct'] < 10 else "Moderate"
            hp_text += f"**{strike} {opt['type']}** ({label})\n"
            hp_text += f"   • Vol: {opt['volume']} ({opt['vol_oi_ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n"
            hp_text += f"   • DTE: {dte}  Distance: {distance} – Prob: {prob_est}\n\n"
        if hp_text:
            add_field_safe(embed, "📈 HIGH PROBABILITY SETUPS (30-45 DTE, Near Money)", hp_text, inline=False)

        lt_text = ""
        for label, exp_str, dte, opt in lottery_options[:10]:
            strike = f"${opt['strike']:.2f}"
            whale = get_whale_emoji(opt['raw_premium'])
            lt_text += f"**{strike} {opt['type']}** ({label})\n"
            lt_text += f"   • Vol: {opt['volume']} ({opt['vol_oi_ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n"
            lt_text += f"   • DTE: {dte}\n\n"
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
                                            all_unusual.append({
                                                'symbol': symbol,
                                                'strike': strike,
                                                'expiration': exp_str,
                                                'dte': dte,
                                                'volume': int(volume),
                                                'oi': int(oi),
                                                'ratio': vol_oi_ratio,
                                                'price': current_price,
                                                'premium': format_premium(volume, last),
                                                'raw_premium': premium,
                                                'distance': distance_pct,
                                                'label': label
                                            })
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

        embed = discord.Embed(
            title="🔥 UNUSUAL OPTIONS ACTIVITY SUMMARY",
            description=f"Found {len(all_unusual)} unusual setups across your watchlist",
            color=0x00ff00
        )

        hp_text = ""
        for opt in high_prob[:8]:
            whale = get_whale_emoji(opt['raw_premium'])
            hp_text += f"**{opt['symbol']} ${opt['strike']:.2f} CALL** ({opt['label']})\n"
            hp_text += f"   • Vol: {opt['volume']} ({opt['ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n"
            hp_text += f"   • DTE: {opt['dte']}  Distance: {opt['distance']:.1f}%\n\n"
        if hp_text:
            add_field_safe(embed, "📈 HIGH PROBABILITY SETUPS (30-45 DTE, Near Money)", hp_text, inline=False)

        lt_text = ""
        for opt in lottery[:12]:
            whale = get_whale_emoji(opt['raw_premium'])
            lt_text += f"**{opt['symbol']} ${opt['strike']:.2f} CALL** ({opt['label']})\n"
            lt_text += f"   • Vol: {opt['volume']} ({opt['ratio']:.1f}x)  Premium: {opt['premium']} {whale}\n"
            lt_text += f"   • DTE: {opt['dte']}\n\n"
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
# MARKET STRUCTURE ANALYSIS - CORRECTED VERSION WITH MULTIPLE EVENTS
# ====================
def find_swings(df, window=5):
    """
    Identify swing highs and lows using a rolling window.
    Returns two lists: (swing_highs, swing_lows) where each is a list of (index, price).
    """
    if len(df) < window * 2 + 1:
        return [], []
    
    highs = df['high'].values
    lows = df['low'].values
    idx = df.index
    
    swing_highs = []
    swing_lows = []
    
    for i in range(window, len(df) - window):
        # Swing high: central high is higher than window on both sides
        if highs[i] == max(highs[i-window:i+window+1]):
            swing_highs.append((idx[i], highs[i]))
        # Swing low: central low is lower than window on both sides
        if lows[i] == min(lows[i-window:i+window+1]):
            swing_lows.append((idx[i], lows[i]))
    
    return swing_highs, swing_lows

def analyze_structure(df, window=5):
    """
    Returns a dictionary with:
        trend: 'uptrend' / 'downtrend' / 'sideways'
        last_event: most recent BOS or CHoCH
        last_event_direction: 'up' or 'down'
        bos_events: list of recent BOS events (up to 3)
        choch_events: list of recent CHoCH events (up to 3)
        description: readable summary
    """
    if len(df) < 50:
        return {
            'trend': 'insufficient data',
            'last_event': None,
            'last_event_direction': None,
            'bos_events': [],
            'choch_events': [],
            'description': 'Not enough data to determine market structure.',
            'event_points': None
        }
    
    highs, lows = find_swings(df, window)
    
    # PRIMARY TREND DETECTION: Compare current price to 40 candles ago
    current_price = df['close'].iloc[-1]
    price_40_ago = df['close'].iloc[-40] if len(df) >= 40 else df['close'].iloc[0]
    
    # Calculate percentage change over 40 periods
    pct_change = (current_price - price_40_ago) / price_40_ago * 100
    
    if pct_change > 3:  # Up more than 3%
        trend = 'uptrend'
    elif pct_change < -3:  # Down more than 3%
        trend = 'downtrend'
    else:
        # Secondary: check swing structure
        if len(highs) >= 2 and highs[-1][1] > highs[-2][1]:
            trend = 'uptrend'
        elif len(lows) >= 2 and lows[-1][1] < lows[-2][1]:
            trend = 'downtrend'
        else:
            trend = 'sideways'
    
    # Track BOS events (continuations)
    bos_events = []
    choch_events = []
    
    # BOS: higher high in uptrend, lower low in downtrend
    for i in range(1, min(len(highs), 10)):
        if highs[i][1] > highs[i-1][1]:
            bos_events.append({
                'type': 'BOS',
                'direction': 'up',
                'price': highs[i][1],
                'date': highs[i][0],
                'index': i
            })
    for i in range(1, min(len(lows), 10)):
        if lows[i][1] < lows[i-1][1]:
            bos_events.append({
                'type': 'BOS',
                'direction': 'down',
                'price': lows[i][1],
                'date': lows[i][0],
                'index': i
            })
    
    # CHoCH: lower low in uptrend (reversal down), higher high in downtrend (reversal up)
    if len(highs) >= 3 and len(lows) >= 3:
        # Check for CHoCH down (lower low during uptrend)
        for i in range(2, min(len(lows), 10)):
            if lows[i][1] < lows[i-1][1]:
                choch_events.append({
                    'type': 'CHoCH',
                    'direction': 'down',
                    'price': lows[i][1],
                    'date': lows[i][0],
                    'index': i
                })
        # Check for CHoCH up (higher high during downtrend)
        for i in range(2, min(len(highs), 10)):
            if highs[i][1] > highs[i-1][1]:
                choch_events.append({
                    'type': 'CHoCH',
                    'direction': 'up',
                    'price': highs[i][1],
                    'date': highs[i][0],
                    'index': i
                })
    
    # Keep only last 3 of each
    bos_events = bos_events[-3:] if len(bos_events) > 3 else bos_events
    choch_events = choch_events[-3:] if len(choch_events) > 3 else choch_events
    
    # Determine most recent event (combine both lists)
    all_events = bos_events + choch_events
    all_events.sort(key=lambda x: x['date'], reverse=True)
    last_event = all_events[0] if all_events else None
    
    last_event_type = last_event['type'] if last_event else None
    last_event_direction = last_event['direction'] if last_event else None
    
    description = f"Trend: {trend}. "
    
    if last_event_type == 'BOS':
        if last_event_direction == 'up':
            description += f"Last event: BOS ↑ (uptrend continuing)."
        else:
            description += f"Last event: BOS ↓ (downtrend continuing)."
    elif last_event_type == 'CHoCH':
        if last_event_direction == 'up':
            description += f"Last event: CHoCH ↑ (reversal to uptrend)."
        else:
            description += f"Last event: CHoCH ↓ (reversal to downtrend)."
    else:
        description += "No clear BOS or CHoCH events detected."
    
    return {
        'trend': trend,
        'last_event': last_event_type,
        'last_event_direction': last_event_direction,
        'bos_events': bos_events,
        'choch_events': choch_events,
        'description': description,
        'event_points': None
    }

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
    for i, (idx, row) in enumerate(chart_data.iterrows()):
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        ax.bar(idx, row['High'] - row['Low'], bottom=row['Low'], width=width, color=color, alpha=0.5)
        ax.bar(idx, row['Close'] - row['Open'], bottom=row['Open'], width=width, color=color, alpha=1.0)
    for idx, price in swing_highs:
        if idx in chart_data.index:
            ax.plot(idx, price, '^', color='lime', markersize=10, zorder=5, linewidth=2)
    for idx, price in swing_lows:
        if idx in chart_data.index:
            ax.plot(idx, price, 'v', color='red', markersize=10, zorder=5, linewidth=2)
    
    # Draw BOS events
    if structure.get('bos_events'):
        for bos in structure['bos_events'][-3:]:
            direction = bos['direction']
            price = bos['price']
            date = bos['date']
            if date in chart_data.index:
                line_color = '#00aaff' if direction == 'up' else '#ff8800'
                ax.axhline(y=price, color=line_color, linestyle='--', linewidth=1.5, alpha=0.7)
                label = f"BOS {direction.upper()}"
                ax.text(date, price, label, fontsize=8, color=line_color,
                        ha='left', va='bottom', bbox=dict(facecolor='#1e1e1e', alpha=0.7, pad=1))
    
    # Draw CHoCH events
    if structure.get('choch_events'):
        for choch in structure['choch_events'][-3:]:
            direction = choch['direction']
            price = choch['price']
            date = choch['date']
            if date in chart_data.index:
                line_color = '#ff00ff' if direction == 'up' else '#ff4444'
                ax.axhline(y=price, color=line_color, linestyle='--', linewidth=2, alpha=0.8)
                label = f"CHoCH {direction.upper()}"
                ax.text(date, price, label, fontsize=9, color=line_color,
                        ha='left', va='top', weight='bold',
                        bbox=dict(facecolor='#1e1e1e', alpha=0.8, pad=2))
    
    # Highlight most recent event
    all_events = (structure.get('bos_events', []) + structure.get('choch_events', []))
    all_events.sort(key=lambda x: x['date'], reverse=True)
    if all_events:
        last = all_events[0]
        price = last['price']
        date = last['date']
        if date in chart_data.index:
            ax.axhline(y=price, color='white', linestyle='-', linewidth=3, alpha=0.9)
            label = f"★ MOST RECENT: {last['type']} {last['direction'].upper()}"
            ax.text(date, price, label, fontsize=10, color='yellow',
                    ha='left', va='bottom', weight='bold',
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
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              facecolor='#333333', edgecolor='white', labelcolor='white', framealpha=0.8)
    
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
# STRUCTURE COMMAND – supports both single symbol and 'all' scan
# ====================
@bot.command(name='structure')
async def market_structure(ctx, ticker: str, timeframe: str = '4h'):
    """Analyse market structure for a single symbol OR scan watchlist.
    Usage: !structure AAPL 4h        (individual symbol)
           !structure all 4h         (scan watchlist)
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

        # Accept "all" to scan watchlist
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

            bos_up = []
            choch_up = []
            bos_down = []
            choch_down = []

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

            embed = discord.Embed(
                title=f"📊 Market Structure Scan – {timeframe.upper()}",
                color=0x3498db,
                timestamp=datetime.now()
            )
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

        # Otherwise, treat as single symbol
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

        # Determine action recommendation based on last event
        if structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'up':
            action = "✅ BUY CALLS – Reversal to uptrend detected"
            action_color = 0x00ff00
        elif structure['last_event'] == 'CHoCH' and structure['last_event_direction'] == 'down':
            action = "🔴 SELL CALLS / BUY PUTS – Reversal to downtrend detected"
            action_color = 0xff0000
        elif structure['last_event'] == 'BOS' and structure['last_event_direction'] == 'up':
            action = "📈 HOLD/ADD CALLS – Uptrend continuing"
            action_color = 0x00cc00
        elif structure['last_event'] == 'BOS' and structure['last_event_direction'] == 'down':
            action = "📉 BUY PUTS – Downtrend continuing"
            action_color = 0xcc0000
        else:
            action = "⏸️ WAIT – No clear signal"
            action_color = 0xffff00

        embed = discord.Embed(
            title=f"📈 Market Structure: {symbol} ({timeframe.upper()})",
            description=f"Current Price: **${current_price:.2f}**\n\n**{action}**",
            color=action_color
        )
        embed.add_field(name="Trend", value=structure['trend'].capitalize(), inline=True)
        
        # Show most recent event
        if structure['last_event']:
            emoji = "🟢" if structure['last_event'] == 'BOS' else "🟠"
            direction = "🔼" if structure['last_event_direction'] == 'up' else "🔽"
            embed.add_field(name="📌 Most Recent Event", value=f"{emoji} {structure['last_event']} {direction}", inline=True)
        
        # Show recent BOS events
        if structure['bos_events']:
            bos_text = ""
            for i, bos in enumerate(reversed(structure['bos_events'][-3:]), 1):
                date_str = bos['date'].strftime('%m/%d %H:%M') if hasattr(bos['date'], 'strftime') else str(bos['date'])[:16]
                arrow = "🔼" if bos['direction'] == 'up' else "🔽"
                bos_text += f"{i}. {arrow} BOS {bos['direction'].upper()} at ${bos['price']:.2f} ({date_str})\n"
            embed.add_field(name="📊 Recent BOS (Break of Structure)", value=bos_text, inline=False)
        
        # Show recent CHoCH events
        if structure['choch_events']:
            choch_text = ""
            for i, choch in enumerate(reversed(structure['choch_events'][-3:]), 1):
                date_str = choch['date'].strftime('%m/%d %H:%M') if hasattr(choch['date'], 'strftime') else str(choch['date'])[:16]
                arrow = "🔼" if choch['direction'] == 'up' else "🔽"
                choch_text += f"{i}. {arrow} CHoCH {choch['direction'].upper()} at ${choch['price']:.2f} ({date_str})\n"
            embed.add_field(name="🔄 Recent CHoCH (Change of Character)", value=choch_text, inline=False)
        
        embed.add_field(name="📖 Analysis", value=structure['description'], inline=False)
        embed.add_field(
            name="💡 What this means",
            value=(
                "**BOS (Break of Structure)**: Trend continues.\n"
                "**CHoCH (Change of Character)**: Trend reverses.\n"
                "• BOS up + uptrend = HOLD/ADD CALLS\n"
                "• BOS down + downtrend = BUY PUTS\n"
                "• CHoCH up + downtrend = BUY CALLS\n"
                "• CHoCH down + uptrend = SELL CALLS / BUY PUTS"
            ),
            inline=False
        )
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
    """Check trend strength - RSI, ADX, volume, and divergence."""
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
        
        # RSI
        rsi = df['rsi'].iloc[-1]
        
        # ADX (trend strength) - 25+ = strong trend
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14).iloc[-1]
        
        # Volume trend
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Price vs EMAs
        ema13 = df['ema13'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]
        price_above_ema13 = current_price > ema13
        price_above_ema50 = current_price > ema50
        price_above_ema200 = current_price > ema200
        
        # Divergence detection
        price_last_5 = df['close'].tail(5).values
        rsi_last_5 = df['rsi'].tail(5).values
        
        bearish_divergence = False
        bullish_divergence = False
        
        if len(price_last_5) >= 2 and len(rsi_last_5) >= 2:
            if price_last_5[-1] > price_last_5[-2] and rsi_last_5[-1] < rsi_last_5[-2]:
                bearish_divergence = True
            if price_last_5[-1] < price_last_5[-2] and rsi_last_5[-1] > rsi_last_5[-2]:
                bullish_divergence = True
        
        if adx > 25:
            trend_strength_msg = f"🟢 STRONG (ADX: {adx:.1f})"
        elif adx > 20:
            trend_strength_msg = f"🟡 MODERATE (ADX: {adx:.1f})"
        else:
            trend_strength_msg = f"🔴 WEAK / RANGING (ADX: {adx:.1f})"
        
        if rsi > 70:
            rsi_status = "🔴 OVERBOUGHT (sell signal possible)"
        elif rsi < 30:
            rsi_status = "🟢 OVERSOLD (buy signal possible)"
        else:
            rsi_status = f"⚪ NEUTRAL ({rsi:.1f})"
        
        if volume_ratio > 1.5:
            volume_status = f"🔊 HIGH volume ({volume_ratio:.1f}x average)"
        elif volume_ratio < 0.5:
            volume_status = f"🔇 LOW volume ({volume_ratio:.1f}x average)"
        else:
            volume_status = f"📊 NORMAL volume ({volume_ratio:.1f}x average)"
        
        embed = discord.Embed(
            title=f"📊 Trend Strength: {symbol} ({timeframe.upper()})",
            description=f"Current Price: **${current_price:.2f}**",
            color=0x3498db,
            timestamp=datetime.now()
        )
        
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
        
        embed.add_field(
            name="💡 Trading Advice",
            value=(
                "• **ADX > 25** = Strong trend (trade with trend)\n"
                "• **ADX < 20** = Weak trend (avoid, wait for breakout)\n"
                "• **RSI > 70** = Overbought (consider taking profits)\n"
                "• **RSI < 30** = Oversold (look for buying opportunities)\n"
                "• **Divergence** = Early reversal warning\n"
                "Use `!structure` for exact BOS/CHoCH signals."
            ),
            inline=False
        )
        
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

            embed = discord.Embed(
                title=f"📉 Demand Zones for {symbol} (30min)",
                description=f"Current Price: **${current_price:.2f}**",
                color=0x00ff00
            )
            if structure:
                struct_text = f"**{structure['trend'].capitalize()}** – {structure['description']}"
                embed.add_field(name="🏛️ Market Structure (4h)", value=struct_text, inline=False)

            for z in zones:
                distance = (current_price - z['level']) / current_price * 100
                status = "🔵 **NEAR**" if abs(distance) < 2 else ""
                date_str = z['date'].strftime('%m/%d') if hasattr(z['date'], 'strftime') else ''
                embed.add_field(
                    name=f"Support at ${z['level']:.2f} ({date_str})",
                    value=f"Distance: {distance:.1f}% {status}\nTouches: {z['strength']}",
                    inline=False
                )

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
                            if price > 100:
                                offset = 5.0
                            elif price > 50:
                                offset = 2.0
                            elif price > 10:
                                offset = 1.0
                            else:
                                offset = max(0.5, price * 0.15)

                            target_strike = price - offset
                            calls = opt_chain.calls
                            if not calls.empty:
                                calls['strike_diff'] = abs(calls['strike'] - target_strike)
                                best_call = calls.loc[calls['strike_diff'].idxmin()]

                                strike = best_call['strike']
                                last = best_call.get('lastPrice', 'N/A')
                                bid = best_call.get('bid', 'N/A')
                                ask = best_call.get('ask', 'N/A')
                                volume = best_call.get('volume', 'N/A')

                                if bid != 'N/A' and ask != 'N/A' and bid > 0 and ask > 0:
                                    premium = (bid + ask) / 2
                                else:
                                    premium = last if last != 'N/A' else None

                                if premium:
                                    breakeven = strike + premium
                                else:
                                    breakeven = 'N/A'

                                option_text = (
                                    f"**Strike:** ${strike:.2f}\n"
                                    f"**Expiration:** {primary_exp}\n"
                                    f"**Last:** {last}\n"
                                    f"**Bid/Ask:** {bid}/{ask}\n"
                                    f"**Volume:** {volume}\n"
                                    f"**Est. Premium:** ${premium:.2f}\n"
                                    f"**Breakeven:** ${breakeven:.2f}" if breakeven != 'N/A' else "Breakeven N/A"
                                )
                                embed.add_field(
                                    name="💡 Suggested Call Option (ITM)",
                                    value=option_text,
                                    inline=False
                                )
                except Exception as e:
                    embed.add_field(name="Options suggestion", value=f"Could not fetch options: {str(e)}", inline=False)

            if structure and structure['last_event'] == 'BOS' and structure['trend'] == 'downtrend':
                embed.add_field(
                    name="⚠️ Trading Advice",
                    value="**Downtrend with BOS – trend continuing DOWN.** Consider PUTS or stay away.",
                    inline=False
                )
            elif structure and structure['last_event'] == 'CHoCH' and structure['trend'] == 'downtrend' and structure['last_event_direction'] == 'up':
                embed.add_field(
                    name="📈 Trading Advice",
                    value="**Change of Character (CHoCH) detected – reversal to UPTREND.** Consider CALLS.",
                    inline=False
                )
            elif structure and structure['trend'] == 'uptrend':
                embed.add_field(
                    name="📈 Trading Advice",
                    value="**Uptrend confirmed.** This demand zone is a potential bounce area. Consider CALLS.",
                    inline=False
                )

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

        embed = discord.Embed(
            title="📈 LEAPS CANDIDATES",
            description="Stocks showing Change of Character (CHoCH) on the weekly chart – potential long-term reversal setups.",
            color=0x00ff00
        )

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
                            bid = best_call.get('bid', 'N/A')
                            ask = best_call.get('ask', 'N/A')
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

            embed.add_field(
                name=f"{symbol} – ${price:.2f}",
                value=f"{structure['description']}\n{rating_info}\n\n💡 Suggested LEAPS: {option_text}",
                inline=False
            )

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
            help_embed = discord.Embed(
                title="🔎 Finviz Scanner Help",
                description="Use `!finviz_scan` followed by a comma‑separated list of filters.\n\n**Example:**\n`!finviz_scan Price: $10 to $20, Option/Short: Optionable, Average Volume: Over 1M, Relative Volume: Over 1.5, Change: Up 3%`\n\n**Available filters (common ones):**\n- `Price: $10 to $20`\n- `Option/Short: Optionable`\n- `Average Volume: Over 1M`\n- `Relative Volume: Over 1.5`\n- `Change: Up 3%`\n- `Market Cap.: +Mid (over $2bln)`\n- `EPS growth this year: Positive`\n- `RSI (14): Over 50`\n- `20-Day Simple Moving Average: Price above SMA20`\n\n**Tip:** You can combine many filters. The scanner will return up to 50 results.",
                color=0x3498db
            )
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
        embed = discord.Embed(
            title="📊 Finviz Scan Results",
            description=f"Filters used: {filters[:200]}{'...' if len(filters) > 200 else ''}",
            color=0x00ff00,
            timestamp=datetime.now()
        )
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
        calls = opt_chain.calls
        puts = opt_chain.puts

        current_price = stock.history(period="1d")['Close'].iloc[-1]
        if calls.empty or puts.empty:
            return "N/A", "N/A"

        calls['diff'] = abs(calls['strike'] - current_price)
        puts['diff'] = abs(puts['strike'] - current_price)
        atm_call = calls.loc[calls['diff'].idxmin()]
        atm_put = puts.loc[puts['diff'].idxmin()]

        call_price = (atm_call['bid'] + atm_call['ask']) / 2 if atm_call['bid'] > 0 and atm_call['ask'] > 0 else atm_call['lastPrice']
        put_price = (atm_put['bid'] + atm_put['ask']) / 2 if atm_put['bid'] > 0 and atm_put['ask'] > 0 else atm_put['lastPrice']
        straddle = call_price + put_price

        expected_pct = (straddle / current_price) * 100

        return f"{expected_pct:.1f}%", "N/A"

    except Exception as e:
        print(f"Error getting earnings stats for {symbol}: {e}")
        return "N/A", "N/A"

@bot.command(name='upcoming')
async def upcoming_events(ctx, ticker: str = None):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now

        if ticker is None:
            watchlist = await load_watchlist()
            stocks = watchlist['stocks']
            if not stocks:
                await ctx.send("No stocks in your watchlist to scan for events.")
                return

            await ctx.send(f"🔍 Scanning all stocks ({len(stocks)}) for upcoming events...")

            econ_events = await fetch_economic_events(days=14)
            if econ_events:
                econ_embed = discord.Embed(
                    title="📆 Upcoming Macroeconomic Events (next 14 days)",
                    color=0x3498db
                )
                for ev in econ_events:
                    date = ev.get('date', 'N/A')
                    event = ev.get('event', 'N/A')
                    country = ev.get('country', '')
                    importance = ev.get('importance', '')
                    star = "★" if importance == "high" else "☆" if importance == "medium" else "·" if importance else ""
                    econ_embed.add_field(
                        name=f"{date} {country}",
                        value=f"{event} {star}",
                        inline=False
                    )
                await ctx.send(embed=econ_embed)

            found_any = False
            for sym in stocks:
                if cancellation_flags.get(ctx.author.id, False):
                    cancellation_flags[ctx.author.id] = False
                    await ctx.send("🛑 Scan cancelled.")
                    break

                try:
                    earnings_task = asyncio.create_task(fetch_earnings_upcoming(sym))
                    dividends_task = asyncio.create_task(fetch_dividends_upcoming(sym))
                    splits_task = asyncio.create_task(fetch_splits_upcoming(sym))
                    ratings_task = asyncio.create_task(fetch_analyst_ratings(sym, limit=3))

                    earnings, dividends, splits, ratings = await asyncio.wait_for(
                        asyncio.gather(earnings_task, dividends_task, splits_task, ratings_task, return_exceptions=True),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    print(f"Timeout fetching data for {sym}, skipping.")
                    continue

                if isinstance(earnings, Exception):
                    earnings = []
                if isinstance(dividends, Exception):
                    dividends = []
                if isinstance(splits, Exception):
                    splits = []
                if isinstance(ratings, Exception):
                    ratings = []

                if earnings or dividends or splits or ratings:
                    found_any = True
                    web_url = get_tradingview_web_link(sym)
                    tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"

                    embed = discord.Embed(
                        title=f"📅 Upcoming Catalysts for {sym}",
                        color=0x00ff00
                    )

                    if earnings:
                        lines = []
                        for e in earnings:
                            date = e.get('date', 'N/A')
                            eps_est = e.get('epsEstimate', 'N/A')
                            time_str = e.get('hour', '')
                            exp_move, hist_avg = await get_earnings_stats(sym, date)
                            lines.append(
                                f"**{date}** {time_str} – EPS Est: {eps_est}\n"
                                f"   • Expected Move: {exp_move}\n"
                                f"   • Historical Avg: {hist_avg}"
                            )
                        embed.add_field(name="📊 Earnings", value="\n".join(lines), inline=False)

                    if dividends:
                        lines = []
                        for d in dividends:
                            ex_date = d.get('exDate', 'N/A')
                            amount = d.get('amount', 'N/A')
                            lines.append(f"**{ex_date}** – Amount: ${amount}")
                        embed.add_field(name="💰 Dividends", value="\n".join(lines), inline=False)

                    if splits:
                        lines = []
                        for s in splits:
                            date = s.get('date', 'N/A')
                            ratio = s.get('splitRatio', '')
                            lines.append(f"**{date}** – {ratio}")
                        embed.add_field(name="🔄 Stock Splits", value="\n".join(lines), inline=False)

                    if ratings:
                        lines = []
                        for r in ratings:
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
                                lines.append(f"**{period}** – {buys} Buy / {h} Hold / {sells} Sell {sentiment}")
                            else:
                                lines.append(f"**{period}** – No data")
                        embed.add_field(name="📈 Analyst Ratings (last 3)", value="\n".join(lines), inline=False)

                    embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
                    await ctx.send(embed=embed)

                await asyncio.sleep(2)

            if not found_any:
                await ctx.send("No upcoming events found for any stock in your watchlist.")
            else:
                await ctx.send("✅ Upcoming events scan complete.")

        else:
            await ctx.send(f"🔍 Fetching upcoming events for **{ticker.upper()}**...")
            earnings = await fetch_earnings_upcoming(ticker.upper())
            dividends = await fetch_dividends_upcoming(ticker.upper())
            splits = await fetch_splits_upcoming(ticker.upper())
            ratings = await fetch_analyst_ratings(ticker.upper(), limit=3)

            if not (earnings or dividends or splits or ratings):
                await ctx.send(f"No upcoming events found for {ticker.upper()} in the next 14 days.")
                return

            web_url = get_tradingview_web_link(ticker.upper())
            tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"

            embed = discord.Embed(
                title=f"📅 Upcoming Catalysts for {ticker.upper()}",
                color=0x00ff00
            )

            if earnings:
                lines = []
                for e in earnings:
                    date = e.get('date', 'N/A')
                    eps_est = e.get('epsEstimate', 'N/A')
                    time_str = e.get('hour', '')
                    exp_move, hist_avg = await get_earnings_stats(ticker.upper(), date)
                    lines.append(
                        f"**{date}** {time_str} – EPS Est: {eps_est}\n"
                        f"   • Expected Move: {exp_move}\n"
                        f"   • Historical Avg: {hist_avg}"
                    )
                embed.add_field(name="📊 Earnings", value="\n".join(lines), inline=False)

            if dividends:
                lines = []
                for d in dividends:
                    ex_date = d.get('exDate', 'N/A')
                    amount = d.get('amount', 'N/A')
                    lines.append(f"**{ex_date}** – Amount: ${amount}")
                embed.add_field(name="💰 Dividends", value="\n".join(lines), inline=False)

            if splits:
                lines = []
                for s in splits:
                    date = s.get('date', 'N/A')
                    ratio = s.get('splitRatio', '')
                    lines.append(f"**{date}** – {ratio}")
                embed.add_field(name="🔄 Stock Splits", value="\n".join(lines), inline=False)

            if ratings:
                lines = []
                for r in ratings:
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
                        lines.append(f"**{period}** – {buys} Buy / {h} Hold / {sells} Sell {sentiment}")
                    else:
                        lines.append(f"**{period}** – No data")
                embed.add_field(name="📈 Analyst Ratings (last 3)", value="\n".join(lines), inline=False)

            embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
            await ctx.send(embed=embed)

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

        signals_list = []
        for i in range(len(df)):
            sig = get_signals(df.iloc[:i+1])
            if sig:
                signals_list.append(sig)

        dates = df.index.tolist()
        equity = [1.0]
        in_position = False
        entry_price = 0
        entry_date = None
        position_type = None
        trades = []
        equity_dates = [dates[0]]

        for i in range(1, len(signals_list)):
            sig = signals_list[i]
            prev_sig = signals_list[i-1]
            today_open = df['open'].iloc[i] if i < len(df) else None

            if not in_position:
                if prev_sig['net_score'] > 0:
                    if today_open is None:
                        continue
                    in_position = True
                    position_type = 'long'
                    entry_price = today_open
                    entry_date = df.index[i]
                    equity[-1] *= (1 - cost)
                elif prev_sig['net_score'] < 0:
                    if today_open is None:
                        continue
                    in_position = True
                    position_type = 'short'
                    entry_price = today_open
                    entry_date = df.index[i]
                    equity[-1] *= (1 - cost)

            elif in_position:
                exit_signal = False
                if position_type == 'long' and prev_sig['net_score'] < 0:
                    exit_signal = True
                elif position_type == 'short' and prev_sig['net_score'] > 0:
                    exit_signal = True

                if exit_signal:
                    exit_price = today_open if today_open is not None else df['close'].iloc[-1]
                    exit_date = df.index[i] if today_open is not None else df.index[-1]

                    if position_type == 'long':
                        ret = (exit_price - entry_price) / entry_price
                    else:
                        ret = (entry_price - exit_price) / entry_price

                    new_equity = equity[-1] * (1 + ret) * (1 - cost)
                    equity.append(new_equity)
                    equity_dates.append(exit_date)

                    trades.append({'entry_date': entry_date, 'exit_date': exit_date, 'type': position_type, 'ret': ret})
                    in_position = False

        if in_position:
            exit_price = df['close'].iloc[-1]
            exit_date = df.index[-1]
            if position_type == 'long':
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price
            new_equity = equity[-1] * (1 + ret) * (1 - cost)
            equity.append(new_equity)
            equity_dates.append(exit_date)
            trades.append({'entry_date': entry_date, 'exit_date': exit_date, 'type': position_type, 'ret': ret})

        if not trades:
            await ctx.send("No trades generated during this period.")
            return

        final_equity = equity[-1]
        total_return = (final_equity - 1) * 100

        winning_trades = [t for t in trades if t['ret'] > 0]
        losing_trades = [t for t in trades if t['ret'] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100

        avg_win = np.mean([t['ret']*100 for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['ret']*100 for t in losing_trades]) if losing_trades else 0

        gross_profit = sum(t['ret'] for t in winning_trades)
        gross_loss = abs(sum(t['ret'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = np.max(drawdown)

        embed = discord.Embed(
            title=f"📈 BACKTEST RESULTS: {symbol.upper()}",
            description=f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}\nTransaction cost: {cost*100:.1f}% per trade",
            color=0x00ff00 if total_return > 0 else 0xff0000
        )
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
                print(f"⚠️ Chart generation failed: {e}")
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
                    print(f"⚠️ Chart generation failed: {e}")
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
        closes_below = after['close'] < level * (1 - touch_tolerance)
        if closes_below.any():
            continue
        touches = after['low'] <= level * (1 + touch_tolerance)
        strength = int(touches.sum())
        if strength >= 1:
            zones.append({
                'level': level,
                'date': idx,
                'strength': strength
            })
    zones.sort(key=lambda x: x['level'])
    return zones

# ====================
# TRADE TRACKING COMMANDS
# ====================
@bot.command(name='track')
async def track_trade(ctx, action: str, symbol: str, *, details: str = None):
    """Track your trades. Usage:
    !track BUY NVDA CALL 200 strike 1.50
    !track SELL NVDA 1.80
    !track CLOSE NVDA profit 15%
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

        action = action.upper()
        symbol = symbol.upper()

        if action in ['BUY', 'OPEN']:
            # Parse details: e.g., "CALL 200 strike 1.50" or "PUT 180 2.00"
            parts = details.split() if details else []
            option_type = parts[0].upper() if parts and parts[0] in ['CALL', 'PUT'] else None
            strike = float(parts[1]) if len(parts) > 1 else None
            premium = float(parts[-1]) if parts else None
            
            trade = {
                'user_id': str(ctx.author.id),
                'symbol': symbol,
                'action': 'BUY',
                'option_type': option_type,
                'strike': strike,
                'premium': premium,
                'entry_date': now,
                'entry_price': premium,
                'status': 'OPEN',
                'strategy': None  # Optional: add strategy tag later
            }
            await trades_collection.insert_one(trade)
            await ctx.send(f"✅ Tracked: **BUY {symbol} {option_type if option_type else ''} {f'${strike} strike ' if strike else ''}at ${premium:.2f}**" if premium else f"✅ Tracked: **BUY {symbol}**")
            
        elif action in ['SELL', 'CLOSE']:
            # Find most recent open trade for this symbol
            open_trade = await trades_collection.find_one(
                {'user_id': str(ctx.author.id), 'symbol': symbol, 'status': 'OPEN'},
                sort=[('entry_date', -1)]
            )
            if not open_trade:
                await ctx.send(f"❌ No open trade found for {symbol}. Use `!track BUY {symbol}` first.")
                return
            
            # Parse exit price or profit percentage
            if details:
                if details.endswith('%'):
                    profit_pct = float(details[:-1])
                    entry_premium = open_trade.get('premium', 0)
                    if entry_premium > 0:
                        exit_premium = entry_premium * (1 + profit_pct / 100)
                    else:
                        exit_premium = None
                else:
                    exit_premium = float(details)
                    profit_pct = ((exit_premium - open_trade.get('premium', 0)) / open_trade.get('premium', 1)) * 100 if open_trade.get('premium') else None
            else:
                exit_premium = None
                profit_pct = None
            
            # Update the trade
            await trades_collection.update_one(
                {'_id': open_trade['_id']},
                {'$set': {
                    'status': 'CLOSED',
                    'exit_date': now,
                    'exit_premium': exit_premium,
                    'profit_pct': profit_pct,
                    'profit_amount': (exit_premium - open_trade.get('premium', 0)) * 100 if exit_premium and open_trade.get('premium') else None
                }}
            )
            
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

@bot.command(name='performance', aliases=['stats', 'perf'])
async def performance(ctx, days: int = 30):
    """Show your trading performance over the last N days (default 30)."""
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        
        # Get all closed trades for this user
        trades = await trades_collection.find({
            'user_id': str(ctx.author.id),
            'status': 'CLOSED',
            'exit_date': {'$gte': cutoff}
        }).to_list(length=1000)
        
        if not trades:
            await ctx.send(f"📭 No closed trades found in the last {days} days. Start tracking with `!track BUY SYMBOL`")
            return
        
        # Calculate stats
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('profit_pct', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit_pct', 0) < 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit_pct', 0) for t in trades)
        avg_win = sum(t.get('profit_pct', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum(t.get('profit_pct', 0) for t in losing_trades) / len(losing_trades)) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.get('profit_pct', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit_pct', 0) for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Best/worst trade
        best_trade = max(trades, key=lambda x: x.get('profit_pct', -999)) if trades else None
        worst_trade = min(trades, key=lambda x: x.get('profit_pct', 999)) if trades else None
        
        # By symbol
        symbol_stats = {}
        for t in trades:
            sym = t['symbol']
            if sym not in symbol_stats:
                symbol_stats[sym] = {'trades': 0, 'wins': 0, 'profit': 0}
            symbol_stats[sym]['trades'] += 1
            symbol_stats[sym]['profit'] += t.get('profit_pct', 0)
            if t.get('profit_pct', 0) > 0:
                symbol_stats[sym]['wins'] += 1
        
        # Build embed
        embed = discord.Embed(
            title=f"📊 Trading Performance (Last {days} days)",
            color=0x00ff00 if total_profit > 0 else 0xff0000,
            timestamp=now
        )
        
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
        
        # Top 3 best symbols
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

@bot.command(name='trades')
async def list_trades(ctx, limit: int = 10):
    """List your recent trades."""
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        trades = await trades_collection.find({
            'user_id': str(ctx.author.id)
        }).sort('entry_date', -1).to_list(length=limit)
        
        if not trades:
            await ctx.send("No trades found. Start tracking with `!track BUY SYMBOL`")
            return
        
        embed = discord.Embed(
            title=f"📋 Recent Trades (Last {len(trades)})",
            color=0x3498db,
            timestamp=datetime.now()
        )
        
        trade_list = ""
        for t in trades:
            status = "🟢 OPEN" if t.get('status') == 'OPEN' else "🔒 CLOSED"
            profit_str = f"{t.get('profit_pct', 0):+.1f}%" if t.get('profit_pct') else "N/A"
            entry_date = t['entry_date'].strftime('%m/%d') if hasattr(t['entry_date'], 'strftime') else str(t['entry_date'])[:10]
            trade_list += f"**{t['symbol']}** {status} | Entry: ${t.get('premium', 0):.2f} | P&L: {profit_str} | {entry_date}\n"
        
        embed.add_field(name="Trades", value=trade_list or "None", inline=False)
        embed.set_footer(text="Use !performance for detailed stats")
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
                "`!scan all [timeframe]`\n"
                "  Scan all watchlist symbols on a single timeframe (5min,15min,30min,1h,4h,daily,weekly)\n"
                "`!scan SYMBOL [timeframe]`\n"
                "  Scan a single symbol on a specific timeframe\n"
                "`!signals`\n"
                "  Scan your ENTIRE watchlist across ALL 7 timeframes (fast, uses Alpaca)\n"
                "`!signal SYMBOL`\n"
                "  Multi‑timeframe report for a single symbol\n\n"
                "**NEWS & EVENTS**\n"
                "`!news TICKER`\n"
                "  Comprehensive, actionable news (analyst actions, product launches, institutional moves)\n"
                "`!worldnews`\n"
                "  Global headlines with market impact analysis and suggested stocks\n"
                "`!upcoming [TICKER]`\n"
                "  Upcoming catalysts (earnings, dividends, splits, analyst ratings, expected move)\n\n"
                "**ZONES & STRUCTURE**\n"
                "`!zone SYMBOL [timeframe]`\n"
                "  Default 30min – shows demand zones with strength‑colored lines and ITM option suggestions. Also includes 4h market structure analysis.\n"
                "`!structure SYMBOL [timeframe]`\n"
                "  Analyse market structure (BOS / CHoCH) on 1h, 4h, daily, or weekly. Includes chart with swing points and event lines.\n"
                "`!structure all [timeframe]`\n"
                "  Scan watchlist for BOS/CHoCH events.\n\n"
                "**TREND STRENGTH**\n"
                "`!strength SYMBOL [timeframe]`\n"
                "  Check trend strength with ADX, RSI, volume, and divergence detection.\n\n"
                "**TACO TRADE**\n"
                "`!taco`\n"
                "  Get actionable trade signal based on Trump's latest post (Trump Always Chickens Out).\n"
                "`!pennant [timeframe]`\n"
                "  Scan watchlist for bullish pennant (flat bottom) patterns.\n\n"
                "**OPTIONS FLOW**\n"
                "`!flow TICKER`\n"
                "  Unusual options activity – high probability setups first\n"
                "`!scanflow`\n"
                "  Scan watchlist for unusual options – high probability first\n\n"
                "**BACKTESTING & LEAPS**\n"
                "`!backtest SYMBOL [days=365]`\n"
                "  Backtest EMA crossover strategy, returns win rate, profit factor, max drawdown\n"
                "`!leaps`\n"
                "  Scan watchlist for LEAPS candidates (stocks with CHoCH on weekly chart)\n\n"
                "**FINVIZ SCANNER**\n"
                "`!finviz_scan [filters]`\n"
                "  Run a custom Finviz scan (e.g., `!finviz_scan Price: $10 to $20, Option/Short: Optionable`)\n"
                "`!cheap_options`\n"
                "  Preset: $10‑20, optionable, avg volume >1M, rel volume >1.5, up 3%\n"
                "`!hype_stocks`\n"
                "  Preset: rel volume >2, up 5%, optionable, avg volume >500K\n"
                "`!cheap_plays`\n"
                "  Alias for `!cheap_options`\n\n"
                "**TRADE TRACKING**\n"
                "`!track BUY SYMBOL CALL/PUT STRIKE PREMIUM`\n"
                "  Record a new trade entry\n"
                "`!track CLOSE SYMBOL EXIT_PRICE`\n"
                "  Close a trade and calculate profit/loss\n"
                "`!performance [days]`\n"
                "  Show win rate, total P&L, best/worst trades\n"
                "`!trades [limit]`\n"
                "  List your recent trades\n\n"
                "**WATCHLIST**\n"
                "`!add SYMBOL`\n"
                "  Add stock or crypto (use BTC/USD for crypto)\n"
                "`!remove SYMBOL`\n"
                "  Remove from watchlist\n"
                "`!list`\n"
                "  Show watchlist\n\n"
                "**UTILITY**\n"
                "`!ping`\n"
                "  Test bot\n"
                "`!stopscan` / `!cancel`\n"
                "  Stop ongoing scan\n"
                "`!help`\n"
                "  This message\n\n"
                "**TIMEFRAMES**\n"
                "5min, 15min, 30min, 1h, 4h, daily, weekly\n\n"
                "💡 **Pro tip:** Focus on High Probability Setups (30‑45 DTE, near money) for consistent wins"
            ),
            color=0x3498db
        )
        embed.set_footer(text="Use !help for this menu")
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send("📚 Commands: !scan, !signals, !signal, !news, !worldnews, !upcoming, !zone, !structure, !strength, !taco, !pennant, !leaps, !finviz_scan, !cheap_options, !hype_stocks, !cheap_plays, !flow, !scanflow, !backtest, !track, !performance, !trades, !add, !remove, !list, !ping, !stopscan, !cancel")
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
                print(f"Error scanning {symbol} for pennant: {e}")
            await asyncio.sleep(0.3)

        if not found:
            await ctx.send(f"📭 No bullish pennants found on {timeframe}.")
            return

        embed = discord.Embed(
            title=f"🚩 Bullish Pennant (Flag) Scan – {timeframe.upper()}",
            description=f"Found {len(found)} stocks with flat-bottom pennants",
            color=0x00ff00,
            timestamp=datetime.now()
        )
        embed.add_field(name="📈 Potential Breakout Candidates", value="\n".join(found[:10]), inline=False)
        embed.add_field(
            name="💡 Trading Advice",
            value="• Wait for price to break ABOVE the descending resistance line.\n"
                  "• Entry on breakout + volume confirmation.\n"
                  "• Stop loss just below the flat support.\n"
                  "• Use `!structure SYMBOL` to see the chart.",
            inline=False
        )
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

        embed = discord.Embed(
            title="🌮 TACO TRADE SIGNAL",
            description="**Trump Always Chickens Out** – trade the reversal",
            color=0xff6600,
            timestamp=datetime.now()
        )

        embed.add_field(name="📢 Latest Truth", value=f"> {trump_post['text'][:500]}{'…' if len(trump_post['text'])>500 else ''}", inline=False)
        embed.add_field(name="📅 Time", value=trump_post['timestamp'], inline=True)
        embed.add_field(name="📊 Sentiment", value=analysis['sentiment'], inline=True)
        embed.add_field(name="🎲 TACO Probability", value=analysis['taco_probability'], inline=False)
        embed.add_field(name="🎯 Affected Sectors", value=analysis['affected_sectors'], inline=True)
        embed.add_field(name="💎 Suggested Stocks", value=analysis['suggested_stocks'], inline=True)

        if "Bearish" in analysis['sentiment'] and "High" in analysis['taco_probability']:
            embed.add_field(
                name="✅ TRADE ACTION",
                value=(
                    "**BUY CALLS** on the suggested stocks.\n"
                    "• Wait for the market to dip on the threat.\n"
                    "• Look for a **flat bottom (bullish pennant)** on the 4h chart.\n"
                    "• Enter when price breaks above descending resistance.\n"
                    "• Set stop loss just below the flat support.\n"
                    "• Target: +10‑20% move when Trump backtracks.\n"
                    "Use `!structure SYMBOL` to see the chart and `!pennant` to find flat bottoms."
                ),
                inline=False
            )
        elif "Bullish" in analysis['sentiment']:
            embed.add_field(
                name="✅ TRADE ACTION",
                value=(
                    "**SELL / TAKE PROFITS** on the suggested stocks.\n"
                    "• Trump is talking peace – the rally may end soon.\n"
                    "• Consider selling calls or buying puts after a bounce.\n"
                    "• Wait for a CHoCH down on the 4h chart.\n"
                    "Use `!structure SYMBOL` to confirm reversal."
                ),
                inline=False
            )
        else:
            embed.add_field(
                name="⏳ TRADE ACTION",
                value=(
                    "No clear TACO signal yet. **Monitor** for threats or peace talks.\n"
                    "When you see a threat, prepare to buy dips. When you see peace talks, prepare to sell.\n"
                    "Use `!worldnews` for broader context."
                ),
                inline=False
            )

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
        print(f"⚠️ Unexpected error in send_symbol_with_chart for {symbol}: {e}")
        await ctx.send(embed=embed)

# ====================
# TRUMP / TACO FUNCTIONS
# ====================
async def fetch_latest_trump_post():
    """Fetch the latest post from Trump's Truth Social using multiple RSS feeds."""
    feeds = [
        "https://trumpstruth.org/feed",           # Community-run, more reliable
        "https://truthsocial.rss.simple-web.org/feed.xml",  # Backup
    ]
    
    for url in feeds:
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
                    root = ET.fromstring(text)
                    # Try RSS 2.0 first
                    item = root.find('.//item')
                    if item is not None:
                        title = item.find('title').text
                        pub_date = item.find('pubDate').text
                        link = item.find('link').text
                        return {
                            'text': title,
                            'timestamp': pub_date,
                            'url': link
                        }
                    # Try Atom
                    entry = root.find('.//{http://www.w3.org/2005/Atom}entry')
                    if entry is not None:
                        title = entry.find('{http://www.w3.org/2005/Atom}title').text
                        pub_date = entry.find('{http://www.w3.org/2005/Atom}updated').text
                        link = entry.find('{http://www.w3.org/2005/Atom}link').get('href')
                        return {
                            'text': title,
                            'timestamp': pub_date,
                            'url': link
                        }
        except Exception as e:
            print(f"Error with feed {url}: {e}")
            continue
    return None

def analyze_trump_post(post_text):
    text_lower = post_text.lower()
    threat_keywords = ['tariff', 'sanction', 'war', 'military', 'bomb', 'strike', 'deadline', 'penalty', 'tax', 'duty', 'ban', 'restrict', 'withdraw']
    backtrack_keywords = ['exemption', 'delay', 'postpone', 'review', 'negotiate', 'deal', 'agreement', 'temporary', 'exception', 'reconsider']

    has_threat = any(kw in text_lower for kw in threat_keywords)
    has_backtrack = any(kw in text_lower for kw in backtrack_keywords)

    if has_threat and not has_backtrack:
        sentiment = "🔴 Bearish"
        taco_prob = "High (expected market dip & reversal)"
        advice = "TACO play likely. Watch for dip – DO NOT buy puts. Wait for dip to buy calls."
    elif has_threat and has_backtrack:
        sentiment = "🟡 Mixed (threat with caveats)"
        taco_prob = "Medium – possible backtrack"
        advice = "Monitor for clarification. Could swing either way."
    elif any(word in text_lower for word in ['good', 'great', 'deal', 'agreement']):
        sentiment = "🟢 Bullish"
        taco_prob = "Low – positive news"
        advice = "Potential rally. Check related sectors."
    else:
        sentiment = "⚪ Neutral"
        taco_prob = "Low – no clear threat or positive"
        advice = "No immediate TACO signal."

    sectors = []
    stocks = []
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

    sectors = list(dict.fromkeys(sectors))
    stocks = list(dict.fromkeys(stocks))[:5]

    return {
        'sentiment': sentiment,
        'taco_probability': taco_prob,
        'affected_sectors': ', '.join(sectors) if sectors else 'General market',
        'suggested_stocks': ', '.join([f"${s}" for s in stocks]) if stocks else '$SPY, $QQQ',
        'advice': advice
    }

# ====================
# MAIN ENTRY POINT
# ====================
async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())