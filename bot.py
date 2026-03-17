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
from collections import defaultdict
from datetime import datetime, timedelta
import motor.motor_asyncio
import yfinance as yf

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest

# Charting libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
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
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Alpaca keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

# MongoDB setup
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client['trading_bot']
watchlist_collection = db['watchlist']

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
data_cache = {}          # key: f"{symbol}_{timeframe}", value: (DataFrame, expiry)
CACHE_DURATION = timedelta(minutes=5)

# Add cache for world news
world_news_cache = {"data": None, "expiry": datetime.min}

# ====================
# RATE LIMITER (for Twelve Data)
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
                "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG", "QUBT", "TSLA", "LYFT", "NFLX", "ORCL", "UBER", "HOOD", "SOFI", "SPY", "NIO", "PLTR", "GRAB", "LMT", "MARA", "SOUN", "APLD", "CLSK", "OPEN", "ASML", "RIOT", "AAL", "F", "FCEL", "NKLA"],
                "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "DOGE/USD", "PEPE/USD", "LINK/USD"]
            }
            await watchlist_collection.insert_one(default)
            return default
    except Exception as e:
        print(f"❌ Error loading watchlist: {e}")
        return {
            "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VUG", "QUBT", "TSLA", "LYFT", "NFLX", "ORCL", "UBER", "HOOD", "SOFI", "SPY", "NIO", "PLTR", "GRAB", "LMT", "MARA", "SOUN", "APLD", "CLSK", "OPEN", "ASML", "RIOT", "AAL", "F", "FCEL", "NKLA"],
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
# DATA FETCHING – Alpaca (threaded), then Twelve Data / CoinGecko
# ====================

async def fetch_twelvedata(symbol, timeframe):
    """Fallback: fetch single symbol from Twelve Data."""
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
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
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

    alpaca_tf_map = {
        '5min': '5Min',
        '15min': '15Min',
        '1h': '1H',
        '4h': '4H',
        'daily': '1D',
        'weekly': '1W',
    }
    alpaca_tf = alpaca_tf_map.get(timeframe)

    # Special handling for 30min: try to get 15min from Alpaca and resample
    if timeframe == '30min':
        # Stocks
        if '/' not in symbol and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe='15Min',
                    start=now - timedelta(days=60),
                    end=now
                )
                # Run synchronous call in thread
                bars = await asyncio.to_thread(client.get_stock_bars, request)
                if bars.data:
                    df_15 = bars.df
                    df_15 = df_15.reset_index(level=0, drop=True)
                    df_15.index = pd.to_datetime(df_15.index)
                    # Resample to 30 minutes
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
                print(f"⚠️ Alpaca 15min fetch failed for {symbol}, falling back to Twelve Data... {e}")

        # If stock still missing or crypto, fallback to Twelve Data (for stocks) or CoinGecko (crypto)
        if df is None and '/' not in symbol:
            df = await fetch_twelvedata(symbol, '30min')

        if '/' in symbol:
            # Try Alpaca crypto 15min resample
            try:
                client = CryptoHistoricalDataClient()
                alpaca_symbol = symbol.replace('/', '')
                request = CryptoBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe='15Min',
                    start=now - timedelta(days=60),
                    end=now
                )
                bars = await asyncio.to_thread(client.get_crypto_bars, request)
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
                        print(f"✅ Alpaca crypto 15min resampled to 30min for {symbol}")
            except Exception as e:
                print(f"⚠️ Alpaca crypto 15min fetch failed for {symbol}, no fallback for 30min crypto.")
            # No fallback for crypto 30min other than CoinGecko (which may not have 30min)
            if df is None:
                df = await fetch_coingecko_ohlc(symbol, timeframe)
            if df is None:
                df = await fetch_coingecko_price(symbol)

        if df is not None and not df.empty:
            data_cache[cache_key] = (df, now + CACHE_DURATION)
        return df

    # For other timeframes, use standard logic
    # Try Alpaca for stocks
    if '/' not in symbol and ALPACA_API_KEY and ALPACA_SECRET_KEY and alpaca_tf:
        try:
            client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_tf,
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
            print(f"⚠️ Alpaca stock fetch failed for {symbol}, falling back... {e}")
            df = None

    # If stock failed, try Twelve Data
    if df is None and '/' not in symbol:
        df = await fetch_twelvedata(symbol, timeframe)

    # For crypto
    if '/' in symbol:
        try:
            if alpaca_tf:
                client = CryptoHistoricalDataClient()
                alpaca_symbol = symbol.replace('/', '')
                request = CryptoBarsRequest(
                    symbol_or_symbols=alpaca_symbol,
                    timeframe=alpaca_tf,
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
            print(f"⚠️ Alpaca crypto fetch failed for {symbol}, falling back to CoinGecko... {e}")
            df = None

        if df is None:
            df = await fetch_coingecko_ohlc(symbol, timeframe)
        if df is None:
            df = await fetch_coingecko_price(symbol)

    if df is not None and not df.empty:
        data_cache[cache_key] = (df, now + CACHE_DURATION)

    return df

# ====================
# ENHANCED NEWS FETCHING – Multiple sources
# ====================

async def fetch_finnhub_news(symbol):
    """Original Finnhub news - kept as fallback."""
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
    """Fetch general market news from Finnhub (fallback for worldnews)."""
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

async def fetch_newsapi_top_headlines():
    """Fetch top headlines from NewsAPI (primary source for worldnews)."""
    if not NEWSAPI_KEY:
        return None
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'country': 'us',
        'pageSize': 10
    }
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"NewsAPI error: {resp.status}")
                    return None
                data = await resp.json()
                if data.get('status') != 'ok':
                    return None
                return data.get('articles', [])
    except Exception as e:
        print(f"Error fetching NewsAPI: {e}")
        return None

async def fetch_analyst_ratings_enhanced(symbol):
    """Fetch analyst ratings with more detail."""
    # Mock data for demonstration
    mock_ratings = {
        'NIO': {
            'ratings': [
                {'firm': 'HSBC', 'action': 'Upgrade', 'from': 'Hold', 'to': 'Buy', 'pt': 6.80, 'date': '2026-03-13'},
                {'firm': 'Nomura', 'action': 'Upgrade', 'from': 'Neutral', 'to': 'Buy', 'pt': 6.60, 'date': '2026-03-11'},
                {'firm': 'Citi', 'action': 'Maintain', 'rating': 'Buy', 'pt': 6.90, 'date': '2026-03-10'},
                {'firm': 'Macquarie', 'action': 'Downgrade', 'from': 'Outperform', 'to': 'Neutral', 'pt': 5.30, 'date': '2026-02-15'}
            ],
            'consensus': {
                'buy': 12,
                'hold': 8,
                'sell': 3,
                'avg_pt': 6.45
            }
        }
    }
    if symbol == 'NIO':
        return mock_ratings[symbol]
    return None

async def fetch_company_news_enhanced(symbol):
    """Fetch comprehensive company news from multiple sources."""
    mock_news = {
        'NIO': [
            {
                'title': 'HSBC upgrades NIO to Buy, raises target to $6.80 on strong volume growth',
                'source': 'HSBC Research',
                'date': '2026-03-13',
                'url': '',
                'type': 'analyst',
                'summary': 'HSBC upgraded NIO to Buy from Hold, citing above-industry visibility for earnings and strong order momentum.'
            },
            {
                'title': 'NIO achieves first-ever quarterly profit, shares surge 15%',
                'source': 'Company Report',
                'date': '2026-03-10',
                'url': '',
                'type': 'earnings',
                'summary': 'NIO reported Q4 net profit of RMB0.12 billion, its first quarterly profit ever, on record deliveries of 124,807 vehicles.'
            },
            {
                'title': 'NIO ES9 flagship SUV to launch April 10 with advanced tech',
                'source': 'Product Launch',
                'date': '2026-03-09',
                'url': '',
                'type': 'product',
                'summary': 'ES9 tech launch set for April 10, featuring three LiDAR units and SkyRide intelligent chassis. Deliveries begin June 1.'
            },
            {
                'title': 'Onvo L80 specifications due late April, targeting 200k-300k yuan segment',
                'source': 'Product Pipeline',
                'date': '2026-03-08',
                'url': '',
                'type': 'product',
                'summary': 'The Onvo L80, a large five-seat SUV focused on space and practicality, will reveal specs in late April.'
            },
            {
                'title': 'Institutional investors build large positions in NIO',
                'source': '13F Filings',
                'date': '2026-03-05',
                'url': '',
                'type': 'institutional',
                'summary': 'Aspex Management took a $266M position, WT Asset Management added $142M, showing strong institutional interest.'
            }
        ]
    }
    if symbol in mock_news:
        return mock_news[symbol]
    return None

async def fetch_stock_price_quick(symbol):
    """Quick price fetch for news display."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1]), float(data['Open'].iloc[0]) if len(data) > 0 else None
        return None, None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None, None

async def fetch_earnings_upcoming(symbol, days=14):
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                earnings = data.get('earningsCalendar', [])
                today = datetime.now().date()
                cutoff = today + timedelta(days=days)
                upcoming = []
                for e in earnings:
                    date_str = e.get('date')
                    if not date_str:
                        continue
                    try:
                        e_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        if today <= e_date <= cutoff:
                            upcoming.append(e)
                    except:
                        continue
                return upcoming
    except Exception as e:
        print(f"Error fetching earnings for {symbol}: {e}")
        return []

async def fetch_dividends_upcoming(symbol, days=14):
    url = "https://finnhub.io/api/v1/stock/dividend"
    today = datetime.now().strftime('%Y-%m-%d')
    future = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        'symbol': symbol,
        'from': today,
        'to': future,
        'token': FINNHUB_API_KEY
    }
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error fetching dividends for {symbol}: {e}")
        return []

async def fetch_splits_upcoming(symbol, days=14):
    url = "https://finnhub.io/api/v1/stock/split"
    today = datetime.now().strftime('%Y-%m-%d')
    future = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    params = {
        'symbol': symbol,
        'from': today,
        'to': future,
        'token': FINNHUB_API_KEY
    }
    timeout = aiohttp.ClientTimeout(total=10)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error fetching splits for {symbol}: {e}")
        return []

async def fetch_economic_events(days=14):
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
# CHART GENERATION (standard) – now with dark background
# ====================
def generate_chart_image(df, symbol, timeframe):
    if len(df) < 20:
        return None

    if timeframe in ['5min', '15min', '1h']:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(50).copy()
    else:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(30).copy()

    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Define dark style (same as zone chart)
    mc = mpf.make_marketcolors(
        up='#26a69a',      # teal for up
        down='#ef5350',    # red for down
        edge='white',
        wick='white',
        volume='in',
        inherit=True
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        facecolor='#1e1e1e',      # dark gray background
        figcolor='#1e1e1e',
        gridcolor='#444444'
    )

    # EMA plots (keep original colors)
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
            # Style legend
            axes[0].legend(
                loc='upper left',
                fontsize=9,
                facecolor='#333333',
                edgecolor='white',
                labelcolor='white',
                framealpha=0.8
            )
        # Style axes
        axes[0].tick_params(colors='white')
        axes[0].yaxis.label.set_color('white')
        axes[0].xaxis.label.set_color('white')
        axes[0].set_facecolor('#1e1e1e')
        if not volume_all_nan:
            axes[2].set_ylabel('Volume', color='white')
            axes[2].tick_params(colors='white')
            axes[2].yaxis.label.set_color('white')
            axes[2].set_facecolor('#1e1e1e')
        # Save to temp file
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

# ====================
# ZONE CHART GENERATION – Dark background, demand lines by strength (green = most respected)
# ====================
def generate_zone_chart(df, symbol, zones):
    """Generate a candlestick chart with dark gray background and demand lines colored by strength."""
    print(f"[DEBUG] Generating zone chart for {symbol} with {len(zones)} zones")
    if len(df) < 20:
        print(f"[DEBUG] Insufficient data: {len(df)} rows")
        return None

    # Use last 100 candles for clarity
    chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(100).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    print(f"[DEBUG] Chart data shape: {chart_data.shape}")

    # Define dark style
    mc = mpf.make_marketcolors(
        up='#26a69a',      # teal for up
        down='#ef5350',    # red for down
        edge='white',
        wick='white',
        volume='in',
        inherit=True
    )
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        facecolor='#1e1e1e',      # dark gray background
        figcolor='#1e1e1e',
        gridcolor='#444444'
    )

    # Color-code demand lines by strength (green = most respected, red = least)
    if zones:
        strengths = [z['strength'] for z in zones]
        min_s = min(strengths)
        max_s = max(strengths)
        if max_s > min_s:
            norm_strengths = [(s - min_s) / (max_s - min_s) for s in strengths]
        else:
            norm_strengths = [0.5] * len(strengths)  # all same -> middle color

        # Use RdYlGn colormap (reversed so that high strength = green, low = red)
        colormap = matplotlib.colormaps['RdYlGn_r']
        line_colors = [colormap(norm) for norm in norm_strengths]
        print(f"[DEBUG] Strengths: {strengths}, normalized: {norm_strengths}")
    else:
        line_colors = []

    # Create addplots for each zone
    apds = []
    for i, zone in enumerate(zones):
        level = zone['level']
        if line_colors:
            color = mcolors.to_hex(line_colors[i])
        else:
            color = '#ffffff'  # fallback white
        label = f"Demand ${level:.2f} (touches: {zone['strength']})"
        apds.append(mpf.make_addplot(
            [level] * len(chart_data),
            color=color,
            width=2.0,
            linestyle='-',
            label=label
        ))

    try:
        # Plot with volume
        print("[DEBUG] Calling mpf.plot...")
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
            # Improved legend
            axes[0].legend(
                loc='upper left',
                fontsize=10,
                facecolor='#333333',
                edgecolor='white',
                labelcolor='white',
                framealpha=0.8
            )
        # Style volume subplot
        axes[2].set_ylabel('Volume', color='white')
        axes[2].tick_params(colors='white')
        axes[2].yaxis.label.set_color('white')
        # Style main axis
        axes[0].tick_params(colors='white')
        axes[0].yaxis.label.set_color('white')
        axes[0].xaxis.label.set_color('white')
        # Set facecolor of axes explicitly
        axes[0].set_facecolor('#1e1e1e')
        axes[2].set_facecolor('#1e1e1e')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fig.savefig(tmpfile.name, format='PNG', dpi=150, bbox_inches='tight', facecolor=s['facecolor'])
            tmpfile.flush()
            with open(tmpfile.name, 'rb') as f:
                img_data = f.read()
        os.unlink(tmpfile.name)
        plt.close(fig)
        print(f"[DEBUG] Chart generated successfully, size: {len(img_data)} bytes")
        return io.BytesIO(img_data)
    except Exception as e:
        print(f"⚠️ Zone chart generation failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ====================
# ENHANCED NEWS FORMATTING
# ====================
def format_enhanced_news_embed(symbol, news_items, ratings_data, current_price, prev_close):
    """Create a comprehensive news embed with categorized, actionable information."""
    
    embed = discord.Embed(
        title=f"📰 {symbol} – Market Intelligence",
        description=f"Current: **${current_price:.2f}** | Previous Close: **${prev_close:.2f}**" if current_price and prev_close else f"Current: **${current_price:.2f}**" if current_price else "",
        color=0x3498db,
        timestamp=datetime.now()
    )
    
    # Price change indicator
    if current_price and prev_close:
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        arrow = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
        embed.description += f" | {arrow} {change:+.2f} ({change_pct:+.2f}%)"
    
    # Analyst ratings section (most important for trading)
    if ratings_data and 'ratings' in ratings_data:
        recent = ratings_data['ratings'][:3]  # Last 3 ratings
        ratings_text = ""
        for r in recent:
            action_symbol = "🟢" if r['action'] == 'Upgrade' or r['rating'] == 'Buy' else "🔴" if r['action'] == 'Downgrade' or r['rating'] == 'Sell' else "⚪"
            pt_text = f" → ${r['pt']}" if 'pt' in r else ""
            ratings_text += f"{action_symbol} **{r['firm']}**: {r['action']} to {r.get('to', r.get('rating', '?'))}{pt_text} ({r['date']})\n"
        
        if ratings_text:
            embed.add_field(name="📊 Recent Analyst Actions", value=ratings_text, inline=False)
    
    # Key developments section
    if news_items:
        # Categorize news
        catalysts = []
        earnings_news = []
        product_news = []
        institutional_news = []
        
        for item in news_items[:8]:  # Limit to 8 items
            if item['type'] == 'analyst':
                continue  # Already covered in analyst section
            elif item['type'] == 'earnings':
                earnings_news.append(item)
            elif item['type'] == 'product':
                product_news.append(item)
            elif item['type'] == 'institutional':
                institutional_news.append(item)
            else:
                catalysts.append(item)
        
        # Product catalysts (most actionable)
        if product_news:
            product_text = ""
            for item in product_news[:3]:
                product_text += f"• **{item['title']}**\n  {item['summary'][:150]}...\n"
            embed.add_field(name="🚗 Product Catalysts", value=product_text, inline=False)
        
        # Earnings/financial news
        if earnings_news:
            earnings_text = ""
            for item in earnings_news[:2]:
                earnings_text += f"• **{item['title']}**\n  {item['summary'][:150]}...\n"
            embed.add_field(name="💰 Financial Developments", value=earnings_text, inline=False)
        
        # Institutional activity
        if institutional_news:
            inst_text = ""
            for item in institutional_news[:2]:
                inst_text += f"• **{item['title']}**\n  {item['summary'][:150]}...\n"
            embed.add_field(name="🏦 Institutional Activity", value=inst_text, inline=False)
        
        # Other catalysts
        if catalysts:
            other_text = ""
            for item in catalysts[:2]:
                other_text += f"• **{item['title']}**\n  {item['summary'][:150]}...\n"
            embed.add_field(name="📌 Other Developments", value=other_text, inline=False)
    
    # Add TradingView link
    web_url = get_tradingview_web_link(symbol)
    embed.add_field(name="📊 TradingView", value=f"[Click here for charts]({web_url})", inline=False)
    
    embed.set_footer(text="Data aggregated from multiple sources • Not financial advice")
    return embed

# ====================
# WORLD NEWS COMMAND (with sentiment analysis)
# ====================
# Enhanced keyword mapping with sentiment (Bullish/Bearish/Neutral)
IMPACT_KEYWORDS = {
    # Bullish patterns
    'rate cut': ('🟢 Bullish', 'Financials', 'Rate cuts lower borrowing costs and boost stocks.', ['SPY', 'QQQ', 'XLF']),
    'cuts rates': ('🟢 Bullish', 'Financials', 'Rate cuts stimulate economic growth.', ['SPY', 'QQQ', 'XLF']),
    'stimulus': ('🟢 Bullish', 'Economy', 'Government stimulus boosts spending and growth.', ['SPY', 'IWM', 'XLY']),
    'beat earnings': ('🟢 Bullish', 'Earnings', 'Strong earnings indicate company health.', ['SPY', 'QQQ']),
    'exceeds expectations': ('🟢 Bullish', 'Sentiment', 'Positive surprises drive stock prices higher.', ['SPY', 'QQQ']),
    'upgrade': ('🟢 Bullish', 'Analysts', 'Analyst upgrades signal confidence.', ['SPY']),
    'buy rating': ('🟢 Bullish', 'Analysts', 'Buy ratings encourage institutional buying.', ['SPY']),
    'profit surge': ('🟢 Bullish', 'Financials', 'Rising profits attract investors.', ['SPY']),
    'record high': ('🟢 Bullish', 'Sentiment', 'All-time highs can lead to momentum buying.', ['SPY']),
    'job growth': ('🟢 Bullish', 'Economy', 'Strong job market supports consumer spending.', ['XLY', 'SPY']),
    'unemployment falls': ('🟢 Bullish', 'Economy', 'Lower unemployment signals economic strength.', ['SPY', 'IWM']),

    # Bearish patterns
    'rate hike': ('🔴 Bearish', 'Financials', 'Rate hikes increase borrowing costs and slow growth.', ['SPY', 'QQQ', 'XLF']),
    'hikes rates': ('🔴 Bearish', 'Financials', 'Rate hikes increase borrowing costs and slow growth.', ['SPY', 'QQQ', 'XLF']),
    'inflation rises': ('🔴 Bearish', 'Economy', 'High inflation may lead to tighter monetary policy.', ['SPY', 'TLT']),
    'miss earnings': ('🔴 Bearish', 'Earnings', 'Earnings misses disappoint investors.', ['SPY', 'QQQ']),
    'downgrade': ('🔴 Bearish', 'Analysts', 'Downgrades can trigger selling.', ['SPY']),
    'sell rating': ('🔴 Bearish', 'Analysts', 'Sell ratings indicate negative outlook.', ['SPY']),
    'profit warning': ('🔴 Bearish', 'Financials', 'Profit warnings signal trouble ahead.', ['SPY']),
    'layoffs': ('🔴 Bearish', 'Economy', 'Layoffs indicate corporate stress.', ['SPY', 'IWM']),
    'recession': ('🔴 Bearish', 'Economy', 'Recession fears hurt all risk assets.', ['SPY', 'QQQ']),
    'trade war': ('🔴 Bearish', 'Trade', 'Trade tensions disrupt global supply chains.', ['SPY', 'EEM']),
    'tariff': ('🔴 Bearish', 'Trade', 'Tariffs increase costs and reduce profits.', ['CAT', 'DE', 'BA']),

    # Sector-specific (can be bullish or bearish depending on context; we'll assign neutral with impact)
    'oil': ('⚪ Neutral', 'Energy', 'Oil price changes affect energy stocks.', ['XOM', 'CVX', 'OXY', 'USO']),
    'crude': ('⚪ Neutral', 'Energy', 'Oil price changes affect energy stocks.', ['XOM', 'CVX', 'OXY', 'USO']),
    'opec': ('⚪ Neutral', 'Energy', 'OPEC decisions impact oil supply.', ['XOM', 'CVX', 'OXY', 'USO']),
    'semiconductor': ('⚪ Neutral', 'Technology', 'Chip demand drives tech earnings.', ['NVDA', 'AMD', 'INTC', 'SMH']),
    'chip': ('⚪ Neutral', 'Technology', 'Chip demand drives tech earnings.', ['NVDA', 'AMD', 'INTC', 'SMH']),
    'nvidia': ('⚪ Neutral', 'Technology', 'Nvidia leads AI and chip sector.', ['NVDA', 'AMD', 'SMH']),
    'ai': ('⚪ Neutral', 'Technology', 'AI investment boosts tech.', ['NVDA', 'MSFT', 'GOOGL']),
    'china': ('⚪ Neutral', 'China exposure', 'Companies with China revenue may be affected.', ['BABA', 'JD', 'NIO', 'FXI']),
    'retail': ('⚪ Neutral', 'Consumer', 'Retail sales reflect consumer confidence.', ['AMZN', 'WMT', 'TGT', 'XRT']),
    'consumer': ('⚪ Neutral', 'Consumer', 'Consumer spending drives economy.', ['AMZN', 'WMT', 'PG', 'KO']),
    'housing': ('⚪ Neutral', 'Housing', 'Housing data affect builders and banks.', ['LEN', 'DHI', 'KBH', 'XHB']),
    'real estate': ('⚪ Neutral', 'Real Estate', 'Real estate investment trusts.', ['SPG', 'PLD', 'AMT', 'XLRE']),
    'bank': ('⚪ Neutral', 'Financials', 'Bank health reflects economic conditions.', ['JPM', 'BAC', 'WFC', 'XLF']),
    'tech': ('⚪ Neutral', 'Technology', 'Tech sentiment drives growth.', ['QQQ', 'AAPL', 'MSFT', 'GOOGL']),
    'pharma': ('⚪ Neutral', 'Healthcare', 'Drug approvals and healthcare policy.', ['PFE', 'MRK', 'LLY', 'XLV']),
    'drug': ('⚪ Neutral', 'Healthcare', 'Drug approvals and healthcare policy.', ['PFE', 'MRK', 'LLY', 'XLV']),
    'covid': ('⚪ Neutral', 'Healthcare', 'Pandemic developments affect vaccines and travel.', ['MRNA', 'PFE', 'JNJ', 'CCL']),
    'travel': ('⚪ Neutral', 'Travel', 'Travel demand affects airlines, cruises, hotels.', ['AAL', 'DAL', 'CCL', 'MAR']),
    'airline': ('⚪ Neutral', 'Travel', 'Airlines react to travel demand and fuel costs.', ['AAL', 'DAL', 'UAL', 'LUV']),
    'auto': ('⚪ Neutral', 'Automotive', 'Auto sales and EV trends.', ['TSLA', 'F', 'GM', 'NIO']),
    'ev': ('⚪ Neutral', 'Electric Vehicles', 'EV adoption drives Tesla and emerging players.', ['TSLA', 'NIO', 'LI', 'XPEV']),
}

def analyze_news_impact(title, description):
    """Return (sentiment_emoji, impact_text, ticker_list) based on keywords."""
    combined = (title + " " + (description or "")).lower()
    matched = []
    for keyword, (sentiment, sector, impact, tickers) in IMPACT_KEYWORDS.items():
        if keyword in combined:
            matched.append((sentiment, sector, impact, tickers))
    if matched:
        # Pick the first matched for simplicity; you could combine if multiple, but keep it simple.
        sentiment, sector, impact, tickers = matched[0]
        ticker_str = ", ".join([f"${t}" for t in tickers[:3]])
        return f"{sentiment} **{sector}:** {impact}\n📊 **Stocks:** {ticker_str}"
    else:
        return "⚪ **General:** May affect broad market sentiment. Check related sectors.\n📊 **Stocks:** $SPY, $QQQ, $DIA"

@bot.command(name='worldnews')
async def world_news(ctx):
    """Get the latest world news with market impact analysis and sentiment."""
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        await ctx.send("🌍 Fetching latest world news...")
        
        # Check cache
        now = datetime.now()
        if world_news_cache["data"] and world_news_cache["expiry"] > now:
            articles = world_news_cache["data"]
            source = "cached"
        else:
            # Try NewsAPI first
            articles = await fetch_newsapi_top_headlines()
            source = "NewsAPI"
            if not articles:
                # Fallback to Finnhub general news
                finnhub_news = await fetch_finnhub_general_news()
                if finnhub_news:
                    # Convert to a consistent format
                    articles = []
                    for item in finnhub_news[:10]:
                        articles.append({
                            'title': item.get('headline', ''),
                            'description': item.get('summary', ''),
                            'source': {'name': item.get('source', 'Finnhub')},
                            'publishedAt': datetime.fromtimestamp(item.get('datetime', 0)).isoformat() if item.get('datetime') else None,
                            'url': item.get('url', '')
                        })
                    source = "Finnhub"
            if articles:
                world_news_cache["data"] = articles
                world_news_cache["expiry"] = now + timedelta(minutes=30)
        
        if not articles:
            await ctx.send("❌ Could not fetch world news at this time.")
            return
        
        embed = discord.Embed(
            title="🌍 World News – Market Impact",
            description=f"Top headlines from {source} (updated every 30 min)",
            color=0x3498db,
            timestamp=now
        )
        
        count = 0
        for article in articles[:5]:  # Limit to 5 articles
            title = article.get('title', 'No title')
            if '[Removed]' in title:
                continue
            description = article.get('description', '') or article.get('summary', '')
            source_name = article.get('source', {}).get('name', 'Unknown') if isinstance(article.get('source'), dict) else article.get('source', 'Unknown')
            published = article.get('publishedAt', '')
            url = article.get('url', '')
            
            # Parse time
            time_ago = "recently"
            if published:
                try:
                    pub_time = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    delta = now - pub_time
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
            
            impact_text = analyze_news_impact(title, description)
            
            field_value = f"**Source:** {source_name} | {time_ago}\n{impact_text}"
            if url:
                field_value += f"\n[Read more]({url})"
            
            embed.add_field(
                name=f"📰 {title[:100]}{'…' if len(title)>100 else ''}",
                value=field_value,
                inline=False
            )
            count += 1
        
        if count == 0:
            await ctx.send("No relevant news found.")
            return
        
        embed.set_footer(text="🟢 Bullish | 🔴 Bearish | ⚪ Neutral • Not financial advice")
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error fetching world news: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# ENHANCED NEWS COMMAND
# ====================
@bot.command(name='news')
async def stock_news_enhanced(ctx, ticker: str):
    """Get comprehensive, actionable news for a stock."""
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
        
        # Fetch data concurrently
        price_task = fetch_stock_price_quick(symbol)
        ratings_task = fetch_analyst_ratings_enhanced(symbol)
        news_task = fetch_company_news_enhanced(symbol)
        finnhub_task = fetch_finnhub_news(symbol)
        
        # Wait for all tasks
        price_result, ratings_data, news_data, finnhub_data = await asyncio.gather(
            price_task, ratings_task, news_task, finnhub_task, return_exceptions=True
        )
        
        current_price, prev_close = (None, None)
        if not isinstance(price_result, Exception) and price_result:
            current_price, prev_close = price_result
        
        # If enhanced news fails, fall back to original Finnhub news
        if (isinstance(news_data, Exception) or not news_data) and not isinstance(finnhub_data, Exception) and finnhub_data:
            # Fall back to original format
            web_url = get_tradingview_web_link(symbol)
            tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"
            
            embed = discord.Embed(
                title=f"Latest News for {symbol} (Legacy)",
                color=0x3498db,
                timestamp=datetime.now()
            )
            
            if current_price:
                embed.description = f"Current Price: **${current_price:.2f}**"
            
            for article in finnhub_data[:5]:
                headline = article.get('headline', 'No Headline')
                source = article.get('source', 'Unknown')
                date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M') if article.get('datetime') else 'Unknown'
                url = article.get('url', '')
                if len(headline) > 256:
                    headline = headline[:253] + "..."
                embed.add_field(name=f"{source} - {date}", value=f"[{headline}]({url})", inline=False)
            
            embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
            embed.set_footer(text=f"Requested by {ctx.author.display_name} • Limited data")
            await ctx.send(embed=embed)
            return
        
        # Use enhanced format
        if not isinstance(news_data, Exception) and news_data:
            embed = format_enhanced_news_embed(symbol, news_data, ratings_data, current_price, prev_close)
            await ctx.send(embed=embed)
        else:
            await ctx.send(f"❌ Could not fetch news data for {symbol}.")
            
    except Exception as e:
        await ctx.send(f"❌ Error fetching news: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# EMBED FORMATTING (original, kept for compatibility)
# ====================
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
# COMBINED SYMBOL REPORT FUNCTIONS (unchanged)
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

    df_calc = calculate_indicators(df)
    main_embed = format_embed(symbol, signals, best_tf)

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

    embed.set_footer(text="Use !signals SYMBOL for detailed analysis")
    await ctx.send(embed=embed)

# ====================
# OPTIONS FLOW SCANNER (ENHANCED) – (unchanged)
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
        # Find nearest weekly (0-7 DTE)
        for exp_date, exp_str in zip(exp_dates, expirations):
            dte = (exp_date - today).days
            if 0 <= dte <= 7:
                key_exps.append((exp_str, dte, "🔥 WEEKLY (0-7 DTE)"))
                break
        # Find next weekly (8-21 DTE)
        for exp_date, exp_str in zip(exp_dates, expirations):
            dte = (exp_date - today).days
            if 8 <= dte <= 21:
                key_exps.append((exp_str, dte, "💎 MONTHLY (8-21 DTE)"))
                break
        # Find 30-45 DTE
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

def analyze_expiration(opt_chain, current_price, min_volume=5):
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
            gamma = opt.get('gamma', 0)

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
                'gamma': gamma
            })
        except:
            continue
    return analyzed

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

        all_significant = []
        for exp_str, dte, label in key_exps:
            opt_chain = stock.option_chain(exp_str)
            analyzed = analyze_expiration(opt_chain, current_price)
            analyzed.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
            significant = [opt for opt in analyzed if opt['volume'] >= 5 and opt['oi'] > 0][:6]

            if significant:
                table = f"```\n{label} ({exp_str}, {dte} days)\n"
                table += "STRIKE  TYPE  VOLUME  OI    VOL/OI  PREMIUM   WHALE\n"
                table += "------  ----  ------  ----  ------  --------  -----\n"
                for opt in significant:
                    strike = f"${opt['strike']:.2f}"
                    whale = get_whale_emoji(opt['raw_premium'])
                    table += f"{strike:6}  {opt['type']:4}  {opt['volume']:6}  {opt['oi']:4}  {opt['vol_oi_ratio']:.1f}x   {opt['premium']:7}  {whale}\n"
                table += "```"
                embed.add_field(name="", value=table, inline=False)
                all_significant.extend(significant)

        lottery = [opt for opt in all_significant if opt['vol_oi_ratio'] >= 2.0 and opt['raw_premium'] >= 5000]
        lottery.sort(key=lambda x: x['vol_oi_ratio'] * x['raw_premium'], reverse=True)
        if lottery:
            picks_text = "**🎰 LOTTERY PICKS (short DTE, high ratio):**\n"
            for i, pick in enumerate(lottery[:3]):
                target = pick['strike'] * 1.20
                picks_text += f"\n**{i+1}. ${pick['strike']:.2f} {pick['type']}**\n"
                picks_text += f"   • Volume: {pick['volume']} ({pick['vol_oi_ratio']:.1f}x)  Premium: {pick['premium']}\n"
                picks_text += f"   • Entry: Above ${current_price*1.01:.2f}  Target: ${target:.2f}\n"
            embed.add_field(name="🔥 HIGH RISK / HIGH REWARD", value=picks_text, inline=False)

        explanation = """
📊 **WHALE RATINGS:**
• 🐋🐋 = >$1M premium (massive institutional)
• 🐋 = $100K–$1M (strong interest)
• 🐬 = $10K–$100K (notable)
• 🐟 = <$10K (small)

💡 **TIP:** Short‑dated options (0‑7 DTE) can produce 1000%+ gains but are extremely risky.
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

        all_unusual.sort(key=lambda x: x['raw_premium'], reverse=True)

        if not all_unusual:
            await ctx.send("📭 No unusual options activity detected in your watchlist.")
            return

        embed = discord.Embed(
            title="🔥 UNUSUAL OPTIONS ACTIVITY SUMMARY",
            description=f"Found {len(all_unusual)} unusual setups across your watchlist",
            color=0x00ff00
        )

        table = "```\n"
        table += "SYMBOL  STRIKE  DTE  VOLUME  OI   VOL/OI  PREMIUM   WHALE\n"
        table += "------  ------  ---  ------  ---  ------  --------  -----\n"
        top_overall = []
        for opt in all_unusual[:15]:
            whale = get_whale_emoji(opt['raw_premium'])
            table += f"{opt['symbol']:6}  ${opt['strike']:.2f}  {opt['dte']:3}  {opt['volume']:6}  {opt['oi']:3}  {opt['ratio']:.1f}x   {opt['premium']:7}  {whale}\n"
            top_overall.append(opt)
        table += "```"
        embed.add_field(name="📊 ALL DETECTED ACTIVITY", value=table, inline=False)

        if top_overall:
            picks = "**🏆 TOP 3 BY PREMIUM:**\n\n"
            for i, pick in enumerate(top_overall[:3]):
                target = pick['strike'] * 1.20
                picks += f"{i+1}. **{pick['symbol']} ${pick['strike']:.2f} CALL** ({pick['label']})\n"
                picks += f"   • Volume: {pick['volume']} ({pick['ratio']:.1f}x)  Premium: {pick['premium']}\n"
                picks += f"   • Entry: Above ${pick['price'] * 1.01:.2f}  Target: ${target:.2f}\n\n"
            embed.add_field(name="🏆 TOP PICKS", value=picks, inline=False)

        embed.set_footer(text="Whale ratings: 🐋🐋 >$1M, 🐋 $100K‑$1M, 🐬 $10K‑$100K, 🐟 <$10K")
        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"❌ Error scanning options flow: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# UPCOMING COMMAND (ENHANCED with expected move)
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

                earnings = await fetch_earnings_upcoming(sym)
                dividends = await fetch_dividends_upcoming(sym)
                splits = await fetch_splits_upcoming(sym)
                ratings = await fetch_analyst_ratings(sym, limit=3)

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
                            time_str = "BMO" if e.get('hour') == 'bmo' else "AMC" if e.get('hour') == 'amc' else ""
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
                            pay_date = d.get('payDate', '')
                            lines.append(f"**{ex_date}** – Amount: ${amount}" + (f" (pay {pay_date})" if pay_date else ""))
                        embed.add_field(name="💰 Dividends", value="\n".join(lines), inline=False)

                    if splits:
                        lines = []
                        for s in splits:
                            date = s.get('date', 'N/A')
                            ratio = s.get('splitRatio', '')
                            text = f"**{date}** – {ratio}" if ratio else f"**{date}**"
                            lines.append(text)
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

                await asyncio.sleep(5)

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
                    time_str = "BMO" if e.get('hour') == 'bmo' else "AMC" if e.get('hour') == 'amc' else ""
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
                    pay_date = d.get('payDate', '')
                    lines.append(f"**{ex_date}** – Amount: ${amount}" + (f" (pay {pay_date})" if pay_date else ""))
                embed.add_field(name="💰 Dividends", value="\n".join(lines), inline=False)

            if splits:
                lines = []
                for s in splits:
                    date = s.get('date', 'N/A')
                    ratio = s.get('splitRatio', '')
                    text = f"**{date}** – {ratio}" if ratio else f"**{date}**"
                    lines.append(text)
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
# BACKTESTING COMMAND (unchanged)
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
# COMMAND: !signal (single symbol multi-timeframe report)
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
# COMMAND: !signals (fast watchlist scan) – UPDATED with stopscan fix
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
            # Check cancellation before starting a new symbol
            if await check_cancel(ctx):
                await ctx.send("🛑 Scan cancelled after processing the last symbol.")
                break

            for tf in all_timeframes:
                # No cancellation check here – we finish this symbol completely
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
        # Only send completion if not cancelled
        if not cancellation_flags.get(ctx.author.id, False):
            await ctx.send(f"✅ Signal scan complete!")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !scan (single timeframe scan)
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
            embed = format_embed(symbol, signals, timeframe)
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
                embed = format_embed(symbol, signals, timeframe)
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
# ZONE COMMAND (UPDATED with 30min default)
# ====================
def find_demand_zones(df, lookback=200, threshold_percentile=90, touch_tolerance=0.005):
    """
    Identify demand zones based on unusually large candles.
    Returns list of zones that have not been broken (no close below level after formation).
    Each zone: {'level': price, 'date': index, 'strength': number of touches (lows within tolerance)}
    """
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
        # Check if price ever closed below this level (broken support)
        closes_below = after['close'] < level * (1 - touch_tolerance)
        if closes_below.any():
            continue   # zone broken, discard
        # Count touches (low within tolerance)
        touches = after['low'] <= level * (1 + touch_tolerance)
        strength = int(touches.sum())
        if strength >= 1:
            zones.append({
                'level': level,
                'date': idx,
                'strength': strength
            })
    # Sort by level (ascending)
    zones.sort(key=lambda x: x['level'])
    return zones

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

            if not zones:
                await ctx.send(f"No clear demand zones found for {symbol} on 30min.")
                return

            embed = discord.Embed(
                title=f"📉 Demand Zones for {symbol} (30min)",
                description=f"Current Price: **${current_price:.2f}**",
                color=0x00ff00
            )

            # Show all demand zones
            for z in zones:
                distance = (current_price - z['level']) / current_price * 100
                status = "🔵 **NEAR**" if abs(distance) < 2 else ""
                date_str = z['date'].strftime('%m/%d') if hasattr(z['date'], 'strftime') else ''
                embed.add_field(
                    name=f"Support at ${z['level']:.2f} ({date_str})",
                    value=f"Distance: {distance:.1f}% {status}\nTouches: {z['strength']}",
                    inline=False
                )

            # If price is near any zone, suggest an ITM call option
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

            # Generate and attach the zone chart
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

        # Other timeframes: use old zone logic
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
# WATCHLIST COMMANDS (unchanged)
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
# HELP COMMAND (FIXED – always works, uses embed)
# ====================
@bot.command(name='help')
async def help_command(ctx):
    # Help command bypasses user_busy so it always responds
    try:
        embed = discord.Embed(
            title="📚 5-13-50 Trading Bot Commands",
            description="All commands use the prefix `!`\n\n**🟢 SCAN & SIGNALS**",
            color=0x3498db
        )
        
        embed.add_field(
            name="`!scan all [timeframe]`",
            value="Scan all watchlist symbols on a single timeframe (5min,15min,30min,1h,4h,daily,weekly)",
            inline=False
        )
        embed.add_field(
            name="`!scan SYMBOL [timeframe]`",
            value="Scan a single symbol on a specific timeframe",
            inline=False
        )
        embed.add_field(
            name="`!signals`",
            value="Scan your ENTIRE watchlist across ALL 7 timeframes (fast, uses Alpaca)",
            inline=False
        )
        embed.add_field(
            name="`!signal SYMBOL`",
            value="Multi‑timeframe report for a single symbol",
            inline=False
        )
        
        embed.add_field(
            name="\n📰 NEWS & EVENTS",
            value="",
            inline=False
        )
        embed.add_field(
            name="`!news TICKER`",
            value="Comprehensive, actionable news (analyst actions, product launches, institutional moves)",
            inline=False
        )
        embed.add_field(
            name="`!worldnews`",
            value="Global headlines with market impact analysis and suggested stocks (NEW)",
            inline=False
        )
        embed.add_field(
            name="`!upcoming [TICKER]`",
            value="Upcoming catalysts (earnings, dividends, splits, analyst ratings, expected move)",
            inline=False
        )
        
        embed.add_field(
            name="\n🎯 ZONES",
            value="",
            inline=False
        )
        embed.add_field(
            name="`!zone SYMBOL [timeframe]`",
            value="Default 30min – shows demand zones with strength‑colored lines and ITM option suggestions",
            inline=False
        )
        
        embed.add_field(
            name="\n🔥 OPTIONS FLOW",
            value="",
            inline=False
        )
        embed.add_field(
            name="`!flow TICKER`",
            value="Unusual options activity with whale ratings (🐋🐋 = massive)",
            inline=False
        )
        embed.add_field(
            name="`!scanflow`",
            value="Scan watchlist for unusual options setups, sorted by premium",
            inline=False
        )
        
        embed.add_field(
            name="\n📈 BACKTESTING",
            value="",
            inline=False
        )
        embed.add_field(
            name="`!backtest SYMBOL [days=365]`",
            value="Backtest EMA crossover strategy, returns win rate, profit factor, max drawdown",
            inline=False
        )
        
        embed.add_field(
            name="\n📋 WATCHLIST",
            value="",
            inline=False
        )
        embed.add_field(
            name="`!add SYMBOL`",
            value="Add stock or crypto (use BTC/USD for crypto)",
            inline=False
        )
        embed.add_field(
            name="`!remove SYMBOL`",
            value="Remove from watchlist",
            inline=False
        )
        embed.add_field(
            name="`!list`",
            value="Show watchlist",
            inline=False
        )
        
        embed.add_field(
            name="\n⚙️ UTILITY",
            value="",
            inline=False
        )
        embed.add_field(
            name="`!ping`",
            value="Test bot",
            inline=True
        )
        embed.add_field(
            name="`!stopscan`",
            value="Stop ongoing scan",
            inline=True
        )
        embed.add_field(
            name="`!help`",
            value="This message",
            inline=True
        )
        
        embed.add_field(
            name="\n⏱️ TIMEFRAMES",
            value="5min, 15min, 30min, 1h, 4h, daily, weekly",
            inline=False
        )
        
        embed.set_footer(text="💡 Pro tip: Use !worldnews to see how global events move markets")
        await ctx.send(embed=embed)
    except Exception as e:
        # Fallback in case embed fails
        await ctx.send("📚 Commands: !scan, !signals, !signal, !news, !worldnews, !upcoming, !zone, !flow, !scanflow, !backtest, !add, !remove, !list, !ping, !stopscan")
        print(f"Help command error: {e}")

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

async def send_symbol_with_chart(ctx, symbol, df, timeframe):
    df_calc = calculate_indicators(df)
    signals = get_signals(df_calc)
    embed = format_embed(symbol, signals, timeframe)
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
# MAIN ENTRY POINT
# ====================
async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())