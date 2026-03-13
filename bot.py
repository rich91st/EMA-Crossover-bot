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
import mplfinance as mpf
import io

warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")

# === CONFIGURATION ===
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI')
PORT = int(os.getenv('PORT', 10000))

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
# DATA FETCHING – Alpaca first, then Twelve Data / CoinGecko
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

    try:
        async with aiohttp.ClientSession() as session:
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
    try:
        async with aiohttp.ClientSession() as session:
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
                bars = client.get_stock_bars(request)
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
                bars = client.get_crypto_bars(request)
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
            bars = client.get_stock_bars(request)
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
                bars = client.get_crypto_bars(request)
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
# FINNHUB NEWS & EVENTS
# ====================

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
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if isinstance(data, list) else None
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return None

async def fetch_earnings_upcoming(symbol, days=14):
    url = "https://finnhub.io/api/v1/calendar/earnings"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
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
    try:
        async with aiohttp.ClientSession() as session:
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
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error fetching splits for {symbol}: {e}")
        return []

async def fetch_analyst_ratings(symbol, limit=3):
    url = "https://finnhub.io/api/v1/stock/recommendation"
    params = {'symbol': symbol, 'token': FINNHUB_API_KEY}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data[:limit] if data else []
    except Exception as e:
        print(f"Error fetching analyst ratings for {symbol}: {e}")
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
    try:
        async with aiohttp.ClientSession() as session:
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
# CHART GENERATION (standard)
# ====================

def generate_chart_image(df, symbol, timeframe):
    if len(df) < 20:
        return None

    if timeframe in ['5min', '15min', '1h']:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(50).copy()
    else:
        chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(30).copy()

    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    volume_all_nan = chart_data['Volume'].isna().all()

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

    mc = mpf.make_marketcolors(up='#00ff88', down='#ff4d4d', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)

    try:
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
            axes[0].legend(loc='upper left')
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG', dpi=120, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        print(f"⚠️ Chart generation failed for {symbol}: {e}")
        return None

# ====================
# ZONE CHART GENERATION – Dark background, candlesticks, color‑coded demand lines
# ====================
def generate_zone_chart(df, symbol, zones):
    """Generate a candlestick chart with black background and demand lines colored by strength."""
    if len(df) < 20:
        return None

    # Use last 100 candles for clarity
    chart_data = df[['open', 'high', 'low', 'close', 'volume']].tail(100).copy()
    chart_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Define a dark style with black background
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
        facecolor='#000000',      # pure black background
        figcolor='#000000',
        gridcolor='#333333'
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
        colormap = cm.get_cmap('RdYlGn_r')
        line_colors = [colormap(norm) for norm in norm_strengths]
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
            axes[0].legend(loc='upper left', fontsize='small', facecolor='#222222', edgecolor='white', labelcolor='white')
        # Style volume subplot
        axes[2].set_ylabel('Volume', color='white')
        axes[2].tick_params(colors='white')
        axes[2].yaxis.label.set_color('white')
        # Style main axis
        axes[0].tick_params(colors='white')
        axes[0].yaxis.label.set_color('white')
        axes[0].xaxis.label.set_color('white')
        # Set facecolor of axes explicitly
        axes[0].set_facecolor('#000000')
        axes[2].set_facecolor('#000000')
        # Save
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG', dpi=150, bbox_inches='tight', facecolor=s.facecolor)
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        print(f"⚠️ Zone chart generation failed for {symbol}: {e}")
        return None

# ====================
# EMBED FORMATTING
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
# OPTIONS FLOW SCANNER (ENHANCED)
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
    """Return a list of (expiration_str, dte, label) for nearest weekly, next weekly, and 30-45 DTE."""
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
    """Return list of significant options with analysis."""
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
    """Enhanced options flow – scans multiple expirations, adds whale rating."""
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
            significant = [opt for opt in analyzed if opt['volume'] >= 5 and opt['oi'] > 0][:6]  # top 6 per expiration

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

        # Top lottery picks (short DTE, high vol/OI, reasonable premium)
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
    """Scan watchlist for unusual options activity across key expirations, with whale ratings."""
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
                    # Only scan weekly and primary (ignore the middle one for scanflow to save time)
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
                                        if distance_pct <= 20:  # near the money
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
                await asyncio.sleep(2)  # be gentle to yfinance
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue

        # Sort by premium (largest first)
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
        for opt in all_unusual[:15]:  # show top 15
            whale = get_whale_emoji(opt['raw_premium'])
            table += f"{opt['symbol']:6}  ${opt['strike']:.2f}  {opt['dte']:3}  {opt['volume']:6}  {opt['oi']:3}  {opt['ratio']:.1f}x   {opt['premium']:7}  {whale}\n"
            top_overall.append(opt)
        table += "```"
        embed.add_field(name="📊 ALL DETECTED ACTIVITY", value=table, inline=False)

        # Top 3 setups by premium
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
    """
    Calculate expected move from options (nearest expiration after earnings)
    Returns (expected_move_str, historical_avg_str).
    """
    try:
        stock = yf.Ticker(symbol)
        earn_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
        today = datetime.now()

        expirations = stock.options
        if not expirations:
            return "N/A", "N/A"

        # Find first expiration after earnings
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

        # ATM straddle
        calls['diff'] = abs(calls['strike'] - current_price)
        puts['diff'] = abs(puts['strike'] - current_price)
        atm_call = calls.loc[calls['diff'].idxmin()]
        atm_put = puts.loc[puts['diff'].idxmin()]

        call_price = (atm_call['bid'] + atm_call['ask']) / 2 if atm_call['bid'] > 0 and atm_call['ask'] > 0 else atm_call['lastPrice']
        put_price = (atm_put['bid'] + atm_put['ask']) / 2 if atm_put['bid'] > 0 and atm_put['ask'] > 0 else atm_put['lastPrice']
        straddle = call_price + put_price

        expected_pct = (straddle / current_price) * 100

        # Historical average move – optional, we'll return N/A for now
        return f"{expected_pct:.1f}%", "N/A"

    except Exception as e:
        print(f"Error getting earnings stats for {symbol}: {e}")
        return "N/A", "N/A"

@bot.command(name='upcoming')
async def upcoming_events(ctx, ticker: str = None):
    """Show upcoming catalysts with expected move from options."""
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
            # Scan all stocks in watchlist
            watchlist = await load_watchlist()
            stocks = watchlist['stocks']
            if not stocks:
                await ctx.send("No stocks in your watchlist to scan for events.")
                return

            await ctx.send(f"🔍 Scanning all stocks ({len(stocks)}) for upcoming events...")

            # Economic events first
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
                    if importance:
                        star = "★" if importance == "high" else "☆" if importance == "medium" else "·"
                    else:
                        star = ""
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
            # Single ticker
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
    """Backtest EMA crossover strategy."""
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
    """Get multi-timeframe signal report for a single symbol."""
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
# COMMAND: !signals (fast watchlist scan)
# ====================
@bot.command(name='signals')
async def signals(ctx):
    """Scan your entire watchlist across all 6 timeframes and report symbols with signals."""
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
                break

            for tf in all_timeframes:
                if await check_cancel(ctx):
                    break
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
        await ctx.send(f"✅ Signal scan complete!")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# COMMAND: !scan (single timeframe scan)
# ====================
@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    """Scan symbols on a single timeframe."""
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
            await asyncio.sleep(1)  # delay between messages

        cancellation_flags[ctx.author.id] = False
        await ctx.send("Scan complete.")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# NEWS COMMAND (unchanged)
# ====================
@bot.command(name='news')
async def stock_news(ctx, ticker: str, limit: int = 5):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        now = datetime.now()
        last = last_command_time.get(ctx.author.id)
        if last and (now - last) < timedelta(seconds=5):
            return
        last_command_time[ctx.author.id] = now
        await ctx.send(f"📰 Fetching the latest **{limit}** news headlines for **{ticker.upper()}**...")
        news_data = await fetch_stock_news(ticker.upper())
        if not news_data or len(news_data) == 0:
            await ctx.send(f"Could not fetch news for {ticker.upper()}.")
            return
        web_url = get_tradingview_web_link(ticker.upper())
        tv_field = f"📊 **View on TradingView:** [Click here]({web_url})"
        embed = discord.Embed(title=f"Latest News for {ticker.upper()}", color=0x3498db)
        for article in news_data[:limit]:
            headline = article.get('headline', 'No Headline')
            source = article.get('source', 'Unknown')
            date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M') if article.get('datetime') else 'Unknown'
            url = article.get('url', '')
            if len(headline) > 256:
                headline = headline[:253] + "..."
            embed.add_field(name=f"{source} - {date}", value=f"[{headline}]({url})", inline=False)
        embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
        embed.set_footer(text=f"Requested by {ctx.author.display_name}")
        await ctx.send(embed=embed)
    finally:
        user_busy[ctx.author.id] = False

# ====================
# ZONE COMMAND (UPDATED with 30min default and improved chart)
# ====================
def find_demand_zones(df, lookback=200, threshold_percentile=90, touch_tolerance=0.005):
    """
    Identify demand zones based on unusually large candles.
    Returns list of zones: each with 'level', 'date', 'strength' (number of touches).
    """
    if len(df) < 50:
        return []
    df = df.iloc[-lookback:].copy()
    df['range'] = df['high'] - df['low']
    # Determine large candle threshold (e.g., 90th percentile of range)
    threshold = np.percentile(df['range'].dropna(), threshold_percentile)
    large_candles = df[df['range'] > threshold]
    zones = []
    for idx, row in large_candles.iterrows():
        level = row['low']
        # Data after this candle (including the candle itself)
        after = df.loc[idx:]
        if len(after) < 2:
            continue
        # Check if price ever closed below this level (broken support)
        closes_below = after['close'] < level * (1 - touch_tolerance)
        if closes_below.any():
            continue
        # Count touches (low within tolerance)
        touches = after['low'] <= level * (1 + touch_tolerance)
        if touches.sum() >= 1:
            zones.append({
                'level': level,
                'date': idx,
                'strength': int(touches.sum())
            })
    # Sort by level (ascending)
    zones.sort(key=lambda x: x['level'])
    return zones

@bot.command(name='zone')
async def zone(ctx, ticker: str, timeframe: str = '30min'):
    """Show support/resistance zones and, for 30min, identify demand zones with option suggestions and chart."""
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

        # Special handling for 30min: detect demand zones
        if timeframe == '30min':
            await ctx.send(f"🔍 Scanning 30‑minute chart for **{symbol}** to find demand zones...")
            df = await fetch_ohlcv(symbol, '30min')
            if df is None or df.empty:
                await ctx.send(f"Could not fetch 30min data for {symbol}.")
                return

            current_price = df['close'].iloc[-1]
            zones = find_demand_zones(df)

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
            if near_zones and '/' not in symbol:  # stocks only (crypto no options)
                best_zone = min(near_zones, key=lambda z: abs(current_price - z['level']))
                try:
                    import yfinance as yf
                    stock = yf.Ticker(symbol)
                    expirations = stock.options
                    if expirations:
                        # Find monthly expiration (30-45 DTE)
                        today = datetime.now().date()
                        primary_exp = None
                        for exp in expirations:
                            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                            dte = (exp_date - today).days
                            if 30 <= dte <= 45:
                                primary_exp = exp
                                break
                        if not primary_exp and expirations:
                            primary_exp = expirations[0]  # fallback to nearest

                        if primary_exp:
                            opt_chain = stock.option_chain(primary_exp)

                            # Determine strike offset based on price
                            price = current_price
                            if price > 100:
                                offset = 5.0
                            elif price > 50:
                                offset = 2.0
                            elif price > 10:
                                offset = 1.0
                            else:
                                offset = max(0.5, price * 0.15)  # 15% for cheap stocks

                            target_strike = price - offset  # ITM call
                            calls = opt_chain.calls
                            if not calls.empty:
                                calls['strike_diff'] = abs(calls['strike'] - target_strike)
                                best_call = calls.loc[calls['strike_diff'].idxmin()]

                                strike = best_call['strike']
                                last = best_call.get('lastPrice', 'N/A')
                                bid = best_call.get('bid', 'N/A')
                                ask = best_call.get('ask', 'N/A')
                                volume = best_call.get('volume', 'N/A')
                                option_symbol = best_call.get('contractSymbol', 'N/A')

                                # Estimate premium (use mid if available)
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

            # Generate and attach the improved zone chart (dark background, color-coded lines)
            chart_buffer = generate_zone_chart(df, symbol, zones)
            if chart_buffer:
                file = discord.File(chart_buffer, filename='zone_chart.png')
                embed.set_image(url='attachment://zone_chart.png')
                embed.set_footer(text="⚠️ Options are risky. This is not financial advice.")
                await ctx.send(embed=embed, file=file)
            else:
                embed.set_footer(text="⚠️ Options are risky. This is not financial advice.")
                await ctx.send(embed=embed)
            return

        # --- Existing zone logic for other timeframes (unchanged) ---
        await ctx.send(f"🔍 Fetching zones for **{symbol}** ({timeframe})...")
        df = await fetch_ohlcv(symbol, timeframe)
        if df is None or df.empty:
            await ctx.send(f"Could not fetch data for {symbol}.")
            return
        df = calculate_indicators(df)
        signals = get_signals(df)
        embed = format_zone_embed(symbol, signals, timeframe)
        await ctx.send(embed=embed)

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
# HELP COMMAND (updated with 30min info)
# ====================
@bot.command(name='help')
async def help_command(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        help_text = """
**5-13-50 Trading Bot Commands**

📊 **SCAN COMMANDS**
`!scan all [5min|15min|30min|1h|4h|daily|weekly]` – Scan all watchlist symbols on a single timeframe
`!scan SYMBOL [timeframe]` – Scan a single symbol on a specific timeframe

🚀 **SIGNAL COMMANDS**
`!signals` – Scan your ENTIRE watchlist across ALL 7 timeframes (5min, 15min, 30min, 1h, 4h, daily, weekly)
   • Shows ONE combined report per symbol with best timeframe chart + clean summary
   • Uses Alpaca for stocks (fast) – no 60‑second waits!

`!signal SYMBOL` – Get a multi‑timeframe report for a single symbol (all 7 timeframes)

📰 **NEWS & EVENTS**
`!news TICKER [limit]` – Fetch latest news headlines
`!upcoming [TICKER]` – Show upcoming catalysts (earnings, dividends, splits, analyst ratings, expected move)

🎯 **ZONES**
`!zone SYMBOL [timeframe]` – Show buy/sell zones based on support/resistance and EMAs.
   • **Default is now 30min** (demand zones + option suggestions if near a zone, with annotated chart).
   • Example: `!zone AAPL` (30min with chart), `!zone AAPL daily` (daily zones).

🔥 **OPTIONS FLOW**
`!flow TICKER` – Check unusual options activity for a specific stock (scans weekly, monthly, primary expirations; includes whale ratings)
`!scanflow` – Scan entire watchlist for unusual options setups (sorts by premium, adds whale ratings)

📈 **BACKTESTING**
`!backtest SYMBOL [days=365]` – Backtest EMA crossover strategy on historical data
   • Enters at next day's open, compounds returns, includes transaction costs
   • Returns win rate, profit factor, max drawdown, and equity curve chart

📋 **WATCHLIST**
`!add SYMBOL` – Add to watchlist (use `BTC/USD` for crypto)
`!remove SYMBOL` – Remove from watchlist
`!list` – Show watchlist

⚙️ **UTILITY**
`!ping` – Test bot
`!stopscan` – Stop ongoing scan
`!help` – This message

⏱️ **TIMEFRAMES (for !scan, !zone, !signal):**
• `5min` – 5‑minute candles
• `15min` – 15‑minute candles
• `30min` – 30‑minute candles (default for !zone)
• `1h` – 1‑hour candles
• `4h` – 4‑hour candles
• `daily` – daily candles
• `weekly` – weekly candles

💡 **PRO TIPS:**
• Use `!signals` to scan your whole watchlist for opportunities – now super fast with Alpaca!
• Use `!signal AAPL` to drill down on a specific symbol.
• Use `!zone AAPL` (default 30min) to see demand zones and get ITM call suggestions with a chart.
• Use `!scanflow` to find explosive options setups before they run (watch for 🐋🐋 whales).
• Use `!upcoming` to see expected moves on earnings dates.
• Use `!backtest` to validate your strategy before risking real money.
        """
        await ctx.send(help_text)
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
    await ctx.send("⏹️ Cancelling scan... (will stop after current symbol)")

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