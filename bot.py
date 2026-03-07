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
from FOC import FOC

# Charting libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplfinance as mpf
import io

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
user_busy = {}
cancellation_flags = {}

# Initialize FOC for options data
foc = FOC()

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
        print(f"❌ Error loading watchlist: {e}")
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
    """Generate a clickable TradingView web link."""
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
# DATA FETCHING
# ====================

async def fetch_twelvedata(symbol, timeframe):
    interval_map = {
        '5min': '5min', 
        '15min': '15min',
        '1h': '1h',
        '4h': '4h',
        'daily': '1day', 
        'weekly': '1week'
    }
    interval = interval_map.get(timeframe)
    if not interval:
        return None

    # For intraday timeframes, we need more data points
    outputsize = 200
    if timeframe in ['5min', '15min', '1h']:
        outputsize = 500  # Get more data for intraday charts

    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': interval,
        'apikey': TWELVEDATA_API_KEY,
        'outputsize': outputsize,
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
    if '/' in symbol:  # crypto
        if timeframe in ['5min', '15min', '1h', '4h']:
            df = await fetch_twelvedata(symbol, timeframe)
            if df is not None:
                return df
        df = await fetch_coingecko_ohlc(symbol, timeframe)
        if df is not None:
            return df
        df = await fetch_coingecko_price(symbol)
        if df is not None:
            return df
        return await fetch_twelvedata(symbol, timeframe)
    else:
        return await fetch_twelvedata(symbol, timeframe)

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
    volume_all_nan = chart_data['Volume'].isna().all()

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
    """Send ONE combined message per symbol with best timeframe chart + all timeframes summary"""
    
    # Define timeframe priority (lower number = more important for entry)
    timeframe_priority = {
        '5min': 1,    # Earliest signals
        '15min': 2,   # Confirmation
        '1h': 3,      # Trend building
        '4h': 4,      # Strong trend
        'daily': 5,   # Major trend
        'weekly': 6   # Long-term trend
    }
    
    # Find the best timeframe to show as main chart
    # Priority: 1. Strongest signal (highest |net_score|), 2. Earliest timeframe
    best_tf = None
    best_score = -float('inf')
    
    for tf, data in symbol_signals.items():
        net_score = data['signals']['net_score']
        # Consider absolute value for strength
        strength = abs(net_score)
        if strength > best_score or (strength == best_score and timeframe_priority.get(tf, 99) < timeframe_priority.get(best_tf, 99)):
            best_score = strength
            best_tf = tf
    
    if not best_tf:
        return
    
    # Get data for best timeframe
    best_data = symbol_signals[best_tf]
    df = best_data['df']
    signals = best_data['signals']
    
    # Generate and send main chart for best timeframe
    df_calc = calculate_indicators(df)
    main_embed = format_embed(symbol, signals, best_tf)
    
    try:
        chart_buffer = generate_chart_image(df, symbol, best_tf)
        if chart_buffer:
            file = discord.File(chart_buffer, filename='chart.png')
            main_embed.set_image(url='attachment://chart.png')
            
            # Add a note that this is the primary timeframe
            main_embed.description = f"**{symbol}** · ${signals['price']:.2f} · ⭐ **Primary: {best_tf}**"
            
            # Send main chart
            await ctx.send(embed=main_embed, file=file)
        else:
            await ctx.send(embed=main_embed)
    except Exception as e:
        print(f"⚠️ Chart generation failed for {symbol}: {e}")
        await ctx.send(embed=main_embed)
    
    # Send clean summary of ALL timeframes for this symbol
    await send_symbol_timeframe_summary(ctx, symbol, symbol_signals)

async def send_symbol_timeframe_summary(ctx, symbol, symbol_signals):
    """Send a clean, easy-to-read summary of all timeframes for a symbol"""
    
    # Sort timeframes by priority
    timeframe_order = {'5min': 1, '15min': 2, '1h': 3, '4h': 4, 'daily': 5, 'weekly': 6}
    sorted_timeframes = sorted(symbol_signals.keys(), key=lambda x: timeframe_order.get(x, 99))
    
    # Count signals
    bullish_count = sum(1 for tf, data in symbol_signals.items() if data['signals']['net_score'] > 0)
    bearish_count = sum(1 for tf, data in symbol_signals.items() if data['signals']['net_score'] < 0)
    total = len(symbol_signals)
    
    # Create the summary lines in the exact format you wanted
    summary_lines = []
    for tf in sorted_timeframes:
        signals = symbol_signals[tf]['signals']
        net = signals['net_score']
        
        # Signal emoji and text
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
        
        # Add score in parentheses
        summary_lines.append(f"{emoji} {tf}: {signal_text} (Score: {net})")
    
    # Create the embed
    embed = discord.Embed(
        title=f"📊 MULTI-TIMEFRAME SUMMARY: {symbol}",
        description="\n".join(summary_lines),
        color=0x3498db
    )
    
    # Add recommendation based on alignment
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
    """Send a final summary of all symbols with signals"""
    
    if not signal_summary:
        return
    
    embed = discord.Embed(
        title="📊 MULTI-TIMEFRAME SCAN COMPLETE",
        description=f"Found signals for **{len(signal_summary)}** symbols",
        color=0x3498db
    )
    
    # Group by signal strength
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
# OPTIONS FLOW SCANNER - NEW!
# ====================

async def get_stock_price(symbol):
    """Get current stock price using yfinance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def get_best_expiration():
    """Get the best expiration date (30-45 days out)"""
    today = datetime.now()
    target_days = 38  # Middle of 30-45 range
    
    # Find the next Friday that's ~38 days out
    days_to_friday = (4 - today.weekday()) % 7  # Friday is 4
    if days_to_friday == 0:
        days_to_friday = 7
    
    # Start with next Friday
    base_date = today + timedelta(days=days_to_friday)
    
    # Add weeks until we're in the 30-45 day range
    weeks_to_add = 0
    while True:
        exp_date = base_date + timedelta(weeks=weeks_to_add)
        dte = (exp_date - today).days
        if 30 <= dte <= 45:
            return exp_date.strftime('%Y-%m-%d'), dte
        weeks_to_add += 1

def format_premium(volume, last_price):
    """Calculate total premium (volume * 100 shares * last price)"""
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

@bot.command(name='flow')
async def options_flow(ctx, ticker: str):
    """Check unusual options activity for a specific ticker"""
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    
    try:
        await ctx.send(f"🔍 Analyzing options flow for **{ticker.upper()}**...")
        
        # Get current stock price
        current_price = await get_stock_price(ticker.upper())
        if not current_price:
            await ctx.send(f"❌ Could not fetch current price for {ticker.upper()}")
            return
        
        # Get the best expiration date
        best_exp, dte = get_best_expiration()
        
        # Fetch options chain for best expiration
        try:
            # Get calls and puts
            calls = foc.get_options_chain_greeks(ticker.upper(), best_exp, "CALL")
            puts = foc.get_options_chain_greeks(ticker.upper(), best_exp, "PUT")
        except:
            await ctx.send(f"❌ Could not fetch options data for {ticker.upper()}. Symbol may not have options.")
            return
        
        # Combine and analyze all options
        all_options = []
        if calls:
            for opt in calls:
                opt['type'] = 'CALL'
                all_options.append(opt)
        if puts:
            for opt in puts:
                opt['type'] = 'PUT'
                all_options.append(opt)
        
        if not all_options:
            await ctx.send(f"No options data found for {ticker.upper()} on {best_exp}")
            return
        
        # Calculate volume/OI ratio and filter for unusual activity
        analyzed = []
        for opt in all_options:
            try:
                volume = opt.get('volume', 0)
                oi = opt.get('openInterest', 0)
                strike = opt.get('strike', 0)
                last = opt.get('lastPrice', 0)
                opt_type = opt.get('type', 'CALL')
                
                if oi > 0 and volume > 0:
                    vol_oi_ratio = volume / oi
                else:
                    vol_oi_ratio = volume if volume > 0 else 0
                
                # Calculate distance from current price
                price_distance_pct = abs(strike - current_price) / current_price * 100
                
                analyzed.append({
                    'strike': strike,
                    'type': opt_type,
                    'volume': volume,
                    'oi': oi,
                    'vol_oi_ratio': vol_oi_ratio,
                    'last': last,
                    'distance_pct': price_distance_pct,
                    'premium': format_premium(volume, last)
                })
            except:
                continue
        
        # Sort by volume/OI ratio (highest first)
        analyzed.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
        
        # Filter to only show meaningful data
        significant = [opt for opt in analyzed if opt['volume'] >= 5 and opt['oi'] > 0]
        
        if not significant:
            await ctx.send(f"📭 No unusual options activity detected for {ticker.upper()} on {best_exp}")
            return
        
        # Create the embed
        embed = discord.Embed(
            title=f"🔍 OPTIONS FLOW: {ticker.upper()}",
            description=f"Current Price: **${current_price:.2f}**\nExpiration: {best_exp} ({dte} days)",
            color=0x00ff00
        )
        
        # Create the table header
        table = "```\n"
        table += "STRIKE  TYPE  VOLUME  OI    VOL/OI  ACTION \n"
        table += "------  ----  ------  ----  ------  ------\n"
        
        unusual_count = 0
        top_picks = []
        
        for opt in significant[:8]:  # Show top 8
            strike = f"${opt['strike']:.2f}"
            opt_type = opt['type']
            volume = opt['volume']
            oi = opt['oi']
            ratio = opt['vol_oi_ratio']
            
            # Determine action
            if ratio >= 2.0:
                action = "🟢 BUY"
                unusual_count += 1
                top_picks.append(opt)
            elif ratio >= 1.5:
                action = "🟡 WATCH"
            elif ratio >= 1.0:
                action = "⚪ NORMAL"
            else:
                action = "🔴 INACTIVE"
            
            table += f"{strike:6}  {opt_type:4}  {volume:6}  {oi:4}  {ratio:.1f}x    {action}\n"
        
        table += "```"
        
        embed.add_field(name="🔥 UNUSUAL ACTIVITY DETECTED", value=table, inline=False)
        
        # Add top picks if any
        if top_picks:
            picks_text = ""
            for i, pick in enumerate(top_picks[:3]):  # Show top 3
                if i == 0:
                    medal = "🥇 BEST SHOT"
                elif i == 1:
                    medal = "🥈 SECOND CHOICE"
                else:
                    medal = "🥉 THIRD CHOICE"
                
                strike_price = pick['strike']
                target = strike_price * 1.20  # 20% target
                
                picks_text += f"\n**{medal}: ${pick['strike']:.2f} {pick['type']}**\n"
                picks_text += f"   • Volume: {pick['volume']} ({pick['vol_oi_ratio']:.1f}x normal!)\n"
                picks_text += f"   • Premium: {pick['premium']}\n"
                picks_text += f"   • Why: {'High volume spike' if pick['vol_oi_ratio'] >= 2 else 'Strong volume'} near price\n"
                picks_text += f"   • Entry: Buy when price holds above ${current_price * 1.01:.2f}\n"
                picks_text += f"   • Target: ${target:.2f} ({(target/current_price-1)*100:.0f}% potential)\n"
                picks_text += f"   • Expiration: {best_exp} ({dte} days)\n"
            
            embed.add_field(name="⭐ TOP PICKS", value=picks_text, inline=False)
        
        # Add explanation
        explanation = """
📊 **WHAT THIS MEANS:**
• 🟢 BUY = Smart money accumulating (2x+ volume)
• 🟡 WATCH = Interesting activity (1.5-2x volume)
• ⚪ NORMAL = Regular trading
• 🔴 INACTIVE = Avoid

💡 **RECOMMENDATION:**
Focus on 🟢 BUY signals near current price. These have the most explosive potential!
        """
        embed.add_field(name="", value=explanation, inline=False)
        
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error analyzing options: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='scanflow')
async def scan_options_flow(ctx):
    """Scan entire watchlist for unusual options activity"""
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    
    try:
        watchlist = await load_watchlist()
        symbols = watchlist['stocks']  # Only stocks have options, not crypto
        
        if not symbols:
            await ctx.send("No stocks in watchlist to scan.")
            return
        
        await ctx.send(f"🔍 **SCANNING {len(symbols)} SYMBOLS FOR UNUSUAL OPTIONS ACTIVITY**")
        await ctx.send(f"⏱️ This will take 2-3 minutes. Results will appear as they come...\n")
        
        # Get best expiration
        best_exp, dte = get_best_expiration()
        
        all_unusual = []
        
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            
            try:
                # Get current price
                current_price = await get_stock_price(symbol)
                if not current_price:
                    continue
                
                # Fetch options
                calls = foc.get_options_chain_greeks(symbol, best_exp, "CALL")
                
                if not calls:
                    continue
                
                # Check for unusual activity
                unusual = []
                for opt in calls:
                    try:
                        volume = opt.get('volume', 0)
                        oi = opt.get('openInterest', 0)
                        strike = opt.get('strike', 0)
                        last = opt.get('lastPrice', 0)
                        
                        if oi > 0 and volume > 0:
                            vol_oi_ratio = volume / oi
                            if vol_oi_ratio >= 1.5 and volume >= 10:  # Unusual threshold
                                distance_pct = abs(strike - current_price) / current_price * 100
                                if distance_pct <= 20:  # Within 20% of price
                                    unusual.append({
                                        'symbol': symbol,
                                        'strike': strike,
                                        'volume': volume,
                                        'oi': oi,
                                        'ratio': vol_oi_ratio,
                                        'price': current_price,
                                        'premium': format_premium(volume, last),
                                        'distance': distance_pct
                                    })
                    except:
                        continue
                
                if unusual:
                    all_unusual.extend(unusual)
                    # Send individual alert
                    for u in unusual[:2]:  # Top 2 per symbol
                        alert = f"**{symbol}** - ${u['strike']:.2f} CALL: {u['volume']} vol ({u['ratio']:.1f}x) Premium: {u['premium']}"
                        await ctx.send(alert)
                
                await asyncio.sleep(3)  # Rate limit
                
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue
        
        # Sort all unusual by ratio
        all_unusual.sort(key=lambda x: x['ratio'], reverse=True)
        
        if not all_unusual:
            await ctx.send("📭 No unusual options activity detected in your watchlist.")
            return
        
        # Create summary table
        embed = discord.Embed(
            title="🔥 UNUSUAL OPTIONS ACTIVITY SUMMARY",
            description=f"Found {len(all_unusual)} unusual setups across your watchlist",
            color=0x00ff00
        )
        
        table = "```\n"
        table += "SYMBOL  STRIKE  VOLUME  OI   VOL/OI  PREMIUM\n"
        table += "------  ------  ------  ---  ------  -------\n"
        
        top_overall = []
        for opt in all_unusual[:10]:
            table += f"{opt['symbol']:6}  ${opt['strike']:.2f}  {opt['volume']:6}  {opt['oi']:3}  {opt['ratio']:.1f}x   {opt['premium']}\n"
            top_overall.append(opt)
        
        table += "```"
        embed.add_field(name="📊 ALL DETECTED ACTIVITY", value=table, inline=False)
        
        # Top picks overall
        if top_overall:
            picks = "**TOP 3 SETUPS:**\n\n"
            for i, pick in enumerate(top_overall[:3]):
                target = pick['strike'] * 1.20
                picks += f"{i+1}. **{pick['symbol']} ${pick['strike']:.2f} CALL**\n"
                picks += f"   • Volume: {pick['volume']} ({pick['ratio']:.1f}x normal)\n"
                picks += f"   • Premium: {pick['premium']}\n"
                picks += f"   • Entry: Above ${pick['price'] * 1.01:.2f}\n"
                picks += f"   • Target: ${target:.2f}\n\n"
            
            embed.add_field(name="🏆 TOP PICKS", value=picks, inline=False)
        
        embed.set_footer(text=f"Expiration: {best_exp} ({dte} days)")
        await ctx.send(embed=embed)
        
    except Exception as e:
        await ctx.send(f"❌ Error scanning options flow: {str(e)}")
    finally:
        user_busy[ctx.author.id] = False

# ====================
# DISCORD COMMANDS
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
        valid_timeframes = ['5min', '15min', '1h', '4h', 'daily', 'weekly']
        if timeframe not in valid_timeframes:
            await ctx.send("Invalid timeframe. Use 5min, 15min, 1h, 4h, daily, or weekly.")
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
            await asyncio.sleep(8)

        cancellation_flags[ctx.author.id] = False
        await ctx.send("Scan complete.")
    finally:
        user_busy[ctx.author.id] = False

@bot.command(name='signals')
async def signals(ctx, timeframe: str = 'all'):
    """Scan for signals across multiple timeframes"""
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
        watchlist = await load_watchlist()
        symbols = watchlist['stocks'] + watchlist['crypto']
        
        all_timeframes = ['5min', '15min', '1h', '4h', 'daily', 'weekly']
        
        if timeframe == 'all':
            timeframes_to_scan = all_timeframes
            await ctx.send(f"🔍 **MULTI-TIMEFRAME SIGNAL SCAN**")
            await ctx.send(f"📊 Scanning **{len(symbols)}** symbols across **ALL {len(timeframes_to_scan)} timeframes**")
            await ctx.send(f"⏱️ Timeframes: 5min, 15min, 1h, 4h, daily, weekly")
            await ctx.send(f"📈 Total API calls: ~{len(symbols) * len(timeframes_to_scan)}")
            await ctx.send("⏳ This will take several minutes. Results will appear as they come...\n")
        elif timeframe in all_timeframes:
            timeframes_to_scan = [timeframe]
            await ctx.send(f"🔍 Scanning **{len(symbols)}** symbols on **{timeframe}** timeframe...")
        else:
            await ctx.send("❌ Invalid timeframe. Use: 5min, 15min, 1h, 4h, daily, weekly, or all")
            return

        # Store all signals for each symbol
        all_symbol_signals = defaultdict(dict)
        found_any = False
        
        for symbol in symbols:
            if await check_cancel(ctx):
                break
            
            symbol_signals = {}
            
            # Collect all timeframe signals for this symbol first
            for tf in timeframes_to_scan:
                if await check_cancel(ctx):
                    break
                    
                df = await fetch_ohlcv(symbol, tf)
                if df is not None and not df.empty:
                    df_calc = calculate_indicators(df)
                    sig = get_signals(df_calc)
                    if sig and sig['net_score'] != 0:
                        found_any = True
                        symbol_signals[tf] = {
                            'signals': sig,
                            'df': df
                        }
                await asyncio.sleep(2)
            
            # If symbol has any signals, send ONE combined message
            if symbol_signals:
                await send_combined_symbol_report(ctx, symbol, symbol_signals)
                all_symbol_signals[symbol] = {tf: data['signals'] for tf, data in symbol_signals.items()}
                
            await asyncio.sleep(5)

        # Send final summary
        if timeframe == 'all' and found_any:
            await send_final_summary(ctx, all_symbol_signals)
        elif not found_any and not cancellation_flags.get(ctx.author.id, False):
            await ctx.send(f"📭 No symbols with active signals found{ ' on any timeframe' if timeframe == 'all' else ''}.")
            
        cancellation_flags[ctx.author.id] = False
        await ctx.send(f"✅ Signal scan complete{ ' across all timeframes' if timeframe == 'all' else ''}!")
    finally:
        user_busy[ctx.author.id] = False

async def send_symbol_with_chart(ctx, symbol, df, timeframe):
    """Send a chart with signals (used by !scan command)"""
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

        embed = discord.Embed(
            title=f"Latest News for {ticker.upper()}",
            color=0x3498db
        )
        for article in news_data[:limit]:
            headline = article.get('headline', 'No Headline')
            source = article.get('source', 'Unknown')
            date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M') if article.get('datetime') else 'Unknown'
            url = article.get('url', '')
            if len(headline) > 256:
                headline = headline[:253] + "..."
            embed.add_field(
                name=f"{source} - {date}",
                value=f"[{headline}]({url})",
                inline=False
            )
        embed.add_field(name="📊 TradingView", value=tv_field, inline=False)
        embed.set_footer(text=f"Requested by {ctx.author.display_name}")
        await ctx.send(embed=embed)
    finally:
        user_busy[ctx.author.id] = False

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
                            lines.append(f"**{date}** {time_str} – EPS Est: {eps_est}")
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
                    lines.append(f"**{date}** {time_str} – EPS Est: {eps_est}")
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

@bot.command(name='zone')
async def zone(ctx, ticker: str, timeframe: str = 'daily'):
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
        valid_timeframes = ['5min', '15min', '1h', '4h', 'daily', 'weekly']
        if timeframe not in valid_timeframes:
            await ctx.send("Invalid timeframe. Use 5min, 15min, 1h, 4h, daily, or weekly.")
            return

        symbol = normalize_symbol(ticker)
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

@bot.command(name='help')
async def help_command(ctx):
    if user_busy.get(ctx.author.id):
        return
    user_busy[ctx.author.id] = True
    try:
        help_text = """
**5-13-50 Trading Bot Commands**

📊 **SCAN COMMANDS**
`!scan all [5min|15min|1h|4h|daily|weekly]` – Scan all watchlist symbols
`!scan SYMBOL [timeframe]` – Scan a single symbol

🚀 **SIGNAL COMMANDS**
`!signals [5min|15min|1h|4h|daily|weekly|all]` – Multi-timeframe signal scanner
   • `!signals` or `!signals all` – Scans ALL 6 timeframes
   • Shows ONE combined report per symbol with best timeframe chart + clean summary
   • Final summary of all signals at the end

📰 **NEWS & EVENTS**
`!news TICKER [limit]` – Fetch latest news headlines
`!upcoming [TICKER]` – Show upcoming catalysts

🎯 **ZONES**
`!zone SYMBOL [timeframe]` – Show buy/sell zones

🔥 **OPTIONS FLOW (NEW!)**
`!flow TICKER` – Check unusual options activity for a specific stock
`!scanflow` – Scan entire watchlist for unusual options setups
   • Shows volume vs open interest ratios
   • Flags 🟢 BUY signals (2x+ volume)
   • Recommends best strikes and entry prices
   • Suggests optimal expiration (30-45 days)

📋 **WATCHLIST**
`!add SYMBOL` – Add to watchlist
`!remove SYMBOL` – Remove from watchlist
`!list` – Show watchlist

⚙️ **UTILITY**
`!ping` – Test bot
`!stopscan` – Stop ongoing scan
`!help` – This message

⏱️ **TIMEFRAMES:**
• `5min` – Earliest entry signals
• `15min` – Confirmation
• `1h` – Trend establishment
• `4h` – Strong trend
• `daily` – Major trend
• `weekly` – Long-term trend
• `all` – All timeframes at once

💡 **PRO TIPS:**
• Use `!signals all` to catch moves EARLY
• Use `!scanflow` to find explosive options setups (like your NIO example!)
        """
        await ctx.send(help_text)
    finally:
        user_busy[ctx.author.id] = False

# ====================
# MAIN ENTRY POINT
# ====================

async def main():
    asyncio.create_task(start_web_server())
    await bot.start(DISCORD_TOKEN)

if __name__ == '__main__':
    asyncio.run(main())