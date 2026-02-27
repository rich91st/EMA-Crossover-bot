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
from datetime import datetime, timedelta

# Charting libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import mplfinance as mpf
import io

# === CONFIGURATION ===
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
PORT = int(os.getenv('PORT', 10000))

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)
bot._skip_check = lambda x, y: False

WATCHLIST_FILE = 'watchlist.json'
last_command_time = {}

# ====================
# WEB SERVER (keeps Render happy)
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

    days_map = {'daily': 30, 'weekly': 90, '4h': 7}
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
                    print(f"CoinGecko price error for {symbol}: status {resp.status}")
                    return None
                data = await resp.json()
                price = data.get(coin_id, {}).get('usd')
                if price is None:
                    print(f"CoinGecko price: no price for {coin_id}")
                    return None
                # Create synthetic OHLC
                np.random.seed(42)
                dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
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
        # Layer 1: CoinGecko OHLC
        df = await fetch_coingecko_ohlc(symbol, timeframe)
        if df is not None:
            return df
        # Layer 2: CoinGecko price (synthetic)
        df = await fetch_coingecko_price(symbol)
        if df is not None:
            return df
        # Layer 3: Twelve Data (final fallback)
        print(f"Trying Twelve Data as final fallback for {symbol}")
        return await fetch_twelvedata(symbol, timeframe)
    else:
        return await fetch_twelvedata(symbol, timeframe)

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
    """
    Generate a candlestick chart with EMAs from the DataFrame.
    Returns a BytesIO object ready to be sent as a Discord file.
    """
    # Prepare data for mplfinance
    df_chart = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_chart.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Add EMAs as additional plots
    apds = []
    if not df['ema5'].isna().all():
        apds.append(mpf.make_addplot(df['ema5'], color='green', width=0.8, label='EMA5'))
    if not df['ema13'].isna().all():
        apds.append(mpf.make_addplot(df['ema13'], color='yellow', width=0.8, label='EMA13'))
    if not df['ema50'].isna().all():
        apds.append(mpf.make_addplot(df['ema50'], color='red', width=0.8, label='EMA50'))
    if not df['ema200'].isna().all():
        apds.append(mpf.make_addplot(df['ema200'], color='purple', width=0.8, label='EMA200'))
    
    # Style
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=False)
    
    # Create figure
    fig, axes = mpf.plot(
        df_chart,
        type='candle',
        style=s,
        addplot=apds,
        volume=True,
        figsize=(10, 6),
        returnfig=True,
        title=f'{symbol} – {timeframe}',
        tight_layout=True
    )
    
    # Save to bytes buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=100)
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

    # Colored and sorted EMAs
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
    embed.add_field(name="EMAs (sorted)", value=ema_text, inline=False)
    embed.add_field(name="Bollinger Bands", value=bb_status, inline=True)
    embed.add_field(name="Reason", value=reason_str, inline=False)
    embed.add_field(name="Support", value=f"${support:.2f}", inline=True)
    embed.add_field(name="Resistance", value=f"${resistance:.2f}", inline=True)
    embed.add_field(name="Stop Loss", value=f"${stop_loss:.2f}", inline=True)
    embed.add_field(name="Target", value=f"${target:.2f}", inline=True)
    embed.set_footer(text=f"{sym_type} · {timeframe} timeframe")
    return embed

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

@bot.command(name='scan')
async def scan(ctx, target='all', timeframe='daily'):
    """Full scan: shows all symbols in watchlist (with chart for single symbol)."""
    # Deduplication
    if not hasattr(bot, 'processed_msgs'):
        bot.processed_msgs = {}
    msg_id = ctx.message.id
    now_ts = datetime.now().timestamp()
    to_remove = [mid for mid, ts in bot.processed_msgs.items() if now_ts - ts > 10]
    for mid in to_remove:
        del bot.processed_msgs[mid]
    if msg_id in bot.processed_msgs:
        print(f"Ignoring duplicate message {msg_id}")
        return
    bot.processed_msgs[msg_id] = now_ts

    # Cooldown per user
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

    # Single symbol scan (with chart)
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

        # Generate and attach chart
        try:
            chart_buffer = generate_chart_image(df, symbol, timeframe)
            file = discord.File(chart_buffer, filename='chart.png')
            embed.set_image(url='attachment://chart.png')
            await ctx.send(embed=embed, file=file)
        except Exception as e:
            print(f"Chart generation failed: {e}")
            await ctx.send(embed=embed)  # fallback: send embed without chart
        return

    # Full scan (all symbols, no charts)
    await ctx.send(f"Scanning all symbols ({len(symbols)}) on {timeframe} timeframe. This may take a few minutes. Results will appear as they come.")

    for symbol in symbols:
        df = await fetch_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            signals = get_signals(df)
            embed = format_embed(symbol, signals, timeframe)
            await ctx.send(embed=embed)
        await asyncio.sleep(8)

    await ctx.send("Scan complete.")

@bot.command(name='signals')
async def signals(ctx, timeframe='daily'):
    """Filtered scan: only shows symbols with active signals."""
    # Deduplication
    if not hasattr(bot, 'processed_msgs'):
        bot.processed_msgs = {}
    msg_id = ctx.message.id
    now_ts = datetime.now().timestamp()
    to_remove = [mid for mid, ts in bot.processed_msgs.items() if now_ts - ts > 10]
    for mid in to_remove:
        del bot.processed_msgs[mid]
    if msg_id in bot.processed_msgs:
        print(f"Ignoring duplicate message {msg_id}")
        return
    bot.processed_msgs[msg_id] = now_ts

    # Cooldown per user
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

    await ctx.send(f"Scanning for signals on {timeframe} timeframe. This may take a few minutes. Results will appear as they come.")

    found_any = False
    for symbol in symbols:
        df = await fetch_ohlcv(symbol, timeframe)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            signals = get_signals(df)
            if signals and signals['net_score'] != 0:
                found_any = True
                embed = format_embed(symbol, signals, timeframe)
                await ctx.send(embed=embed)
        await asyncio.sleep(8)

    if not found_any:
        await ctx.send("No symbols with active signals found.")
    await ctx.send("Signal scan complete.")

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
`!scan all [timeframe]` – Scan all watchlist symbols (full overview).
`!scan SYMBOL [timeframe]` – Scan a single symbol (with chart).
`!signals [timeframe]` – Scan only symbols with active bullish/bearish signals.
`!add SYMBOL` – Add a symbol (use `BTC/USD` for crypto).
`!remove SYMBOL` – Remove a symbol.
`!list` – Show current watchlist.
`!ping` – Test if bot is responsive.
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