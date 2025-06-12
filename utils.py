import pandas as pd
import numpy as np
import ccxt
import os
import requests
import config
import logging
# 自动创建目录
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    filename='logs/live_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_info(msg):
    print(msg)
    logging.info(msg)
    send_telegram(msg)

def log_error(msg):
    print("❌", msg)
    logging.error(msg)
    send_telegram(f"❌ {msg}")



# 拉取历史数据
def fetch_binance_data(symbol='SOL/USDT', timeframe='30m', limit=3000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 增强版特征工程
def add_indicators(df):
    # 基础技术指标
    df['return'] = df['close'].pct_change(12)
    df['ema_short'] = df['close'].ewm(span=20).mean()
    df['ema_long'] = df['close'].ewm(span=60).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(window=14).mean()

    # MACD
    ema_fast = df['close'].ewm(span=12).mean()
    ema_slow = df['close'].ewm(span=26).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['close'].rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

    # 增强指标部分（新增因子）

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # CCI
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci'] = (tp - ma) / (0.015 * md)

    # Stochastic Oscillator
    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['stoch'] = 100 * (df['close'] - low_min) / (high_max - low_min)

    # Volatility (简单标准差)
    df['volatility'] = df['close'].rolling(window=20).std()

    # 短期收益变化（预测未来趋势用）
    df['price_change'] = df['close'].pct_change(3)

    # 清洗缺失
    df.dropna(inplace=True)
    return df

# 完整特征列表
def get_feature_columns():
    return [
        'return', 'ema_short', 'ema_long', 'rsi', 'atr',
        'macd', 'macd_signal', 'boll_mid', 'boll_upper', 'boll_lower',
        'obv', 'cci', 'stoch', 'volatility', 'price_change', 'volume'
    ]


# Telegram通知统一封装
def send_telegram(message):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram通知失败: {e}")