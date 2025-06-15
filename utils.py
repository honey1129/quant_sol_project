import pandas as pd
import numpy as np
import ccxt
import os
import requests
import config
import logging
import time

from features import get_feature_columns

# 自动创建目录
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

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



# 趋势过滤器（简单EMA斜率过滤）
def trend_filter(df, window=20):
    ema = df['close'].ewm(span=window).mean()
    slope = ema.diff()
    return slope

# 动态阈值波动率过滤
def volatility_filter(df, window=20):
    vol = df['close'].pct_change().rolling(window).std()
    return vol

# 特征数据准备（预测时用）
def prepare_features(df):
    df = df.copy()
    features = get_feature_columns()
    return df[features].astype(float)

# 简单资金曲线计算 (仅用于快速debug)
def calc_equity_curve(df, returns):
    curve = (1 + returns).cumprod()
    return pd.DataFrame({'timestamp': df.index, 'equity': curve}).set_index('timestamp')

# Telegram通知统一封装
def send_telegram(message):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram通知失败: {e}")