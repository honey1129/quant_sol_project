import time
import pandas as pd
import joblib
import traceback
import logging
from utils import add_indicators, get_feature_columns, send_telegram  # 你已有的工具函数
import config
from okx_api import OKXClient
import os
client = OKXClient()

# 自动创建日志目录
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


# 获取历史K线数据
def fetch_ohlcv(max_retry=3, sleep_sec=1):
    for attempt in range(max_retry):
        try:
            raw_data = client.market_api.get_candlesticks(instId=config.SYMBOL, bar='1H', limit=100)['data']
            raw_data = list(reversed(raw_data))
            df = pd.DataFrame(raw_data)
            df = df.iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            return df
        except Exception as e:
            print(f"⚠ 拉取K线失败，第{attempt+1}次重试: {e}")
            time.sleep(sleep_sec)
    raise Exception("❌ 超过最大重试次数，fetch_ohlcv() 彻底失败")

# 风控逻辑（止盈止损）
def risk_control(side, entry_price, size):
    market_price = client.get_price()

    change_pct = (market_price - entry_price) / entry_price
    pnl_pct = change_pct if side == 'long' else -change_pct

    profit_amount = (market_price - entry_price) * size

    if pnl_pct >= config.TAKE_PROFIT:
        if side == 'long':
            client.close_long(size)
        else:
            client.close_short(size)
        log_info(f"✅ {side.upper()} 仓止盈平仓，收益: {pnl_pct * 100:.2f}%, 盈利金额: {profit_amount:.2f} USD")

    elif pnl_pct <= -config.STOP_LOSS:
        if side == 'long':
            client.close_long(size)
        else:
            client.close_short(size)
        log_info(f"❌ {side.upper()} 仓止损平仓，收益: {pnl_pct * 100:.2f}%, 盈亏金额: {profit_amount:.2f} USD")

    else:
        log_info(
            f"🔄 {side.upper()} 仓监控中，无平仓动作。当前收益: {pnl_pct * 100:.2f}%, 当前盈亏: {profit_amount:.2f} USD")


# 模型预测信号
def predict_signal(model, df):
    features = get_feature_columns()
    X_live = df[features].iloc[-1:].astype(float)
    prob = model.predict_proba(X_live)[0]
    long_prob, short_prob = prob[1], prob[0]

    log_info(f"实时预测 - 多: {long_prob:.3f} 空: {short_prob:.3f}")

    if long_prob > config.THRESHOLD_LONG:
        return 'long'
    elif short_prob > config.THRESHOLD_SHORT:
        return 'short'
    else:
        return 'neutral'

# 下单逻辑

def place_order(signal):
    price = client.get_price()
    size = round(config.POSITION_SIZE * config.LEVERAGE / price, 3)

    if signal == 'long':
        client.open_long(size)
        log_info(f"✅ 开多仓: {size}")

    elif signal == 'short':
        client.open_short(size)
        log_info(f"✅ 开空仓: {size}")

    else:
        log_info("当前无信号，继续观望。")

# 主逻辑
def run():
    try:
        df = fetch_ohlcv()  # 你已有完整的K线数据抓取逻辑
        df = add_indicators(df)
        model = joblib.load(config.MODEL_PATH)
        account_balance  = client.get_account_balance()
        log_info(f"📊 当前账户余额: {account_balance['data'][0]['totalEq']} USDT")
        side, size, entry_price = client.get_position()
        log_info(f"📊 当前仓位: {side} | 仓位: {size} | 开仓价: {entry_price}")

        if side != 'none':
            risk_control(side, entry_price, size)
        else:
            signal = predict_signal(model, df)
            place_order(signal)

    except Exception as e:
        log_error(f"实盘运行异常: {e}")
        log_error(traceback.format_exc())

if __name__ == '__main__':
    run()
