import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import config
from live_trading_monitor import log_info
from utils import fetch_binance_data, add_indicators, get_feature_columns
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac通用
matplotlib.rcParams['axes.unicode_minus'] = False

from exchange_helper import init_okx

exchange = init_okx()

# 自动创建日志目录
os.makedirs("logs", exist_ok=True)

# 增强版回测核心逻辑
def backtest():
    # 拉取更长历史数据用于回测
    df = fetch_binance_data(limit=3000)
    df = add_indicators(df)
    features = get_feature_columns()

    # 加载新训练好的模型
    model = joblib.load(config.MODEL_PATH)

    balance = config.INITIAL_BALANCE
    position = 0
    entry_price = 0
    balances = []

    for idx, row in df.iterrows():
        X_row = pd.DataFrame([row[features].values], columns=features).astype(float)
        prob = model.predict_proba(X_row)[0]
        long_prob, short_prob = prob[1], prob[0]
        price = row['close']

        # 风控平仓逻辑
        if position != 0:
            change_pct = (price - entry_price) / entry_price
            if position > 0:
                if change_pct >= config.TAKE_PROFIT or change_pct <= -config.STOP_LOSS:
                    pnl = (price - entry_price) * abs(position)
                    balance += pnl
                    position = 0
                    entry_price = 0
            else:
                change_pct = (entry_price - price) / entry_price
                if change_pct >= config.TAKE_PROFIT or change_pct <= -config.STOP_LOSS:
                    pnl = (entry_price - price) * abs(position)
                    balance += pnl
                    position = 0
                    entry_price = 0

        # 无仓时执行新模型信号判定开仓
        if position == 0:
            size = config.POSITION_SIZE * config.LEVERAGE / price
            if long_prob > config.THRESHOLD_LONG:
                position = size
                entry_price = price
            elif short_prob > config.THRESHOLD_SHORT:
                position = -size
                entry_price = price

        balances.append(balance)

    df['balance'] = balances

    # 性能评估指标
    final_balance = df['balance'].iloc[-1]
    returns = df['balance'].pct_change().dropna()
    cum_return = (final_balance - config.INITIAL_BALANCE) / config.INITIAL_BALANCE
    max_dd = ((df['balance'].cummax() - df['balance']) / df['balance'].cummax()).max()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(24*365) if returns.std() > 0 else 0

    log_info("增强版回测结果：")
    log_info(f"最终收益: {cum_return * 100:.2f}%")
    log_info(f"最大回撤: {max_dd * 100:.2f}%")
    log_info(f"年化夏普: {sharpe:.2f}")

    # 资金曲线可视化
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['balance'], label='Balance Curve')
    plt.title("增强版资金曲线")
    plt.xlabel("时间")
    plt.ylabel("余额 (USDT)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    backtest()
