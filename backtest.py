import numpy as np
import joblib
import matplotlib.pyplot as plt
import config
from okx_api import OKXClient
from ml_feature_engineering import merge_multi_period_features

# 参数配置
initial_balance = config.INITIAL_BALANCE
leverage = config.LEVERAGE
fee_rate = config.FEE_RATE
TAKE_PROFIT_PCT = 0.05
STOP_LOSS_PCT = 0.02
CONFIDENCE_THRESHOLD = 0.55
VERY_STRONG_THRESHOLD = 0.65
DECAY_FACTOR = 0.9
DECAY_WINDOW = 5
ADJUST_UNIT = 10  # 每次调仓最小单位 (USDT)

# 仓位目标计算
def compute_target_position_ratio(prob, vol, base_vol, balance, peak_balance, recent_losses, min_pos=0.05, max_pos=0.25):
    signal_strength = max(0, prob - 0.5) * 2
    volatility_factor = base_vol / vol if vol > 0 else 1
    drawdown = 1 - balance / peak_balance
    base_ratio = 0.2 * signal_strength * volatility_factor * (1 - drawdown)
    for _ in range(recent_losses):
        base_ratio *= DECAY_FACTOR
    return min(max(base_ratio, min_pos), max_pos)

# 最大回撤计算
def calculate_max_drawdown(equity_curve):
    high_water_mark = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - high_water_mark) / high_water_mark
    return drawdowns.min()

# 回测核心逻辑（动态持仓版）
def backtest_ml_dynamic_position():
    fetcher = OKXClient()
    data_dict = fetcher.fetch_data()
    merged_df = merge_multi_period_features(data_dict)
    merged_df.dropna(inplace=True)

    model = joblib.load(config.MODEL_PATH)
    features = [col for col in merged_df.columns if col not in ['future_return', 'target']]
    X = merged_df[features].astype(float)
    merged_df['prob_long'] = model.predict_proba(X)[:, 1]
    merged_df['rolling_vol'] = merged_df['5m_close'].pct_change().rolling(window=30).std()
    base_vol = merged_df['rolling_vol'].mean()

    balance = initial_balance
    peak_balance = initial_balance
    position = 0
    entry_price = 0
    margin_used = 0
    balance_curve = []
    trade_log = []
    win_count, lose_count = 0, 0
    loss_streak = 0

    for idx, row in merged_df.iterrows():
        price = row['5m_close']
        prob = row['prob_long']
        vol = row['rolling_vol']
        peak_balance = max(balance, peak_balance)

        target_ratio = compute_target_position_ratio(prob, vol, base_vol, balance, peak_balance, loss_streak)
        target_value = balance * leverage * target_ratio
        current_value = position * price

        # 计算应调仓位差
        delta_value = target_value - current_value

        if abs(delta_value) >= ADJUST_UNIT:
            delta_contract = delta_value / price
            # 仓位调整时更新entry_price (简单加权更新)
            if position + delta_contract != 0:
                entry_price = (entry_price * position + price * delta_contract) / (position + delta_contract)
            else:
                entry_price = 0

            position += delta_contract
            margin_change = (abs(delta_value) / leverage)
            margin_used += margin_change if delta_value > 0 else -margin_change
            balance -= margin_change if delta_value > 0 else -margin_change
            balance -= abs(delta_value) * fee_rate

            trade_log.append((idx, '调仓', price, balance, position))

        # 止盈止损逻辑（每根K线都检查）
        if position != 0:
            pnl_pct = (price - entry_price) / entry_price
            if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT:
                pnl = position * (price - entry_price)
                balance += margin_used + pnl
                balance -= abs(position * price) * fee_rate

                if pnl >= 0:
                    win_count += 1
                    loss_streak = 0
                else:
                    lose_count += 1
                    loss_streak = min(loss_streak + 1, DECAY_WINDOW)

                trade_log.append((idx, '止盈止损平仓', price, balance, position))
                position = 0
                margin_used = 0
                entry_price = 0

        balance_curve.append(balance + position * price if position != 0 else balance)

    # 收盘强平
    if position != 0:
        pnl = position * (price - entry_price)
        balance += margin_used + pnl
        balance -= abs(position * price) * fee_rate
        trade_log.append((idx, '收盘平仓', price, balance, position))

    final_balance = balance
    profit_pct = (final_balance - initial_balance) / initial_balance * 100
    max_dd = calculate_max_drawdown(np.array(balance_curve))
    sharpe = np.mean(np.diff(np.log(balance_curve))) / np.std(np.diff(np.log(balance_curve))) * np.sqrt(365*24)
    total_trades = win_count + lose_count + 1e-9
    win_rate = win_count / total_trades

    print("\n✅ 多周期机器学习动态持仓回测完成")
    print(f"最终资金: {final_balance:.2f}")
    print(f"总收益率: {profit_pct:.2f}%")
    print(f"最大回撤: {max_dd*100:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"交易次数: {int(total_trades)}")
    print(f"胜率: {win_rate*100:.2f}%")

    plt.figure(figsize=(10,6))
    plt.plot(balance_curve)
    plt.title("资金曲线 - 动态持仓增强版")
    plt.xlabel("周期")
    plt.ylabel("资金")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    backtest_ml_dynamic_position()
