import os
import joblib
import traceback
import numpy as np
from core.signal_engine import SignalSmoother, bayesian_weighted_predict, load_models
from utils.utils import log_info, log_error, BASE_DIR
from config import config
from core.okx_api import OKXClient
from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features
from core.position_manager import PositionManager

# 初始化对象
client = OKXClient()
position_manager = PositionManager()
smoother = SignalSmoother(alpha=float(config.SMOOTH_ALPHA))  # alpha注意强转为float

# 止盈止损逻辑
def risk_control():
    market_price = client.get_price()
    long_pos, short_pos = client.get_position()

    # 多仓风控
    if long_pos['size'] > 0:
        entry_price = long_pos['entry_price']
        size = long_pos['size']
        pnl_pct = (market_price - entry_price) / entry_price
        usd_amount = size * entry_price / config.LEVERAGE

        if pnl_pct >= config.TAKE_PROFIT:
            client.close_long(usd_amount, config.LEVERAGE)
            log_info(f"✅ LONG 止盈平仓，收益率: {pnl_pct*100:.2f}%")
        elif pnl_pct <= -config.STOP_LOSS:
            client.close_long(usd_amount, config.LEVERAGE)
            log_info(f"❌ LONG 止损平仓，收益率: {pnl_pct*100:.2f}%")

    # 空仓风控
    if short_pos['size'] > 0:
        entry_price = short_pos['entry_price']
        size = short_pos['size']
        pnl_pct = (entry_price - market_price) / entry_price
        usd_amount = size * entry_price / config.LEVERAGE

        if pnl_pct >= config.TAKE_PROFIT:
            client.close_short(usd_amount, config.LEVERAGE)
            log_info(f"✅ SHORT 止盈平仓，收益率: {pnl_pct*100:.2f}%")
        elif pnl_pct <= -config.STOP_LOSS:
            client.close_short(usd_amount, config.LEVERAGE)
            log_info(f"❌ SHORT 止损平仓，收益率: {pnl_pct*100:.2f}%")


# 预测信号模块
def predict_signal(model_dict, model_weights):
    data_dict = client.fetch_data()
    merged_df = merge_multi_period_features(data_dict)
    merged_df = add_advanced_features(merged_df)

    feature_path = os.path.join(BASE_DIR, config.FEATURE_LIST_PATH)
    feature_cols = joblib.load(feature_path)

    prob = bayesian_weighted_predict(model_dict, merged_df, feature_cols, model_weights)
    smoothed_prob = smoother.smooth(prob)

    long_prob, short_prob = smoothed_prob[1], smoothed_prob[0]
    money_flow_ratio = merged_df['money_flow_ratio'].iloc[-1]
    merged_df['log_return'] = np.log(merged_df['5m_close'] / merged_df['5m_close'].shift(1))
    volatility = merged_df['log_return'].rolling(288).std().iloc[-1] * np.sqrt(288)

    log_info(f"实时预测 - 多: {long_prob:.3f} 空: {short_prob:.3f} (平滑后)")
    log_info(f"特征监控 - 资金流: {money_flow_ratio:.3f} 波动率: {volatility:.5f}")

    return long_prob, short_prob, money_flow_ratio, volatility

# 仓位调整核心
def adjust_position(long_prob, short_prob, money_flow_ratio, volatility):
    account_balance = client.get_account_balance()
    usdt_balance = float(account_balance['data'][0]['availEq'])

    long_pos, short_pos = client.get_position()

    adjusted_balance = position_manager.volatility_adjust_balance(usdt_balance, volatility)
    max_position_value = usdt_balance * float(config.MAX_POSITION_RATIO)
    MIN_ADJUST_AMOUNT = float(config.MIN_ADJUST_AMOUNT)

    # 多仓逻辑
    if long_prob > config.THRESHOLD_LONG:
        target_ratio = position_manager.calculate_target_ratio(long_prob, money_flow_ratio, volatility)
        target_value = min(adjusted_balance * target_ratio, max_position_value)
        delta_value = target_value - long_pos['size'] * long_pos['entry_price']
        delta_principal = delta_value / config.LEVERAGE

        if delta_principal > MIN_ADJUST_AMOUNT:
            client.open_long(delta_principal, config.LEVERAGE)
        elif delta_principal < -MIN_ADJUST_AMOUNT:
            client.close_long(abs(delta_principal), config.LEVERAGE)

    # 空仓逻辑
    if short_prob > config.THRESHOLD_SHORT:
        target_ratio = position_manager.calculate_target_ratio(short_prob, money_flow_ratio, volatility)
        target_value = min(adjusted_balance * target_ratio, max_position_value)
        delta_value = target_value - short_pos['size'] * short_pos['entry_price']
        delta_principal = delta_value / config.LEVERAGE

        if delta_principal > MIN_ADJUST_AMOUNT:
            client.open_short(delta_principal, config.LEVERAGE)
        elif delta_principal < -MIN_ADJUST_AMOUNT:
            client.close_short(abs(delta_principal), config.LEVERAGE)


# 主运行入口
def run():
    try:
        # 1. 模型加载
        model_paths = {name: os.path.join(BASE_DIR, path) for name, path in config.MODEL_PATHS.items()}
        model_dict = load_models(model_paths)
        model_weights = config.MODEL_WEIGHTS

        # 2. 风控止盈止损（先平仓，避免后续重复调整）
        risk_control()

        # 3. 获取新预测信号
        long_prob, short_prob, money_flow_ratio, volatility = predict_signal(model_dict, model_weights)

        # 4. 根据最新信号动态调仓
        adjust_position(long_prob, short_prob, money_flow_ratio, volatility)

    except Exception as e:
        log_error(f"实盘运行异常: {e}")
        log_error(traceback.format_exc())


if __name__ == '__main__':
    run()
