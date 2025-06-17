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
def risk_control(side, entry_price, size):
    market_price = client.get_price()
    change_pct = (market_price - entry_price) / entry_price
    pnl_pct = change_pct if side == 'long' else -change_pct
    profit_amount = (market_price - entry_price) * size
    usd_amount = size * entry_price / config.LEVERAGE
    if pnl_pct >= config.TAKE_PROFIT:
        if side == 'long':
            client.close_long(usd_amount, config.LEVERAGE)
        else:
            client.close_short(usd_amount, config.LEVERAGE)
        log_info(f"✅ {side.upper()} 仓止盈平仓，收益率: {pnl_pct * 100:.2f}%, 盈利金额: {profit_amount:.2f} USD")

    elif pnl_pct <= -config.STOP_LOSS:
        if side == 'long':
            client.close_long(usd_amount, config.LEVERAGE)
        else:
            client.close_short(usd_amount, config.LEVERAGE)
        log_info(f"❌ {side.upper()} 仓止损平仓，收益率: {pnl_pct * 100:.2f}%, 盈亏金额: {profit_amount:.2f} USD")
    else:
        log_info(f"🔄 {side.upper()} 仓监控中，无平仓动作。当前收益率: {pnl_pct * 100:.2f}%, 当前盈亏: {profit_amount:.2f} USD")

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
    details = account_balance['data'][0]['details']
    usdt_detail = next((d for d in details if d['ccy'] == 'USDT'), None)
    total_balance = float(usdt_detail['eq']) if usdt_detail else 0.0

    side, current_size, entry_price = client.get_position()
    current_value = current_size * entry_price if entry_price > 0 else 0
    max_position_value = total_balance * float(config.MAX_POSITION_RATIO)
    MIN_ADJUST_AMOUNT = float(config.MIN_ADJUST_AMOUNT)

    # 信号反转平仓逻辑
    if long_prob > config.THRESHOLD_LONG and side == 'short':
        if current_size > 0:
            principal_amount = current_value / config.LEVERAGE
            client.close_short(principal_amount, config.LEVERAGE)
            log_info(f"🔄 信号反转，已平空仓: 本金 {principal_amount} USD")
        side, current_size, entry_price = 'none', 0, 0

    if short_prob > config.THRESHOLD_SHORT and side == 'long':
        if current_size > 0:
            principal_amount = current_value / config.LEVERAGE
            client.close_long(principal_amount, config.LEVERAGE)
            log_info(f"🔄 信号反转，已平多仓: 本金 {principal_amount} USD")
        side, current_size, entry_price = 'none', 0, 0

    current_value = current_size * entry_price if entry_price > 0 else 0
    adjusted_balance = position_manager.volatility_adjust_balance(total_balance, volatility)

    # 多头逻辑
    if long_prob > config.THRESHOLD_LONG:
        target_ratio = position_manager.calculate_target_ratio(long_prob, money_flow_ratio, volatility)
        target_value = min(adjusted_balance * target_ratio, max_position_value)
        delta_value = target_value - current_value
        delta_principal = delta_value / config.LEVERAGE

        if abs(delta_principal) >= MIN_ADJUST_AMOUNT:
            if delta_principal > 0:
                client.open_long(delta_principal, config.LEVERAGE)
                log_info(f"📈 加多仓: {delta_principal} USD 本金")
            else:
                client.close_long(abs(delta_principal), config.LEVERAGE)
                log_info(f"📉 减多仓: {abs(delta_principal)} USD 本金")
        else:
            log_info("🟢 多仓已达目标，无需调整")

    # 空头逻辑
    elif short_prob > config.THRESHOLD_SHORT:
        target_ratio = position_manager.calculate_target_ratio(short_prob, money_flow_ratio, volatility)
        target_value = min(adjusted_balance * target_ratio, max_position_value)
        delta_value = target_value - current_value
        delta_principal = delta_value / config.LEVERAGE

        if abs(delta_principal) >= MIN_ADJUST_AMOUNT:
            if delta_principal > 0:
                client.open_short(delta_principal, config.LEVERAGE)
                log_info(f"📈 加空仓: {delta_principal} USD 本金")
            else:
                client.close_short(abs(delta_principal), config.LEVERAGE)
                log_info(f"📉 减空仓: {abs(delta_principal)} USD 本金")
        else:
            log_info("🟢 空仓已达目标，无需调整")
    else:
        log_info("📊 当前无明显信号，仓位保持不变")

# 主运行入口
def run():
    try:
        model_paths = {name: os.path.join(BASE_DIR, path) for name, path in config.MODEL_PATHS.items()}
        model_dict = load_models(model_paths)
        model_weights = config.MODEL_WEIGHTS

        side, size, entry_price = client.get_position()
        if side != 'none':
            risk_control(side, entry_price, size)

        long_prob, short_prob, money_flow_ratio, volatility = predict_signal(model_dict, model_weights)
        adjust_position(long_prob, short_prob, money_flow_ratio, volatility)

    except Exception as e:
        log_error(f"实盘运行异常: {e}")
        log_error(traceback.format_exc())

if __name__ == '__main__':
    run()
