import joblib
import traceback
from utils import log_info, log_error
import config
from okx_api import OKXClient
from ml_feature_engineering import merge_multi_period_features
from position_manager import PositionManager

client = OKXClient()
position_manager = PositionManager()

# 风控逻辑（止盈止损）
def risk_control(side, entry_price, size):
    market_price = client.get_price()

    change_pct = (market_price - entry_price) / entry_price
    pnl_pct = change_pct if side == 'long' else -change_pct
    profit_amount = (market_price - entry_price) * size

    if pnl_pct >= config.TAKE_PROFIT:
        if side == 'long':
            client.close_long(size, config.LEVERAGE)
        else:
            client.close_short(size, config.LEVERAGE)
        log_info(f"✅ {side.upper()} 仓止盈平仓，收益率: {pnl_pct * 100:.2f}%, 盈利金额: {profit_amount:.2f} USD")

    elif pnl_pct <= -config.STOP_LOSS:
        if side == 'long':
            client.close_long(size, config.LEVERAGE)
        else:
            client.close_short(size, config.LEVERAGE)
        log_info(f"❌ {side.upper()} 仓止损平仓，收益率: {pnl_pct * 100:.2f}%, 盈亏金额: {profit_amount:.2f} USD")
    else:
        log_info(f"🔄 {side.upper()} 仓监控中，无平仓动作。当前收益率: {pnl_pct * 100:.2f}%, 当前盈亏: {profit_amount:.2f} USD")

# 多周期机器学习模型预测信号
def predict_signal(model):
    data_dict = client.fetch_data()
    merged_df = merge_multi_period_features(data_dict)

    # 核心变化在这里：实盘加载训练时保存的特征列
    feature_cols = joblib.load('models/feature_list.pkl')

    # 只取训练时使用过的特征列，保持和训练时完全一致
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)

    prob = model.predict_proba(X_live)[0]
    long_prob, short_prob = prob[1], prob[0]

    log_info(f"实时预测 - 多: {long_prob:.3f} 空: {short_prob:.3f}")

    return long_prob, short_prob

# 仓位动态调整核心逻辑
def adjust_position(model):
    long_prob, short_prob = predict_signal(model)
    account_balance = client.get_account_balance()
    total_balance = float(account_balance['data'][0]['totalEq'])

    side, current_size, entry_price = client.get_position()
    current_value = current_size * entry_price  # 当前持仓价值

    # 判断信号方向
    if long_prob > config.THRESHOLD_LONG:
        target_ratio = position_manager.calculate_target_ratio(long_prob)
        delta = position_manager.calculate_adjust_amount(total_balance, current_value, target_ratio)

        if delta > 0:
            client.open_long(delta, config.LEVERAGE)
            log_info(f"📈 动态加多仓: {delta} USD")
        elif delta < 0:
            client.close_long(abs(delta), config.LEVERAGE)
            log_info(f"📉 动态减多仓: {abs(delta)} USD")
        else:
            log_info("当前多仓已达目标仓位，无需调整。")

    elif short_prob > config.THRESHOLD_SHORT:
        target_ratio = position_manager.calculate_target_ratio(short_prob)
        delta = position_manager.calculate_adjust_amount(total_balance, current_value, target_ratio)

        if delta > 0:
            client.open_short(delta, config.LEVERAGE)
            log_info(f"📈 动态加空仓: {delta} USD")
        elif delta < 0:
            client.close_short(abs(delta), config.LEVERAGE)
            log_info(f"📉 动态减空仓: {abs(delta)} USD")
        else:
            log_info("当前空仓已达目标仓位，无需调整。")
    else:
        log_info("当前无明显信号，暂不调整仓位。")

# 主逻辑入口
def run():
    try:
        model = joblib.load(config.MODEL_PATH)

        # 风控模块仍然保留
        side, size, entry_price = client.get_position()
        if side != 'none':
            risk_control(side, entry_price, size)

        # 核心动态仓位管理模块
        adjust_position(model)

    except Exception as e:
        log_error(f"实盘运行异常: {e}")
        log_error(traceback.format_exc())

if __name__ == '__main__':
    run()
