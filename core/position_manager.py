from config import config

class PositionManager:
    def __init__(self):
        self.min_ratio = config.POSITION_MIN
        self.max_ratio = config.POSITION_MAX
        self.adjust_unit = config.ADJUST_UNIT

    # Kelly公式计算
    def kelly_fraction(self, prob, reward_risk=2.5):
        kelly = ((prob * (reward_risk + 1)) - 1) / reward_risk
        return max(0, min(kelly, 1))

    # 波动率动态调整账户余额
    def volatility_adjust_balance(self, total_balance, volatility):
        target_vol = config.TARGET_VOL
        adjust_factor = target_vol / (volatility + 1e-6)
        adjust_factor = min(1.5, max(0.5, adjust_factor))
        return total_balance * adjust_factor

    # 多因子评分 (可扩展因子体系)
    def multi_factor_score(self, prob, money_flow_ratio, volatility):
        score = (
            0.5 * prob +
            0.3 * (money_flow_ratio / 5) +  # money_flow_ratio 通常在 0-5之间
            0.2 * (0.02 / (volatility + 1e-6))
        )
        return max(0, min(score, 1))

    # 最终目标仓位比例
    def calculate_target_ratio(self, prob, money_flow_ratio, volatility):
        signal_strength = max(0, prob - 0.5) * 2
        kelly_weight = self.kelly_fraction(prob)
        multi_factor = self.multi_factor_score(prob, money_flow_ratio, volatility)
        blended_ratio = self.min_ratio + signal_strength * (self.max_ratio - self.min_ratio)
        final_ratio = blended_ratio * kelly_weight * multi_factor
        return round(final_ratio, 4)

    # 实际调仓金额（按最小调整单位控制）
    def calculate_adjust_amount(self, account_balance, current_position_value, target_ratio):
        target_amount = account_balance * target_ratio
        delta = target_amount - current_position_value

        if abs(delta) < self.adjust_unit:
            return 0
        else:
            return round(delta, 2)
