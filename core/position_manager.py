# position_manager.py

import math

from config import config

class PositionManager:
    def __init__(self, min_ratio=None, max_ratio=None):
        self.min_ratio = float(config.POSITION_MIN if min_ratio is None else min_ratio)
        self.max_ratio = float(config.POSITION_MAX if max_ratio is None else max_ratio)
        self.adjust_unit = config.ADJUST_UNIT

    def set_bounds(self, *, min_ratio=None, max_ratio=None):
        if min_ratio is not None:
            self.min_ratio = float(min_ratio)
        if max_ratio is not None:
            self.max_ratio = max(float(max_ratio), self.min_ratio)

    # Kelly公式计算
    def kelly_fraction(self, prob, reward_risk):
        if reward_risk <= 0:
            return 0.0
        kelly = ((prob * (reward_risk + 1)) - 1) / reward_risk
        return max(0, min(kelly, 0.75))

    # 波动率动态调整账户余额
    def volatility_adjust_balance(self, total_balance, volatility):
        target_vol = config.TARGET_VOL
        adjust_factor = target_vol / (volatility + 1e-6)
        adjust_factor = min(1.5, max(0.5, adjust_factor))
        return total_balance * adjust_factor

    # 多因子评分 (可扩展因子体系)
    def multi_factor_score(self, prob, money_flow_ratio, volatility):
        try:
            money_flow_ratio = float(money_flow_ratio)
        except (TypeError, ValueError):
            money_flow_ratio = 1.0
        try:
            volatility = float(volatility)
        except (TypeError, ValueError):
            volatility = config.TARGET_VOL

        if not math.isfinite(money_flow_ratio):
            money_flow_ratio = 1.0
        if not math.isfinite(volatility) or volatility <= 0:
            volatility = config.TARGET_VOL

        # 把资金流限制在稳健区间，避免极端缩量/放量把仓位评分顶满。
        money_flow_clamped = min(1.5, max(0.5, money_flow_ratio))
        money_flow_score = (money_flow_clamped - 0.5) / 1.0

        # 低波动不再无限放大仓位，最高只给满分 1.0。
        volatility_score = min(1.0, config.TARGET_VOL / max(volatility, 1e-6))

        score = (
            0.5 * prob +
            0.25 * money_flow_score +
            0.25 * volatility_score
        )
        return max(0, min(score, 1))

    # 最终目标仓位比例
    def calculate_target_ratio(self, prob, money_flow_ratio, volatility,reward_risk=1.0):
        signal_strength = max(0, prob - 0.5) * 2
        kelly_weight = self.kelly_fraction(prob,reward_risk)
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
            delta_rounded = round(delta / self.adjust_unit) * self.adjust_unit
            return round(delta_rounded, 2)
