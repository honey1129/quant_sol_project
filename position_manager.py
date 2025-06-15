# position_manager.py

import config

class PositionManager:
    def __init__(self):
        self.min_ratio = config.POSITION_MIN
        self.max_ratio = config.POSITION_MAX
        self.adjust_unit = config.ADJUST_UNIT  # 每次加减仓单位（USDT）

    def calculate_target_ratio(self, signal_prob):
        """
        将模型的预测概率映射为目标仓位比例（0 ~ 1）
        """
        signal_strength = max(0, signal_prob - 0.5) * 2  # 映射 0 ~ 1
        target_ratio = self.min_ratio + signal_strength * (self.max_ratio - self.min_ratio)
        return round(target_ratio, 4)

    def calculate_adjust_amount(self, account_balance, current_position_value, target_ratio):
        """
        根据账户余额、当前持仓金额、目标仓位比例，计算加减仓金额
        """
        target_amount = account_balance * config.LEVERAGE * target_ratio
        delta = target_amount - current_position_value

        # 按照调仓单位调整，防止频繁小幅调整
        if abs(delta) < self.adjust_unit:
            return 0
        else:
            return round(delta, 2)
