import math
from statistics import mean

from config import config


def get_configured_reward_risk():
    reward_risk = float(getattr(config, "KELLY_REWARD_RISK", 1.0))
    if not math.isfinite(reward_risk) or reward_risk <= 0:
        raise ValueError(f"KELLY_REWARD_RISK 必须为正数: {reward_risk}")
    return reward_risk


class RewardRiskEstimator:
    def __init__(self, min_trades=12, default_rr=1.8):
        self.min_trades = min_trades
        self.default_rr = default_rr
        self.trades = []

    def batch_update(self, trades):
        self.trades = trades[-100:]  # 只看最近 100 笔

    def estimate(self):
        if len(self.trades) < self.min_trades:
            return self.default_rr

        wins = [r for r in self.trades if r > 0]
        losses = [-r for r in self.trades if r < 0]

        if not wins or not losses:
            return self.default_rr

        avg_win = mean(wins)
        avg_loss = mean(losses)

        rr = avg_win / avg_loss
        return max(0.8, min(rr, 3.5))
