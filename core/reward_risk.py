# reward_risk.py

import numpy as np

class RewardRiskEstimator:
    def __init__(self, min_trades=20, default_rr=1.0):
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

        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        rr = avg_win / avg_loss
        return max(0.3, min(rr, 3.0))
