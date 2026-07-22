import unittest

import pandas as pd

from research.directional_v2_backtest import (
    evaluate_forward_result,
    run_directional_backtest,
)


def experiment_spec():
    return {
        "signal": {
            "minimum_direction_probability": 0.55,
            "minimum_direction_probability_gap": 0.10,
            "minimum_advantage_over_flat": 0.05,
            "hard_blocked_directions": [],
        },
        "execution": {
            "initial_balance": 1000.0,
            "position_notional_ratio": 0.15,
            "leverage": 1.0,
            "maximum_hold_bars": 2,
            "fee_rate_per_side": 0.0,
            "slippage_bps_per_side": 0.0,
            "take_profit_pct": 0.01,
            "stop_loss_pct": 0.01,
        },
        "holdout": {"minimum_closed_trades": 1},
        "evaluation": {
            "minimum_profit_factor": 1.2,
            "minimum_net_pnl_after_costs": 0.0,
            "maximum_drawdown_pct": -5.0,
            "minimum_positive_week_ratio": 0.6,
            "result_if_insufficient_sample": "WATCH",
            "result_if_any_decisive_gate_fails": "ELIMINATE",
            "result_if_all_gates_pass": "KEEP",
        },
    }


def market_data():
    return pd.DataFrame(
        [
            {"5m_open": 100.0, "5m_high": 100.2, "5m_low": 99.8, "5m_close": 100.0},
            {"5m_open": 100.0, "5m_high": 101.2, "5m_low": 99.8, "5m_close": 101.0},
            {"5m_open": 101.0, "5m_high": 101.2, "5m_low": 100.8, "5m_close": 101.0},
        ],
        index=pd.date_range("2026-01-05", periods=3, freq="5min", tz="UTC"),
    )


class DirectionalV2BacktestTests(unittest.TestCase):
    def test_long_signal_executes_on_next_open_and_takes_profit(self):
        data = market_data()
        probabilities = pd.DataFrame(
            [
                {"flat": 0.1, "long": 0.8, "short": 0.1},
                {"flat": 1.0, "long": 0.0, "short": 0.0},
                {"flat": 1.0, "long": 0.0, "short": 0.0},
            ],
            index=data.index,
        )

        summary = run_directional_backtest(data, probabilities, experiment_spec())

        self.assertEqual(summary["closed_trade_count"], 1)
        self.assertEqual(summary["trades"][0]["direction"], "long")
        self.assertEqual(summary["trades"][0]["reason"], "TP")
        self.assertGreater(summary["net_pnl_after_costs"], 0)

    def test_flat_probabilities_do_not_trade(self):
        data = market_data()
        probabilities = pd.DataFrame(
            [{"flat": 1.0, "long": 0.0, "short": 0.0}] * len(data),
            index=data.index,
        )

        summary = run_directional_backtest(data, probabilities, experiment_spec())

        self.assertEqual(summary["closed_trade_count"], 0)
        self.assertEqual(summary["net_pnl_after_costs"], 0.0)

    def test_forward_gate_waits_for_sample(self):
        spec = experiment_spec()
        spec["holdout"]["minimum_closed_trades"] = 30
        summary = {
            "closed_trade_count": 2,
            "net_pnl_after_costs": -1.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": -1.0,
            "positive_week_ratio": 0.0,
        }

        decision = evaluate_forward_result(summary, summary, spec)

        self.assertEqual(decision["verdict"], "WATCH")

    def test_forward_gate_keeps_only_profitable_robust_result(self):
        spec = experiment_spec()
        summary = {
            "closed_trade_count": 10,
            "net_pnl_after_costs": 20.0,
            "profit_factor": 1.5,
            "max_drawdown_pct": -2.0,
            "positive_week_ratio": 0.75,
        }
        baseline = {"net_pnl_after_costs": 5.0}

        decision = evaluate_forward_result(summary, baseline, spec)

        self.assertEqual(decision["verdict"], "KEEP")
        self.assertEqual(decision["failed"], [])


if __name__ == "__main__":
    unittest.main()
