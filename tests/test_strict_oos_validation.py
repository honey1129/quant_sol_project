import unittest

from backtest.backtest import Backtester
from run import retrain_models
from run import strict_oos_validation


def performance_summary(
    *,
    closed=30,
    net=120.0,
    profit_factor=1.5,
    drawdown=-2.0,
    positive_fold_ratio=0.75,
    baseline_net=40.0,
):
    wins = max(1, closed // 2)
    losses = max(0, closed - wins)
    return {
        "closed_trade_count": closed,
        "net_pnl_after_costs": net,
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown,
        "positive_fold_ratio": positive_fold_ratio,
        "trend_baseline": {"net_pnl_after_costs": baseline_net},
        "closed_trade_attribution": {
            "by_direction": {
                "long": {
                    "closed_trade_count": closed,
                    "winning_trade_count": wins,
                    "losing_trade_count": losses,
                    "gross_profit": 180.0,
                    "gross_loss": 90.0,
                    "net_pnl_after_costs": net,
                    "profit_factor": 2.0,
                    "win_rate_pct": wins / closed * 100.0,
                }
            },
            "by_entry_regime": {},
        },
    }


class StrictOosValidationTests(unittest.TestCase):
    def test_parse_positive_ints_sorts_and_deduplicates(self):
        self.assertEqual(
            strict_oos_validation.parse_positive_ints("90,30,60,30"),
            [30, 60, 90],
        )

    def test_insufficient_sample_is_watch_not_eliminate(self):
        windows = {
            "30": performance_summary(closed=2),
            "60": performance_summary(closed=4),
            "90": performance_summary(closed=6),
        }

        decision = strict_oos_validation.evaluate_strategy(
            windows,
            min_closed_trades=30,
        )

        self.assertEqual(decision["verdict"], "WATCH")
        self.assertEqual(decision["reason"], "insufficient_oos_closed_trades")

    def test_sufficient_but_unprofitable_sample_is_eliminated(self):
        windows = {
            "30": performance_summary(closed=10, net=-10.0, profit_factor=0.8),
            "60": performance_summary(closed=20, net=-20.0, profit_factor=0.7),
            "90": performance_summary(closed=30, net=-30.0, profit_factor=0.6),
        }

        decision = strict_oos_validation.evaluate_strategy(windows)

        self.assertEqual(decision["verdict"], "ELIMINATE")
        self.assertGreater(decision["failed_gate_count"], 0)

    def test_sufficient_long_window_can_eliminate_when_recent_sample_is_small(self):
        windows = {
            "30": performance_summary(closed=2, net=-2.0, profit_factor=0.5),
            "60": performance_summary(closed=20, net=-20.0, profit_factor=0.7),
            "90": performance_summary(closed=30, net=-30.0, profit_factor=0.6),
        }

        decision = strict_oos_validation.evaluate_strategy(windows)

        self.assertEqual(decision["verdict"], "ELIMINATE")
        self.assertTrue(any(
            gate["scope"] == "90d" and gate["decisive"] and not gate["passed"]
            for gate in decision["gates"]
        ))

    def test_robust_strategy_is_kept(self):
        windows = {
            "30": performance_summary(closed=10, net=50.0),
            "60": performance_summary(closed=20, net=90.0),
            "90": performance_summary(closed=30, net=130.0),
        }

        decision = strict_oos_validation.evaluate_strategy(windows)

        self.assertEqual(decision["verdict"], "KEEP")
        self.assertEqual(decision["failed_gate_count"], 0)

    def test_losing_mature_direction_rejects_strategy(self):
        windows = {
            "30": performance_summary(closed=10),
            "60": performance_summary(closed=20),
            "90": performance_summary(closed=30),
        }
        windows["90"]["closed_trade_attribution"]["by_direction"]["short"] = {
            "closed_trade_count": 10,
            "winning_trade_count": 2,
            "losing_trade_count": 8,
            "gross_profit": 20.0,
            "gross_loss": 80.0,
            "net_pnl_after_costs": -60.0,
            "profit_factor": 0.25,
            "win_rate_pct": 20.0,
        }

        decision = strict_oos_validation.evaluate_strategy(windows)

        self.assertEqual(decision["verdict"], "ELIMINATE")
        self.assertTrue(any(
            gate["scope"] == "90d:by_direction:short" and not gate["passed"]
            for gate in decision["gates"]
        ))

    def test_closed_trade_attribution_uses_net_pnl(self):
        grouped = Backtester._summarize_closed_trade_group([
            {"direction": "long", "net_pnl_after_costs": 10.0},
            {"direction": "long", "net_pnl_after_costs": -4.0},
            {"direction": "short", "net_pnl_after_costs": 2.0},
        ], "direction")

        self.assertEqual(grouped["long"]["closed_trade_count"], 2)
        self.assertAlmostEqual(grouped["long"]["net_pnl_after_costs"], 6.0)
        self.assertAlmostEqual(grouped["long"]["profit_factor"], 2.5)
        self.assertAlmostEqual(grouped["short"]["win_rate_pct"], 100.0)

    def test_walk_forward_aggregation_combines_trade_attribution(self):
        base = {
            "max_drawdown_pct": -1.0,
            "trade_count": 2,
            "closed_trade_count": 1,
            "winning_trade_count": 1,
            "losing_trade_count": 0,
            "gross_profit": 5.0,
            "gross_loss": 0.0,
            "net_pnl_after_costs": 5.0,
            "closed_trade_attribution": {
                "by_direction": {
                    "long": {
                        "closed_trade_count": 1,
                        "winning_trade_count": 1,
                        "losing_trade_count": 0,
                        "gross_profit": 5.0,
                        "gross_loss": 0.0,
                        "net_pnl_after_costs": 5.0,
                    }
                }
            },
        }

        summary = retrain_models.aggregate_backtest_summaries([base, base])
        long_stats = summary["closed_trade_attribution"]["by_direction"]["long"]

        self.assertEqual(long_stats["closed_trade_count"], 2)
        self.assertAlmostEqual(long_stats["net_pnl_after_costs"], 10.0)
        self.assertEqual(long_stats["profit_factor"], float("inf"))


if __name__ == "__main__":
    unittest.main()
