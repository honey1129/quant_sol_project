import unittest

import pandas as pd

from run import rule_edge_diagnostics as diag
from train import train as train_module


class RuleEdgeDiagnosticsTests(unittest.TestCase):
    def test_summarize_group_classifies_positive_and_negative_edge(self):
        index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        group = pd.DataFrame({
            "label_direction": ["long"] * 4 + ["short"] * 4,
            "label_regime": ["trend_long"] * 4 + ["trend_short"] * 4,
            "label_outcome": ["TP", "TP", "SL", "TIMEOUT_WEAK_POSITIVE", "SL", "SL", "TP", "TIMEOUT_WEAK_NEGATIVE"],
            "label_reject_reason": ["accepted", "accepted", "outcome_sl", "accepted", "outcome_sl", "outcome_sl", "accepted", "timeout_weak_negative_net_return"],
            "label_net_return": [0.014, 0.014, -0.016, 0.003, -0.016, -0.016, 0.014, -0.004],
            "label_gross_return": [0.016, 0.016, -0.014, 0.005, -0.014, -0.014, 0.016, -0.002],
            "label_mfe": [0.016] * 8,
            "label_mae": [0.002] * 8,
            "label_mae_ratio": [0.1] * 8,
            "label_mfe_mae_ratio": [8.0] * 8,
        }, index=index)
        strict = pd.DataFrame({
            "target": [
                train_module.TARGET_TRADE,
                train_module.TARGET_TRADE,
                train_module.TARGET_NO_TRADE,
                train_module.TARGET_NO_TRADE,
                train_module.TARGET_NO_TRADE,
                train_module.TARGET_NO_TRADE,
                train_module.TARGET_TRADE,
                train_module.TARGET_NO_TRADE,
            ],
        }, index=index)

        long_summary = diag.summarize_group(
            group.iloc[:4],
            strict,
            min_rows=3,
            min_profit_factor=1.05,
            min_mean_net_return=0.0,
        )
        short_summary = diag.summarize_group(
            group.iloc[4:],
            strict,
            min_rows=3,
            min_profit_factor=1.05,
            min_mean_net_return=0.0,
        )

        self.assertEqual(long_summary["recommendation"]["status"], "positive_edge")
        self.assertEqual(short_summary["recommendation"]["status"], "no_edge")
        self.assertGreater(long_summary["mean_net_return"], 0.0)
        self.assertLess(short_summary["mean_net_return"], 0.0)
        self.assertEqual(long_summary["quality_label_summary"]["strict_trade_rows"], 2)

    def test_final_decision_requires_validation_and_oos_confirmation(self):
        report = {
            "splits": {
                "validation": {
                    "by_direction": {
                        "long": {"recommendation": {"status": "no_edge"}},
                        "short": {"recommendation": {"status": "positive_edge"}},
                    },
                },
                "oos": {
                    "by_direction": {
                        "long": {"recommendation": {"status": "no_edge"}},
                        "short": {"recommendation": {"status": "positive_edge"}},
                    },
                },
            },
        }

        decision = diag.final_decision(report)

        self.assertEqual(decision["overall_action"], "test_short_only_ml_quality_filter")
        self.assertEqual(decision["directions"]["long"]["action"], "disable_direction_or_rework_base_rule_before_ml")
        self.assertEqual(decision["directions"]["short"]["action"], "keep_direction_and_train_quality_filter")


if __name__ == "__main__":
    unittest.main()
