import argparse
import unittest

import pandas as pd

from run import rule_breakout_flow_stability as stability


def make_args(**overrides):
    defaults = {
        "rows": 0,
        "direction": "short",
        "trend_gaps": "0.003",
        "regime_gap_multipliers": "1.0",
        "tp_sl_pairs": "0.012:0.010",
        "allow_high_vol_values": "0",
        "allow_range_values": "1",
        "breakout_lookbacks": "24",
        "flow_min_values": "1.2",
        "max_candidates": 0,
        "min_split_rows": 10,
        "min_period_rows": 2,
        "min_active_periods": 2,
        "min_profit_factor": 1.05,
        "min_mean_net_return": 0.0,
        "min_positive_period_ratio": 0.5,
        "top_n": 3,
        "output": None,
        "progress": False,
        "verbose_candidates": False,
        "print_json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class RuleBreakoutFlowStabilityTests(unittest.TestCase):
    def test_build_sweep_args_forces_breakout_flow(self):
        sweep_args = stability.build_sweep_args(make_args(
            breakout_lookbacks="12,24",
            flow_min_values="1.0,1.2",
            max_candidates=8,
        ))

        self.assertEqual(sweep_args.entry_filters, "breakout_flow")
        self.assertEqual(sweep_args.breakout_lookbacks, "12,24")
        self.assertEqual(sweep_args.flow_min_values, "1.0,1.2")
        self.assertEqual(sweep_args.max_candidates, 8)

    def test_summarize_period_buckets_counts_positive_periods(self):
        index = pd.to_datetime([
            "2026-01-01 00:00",
            "2026-01-02 00:00",
            "2026-02-01 00:00",
            "2026-02-02 00:00",
            "2026-03-01 00:00",
            "2026-03-02 00:00",
        ], utc=True)
        data = pd.DataFrame({
            "label_direction": ["short"] * 6,
            "label_regime": ["trend_short"] * 6,
            "label_outcome": ["TP", "TIMEOUT_WEAK_POSITIVE", "SL", "SL", "TP", "SL"],
            "label_reject_reason": ["accepted"] * 6,
            "label_net_return": [0.010, 0.002, -0.006, -0.004, 0.008, -0.001],
            "label_gross_return": [0.012, 0.004, -0.004, -0.002, 0.010, 0.001],
            "label_mfe": [0.012] * 6,
            "label_mae": [0.002] * 6,
            "label_mae_ratio": [0.2] * 6,
            "label_mfe_mae_ratio": [6.0] * 6,
            "diagnostic_split": ["train", "train", "validation", "validation", "oos", "oos"],
        }, index=index)

        summary = stability.summarize_period_buckets(
            data,
            "short",
            freq="M",
            min_rows=2,
            min_profit_factor=1.05,
            min_mean_net_return=0.0,
        )

        self.assertEqual(summary["candidate_period_count"], 3)
        self.assertEqual(summary["covered_period_count"], 3)
        self.assertEqual(summary["positive_period_count"], 2)
        self.assertAlmostEqual(summary["positive_period_ratio"], 2 / 3)
        self.assertEqual(summary["worst_period"]["period"], "2026-02")

    def test_classify_stability_marks_oos_only_unconfirmed(self):
        args = make_args(min_split_rows=10, min_active_periods=2, min_positive_period_ratio=0.5)
        split_metrics = {
            "validation": {
                "candidate_rows": 12,
                "mean_net_return": -0.0001,
                "profit_factor": 0.95,
            },
            "oos": {
                "candidate_rows": 20,
                "mean_net_return": 0.002,
                "profit_factor": 1.4,
            },
        }
        period_stability = {
            "monthly": {
                "covered_period_count": 3,
                "positive_period_ratio": 0.67,
            },
        }

        decision = stability.classify_stability(split_metrics, period_stability, args)

        self.assertEqual(decision["status"], "oos_only_unconfirmed")
        self.assertTrue(decision["oos_pass"])
        self.assertFalse(decision["validation_pass"])

    def test_classify_stability_marks_stable_positive(self):
        args = make_args(min_split_rows=10, min_active_periods=2, min_positive_period_ratio=0.5)
        split_metrics = {
            "validation": {
                "candidate_rows": 12,
                "mean_net_return": 0.0005,
                "profit_factor": 1.1,
            },
            "oos": {
                "candidate_rows": 20,
                "mean_net_return": 0.002,
                "profit_factor": 1.4,
            },
        }
        period_stability = {
            "monthly": {
                "covered_period_count": 3,
                "positive_period_ratio": 0.67,
            },
        }

        decision = stability.classify_stability(split_metrics, period_stability, args)

        self.assertEqual(decision["status"], "stable_positive")
        self.assertEqual(decision["action"], "eligible_for_short_only_paper_rule_test")


if __name__ == "__main__":
    unittest.main()
