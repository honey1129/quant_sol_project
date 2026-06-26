import argparse
import unittest
from unittest.mock import patch

import pandas as pd

from run import rule_edge_sweep as sweep


def make_args(**overrides):
    defaults = {
        "rows": 0,
        "trend_gaps": "0.002,0.003",
        "regime_gap_multipliers": "1.0,1.5",
        "tp_sl_pairs": "0.012:0.010,0.016:0.014",
        "allow_high_vol_values": "0,1",
        "allow_range_values": "1",
        "entry_filters": "none",
        "pullback_pct_values": "0.003,0.006",
        "breakout_lookbacks": "12,24",
        "flow_min_values": "1.0,1.2",
        "low_vol_max_values": "0.003,0.005",
        "max_candidates": 0,
        "min_rows": 2,
        "min_profit_factor": 1.05,
        "min_mean_net_return": 0.0,
        "top_n": 3,
        "output": None,
        "progress": False,
        "verbose_candidates": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class RuleEdgeSweepTests(unittest.TestCase):
    def test_build_candidates_expands_parameter_grid(self):
        candidates = sweep.build_candidates(make_args())

        self.assertEqual(len(candidates), 16)
        self.assertEqual(candidates[0]["params"]["TREND_FILTER_MIN_GAP"], 0.002)
        self.assertIn("MODEL_LABEL_TAKE_PROFIT", candidates[0]["params"])
        self.assertEqual(candidates[0]["entry_filter"]["name"], "none")

    def test_build_candidates_expands_entry_filter_grid(self):
        candidates = sweep.build_candidates(make_args(
            trend_gaps="0.002",
            regime_gap_multipliers="1.0",
            tp_sl_pairs="0.016:0.014",
            allow_high_vol_values="0",
            allow_range_values="1",
            entry_filters="none,pullback,flow,pullback_flow",
            pullback_pct_values="0.003,0.006",
            flow_min_values="1.0,1.2",
        ))

        self.assertEqual(len(candidates), 9)
        names = {candidate["entry_filter"]["name"] for candidate in candidates}
        self.assertEqual(names, {"none", "pullback", "flow", "pullback_flow"})
        self.assertTrue(any("efpullback_flow" in candidate["name"] for candidate in candidates))

    def test_max_candidates_preserves_entry_filter_coverage(self):
        candidates = sweep.build_candidates(make_args(
            trend_gaps="0.002,0.003",
            regime_gap_multipliers="1.0,1.5",
            tp_sl_pairs="0.016:0.014",
            allow_high_vol_values="0,1",
            allow_range_values="1",
            entry_filters="none,pullback,breakout,flow",
            pullback_pct_values="0.003",
            breakout_lookbacks="12",
            flow_min_values="1.0",
            max_candidates=4,
        ))

        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            {candidate["entry_filter"]["name"] for candidate in candidates},
            {"none", "pullback", "breakout", "flow"},
        )

    def test_apply_entry_filter_blocks_failed_pullback_and_flow_rows(self):
        index = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
        labeled = pd.DataFrame({
            "label_direction": ["long", "long", "short", "short", "none"],
            "target": [1, 1, 1, 1, 0],
            "label_reject_reason": ["accepted"] * 5,
            "5m_close": [100.2, 101.0, 99.8, 98.0, 100.0],
            "5m_ema_20": [100.0] * 5,
            "money_flow_ratio": [1.1, 1.3, 1.2, 1.4, 0.8],
            "volume_ratio": [0.8, 0.8, 0.8, 0.8, 0.8],
            "volatility_15": [0.002] * 5,
        }, index=index)

        filtered, pass_mask = sweep.apply_entry_filter(labeled, {
            "name": "pullback_flow",
            "params": {
                "pullback_pct": 0.003,
                "flow_min": 1.2,
            },
        })

        self.assertFalse(pass_mask.iloc[0])
        self.assertFalse(pass_mask.iloc[1])
        self.assertTrue(pass_mask.iloc[2])
        self.assertFalse(pass_mask.iloc[3])
        self.assertEqual(filtered["label_direction"].tolist(), ["none", "none", "short", "none", "none"])
        self.assertEqual(filtered["target"].tolist(), [0, 0, 1, 0, 0])

    def test_apply_entry_filter_supports_breakout_by_direction(self):
        index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
        labeled = pd.DataFrame({
            "label_direction": ["none", "long", "long", "none", "short", "short"],
            "5m_close": [100.0, 101.0, 100.5, 99.0, 98.0, 99.0],
            "5m_high": [100.2, 100.8, 101.2, 100.7, 98.8, 99.5],
            "5m_low": [99.8, 100.5, 100.1, 98.8, 98.5, 98.7],
        }, index=index)

        mask = sweep.entry_filter_mask(labeled, {
            "name": "breakout",
            "params": {"breakout_lookback": 1},
        })

        self.assertTrue(mask.iloc[1])
        self.assertFalse(mask.iloc[2])
        self.assertTrue(mask.iloc[4])
        self.assertFalse(mask.iloc[5])

    def test_summarize_candidate_result_marks_passing_direction(self):
        report = {
            "splits": {
                "validation": {
                    "by_direction": {
                        "long": {
                            "recommendation": {"status": "positive_edge", "action": "ok"},
                            "candidate_rows": 10,
                            "mean_net_return": 0.002,
                            "profit_factor": 1.4,
                            "net_win_rate": 0.6,
                            "tp_rate": 0.3,
                            "sl_rate": 0.1,
                            "timeout_rate": 0.6,
                            "sum_net_return": 0.02,
                        },
                        "short": {
                            "recommendation": {"status": "no_edge", "action": "disable"},
                            "candidate_rows": 10,
                            "mean_net_return": -0.001,
                            "profit_factor": 0.8,
                        },
                    },
                },
                "oos": {
                    "by_direction": {
                        "long": {
                            "recommendation": {"status": "positive_edge", "action": "ok"},
                            "candidate_rows": 8,
                            "mean_net_return": 0.001,
                            "profit_factor": 1.2,
                            "net_win_rate": 0.5,
                            "tp_rate": 0.25,
                            "sl_rate": 0.1,
                            "timeout_rate": 0.65,
                            "sum_net_return": 0.008,
                        },
                        "short": {
                            "recommendation": {"status": "no_edge", "action": "disable"},
                            "candidate_rows": 8,
                            "mean_net_return": -0.002,
                            "profit_factor": 0.7,
                        },
                    },
                },
            },
        }
        candidate = {
            "name": "demo",
            "params": {
                "TREND_FILTER_MIN_GAP": 0.002,
                "REGIME_TREND_GAP_THRESHOLD": 0.002,
                "REGIME_HIGH_VOL_ALLOW_TRADES": False,
                "REGIME_RANGE_ALLOW_TRADES": True,
                "MODEL_LABEL_TAKE_PROFIT": 0.016,
                "MODEL_LABEL_STOP_LOSS": 0.014,
            },
        }

        result = sweep.summarize_candidate_result(candidate, report)

        self.assertTrue(result["passed"])
        self.assertEqual(result["passing_directions"], ["long"])
        self.assertEqual(result["best_direction"], "long")
        self.assertAlmostEqual(result["best_score"]["worst_mean_net_return"], 0.001)
        self.assertEqual(result["entry_filter"]["name"], "none")

    def test_run_sweep_ranks_positive_candidates(self):
        feature_data = pd.DataFrame({"placeholder": [1, 2, 3]})
        positive_report = {
            "splits": {
                "validation": {
                    "by_direction": {
                        "long": {
                            "recommendation": {"status": "positive_edge", "action": "ok"},
                            "candidate_rows": 10,
                            "mean_net_return": 0.002,
                            "profit_factor": 1.4,
                        },
                        "short": {
                            "recommendation": {"status": "no_edge", "action": "disable"},
                            "candidate_rows": 10,
                            "mean_net_return": -0.001,
                            "profit_factor": 0.8,
                        },
                    },
                },
                "oos": {
                    "by_direction": {
                        "long": {
                            "recommendation": {"status": "positive_edge", "action": "ok"},
                            "candidate_rows": 8,
                            "mean_net_return": 0.001,
                            "profit_factor": 1.2,
                        },
                        "short": {
                            "recommendation": {"status": "no_edge", "action": "disable"},
                            "candidate_rows": 8,
                            "mean_net_return": -0.002,
                            "profit_factor": 0.7,
                        },
                    },
                },
            },
            "decision": {},
        }

        with patch("run.rule_edge_sweep.diag.load_feature_data", return_value=feature_data):
            with patch("run.rule_edge_sweep.build_edge_labeled", return_value=(feature_data, {})):
                with patch("run.rule_edge_sweep.build_candidate_report", return_value=positive_report):
                    report = sweep.run_sweep(make_args(
                        trend_gaps="0.002",
                        regime_gap_multipliers="1.0",
                        tp_sl_pairs="0.016:0.014",
                        allow_high_vol_values="0",
                        allow_range_values="1",
                    ))

        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["positive_candidate_count"], 1)
        self.assertEqual(report["positive_candidates"][0]["passing_directions"], ["long"])


if __name__ == "__main__":
    unittest.main()
