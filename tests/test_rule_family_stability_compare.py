import argparse
import unittest
from unittest.mock import patch

import pandas as pd

from run import rule_family_stability_compare as compare


def make_args(**overrides):
    defaults = {
        "rows": 0,
        "windows": "",
        "families": "breakout_flow,failed_breakout",
        "directions": "short",
        "trend_gaps": "0.003",
        "regime_gap_multipliers": "1.0",
        "tp_sl_pairs": "0.012:0.010",
        "allow_high_vol_values": "0",
        "allow_range_values": "1",
        "pullback_pct_values": "0.003",
        "breakout_lookbacks": "12",
        "failed_breakout_reclaim_pct_values": "0",
        "flow_min_values": "1.2",
        "low_vol_max_values": "0.003",
        "volatility_min_values": "",
        "trend_gap_min_values": "",
        "max_candidates_per_family": 2,
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


class RuleFamilyStabilityCompareTests(unittest.TestCase):
    def test_build_family_sweep_args_uses_family_as_entry_filter(self):
        sweep_args = compare.build_family_sweep_args(
            make_args(
                flow_min_values="1.0,1.2",
                failed_breakout_reclaim_pct_values="0,0.001",
            ),
            "failed_breakout_flow",
        )

        self.assertEqual(sweep_args.entry_filters, "failed_breakout_flow")
        self.assertEqual(sweep_args.flow_min_values, "1.0,1.2")
        self.assertEqual(sweep_args.failed_breakout_reclaim_pct_values, "0,0.001")
        self.assertEqual(sweep_args.max_candidates, 2)

    def test_summarize_family_results_groups_and_ranks_best_candidate(self):
        base_candidate = {
            "target_direction_splits": {
                "validation": {"mean_net_return": 0.001, "profit_factor": 1.2, "candidate_rows": 12},
                "oos": {"mean_net_return": 0.001, "profit_factor": 1.2, "candidate_rows": 12},
            },
            "period_stability": {"monthly": {"positive_period_ratio": 0.5}},
        }
        results = [
            {
                **base_candidate,
                "name": "breakout",
                "rule_family": "breakout_flow",
                "stability_decision": {"status": "no_confirmed_edge"},
            },
            {
                **base_candidate,
                "name": "failed",
                "rule_family": "failed_breakout",
                "target_direction_splits": {
                    "validation": {"mean_net_return": 0.002, "profit_factor": 1.4, "candidate_rows": 12},
                    "oos": {"mean_net_return": 0.0015, "profit_factor": 1.3, "candidate_rows": 12},
                },
                "stability_decision": {"status": "stable_positive"},
            },
        ]

        summaries = compare.summarize_family_results(results)

        self.assertEqual(summaries[0]["family"], "failed_breakout")
        self.assertEqual(summaries[0]["stable_positive_count"], 1)
        self.assertEqual(summaries[0]["best_candidate"]["name"], "failed")

    def test_run_compare_builds_candidates_for_each_family_and_direction(self):
        feature_data = pd.DataFrame({"placeholder": [1, 2, 3]})
        built_candidates = {
            "breakout_flow": [{
                "name": "bo",
                "params": {},
                "entry_filter": {"name": "breakout_flow", "params": {}},
            }],
            "failed_breakout": [{
                "name": "fb",
                "params": {},
                "entry_filter": {"name": "failed_breakout", "params": {}},
            }],
        }

        def fake_build_candidates(sweep_args):
            return built_candidates[sweep_args.entry_filters]

        def fake_stability(candidate, feature_data_arg, args_arg, sweep_args_arg, label_cache):
            return {
                "name": candidate["name"],
                "target_direction": args_arg.direction,
                "entry_filter_summary": {"candidate_rows_after": 1, "candidate_rows_before": 2},
                "target_direction_splits": {
                    "validation": {"mean_net_return": 0.001, "profit_factor": 1.2, "candidate_rows": 10},
                    "oos": {"mean_net_return": 0.001, "profit_factor": 1.2, "candidate_rows": 10},
                },
                "period_stability": {"monthly": {"positive_period_ratio": 0.5}},
                "stability_decision": {"status": "stable_positive"},
            }

        with patch("run.rule_family_stability_compare.diag.load_feature_data", return_value=feature_data):
            with patch("run.rule_family_stability_compare.sweep.build_candidates", side_effect=fake_build_candidates):
                with patch("run.rule_family_stability_compare.stability.build_candidate_stability", side_effect=fake_stability):
                    report = compare.run_compare(make_args(directions="short,long"))

        self.assertEqual(report["candidate_count"], 4)
        self.assertEqual(report["status_counts"]["stable_positive"], 4)
        self.assertEqual({item["rule_family"] for item in report["all_candidates"]}, {"breakout_flow", "failed_breakout"})
        self.assertEqual({item["target_direction"] for item in report["all_candidates"]}, {"short", "long"})


if __name__ == "__main__":
    unittest.main()
