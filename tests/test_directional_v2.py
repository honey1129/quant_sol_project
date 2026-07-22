import unittest
from datetime import datetime, timezone

import pandas as pd

from research import directional_v2


def label_spec(**overrides):
    spec = {
        "lookahead_bars": 1,
        "take_profit_pct": 0.01,
        "stop_loss_pct": 0.01,
        "round_trip_fee_rate": 0.001,
        "round_trip_slippage_rate": 0.0,
        "minimum_net_return": 0.002,
        "mae_penalty": 0.0,
        "minimum_direction_score_gap": 0.002,
    }
    spec.update(overrides)
    return spec


def market_frame(rows):
    return pd.DataFrame(
        rows,
        index=pd.date_range("2026-01-01", periods=len(rows), freq="5min", tz="UTC"),
    )


class DirectionalV2Tests(unittest.TestCase):
    def test_frozen_spec_hash_matches(self):
        digest = directional_v2.verify_frozen_spec()
        self.assertEqual(len(digest), 64)

    def test_labels_use_next_bar_open_and_choose_long(self):
        data = market_frame([
            {"5m_open": 50.0, "5m_high": 51.0, "5m_low": 49.0, "5m_close": 50.0},
            {"5m_open": 100.0, "5m_high": 102.0, "5m_low": 99.5, "5m_close": 101.0},
        ])

        labeled = directional_v2.build_directional_labels(
            data,
            {"label": label_spec()},
        )

        self.assertEqual(len(labeled), 1)
        self.assertEqual(labeled.iloc[0]["label_v2_entry_price"], 100.0)
        self.assertEqual(labeled.iloc[0]["target_v2"], directional_v2.TARGET_LONG)
        self.assertEqual(labeled.iloc[0]["label_v2_direction"], "long")

    def test_labels_choose_short_without_trend_gate(self):
        data = market_frame([
            {"5m_open": 100.0, "5m_high": 101.0, "5m_low": 99.0, "5m_close": 100.0},
            {"5m_open": 100.0, "5m_high": 100.5, "5m_low": 98.0, "5m_close": 99.0},
        ])

        labeled = directional_v2.build_directional_labels(
            data,
            {"label": label_spec()},
        )

        self.assertEqual(labeled.iloc[0]["target_v2"], directional_v2.TARGET_SHORT)
        self.assertEqual(labeled.iloc[0]["label_v2_direction"], "short")

    def test_same_bar_take_profit_and_stop_loss_uses_stop_first(self):
        future = market_frame([
            {"5m_open": 100.0, "5m_high": 102.0, "5m_low": 98.0, "5m_close": 100.0},
        ])

        quality = directional_v2.simulate_direction_quality(
            100.0,
            future,
            "long",
            label_spec(),
        )

        self.assertEqual(quality["outcome"], "SL")
        self.assertLess(quality["net_return"], 0)

    def test_ambiguous_profitable_directions_become_flat(self):
        spec = label_spec(
            take_profit_pct=0.01,
            stop_loss_pct=0.02,
            round_trip_fee_rate=0.0,
            minimum_net_return=0.0,
            minimum_direction_score_gap=0.002,
        )
        future = market_frame([
            {"5m_open": 100.0, "5m_high": 101.2, "5m_low": 99.5, "5m_close": 101.0},
            {"5m_open": 101.0, "5m_high": 101.0, "5m_low": 98.8, "5m_close": 99.0},
        ])
        long_quality = directional_v2.simulate_direction_quality(100.0, future, "long", spec)
        short_quality = directional_v2.simulate_direction_quality(100.0, future, "short", spec)

        target, direction, reason = directional_v2.choose_directional_target(
            long_quality,
            short_quality,
            spec,
        )

        self.assertEqual(target, directional_v2.TARGET_FLAT)
        self.assertEqual(direction, "flat")
        self.assertEqual(reason, "ambiguous_direction")

    def test_signal_requires_probability_and_flat_advantage(self):
        signal_spec = {
            "minimum_direction_probability": 0.55,
            "minimum_direction_probability_gap": 0.10,
            "minimum_advantage_over_flat": 0.05,
            "hard_blocked_directions": [],
        }

        accepted = directional_v2.select_directional_signal(
            {"flat": 0.20, "long": 0.65, "short": 0.15},
            signal_spec,
        )
        rejected = directional_v2.select_directional_signal(
            {"flat": 0.44, "long": 0.50, "short": 0.06},
            signal_spec,
        )

        self.assertEqual(accepted["direction"], "long")
        self.assertEqual(accepted["reason"], "accepted")
        self.assertEqual(rejected["direction"], "flat")
        self.assertEqual(rejected["reason"], "probability_below_minimum")

    def test_holdout_status_never_allows_early_final_evaluation(self):
        spec = directional_v2.load_experiment_spec()
        collecting = directional_v2.forward_holdout_status(
            spec,
            now=datetime(2026, 8, 1, tzinfo=timezone.utc),
            closed_trades=100,
        )
        insufficient = directional_v2.forward_holdout_status(
            spec,
            now=datetime(2026, 8, 23, tzinfo=timezone.utc),
            closed_trades=29,
        )
        ready = directional_v2.forward_holdout_status(
            spec,
            now=datetime(2026, 8, 23, tzinfo=timezone.utc),
            closed_trades=30,
        )

        self.assertEqual(collecting["state"], "COLLECTING_FORWARD_DATA")
        self.assertFalse(collecting["final_evaluation_allowed"])
        self.assertEqual(insufficient["state"], "WATCH_INSUFFICIENT_TRADES")
        self.assertEqual(ready["state"], "READY_FOR_FINAL_EVALUATION")
        self.assertTrue(ready["final_evaluation_allowed"])


if __name__ == "__main__":
    unittest.main()
