import json
import os
import tempfile
import unittest

import pandas as pd

from train import train as train_module


class TrainingMetadataTests(unittest.TestCase):
    def test_build_training_metadata_includes_hashes_and_metrics(self):
        index = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
        X = pd.DataFrame({"a": range(20), "b": range(20, 40)}, index=index)
        y = pd.Series([0, 1] * 10, index=index)
        with tempfile.NamedTemporaryFile(delete=False) as artifact:
            artifact.write(b"artifact-bytes")
            artifact_path = artifact.name
        try:
            metadata = train_module.build_training_metadata(
                X=X,
                y=y,
                feature_cols=["a", "b"],
                train_end=10,
                validation_start=12,
                validation_end=16,
                oos_start=18,
                original_train_rows=10,
                balanced_train_rows=8,
                validation_metrics={"lgb_v1": {"accuracy": 0.55}},
                artifact_paths=[artifact_path],
                sample_weight_summary={"method": "unit", "rows": 10},
            )
        finally:
            os.unlink(artifact_path)

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["feature_count"], 2)
        self.assertIn("feature_columns_sha256", metadata)
        self.assertEqual(metadata["validation_metrics"]["lgb_v1"]["accuracy"], 0.55)
        self.assertIn("label_mode", metadata)
        self.assertEqual(metadata["label_mode"], "tradable_quality")
        self.assertIn("label_filter_summary", metadata)
        self.assertEqual(metadata["training_balance_strategy"], "sample_weight_direction_then_regime_recency")
        self.assertEqual(metadata["sample_weight_summary"]["method"], "unit")
        self.assertEqual(metadata["evaluation_sample_weight_summary"], {})
        self.assertEqual(metadata["label_distribution"]["all"], {"0": 10, "1": 10})
        self.assertTrue(metadata["final_train_on_validation"])
        self.assertEqual(metadata["final_train_rows"], 10)
        self.assertEqual(metadata["validation_rows"], 4)
        self.assertEqual(metadata["oos_rows"], 2)
        self.assertTrue(metadata["artifact_hashes"])

    def test_tradable_quality_labels_turn_counter_trend_into_no_trade(self):
        index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
        close = pd.Series([100, 103, 100, 97, 100, 103], index=index, dtype=float)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close * 1.01,
            "15m_ema_60": close * 1.05,
        }, index=index)

        labeled = train_module.create_labels(
            df,
            future_window=1,
            threshold=0.01,
            tradable_only=True,
        )

        self.assertGreater(labeled.attrs["label_filter_summary"]["blocked_rows"], 0)
        self.assertIn(2, set(labeled["target"].astype(int)))
        self.assertIn(0, set(labeled["target"].astype(int)))
        self.assertGreater(len(labeled), 0)

    def test_quality_labels_keep_small_moves_as_no_trade(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.1, 100.0, 100.1], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close,
            "15m_ema_60": close,
        }, index=index)

        labeled = train_module.create_labels(
            df,
            future_window=1,
            threshold=0.01,
            tradable_only=True,
        )

        self.assertEqual(set(labeled["target"].astype(int)), {2})

    def test_raw_labels_can_keep_counter_trend_long_for_ab(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([100, 103, 100], index=index, dtype=float)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close * 1.01,
            "15m_ema_60": close * 1.05,
        }, index=index)

        labeled = train_module.create_labels(
            df,
            future_window=1,
            threshold=0.01,
            tradable_only=False,
        )

        self.assertEqual(set(labeled["target"].astype(int)), {0, 1})
        self.assertFalse(labeled.attrs["label_filter_summary"]["enabled"])

    def test_balance_samples_preserves_order_and_returns_weights(self):
        index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "regime_trend_long": [1, 1, 1, 0, 0, 0],
            "regime_trend_short": [0, 0, 0, 1, 1, 1],
            "regime_range_high_vol": [0, 0, 0, 0, 0, 0],
            "is_high_vol": [0, 0, 0, 0, 0, 0],
        }, index=index)
        y = pd.Series([1, 1, 0, 0, 0, 1], index=index)

        X_balanced, y_balanced, sample_weight, summary = train_module.balance_samples(X, y)

        self.assertEqual(list(X_balanced.index), list(index))
        self.assertEqual(list(y_balanced.index), list(index))
        self.assertEqual(list(sample_weight.index), list(index))
        self.assertEqual(len(X_balanced), len(X))
        self.assertEqual(summary["rows"], len(y))
        self.assertEqual(summary["method"], "direction_then_regime_inverse_frequency_with_recency")

    def test_sample_weights_balance_directions_before_regime_groups(self):
        index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "regime_trend_long": [1, 1, 1, 1, 1, 1, 0, 0],
            "regime_trend_short": [0, 0, 0, 0, 0, 0, 1, 0],
            "regime_range_high_vol": [0, 0, 0, 0, 0, 0, 0, 1],
            "is_high_vol": [0, 0, 0, 0, 0, 0, 0, 1],
        }, index=index)
        y = pd.Series([1, 1, 1, 1, 0, 0, 0, 2], index=index)

        sample_weight, _ = train_module.build_sample_weights(
            X,
            y,
            recent_boost=0.0,
            min_weight=0.0,
            max_weight=1000.0,
        )
        regimes = train_module.infer_sample_regimes(X)
        directions = y.astype(int).map(train_module._target_direction)
        direction_totals = sample_weight.groupby(directions).sum()
        group_totals = sample_weight.groupby(regimes + ":" + directions).sum()

        self.assertEqual(len(direction_totals), 3)
        for value in direction_totals:
            self.assertAlmostEqual(value, 8 / 3)
        self.assertAlmostEqual(group_totals["trend_long:short"], group_totals["trend_short:short"])

    def test_write_json_atomic_round_trips_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metadata.json")
            train_module.write_json_atomic(path, {"schema_version": 2, "ok": True})
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            self.assertEqual(payload["schema_version"], 2)
            self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
