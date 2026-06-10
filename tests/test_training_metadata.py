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
            )
        finally:
            os.unlink(artifact_path)

        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["feature_count"], 2)
        self.assertIn("feature_columns_sha256", metadata)
        self.assertEqual(metadata["validation_metrics"]["lgb_v1"]["accuracy"], 0.55)
        self.assertIn("label_mode", metadata)
        self.assertIn("label_filter_summary", metadata)
        self.assertEqual(metadata["label_distribution"]["all"], {"0": 10, "1": 10})
        self.assertEqual(metadata["validation_rows"], 4)
        self.assertEqual(metadata["oos_rows"], 2)
        self.assertTrue(metadata["artifact_hashes"])

    def test_tradable_labels_drop_counter_trend_long(self):
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
        self.assertTrue((labeled["target"] == 0).all())
        self.assertGreater(len(labeled), 0)

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
