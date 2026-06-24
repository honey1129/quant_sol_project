import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from train import train as train_module


class TinyBinaryEstimator:
    classes_ = [0, 1]

    def __init__(self):
        self.fit_rows = 0

    def fit(self, X, y, sample_weight=None):
        self.fit_rows = len(y)
        self.trade_rate = float(pd.Series(y).astype(int).mean()) if len(y) else 0.0
        return self

    def predict(self, X):
        return [1 if self.trade_rate >= 0.5 else 0 for _ in range(len(X))]

    def predict_proba(self, X):
        return [[1.0 - self.trade_rate, self.trade_rate] for _ in range(len(X))]


class FeatureProbabilityEstimator:
    classes_ = [0, 1]

    def fit(self, X, y, sample_weight=None):
        self.fit_rows = len(y)
        return self

    def predict(self, X):
        return [1 if row[0] >= 0.5 else 0 for row in self.predict_proba(X)]

    def predict_proba(self, X):
        score = pd.Series(X["score"]).astype(float).clip(0.01, 0.99)
        return [[1.0 - float(value), float(value)] for value in score]


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
        # label_mode 由 .env 的 tradable/no_trade 开关决定;断言它与当前配置一致,
        # 而非硬编码某个值,避免随 .env 变更而误报。
        self.assertEqual(metadata["label_mode"], train_module._label_mode())
        self.assertIn("label_filter_summary", metadata)
        self.assertEqual(
            metadata["training_balance_strategy"],
            "sample_weight_binary_target_regime_direction_recency",
        )
        self.assertEqual(metadata["sample_weight_summary"]["method"], "unit")
        self.assertEqual(metadata["evaluation_sample_weight_summary"], {})
        self.assertEqual(metadata["direction_quality_models"], {})
        self.assertEqual(metadata["label_distribution"]["all"], {"0": 10, "1": 10})
        self.assertEqual(metadata["target_schema"], "binary_trade_quality")
        self.assertEqual(metadata["target_labels"], {"0": "no_trade", "1": "trade"})
        self.assertEqual(metadata["label_lookahead_bars"], train_module._label_lookahead_bars())
        self.assertEqual(metadata["label_take_profit"], train_module._label_take_profit())
        self.assertEqual(metadata["label_stop_loss"], train_module._label_stop_loss())
        self.assertEqual(metadata["label_estimated_round_trip_cost"], train_module._round_trip_cost_ratio())
        self.assertEqual(metadata["label_min_net_return"], train_module._label_min_net_return())
        self.assertEqual(metadata["label_max_mae_ratio"], train_module._label_max_mae_ratio())
        self.assertEqual(metadata["label_require_regime_allowed"], train_module._label_require_regime_allowed())
        self.assertIn("label_quality_summary", metadata)
        self.assertTrue(metadata["final_train_on_validation"])
        self.assertEqual(metadata["final_train_rows"], 10)
        self.assertEqual(metadata["validation_rows"], 4)
        self.assertEqual(metadata["oos_rows"], 2)
        self.assertTrue(metadata["artifact_hashes"])

    def test_realistic_quality_labels_mark_only_tp_before_sl_as_trade(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        close = pd.Series([100, 100, 100, 100], index=index, dtype=float)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 102.0, 100.2, 100.2],
            "5m_low": [100.0, 99.8, 98.5, 99.8],
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close * 0.99,
            "15m_ema_60": close * 0.98,
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_LABEL_USE_REALISTIC": "1",
            "MODEL_LABEL_LOOKAHEAD_BARS": "1",
            "MODEL_LABEL_TAKE_PROFIT": "0.01",
            "MODEL_LABEL_STOP_LOSS": "0.01",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        self.assertIn(1, set(labeled["target"].astype(int)))
        self.assertIn(0, set(labeled["target"].astype(int)))
        self.assertGreater(len(labeled), 0)
        self.assertIn("label_net_return", labeled.columns)
        self.assertIn("label_quality_summary", labeled.attrs)
        self.assertGreater(labeled.attrs["label_quality_summary"]["trade_rows"], 0)

    def test_quality_labels_keep_small_moves_as_no_trade(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.1, 100.0, 100.1], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": close + 0.1,
            "5m_low": close - 0.1,
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close,
            "15m_ema_60": close,
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_LABEL_USE_REALISTIC": "1",
            "MODEL_LABEL_LOOKAHEAD_BARS": "1",
            "MODEL_LABEL_TAKE_PROFIT": "0.01",
            "MODEL_LABEL_STOP_LOSS": "0.01",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        self.assertEqual(set(labeled["target"].astype(int)), {0})

    def test_realistic_labels_respect_regime_trade_blocks(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([102, 102, 102], index=index, dtype=float)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [102.0, 104.0, 104.0],
            "5m_low": [102.0, 101.8, 101.8],
            "5m_atr": 1.0,
            "volatility_15": 0.02,
            "money_flow_ratio": 2.5,
            "15m_ema_20": close,
            "15m_ema_60": close * 0.98,
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_LABEL_USE_REALISTIC": "1",
            "MODEL_LABEL_LOOKAHEAD_BARS": "1",
            "MODEL_LABEL_TAKE_PROFIT": "0.01",
            "MODEL_LABEL_STOP_LOSS": "0.01",
            "MODEL_LABEL_REQUIRE_REGIME_ALLOWED": "1",
        }):
            with patch("train.train.config.REGIME_FILTER_ENABLED", True):
                with patch("train.train.config.REGIME_HIGH_VOL_ALLOW_TRADES", False):
                    with patch("train.train.config.REGIME_TREND_GAP_THRESHOLD", 0.03):
                        labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        self.assertEqual(set(labeled["target"].astype(int)), {0})
        self.assertIn("regime_block", " ".join(labeled["label_reject_reason"].astype(str)))

    def test_threshold_labels_mark_large_moves_as_trade(self):
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

        with patch.dict(os.environ, {"MODEL_LABEL_USE_REALISTIC": "0"}):
            labeled = train_module.create_labels(
                df,
                future_window=1,
                threshold=0.01,
                tradable_only=False,
            )

        self.assertEqual(set(labeled["target"].astype(int)), {1})
        self.assertFalse(labeled.attrs["label_filter_summary"]["enabled"])

    def test_label_diagnostics_are_excluded_from_model_features(self):
        df = pd.DataFrame({
            "rsi": [50.0],
            "label_net_return": [0.02],
            "label_mfe": [0.03],
            "target": [1],
        })

        feature_cols = train_module.model_feature_columns(df)

        self.assertEqual(feature_cols, ["rsi"])

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
        self.assertEqual(summary["method"], "binary_target_regime_direction_with_recency")

    def test_sample_weights_balance_targets_and_regime_direction_groups(self):
        index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "regime_trend_long": [1, 1, 1, 1, 1, 1, 0, 0],
            "regime_trend_short": [0, 0, 0, 0, 0, 0, 1, 0],
            "regime_range_high_vol": [0, 0, 0, 0, 0, 0, 0, 1],
            "is_high_vol": [0, 0, 0, 0, 0, 0, 0, 1],
        }, index=index)
        y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0], index=index)
        sample_context = pd.DataFrame({
            "label_regime": [
                "trend_long", "trend_long", "trend_long", "trend_long",
                "trend_long", "trend_long", "trend_short", "range_high_vol",
            ],
            "label_direction": ["long", "long", "long", "long", "long", "long", "short", "none"],
        }, index=index)

        with patch("train.train.config.MODEL_TRADE_SAMPLE_WEIGHT_MULTIPLIER", 1.0):
            with patch("train.train.config.MODEL_NO_TRADE_SAMPLE_WEIGHT_MULTIPLIER", 1.0):
                sample_weight, _ = train_module.build_sample_weights(
                    X,
                    y,
                    sample_context=sample_context,
                    recent_boost=0.0,
                    min_weight=0.0,
                    max_weight=1000.0,
                )

        targets = y.astype(int).map(train_module._target_direction)
        groups = targets + ":" + sample_context["label_regime"] + ":" + sample_context["label_direction"]
        target_totals = sample_weight.groupby(targets).sum()
        group_totals = sample_weight.groupby(groups).sum()

        self.assertAlmostEqual(target_totals["trade"], target_totals["no_trade"])
        self.assertAlmostEqual(group_totals["no_trade:trend_long:long"], group_totals["no_trade:trend_short:short"])
        self.assertAlmostEqual(group_totals["no_trade:trend_short:short"], group_totals["no_trade:range_high_vol:none"])

    def test_direction_quality_bundle_trains_long_and_short_submodels(self):
        index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "trend_bias_num": [1.0] * 6 + [-1.0] * 6,
            "regime_trend_long": [1.0] * 6 + [0.0] * 6,
            "regime_trend_short": [0.0] * 6 + [1.0] * 6,
        }, index=index)
        y = pd.Series([1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], index=index)
        context = pd.DataFrame({
            "label_direction": ["long"] * 6 + ["short"] * 6,
            "label_regime": ["trend_long"] * 6 + ["trend_short"] * 6,
            "label_outcome": ["TP", "SL", "TP", "TIMEOUT", "SL", "TP"] * 2,
            "label_reject_reason": ["accepted", "outcome_sl", "accepted", "outcome_timeout", "outcome_sl", "accepted"] * 2,
        }, index=index)

        with patch("train.train.build_model_estimators", return_value={
            "lgb_v1": TinyBinaryEstimator(),
            "xgb_v1": TinyBinaryEstimator(),
            "rf_v1": TinyBinaryEstimator(),
        }):
            with patch("train.train.config.MODEL_DIRECTION_QUALITY_MIN_ROWS", 1):
                with patch("train.train.config.MODEL_DIRECTION_QUALITY_MIN_TRADE_ROWS", 1):
                    models, _, _, _, summary = train_module.train_direction_quality_bundle(
                        X,
                        y,
                        sample_context=context,
                    )

        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["enabled_directions"], ["long", "short"])
        self.assertTrue(summary["directions"]["long"]["enabled"])
        self.assertTrue(summary["directions"]["short"]["enabled"])
        self.assertTrue(models["lgb_v1"].direction_quality_enabled)
        self.assertEqual(models["lgb_v1"].trained_directions, ["long", "short"])

    def test_direction_quality_bundle_trains_direction_calibrators(self):
        index = pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "score": [
                0.05, 0.10, 0.15, 0.20, 0.75, 0.80, 0.85, 0.90,
                0.12, 0.18, 0.78, 0.88, 0.08, 0.22, 0.82, 0.92,
                0.05, 0.95, 0.10, 0.90,
            ] * 2,
            "trend_bias_num": [1.0] * 20 + [-1.0] * 20,
            "regime_trend_long": [1.0] * 20 + [0.0] * 20,
            "regime_trend_short": [0.0] * 20 + [1.0] * 20,
        }, index=index)
        y = pd.Series(
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1] * 2,
            index=index,
        )
        context = pd.DataFrame({
            "label_direction": ["long"] * 20 + ["short"] * 20,
            "label_regime": ["trend_long"] * 20 + ["trend_short"] * 20,
            "label_outcome": ["SL", "TIMEOUT", "SL", "TIMEOUT", "TP"] * 8,
            "label_reject_reason": ["outcome_sl", "outcome_timeout", "outcome_sl", "outcome_timeout", "accepted"] * 8,
        }, index=index)

        with patch("train.train.build_model_estimators", return_value={
            "lgb_v1": FeatureProbabilityEstimator(),
            "xgb_v1": FeatureProbabilityEstimator(),
            "rf_v1": FeatureProbabilityEstimator(),
        }):
            with patch("train.train.config.MODEL_DIRECTION_QUALITY_MIN_ROWS", 4):
                with patch("train.train.config.MODEL_DIRECTION_QUALITY_MIN_TRADE_ROWS", 2):
                    with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION", "sigmoid"):
                        with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_RATIO", 0.25):
                            with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_MIN_ROWS", 4):
                                with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_MIN_POSITIVES", 1):
                                    with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_MIN_NEGATIVES", 1):
                                        models, _, _, _, summary = train_module.train_direction_quality_bundle(
                                            X,
                                            y,
                                            sample_context=context,
                                        )

        self.assertEqual(summary["calibration_method"], "sigmoid")
        self.assertEqual(summary["calibrated_directions"], ["long", "short"])
        self.assertEqual(models["lgb_v1"].calibrated_directions, ["long", "short"])
        self.assertTrue(summary["directions"]["long"]["calibration"]["lgb_v1"]["active"])
        self.assertGreater(summary["directions"]["long"]["calibration"]["lgb_v1"]["fitted_rows"], 0)

    def test_direction_quality_bundle_trains_regime_calibrators(self):
        index = pd.date_range("2026-01-01", periods=48, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "score": [
                0.10, 0.20, 0.80, 0.90,
                0.12, 0.22, 0.82, 0.92,
                0.14, 0.24, 0.84, 0.94,
            ] * 4,
            "trend_bias_num": [1.0] * 24 + [-1.0] * 24,
            "regime_trend_long": [1.0] * 24 + [0.0] * 24,
            "regime_trend_short": [0.0] * 24 + [1.0] * 24,
            "regime_range_high_vol": [0.0] * 48,
        }, index=index)
        y = pd.Series([0, 0, 1, 1] * 12, index=index)
        context = pd.DataFrame({
            "label_direction": ["long"] * 24 + ["short"] * 24,
            "label_regime": ["trend_long"] * 24 + ["trend_short"] * 24,
            "label_outcome": ["SL", "TIMEOUT", "TP", "TP"] * 12,
            "label_reject_reason": ["outcome_sl", "outcome_timeout", "accepted", "accepted"] * 12,
        }, index=index)

        with patch("train.train.build_model_estimators", return_value={
            "lgb_v1": FeatureProbabilityEstimator(),
            "xgb_v1": FeatureProbabilityEstimator(),
            "rf_v1": FeatureProbabilityEstimator(),
        }):
            with patch("train.train.config.MODEL_DIRECTION_QUALITY_MIN_ROWS", 4):
                with patch("train.train.config.MODEL_DIRECTION_QUALITY_MIN_TRADE_ROWS", 2):
                    with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION", "sigmoid"):
                        with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_RATIO", 0.25):
                            with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_MIN_ROWS", 4):
                                with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_MIN_POSITIVES", 1):
                                    with patch("train.train.config.MODEL_DIRECTION_QUALITY_CALIBRATION_MIN_NEGATIVES", 1):
                                        with patch("train.train.config.MODEL_DIRECTION_QUALITY_REGIME_CALIBRATION", True):
                                            with patch("train.train.config.MODEL_DIRECTION_QUALITY_REGIME_CALIBRATION_MIN_ROWS", 4):
                                                with patch("train.train.config.MODEL_DIRECTION_QUALITY_REGIME_CALIBRATION_MIN_POSITIVES", 1):
                                                    with patch("train.train.config.MODEL_DIRECTION_QUALITY_REGIME_CALIBRATION_MIN_NEGATIVES", 1):
                                                        models, _, _, _, summary = train_module.train_direction_quality_bundle(
                                                            X,
                                                            y,
                                                            sample_context=context,
                                                        )

        self.assertTrue(summary["regime_calibration_enabled"])
        self.assertEqual(summary["calibrated_direction_regimes"], ["long:trend_long", "short:trend_short"])
        self.assertEqual(models["lgb_v1"].calibrated_direction_regimes, ["long:trend_long", "short:trend_short"])
        self.assertTrue(
            summary["directions"]["short"]["regime_calibration"]["lgb_v1"]["trend_short"]["active"]
        )

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
