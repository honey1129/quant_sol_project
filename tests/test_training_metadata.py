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
    def test_probability_calibrator_rejects_negative_sigmoid_slope(self):
        calibrator = train_module.fit_binary_probability_calibrator(
            [0.10, 0.20, 0.80, 0.90],
            [1, 1, 0, 0],
            method="sigmoid",
            min_rows=4,
            min_positive_rows=1,
            min_negative_rows=1,
        )

        self.assertFalse(calibrator.active)
        self.assertEqual(calibrator.summary()["fallback_reason"], "calibration_negative_slope")

    def test_probability_calibrator_allows_inverse_sigmoid_when_enabled(self):
        calibrator = train_module.fit_binary_probability_calibrator(
            [0.10, 0.20, 0.80, 0.90],
            [1, 1, 0, 0],
            method="sigmoid",
            min_rows=4,
            min_positive_rows=1,
            min_negative_rows=1,
            allow_negative_slope=True,
        )

        summary = calibrator.summary()
        self.assertTrue(calibrator.active)
        self.assertTrue(summary["inverted"])
        self.assertLess(summary["coef"], 0.0)

    def test_probability_calibrator_keeps_positive_sigmoid_slope(self):
        calibrator = train_module.fit_binary_probability_calibrator(
            [0.10, 0.20, 0.80, 0.90],
            [0, 0, 1, 1],
            method="sigmoid",
            min_rows=4,
            min_positive_rows=1,
            min_negative_rows=1,
        )

        self.assertTrue(calibrator.active)
        self.assertGreater(calibrator.summary()["coef"], 0.0)

    def test_model_estimators_accept_lightweight_estimator_config(self):
        default_models = train_module.build_model_estimators()
        lightweight_models = train_module.build_model_estimators({
            "lgb_n_estimators": 11,
            "xgb_n_estimators": 12,
            "rf_n_estimators": 13,
        })

        self.assertEqual(default_models["lgb_v1"].n_estimators, 160)
        self.assertEqual(default_models["xgb_v1"].n_estimators, 160)
        self.assertEqual(default_models["rf_v1"].n_estimators, 100)
        self.assertEqual(default_models["lgb_v1"].verbosity, -1)
        self.assertEqual(default_models["xgb_v1"].verbosity, 0)
        self.assertEqual(default_models["rf_v1"].n_jobs, -1)
        self.assertEqual(lightweight_models["lgb_v1"].n_estimators, 11)
        self.assertEqual(lightweight_models["xgb_v1"].n_estimators, 12)
        self.assertEqual(lightweight_models["rf_v1"].n_estimators, 13)

    def test_validation_estimator_config_uses_lightweight_settings(self):
        with patch.dict(os.environ, {
            "MODEL_VALIDATION_LIGHTWEIGHT_TRAINING": "1",
            "MODEL_VALIDATION_LGB_ESTIMATORS": "21",
            "MODEL_VALIDATION_XGB_ESTIMATORS": "22",
            "MODEL_VALIDATION_RF_ESTIMATORS": "23",
        }):
            estimator_config = train_module.validation_estimator_config()

        self.assertEqual(estimator_config, {
            "lgb_n_estimators": 21,
            "xgb_n_estimators": 22,
            "rf_n_estimators": 23,
        })

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
            "sample_weight_binary_target_regime_direction_hard_negative_recency",
        )
        self.assertEqual(metadata["sample_weight_summary"]["method"], "unit")
        self.assertEqual(metadata["evaluation_sample_weight_summary"], {})
        self.assertEqual(metadata["validation_gate_summary"], {})
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
        self.assertEqual(metadata["label_timeout_as_trade"], train_module._label_timeout_as_trade())
        self.assertEqual(
            metadata["label_timeout_weak_positive_as_trade"],
            train_module._label_timeout_weak_positive_as_trade(),
        )
        self.assertEqual(metadata["label_timeout_min_net_return"], train_module._label_timeout_min_net_return())
        self.assertEqual(metadata["label_timeout_max_mae_ratio"], train_module._label_timeout_max_mae_ratio())
        self.assertEqual(metadata["label_require_regime_allowed"], train_module._label_require_regime_allowed())
        self.assertIn("label_quality_summary", metadata)
        self.assertTrue(metadata["final_train_on_validation"])
        self.assertEqual(metadata["final_train_rows"], 10)
        self.assertEqual(metadata["validation_rows"], 4)
        self.assertEqual(metadata["oos_rows"], 2)
        self.assertTrue(metadata["artifact_hashes"])

    def test_write_candidate_training_metadata_marks_diagnostic_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidate_training_metadata.json")
            metadata = {
                "schema_version": 2,
                "artifact_hashes": {"models/lgb_model.pkl": "old"},
                "validation_gate_summary": {"trade_recall": 0.0},
            }
            with patch("train.train.candidate_training_metadata_path", candidate_path):
                train_module.write_candidate_training_metadata(metadata)

            with open(candidate_path, "r", encoding="utf-8") as file:
                saved = json.load(file)

        self.assertEqual(saved["candidate_status"], "validation_gate_pending")
        self.assertFalse(saved["candidate_artifacts_written"])
        self.assertEqual(saved["validation_gate_summary"]["trade_recall"], 0.0)

    def test_write_candidate_training_metadata_preserves_explicit_failure_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = os.path.join(tmpdir, "candidate_training_metadata.json")
            with patch("train.train.candidate_training_metadata_path", candidate_path):
                train_module.write_candidate_training_metadata({
                    "candidate_status": "validation_gate_failed",
                    "validation_gate_failure_reason": "precision too low",
                })

            with open(candidate_path, "r", encoding="utf-8") as file:
                saved = json.load(file)

        self.assertEqual(saved["candidate_status"], "validation_gate_failed")
        self.assertEqual(saved["validation_gate_failure_reason"], "precision too low")
        self.assertFalse(saved["candidate_artifacts_written"])

    def test_validation_gate_summary_uses_probability_threshold(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        X = pd.DataFrame({"score": [0.10, 0.80, 0.90, 0.20]}, index=index)
        y = pd.Series([0, 1, 1, 0], index=index)
        context = pd.DataFrame({
            "label_direction": ["long"] * 4,
            "label_regime": ["trend_long"] * 4,
        }, index=index)

        summary = train_module.build_validation_gate_summary(
            {"lgb_v1": FeatureProbabilityEstimator()},
            {"lgb_v1": 1.0},
            X,
            y,
            threshold=0.5,
            sample_context=context,
        )

        self.assertEqual(summary["trade_rows"], 2)
        self.assertEqual(summary["predicted_trade_rows"], 2)
        self.assertAlmostEqual(summary["trade_precision"], 1.0)
        self.assertAlmostEqual(summary["trade_recall"], 1.0)

    def test_validation_gate_auto_threshold_uses_live_entry_threshold(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        X = pd.DataFrame({"score": [0.40, 0.54, 0.57, 0.80]}, index=index)
        y = pd.Series([0, 0, 1, 1], index=index)
        context = pd.DataFrame({
            "label_direction": ["long"] * 4,
            "label_regime": ["trend_long"] * 4,
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_RETRAIN_VALIDATION_GATE_THRESHOLD": "auto",
        }):
            with patch("train.train.config.MODEL_WALK_FORWARD_DIAGNOSTIC_THRESHOLD", 0.35):
                with patch("train.train.config.THRESHOLD_LONG", 0.56):
                    with patch("train.train.config.THRESHOLD_SHORT", 0.60):
                        summary = train_module.build_validation_gate_summary(
                            {"lgb_v1": FeatureProbabilityEstimator()},
                            {"lgb_v1": 1.0},
                            X,
                            y,
                            sample_context=context,
                        )

        self.assertAlmostEqual(summary["decision_threshold"], 0.56)
        self.assertEqual(summary["predicted_trade_rows"], 2)
        self.assertAlmostEqual(summary["trade_precision"], 1.0)

    def test_validation_gate_threshold_sweep_recommends_precision_target(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        X = pd.DataFrame({"score": [0.40, 0.55, 0.62, 0.80]}, index=index)
        y = pd.Series([0, 0, 1, 1], index=index)
        context = pd.DataFrame({
            "label_direction": ["long", "long", "short", "short"],
            "label_regime": ["trend_long", "trend_long", "trend_short", "trend_short"],
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_RETRAIN_VALIDATION_GATE_THRESHOLD_SWEEP": "0.50,0.60,0.70",
            "MODEL_RETRAIN_VALIDATION_GATE_TARGET_PRECISION": "1.0",
        }):
            summary = train_module.build_validation_gate_summary(
                {"lgb_v1": FeatureProbabilityEstimator()},
                {"lgb_v1": 1.0},
                X,
                y,
                threshold=0.5,
                sample_context=context,
            )

        sweep = summary["threshold_sweep"]
        self.assertEqual([item["threshold"] for item in sweep["candidates"]], [0.5, 0.6, 0.7])
        self.assertEqual(sweep["recommended"]["threshold"], 0.6)
        self.assertAlmostEqual(sweep["recommended"]["trade_precision"], 1.0)
        self.assertEqual(
            sweep["recommended"]["direction_metrics"]["short"]["predicted_trade_rows"],
            2,
        )
        self.assertIn("lgb_v1", summary["model_threshold_sweeps"])
        self.assertEqual(summary["model_threshold_sweeps"]["lgb_v1"]["recommended"]["threshold"], 0.6)

    def test_validation_gate_summary_includes_probability_separability_diagnostics(self):
        index = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
        X = pd.DataFrame({"score": [0.05, 0.10, 0.20, 0.75, 0.85, 0.95]}, index=index)
        y = pd.Series([0, 0, 0, 1, 1, 1], index=index)
        context = pd.DataFrame({
            "label_direction": ["long", "long", "short", "long", "short", "short"],
            "label_regime": ["trend_long", "trend_long", "trend_short"] * 2,
        }, index=index)

        summary = train_module.build_validation_gate_summary(
            {"lgb_v1": FeatureProbabilityEstimator()},
            {"lgb_v1": 1.0},
            X,
            y,
            threshold=0.5,
            sample_context=context,
        )

        diagnostics = summary["separability_diagnostics"]
        self.assertAlmostEqual(diagnostics["roc_auc"], 1.0)
        self.assertAlmostEqual(diagnostics["average_precision"], 1.0)
        self.assertEqual(diagnostics["ranking_signal"], "positive")
        self.assertGreater(diagnostics["mean_gap"], 0.0)
        self.assertEqual(diagnostics["top_bucket_precision"]["top_10pct"]["precision"], 1.0)
        self.assertIn("long", diagnostics["by_direction"])
        self.assertIn("short:trend_short", diagnostics["by_direction_regime"])
        self.assertAlmostEqual(
            summary["model_separability_diagnostics"]["lgb_v1"]["roc_auc"],
            1.0,
        )

    def test_validation_gate_recommendations_flag_inverted_direction_and_best_model(self):
        separability = {
            "top_bucket_precision": {"top_10pct": {"precision": 0.08}},
            "by_direction": {
                "long": {
                    "rows": 100,
                    "trade_rows": 10,
                    "trade_rate": 0.10,
                    "roc_auc": 0.30,
                    "average_precision": 0.06,
                    "ranking_signal": "inverted",
                    "mean_gap": -0.02,
                    "top_bucket_precision": {
                        "top_10pct": {
                            "rows": 10,
                            "trade_rows": 0,
                            "precision": 0.0,
                            "lift_vs_base_rate": 0.0,
                        },
                    },
                },
                "short": {
                    "rows": 100,
                    "trade_rows": 10,
                    "trade_rate": 0.10,
                    "roc_auc": 0.60,
                    "average_precision": 0.12,
                    "ranking_signal": "positive",
                    "mean_gap": 0.02,
                    "top_bucket_precision": {
                        "top_10pct": {
                            "rows": 10,
                            "trade_rows": 1,
                            "precision": 0.10,
                            "lift_vs_base_rate": 1.0,
                        },
                    },
                },
            },
        }
        model_separability = {
            "xgb_v1": {
                "by_direction": {
                    "short": {
                        "rows": 100,
                        "trade_rows": 10,
                        "trade_rate": 0.10,
                        "roc_auc": 0.72,
                        "average_precision": 0.24,
                        "ranking_signal": "positive",
                        "mean_gap": 0.04,
                        "top_bucket_precision": {
                            "top_10pct": {
                                "rows": 10,
                                "trade_rows": 3,
                                "precision": 0.30,
                                "lift_vs_base_rate": 3.0,
                            },
                        },
                    },
                },
            },
            "rf_v1": {
                "by_direction": {
                    "short": {
                        "rows": 100,
                        "trade_rows": 10,
                        "trade_rate": 0.10,
                        "roc_auc": 0.45,
                        "average_precision": 0.08,
                        "ranking_signal": "inverted",
                        "mean_gap": -0.01,
                        "top_bucket_precision": {
                            "top_10pct": {
                                "rows": 10,
                                "trade_rows": 0,
                                "precision": 0.0,
                                "lift_vs_base_rate": 0.0,
                            },
                        },
                    },
                },
            },
        }

        with patch.dict(os.environ, {
            "MODEL_RETRAIN_VALIDATION_GATE_TARGET_PRECISION": "0.25",
        }):
            recommendations = train_module._validation_gate_diagnostic_recommendations(
                separability,
                model_separability,
                {"recommended": None},
            )

        self.assertTrue(recommendations["do_not_relax_threshold"])
        self.assertEqual(recommendations["directions"]["long"]["status"], "unusable")
        self.assertIn(
            "inverted_probability_ranking",
            recommendations["directions"]["long"]["reason_codes"],
        )
        self.assertEqual(
            recommendations["directions"]["short"]["recommended_model_weights"],
            {"xgb_v1": 1.0},
        )
        self.assertEqual(
            recommendations["recommended_env_overrides"]["MODEL_DIRECTION_MODEL_WEIGHTS"],
            "short=xgb_v1:1.0",
        )

    def test_validation_gate_summary_includes_group_error_diagnostics(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        X = pd.DataFrame({"score": [0.80, 0.20, 0.90, 0.10]}, index=index)
        y = pd.Series([0, 1, 0, 1], index=index)
        context = pd.DataFrame({
            "label_direction": ["short", "long", "short", "long"],
            "label_regime": ["trend_short", "trend_long", "trend_short", "trend_long"],
            "label_outcome": ["SL", "TP", "TIMEOUT_WEAK_NEGATIVE", "TP"],
            "label_reject_reason": [
                "outcome_sl",
                "accepted",
                "timeout_weak_negative_net_return",
                "accepted",
            ],
        }, index=index)

        summary = train_module.build_validation_gate_summary(
            {"lgb_v1": FeatureProbabilityEstimator()},
            {"lgb_v1": 1.0},
            X,
            y,
            threshold=0.5,
            sample_context=context,
        )

        diagnostics = summary["group_diagnostics"]
        self.assertEqual(diagnostics["false_positive_outcome_counts"], {
            "SL": 1,
            "TIMEOUT_WEAK_NEGATIVE": 1,
        })
        self.assertEqual(diagnostics["false_negative_direction_counts"], {"long": 2})
        self.assertEqual(diagnostics["by_direction"]["short"]["fp"], 2)
        self.assertEqual(diagnostics["by_direction_regime"]["long:trend_long"]["fn"], 2)
        model_diagnostics = summary["model_group_diagnostics"]["lgb_v1"]
        self.assertEqual(model_diagnostics["by_direction"]["short"]["fp"], 2)
        self.assertEqual(model_diagnostics["by_direction_regime"]["long:trend_long"]["fn"], 2)

    def test_validation_gate_summary_maps_binary_probability_to_tradable_direction(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        X = pd.DataFrame({"score": [0.90, 0.90, 0.90]}, index=index)
        y = pd.Series([0, 1, 0], index=index)
        context = pd.DataFrame({
            "label_direction": ["none", "long", "none"],
            "label_regime": ["range", "trend_long", "range_high_vol"],
            "label_outcome": ["NO_DIRECTION", "TP", "NO_DIRECTION"],
            "label_reject_reason": ["neutral_trend", "accepted", "neutral_trend"],
        }, index=index)

        summary = train_module.build_validation_gate_summary(
            {"lgb_v1": FeatureProbabilityEstimator()},
            {"lgb_v1": 1.0},
            X,
            y,
            threshold=0.5,
            sample_context=context,
        )

        self.assertEqual(summary["predicted_trade_rows"], 1)
        self.assertAlmostEqual(summary["trade_precision"], 1.0)
        self.assertEqual(summary["group_diagnostics"]["by_direction"]["none"]["fp"], 0)

    def test_validation_gate_rejects_collapsed_trade_predictions(self):
        summary = {
            "trade_rows": 5,
            "predicted_trade_rows": 0,
            "trade_recall": 0.0,
        }

        with patch.dict(os.environ, {
            "MODEL_RETRAIN_VALIDATION_GATE_ENABLED": "1",
            "MODEL_RETRAIN_MIN_VALIDATION_TRADE_RECALL": "0.01",
            "MODEL_RETRAIN_MIN_VALIDATION_PREDICTED_TRADES": "1",
        }):
            with self.assertRaises(ValueError) as context:
                train_module.validate_retrain_validation_gate(summary)

        self.assertIn("验证集候选交易数不足", str(context.exception))

    def test_validation_gate_rejects_low_trade_precision(self):
        summary = {
            "trade_rows": 20,
            "predicted_trade_rows": 10,
            "trade_recall": 0.10,
            "trade_precision": 0.20,
        }

        with patch.dict(os.environ, {
            "MODEL_RETRAIN_VALIDATION_GATE_ENABLED": "1",
            "MODEL_RETRAIN_MIN_VALIDATION_TRADE_RECALL": "0.01",
            "MODEL_RETRAIN_MIN_VALIDATION_TRADE_PRECISION": "0.25",
            "MODEL_RETRAIN_MIN_VALIDATION_PREDICTED_TRADES": "1",
        }):
            with self.assertRaises(ValueError) as context:
                train_module.validate_retrain_validation_gate(summary)

        self.assertIn("验证集 trade precision 过低", str(context.exception))

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

    def test_long_trend_weak_tp_is_no_trade_by_default(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.2, 100.8, 101.7], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 100.3, 100.9, 101.7],
            "5m_low": [100.0, 99.0, 100.0, 100.8],
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close * 0.99,
            "15m_ema_60": close * 0.98,
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_LABEL_USE_REALISTIC": "1",
            "MODEL_LABEL_LOOKAHEAD_BARS": "3",
            "MODEL_LABEL_TAKE_PROFIT": "0.016",
            "MODEL_LABEL_STOP_LOSS": "0.014",
            "MODEL_LABEL_LONG_TREND_WEAK_TP_AS_TRADE": "0",
            "MODEL_LABEL_LONG_TREND_STRONG_MAX_EXIT_BARS": "2",
            "MODEL_LABEL_LONG_TREND_STRONG_MAX_MAE_RATIO": "0.50",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        first = labeled.iloc[0]
        self.assertEqual(int(first["target"]), 0)
        self.assertEqual(first["label_outcome"], "TP_WEAK_LONG_TREND")
        self.assertEqual(first["label_reject_reason"], "long_trend_weak_tp_slow")

    def test_long_trend_strong_tp_remains_trade(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([100.0, 101.7, 101.7], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 101.7, 101.7],
            "5m_low": [100.0, 100.0, 101.0],
            "5m_atr": 0.1,
            "volatility_15": 0.0,
            "money_flow_ratio": 1.0,
            "15m_ema_20": close * 0.99,
            "15m_ema_60": close * 0.98,
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_LABEL_USE_REALISTIC": "1",
            "MODEL_LABEL_LOOKAHEAD_BARS": "2",
            "MODEL_LABEL_TAKE_PROFIT": "0.016",
            "MODEL_LABEL_STOP_LOSS": "0.014",
            "MODEL_LABEL_LONG_TREND_WEAK_TP_AS_TRADE": "0",
            "MODEL_LABEL_LONG_TREND_STRONG_MAX_EXIT_BARS": "2",
            "MODEL_LABEL_LONG_TREND_STRONG_MAX_MAE_RATIO": "0.50",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        first = labeled.iloc[0]
        self.assertEqual(int(first["target"]), 1)
        self.assertEqual(first["label_outcome"], "TP_STRONG_LONG_TREND")
        self.assertEqual(first["label_reject_reason"], "accepted")

    def test_realistic_quality_labels_keep_timeout_weak_positive_as_no_trade_by_default(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.5, 100.5], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 100.6, 100.6],
            "5m_low": [100.0, 99.8, 100.0],
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
            "MODEL_LABEL_TIMEOUT_AS_TRADE": "1",
            "MODEL_LABEL_TIMEOUT_MIN_NET_RETURN": "0.0",
            "MODEL_LABEL_TIMEOUT_MAX_MAE_RATIO": "0.6",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        self.assertEqual(len(labeled), 1)
        filter_summary = labeled.attrs["label_filter_summary"]
        self.assertTrue(filter_summary["enabled"])
        self.assertEqual(filter_summary["ignored_rows"], 1)
        self.assertEqual(filter_summary["ignored_outcome_counts"], {"TIMEOUT_WEAK_POSITIVE": 1})
        self.assertEqual(filter_summary["ignored_reason_counts"], {"timeout_weak_positive_not_trade": 1})
        self.assertEqual(set(labeled["label_outcome"]), {"TIMEOUT_WEAK_NEGATIVE"})

    def test_realistic_quality_labels_allow_timeout_weak_positive_as_trade_when_enabled(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.5, 100.5], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 100.6, 100.6],
            "5m_low": [100.0, 99.8, 100.0],
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
            "MODEL_LABEL_TIMEOUT_AS_TRADE": "1",
            "MODEL_LABEL_TIMEOUT_WEAK_POSITIVE_AS_TRADE": "1",
            "MODEL_LABEL_TIMEOUT_MIN_NET_RETURN": "0.0",
            "MODEL_LABEL_TIMEOUT_MAX_MAE_RATIO": "0.6",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        first = labeled.iloc[0]
        self.assertEqual(int(first["target"]), 1)
        self.assertEqual(first["label_outcome"], "TIMEOUT_WEAK_POSITIVE")
        self.assertEqual(first["label_reject_reason"], "accepted")

    def test_realistic_quality_labels_reject_timeout_below_min_net_return(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.5, 100.5], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 100.6, 100.6],
            "5m_low": [100.0, 99.8, 100.0],
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
            "MODEL_LABEL_TIMEOUT_AS_TRADE": "1",
            "MODEL_LABEL_TIMEOUT_MIN_NET_RETURN": "0.004",
            "MODEL_LABEL_TIMEOUT_MAX_MAE_RATIO": "0.6",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        first = labeled.iloc[0]
        self.assertEqual(int(first["target"]), 0)
        self.assertEqual(first["label_outcome"], "TIMEOUT_WEAK_NEGATIVE")
        self.assertEqual(first["label_reject_reason"], "timeout_weak_negative_net_return")

    def test_realistic_quality_labels_split_timeout_weak_negative(self):
        index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        close = pd.Series([100.0, 100.5, 100.5], index=index)
        df = pd.DataFrame({
            "5m_close": close,
            "5m_high": [100.0, 100.6, 100.6],
            "5m_low": [100.0, 99.2, 100.0],
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
            "MODEL_LABEL_TIMEOUT_AS_TRADE": "1",
            "MODEL_LABEL_TIMEOUT_MIN_NET_RETURN": "0.0",
            "MODEL_LABEL_TIMEOUT_MAX_MAE_RATIO": "0.6",
        }):
            labeled = train_module.create_labels(df, future_window=1, threshold=0.01)

        first = labeled.iloc[0]
        self.assertEqual(int(first["target"]), 0)
        self.assertEqual(first["label_outcome"], "TIMEOUT_WEAK_NEGATIVE")
        self.assertEqual(first["label_reject_reason"], "timeout_weak_negative_mae")

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
        self.assertEqual(summary["method"], "binary_target_regime_direction_hard_negative_with_recency")

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

        with patch.dict(os.environ, {
            "MODEL_DIRECTION_TRADE_SAMPLE_WEIGHT_MULTIPLIERS": "",
            "MODEL_DIRECTION_HARD_NEGATIVE_SAMPLE_WEIGHT_MULTIPLIERS": "",
        }):
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

    def test_sample_weights_boost_sl_and_timeout_weak_negative_rows(self):
        index = pd.date_range("2026-01-01", periods=7, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "trend_bias_num": [1.0] * 7,
            "regime_trend_long": [1.0] * 7,
            "regime_trend_short": [0.0] * 7,
            "regime_range_high_vol": [0.0] * 7,
        }, index=index)
        y = pd.Series([1, 1, 0, 0, 0, 0, 0], index=index)
        sample_context = pd.DataFrame({
            "label_regime": ["trend_long"] * 7,
            "label_direction": ["long"] * 7,
            "label_outcome": [
                "TP", "TIMEOUT_WEAK_POSITIVE", "SL", "TIMEOUT_WEAK_NEGATIVE",
                "TP_WEAK_LONG_TREND", "NO_DIRECTION", "RULE_BLOCK",
            ],
            "label_reject_reason": [
                "accepted",
                "accepted",
                "outcome_sl",
                "timeout_weak_negative_net_return",
                "long_trend_weak_tp_slow",
                "neutral_trend",
                "regime_block:range_high_vol",
            ],
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_HARD_NEGATIVE_SAMPLE_WEIGHT_MULTIPLIER": "3.0",
            "MODEL_DIRECTION_TRADE_SAMPLE_WEIGHT_MULTIPLIERS": "",
            "MODEL_DIRECTION_HARD_NEGATIVE_SAMPLE_WEIGHT_MULTIPLIERS": "",
        }):
            with patch("train.train.config.MODEL_TRADE_SAMPLE_WEIGHT_MULTIPLIER", 1.0):
                with patch("train.train.config.MODEL_NO_TRADE_SAMPLE_WEIGHT_MULTIPLIER", 1.0):
                    sample_weight, summary = train_module.build_sample_weights(
                        X,
                        y,
                        sample_context=sample_context,
                        recent_boost=0.0,
                        min_weight=0.0,
                        max_weight=1000.0,
                    )

        hard_negative_mask = train_module.infer_hard_negative_mask(y, sample_context=sample_context)
        ordinary_no_trade_mask = (y == 0) & ~hard_negative_mask

        self.assertEqual(summary["hard_negative_multiplier"], 3.0)
        self.assertEqual(summary["hard_negative_rows"], 3)
        self.assertGreater(sample_weight.loc[hard_negative_mask].mean(), sample_weight.loc[ordinary_no_trade_mask].mean())
        self.assertGreater(sample_weight.loc[hard_negative_mask].mean(), sample_weight.loc[y == 1].mean())
        self.assertGreater(summary["hard_negative_weight_mean"], 0.0)

    def test_sample_weights_apply_direction_specific_trade_and_hard_negative_boosts(self):
        index = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        X = pd.DataFrame({
            "trend_bias_num": [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            "regime_trend_long": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "regime_trend_short": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "regime_range_high_vol": [0.0] * 8,
        }, index=index)
        y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0], index=index)
        sample_context = pd.DataFrame({
            "label_regime": [
                "trend_long", "trend_long", "trend_short", "trend_short",
                "trend_long", "trend_long", "trend_short", "trend_short",
            ],
            "label_direction": ["long", "long", "short", "short", "long", "long", "short", "short"],
            "label_outcome": ["TP", "TP", "TP", "TP", "SL", "SL", "SL", "SL"],
            "label_reject_reason": [
                "accepted", "accepted", "accepted", "accepted",
                "outcome_sl", "outcome_sl", "outcome_sl", "outcome_sl",
            ],
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_HARD_NEGATIVE_SAMPLE_WEIGHT_MULTIPLIER": "1.0",
            "MODEL_DIRECTION_TRADE_SAMPLE_WEIGHT_MULTIPLIERS": "long:trend_long=2.0",
            "MODEL_DIRECTION_HARD_NEGATIVE_SAMPLE_WEIGHT_MULTIPLIERS": "short:trend_short=3.0",
        }):
            with patch("train.train.config.MODEL_TRADE_SAMPLE_WEIGHT_MULTIPLIER", 1.0):
                with patch("train.train.config.MODEL_NO_TRADE_SAMPLE_WEIGHT_MULTIPLIER", 1.0):
                    sample_weight, summary = train_module.build_sample_weights(
                        X,
                        y,
                        sample_context=sample_context,
                        recent_boost=0.0,
                        min_weight=0.0,
                        max_weight=1000.0,
                    )

        long_trade_mask = (y == 1) & (sample_context["label_direction"] == "long")
        short_trade_mask = (y == 1) & (sample_context["label_direction"] == "short")
        long_hard_negative_mask = (
            (y == 0)
            & (sample_context["label_direction"] == "long")
            & (sample_context["label_outcome"] == "SL")
        )
        short_hard_negative_mask = (
            (y == 0)
            & (sample_context["label_direction"] == "short")
            & (sample_context["label_outcome"] == "SL")
        )

        self.assertGreater(sample_weight.loc[long_trade_mask].mean(), sample_weight.loc[short_trade_mask].mean())
        self.assertGreater(
            sample_weight.loc[short_hard_negative_mask].mean(),
            sample_weight.loc[long_hard_negative_mask].mean(),
        )
        self.assertEqual(summary["direction_trade_multipliers"], {"long:trend_long": 2.0})
        self.assertEqual(summary["direction_hard_negative_multipliers"], {"short:trend_short": 3.0})
        self.assertEqual(summary["direction_trade_multiplier_effects"]["long:trend_long"]["rows"], 2)
        self.assertEqual(summary["direction_hard_negative_multiplier_effects"]["short:trend_short"]["rows"], 2)

    def test_direction_weight_defaults_do_not_boost_trade_samples(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(train_module._direction_trade_sample_weight_multipliers(), {})
            self.assertEqual(
                train_module._direction_hard_negative_sample_weight_multipliers(),
                {("long", "trend_long"): 2.0, ("short", "trend_short"): 2.0},
            )

        with patch.dict(os.environ, {
            "MODEL_DIRECTION_TRADE_SAMPLE_WEIGHT_MULTIPLIERS": "",
            "MODEL_DIRECTION_HARD_NEGATIVE_SAMPLE_WEIGHT_MULTIPLIERS": "",
        }, clear=False):
            trade_multipliers = train_module._direction_trade_sample_weight_multipliers()
            hard_negative_multipliers = train_module._direction_hard_negative_sample_weight_multipliers()

        self.assertEqual(trade_multipliers, {})
        self.assertEqual(hard_negative_multipliers, {})

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

    def test_direction_quality_inverse_calibration_is_direction_limited(self):
        index = pd.date_range("2026-01-01", periods=40, freq="5min", tz="UTC")
        model_scores = [
            0.20, 0.80, 0.25, 0.75, 0.30,
            0.70, 0.35, 0.65, 0.40, 0.60,
            0.45, 0.55, 0.50, 0.58, 0.42,
        ]
        inverse_calibration_scores = [0.10, 0.20, 0.80, 0.90, 0.95]
        direction_scores = model_scores + inverse_calibration_scores
        model_targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        inverse_calibration_targets = [1, 1, 0, 0, 0]
        direction_targets = model_targets + inverse_calibration_targets
        X = pd.DataFrame({
            "score": direction_scores + direction_scores,
            "trend_bias_num": [1.0] * 20 + [-1.0] * 20,
            "regime_trend_long": [1.0] * 20 + [0.0] * 20,
            "regime_trend_short": [0.0] * 20 + [1.0] * 20,
            "regime_range_high_vol": [0.0] * 40,
        }, index=index)
        y = pd.Series(direction_targets + direction_targets, index=index)
        context = pd.DataFrame({
            "label_direction": ["long"] * 20 + ["short"] * 20,
            "label_regime": ["trend_long"] * 20 + ["trend_short"] * 20,
            "label_outcome": ["TP" if value else "SL" for value in y],
            "label_reject_reason": ["accepted" if value else "outcome_sl" for value in y],
        }, index=index)

        with patch.dict(os.environ, {
            "MODEL_DIRECTION_QUALITY_ALLOW_INVERSE_CALIBRATION": "1",
            "MODEL_DIRECTION_QUALITY_INVERSE_CALIBRATION_DIRECTIONS": "short",
        }):
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

        long_calibration = summary["directions"]["long"]["calibration"]["lgb_v1"]
        short_calibration = summary["directions"]["short"]["calibration"]["lgb_v1"]
        self.assertEqual(summary["inverse_calibration_directions"], ["short"])
        self.assertFalse(summary["directions"]["long"]["allow_inverse_calibration"])
        self.assertTrue(summary["directions"]["short"]["allow_inverse_calibration"])
        self.assertFalse(long_calibration["active"])
        self.assertEqual(long_calibration["fallback_reason"], "calibration_negative_slope")
        self.assertTrue(short_calibration["active"])
        self.assertTrue(short_calibration["inverted"])
        self.assertEqual(models["lgb_v1"].calibrated_directions, ["short"])
        self.assertEqual(models["lgb_v1"].calibrated_direction_regimes, ["short:trend_short"])

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
