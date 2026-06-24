import unittest
import os
import sys
import tempfile
import types
from types import SimpleNamespace
from unittest.mock import patch

try:
    import joblib  # noqa: F401
except ModuleNotFoundError:
    fake_joblib = types.ModuleType("joblib")
    fake_joblib.load = lambda path: object()
    sys.modules["joblib"] = fake_joblib

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

if "requests" not in sys.modules:
    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = fake_requests

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    fake_pandas = types.ModuleType("pandas")

    class FakeTimestamp:
        def __init__(self, value):
            self.value = value
            self.tzinfo = None

        def tz_convert(self, tz):
            return self

        def tz_localize(self, tz):
            return self

    fake_pandas.Timestamp = FakeTimestamp
    sys.modules["pandas"] = fake_pandas

from run import retrain_models


HAS_REAL_PANDAS = hasattr(retrain_models.pd, "DataFrame")


def build_summary(**overrides):
    summary = {
        "return_pct": 8.0,
        "max_drawdown_pct": -2.0,
        "closed_trade_count": 40,
        "win_rate_pct": 52.5,
        "profit_factor": 1.25,
        "avg_win_loss_ratio": 1.1,
        "net_pnl_after_costs": 80.0,
    }
    summary.update(overrides)
    return summary


class RetrainBacktestValidationTests(unittest.TestCase):
    def test_validation_accepts_trade_performance_metrics(self):
        retrain_models.validate_backtest_summary(build_summary())

    def test_validation_rejects_zero_trade_model(self):
        with self.assertRaisesRegex(RuntimeError, "回测收益未达标|平仓交易数不足"):
            retrain_models.validate_backtest_summary(
                build_summary(
                    return_pct=0.0,
                    max_drawdown_pct=0.0,
                    closed_trade_count=0,
                    win_rate_pct=0.0,
                    profit_factor=0.0,
                    avg_win_loss_ratio=0.0,
                    net_pnl_after_costs=0.0,
                )
            )

    def test_validation_requires_positive_profit_factor_even_when_config_is_loose(self):
        with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_CLOSED_TRADES", 0):
            with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_PROFIT_FACTOR", 0.0):
                with self.assertRaisesRegex(RuntimeError, "盈利因子必须大于1"):
                    retrain_models.validate_backtest_summary(
                        build_summary(
                            closed_trade_count=1,
                            profit_factor=1.0,
                        )
                    )

    def test_validation_uses_configured_metric_thresholds(self):
        with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_PROFIT_FACTOR", 1.4):
            with self.assertRaisesRegex(RuntimeError, "盈利因子未达标"):
                retrain_models.validate_backtest_summary(build_summary(profit_factor=1.25))


    def test_validation_rejects_low_net_pnl_after_costs(self):
        with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_NET_PNL_AFTER_COSTS", 1.0):
            with self.assertRaisesRegex(RuntimeError, "手续费后收益未达标"):
                retrain_models.validate_backtest_summary(
                    build_summary(net_pnl_after_costs=0.5)
                )

    def test_validation_rejects_low_positive_return(self):
        with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_RETURN_PCT", 0.05):
            with self.assertRaisesRegex(RuntimeError, "回测收益未达标"):
                retrain_models.validate_backtest_summary(
                    build_summary(return_pct=0.01)
                )

    def test_validation_rejects_low_avg_win_loss_ratio(self):
        with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_AVG_WIN_LOSS_RATIO", 0.9):
            with self.assertRaisesRegex(RuntimeError, "平均盈亏比未达标"):
                retrain_models.validate_backtest_summary(
                    build_summary(avg_win_loss_ratio=0.85)
                )

    def test_new_model_must_improve_over_old_model(self):
        old_summary = build_summary(
            net_pnl_after_costs=80.0,
            profit_factor=1.25,
            max_drawdown_pct=-2.0,
        )
        new_summary = build_summary(
            net_pnl_after_costs=90.0,
            profit_factor=1.35,
            max_drawdown_pct=-1.5,
        )

        comparison = retrain_models.validate_new_model_improvement(new_summary, old_summary)

        self.assertAlmostEqual(comparison["net_pnl_after_costs"]["delta"], 10.0)
        self.assertAlmostEqual(comparison["profit_factor"]["delta"], 0.10)
        self.assertAlmostEqual(comparison["max_drawdown_pct"]["delta"], 0.5)

    def test_new_model_rejects_when_any_required_metric_is_not_better(self):
        old_summary = build_summary(
            net_pnl_after_costs=80.0,
            profit_factor=1.25,
            max_drawdown_pct=-2.0,
        )
        new_summary = build_summary(
            net_pnl_after_costs=90.0,
            profit_factor=1.25,
            max_drawdown_pct=-1.5,
        )

        with self.assertRaisesRegex(RuntimeError, "盈利因子未优于旧模型"):
            retrain_models.validate_new_model_improvement(new_summary, old_summary)

    def test_new_model_rejects_equal_infinite_profit_factor(self):
        old_summary = build_summary(profit_factor=float("inf"))
        new_summary = build_summary(
            net_pnl_after_costs=90.0,
            profit_factor=float("inf"),
            max_drawdown_pct=-1.5,
        )

        with self.assertRaisesRegex(RuntimeError, "盈利因子未优于旧模型"):
            retrain_models.validate_new_model_improvement(new_summary, old_summary)

    def test_oos_restriction_requires_training_metadata(self):
        with self.assertRaisesRegex(RuntimeError, "训练元数据缺失"):
            retrain_models.restrict_backtester_to_oos(object(), None)

    def test_preserve_candidate_training_metadata_copies_metadata_to_backup_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = os.path.join(tmpdir, "project")
            backup_dir = os.path.join(tmpdir, "backup")
            metadata_path = os.path.join(base_dir, "models", "training_metadata.json")
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as file:
                file.write('{"oos_start":"2026-01-01T00:00:00+00:00"}')

            with patch("run.retrain_models.BASE_DIR", base_dir):
                with patch("run.retrain_models.config.TRAINING_METADATA_PATH", "models/training_metadata.json"):
                    preserved_path = retrain_models.preserve_candidate_training_metadata(backup_dir)

            self.assertEqual(
                preserved_path,
                os.path.join(backup_dir, "candidate_training_metadata.json"),
            )
            with open(preserved_path, "r", encoding="utf-8") as file:
                self.assertIn("oos_start", file.read())

    def test_preserve_candidate_training_metadata_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("run.retrain_models.BASE_DIR", tmpdir):
                with patch("run.retrain_models.config.TRAINING_METADATA_PATH", "models/training_metadata.json"):
                    self.assertIsNone(
                        retrain_models.preserve_candidate_training_metadata(
                            os.path.join(tmpdir, "backup")
                        )
                    )

    @unittest.skipUnless(HAS_REAL_PANDAS, "requires pandas")
    def test_oos_restriction_cuts_data_and_funding_to_candidate_oos(self):
        pd = retrain_models.pd
        index = pd.date_range("2026-01-01", periods=8, freq="5min")
        context = SimpleNamespace(
            data=pd.DataFrame({"5m_close": range(8)}, index=index),
            price_series=None,
            funding_history=pd.DataFrame({
                "funding_time": [index[1], index[5]],
                "funding_rate": [0.0001, 0.0002],
            }),
        )

        with patch("run.retrain_models.config.MODEL_RETRAIN_MIN_OOS_ROWS", 3):
            retrain_models.restrict_backtester_to_oos(
                context,
                {"oos_start": index[4].isoformat()},
            )

        self.assertEqual(list(context.data.index), list(index[4:]))
        self.assertEqual(list(context.price_series), list(range(4, 8)))
        self.assertEqual(list(context.funding_history["funding_time"]), [index[5]])

    @unittest.skipUnless(HAS_REAL_PANDAS, "requires pandas")
    def test_walk_forward_slices_keep_purge_gap_before_each_validation_fold(self):
        pd = retrain_models.pd
        index = pd.date_range("2026-01-01", periods=100, freq="5min")
        metadata = {
            "validation_start": index[60].isoformat(),
            "validation_end": index[89].isoformat(),
            "purge_bars": 2,
        }

        with patch("run.retrain_models.config.MODEL_WALK_FORWARD_FOLDS", 3):
            with patch("run.retrain_models.config.MODEL_WALK_FORWARD_MIN_FOLDS", 2):
                with patch("run.retrain_models.config.MODEL_WALK_FORWARD_MIN_VALIDATION_ROWS", 10):
                    slices = retrain_models.build_walk_forward_slices(index, metadata)

        self.assertEqual(len(slices), 3)
        self.assertEqual(slices[0]["train_end_pos"], 58)
        self.assertEqual(slices[0]["validation_start_pos"], 60)
        self.assertEqual(slices[1]["train_end_pos"], 68)
        self.assertEqual(slices[1]["validation_start_pos"], 70)
        self.assertEqual(slices[2]["train_end_pos"], 78)
        self.assertEqual(slices[2]["validation_start_pos"], 80)

    def test_walk_forward_aggregation_uses_gross_profit_loss_not_accuracy(self):
        summary = retrain_models.aggregate_backtest_summaries([
            build_summary(
                max_drawdown_pct=-1.0,
                trade_count=4,
                closed_trade_count=3,
                winning_trade_count=2,
                losing_trade_count=1,
                gross_profit=30.0,
                gross_loss=10.0,
                net_pnl_after_costs=20.0,
                fees_paid=2.0,
                slippage_cost=1.0,
                funding_pnl=0.5,
            ),
            build_summary(
                max_drawdown_pct=-3.0,
                trade_count=3,
                closed_trade_count=2,
                winning_trade_count=1,
                losing_trade_count=1,
                gross_profit=15.0,
                gross_loss=10.0,
                net_pnl_after_costs=5.0,
                fees_paid=1.5,
                slippage_cost=0.5,
                funding_pnl=-0.2,
            ),
        ])

        self.assertEqual(summary["fold_count"], 2)
        self.assertEqual(summary["closed_trade_count"], 5)
        self.assertAlmostEqual(summary["win_rate_pct"], 60.0)
        self.assertAlmostEqual(summary["profit_factor"], 2.25)
        self.assertAlmostEqual(summary["net_pnl_after_costs"], 25.0)
        self.assertAlmostEqual(summary["max_drawdown_pct"], -3.0)
    def test_regime_gate_rejects_trend_short_long_bias(self):
        summary = build_summary(
            decision_regime_signal_summary={
                "trend_short": {
                    "rows": 100,
                    "dominant_long_count": 85,
                    "dominant_short_count": 15,
                    "dominant_long_pct": 85.0,
                    "dominant_short_pct": 15.0,
                }
            }
        )

        with patch("run.retrain_models.config.MODEL_RETRAIN_REGIME_GATE_ENABLED", True):
            with patch("run.retrain_models.config.MODEL_RETRAIN_REGIME_GATE_MIN_ROWS", 30):
                with patch("run.retrain_models.config.MODEL_RETRAIN_MAX_TREND_SHORT_LONG_DOMINANCE_PCT", 80.0):
                    with self.assertRaisesRegex(RuntimeError, "trend_short 中候选模型过度偏多"):
                        retrain_models.validate_backtest_summary(summary)

    def test_regime_gate_rejects_trend_long_short_bias(self):
        summary = build_summary(
            decision_regime_signal_summary={
                "trend_long": {
                    "rows": 100,
                    "dominant_long_count": 10,
                    "dominant_short_count": 90,
                    "dominant_long_pct": 10.0,
                    "dominant_short_pct": 90.0,
                }
            }
        )

        with patch("run.retrain_models.config.MODEL_RETRAIN_REGIME_GATE_ENABLED", True):
            with patch("run.retrain_models.config.MODEL_RETRAIN_REGIME_GATE_MIN_ROWS", 30):
                with patch("run.retrain_models.config.MODEL_RETRAIN_MAX_TREND_LONG_SHORT_DOMINANCE_PCT", 80.0):
                    with self.assertRaisesRegex(RuntimeError, "trend_long 中候选模型过度偏空"):
                        retrain_models.validate_backtest_summary(summary)

    def test_regime_gate_ignores_small_samples(self):
        summary = build_summary(
            decision_regime_signal_summary={
                "trend_short": {
                    "rows": 10,
                    "dominant_long_count": 10,
                    "dominant_short_count": 0,
                    "dominant_long_pct": 100.0,
                    "dominant_short_pct": 0.0,
                }
            }
        )

        with patch("run.retrain_models.config.MODEL_RETRAIN_REGIME_GATE_ENABLED", True):
            with patch("run.retrain_models.config.MODEL_RETRAIN_REGIME_GATE_MIN_ROWS", 30):
                retrain_models.validate_backtest_summary(summary)

    def test_walk_forward_aggregation_combines_regime_signal_summaries(self):
        summary = retrain_models.aggregate_backtest_summaries([
            build_summary(
                max_drawdown_pct=-1.0,
                trade_count=1,
                closed_trade_count=1,
                winning_trade_count=1,
                losing_trade_count=0,
                gross_profit=10.0,
                gross_loss=0.0,
                net_pnl_after_costs=10.0,
                decision_regime_signal_summary={
                    "trend_short": {
                        "rows": 40,
                        "dominant_long_count": 30,
                        "dominant_short_count": 10,
                    }
                },
            ),
            build_summary(
                max_drawdown_pct=-2.0,
                trade_count=1,
                closed_trade_count=1,
                winning_trade_count=0,
                losing_trade_count=1,
                gross_profit=0.0,
                gross_loss=5.0,
                net_pnl_after_costs=-5.0,
                decision_regime_signal_summary={
                    "trend_short": {
                        "rows": 60,
                        "dominant_long_count": 55,
                        "dominant_short_count": 5,
                    }
                },
            ),
        ])

        regime = summary["decision_regime_signal_summary"]["trend_short"]
        self.assertEqual(regime["rows"], 100)
        self.assertEqual(regime["dominant_long_count"], 85)
        self.assertAlmostEqual(regime["dominant_long_pct"], 85.0)

    @unittest.skipUnless(HAS_REAL_PANDAS, "requires pandas")
    def test_walk_forward_fold_diagnostics_include_label_and_prediction_quality(self):
        pd = retrain_models.pd

        class FakeModel:
            classes_ = [0, 1]

            def predict(self, X):
                return [1 if value >= 0.5 else 0 for value in X["feature"]]

            def predict_proba(self, X):
                rows = []
                for value in X["feature"]:
                    trade_prob = 0.8 if value >= 0.5 else 0.2
                    rows.append([1.0 - trade_prob, trade_prob])
                return rows

        index = pd.date_range("2026-01-01", periods=4, freq="5min")
        validation_df = pd.DataFrame({
            "feature": [0.8, 0.2, 0.7, 0.1],
            "target": [1, 0, 1, 0],
            "label_regime": ["trend_long", "trend_long", "trend_short", "range"],
            "label_direction": ["long", "long", "short", "none"],
            "label_outcome": ["TP", "TIMEOUT", "TP", "NO_DIRECTION"],
            "label_reject_reason": ["accepted", "outcome_timeout", "accepted", "neutral_trend"],
            "5m_close": [102.0, 102.0, 98.0, 100.0],
            "15m_ema_20": [102.0, 102.0, 98.0, 100.0],
            "15m_ema_60": [100.0, 100.0, 100.0, 100.0],
        }, index=index)
        train_df = validation_df.copy()
        diagnostics = retrain_models.build_walk_forward_fold_diagnostics(
            {"fold": 1},
            train_df,
            validation_df,
            ["feature"],
            {"fake": FakeModel()},
            {"fake": 1.0},
            {"target_schema": "binary_trade_quality"},
        )

        self.assertEqual(diagnostics["fold"], 1)
        self.assertEqual(diagnostics["validation"]["target_counts"], {"0": 2, "1": 2})
        self.assertEqual(diagnostics["ensemble"]["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertAlmostEqual(diagnostics["ensemble"]["trade_precision"], 1.0)
        self.assertAlmostEqual(diagnostics["ensemble"]["trade_recall"], 1.0)
        self.assertIn("trend_long", diagnostics["by_regime"])
        self.assertIn("signal_direction_counts", diagnostics["ensemble"])
        self.assertIn("predicted_trade_direction_counts", diagnostics["ensemble"])

    @unittest.skipUnless(HAS_REAL_PANDAS, "requires pandas")
    def test_walk_forward_fold_diagnostics_use_configurable_decision_threshold(self):
        pd = retrain_models.pd

        class CalibratedScaleModel:
            classes_ = [0, 1]

            def predict(self, X):
                return [0 for _ in range(len(X))]

            def predict_proba(self, X):
                return [[0.7, 0.3] for _ in range(len(X))]

        index = pd.date_range("2026-01-01", periods=2, freq="5min")
        validation_df = pd.DataFrame({
            "feature": [1.0, 1.0],
            "target": [1, 0],
            "5m_close": [102.0, 102.0],
            "15m_ema_20": [102.0, 102.0],
            "15m_ema_60": [100.0, 100.0],
        }, index=index)

        diagnostics = retrain_models.build_walk_forward_fold_diagnostics(
            {"fold": 1},
            validation_df.copy(),
            validation_df,
            ["feature"],
            {"fake": CalibratedScaleModel()},
            {"fake": 1.0},
            {"target_schema": "binary_trade_quality"},
            decision_threshold=0.25,
        )

        self.assertEqual(diagnostics["ensemble"]["decision_threshold"], 0.25)
        self.assertEqual(diagnostics["ensemble"]["prediction_counts"], {"1": 2})
        self.assertEqual(diagnostics["ensemble"]["confusion_matrix"], [[0, 1], [0, 1]])

    def test_walk_forward_threshold_candidates_include_low_scale_and_current_config(self):
        with patch("run.retrain_models.config.MODEL_WALK_FORWARD_THRESHOLD_SWEEP_THRESHOLDS", "0.12,0.30"):
            with patch("run.retrain_models.config.MODEL_WALK_FORWARD_THRESHOLD_SWEEP_GAPS", "0.00"):
                with patch("run.retrain_models.config.MODEL_WALK_FORWARD_THRESHOLD_SWEEP_MIN_TARGET_RATIOS", "0.005"):
                    with patch("run.retrain_models.config.MODEL_WALK_FORWARD_THRESHOLD_SWEEP_POSITION_CENTERS", "0.05"):
                        with patch("run.retrain_models.config.THRESHOLD_LONG", 0.56):
                            with patch("run.retrain_models.config.THRESHOLD_SHORT", 0.56):
                                with patch("run.retrain_models.config.SIGNAL_MIN_PROB_DIFF", 0.12):
                                    with patch("run.retrain_models.config.MIN_SIGNAL_TARGET_RATIO", 0.04):
                                        with patch("run.retrain_models.config.POSITION_PROBABILITY_CENTER", 0.45):
                                            candidates = retrain_models.build_walk_forward_threshold_candidates()

        names = {candidate["name"] for candidate in candidates}
        self.assertIn("current", names)
        self.assertIn("tl0.12_ts0.12_gap0.00_mt0.005_pc0.05", names)
        low_candidate = next(
            candidate for candidate in candidates
            if candidate["name"] == "tl0.12_ts0.12_gap0.00_mt0.005_pc0.05"
        )
        self.assertEqual(low_candidate["overrides"]["BACKTEST_MIN_ADJUST_AMOUNT"], 5.0)


if __name__ == "__main__":
    unittest.main()
