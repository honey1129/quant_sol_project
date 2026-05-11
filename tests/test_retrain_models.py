import unittest
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

if "joblib" not in sys.modules:
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
        with self.assertRaisesRegex(RuntimeError, "平仓交易数不足"):
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


if __name__ == "__main__":
    unittest.main()
