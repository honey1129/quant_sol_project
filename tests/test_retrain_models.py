import unittest
import sys
import types
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

from run import retrain_models


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


if __name__ == "__main__":
    unittest.main()
