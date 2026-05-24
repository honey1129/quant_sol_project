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
from utils import utils


class ImportantNotificationTests(unittest.TestCase):
    def test_log_info_and_error_do_not_send_telegram(self):
        with patch("utils.utils.send_telegram") as mock_send:
            utils.log_info("ordinary info")
            utils.log_error("ordinary error")

        mock_send.assert_not_called()

    def test_notify_important_sends_telegram(self):
        with patch("utils.utils.send_telegram") as mock_send:
            utils.notify_important("important")

        mock_send.assert_called_once_with("important")

    def test_retrain_success_notification_contains_oos_metrics(self):
        message = retrain_models.format_retrain_success_notification(
            {
                "net_pnl_after_costs": 12.34,
                "net_return_pct_after_costs": 1.23,
                "max_drawdown_pct": -0.5,
                "win_rate_pct": 55.0,
                "profit_factor": 1.2345,
                "closed_trade_count": 12,
                "comparison": {
                    "net_pnl_after_costs": {"delta": 3.0},
                    "profit_factor": {"delta": 0.11},
                    "max_drawdown_pct": {"delta": 0.2},
                },
            },
            "/tmp/retrain.log",
        )

        self.assertIn("模型重训成功", message)
        self.assertIn("OOS手续费后收益: 12.34 USDT", message)
        self.assertIn("新旧同场差值", message)


if __name__ == "__main__":
    unittest.main()
