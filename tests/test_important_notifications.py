import unittest
import sys
import types
from collections import Counter
from unittest.mock import patch

if "joblib" not in sys.modules:
    fake_joblib = types.ModuleType("joblib")
    fake_joblib.load = lambda path: object()
    sys.modules["joblib"] = fake_joblib

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.log = lambda value: value
    fake_numpy.asarray = lambda value, dtype=None: value
    fake_numpy.zeros_like = lambda value, dtype=None: [0.0 for _ in value]
    sys.modules["numpy"] = fake_numpy

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

if "requests" not in sys.modules:
    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = fake_requests

if "okx" not in sys.modules:
    fake_okx = types.ModuleType("okx")
    sys.modules["okx"] = fake_okx
    for module_name, api_name in {
        "okx.Account": "AccountAPI",
        "okx.Trade": "TradeAPI",
        "okx.MarketData": "MarketAPI",
        "okx.PublicData": "PublicAPI",
        "okx.TradingData": "TradingDataAPI",
    }.items():
        fake_module = types.ModuleType(module_name)
        setattr(fake_module, api_name, type(api_name, (), {"__init__": lambda self, *args, **kwargs: None}))
        sys.modules[module_name] = fake_module

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
    fake_pandas.Series = type("Series", (), {})
    sys.modules["pandas"] = fake_pandas

from run import retrain_models
from run.live_trading_monitor import LiveTrader
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

    def test_runtime_summary_notification_is_human_readable(self):
        trader = LiveTrader.__new__(LiveTrader)
        trader.hold_reason_counts = Counter({"Cooldown": 10, "WeakSignal": 2})

        message = trader._format_runtime_summary_notification(
            bar_ts="2026-06-10T18:20:00+00:00",
            price=63.74,
            equity=93854.22,
            position_snapshot={
                "direction": "flat",
                "net_qty": 0,
                "entry_price": 0,
                "notional": 0,
            },
            signal_snapshot={
                "long_prob": 0.868,
                "short_prob": 0.132,
                "regime": "range_high_vol",
                "trend_bias": "neutral",
            },
            decision={
                "action": "HOLD",
                "reason": "Cooldown(27)",
                "target_ratio": 0,
                "raw_target_ratio": 0,
                "expected_net_edge": None,
                "take_profit": 0.029129,
                "stop_loss": 0.011509,
                "risk": {},
            },
        )

        self.assertIn("本轮动作: 暂不交易", message)
        self.assertIn("冷却中，剩余 27 根K线", message)
        self.assertIn("模拟盘权益（虚拟资金） 93854.22 USDT", message)
        self.assertIn("做多 86.8%", message)
        self.assertIn("行情 高波动震荡，趋势 中性", message)
        self.assertIn("最近HOLD原因: 冷却中，避免刚交易完立刻反复进出 10次", message)
        self.assertNotIn("long=0.868", message)
        self.assertNotIn("target=0.0000", message)


if __name__ == "__main__":
    unittest.main()
