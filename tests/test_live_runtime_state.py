import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.okx_api import build_client_order_id, order_is_acknowledged
from run.live_trading_monitor import (
    LiveTrader,
    load_last_bar_ts,
    persist_last_bar_ts,
    should_emit_interval_log,
)


class LiveRuntimeStateTests(unittest.TestCase):
    def test_persist_last_bar_ts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "live_state.json")
            ts = pd.Timestamp("2026-04-23 10:15:00")

            persist_last_bar_ts(state_path, ts)
            loaded = load_last_bar_ts(state_path)

            self.assertEqual(loaded, ts)

    def test_load_last_bar_ts_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "missing.json")
            self.assertIsNone(load_last_bar_ts(state_path))

    def test_should_emit_interval_log_when_never_logged(self):
        self.assertTrue(should_emit_interval_log(None, 100.0, 30.0))

    def test_should_emit_interval_log_only_after_interval(self):
        self.assertFalse(should_emit_interval_log(100.0, 120.0, 30.0))
        self.assertTrue(should_emit_interval_log(100.0, 130.0, 30.0))

    def test_write_dashboard_snapshot_preserves_last_position_when_not_provided(self):
        trader = LiveTrader.__new__(LiveTrader)
        trader.loop_count = 3
        trader.same_bar_skip_count = 2
        trader.heartbeat_log_interval_sec = 30.0
        trader.last_bar_ts = pd.Timestamp("2026-04-25 12:00:00")
        trader.last_bar_snapshot = {}
        trader.last_execution = {}
        trader.last_signal_snapshot = {"long_prob": 0.61, "short_prob": 0.39}
        trader.last_dashboard_account = {"total_eq": 1000.0, "avail_eq": 900.0}
        trader.last_position_snapshot = {"direction": "long", "net_qty": 2.5, "entry_price": 150.0}

        with patch("run.live_trading_monitor.write_runtime_dashboard_snapshot") as mock_write:
            trader._write_dashboard_snapshot(
                runtime_status="waiting_next_bar",
                latest_closed_bar_ts=pd.Timestamp("2026-04-25 12:05:00"),
                current_price=155.0,
                decision={"action": "WAIT_SAME_BAR", "reason": "SameClosedBarSkip"},
            )

        payload = mock_write.call_args.kwargs.get("snapshot") or mock_write.call_args.args[0]
        self.assertEqual(payload["position"]["direction"], "long")
        self.assertAlmostEqual(payload["position"]["net_qty"], 2.5)


class OkxOrderHelperTests(unittest.TestCase):
    def test_build_client_order_id_is_short_and_unique(self):
        cl_ord_id_1 = build_client_order_id("SOL-USDT-SWAP", "buy", "long", False)
        cl_ord_id_2 = build_client_order_id("SOL-USDT-SWAP", "buy", "long", False)

        self.assertLessEqual(len(cl_ord_id_1), 32)
        self.assertLessEqual(len(cl_ord_id_2), 32)
        self.assertNotEqual(cl_ord_id_1, cl_ord_id_2)

    def test_order_is_acknowledged_accepts_active_and_filled_states(self):
        self.assertTrue(order_is_acknowledged({"state": "live"}))
        self.assertTrue(order_is_acknowledged({"state": "partially_filled"}))
        self.assertTrue(order_is_acknowledged({"state": "filled"}))
        self.assertFalse(order_is_acknowledged({"state": "canceled"}))


if __name__ == "__main__":
    unittest.main()
