import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.okx_api import (
    OKXClient,
    build_client_order_id,
    order_is_acknowledged,
    order_is_filled,
)
from run.live_trading_monitor import (
    LiveTrader,
    load_last_bar_ts,
    load_runtime_state,
    persist_last_bar_ts,
    persist_runtime_state,
    should_emit_interval_log,
)


class LiveRuntimeStateTests(unittest.TestCase):
    def test_persist_last_bar_ts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "live_state.json")
            ts = pd.Timestamp("2026-04-23 10:15:00", tz="UTC")

            persist_last_bar_ts(state_path, ts)
            loaded = load_last_bar_ts(state_path)

            self.assertEqual(loaded, ts)

    def test_persist_runtime_state_round_trip_with_hold_bars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "live_state.json")
            ts = pd.Timestamp("2026-04-23 10:15:00", tz="UTC")

            persist_runtime_state(state_path, last_bar_ts=ts, hold_bars=7)
            state = load_runtime_state(state_path)

            self.assertEqual(state["last_bar_ts"], ts)
            self.assertEqual(state["hold_bars"], 7)

    def test_load_runtime_state_legacy_payload_without_hold_bars(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "live_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"last_bar_ts": "2026-04-23T10:15:00+00:00"}, f)

            state = load_runtime_state(state_path)

            self.assertIsNotNone(state["last_bar_ts"])
            self.assertEqual(state["hold_bars"], 0)

    def test_load_runtime_state_missing_file_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "missing.json")
            state = load_runtime_state(state_path)
            self.assertIsNone(state["last_bar_ts"])
            self.assertEqual(state["hold_bars"], 0)

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

    def test_order_is_filled_only_for_filled_states(self):
        self.assertTrue(order_is_filled({"state": "filled"}))
        self.assertTrue(order_is_filled({"state": "partially_filled"}))
        self.assertFalse(order_is_filled({"state": "live"}))
        self.assertFalse(order_is_filled({"state": "canceled"}))
        self.assertFalse(order_is_filled(None))

    def test_wait_until_filled_returns_when_state_becomes_filled(self):
        client = OKXClient.__new__(OKXClient)
        sequence = iter([
            {"state": "live", "ordId": "1"},
            {"state": "filled", "ordId": "1"},
        ])

        with patch.object(OKXClient, "get_order_by_client_id", side_effect=lambda cid: next(sequence)):
            order = client.wait_until_filled("cid", timeout_sec=2.0, poll_interval_sec=0.0)

        self.assertIsNotNone(order)
        self.assertEqual(order["state"], "filled")

    def test_wait_until_filled_returns_none_on_terminal_failure(self):
        client = OKXClient.__new__(OKXClient)

        with patch.object(OKXClient, "get_order_by_client_id", return_value={"state": "canceled"}):
            order = client.wait_until_filled("cid", timeout_sec=2.0, poll_interval_sec=0.0)

        self.assertIsNone(order)

    def test_wait_until_filled_times_out_when_never_filled(self):
        client = OKXClient.__new__(OKXClient)

        with patch.object(OKXClient, "get_order_by_client_id", return_value={"state": "live"}):
            order = client.wait_until_filled("cid", timeout_sec=0.05, poll_interval_sec=0.01)

        self.assertIsNone(order)


if __name__ == "__main__":
    unittest.main()
