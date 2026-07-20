import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from core.okx_api import (
    OKXClient,
    OrderStateUnknownError,
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
    net_position_from_sides,
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

            persist_runtime_state(
                state_path,
                last_bar_ts=ts,
                hold_bars=7,
                cooldown_bars_remaining=3,
                reverse_signal_bars=2,
                loss_guard_exit_bars=1,
                position_qty=-2.5,
                entry_price=145.2,
                take_profit=0.02,
                stop_loss=0.01,
                active_algo_id="algo-1",
            )
            state = load_runtime_state(state_path)

            self.assertEqual(state["last_bar_ts"], ts)
            self.assertEqual(state["hold_bars"], 7)
            self.assertEqual(state["cooldown_bars_remaining"], 3)
            self.assertEqual(state["reverse_signal_bars"], 2)
            self.assertEqual(state["loss_guard_exit_bars"], 1)
            self.assertEqual(state["position_qty"], -2.5)
            self.assertEqual(state["entry_price"], 145.2)
            self.assertEqual(state["take_profit"], 0.02)
            self.assertEqual(state["stop_loss"], 0.01)
            self.assertEqual(state["active_algo_id"], "algo-1")

    def test_load_runtime_state_legacy_payload_without_hold_bars(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "live_state.json")
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump({"last_bar_ts": "2026-04-23T10:15:00+00:00"}, f)

            state = load_runtime_state(state_path)

            self.assertIsNotNone(state["last_bar_ts"])
            self.assertEqual(state["hold_bars"], 0)
            self.assertEqual(state["cooldown_bars_remaining"], 0)
            self.assertEqual(state["reverse_signal_bars"], 0)
            self.assertEqual(state["loss_guard_exit_bars"], 0)
            self.assertIsNone(state["position_qty"])
            self.assertEqual(state["active_algo_id"], "")

    def test_load_runtime_state_missing_file_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "missing.json")
            state = load_runtime_state(state_path)
            self.assertIsNone(state["last_bar_ts"])
            self.assertEqual(state["hold_bars"], 0)
            self.assertEqual(state["reverse_signal_bars"], 0)
            self.assertEqual(state["loss_guard_exit_bars"], 0)

    def test_load_last_bar_ts_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "missing.json")
            self.assertIsNone(load_last_bar_ts(state_path))

    def test_net_position_from_sides_handles_short_only_position(self):
        qty, entry = net_position_from_sides(
            {"size": 0.0, "entry_price": 0.0},
            {"size": 3.5, "entry_price": 142.0},
        )

        self.assertEqual(qty, -3.5)
        self.assertEqual(entry, 142.0)

    def test_net_position_from_sides_rejects_dual_side_position(self):
        qty, entry = net_position_from_sides(
            {"size": 1.0, "entry_price": 140.0},
            {"size": 2.0, "entry_price": 141.0},
        )

        self.assertIsNone(qty)
        self.assertIsNone(entry)

    def test_should_emit_interval_log_when_never_logged(self):
        self.assertTrue(should_emit_interval_log(None, 100.0, 30.0))

    def test_should_emit_interval_log_only_after_interval(self):
        self.assertFalse(should_emit_interval_log(100.0, 120.0, 30.0))
        self.assertTrue(should_emit_interval_log(100.0, 130.0, 30.0))

    def test_realtime_risk_timing_tracks_actual_interval_and_slow_checks(self):
        class Core:
            @staticmethod
            def get_state():
                return 0.0, 0.0, 0

        trader = LiveTrader.__new__(LiveTrader)
        trader.core = Core()

        with patch("run.live_trading_monitor.config.POLL_SEC", 1):
            with patch("run.live_trading_monitor.config.RISK_LOOP_WARN_SEC", 3.0):
                trader._record_realtime_risk_timing(100.0, 101.0)
                trader._record_realtime_risk_timing(105.0, 106.0)

        self.assertEqual(trader.risk_check_count, 2)
        self.assertAlmostEqual(trader.risk_check_last_duration_ms, 1000.0)
        self.assertAlmostEqual(trader.risk_check_last_interval_ms, 5000.0)
        self.assertAlmostEqual(trader.risk_check_max_duration_ms, 1000.0)
        self.assertAlmostEqual(trader.risk_check_max_interval_ms, 5000.0)
        self.assertEqual(trader.risk_check_slow_count, 1)
        self.assertTrue(trader.risk_check_slow_active)

    def test_realtime_risk_timing_logs_slow_check_while_position_is_open(self):
        class Core:
            @staticmethod
            def get_state():
                return 2.0, 100.0, 1

        trader = LiveTrader.__new__(LiveTrader)
        trader.core = Core()
        trader.risk_check_last_started_at = 100.0

        with patch("run.live_trading_monitor.config.POLL_SEC", 1):
            with patch("run.live_trading_monitor.config.RISK_LOOP_WARN_SEC", 3.0):
                with patch("run.live_trading_monitor.log_error") as mock_log_error:
                    trader._record_realtime_risk_timing(105.0, 106.0)

        mock_log_error.assert_called_once()
        self.assertIn("interval=5000ms", mock_log_error.call_args.args[0])

    def test_realtime_risk_uses_fresh_websocket_position_and_price(self):
        class Client:
            def get_position(self):
                raise AssertionError("REST position should not be used")

            def get_price(self):
                raise AssertionError("REST price should not be used")

            def close_long_sz(self, qty, leverage, known_position_size=None):
                self.closed = (qty, leverage, known_position_size)
                return {"state": "filled"}

        class Stream:
            @staticmethod
            def get_position(max_age_sec):
                return (
                    {"size": 2.0, "entry_price": 100.0},
                    {"size": 0.0, "entry_price": 0.0},
                )

            @staticmethod
            def get_price(max_age_sec):
                return 98.0

        class Core:
            @staticmethod
            def get_state():
                return 2.0, 100.0, 1

            @staticmethod
            def get_risk_thresholds():
                return 0.03, 0.01

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()
        trader.realtime_stream = Stream()
        trader.core = Core()
        trader._active_algo_id = ""
        trader._tpsl_coverage_verified = True
        trader._last_tpsl_reconcile_at = None
        trader.cooldown_bars_remaining = 0
        trader.reverse_signal_bars = 0
        trader.loss_guard_exit_bars = 0
        trader.last_bar_ts = pd.Timestamp("2026-07-12T00:00:00Z")
        trader.last_execution = {}
        trader.last_dashboard_account = {}
        trader.last_signal_snapshot = {}

        with patch("run.live_trading_monitor.config.LEVERAGE", 3):
            with patch("run.live_trading_monitor.config.STOP_LOSS_COOLDOWN_BARS", 12):
                with patch.object(trader, "_sync_after_trade", return_value=True):
                    with patch.object(trader, "_persist_last_bar_state"):
                        with patch.object(trader, "_record_trade_execution", return_value={"action": "CLOSE"}):
                            with patch("run.live_trading_monitor.notify_important"):
                                triggered = trader.run_realtime_risk_check()

        self.assertTrue(triggered)
        self.assertEqual(trader.client.closed, (2.0, 3, 2.0))
        self.assertEqual(trader.last_risk_position_source, "websocket")
        self.assertEqual(trader.last_risk_price_source, "websocket")

    def test_run_once_fetches_features_before_processing_them(self):
        trader = LiveTrader.__new__(LiveTrader)
        latest_features = object()

        with patch.object(trader, "_get_latest_features", return_value=latest_features) as mock_fetch:
            with patch.object(trader, "_process_latest_features", return_value="processed") as mock_process:
                result = trader.run_once_on_new_bar()

        self.assertEqual(result, "processed")
        mock_fetch.assert_called_once_with()
        mock_process.assert_called_once_with(latest_features)

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
        trader.risk_check_count = 12
        trader.risk_check_last_duration_ms = 240.0
        trader.risk_check_last_interval_ms = 1240.0
        trader.risk_check_max_duration_ms = 800.0
        trader.risk_check_max_interval_ms = 2400.0
        trader.risk_check_slow_count = 0
        trader.risk_check_slow_active = False

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
        self.assertEqual(payload["runtime"]["risk_check_count"], 12)
        self.assertAlmostEqual(payload["runtime"]["risk_check_last_interval_ms"], 1240.0)

    def test_live_reward_risk_uses_configured_value(self):
        trader = LiveTrader.__new__(LiveTrader)
        trader.client = object()

        with patch("run.live_trading_monitor.config.KELLY_REWARD_RISK", 2.8):
            self.assertAlmostEqual(trader._load_reward_risk(), 2.8)

    def test_loss_guard_reason_is_humanized_for_short_block(self):
        text = LiveTrader._humanize_reason("LossGuardDirection(short)")

        self.assertIn("禁止做空新开仓", text)

    def test_long_entry_guard_reason_is_humanized_for_weak_gap(self):
        text = LiveTrader._humanize_reason("LongEntryGuard(weak_trend_gap=0.2500%)")

        self.assertIn("趋势强度不足", text)
        self.assertIn("暂不开多", text)

    def test_long_entry_guard_reason_is_humanized_for_overheat_money_flow(self):
        text = LiveTrader._humanize_reason("LongEntryGuard(overheat_money_flow=3.200)")

        self.assertIn("资金流过热", text)
        self.assertIn("暂不开多", text)

    def test_loss_guard_reason_key_groups_parameterized_reasons(self):
        trader = LiveTrader.__new__(LiveTrader)

        self.assertEqual(trader._hold_reason_key("LossGuardDirection(short)"), "LossGuardDirection")
        self.assertEqual(trader._hold_reason_key("LossGuardRegime(range_high_vol)"), "LossGuardRegime")
        self.assertEqual(trader._hold_reason_key("LongEntryGuard(weak_trend_gap=0.2500%)"), "LongEntryGuard")

    def test_execute_delta_blocks_short_new_entry_when_loss_guard_blocks_short(self):
        class Client:
            def get_position(self):
                return (
                    {"size": 0.0, "entry_price": 0.0},
                    {"size": 0.0, "entry_price": 0.0},
                )

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()

        with patch("run.live_trading_monitor.config.LOSS_GUARD_BLOCK_DIRECTIONS", ["short"]):
            with patch("run.live_trading_monitor.log_error") as mock_log:
                success = trader._execute_delta(
                    current_pos_qty=0.0,
                    delta_qty=-1.5,
                    decision={"reason": "LossGuardDirection(short)"},
                )

        self.assertFalse(success)
        mock_log.assert_called_once()

    def test_execute_delta_allows_short_reduction_when_loss_guard_blocks_short(self):
        class Client:
            def get_position(self):
                return (
                    {"size": 0.0, "entry_price": 0.0},
                    {"size": 2.0, "entry_price": 100.0},
                )

            def close_short_sz(self, qty, leverage):
                self.closed = (qty, leverage)
                return True

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()

        with patch("run.live_trading_monitor.config.LOSS_GUARD_BLOCK_DIRECTIONS", ["short"]):
            with patch("run.live_trading_monitor.config.LEVERAGE", 3):
                success = trader._execute_delta(current_pos_qty=-2.0, delta_qty=1.0)

        self.assertTrue(success)
        self.assertEqual(trader.client.closed, (1.0, 3))

    def test_execute_delta_rejects_when_exchange_position_changed(self):
        class Client:
            def __init__(self):
                self.opened = False

            def get_position(self):
                return (
                    {"size": 1.0, "entry_price": 100.0},
                    {"size": 0.0, "entry_price": 0.0},
                )

            def open_short_sz(self, qty, leverage):
                self.opened = True
                return True

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()

        with patch("run.live_trading_monitor.config.LOT_SIZE", 0.01):
            success = trader._execute_delta(current_pos_qty=0.0, delta_qty=-1.0)

        self.assertFalse(success)
        self.assertFalse(trader.client.opened)

    def test_startup_tpsl_reconciliation_adopts_matching_short_order(self):
        class Client:
            def list_pending_tpsl_algo_orders(self):
                return [{"algoId": "algo-short", "posSide": "short", "sz": "2.0"}]

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()
        trader._startup_position_verified = True
        trader._startup_position_qty = -2.0
        trader._startup_entry_price = 140.0
        trader._active_algo_id = ""
        trader._tpsl_coverage_verified = False

        with patch("run.live_trading_monitor.config.EXCHANGE_TPSL_ENABLED", True):
            with patch("run.live_trading_monitor.config.LOT_SIZE", 0.01):
                reconciled = trader._reconcile_exchange_tpsl_on_startup()

        self.assertTrue(reconciled)
        self.assertTrue(trader._tpsl_coverage_verified)
        self.assertEqual(trader._active_algo_id, "algo-short")

    def test_realtime_risk_check_closes_long_without_waiting_for_new_bar(self):
        class Client:
            def __init__(self):
                self.closed = None
                self.events = []

            def get_position(self):
                return (
                    {"size": 2.0, "entry_price": 100.0},
                    {"size": 0.0, "entry_price": 0.0},
                )

            def get_price(self):
                return 98.0

            def close_long_sz(self, qty, leverage, known_position_size=None):
                self.events.append("close")
                self.closed = (qty, leverage, known_position_size)
                return {"state": "filled"}

            def cancel_algo_order(self, algo_id):
                self.events.append("cancel")
                return True

        class Core:
            def get_state(self):
                return 2.0, 100.0, 1

            def get_risk_thresholds(self):
                return 0.03, 0.01

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()
        trader.core = Core()
        trader._active_algo_id = "algo-1"
        trader._tpsl_coverage_verified = True
        trader._last_tpsl_reconcile_at = None
        trader.cooldown_bars_remaining = 0
        trader.reverse_signal_bars = 0
        trader.loss_guard_exit_bars = 0
        trader.last_bar_ts = pd.Timestamp("2026-07-12T00:00:00Z")
        trader.last_execution = {}
        trader.last_dashboard_account = {}
        trader.last_signal_snapshot = {}

        with patch("run.live_trading_monitor.config.EXCHANGE_TPSL_ENABLED", True):
            with patch("run.live_trading_monitor.config.LEVERAGE", 3):
                with patch("run.live_trading_monitor.config.STOP_LOSS_COOLDOWN_BARS", 12):
                    with patch.object(trader, "_sync_after_trade", return_value=True):
                        with patch.object(trader, "_persist_last_bar_state"):
                            with patch.object(trader, "_record_trade_execution", return_value={"action": "CLOSE"}) as record:
                                with patch("run.live_trading_monitor.notify_important"):
                                    triggered = trader.run_realtime_risk_check()

        self.assertTrue(triggered)
        self.assertEqual(trader.client.closed, (2.0, 3, 2.0))
        self.assertEqual(trader.client.events, ["close", "cancel"])
        self.assertEqual(trader._active_algo_id, "")
        self.assertEqual(trader.cooldown_bars_remaining, 12)
        self.assertAlmostEqual(record.call_args.kwargs["reference_price"], 98.0)
        execution_context = record.call_args.kwargs["execution_context"]
        self.assertEqual(execution_context["trigger_source"], "local_realtime_risk")
        self.assertEqual(execution_context["trigger_type"], "sl")
        self.assertAlmostEqual(execution_context["trigger_price"], 98.0)
        self.assertAlmostEqual(execution_context["threshold_price"], 99.0)
        self.assertGreaterEqual(execution_context["order_round_trip_ms"], 0.0)

    def test_exchange_tpsl_execution_is_recorded_and_clears_algo(self):
        class Client:
            def fetch_algo_child_orders(self, algo_id):
                return (
                    {
                        "algoId": algo_id,
                        "state": "effective",
                        "actualSide": "tp",
                        "triggerTime": "1784293908300",
                        "tpTriggerPx": "73.70",
                        "slTriggerPx": "76.10",
                    },
                    [{
                        "ordId": "order-1",
                        "state": "filled",
                        "avgPx": "73.64",
                        "accFillSz": "30.83",
                        "side": "buy",
                        "posSide": "short",
                        "reduceOnly": "true",
                    }],
                )

        class Core:
            def set_state(self, *args, **kwargs):
                self.state = (args, kwargs)

        trader = LiveTrader.__new__(LiveTrader)
        trader.client = Client()
        trader.core = Core()
        trader._active_algo_id = "algo-1"
        trader._tpsl_coverage_verified = True
        trader.hold_bars = 10
        trader.cooldown_bars_remaining = 0
        trader.reverse_signal_bars = 0
        trader.loss_guard_exit_bars = 0
        trader.last_bar_ts = pd.Timestamp("2026-07-17T13:10:00Z")
        trader.last_dashboard_account = {"total_eq": 1000.0}
        trader.last_signal_snapshot = {"short_prob": 0.9}

        with patch("run.live_trading_monitor.config.LOT_SIZE", 0.01):
            with patch("run.live_trading_monitor.config.TAKE_PROFIT_COOLDOWN_BARS", 6):
                with patch("run.live_trading_monitor.trade_record_exists", return_value=False):
                    with patch.object(trader, "_get_account_snapshot", return_value={"total_eq": 1059.0}):
                        with patch.object(trader, "_record_trade_execution", return_value={}) as record:
                            with patch.object(trader, "_persist_last_bar_state"):
                                with patch("run.live_trading_monitor.notify_important"):
                                    reconciled = trader._reconcile_exchange_tpsl_execution(
                                        0.0,
                                        0.0,
                                        tracked_pos_qty=-30.83,
                                        tracked_entry_price=75.63,
                                    )

        self.assertTrue(reconciled)
        self.assertEqual(trader._active_algo_id, "")
        self.assertTrue(trader._tpsl_coverage_verified)
        self.assertEqual(trader.cooldown_bars_remaining, 6)
        self.assertEqual(record.call_args.kwargs["reason"], "TakeProfitExchange")
        self.assertAlmostEqual(record.call_args.kwargs["delta_qty"], 30.83)
        self.assertAlmostEqual(record.call_args.kwargs["reference_price"], 73.70)
        execution_context = record.call_args.kwargs["execution_context"]
        self.assertEqual(execution_context["trigger_source"], "exchange_oco")
        self.assertEqual(execution_context["trigger_type"], "tp")
        self.assertEqual(execution_context["trigger_detected_at"], "1784293908300")
        self.assertAlmostEqual(execution_context["threshold_price"], 73.70)


class OkxOrderHelperTests(unittest.TestCase):
    def test_reduce_only_size_order_skips_preflight_query_on_first_attempt(self):
        client = OKXClient.__new__(OKXClient)

        class TradeAPI:
            @staticmethod
            def place_order(**kwargs):
                return {"code": "0", "data": [{"ordId": "order-1"}]}

        client.trade_api = TradeAPI()
        with patch("core.okx_api.config.LOT_SIZE", 0.01):
            with patch.object(client, "get_order_by_client_id", side_effect=AssertionError("unexpected preflight query")):
                with patch.object(client, "wait_until_filled", return_value={"state": "filled"}):
                    result = client.place_order_with_size(
                        "sell",
                        "long",
                        2.0,
                        3,
                        reduce_only=True,
                        max_retry=1,
                    )

        self.assertEqual(result["state"], "filled")

    def test_close_long_uses_known_position_without_rest_query(self):
        client = OKXClient.__new__(OKXClient)
        with patch.object(client, "get_position", side_effect=AssertionError("unexpected position query")):
            with patch.object(client, "place_order_with_size", return_value={"state": "filled"}) as place:
                result = client.close_long_sz(2.0, 3, known_position_size=2.5)

        self.assertEqual(result["state"], "filled")
        place.assert_called_once_with("sell", "long", 2.0, 3, reduce_only=True)

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
        self.assertFalse(order_is_filled({"state": "partially_filled"}))
        self.assertFalse(order_is_filled({"state": "live"}))
        self.assertFalse(order_is_filled({"state": "canceled"}))
        self.assertFalse(order_is_filled(None))

    def test_cancel_algo_order_accepts_already_terminal_51400(self):
        client = OKXClient.__new__(OKXClient)
        client.trade_api = type("TradeAPI", (), {
            "cancel_algo_order": staticmethod(
                lambda params: {"code": "1", "data": [{"sCode": "51400"}]}
            )
        })()
        client._call_with_retry = lambda _label, func: func()

        self.assertTrue(client.cancel_algo_order("algo-filled"))

    def test_fetch_algo_child_orders_deduplicates_fallback_order_id(self):
        client = OKXClient.__new__(OKXClient)
        detail = {"algoId": "algo-1", "ordId": "order-1", "ordIdList": ["order-1"]}

        with patch.object(OKXClient, "get_algo_order_details", return_value=detail):
            with patch.object(OKXClient, "get_order_by_id", return_value={"ordId": "order-1"}) as get_order:
                returned_detail, orders = client.fetch_algo_child_orders("algo-1")

        self.assertEqual(returned_detail, detail)
        self.assertEqual(orders, [{"ordId": "order-1"}])
        get_order.assert_called_once_with("order-1")

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

        class TradeAPI:
            def __init__(self):
                self.canceled = False

            def cancel_order(self, **kwargs):
                self.canceled = True
                return {"code": "0", "data": [{"sCode": "0"}]}

        client.trade_api = TradeAPI()

        def get_order(_):
            state = "canceled" if client.trade_api.canceled else "live"
            return {"state": state, "ordId": "1"}

        with patch.object(OKXClient, "get_order_by_client_id", side_effect=get_order):
            order = client.wait_until_filled(
                "cid",
                timeout_sec=0.01,
                poll_interval_sec=0.0,
                cancel_confirm_timeout_sec=0.05,
            )

        self.assertIsNone(order)
        self.assertTrue(client.trade_api.canceled)

    def test_wait_until_filled_raises_when_cancel_never_reaches_terminal_state(self):
        client = OKXClient.__new__(OKXClient)

        class TradeAPI:
            def cancel_order(self, **kwargs):
                return {"code": "0", "data": [{"sCode": "0"}]}

        client.trade_api = TradeAPI()
        with patch.object(OKXClient, "get_order_by_client_id", return_value={"state": "live", "ordId": "1"}):
            with self.assertRaises(OrderStateUnknownError):
                client.wait_until_filled(
                    "cid",
                    timeout_sec=0.01,
                    poll_interval_sec=0.0,
                    cancel_confirm_timeout_sec=0.01,
                )

    def test_wait_until_filled_returns_canceled_partial_fill(self):
        client = OKXClient.__new__(OKXClient)
        order_payload = {"state": "canceled", "ordId": "1", "accFillSz": "0.5"}

        with patch.object(OKXClient, "get_order_by_client_id", return_value=order_payload):
            order = client.wait_until_filled("cid", timeout_sec=1.0, poll_interval_sec=0.0)

        self.assertTrue(order["_partial_fill"])
        self.assertEqual(order["accFillSz"], "0.5")

    def test_wait_until_filled_remembers_partial_fill_when_terminal_payload_omits_size(self):
        client = OKXClient.__new__(OKXClient)
        sequence = iter([
            {"state": "partially_filled", "ordId": "1", "accFillSz": "0.5"},
            {"state": "canceled", "ordId": "1"},
        ])

        with patch.object(OKXClient, "get_order_by_client_id", side_effect=lambda _: next(sequence)):
            order = client.wait_until_filled("cid", timeout_sec=1.0, poll_interval_sec=0.0)

        self.assertTrue(order["_partial_fill"])
        self.assertEqual(order["accFillSz"], "0.5")
        self.assertEqual(order["state"], "canceled")


if __name__ == "__main__":
    unittest.main()
