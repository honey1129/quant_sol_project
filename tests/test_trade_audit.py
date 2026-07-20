import json
import os
import tempfile
import unittest

from utils.trade_audit import (
    append_trade_record,
    build_trade_record,
    load_trade_records,
    trade_record_exists,
    write_daily_report,
)


class TradeAuditTests(unittest.TestCase):
    def test_build_trade_record_extracts_fill_and_pnl(self):
        order = {
            "ordId": "123",
            "clOrdId": "cid123",
            "state": "filled",
            "side": "sell",
            "posSide": "long",
            "reduceOnly": "true",
            "avgPx": "105",
            "accFillSz": "2",
            "_fills": [
                {"fillPx": "104", "fillSz": "1", "fee": "-0.052", "feeCcy": "USDT"},
                {"fillPx": "106", "fillSz": "1", "fee": "-0.053", "feeCcy": "USDT"},
            ],
        }

        record = build_trade_record(
            order,
            bar_ts="2026-05-05T00:00:00Z",
            action="CLOSE",
            reason="TP/SL",
            delta_qty=-2.0,
            reference_price=103.0,
            pos_qty_before=2.0,
            entry_price_before=100.0,
            pos_qty_after=0.0,
            entry_price_after=0.0,
            account_before={"total_eq": 1000, "avail_eq": 900, "sizing_eq": 950},
            account_after={"total_eq": 1010, "avail_eq": 910, "sizing_eq": 960},
            signal_snapshot={"long_prob": 0.6, "trend_bias": "long", "trend_gap": 0.01},
            decision={"action": "CLOSE"},
        )

        self.assertEqual(record["fill_size"], 2.0)
        self.assertAlmostEqual(record["fill_price"], 105.0)
        self.assertAlmostEqual(record["gross_realized_pnl"], 10.0)
        self.assertAlmostEqual(record["fee_abs"], 0.105)
        self.assertAlmostEqual(record["net_realized_pnl"], 9.895)
        self.assertEqual(record["trade_date"], "2026-05-05")
        self.assertAlmostEqual(record["avail_eq_before"], 900.0)
        self.assertAlmostEqual(record["sizing_eq_after"], 960.0)
        self.assertEqual(record["risk_context"]["trend_bias"], "long")
        self.assertEqual(record["schema_version"], 2)

    def test_build_trade_record_calculates_realtime_execution_quality(self):
        record = build_trade_record(
            {
                "state": "filled",
                "side": "sell",
                "posSide": "long",
                "avgPx": "98.7",
                "accFillSz": "2",
                "fillTime": "1784304000250",
            },
            bar_ts="2026-07-17T16:00:00Z",
            action="CLOSE",
            reason="StopLossRealtime",
            delta_qty=-2.0,
            reference_price=98.8,
            pos_qty_before=2.0,
            entry_price_before=100.0,
            pos_qty_after=0.0,
            entry_price_after=0.0,
            account_before={},
            account_after={},
            signal_snapshot={},
            decision={"source": "local_realtime_risk"},
            execution_context={
                "trigger_source": "local_realtime_risk",
                "trigger_type": "sl",
                "trigger_detected_at": "1784304000000",
                "trigger_price": 98.8,
                "threshold_price": 99.0,
                "order_round_trip_ms": 180.5,
            },
        )

        quality = record["execution_quality"]
        self.assertEqual(quality["trigger_source"], "local_realtime_risk")
        self.assertEqual(quality["trigger_type"], "sl")
        self.assertAlmostEqual(quality["trigger_to_fill_ms"], 250.0)
        self.assertAlmostEqual(quality["order_round_trip_ms"], 180.5)
        self.assertAlmostEqual(quality["detection_slippage_bps"], (99.0 - 98.8) / 99.0 * 10000)
        self.assertAlmostEqual(quality["execution_slippage_bps"], (98.8 - 98.7) / 98.8 * 10000)
        self.assertAlmostEqual(quality["threshold_to_fill_slippage_bps"], (99.0 - 98.7) / 99.0 * 10000)

    def test_append_and_load_records_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fills.jsonl")
            append_trade_record({"trade_date": "2026-05-05", "action": "OPEN"}, path=path)
            append_trade_record({"trade_date": "2026-05-05", "action": "CLOSE"}, path=path)

            records = load_trade_records(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["action"], "OPEN")

    def test_trade_record_exists_matches_exchange_order_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fills.jsonl")
            append_trade_record({"ord_id": "order-1", "cl_ord_id": "client-1"}, path=path)

            self.assertTrue(trade_record_exists(ord_id="order-1", path=path))
            self.assertTrue(trade_record_exists(cl_ord_id="client-1", path=path))
            self.assertFalse(trade_record_exists(ord_id="order-2", path=path))

    def test_daily_report_writes_markdown_and_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records_path = os.path.join(tmpdir, "fills.jsonl")
            report_dir = os.path.join(tmpdir, "reports")
            records = [
                {
                    "trade_date": "2026-05-05",
                    "executed_at": "2026-05-05T01:00:00+00:00",
                    "action": "OPEN",
                    "reason": "OpenFromFlat",
                    "pos_side": "long",
                    "notional": 100.0,
                    "gross_realized_pnl": 0.0,
                    "net_realized_pnl": -0.05,
                    "fee_abs": 0.05,
                    "slippage_value": 0.0,
                    "closed_qty": 0.0,
                    "equity_before": 1000.0,
                    "equity_after": 999.95,
                },
                {
                    "trade_date": "2026-05-05",
                    "executed_at": "2026-05-05T03:00:00+00:00",
                    "action": "CLOSE",
                    "reason": "TP/SL",
                    "pos_side": "long",
                    "notional": 105.0,
                    "gross_realized_pnl": 10.0,
                    "net_realized_pnl": 9.90,
                    "fee_abs": 0.10,
                    "slippage_value": 0.0,
                    "closed_qty": 1.0,
                    "equity_before": 999.95,
                    "equity_after": 1009.85,
                },
            ]
            with open(records_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            summary, json_path, md_path = write_daily_report(
                "2026-05-05",
                records_path=records_path,
                report_dir=report_dir,
                latest_report_path=os.path.join(tmpdir, "latest.md"),
            )

            self.assertEqual(summary["record_count"], 2)
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))
            with open(md_path, "r", encoding="utf-8") as f:
                report = f.read()
            self.assertIn("每日交易复盘", report)
            self.assertIn("阈值滑点(bps)", report)
            self.assertIn("触发到成交(ms)", report)


if __name__ == "__main__":
    unittest.main()
