import sys
import types
import unittest

import pandas as pd

try:
    import okx.Account  # noqa: F401
except ModuleNotFoundError:
    fake_okx = types.ModuleType("okx")
    fake_okx.__path__ = []
    sys.modules["okx"] = fake_okx
    for name in ("Account", "Trade", "MarketData", "PublicData", "TradingData"):
        module = types.ModuleType(f"okx.{name}")
        sys.modules[f"okx.{name}"] = module

from run import audit_market_data as audit


def build_frame(index, closes=None, confirm="1"):
    closes = closes or [100.0 + i for i in range(len(index))]
    rows = []
    for ts, close in zip(index, closes):
        rows.append({
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
            "confirm": confirm,
        })
    return pd.DataFrame(rows).set_index("timestamp")


class MarketDataAuditTests(unittest.TestCase):
    def test_audit_ohlcv_detects_gap_duplicate_unconfirmed_and_invalid_rows(self):
        index = pd.to_datetime([
            "2026-01-01 00:00:00+00:00",
            "2026-01-01 00:05:00+00:00",
            "2026-01-01 00:05:00+00:00",
            "2026-01-01 00:20:00+00:00",
        ])
        frame = build_frame(index)
        frame.iloc[2, frame.columns.get_loc("confirm")] = "0"
        frame.iloc[3, frame.columns.get_loc("high")] = 50.0

        report = audit.audit_ohlcv_frame(
            frame,
            "5m",
            now_ts=pd.Timestamp("2026-01-01 01:00:00+00:00"),
        )
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "error")
        self.assertIn("duplicate_timestamps", codes)
        self.assertIn("missing_bars", codes)
        self.assertIn("invalid_ohlcv", codes)
        self.assertIn("unconfirmed_exchange_bars", codes)
        self.assertEqual(report["missing_bars"], 2)

    def test_cross_interval_alignment_flags_close_mismatch(self):
        base_index = pd.date_range("2026-01-01 00:00:00", periods=6, freq="5min", tz="UTC")
        high_index = pd.date_range("2026-01-01 00:00:00", periods=2, freq="15min", tz="UTC")
        base = build_frame(base_index, closes=[100, 101, 102, 103, 104, 105])
        high = build_frame(high_index, closes=[102, 999])

        report = audit.build_cross_interval_alignment(
            {"5m": base, "15m": high},
            base_interval="5m",
            tolerance_pct=0.02,
        )
        item = report["results"]["15m"]

        self.assertEqual(report["status"], "warning")
        self.assertEqual(item["rows"], 2)
        self.assertEqual(item["breach_count"], 1)
        self.assertEqual(item["issues"][0]["code"], "cross_interval_close_mismatch")

    def test_compare_close_frames_reports_price_deviation(self):
        index = pd.date_range("2026-01-01 00:00:00", periods=3, freq="5min", tz="UTC")
        okx = build_frame(index, closes=[100.0, 101.0, 102.0])
        binance = build_frame(index, closes=[100.0, 101.0, 110.0])

        report = audit.compare_close_frames(
            okx,
            binance,
            left_name="okx",
            right_name="binance",
            tolerance_pct=1.0,
        )

        self.assertEqual(report["status"], "warning")
        self.assertEqual(report["breach_count"], 1)
        self.assertEqual(report["issues"][0]["code"], "price_deviation_above_tolerance")

    def test_okx_symbol_to_binance_removes_swap_suffix(self):
        self.assertEqual(audit.okx_symbol_to_binance("SOL-USDT-SWAP"), "SOLUSDT")


if __name__ == "__main__":
    unittest.main()
