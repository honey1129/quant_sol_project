import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from monitoring.hourly_performance_report import (
    format_performance_report,
    parse_live_fills,
)


class HourlyPerformanceReportTests(unittest.TestCase):
    def test_report_uses_simple_summary_and_realized_usdt(self):
        stats_24h = {
            "trades": [
                {
                    "time": datetime(2026, 6, 27, 3, 31, tzinfo=timezone.utc),
                    "reason": "TakeProfit",
                    "net_pnl": 507.33,
                },
                {
                    "time": datetime(2026, 6, 26, 22, 25, tzinfo=timezone.utc),
                    "reason": "TakeProfit",
                    "net_pnl": 646.16,
                },
                {
                    "time": datetime(2026, 6, 26, 18, 0, tzinfo=timezone.utc),
                    "reason": "StopLoss",
                    "net_pnl": -253.29,
                },
            ],
            "take_profits": 2,
            "stop_losses": 1,
            "profit_count": 2,
            "loss_count": 1,
            "flat_count": 0,
            "total_trades": 3,
            "win_rate": 66.666,
            "total_pnl": 0.96,
            "net_pnl": 900.20,
            "return_pct": 0.96,
            "avg_pnl": 300.066,
        }
        stats_today = {
            "trades": [stats_24h["trades"][0]],
            "take_profits": 1,
            "stop_losses": 0,
            "profit_count": 1,
            "loss_count": 0,
            "flat_count": 0,
            "total_trades": 1,
            "win_rate": 100.0,
            "total_pnl": 0.54,
            "net_pnl": 507.33,
            "return_pct": 0.54,
            "avg_pnl": 507.33,
        }

        report = format_performance_report(stats_24h, stats_today, "简单规则模式")

        self.assertIn("结论: 最近24小时盈利，2赚1亏，含手续费净盈亏 +900.20 USDT。", report)
        self.assertIn("交易: 3笔 | 2赚1亏 | 胜率 66.7%", report)
        self.assertIn("含手续费净盈亏（区间内全部成交）: +900.20 USDT (+0.96%)", report)
        self.assertIn("06-27 11:31 止盈 平仓记录净PnL +507.33 USDT", report)
        self.assertIn("06-27 02:00 止损 平仓记录净PnL -253.29 USDT", report)
        self.assertIn("提示: 当前样本量较少，仅记录结果，不据此调整策略参数。", report)
        self.assertNotIn("N/A", report)
        self.assertNotIn("回测基线", report)
        self.assertNotIn("优秀", report)

    def test_report_handles_no_trades_plainly(self):
        empty = {
            "trades": [],
            "take_profits": 0,
            "stop_losses": 0,
            "profit_count": 0,
            "loss_count": 0,
            "flat_count": 0,
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "net_pnl": 0.0,
            "return_pct": None,
            "avg_pnl": None,
        }

        report = format_performance_report(empty, empty, "简单规则模式")

        self.assertIn("结论: 最近24小时没有平仓交易", report)
        self.assertIn("最近24小时: 暂无平仓交易", report)

    def test_parse_live_fills_summarizes_closed_trades(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "live_fills.jsonl")
            rows = [
                {
                    "executed_at": "2026-06-26T22:00:00+00:00",
                    "action": "OPEN",
                    "equity_before": 93000.0,
                    "net_realized_pnl": -8.0,
                },
                {
                    "executed_at": "2026-06-26T23:00:00+00:00",
                    "action": "CLOSE",
                    "reason": "TakeProfit",
                    "net_realized_pnl": 120.0,
                    "equity_before": 93000.0,
                    "pos_side": "long",
                },
                {
                    "executed_at": "2026-06-27T01:00:00+00:00",
                    "action": "CLOSE",
                    "reason": "StopLoss",
                    "net_realized_pnl": -30.0,
                    "equity_before": 93120.0,
                    "pos_side": "long",
                },
            ]
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            stats = parse_live_fills(
                path,
                hours=24,
                now=datetime(2026, 6, 27, 2, 0, tzinfo=timezone.utc),
            )

        self.assertEqual(stats["total_trades"], 2)
        self.assertEqual(stats["profit_count"], 1)
        self.assertEqual(stats["loss_count"], 1)
        self.assertAlmostEqual(stats["net_pnl"], 82.0)
        self.assertAlmostEqual(stats["win_rate"], 50.0)


if __name__ == "__main__":
    unittest.main()
