import argparse
import json
import os
import tempfile
import unittest

from run import market_condition_loss_diagnostics as diag


def make_record(
    *,
    action,
    ts,
    side,
    reason,
    net_pnl,
    closed_qty,
    regime,
    trend,
    long_prob,
    short_prob,
    volatility=0.003,
    atr_ratio=0.004,
    trend_gap=0.004,
    money_flow_ratio=1.0,
):
    return {
        "action": action,
        "executed_at": ts,
        "trade_date": ts[:10],
        "pos_side": side,
        "reason": reason,
        "net_realized_pnl": net_pnl,
        "fee_abs": abs(net_pnl) * 0.01,
        "slippage_value": 1.0,
        "closed_qty": closed_qty,
        "fill_price": 100.0,
        "notional": 1000.0,
        "signal": {
            "regime": regime,
            "trend_bias": trend,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "volatility": volatility,
            "atr_ratio": atr_ratio,
            "trend_gap": trend_gap,
            "money_flow_ratio": money_flow_ratio,
        },
        "decision": {
            "market_regime": regime,
            "trend_bias": trend,
        },
    }


def make_args(path, **overrides):
    defaults = {
        "records_path": path,
        "days": 30,
        "since": None,
        "min_trades": 2,
        "min_total_loss": 100.0,
        "min_avg_loss": 40.0,
        "max_win_rate": 0.35,
        "min_profit_factor": 0.8,
        "top_n": 10,
        "dimensions": None,
        "output": None,
        "markdown_output": None,
        "print_json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class MarketConditionLossDiagnosticsTests(unittest.TestCase):
    def test_build_completed_trades_pairs_open_and_close(self):
        records = [
            make_record(
                action="OPEN",
                ts="2026-06-01T00:00:00+00:00",
                side="short",
                reason="OpenFromFlat",
                net_pnl=-5,
                closed_qty=0,
                regime="trend_short",
                trend="short",
                long_prob=0.1,
                short_prob=0.9,
            ),
            make_record(
                action="CLOSE",
                ts="2026-06-01T01:00:00+00:00",
                side="short",
                reason="StopLoss",
                net_pnl=-120,
                closed_qty=1,
                regime="range_high_vol",
                trend="neutral",
                long_prob=0.5,
                short_prob=0.5,
            ),
        ]

        trades = diag.build_completed_trades(records)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "short")
        self.assertEqual(trades[0]["entry_regime"], "trend_short")
        self.assertEqual(trades[0]["exit_regime"], "range_high_vol")
        self.assertEqual(trades[0]["regime_transition"], "trend_short->range_high_vol")
        self.assertAlmostEqual(trades[0]["net_pnl"], -125.0)
        self.assertEqual(trades[0]["entry_alignment"], "aligned")

    def test_report_recommends_blocking_lossy_short_regime(self):
        records = []
        for index in range(2):
            records.append(make_record(
                action="OPEN",
                ts=f"2026-06-0{index + 1}T00:00:00+00:00",
                side="short",
                reason="OpenFromFlat",
                net_pnl=-5,
                closed_qty=0,
                regime="trend_short",
                trend="short",
                long_prob=0.1,
                short_prob=0.9,
            ))
            records.append(make_record(
                action="CLOSE",
                ts=f"2026-06-0{index + 1}T01:00:00+00:00",
                side="short",
                reason="StopLoss",
                net_pnl=-120,
                closed_qty=1,
                regime="trend_short",
                trend="short",
                long_prob=0.1,
                short_prob=0.9,
            ))
        records.append(make_record(
            action="OPEN",
            ts="2026-06-03T00:00:00+00:00",
            side="long",
            reason="OpenFromFlat",
            net_pnl=-5,
            closed_qty=0,
            regime="trend_long",
            trend="long",
            long_prob=0.9,
            short_prob=0.1,
        ))
        records.append(make_record(
            action="CLOSE",
            ts="2026-06-03T01:00:00+00:00",
            side="long",
            reason="TakeProfit",
            net_pnl=260,
            closed_qty=1,
            regime="trend_long",
            trend="long",
            long_prob=0.9,
            short_prob=0.1,
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fills.jsonl")
            with open(path, "w", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record) + "\n")

            report = diag.build_report(make_args(path, dimensions=["entry_side_regime", "side"]))

        expressions = {item["filter_expression"]: item["action"] for item in report["recommendations"]}
        self.assertEqual(expressions["side == 'short' and entry_regime == 'trend_short'"], "block_new_entries")
        self.assertEqual(report["totals"]["trade_count"], 3)
        self.assertAlmostEqual(report["totals"]["net_pnl"], 5.0)

    def test_format_markdown_includes_recommendations(self):
        report = {
            "completed_trade_count": 2,
            "period": {"start": "2026-06-01T00:00:00+00:00", "end": "2026-06-02T00:00:00+00:00"},
            "totals": {"net_pnl": -200, "win_rate": 0.0, "profit_factor": 0.0, "fee_abs": 10},
            "recommendations": [{
                "action": "block_new_entries",
                "filter_expression": "side == 'short'",
                "reason_codes": ["total_loss_exceeds_threshold"],
                "summary": {
                    "trade_count": 2,
                    "win_rate": 0.0,
                    "net_pnl": -200,
                    "avg_net_pnl": -100,
                    "profit_factor": 0.0,
                },
            }],
        }

        markdown = diag.format_markdown(report)

        self.assertIn("Market Condition Loss Diagnostics", markdown)
        self.assertIn("side == 'short'", markdown)
        self.assertIn("block_new_entries", markdown)


if __name__ == "__main__":
    unittest.main()
