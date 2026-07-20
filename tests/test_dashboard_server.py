import os
import tempfile
import unittest
from unittest.mock import patch

from run import dashboard_server


class DashboardServerTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.env_path = os.path.join(self.tmpdir.name, ".env")
        with open(self.env_path, "w", encoding="utf-8") as f:
            f.write(
                "INTERVALS=5m,15m,1H\n"
                "WINDOWS=5m:5000,15m:5000,1H:2000\n"
                "MA_PERIOD=34\n"
                "RSI_PERIOD=14\n"
                "ATR_STOP_LOSS_MULTIPLIER=2.2\n"
                "STOP_LOSS=0.01\n"
                "TAKE_PROFIT=0.02\n"
                "POSITION_SIZE=50\n"
                "LEVERAGE=3\n"
            )
        self.original_build_dashboard_bundle = dashboard_server.build_dashboard_bundle
        self.original_env_values = {
            "INTERVALS": os.environ.get("INTERVALS"),
            "WINDOWS": os.environ.get("WINDOWS"),
            "MA_PERIOD": os.environ.get("MA_PERIOD"),
            "RSI_PERIOD": os.environ.get("RSI_PERIOD"),
            "ATR_STOP_LOSS_MULTIPLIER": os.environ.get("ATR_STOP_LOSS_MULTIPLIER"),
            "STOP_LOSS": os.environ.get("STOP_LOSS"),
            "TAKE_PROFIT": os.environ.get("TAKE_PROFIT"),
            "POSITION_SIZE": os.environ.get("POSITION_SIZE"),
            "LEVERAGE": os.environ.get("LEVERAGE"),
        }
        self.original_config_values = {
            "INTERVALS": list(getattr(dashboard_server.config, "INTERVALS", [])),
            "WINDOWS": dict(getattr(dashboard_server.config, "WINDOWS", {})),
            "MA_PERIOD": getattr(dashboard_server.config, "MA_PERIOD", 34),
            "RSI_PERIOD": getattr(dashboard_server.config, "RSI_PERIOD", 14),
            "ATR_STOP_LOSS_MULTIPLIER": getattr(dashboard_server.config, "ATR_STOP_LOSS_MULTIPLIER", 2.2),
            "STOP_LOSS": getattr(dashboard_server.config, "STOP_LOSS", 0.01),
            "TAKE_PROFIT": getattr(dashboard_server.config, "TAKE_PROFIT", 0.02),
            "POSITION_SIZE": getattr(dashboard_server.config, "POSITION_SIZE", 50),
            "LEVERAGE": getattr(dashboard_server.config, "LEVERAGE", 3),
        }

    def tearDown(self):
        dashboard_server.build_dashboard_bundle = self.original_build_dashboard_bundle
        for key, value in self.original_config_values.items():
            setattr(dashboard_server.config, key, value)
        for key, value in self.original_env_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmpdir.cleanup()

    def test_enrich_status_with_log_fallbacks(self):
        log_lines = [
            "2026-04-24 08:08:38,751 - INFO - 已恢复最近处理 bar: 2026-04-24 06:00:00",
            "2026-04-24 08:10:34,081 - INFO - 新bar=2026-04-24 06:05:00 price=85.6200 long=0.389 short=0.611 mf=0.118 vol=0.000992 atr_ratio=0.1802%",
            "2026-04-24 10:47:40,655 - ERROR - 实盘循环异常: ❌ 无法拉取任何K线数据，请检查API权限/网络",
        ]

        status = dashboard_server.enrich_status_with_fallbacks({}, log_lines, log_lines)

        self.assertEqual(status["runtime"]["last_status"], "error")
        self.assertIn("无法拉取任何K线数据", status["runtime"]["last_error"])
        self.assertAlmostEqual(status["market"]["last_price"], 85.62)
        self.assertAlmostEqual(status["signal"]["long_prob"], 0.389)
        self.assertAlmostEqual(status["signal"]["short_prob"], 0.611)
        self.assertEqual(status["bar"]["last_processed_bar_ts"], "2026-04-23T22:00:00+00:00")
        self.assertEqual(status["bar"]["latest_closed_bar_ts"], "2026-04-23T22:05:00+00:00")

    def test_parse_latest_backtest_summary_uses_latest_block(self):
        log_lines = [
            "2026-04-24 10:06:04,692 - INFO - 回测完成 ✅",
            "2026-04-24 10:06:04,692 - INFO - 累计收益: 165.50 USDT (16.55%)",
            "2026-04-24 10:06:04,692 - INFO - 最大回撤: -0.19%",
            "2026-04-24 10:06:04,692 - INFO - 交易次数: 645",
            "2026-04-24 10:39:11,430 - INFO - 回测完成 ✅",
            "2026-04-24 10:39:11,430 - INFO - 期末净值: 1142.97 USDT",
            "2026-04-24 10:39:11,430 - INFO - 累计收益: 142.97 USDT (14.30%)",
            "2026-04-24 10:39:11,430 - INFO - 最大回撤: -0.31%",
            "2026-04-24 10:39:11,430 - INFO - 交易次数: 425",
            "2026-04-24 10:39:11,430 - INFO - 平仓交易数: 120",
            "2026-04-24 10:39:11,430 - INFO - 胜率: 51.67%",
            "2026-04-24 10:39:11,430 - INFO - 盈利因子: 1.2345",
            "2026-04-24 10:39:11,430 - INFO - 平均盈亏比: 1.1200",
            "2026-04-24 10:39:11,430 - INFO - 平均平仓净PnL: 1.19 USDT",
            "2026-04-24 10:39:11,430 - INFO - 手续费后收益: 142.97 USDT (14.30%)",
            "2026-04-24 10:39:11,430 - INFO - 手续费合计: 27.02 USDT",
            "2026-04-24 10:39:11,430 - INFO - 滑点成本合计: 14.96 USDT",
        ]

        summary = dashboard_server.parse_latest_backtest_summary(log_lines)

        self.assertAlmostEqual(summary["final_equity"], 1142.97)
        self.assertAlmostEqual(summary["return_pct"], 14.30)
        self.assertAlmostEqual(summary["max_drawdown_pct"], -0.31)
        self.assertEqual(summary["trade_count"], 425)
        self.assertEqual(summary["closed_trade_count"], 120)
        self.assertAlmostEqual(summary["win_rate_pct"], 51.67)
        self.assertAlmostEqual(summary["profit_factor"], 1.2345)
        self.assertAlmostEqual(summary["avg_win_loss_ratio"], 1.12)
        self.assertAlmostEqual(summary["avg_closed_trade_pnl"], 1.19)
        self.assertAlmostEqual(summary["net_pnl_after_costs"], 142.97)
        self.assertAlmostEqual(summary["net_return_pct_after_costs"], 14.30)
        self.assertAlmostEqual(summary["fees_paid"], 27.02)
        self.assertAlmostEqual(summary["slippage_cost"], 14.96)

    def test_build_metrics_snapshot_counts_single_position_as_one(self):
        metrics = dashboard_server.build_metrics_snapshot(
            {
                "account": {"total_eq": 1000},
                "performance": {},
                "position": {"direction": "long", "net_qty": 2.5},
            },
            history=[],
            risk_snapshot={"risk_level": "Low"},
            backtest_summary={},
            backtest_csv_metrics={},
        )

        self.assertEqual(metrics["open_positions"], 1)

    def test_build_metrics_snapshot_counts_mixed_position_legs(self):
        metrics = dashboard_server.build_metrics_snapshot(
            {
                "account": {"total_eq": 1000},
                "performance": {},
                "position": {
                    "direction": "mixed",
                    "net_qty": 0,
                    "long_qty": 3.0,
                    "short_qty": 3.0,
                },
            },
            history=[],
            risk_snapshot={"risk_level": "High"},
            backtest_summary={},
            backtest_csv_metrics={},
        )

        self.assertEqual(metrics["open_positions"], 2)

    def test_parse_recent_trade_rows_returns_latest_first(self):
        log_lines = [
            "2026-04-24 10:10:00,000 - INFO - 执行开仓: target_ratio=0.240, qty=1.250000",
            "2026-04-24 10:15:00,000 - INFO - 执行调仓: delta_qty=-0.450000, reason=SameDirRebalance",
            "2026-04-24 10:20:00,000 - ERROR - 调仓未成交，已重新同步仓位: delta_qty=-0.300000, reason=SameDirRebalance",
        ]
        status = {
            "market": {"symbol": "SOL-USDT-SWAP", "last_price": 86.25},
            "position": {"entry_price": 85.75},
            "signal": {"long_prob": 0.41, "short_prob": 0.59},
        }

        trades = dashboard_server.parse_recent_trade_rows(log_lines, status, limit=3)

        self.assertEqual(len(trades), 3)
        self.assertEqual(trades[0]["status"], "Canceled")
        self.assertEqual(trades[0]["side"], "Short")
        self.assertEqual(trades[0]["pnl_source"], "not_recorded")
        self.assertEqual(trades[1]["reason"], "SameDirRebalance")
        self.assertEqual(trades[1]["entry_source"], "not_recorded")
        self.assertEqual(trades[2]["side"], "Short")
        self.assertEqual(trades[2]["entry_source"], "position_snapshot")

    def test_save_strategy_params_persists_env_and_refreshes_config(self):
        dashboard_server.build_dashboard_bundle = lambda: {
            "strategy_params": dashboard_server.build_strategy_params(),
        }

        result = dashboard_server.save_strategy_params(
            {
                "timeframe": "4h",
                "maPeriod": 55,
                "rsiPeriod": 21,
                "atrMultiplier": 1.8,
                "stopLossPct": 1.5,
                "takeProfitPct": 3.8,
                "positionSizePct": 75,
                "maxLeverage": 6,
            },
            env_path=self.env_path,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["restart_required"])
        self.assertEqual(dashboard_server.config.INTERVALS[0], "4H")
        self.assertEqual(dashboard_server.config.MA_PERIOD, 55)
        self.assertEqual(dashboard_server.config.RSI_PERIOD, 21)
        self.assertAlmostEqual(dashboard_server.config.ATR_STOP_LOSS_MULTIPLIER, 1.8)
        self.assertAlmostEqual(dashboard_server.config.STOP_LOSS, 0.015)
        self.assertAlmostEqual(dashboard_server.config.TAKE_PROFIT, 0.038)
        self.assertAlmostEqual(dashboard_server.config.POSITION_SIZE, 75.0)
        self.assertEqual(dashboard_server.config.LEVERAGE, 6)
        self.assertEqual(result["saved_params"]["timeframe"], "4H")

        with open(self.env_path, "r", encoding="utf-8") as f:
            env_text = f.read()

        self.assertIn("INTERVALS=4H,5m,15m,1H", env_text)
        self.assertIn("MA_PERIOD=55", env_text)
        self.assertIn("RSI_PERIOD=21", env_text)
        self.assertIn("ATR_STOP_LOSS_MULTIPLIER=1.8", env_text)
        self.assertIn("STOP_LOSS=0.015", env_text)
        self.assertIn("TAKE_PROFIT=0.038", env_text)
        self.assertIn("POSITION_SIZE=75", env_text)
        self.assertIn("LEVERAGE=6", env_text)

    def test_validate_strategy_params_payload_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            dashboard_server.validate_strategy_params_payload(
                {
                    "timeframe": "bad",
                    "maPeriod": 0,
                    "rsiPeriod": 0,
                    "atrMultiplier": 0,
                    "stopLossPct": -1,
                    "takeProfitPct": 0,
                    "positionSizePct": -5,
                    "maxLeverage": 0,
                }
            )

    def test_restart_strategy_process_uses_custom_command(self):
        with patch.dict(os.environ, {"DASHBOARD_STRATEGY_RESTART_CMD": "echo restart-ok"}, clear=False):
            with patch("run.dashboard_server.subprocess.run") as mock_run:
                with patch("run.dashboard_server.build_dashboard_bundle", return_value={"status": {"runtime": {"last_status": "starting"}}}):
                    mock_run.return_value = dashboard_server.subprocess.CompletedProcess(
                        ["echo", "restart-ok"],
                        0,
                        stdout="restart-ok\n",
                        stderr="",
                    )
                    result = dashboard_server.restart_strategy_process()

        self.assertTrue(result["ok"])
        self.assertEqual(result["command_mode"], "custom")
        self.assertEqual(result["command"], ["echo", "restart-ok"])
        self.assertIn("自定义策略重启命令", result["message"])
        mock_run.assert_called_once()

    def test_restart_strategy_process_uses_pm2_by_default(self):
        with patch.dict(os.environ, {"DASHBOARD_STRATEGY_RESTART_CMD": "", "DASHBOARD_STRATEGY_PM2_APP": "quant_okx_paper"}, clear=False):
            with patch("run.dashboard_server.shutil.which", return_value="/usr/bin/pm2"):
                with patch("run.dashboard_server.subprocess.run") as mock_run:
                    with patch("run.dashboard_server.build_dashboard_bundle", return_value={"status": {"runtime": {"last_status": "starting"}}}):
                        mock_run.return_value = dashboard_server.subprocess.CompletedProcess(
                            ["/usr/bin/pm2", "restart", "quant_okx_paper", "--update-env"],
                            0,
                            stdout="[PM2] restart triggered\n",
                            stderr="",
                        )
                        result = dashboard_server.restart_strategy_process()

        self.assertTrue(result["ok"])
        self.assertEqual(result["command_mode"], "pm2")
        self.assertEqual(result["command"], ["/usr/bin/pm2", "restart", "quant_okx_paper", "--update-env"])
        self.assertIn("PM2", result["message"])
        mock_run.assert_called_once()

    def test_restart_strategy_process_requires_pm2_or_custom_command(self):
        with patch.dict(os.environ, {"DASHBOARD_STRATEGY_RESTART_CMD": ""}, clear=False):
            with patch("run.dashboard_server.shutil.which", return_value=None):
                with self.assertRaises(RuntimeError):
                    dashboard_server.restart_strategy_process()


if __name__ == "__main__":
    unittest.main()

class DashboardObservabilityTests(unittest.TestCase):
    def test_model_observability_reports_metadata_and_failed_retrain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "models", "training_metadata.json")
            logs_dir = os.path.join(tmpdir, "logs")
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            os.makedirs(logs_dir, exist_ok=True)
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write('{"schema_version":2,"source":"unit","symbol":"SOL-USDT-SWAP","artifact_hashes":{"a":"b"},"feature_count":3,"oos_start":"2026-01-01T00:00:00+00:00"}')
            with open(os.path.join(logs_dir, "model_retrain_state.json"), "w", encoding="utf-8") as f:
                f.write('{"last_status":"failed","last_error":"gate failed","last_log_path":"logs/model_retrain_x.log"}')
            log_path = os.path.join(logs_dir, "model_retrain_x.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("FAILED: gate failed\n")

            with patch("run.dashboard_server.BASE_DIR", tmpdir), patch("run.dashboard_server.LOGS_DIR", logs_dir):
                with patch.object(dashboard_server.config, "TRAINING_METADATA_PATH", "models/training_metadata.json"):
                    snapshot = dashboard_server.build_model_observability_snapshot()

            self.assertEqual(snapshot["health"], "warning")
            self.assertIn("last_retrain_failed", snapshot["warnings"])
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertEqual(snapshot["artifact_hash_count"], 1)
            self.assertTrue(snapshot["metadata_file"]["exists"])
            self.assertTrue(snapshot["latest_retrain_log_file"]["exists"])

    def test_observability_escalates_runtime_error(self):
        with patch("run.dashboard_server.build_model_observability_snapshot", return_value={"health": "ok", "warnings": []}):
            snapshot = dashboard_server.build_observability_snapshot({"runtime": {"last_error": "boom"}})
        self.assertEqual(snapshot["health"], "error")
        self.assertEqual(snapshot["alerts"][0]["code"], "runtime_last_error")

    def test_observability_warns_on_slow_risk_loop_with_open_position(self):
        status = {
            "runtime": {
                "risk_check_slow_active": True,
                "risk_check_last_interval_ms": 5100.0,
                "risk_check_last_duration_ms": 900.0,
            },
            "position": {"net_qty": -2.0},
        }
        with patch("run.dashboard_server.build_model_observability_snapshot", return_value={"health": "ok", "warnings": []}):
            snapshot = dashboard_server.build_observability_snapshot(status)

        self.assertEqual(snapshot["health"], "warning")
        self.assertEqual(snapshot["alerts"][0]["code"], "risk_loop_latency")

    def test_observability_ignores_slow_risk_loop_while_flat(self):
        status = {
            "runtime": {"risk_check_slow_active": True},
            "position": {"net_qty": 0.0},
        }
        with patch("run.dashboard_server.build_model_observability_snapshot", return_value={"health": "ok", "warnings": []}):
            snapshot = dashboard_server.build_observability_snapshot(status)

        self.assertEqual(snapshot["health"], "ok")
        self.assertEqual(snapshot["alerts"], [])

    def test_observability_warns_on_stale_websocket_with_open_position(self):
        status = {
            "runtime": {
                "ws_ticker_connected": False,
                "ws_position_connected": True,
            },
            "position": {"net_qty": 2.0},
        }
        with patch("run.dashboard_server.build_model_observability_snapshot", return_value={"health": "ok", "warnings": []}):
            snapshot = dashboard_server.build_observability_snapshot(status)

        self.assertEqual(snapshot["health"], "warning")
        self.assertEqual(snapshot["alerts"][0]["code"], "risk_websocket_unavailable")

    def test_enrich_status_ignores_tmp_runtime_dashboard_corruption_from_tests(self):
        log_lines = [
            "2026-05-17 21:20:00,000 - ERROR - ⚠ runtime_dashboard JSON 损坏，已备份到 /tmp/tmpabc/history.json.corrupt-20260517T192008Z: Expecting value",
            "2026-05-17 21:20:10,000 - INFO - 心跳: 运行中，最近已处理bar=2026-05-17 21:15:00, 当前最新已收盘bar=2026-05-17 21:15:00, 连续跳过同bar次数=1",
        ]
        status = dashboard_server.enrich_status_with_fallbacks({}, log_lines, log_lines)
        self.assertIsNone(status["runtime"].get("last_error"))
        self.assertEqual(status["runtime"]["last_status"], "waiting_next_bar")
