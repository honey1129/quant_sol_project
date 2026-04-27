import glob
import json
import os
import tempfile
import unittest

from utils import runtime_dashboard as dashboard


class RuntimeDashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.original_status_path = dashboard.RUNTIME_DASHBOARD_STATUS_PATH
        self.original_history_path = dashboard.RUNTIME_DASHBOARD_HISTORY_PATH
        self.original_baseline_path = dashboard.RUNTIME_DASHBOARD_BASELINE_PATH

        dashboard.RUNTIME_DASHBOARD_STATUS_PATH = os.path.join(self.tmpdir.name, "status.json")
        dashboard.RUNTIME_DASHBOARD_HISTORY_PATH = os.path.join(self.tmpdir.name, "history.json")
        dashboard.RUNTIME_DASHBOARD_BASELINE_PATH = os.path.join(self.tmpdir.name, "baseline.json")

    def tearDown(self):
        dashboard.RUNTIME_DASHBOARD_STATUS_PATH = self.original_status_path
        dashboard.RUNTIME_DASHBOARD_HISTORY_PATH = self.original_history_path
        dashboard.RUNTIME_DASHBOARD_BASELINE_PATH = self.original_baseline_path
        self.tmpdir.cleanup()

    def test_snapshot_tracks_baseline_return_and_drawdown(self):
        snapshot_1 = {
            "runtime": {"last_status": "starting"},
            "account": {"total_eq": 1000, "avail_eq": 900},
        }
        snapshot_2 = {
            "runtime": {"last_status": "running"},
            "account": {"total_eq": 1100, "avail_eq": 980},
        }
        snapshot_3 = {
            "runtime": {"last_status": "running"},
            "account": {"total_eq": 1050, "avail_eq": 930},
        }

        dashboard.write_runtime_dashboard_snapshot(
            snapshot_1,
            history_point={"bar_ts": "2026-04-24T06:00:00", "total_eq": 1000, "avail_eq": 900},
        )
        dashboard.write_runtime_dashboard_snapshot(
            snapshot_2,
            history_point={"bar_ts": "2026-04-24T06:05:00", "total_eq": 1100, "avail_eq": 980},
        )
        payload = dashboard.write_runtime_dashboard_snapshot(
            snapshot_3,
            history_point={"bar_ts": "2026-04-24T06:10:00", "total_eq": 1050, "avail_eq": 930},
        )

        self.assertAlmostEqual(payload["performance"]["baseline_total_eq"], 1000.0)
        self.assertAlmostEqual(payload["performance"]["net_pnl"], 50.0)
        self.assertAlmostEqual(payload["performance"]["return_pct"], 5.0)
        self.assertAlmostEqual(payload["performance"]["drawdown_pct"], (1050.0 - 1100.0) / 1100.0 * 100.0)
        self.assertEqual(payload["performance"]["history_points"], 3)

    def test_same_bar_history_point_overwrites_last_record(self):
        dashboard.write_runtime_dashboard_snapshot(
            {
                "runtime": {"last_status": "running"},
                "account": {"total_eq": 1000, "avail_eq": 900},
            },
            history_point={"bar_ts": "2026-04-24T06:00:00", "total_eq": 1000, "avail_eq": 900},
        )

        dashboard.write_runtime_dashboard_snapshot(
            {
                "runtime": {"last_status": "waiting_next_bar"},
                "account": {"total_eq": 1005, "avail_eq": 910},
            },
            history_point={"bar_ts": "2026-04-24T06:00:00", "total_eq": 1005, "avail_eq": 910},
        )

        history = dashboard.load_runtime_dashboard_history()
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(history[0]["total_eq"], 1005.0)
        self.assertAlmostEqual(history[0]["avail_eq"], 910.0)

    def test_corrupt_baseline_is_backed_up_and_not_silently_reset(self):
        # 第一次正常写入基线
        dashboard.write_runtime_dashboard_snapshot(
            {"runtime": {"last_status": "running"}, "account": {"total_eq": 1000, "avail_eq": 900}},
            history_point={"bar_ts": "2026-04-24T06:00:00", "total_eq": 1000, "avail_eq": 900},
        )
        with open(dashboard.RUNTIME_DASHBOARD_BASELINE_PATH, "r", encoding="utf-8") as f:
            self.assertAlmostEqual(json.load(f)["baseline_total_eq"], 1000.0)

        # 模拟基线文件损坏
        with open(dashboard.RUNTIME_DASHBOARD_BASELINE_PATH, "w", encoding="utf-8") as f:
            f.write("{not valid json")

        # 写入新快照：当下 total_eq=2000，但旧基线已损坏
        payload = dashboard.write_runtime_dashboard_snapshot(
            {"runtime": {"last_status": "running"}, "account": {"total_eq": 2000, "avail_eq": 1800}},
            history_point={"bar_ts": "2026-04-24T06:05:00", "total_eq": 2000, "avail_eq": 1800},
        )

        # 损坏的基线应被备份为 .corrupt-* 文件
        backups = glob.glob(dashboard.RUNTIME_DASHBOARD_BASELINE_PATH + ".corrupt-*")
        self.assertEqual(len(backups), 1)

        # 关键：基线**未**被静默重置为 2000；performance.baseline_total_eq 为 None
        self.assertIsNone(payload["performance"]["baseline_total_eq"])
        self.assertIsNone(payload["performance"]["net_pnl"])
        self.assertIsNone(payload["performance"]["return_pct"])

        # 基线文件保持缺失（等待人工恢复 corrupt 备份）
        self.assertFalse(os.path.exists(dashboard.RUNTIME_DASHBOARD_BASELINE_PATH))

    def test_corrupt_history_is_backed_up_and_recovers(self):
        # 模拟 history 文件损坏
        os.makedirs(os.path.dirname(dashboard.RUNTIME_DASHBOARD_HISTORY_PATH), exist_ok=True)
        with open(dashboard.RUNTIME_DASHBOARD_HISTORY_PATH, "w", encoding="utf-8") as f:
            f.write("garbage")

        # 写入仍能正常完成
        payload = dashboard.write_runtime_dashboard_snapshot(
            {"runtime": {"last_status": "running"}, "account": {"total_eq": 1000, "avail_eq": 900}},
            history_point={"bar_ts": "2026-04-24T06:00:00", "total_eq": 1000, "avail_eq": 900},
        )
        self.assertEqual(payload["performance"]["history_points"], 1)

        backups = glob.glob(dashboard.RUNTIME_DASHBOARD_HISTORY_PATH + ".corrupt-*")
        self.assertEqual(len(backups), 1)

    def test_baseline_initializes_normally_when_file_missing(self):
        self.assertFalse(os.path.exists(dashboard.RUNTIME_DASHBOARD_BASELINE_PATH))

        payload = dashboard.write_runtime_dashboard_snapshot(
            {"runtime": {"last_status": "starting"}, "account": {"total_eq": 1500, "avail_eq": 1300}},
        )

        self.assertAlmostEqual(payload["performance"]["baseline_total_eq"], 1500.0)
        self.assertTrue(os.path.exists(dashboard.RUNTIME_DASHBOARD_BASELINE_PATH))


if __name__ == "__main__":
    unittest.main()
