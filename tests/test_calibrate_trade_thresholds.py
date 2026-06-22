import unittest
import os
import tempfile

import pandas as pd

from run import calibrate_trade_thresholds as calibration


class TradeThresholdCalibrationTests(unittest.TestCase):
    def test_isotonic_probability_calibrator_adds_calibrated_columns(self):
        data = pd.DataFrame({
            "target": [0, 0, 2, 1, 1, 1],
            "long_prob": [0.05, 0.15, 0.35, 0.55, 0.75, 0.9],
            "short_prob": [0.9, 0.7, 0.4, 0.25, 0.15, 0.05],
        })

        calibrators = calibration.fit_probability_calibrators(data, "isotonic")
        out = calibration.apply_probability_calibrators(data, calibrators)

        self.assertIn("long_prob_calibrated", out.columns)
        self.assertIn("short_prob_calibrated", out.columns)
        self.assertTrue(out["long_prob_calibrated"].between(0.0, 1.0).all())
        self.assertTrue(out["short_prob_calibrated"].between(0.0, 1.0).all())
        self.assertTrue(calibrators["long"].summary()["active"])
        self.assertTrue(calibrators["short"].summary()["active"])

    def test_probability_calibrator_falls_back_for_single_class_source(self):
        data = pd.DataFrame({
            "target": [1, 1, 1],
            "long_prob": [0.4, 0.6, 0.8],
            "short_prob": [0.2, 0.1, 0.05],
        })

        calibrator = calibration.fit_direction_probability_calibrator(data, "long", "sigmoid")
        out = calibrator.predict(data["long_prob"])

        self.assertFalse(calibrator.summary()["active"])
        self.assertEqual(calibrator.summary()["fallback_reason"], "single_class_calibration_data")
        self.assertEqual(list(out), data["long_prob"].tolist())

    def test_probability_calibration_uses_actual_direction_label_when_present(self):
        data = pd.DataFrame({
            "target": [0, 1],
            "actual_label": [2, 0],
            "long_prob": [0.1, 0.2],
            "short_prob": [0.8, 0.7],
        })

        self.assertEqual(
            calibration.direction_target_series(data, "short").tolist(),
            [0, 1],
        )
        report = calibration.calibration_bins(data, "short", [0.0, 1.0])

        self.assertEqual(report["bins"][0]["rows"], 2)
        self.assertAlmostEqual(report["bins"][0]["hit_rate"], 0.5)

    def test_calibration_source_falls_back_when_validation_metadata_missing(self):
        selected = pd.DataFrame({"target": [0], "long_prob": [0.2], "short_prob": [0.7]})
        labeled = pd.DataFrame({
            "target": [0, 1],
            "long_prob": [0.2, 0.8],
            "short_prob": [0.7, 0.1],
        })

        source, reason = calibration.select_probability_calibration_source(
            labeled,
            {},
            "validation",
            selected,
        )

        self.assertEqual(reason, "validation_metadata_missing_used_selected")
        self.assertEqual(source.to_dict("list"), selected.to_dict("list"))

    def test_use_calibrated_probability_columns_preserves_raw_columns(self):
        data = pd.DataFrame({
            "long_prob": [0.4],
            "short_prob": [0.3],
            "long_prob_calibrated": [0.7],
            "short_prob_calibrated": [0.2],
        })

        out = calibration.use_calibrated_probability_columns(data)

        self.assertEqual(float(out.loc[0, "long_prob_raw"]), 0.4)
        self.assertEqual(float(out.loc[0, "short_prob_raw"]), 0.3)
        self.assertEqual(float(out.loc[0, "long_prob"]), 0.7)
        self.assertEqual(float(out.loc[0, "short_prob"]), 0.2)
        self.assertEqual(out.loc[0, "pred_direction"], "long")

    def test_label_strength_summary_reports_trade_density_and_direction_balance(self):
        index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        data = pd.DataFrame({
            "target": [1, 1, 0, 0],
            "diag_trend_bias": ["long", "short", "long", "neutral"],
            "diag_regime": ["trend_long", "trend_short", "trend_long", "range_high_vol"],
        }, index=index)
        candidate = {"name": "lh48_tp0.028_sl0.012", "lookahead_bars": 48, "take_profit": 0.028, "stop_loss": 0.012}

        summary = calibration.summarize_label_strength(
            data,
            candidate,
            target_trade_pct=50.0,
            min_trade_rows=1,
        )

        self.assertEqual(summary["trade_rows"], 2)
        self.assertAlmostEqual(summary["trade_pct"], 50.0)
        self.assertEqual(summary["trade_direction_counts"]["long"], 1)
        self.assertEqual(summary["trade_direction_counts"]["short"], 1)
        self.assertAlmostEqual(summary["direction_imbalance_pct"], 0.0)
        self.assertEqual(summary["by_regime"]["trend_long"]["trade_rows"], 1)

    def test_build_label_strength_candidates_crosses_inputs(self):
        candidates = calibration.build_label_strength_candidates([24, 48], [0.02], [0.01, 0.012])

        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0]["name"], "lh24_tp0.020_sl0.010")

    def test_write_report_replaces_atomically_without_tmp_leftover(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")

            calibration.write_report({"ok": True}, path)
            calibration.write_report({"ok": False}, path)

            with open(path, "r", encoding="utf-8") as file:
                self.assertIn('"ok": false', file.read())
            self.assertFalse(os.path.exists(f"{path}.tmp"))


if __name__ == "__main__":
    unittest.main()
