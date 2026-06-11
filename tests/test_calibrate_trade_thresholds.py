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
