import unittest

from core.position_manager import PositionManager


class PositionManagerTests(unittest.TestCase):
    def test_probability_center_controls_signal_strength(self):
        default_pm = PositionManager(min_ratio=0.08, max_ratio=0.45, probability_center=0.50)
        calibrated_pm = PositionManager(min_ratio=0.08, max_ratio=0.45, probability_center=0.40)

        self.assertEqual(default_pm.signal_strength(0.50), 0.0)
        self.assertGreater(calibrated_pm.signal_strength(0.60), default_pm.signal_strength(0.60))

    def test_invalid_probability_center_is_rejected(self):
        with self.assertRaises(ValueError):
            PositionManager(probability_center=1.0)


if __name__ == "__main__":
    unittest.main()
