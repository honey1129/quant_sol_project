import unittest

import numpy as np
import pandas as pd

from research import directional_v2_training


class FakeModel:
    classes_ = np.asarray([0, 1, 2])

    def predict_proba(self, X):
        return np.asarray([
            [0.70, 0.20, 0.10],
            [0.10, 0.65, 0.25],
        ])[:len(X)]


class DirectionalV2TrainingTests(unittest.TestCase):
    def test_class_weights_are_inverse_frequency_and_capped(self):
        target = pd.Series([0] * 90 + [1] * 9 + [2])

        weights = directional_v2_training.class_weight_map(target, 8.0)

        self.assertLess(weights[0], weights[1])
        self.assertEqual(weights[2], 8.0)

    def test_temporal_split_keeps_purge_gap(self):
        train_end, validation_start = directional_v2_training.temporal_development_split(
            1000,
            validation_ratio=0.2,
            purge_bars=24,
        )

        self.assertEqual(validation_start, 800)
        self.assertEqual(train_end, 776)

    def test_probability_frame_uses_explicit_flat_long_short_classes(self):
        X = pd.DataFrame(
            {"feature": [1.0, 2.0]},
            index=pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
        )

        probabilities = directional_v2_training.probability_frame(FakeModel(), X)

        self.assertEqual(list(probabilities.columns), ["flat", "long", "short"])
        self.assertAlmostEqual(probabilities.iloc[1]["long"], 0.65)
        self.assertAlmostEqual(probabilities.iloc[1]["short"], 0.25)


if __name__ == "__main__":
    unittest.main()
