import math
import sys
import types
import unittest
from unittest.mock import patch


if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

if "requests" not in sys.modules:
    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = fake_requests

try:
    import joblib  # noqa: F401
except ModuleNotFoundError:
    fake_joblib = types.ModuleType("joblib")
    fake_joblib.load = lambda path: object()
    sys.modules["joblib"] = fake_joblib

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    fake_numpy = types.ModuleType("numpy")

    class FakeArray:
        def __init__(self, values):
            self.values = [float(value) for value in values]

        def __iter__(self):
            return iter(self.values)

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

        def __repr__(self):
            return repr(self.values)

        def __mul__(self, other):
            return FakeArray([value * float(other) for value in self.values])

        __rmul__ = __mul__

        def __add__(self, other):
            return FakeArray([left + right for left, right in zip(self.values, other)])

        def __iadd__(self, other):
            self.values = [left + right for left, right in zip(self.values, other)]
            return self

        def __truediv__(self, other):
            return FakeArray([value / float(other) for value in self.values])

    fake_numpy.asarray = lambda values, dtype=None: FakeArray(values)
    fake_numpy.zeros_like = lambda values, dtype=None: FakeArray([0.0 for _ in values])
    fake_numpy.clip = lambda value, lower, upper: max(lower, min(value, upper))
    fake_numpy.sign = lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
    fake_numpy.mean = lambda values, axis=0: FakeArray([
        sum(row[index] for row in values) / len(values)
        for index in range(len(values[0]))
    ])
    fake_numpy.isfinite = math.isfinite
    sys.modules["numpy"] = fake_numpy

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = lambda values, columns=None: values
    sys.modules["pandas"] = fake_pandas

from core import signal_engine
from core.direction_quality import DirectionQualityModel, BinaryProbabilityCalibrator


class StubModel:
    def __init__(self, probability, classes=None):
        self.probability = probability
        if classes is not None:
            self.classes_ = classes

    def predict_proba(self, X):
        return [self.probability]


class BatchStubModel:
    def __init__(self, probabilities, classes=None):
        self.probabilities = probabilities
        if classes is not None:
            self.classes_ = classes

    def predict_proba(self, X):
        return self.probabilities[:len(X)]


class WeightedPredictProbaTests(unittest.TestCase):
    def test_missing_weight_uses_default_and_extra_weight_is_ignored(self):
        models = {
            "lgb_v1": StubModel([0.2, 0.8]),
            "xgb_v1": StubModel([0.6, 0.4]),
        }

        out = signal_engine.weighted_predict_proba(
            models,
            object(),
            {
                "lgb_v1": 0.75,
                "unused_model": 99.0,
            },
        )

        expected_short = (0.2 * 0.75 + 0.6 * 1.0) / 1.75
        expected_long = (0.8 * 0.75 + 0.4 * 1.0) / 1.75
        self.assertAlmostEqual(float(out[0]), expected_short)
        self.assertAlmostEqual(float(out[1]), expected_long)

    def test_multiclass_no_trade_probability_lowers_directional_probs(self):
        models = {
            "lgb_v1": StubModel([0.10, 0.30, 0.60], classes=[0, 1, 2]),
            "xgb_v1": StubModel([0.70, 0.20, 0.10], classes=[2, 0, 1]),
        }

        out = signal_engine.weighted_predict_proba(
            models,
            object(),
            {"lgb_v1": 1.0, "xgb_v1": 1.0},
        )

        self.assertAlmostEqual(float(out[0]), 0.15)
        self.assertAlmostEqual(float(out[1]), 0.20)

    def test_binary_trade_quality_maps_trade_prob_to_trend_direction(self):
        models = {"lgb_v1": StubModel([0.80, 0.20], classes=[0, 1])}

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            long_out = signal_engine.weighted_predict_proba(
                models,
                object(),
                {"lgb_v1": 1.0},
                trend_bias="long",
                model_metadata={"target_schema": "binary_trade_quality"},
            )
            short_out = signal_engine.weighted_predict_proba(
                models,
                object(),
                {"lgb_v1": 1.0},
                trend_bias="short",
                model_metadata={"target_schema": "binary_trade_quality"},
            )
            neutral_out = signal_engine.weighted_predict_proba(
                models,
                object(),
                {"lgb_v1": 1.0},
                trend_bias="neutral",
                model_metadata={"target_schema": "binary_trade_quality"},
            )

        self.assertAlmostEqual(float(long_out[0]), 0.0)
        self.assertAlmostEqual(float(long_out[1]), 0.20)
        self.assertAlmostEqual(float(short_out[0]), 0.20)
        self.assertAlmostEqual(float(short_out[1]), 0.0)
        self.assertAlmostEqual(float(neutral_out[0]), 0.0)
        self.assertAlmostEqual(float(neutral_out[1]), 0.0)

    def test_binary_trade_quality_scales_rare_quality_probability_to_execution_probability(self):
        models = {"lgb_v1": StubModel([0.875, 0.125], classes=[0, 1])}
        metadata = {
            "target_schema": "binary_trade_quality",
            "label_quality_summary": {
                "by_regime": {
                    "trend_long": {"trade_pct": 12.5},
                },
            },
        }

        out = signal_engine.weighted_predict_proba(
            models,
            object(),
            {"lgb_v1": 1.0},
            trend_bias="long",
            model_metadata=metadata,
        )

        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[1]), 0.50)

    def test_binary_trade_quality_execution_scale_can_use_lift_above_base_rate(self):
        models = {"lgb_v1": StubModel([0.80, 0.20], classes=[0, 1])}
        metadata = {
            "target_schema": "binary_trade_quality",
            "label_quality_summary": {
                "by_regime": {
                    "trend_long": {"trade_pct": 10.0},
                },
            },
        }

        out = signal_engine.weighted_predict_proba(
            models,
            object(),
            {"lgb_v1": 1.0},
            trend_bias="long",
            model_metadata=metadata,
        )

        self.assertAlmostEqual(float(out[1]), 2.25 / 3.25, places=6)

    def test_batch_binary_trade_quality_maps_each_row_trend_direction(self):
        import pandas as pd

        models = {
            "lgb_v1": BatchStubModel(
                [[0.80, 0.20], [0.30, 0.70], [0.10, 0.90]],
                classes=[0, 1],
            )
        }

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            out = signal_engine.weighted_predict_proba_batch(
                models,
                pd.DataFrame({"feature": [1.0, 2.0, 3.0]}),
                {"lgb_v1": 1.0},
                trend_biases=["long", "short", "neutral"],
                model_metadata={"target_schema": "binary_trade_quality"},
            )

        self.assertAlmostEqual(float(out[0][0]), 0.0)
        self.assertAlmostEqual(float(out[0][1]), 0.20)
        self.assertAlmostEqual(float(out[1][0]), 0.70)
        self.assertAlmostEqual(float(out[1][1]), 0.0)
        self.assertAlmostEqual(float(out[2][0]), 0.0)
        self.assertAlmostEqual(float(out[2][1]), 0.0)

    def test_direction_model_weight_overrides_single_and_batch_predictions(self):
        import pandas as pd

        models = {
            "lgb_v1": BatchStubModel([[0.80, 0.20], [0.80, 0.20]], classes=[0, 1]),
            "rf_v1": BatchStubModel([[0.20, 0.80], [0.20, 0.80]], classes=[0, 1]),
        }
        X = pd.DataFrame({"feature": [1.0, 2.0]})
        direction_weights = {
            "long": "rf_v1:1.0",
            "short": "lgb_v1:1.0",
        }

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            batch = signal_engine.weighted_predict_proba_batch(
                models,
                X,
                {"lgb_v1": 0.5, "rf_v1": 0.5},
                trend_biases=["long", "short"],
                model_metadata={"target_schema": "binary_trade_quality"},
                direction_model_weights=direction_weights,
            )
            single_long = signal_engine.weighted_predict_proba(
                models,
                X.iloc[:1],
                {"lgb_v1": 0.5, "rf_v1": 0.5},
                trend_bias="long",
                model_metadata={"target_schema": "binary_trade_quality"},
                direction_model_weights=direction_weights,
            )

        self.assertAlmostEqual(float(batch[0][1]), 0.80)
        self.assertAlmostEqual(float(batch[0][0]), 0.0)
        self.assertAlmostEqual(float(batch[1][0]), 0.20)
        self.assertAlmostEqual(float(batch[1][1]), 0.0)
        self.assertAlmostEqual(float(single_long[1]), 0.80)

    def test_batch_and_single_row_predictions_match(self):
        import pandas as pd

        models = {
            "lgb_v1": BatchStubModel([[0.60, 0.40], [0.20, 0.80]], classes=[0, 1]),
            "xgb_v1": BatchStubModel([[0.50, 0.50], [0.70, 0.30]], classes=[0, 1]),
        }
        X = pd.DataFrame({"feature": [1.0, 2.0]})
        weights = {"lgb_v1": 0.75, "xgb_v1": 0.25}

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            batch = signal_engine.weighted_predict_proba_batch(
                models,
                X,
                weights,
                trend_biases=["long", "short"],
                model_metadata={"target_schema": "binary_trade_quality"},
            )
            first = signal_engine.weighted_predict_proba(
                {"lgb_v1": BatchStubModel([[0.60, 0.40]], classes=[0, 1]), "xgb_v1": BatchStubModel([[0.50, 0.50]], classes=[0, 1])},
                X.iloc[:1],
                weights,
                trend_bias="long",
                model_metadata={"target_schema": "binary_trade_quality"},
            )
            second = signal_engine.weighted_predict_proba(
                {"lgb_v1": BatchStubModel([[0.20, 0.80]], classes=[0, 1]), "xgb_v1": BatchStubModel([[0.70, 0.30]], classes=[0, 1])},
                X.iloc[1:2],
                weights,
                trend_bias="short",
                model_metadata={"target_schema": "binary_trade_quality"},
            )

        self.assertAlmostEqual(float(batch[0][0]), float(first[0]))
        self.assertAlmostEqual(float(batch[0][1]), float(first[1]))
        self.assertAlmostEqual(float(batch[1][0]), float(second[0]))
        self.assertAlmostEqual(float(batch[1][1]), float(second[1]))

    def test_direction_quality_model_uses_direction_specific_submodel(self):
        import pandas as pd

        model = DirectionQualityModel(
            StubModel([0.80, 0.20], classes=[0, 1]),
            direction_models={
                "long": StubModel([0.10, 0.90], classes=[0, 1]),
                "short": StubModel([0.25, 0.75], classes=[0, 1]),
            },
        )

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            long_out = signal_engine.weighted_predict_proba(
                {"lgb_v1": model},
                pd.DataFrame({"trend_bias_num": [1.0]}),
                {"lgb_v1": 1.0},
                trend_bias="long",
                model_metadata={"target_schema": "binary_trade_quality"},
            )
            short_out = signal_engine.weighted_predict_proba(
                {"lgb_v1": model},
                pd.DataFrame({"trend_bias_num": [-1.0]}),
                {"lgb_v1": 1.0},
                trend_bias="short",
                model_metadata={"target_schema": "binary_trade_quality"},
            )

        self.assertAlmostEqual(float(long_out[0]), 0.0)
        self.assertAlmostEqual(float(long_out[1]), 0.90)
        self.assertAlmostEqual(float(short_out[0]), 0.75)
        self.assertAlmostEqual(float(short_out[1]), 0.0)

    def test_direction_quality_model_applies_direction_calibrator(self):
        import pandas as pd

        class ScaleCalibrator(BinaryProbabilityCalibrator):
            @property
            def active(self):
                return True

            def predict_trade_probability(self, trade_probability):
                return [float(value) * 0.5 for value in trade_probability]

        model = DirectionQualityModel(
            StubModel([0.80, 0.20], classes=[0, 1]),
            direction_models={"long": StubModel([0.10, 0.90], classes=[0, 1])},
            direction_calibrators={"long": ScaleCalibrator(method="custom", direction="long")},
        )

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            long_out = signal_engine.weighted_predict_proba(
                {"lgb_v1": model},
                pd.DataFrame({"trend_bias_num": [1.0]}),
                {"lgb_v1": 1.0},
                trend_bias="long",
                model_metadata={"target_schema": "binary_trade_quality"},
            )

        self.assertAlmostEqual(float(long_out[0]), 0.0)
        self.assertAlmostEqual(float(long_out[1]), 0.45)

    def test_direction_quality_model_prefers_regime_calibrator(self):
        import pandas as pd

        class FixedCalibrator(BinaryProbabilityCalibrator):
            def __init__(self, value, **kwargs):
                super().__init__(method="custom", **kwargs)
                self.value = value

            @property
            def active(self):
                return True

            def predict_trade_probability(self, trade_probability):
                return [self.value for _ in trade_probability]

        model = DirectionQualityModel(
            StubModel([0.80, 0.20], classes=[0, 1]),
            direction_models={"short": StubModel([0.10, 0.90], classes=[0, 1])},
            direction_calibrators={"short": FixedCalibrator(0.40, direction="short")},
            direction_regime_calibrators={
                "short": {
                    "trend_short": FixedCalibrator(0.12, direction="short", regime="trend_short"),
                },
            },
        )

        with patch("core.signal_engine.config.MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", False):
            trend_short_out = signal_engine.weighted_predict_proba(
                {"lgb_v1": model},
                pd.DataFrame({
                    "trend_bias_num": [-1.0],
                    "regime_trend_short": [1.0],
                    "regime_trend_long": [0.0],
                }),
                {"lgb_v1": 1.0},
                trend_bias="short",
                model_metadata={"target_schema": "binary_trade_quality"},
            )
            range_out = signal_engine.weighted_predict_proba(
                {"lgb_v1": model},
                pd.DataFrame({
                    "trend_bias_num": [-1.0],
                    "regime_trend_short": [0.0],
                    "regime_trend_long": [0.0],
                }),
                {"lgb_v1": 1.0},
                trend_bias="short",
                model_metadata={"target_schema": "binary_trade_quality"},
            )

        self.assertAlmostEqual(float(trend_short_out[0]), 0.12)
        self.assertAlmostEqual(float(trend_short_out[1]), 0.0)
        self.assertAlmostEqual(float(range_out[0]), 0.40)
        self.assertAlmostEqual(float(range_out[1]), 0.0)
        self.assertEqual(model.calibrated_direction_regimes, ["short:trend_short"])

    def test_empty_models_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "模型列表为空"):
            signal_engine.weighted_predict_proba({}, object(), {})

    def test_all_zero_used_weights_are_rejected(self):
        models = {"lgb_v1": StubModel([0.2, 0.8])}

        with self.assertRaisesRegex(ValueError, "权重总和必须大于 0"):
            signal_engine.weighted_predict_proba(models, object(), {"lgb_v1": 0.0})

    def test_negative_weight_is_rejected(self):
        models = {"lgb_v1": StubModel([0.2, 0.8])}

        with self.assertRaisesRegex(ValueError, "非负有限数"):
            signal_engine.weighted_predict_proba(models, object(), {"lgb_v1": -1.0})


if __name__ == "__main__":
    unittest.main()
