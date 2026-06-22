import math
import sys
import types
import unittest


if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

if "requests" not in sys.modules:
    fake_requests = types.ModuleType("requests")
    fake_requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = fake_requests

if "joblib" not in sys.modules:
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


class StubModel:
    def __init__(self, probability, classes=None):
        self.probability = probability
        if classes is not None:
            self.classes_ = classes

    def predict_proba(self, X):
        return [self.probability]


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
