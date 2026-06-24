import math

import numpy as np


def _binary_probabilities(model, X):
    raw = np.asarray(model.predict_proba(X), dtype=float)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"方向质量模型返回的概率维度不正确: {raw.shape!r}")

    classes = list(getattr(model, "classes_", range(raw.shape[1])))
    no_trade = raw[:, classes.index(0)] if 0 in classes else np.zeros(len(X), dtype=float)
    trade = raw[:, classes.index(1)] if 1 in classes else np.zeros(len(X), dtype=float)
    return np.column_stack([no_trade, trade]).astype(float)


class DirectionQualityModel:
    """Binary trade-quality model with direction-specific submodels.

    The global model remains a fallback. When `trend_bias_num` is available in
    the feature frame, rows with long/short trend bias use the corresponding
    direction quality model. This lets long and short learn separate quality
    rankings while preserving the existing binary [no_trade, trade] interface.
    """

    classes_ = np.asarray([0, 1], dtype=int)

    def __init__(self, global_model, direction_models=None, diagnostics=None):
        self.global_model = global_model
        self.direction_models = dict(direction_models or {})
        self.diagnostics = dict(diagnostics or {})
        self.classes_ = np.asarray([0, 1], dtype=int)

    def _trend_bias_values(self, X):
        if hasattr(X, "columns") and "trend_bias_num" in X.columns:
            values = np.asarray(X["trend_bias_num"], dtype=float)
        else:
            values = np.zeros(len(X), dtype=float)
        values = np.asarray([0.0 if not math.isfinite(float(v)) else float(v) for v in values])
        return values

    def predict_proba(self, X):
        probs = _binary_probabilities(self.global_model, X)
        trend_bias = self._trend_bias_values(X)

        for direction, sign in (("long", 1.0), ("short", -1.0)):
            model = self.direction_models.get(direction)
            if model is None:
                continue
            mask = trend_bias > 0 if sign > 0 else trend_bias < 0
            if not np.any(mask):
                continue
            if hasattr(X, "iloc"):
                subset = X.iloc[np.where(mask)[0]]
            else:
                subset = np.asarray(X)[mask]
            probs[mask] = _binary_probabilities(model, subset)

        return np.clip(probs, 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def direction_quality_enabled(self):
        return True

    @property
    def trained_directions(self):
        return sorted(str(direction) for direction in self.direction_models.keys())
