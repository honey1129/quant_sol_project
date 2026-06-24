import math

import numpy as np


class BinaryProbabilityCalibrator:
    """Calibrate a binary model's trade score while keeping the same interface."""

    def __init__(
        self,
        *,
        method="none",
        model=None,
        direction=None,
        regime=None,
        fallback_reason=None,
        fitted_rows=0,
        positive_rows=0,
        negative_rows=0,
        weighted=False,
    ):
        self.method = str(method or "none").lower()
        self.model = model
        self.direction = direction
        self.regime = regime
        self.fallback_reason = fallback_reason
        self.fitted_rows = int(fitted_rows or 0)
        self.positive_rows = int(positive_rows or 0)
        self.negative_rows = int(negative_rows or 0)
        self.weighted = bool(weighted)

    @property
    def active(self):
        return self.model is not None and self.fallback_reason is None

    def predict_trade_probability(self, trade_probability):
        values = np.asarray(trade_probability, dtype=float)
        values = np.clip(values, 0.0, 1.0)
        if not self.active:
            return values

        if self.method == "sigmoid":
            calibrated = self.model.predict_proba(values.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            calibrated = self.model.predict(values)
        else:
            return values
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    def summary(self):
        payload = {
            "method": self.method,
            "direction": self.direction,
            "regime": self.regime,
            "active": bool(self.active),
            "fallback_reason": self.fallback_reason,
            "fitted_rows": int(self.fitted_rows),
            "positive_rows": int(self.positive_rows),
            "negative_rows": int(self.negative_rows),
            "weighted": bool(self.weighted),
        }
        if self.active and self.method == "sigmoid":
            payload["coef"] = float(self.model.coef_[0][0])
            payload["intercept"] = float(self.model.intercept_[0])
        elif self.active and self.method == "isotonic":
            payload["x_thresholds"] = [float(value) for value in self.model.X_thresholds_]
            payload["y_thresholds"] = [float(value) for value in self.model.y_thresholds_]
        return payload


def fit_binary_probability_calibrator(
    trade_probability,
    y,
    *,
    method="sigmoid",
    direction=None,
    regime=None,
    sample_weight=None,
    min_rows=50,
    min_positive_rows=5,
    min_negative_rows=5,
):
    method = str(method or "none").lower()
    if method == "none":
        return BinaryProbabilityCalibrator(
            method=method,
            direction=direction,
            regime=regime,
            fallback_reason="disabled",
        )

    probs = np.asarray(trade_probability, dtype=float).reshape(-1)
    targets = np.asarray(y, dtype=int).reshape(-1)
    if len(probs) != len(targets):
        raise ValueError("校准概率和标签长度不一致")

    finite_mask = np.isfinite(probs)
    probs = probs[finite_mask]
    targets = targets[finite_mask]
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != len(finite_mask):
            raise ValueError("校准权重和标签长度不一致")
        weights = weights[finite_mask]

    fitted_rows = int(len(targets))
    positive_rows = int((targets == 1).sum())
    negative_rows = int(fitted_rows - positive_rows)
    base_payload = {
        "method": method,
        "direction": direction,
        "regime": regime,
        "fitted_rows": fitted_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "weighted": weights is not None,
    }

    if fitted_rows < int(min_rows):
        return BinaryProbabilityCalibrator(
            **base_payload,
            fallback_reason="calibration_rows_below_minimum",
        )
    if positive_rows < int(min_positive_rows):
        return BinaryProbabilityCalibrator(
            **base_payload,
            fallback_reason="calibration_positive_rows_below_minimum",
        )
    if negative_rows < int(min_negative_rows):
        return BinaryProbabilityCalibrator(
            **base_payload,
            fallback_reason="calibration_negative_rows_below_minimum",
        )
    if positive_rows == 0 or negative_rows == 0:
        return BinaryProbabilityCalibrator(
            **base_payload,
            fallback_reason="single_class_calibration_data",
        )

    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver="lbfgs")
        model.fit(probs.reshape(-1, 1), targets, sample_weight=weights)
    elif method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(probs, targets, sample_weight=weights)
    else:
        raise ValueError(f"不支持的方向质量概率校准方法: {method}")

    return BinaryProbabilityCalibrator(**base_payload, model=model)


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

    def __init__(
        self,
        global_model,
        direction_models=None,
        direction_calibrators=None,
        direction_regime_calibrators=None,
        diagnostics=None,
    ):
        self.global_model = global_model
        self.direction_models = {
            str(direction).strip().lower(): model
            for direction, model in dict(direction_models or {}).items()
        }
        self.direction_calibrators = {
            str(direction).strip().lower(): calibrator
            for direction, calibrator in dict(direction_calibrators or {}).items()
        }
        self.direction_regime_calibrators = {}
        for direction, by_regime in dict(direction_regime_calibrators or {}).items():
            direction_key = str(direction).strip().lower()
            self.direction_regime_calibrators[direction_key] = {
                str(regime).strip().lower(): calibrator
                for regime, calibrator in dict(by_regime or {}).items()
            }
        self.diagnostics = dict(diagnostics or {})
        self.classes_ = np.asarray([0, 1], dtype=int)

    def _trend_bias_values(self, X):
        if hasattr(X, "columns") and "trend_bias_num" in X.columns:
            values = np.asarray(X["trend_bias_num"], dtype=float)
        else:
            values = np.zeros(len(X), dtype=float)
        values = np.asarray([0.0 if not math.isfinite(float(v)) else float(v) for v in values])
        return values

    def _regime_values(self, X):
        regimes = np.asarray(["unknown"] * len(X), dtype=object)
        if not hasattr(X, "columns"):
            return regimes

        if "label_regime" in X.columns:
            return np.asarray(
                [
                    str(value or "unknown").strip().lower()
                    for value in X["label_regime"].fillna("unknown")
                ],
                dtype=object,
            )

        regimes = np.asarray(["range"] * len(X), dtype=object)
        if "regime_range_high_vol" in X.columns:
            mask = np.asarray(X["regime_range_high_vol"].astype(float) > 0.5, dtype=bool)
            regimes[mask] = "range_high_vol"
        if "regime_trend_long" in X.columns:
            mask = np.asarray(X["regime_trend_long"].astype(float) > 0.5, dtype=bool)
            regimes[mask] = "trend_long"
        if "regime_trend_short" in X.columns:
            mask = np.asarray(X["regime_trend_short"].astype(float) > 0.5, dtype=bool)
            regimes[mask] = "trend_short"
        return regimes

    def _calibrate_probabilities(self, direction, probs, X=None):
        probs = np.asarray(probs, dtype=float)
        calibrated = probs.copy()
        applied = np.zeros(len(probs), dtype=bool)

        by_regime = self.direction_regime_calibrators.get(direction) or {}
        if by_regime and X is not None:
            regimes = self._regime_values(X)
            for regime, calibrator in by_regime.items():
                if calibrator is None or not bool(getattr(calibrator, "active", False)):
                    continue
                mask = regimes == str(regime).strip().lower()
                if not np.any(mask):
                    continue
                trade_probability = np.asarray(
                    calibrator.predict_trade_probability(probs[mask, 1]),
                    dtype=float,
                )
                calibrated[mask] = np.column_stack([1.0 - trade_probability, trade_probability]).astype(float)
                applied[mask] = True

        calibrator = self.direction_calibrators.get(direction)
        if calibrator is None or not bool(getattr(calibrator, "active", False)):
            return calibrated
        fallback_mask = ~applied
        if not np.any(fallback_mask):
            return calibrated
        trade_probability = np.asarray(
            calibrator.predict_trade_probability(probs[fallback_mask, 1]),
            dtype=float,
        )
        calibrated[fallback_mask] = np.column_stack([1.0 - trade_probability, trade_probability]).astype(float)
        return calibrated

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
            direction_probs = _binary_probabilities(model, subset)
            probs[mask] = self._calibrate_probabilities(direction, direction_probs, subset)

        return np.clip(probs, 0.0, 1.0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def direction_quality_enabled(self):
        return True

    @property
    def trained_directions(self):
        return sorted(str(direction) for direction in self.direction_models.keys())

    @property
    def calibrated_directions(self):
        return sorted(
            str(direction)
            for direction, calibrator in self.direction_calibrators.items()
            if bool(getattr(calibrator, "active", False))
        )

    @property
    def calibrated_direction_regimes(self):
        return sorted(
            f"{direction}:{regime}"
            for direction, by_regime in self.direction_regime_calibrators.items()
            for regime, calibrator in by_regime.items()
            if bool(getattr(calibrator, "active", False))
        )
