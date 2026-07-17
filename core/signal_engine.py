# signal_engine.py

import os
import math
import numpy as np
import joblib
import pandas as pd
from config import config
from utils.utils import BASE_DIR

# 多模型加载（支持绝对路径）
def load_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        full_path = os.path.join(BASE_DIR, path)
        models[name] = joblib.load(full_path)
    return models

# 简单平均融合
def ensemble_predict(models, merged_df, feature_cols):
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)
    X_live = pd.DataFrame(X_live, columns=feature_cols)

    return weighted_predict_proba(
        models,
        X_live,
        {name: 1.0 for name in models},
    )

# 贝叶斯加权融合
def bayesian_weighted_predict(models, merged_df, feature_cols, model_weights):
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)
    X_live = pd.DataFrame(X_live, columns=feature_cols)
    return weighted_predict_proba(models, X_live, model_weights)


def _metadata_label_mode(model_metadata):
    if not isinstance(model_metadata, dict):
        return ""
    return str(model_metadata.get("target_schema") or model_metadata.get("label_mode") or "").lower()


def _is_binary_trade_quality_model(model_metadata):
    label_mode = _metadata_label_mode(model_metadata)
    return label_mode == "binary_trade_quality" or label_mode.startswith("binary_")


def _model_is_direction_quality(model):
    return bool(getattr(model, "direction_quality_enabled", False))


def _class_probability(raw_prob, classes, label):
    return float(raw_prob[classes.index(label)]) if label in classes else 0.0


def _class_probabilities(raw_prob, classes, label):
    raw_prob = np.asarray(raw_prob, dtype=float)
    return raw_prob[:, classes.index(label)] if label in classes else np.zeros(raw_prob.shape[0], dtype=float)


def _parse_model_weight_map(raw_value):
    if isinstance(raw_value, dict):
        items = raw_value.items()
    else:
        items = []
        for item in str(raw_value or "").replace("|", ",").split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.rsplit(":", 1)
            items.append((key, value))

    parsed = {}
    for key, value in items:
        try:
            parsed[str(key).strip()] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"方向模型权重不是有效数字: {key}={value!r}")
    return parsed


def _direction_model_weight_overrides(raw_value):
    overrides = {}
    if not raw_value:
        return overrides
    if isinstance(raw_value, dict):
        items = raw_value.items()
    else:
        items = []
        for item in str(raw_value or "").split(","):
            item = item.strip()
            if not item:
                continue
            direction, weights = item.split("=", 1)
            items.append((direction, weights))
    for direction, weights in items:
        direction_key = str(direction).strip().lower()
        if direction_key not in {"long", "short"}:
            continue
        overrides[direction_key] = _parse_model_weight_map(weights)
    return overrides


def _model_weight_for_direction(model_weights, direction_model_weights, model_name, direction):
    direction = str(direction or "").strip().lower()
    if direction in direction_model_weights:
        return direction_model_weights[direction].get(model_name, 0.0)
    return model_weights.get(model_name, 1.0)


def _quality_probability_execution_scale_enabled():
    return bool(getattr(config, "MODEL_QUALITY_PROBABILITY_EXECUTION_SCALE_ENABLED", True))


def _quality_probability_execution_anchor():
    value = float(getattr(config, "MODEL_QUALITY_PROBABILITY_EXECUTION_ANCHOR", 0.50))
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _quality_probability_execution_temperature():
    value = float(getattr(config, "MODEL_QUALITY_PROBABILITY_EXECUTION_TEMPERATURE", 1.0))
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return value


def _quality_probability_config_base_rate():
    value = float(getattr(config, "MODEL_QUALITY_PROBABILITY_BASE_RATE", 0.0))
    if not math.isfinite(value) or value <= 0:
        return None
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _quality_probability_min_base_rate():
    value = float(getattr(config, "MODEL_QUALITY_PROBABILITY_MIN_BASE_RATE", 0.01))
    if not math.isfinite(value) or value <= 0:
        return 1e-6
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _quality_probability_max_base_rate():
    value = float(getattr(config, "MODEL_QUALITY_PROBABILITY_MAX_BASE_RATE", 0.50))
    if not math.isfinite(value) or value <= 0:
        return 1.0 - 1e-6
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _clean_probability(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _trade_pct_to_probability(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    if value > 1.0:
        value /= 100.0
    return float(np.clip(value, 1e-6, 1.0 - 1e-6))


def _quality_base_rate_from_model(model, direction):
    diagnostics = getattr(model, "diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None
    direction_summary = diagnostics.get(str(direction or "").lower())
    if not isinstance(direction_summary, dict):
        direction_summary = diagnostics.get("directions", {}).get(str(direction or "").lower())
    if not isinstance(direction_summary, dict):
        return None
    return _trade_pct_to_probability(direction_summary.get("trade_pct"))


def _quality_base_rate_from_metadata(model_metadata, direction):
    if not isinstance(model_metadata, dict):
        return None
    quality_summary = model_metadata.get("label_quality_summary") or {}
    if not isinstance(quality_summary, dict):
        return None

    direction_key = str(direction or "").strip().lower()
    regime_key = {"long": "trend_long", "short": "trend_short"}.get(direction_key)
    by_regime = quality_summary.get("by_regime") or {}
    if regime_key and isinstance(by_regime, dict):
        regime_summary = by_regime.get(regime_key) or {}
        if isinstance(regime_summary, dict):
            base_rate = _trade_pct_to_probability(regime_summary.get("trade_pct"))
            if base_rate is not None:
                return base_rate

    return _trade_pct_to_probability(quality_summary.get("trade_pct"))


def _quality_probability_base_rate(model_metadata, model=None, direction=None):
    configured = _quality_probability_config_base_rate()
    if configured is not None:
        return configured

    base_rate = _quality_base_rate_from_model(model, direction)
    if base_rate is None:
        base_rate = _quality_base_rate_from_metadata(model_metadata, direction)
    if base_rate is None:
        return None

    min_base = _quality_probability_min_base_rate()
    max_base = max(min_base, _quality_probability_max_base_rate())
    return float(np.clip(base_rate, min_base, max_base))


def _quality_probability_to_execution_probability(trade_probability, *, model_metadata=None, model=None, direction=None):
    values = np.clip(np.asarray(trade_probability, dtype=float), 0.0, 1.0)
    if not _quality_probability_execution_scale_enabled():
        return values

    base_rate = _quality_probability_base_rate(model_metadata, model=model, direction=direction)
    if base_rate is None:
        return values

    anchor = _quality_probability_execution_anchor()
    temperature = _quality_probability_execution_temperature()
    clipped_values = np.clip(values, 1e-6, 1.0 - 1e-6)
    quality_odds = clipped_values / (1.0 - clipped_values)
    base_odds = base_rate / (1.0 - base_rate)
    anchor_odds = anchor / (1.0 - anchor)
    lift = np.maximum(quality_odds / max(base_odds, 1e-12), 1e-12)
    execution_odds = anchor_odds * np.power(lift, temperature)
    return np.clip(execution_odds / (1.0 + execution_odds), 0.0, 1.0)


def _validate_model_weight(model_name, raw_weight):
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        raise ValueError(f"模型 {model_name} 的权重不是有效数字: {raw_weight!r}")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"模型 {model_name} 的权重必须是非负有限数: {weight!r}")
    return weight


def _binary_trade_quality_to_directional(trade_prob, no_trade_prob, trend_bias, *, model_metadata=None, model=None):
    """Map trade/no-trade quality into directional strategy probabilities.

    Binary quality models answer "should we take the rule-selected trend trade?"
    They do not predict short vs long directly. The trend/regime filters still own
    direction, while no-trade probability keeps both directional probabilities below
    entry thresholds when quality is weak.
    """
    trend_bias = str(trend_bias or "").lower()
    trade_prob = float(_quality_probability_to_execution_probability(
        [trade_prob],
        model_metadata=model_metadata,
        model=model,
        direction=trend_bias,
    )[0])
    trade_prob = max(0.0, min(1.0, float(trade_prob)))

    if trend_bias == "long":
        return np.asarray([0.0, trade_prob], dtype=float)
    if trend_bias == "short":
        return np.asarray([trade_prob, 0.0], dtype=float)
    return np.asarray([0.0, 0.0], dtype=float)


def _binary_trade_quality_to_directional_batch(trade_prob, no_trade_prob, trend_biases, *, model_metadata=None, model=None):
    trade_prob = np.clip(np.asarray(trade_prob, dtype=float).reshape(-1), 0.0, 1.0)
    trend_values = [str(value or "").lower() for value in (trend_biases or [])]
    if len(trend_values) != len(trade_prob):
        trend_values = ["neutral"] * len(trade_prob)

    prob = np.zeros((len(trade_prob), 2), dtype=float)
    for direction in ("long", "short"):
        mask = np.asarray([trend_bias == direction for trend_bias in trend_values], dtype=bool)
        if not np.any(mask):
            continue
        trade_prob[mask] = _quality_probability_to_execution_probability(
            trade_prob[mask],
            model_metadata=model_metadata,
            model=model,
            direction=direction,
        )

    for index, trend_bias in enumerate(trend_values):
        if trend_bias == "long":
            prob[index, 1] = trade_prob[index]
        elif trend_bias == "short":
            prob[index, 0] = trade_prob[index]
    return prob


def weighted_predict_proba_batch(
    models,
    X,
    model_weights=None,
    *,
    trend_biases=None,
    model_metadata=None,
    direction_model_weights=None,
):
    """Return an Nx2 array of weighted [short_prob, long_prob] probabilities."""
    if not models:
        raise ValueError("模型列表为空，无法生成预测概率")

    model_weights = model_weights or {}
    weighted_sum = None
    row_count = len(X)
    direction_model_weights = _direction_model_weight_overrides(direction_model_weights)
    trend_values = [str(value or "").lower() for value in (trend_biases or [])]
    if len(trend_values) != row_count:
        trend_values = ["neutral"] * row_count
    weight_totals = np.zeros(row_count, dtype=float)

    for name, model in models.items():
        raw_prob = np.asarray(model.predict_proba(X), dtype=float)
        if raw_prob.ndim != 2 or raw_prob.shape[1] < 2:
            raise ValueError(f"模型 {name} 返回的概率维度不足: {raw_prob!r}")
        if raw_prob.shape[0] != row_count:
            raise ValueError(
                f"模型 {name} 返回的概率行数不一致: rows={raw_prob.shape[0]} expected={row_count}"
            )
        if not np.all(np.isfinite(raw_prob)):
            raise ValueError(f"模型 {name} 返回了非有限概率: {raw_prob!r}")

        classes = list(getattr(model, "classes_", range(raw_prob.shape[1])))
        if (_is_binary_trade_quality_model(model_metadata) or _model_is_direction_quality(model)) and 2 not in classes:
            prob = _binary_trade_quality_to_directional_batch(
                trade_prob=_class_probabilities(raw_prob, classes, 1),
                no_trade_prob=_class_probabilities(raw_prob, classes, 0),
                trend_biases=trend_values,
                model_metadata=model_metadata,
                model=model,
            )
        else:
            prob = np.column_stack([
                _class_probabilities(raw_prob, classes, 0),
                _class_probabilities(raw_prob, classes, 1),
            ]).astype(float)

        row_weights = np.asarray([
            _validate_model_weight(
                name,
                _model_weight_for_direction(model_weights, direction_model_weights, name, trend_bias),
            )
            for trend_bias in trend_values
        ], dtype=float)
        if not np.any(row_weights > 0):
            continue
        if weighted_sum is None:
            weighted_sum = np.zeros_like(prob, dtype=float)
        weighted_sum += prob * row_weights.reshape(-1, 1)
        weight_totals += row_weights

    if weighted_sum is None or not np.all(weight_totals > 0):
        raise ValueError("实际参与预测的模型权重总和必须大于 0")

    return weighted_sum / weight_totals.reshape(-1, 1)


def weighted_predict_proba(
    models,
    X,
    model_weights=None,
    *,
    trend_bias=None,
    model_metadata=None,
    direction_model_weights=None,
):
    """Return weighted [short_prob, long_prob] over the models that actually predict.

    Supported label schemas:
      - legacy directional: 0=short, 1=long, optional 2=no_trade
      - binary trade quality: 0=no_trade, 1=trade, direction supplied by trend_bias
    """
    if not models:
        raise ValueError("模型列表为空，无法生成预测概率")

    model_weights = model_weights or {}
    weighted_sum = None
    used_weight_total = 0.0
    direction_model_weights = _direction_model_weight_overrides(direction_model_weights)

    for name, model in models.items():
        weight = _validate_model_weight(
            name,
            _model_weight_for_direction(model_weights, direction_model_weights, name, trend_bias),
        )
        if weight == 0:
            continue

        raw_prob = np.asarray(model.predict_proba(X)[0], dtype=float)
        if len(raw_prob) < 2:
            raise ValueError(f"模型 {name} 返回的概率维度不足: {raw_prob!r}")
        if not all(math.isfinite(float(value)) for value in raw_prob):
            raise ValueError(f"模型 {name} 返回了非有限概率: {raw_prob!r}")

        classes = list(getattr(model, "classes_", range(len(raw_prob))))
        if (_is_binary_trade_quality_model(model_metadata) or _model_is_direction_quality(model)) and 2 not in classes:
            prob = _binary_trade_quality_to_directional(
                trade_prob=_class_probability(raw_prob, classes, 1),
                no_trade_prob=_class_probability(raw_prob, classes, 0),
                trend_bias=trend_bias,
                model_metadata=model_metadata,
                model=model,
            )
        else:
            prob = np.asarray([
                _class_probability(raw_prob, classes, 0),
                _class_probability(raw_prob, classes, 1),
            ], dtype=float)

        if weighted_sum is None:
            weighted_sum = np.zeros_like(prob, dtype=float)
        weighted_sum += prob * weight
        used_weight_total += weight

    if weighted_sum is None or used_weight_total <= 0:
        raise ValueError("实际参与预测的模型权重总和必须大于 0")

    return weighted_sum / used_weight_total


# 信号平滑去噪模块
class SignalSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_prob = None

    def smooth(self, new_prob):
        if self.smoothed_prob is None:
            self.smoothed_prob = new_prob
        else:
            self.smoothed_prob = self.alpha * new_prob + (1 - self.alpha) * self.smoothed_prob
        return self.smoothed_prob
