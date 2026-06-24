# signal_engine.py

import os
import math
import numpy as np
import joblib
import pandas as pd
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


def _binary_trade_quality_to_directional(trade_prob, no_trade_prob, trend_bias):
    """Map trade/no-trade quality into directional strategy probabilities.

    Binary quality models answer "should we take the rule-selected trend trade?"
    They do not predict short vs long directly. The trend/regime filters still own
    direction, while no-trade probability keeps both directional probabilities below
    entry thresholds when quality is weak.
    """
    trend_bias = str(trend_bias or "").lower()
    trade_prob = max(0.0, min(1.0, float(trade_prob)))

    if trend_bias == "long":
        return np.asarray([0.0, trade_prob], dtype=float)
    if trend_bias == "short":
        return np.asarray([trade_prob, 0.0], dtype=float)
    return np.asarray([0.0, 0.0], dtype=float)


def weighted_predict_proba(models, X, model_weights=None, *, trend_bias=None, model_metadata=None):
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

    for name, model in models.items():
        try:
            weight = float(model_weights.get(name, 1.0))
        except (TypeError, ValueError):
            raise ValueError(f"模型 {name} 的权重不是有效数字: {model_weights.get(name)!r}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"模型 {name} 的权重必须是非负有限数: {weight!r}")
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
