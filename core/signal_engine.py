import os
import numpy as np
import joblib
import pandas as pd

from config import config
from utils.utils import BASE_DIR

# ============================
# 多模型加载（支持绝对路径）
# ============================
def load_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        full_path = os.path.join(BASE_DIR, path)
        models[name] = joblib.load(full_path)
    return models

# ============================
# 简单平均融合
# ============================
def ensemble_predict(models, merged_df, feature_cols):
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)
    X_live = pd.DataFrame(X_live, columns=feature_cols)  # ✅ 加入这一行保证 feature names 一致

    predictions = []

    for name, model in models.items():
        prob = model.predict_proba(X_live)[0]
        predictions.append(prob)

    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

# ============================
# 贝叶斯加权融合
# ============================
def bayesian_weighted_predict(models, merged_df, feature_cols, model_weights):
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)
    X_live = pd.DataFrame(X_live, columns=feature_cols)  # ✅ 加入这一行保证 feature names 一致
    weighted_sum = np.zeros(2)
    total_weight = sum(model_weights.values())

    for name, model in models.items():
        prob = model.predict_proba(X_live)[0]
        weight = model_weights.get(name, 1.0)
        weighted_sum += prob * weight

    avg_pred = weighted_sum / total_weight
    return avg_pred

# ============================
# 信号平滑去噪模块
# ============================
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
