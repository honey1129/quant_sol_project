import numpy as np
import joblib

# ============================
# 多模型加载（适配新版路径管理）
# ============================
def load_models(model_paths):
    """
    一次性批量加载多个模型文件
    """
    models = {}
    for name, path in model_paths.items():
        models[name] = joblib.load(path)
    return models

# ============================
# 简单平均融合
# ============================
def ensemble_predict(models, merged_df, feature_cols):
    """
    多模型简单平均融合
    """
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)
    predictions = []

    for name, model in models.items():
        prob = model.predict_proba(X_live)[0]  # 概率输出 [做多, 做空]
        predictions.append(prob)

    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

# ============================
# 贝叶斯加权融合
# ============================
def bayesian_weighted_predict(models, merged_df, feature_cols, model_weights):
    """
    多模型贝叶斯加权融合 (你主用逻辑)
    """
    X_live = merged_df[feature_cols].iloc[-1:].astype(float)
    weighted_sum = np.zeros(2)
    total_weight = sum(model_weights.values())

    for name, model in models.items():
        prob = model.predict_proba(X_live)[0]
        weight = model_weights.get(name, 1.0)
        weighted_sum += prob * weight

    avg_pred = weighted_sum / total_weight
    return avg_pred

# ============================
# 指数平滑信号去噪模块
# ============================
class SignalSmoother:
    """
    简单信号去噪平滑器（适合连续实盘平滑信号跳动）
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.smoothed_prob = None

    def smooth(self, new_prob):
        if self.smoothed_prob is None:
            self.smoothed_prob = new_prob
        else:
            self.smoothed_prob = self.alpha * new_prob + (1 - self.alpha) * self.smoothed_prob
        return self.smoothed_prob
