# predict_engine.py

import os
import joblib
import numpy as np
import pandas as pd

from config import config
from core.ml_feature_engineering import merge_multi_period_features
from core.okx_api import OKXClient
from utils.utils import BASE_DIR

class MultiPeriodSignalPredictor:
    def __init__(self):
        self.fetcher = OKXClient()
        self.model_paths = {name: os.path.join(BASE_DIR, path) for name, path in config.MODEL_PATHS.items()}
        self.models = {name: joblib.load(path) for name, path in self.model_paths.items()}
        self.model_weights = config.MODEL_WEIGHTS

    def get_latest_signal(self):
        # 多周期拉取数据
        data_dict = self.fetcher.fetch_data()
        merged_df = merge_multi_period_features(data_dict)

        # 获取最近一行数据
        feature_cols = joblib.load(os.path.join(BASE_DIR, config.FEATURE_LIST_PATH))
        X_live = merged_df[feature_cols].iloc[-1:].astype(float)
        X_live = pd.DataFrame(X_live, columns=feature_cols)

        # 多模型融合预测
        weighted_sum = np.zeros(2)
        total_weight = sum(self.model_weights.values())

        for name, model in self.models.items():
            prob = model.predict_proba(X_live)[0]
            weight = self.model_weights.get(name, 1.0)
            weighted_sum += prob * weight

        avg_prob = weighted_sum / total_weight
        long_prob, short_prob = avg_prob[1], avg_prob[0]

        print(f"实时预测概率 => 多头: {long_prob:.3f} | 空头: {short_prob:.3f}")

        # 阈值判定信号
        if long_prob > config.THRESHOLD_LONG:
            return 'long'
        elif short_prob > config.THRESHOLD_SHORT:
            return 'short'
        else:
            return 'neutral'

# 示例
if __name__ == '__main__':
    predictor = MultiPeriodSignalPredictor()
    signal = predictor.get_latest_signal()
    print(f"✅ 当前信号: {signal.upper()}")
