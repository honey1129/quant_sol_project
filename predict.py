# predict_engine.py

import joblib
import config
from ml_feature_engineering import merge_multi_period_features
from okx_api import OKXClient


class MultiPeriodSignalPredictor:
    def __init__(self):
        self.model = joblib.load(config.MODEL_PATH)
        self.fetcher = OKXClient()

    def get_latest_signal(self):
        # 多周期拉取数据
        data_dict = self.fetcher.fetch_data()
        merged_df = merge_multi_period_features(data_dict)

        # 获取最近一行数据
        X_live = merged_df.drop(columns=['future_return', 'target'], errors='ignore').iloc[-1:].astype(float)

        # 模型预测
        prob = self.model.predict_proba(X_live)[0]
        long_prob, short_prob = prob[1], prob[0]

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
