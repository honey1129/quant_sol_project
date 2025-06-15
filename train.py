import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import config
import os
from ml_feature_engineering import merge_multi_period_features
from okx_api import OKXClient

# 自动创建模型目录
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

def create_labels(df, future_window=5, threshold=0.002):
    """
    标签逻辑：
    - 未来5根5mK线收益率
    """
    df['future_return'] = df['5m_close'].shift(-future_window) / df['5m_close'] - 1
    df['target'] = np.where(df['future_return'] > threshold, 1,
                     np.where(df['future_return'] < -threshold, 0, np.nan))
    df.dropna(inplace=True)
    return df

def balance_samples(X, y):
    """
    简单平衡多空样本数量
    """
    df = pd.concat([X, y.rename('target')], axis=1)
    long_df = df[df['target'] == 1]
    short_df = df[df['target'] == 0]

    min_count = min(len(long_df), len(short_df))
    long_sample = resample(long_df, n_samples=min_count, replace=False, random_state=42)
    short_sample = resample(short_df, n_samples=min_count, replace=False, random_state=42)

    balanced_df = pd.concat([long_sample, short_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # 打乱
    return balanced_df.drop('target', axis=1), balanced_df['target']

def train():
    clint = OKXClient()
    # 多周期数据拉取
    data_dict = clint.fetch_data()

    # 多周期特征工程
    merged_df = merge_multi_period_features(data_dict)

    # 打标签
    merged_df = create_labels(merged_df, future_window=5, threshold=0.002)

    # 所有特征列：除了 'future_return', 'target' 之外的列
    feature_cols = [col for col in merged_df.columns if col not in ['future_return', 'target']]
    X = merged_df[feature_cols].astype(float)
    y = merged_df['target']

    # 样本平衡
    X_bal, y_bal = balance_samples(X, y)

    # 划分训练集、测试集
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, shuffle=False)

    # LightGBM 模型训练
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ 模型准确率: {acc:.4f}")
    print("分类报告:\n", classification_report(y_test, y_pred, digits=4))

    # 保存模型
    joblib.dump(model, config.MODEL_PATH)
    print(f"✅ 模型已保存至: {config.MODEL_PATH}")

    # 【新增】保存训练时使用的特征列表
    joblib.dump(feature_cols, 'models/feature_list.pkl')
    print("✅ 特征列已保存至: models/feature_list.pkl")

if __name__ == '__main__':
    train()
