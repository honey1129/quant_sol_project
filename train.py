import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report
import config
from utils import fetch_binance_data, add_indicators, get_feature_columns
import os

# 自动创建模型目录
os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

def create_labels(df, future_window=5, threshold=0.002):
    """
    增强版标签设计：
    - 未来5根K线收益率
    - 超过正阈值 -> 多头信号
    - 低于负阈值 -> 空头信号
    - 中性不交易信号（丢弃）
    """
    df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1

    # 多头 = 1, 空头 = 0
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

    # 取两者最小数量做平衡
    min_count = min(len(long_df), len(short_df))
    long_sample = resample(long_df, n_samples=min_count, replace=False, random_state=42)
    short_sample = resample(short_df, n_samples=min_count, replace=False, random_state=42)

    balanced_df = pd.concat([long_sample, short_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42)  # 打乱顺序

    return balanced_df.drop('target', axis=1), balanced_df['target']

def train():
    # 拉取数据
    df = fetch_binance_data(limit=3000)  # 拉长历史长度，提升训练稳定性
    df = add_indicators(df)
    df = create_labels(df)

    features = get_feature_columns()
    X = df[features].astype(float)
    y = df['target']

    # 样本平衡
    X_balanced, y_balanced = balance_samples(X, y)

    # 切分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, shuffle=False)

    # 训练模型
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)

    # 验证模型效果
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {acc:.4f}")
    print("分类报告：\n", classification_report(y_test, y_pred, digits=4))

    # 保存模型
    joblib.dump(model, config.MODEL_PATH)
    print("模型已保存到:", config.MODEL_PATH)

if __name__ == '__main__':
    train()
