import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import config
import os
import xgboost as xgb

from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features
from core.okx_api import OKXClient
from utils.utils import log_info, BASE_DIR

# 统一拼接绝对路径
lgb_path = os.path.join(BASE_DIR,config.MODEL_PATHS.get("lgb_v1"))
xgb_path = os.path.join(BASE_DIR, config.MODEL_PATHS.get("xgb_v1"))
rf_path  = os.path.join(BASE_DIR, config.MODEL_PATHS.get("rf_v1"))
feature_path = os.path.join(BASE_DIR, config.FEATURE_LIST_PATH)

def create_labels(df, future_window=5, threshold=0.002):
    df['future_return'] = df['5m_close'].shift(-future_window) / df['5m_close'] - 1
    df['target'] = np.where(df['future_return'] > threshold, 1,
                     np.where(df['future_return'] < -threshold, 0, np.nan))
    df.dropna(subset=['target'], inplace=True)
    return df

def balance_samples(X, y):
    df = pd.concat([X, y.rename('target')], axis=1)
    long_df = df[df['target'] == 1]
    short_df = df[df['target'] == 0]
    min_count = min(len(long_df), len(short_df))
    long_sample = resample(long_df, n_samples=min_count, replace=False, random_state=42)
    short_sample = resample(short_df, n_samples=min_count, replace=False, random_state=42)
    balanced_df = pd.concat([long_sample, short_sample])
    balanced_df = balanced_df.sample(frac=1, random_state=42)
    return balanced_df.drop('target', axis=1), balanced_df['target']

def train():
    client = OKXClient()
    data_dict = client.fetch_data()
    merged_df = merge_multi_period_features(data_dict)
    merged_df = add_advanced_features(merged_df)
    merged_df = create_labels(merged_df, future_window=5, threshold=0.002)

    feature_cols = [col for col in merged_df.columns if col not in ['future_return', 'target']]
    X = merged_df[feature_cols].astype(float)
    y = merged_df['target']

    X_bal, y_bal = balance_samples(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, shuffle=False)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=5,
        min_split_gain=0.0,
        force_col_wise=True,
        random_state=42
    )

    lgb_model.fit(X_train, y_train)
    joblib.dump(lgb_model, lgb_path)
    log_info(f"✅ LGB 模型已保存至: {lgb_path}")

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, xgb_path)
    log_info(f"✅ XGB 模型已保存至: {xgb_path}")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, rf_path)
    log_info(f"✅ RF 模型已保存至: {rf_path}")

    # 评估示例（以LightGBM为例）
    y_pred = lgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log_info(f"✅ 模型准确率: {acc:.4f}")
    log_info(f"分类报告:\n{classification_report(y_test, y_pred, digits=4)}")

    joblib.dump(feature_cols, feature_path)
    log_info(f"✅ 特征列已保存至: {feature_path}")

if __name__ == '__main__':
    train()
