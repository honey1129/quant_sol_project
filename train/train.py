import json
import hashlib
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import config
import os
import xgboost as xgb

from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features
from core.okx_api import OKXClient
from core.regime_filter import derive_market_regime, regime_allows_direction
from core.trend_filter import derive_trend_context, trend_allows_direction
from utils.utils import log_info, BASE_DIR

# 统一拼接绝对路径
lgb_path = os.path.join(BASE_DIR,config.MODEL_PATHS.get("lgb_v1"))
xgb_path = os.path.join(BASE_DIR, config.MODEL_PATHS.get("xgb_v1"))
rf_path  = os.path.join(BASE_DIR, config.MODEL_PATHS.get("rf_v1"))
feature_path = os.path.join(BASE_DIR, config.FEATURE_LIST_PATH)
training_metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH)

def _target_direction(target):
    if int(target) == 1:
        return "long"
    return "short"


def _row_atr_ratio(row):
    close_price = row.get("5m_close")
    atr_value = row.get("5m_atr")
    if pd.isna(close_price) or pd.isna(atr_value):
        return None
    close_price = float(close_price)
    if close_price <= 0:
        return None
    return float(atr_value) / close_price


def _label_trade_context(row):
    trend_context = derive_trend_context(
        row,
        interval=config.TREND_FILTER_INTERVAL,
        fast_col=config.TREND_FILTER_FAST_COL,
        slow_col=config.TREND_FILTER_SLOW_COL,
        min_gap=config.TREND_FILTER_MIN_GAP,
    )
    regime_context = derive_market_regime(
        trend_bias=trend_context.get("trend_bias"),
        trend_gap=trend_context.get("trend_gap"),
        volatility=row.get("volatility_15"),
        atr_ratio=_row_atr_ratio(row),
        money_flow_ratio=row.get("money_flow_ratio"),
        trend_gap_threshold=config.REGIME_TREND_GAP_THRESHOLD,
        high_vol_atr_threshold=config.REGIME_HIGH_VOL_ATR_THRESHOLD,
        high_volatility_threshold=config.REGIME_HIGH_VOLATILITY_THRESHOLD,
        money_flow_extreme_threshold=config.REGIME_MONEY_FLOW_EXTREME_THRESHOLD,
    )
    return trend_context, regime_context


def _target_is_tradable(row):
    direction = _target_direction(row["target"])
    trend_context, regime_context = _label_trade_context(row)
    regime = regime_context.get("regime")
    trend_bias = trend_context.get("trend_bias")

    if bool(config.REGIME_FILTER_ENABLED):
        if (
            bool(config.REGIME_TREND_AGAINST_BLOCK)
            or str(regime or "").lower() not in {"trend_long", "trend_short"}
        ):
            if not regime_allows_direction(
                regime,
                direction,
                allow_range=bool(config.REGIME_RANGE_ALLOW_TRADES),
                allow_high_vol=bool(config.REGIME_HIGH_VOL_ALLOW_TRADES),
            ):
                return False

    if bool(config.TREND_FILTER_ENABLED) and not trend_allows_direction(direction, trend_bias):
        return False

    return True


def _tradable_label_filter_summary(raw_df, filtered_df, blocked_mask):
    raw_counts = raw_df["target"].astype(int).map(_target_direction).value_counts().to_dict()
    kept_counts = filtered_df["target"].astype(int).map(_target_direction).value_counts().to_dict()
    blocked_df = raw_df[blocked_mask].copy()
    blocked_counts = blocked_df["target"].astype(int).map(_target_direction).value_counts().to_dict()
    return {
        "enabled": True,
        "raw_rows": int(len(raw_df)),
        "kept_rows": int(len(filtered_df)),
        "blocked_rows": int(len(blocked_df)),
        "raw_direction_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "kept_direction_counts": {str(k): int(v) for k, v in kept_counts.items()},
        "blocked_direction_counts": {str(k): int(v) for k, v in blocked_counts.items()},
    }


def create_labels(df, future_window=5, threshold=0.002, tradable_only=None):
    df = df.copy()
    df['future_return'] = df['5m_close'].shift(-future_window) / df['5m_close'] - 1
    df['target'] = np.where(df['future_return'] > threshold, 1,
                     np.where(df['future_return'] < -threshold, 0, np.nan))
    df.dropna(subset=['target'], inplace=True)
    df["target"] = df["target"].astype(int)
    tradable_only = bool(config.MODEL_TRAIN_TRADABLE_LABELS if tradable_only is None else tradable_only)
    if tradable_only:
        allowed_mask = df.apply(_target_is_tradable, axis=1).astype(bool)
        blocked_mask = ~allowed_mask
        filtered_df = df[allowed_mask].copy()
        filtered_df.attrs["label_filter_summary"] = _tradable_label_filter_summary(
            df,
            filtered_df,
            blocked_mask,
        )
        return filtered_df

    df.attrs["label_filter_summary"] = {
        "enabled": False,
        "raw_rows": int(len(df)),
        "kept_rows": int(len(df)),
        "blocked_rows": 0,
    }
    return df

def infer_sample_regimes(X):
    if X.empty:
        return pd.Series(dtype="object", index=X.index)

    explicit_cols = {
        "regime_trend_long",
        "regime_trend_short",
        "regime_range_high_vol",
        "is_high_vol",
    }
    if explicit_cols & set(X.columns):
        regimes = pd.Series("range", index=X.index, dtype="object")

        def enabled(col):
            if col not in X.columns:
                return pd.Series(False, index=X.index)
            return pd.to_numeric(X[col], errors="coerce").fillna(0.0) >= 0.5

        regimes.loc[enabled("is_high_vol")] = "range_high_vol"
        regimes.loc[enabled("regime_range_high_vol")] = "range_high_vol"
        regimes.loc[enabled("regime_trend_long")] = "trend_long"
        regimes.loc[enabled("regime_trend_short")] = "trend_short"
        return regimes

    return X.apply(
        lambda row: str(_label_trade_context(row)[1].get("regime") or "unknown").lower(),
        axis=1,
    )


def build_sample_weights(X, y, *, recent_boost=None, min_weight=None, max_weight=None):
    y = y.astype(int)
    if len(y) == 0:
        return pd.Series(dtype=float, index=y.index, name="sample_weight"), {
            "enabled": True,
            "method": "regime_direction_inverse_frequency_with_recency",
            "rows": 0,
        }

    recent_boost = max(
        0.0,
        float(config.MODEL_RECENT_SAMPLE_WEIGHT_BOOST if recent_boost is None else recent_boost),
    )
    min_weight = float(config.MODEL_SAMPLE_WEIGHT_MIN if min_weight is None else min_weight)
    max_weight = config.MODEL_SAMPLE_WEIGHT_MAX if max_weight is None else max_weight
    max_weight = None if max_weight is None else float(max_weight)
    if max_weight is not None and max_weight < min_weight:
        raise ValueError("MODEL_SAMPLE_WEIGHT_MAX 不能小于 MODEL_SAMPLE_WEIGHT_MIN")

    regimes = infer_sample_regimes(X).reindex(y.index).fillna("unknown")
    directions = y.map(_target_direction)
    group_df = pd.DataFrame({
        "regime": regimes.astype(str),
        "direction": directions.astype(str),
    }, index=y.index)
    group_labels = group_df["regime"] + ":" + group_df["direction"]
    group_counts = group_labels.value_counts(sort=False)
    group_count = max(1, int(len(group_counts)))
    base_weights = group_labels.map(
        lambda group: len(y) / (group_count * float(group_counts[group]))
    ).astype(float)

    if len(base_weights) == 1:
        recency_weights = np.array([1.0 + recent_boost], dtype=float)
    else:
        recency_weights = np.linspace(1.0, 1.0 + recent_boost, len(base_weights))
    weights = pd.Series(base_weights.to_numpy() * recency_weights, index=y.index, name="sample_weight")

    if min_weight > 0 or max_weight is not None:
        weights = weights.clip(lower=min_weight, upper=max_weight)
    mean_weight = float(weights.mean())
    if mean_weight > 0:
        weights = weights / mean_weight

    weighted_groups = group_df.assign(sample_weight=weights)
    group_weight_mean = weighted_groups.groupby(["regime", "direction"])["sample_weight"].mean()
    summary = {
        "enabled": True,
        "method": "regime_direction_inverse_frequency_with_recency",
        "rows": int(len(y)),
        "recent_boost": float(recent_boost),
        "clip_min": float(min_weight),
        "clip_max": None if max_weight is None else float(max_weight),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "group_counts": {str(k): int(v) for k, v in group_counts.sort_index().items()},
        "group_weight_mean": {
            f"{regime}:{direction}": float(value)
            for (regime, direction), value in group_weight_mean.sort_index().items()
        },
    }
    return weights, summary


def balance_samples(X, y):
    sample_weight, weight_summary = build_sample_weights(X, y)
    return X.copy(), y.copy(), sample_weight, weight_summary

def evaluate_model(model, model_name, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    log_info(f"✅ {model_name} 准确率: {acc:.4f}")
    log_info(f"分类报告:\n{classification_report(y_test, y_pred, digits=4)}")
    return {
        "accuracy": float(acc),
        "classification_report": _json_safe(report),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def build_training_metadata(*, X, y, feature_cols, train_end, validation_start, validation_end, oos_start, original_train_rows, balanced_train_rows, validation_metrics, artifact_paths, label_filter_summary=None, sample_weight_summary=None):
    artifact_hashes = {
        os.path.relpath(path, BASE_DIR): sha256_file(path)
        for path in artifact_paths
        if os.path.exists(path)
    }
    label_distribution = {
        "all": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train": {str(k): int(v) for k, v in y.iloc[:train_end].value_counts().sort_index().items()},
        "validation": {str(k): int(v) for k, v in y.iloc[validation_start:validation_end].value_counts().sort_index().items()},
        "oos": {str(k): int(v) for k, v in y.iloc[oos_start:].value_counts().sort_index().items()},
    }
    return {
        "schema_version": 2,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "source": "train.train",
        "symbol": config.SYMBOL,
        "intervals": list(config.INTERVALS),
        "model_paths": dict(config.MODEL_PATHS),
        "model_weights": dict(config.MODEL_WEIGHTS),
        "feature_list_path": config.FEATURE_LIST_PATH,
        "training_metadata_path": config.TRAINING_METADATA_PATH,
        "artifact_hashes": artifact_hashes,
        "feature_count": int(len(feature_cols)),
        "feature_columns_sha256": hashlib.sha256("\n".join(feature_cols).encode("utf-8")).hexdigest(),
        "label_distribution": label_distribution,
        "validation_metrics": validation_metrics,
        "label_future_window": int(config.MODEL_LABEL_FUTURE_WINDOW),
        "label_threshold": float(config.MODEL_LABEL_THRESHOLD),
        "label_mode": "tradable_binary" if bool(config.MODEL_TRAIN_TRADABLE_LABELS) else "raw_binary",
        "label_filter_summary": label_filter_summary or {},
        "training_balance_strategy": "sample_weight_regime_direction_recency",
        "sample_weight_summary": sample_weight_summary or {},
        "train_ratio": float(config.MODEL_TRAIN_RATIO),
        "validation_ratio": float(config.MODEL_VALIDATION_RATIO),
        "purge_bars": int(config.MODEL_PURGE_BARS),
        "row_count": int(len(X)),
        "train_rows": int(original_train_rows),
        "balanced_train_rows": int(balanced_train_rows),
        "validation_rows": int(validation_end - validation_start),
        "oos_rows": int(len(X.iloc[oos_start:])),
        "train_start": X.index[0].isoformat(),
        "train_end": X.index[train_end - 1].isoformat(),
        "validation_start": X.index[validation_start].isoformat(),
        "validation_end": X.index[validation_end - 1].isoformat(),
        "oos_start": X.index[oos_start].isoformat(),
        "oos_end": X.index[-1].isoformat(),
    }


def build_model_estimators():
    return {
        "lgb_v1": lgb.LGBMClassifier(
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
        ),
        "xgb_v1": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42
        ),
        "rf_v1": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
    }


def train_model_bundle(X_train, y_train):
    X_balanced, y_balanced, sample_weight, sample_weight_summary = balance_samples(X_train, y_train)
    X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)

    models = build_model_estimators()
    for model in models.values():
        model.fit(X_balanced, y_balanced, sample_weight=sample_weight)
    return models, X_balanced, y_balanced, sample_weight_summary


def build_time_splits(length):
    train_ratio = float(config.MODEL_TRAIN_RATIO)
    validation_ratio = float(config.MODEL_VALIDATION_RATIO)
    purge_bars = max(0, int(config.MODEL_PURGE_BARS))

    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("MODEL_TRAIN_RATIO 和 MODEL_VALIDATION_RATIO 必须为正，且总和小于 1")

    train_end = int(length * train_ratio)
    validation_start = train_end + purge_bars
    validation_end = int(length * (train_ratio + validation_ratio))
    oos_start = validation_end + purge_bars

    if train_end <= 0 or validation_start >= validation_end or oos_start >= length:
        raise ValueError("样本量不足，无法切分 train/validation/oos")

    return train_end, validation_start, validation_end, oos_start


def train():
    client = OKXClient()
    data_dict = client.fetch_data()
    merged_df = merge_multi_period_features(data_dict)
    merged_df = add_advanced_features(merged_df)
    merged_df = merged_df.dropna().copy()
    merged_df = create_labels(
        merged_df,
        future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
        threshold=float(config.MODEL_LABEL_THRESHOLD),
    )
    label_filter_summary = merged_df.attrs.get("label_filter_summary", {})
    if label_filter_summary:
        log_info(f"标签过滤摘要: {json.dumps(label_filter_summary, ensure_ascii=False, sort_keys=True)}")

    feature_cols = [col for col in merged_df.columns if col not in ['future_return', 'target']]
    X = merged_df[feature_cols].astype(float)
    y = merged_df['target']

    train_end, validation_start, validation_end, oos_start = build_time_splits(len(X))
    if len(X.iloc[oos_start:]) < int(config.MODEL_RETRAIN_MIN_OOS_ROWS):
        raise ValueError(
            f"OOS样本不足: rows={len(X.iloc[oos_start:])} < {int(config.MODEL_RETRAIN_MIN_OOS_ROWS)}"
        )

    original_train_rows = train_end
    X_train = X.iloc[:train_end].copy()
    X_test = X.iloc[validation_start:validation_end].copy()
    y_train = y.iloc[:train_end].copy()
    y_test = y.iloc[validation_start:validation_end].copy()

    # 只在训练集内部计算样本权重，避免把未来样本混回训练过程。
    models, X_train, y_train, sample_weight_summary = train_model_bundle(X_train, y_train)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    lgb_model = models["lgb_v1"]
    joblib.dump(lgb_model, lgb_path)
    log_info(f"✅ LGB 模型已保存至: {lgb_path}")

    xgb_model = models["xgb_v1"]
    joblib.dump(xgb_model, xgb_path)
    log_info(f"✅ XGB 模型已保存至: {xgb_path}")

    rf_model = models["rf_v1"]
    joblib.dump(rf_model, rf_path)
    log_info(f"✅ RF 模型已保存至: {rf_path}")

    validation_metrics = {
        "lgb_v1": evaluate_model(lgb_model, "LightGBM", X_test, y_test),
        "xgb_v1": evaluate_model(xgb_model, "XGBoost", X_test, y_test),
        "rf_v1": evaluate_model(rf_model, "RandomForest", X_test, y_test),
    }

    joblib.dump(feature_cols, feature_path)
    log_info(f"✅ 特征列已保存至: {feature_path}")

    metadata = build_training_metadata(
        X=X,
        y=y,
        feature_cols=feature_cols,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        oos_start=oos_start,
        original_train_rows=original_train_rows,
        balanced_train_rows=len(X_train),
        validation_metrics=validation_metrics,
        artifact_paths=[lgb_path, xgb_path, rf_path, feature_path],
        label_filter_summary=label_filter_summary,
        sample_weight_summary=sample_weight_summary,
    )
    write_json_atomic(training_metadata_path, metadata)
    log_info(f"✅ 训练元数据已保存至: {training_metadata_path}")
    log_info(
        "样本切分: "
        f"train={metadata['train_rows']} validation={metadata['validation_rows']} "
        f"oos={metadata['oos_rows']} oos_start={metadata['oos_start']}"
    )

if __name__ == '__main__':
    train()
