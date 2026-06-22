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

from core.ml_feature_engineering import merge_multi_period_features, add_advanced_features, model_feature_columns
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


TARGET_NO_TRADE = 0
TARGET_TRADE = 1
TARGET_DIRECTIONS = {
    TARGET_NO_TRADE: "no_trade",
    TARGET_TRADE: "trade",
}


def _target_direction(target):
    return TARGET_DIRECTIONS.get(int(target), "unknown")


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _label_use_realistic():
    return _env_bool("MODEL_LABEL_USE_REALISTIC", config.MODEL_LABEL_USE_REALISTIC)


def _label_lookahead_bars():
    return int(os.getenv("MODEL_LABEL_LOOKAHEAD_BARS", str(config.MODEL_LABEL_LOOKAHEAD_BARS)))


def _label_take_profit():
    return float(os.getenv("MODEL_LABEL_TAKE_PROFIT", str(config.MODEL_LABEL_TAKE_PROFIT)))


def _label_stop_loss():
    return float(os.getenv("MODEL_LABEL_STOP_LOSS", str(config.MODEL_LABEL_STOP_LOSS)))


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
    if direction == "no_trade":
        return True

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


def _label_mode():
    use_realistic = _label_use_realistic()
    return "binary_realistic" if use_realistic else "binary_threshold"


def _check_tp_sl_outcome(entry_price, future_bars, direction, take_profit_pct, stop_loss_pct):
    """
    检查从entry_price开仓,在future_bars期间,能否先触达止盈而不先触发止损

    Returns:
        'TP': 先触达止盈
        'SL': 先触发止损
        'TIMEOUT': 在给定时间内都没触达
    """
    if len(future_bars) == 0:
        return 'TIMEOUT'

    highs = future_bars['5m_high'].values
    lows = future_bars['5m_low'].values

    if direction == 'long':
        tp_price = entry_price * (1 + take_profit_pct)
        sl_price = entry_price * (1 - stop_loss_pct)

        for high, low in zip(highs, lows):
            # 悲观假设:同一根K线内,先看止损再看止盈
            if low <= sl_price:
                return 'SL'
            if high >= tp_price:
                return 'TP'
        return 'TIMEOUT'

    else:  # short
        tp_price = entry_price * (1 - take_profit_pct)
        sl_price = entry_price * (1 + stop_loss_pct)

        for high, low in zip(highs, lows):
            # 悲观假设:同一根K线内,先看止损再看止盈
            if high >= sl_price:
                return 'SL'
            if low <= tp_price:
                return 'TP'
        return 'TIMEOUT'


def create_labels(df, future_window=5, threshold=0.002, tradable_only=None, include_no_trade=None):
    """
    创建训练标签,二分类版本: trade vs no_trade

    逻辑:
        - 根据当前 trend_bias 确定交易方向
        - 检查该方向未来能否先触达止盈(不先止损)
        - 如果能止盈 -> TARGET_TRADE (1)
        - 否则 -> TARGET_NO_TRADE (0)

    这样模型只需要学习"什么时候该交易",方向由 trend filter 决定
    """
    df = df.copy()
    use_realistic = _label_use_realistic()

    if use_realistic:
        # 二分类realistic标签
        lookahead_bars = _label_lookahead_bars()
        take_profit_pct = _label_take_profit()
        stop_loss_pct = _label_stop_loss()

        log_info(f"使用二分类realistic标签: lookahead={lookahead_bars}根K线, TP={take_profit_pct:.2%}, SL={stop_loss_pct:.2%}")

        labels = []
        for i in range(len(df)):
            if i + lookahead_bars >= len(df):
                labels.append(np.nan)
                continue

            entry_price = df.iloc[i]['5m_close']
            future_bars = df.iloc[i+1:i+lookahead_bars+1]

            # 从特征推断 trend_bias
            trend_context, _ = _label_trade_context(df.iloc[i])
            trend_bias = trend_context.get('trend_bias', 'neutral')

            # 根据 trend 决定交易方向
            if trend_bias == 'long':
                direction = 'long'
            elif trend_bias == 'short':
                direction = 'short'
            else:
                # neutral 时不交易
                labels.append(TARGET_NO_TRADE)
                continue

            # 检查该方向能否止盈
            outcome = _check_tp_sl_outcome(entry_price, future_bars, direction, take_profit_pct, stop_loss_pct)

            if outcome == 'TP':
                labels.append(TARGET_TRADE)
            else:
                labels.append(TARGET_NO_TRADE)

        df['target'] = labels
        df = df[~df['target'].isna()].copy()

    else:
        # 旧逻辑保留兼容(但改成二分类)
        log_info(f"使用threshold标签(二分类): future_window={future_window}, threshold={threshold:.2%}")
        df['future_return'] = df['5m_close'].shift(-future_window) / df['5m_close'] - 1

        # 任何方向超过threshold都算 trade
        df['target'] = np.where(
            (df['future_return'] > threshold) | (df['future_return'] < -threshold),
            TARGET_TRADE,
            TARGET_NO_TRADE
        )
        df.dropna(subset=['future_return'], inplace=True)

    df["target"] = df["target"].astype(int)

    # tradable_only filter 不再需要(已经在标签生成时考虑了 trend)
    tradable_only = bool(config.MODEL_TRAIN_TRADABLE_LABELS if tradable_only is None else tradable_only)
    if tradable_only:
        log_info("⚠️ 二分类模式下 tradable_only filter 已集成到标签生成,跳过额外过滤")

    df.attrs["label_filter_summary"] = {
        "enabled": False,
        "raw_rows": int(len(df)),
        "kept_rows": int(len(df)),
        "blocked_rows": 0,
        "raw_direction_counts": {
            str(k): int(v)
            for k, v in df["target"].astype(int).map(_target_direction).value_counts().to_dict().items()
        },
        "kept_direction_counts": {
            str(k): int(v)
            for k, v in df["target"].astype(int).map(_target_direction).value_counts().to_dict().items()
        },
    }
    return df

def infer_sample_regimes(X):
    """二分类模式下不需要按regime分组,直接返回统一regime"""
    return pd.Series("all", index=X.index, dtype="object")


def build_sample_weights(X, y, *, recent_boost=None, min_weight=None, max_weight=None):
    """
    二分类样本权重: 简化为按类别反频率加权 + 时间衰减
    """
    y = y.astype(int)
    if len(y) == 0:
        return pd.Series(dtype=float, index=y.index, name="sample_weight"), {
            "enabled": True,
            "method": "binary_inverse_frequency_with_recency",
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

    # 二分类: 只有 trade 和 no_trade
    trade_multiplier = max(0.0, float(config.MODEL_TRADE_SAMPLE_WEIGHT_MULTIPLIER))
    no_trade_multiplier = max(0.0, float(config.MODEL_NO_TRADE_SAMPLE_WEIGHT_MULTIPLIER))

    directions = y.map(_target_direction)
    direction_counts = directions.value_counts(sort=False)
    direction_count = max(1, int(len(direction_counts)))

    def direction_weight(direction):
        multiplier = no_trade_multiplier if direction == "no_trade" else trade_multiplier
        return (len(y) / (direction_count * float(direction_counts[direction]))) * multiplier

    base_weights = directions.map(direction_weight).astype(float)

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

    direction_weight_mean = pd.DataFrame({"direction": directions, "sample_weight": weights}).groupby("direction")["sample_weight"].mean()
    direction_weight_total = pd.DataFrame({"direction": directions, "sample_weight": weights}).groupby("direction")["sample_weight"].sum()

    summary = {
        "enabled": True,
        "method": "binary_inverse_frequency_with_recency",
        "rows": int(len(y)),
        "recent_boost": float(recent_boost),
        "trade_multiplier": float(trade_multiplier),
        "no_trade_multiplier": float(no_trade_multiplier),
        "clip_min": float(min_weight),
        "clip_max": None if max_weight is None else float(max_weight),
        "mean": float(weights.mean()),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "direction_counts": {str(k): int(v) for k, v in direction_counts.sort_index().items()},
        "direction_weight_mean": {
            str(direction): float(value)
            for direction, value in direction_weight_mean.sort_index().items()
        },
        "direction_weight_total": {
            str(direction): float(value)
            for direction, value in direction_weight_total.sort_index().items()
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


def build_training_metadata(*, X, y, feature_cols, train_end, validation_start, validation_end, oos_start, original_train_rows, balanced_train_rows, validation_metrics, artifact_paths, label_filter_summary=None, sample_weight_summary=None, evaluation_sample_weight_summary=None, final_train_end=None):
    final_train_end = train_end if final_train_end is None else int(final_train_end)
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
        "label_use_realistic": bool(_label_use_realistic()),
        "label_lookahead_bars": int(_label_lookahead_bars()),
        "label_take_profit": float(_label_take_profit()),
        "label_stop_loss": float(_label_stop_loss()),
        "target_schema": "binary_trade_quality",
        "target_labels": {
            str(TARGET_NO_TRADE): "no_trade",
            str(TARGET_TRADE): "trade",
        },
        "label_mode": _label_mode(),
        "label_filter_summary": label_filter_summary or {},
        "training_balance_strategy": "sample_weight_binary_quality_recency",
        "sample_weight_summary": sample_weight_summary or {},
        "evaluation_sample_weight_summary": evaluation_sample_weight_summary or {},
        "train_ratio": float(config.MODEL_TRAIN_RATIO),
        "validation_ratio": float(config.MODEL_VALIDATION_RATIO),
        "purge_bars": int(config.MODEL_PURGE_BARS),
        "final_train_on_validation": bool(config.MODEL_FINAL_TRAIN_ON_VALIDATION),
        "row_count": int(len(X)),
        "train_rows": int(original_train_rows),
        "balanced_train_rows": int(balanced_train_rows),
        "final_train_rows": int(final_train_end),
        "validation_rows": int(validation_end - validation_start),
        "oos_rows": int(len(X.iloc[oos_start:])),
        "train_start": X.index[0].isoformat(),
        "train_end": X.index[train_end - 1].isoformat(),
        "final_train_start": X.index[0].isoformat(),
        "final_train_end": X.index[final_train_end - 1].isoformat(),
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
    rubik_data = None
    if bool(config.MODEL_USE_RUBIK_FEATURES):
        rubik_data = client.fetch_rubik_data(period=config.MODEL_RUBIK_PERIOD)
        log_info(f"已拉取 Rubik 特征数据 (period={config.MODEL_RUBIK_PERIOD})")
    merged_df = add_advanced_features(merged_df, rubik_data=rubik_data)
    merged_df = merged_df.dropna().copy()
    merged_df = create_labels(
        merged_df,
        future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
        threshold=float(config.MODEL_LABEL_THRESHOLD),
    )
    label_filter_summary = merged_df.attrs.get("label_filter_summary", {})
    if label_filter_summary:
        log_info(f"标签过滤摘要: {json.dumps(label_filter_summary, ensure_ascii=False, sort_keys=True)}")

    # 只把平稳特征喂给模型；绝对价格/量级列与 confirm 标志保留在 df 中供下游使用，
    # 但通过 model_feature_columns 排除，避免模型记忆训练期价位带。
    feature_cols = model_feature_columns(merged_df)
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

    # 只在训练集内部计算评估模型的样本权重，避免把未来样本混回验证过程。
    eval_models, X_eval_train, _, evaluation_sample_weight_summary = train_model_bundle(X_train, y_train)
    X_test = pd.DataFrame(X_test, columns=feature_cols)

    validation_metrics = {
        "lgb_v1": evaluate_model(eval_models["lgb_v1"], "LightGBM", X_test, y_test),
        "xgb_v1": evaluate_model(eval_models["xgb_v1"], "XGBoost", X_test, y_test),
        "rf_v1": evaluate_model(eval_models["rf_v1"], "RandomForest", X_test, y_test),
    }

    final_train_end = validation_end if bool(config.MODEL_FINAL_TRAIN_ON_VALIDATION) else train_end
    X_final_train = X.iloc[:final_train_end].copy()
    y_final_train = y.iloc[:final_train_end].copy()
    models, X_final_train, _, sample_weight_summary = train_model_bundle(X_final_train, y_final_train)

    lgb_model = models["lgb_v1"]
    joblib.dump(lgb_model, lgb_path)
    log_info(f"✅ LGB 模型已保存至: {lgb_path}")

    xgb_model = models["xgb_v1"]
    joblib.dump(xgb_model, xgb_path)
    log_info(f"✅ XGB 模型已保存至: {xgb_path}")

    rf_model = models["rf_v1"]
    joblib.dump(rf_model, rf_path)
    log_info(f"✅ RF 模型已保存至: {rf_path}")

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
        balanced_train_rows=len(X_final_train),
        validation_metrics=validation_metrics,
        artifact_paths=[lgb_path, xgb_path, rf_path, feature_path],
        label_filter_summary=label_filter_summary,
        sample_weight_summary=sample_weight_summary,
        evaluation_sample_weight_summary=evaluation_sample_weight_summary,
        final_train_end=final_train_end,
    )
    write_json_atomic(training_metadata_path, metadata)
    log_info(f"✅ 训练元数据已保存至: {training_metadata_path}")
    log_info(
        "样本切分: "
        f"train={metadata['train_rows']} validation={metadata['validation_rows']} "
        f"final_train={metadata['final_train_rows']} "
        f"oos={metadata['oos_rows']} oos_start={metadata['oos_start']}"
    )

if __name__ == '__main__':
    train()
