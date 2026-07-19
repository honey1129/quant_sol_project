import argparse
import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from config import config
from utils.utils import BASE_DIR, LOGS_DIR, notify_important


LOCK_PATH = os.path.join(LOGS_DIR, "model_retrain.lock")
STATE_PATH = os.path.join(LOGS_DIR, "model_retrain_state.json")
BACKUP_ROOT = os.path.join(BASE_DIR, "models", "backups")
ABSOLUTE_MIN_CLOSED_TRADES = 1
ABSOLUTE_MIN_PROFIT_FACTOR = 1.0
HARD_MIN_CLOSED_TRADES = max(
    ABSOLUTE_MIN_CLOSED_TRADES,
    int(getattr(config, "MODEL_RETRAIN_HARD_MIN_CLOSED_TRADES", ABSOLUTE_MIN_CLOSED_TRADES)),
)
HARD_MIN_PROFIT_FACTOR = max(
    ABSOLUTE_MIN_PROFIT_FACTOR,
    float(getattr(config, "MODEL_RETRAIN_HARD_MIN_PROFIT_FACTOR", ABSOLUTE_MIN_PROFIT_FACTOR)),
)
IMPROVEMENT_EPSILON = 1e-9


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def pid_is_running(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def acquire_lock():
    if os.path.exists(LOCK_PATH):
        lock = read_json(LOCK_PATH, {})
        pid = lock.get("pid")
        if pid and pid_is_running(pid):
            raise RuntimeError(f"模型重训已在运行: pid={pid}")
    write_json_atomic(LOCK_PATH, {"pid": os.getpid(), "started_at": utc_now_iso()})


def release_lock():
    try:
        os.remove(LOCK_PATH)
    except FileNotFoundError:
        pass


def artifact_paths():
    paths = []
    for rel_path in config.MODEL_PATHS.values():
        paths.append(os.path.join(BASE_DIR, rel_path))
    paths.append(os.path.join(BASE_DIR, config.FEATURE_LIST_PATH))
    paths.append(os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH))
    return sorted(set(paths))


def make_backup(run_id):
    backup_dir = os.path.join(BACKUP_ROOT, f"retrain_{run_id}")
    manifest = []
    os.makedirs(backup_dir, exist_ok=True)

    for src_path in artifact_paths():
        rel_path = os.path.relpath(src_path, BASE_DIR)
        dst_path = os.path.join(backup_dir, rel_path)
        exists = os.path.exists(src_path)
        manifest.append({"path": rel_path, "exists": exists})
        if exists:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

    write_json_atomic(os.path.join(backup_dir, "manifest.json"), manifest)
    return backup_dir, manifest


def restore_backup(backup_dir, manifest):
    for item in manifest:
        rel_path = item["path"]
        dst_path = os.path.join(BASE_DIR, rel_path)
        src_path = os.path.join(backup_dir, rel_path)
        if item.get("exists") and os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
        elif os.path.exists(dst_path):
            os.remove(dst_path)


def preserve_candidate_training_metadata(backup_dir):
    """Keep failed candidate metadata for diagnostics before restoring old artifacts."""
    if not backup_dir:
        return None

    metadata_dir = os.path.dirname(os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH))
    candidate_metadata_path = os.path.join(metadata_dir, "candidate_training_metadata.json")
    metadata_path = candidate_metadata_path
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH)
    if not os.path.exists(metadata_path):
        return None

    dst_path = os.path.join(backup_dir, "candidate_training_metadata.json")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(metadata_path, dst_path)
    return dst_path


def update_candidate_training_metadata(**updates):
    metadata_dir = os.path.dirname(os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH))
    candidate_metadata_path = os.path.join(metadata_dir, "candidate_training_metadata.json")
    metadata_path = candidate_metadata_path
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH)
    payload = read_json(metadata_path, {})
    if not isinstance(payload, dict):
        payload = {}
    payload.update(updates)
    write_json_atomic(candidate_metadata_path, payload)
    return candidate_metadata_path


def remove_candidate_training_metadata():
    metadata_dir = os.path.dirname(os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH))
    candidate_metadata_path = os.path.join(metadata_dir, "candidate_training_metadata.json")
    try:
        os.remove(candidate_metadata_path)
    except FileNotFoundError:
        pass


def prune_backups(keep_count):
    keep_count = max(1, int(keep_count))
    if not os.path.isdir(BACKUP_ROOT):
        return

    backup_dirs = []
    for name in os.listdir(BACKUP_ROOT):
        path = os.path.join(BACKUP_ROOT, name)
        if name.startswith("retrain_") and os.path.isdir(path):
            backup_dirs.append(path)

    backup_dirs.sort(reverse=True)
    for old_path in backup_dirs[keep_count:]:
        shutil.rmtree(old_path, ignore_errors=True)


def validate_artifacts():
    loaded = []
    joblib_paths = [
        *(os.path.join(BASE_DIR, rel_path) for rel_path in config.MODEL_PATHS.values()),
        os.path.join(BASE_DIR, config.FEATURE_LIST_PATH),
    ]
    for path in sorted(set(joblib_paths)):
        if not os.path.exists(path):
            raise RuntimeError(f"训练后缺少模型产物: {path}")
        joblib.load(path)
        loaded.append(os.path.relpath(path, BASE_DIR))

    metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH)
    if not os.path.exists(metadata_path):
        raise RuntimeError(f"训练后缺少模型产物: {metadata_path}")
    metadata = read_json(metadata_path, {})
    if not metadata:
        raise RuntimeError(f"训练元数据无效: {metadata_path}")
    # New training runs must include strict OOS split fields. Legacy artifacts that predate
    # metadata tracking are allowed only when explicitly backfilled with artifact hashes;
    # they are useful for version traceability but are not sufficient for retrain promotion.
    if not metadata.get("oos_start") and not metadata.get("artifact_hashes"):
        raise RuntimeError(f"训练元数据无效或缺少 oos_start: {metadata_path}")
    loaded.append(os.path.relpath(metadata_path, BASE_DIR))
    return loaded


def load_model_bundle(root_dir):
    models = {}
    for name, rel_path in config.MODEL_PATHS.items():
        path = os.path.join(root_dir, rel_path)
        if not os.path.exists(path):
            raise RuntimeError(f"缺少模型文件: {path}")
        models[name] = joblib.load(path)

    feature_path = os.path.join(root_dir, config.FEATURE_LIST_PATH)
    if not os.path.exists(feature_path):
        raise RuntimeError(f"缺少特征列文件: {feature_path}")

    metadata_path = os.path.join(root_dir, config.TRAINING_METADATA_PATH)
    metadata = None
    if os.path.exists(metadata_path):
        metadata = read_json(metadata_path, {})

    return {
        "models": models,
        "feature_cols": joblib.load(feature_path),
        "metadata": metadata,
        "root_dir": root_dir,
    }


def append_log_header(log_file, title):
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"\n===== {title} {utc_now_iso()} =====\n")


def run_subprocess(args, log_file):
    append_log_header(log_file, "subprocess")
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"$ {' '.join(args)}\n")
        file.flush()
        return subprocess.run(
            args,
            cwd=BASE_DIR,
            stdout=file,
            stderr=subprocess.STDOUT,
            text=True,
        ).returncode


def run_backtest_validation(log_file, backup_dir=None):
    from backtest.backtest import Backtester

    new_bundle = load_model_bundle(BASE_DIR)
    append_log_header(log_file, "backtest_validation_context")
    base_interval = config.INTERVALS[0] if config.INTERVALS else "5m"
    window = config.WINDOWS.get(base_interval, 1000)
    with open(log_file, "a", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file):
            context_backtester = Backtester(
                "multi_period",
                window,
                enable_csv_dump=False,
                show_progress=False,
                emit_diagnostics=False,
            )
    walk_forward_summary = run_walk_forward_validation(
        log_file,
        context_backtester,
        new_bundle.get("metadata"),
        new_bundle["feature_cols"],
    )
    context_backtester = restrict_backtester_to_oos(context_backtester, new_bundle.get("metadata"))

    old_summary = None
    if backup_dir is not None:
        old_bundle = load_model_bundle(backup_dir)
        old_summary = run_backtest_with_bundle(
            log_file,
            "backtest_baseline_old_model",
            context_backtester,
            old_bundle,
        )

    summary = run_backtest_with_bundle(
        log_file,
        "backtest_candidate_new_model",
        context_backtester,
        new_bundle,
    )
    if not summary:
        raise RuntimeError("回测验证未返回 summary")

    validate_backtest_summary(summary)
    comparison = None
    if old_summary is not None:
        comparison = validate_new_model_improvement(summary, old_summary)
        summary["baseline_summary"] = old_summary
        summary["comparison"] = comparison
    if walk_forward_summary is not None:
        summary["walk_forward_summary"] = walk_forward_summary
    return summary


def coerce_timestamp_for_index(value, index):
    timestamp = pd.Timestamp(value)
    index_tz = getattr(index, "tz", None)
    if index_tz is None and timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    if index_tz is not None and timestamp.tzinfo is None:
        return timestamp.tz_localize(index_tz)
    return timestamp


def filter_funding_history(funding_history, start_ts, end_ts=None):
    if funding_history is None or funding_history.empty:
        return funding_history

    filtered = funding_history[funding_history["funding_time"] >= start_ts]
    if end_ts is not None:
        filtered = filtered[filtered["funding_time"] <= end_ts]
    return filtered.copy()


def restrict_backtester_to_oos(context_backtester, metadata):
    if not metadata:
        raise RuntimeError("训练元数据缺失，无法执行严格样本外回测")

    oos_start = metadata.get("oos_start")
    if not oos_start:
        raise RuntimeError("训练元数据缺少 oos_start，无法执行严格样本外回测")

    oos_start_ts = coerce_timestamp_for_index(oos_start, context_backtester.data.index)
    oos_data = context_backtester.data.loc[context_backtester.data.index >= oos_start_ts].copy()
    min_oos_rows = int(config.MODEL_RETRAIN_MIN_OOS_ROWS)
    if len(oos_data) < min_oos_rows:
        raise RuntimeError(f"OOS回测样本不足: rows={len(oos_data)} < {min_oos_rows}")

    context_backtester.data = oos_data
    context_backtester.price_series = oos_data["5m_close"]
    context_backtester.funding_history = filter_funding_history(
        context_backtester.funding_history,
        oos_data.index.min(),
    )
    return context_backtester


def build_walk_forward_slices(index, metadata):
    if not metadata:
        raise RuntimeError("训练元数据缺失，无法执行 walk-forward 验证")

    required_keys = ("validation_start", "validation_end")
    missing = [key for key in required_keys if not metadata.get(key)]
    if missing:
        raise RuntimeError(f"训练元数据缺少 walk-forward 字段: {','.join(missing)}")

    validation_start = coerce_timestamp_for_index(metadata["validation_start"], index)
    validation_end = coerce_timestamp_for_index(metadata["validation_end"], index)
    validation_start_pos = int(index.searchsorted(validation_start, side="left"))
    validation_end_pos = int(index.searchsorted(validation_end, side="right"))
    validation_rows = validation_end_pos - validation_start_pos

    min_validation_rows = max(1, int(config.MODEL_WALK_FORWARD_MIN_VALIDATION_ROWS))
    requested_folds = max(1, int(config.MODEL_WALK_FORWARD_FOLDS))
    possible_folds = validation_rows // min_validation_rows
    fold_count = min(requested_folds, possible_folds)
    min_folds = max(1, int(config.MODEL_WALK_FORWARD_MIN_FOLDS))
    if fold_count < min_folds:
        raise RuntimeError(
            "walk-forward 验证折数不足: "
            f"folds={fold_count}, required={min_folds}, validation_rows={validation_rows}, "
            f"min_validation_rows={min_validation_rows}"
        )

    purge_bars = max(0, int(metadata.get("purge_bars", config.MODEL_PURGE_BARS)))
    base_size = validation_rows // fold_count
    remainder = validation_rows % fold_count
    slices = []
    cursor = validation_start_pos

    for fold_idx in range(fold_count):
        fold_size = base_size + (1 if fold_idx < remainder else 0)
        validation_fold_start = cursor
        validation_fold_end = cursor + fold_size
        train_end = validation_fold_start - purge_bars
        if train_end <= 0:
            raise RuntimeError(
                "walk-forward 训练样本不足: "
                f"fold={fold_idx + 1}, train_rows={train_end}, purge_bars={purge_bars}"
            )
        slices.append({
            "fold": fold_idx + 1,
            "train_start_pos": 0,
            "train_end_pos": train_end,
            "validation_start_pos": validation_fold_start,
            "validation_end_pos": validation_fold_end,
        })
        cursor = validation_fold_end

    return slices


def aggregate_regime_signal_summaries(summaries):
    aggregate = {}
    for summary in summaries:
        regime_summary = summary.get("decision_regime_signal_summary") or {}
        for regime, stats in regime_summary.items():
            item = aggregate.setdefault(str(regime), {
                "rows": 0,
                "dominant_long_count": 0,
                "dominant_short_count": 0,
            })
            item["rows"] += int(stats.get("rows", 0))
            item["dominant_long_count"] += int(stats.get("dominant_long_count", 0))
            item["dominant_short_count"] += int(stats.get("dominant_short_count", 0))

    for item in aggregate.values():
        rows = max(int(item.get("rows", 0)), 1)
        item["dominant_long_pct"] = item["dominant_long_count"] / rows * 100.0
        item["dominant_short_pct"] = item["dominant_short_count"] / rows * 100.0
    return aggregate


def aggregate_edge_gate_summaries(summaries):
    counts = {}
    for summary in summaries:
        edge_summary = summary.get("decision_edge_gate_summary") or {}
        for key, count in (edge_summary.get("counts") or {}).items():
            counts[str(key)] = counts.get(str(key), 0) + int(count)

    total = sum(counts.values())
    passed = int(counts.get("pass", 0))
    failed = int(counts.get("fail", 0))
    return {
        "counts": counts,
        "pass_pct": float(passed / total * 100.0) if total else 0.0,
        "fail_pct": float(failed / total * 100.0) if total else 0.0,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def _direction_counts_from_probabilities(long_probs, short_probs):
    counts = {"long": 0, "short": 0, "flat": 0}
    for long_prob, short_prob in zip(long_probs, short_probs):
        long_prob = float(long_prob)
        short_prob = float(short_prob)
        if long_prob > short_prob:
            counts["long"] += 1
        elif short_prob > long_prob:
            counts["short"] += 1
        else:
            counts["flat"] += 1
    total = max(1, sum(counts.values()))
    return {
        **counts,
        "long_pct": counts["long"] / total * 100.0,
        "short_pct": counts["short"] / total * 100.0,
        "flat_pct": counts["flat"] / total * 100.0,
    }


def _active_direction_counts(long_probs, short_probs, is_active):
    counts = {"long": 0, "short": 0, "flat": 0}
    for long_prob, short_prob, active in zip(long_probs, short_probs, is_active):
        if not bool(active):
            counts["flat"] += 1
        elif float(long_prob) > float(short_prob):
            counts["long"] += 1
        elif float(short_prob) > float(long_prob):
            counts["short"] += 1
        else:
            counts["flat"] += 1
    total = max(1, sum(counts.values()))
    return {
        **counts,
        "long_pct": counts["long"] / total * 100.0,
        "short_pct": counts["short"] / total * 100.0,
        "flat_pct": counts["flat"] / total * 100.0,
    }


def _target_counts(y):
    return {str(k): int(v) for k, v in y.astype(int).value_counts().sort_index().items()}


def _series_counts(df, col):
    if col not in df:
        return {}
    return {str(k): int(v) for k, v in df[col].fillna("unknown").astype(str).value_counts().sort_index().items()}


def _quantiles(values):
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    if series.empty:
        return {}
    return {
        "mean": float(series.mean()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def _prediction_slice_summary(df, mask):
    subset = df.loc[mask].copy()
    return {
        "rows": int(len(subset)),
        "regime_counts": _series_counts(subset, "label_regime"),
        "direction_counts": _series_counts(subset, "label_direction"),
        "outcome_counts": _series_counts(subset, "label_outcome"),
        "reject_reason_counts": _series_counts(subset, "label_reject_reason"),
        "trade_prob_quantiles": _quantiles(subset.get("_trade_prob", [])),
    }


def _label_quality_summary(df):
    rows = int(len(df))
    if rows == 0 or "target" not in df:
        return {"rows": rows, "trade_rows": 0, "no_trade_rows": rows, "trade_pct": 0.0}
    y = df["target"].astype(int)
    trade_rows = int((y == 1).sum())
    summary = {
        "rows": rows,
        "trade_rows": trade_rows,
        "no_trade_rows": int(rows - trade_rows),
        "trade_pct": float(trade_rows / rows * 100.0) if rows else 0.0,
        "target_counts": _target_counts(y),
        "direction_counts": _series_counts(df, "label_direction"),
        "trend_counts": _series_counts(df, "label_trend_bias"),
        "regime_counts": _series_counts(df, "label_regime"),
        "outcome_counts": _series_counts(df, "label_outcome"),
        "reject_reason_counts": _series_counts(df, "label_reject_reason"),
    }
    return summary


def _confusion_matrix(y_true, y_pred):
    true_series = pd.Series(y_true).astype(int).reset_index(drop=True)
    pred_series = pd.Series(y_pred).astype(int).reset_index(drop=True)
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(true_series, pred_series):
        if actual in {0, 1} and predicted in {0, 1}:
            matrix[int(actual)][int(predicted)] += 1
    return matrix


def _safe_classification_report(y_true, y_pred):
    matrix = _confusion_matrix(y_true, y_pred)
    total = sum(sum(row) for row in matrix)
    report = {}
    for label, name in ((0, "no_trade"), (1, "trade")):
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = matrix[label][1 - label]
        support = sum(matrix[label])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1),
            "support": int(support),
        }
    accuracy = (matrix[0][0] + matrix[1][1]) / total if total else 0.0
    report["accuracy"] = float(accuracy)
    report["macro avg"] = {
        "precision": float((report["no_trade"]["precision"] + report["trade"]["precision"]) / 2.0),
        "recall": float((report["no_trade"]["recall"] + report["trade"]["recall"]) / 2.0),
        "f1-score": float((report["no_trade"]["f1-score"] + report["trade"]["f1-score"]) / 2.0),
        "support": int(total),
    }
    return report


def _binary_metrics(y_true, y_pred):
    matrix = _confusion_matrix(y_true, y_pred)
    tp = matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "trade_precision": float(precision),
        "trade_recall": float(recall),
        "trade_f1": float(f1),
    }


def walk_forward_diagnostic_threshold():
    try:
        threshold = float(getattr(config, "MODEL_WALK_FORWARD_DIAGNOSTIC_THRESHOLD", 0.5))
    except (TypeError, ValueError):
        threshold = 0.5
    if not math.isfinite(threshold):
        return 0.5
    return max(0.0, min(1.0, threshold))


def walk_forward_fail_fast_enabled():
    return bool(getattr(config, "MODEL_WALK_FORWARD_FAIL_FAST", True))


def walk_forward_estimator_config():
    if not bool(getattr(config, "MODEL_WALK_FORWARD_LIGHTWEIGHT_TRAINING", True)):
        return None
    return {
        "lgb_n_estimators": max(1, int(getattr(config, "MODEL_WALK_FORWARD_LGB_ESTIMATORS", 40))),
        "xgb_n_estimators": max(1, int(getattr(config, "MODEL_WALK_FORWARD_XGB_ESTIMATORS", 40))),
        "rf_n_estimators": max(1, int(getattr(config, "MODEL_WALK_FORWARD_RF_ESTIMATORS", 30))),
    }


def write_walk_forward_stage_timing(log_file, fold_number, stage, elapsed_sec):
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(
            "walk_forward_stage_timing "
            f"fold={fold_number} stage={stage} elapsed_sec={float(elapsed_sec):.2f}\n"
        )


def _probability_scale_diagnostics(trade_prob, y_true, y_pred, threshold):
    series = pd.to_numeric(pd.Series(trade_prob), errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)
    rows = int(len(series))
    if rows == 0:
        return {
            "rows": 0,
            "near_threshold_band": float(0.0),
            "above_threshold_pct": 0.0,
            "p90_below_threshold": True,
            "p95_below_threshold": True,
            "true_trade_p90_below_threshold": True,
            "active_trade_pct": 0.0,
            "collapse_warning": False,
        }

    threshold = max(0.0, min(1.0, float(threshold)))
    y_true_series = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).astype(int).reset_index(drop=True)
    band = min(0.10, max(0.02, threshold * 0.20))
    p90 = float(series.quantile(0.90))
    p95 = float(series.quantile(0.95))
    true_trade = series.loc[y_true_series == 1]
    true_trade_p90 = float(true_trade.quantile(0.90)) if not true_trade.empty else 0.0
    above_threshold_count = int((series >= threshold).sum())
    active_trade_count = int((y_pred_series == 1).sum())
    true_trade_rows = int((y_true_series == 1).sum())
    collapse_warning = bool(
        true_trade_rows > 0
        and active_trade_count <= max(1, int(rows * 0.02))
        and p95 < threshold
    )
    return {
        "rows": rows,
        "near_threshold_band": float(band),
        "above_threshold_count": above_threshold_count,
        "above_threshold_pct": float(above_threshold_count / rows * 100.0),
        "active_trade_count": active_trade_count,
        "active_trade_pct": float(active_trade_count / rows * 100.0),
        "p90": p90,
        "p95": p95,
        "threshold": float(threshold),
        "p90_below_threshold": bool(p90 < threshold),
        "p95_below_threshold": bool(p95 < threshold),
        "true_trade_rows": true_trade_rows,
        "true_trade_p90": float(true_trade_p90),
        "true_trade_p90_below_threshold": bool(true_trade_p90 < threshold),
        "rows_near_threshold": int(((series >= threshold - band) & (series < threshold)).sum()),
        "collapse_warning": collapse_warning,
    }


def _model_group_diagnostics(validation_df, model_probability_frames, decision_threshold):
    if not model_probability_frames:
        return {}

    diagnostics = {}
    for name, proba in model_probability_frames.items():
        frame = pd.DataFrame(proba).reindex(validation_df.index)
        if "long_prob" not in frame:
            frame["long_prob"] = 0.0
        if "short_prob" not in frame:
            frame["short_prob"] = 0.0
        trade_prob = frame[["long_prob", "short_prob"]].astype(float).max(axis=1)
        y_pred = (trade_prob >= float(decision_threshold)).astype(int)
        working = validation_df.copy()
        working["_pred_target"] = y_pred.to_numpy()
        working["_trade_prob"] = trade_prob.reindex(validation_df.index).fillna(0.0).to_numpy()

        def summarize(group_cols):
            groups = {}
            for key, group in working.groupby(group_cols, dropna=False, sort=True):
                if not isinstance(key, tuple):
                    key = (key,)
                label = ":".join(str(part) for part in key)
                group_true = group["target"].astype(int)
                group_pred = group["_pred_target"].astype(int)
                groups[label] = {
                    "rows": int(len(group)),
                    "target_counts": _target_counts(group_true),
                    "prediction_counts": _target_counts(group_pred),
                    "confusion_matrix": _confusion_matrix(group_true, group_pred),
                    "trade_prob_quantiles": _quantiles(group["_trade_prob"]),
                    **_binary_metrics(group_true, group_pred),
                }
            return groups

        item = {
            "by_direction": summarize(["label_direction"]) if "label_direction" in working else {},
            "by_regime": summarize(["label_regime"]) if "label_regime" in working else {},
        }
        if "label_direction" in working and "label_regime" in working:
            item["by_direction_regime"] = summarize(["label_direction", "label_regime"])
        diagnostics[str(name)] = item
    return _json_safe(diagnostics)


def build_walk_forward_fold_diagnostics(
    fold,
    train_df,
    validation_df,
    feature_cols,
    fold_models,
    model_weights,
    metadata,
    *,
    decision_threshold=None,
    precomputed_probabilities=None,
    include_model_diagnostics=True,
    direction_model_weights=None,
):
    from core import signal_engine
    from core.trend_filter import derive_trend_context

    decision_threshold = (
        walk_forward_diagnostic_threshold()
        if decision_threshold is None
        else max(0.0, min(1.0, float(decision_threshold)))
    )
    X_validation = validation_df[feature_cols].astype(float)
    y_true = validation_df["target"].astype(int)
    model_predictions = {}
    model_probability_frames = {}
    proba_sum = None
    used_weight_total = 0.0
    trend_biases = []
    precomputed_probabilities = (
        precomputed_probabilities.reindex(validation_df.index)
        if precomputed_probabilities is not None
        else None
    )

    for _, row in validation_df.iterrows():
        trend_context = derive_trend_context(
            row,
            interval=config.TREND_FILTER_INTERVAL,
            fast_col=config.TREND_FILTER_FAST_COL,
            slow_col=config.TREND_FILTER_SLOW_COL,
            min_gap=config.TREND_FILTER_MIN_GAP,
        )
        trend_biases.append(str(trend_context.get("trend_bias") or "neutral"))

    for name, model in fold_models.items():
        if include_model_diagnostics:
            raw_pred = pd.Series(model.predict(X_validation), index=X_validation.index).astype(int)
            model_predictions[name] = {
                "prediction_counts": _target_counts(raw_pred),
                "classification_report": _json_safe(_safe_classification_report(y_true, raw_pred)),
                "confusion_matrix": _confusion_matrix(y_true, raw_pred),
                **_binary_metrics(y_true, raw_pred),
            }

        try:
            weight = float(model_weights.get(name, 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0:
            continue
        if precomputed_probabilities is not None:
            continue
        weighted_rows = []
        for row_idx in range(len(X_validation)):
            row_frame = X_validation.iloc[row_idx:row_idx + 1]
            directional = signal_engine.weighted_predict_proba(
                {name: model},
                row_frame,
                {name: 1.0},
                trend_bias=trend_biases[row_idx],
                model_metadata=metadata,
                direction_model_weights=direction_model_weights,
            )
            weighted_rows.append(directional)
        model_proba = pd.DataFrame(weighted_rows, columns=["short_prob", "long_prob"], index=X_validation.index)
        model_probability_frames[name] = model_proba
        if proba_sum is None:
            proba_sum = model_proba * weight
        else:
            proba_sum = proba_sum.add(model_proba * weight, fill_value=0.0)
        used_weight_total += weight

    if precomputed_probabilities is not None:
        ensemble_proba = pd.DataFrame(
            {
                "short_prob": pd.to_numeric(
                    precomputed_probabilities["short_prob"],
                    errors="coerce",
                ).fillna(0.0).astype(float),
                "long_prob": pd.to_numeric(
                    precomputed_probabilities["long_prob"],
                    errors="coerce",
                ).fillna(0.0).astype(float),
            },
            index=validation_df.index,
        )
        trade_prob = ensemble_proba.max(axis=1)
        y_pred = (trade_prob >= decision_threshold).astype(int)
        signal_direction_counts = _direction_counts_from_probabilities(
            ensemble_proba["long_prob"],
            ensemble_proba["short_prob"],
        )
        predicted_trade_direction_counts = _active_direction_counts(
            ensemble_proba["long_prob"],
            ensemble_proba["short_prob"],
            y_pred,
        )
    elif proba_sum is not None and used_weight_total > 0:
        ensemble_proba = proba_sum / used_weight_total
        trade_prob = ensemble_proba.max(axis=1)
        y_pred = (trade_prob >= decision_threshold).astype(int)
        signal_direction_counts = _direction_counts_from_probabilities(
            ensemble_proba["long_prob"],
            ensemble_proba["short_prob"],
        )
        predicted_trade_direction_counts = _active_direction_counts(
            ensemble_proba["long_prob"],
            ensemble_proba["short_prob"],
            y_pred,
        )
    else:
        y_pred = pd.Series(0, index=X_validation.index, dtype=int)
        trade_prob = pd.Series(0.0, index=X_validation.index, dtype=float)
        ensemble_proba = pd.DataFrame(
            {"short_prob": 0.0, "long_prob": 0.0},
            index=X_validation.index,
        )
        signal_direction_counts = {"long": 0, "short": 0, "flat": int(len(X_validation)), "long_pct": 0.0, "short_pct": 0.0, "flat_pct": 100.0}
        predicted_trade_direction_counts = dict(signal_direction_counts)

    validation_with_pred = validation_df.copy()
    validation_with_pred["_pred_target"] = y_pred.to_numpy()
    validation_with_pred["_trade_prob"] = trade_prob.reindex(validation_df.index).fillna(0.0).to_numpy()
    by_regime = {}
    if "label_regime" in validation_with_pred:
        for regime, group in validation_with_pred.groupby(validation_with_pred["label_regime"].fillna("unknown").astype(str), sort=True):
            group_true = group["target"].astype(int)
            group_pred = group["_pred_target"].astype(int)
            by_regime[str(regime)] = {
                "rows": int(len(group)),
                "target_counts": _target_counts(group_true),
                "prediction_counts": _target_counts(group_pred),
                "confusion_matrix": _confusion_matrix(group_true, group_pred),
                "trade_prob_quantiles": _quantiles(group["_trade_prob"]),
                **_binary_metrics(group_true, group_pred),
            }

    actual_trade = validation_with_pred["target"].astype(int) == 1
    predicted_trade = validation_with_pred["_pred_target"].astype(int) == 1
    trade_prob_quantiles = {
        "all": _quantiles(validation_with_pred["_trade_prob"]),
        "true_trade": _quantiles(validation_with_pred.loc[actual_trade, "_trade_prob"]),
        "true_no_trade": _quantiles(validation_with_pred.loc[~actual_trade, "_trade_prob"]),
        "true_positive": _quantiles(validation_with_pred.loc[actual_trade & predicted_trade, "_trade_prob"]),
        "false_negative": _quantiles(validation_with_pred.loc[actual_trade & ~predicted_trade, "_trade_prob"]),
        "false_positive": _quantiles(validation_with_pred.loc[~actual_trade & predicted_trade, "_trade_prob"]),
        "true_negative": _quantiles(validation_with_pred.loc[~actual_trade & ~predicted_trade, "_trade_prob"]),
    }
    error_slices = {
        "false_negative": _prediction_slice_summary(validation_with_pred, actual_trade & ~predicted_trade),
        "false_positive": _prediction_slice_summary(validation_with_pred, ~actual_trade & predicted_trade),
    }

    diagnostics = {
        "fold": int(fold["fold"]),
        "train": {
            "rows": int(len(train_df)),
            "target_counts": _target_counts(train_df["target"].astype(int)),
            "label_quality_summary": _label_quality_summary(train_df),
        },
        "validation": {
            "rows": int(len(validation_df)),
            "target_counts": _target_counts(y_true),
            "label_quality_summary": _label_quality_summary(validation_df),
            "trend_counts": {str(k): int(v) for k, v in pd.Series(trend_biases).value_counts().sort_index().items()},
            "regime_counts": (
                {str(k): int(v) for k, v in validation_df["label_regime"].fillna("unknown").astype(str).value_counts().sort_index().items()}
                if "label_regime" in validation_df else {}
            ),
            "direction_counts": (
                {str(k): int(v) for k, v in validation_df["label_direction"].fillna("unknown").astype(str).value_counts().sort_index().items()}
                if "label_direction" in validation_df else {}
            ),
            "outcome_counts": (
                {str(k): int(v) for k, v in validation_df["label_outcome"].fillna("unknown").astype(str).value_counts().sort_index().items()}
                if "label_outcome" in validation_df else {}
            ),
            "reject_reason_counts": (
                {str(k): int(v) for k, v in validation_df["label_reject_reason"].fillna("unknown").astype(str).value_counts().sort_index().items()}
                if "label_reject_reason" in validation_df else {}
            ),
        },
        "ensemble": {
            "decision_threshold": float(decision_threshold),
            "prediction_counts": _target_counts(pd.Series(y_pred)),
            "confusion_matrix": _confusion_matrix(y_true, y_pred),
            "classification_report": _json_safe(_safe_classification_report(y_true, y_pred)),
            "signal_direction_counts": signal_direction_counts,
            "predicted_trade_direction_counts": predicted_trade_direction_counts,
            "probability_scale_diagnostics": _probability_scale_diagnostics(
                trade_prob,
                y_true,
                y_pred,
                decision_threshold,
            ),
            "trade_prob_quantiles": trade_prob_quantiles,
            "error_slices": error_slices,
            **_binary_metrics(y_true, y_pred),
        },
        "by_regime": by_regime,
        "models": model_predictions,
        "model_group_diagnostics": _model_group_diagnostics(
            validation_df,
            model_probability_frames,
            decision_threshold,
        ),
    }
    return _json_safe(diagnostics)


def write_walk_forward_fold_diagnostics(log_file, diagnostics):
    ensemble = diagnostics.get("ensemble") or {}
    validation = diagnostics.get("validation") or {}
    scale_diagnostics = ensemble.get("probability_scale_diagnostics") or {}
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(
            "fold_label_diagnostics "
            f"fold={diagnostics.get('fold')} "
            f"train_targets={diagnostics.get('train', {}).get('target_counts', {})} "
            f"validation_targets={validation.get('target_counts', {})} "
            f"validation_regimes={validation.get('regime_counts', {})} "
            f"validation_rejects={validation.get('reject_reason_counts', {})}\n"
        )
        file.write(
            "fold_prediction_diagnostics "
            f"fold={diagnostics.get('fold')} "
            f"decision_threshold={float(ensemble.get('decision_threshold', 0.0)):.4f} "
            f"predictions={ensemble.get('prediction_counts', {})} "
            f"trade_precision={float(ensemble.get('trade_precision', 0.0)):.4f} "
            f"trade_recall={float(ensemble.get('trade_recall', 0.0)):.4f} "
            f"trade_f1={float(ensemble.get('trade_f1', 0.0)):.4f} "
            f"signal_directions={ensemble.get('signal_direction_counts', {})} "
            f"active_directions={ensemble.get('predicted_trade_direction_counts', {})} "
            f"prob_scale={scale_diagnostics} "
            f"trade_prob_q={ensemble.get('trade_prob_quantiles', {})} "
            f"errors={ensemble.get('error_slices', {})} "
            f"confusion_matrix={ensemble.get('confusion_matrix', [])}\n"
        )
        file.write("fold_diagnostics_json " + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True) + "\n")


def parse_float_candidates(raw_value, defaults):
    if raw_value is None or str(raw_value).strip() == "":
        values = list(defaults)
    else:
        values = []
        for item in str(raw_value).split(","):
            item = item.strip()
            if not item:
                continue
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
    return sorted(set(float(value) for value in values if math.isfinite(float(value))))


def _threshold_sweep_enabled():
    return bool(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_ENABLED", True))


def _threshold_sweep_candidate_limit():
    try:
        return max(1, int(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_MAX_CANDIDATES", 48)))
    except (TypeError, ValueError):
        return 48


def _threshold_sweep_top_n():
    try:
        return max(1, int(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_TOP_N", 5)))
    except (TypeError, ValueError):
        return 5


def _threshold_sweep_early_stop_enabled():
    return bool(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_EARLY_STOP_ENABLED", True))


def _threshold_sweep_early_stop_patience():
    try:
        return max(0, int(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_EARLY_STOP_PATIENCE", 16)))
    except (TypeError, ValueError):
        return 16


def _threshold_sweep_early_stop_min_closed_trades():
    try:
        configured = int(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_EARLY_STOP_MIN_CLOSED_TRADES", 1))
    except (TypeError, ValueError):
        configured = 1
    return max(HARD_MIN_CLOSED_TRADES, configured)


def _threshold_sweep_early_stop_min_profit_factor():
    try:
        configured = float(getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_EARLY_STOP_MIN_PROFIT_FACTOR", 1.05))
    except (TypeError, ValueError):
        configured = 1.05
    if not math.isfinite(configured):
        configured = 1.05
    return max(HARD_MIN_PROFIT_FACTOR, configured)


def _current_threshold_sweep_overrides():
    min_target_ratio = float(getattr(config, "MIN_SIGNAL_TARGET_RATIO", 0.0))
    return {
        "THRESHOLD_LONG": float(config.THRESHOLD_LONG),
        "THRESHOLD_SHORT": float(config.THRESHOLD_SHORT),
        "SIGNAL_MIN_PROB_DIFF": float(config.SIGNAL_MIN_PROB_DIFF),
        "MIN_SIGNAL_TARGET_RATIO": min_target_ratio,
        "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": float(
            getattr(config, "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO", min_target_ratio)
        ),
        "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": float(
            getattr(config, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO", min_target_ratio)
        ),
        "POSITION_PROBABILITY_CENTER": float(config.POSITION_PROBABILITY_CENTER),
        "BACKTEST_MIN_ADJUST_AMOUNT": float(config.BACKTEST_MIN_ADJUST_AMOUNT),
    }


def threshold_sweep_candidate_key(overrides):
    return tuple(
        round(float(overrides[key]), 8)
        for key in (
            "THRESHOLD_LONG",
            "THRESHOLD_SHORT",
            "SIGNAL_MIN_PROB_DIFF",
            "MIN_SIGNAL_TARGET_RATIO",
            "POSITION_PROBABILITY_CENTER",
            "BACKTEST_MIN_ADJUST_AMOUNT",
        )
    )


def downsample_threshold_sweep_candidates(candidates, limit):
    candidates = list(candidates or [])
    limit = max(1, int(limit))
    if len(candidates) <= limit:
        return candidates

    selected = {0}
    if limit >= 2 and len(candidates) > 1:
        selected.add(1)

    remaining_slots = limit - len(selected)
    remaining_indices = list(range(2, len(candidates)))
    if remaining_slots > 0 and remaining_indices:
        if remaining_slots >= len(remaining_indices):
            selected.update(remaining_indices)
        else:
            denominator = max(1, remaining_slots - 1)
            max_offset = len(remaining_indices) - 1
            for slot in range(remaining_slots):
                offset = round(slot * max_offset / denominator)
                selected.add(remaining_indices[offset])
            # Rounding can collide on tiny grids; fill deterministically from both ends.
            left = 0
            right = len(remaining_indices) - 1
            take_left = True
            while len(selected) < limit and left <= right:
                selected.add(remaining_indices[left if take_left else right])
                if take_left:
                    left += 1
                else:
                    right -= 1
                take_left = not take_left

    return [candidates[index] for index in sorted(selected)[:limit]]


def build_walk_forward_threshold_candidates():
    current = _current_threshold_sweep_overrides()
    thresholds = parse_float_candidates(
        getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_THRESHOLDS", ""),
        [0.12, 0.20, 0.30, 0.40, 0.50],
    )
    thresholds.extend([current["THRESHOLD_LONG"], current["THRESHOLD_SHORT"]])
    thresholds = sorted(set(max(0.0, min(1.0, float(value))) for value in thresholds))
    gaps = parse_float_candidates(
        getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_GAPS", ""),
        [0.0, 0.08, 0.12],
    )
    gaps.append(current["SIGNAL_MIN_PROB_DIFF"])
    gaps = sorted(set(max(0.0, min(1.0, float(value))) for value in gaps))
    min_target_ratios = parse_float_candidates(
        getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_MIN_TARGET_RATIOS", ""),
        [0.005, 0.010],
    )
    min_target_ratios.append(current["MIN_SIGNAL_TARGET_RATIO"])
    min_target_ratios = sorted(set(max(0.0, float(value)) for value in min_target_ratios))
    position_centers = parse_float_candidates(
        getattr(config, "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_POSITION_CENTERS", ""),
        [0.05, 0.10, 0.20, 0.30],
    )
    position_centers.append(current["POSITION_PROBABILITY_CENTER"])
    position_centers = sorted(set(max(0.0, min(0.99, float(value))) for value in position_centers))

    candidates = []
    seen = set()

    def add_candidate(name, overrides):
        key = threshold_sweep_candidate_key(overrides)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"name": name, "overrides": dict(overrides)})

    add_candidate("current", current)
    for threshold in thresholds:
        for gap in gaps:
            for min_target_ratio in min_target_ratios:
                for position_center in position_centers:
                    backtest_min_adjust = min(
                        float(config.MIN_ADJUST_AMOUNT),
                        float(config.INITIAL_BALANCE) * float(min_target_ratio),
                    )
                    overrides = {
                        "THRESHOLD_LONG": float(threshold),
                        "THRESHOLD_SHORT": float(threshold),
                        "SIGNAL_MIN_PROB_DIFF": float(gap),
                        "MIN_SIGNAL_TARGET_RATIO": float(min_target_ratio),
                        "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": float(min_target_ratio),
                        "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": float(min_target_ratio),
                        "POSITION_PROBABILITY_CENTER": float(position_center),
                        "BACKTEST_MIN_ADJUST_AMOUNT": float(backtest_min_adjust),
                    }
                    add_candidate(
                        (
                            f"tl{threshold:.2f}_ts{threshold:.2f}_gap{gap:.2f}_"
                            f"mt{min_target_ratio:.3f}_pc{position_center:.2f}"
                        ),
                        overrides,
                    )
    return downsample_threshold_sweep_candidates(candidates, _threshold_sweep_candidate_limit())


def apply_config_overrides(overrides):
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    return originals


def restore_config_overrides(originals):
    for key, value in originals.items():
        setattr(config, key, value)


def compact_walk_forward_candidate_summary(summary):
    keys = [
        "final_equity",
        "return_pct",
        "max_drawdown_pct",
        "trade_count",
        "closed_trade_count",
        "winning_trade_count",
        "losing_trade_count",
        "win_rate_pct",
        "profit_factor",
        "avg_win_loss_ratio",
        "avg_closed_trade_pnl",
        "net_pnl_after_costs",
        "fees_paid",
        "slippage_cost",
        "funding_pnl",
        "take_profit_count",
        "stop_loss_count",
        "decision_action_counts",
        "decision_reason_top",
        "decision_direction_counts",
        "decision_regime_counts",
        "decision_probability_quantiles",
        "decision_edge_gate_summary",
        "decision_gate_config",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def summarize_threshold_sweep_candidate(summary):
    if not summary:
        return {
            "name": None,
            "closed": 0,
            "net": 0.0,
            "pf": 0.0,
            "top_reason": "-",
            "actions": {},
        }

    reason_top = summary.get("decision_reason_top") or []
    top_reason = "-"
    if reason_top:
        first = reason_top[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            top_reason = f"{first[0]}:{first[1]}"
        else:
            top_reason = str(first)

    return {
        "name": summary.get("name"),
        "closed": int(summary.get("closed_trade_count") or 0),
        "net": float(summary.get("net_pnl_after_costs") or 0.0),
        "pf": float(summary.get("profit_factor") or 0.0),
        "top_reason": top_reason,
        "actions": summary.get("decision_action_counts") or {},
        "edge_gate": summary.get("decision_edge_gate_summary") or {},
    }


def threshold_sweep_override_diff(current, best):
    current_overrides = (current or {}).get("overrides") or {}
    best_overrides = (best or {}).get("overrides") or {}
    keys = [
        "THRESHOLD_LONG",
        "THRESHOLD_SHORT",
        "SIGNAL_MIN_PROB_DIFF",
        "MIN_SIGNAL_TARGET_RATIO",
        "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO",
        "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO",
        "POSITION_PROBABILITY_CENTER",
        "BACKTEST_MIN_ADJUST_AMOUNT",
    ]
    diff = {}
    for key in keys:
        if key not in current_overrides and key not in best_overrides:
            continue
        current_value = current_overrides.get(key)
        best_value = best_overrides.get(key)
        if current_value != best_value:
            diff[key] = [current_value, best_value]
    return diff


def threshold_sweep_score(summary):
    closed = int(summary.get("closed_trade_count") or 0)
    net = float(summary.get("net_pnl_after_costs") or 0.0)
    pf = float(summary.get("profit_factor") or 0.0)
    drawdown = abs(float(summary.get("max_drawdown_pct") or 0.0))
    enough_trades = 1 if closed >= HARD_MIN_CLOSED_TRADES else 0
    positive_pf = 1 if pf > HARD_MIN_PROFIT_FACTOR else 0
    return (enough_trades, positive_pf, net, pf, -drawdown, closed)


def threshold_sweep_candidate_is_good_enough(summary):
    closed = int(summary.get("closed_trade_count") or 0)
    net = float(summary.get("net_pnl_after_costs") or 0.0)
    pf = float(summary.get("profit_factor") or 0.0)
    return (
        closed >= _threshold_sweep_early_stop_min_closed_trades()
        and net > 0.0
        and pf >= _threshold_sweep_early_stop_min_profit_factor()
    )


def run_backtest_with_overrides(backtester_kwargs, overrides):
    from backtest.backtest import Backtester

    backtester_kwargs = dict(backtester_kwargs)
    use_precomputed_probabilities = bool(backtester_kwargs.pop("use_precomputed_probabilities", False))
    originals = apply_config_overrides(overrides)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            backtester = Backtester(**backtester_kwargs)
            original_predict_row = backtester._predict_row
            if use_precomputed_probabilities:
                backtester._predict_row = lambda row: (row["long_prob"], row["short_prob"])
            try:
                summary = backtester.run_backtest()
            finally:
                backtester._predict_row = original_predict_row
        return compact_walk_forward_candidate_summary(summary or {})
    finally:
        restore_config_overrides(originals)


def run_walk_forward_threshold_sweep(backtester_kwargs, candidates):
    if not candidates:
        return {"enabled": False, "reason": "no_candidates", "candidate_count": 0}

    results = []
    best = None
    best_score = None
    current = None
    no_improvement_count = 0
    stopped_early = False
    early_stop_reason = None
    early_stop_enabled = _threshold_sweep_early_stop_enabled()
    early_stop_patience = _threshold_sweep_early_stop_patience()

    for candidate in candidates:
        summary = run_backtest_with_overrides(backtester_kwargs, candidate["overrides"])
        summary["name"] = candidate["name"]
        summary["overrides"] = candidate["overrides"]
        results.append(summary)
        if candidate.get("name") == "current":
            current = summary

        score = threshold_sweep_score(summary)
        if best_score is None or score > best_score:
            best = summary
            best_score = score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if (
            early_stop_enabled
            and early_stop_patience > 0
            and best is not None
            and threshold_sweep_candidate_is_good_enough(best)
            and no_improvement_count >= early_stop_patience
        ):
            stopped_early = True
            early_stop_reason = (
                f"no_improvement_patience_{early_stop_patience}_after_good_candidate"
            )
            break

    ranked = sorted(results, key=threshold_sweep_score, reverse=True)
    top_n = _threshold_sweep_top_n()
    best = ranked[0] if ranked else None
    return {
        "enabled": True,
        "candidate_count": int(len(candidates)),
        "evaluated_count": int(len(results)),
        "stopped_early": bool(stopped_early),
        "early_stop_reason": early_stop_reason,
        "early_stop_config": {
            "enabled": bool(early_stop_enabled),
            "patience": int(early_stop_patience),
            "min_closed_trades": int(_threshold_sweep_early_stop_min_closed_trades()),
            "min_profit_factor": float(_threshold_sweep_early_stop_min_profit_factor()),
        },
        "top_n": int(top_n),
        "current": current,
        "best": best,
        "recommended": ranked[:top_n],
    }


def write_walk_forward_threshold_sweep(log_file, fold_number, sweep_summary):
    if not sweep_summary or not sweep_summary.get("enabled"):
        return
    best = sweep_summary.get("best") or {}
    current = sweep_summary.get("current") or {}
    current_summary = summarize_threshold_sweep_candidate(current)
    best_summary = summarize_threshold_sweep_candidate(best)
    override_diff = threshold_sweep_override_diff(current, best)
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(
            "walk_forward_threshold_sweep "
            f"fold={fold_number} "
            f"candidates={sweep_summary.get('candidate_count', 0)} "
            f"evaluated={sweep_summary.get('evaluated_count', 0)} "
            f"stopped_early={int(bool(sweep_summary.get('stopped_early')))} "
            f"best={best.get('name')} "
            f"closed={best.get('closed_trade_count', 0)} "
            f"net={float(best.get('net_pnl_after_costs') or 0.0):.2f} "
            f"pf={float(best.get('profit_factor') or 0.0):.4f} "
            f"overrides={best.get('overrides', {})}\n"
        )
        file.write(
            "walk_forward_threshold_sweep_compare "
            f"fold={fold_number} "
            f"current_closed={current_summary['closed']} "
            f"current_net={current_summary['net']:.2f} "
            f"current_pf={current_summary['pf']:.4f} "
            f"current_top_reason={current_summary['top_reason']} "
            f"current_actions={current_summary['actions']} "
            f"best={best_summary['name']} "
            f"best_closed={best_summary['closed']} "
            f"best_net={best_summary['net']:.2f} "
            f"best_pf={best_summary['pf']:.4f} "
            f"best_top_reason={best_summary['top_reason']} "
            f"best_actions={best_summary['actions']} "
            f"override_diff={override_diff}\n"
        )
        file.write(
            "walk_forward_threshold_sweep_json "
            + json.dumps(sweep_summary, ensure_ascii=False, sort_keys=True)
            + "\n"
        )


def add_walk_forward_probabilities(data, feature_cols, fold_models, model_weights, metadata, direction_model_weights=None):
    from core import signal_engine
    from core.trend_filter import derive_trend_context

    predicted = data.copy()
    if predicted.empty:
        predicted["long_prob"] = 0.0
        predicted["short_prob"] = 0.0
        return predicted

    X = predicted[feature_cols].astype(float)
    trend_biases = []
    for _, row in predicted.iterrows():
        trend_context = derive_trend_context(
            row,
            interval=config.TREND_FILTER_INTERVAL,
            fast_col=config.TREND_FILTER_FAST_COL,
            slow_col=config.TREND_FILTER_SLOW_COL,
            min_gap=config.TREND_FILTER_MIN_GAP,
        )
        trend_biases.append(str(trend_context.get("trend_bias") or "neutral"))

    avg_pred = signal_engine.weighted_predict_proba_batch(
        fold_models,
        X,
        model_weights,
        trend_biases=trend_biases,
        model_metadata=metadata,
        direction_model_weights=direction_model_weights,
    )
    probability_frame = pd.DataFrame(avg_pred, columns=["short_prob", "long_prob"], index=predicted.index)
    predicted[["long_prob", "short_prob"]] = probability_frame[["long_prob", "short_prob"]]
    return predicted


def aggregate_backtest_summaries(summaries):
    fold_count = len(summaries)
    closed_trade_count = sum(int(item.get("closed_trade_count", 0)) for item in summaries)
    winning_trade_count = sum(int(item.get("winning_trade_count", 0)) for item in summaries)
    losing_trade_count = sum(int(item.get("losing_trade_count", 0)) for item in summaries)
    gross_profit = sum(float(item.get("gross_profit", 0.0)) for item in summaries)
    gross_loss = sum(float(item.get("gross_loss", 0.0)) for item in summaries)
    net_pnl_after_costs = sum(float(item.get("net_pnl_after_costs", 0.0)) for item in summaries)
    fees_paid = sum(float(item.get("fees_paid", 0.0)) for item in summaries)
    slippage_cost = sum(float(item.get("slippage_cost", 0.0)) for item in summaries)
    funding_pnl = sum(float(item.get("funding_pnl", 0.0)) for item in summaries)
    trade_count = sum(int(item.get("trade_count", 0)) for item in summaries)
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )
    avg_win = gross_profit / winning_trade_count if winning_trade_count else 0.0
    avg_loss = gross_loss / losing_trade_count if losing_trade_count else 0.0
    avg_win_loss_ratio = (
        avg_win / avg_loss
        if avg_loss > 0
        else (float("inf") if avg_win > 0 else 0.0)
    )
    win_rate_pct = (
        winning_trade_count / closed_trade_count * 100.0
        if closed_trade_count
        else 0.0
    )
    avg_closed_trade_pnl = (
        net_pnl_after_costs / closed_trade_count
        if closed_trade_count
        else 0.0
    )
    capital_base = max(1, fold_count) * float(config.INITIAL_BALANCE)

    return {
        "fold_count": int(fold_count),
        "final_equity": float(config.INITIAL_BALANCE + net_pnl_after_costs),
        "pnl": float(net_pnl_after_costs),
        "return_pct": float(net_pnl_after_costs / capital_base * 100.0),
        "max_drawdown_pct": float(min(item.get("max_drawdown_pct", 0.0) for item in summaries)),
        "trade_count": int(trade_count),
        "closed_trade_count": int(closed_trade_count),
        "winning_trade_count": int(winning_trade_count),
        "losing_trade_count": int(losing_trade_count),
        "win_rate_pct": float(win_rate_pct),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_win_loss_ratio": float(avg_win_loss_ratio),
        "avg_closed_trade_pnl": float(avg_closed_trade_pnl),
        "net_pnl_after_costs": float(net_pnl_after_costs),
        "net_return_pct_after_costs": float(net_pnl_after_costs / capital_base * 100.0),
        "fees_paid": float(fees_paid),
        "slippage_cost": float(slippage_cost),
        "funding_pnl": float(funding_pnl),
        "decision_regime_signal_summary": aggregate_regime_signal_summaries(summaries),
        "decision_edge_gate_summary": aggregate_edge_gate_summaries(summaries),
        "folds": summaries,
    }


def validate_walk_forward_summary(summary):
    min_folds = max(1, int(config.MODEL_WALK_FORWARD_MIN_FOLDS))
    if int(summary.get("fold_count", 0)) < min_folds:
        raise RuntimeError(
            f"walk-forward 折数不足: fold_count={summary.get('fold_count', 0)} < {min_folds}"
        )

    for fold in summary.get("folds", []):
        if int(fold.get("closed_trade_count", 0)) < HARD_MIN_CLOSED_TRADES:
            raise RuntimeError(
                "walk-forward 单折平仓交易数不足: "
                f"fold={fold.get('fold')}, closed_trade_count={fold.get('closed_trade_count', 0)}"
            )
        if float(fold.get("profit_factor", 0.0)) <= HARD_MIN_PROFIT_FACTOR:
            raise RuntimeError(
                "walk-forward 单折盈利因子必须大于1: "
                f"fold={fold.get('fold')}, profit_factor={fold.get('profit_factor', 0.0):.4f}"
            )

    validate_backtest_summary(summary)


def walk_forward_fold_failure_reason(fold_summary):
    fold_id = fold_summary.get("fold")
    closed_trades = int(fold_summary.get("closed_trade_count", 0))
    if closed_trades < HARD_MIN_CLOSED_TRADES:
        return (
            "walk-forward 单折平仓交易数不足: "
            f"fold={fold_id}, closed_trade_count={closed_trades}"
        )
    profit_factor = float(fold_summary.get("profit_factor", 0.0))
    if profit_factor <= HARD_MIN_PROFIT_FACTOR:
        return (
            "walk-forward 单折盈利因子必须大于1: "
            f"fold={fold_id}, profit_factor={profit_factor:.4f}"
        )
    return None


def run_walk_forward_validation(log_file, context_backtester, metadata, feature_cols):
    if not bool(config.MODEL_WALK_FORWARD_ENABLED):
        return None
    if not metadata:
        raise RuntimeError("训练元数据缺失，无法执行 walk-forward 验证")

    from backtest.backtest import Backtester
    from train.train import create_labels, train_direction_quality_bundle

    append_log_header(log_file, "walk_forward_validation")
    labeled_data = create_labels(
        context_backtester.data.copy(),
        future_window=int(metadata.get("label_future_window", config.MODEL_LABEL_FUTURE_WINDOW)),
        threshold=float(metadata.get("label_threshold", config.MODEL_LABEL_THRESHOLD)),
    )
    missing_cols = [col for col in feature_cols if col not in labeled_data.columns]
    if missing_cols:
        raise RuntimeError(
            "walk-forward 验证缺少特征列: "
            f"{','.join(missing_cols[:10])}"
        )

    slices = build_walk_forward_slices(labeled_data.index, metadata)
    fold_summaries = []
    threshold_candidates = (
        build_walk_forward_threshold_candidates()
        if _threshold_sweep_enabled()
        else []
    )
    estimator_config = walk_forward_estimator_config()

    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"walk-forward folds={len(slices)}\n")
        file.write(
            "walk-forward estimator_config "
            f"lightweight={int(estimator_config is not None)} "
            f"config={estimator_config or {}}\n"
        )
        if _threshold_sweep_enabled():
            file.write(
                "walk-forward threshold_sweep "
                f"enabled=1 candidates={len(threshold_candidates)} "
                f"diagnostic_threshold={walk_forward_diagnostic_threshold():.4f}\n"
            )

    for fold in slices:
        fold_started_at = time.monotonic()
        train_df = labeled_data.iloc[fold["train_start_pos"]:fold["train_end_pos"]].copy()
        validation_df = labeled_data.iloc[
            fold["validation_start_pos"]:fold["validation_end_pos"]
        ].copy()
        if train_df.empty or validation_df.empty:
            raise RuntimeError(f"walk-forward fold={fold['fold']} 样本为空")

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df["target"]
        stage_started_at = time.monotonic()
        fold_models, _, _, _, _ = train_direction_quality_bundle(
            X_train,
            y_train,
            sample_context=train_df,
            estimator_config=estimator_config,
        )
        train_elapsed_sec = time.monotonic() - stage_started_at
        write_walk_forward_stage_timing(
            log_file,
            fold["fold"],
            "train_models",
            train_elapsed_sec,
        )
        fold_data = context_backtester.data.loc[
            (context_backtester.data.index >= validation_df.index.min()) &
            (context_backtester.data.index <= validation_df.index.max())
        ].copy()
        fold_funding_history = filter_funding_history(
            context_backtester.funding_history,
            fold_data.index.min(),
            fold_data.index.max(),
        )
        stage_started_at = time.monotonic()
        fold_predicted_data = add_walk_forward_probabilities(
            fold_data,
            feature_cols,
            fold_models,
            config.MODEL_WEIGHTS,
            metadata,
            direction_model_weights=getattr(config, "MODEL_DIRECTION_MODEL_WEIGHTS", {}),
        )
        probabilities_elapsed_sec = time.monotonic() - stage_started_at
        write_walk_forward_stage_timing(
            log_file,
            fold["fold"],
            "precompute_probabilities",
            probabilities_elapsed_sec,
        )

        stage_started_at = time.monotonic()
        fold_diagnostics = build_walk_forward_fold_diagnostics(
            fold,
            train_df,
            validation_df,
            feature_cols,
            fold_models,
            config.MODEL_WEIGHTS,
            metadata,
            precomputed_probabilities=fold_predicted_data[["long_prob", "short_prob"]],
            include_model_diagnostics=bool(
                getattr(config, "MODEL_WALK_FORWARD_MODEL_DIAGNOSTICS", False)
            ),
            direction_model_weights=getattr(config, "MODEL_DIRECTION_MODEL_WEIGHTS", {}),
        )
        diagnostics_elapsed_sec = time.monotonic() - stage_started_at
        write_walk_forward_stage_timing(
            log_file,
            fold["fold"],
            "diagnostics",
            diagnostics_elapsed_sec,
        )
        write_walk_forward_fold_diagnostics(log_file, fold_diagnostics)

        append_log_header(log_file, f"walk_forward_fold_{fold['fold']}")
        stage_started_at = time.monotonic()
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(
                "fold_range "
                f"fold={fold['fold']} train_rows={len(train_df)} validation_rows={len(fold_data)} "
                f"validation_start={validation_df.index.min().isoformat()} "
                f"validation_end={validation_df.index.max().isoformat()}\n"
            )
            with contextlib.redirect_stdout(file):
                backtester = Backtester(
                    context_backtester.interval,
                    context_backtester.window,
                    data_dict=context_backtester.data_dict,
                    reward_risk=context_backtester.reward_risk,
                    precomputed_data=fold_predicted_data,
                    feature_cols=feature_cols,
                    models=fold_models,
                    model_weights=config.MODEL_WEIGHTS,
                    model_metadata=metadata,
                    funding_history=fold_funding_history,
                    enable_csv_dump=False,
                    show_progress=False,
                    emit_diagnostics=False,
                )
                original_predict_row = backtester._predict_row
                backtester._predict_row = lambda row: (row["long_prob"], row["short_prob"])
                try:
                    fold_summary = backtester.run_backtest()
                finally:
                    backtester._predict_row = original_predict_row
        backtest_elapsed_sec = time.monotonic() - stage_started_at
        write_walk_forward_stage_timing(
            log_file,
            fold["fold"],
            "backtest_current",
            backtest_elapsed_sec,
        )

        if not fold_summary:
            raise RuntimeError(f"walk-forward fold={fold['fold']} 未返回 summary")
        threshold_sweep = None
        threshold_sweep_elapsed_sec = 0.0
        if _threshold_sweep_enabled() and threshold_candidates and fold_predicted_data is not None:
            stage_started_at = time.monotonic()
            threshold_sweep = run_walk_forward_threshold_sweep(
                {
                    "interval": context_backtester.interval,
                    "window": context_backtester.window,
                    "data_dict": context_backtester.data_dict,
                    "reward_risk": context_backtester.reward_risk,
                    "precomputed_data": fold_predicted_data,
                    "feature_cols": feature_cols,
                    "models": fold_models,
                    "model_weights": config.MODEL_WEIGHTS,
                    "model_metadata": metadata,
                    "funding_history": fold_funding_history,
                    "enable_csv_dump": False,
                    "show_progress": False,
                    "emit_diagnostics": False,
                    "use_precomputed_probabilities": True,
                },
                threshold_candidates,
            )
            threshold_sweep_elapsed_sec = time.monotonic() - stage_started_at
            write_walk_forward_threshold_sweep(log_file, fold["fold"], threshold_sweep)
            write_walk_forward_stage_timing(
                log_file,
                fold["fold"],
                "threshold_sweep",
                threshold_sweep_elapsed_sec,
            )
        fold_summary.update({
            "fold": fold["fold"],
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(fold_data)),
            "validation_start": validation_df.index.min().isoformat(),
            "validation_end": validation_df.index.max().isoformat(),
            "fold_diagnostics": fold_diagnostics,
            "elapsed_sec": float(time.monotonic() - fold_started_at),
            "stage_timing": {
                "train_models": float(train_elapsed_sec),
                "diagnostics": float(diagnostics_elapsed_sec),
                "precompute_probabilities": float(probabilities_elapsed_sec),
                "backtest_current": float(backtest_elapsed_sec),
                "threshold_sweep": float(threshold_sweep_elapsed_sec),
            },
        })
        if threshold_sweep is not None:
            fold_summary["threshold_sweep"] = threshold_sweep
        fold_summaries.append(fold_summary)
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(
                "walk_forward_fold_timing "
                f"fold={fold['fold']} "
                f"elapsed_sec={fold_summary['elapsed_sec']:.2f} "
                f"train_models_sec={train_elapsed_sec:.2f} "
                f"diagnostics_sec={diagnostics_elapsed_sec:.2f} "
                f"precompute_probabilities_sec={probabilities_elapsed_sec:.2f} "
                f"backtest_current_sec={backtest_elapsed_sec:.2f} "
                f"threshold_sweep_sec={threshold_sweep_elapsed_sec:.2f} "
                f"threshold_sweep_evaluated="
                f"{(threshold_sweep or {}).get('evaluated_count', 0)}\n"
            )

        failure_reason = walk_forward_fold_failure_reason(fold_summary)
        if failure_reason and walk_forward_fail_fast_enabled():
            partial_summary = aggregate_backtest_summaries(fold_summaries)
            partial_summary["failed"] = True
            partial_summary["failure_reason"] = failure_reason
            partial_summary["failed_fold"] = fold["fold"]
            update_candidate_training_metadata(
                walk_forward_failure_summary=partial_summary,
                candidate_status="walk_forward_failed",
            )
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(
                    "walk_forward_fail_fast "
                    f"fold={fold['fold']} reason={failure_reason}\n"
                )
            raise RuntimeError(failure_reason)

    summary = aggregate_backtest_summaries(fold_summaries)
    validate_walk_forward_summary(summary)
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(
            "walk-forward summary "
            f"folds={summary['fold_count']} net_pnl_after_costs={summary['net_pnl_after_costs']:.2f} "
            f"profit_factor={summary['profit_factor']:.4f} "
            f"max_drawdown_pct={summary['max_drawdown_pct']:.2f}\n"
        )
    return summary


def run_backtest_with_bundle(log_file, title, context_backtester, bundle):
    from backtest.backtest import Backtester

    append_log_header(log_file, title)
    with open(log_file, "a", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file):
            backtester = Backtester(
                context_backtester.interval,
                context_backtester.window,
                data_dict=context_backtester.data_dict,
                reward_risk=context_backtester.reward_risk,
                precomputed_data=context_backtester.data,
                feature_cols=bundle["feature_cols"],
                models=bundle["models"],
                model_weights=config.MODEL_WEIGHTS,
                model_metadata=bundle.get("metadata") or {},
                funding_history=context_backtester.funding_history,
                enable_csv_dump=False,
                show_progress=False,
                emit_diagnostics=False,
            )
            return backtester.run_backtest()


def validate_regime_signal_summary(summary):
    if not bool(getattr(config, "MODEL_RETRAIN_REGIME_GATE_ENABLED", True)):
        return

    regime_summary = summary.get("decision_regime_signal_summary") or {}
    if not regime_summary:
        return

    min_rows = max(1, int(getattr(config, "MODEL_RETRAIN_REGIME_GATE_MIN_ROWS", 30)))
    max_trend_short_long_pct = float(getattr(config, "MODEL_RETRAIN_MAX_TREND_SHORT_LONG_DOMINANCE_PCT", 80.0))
    max_trend_long_short_pct = float(getattr(config, "MODEL_RETRAIN_MAX_TREND_LONG_SHORT_DOMINANCE_PCT", 80.0))

    trend_short = regime_summary.get("trend_short") or {}
    if int(trend_short.get("rows", 0)) >= min_rows:
        long_pct = float(trend_short.get("dominant_long_pct", 0.0))
        if long_pct > max_trend_short_long_pct:
            raise RuntimeError(
                "Regime分层检查失败: trend_short 中候选模型过度偏多 "
                f"dominant_long_pct={long_pct:.2f} > {max_trend_short_long_pct:.2f}, "
                f"rows={trend_short.get('rows', 0)}"
            )

    trend_long = regime_summary.get("trend_long") or {}
    if int(trend_long.get("rows", 0)) >= min_rows:
        short_pct = float(trend_long.get("dominant_short_pct", 0.0))
        if short_pct > max_trend_long_short_pct:
            raise RuntimeError(
                "Regime分层检查失败: trend_long 中候选模型过度偏空 "
                f"dominant_short_pct={short_pct:.2f} > {max_trend_long_short_pct:.2f}, "
                f"rows={trend_long.get('rows', 0)}"
            )


def validate_backtest_summary(summary):
    min_return_pct = float(config.MODEL_RETRAIN_MIN_RETURN_PCT)
    max_drawdown_pct = float(config.MODEL_RETRAIN_MAX_DRAWDOWN_PCT)
    min_closed_trades = max(HARD_MIN_CLOSED_TRADES, int(config.MODEL_RETRAIN_MIN_CLOSED_TRADES))
    min_win_rate_pct = float(config.MODEL_RETRAIN_MIN_WIN_RATE_PCT)
    min_profit_factor = max(HARD_MIN_PROFIT_FACTOR, float(config.MODEL_RETRAIN_MIN_PROFIT_FACTOR))
    min_avg_win_loss_ratio = float(config.MODEL_RETRAIN_MIN_AVG_WIN_LOSS_RATIO)
    min_net_pnl_after_costs = float(config.MODEL_RETRAIN_MIN_NET_PNL_AFTER_COSTS)

    if float(summary.get("return_pct", 0.0)) < min_return_pct:
        raise RuntimeError(
            f"回测收益未达标: return_pct={summary.get('return_pct', 0.0):.2f} < {min_return_pct:.2f}"
        )
    if float(summary.get("max_drawdown_pct", 0.0)) < max_drawdown_pct:
        raise RuntimeError(
            f"回测回撤超限: max_drawdown_pct={summary.get('max_drawdown_pct', 0.0):.2f} < {max_drawdown_pct:.2f}"
        )
    if int(summary.get("closed_trade_count", 0)) < min_closed_trades:
        raise RuntimeError(
            f"平仓交易数不足: closed_trade_count={summary.get('closed_trade_count', 0)} < {min_closed_trades}"
        )
    if float(summary.get("win_rate_pct", 0.0)) < min_win_rate_pct:
        raise RuntimeError(
            f"回测胜率未达标: win_rate_pct={summary.get('win_rate_pct', 0.0):.2f} < {min_win_rate_pct:.2f}"
        )
    if float(summary.get("profit_factor", 0.0)) <= HARD_MIN_PROFIT_FACTOR:
        raise RuntimeError(
            "盈利因子必须大于1: "
            f"profit_factor={summary.get('profit_factor', 0.0):.4f} <= {HARD_MIN_PROFIT_FACTOR:.4f}"
        )
    if float(summary.get("profit_factor", 0.0)) < min_profit_factor:
        raise RuntimeError(
            f"盈利因子未达标: profit_factor={summary.get('profit_factor', 0.0):.4f} < {min_profit_factor:.4f}"
        )
    if float(summary.get("avg_win_loss_ratio", 0.0)) < min_avg_win_loss_ratio:
        raise RuntimeError(
            "平均盈亏比未达标: "
            f"avg_win_loss_ratio={summary.get('avg_win_loss_ratio', 0.0):.4f} < {min_avg_win_loss_ratio:.4f}"
        )
    if float(summary.get("net_pnl_after_costs", 0.0)) < min_net_pnl_after_costs:
        raise RuntimeError(
            "手续费后收益未达标: "
            f"net_pnl_after_costs={summary.get('net_pnl_after_costs', 0.0):.2f} < {min_net_pnl_after_costs:.2f}"
        )
    validate_regime_signal_summary(summary)


def validate_new_model_improvement(new_summary, old_summary):
    comparisons = {
        "net_pnl_after_costs": "手续费后收益",
        "profit_factor": "盈利因子",
        "max_drawdown_pct": "最大回撤",
    }
    comparison = {}

    for key, label in comparisons.items():
        new_value = float(new_summary.get(key, 0.0))
        old_value = float(old_summary.get(key, 0.0))
        delta = new_value - old_value
        comparison[key] = {
            "old": old_value,
            "new": new_value,
            "delta": delta,
        }
        if not value_is_strictly_better(new_value, old_value):
            raise RuntimeError(
                f"新模型{label}未优于旧模型: {key} new={new_value:.4f}, old={old_value:.4f}"
            )

    return comparison


def value_is_strictly_better(new_value, old_value):
    if math.isnan(new_value) or math.isnan(old_value):
        return False
    if math.isinf(new_value) or math.isinf(old_value):
        return new_value > old_value
    return (new_value - old_value) > IMPROVEMENT_EPSILON


def fmt_metric(value, digits=2):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def format_retrain_success_notification(backtest_summary, log_file):
    lines = ["[模型重训成功] 新模型已替换线上模型"]
    if backtest_summary:
        lines.extend([
            f"OOS手续费后收益: {fmt_metric(backtest_summary.get('net_pnl_after_costs'))} USDT",
            f"OOS收益率: {fmt_metric(backtest_summary.get('net_return_pct_after_costs'))}%",
            f"OOS最大回撤: {fmt_metric(backtest_summary.get('max_drawdown_pct'))}%",
            f"OOS胜率: {fmt_metric(backtest_summary.get('win_rate_pct'))}%",
            f"OOS盈利因子: {fmt_metric(backtest_summary.get('profit_factor'), 4)}",
            f"OOS平仓交易数: {backtest_summary.get('closed_trade_count', '-')}",
        ])
        comparison = backtest_summary.get("comparison") or {}
        if comparison:
            lines.extend([
                "新旧同场差值:",
                f"net_pnl_after_costs: {fmt_metric(comparison.get('net_pnl_after_costs', {}).get('delta'))}",
                f"profit_factor: {fmt_metric(comparison.get('profit_factor', {}).get('delta'), 4)}",
                f"max_drawdown_pct: {fmt_metric(comparison.get('max_drawdown_pct', {}).get('delta'))}",
            ])
        walk_forward_summary = backtest_summary.get("walk_forward_summary") or {}
        if walk_forward_summary:
            lines.append(
                "walk-forward: "
                f"folds={walk_forward_summary.get('fold_count', '-')}, "
                f"net={fmt_metric(walk_forward_summary.get('net_pnl_after_costs'))}, "
                f"PF={fmt_metric(walk_forward_summary.get('profit_factor'), 4)}"
            )
    lines.append(f"log: {log_file}")
    return "\n".join(lines)


def format_retrain_failure_notification(error, log_file):
    return (
        "[模型重训失败] 已回滚旧模型\n"
        f"原因: {error}\n"
        f"log: {log_file}"
    )


def write_state(**updates):
    state = read_json(STATE_PATH, {})
    state.update(updates)
    write_json_atomic(STATE_PATH, state)


def retrain_once(*, validate_backtest=None):
    os.makedirs(LOGS_DIR, exist_ok=True)
    run_id = timestamp_id()
    log_file = os.path.join(LOGS_DIR, f"model_retrain_{run_id}.log")
    validate_backtest = (
        bool(config.MODEL_RETRAIN_VALIDATE_BACKTEST)
        if validate_backtest is None
        else bool(validate_backtest)
    )

    acquire_lock()
    backup_dir = None
    manifest = []
    started_at = utc_now_iso()
    write_state(
        last_attempt_at=started_at,
        last_status="running",
        last_log_path=log_file,
    )

    try:
        backup_dir, manifest = make_backup(run_id)
        train_returncode = run_subprocess([sys.executable, "-m", "train.train"], log_file)
        if train_returncode != 0:
            raise RuntimeError(f"训练命令失败: exit_code={train_returncode}")

        loaded_artifacts = validate_artifacts()
        backtest_summary = None
        if validate_backtest:
            backtest_summary = run_backtest_validation(log_file, backup_dir=backup_dir)

        prune_backups(config.MODEL_RETRAIN_KEEP_BACKUPS)
        finished_at = utc_now_iso()
        write_state(
            last_success_at=finished_at,
            last_finished_at=finished_at,
            last_status="success",
            last_log_path=log_file,
            last_backup_path=backup_dir,
            loaded_artifacts=loaded_artifacts,
            backtest_summary=backtest_summary,
        )
        print(f"模型重训成功: log={log_file}")
        if backtest_summary:
            print(
                "回测验证通过: "
                f"return={backtest_summary['return_pct']:.2f}%, "
                f"maxDD={backtest_summary['max_drawdown_pct']:.2f}%"
            )
        notify_important(format_retrain_success_notification(backtest_summary, log_file))
        return 0
    except Exception as exc:
        candidate_metadata_path = None
        if backup_dir and manifest:
            candidate_metadata_path = preserve_candidate_training_metadata(backup_dir)
            restore_backup(backup_dir, manifest)
            remove_candidate_training_metadata()
        finished_at = utc_now_iso()
        state_updates = {
            "last_finished_at": finished_at,
            "last_status": "failed",
            "last_error": str(exc),
            "last_log_path": log_file,
            "last_backup_path": backup_dir,
        }
        if candidate_metadata_path:
            state_updates["last_candidate_metadata_path"] = candidate_metadata_path
        write_state(**state_updates)
        with open(log_file, "a", encoding="utf-8") as file:
            if candidate_metadata_path:
                file.write(f"\n候选训练元数据已保留: {candidate_metadata_path}\n")
            file.write(f"\nFAILED: {exc}\n")
        print(f"模型重训失败，已回滚旧模型: {exc}")
        print(f"详情日志: {log_file}")
        notify_important(format_retrain_failure_notification(exc, log_file))
        return 1
    finally:
        release_lock()


def main():
    parser = argparse.ArgumentParser(description="Safely retrain model artifacts from latest market data")
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Only train and validate model files, without backtest gating",
    )
    args = parser.parse_args()
    return retrain_once(validate_backtest=not args.skip_backtest)


if __name__ == "__main__":
    raise SystemExit(main())
