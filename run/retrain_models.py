import argparse
import contextlib
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
HARD_MIN_CLOSED_TRADES = 1
HARD_MIN_PROFIT_FACTOR = 1.0
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

    metadata_path = os.path.join(BASE_DIR, config.TRAINING_METADATA_PATH)
    if not os.path.exists(metadata_path):
        return None

    dst_path = os.path.join(backup_dir, "candidate_training_metadata.json")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(metadata_path, dst_path)
    return dst_path


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
    if not metadata or not metadata.get("oos_start"):
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


def run_walk_forward_validation(log_file, context_backtester, metadata, feature_cols):
    if not bool(config.MODEL_WALK_FORWARD_ENABLED):
        return None
    if not metadata:
        raise RuntimeError("训练元数据缺失，无法执行 walk-forward 验证")

    from backtest.backtest import Backtester
    from train.train import create_labels, train_model_bundle

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

    with open(log_file, "a", encoding="utf-8") as file:
        file.write(f"walk-forward folds={len(slices)}\n")

    for fold in slices:
        train_df = labeled_data.iloc[fold["train_start_pos"]:fold["train_end_pos"]].copy()
        validation_df = labeled_data.iloc[
            fold["validation_start_pos"]:fold["validation_end_pos"]
        ].copy()
        if train_df.empty or validation_df.empty:
            raise RuntimeError(f"walk-forward fold={fold['fold']} 样本为空")

        X_train = train_df[feature_cols].astype(float)
        y_train = train_df["target"]
        fold_models, _, _ = train_model_bundle(X_train, y_train)
        fold_data = context_backtester.data.loc[
            (context_backtester.data.index >= validation_df.index.min()) &
            (context_backtester.data.index <= validation_df.index.max())
        ].copy()
        fold_funding_history = filter_funding_history(
            context_backtester.funding_history,
            fold_data.index.min(),
            fold_data.index.max(),
        )

        append_log_header(log_file, f"walk_forward_fold_{fold['fold']}")
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
                    precomputed_data=fold_data,
                    feature_cols=feature_cols,
                    models=fold_models,
                    model_weights=config.MODEL_WEIGHTS,
                    funding_history=fold_funding_history,
                    enable_csv_dump=False,
                    show_progress=False,
                    emit_diagnostics=False,
                )
                fold_summary = backtester.run_backtest()

        if not fold_summary:
            raise RuntimeError(f"walk-forward fold={fold['fold']} 未返回 summary")
        fold_summary.update({
            "fold": fold["fold"],
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(fold_data)),
            "validation_start": validation_df.index.min().isoformat(),
            "validation_end": validation_df.index.max().isoformat(),
        })
        fold_summaries.append(fold_summary)

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
                funding_history=context_backtester.funding_history,
                enable_csv_dump=False,
                show_progress=False,
                emit_diagnostics=False,
            )
            return backtester.run_backtest()


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
