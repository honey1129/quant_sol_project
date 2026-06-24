import argparse
import contextlib
import io
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.backtest import Backtester
from config import config
from core import signal_engine
from core.regime_filter import derive_market_regime
from core.trend_filter import derive_trend_context
from train.train import create_labels
from utils.utils import BASE_DIR, LOGS_DIR, log_info


DIRECTION_LABELS = {0: "short", 1: "long", 2: "no_trade"}
BINARY_LABELS = {0: "no_trade", 1: "trade"}


def is_binary_trade_quality(metadata):
    label_mode = str((metadata or {}).get("target_schema") or (metadata or {}).get("label_mode") or "").lower()
    return label_mode == "binary_trade_quality" or label_mode.startswith("binary_")


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def resolve_artifact_path(root_dir, rel_path):
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(root_dir, rel_path)


def read_json(path, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model_bundle(model_root):
    root_dir = os.path.abspath(model_root or BASE_DIR)
    models = {}
    for name, rel_path in config.MODEL_PATHS.items():
        models[name] = joblib.load(resolve_artifact_path(root_dir, rel_path))

    feature_cols = joblib.load(resolve_artifact_path(root_dir, config.FEATURE_LIST_PATH))
    metadata = read_json(resolve_artifact_path(root_dir, config.TRAINING_METADATA_PATH), default={})
    return {
        "root_dir": root_dir,
        "models": models,
        "feature_cols": feature_cols,
        "metadata": metadata or {},
        "model_weights": dict(config.MODEL_WEIGHTS),
    }


def weighted_predict_matrix(models, X, model_weights, *, trend_biases=None, model_metadata=None):
    binary_quality = is_binary_trade_quality(model_metadata)
    trend_biases = list(trend_biases) if trend_biases is not None else [None] * len(X)
    rows = []
    for row_idx in range(len(X)):
        row_frame = X.iloc[row_idx:row_idx + 1] if hasattr(X, "iloc") else X[row_idx:row_idx + 1]
        directional = signal_engine.weighted_predict_proba(
            models,
            row_frame,
            model_weights,
            trend_bias=trend_biases[row_idx],
            model_metadata=model_metadata,
        )
        short_prob = float(directional[0])
        long_prob = float(directional[1])
        if binary_quality:
            no_trade_prob = max(0.0, min(1.0, 1.0 - max(short_prob, long_prob)))
        else:
            no_trade_prob = 0.0
        rows.append([short_prob, long_prob, no_trade_prob])
    return np.asarray(rows, dtype=float)


def enrich_regime_context(data):
    rows = []
    for _, row in data.iterrows():
        volatility = row.get("volatility_15")
        money_flow_ratio = row.get("money_flow_ratio")
        close_price = row.get("5m_close")
        atr_value = row.get("5m_atr")
        atr_ratio = None
        if pd.notna(atr_value) and pd.notna(close_price) and float(close_price) > 0:
            atr_ratio = float(atr_value) / float(close_price)

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
            volatility=volatility,
            atr_ratio=atr_ratio,
            money_flow_ratio=money_flow_ratio,
            trend_gap_threshold=config.REGIME_TREND_GAP_THRESHOLD,
            high_vol_atr_threshold=config.REGIME_HIGH_VOL_ATR_THRESHOLD,
            high_volatility_threshold=config.REGIME_HIGH_VOLATILITY_THRESHOLD,
            money_flow_extreme_threshold=config.REGIME_MONEY_FLOW_EXTREME_THRESHOLD,
        )
        rows.append({
            "diag_trend_bias": trend_context.get("trend_bias"),
            "diag_trend_gap": trend_context.get("trend_gap"),
            "diag_regime": regime_context.get("regime"),
            "diag_regime_reason": regime_context.get("regime_reason"),
            "diag_atr_ratio": atr_ratio,
        })

    context_df = pd.DataFrame(rows, index=data.index)
    return pd.concat([data, context_df], axis=1)


def add_predictions(data, bundle):
    X = data[bundle["feature_cols"]].astype(float)
    metadata = bundle.get("metadata") or {}
    probs = weighted_predict_matrix(
        bundle["models"],
        X,
        bundle["model_weights"],
        trend_biases=data.get("diag_trend_bias"),
        model_metadata=metadata,
    )
    data = data.copy()
    data["short_prob"] = probs[:, 0]
    data["long_prob"] = probs[:, 1]
    data["no_trade_prob"] = probs[:, 2]
    data["pred_label"] = np.argmax(probs, axis=1)
    data["pred_direction"] = data["pred_label"].map(DIRECTION_LABELS)
    if is_binary_trade_quality(metadata):
        actual_labels = []
        for _, row in data.iterrows():
            if int(row["target"]) == 1 and row.get("diag_trend_bias") == "long":
                actual_labels.append(1)
            elif int(row["target"]) == 1 and row.get("diag_trend_bias") == "short":
                actual_labels.append(0)
            else:
                actual_labels.append(2)
        data["actual_label"] = actual_labels
        data["actual_binary_label"] = data["target"].astype(int).map(BINARY_LABELS)
    else:
        data["actual_label"] = data["target"].astype(int)
        data["actual_binary_label"] = None
    data["actual_direction"] = data["actual_label"].astype(int).map(DIRECTION_LABELS)
    data["prob_gap"] = (data["long_prob"] - data["short_prob"]).abs()
    return data


def select_split(data, metadata, split, rows):
    selected = data
    if split == "validation" and metadata.get("validation_start") and metadata.get("validation_end"):
        start = pd.Timestamp(metadata["validation_start"])
        end = pd.Timestamp(metadata["validation_end"])
        selected = selected[(selected.index >= start) & (selected.index <= end)]
    elif split == "oos" and metadata.get("oos_start"):
        start = pd.Timestamp(metadata["oos_start"])
        selected = selected[selected.index >= start]
    elif split not in {"all", "validation", "oos"}:
        raise ValueError(f"不支持的 split: {split}")

    if rows is not None and rows > 0:
        selected = selected.tail(int(rows))
    return selected.copy()


def confusion_payload(y_true, y_pred):
    y_true = [int(value) for value in y_true]
    y_pred = [int(value) for value in y_pred]
    counts = Counter(zip(y_true, y_pred))
    return {
        f"actual_{DIRECTION_LABELS[actual]}": {
            f"pred_{DIRECTION_LABELS[pred]}": int(counts.get((actual, pred), 0))
            for pred in (0, 1, 2)
        }
        for actual in (0, 1, 2)
    }


def direction_distribution(values):
    counts = Counter(str(value) for value in values)
    total = max(sum(counts.values()), 1)
    return {
        direction: {
            "count": int(count),
            "pct": float(count / total * 100.0),
        }
        for direction, count in sorted(counts.items())
    }


def quantiles(values):
    cleaned = pd.Series(values).dropna()
    if cleaned.empty:
        return {}
    return {
        "p10": float(cleaned.quantile(0.10)),
        "p50": float(cleaned.quantile(0.50)),
        "p90": float(cleaned.quantile(0.90)),
    }


def classification_metrics(group):
    y_true = group["actual_label"].astype(int)
    y_pred = group["pred_label"].astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return {
        "rows": int(len(group)),
        "label_distribution": direction_distribution(group["actual_direction"]),
        "signal_direction_distribution": direction_distribution(group["pred_direction"]),
        "confusion_matrix": confusion_payload(y_true, y_pred),
        "short": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "long": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
        "no_trade": {
            "precision": float(precision[2]),
            "recall": float(recall[2]),
            "f1": float(f1[2]),
            "support": int(support[2]),
        },
        "prob_gap_quantiles": quantiles(group["prob_gap"]),
        "avg_long_prob": float(group["long_prob"].mean()),
        "avg_short_prob": float(group["short_prob"].mean()),
        "avg_no_trade_prob": float(group["no_trade_prob"].mean()),
    }


def build_classification_report(data):
    report = {"all": classification_metrics(data)}
    by_regime = {}
    for regime, group in data.groupby("diag_regime", dropna=False):
        by_regime[str(regime or "unknown")] = classification_metrics(group)
    report["by_regime"] = by_regime
    return report


def compact_backtest_summary(summary):
    keys = [
        "final_equity",
        "return_pct",
        "max_drawdown_pct",
        "trade_count",
        "closed_trade_count",
        "win_rate_pct",
        "profit_factor",
        "avg_win_loss_ratio",
        "net_pnl_after_costs",
        "net_return_pct_after_costs",
        "fees_paid",
        "slippage_cost",
        "funding_pnl",
        "take_profit_count",
        "stop_loss_count",
        "decision_action_counts",
        "decision_reason_top",
        "decision_direction_counts",
        "decision_regime_counts",
        "decision_regime_signal_summary",
        "decision_regime_reason_top",
        "decision_direction_reason_top",
        "decision_probability_quantiles",
        "decision_gate_config",
        "decision_hold_examples",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def run_backtest(seed_bt, data):
    if len(data) < 2:
        return {"skipped": True, "reason": "rows<2", "rows": int(len(data))}

    bt = Backtester(
        "multi_period",
        seed_bt.window,
        data_dict=seed_bt.data_dict,
        reward_risk=seed_bt.reward_risk,
        precomputed_data=data,
        feature_cols=seed_bt.feature_cols,
        models=seed_bt.models,
        model_weights=seed_bt.model_weights,
        funding_history=seed_bt.funding_history,
        enable_csv_dump=False,
        show_progress=False,
        emit_diagnostics=False,
    )
    original_predict_row = bt._predict_row
    bt._predict_row = lambda row: (row["long_prob"], row["short_prob"])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            summary = bt.run_backtest()
    finally:
        bt._predict_row = original_predict_row
    return compact_backtest_summary(summary or {})


def build_backtest_report(seed_bt, data, min_regime_rows, include_regime_backtests=True):
    report = {
        "all": run_backtest(seed_bt, data),
        "by_regime": {},
        "per_regime_note": (
            "Per-regime backtests filter rows by regime and may be non-contiguous; "
            "use them for localization, not as promotion metrics."
        ),
    }
    if not include_regime_backtests:
        return report

    for regime, group in data.groupby("diag_regime", dropna=False):
        regime = str(regime or "unknown")
        if len(group) < min_regime_rows:
            report["by_regime"][regime] = {
                "skipped": True,
                "reason": f"rows<{min_regime_rows}",
                "rows": int(len(group)),
            }
            continue
        report["by_regime"][regime] = run_backtest(seed_bt, group.copy())
    return report


def create_seed_backtester(bundle):
    base_interval = config.INTERVALS[0] if config.INTERVALS else "5m"
    window = config.WINDOWS.get(base_interval, 1000)
    return Backtester(
        "multi_period",
        window,
        feature_cols=bundle["feature_cols"],
        models=bundle["models"],
        model_weights=bundle["model_weights"],
        enable_csv_dump=False,
        show_progress=False,
        emit_diagnostics=False,
    )


def build_report(args):
    bundle = load_model_bundle(args.model_root)
    seed_bt = create_seed_backtester(bundle)

    labeled = create_labels(
        seed_bt.data.copy(),
        future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
        threshold=float(config.MODEL_LABEL_THRESHOLD),
        tradable_only=not bool(args.raw_labels),
    )
    label_filter_summary = labeled.attrs.get("label_filter_summary", {})
    labeled = enrich_regime_context(labeled)
    labeled = add_predictions(labeled, bundle)
    selected = select_split(labeled, bundle["metadata"], args.split, args.rows)
    if selected.empty:
        raise RuntimeError("诊断样本为空，请调整 --split 或 --rows")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_root": bundle["root_dir"],
        "split": args.split,
        "rows": int(len(selected)),
        "start": selected.index.min().isoformat(),
        "end": selected.index.max().isoformat(),
        "label_future_window": int(config.MODEL_LABEL_FUTURE_WINDOW),
        "label_threshold": float(config.MODEL_LABEL_THRESHOLD),
        "label_mode": (
            ("raw_quality" if bool(config.MODEL_TRAIN_NO_TRADE_LABELS) else "raw_binary")
            if bool(args.raw_labels)
            else ("tradable_quality" if bool(config.MODEL_TRAIN_NO_TRADE_LABELS) else "tradable_binary")
        ),
        "label_filter_summary": label_filter_summary,
        "model_weights": bundle["model_weights"],
        "metadata": {
            "created_at": bundle["metadata"].get("created_at"),
            "validation_start": bundle["metadata"].get("validation_start"),
            "validation_end": bundle["metadata"].get("validation_end"),
            "oos_start": bundle["metadata"].get("oos_start"),
            "oos_end": bundle["metadata"].get("oos_end"),
            "feature_count": bundle["metadata"].get("feature_count"),
        },
        "classification": build_classification_report(selected),
        "backtest": build_backtest_report(
            seed_bt,
            selected,
            min_regime_rows=int(args.min_regime_backtest_rows),
            include_regime_backtests=not args.no_regime_backtests,
        ),
    }
    return json_safe(report)


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"training_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def print_summary(report, path):
    log_info(
        "训练诊断完成: "
        f"rows={report['rows']} split={report['split']} "
        f"range={report['start']}..{report['end']}"
    )
    for regime, metrics in report["classification"]["by_regime"].items():
        signal_dist = metrics.get("signal_direction_distribution", {})
        log_info(
            "regime诊断 "
            f"{regime}: rows={metrics['rows']} "
            f"long_precision={metrics['long']['precision']:.3f} "
            f"long_recall={metrics['long']['recall']:.3f} "
            f"short_precision={metrics['short']['precision']:.3f} "
            f"short_recall={metrics['short']['recall']:.3f} "
            f"no_trade_precision={metrics['no_trade']['precision']:.3f} "
            f"no_trade_recall={metrics['no_trade']['recall']:.3f} "
            f"signals={signal_dist}"
        )
    backtest = report["backtest"]["all"]
    log_info(
        "整体诊断回测: "
        f"closed={backtest.get('closed_trade_count')} "
        f"win_rate={backtest.get('win_rate_pct')} "
        f"pf={backtest.get('profit_factor')} "
        f"net={backtest.get('net_pnl_after_costs')}"
    )
    if backtest.get("decision_reason_top"):
        log_info(f"交易门槛原因TOP: {backtest.get('decision_reason_top')}")
    if backtest.get("decision_probability_quantiles"):
        log_info(f"交易门槛分位: {backtest.get('decision_probability_quantiles')}")
    log_info(f"训练诊断报告: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="输出模型按 regime 分层的分类指标和诊断回测结果")
    parser.add_argument("--model-root", default=BASE_DIR, help="模型产物根目录，默认项目根目录")
    parser.add_argument("--split", choices=["all", "validation", "oos"], default="all", help="诊断样本切片")
    parser.add_argument("--rows", type=int, default=int(os.getenv("TRAIN_DIAG_ROWS", "3000")), help="使用切片尾部 N 行；<=0 表示全量")
    parser.add_argument("--min-regime-backtest-rows", type=int, default=30, help="分 regime 回测最少样本数")
    parser.add_argument("--no-regime-backtests", action="store_true", help="只跑整体回测，不跑分 regime 过滤回测")
    parser.add_argument("--raw-labels", action="store_true", help="使用原始涨跌标签，不按交易门禁过滤")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.rows is not None and args.rows <= 0:
        args.rows = None
    report = build_report(args)
    path = write_report(report, args.output)
    print_summary(report, path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
