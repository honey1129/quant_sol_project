import argparse
import contextlib
import io
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.backtest import Backtester
from config import config
from run import training_diagnostics as td
from utils.utils import LOGS_DIR, log_info


def parse_float_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        result.append(float(item))
    return result


def apply_overrides(overrides):
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    return originals


def restore_overrides(originals):
    for key, value in originals.items():
        setattr(config, key, value)


def load_diagnostic_data(model_root, split, rows, raw_labels=False):
    bundle = td.load_model_bundle(model_root)
    seed_bt = td.create_seed_backtester(bundle)
    labeled = td.create_labels(
        seed_bt.data.copy(),
        future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
        threshold=float(config.MODEL_LABEL_THRESHOLD),
        tradable_only=not bool(raw_labels),
    )
    labeled = td.enrich_regime_context(labeled)
    labeled = td.add_predictions(labeled, bundle)
    selected = td.select_split(labeled, bundle["metadata"], split, rows)
    if selected.empty:
        raise RuntimeError("校准样本为空，请调整 --split 或 --rows")
    return bundle, seed_bt, selected


def calibration_bins(data, direction, bins):
    label = 1 if direction == "long" else 0
    prob_col = "long_prob" if direction == "long" else "short_prob"
    target = (data["target"].astype(int) == label).astype(float)
    probs = data[prob_col].astype(float)
    brier = float(np.mean((probs.to_numpy() - target.to_numpy()) ** 2))

    rows = []
    ece = 0.0
    total = max(1, len(data))
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper >= 1.0:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        bucket = data[mask]
        if bucket.empty:
            continue
        bucket_target = target[mask]
        avg_prob = float(probs[mask].mean())
        hit_rate = float(bucket_target.mean())
        rows.append({
            "bin": f"[{lower:.2f},{upper:.2f}{']' if upper >= 1.0 else ')'}",
            "rows": int(len(bucket)),
            "avg_prob": avg_prob,
            "hit_rate": hit_rate,
            "error": float(avg_prob - hit_rate),
        })
        ece += len(bucket) / total * abs(avg_prob - hit_rate)

    return {
        "direction": direction,
        "rows": int(len(data)),
        "brier": brier,
        "ece": float(ece),
        "bins": rows,
    }


def build_probability_calibration_report(data, bins):
    report = {
        "all": {
            "long": calibration_bins(data, "long", bins),
            "short": calibration_bins(data, "short", bins),
        },
        "by_regime": {},
    }
    for regime, group in data.groupby("diag_regime", dropna=False):
        regime = str(regime or "unknown")
        report["by_regime"][regime] = {
            "long": calibration_bins(group, "long", bins),
            "short": calibration_bins(group, "short", bins),
        }
    return report


def weak_signal_gate_counts(data, threshold_long, threshold_short, signal_min_prob_diff):
    long_prob = data["long_prob"].astype(float)
    short_prob = data["short_prob"].astype(float)
    gap = (long_prob - short_prob).abs()
    long_mask = (long_prob >= short_prob) & (long_prob > threshold_long) & (gap >= signal_min_prob_diff)
    short_mask = (short_prob > long_prob) & (short_prob > threshold_short) & (gap >= signal_min_prob_diff)
    return {
        "rows": int(len(data)),
        "long_gate_count": int(long_mask.sum()),
        "short_gate_count": int(short_mask.sum()),
        "gate_count": int((long_mask | short_mask).sum()),
        "long_gate_pct": float(long_mask.mean() * 100.0) if len(data) else 0.0,
        "short_gate_pct": float(short_mask.mean() * 100.0) if len(data) else 0.0,
        "gate_pct": float((long_mask | short_mask).mean() * 100.0) if len(data) else 0.0,
    }


def compact_summary(summary):
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
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def run_candidate(seed_bt, data, overrides):
    originals = apply_overrides(overrides)
    try:
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
        return compact_summary(summary or {})
    finally:
        restore_overrides(originals)


def score_candidate(item, min_closed_trades):
    closed = int(item.get("closed_trade_count") or 0)
    net = float(item.get("net_pnl_after_costs") or 0.0)
    pf = float(item.get("profit_factor") or 0.0)
    drawdown = abs(float(item.get("max_drawdown_pct") or 0.0))
    enough_trades = 1 if closed >= min_closed_trades else 0
    return (enough_trades, net, pf, -drawdown, closed)


def build_candidates(long_thresholds, short_thresholds, gaps, asymmetric=False):
    candidates = []
    if asymmetric:
        threshold_pairs = [(long, short) for long in long_thresholds for short in short_thresholds]
    else:
        shared = sorted(set(long_thresholds) | set(short_thresholds))
        threshold_pairs = [(value, value) for value in shared]

    for long_threshold, short_threshold in threshold_pairs:
        for gap in gaps:
            candidates.append({
                "name": f"tl{long_threshold:.2f}_ts{short_threshold:.2f}_gap{gap:.2f}",
                "overrides": {
                    "THRESHOLD_LONG": float(long_threshold),
                    "THRESHOLD_SHORT": float(short_threshold),
                    "SIGNAL_MIN_PROB_DIFF": float(gap),
                },
            })
    return candidates


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"trade_threshold_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(td.json_safe(report), file, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def print_top_results(results, limit=12):
    headers = [
        "name",
        "closed",
        "trades",
        "net",
        "pf",
        "win%",
        "maxDD%",
        "gate",
        "top_reason",
    ]
    print(",".join(headers))
    for item in results[:limit]:
        top_reason = "-"
        if item.get("decision_reason_top"):
            top_reason = str(item["decision_reason_top"][0])
        print(
            f"{item['name']},"
            f"{int(item.get('closed_trade_count') or 0)},"
            f"{int(item.get('trade_count') or 0)},"
            f"{float(item.get('net_pnl_after_costs') or 0.0):.2f},"
            f"{float(item.get('profit_factor') or 0.0):.3f},"
            f"{float(item.get('win_rate_pct') or 0.0):.2f},"
            f"{float(item.get('max_drawdown_pct') or 0.0):.2f},"
            f"{int(item.get('weak_signal_gate_counts', {}).get('gate_count') or 0)},"
            f"\"{top_reason}\""
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="校准交易方向阈值、概率差阈值，并输出概率校准表")
    parser.add_argument("--model-root", default=td.BASE_DIR, help="模型产物根目录，默认项目根目录")
    parser.add_argument("--split", choices=["all", "validation", "oos"], default="all", help="校准样本切片")
    parser.add_argument("--rows", type=int, default=int(os.getenv("THRESHOLD_CALIBRATION_ROWS", "927")), help="使用切片尾部 N 行；<=0 表示全量")
    parser.add_argument("--long-thresholds", default=os.getenv("THRESHOLD_CALIBRATION_LONGS"), help="逗号分隔 long 阈值")
    parser.add_argument("--short-thresholds", default=os.getenv("THRESHOLD_CALIBRATION_SHORTS"), help="逗号分隔 short 阈值")
    parser.add_argument("--gaps", default=os.getenv("THRESHOLD_CALIBRATION_GAPS"), help="逗号分隔 SIGNAL_MIN_PROB_DIFF 值")
    parser.add_argument("--bins", default=os.getenv("THRESHOLD_CALIBRATION_BINS"), help="逗号分隔概率校准 bin 边界")
    parser.add_argument("--asymmetric", action="store_true", help="跑 long/short 阈值笛卡尔积；默认使用对称阈值")
    parser.add_argument("--raw-labels", action="store_true", help="使用原始涨跌标签，不按交易门禁过滤")
    parser.add_argument("--min-closed-trades", type=int, default=int(os.getenv("THRESHOLD_CALIBRATION_MIN_CLOSED_TRADES", "5")), help="推荐排序最低平仓笔数")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.rows is not None and args.rows <= 0:
        args.rows = None

    default_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, float(config.THRESHOLD_LONG)]
    default_gaps = [0.08, 0.12, 0.16, 0.20, float(config.SIGNAL_MIN_PROB_DIFF)]
    long_thresholds = parse_float_list(args.long_thresholds, default_thresholds)
    short_thresholds = parse_float_list(args.short_thresholds, default_thresholds)
    gaps = parse_float_list(args.gaps, default_gaps)
    bins = parse_float_list(args.bins, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    bundle, seed_bt, data = load_diagnostic_data(args.model_root, args.split, args.rows, raw_labels=args.raw_labels)
    candidates = build_candidates(long_thresholds, short_thresholds, gaps, asymmetric=bool(args.asymmetric))

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_root": os.path.abspath(args.model_root),
        "split": args.split,
        "rows": int(len(data)),
        "start": data.index.min().isoformat(),
        "end": data.index.max().isoformat(),
        "metadata": {
            "created_at": bundle["metadata"].get("created_at"),
            "validation_start": bundle["metadata"].get("validation_start"),
            "validation_end": bundle["metadata"].get("validation_end"),
            "oos_start": bundle["metadata"].get("oos_start"),
            "oos_end": bundle["metadata"].get("oos_end"),
            "feature_count": bundle["metadata"].get("feature_count"),
        },
        "base_gate_config": {
            "threshold_long": float(config.THRESHOLD_LONG),
            "threshold_short": float(config.THRESHOLD_SHORT),
            "signal_min_prob_diff": float(config.SIGNAL_MIN_PROB_DIFF),
            "min_signal_target_ratio": float(config.MIN_SIGNAL_TARGET_RATIO),
            "min_adjust_amount": float(config.MIN_ADJUST_AMOUNT),
            "min_expected_net_edge": float(config.MIN_EXPECTED_NET_EDGE),
        },
        "probability_calibration": build_probability_calibration_report(data, bins),
        "candidates": [],
    }

    path = write_report(report, args.output)
    log_info(f"阈值校准开始: candidates={len(candidates)} rows={len(data)}")
    for idx, candidate in enumerate(candidates, start=1):
        overrides = candidate["overrides"]
        summary = run_candidate(seed_bt, data.copy(), overrides)
        summary["name"] = candidate["name"]
        summary["overrides"] = overrides
        summary["weak_signal_gate_counts"] = weak_signal_gate_counts(
            data,
            overrides["THRESHOLD_LONG"],
            overrides["THRESHOLD_SHORT"],
            overrides["SIGNAL_MIN_PROB_DIFF"],
        )
        report["candidates"].append(summary)
        if idx % 10 == 0 or idx == len(candidates):
            write_report(report, path)
            log_info(f"阈值校准进度: {idx}/{len(candidates)}")

    ranked = sorted(
        report["candidates"],
        key=lambda item: score_candidate(item, int(args.min_closed_trades)),
        reverse=True,
    )
    report["recommended"] = ranked[:10]
    path = write_report(report, path)
    log_info(f"阈值校准报告: {path}")
    print_top_results(ranked)
    print(json.dumps(td.json_safe({"report_path": path, "recommended": ranked[:10]}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
