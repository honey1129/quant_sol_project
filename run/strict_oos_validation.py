import argparse
import contextlib
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

from backtest.backtest import Backtester
from config import config
from run.retrain_models import (
    aggregate_backtest_summaries,
    load_model_bundle,
    run_walk_forward_validation,
    write_json_atomic,
)
from utils.utils import BASE_DIR, LOGS_DIR


REPORT_DIR = os.path.join(LOGS_DIR, "strict_oos")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_positive_ints(value):
    values = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise ValueError("window days must be positive")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError("at least one window day is required")
    return sorted(values)


def bars_per_day(interval):
    text = str(interval or "").strip()
    if text.endswith("m"):
        return 1440.0 / int(text[:-1])
    if text.endswith("H") or text.endswith("h"):
        return 24.0 / int(text[:-1])
    if text.endswith("D") or text.endswith("d"):
        return 1.0 / int(text[:-1])
    raise ValueError(f"unsupported interval: {interval}")


def required_data_windows(total_days, warmup_days=10):
    requested_days = int(total_days) + int(warmup_days)
    windows = dict(config.WINDOWS)
    for interval in config.INTERVALS:
        required = int(math.ceil(requested_days * bars_per_day(interval)))
        windows[interval] = max(int(windows.get(interval, 0)), required)
    return windows


@contextlib.contextmanager
def temporary_config(**overrides):
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(config, key, value)


def _parse_timestamp(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _window_fold_summaries(folds, end_ts, days):
    cutoff = end_ts - pd.Timedelta(days=int(days))
    selected = []
    for fold in folds:
        start = _parse_timestamp(fold["validation_start"])
        if start >= cutoff:
            selected.append(fold)
    return selected


def _summarize_window(folds, days):
    summary = aggregate_backtest_summaries(folds)
    positive_folds = sum(
        1 for fold in folds if float(fold.get("net_pnl_after_costs", 0.0)) > 0
    )
    baseline_folds = [
        fold["trend_baseline_summary"]
        for fold in folds
        if fold.get("trend_baseline_summary")
    ]
    baseline = aggregate_backtest_summaries(baseline_folds) if baseline_folds else None
    summary.update({
        "window_days": int(days),
        "period_start": min(fold["validation_start"] for fold in folds),
        "period_end": max(fold["validation_end"] for fold in folds),
        "positive_fold_count": int(positive_folds),
        "positive_fold_ratio": float(positive_folds / len(folds)),
        "trend_baseline": baseline,
    })
    return summary


def build_window_summaries(walk_forward_summary, window_days):
    folds = list(walk_forward_summary.get("folds") or [])
    if not folds:
        raise RuntimeError("strict OOS audit has no completed folds")
    end_ts = max(_parse_timestamp(fold["validation_end"]) for fold in folds)
    summaries = {}
    for days in sorted(window_days):
        selected = _window_fold_summaries(folds, end_ts, days)
        if not selected:
            raise RuntimeError(f"strict OOS window has no folds: {days}d")
        summaries[str(days)] = _summarize_window(selected, days)
    return summaries


def _gate(name, passed, actual, required, scope, *, decisive=True):
    return {
        "name": name,
        "scope": scope,
        "passed": bool(passed),
        "decisive": bool(decisive),
        "actual": actual,
        "required": required,
    }


def evaluate_strategy(
    window_summaries,
    *,
    min_closed_trades=30,
    min_profit_factor=1.20,
    max_drawdown_pct=-5.0,
    min_positive_fold_ratio=0.60,
    min_group_closed_trades=10,
):
    ordered_days = sorted(int(key) for key in window_summaries)
    max_days = max(ordered_days)
    gates = []
    sample_complete = True
    longest_sample_complete = False

    for days in ordered_days:
        summary = window_summaries[str(days)]
        required_trades = max(
            1,
            int(math.ceil(min_closed_trades * days / max_days)),
        )
        closed = int(summary.get("closed_trade_count", 0))
        has_sample = closed >= required_trades
        sample_complete = sample_complete and has_sample
        if days == max_days:
            longest_sample_complete = has_sample
        gates.append(_gate(
            "minimum_closed_trades",
            has_sample,
            closed,
            required_trades,
            f"{days}d",
            decisive=False,
        ))
        if not has_sample:
            continue

        net_pnl = float(summary.get("net_pnl_after_costs", 0.0))
        profit_factor = float(summary.get("profit_factor", 0.0))
        drawdown = float(summary.get("max_drawdown_pct", 0.0))
        positive_ratio = float(summary.get("positive_fold_ratio", 0.0))
        baseline = summary.get("trend_baseline") or {}
        baseline_net = float(baseline.get("net_pnl_after_costs", 0.0))
        gates.extend([
            _gate("positive_after_costs", net_pnl > 0, net_pnl, "> 0", f"{days}d"),
            _gate(
                "minimum_profit_factor",
                profit_factor >= min_profit_factor,
                profit_factor,
                min_profit_factor,
                f"{days}d",
            ),
            _gate(
                "maximum_drawdown",
                drawdown >= max_drawdown_pct,
                drawdown,
                f">= {max_drawdown_pct}",
                f"{days}d",
            ),
            _gate(
                "fold_consistency",
                positive_ratio >= min_positive_fold_ratio,
                positive_ratio,
                min_positive_fold_ratio,
                f"{days}d",
            ),
            _gate(
                "beat_trend_baseline",
                net_pnl > baseline_net,
                net_pnl - baseline_net,
                "> 0",
                f"{days}d",
            ),
        ])

    longest = window_summaries[str(max_days)]
    attribution = longest.get("closed_trade_attribution") or {}
    for dimension in ("by_direction", "by_entry_regime"):
        for group, stats in sorted((attribution.get(dimension) or {}).items()):
            closed = int(stats.get("closed_trade_count", 0))
            if closed < min_group_closed_trades:
                continue
            net_pnl = float(stats.get("net_pnl_after_costs", 0.0))
            profit_factor = float(stats.get("profit_factor", 0.0))
            scope = f"{max_days}d:{dimension}:{group}"
            gates.extend([
                _gate("group_positive_after_costs", net_pnl > 0, net_pnl, "> 0", scope),
                _gate("group_profit_factor", profit_factor > 1.0, profit_factor, "> 1.0", scope),
            ])

    decisive_failures = [
        gate for gate in gates if gate["decisive"] and not gate["passed"]
    ]
    if not longest_sample_complete:
        verdict = "WATCH"
        reason = "insufficient_oos_closed_trades"
    elif decisive_failures:
        verdict = "ELIMINATE"
        reason = "failed_robustness_gates"
    elif not sample_complete:
        verdict = "WATCH"
        reason = "insufficient_recent_window_trades"
    else:
        verdict = "KEEP"
        reason = "passed_all_robustness_gates"

    return {
        "verdict": verdict,
        "reason": reason,
        "sample_complete": bool(sample_complete),
        "failed_gate_count": len(decisive_failures),
        "gates": gates,
        "thresholds": {
            "min_closed_trades": int(min_closed_trades),
            "min_profit_factor": float(min_profit_factor),
            "max_drawdown_pct": float(max_drawdown_pct),
            "min_positive_fold_ratio": float(min_positive_fold_ratio),
            "min_group_closed_trades": int(min_group_closed_trades),
        },
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def format_markdown(report):
    decision = report["decision"]
    lines = [
        "# Strict OOS Strategy Audit",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Verdict: **{decision['verdict']}**",
        f"- Reason: `{decision['reason']}`",
        f"- OOS folds: {report['walk_forward']['fold_count']}",
        "",
        "## Rolling windows",
        "",
        "| Window | Closed | Net after costs | PF | Max DD | Positive folds | Trend baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for days in sorted(report["windows"], key=int):
        summary = report["windows"][days]
        baseline = summary.get("trend_baseline") or {}
        lines.append(
            f"| {days}d | {summary['closed_trade_count']} | "
            f"{summary['net_pnl_after_costs']:.2f} | {summary['profit_factor']:.4f} | "
            f"{summary['max_drawdown_pct']:.2f}% | "
            f"{summary['positive_fold_ratio']:.1%} | "
            f"{float(baseline.get('net_pnl_after_costs', 0.0)):.2f} |"
        )

    lines.extend([
        "",
        "## Failed gates",
        "",
        "| Scope | Gate | Actual | Required |",
        "| --- | --- | ---: | --- |",
    ])
    failed = [gate for gate in decision["gates"] if not gate["passed"]]
    if failed:
        for gate in failed:
            lines.append(
                f"| {gate['scope']} | {gate['name']} | {gate['actual']} | {gate['required']} |"
            )
    else:
        lines.append("| all | none | - | - |")

    longest_days = str(max(int(key) for key in report["windows"]))
    attribution = report["windows"][longest_days].get("closed_trade_attribution") or {}
    for dimension, title in (
        ("by_direction", "Direction attribution"),
        ("by_entry_regime", "Entry regime attribution"),
    ):
        lines.extend([
            "",
            f"## {title}",
            "",
            "| Group | Closed | Net after costs | PF | Win rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        groups = attribution.get(dimension) or {}
        if not groups:
            lines.append("| none | 0 | 0.00 | 0.0000 | 0.0% |")
        for group, stats in sorted(groups.items()):
            lines.append(
                f"| {group} | {stats['closed_trade_count']} | "
                f"{stats['net_pnl_after_costs']:.2f} | {stats['profit_factor']:.4f} | "
                f"{stats['win_rate_pct']:.1f}% |"
            )
    return "\n".join(lines) + "\n"


def write_report(report, output_dir=REPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"strict_oos_{run_id}.json")
    markdown_path = os.path.join(output_dir, f"strict_oos_{run_id}.md")
    payload = _json_safe(report)
    write_json_atomic(json_path, payload)
    write_json_atomic(os.path.join(output_dir, "latest.json"), payload)
    markdown = format_markdown(report)
    for path in (markdown_path, os.path.join(output_dir, "latest.md")):
        with open(path, "w", encoding="utf-8") as file:
            file.write(markdown)
    return json_path, markdown_path


def run_audit(args):
    if args.train_days <= 0 or args.oos_days <= 0 or args.fold_days <= 0:
        raise ValueError("train-days, oos-days and fold-days must be positive")
    if args.min_closed_trades <= 0 or args.min_group_closed_trades <= 0:
        raise ValueError("closed-trade thresholds must be positive")
    windows = parse_positive_ints(args.windows)
    oos_days = max(max(windows), int(args.oos_days))
    fold_count = int(math.ceil(oos_days / args.fold_days))
    data_windows = required_data_windows(args.train_days + oos_days)
    base_interval = config.INTERVALS[0]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "audit.log")

    overrides = {
        "WINDOWS": data_windows,
        "MODEL_WALK_FORWARD_ENABLED": True,
        "MODEL_WALK_FORWARD_FOLDS": fold_count,
        "MODEL_WALK_FORWARD_MIN_FOLDS": fold_count,
        "MODEL_WALK_FORWARD_MIN_VALIDATION_ROWS": max(
            100,
            int(args.fold_days * bars_per_day(base_interval) * 0.80),
        ),
        "MODEL_WALK_FORWARD_FAIL_FAST": False,
        "MODEL_WALK_FORWARD_THRESHOLD_SWEEP_ENABLED": False,
    }

    with temporary_config(**overrides):
        context = Backtester(
            "multi_period",
            data_windows[base_interval],
            enable_csv_dump=False,
            show_progress=False,
            emit_diagnostics=False,
        )
        bundle = load_model_bundle(BASE_DIR)
        data_end = context.data.index.max()
        audit_start = data_end - pd.Timedelta(days=oos_days)
        train_start = audit_start - pd.Timedelta(days=args.train_days)
        scoped_data = context.data.loc[context.data.index >= train_start].copy()
        actual_train_days = (
            (audit_start - scoped_data.index.min()).total_seconds() / 86400.0
            if not scoped_data.empty
            else 0.0
        )
        if scoped_data.empty or actual_train_days < (args.train_days - 1.0):
            raise RuntimeError(
                "strict OOS training history is insufficient: "
                f"requested_train_days={args.train_days}, "
                f"actual_train_days={actual_train_days:.2f}, "
                f"available_start={context.data.index.min().isoformat()}"
            )
        context.data = scoped_data
        context.price_series = scoped_data["5m_close"]
        metadata = dict(bundle.get("metadata") or {})
        metadata.update({
            "validation_start": audit_start.isoformat(),
            "validation_end": data_end.isoformat(),
            "purge_bars": int(config.MODEL_PURGE_BARS),
        })
        walk_forward = run_walk_forward_validation(
            log_path,
            context,
            metadata,
            bundle["feature_cols"],
            enforce_gates=False,
            include_trend_baseline=True,
        )

    window_summaries = build_window_summaries(walk_forward, windows)
    decision = evaluate_strategy(
        window_summaries,
        min_closed_trades=args.min_closed_trades,
        min_profit_factor=args.min_profit_factor,
        max_drawdown_pct=args.max_drawdown_pct,
        min_positive_fold_ratio=args.min_positive_fold_ratio,
        min_group_closed_trades=args.min_group_closed_trades,
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "symbol": config.SYMBOL,
        "intervals": list(config.INTERVALS),
        "data_windows": data_windows,
        "audit_config": {
            "train_days": int(args.train_days),
            "oos_days": int(oos_days),
            "fold_days": int(args.fold_days),
            "window_days": windows,
            "threshold_sweep_enabled": False,
            "trend_baseline_enabled": True,
        },
        "walk_forward": walk_forward,
        "windows": window_summaries,
        "decision": decision,
    }
    return report, write_report(report, output_dir=output_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run a strict rolling out-of-sample strategy audit.",
    )
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--oos-days", type=int, default=90)
    parser.add_argument("--fold-days", type=int, default=10)
    parser.add_argument("--windows", default="30,60,90")
    parser.add_argument("--min-closed-trades", type=int, default=30)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--max-drawdown-pct", type=float, default=-5.0)
    parser.add_argument("--min-positive-fold-ratio", type=float, default=0.60)
    parser.add_argument("--min-group-closed-trades", type=int, default=10)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    return parser


def main():
    args = build_parser().parse_args()
    report, paths = run_audit(args)
    print(json.dumps({
        "verdict": report["decision"]["verdict"],
        "reason": report["decision"]["reason"],
        "json_path": paths[0],
        "markdown_path": paths[1],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
