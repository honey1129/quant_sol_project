import argparse
import contextlib
import io
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import rule_edge_diagnostics as diag
from run import rule_edge_sweep as sweep
from utils.utils import LOGS_DIR, log_info


STATUS_RANK = {
    "stable_positive": 5,
    "weak_positive_low_pf_or_periods": 4,
    "oos_only_unconfirmed": 3,
    "validation_only_failed_oos": 2,
    "no_confirmed_edge": 1,
    "insufficient_samples": 0,
}


def parse_windows(raw_value):
    windows = {}
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"窗口格式应为 interval:rows，实际为 {item!r}")
        interval, rows = item.split(":", 1)
        windows[interval.strip()] = int(rows.strip())
    return windows


@contextmanager
def temporary_windows(raw_windows):
    from config import config

    overrides = parse_windows(raw_windows)
    if not overrides:
        yield dict(config.WINDOWS)
        return

    original_windows = dict(config.WINDOWS)
    original_env = os.environ.get("WINDOWS")
    try:
        merged = dict(config.WINDOWS)
        merged.update(overrides)
        config.WINDOWS = merged
        os.environ["WINDOWS"] = ",".join(f"{key}:{value}" for key, value in merged.items())
        yield dict(config.WINDOWS)
    finally:
        config.WINDOWS = original_windows
        if original_env is None:
            os.environ.pop("WINDOWS", None)
        else:
            os.environ["WINDOWS"] = original_env


def build_sweep_args(args):
    return argparse.Namespace(
        rows=args.rows,
        trend_gaps=args.trend_gaps,
        regime_gap_multipliers=args.regime_gap_multipliers,
        tp_sl_pairs=args.tp_sl_pairs,
        allow_high_vol_values=args.allow_high_vol_values,
        allow_range_values=args.allow_range_values,
        entry_filters="breakout_flow",
        pullback_pct_values="",
        breakout_lookbacks=args.breakout_lookbacks,
        flow_min_values=args.flow_min_values,
        low_vol_max_values="",
        volatility_min_values=args.volatility_min_values,
        trend_gap_min_values=args.trend_gap_min_values,
        max_candidates=args.max_candidates,
        min_rows=args.min_split_rows,
        min_profit_factor=args.min_profit_factor,
        min_mean_net_return=args.min_mean_net_return,
        top_n=args.top_n,
        output=args.output,
        progress=args.progress,
        verbose_candidates=args.verbose_candidates,
    )


def direction_values(data):
    return (
        data.get("label_direction", pd.Series(index=data.index, dtype=object))
        .fillna("none")
        .astype(str)
        .str.lower()
    )


def period_labels(data, freq):
    if data.empty:
        return pd.Series(index=data.index, dtype=object)
    index = pd.DatetimeIndex(pd.to_datetime(data.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_convert(None)
    return pd.Series(index.to_period(freq).astype(str), index=data.index, dtype=object)


def metric_passes(summary, *, min_rows, min_profit_factor, min_mean_net_return):
    rows = int(summary.get("candidate_rows", 0) or 0)
    mean_net = float(summary.get("mean_net_return", 0.0) or 0.0)
    profit_factor = sweep.pf_numeric(summary.get("profit_factor", 0.0))
    return rows >= min_rows and mean_net > min_mean_net_return and profit_factor >= min_profit_factor


def summarize_period_buckets(data, direction, *, freq, min_rows, min_profit_factor, min_mean_net_return):
    data = data.copy()
    calendar_labels = period_labels(data, freq)
    calendar_period_count = int(calendar_labels.nunique(dropna=True))

    target = data.loc[direction_values(data) == str(direction).lower()].copy()
    if target.empty:
        return diag.json_safe({
            "freq": freq,
            "direction": direction,
            "calendar_period_count": calendar_period_count,
            "candidate_period_count": 0,
            "covered_period_count": 0,
            "positive_period_count": 0,
            "positive_period_ratio": 0.0,
            "active_period_coverage": 0.0,
            "periods": [],
        })

    target_labels = period_labels(target, freq)
    periods = []
    for period, group in target.groupby(target_labels, sort=True):
        summary = diag.summarize_group(
            group,
            strict_labeled=None,
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        )
        summary["period"] = str(period)
        summary["start"] = group.index.min().isoformat()
        summary["end"] = group.index.max().isoformat()
        summary["diagnostic_split_counts"] = diag.value_counts(group.get("diagnostic_split"))
        summary["metric_positive"] = metric_passes(
            summary,
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        )
        periods.append(summary)

    covered = [item for item in periods if int(item.get("candidate_rows", 0) or 0) >= int(min_rows)]
    positive = [item for item in covered if item.get("metric_positive")]
    means = [float(item.get("mean_net_return", 0.0) or 0.0) for item in covered]
    candidate_period_count = int(len(periods))
    covered_period_count = int(len(covered))
    positive_period_count = int(len(positive))

    return diag.json_safe({
        "freq": freq,
        "direction": direction,
        "calendar_period_count": calendar_period_count,
        "candidate_period_count": candidate_period_count,
        "covered_period_count": covered_period_count,
        "positive_period_count": positive_period_count,
        "positive_period_ratio": float(positive_period_count / covered_period_count) if covered_period_count else 0.0,
        "active_period_coverage": float(candidate_period_count / calendar_period_count) if calendar_period_count else 0.0,
        "worst_period": min(covered, key=lambda item: float(item.get("mean_net_return", 0.0) or 0.0), default=None),
        "best_period": max(covered, key=lambda item: float(item.get("mean_net_return", 0.0) or 0.0), default=None),
        "min_period_mean_net_return": min(means) if means else 0.0,
        "median_period_mean_net_return": float(pd.Series(means).median()) if means else 0.0,
        "periods": periods,
    })


def numeric_quantile_bucket(series, labels=("low", "mid", "high")):
    values = pd.to_numeric(series, errors="coerce").replace([float("inf"), float("-inf")], pd.NA)
    non_null = values.dropna()
    result = pd.Series("unknown", index=series.index, dtype=object)
    if non_null.empty:
        return result
    unique_count = int(non_null.nunique(dropna=True))
    if unique_count < len(labels):
        median = float(non_null.median())
        result.loc[values.notna()] = values.loc[values.notna()].map(
            lambda item: "high" if float(item) >= median else "low"
        )
        return result
    try:
        bucketed = pd.qcut(non_null, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        median = float(non_null.median())
        result.loc[values.notna()] = values.loc[values.notna()].map(
            lambda item: "high" if float(item) >= median else "low"
        )
        return result

    bucketed = bucketed.astype(str)
    result.loc[bucketed.index] = bucketed
    return result


def regime_bucket(data):
    if "label_regime" in data:
        return data["label_regime"].fillna("unknown").astype(str).str.lower()

    regimes = pd.Series("range", index=data.index, dtype=object)
    if "is_high_vol" in data:
        regimes = regimes.mask(pd.to_numeric(data["is_high_vol"], errors="coerce").fillna(0.0) > 0.5, "range_high_vol")
    if "regime_range_high_vol" in data:
        regimes = regimes.mask(pd.to_numeric(data["regime_range_high_vol"], errors="coerce").fillna(0.0) > 0.5, "range_high_vol")
    if "regime_trend_long" in data:
        regimes = regimes.mask(pd.to_numeric(data["regime_trend_long"], errors="coerce").fillna(0.0) > 0.5, "trend_long")
    if "regime_trend_short" in data:
        regimes = regimes.mask(pd.to_numeric(data["regime_trend_short"], errors="coerce").fillna(0.0) > 0.5, "trend_short")
    return regimes


def summarize_grouped_candidates(data, direction, group_labels, *, min_rows, min_profit_factor, min_mean_net_return):
    target = data.loc[direction_values(data) == str(direction).lower()].copy()
    group_labels = group_labels.reindex(data.index).fillna("unknown").astype(str)
    if target.empty:
        return {
            "group_count": 0,
            "covered_group_count": 0,
            "positive_group_count": 0,
            "positive_group_ratio": 0.0,
            "groups": [],
        }

    groups = []
    for group_name, group in target.groupby(group_labels.reindex(target.index), sort=True):
        summary = diag.summarize_group(
            group,
            strict_labeled=None,
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        )
        summary["group"] = str(group_name)
        summary["start"] = group.index.min().isoformat()
        summary["end"] = group.index.max().isoformat()
        summary["diagnostic_split_counts"] = diag.value_counts(group.get("diagnostic_split"))
        summary["metric_positive"] = metric_passes(
            summary,
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        )
        groups.append(summary)

    covered = [item for item in groups if int(item.get("candidate_rows", 0) or 0) >= int(min_rows)]
    positive = [item for item in covered if item.get("metric_positive")]
    return diag.json_safe({
        "group_count": int(len(groups)),
        "covered_group_count": int(len(covered)),
        "positive_group_count": int(len(positive)),
        "positive_group_ratio": float(len(positive) / len(covered)) if covered else 0.0,
        "worst_group": min(covered, key=lambda item: float(item.get("mean_net_return", 0.0) or 0.0), default=None),
        "best_group": max(covered, key=lambda item: float(item.get("mean_net_return", 0.0) or 0.0), default=None),
        "groups": groups,
    })


def summarize_market_state_breakdown(data, direction, *, min_rows, min_profit_factor, min_mean_net_return):
    data = data.copy()
    breakdown = {
        "by_regime": summarize_grouped_candidates(
            data,
            direction,
            regime_bucket(data),
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        ),
        "by_volatility_15_bucket": summarize_grouped_candidates(
            data,
            direction,
            numeric_quantile_bucket(data.get("volatility_15", pd.Series(index=data.index, dtype=float))),
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        ),
        "by_money_flow_bucket": summarize_grouped_candidates(
            data,
            direction,
            numeric_quantile_bucket(data.get("money_flow_ratio", pd.Series(index=data.index, dtype=float))),
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        ),
        "by_volume_bucket": summarize_grouped_candidates(
            data,
            direction,
            numeric_quantile_bucket(data.get("volume_ratio", pd.Series(index=data.index, dtype=float))),
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        ),
        "by_trend_gap_bucket": summarize_grouped_candidates(
            data,
            direction,
            numeric_quantile_bucket(data.get("trend_gap_abs", pd.Series(index=data.index, dtype=float))),
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        ),
    }

    month_regime = period_labels(data, "M").astype(str) + ":" + regime_bucket(data).astype(str)
    breakdown["by_month_regime"] = summarize_grouped_candidates(
        data,
        direction,
        month_regime,
        min_rows=min_rows,
        min_profit_factor=min_profit_factor,
        min_mean_net_return=min_mean_net_return,
    )
    return diag.json_safe(breakdown)


def split_direction_metrics(report, direction):
    metrics = {}
    for split_name in ("all", "train", "validation", "oos"):
        metrics[split_name] = sweep.metric_snapshot(
            report["splits"][split_name]["by_direction"][direction]
        )
    return metrics


def split_metric_passes(metric, *, min_rows, min_profit_factor, min_mean_net_return):
    rows = int(metric.get("candidate_rows", 0) or 0)
    mean_net = float(metric.get("mean_net_return", 0.0) or 0.0)
    profit_factor = sweep.pf_numeric(metric.get("profit_factor", 0.0))
    return rows >= min_rows and mean_net > min_mean_net_return and profit_factor >= min_profit_factor


def classify_stability(split_metrics, period_stability, args):
    validation = split_metrics["validation"]
    oos = split_metrics["oos"]
    monthly = period_stability["monthly"]
    reasons = []

    if int(validation.get("candidate_rows", 0) or 0) < int(args.min_split_rows):
        reasons.append("validation_insufficient_rows")
    if int(oos.get("candidate_rows", 0) or 0) < int(args.min_split_rows):
        reasons.append("oos_insufficient_rows")
    if int(monthly.get("covered_period_count", 0) or 0) < int(args.min_active_periods):
        reasons.append("insufficient_active_months")
    if float(monthly.get("positive_period_ratio", 0.0) or 0.0) < float(args.min_positive_period_ratio):
        reasons.append("low_positive_month_ratio")

    validation_pass = split_metric_passes(
        validation,
        min_rows=args.min_split_rows,
        min_profit_factor=args.min_profit_factor,
        min_mean_net_return=args.min_mean_net_return,
    )
    oos_pass = split_metric_passes(
        oos,
        min_rows=args.min_split_rows,
        min_profit_factor=args.min_profit_factor,
        min_mean_net_return=args.min_mean_net_return,
    )
    monthly_pass = (
        int(monthly.get("covered_period_count", 0) or 0) >= int(args.min_active_periods)
        and float(monthly.get("positive_period_ratio", 0.0) or 0.0) >= float(args.min_positive_period_ratio)
    )

    if "validation_insufficient_rows" in reasons or "oos_insufficient_rows" in reasons:
        status = "insufficient_samples"
        action = "collect_more_samples_or_relax_filter_for_diagnostics_only"
    elif validation_pass and oos_pass and monthly_pass:
        status = "stable_positive"
        action = "eligible_for_short_only_paper_rule_test"
    elif validation_pass and oos_pass:
        status = "weak_positive_low_pf_or_periods"
        action = "continue_period_stability_work_before_ml"
    elif oos_pass and not validation_pass:
        status = "oos_only_unconfirmed"
        action = "do_not_promote; test_adjacent_breakout_flow_variants"
    elif validation_pass and not oos_pass:
        status = "validation_only_failed_oos"
        action = "reject_or_rework_oos_robustness"
    else:
        status = "no_confirmed_edge"
        action = "reject_or_rework_rule"

    return {
        "status": status,
        "action": action,
        "reason_codes": reasons,
        "validation_pass": bool(validation_pass),
        "oos_pass": bool(oos_pass),
        "monthly_pass": bool(monthly_pass),
        "thresholds": {
            "min_split_rows": int(args.min_split_rows),
            "min_profit_factor": float(args.min_profit_factor),
            "min_mean_net_return": float(args.min_mean_net_return),
            "min_active_periods": int(args.min_active_periods),
            "min_positive_period_ratio": float(args.min_positive_period_ratio),
        },
    }


def result_sort_key(item):
    decision = item.get("stability_decision", {})
    split_metrics = item.get("target_direction_splits", {})
    validation = split_metrics.get("validation", {})
    oos = split_metrics.get("oos", {})
    monthly = item.get("period_stability", {}).get("monthly", {})
    validation_mean = float(validation.get("mean_net_return", 0.0) or 0.0)
    oos_mean = float(oos.get("mean_net_return", 0.0) or 0.0)
    validation_pf = sweep.pf_numeric(validation.get("profit_factor", 0.0))
    oos_pf = sweep.pf_numeric(oos.get("profit_factor", 0.0))
    return (
        STATUS_RANK.get(decision.get("status"), -1),
        min(validation_mean, oos_mean),
        min(validation_pf, oos_pf),
        float(monthly.get("positive_period_ratio", 0.0) or 0.0),
        int(validation.get("candidate_rows", 0) or 0) + int(oos.get("candidate_rows", 0) or 0),
    )


def build_candidate_stability(candidate, feature_data, args, sweep_args, label_cache):
    started = time.perf_counter()
    with sweep.temporary_config_and_env(candidate["params"]):
        params_key = sweep.candidate_params_key(candidate)

        def build_frames():
            if params_key not in label_cache:
                label_cache[params_key] = sweep.build_edge_labeled(feature_data)
            return label_cache[params_key]

        if args.verbose_candidates:
            edge_labeled, split_config = build_frames()
            candidate_report = sweep.build_candidate_report(
                feature_data,
                sweep_args,
                candidate.get("entry_filter"),
                edge_labeled=edge_labeled,
                split_config=split_config,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                edge_labeled, split_config = build_frames()
                candidate_report = sweep.build_candidate_report(
                    feature_data,
                    sweep_args,
                    candidate.get("entry_filter"),
                    edge_labeled=edge_labeled,
                    split_config=split_config,
                )

        filtered, _ = sweep.apply_entry_filter(edge_labeled, candidate.get("entry_filter"))
        selected = filtered.tail(args.rows).copy() if args.rows and args.rows > 0 else filtered.copy()

    result = sweep.summarize_candidate_result(candidate, candidate_report, elapsed_sec=time.perf_counter() - started)
    split_metrics = split_direction_metrics(candidate_report, args.direction)
    period_stability = {
        "monthly": summarize_period_buckets(
            selected,
            args.direction,
            freq="M",
            min_rows=args.min_period_rows,
            min_profit_factor=args.min_profit_factor,
            min_mean_net_return=args.min_mean_net_return,
        ),
        "quarterly": summarize_period_buckets(
            selected,
            args.direction,
            freq="Q",
            min_rows=args.min_period_rows,
            min_profit_factor=args.min_profit_factor,
            min_mean_net_return=args.min_mean_net_return,
        ),
    }
    result["target_direction"] = args.direction
    result["target_direction_splits"] = split_metrics
    result["period_stability"] = period_stability
    result["market_state_breakdown"] = summarize_market_state_breakdown(
        selected,
        args.direction,
        min_rows=args.min_period_rows,
        min_profit_factor=args.min_profit_factor,
        min_mean_net_return=args.min_mean_net_return,
    )
    result["stability_decision"] = classify_stability(split_metrics, period_stability, args)
    return diag.json_safe(result)


def run_stability(args):
    sweep_args = build_sweep_args(args)
    candidates = sweep.build_candidates(sweep_args)
    if not candidates:
        raise RuntimeError("没有生成任何 breakout_flow 稳定性候选")

    with temporary_windows(args.windows) as active_windows:
        feature_data = diag.load_feature_data()
        feature_range = {
            "rows": int(len(feature_data)),
            "start": feature_data.index.min().isoformat() if len(feature_data) else None,
            "end": feature_data.index.max().isoformat() if len(feature_data) else None,
        }
        label_cache = {}
        results = []
        for index, candidate in enumerate(candidates, start=1):
            result = build_candidate_stability(candidate, feature_data, args, sweep_args, label_cache)
            results.append(result)
            if args.progress:
                split = result["target_direction_splits"]
                val = split["validation"]
                oos = split["oos"]
                log_info(
                    "breakout_flow稳定性 "
                    f"{index}/{len(candidates)} {candidate['name']} "
                    f"status={result['stability_decision']['status']} "
                    f"val={val['mean_net_return']:+.4%}/{val['profit_factor']} rows={val['candidate_rows']} "
                    f"oos={oos['mean_net_return']:+.4%}/{oos['profit_factor']} rows={oos['candidate_rows']}"
                )

    ranked = sorted(results, key=result_sort_key, reverse=True)
    return diag.json_safe({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "diagnostic": "breakout_flow_stability",
        "target_direction": args.direction,
        "feature_range": feature_range,
        "candidate_count": int(len(results)),
        "status_counts": diag.value_counts(pd.Series([item["stability_decision"]["status"] for item in results])),
        "settings": {
            "rows": int(args.rows),
            "windows": dict(active_windows),
            "trend_gaps": sweep.parse_float_list(args.trend_gaps),
            "regime_gap_multipliers": sweep.parse_float_list(args.regime_gap_multipliers),
            "tp_sl_pairs": sweep.parse_tp_sl_pairs(args.tp_sl_pairs),
            "allow_high_vol_values": sweep.parse_bool_list(args.allow_high_vol_values),
            "allow_range_values": sweep.parse_bool_list(args.allow_range_values),
            "breakout_lookbacks": sweep.parse_int_list(args.breakout_lookbacks),
            "flow_min_values": sweep.parse_float_list(args.flow_min_values),
            "volatility_min_values": sweep.parse_float_list(args.volatility_min_values),
            "trend_gap_min_values": sweep.parse_float_list(args.trend_gap_min_values),
            "max_candidates": int(args.max_candidates or 0),
            "min_split_rows": int(args.min_split_rows),
            "min_period_rows": int(args.min_period_rows),
            "min_active_periods": int(args.min_active_periods),
            "min_profit_factor": float(args.min_profit_factor),
            "min_mean_net_return": float(args.min_mean_net_return),
            "min_positive_period_ratio": float(args.min_positive_period_ratio),
            "top_n": int(args.top_n),
        },
        "top_candidates": ranked[: int(args.top_n)],
        "all_candidates": ranked,
    })


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"breakout_flow_stability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def print_candidate_line(item):
    split = item["target_direction_splits"]
    validation = split["validation"]
    oos = split["oos"]
    monthly = item["period_stability"]["monthly"]
    regime = item.get("market_state_breakdown", {}).get("by_regime", {})
    best_regime = regime.get("best_group") or {}
    worst_regime = regime.get("worst_group") or {}
    decision = item["stability_decision"]
    print(
        f"{item['name']} status={decision['status']} "
        f"val={validation['mean_net_return']:+.4%}/pf{validation['profit_factor']}/rows{validation['candidate_rows']} "
        f"oos={oos['mean_net_return']:+.4%}/pf{oos['profit_factor']}/rows{oos['candidate_rows']} "
        f"months={monthly['positive_period_count']}/{monthly['covered_period_count']} "
        f"best_regime={best_regime.get('group', '-')}:{float(best_regime.get('mean_net_return', 0.0) or 0.0):+.4%} "
        f"worst_regime={worst_regime.get('group', '-')}:{float(worst_regime.get('mean_net_return', 0.0) or 0.0):+.4%} "
        f"active={monthly['active_period_coverage']:.2%} "
        f"filter_pass={item['entry_filter_summary'].get('candidate_rows_after')}/{item['entry_filter_summary'].get('candidate_rows_before')}"
    )


def print_summary(report, path):
    log_info(
        "breakout_flow稳定性诊断完成: "
        f"rows={report.get('feature_range', {}).get('rows')} "
        f"range={report.get('feature_range', {}).get('start')}..{report.get('feature_range', {}).get('end')} "
        f"candidates={report['candidate_count']} status_counts={report['status_counts']} report={path}"
    )
    print("\n排名靠前的 breakout_flow 稳定性候选:")
    for item in report["top_candidates"]:
        print_candidate_line(item)
    print(f"\nreport_path={path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="聚焦 breakout_flow 入场过滤，按 validation/OOS 和月份/季度诊断 short edge 稳定性")
    parser.add_argument("--rows", type=int, default=int(os.getenv("BREAKOUT_FLOW_STABILITY_ROWS", "0")), help="仅使用最后 N 行；<=0 表示全量")
    parser.add_argument("--windows", default=os.getenv("BREAKOUT_FLOW_STABILITY_WINDOWS", ""), help="临时覆盖拉取窗口，例如 5m:30000,15m:10000,1H:3000")
    parser.add_argument("--direction", default=os.getenv("BREAKOUT_FLOW_STABILITY_DIRECTION", "short"), choices=["long", "short"], help="诊断方向")
    parser.add_argument("--trend-gaps", default=os.getenv("BREAKOUT_FLOW_STABILITY_TREND_GAPS", "0.0025,0.003,0.0035"), help="TREND_FILTER_MIN_GAP 候选")
    parser.add_argument("--regime-gap-multipliers", default=os.getenv("BREAKOUT_FLOW_STABILITY_REGIME_MULTIPLIERS", "1.0,1.5"), help="REGIME_TREND_GAP_THRESHOLD = trend_gap * multiplier")
    parser.add_argument("--tp-sl-pairs", default=os.getenv("BREAKOUT_FLOW_STABILITY_TP_SL", "0.010:0.008,0.012:0.010,0.014:0.010"), help="TP:SL 候选，逗号分隔")
    parser.add_argument("--allow-high-vol-values", default=os.getenv("BREAKOUT_FLOW_STABILITY_ALLOW_HIGH_VOL", "0"), help="是否允许 range_high_vol 候选")
    parser.add_argument("--allow-range-values", default=os.getenv("BREAKOUT_FLOW_STABILITY_ALLOW_RANGE", "1"), help="是否允许 range 候选")
    parser.add_argument("--breakout-lookbacks", default=os.getenv("BREAKOUT_FLOW_STABILITY_LOOKBACKS", "12,18,24,36"), help="breakout 前高/前低回看根数")
    parser.add_argument("--flow-min-values", default=os.getenv("BREAKOUT_FLOW_STABILITY_FLOW_MINS", "1.0,1.1,1.2,1.3"), help="money_flow_ratio 或 volume_ratio 最小值")
    parser.add_argument("--volatility-min-values", default=os.getenv("BREAKOUT_FLOW_STABILITY_VOLATILITY_MINS", ""), help="可选 volatility_15 最小值状态门槛")
    parser.add_argument("--trend-gap-min-values", default=os.getenv("BREAKOUT_FLOW_STABILITY_TREND_GAP_MINS", ""), help="可选 trend_gap_abs 最小值状态门槛")
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("BREAKOUT_FLOW_STABILITY_MAX_CANDIDATES", "72")), help=">0 时确定性抽样最多 N 个候选")
    parser.add_argument("--min-split-rows", type=int, default=int(os.getenv("BREAKOUT_FLOW_STABILITY_MIN_SPLIT_ROWS", "10")), help="validation/OOS 方向最少候选样本数")
    parser.add_argument("--min-period-rows", type=int, default=int(os.getenv("BREAKOUT_FLOW_STABILITY_MIN_PERIOD_ROWS", "4")), help="单月/单季度最少候选样本数")
    parser.add_argument("--min-active-periods", type=int, default=int(os.getenv("BREAKOUT_FLOW_STABILITY_MIN_ACTIVE_PERIODS", "3")), help="最少有效月份数")
    parser.add_argument("--min-profit-factor", type=float, default=float(os.getenv("BREAKOUT_FLOW_STABILITY_MIN_PF", "1.05")), help="正 edge 最低 PF")
    parser.add_argument("--min-mean-net-return", type=float, default=float(os.getenv("BREAKOUT_FLOW_STABILITY_MIN_MEAN", "0.0")), help="正 edge 最低平均净收益")
    parser.add_argument("--min-positive-period-ratio", type=float, default=float(os.getenv("BREAKOUT_FLOW_STABILITY_MIN_POSITIVE_PERIOD_RATIO", "0.50")), help="有效月份里正 edge 月份最低占比")
    parser.add_argument("--top-n", type=int, default=int(os.getenv("BREAKOUT_FLOW_STABILITY_TOP_N", "12")), help="打印和报告保留的前 N 个候选")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    parser.add_argument("--progress", action="store_true", help="逐候选打印进度")
    parser.add_argument("--verbose-candidates", action="store_true", help="不隐藏每个候选打标签日志")
    parser.add_argument("--print-json", action="store_true", help="同时打印完整 JSON 报告")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_stability(args)
    path = write_report(report, args.output)
    print_summary(report, path)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
