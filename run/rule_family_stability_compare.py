import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import rule_breakout_flow_stability as stability
from run import rule_edge_diagnostics as diag
from run import rule_edge_sweep as sweep
from utils.utils import LOGS_DIR, log_info


DEFAULT_FAMILIES = (
    "breakout_flow",
    "pullback_flow",
    "breakout",
    "failed_breakout",
    "failed_breakout_flow",
    "pullback_continuation",
    "pullback_continuation_flow",
)


def parse_name_list(raw_value):
    return sweep.parse_str_list(raw_value)


def build_family_sweep_args(args, family):
    return argparse.Namespace(
        rows=args.rows,
        trend_gaps=args.trend_gaps,
        regime_gap_multipliers=args.regime_gap_multipliers,
        tp_sl_pairs=args.tp_sl_pairs,
        allow_high_vol_values=args.allow_high_vol_values,
        allow_range_values=args.allow_range_values,
        entry_filters=family,
        pullback_pct_values=args.pullback_pct_values,
        breakout_lookbacks=args.breakout_lookbacks,
        failed_breakout_reclaim_pct_values=args.failed_breakout_reclaim_pct_values,
        flow_min_values=args.flow_min_values,
        low_vol_max_values=args.low_vol_max_values,
        volatility_min_values=args.volatility_min_values,
        trend_gap_min_values=args.trend_gap_min_values,
        max_candidates=args.max_candidates_per_family,
        min_rows=args.min_split_rows,
        min_profit_factor=args.min_profit_factor,
        min_mean_net_return=args.min_mean_net_return,
        top_n=args.top_n,
        output=args.output,
        progress=args.progress,
        verbose_candidates=args.verbose_candidates,
    )


def candidate_family(candidate):
    return sweep.normalize_entry_filter(candidate.get("entry_filter")).get("name", "none")


def candidate_status(item):
    return (item.get("stability_decision") or {}).get("status", "unknown")


def summarize_family_results(results):
    summaries = []
    for family, group in pd.Series(results, dtype=object).groupby(
        [item.get("rule_family", "unknown") for item in results],
        sort=True,
    ):
        group_items = list(group)
        ranked = sorted(group_items, key=stability.result_sort_key, reverse=True)
        status_counts = diag.value_counts(pd.Series([candidate_status(item) for item in group_items]))
        stable_count = int(status_counts.get("stable_positive", 0))
        weak_count = int(status_counts.get("weak_positive_low_pf_or_periods", 0))
        oos_only_count = int(status_counts.get("oos_only_unconfirmed", 0))
        best = ranked[0] if ranked else None
        summaries.append(diag.json_safe({
            "family": str(family),
            "candidate_count": int(len(group_items)),
            "status_counts": status_counts,
            "stable_positive_count": stable_count,
            "weak_positive_count": weak_count,
            "oos_only_unconfirmed_count": oos_only_count,
            "best_status": candidate_status(best or {}),
            "best_candidate": best,
        }))

    return sorted(
        summaries,
        key=lambda item: stability.result_sort_key(item.get("best_candidate") or {}),
        reverse=True,
    )


def build_direction_args(args, direction):
    direction_args = argparse.Namespace(**vars(args))
    direction_args.direction = direction
    return direction_args


def index_bound_iso(index, bound):
    if len(index) == 0:
        return None
    value = index.min() if bound == "min" else index.max()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def run_compare(args):
    families = parse_name_list(args.families)
    directions = parse_name_list(args.directions)
    if not families:
        raise RuntimeError("没有指定任何规则族")
    if not directions:
        raise RuntimeError("没有指定任何诊断方向")

    for family in families:
        if family not in sweep.ENTRY_FILTER_COMPONENTS:
            raise ValueError(f"未知规则族: {family!r}")
    for direction in directions:
        if direction not in {"long", "short"}:
            raise ValueError(f"未知方向: {direction!r}")

    with stability.temporary_windows(args.windows) as active_windows:
        feature_data = diag.load_feature_data()
        feature_range = {
            "rows": int(len(feature_data)),
            "start": index_bound_iso(feature_data.index, "min"),
            "end": index_bound_iso(feature_data.index, "max"),
        }
        label_cache = {}
        results = []
        planned = []
        for family in families:
            sweep_args = build_family_sweep_args(args, family)
            candidates = sweep.build_candidates(sweep_args)
            for candidate in candidates:
                planned.append((family, sweep_args, candidate))

        if not planned:
            raise RuntimeError("没有生成任何规则族候选")

        total = len(planned) * len(directions)
        completed = 0
        for family, sweep_args, candidate in planned:
            for direction in directions:
                completed += 1
                result = stability.build_candidate_stability(
                    candidate,
                    feature_data,
                    build_direction_args(args, direction),
                    sweep_args,
                    label_cache,
                )
                result["rule_family"] = family
                result["candidate_family"] = candidate_family(candidate)
                results.append(result)
                if args.progress:
                    split = result["target_direction_splits"]
                    validation = split["validation"]
                    oos = split["oos"]
                    monthly = result["period_stability"]["monthly"]
                    log_info(
                        "规则族稳定性 "
                        f"{completed}/{total} family={family} direction={direction} "
                        f"status={candidate_status(result)} "
                        f"val={validation['mean_net_return']:+.4%}/{validation['profit_factor']} rows={validation['candidate_rows']} "
                        f"oos={oos['mean_net_return']:+.4%}/{oos['profit_factor']} rows={oos['candidate_rows']} "
                        f"months={monthly['positive_period_count']}/{monthly['covered_period_count']}"
                    )

    ranked = sorted(results, key=stability.result_sort_key, reverse=True)
    stable = [item for item in ranked if candidate_status(item) == "stable_positive"]
    weak = [item for item in ranked if candidate_status(item) == "weak_positive_low_pf_or_periods"]
    return diag.json_safe({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "diagnostic": "rule_family_stability_compare",
        "feature_range": feature_range,
        "directions": directions,
        "families": families,
        "candidate_count": int(len(results)),
        "status_counts": diag.value_counts(pd.Series([candidate_status(item) for item in results])),
        "settings": {
            "rows": int(args.rows),
            "windows": dict(active_windows),
            "trend_gaps": sweep.parse_float_list(args.trend_gaps),
            "regime_gap_multipliers": sweep.parse_float_list(args.regime_gap_multipliers),
            "tp_sl_pairs": sweep.parse_tp_sl_pairs(args.tp_sl_pairs),
            "allow_high_vol_values": sweep.parse_bool_list(args.allow_high_vol_values),
            "allow_range_values": sweep.parse_bool_list(args.allow_range_values),
            "pullback_pct_values": sweep.parse_float_list(args.pullback_pct_values),
            "breakout_lookbacks": sweep.parse_int_list(args.breakout_lookbacks),
            "failed_breakout_reclaim_pct_values": sweep.parse_float_list(args.failed_breakout_reclaim_pct_values),
            "flow_min_values": sweep.parse_float_list(args.flow_min_values),
            "low_vol_max_values": sweep.parse_float_list(args.low_vol_max_values),
            "volatility_min_values": sweep.parse_float_list(args.volatility_min_values),
            "trend_gap_min_values": sweep.parse_float_list(args.trend_gap_min_values),
            "max_candidates_per_family": int(args.max_candidates_per_family or 0),
            "min_split_rows": int(args.min_split_rows),
            "min_period_rows": int(args.min_period_rows),
            "min_active_periods": int(args.min_active_periods),
            "min_profit_factor": float(args.min_profit_factor),
            "min_mean_net_return": float(args.min_mean_net_return),
            "min_positive_period_ratio": float(args.min_positive_period_ratio),
            "top_n": int(args.top_n),
        },
        "family_summaries": summarize_family_results(ranked),
        "stable_candidates": stable,
        "weak_candidates": weak,
        "top_candidates": ranked[: int(args.top_n)],
        "all_candidates": ranked,
    })


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"rule_family_stability_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
    print(
        f"{item['rule_family']}:{item['name']} dir={item['target_direction']} "
        f"status={candidate_status(item)} "
        f"val={validation['mean_net_return']:+.4%}/pf{validation['profit_factor']}/rows{validation['candidate_rows']} "
        f"oos={oos['mean_net_return']:+.4%}/pf{oos['profit_factor']}/rows{oos['candidate_rows']} "
        f"months={monthly['positive_period_count']}/{monthly['covered_period_count']} "
        f"filter_pass={item['entry_filter_summary'].get('candidate_rows_after')}/"
        f"{item['entry_filter_summary'].get('candidate_rows_before')}"
    )


def print_summary(report, path):
    log_info(
        "规则族稳定性对照完成: "
        f"rows={report.get('feature_range', {}).get('rows')} "
        f"range={report.get('feature_range', {}).get('start')}..{report.get('feature_range', {}).get('end')} "
        f"candidates={report['candidate_count']} status_counts={report['status_counts']} report={path}"
    )
    print("\n规则族概览:")
    for family in report["family_summaries"]:
        best = family.get("best_candidate") or {}
        print(
            f"{family['family']} candidates={family['candidate_count']} "
            f"status_counts={family['status_counts']} best={family['best_status']} "
            f"best_name={best.get('name', '-')}"
        )

    print("\n排名靠前的规则族候选:")
    for item in report["top_candidates"]:
        print_candidate_line(item)
    print(f"\nreport_path={path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="对照多个入场规则族的 validation/OOS/月度稳定性，筛掉只在单月有效的假 edge")
    parser.add_argument("--rows", type=int, default=int(os.getenv("RULE_FAMILY_STABILITY_ROWS", "0")), help="仅使用最后 N 行；<=0 表示全量")
    parser.add_argument("--windows", default=os.getenv("RULE_FAMILY_STABILITY_WINDOWS", ""), help="临时覆盖拉取窗口，例如 5m:30000,15m:10000,1H:3000")
    parser.add_argument("--families", default=os.getenv("RULE_FAMILY_STABILITY_FAMILIES", ",".join(DEFAULT_FAMILIES)), help="规则族/entry filter 名称，逗号分隔")
    parser.add_argument("--directions", default=os.getenv("RULE_FAMILY_STABILITY_DIRECTIONS", "short"), help="诊断方向，short,long")
    parser.add_argument("--trend-gaps", default=os.getenv("RULE_FAMILY_STABILITY_TREND_GAPS", "0.0025,0.003,0.0035"), help="TREND_FILTER_MIN_GAP 候选")
    parser.add_argument("--regime-gap-multipliers", default=os.getenv("RULE_FAMILY_STABILITY_REGIME_MULTIPLIERS", "1.0,1.5"), help="REGIME_TREND_GAP_THRESHOLD = trend_gap * multiplier")
    parser.add_argument("--tp-sl-pairs", default=os.getenv("RULE_FAMILY_STABILITY_TP_SL", "0.010:0.008,0.012:0.010,0.014:0.010"), help="TP:SL 候选，逗号分隔")
    parser.add_argument("--allow-high-vol-values", default=os.getenv("RULE_FAMILY_STABILITY_ALLOW_HIGH_VOL", "0"), help="是否允许 range_high_vol 候选")
    parser.add_argument("--allow-range-values", default=os.getenv("RULE_FAMILY_STABILITY_ALLOW_RANGE", "1"), help="是否允许 range 候选")
    parser.add_argument("--pullback-pct-values", default=os.getenv("RULE_FAMILY_STABILITY_PULLBACK_PCTS", "0.003,0.006"), help="pullback/continuation 回踩幅度候选")
    parser.add_argument("--breakout-lookbacks", default=os.getenv("RULE_FAMILY_STABILITY_LOOKBACKS", "12,18,24,36"), help="breakout/failed_breakout 前高前低回看根数")
    parser.add_argument("--failed-breakout-reclaim-pct-values", default=os.getenv("RULE_FAMILY_STABILITY_FAILED_BREAKOUT_RECLAIM_PCTS", "0,0.001"), help="failed_breakout 收回前高/前低后的最小内收比例候选")
    parser.add_argument("--flow-min-values", default=os.getenv("RULE_FAMILY_STABILITY_FLOW_MINS", "1.0,1.1,1.2,1.3"), help="money_flow_ratio 或 volume_ratio 最小值")
    parser.add_argument("--low-vol-max-values", default=os.getenv("RULE_FAMILY_STABILITY_LOW_VOL_MAXES", "0.003,0.005"), help="low_vol 最大 volatility_15")
    parser.add_argument("--volatility-min-values", default=os.getenv("RULE_FAMILY_STABILITY_VOLATILITY_MINS", ""), help="可选 volatility_15 最小值状态门槛")
    parser.add_argument("--trend-gap-min-values", default=os.getenv("RULE_FAMILY_STABILITY_TREND_GAP_MINS", ""), help="可选 trend_gap_abs 最小值状态门槛")
    parser.add_argument("--max-candidates-per-family", type=int, default=int(os.getenv("RULE_FAMILY_STABILITY_MAX_CANDIDATES_PER_FAMILY", "24")), help="每个规则族最多抽样候选数；<=0 表示全量")
    parser.add_argument("--min-split-rows", type=int, default=int(os.getenv("RULE_FAMILY_STABILITY_MIN_SPLIT_ROWS", "10")), help="validation/OOS 方向最少候选样本数")
    parser.add_argument("--min-period-rows", type=int, default=int(os.getenv("RULE_FAMILY_STABILITY_MIN_PERIOD_ROWS", "4")), help="单月/单季度最少候选样本数")
    parser.add_argument("--min-active-periods", type=int, default=int(os.getenv("RULE_FAMILY_STABILITY_MIN_ACTIVE_PERIODS", "3")), help="最少有效月份数")
    parser.add_argument("--min-profit-factor", type=float, default=float(os.getenv("RULE_FAMILY_STABILITY_MIN_PF", "1.05")), help="正 edge 最低 PF")
    parser.add_argument("--min-mean-net-return", type=float, default=float(os.getenv("RULE_FAMILY_STABILITY_MIN_MEAN", "0.0")), help="正 edge 最低平均净收益")
    parser.add_argument("--min-positive-period-ratio", type=float, default=float(os.getenv("RULE_FAMILY_STABILITY_MIN_POSITIVE_PERIOD_RATIO", "0.50")), help="有效月份里正 edge 月份最低占比")
    parser.add_argument("--top-n", type=int, default=int(os.getenv("RULE_FAMILY_STABILITY_TOP_N", "12")), help="打印和报告保留的前 N 个候选")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    parser.add_argument("--progress", action="store_true", help="逐候选打印进度")
    parser.add_argument("--verbose-candidates", action="store_true", help="不隐藏每个候选打标签日志")
    parser.add_argument("--print-json", action="store_true", help="同时打印完整 JSON 报告")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_compare(args)
    path = write_report(report, args.output)
    print_summary(report, path)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
