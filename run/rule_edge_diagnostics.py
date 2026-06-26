import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from core.ml_feature_engineering import add_advanced_features, merge_multi_period_features
from core.okx_api import OKXClient
from train import train as train_module
from utils.utils import LOGS_DIR, log_info


TRADE_DIRECTIONS = {"long", "short"}


@contextmanager
def temporary_env(overrides):
    original = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def value_counts(series):
    if series is None:
        return {}
    return {
        str(key): int(value)
        for key, value in series.fillna("unknown").astype(str).value_counts().sort_index().items()
    }


def quantiles(series):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {}
    return {
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def add_split_column(data):
    data = data.copy()
    train_end, validation_start, validation_end, oos_start = train_module.build_time_splits(len(data))
    data["diagnostic_split"] = "purge"
    data.iloc[:train_end, data.columns.get_loc("diagnostic_split")] = "train"
    data.iloc[validation_start:validation_end, data.columns.get_loc("diagnostic_split")] = "validation"
    data.iloc[oos_start:, data.columns.get_loc("diagnostic_split")] = "oos"
    return data, {
        "train_end": int(train_end),
        "validation_start": int(validation_start),
        "validation_end": int(validation_end),
        "oos_start": int(oos_start),
        "purge_bars": int(config.MODEL_PURGE_BARS),
    }


def profit_factor(net_returns):
    values = pd.to_numeric(net_returns, errors="coerce").dropna()
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(values[values < 0].sum())
    if gross_loss < 0:
        return float(gross_profit / abs(gross_loss))
    if gross_profit > 0:
        return float("inf")
    return 0.0


def direction_candidate_mask(data):
    return data["label_direction"].fillna("none").astype(str).str.lower().isin(TRADE_DIRECTIONS)


def tp_mask(data):
    return data["label_outcome"].fillna("").astype(str).str.upper().str.startswith("TP")


def summarize_rule_edge(group):
    group = group.copy()
    rows = int(len(group))
    if rows == 0:
        return {
            "rows": 0,
            "candidate_rows": 0,
            "status": "insufficient_rows",
        }

    candidate_mask = direction_candidate_mask(group)
    candidates = group.loc[candidate_mask].copy()
    candidate_rows = int(len(candidates))
    net_return = pd.to_numeric(candidates.get("label_net_return"), errors="coerce")
    gross_return = pd.to_numeric(candidates.get("label_gross_return"), errors="coerce")
    positive_rows = int((net_return > 0).sum())
    negative_rows = int((net_return < 0).sum())
    flat_rows = int((net_return == 0).sum())
    tp_rows = int(tp_mask(candidates).sum()) if candidate_rows else 0
    sl_rows = int((candidates.get("label_outcome", pd.Series(index=candidates.index, dtype=object)).astype(str).str.upper() == "SL").sum())
    timeout_rows = int(candidates.get("label_outcome", pd.Series(index=candidates.index, dtype=object)).astype(str).str.upper().str.startswith("TIMEOUT").sum())

    return json_safe({
        "rows": rows,
        "candidate_rows": candidate_rows,
        "candidate_pct": float(candidate_rows / rows * 100.0) if rows else 0.0,
        "positive_net_rows": positive_rows,
        "negative_net_rows": negative_rows,
        "flat_net_rows": flat_rows,
        "net_win_rate": float(positive_rows / candidate_rows) if candidate_rows else 0.0,
        "tp_rows": tp_rows,
        "sl_rows": sl_rows,
        "timeout_rows": timeout_rows,
        "tp_rate": float(tp_rows / candidate_rows) if candidate_rows else 0.0,
        "sl_rate": float(sl_rows / candidate_rows) if candidate_rows else 0.0,
        "timeout_rate": float(timeout_rows / candidate_rows) if candidate_rows else 0.0,
        "mean_gross_return": float(gross_return.mean()) if candidate_rows else 0.0,
        "mean_net_return": float(net_return.mean()) if candidate_rows else 0.0,
        "median_net_return": float(net_return.median()) if candidate_rows else 0.0,
        "sum_net_return": float(net_return.sum()) if candidate_rows else 0.0,
        "profit_factor": profit_factor(net_return),
        "net_return_quantiles": quantiles(net_return),
        "mfe_quantiles": quantiles(candidates.get("label_mfe")),
        "mae_quantiles": quantiles(candidates.get("label_mae")),
        "mae_ratio_quantiles": quantiles(candidates.get("label_mae_ratio")),
        "mfe_mae_ratio_quantiles": quantiles(candidates.get("label_mfe_mae_ratio")),
        "direction_counts": value_counts(candidates.get("label_direction")),
        "regime_counts": value_counts(candidates.get("label_regime")),
        "outcome_counts": value_counts(candidates.get("label_outcome")),
        "reject_reason_counts": value_counts(candidates.get("label_reject_reason")),
    })


def quality_label_summary(group, strict_labeled):
    if strict_labeled is None or group.empty:
        return {
            "strict_label_rows": 0,
            "strict_trade_rows": 0,
            "strict_trade_pct": 0.0,
            "ignored_or_missing_rows": int(len(group)),
        }
    strict = strict_labeled.reindex(group.index)
    target = pd.to_numeric(strict.get("target"), errors="coerce")
    valid = target.notna()
    trade_rows = int((target.loc[valid].astype(int) == train_module.TARGET_TRADE).sum())
    valid_rows = int(valid.sum())
    return {
        "strict_label_rows": valid_rows,
        "strict_trade_rows": trade_rows,
        "strict_trade_pct": float(trade_rows / valid_rows * 100.0) if valid_rows else 0.0,
        "ignored_or_missing_rows": int(len(group) - valid_rows),
    }


def classify_edge(summary, *, min_rows, min_profit_factor, min_mean_net_return):
    candidate_rows = int(summary.get("candidate_rows", 0) or 0)
    mean_net = float(summary.get("mean_net_return", 0.0) or 0.0)
    pf_value = summary.get("profit_factor", 0.0)
    pf_numeric = float("inf") if pf_value == "inf" else float(pf_value or 0.0)
    reason_codes = []

    if candidate_rows < min_rows:
        reason_codes.append("insufficient_candidate_rows")
    if mean_net <= min_mean_net_return:
        reason_codes.append("non_positive_mean_net_return")
    if pf_numeric < min_profit_factor:
        reason_codes.append("profit_factor_below_threshold")

    if "insufficient_candidate_rows" in reason_codes:
        status = "insufficient_rows"
        action = "collect_more_samples"
    elif reason_codes:
        status = "no_edge"
        action = "disable_or_rework_rule"
    else:
        status = "positive_edge"
        action = "eligible_for_ml_quality_filter"

    return {
        "status": status,
        "action": action,
        "reason_codes": reason_codes,
        "thresholds": {
            "min_rows": int(min_rows),
            "min_profit_factor": float(min_profit_factor),
            "min_mean_net_return": float(min_mean_net_return),
        },
    }


def summarize_group(group, strict_labeled, *, min_rows, min_profit_factor, min_mean_net_return):
    summary = summarize_rule_edge(group)
    summary["quality_label_summary"] = quality_label_summary(group.loc[direction_candidate_mask(group)], strict_labeled)
    summary["recommendation"] = classify_edge(
        summary,
        min_rows=min_rows,
        min_profit_factor=min_profit_factor,
        min_mean_net_return=min_mean_net_return,
    )
    return json_safe(summary)


def split_report(edge_labeled, strict_labeled, split_name, *, min_rows, min_profit_factor, min_mean_net_return):
    if split_name == "all":
        split_data = edge_labeled
    else:
        split_data = edge_labeled[edge_labeled["diagnostic_split"] == split_name]

    report = {
        "summary": summarize_group(
            split_data,
            strict_labeled,
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        ),
        "by_direction": {},
        "by_direction_regime": {},
    }

    direction_values = split_data.get("label_direction", pd.Series(index=split_data.index, dtype=object)).fillna("none").astype(str).str.lower()
    for direction in ("long", "short"):
        direction_group = split_data.loc[direction_values == direction]
        report["by_direction"][direction] = summarize_group(
            direction_group,
            strict_labeled,
            min_rows=min_rows,
            min_profit_factor=min_profit_factor,
            min_mean_net_return=min_mean_net_return,
        )

    candidate_data = split_data.loc[direction_candidate_mask(split_data)].copy()
    if not candidate_data.empty:
        for (direction, regime), group in candidate_data.groupby(["label_direction", "label_regime"], dropna=False, sort=True):
            key = f"{str(direction).lower()}:{str(regime).lower()}"
            report["by_direction_regime"][key] = summarize_group(
                group,
                strict_labeled,
                min_rows=min_rows,
                min_profit_factor=min_profit_factor,
                min_mean_net_return=min_mean_net_return,
            )

    return json_safe(report)


def final_decision(report):
    validation = report["splits"].get("validation", {})
    oos = report["splits"].get("oos", {})
    decisions = {}
    for direction in ("long", "short"):
        validation_status = (
            validation.get("by_direction", {})
            .get(direction, {})
            .get("recommendation", {})
            .get("status")
        )
        oos_status = (
            oos.get("by_direction", {})
            .get(direction, {})
            .get("recommendation", {})
            .get("status")
        )
        if validation_status == "positive_edge" and oos_status == "positive_edge":
            action = "keep_direction_and_train_quality_filter"
            status = "positive_edge"
        elif validation_status == "insufficient_rows" or oos_status == "insufficient_rows":
            action = "do_not_promote_until_more_samples"
            status = "inconclusive"
        else:
            action = "disable_direction_or_rework_base_rule_before_ml"
            status = "no_confirmed_edge"
        decisions[direction] = {
            "status": status,
            "action": action,
            "validation_status": validation_status,
            "oos_status": oos_status,
        }

    positive_directions = [
        direction for direction, item in decisions.items()
        if item["status"] == "positive_edge"
    ]
    if not positive_directions:
        overall_action = "stop_ml_tuning_and_rework_base_strategy"
    elif positive_directions == ["short"]:
        overall_action = "test_short_only_ml_quality_filter"
    elif positive_directions == ["long"]:
        overall_action = "test_long_only_ml_quality_filter"
    else:
        overall_action = "train_direction_specific_quality_filters"

    return {
        "overall_action": overall_action,
        "positive_directions": positive_directions,
        "directions": decisions,
    }


def build_labeled_frames(feature_data):
    strict_labeled = train_module.create_labels(
        feature_data.copy(),
        future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
        threshold=float(config.MODEL_LABEL_THRESHOLD),
    )
    with temporary_env({"MODEL_LABEL_TIMEOUT_WEAK_POSITIVE_AS_TRADE": "1"}):
        edge_labeled = train_module.create_labels(
            feature_data.copy(),
            future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
            threshold=float(config.MODEL_LABEL_THRESHOLD),
        )
    edge_labeled, split_config = add_split_column(edge_labeled)
    return edge_labeled, strict_labeled, split_config


def load_feature_data():
    client = OKXClient()
    data_dict = client.fetch_data()
    merged_df = merge_multi_period_features(data_dict)
    return add_advanced_features(merged_df).dropna().copy()


def build_report(args, feature_data=None):
    feature_data = load_feature_data() if feature_data is None else feature_data.copy()
    edge_labeled, strict_labeled, split_config = build_labeled_frames(feature_data)
    selected = edge_labeled.tail(args.rows).copy() if args.rows and args.rows > 0 else edge_labeled.copy()
    strict_selected = strict_labeled.reindex(selected.index)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "diagnostic": "rule_edge_without_ml",
        "note": (
            "Candidate-level ex-post rule edge. It opens a hypothetical trade at every "
            "rule-allowed bar and does not model overlapping positions or cooldowns."
        ),
        "rows": int(len(selected)),
        "start": selected.index.min().isoformat() if len(selected) else None,
        "end": selected.index.max().isoformat() if len(selected) else None,
        "config": {
            "label_lookahead_bars": int(train_module._label_lookahead_bars()),
            "label_take_profit": float(train_module._label_take_profit()),
            "label_stop_loss": float(train_module._label_stop_loss()),
            "round_trip_cost": float(train_module._round_trip_cost_ratio()),
            "timeout_min_net_return": float(train_module._label_timeout_min_net_return()),
            "timeout_max_mae_ratio": float(train_module._label_timeout_max_mae_ratio()),
            "long_trend_strong_max_exit_bars": int(train_module._label_long_trend_strong_max_exit_bars()),
            "long_trend_strong_max_mae_ratio": float(train_module._label_long_trend_strong_max_mae_ratio()),
        },
        "split_config": split_config,
        "strict_label_quality_summary": strict_labeled.attrs.get("label_quality_summary", {}),
        "strict_label_filter_summary": strict_labeled.attrs.get("label_filter_summary", {}),
        "edge_label_quality_summary": edge_labeled.attrs.get("label_quality_summary", {}),
        "splits": {},
    }

    for split_name in ("all", "train", "validation", "oos"):
        report["splits"][split_name] = split_report(
            selected,
            strict_selected,
            split_name,
            min_rows=int(args.min_rows),
            min_profit_factor=float(args.min_profit_factor),
            min_mean_net_return=float(args.min_mean_net_return),
        )
    report["decision"] = final_decision(report)
    return json_safe(report)


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"rule_edge_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def print_summary(report, path):
    decision = report.get("decision", {})
    log_info(
        "规则基线诊断完成: "
        f"rows={report.get('rows')} range={report.get('start')}..{report.get('end')} "
        f"decision={decision.get('overall_action')}"
    )
    for split_name in ("validation", "oos"):
        split = report["splits"][split_name]
        log_info(f"{split_name} 规则基线:")
        for direction in ("long", "short"):
            item = split["by_direction"][direction]
            rec = item.get("recommendation", {})
            log_info(
                f"  {direction}: status={rec.get('status')} "
                f"rows={item.get('candidate_rows')} "
                f"mean_net={item.get('mean_net_return', 0.0):+.4%} "
                f"pf={item.get('profit_factor')} "
                f"net_win={item.get('net_win_rate', 0.0):.2%} "
                f"tp={item.get('tp_rate', 0.0):.2%} "
                f"action={rec.get('action')}"
            )
    log_info(f"规则基线诊断报告: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="不用 ML，诊断 trend/regime 基础规则本身是否有交易 edge")
    parser.add_argument("--rows", type=int, default=int(os.getenv("RULE_EDGE_DIAG_ROWS", "0")), help="仅使用最后 N 行；<=0 表示全量")
    parser.add_argument("--min-rows", type=int, default=int(os.getenv("RULE_EDGE_MIN_ROWS", "30")), help="方向/分组最少候选交易样本数")
    parser.add_argument("--min-profit-factor", type=float, default=float(os.getenv("RULE_EDGE_MIN_PROFIT_FACTOR", "1.05")), help="判定正 edge 的最低 PF")
    parser.add_argument("--min-mean-net-return", type=float, default=float(os.getenv("RULE_EDGE_MIN_MEAN_NET_RETURN", "0.0")), help="判定正 edge 的最低平均净收益")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args)
    path = write_report(report, args.output)
    print_summary(report, path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
