import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from run import rule_edge_diagnostics as diag
from train import train as train_module
from utils.utils import LOGS_DIR, log_info


ENTRY_FILTER_COMPONENTS = {
    "none": (),
    "pullback": ("pullback",),
    "breakout": ("breakout",),
    "flow": ("flow",),
    "low_vol": ("low_vol",),
    "pullback_flow": ("pullback", "flow"),
    "breakout_flow": ("breakout", "flow"),
    "low_vol_flow": ("low_vol", "flow"),
}

SWEEP_CONFIG_KEYS = {
    "TREND_FILTER_MIN_GAP",
    "REGIME_TREND_GAP_THRESHOLD",
    "REGIME_HIGH_VOL_ALLOW_TRADES",
    "REGIME_RANGE_ALLOW_TRADES",
    "MODEL_LABEL_TAKE_PROFIT",
    "MODEL_LABEL_STOP_LOSS",
}


@contextmanager
def temporary_config_and_env(overrides):
    original_config = {}
    original_env = {}
    try:
        for key, value in overrides.items():
            if hasattr(config, key):
                original_config[key] = getattr(config, key)
                setattr(config, key, value)
            original_env[key] = os.environ.get(key)
            os.environ[key] = _env_value(value)
        yield
    finally:
        for key, value in original_config.items():
            setattr(config, key, value)
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _env_value(value):
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def parse_float_list(raw_value):
    values = []
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def parse_int_list(raw_value):
    values = []
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def parse_str_list(raw_value):
    values = []
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append(item)
    return values


def parse_bool_list(raw_value):
    values = []
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        values.append(item in {"1", "true", "yes", "on"})
    return values


def parse_tp_sl_pairs(raw_value):
    pairs = []
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"TP/SL 候选格式应为 take_profit:stop_loss，实际为 {item!r}")
        take_profit, stop_loss = item.split(":", 1)
        pairs.append((float(take_profit), float(stop_loss)))
    return pairs


def pf_numeric(value):
    if value == "inf":
        return float("inf")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return float("inf") if value > 0 else 0.0
    return value


def _compact_float(value, digits=4):
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".").replace(".", "p")


def entry_filter_name(entry_filter):
    entry_filter = normalize_entry_filter(entry_filter)
    name = entry_filter["name"]
    params = entry_filter["params"]
    if name == "none":
        return "efnone"

    parts = [f"ef{name}"]
    if "pullback_pct" in params:
        parts.append(f"pb{_compact_float(params['pullback_pct'])}")
    if "breakout_lookback" in params:
        parts.append(f"bo{int(params['breakout_lookback'])}")
    if "flow_min" in params:
        parts.append(f"fl{_compact_float(params['flow_min'], digits=2)}")
    if "low_vol_max" in params:
        parts.append(f"lv{_compact_float(params['low_vol_max'])}")
    return "_".join(parts)


def candidate_name(params, entry_filter=None):
    hv = "hv1" if params["REGIME_HIGH_VOL_ALLOW_TRADES"] else "hv0"
    rg = "range1" if params["REGIME_RANGE_ALLOW_TRADES"] else "range0"
    base_name = (
        f"tg{params['TREND_FILTER_MIN_GAP']:.4f}_"
        f"rg{params['REGIME_TREND_GAP_THRESHOLD']:.4f}_"
        f"tp{params['MODEL_LABEL_TAKE_PROFIT']:.3f}_"
        f"sl{params['MODEL_LABEL_STOP_LOSS']:.3f}_"
        f"{hv}_{rg}"
    ).replace(".", "p")
    return f"{base_name}_{entry_filter_name(entry_filter)}"


def normalize_entry_filter(entry_filter):
    if not entry_filter:
        return {"name": "none", "params": {}}
    name = str(entry_filter.get("name") or "none").strip().lower()
    params = dict(entry_filter.get("params") or {})
    if name not in ENTRY_FILTER_COMPONENTS:
        raise ValueError(f"未知 entry filter: {name!r}")
    return {"name": name, "params": params}


def build_entry_filter_candidates(args):
    filter_names = parse_str_list(args.entry_filters)
    pullback_values = parse_float_list(args.pullback_pct_values)
    breakout_values = parse_int_list(args.breakout_lookbacks)
    flow_values = parse_float_list(args.flow_min_values)
    low_vol_values = parse_float_list(args.low_vol_max_values)

    filters = []
    for filter_name in filter_names:
        if filter_name not in ENTRY_FILTER_COMPONENTS:
            raise ValueError(f"未知 entry filter: {filter_name!r}")
        components = ENTRY_FILTER_COMPONENTS[filter_name]
        if not components:
            filters.append({"name": "none", "params": {}})
            continue

        pullback_grid = pullback_values if "pullback" in components else [None]
        breakout_grid = breakout_values if "breakout" in components else [None]
        flow_grid = flow_values if "flow" in components else [None]
        low_vol_grid = low_vol_values if "low_vol" in components else [None]

        for pullback_pct, breakout_lookback, flow_min, low_vol_max in product(
            pullback_grid,
            breakout_grid,
            flow_grid,
            low_vol_grid,
        ):
            params = {}
            if pullback_pct is not None:
                params["pullback_pct"] = float(pullback_pct)
            if breakout_lookback is not None:
                params["breakout_lookback"] = int(breakout_lookback)
            if flow_min is not None:
                params["flow_min"] = float(flow_min)
            if low_vol_max is not None:
                params["low_vol_max"] = float(low_vol_max)
            filters.append({"name": filter_name, "params": params})
    return filters


def _even_sample(items, limit):
    items = list(items)
    limit = int(limit or 0)
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]

    step = (len(items) - 1) / float(limit - 1)
    indexes = []
    seen = set()
    for i in range(limit):
        index = min(len(items) - 1, int(round(i * step)))
        if index not in seen:
            indexes.append(index)
            seen.add(index)
    for index in range(len(items)):
        if len(indexes) >= limit:
            break
        if index not in seen:
            indexes.append(index)
            seen.add(index)
    return [items[index] for index in sorted(indexes[:limit])]


def limit_candidates(candidates, max_candidates):
    max_candidates = int(max_candidates or 0)
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates

    groups = {}
    group_order = []
    for candidate in candidates:
        key = entry_filter_name(candidate.get("entry_filter"))
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(candidate)

    if len(group_order) <= 1:
        return _even_sample(candidates, max_candidates)

    if max_candidates < len(group_order):
        selected_groups = _even_sample(group_order, max_candidates)
        return [groups[key][len(groups[key]) // 2] for key in selected_groups]

    allocations = {key: 1 for key in group_order}
    remaining = max_candidates - len(group_order)
    while remaining > 0:
        added = False
        for key in group_order:
            if allocations[key] < len(groups[key]):
                allocations[key] += 1
                remaining -= 1
                added = True
                if remaining <= 0:
                    break
        if not added:
            break

    selected = []
    for key in group_order:
        selected.extend(_even_sample(groups[key], allocations[key]))
    return selected[:max_candidates]


def build_candidates(args):
    trend_gaps = parse_float_list(args.trend_gaps)
    regime_gap_multipliers = parse_float_list(args.regime_gap_multipliers)
    tp_sl_pairs = parse_tp_sl_pairs(args.tp_sl_pairs)
    allow_high_vol_values = parse_bool_list(args.allow_high_vol_values)
    allow_range_values = parse_bool_list(args.allow_range_values)
    entry_filters = build_entry_filter_candidates(args)

    candidates = []
    for trend_gap, regime_multiplier, (take_profit, stop_loss), allow_high_vol, allow_range, entry_filter in product(
        trend_gaps,
        regime_gap_multipliers,
        tp_sl_pairs,
        allow_high_vol_values,
        allow_range_values,
        entry_filters,
    ):
        params = {
            "TREND_FILTER_MIN_GAP": float(trend_gap),
            "REGIME_TREND_GAP_THRESHOLD": float(trend_gap * regime_multiplier),
            "REGIME_HIGH_VOL_ALLOW_TRADES": bool(allow_high_vol),
            "REGIME_RANGE_ALLOW_TRADES": bool(allow_range),
            "MODEL_LABEL_TAKE_PROFIT": float(take_profit),
            "MODEL_LABEL_STOP_LOSS": float(stop_loss),
        }
        candidates.append({
            "name": candidate_name(params, entry_filter),
            "params": params,
            "entry_filter": normalize_entry_filter(entry_filter),
        })

    return limit_candidates(candidates, args.max_candidates)


def numeric_series(data, column, default=np.nan):
    if column in data:
        return pd.to_numeric(data[column], errors="coerce")
    return pd.Series(default, index=data.index, dtype=float)


def _direction_values(data):
    return data.get("label_direction", pd.Series(index=data.index, dtype=object)).fillna("none").astype(str).str.lower()


def entry_filter_mask(data, entry_filter):
    entry_filter = normalize_entry_filter(entry_filter)
    filter_name = entry_filter["name"]
    components = ENTRY_FILTER_COMPONENTS[filter_name]
    directions = _direction_values(data)
    candidate_mask = directions.isin(diag.TRADE_DIRECTIONS)
    pass_mask = pd.Series(True, index=data.index, dtype=bool)
    if not components:
        return pass_mask

    close = numeric_series(data, "5m_close")
    high = numeric_series(data, "5m_high", default=np.nan).fillna(close)
    low = numeric_series(data, "5m_low", default=np.nan).fillna(close)
    params = entry_filter["params"]

    if "pullback" in components:
        pullback_pct = float(params.get("pullback_pct", 0.0))
        ema20 = numeric_series(data, "5m_ema_20")
        long_ok = close <= ema20 * (1.0 + pullback_pct)
        short_ok = close >= ema20 * (1.0 - pullback_pct)
        component_ok = ((directions == "long") & long_ok) | ((directions == "short") & short_ok)
        pass_mask &= (~candidate_mask) | component_ok.fillna(False)

    if "breakout" in components:
        lookback = max(1, int(params.get("breakout_lookback", 12)))
        prev_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
        prev_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
        long_ok = close >= prev_high
        short_ok = close <= prev_low
        component_ok = ((directions == "long") & long_ok) | ((directions == "short") & short_ok)
        pass_mask &= (~candidate_mask) | component_ok.fillna(False)

    if "flow" in components:
        flow_min = float(params.get("flow_min", 1.0))
        money_flow = numeric_series(data, "money_flow_ratio")
        volume_ratio = numeric_series(data, "volume_ratio")
        component_ok = (money_flow >= flow_min) | (volume_ratio >= flow_min)
        pass_mask &= (~candidate_mask) | component_ok.fillna(False)

    if "low_vol" in components:
        low_vol_max = float(params.get("low_vol_max", 0.0))
        volatility = numeric_series(data, "volatility_15")
        component_ok = volatility <= low_vol_max
        pass_mask &= (~candidate_mask) | component_ok.fillna(False)

    return pass_mask.astype(bool)


def _filter_summary_for_group(data, pass_mask):
    directions = _direction_values(data)
    candidate_mask = directions.isin(diag.TRADE_DIRECTIONS)
    before = int(candidate_mask.sum())
    after = int((candidate_mask & pass_mask).sum())
    return {
        "candidate_rows_before": before,
        "candidate_rows_after": after,
        "removed_rows": int(before - after),
        "pass_rate": float(after / before) if before else 0.0,
    }


def summarize_entry_filter(original_data, pass_mask, entry_filter):
    entry_filter = normalize_entry_filter(entry_filter)
    pass_mask = pass_mask.reindex(original_data.index).fillna(False).astype(bool)
    summary = {
        "name": entry_filter["name"],
        "params": entry_filter["params"],
        **_filter_summary_for_group(original_data, pass_mask),
        "by_direction": {},
        "by_split": {},
    }

    directions = _direction_values(original_data)
    for direction in ("long", "short"):
        group = original_data.loc[directions == direction]
        summary["by_direction"][direction] = _filter_summary_for_group(group, pass_mask.reindex(group.index))

    if "diagnostic_split" in original_data:
        for split_name, group in original_data.groupby("diagnostic_split", sort=True):
            summary["by_split"][str(split_name)] = _filter_summary_for_group(group, pass_mask.reindex(group.index))

    return diag.json_safe(summary)


def apply_entry_filter(edge_labeled, entry_filter):
    entry_filter = normalize_entry_filter(entry_filter)
    filtered = edge_labeled.copy()
    pass_mask = entry_filter_mask(filtered, entry_filter)
    candidate_mask = _direction_values(filtered).isin(diag.TRADE_DIRECTIONS)
    blocked_mask = candidate_mask & ~pass_mask
    if blocked_mask.any():
        filtered.loc[blocked_mask, "label_direction"] = "none"
        if "target" in filtered:
            filtered.loc[blocked_mask, "target"] = train_module.TARGET_NO_TRADE
        if "label_reject_reason" in filtered:
            filtered.loc[blocked_mask, "label_reject_reason"] = f"entry_filter_{entry_filter['name']}"
    filtered["entry_filter_name"] = entry_filter["name"]
    filtered["entry_filter_pass"] = np.where(candidate_mask, pass_mask, np.nan)
    return filtered, pass_mask


def build_edge_labeled(feature_data):
    with diag.temporary_env({"MODEL_LABEL_TIMEOUT_WEAK_POSITIVE_AS_TRADE": "1"}):
        edge_labeled = train_module.create_labels(
            feature_data.copy(),
            future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
            threshold=float(config.MODEL_LABEL_THRESHOLD),
        )
    return diag.add_split_column(edge_labeled)


def build_candidate_report(feature_data, args, entry_filter=None, edge_labeled=None, split_config=None):
    entry_filter = normalize_entry_filter(entry_filter)
    if edge_labeled is None or split_config is None:
        edge_labeled, split_config = build_edge_labeled(feature_data)
    filtered, pass_mask = apply_entry_filter(edge_labeled, entry_filter)
    selected_original = edge_labeled.tail(args.rows).copy() if args.rows and args.rows > 0 else edge_labeled.copy()
    selected = filtered.tail(args.rows).copy() if args.rows and args.rows > 0 else filtered.copy()
    selected_pass_mask = pass_mask.reindex(selected.index).fillna(False).astype(bool)

    report = {
        "split_config": split_config,
        "entry_filter": entry_filter,
        "entry_filter_summary": summarize_entry_filter(selected_original, selected_pass_mask, entry_filter),
        "splits": {},
    }
    for split_name in ("all", "train", "validation", "oos"):
        report["splits"][split_name] = diag.split_report(
            selected,
            strict_labeled=None,
            split_name=split_name,
            min_rows=int(args.min_rows),
            min_profit_factor=float(args.min_profit_factor),
            min_mean_net_return=float(args.min_mean_net_return),
        )
    report["decision"] = diag.final_decision(report)
    return diag.json_safe(report)


def metric_snapshot(item):
    rec = item.get("recommendation", {})
    return {
        "status": rec.get("status"),
        "action": rec.get("action"),
        "reason_codes": rec.get("reason_codes", []),
        "candidate_rows": int(item.get("candidate_rows", 0) or 0),
        "mean_net_return": float(item.get("mean_net_return", 0.0) or 0.0),
        "profit_factor": item.get("profit_factor", 0.0),
        "net_win_rate": float(item.get("net_win_rate", 0.0) or 0.0),
        "tp_rate": float(item.get("tp_rate", 0.0) or 0.0),
        "sl_rate": float(item.get("sl_rate", 0.0) or 0.0),
        "timeout_rate": float(item.get("timeout_rate", 0.0) or 0.0),
        "sum_net_return": float(item.get("sum_net_return", 0.0) or 0.0),
    }


def direction_score(validation_snapshot, oos_snapshot):
    validation_pf = pf_numeric(validation_snapshot.get("profit_factor"))
    oos_pf = pf_numeric(oos_snapshot.get("profit_factor"))
    validation_mean = float(validation_snapshot.get("mean_net_return", 0.0) or 0.0)
    oos_mean = float(oos_snapshot.get("mean_net_return", 0.0) or 0.0)
    return {
        "worst_profit_factor": min(validation_pf, oos_pf),
        "worst_mean_net_return": min(validation_mean, oos_mean),
        "combined_mean_net_return": float(validation_mean + oos_mean),
        "combined_candidate_rows": int(validation_snapshot.get("candidate_rows", 0) or 0)
        + int(oos_snapshot.get("candidate_rows", 0) or 0),
    }


def summarize_candidate_result(candidate, report, elapsed_sec=None):
    directions = {}
    passing_directions = []
    for direction in ("long", "short"):
        validation = metric_snapshot(report["splits"]["validation"]["by_direction"][direction])
        oos = metric_snapshot(report["splits"]["oos"]["by_direction"][direction])
        score = direction_score(validation, oos)
        passed = validation.get("status") == "positive_edge" and oos.get("status") == "positive_edge"
        if passed:
            passing_directions.append(direction)
        directions[direction] = {
            "passed": bool(passed),
            "validation": validation,
            "oos": oos,
            "score": score,
        }

    best_direction = sorted(
        directions,
        key=lambda item: (
            directions[item]["passed"],
            directions[item]["score"]["worst_mean_net_return"],
            directions[item]["score"]["worst_profit_factor"],
            directions[item]["score"]["combined_candidate_rows"],
        ),
        reverse=True,
    )[0]
    result = {
        "name": candidate["name"],
        "params": candidate["params"],
        "entry_filter": candidate.get("entry_filter", {"name": "none", "params": {}}),
        "entry_filter_summary": report.get("entry_filter_summary", {}),
        "passed": bool(passing_directions),
        "passing_directions": passing_directions,
        "best_direction": best_direction,
        "best_score": directions[best_direction]["score"],
        "directions": directions,
    }
    if elapsed_sec is not None:
        result["elapsed_sec"] = float(elapsed_sec)
    return diag.json_safe(result)


def candidate_sort_key(item):
    score = item.get("best_score", {})
    return (
        bool(item.get("passed")),
        float(score.get("worst_mean_net_return", 0.0) or 0.0),
        pf_numeric(score.get("worst_profit_factor", 0.0)),
        int(score.get("combined_candidate_rows", 0) or 0),
    )


def candidate_params_key(candidate):
    return json.dumps(candidate["params"], sort_keys=True, separators=(",", ":"))


def run_candidate(candidate, feature_data, args, label_cache=None):
    started = time.perf_counter()
    with temporary_config_and_env(candidate["params"]):
        params_key = candidate_params_key(candidate)

        def build_report_for_candidate():
            if label_cache is not None and params_key in label_cache:
                edge_labeled, split_config = label_cache[params_key]
            else:
                edge_labeled, split_config = build_edge_labeled(feature_data)
                if label_cache is not None:
                    label_cache[params_key] = (edge_labeled, split_config)
            return build_candidate_report(
                feature_data,
                args,
                candidate.get("entry_filter"),
                edge_labeled=edge_labeled,
                split_config=split_config,
            )

        if args.verbose_candidates:
            report = build_report_for_candidate()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                report = build_report_for_candidate()
    return summarize_candidate_result(candidate, report, elapsed_sec=time.perf_counter() - started)


def run_sweep(args):
    candidates = build_candidates(args)
    if not candidates:
        raise RuntimeError("没有生成任何 sweep 候选")

    feature_data = diag.load_feature_data()
    label_cache = {}
    results = []
    for index, candidate in enumerate(candidates, start=1):
        result = run_candidate(candidate, feature_data, args, label_cache=label_cache)
        results.append(result)
        if args.progress:
            log_info(
                "规则参数sweep "
                f"{index}/{len(candidates)} {candidate['name']} "
                f"passed={result['passing_directions']} "
                f"best={result['best_direction']} "
                f"worst_mean={result['best_score']['worst_mean_net_return']:+.4%} "
                f"worst_pf={result['best_score']['worst_profit_factor']}"
            )

    ranked = sorted(results, key=candidate_sort_key, reverse=True)
    positive = [item for item in ranked if item.get("passed")]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "diagnostic": "rule_edge_parameter_sweep",
        "candidate_count": int(len(results)),
        "positive_candidate_count": int(len(positive)),
        "settings": {
            "rows": args.rows,
            "min_rows": int(args.min_rows),
            "min_profit_factor": float(args.min_profit_factor),
            "min_mean_net_return": float(args.min_mean_net_return),
            "trend_gaps": parse_float_list(args.trend_gaps),
            "regime_gap_multipliers": parse_float_list(args.regime_gap_multipliers),
            "tp_sl_pairs": parse_tp_sl_pairs(args.tp_sl_pairs),
            "allow_high_vol_values": parse_bool_list(args.allow_high_vol_values),
            "allow_range_values": parse_bool_list(args.allow_range_values),
            "entry_filters": parse_str_list(args.entry_filters),
            "pullback_pct_values": parse_float_list(args.pullback_pct_values),
            "breakout_lookbacks": parse_int_list(args.breakout_lookbacks),
            "flow_min_values": parse_float_list(args.flow_min_values),
            "low_vol_max_values": parse_float_list(args.low_vol_max_values),
            "max_candidates": int(args.max_candidates or 0),
            "top_n": int(args.top_n),
        },
        "positive_candidates": positive,
        "top_candidates": ranked[: int(args.top_n)],
        "all_candidates": ranked,
    }
    return diag.json_safe(report)


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"rule_edge_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def print_candidate_line(item):
    direction = item["best_direction"]
    validation = item["directions"][direction]["validation"]
    oos = item["directions"][direction]["oos"]
    params = item["params"]
    entry_filter = normalize_entry_filter(item.get("entry_filter"))
    filter_summary = item.get("entry_filter_summary", {})
    print(
        f"{item['name']} dir={direction} passed={item['passed']} "
        f"val_mean={validation['mean_net_return']:+.4%} val_pf={validation['profit_factor']} "
        f"oos_mean={oos['mean_net_return']:+.4%} oos_pf={oos['profit_factor']} "
        f"rows={validation['candidate_rows']}/{oos['candidate_rows']} "
        f"filter={entry_filter_name(entry_filter)} "
        f"filter_pass={filter_summary.get('candidate_rows_after', 0)}/{filter_summary.get('candidate_rows_before', 0)} "
        f"tg={params['TREND_FILTER_MIN_GAP']:.4f} "
        f"rg={params['REGIME_TREND_GAP_THRESHOLD']:.4f} "
        f"tp/sl={params['MODEL_LABEL_TAKE_PROFIT']:.3f}/{params['MODEL_LABEL_STOP_LOSS']:.3f} "
        f"high_vol={params['REGIME_HIGH_VOL_ALLOW_TRADES']}"
    )


def print_summary(report, path):
    log_info(
        "规则参数sweep完成: "
        f"candidates={report['candidate_count']} "
        f"positive={report['positive_candidate_count']} "
        f"report={path}"
    )
    if report["positive_candidates"]:
        print("\npositive_candidates")
        for item in report["positive_candidates"][: report["settings"].get("top_n", 10)]:
            print_candidate_line(item)
    else:
        print("\n没有候选同时通过 validation + OOS 正期望门槛。排名靠前的候选:")
        for item in report["top_candidates"]:
            print_candidate_line(item)
    print(f"\nreport_path={path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="扫描基础 trend/regime + TP/SL + 入场过滤规则，寻找 validation 和 OOS 同时正期望的候选")
    parser.add_argument("--rows", type=int, default=int(os.getenv("RULE_EDGE_SWEEP_ROWS", "0")), help="仅使用最后 N 行；<=0 表示全量")
    parser.add_argument("--trend-gaps", default=os.getenv("RULE_EDGE_SWEEP_TREND_GAPS", "0.002,0.003"), help="TREND_FILTER_MIN_GAP 候选")
    parser.add_argument("--regime-gap-multipliers", default=os.getenv("RULE_EDGE_SWEEP_REGIME_GAP_MULTIPLIERS", "1.0,1.5"), help="REGIME_TREND_GAP_THRESHOLD = trend_gap * multiplier")
    parser.add_argument("--tp-sl-pairs", default=os.getenv("RULE_EDGE_SWEEP_TP_SL_PAIRS", "0.012:0.010,0.016:0.014,0.020:0.014"), help="TP:SL 候选，逗号分隔")
    parser.add_argument("--allow-high-vol-values", default=os.getenv("RULE_EDGE_SWEEP_ALLOW_HIGH_VOL", "0,1"), help="是否允许 range_high_vol 候选，0/1 逗号分隔")
    parser.add_argument("--allow-range-values", default=os.getenv("RULE_EDGE_SWEEP_ALLOW_RANGE", "1"), help="是否允许 range 候选，0/1 逗号分隔")
    parser.add_argument("--entry-filters", default=os.getenv("RULE_EDGE_SWEEP_ENTRY_FILTERS", "none,pullback,breakout,flow,pullback_flow,breakout_flow"), help="入场过滤器候选，逗号分隔")
    parser.add_argument("--pullback-pct-values", default=os.getenv("RULE_EDGE_SWEEP_PULLBACK_PCTS", "0.003,0.006"), help="pullback 最大偏离 5m_ema_20 候选")
    parser.add_argument("--breakout-lookbacks", default=os.getenv("RULE_EDGE_SWEEP_BREAKOUT_LOOKBACKS", "12,24"), help="breakout 前高/前低回看根数候选")
    parser.add_argument("--flow-min-values", default=os.getenv("RULE_EDGE_SWEEP_FLOW_MINS", "1.0,1.2"), help="money_flow_ratio 或 volume_ratio 最小值候选")
    parser.add_argument("--low-vol-max-values", default=os.getenv("RULE_EDGE_SWEEP_LOW_VOL_MAXES", "0.003,0.005"), help="volatility_15 最大值候选")
    parser.add_argument("--max-candidates", type=int, default=int(os.getenv("RULE_EDGE_SWEEP_MAX_CANDIDATES", "120")), help=">0 时确定性抽样最多 N 个候选")
    parser.add_argument("--min-rows", type=int, default=int(os.getenv("RULE_EDGE_MIN_ROWS", "100")), help="方向最少候选交易样本数")
    parser.add_argument("--min-profit-factor", type=float, default=float(os.getenv("RULE_EDGE_MIN_PROFIT_FACTOR", "1.05")), help="正 edge 最低 PF")
    parser.add_argument("--min-mean-net-return", type=float, default=float(os.getenv("RULE_EDGE_MIN_MEAN_NET_RETURN", "0.0")), help="正 edge 最低平均净收益")
    parser.add_argument("--top-n", type=int, default=int(os.getenv("RULE_EDGE_SWEEP_TOP_N", "10")), help="打印和报告保留的前 N 个候选")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    parser.add_argument("--progress", action="store_true", help="逐候选打印进度")
    parser.add_argument("--verbose-candidates", action="store_true", help="不隐藏每个候选打标签日志")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_sweep(args)
    path = write_report(report, args.output)
    print_summary(report, path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
