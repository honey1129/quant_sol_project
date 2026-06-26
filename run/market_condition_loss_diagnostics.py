import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.trade_audit import LIVE_FILLS_PATH, load_trade_records, safe_float, safe_optional_float
from utils.utils import DISPLAY_TIMEZONE, LOGS_DIR, log_info


GROUP_DIMENSIONS = (
    "side",
    "entry_regime",
    "entry_trend",
    "entry_side_regime",
    "entry_side_trend",
    "entry_alignment",
    "entry_hour",
    "entry_prob_bucket",
    "entry_volatility_bucket",
    "entry_atr_bucket",
    "entry_trend_gap_bucket",
    "entry_money_flow_bucket",
    "exit_regime",
    "regime_transition",
    "exit_reason",
)


def parse_timestamp(value):
    if value in ("", None):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def profit_factor(values):
    positives = sum(float(value) for value in values if float(value) > 0)
    negatives = sum(float(value) for value in values if float(value) < 0)
    if negatives < 0:
        return positives / abs(negatives)
    if positives > 0:
        return float("inf")
    return 0.0


def numeric_bucket(value, *, low, high, labels=("low", "mid", "high")):
    value = safe_optional_float(value)
    if value is None:
        return "unknown"
    if value < low:
        return labels[0]
    if value < high:
        return labels[1]
    return labels[2]


def prob_bucket(long_prob, short_prob):
    long_prob = safe_optional_float(long_prob)
    short_prob = safe_optional_float(short_prob)
    if long_prob is None or short_prob is None:
        return "unknown"
    gap = abs(long_prob - short_prob)
    dominant = max(long_prob, short_prob)
    if gap < 0.10:
        return "neutral"
    if dominant >= 0.85:
        return "very_strong"
    if dominant >= 0.70:
        return "strong"
    return "weak"


def display_hour(ts):
    if ts is None:
        return "unknown"
    return pd.Timestamp(ts).tz_convert(DISPLAY_TIMEZONE).strftime("%H")


def utc_hour(ts):
    if ts is None:
        return "unknown"
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%H")


def _first_value(*values, default="unknown"):
    for value in values:
        if value not in ("", None):
            return str(value)
    return default


def extract_condition(record):
    record = record if isinstance(record, dict) else {}
    signal = record.get("signal") if isinstance(record.get("signal"), dict) else {}
    decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    risk = record.get("risk_context") if isinstance(record.get("risk_context"), dict) else {}
    ts = parse_timestamp(record.get("executed_at") or record.get("bar_ts"))

    regime = _first_value(decision.get("market_regime"), signal.get("regime"))
    trend = _first_value(decision.get("trend_bias"), signal.get("trend_bias"), risk.get("trend_bias"))
    long_prob = safe_optional_float(signal.get("long_prob"))
    short_prob = safe_optional_float(signal.get("short_prob"))
    volatility = safe_optional_float(signal.get("volatility"))
    atr_ratio = safe_optional_float(signal.get("atr_ratio"))
    trend_gap = safe_optional_float(signal.get("trend_gap"))
    money_flow = safe_optional_float(signal.get("money_flow_ratio"))

    return {
        "ts": ts.isoformat() if ts else None,
        "hour_display": display_hour(ts),
        "hour_utc": utc_hour(ts),
        "regime": regime,
        "trend": trend,
        "long_prob": long_prob,
        "short_prob": short_prob,
        "prob_bucket": prob_bucket(long_prob, short_prob),
        "volatility": volatility,
        "volatility_bucket": numeric_bucket(volatility, low=0.002, high=0.004),
        "atr_ratio": atr_ratio,
        "atr_bucket": numeric_bucket(atr_ratio, low=0.003, high=0.006),
        "trend_gap": trend_gap,
        "trend_gap_bucket": numeric_bucket(
            abs(trend_gap) if trend_gap is not None else None,
            low=0.003,
            high=0.008,
            labels=("flat", "medium", "wide"),
        ),
        "money_flow_ratio": money_flow,
        "money_flow_bucket": numeric_bucket(
            money_flow,
            low=0.8,
            high=1.2,
            labels=("thin", "normal", "active"),
        ),
    }


def trade_alignment(side, trend):
    side = str(side or "").lower()
    trend = str(trend or "").lower()
    if trend in {"neutral", "range", "unknown", "none", ""}:
        return "neutral"
    if side == trend:
        return "aligned"
    if trend in {"long", "short"}:
        return "counter_trend"
    return "unknown"


def is_open_record(record):
    return str(record.get("action") or "").upper() == "OPEN"


def is_close_record(record):
    if safe_float(record.get("closed_qty"), 0.0) > 0:
        return True
    return str(record.get("action") or "").upper() == "CLOSE"


def build_completed_trades(records):
    parsed = []
    for record in records:
        ts = parse_timestamp(record.get("executed_at") or record.get("bar_ts"))
        if ts is None:
            continue
        item = dict(record)
        item["_ts"] = ts
        parsed.append(item)
    parsed.sort(key=lambda item: item["_ts"])

    open_by_side = defaultdict(list)
    trades = []
    for record in parsed:
        side = str(record.get("pos_side") or "unknown").lower()
        if is_open_record(record):
            open_by_side[side].append(record)
            continue
        if not is_close_record(record):
            continue

        entry = open_by_side[side].pop(0) if open_by_side[side] else None
        entry_condition = extract_condition(entry or record)
        exit_condition = extract_condition(record)
        entry_ts = parse_timestamp((entry or {}).get("executed_at") or (entry or {}).get("bar_ts"))
        exit_ts = parse_timestamp(record.get("executed_at") or record.get("bar_ts"))
        hold_minutes = None
        if entry_ts is not None and exit_ts is not None:
            hold_minutes = (exit_ts - entry_ts).total_seconds() / 60.0

        close_net = safe_float(record.get("net_realized_pnl"), 0.0)
        entry_net = safe_float((entry or {}).get("net_realized_pnl"), 0.0)
        net_pnl = close_net + entry_net
        close_fee = safe_float(record.get("fee_abs"), 0.0)
        entry_fee = safe_float((entry or {}).get("fee_abs"), 0.0)
        close_slippage = safe_float(record.get("slippage_value"), 0.0)
        entry_slippage = safe_float((entry or {}).get("slippage_value"), 0.0)

        trade = {
            "entry_time": entry_ts.isoformat() if entry_ts else None,
            "exit_time": exit_ts.isoformat() if exit_ts else None,
            "side": side,
            "paired": entry is not None,
            "entry_reason": (entry or {}).get("reason"),
            "exit_reason": str(record.get("reason") or "unknown"),
            "entry_price": safe_optional_float((entry or {}).get("fill_price")),
            "exit_price": safe_optional_float(record.get("fill_price")),
            "closed_qty": safe_float(record.get("closed_qty"), 0.0),
            "notional": safe_float(record.get("notional"), 0.0),
            "net_pnl": net_pnl,
            "close_net_pnl": close_net,
            "entry_net_pnl": entry_net,
            "fee_abs": entry_fee + close_fee,
            "slippage_value": entry_slippage + close_slippage,
            "hold_minutes": hold_minutes,
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
        }
        trade.update({
            "entry_regime": entry_condition["regime"],
            "entry_trend": entry_condition["trend"],
            "entry_side_regime": f"{side}:{entry_condition['regime']}",
            "entry_side_trend": f"{side}:{entry_condition['trend']}",
            "entry_alignment": trade_alignment(side, entry_condition["trend"]),
            "entry_hour": entry_condition["hour_display"],
            "entry_prob_bucket": entry_condition["prob_bucket"],
            "entry_volatility_bucket": entry_condition["volatility_bucket"],
            "entry_atr_bucket": entry_condition["atr_bucket"],
            "entry_trend_gap_bucket": entry_condition["trend_gap_bucket"],
            "entry_money_flow_bucket": entry_condition["money_flow_bucket"],
            "exit_regime": exit_condition["regime"],
            "regime_transition": f"{entry_condition['regime']}->{exit_condition['regime']}",
        })
        trades.append(trade)
    return trades


def filter_trades_by_window(trades, *, days=None, since=None):
    if since:
        cutoff = parse_timestamp(since)
    elif days and trades:
        latest = max(parse_timestamp(item.get("exit_time")) for item in trades if item.get("exit_time"))
        cutoff = latest - timedelta(days=float(days)) if latest else None
    else:
        cutoff = None
    if cutoff is None:
        return list(trades), None
    filtered = [
        item for item in trades
        if parse_timestamp(item.get("exit_time")) is not None and parse_timestamp(item.get("exit_time")) >= cutoff
    ]
    return filtered, cutoff


def summarize_trades(trades):
    pnls = [safe_float(item.get("net_pnl"), 0.0) for item in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    fees = [safe_float(item.get("fee_abs"), 0.0) for item in trades]
    slippage = [safe_float(item.get("slippage_value"), 0.0) for item in trades]
    count = len(trades)
    total_net = sum(pnls)
    return json_safe({
        "trade_count": count,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": len(wins) / count if count else 0.0,
        "net_pnl": total_net,
        "avg_net_pnl": total_net / count if count else 0.0,
        "median_net_pnl": float(pd.Series(pnls).median()) if pnls else 0.0,
        "profit_factor": profit_factor(pnls),
        "fee_abs": sum(fees),
        "slippage_value": sum(slippage),
        "avg_hold_minutes": sum(
            safe_float(item.get("hold_minutes"), 0.0) for item in trades if item.get("hold_minutes") is not None
        ) / sum(1 for item in trades if item.get("hold_minutes") is not None)
        if any(item.get("hold_minutes") is not None for item in trades)
        else None,
        "exit_reason_counts": dict(Counter(str(item.get("exit_reason") or "unknown") for item in trades)),
        "side_counts": dict(Counter(str(item.get("side") or "unknown") for item in trades)),
    })


def summarize_group(trades, dimension):
    groups = defaultdict(list)
    for trade in trades:
        groups[str(trade.get(dimension) or "unknown")].append(trade)
    rows = []
    for value, items in groups.items():
        summary = summarize_trades(items)
        summary["dimension"] = dimension
        summary["value"] = value
        rows.append(summary)
    return sorted(
        rows,
        key=lambda item: (
            float(item.get("net_pnl", 0.0) or 0.0),
            float(item.get("avg_net_pnl", 0.0) or 0.0),
            -int(item.get("trade_count", 0) or 0),
        ),
    )


def pf_numeric(value):
    if value == "inf":
        return float("inf")
    return safe_float(value, 0.0)


def classify_loss_condition(summary, args):
    trades = int(summary.get("trade_count", 0) or 0)
    net_pnl = float(summary.get("net_pnl", 0.0) or 0.0)
    avg_net = float(summary.get("avg_net_pnl", 0.0) or 0.0)
    win_rate = float(summary.get("win_rate", 0.0) or 0.0)
    pf = pf_numeric(summary.get("profit_factor", 0.0))
    reasons = []

    if trades < int(args.min_trades):
        return "watch_more_samples", ["insufficient_trades"]
    if net_pnl <= -float(args.min_total_loss):
        reasons.append("total_loss_exceeds_threshold")
    if avg_net <= -float(args.min_avg_loss):
        reasons.append("avg_loss_exceeds_threshold")
    if win_rate <= float(args.max_win_rate):
        reasons.append("low_win_rate")
    if pf < float(args.min_profit_factor):
        reasons.append("profit_factor_below_threshold")

    if "total_loss_exceeds_threshold" in reasons and "low_win_rate" in reasons:
        return "block_new_entries", reasons
    if "avg_loss_exceeds_threshold" in reasons and "profit_factor_below_threshold" in reasons:
        return "reduce_size_or_require_confirmation", reasons
    if net_pnl < 0:
        return "watch_or_reduce", reasons or ["negative_net_pnl"]
    return "allow", []


def filter_expression(dimension, value):
    if dimension == "side":
        return f"side == {value!r}"
    if dimension == "entry_regime":
        return f"entry_regime == {value!r}"
    if dimension == "entry_trend":
        return f"entry_trend == {value!r}"
    if dimension == "entry_side_regime" and ":" in value:
        side, regime = value.split(":", 1)
        return f"side == {side!r} and entry_regime == {regime!r}"
    if dimension == "entry_side_trend" and ":" in value:
        side, trend = value.split(":", 1)
        return f"side == {side!r} and entry_trend == {trend!r}"
    if dimension == "entry_alignment":
        return f"entry_alignment == {value!r}"
    if dimension == "entry_hour":
        return f"entry_hour_display == {value!r}"
    return f"{dimension} == {value!r}"


def build_recommendations(group_summaries, args):
    action_rank = {
        "block_new_entries": 3,
        "reduce_size_or_require_confirmation": 2,
        "watch_or_reduce": 1,
    }
    recommendations = []
    for dimension, rows in group_summaries.items():
        for summary in rows:
            action, reasons = classify_loss_condition(summary, args)
            if action not in action_rank:
                continue
            item = {
                "action": action,
                "dimension": dimension,
                "value": summary["value"],
                "filter_expression": filter_expression(dimension, summary["value"]),
                "reason_codes": reasons,
                "summary": summary,
            }
            recommendations.append(item)
    return sorted(
        recommendations,
        key=lambda item: (
            action_rank.get(item["action"], 0),
            -abs(float(item["summary"].get("net_pnl", 0.0) or 0.0)),
            int(item["summary"].get("trade_count", 0) or 0),
        ),
        reverse=True,
    )


def build_report(args):
    records = load_trade_records(args.records_path)
    completed = build_completed_trades(records)
    trades, cutoff = filter_trades_by_window(completed, days=args.days, since=args.since)
    dimensions = tuple(args.dimensions) if getattr(args, "dimensions", None) else GROUP_DIMENSIONS
    group_summaries = {
        dimension: summarize_group(trades, dimension)
        for dimension in dimensions
    }
    recommendations = build_recommendations(group_summaries, args)
    period_start = min((parse_timestamp(item.get("exit_time")) for item in trades if item.get("exit_time")), default=None)
    period_end = max((parse_timestamp(item.get("exit_time")) for item in trades if item.get("exit_time")), default=None)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "diagnostic": "market_condition_loss_diagnostics",
        "source_path": args.records_path,
        "record_count": len(records),
        "completed_trade_count_all": len(completed),
        "completed_trade_count": len(trades),
        "period": {
            "cutoff": cutoff.isoformat() if cutoff else None,
            "start": period_start.isoformat() if period_start else None,
            "end": period_end.isoformat() if period_end else None,
        },
        "settings": {
            "days": args.days,
            "since": args.since,
            "min_trades": int(args.min_trades),
            "min_total_loss": float(args.min_total_loss),
            "min_avg_loss": float(args.min_avg_loss),
            "max_win_rate": float(args.max_win_rate),
            "min_profit_factor": float(args.min_profit_factor),
            "top_n": int(args.top_n),
            "dimensions": list(dimensions),
        },
        "totals": summarize_trades(trades),
        "recommendations": recommendations[: int(args.top_n)],
        "all_recommendations": recommendations,
        "group_summaries": group_summaries,
        "recent_completed_trades": trades[-int(args.top_n):],
    }
    return json_safe(report)


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"market_condition_loss_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    return output_path


def format_markdown(report):
    totals = report.get("totals") or {}
    lines = [
        "# Market Condition Loss Diagnostics",
        "",
        "## Summary",
        "",
        f"- Completed trades: {report.get('completed_trade_count', 0)}",
        f"- Period: {report.get('period', {}).get('start')} .. {report.get('period', {}).get('end')}",
        f"- Net PnL: {safe_float(totals.get('net_pnl')):.2f} USDT",
        f"- Win rate: {safe_float(totals.get('win_rate')):.1%}",
        f"- Profit factor: {totals.get('profit_factor')}",
        f"- Fees: {safe_float(totals.get('fee_abs')):.2f} USDT",
        "",
        "## Recommendations",
        "",
        "| Action | Condition | Trades | Win Rate | Net PnL | Avg PnL | PF | Reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    recommendations = report.get("recommendations") or []
    if not recommendations:
        lines.append("| allow | No loss condition crossed thresholds | 0 | 0.0% | 0.00 | 0.00 | 0 | - |")
    for item in recommendations:
        summary = item.get("summary") or {}
        lines.append(
            "| "
            f"{item.get('action')} | `{item.get('filter_expression')}` | "
            f"{int(summary.get('trade_count', 0) or 0)} | "
            f"{safe_float(summary.get('win_rate')):.1%} | "
            f"{safe_float(summary.get('net_pnl')):.2f} | "
            f"{safe_float(summary.get('avg_net_pnl')):.2f} | "
            f"{summary.get('profit_factor')} | "
            f"{', '.join(item.get('reason_codes') or [])} |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(report, output_path=None, json_path=None):
    if output_path is None:
        if json_path:
            output_path = os.path.splitext(json_path)[0] + ".md"
        else:
            output_path = os.path.join(
                LOGS_DIR,
                f"market_condition_loss_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(format_markdown(report))
    return output_path


def print_summary(report, json_path, md_path):
    totals = report.get("totals") or {}
    log_info(
        "反向过滤诊断完成: "
        f"trades={report.get('completed_trade_count')} "
        f"net={safe_float(totals.get('net_pnl')):+.2f} "
        f"win_rate={safe_float(totals.get('win_rate')):.1%} "
        f"recommendations={len(report.get('recommendations') or [])} "
        f"json={json_path} markdown={md_path}"
    )
    print("\nTop loss-condition recommendations:")
    for item in report.get("recommendations", []):
        summary = item["summary"]
        print(
            f"{item['action']} {item['filter_expression']} "
            f"trades={summary['trade_count']} win={safe_float(summary['win_rate']):.1%} "
            f"net={safe_float(summary['net_pnl']):+.2f} avg={safe_float(summary['avg_net_pnl']):+.2f} "
            f"pf={summary['profit_factor']} reasons={','.join(item['reason_codes'])}"
        )
    print(f"\njson_path={json_path}")
    print(f"markdown_path={md_path}")


def parse_dimensions(raw_value):
    values = []
    for item in str(raw_value or "").replace("|", ",").split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Find market conditions where live completed trades lose money systematically")
    parser.add_argument("--records-path", default=LIVE_FILLS_PATH, help="live_fills.jsonl path")
    parser.add_argument("--days", type=float, default=float(os.getenv("MARKET_CONDITION_DIAG_DAYS", "7")), help="Use last N days relative to latest fill; ignored when --since is set")
    parser.add_argument("--since", default=None, help="UTC/display parseable timestamp lower bound for exit_time")
    parser.add_argument("--min-trades", type=int, default=int(os.getenv("MARKET_CONDITION_DIAG_MIN_TRADES", "3")), help="Minimum completed trades before a condition can be blocked")
    parser.add_argument("--min-total-loss", type=float, default=float(os.getenv("MARKET_CONDITION_DIAG_MIN_TOTAL_LOSS", "100")), help="Total loss threshold in USDT")
    parser.add_argument("--min-avg-loss", type=float, default=float(os.getenv("MARKET_CONDITION_DIAG_MIN_AVG_LOSS", "50")), help="Average loss threshold in USDT")
    parser.add_argument("--max-win-rate", type=float, default=float(os.getenv("MARKET_CONDITION_DIAG_MAX_WIN_RATE", "0.35")), help="Win-rate threshold for block recommendations")
    parser.add_argument("--min-profit-factor", type=float, default=float(os.getenv("MARKET_CONDITION_DIAG_MIN_PF", "0.80")), help="Profit-factor threshold for reduce recommendations")
    parser.add_argument("--top-n", type=int, default=int(os.getenv("MARKET_CONDITION_DIAG_TOP_N", "15")), help="Top recommendations/trades to print")
    parser.add_argument("--dimensions", type=parse_dimensions, default=None, help="Optional comma-separated dimensions to evaluate")
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument("--markdown-output", default=None, help="Markdown report path")
    parser.add_argument("--print-json", action="store_true", help="Print full JSON report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = build_report(args)
    json_path = write_report(report, args.output)
    md_path = write_markdown(report, args.markdown_output, json_path=json_path)
    print_summary(report, json_path, md_path)
    if args.print_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
