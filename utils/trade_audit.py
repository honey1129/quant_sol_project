import json
import math
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from config import config
from utils.utils import DISPLAY_TIMEZONE, LOGS_DIR


LIVE_FILLS_PATH = os.path.join(LOGS_DIR, "live_fills.jsonl")
DAILY_REPORT_DIR = os.path.join(LOGS_DIR, "daily_reports")
LATEST_DAILY_REPORT_PATH = os.path.join(LOGS_DIR, "daily_report_latest.md")


def safe_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def safe_optional_float(value):
    try:
        if value in ("", None):
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def normalize_ts(value=None):
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def normalize_event_ts(value):
    if value in ("", None):
        return None
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            return pd.to_datetime(float(value), unit="ms", utc=True).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None
    return normalize_ts(value)


def elapsed_ms(start, end):
    start_ts = normalize_event_ts(start)
    end_ts = normalize_event_ts(end)
    if start_ts is None or end_ts is None:
        return None
    try:
        value = (pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds() * 1000.0
    except Exception:
        return None
    return float(value) if value >= 0 else None


def display_date(value):
    ts = pd.Timestamp(normalize_ts(value))
    return ts.tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d")


def _extract_timestamp(order, fills, fallback_ts):
    candidates = []
    for source in [order] + list(fills):
        for key in ("fillTime", "uTime", "cTime", "ts"):
            raw = source.get(key) if isinstance(source, dict) else None
            if raw not in ("", None):
                candidates.append(raw)

    for raw in candidates:
        try:
            normalized = normalize_event_ts(raw)
            if normalized is not None:
                return normalized
        except Exception:
            continue
    return normalize_ts(fallback_ts)


def _aggregate_fills(order):
    order = order if isinstance(order, dict) else {}
    fills = order.get("_fills") or []

    fill_size = 0.0
    notional = 0.0
    fee_signed = 0.0
    fee_found = False
    fee_currency = None

    for fill in fills:
        size = abs(safe_float(fill.get("fillSz"), 0.0))
        price = safe_float(fill.get("fillPx"), 0.0)
        if size > 0 and price > 0:
            fill_size += size
            notional += size * price
        if fill.get("fee") not in ("", None):
            fee_signed += safe_float(fill.get("fee"), 0.0)
            fee_found = True
        if not fee_currency and fill.get("feeCcy"):
            fee_currency = fill.get("feeCcy")

    if fill_size <= 0:
        fill_size = abs(safe_float(order.get("accFillSz"), 0.0))
    if fill_size <= 0:
        fill_size = abs(safe_float(order.get("fillSz"), 0.0))
    if fill_size <= 0:
        fill_size = abs(safe_float(order.get("sz"), 0.0))

    fill_price = None
    if notional > 0 and fill_size > 0:
        fill_price = notional / fill_size
    else:
        fill_price = safe_optional_float(order.get("avgPx"))
        if fill_price is None:
            fill_price = safe_optional_float(order.get("fillPx"))

    if fill_price is not None and notional <= 0 and fill_size > 0:
        notional = fill_size * fill_price

    if not fee_found and order.get("fee") not in ("", None):
        fee_signed = safe_float(order.get("fee"), 0.0)
        fee_found = True
        fee_currency = fee_currency or order.get("feeCcy")

    fee_source = "exchange"
    if not fee_found and notional > 0:
        fee_signed = -abs(notional * float(config.FEE_RATE))
        fee_source = "estimated"
    elif not fee_found:
        fee_source = "not_recorded"

    return {
        "fills": fills,
        "fill_size": float(fill_size),
        "fill_price": fill_price,
        "notional": float(notional),
        "fee_signed": float(fee_signed),
        "fee_abs": abs(float(fee_signed)),
        "fee_currency": fee_currency or "USDT",
        "fee_source": fee_source,
    }


def _infer_order_side(order, delta_qty):
    side = str((order or {}).get("side", "") or "").lower()
    if side in {"buy", "sell"}:
        return side
    return "buy" if float(delta_qty or 0.0) > 0 else "sell"


def _infer_pos_side(order, pos_qty_before, delta_qty):
    pos_side = str((order or {}).get("posSide", "") or "").lower()
    if pos_side in {"long", "short"}:
        return pos_side
    if pos_qty_before > 0 or delta_qty > 0:
        return "long"
    return "short"


def _calculate_realized_pnl(action, delta_qty, pos_qty_before, entry_price_before, fill_price, fill_size):
    if fill_price is None or entry_price_before <= 0 or pos_qty_before == 0:
        return 0.0, 0.0

    action = str(action or "").upper()
    max_closed_qty = abs(fill_size) if fill_size and fill_size > 0 else abs(delta_qty)
    closed_qty = 0.0
    if action == "CLOSE":
        closed_qty = min(abs(pos_qty_before), abs(delta_qty), max_closed_qty)
    elif action == "REBALANCE" and (delta_qty * pos_qty_before) < 0:
        closed_qty = min(abs(delta_qty), abs(pos_qty_before), max_closed_qty)

    if closed_qty <= 0:
        return 0.0, 0.0

    signed_closed_pos = math.copysign(closed_qty, pos_qty_before)
    gross_realized_pnl = (fill_price - entry_price_before) * signed_closed_pos
    return closed_qty, float(gross_realized_pnl)


def _calculate_slippage(side, fill_size, fill_price, reference_price):
    if fill_price is None or reference_price is None or fill_size <= 0 or reference_price <= 0:
        return None, None
    if side == "buy":
        slippage_value = (fill_price - reference_price) * fill_size
    else:
        slippage_value = (reference_price - fill_price) * fill_size
    slippage_bps = (slippage_value / max(fill_size * reference_price, 1e-12)) * 10000.0
    return float(slippage_value), float(slippage_bps)


def build_trade_record(
    order,
    *,
    bar_ts,
    action,
    reason,
    delta_qty,
    reference_price,
    pos_qty_before,
    entry_price_before,
    pos_qty_after,
    entry_price_after,
    account_before,
    account_after,
    signal_snapshot,
    decision,
    execution_context=None,
):
    order = order if isinstance(order, dict) else {}
    fills = order.get("_fills") or []
    aggregated = _aggregate_fills(order)
    fill_price = aggregated["fill_price"]
    fill_size = aggregated["fill_size"]
    side = _infer_order_side(order, delta_qty)
    pos_side = _infer_pos_side(order, pos_qty_before, delta_qty)
    fee_signed = aggregated["fee_signed"]
    closed_qty, gross_realized_pnl = _calculate_realized_pnl(
        action,
        float(delta_qty),
        float(pos_qty_before),
        float(entry_price_before),
        fill_price,
        fill_size,
    )
    net_realized_pnl = gross_realized_pnl + fee_signed
    slippage_value, slippage_bps = _calculate_slippage(
        side,
        fill_size,
        fill_price,
        safe_optional_float(reference_price),
    )

    before_eq = safe_optional_float((account_before or {}).get("total_eq"))
    after_eq = safe_optional_float((account_after or {}).get("total_eq"))
    before_avail_eq = safe_optional_float((account_before or {}).get("avail_eq"))
    after_avail_eq = safe_optional_float((account_after or {}).get("avail_eq"))
    before_sizing_eq = safe_optional_float((account_before or {}).get("sizing_eq"))
    after_sizing_eq = safe_optional_float((account_after or {}).get("sizing_eq"))
    before_equity_usdt = safe_optional_float((account_before or {}).get("equity_usdt"))
    after_equity_usdt = safe_optional_float((account_after or {}).get("equity_usdt"))
    before_cash_balance_usdt = safe_optional_float((account_before or {}).get("cash_balance_usdt"))
    after_cash_balance_usdt = safe_optional_float((account_after or {}).get("cash_balance_usdt"))
    equity_delta = None
    if before_eq is not None and after_eq is not None:
        equity_delta = after_eq - before_eq
    equity_usdt_delta = None
    if before_equity_usdt is not None and after_equity_usdt is not None:
        equity_usdt_delta = after_equity_usdt - before_equity_usdt

    executed_at = _extract_timestamp(order, fills, bar_ts)
    execution_context = execution_context if isinstance(execution_context, dict) else {}
    trigger_detected_at = normalize_event_ts(execution_context.get("trigger_detected_at"))
    trigger_price = safe_optional_float(execution_context.get("trigger_price"))
    threshold_price = safe_optional_float(execution_context.get("threshold_price"))
    order_round_trip_ms = safe_optional_float(execution_context.get("order_round_trip_ms"))
    if order_round_trip_ms is not None and order_round_trip_ms < 0:
        order_round_trip_ms = None

    _, detection_slippage_bps = _calculate_slippage(
        side,
        fill_size,
        trigger_price,
        threshold_price,
    )
    _, threshold_to_fill_slippage_bps = _calculate_slippage(
        side,
        fill_size,
        fill_price,
        threshold_price,
    )
    execution_quality = {
        "trigger_source": execution_context.get("trigger_source"),
        "trigger_type": execution_context.get("trigger_type"),
        "trigger_detected_at": trigger_detected_at,
        "trigger_price": trigger_price,
        "threshold_price": threshold_price,
        "trigger_to_fill_ms": elapsed_ms(trigger_detected_at, executed_at),
        "order_round_trip_ms": order_round_trip_ms,
        "detection_slippage_bps": detection_slippage_bps,
        "execution_slippage_bps": slippage_bps,
        "threshold_to_fill_slippage_bps": threshold_to_fill_slippage_bps,
    }
    record = {
        "schema_version": 2,
        "executed_at": executed_at,
        "trade_date": display_date(executed_at),
        "bar_ts": normalize_ts(bar_ts),
        "symbol": getattr(config, "SYMBOL", None),
        "action": str(action or "").upper(),
        "reason": reason,
        "side": side,
        "pos_side": pos_side,
        "reduce_only": str(order.get("reduceOnly", "")).lower() == "true" if order else None,
        "ord_id": order.get("ordId"),
        "cl_ord_id": order.get("clOrdId"),
        "state": order.get("state"),
        "fill_price": fill_price,
        "fill_size": fill_size,
        "notional": aggregated["notional"],
        "fee_signed": fee_signed,
        "fee_abs": aggregated["fee_abs"],
        "fee_currency": aggregated["fee_currency"],
        "fee_source": aggregated["fee_source"],
        "slippage_value": slippage_value,
        "slippage_bps": slippage_bps,
        "reference_price": safe_optional_float(reference_price),
        "delta_qty": float(delta_qty),
        "pos_qty_before": float(pos_qty_before),
        "entry_price_before": float(entry_price_before),
        "pos_qty_after": float(pos_qty_after),
        "entry_price_after": float(entry_price_after),
        "closed_qty": closed_qty,
        "gross_realized_pnl": gross_realized_pnl,
        "net_realized_pnl": net_realized_pnl,
        "equity_before": before_eq,
        "equity_after": after_eq,
        "equity_delta": equity_delta,
        "equity_usdt_before": before_equity_usdt,
        "equity_usdt_after": after_equity_usdt,
        "equity_usdt_delta": equity_usdt_delta,
        "cash_balance_usdt_before": before_cash_balance_usdt,
        "cash_balance_usdt_after": after_cash_balance_usdt,
        "avail_eq_before": before_avail_eq,
        "avail_eq_after": after_avail_eq,
        "sizing_eq_before": before_sizing_eq,
        "sizing_eq_after": after_sizing_eq,
        "risk_context": {
            "leverage": safe_optional_float(getattr(config, "LEVERAGE", None)),
            "live_margin_usage_ratio": safe_optional_float(getattr(config, "LIVE_MARGIN_USAGE_RATIO", None)),
            "live_min_free_margin_usdt": safe_optional_float(getattr(config, "LIVE_MIN_FREE_MARGIN_USDT", None)),
            "trend_filter_enabled": bool(getattr(config, "TREND_FILTER_ENABLED", False)),
            "trend_bias": (signal_snapshot or {}).get("trend_bias"),
            "trend_gap": (signal_snapshot or {}).get("trend_gap"),
            "take_profit": safe_optional_float(getattr(config, "TAKE_PROFIT", None)),
            "stop_loss": safe_optional_float(getattr(config, "STOP_LOSS", None)),
        },
        "execution_quality": execution_quality,
        "signal": signal_snapshot or {},
        "decision": decision or {},
        "raw_order": order,
        "fills": fills,
    }
    return record


def append_trade_record(record, path=LIVE_FILLS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def trade_record_exists(*, ord_id=None, cl_ord_id=None, path=LIVE_FILLS_PATH):
    ord_id = str(ord_id or "")
    cl_ord_id = str(cl_ord_id or "")
    if not ord_id and not cl_ord_id:
        return False
    for record in load_trade_records(path):
        if ord_id and str(record.get("ord_id") or "") == ord_id:
            return True
        if cl_ord_id and str(record.get("cl_ord_id") or "") == cl_ord_id:
            return True
    return False


def load_trade_records(path=LIVE_FILLS_PATH):
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _empty_bucket():
    return {
        "count": 0,
        "notional": 0.0,
        "gross_realized_pnl": 0.0,
        "net_realized_pnl": 0.0,
        "fee_abs": 0.0,
        "slippage_value": 0.0,
        "wins": 0,
        "losses": 0,
    }


def _add_to_bucket(bucket, record):
    bucket["count"] += 1
    bucket["notional"] += safe_float(record.get("notional"), 0.0)
    bucket["gross_realized_pnl"] += safe_float(record.get("gross_realized_pnl"), 0.0)
    bucket["net_realized_pnl"] += safe_float(record.get("net_realized_pnl"), 0.0)
    bucket["fee_abs"] += safe_float(record.get("fee_abs"), 0.0)
    bucket["slippage_value"] += safe_float(record.get("slippage_value"), 0.0)
    if safe_float(record.get("closed_qty"), 0.0) > 0:
        pnl = safe_float(record.get("net_realized_pnl"), 0.0)
        if pnl > 0:
            bucket["wins"] += 1
        elif pnl < 0:
            bucket["losses"] += 1


def summarize_daily_records(records, trade_date):
    day_records = [r for r in records if r.get("trade_date") == trade_date]
    day_records.sort(key=lambda r: r.get("executed_at") or "")

    by_action = defaultdict(_empty_bucket)
    by_reason = defaultdict(_empty_bucket)
    by_side = defaultdict(_empty_bucket)
    summary = _empty_bucket()

    for record in day_records:
        _add_to_bucket(summary, record)
        _add_to_bucket(by_action[str(record.get("action") or "UNKNOWN")], record)
        _add_to_bucket(by_reason[str(record.get("reason") or "UNKNOWN")], record)
        _add_to_bucket(by_side[str(record.get("pos_side") or "unknown")], record)

    closing_records = [r for r in day_records if safe_float(r.get("closed_qty"), 0.0) > 0]
    usdt_equity_records = [
        record
        for record in day_records
        if record.get("equity_usdt_before") is not None
        and record.get("equity_usdt_after") is not None
    ]
    if usdt_equity_records:
        first_equity = usdt_equity_records[0].get("equity_usdt_before")
        last_equity = usdt_equity_records[-1].get("equity_usdt_after")
        equity_source = "usdt_equity"
        equity_currency = "USDT"
    else:
        first_equity = day_records[0].get("equity_before") if day_records else None
        last_equity = day_records[-1].get("equity_after") if day_records else None
        equity_source = "usd_total_equity"
        equity_currency = "USD"
    equity_delta = None
    if first_equity is not None and last_equity is not None:
        equity_delta = safe_float(last_equity) - safe_float(first_equity)

    return {
        "trade_date": trade_date,
        "record_count": len(day_records),
        "closing_trade_count": len(closing_records),
        "first_equity": first_equity,
        "last_equity": last_equity,
        "equity_delta": equity_delta,
        "equity_source": equity_source,
        "equity_currency": equity_currency,
        "totals": dict(summary),
        "by_action": {k: dict(v) for k, v in sorted(by_action.items())},
        "by_reason": {k: dict(v) for k, v in sorted(by_reason.items())},
        "by_side": {k: dict(v) for k, v in sorted(by_side.items())},
        "last_records": day_records[-10:],
    }


def _fmt(value, digits=2):
    if value is None:
        return "n/a"
    return f"{safe_float(value):.{digits}f}"


def _format_bucket_table(title, buckets):
    lines = [f"## {title}", "", "| 项目 | 笔数 | 名义金额 | 毛实现PnL | 净实现PnL | 手续费 | 滑点 | 胜/负 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    if not buckets:
        lines.append("| 无 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0/0 |")
    for name, bucket in buckets.items():
        lines.append(
            "| "
            f"{name} | {bucket['count']} | {_fmt(bucket['notional'])} | "
            f"{_fmt(bucket['gross_realized_pnl'])} | {_fmt(bucket['net_realized_pnl'])} | "
            f"{_fmt(bucket['fee_abs'])} | {_fmt(bucket['slippage_value'])} | "
            f"{bucket['wins']}/{bucket['losses']} |"
        )
    return lines


def format_daily_report_markdown(summary):
    totals = summary["totals"]
    equity_currency = summary.get("equity_currency") or "USD"
    lines = [
        f"# 每日交易复盘 {summary['trade_date']}",
        "",
        "## 总览",
        "",
        f"- 成交记录数: {summary['record_count']}",
        f"- 平仓/减仓记录数: {summary['closing_trade_count']}",
        f"- 起始权益: {_fmt(summary.get('first_equity'))} {equity_currency}",
        f"- 结束权益: {_fmt(summary.get('last_equity'))} {equity_currency}",
        f"- 权益变化: {_fmt(summary.get('equity_delta'))} {equity_currency}",
        f"- 毛实现PnL: {_fmt(totals.get('gross_realized_pnl'))} USDT",
        f"- 净实现PnL: {_fmt(totals.get('net_realized_pnl'))} USDT",
        f"- 手续费: {_fmt(totals.get('fee_abs'))} USDT",
        f"- 滑点成本: {_fmt(totals.get('slippage_value'))} USDT",
        "",
    ]
    lines.extend(_format_bucket_table("按动作归因", summary["by_action"]))
    lines.append("")
    lines.extend(_format_bucket_table("按出场/交易原因归因", summary["by_reason"]))
    lines.append("")
    lines.extend(_format_bucket_table("按多空方向归因", summary["by_side"]))
    lines.append("")
    lines.append("## 最近成交")
    lines.append("")
    lines.append("| 时间 | 动作 | 原因 | 方向 | 均价 | 数量 | 净实现PnL | 手续费 | 阈值滑点(bps) | 触发到成交(ms) |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for record in summary.get("last_records", []):
        execution_quality = record.get("execution_quality") or {}
        lines.append(
            "| "
            f"{record.get('executed_at')} | {record.get('action')} | {record.get('reason')} | "
            f"{record.get('pos_side')} | {_fmt(record.get('fill_price'), 4)} | "
            f"{_fmt(record.get('fill_size'), 6)} | {_fmt(record.get('net_realized_pnl'))} | "
            f"{_fmt(record.get('fee_abs'))} | "
            f"{_fmt(execution_quality.get('threshold_to_fill_slippage_bps'), 2)} | "
            f"{_fmt(execution_quality.get('trigger_to_fill_ms'), 1)} |"
        )
    return "\n".join(lines) + "\n"


def write_daily_report(
    trade_date=None,
    records_path=LIVE_FILLS_PATH,
    report_dir=DAILY_REPORT_DIR,
    latest_report_path=LATEST_DAILY_REPORT_PATH,
):
    records = load_trade_records(records_path)
    if trade_date is None:
        trade_date = display_date(datetime.now(timezone.utc))
    summary = summarize_daily_records(records, trade_date)

    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, f"{trade_date}.json")
    md_path = os.path.join(report_dir, f"{trade_date}.md")
    tmp_json_path = f"{json_path}.tmp"
    tmp_md_path = f"{md_path}.tmp"

    with open(tmp_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_json_path, json_path)

    with open(tmp_md_path, "w", encoding="utf-8") as f:
        f.write(format_daily_report_markdown(summary))
    os.replace(tmp_md_path, md_path)

    if latest_report_path:
        os.makedirs(os.path.dirname(latest_report_path), exist_ok=True)
        shutil.copyfile(md_path, latest_report_path)
    return summary, json_path, md_path
