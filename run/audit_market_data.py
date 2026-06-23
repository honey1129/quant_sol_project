"""Audit OKX market data quality and optionally compare it with Binance futures.

Usage:
    PYTHONPATH=. python -m run.audit_market_data --rows 500 --compare-binance
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from core.ml_feature_engineering import interval_to_timedelta
from core.okx_api import OKXClient
from utils.utils import LOGS_DIR, log_error, log_info


BINANCE_INTERVAL_MAP = {
    "1H": "1h",
}


def normalize_utc_index(df):
    frame = df.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame.set_index("timestamp", inplace=True)
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame.index = index
    frame.sort_index(inplace=True)
    return frame


def _as_float_series(frame, column):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _issue(severity, code, message, **extra):
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    payload.update(extra)
    return payload


def audit_ohlcv_frame(df, interval, *, now_ts=None, max_examples=8):
    interval_delta = interval_to_timedelta(interval)
    if now_ts is None:
        now_ts = pd.Timestamp.utcnow()
    now_ts = pd.Timestamp(now_ts)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")

    if df is None or df.empty:
        return {
            "interval": interval,
            "rows": 0,
            "issues": [_issue("error", "empty_data", "K线数据为空")],
            "status": "error",
        }

    frame = normalize_utc_index(df)
    index = pd.DatetimeIndex(frame.index)
    duplicate_mask = index.duplicated(keep=False)
    duplicate_examples = [ts.isoformat() for ts in index[duplicate_mask][:max_examples]]
    deduped = frame[~index.duplicated(keep="last")].copy()
    deduped.sort_index(inplace=True)
    idx = pd.DatetimeIndex(deduped.index)

    diffs = idx.to_series().diff().dropna()
    long_gaps = diffs[diffs > interval_delta]
    short_gaps = diffs[diffs < interval_delta]
    missing_bars = int(
        sum(max(0, int(round(diff / interval_delta)) - 1) for diff in long_gaps)
    )
    gap_examples = []
    for ts, diff in long_gaps.head(max_examples).items():
        prev_ts = idx[idx.get_loc(ts) - 1]
        gap_examples.append({
            "from": prev_ts.isoformat(),
            "to": ts.isoformat(),
            "gap_minutes": float(diff / pd.Timedelta(minutes=1)),
            "missing_bars": max(0, int(round(diff / interval_delta)) - 1),
        })

    open_ = _as_float_series(deduped, "open")
    high = _as_float_series(deduped, "high")
    low = _as_float_series(deduped, "low")
    close = _as_float_series(deduped, "close")
    volume = _as_float_series(deduped, "volume")
    invalid_high_low = high < low
    invalid_high_body = high < pd.concat([open_, close], axis=1).max(axis=1)
    invalid_low_body = low > pd.concat([open_, close], axis=1).min(axis=1)
    non_positive_price = (open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)
    negative_volume = volume < 0
    nan_ohlcv = pd.concat([open_, high, low, close, volume], axis=1).isna().any(axis=1)
    ohlc_invalid_mask = (
        invalid_high_low
        | invalid_high_body
        | invalid_low_body
        | non_positive_price
        | negative_volume
        | nan_ohlcv
    )

    confirm_counts = {}
    unconfirmed_count = 0
    if "confirm" in deduped.columns:
        confirm_counts = {
            str(key): int(value)
            for key, value in deduped["confirm"].astype(str).value_counts().sort_index().items()
        }
        unconfirmed_count = int((deduped["confirm"].astype(str) != "1").sum())

    inferred_open_mask = (idx + interval_delta) > now_ts
    inferred_open_count = int(inferred_open_mask.sum())
    last_ts = idx[-1]
    last_bar_close_time = last_ts + interval_delta
    freshness_lag_seconds = float((now_ts - last_bar_close_time).total_seconds())

    issues = []
    if duplicate_examples:
        issues.append(_issue(
            "error",
            "duplicate_timestamps",
            "存在重复K线时间戳",
            count=int(duplicate_mask.sum()),
            examples=duplicate_examples,
        ))
    if len(long_gaps) > 0:
        issues.append(_issue(
            "error",
            "missing_bars",
            "K线时间序列存在缺口",
            gap_count=int(len(long_gaps)),
            missing_bars=missing_bars,
            examples=gap_examples,
        ))
    if len(short_gaps) > 0:
        issues.append(_issue(
            "error",
            "short_or_overlapping_bars",
            "K线间隔短于预期，可能有乱序或重复",
            count=int(len(short_gaps)),
            examples=[ts.isoformat() for ts in short_gaps.index[:max_examples]],
        ))
    if int(ohlc_invalid_mask.sum()) > 0:
        issues.append(_issue(
            "error",
            "invalid_ohlcv",
            "OHLCV 字段存在非法值",
            count=int(ohlc_invalid_mask.sum()),
            examples=[ts.isoformat() for ts in idx[ohlc_invalid_mask.to_numpy()][:max_examples]],
        ))
    if unconfirmed_count > 0:
        issues.append(_issue(
            "warning",
            "unconfirmed_exchange_bars",
            "交易所返回未确认K线，训练/回测应过滤",
            count=unconfirmed_count,
            confirm_counts=confirm_counts,
        ))
    if inferred_open_count > 0:
        issues.append(_issue(
            "warning",
            "inferred_open_tail_bars",
            "按当前时间推断尾部仍有未收盘K线",
            count=inferred_open_count,
        ))
    if freshness_lag_seconds > interval_delta.total_seconds() * 3:
        issues.append(_issue(
            "warning",
            "stale_tail",
            "最新K线距离当前时间偏久",
            lag_seconds=freshness_lag_seconds,
        ))

    status = "ok"
    if any(item["severity"] == "error" for item in issues):
        status = "error"
    elif issues:
        status = "warning"

    expected_rows = int(round((idx[-1] - idx[0]) / interval_delta)) + 1 if len(idx) else 0
    close_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "interval": interval,
        "status": status,
        "rows": int(len(deduped)),
        "raw_rows": int(len(frame)),
        "expected_rows_between_start_end": expected_rows,
        "start": idx[0].isoformat(),
        "end": idx[-1].isoformat(),
        "duplicate_timestamp_count": int(duplicate_mask.sum()),
        "gap_count": int(len(long_gaps)),
        "missing_bars": missing_bars,
        "short_gap_count": int(len(short_gaps)),
        "unconfirmed_count": unconfirmed_count,
        "confirm_counts": confirm_counts,
        "inferred_open_tail_count": inferred_open_count,
        "freshness_lag_seconds": freshness_lag_seconds,
        "close_return_abs_quantiles": {
            "p50": float(close_returns.abs().quantile(0.50)) if not close_returns.empty else None,
            "p95": float(close_returns.abs().quantile(0.95)) if not close_returns.empty else None,
            "p99": float(close_returns.abs().quantile(0.99)) if not close_returns.empty else None,
            "max": float(close_returns.abs().max()) if not close_returns.empty else None,
        },
        "issues": issues,
    }


def okx_symbol_to_binance(symbol):
    parts = str(symbol).upper().split("-")
    if len(parts) >= 2:
        return f"{parts[0]}{parts[1]}"
    return str(symbol).upper().replace("-", "")


def okx_interval_to_binance(interval):
    return BINANCE_INTERVAL_MAP.get(str(interval), str(interval).lower())


def fetch_binance_futures_klines(symbol, interval, limit=500, timeout=10):
    url = "https://fapi.binance.com/fapi/v1/klines"
    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": okx_interval_to_binance(interval),
            "limit": min(max(int(limit), 1), 1500),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json()
    normalized = []
    for row in rows:
        normalized.append({
            "timestamp": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5],
        })
    frame = pd.DataFrame(normalized)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.set_index("timestamp", inplace=True)
    frame.sort_index(inplace=True)
    return frame


def compare_close_frames(left, right, *, left_name, right_name, tolerance_pct=0.5, max_examples=8):
    left_frame = normalize_utc_index(left)
    right_frame = normalize_utc_index(right)
    joined = pd.DataFrame({
        f"{left_name}_close": _as_float_series(left_frame, "close"),
    }).join(
        pd.DataFrame({f"{right_name}_close": _as_float_series(right_frame, "close")}),
        how="inner",
    ).dropna()

    if joined.empty:
        return {
            "status": "warning",
            "rows": 0,
            "issues": [_issue("warning", "no_overlap", "两个数据源没有重叠时间戳")],
        }

    left_close = joined[f"{left_name}_close"]
    right_close = joined[f"{right_name}_close"]
    pct_diff = (left_close - right_close).abs() / right_close.replace(0, np.nan) * 100.0
    pct_diff = pct_diff.replace([np.inf, -np.inf], np.nan).dropna()
    breaches = pct_diff[pct_diff > float(tolerance_pct)]
    issues = []
    if not breaches.empty:
        issues.append(_issue(
            "warning",
            "price_deviation_above_tolerance",
            "两个市场收盘价偏差超过阈值；合约标的不同会有合理基差",
            tolerance_pct=float(tolerance_pct),
            breach_count=int(len(breaches)),
            examples=[
                {
                    "timestamp": ts.isoformat(),
                    "pct_diff": float(value),
                    f"{left_name}_close": float(left_close.loc[ts]),
                    f"{right_name}_close": float(right_close.loc[ts]),
                }
                for ts, value in breaches.sort_values(ascending=False).head(max_examples).items()
            ],
        ))

    return {
        "status": "warning" if issues else "ok",
        "rows": int(len(joined)),
        "start": joined.index.min().isoformat(),
        "end": joined.index.max().isoformat(),
        "tolerance_pct": float(tolerance_pct),
        "abs_pct_diff": {
            "mean": float(pct_diff.mean()) if not pct_diff.empty else None,
            "p50": float(pct_diff.quantile(0.50)) if not pct_diff.empty else None,
            "p95": float(pct_diff.quantile(0.95)) if not pct_diff.empty else None,
            "max": float(pct_diff.max()) if not pct_diff.empty else None,
        },
        "breach_count": int(len(breaches)),
        "issues": issues,
    }


def build_cross_interval_alignment(okx_frames, *, base_interval="5m", tolerance_pct=0.02, max_examples=8):
    if base_interval not in okx_frames:
        return {"status": "skipped", "reason": f"missing_base_interval:{base_interval}"}

    base = normalize_utc_index(okx_frames[base_interval])
    base_delta = interval_to_timedelta(base_interval)
    base_close = _as_float_series(base, "close")
    results = {}
    issues = []

    for interval, frame in okx_frames.items():
        if interval == base_interval:
            continue
        high = normalize_utc_index(frame)
        interval_delta = interval_to_timedelta(interval)
        rows = []
        for high_ts, high_row in high.iterrows():
            base_ts = high_ts + interval_delta - base_delta
            if base_ts not in base_close.index:
                continue
            high_close = float(high_row["close"])
            ref_close = float(base_close.loc[base_ts])
            pct_diff = abs(high_close - ref_close) / max(abs(ref_close), 1e-12) * 100.0
            rows.append((high_ts, base_ts, high_close, ref_close, pct_diff))

        if not rows:
            result = {
                "status": "warning",
                "rows": 0,
                "issues": [_issue("warning", "no_cross_interval_overlap", "跨周期没有可对齐样本")],
            }
            results[interval] = result
            issues.extend(result["issues"])
            continue

        diffs = pd.Series([item[4] for item in rows])
        breaches = [item for item in rows if item[4] > float(tolerance_pct)]
        result = {
            "status": "warning" if breaches else "ok",
            "rows": int(len(rows)),
            "tolerance_pct": float(tolerance_pct),
            "abs_pct_diff": {
                "mean": float(diffs.mean()),
                "p95": float(diffs.quantile(0.95)),
                "max": float(diffs.max()),
            },
            "breach_count": int(len(breaches)),
            "issues": [],
        }
        if breaches:
            examples = sorted(breaches, key=lambda item: item[4], reverse=True)[:max_examples]
            result["issues"].append(_issue(
                "warning",
                "cross_interval_close_mismatch",
                "高周期 close 与同一收盘时刻的 5m close 不一致",
                examples=[
                    {
                        "high_interval_open": high_ts.isoformat(),
                        "base_interval_open": base_ts.isoformat(),
                        "high_close": high_close,
                        "base_close": ref_close,
                        "pct_diff": pct_diff,
                    }
                    for high_ts, base_ts, high_close, ref_close, pct_diff in examples
                ],
            ))
        results[interval] = result
        issues.extend(result["issues"])

    return {
        "status": "warning" if issues else "ok",
        "base_interval": base_interval,
        "results": results,
        "issues": issues,
    }


def parse_intervals(value):
    if not value:
        return list(config.INTERVALS)
    return [item.strip() for item in str(value).split(",") if item.strip()]


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"market_data_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)
    return output_path


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


def build_report(args):
    intervals = parse_intervals(args.intervals)
    client = OKXClient()
    okx_frames = {}
    interval_reports = {}
    binance_reports = {}

    for interval in intervals:
        log_info(f"拉取 OKX K线: symbol={args.symbol} interval={interval} rows={args.rows}")
        frame = client.fetch_ohlcv(args.symbol, bar=interval, max_limit=int(args.rows))
        okx_frames[interval] = frame.set_index("timestamp") if "timestamp" in frame.columns else frame
        interval_reports[interval] = audit_ohlcv_frame(okx_frames[interval], interval)

        if args.compare_binance:
            binance_symbol = args.binance_symbol or okx_symbol_to_binance(args.symbol)
            try:
                log_info(f"拉取 Binance USDT-M K线: symbol={binance_symbol} interval={interval}")
                binance_frame = fetch_binance_futures_klines(
                    binance_symbol,
                    interval,
                    limit=int(args.rows),
                    timeout=float(args.http_timeout),
                )
                binance_reports[interval] = compare_close_frames(
                    okx_frames[interval],
                    binance_frame,
                    left_name="okx",
                    right_name="binance",
                    tolerance_pct=float(args.price_tolerance_pct),
                )
            except Exception as exc:
                log_error(f"Binance 对照失败: interval={interval}, err={exc}")
                binance_reports[interval] = {
                    "status": "error",
                    "issues": [_issue("error", "binance_compare_failed", str(exc))],
                }

    cross_interval = build_cross_interval_alignment(
        okx_frames,
        base_interval=args.base_interval,
        tolerance_pct=float(args.alignment_tolerance_pct),
    )
    sections = [*interval_reports.values(), cross_interval, *binance_reports.values()]
    error_count = sum(
        1
        for section in sections
        for issue in section.get("issues", [])
        if issue.get("severity") == "error"
    )
    warning_count = sum(
        1
        for section in sections
        for issue in section.get("issues", [])
        if issue.get("severity") == "warning"
    )
    return {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "symbol": args.symbol,
        "rows_requested": int(args.rows),
        "intervals": intervals,
        "okx": interval_reports,
        "cross_interval_alignment": cross_interval,
        "binance_compare_enabled": bool(args.compare_binance),
        "binance_symbol": args.binance_symbol or okx_symbol_to_binance(args.symbol),
        "binance": binance_reports,
        "summary": {
            "status": "error" if error_count else ("warning" if warning_count else "ok"),
            "error_count": int(error_count),
            "warning_count": int(warning_count),
        },
    }


def print_summary(report, path):
    summary = report["summary"]
    log_info(
        "行情数据审计完成: "
        f"status={summary['status']} errors={summary['error_count']} warnings={summary['warning_count']}"
    )
    for interval, item in report["okx"].items():
        log_info(
            f"OKX {interval}: status={item['status']} rows={item['rows']} "
            f"gaps={item['gap_count']} duplicates={item['duplicate_timestamp_count']} "
            f"unconfirmed={item['unconfirmed_count']} start={item.get('start')} end={item.get('end')}"
        )
        for issue in item.get("issues", [])[:3]:
            log_info(f"  - {issue['severity']} {issue['code']}: {issue['message']}")

    cross = report.get("cross_interval_alignment", {})
    log_info(f"跨周期对齐: status={cross.get('status')} base={cross.get('base_interval')}")
    for interval, item in (cross.get("results") or {}).items():
        diff = item.get("abs_pct_diff") or {}
        log_info(
            f"  {interval}: rows={item.get('rows')} breach={item.get('breach_count')} "
            f"p95_diff={diff.get('p95')}"
        )

    if report.get("binance_compare_enabled"):
        for interval, item in report.get("binance", {}).items():
            diff = item.get("abs_pct_diff") or {}
            log_info(
                f"Binance对照 {interval}: status={item.get('status')} rows={item.get('rows')} "
                f"breach={item.get('breach_count')} p95_diff={diff.get('p95')}"
            )
    log_info(f"审计报告: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="审计 OKX 行情K线连续性、完整性，并可选对照 Binance")
    parser.add_argument("--symbol", default=config.SYMBOL, help="OKX instId，例如 SOL-USDT-SWAP")
    parser.add_argument("--intervals", default=",".join(config.INTERVALS), help="逗号分隔周期，例如 5m,15m,1H")
    parser.add_argument("--rows", type=int, default=500, help="每个周期拉取的K线数量")
    parser.add_argument("--base-interval", default="5m", help="跨周期对齐使用的基础周期")
    parser.add_argument("--alignment-tolerance-pct", type=float, default=0.02, help="OKX 跨周期 close 允许偏差百分比")
    parser.add_argument("--compare-binance", action="store_true", help="额外拉 Binance USDT-M 永续K线做收盘价偏差对照")
    parser.add_argument("--binance-symbol", default=None, help="Binance USDT-M symbol，默认从 OKX symbol 推断")
    parser.add_argument("--price-tolerance-pct", type=float, default=0.50, help="OKX vs Binance 收盘价偏差告警阈值百分比")
    parser.add_argument("--http-timeout", type=float, default=10.0, help="外部 HTTP 请求超时秒数")
    parser.add_argument("--fail-on-issues", action="store_true", help="发现 error 级问题时以非0退出")
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = json_safe(build_report(args))
    path = write_report(report, args.output)
    print_summary(report, path)
    if args.fail_on_issues and report["summary"]["error_count"] > 0:
        raise SystemExit(2)
    print(json.dumps({"report_path": path, "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
