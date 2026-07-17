"""
每小时胜率监控脚本

功能:
- 统计最近24小时的交易胜率
- 对比ML模式和简单规则模式
- 发送Telegram通知
- 可通过cron每小时运行
"""
import os
import sys
import re
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.utils import DISPLAY_TIMEZONE, log_info, send_telegram


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _parse_iso_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _same_timezone(value, reference):
    if value.tzinfo is None and reference.tzinfo is not None:
        return value.replace(tzinfo=reference.tzinfo)
    if value.tzinfo is not None and reference.tzinfo is None:
        return value.replace(tzinfo=None)
    if value.tzinfo is not None and reference.tzinfo is not None:
        return value.astimezone(reference.tzinfo)
    return value


def _reason_label(reason):
    raw = str(reason or "-")
    key = raw.split("(", 1)[0]
    labels = {
        "TakeProfit": "止盈",
        "StopLoss": "止损",
        "LossGuardExit": "风控退出",
        "ReverseClose": "反向平仓",
        "ConsecutiveReverseClose": "连续反向平仓",
        "TP/SL": "平仓",
    }
    return labels.get(key, raw)


def _fmt_signed_usdt(value):
    value = _safe_float(value)
    if value is None:
        return "未记录盈亏"
    return f"{value:+.2f} USDT"


def _fmt_signed_pct(value):
    value = _safe_float(value)
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def _fmt_trade_pnl(trade):
    if trade.get("net_pnl") is not None:
        return _fmt_signed_usdt(trade.get("net_pnl"))
    if trade.get("pnl") is not None:
        return f"{float(trade['pnl']) * 100:+.2f}%"
    return "未记录盈亏"


def _empty_stats(source):
    return {
        "trades": [],
        "take_profits": 0,
        "stop_losses": 0,
        "profit_count": 0,
        "loss_count": 0,
        "flat_count": 0,
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "net_pnl": 0.0,
        "return_pct": None,
        "avg_pnl": None,
        "source": source,
    }


def parse_live_fills(fill_path, hours=24, now=None):
    """Parse structured fill records and summarize closed trades."""
    if not os.path.exists(fill_path):
        return None

    rows = []
    with open(fill_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_iso_datetime(record.get("executed_at") or record.get("bar_ts"))
            if ts is None:
                continue
            record["_ts"] = ts
            rows.append(record)

    if not rows:
        return _empty_stats("live_fills")

    rows.sort(key=lambda item: item["_ts"])
    if now is None:
        latest_ts = max(item["_ts"] for item in rows)
        now = datetime.now(latest_ts.tzinfo) if latest_ts.tzinfo is not None else datetime.now()
    rows = [{**item, "_ts": _same_timezone(item["_ts"], now)} for item in rows]
    cutoff_time = now - timedelta(hours=hours)
    window_rows = [record for record in rows if record["_ts"] >= cutoff_time]

    closes = []
    for record in window_rows:
        if str(record.get("action") or "").upper() != "CLOSE":
            continue
        net_pnl = _safe_float(record.get("net_realized_pnl"))
        closes.append({
            "time": record["_ts"],
            "reason": record.get("reason") or (record.get("decision") or {}).get("reason"),
            "net_pnl": net_pnl,
            "side": record.get("pos_side"),
            "price": _safe_float(record.get("fill_price")),
            "entry_price": _safe_float(record.get("entry_price_before")),
        })

    if not closes:
        return _empty_stats("live_fills")

    take_profits = sum(1 for trade in closes if str(trade.get("reason")) == "TakeProfit")
    stop_losses = sum(1 for trade in closes if str(trade.get("reason")) == "StopLoss")
    profit_count = sum(1 for trade in closes if (trade.get("net_pnl") or 0.0) > 0)
    loss_count = sum(1 for trade in closes if (trade.get("net_pnl") or 0.0) < 0)
    flat_count = len(closes) - profit_count - loss_count
    net_pnl = sum(_safe_float(record.get("net_realized_pnl")) or 0.0 for record in window_rows)

    first_equity = None
    for record in window_rows:
        first_equity = _safe_float(record.get("equity_before"))
        if first_equity:
            break
    return_pct = (net_pnl / first_equity * 100.0) if first_equity else None

    return {
        "trades": closes,
        "take_profits": take_profits,
        "stop_losses": stop_losses,
        "profit_count": profit_count,
        "loss_count": loss_count,
        "flat_count": flat_count,
        "total_trades": len(closes),
        "win_rate": profit_count / len(closes) * 100.0,
        "total_pnl": 0.0 if return_pct is None else return_pct,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "avg_pnl": net_pnl / len(closes),
        "source": "live_fills",
    }


def parse_trade_log(log_path, hours=24):
    """
    解析交易日志，提取止盈止损记录

    Returns:
        {
            'trades': [{'time', 'reason', 'pnl', 'price', 'entry_price'}],
            'take_profits': int,
            'stop_losses': int,
            'win_rate': float,
            'total_pnl': float,
        }
    """
    if not os.path.exists(log_path):
        return None

    cutoff_time = datetime.now() - timedelta(hours=hours)
    trades = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 找到所有平仓记录
    entry_prices = {}  # bar_time -> entry_price
    for i, line in enumerate(lines):
        try:
            # 提取时间戳
            time_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not time_match:
                continue

            timestamp_str = time_match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')

            if timestamp < cutoff_time:
                continue

            # 提取开仓价格
            if '执行开仓' in line or '执行加仓' in line:
                entry_match = re.search(r'entry_price=([0-9.]+)', line)
                if entry_match:
                    entry_price = float(entry_match.group(1))
                    entry_prices[timestamp_str] = entry_price

            # 提取平仓记录
            if '执行平仓' in line:
                reason_match = re.search(r'reason=(\w+)', line)
                if not reason_match:
                    continue

                reason = reason_match.group(1)
                if reason not in ['TakeProfit', 'StopLoss']:
                    continue

                # 往前找最近的bar价格
                price = None
                for j in range(i-1, max(0, i-20), -1):
                    price_match = re.search(r'price=([0-9.]+)', lines[j])
                    if price_match:
                        price = float(price_match.group(1))
                        break

                # 找对应的入场价
                entry_price = None
                for t in sorted(entry_prices.keys(), reverse=True):
                    if t <= timestamp_str:
                        entry_price = entry_prices[t]
                        break

                # 计算PnL（简化版，不考虑方向）
                pnl = None
                if price and entry_price:
                    pnl_pct = (price - entry_price) / entry_price
                    pnl = pnl_pct

                trades.append({
                    'time': timestamp,
                    'reason': reason,
                    'price': price,
                    'entry_price': entry_price,
                    'pnl': pnl,
                })

        except Exception as e:
            continue

    # 统计
    tp_count = len([t for t in trades if t['reason'] == 'TakeProfit'])
    sl_count = len([t for t in trades if t['reason'] == 'StopLoss'])
    total = tp_count + sl_count

    win_rate = (tp_count / total * 100) if total > 0 else 0.0

    # 计算总盈亏（简化版）
    total_pnl = sum([t['pnl'] for t in trades if t['pnl'] is not None])

    return {
        'trades': trades,
        'take_profits': tp_count,
        'stop_losses': sl_count,
        'profit_count': tp_count,
        'loss_count': sl_count,
        'flat_count': 0,
        'total_trades': total,
        'win_rate': win_rate,
        'total_pnl': total_pnl * 100,  # 转为百分比
        'net_pnl': None,
        'return_pct': total_pnl * 100,
        'avg_pnl': None,
        'source': 'live_trading_log',
    }


def load_trade_stats(base_dir, hours=24, now=None):
    fill_path = os.path.join(base_dir, "logs", "live_fills.jsonl")
    stats = parse_live_fills(fill_path, hours=hours, now=now)
    if stats is not None:
        return stats

    log_path = os.path.join(base_dir, "logs", "live_trading.log")
    return parse_trade_log(log_path, hours=hours)


def _format_pnl_line(stats):
    if stats.get("net_pnl") is not None:
        pct = ""
        if stats.get("return_pct") is not None:
            pct = f" ({_fmt_signed_pct(stats.get('return_pct'))})"
        return f"净盈亏: {_fmt_signed_usdt(stats.get('net_pnl'))}{pct}"
    return f"累计盈亏: {_fmt_signed_pct(stats.get('total_pnl'))}"


def _format_avg_line(stats):
    if stats.get("avg_pnl") is None:
        return None
    return f"平均每笔: {_fmt_signed_usdt(stats.get('avg_pnl'))}"


def _format_stats_block(title, stats):
    if not stats or stats["total_trades"] == 0:
        return f"{title}: 暂无平仓交易"

    lines = [
        title,
        (
            f"交易: {stats['total_trades']}笔 | "
            f"{stats.get('profit_count', 0)}赚{stats.get('loss_count', 0)}亏 | "
            f"胜率 {stats['win_rate']:.1f}%"
        ),
        _format_pnl_line(stats),
    ]
    avg_line = _format_avg_line(stats)
    if avg_line:
        lines.append(avg_line)
    return "\n".join(lines)


def _format_conclusion(stats_24h):
    if not stats_24h or stats_24h["total_trades"] == 0:
        return "结论: 最近24小时没有平仓交易，继续等待信号。"

    net_pnl = stats_24h.get("net_pnl")
    if net_pnl is not None:
        if net_pnl > 0:
            status = "盈利"
        elif net_pnl < 0:
            status = "亏损"
        else:
            status = "持平"
        pnl_text = _fmt_signed_usdt(net_pnl)
    else:
        total_pnl = stats_24h.get("total_pnl")
        if total_pnl > 0:
            status = "盈利"
        elif total_pnl < 0:
            status = "亏损"
        else:
            status = "持平"
        pnl_text = _fmt_signed_pct(total_pnl)

    return (
        f"结论: 最近24小时{status}，"
        f"{stats_24h.get('profit_count', 0)}赚{stats_24h.get('loss_count', 0)}亏，"
        f"{pnl_text}。"
    )


def format_performance_report(stats_24h, stats_today, mode_name):
    """格式化性能报告: 先给结论，再给必要数字。"""
    stats_24h = stats_24h or _empty_stats("none")
    stats_today = stats_today or _empty_stats("none")
    now = datetime.now(DISPLAY_TIMEZONE)
    now_text = now.strftime("%Y-%m-%d %H:%M")
    report = [
        f"**{mode_name}运行简报**",
        f"时间: {now_text}",
        "",
        _format_conclusion(stats_24h),
        "",
        _format_stats_block("最近24小时", stats_24h),
        "",
        _format_stats_block(f"今日({now.strftime('%m-%d')})", stats_today),
    ]

    if stats_24h['trades']:
        report.extend(["", "最近3笔"])
        for trade in sorted(stats_24h['trades'], key=lambda x: x['time'], reverse=True)[:3]:
            report.append(
                f"{trade['time'].strftime('%H:%M')} "
                f"{_reason_label(trade.get('reason'))} "
                f"{_fmt_trade_pnl(trade)}"
            )

    report.extend(["", "建议: 继续观察3-5天，不因为短期盈利放松风控。"])
    return "\n".join(report)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 检测当前模式
    mode_name = "简单规则模式" if bool(config.USE_SIMPLE_RULE_MODE) else "ML模式"

    # 统计最近24小时
    stats_24h = load_trade_stats(base_dir, hours=24)
    if stats_24h is None:
        log_info("交易日志不存在，跳过监控")
        return

    # 统计今日（从0点开始）
    now = datetime.now(DISPLAY_TIMEZONE)
    hours_since_midnight = now.hour + now.minute / 60.0
    stats_today = load_trade_stats(base_dir, hours=hours_since_midnight, now=now)

    # 生成报告
    report = format_performance_report(stats_24h, stats_today, mode_name)

    # 发送Telegram
    try:
        send_telegram(report)
        log_info(f"性能报告已发送 - {mode_name}")
        print(report)
    except Exception as e:
        log_info(f"发送Telegram失败: {e}")
        print(report)

    # 返回统计结果供其他脚本使用
    return stats_24h


if __name__ == '__main__':
    main()
