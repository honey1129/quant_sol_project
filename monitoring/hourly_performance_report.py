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
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from utils.utils import log_info, send_telegram


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
        'total_trades': total,
        'win_rate': win_rate,
        'total_pnl': total_pnl * 100,  # 转为百分比
    }


def format_performance_report(stats_24h, stats_today, mode_name):
    """格式化性能报告"""
    if not stats_24h or stats_24h['total_trades'] == 0:
        return f"""
🤖 **{mode_name} 性能报告**
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 **最近24小时**: 无交易
📊 **今日**: 无交易

💡 等待交易信号...
"""

    report = f"""
🤖 **{mode_name} 性能报告**
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 **最近24小时**
• 总交易: {stats_24h['total_trades']} 笔
• 止盈: {stats_24h['take_profits']} 笔
• 止损: {stats_24h['stop_losses']} 笔
• 胜率: **{stats_24h['win_rate']:.1f}%**
• 累计盈亏: {stats_24h['total_pnl']:+.2f}%
"""

    if stats_today and stats_today['total_trades'] > 0:
        report += f"""
📊 **今日** ({datetime.now().strftime('%m-%d')})
• 总交易: {stats_today['total_trades']} 笔
• 胜率: {stats_today['win_rate']:.1f}%
• 累计盈亏: {stats_today['total_pnl']:+.2f}%
"""

    # 最近3笔交易
    if stats_24h['trades']:
        report += "\n📝 **最近3笔交易**\n"
        for trade in sorted(stats_24h['trades'], key=lambda x: x['time'], reverse=True)[:3]:
            result_emoji = "✅" if trade['reason'] == 'TakeProfit' else "❌"
            pnl_str = f"{trade['pnl']*100:+.2f}%" if trade['pnl'] else "N/A"
            report += f"• {trade['time'].strftime('%H:%M')} {result_emoji} {trade['reason']} {pnl_str}\n"

    # 告警
    if stats_24h['total_trades'] >= 5:
        if stats_24h['win_rate'] < 20:
            report += "\n⚠️ **警告**: 胜率低于20%，建议检查策略\n"
        elif stats_24h['win_rate'] > 40:
            report += "\n🎉 **优秀**: 胜率超过40%！\n"

    # 对比基线
    baseline_winrate = 35.1 if mode_name == "简单规则模式" else 12.5
    if stats_24h['total_trades'] >= 3:
        diff = stats_24h['win_rate'] - baseline_winrate
        if abs(diff) > 5:
            trend = "高于" if diff > 0 else "低于"
            report += f"\n📈 胜率{trend}回测基线({baseline_winrate:.1f}%) {abs(diff):.1f}个百分点\n"

    return report


def main():
    # 日志路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base_dir, 'logs', 'live_trading.log')

    if not os.path.exists(log_path):
        log_info("日志文件不存在，跳过监控")
        return

    # 检测当前模式
    mode_name = "简单规则模式" if bool(config.USE_SIMPLE_RULE_MODE) else "ML模式"

    # 统计最近24小时
    stats_24h = parse_trade_log(log_path, hours=24)

    # 统计今日（从0点开始）
    now = datetime.now()
    hours_since_midnight = now.hour + now.minute / 60.0
    stats_today = parse_trade_log(log_path, hours=hours_since_midnight)

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
