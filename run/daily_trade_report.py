import argparse

from utils.trade_audit import write_daily_report
from utils.utils import notify_important


def fmt(value, digits=2):
    try:
        if value is None:
            return "-"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def format_daily_report_notification(summary, md_path):
    totals = summary.get("totals") or {}
    return "\n".join([
        f"[每日交易复盘] {summary.get('trade_date')}",
        f"成交记录数: {summary.get('record_count', 0)}",
        f"平仓/减仓记录数: {summary.get('closing_trade_count', 0)}",
        f"权益变化: {fmt(summary.get('equity_delta'))} USDT",
        f"净实现PnL: {fmt(totals.get('net_realized_pnl'))} USDT",
        f"手续费: {fmt(totals.get('fee_abs'))} USDT",
        f"滑点成本: {fmt(totals.get('slippage_value'))} USDT",
        f"report: {md_path}",
    ])


def main():
    parser = argparse.ArgumentParser(description="Generate daily trade report from live fill records")
    parser.add_argument("--date", dest="trade_date", default=None, help="Trade date in YYYY-MM-DD format")
    parser.add_argument(
        "--records-path",
        dest="records_path",
        default=None,
        help="Optional JSONL path for live fill records",
    )
    parser.add_argument(
        "--report-dir",
        dest="report_dir",
        default=None,
        help="Optional output directory for daily reports",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.records_path:
        kwargs["records_path"] = args.records_path
    if args.report_dir:
        kwargs["report_dir"] = args.report_dir

    summary, json_path, md_path = write_daily_report(args.trade_date, **kwargs)
    print(f"ok trade_date={summary['trade_date']}")
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    notify_important(format_daily_report_notification(summary, md_path))


if __name__ == "__main__":
    main()
