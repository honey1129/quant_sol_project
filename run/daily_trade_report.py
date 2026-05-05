import argparse

from utils.trade_audit import write_daily_report


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


if __name__ == "__main__":
    main()
