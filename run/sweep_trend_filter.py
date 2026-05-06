import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import config
from run.compare_trend_filter import build_seed_backtester, run_candidate
from utils.utils import LOGS_DIR


def candidate_overrides():
    candidates = [("trend_filter_off", {"TREND_FILTER_ENABLED": False})]
    intervals = ["1H", "15m"]
    min_gaps = [0.001, 0.002, 0.003, 0.005, 0.008]

    for interval in intervals:
        for min_gap in min_gaps:
            name = f"trend_filter_{interval}_gap_{min_gap:.4f}".replace(".", "p")
            candidates.append(
                (
                    name,
                    {
                        "TREND_FILTER_ENABLED": True,
                        "TREND_FILTER_INTERVAL": interval,
                        "TREND_FILTER_FAST_COL": config.TREND_FILTER_FAST_COL,
                        "TREND_FILTER_SLOW_COL": config.TREND_FILTER_SLOW_COL,
                        "TREND_FILTER_MIN_GAP": min_gap,
                    },
                )
            )
    return candidates


def add_baseline_deltas(results):
    baseline = next((item for item in results if item["name"] == "trend_filter_off"), None)
    if baseline is None:
        return results

    for item in results:
        item["delta_vs_off"] = {
            "final_equity": float(item["final_equity"] - baseline["final_equity"]),
            "return_pct": float(item["return_pct"] - baseline["return_pct"]),
            "max_drawdown_pct": float(item["max_drawdown_pct"] - baseline["max_drawdown_pct"]),
            "trade_count": int(item["trade_count"] - baseline["trade_count"]),
            "fees_paid": float(item["fees_paid"] - baseline["fees_paid"]),
            "slippage_cost": float(item["slippage_cost"] - baseline["slippage_cost"]),
        }
    return results


def sort_results(results):
    return sorted(
        results,
        key=lambda item: (
            item["final_equity"],
            item["max_drawdown_pct"],
            -item["trade_count"],
        ),
        reverse=True,
    )


def write_report(results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOGS_DIR, f"trend_filter_sweep_{ts}.json")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ranking_rule": "final_equity desc, max_drawdown_pct desc, trade_count asc",
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return path


def print_table(results):
    print("rank,name,equity,return%,maxDD%,trades,delta_equity,delta_return%,delta_maxDD%,delta_trades,fees,slip")
    for idx, item in enumerate(results, start=1):
        delta = item.get("delta_vs_off", {})
        print(
            f"{idx},"
            f"{item['name']},"
            f"{item['final_equity']:.2f},"
            f"{item['return_pct']:.2f},"
            f"{item['max_drawdown_pct']:.2f},"
            f"{item['trade_count']},"
            f"{delta.get('final_equity', 0.0):.2f},"
            f"{delta.get('return_pct', 0.0):.2f},"
            f"{delta.get('max_drawdown_pct', 0.0):.2f},"
            f"{delta.get('trade_count', 0)},"
            f"{item['fees_paid']:.2f},"
            f"{item['slippage_cost']:.2f}"
        )


def print_recommendation(results):
    baseline = next((item for item in results if item["name"] == "trend_filter_off"), None)
    best = results[0] if results else None
    if baseline is None or best is None:
        return

    print("\nrecommendation")
    if best["name"] == baseline["name"]:
        print("TREND_FILTER_ENABLED=0")
        print("reason=baseline_off_has_best_final_equity")
        return

    delta = best["delta_vs_off"]
    if delta["final_equity"] > 0 and delta["max_drawdown_pct"] >= -0.1:
        print("TREND_FILTER_ENABLED=1")
        for key, value in best["overrides"].items():
            print(f"{key}={value}")
        print("reason=best_filter_improves_equity_without_material_drawdown_penalty")
    else:
        print("TREND_FILTER_ENABLED=0")
        print("reason=best_filter_does_not_clear_equity_drawdown_gate")


def main():
    config.TELEGRAM_ENABLED = False
    seed_bt = build_seed_backtester()
    raw_results = [
        run_candidate(seed_bt, name, overrides)
        for name, overrides in candidate_overrides()
    ]
    ranked_results = sort_results(add_baseline_deltas(raw_results))
    print_table(ranked_results)
    print_recommendation(ranked_results)
    report_path = write_report(ranked_results)
    print(f"\nreport_path={report_path}")


if __name__ == "__main__":
    main()
