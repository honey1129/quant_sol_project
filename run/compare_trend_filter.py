import contextlib
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.backtest import Backtester
from config import config
from utils.utils import LOGS_DIR, log_info


def apply_overrides(overrides):
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    return originals


def restore_overrides(originals):
    for key, value in originals.items():
        setattr(config, key, value)


def build_seed_backtester():
    base_interval = config.INTERVALS[0] if config.INTERVALS else "5m"
    window = config.WINDOWS.get(base_interval, 1000)
    log_info("准备共享回测缓存，用于趋势过滤 A/B")
    return Backtester(
        "multi_period",
        window,
        enable_csv_dump=False,
        show_progress=False,
        emit_diagnostics=False,
    )


def summarize_actions(trade_log):
    counts = Counter(str(item[1]) for item in trade_log)
    return {
        "open_long": int(counts.get("开多", 0)),
        "open_short": int(counts.get("开空", 0)),
        "add_long": int(counts.get("加多", 0)),
        "add_short": int(counts.get("加空", 0)),
        "reduce_long": int(counts.get("减多", 0)),
        "reduce_short": int(counts.get("减空", 0)),
        "close": int(counts.get("平仓", 0)),
        "reverse_close": int(counts.get("反向平仓", 0)),
    }


def run_candidate(seed_bt, name, overrides):
    originals = apply_overrides(overrides)
    try:
        bt = Backtester(
            "multi_period",
            seed_bt.window,
            data_dict=seed_bt.data_dict,
            reward_risk=seed_bt.reward_risk,
            precomputed_data=seed_bt.data,
            feature_cols=seed_bt.feature_cols,
            models=seed_bt.models,
            model_weights=seed_bt.model_weights,
            funding_history=seed_bt.funding_history,
            enable_csv_dump=False,
            show_progress=False,
            emit_diagnostics=False,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            summary = bt.run_backtest()
        summary["name"] = name
        summary["overrides"] = overrides
        summary["action_counts"] = summarize_actions(bt.trade_log)
        return summary
    finally:
        restore_overrides(originals)


def write_report(results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOGS_DIR, f"trend_filter_ab_{ts}.json")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "trend_filter_config": {
            "interval": config.TREND_FILTER_INTERVAL,
            "fast_col": config.TREND_FILTER_FAST_COL,
            "slow_col": config.TREND_FILTER_SLOW_COL,
            "min_gap": config.TREND_FILTER_MIN_GAP,
        },
        "base_config": {
            "initial_balance": config.INITIAL_BALANCE,
            "threshold_long": config.THRESHOLD_LONG,
            "threshold_short": config.THRESHOLD_SHORT,
            "signal_min_prob_diff": config.SIGNAL_MIN_PROB_DIFF,
            "reverse_signal_min_prob_diff": config.REVERSE_SIGNAL_MIN_PROB_DIFF,
            "min_adjust_amount": config.MIN_ADJUST_AMOUNT,
            "add_threshold": config.ADD_THRESHOLD,
            "max_rebalance_ratio": config.MAX_REBALANCE_RATIO,
            "trade_cooldown_bars": config.TRADE_COOLDOWN_BARS,
        },
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return path


def print_table(results):
    headers = [
        "name",
        "equity",
        "return%",
        "maxDD%",
        "trades",
        "openL",
        "openS",
        "revClose",
        "fees",
        "slip",
    ]
    print(",".join(headers))
    for item in results:
        actions = item["action_counts"]
        print(
            f"{item['name']},"
            f"{item['final_equity']:.2f},"
            f"{item['return_pct']:.2f},"
            f"{item['max_drawdown_pct']:.2f},"
            f"{item['trade_count']},"
            f"{actions['open_long']},"
            f"{actions['open_short']},"
            f"{actions['reverse_close']},"
            f"{item['fees_paid']:.2f},"
            f"{item['slippage_cost']:.2f}"
        )


def print_delta(results):
    by_name = {item["name"]: item for item in results}
    off = by_name.get("trend_filter_off")
    on = by_name.get("trend_filter_on")
    if not off or not on:
        return

    print("\ntrend_filter_on_minus_off")
    print(f"final_equity_delta={on['final_equity'] - off['final_equity']:.2f}")
    print(f"return_pct_delta={on['return_pct'] - off['return_pct']:.2f}")
    print(f"max_drawdown_pct_delta={on['max_drawdown_pct'] - off['max_drawdown_pct']:.2f}")
    print(f"trade_count_delta={on['trade_count'] - off['trade_count']}")
    print(f"open_short_delta={on['action_counts']['open_short'] - off['action_counts']['open_short']}")


def main():
    config.TELEGRAM_ENABLED = False
    seed_bt = build_seed_backtester()
    candidates = [
        ("trend_filter_off", {"TREND_FILTER_ENABLED": False}),
        ("trend_filter_on", {"TREND_FILTER_ENABLED": True}),
    ]

    results = [run_candidate(seed_bt, name, overrides) for name, overrides in candidates]
    print_table(results)
    print_delta(results)
    report_path = write_report(results)
    print(f"\nreport_path={report_path}")


if __name__ == "__main__":
    main()
