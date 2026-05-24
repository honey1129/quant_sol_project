import contextlib
import io
import json
import os
import sys
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
    log_info("准备共享回测缓存，用于 SmallTarget regime A/B")
    return Backtester(
        "multi_period",
        window,
        enable_csv_dump=False,
        show_progress=False,
        emit_diagnostics=False,
    )


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
        return {
            "name": name,
            "overrides": overrides,
            "trade_count": summary.get("trade_count"),
            "closed_trade_count": summary.get("closed_trade_count"),
            "net_pnl_after_costs": summary.get("net_pnl_after_costs"),
            "return_pct": summary.get("return_pct"),
            "max_drawdown_pct": summary.get("max_drawdown_pct"),
            "win_rate_pct": summary.get("win_rate_pct"),
            "profit_factor": summary.get("profit_factor"),
            "take_profit_count": summary.get("take_profit_count"),
            "stop_loss_count": summary.get("stop_loss_count"),
            "fees_paid": summary.get("fees_paid"),
            "slippage_cost": summary.get("slippage_cost"),
            "decision_action_counts": summary.get("decision_action_counts"),
            "decision_reason_top": summary.get("decision_reason_top"),
            "decision_regime_signal_summary": summary.get("decision_regime_signal_summary"),
        }
    finally:
        restore_overrides(originals)


def main():
    seed_bt = build_seed_backtester()
    baseline_min = float(config.MIN_SIGNAL_TARGET_RATIO)
    candidates = [
        ("baseline", {
            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": baseline_min,
            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": baseline_min,
            "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.35,
        }),
        ("regime_min_005", {
            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": 0.05,
            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": 0.05,
            "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.35,
        }),
        ("high_vol_mult_045", {
            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": baseline_min,
            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": baseline_min,
            "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.45,
        }),
        ("high_vol_mult_050", {
            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": baseline_min,
            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": baseline_min,
            "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.50,
        }),
        ("regime_min_005_high_vol_mult_045", {
            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": 0.05,
            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": 0.05,
            "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.45,
        }),
        ("regime_min_005_high_vol_mult_050", {
            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": 0.05,
            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": 0.05,
            "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.50,
        }),
    ]
    results = []
    for name, overrides in candidates:
        log_info(f"运行 SmallTarget A/B: {name} {overrides}")
        results.append(run_candidate(seed_bt, name, overrides))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOGS_DIR, f"smalltarget_regime_ab_{ts}.json")
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "results": results}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    log_info(f"SmallTarget regime A/B 报告: {path}")


if __name__ == "__main__":
    main()
