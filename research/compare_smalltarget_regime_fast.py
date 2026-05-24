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


def run_candidate(seed_bt, data, name, overrides):
    originals = apply_overrides(overrides)
    try:
        bt = Backtester(
            "multi_period",
            seed_bt.window,
            data_dict=seed_bt.data_dict,
            reward_risk=seed_bt.reward_risk,
            precomputed_data=data,
            feature_cols=seed_bt.feature_cols,
            models=seed_bt.models,
            model_weights=seed_bt.model_weights,
            funding_history=seed_bt.funding_history,
            enable_csv_dump=False,
            show_progress=False,
            emit_diagnostics=False,
        )
        bt.data[["long_prob", "short_prob"]] = data[["long_prob", "short_prob"]]
        original_predict_row = bt._predict_row
        bt._predict_row = lambda row: (row["long_prob"], row["short_prob"])
        with contextlib.redirect_stdout(io.StringIO()):
            summary = bt.run_backtest()
        bt._predict_row = original_predict_row
        return {
            "name": name,
            "overrides": overrides,
            "trade_count": summary.get("trade_count"),
            "closed_trade_count": summary.get("closed_trade_count"),
            "net_pnl_after_costs": summary.get("net_pnl_after_costs"),
            "max_drawdown_pct": summary.get("max_drawdown_pct"),
            "win_rate_pct": summary.get("win_rate_pct"),
            "profit_factor": summary.get("profit_factor"),
            "take_profit_count": summary.get("take_profit_count"),
            "stop_loss_count": summary.get("stop_loss_count"),
            "decision_action_counts": summary.get("decision_action_counts"),
            "decision_reason_top": summary.get("decision_reason_top"),
        }
    finally:
        restore_overrides(originals)


def main():
    base_interval = config.INTERVALS[0] if config.INTERVALS else "5m"
    window = config.WINDOWS.get(base_interval, 1000)
    seed_bt = Backtester("multi_period", window, enable_csv_dump=False, show_progress=False, emit_diagnostics=False)
    data = seed_bt.data.tail(int(os.getenv("SMALLTARGET_AB_ROWS", "3000"))).copy()
    log_info(f"预计算快速 A/B 信号: rows={len(data)}")
    data[["long_prob", "short_prob"]] = data.apply(seed_bt._predict_row, axis=1, result_type="expand")
    baseline_min = float(config.MIN_SIGNAL_TARGET_RATIO)
    candidates = [
        ("baseline", {"REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": baseline_min, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": baseline_min, "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.35}),
        ("regime_min_005", {"REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": 0.05, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": 0.05, "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.35}),
        ("high_vol_mult_045", {"REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": baseline_min, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": baseline_min, "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.45}),
        ("high_vol_mult_050", {"REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": baseline_min, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": baseline_min, "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.50}),
        ("regime_min_005_high_vol_mult_045", {"REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": 0.05, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": 0.05, "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.45}),
        ("regime_min_005_high_vol_mult_050", {"REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": 0.05, "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": 0.05, "REGIME_HIGH_VOL_TARGET_MULTIPLIER": 0.50}),
    ]
    path = os.path.join(LOGS_DIR, f"smalltarget_regime_ab_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report = {"created_at": datetime.now().isoformat(timespec="seconds"), "rows": len(data), "start": str(data.index.min()), "end": str(data.index.max()), "results": []}
    for name, overrides in candidates:
        log_info(f"运行快速 SmallTarget A/B: {name}")
        report["results"].append(run_candidate(seed_bt, data, name, overrides))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    log_info(f"快速 SmallTarget regime A/B 报告: {path}")


if __name__ == "__main__":
    main()
