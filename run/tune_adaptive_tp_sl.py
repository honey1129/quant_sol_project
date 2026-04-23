import contextlib
import io

from backtest.backtest import Backtester
from config import config
from utils.utils import log_info


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
    log_info("准备共享回测缓存，用于参数扫描")
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
        summary["name"] = name
        summary["overrides"] = overrides
        return summary
    finally:
        restore_overrides(originals)


def main():
    candidates = [
        (
            "baseline",
            {
                "ATR_TAKE_PROFIT_MULTIPLIER": config.ATR_TAKE_PROFIT_MULTIPLIER,
                "ATR_STOP_LOSS_MULTIPLIER": config.ATR_STOP_LOSS_MULTIPLIER,
                "VOLATILITY_TAKE_PROFIT_MULTIPLIER": config.VOLATILITY_TAKE_PROFIT_MULTIPLIER,
                "VOLATILITY_STOP_LOSS_MULTIPLIER": config.VOLATILITY_STOP_LOSS_MULTIPLIER,
                "ADAPTIVE_TAKE_PROFIT_MIN": config.ADAPTIVE_TAKE_PROFIT_MIN,
                "ADAPTIVE_TAKE_PROFIT_MAX": config.ADAPTIVE_TAKE_PROFIT_MAX,
                "ADAPTIVE_STOP_LOSS_MIN": config.ADAPTIVE_STOP_LOSS_MIN,
                "ADAPTIVE_STOP_LOSS_MAX": config.ADAPTIVE_STOP_LOSS_MAX,
            },
        ),
        (
            "balanced_wider",
            {
                "ATR_TAKE_PROFIT_MULTIPLIER": 3.5,
                "ATR_STOP_LOSS_MULTIPLIER": 1.8,
                "VOLATILITY_TAKE_PROFIT_MULTIPLIER": 5.0,
                "VOLATILITY_STOP_LOSS_MULTIPLIER": 2.2,
                "ADAPTIVE_TAKE_PROFIT_MIN": 0.007,
                "ADAPTIVE_TAKE_PROFIT_MAX": 0.035,
                "ADAPTIVE_STOP_LOSS_MIN": 0.0045,
                "ADAPTIVE_STOP_LOSS_MAX": 0.022,
            },
        ),
        (
            "trend_wider",
            {
                "ATR_TAKE_PROFIT_MULTIPLIER": 4.0,
                "ATR_STOP_LOSS_MULTIPLIER": 1.8,
                "VOLATILITY_TAKE_PROFIT_MULTIPLIER": 6.0,
                "VOLATILITY_STOP_LOSS_MULTIPLIER": 2.2,
                "ADAPTIVE_TAKE_PROFIT_MIN": 0.008,
                "ADAPTIVE_TAKE_PROFIT_MAX": 0.04,
                "ADAPTIVE_STOP_LOSS_MIN": 0.0045,
                "ADAPTIVE_STOP_LOSS_MAX": 0.022,
            },
        ),
        (
            "fewer_trades",
            {
                "ATR_TAKE_PROFIT_MULTIPLIER": 5.0,
                "ATR_STOP_LOSS_MULTIPLIER": 2.2,
                "VOLATILITY_TAKE_PROFIT_MULTIPLIER": 7.0,
                "VOLATILITY_STOP_LOSS_MULTIPLIER": 2.6,
                "ADAPTIVE_TAKE_PROFIT_MIN": 0.009,
                "ADAPTIVE_TAKE_PROFIT_MAX": 0.045,
                "ADAPTIVE_STOP_LOSS_MIN": 0.0055,
                "ADAPTIVE_STOP_LOSS_MAX": 0.025,
            },
        ),
        (
            "tight_stop",
            {
                "ATR_TAKE_PROFIT_MULTIPLIER": 4.0,
                "ATR_STOP_LOSS_MULTIPLIER": 1.4,
                "VOLATILITY_TAKE_PROFIT_MULTIPLIER": 6.0,
                "VOLATILITY_STOP_LOSS_MULTIPLIER": 1.8,
                "ADAPTIVE_TAKE_PROFIT_MIN": 0.008,
                "ADAPTIVE_TAKE_PROFIT_MAX": 0.04,
                "ADAPTIVE_STOP_LOSS_MIN": 0.004,
                "ADAPTIVE_STOP_LOSS_MAX": 0.018,
            },
        ),
    ]

    seed_bt = build_seed_backtester()

    results = []
    for name, overrides in candidates:
        result = run_candidate(seed_bt, name, overrides)
        results.append(result)

    results.sort(
        key=lambda item: (
            item["final_equity"],
            -item["max_drawdown_pct"],
            -item["trade_count"],
        ),
        reverse=True,
    )

    print("name,final_equity,return_pct,max_drawdown_pct,trade_count,tp_count,sl_count,fees,slippage")
    for item in results:
        print(
            f"{item['name']},"
            f"{item['final_equity']:.2f},"
            f"{item['return_pct']:.2f},"
            f"{item['max_drawdown_pct']:.2f},"
            f"{item['trade_count']},"
            f"{item['take_profit_count']},"
            f"{item['stop_loss_count']},"
            f"{item['fees_paid']:.2f},"
            f"{item['slippage_cost']:.2f}"
        )

    best = results[0]
    print("\nrecommended_overrides")
    for key, value in best["overrides"].items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
