from collections import defaultdict

import pandas as pd

from backtest.backtest import resolve_intrabar_tp_sl
from core.trend_filter import derive_trend_context
from research.directional_v2 import select_directional_signal


def _execution_price(reference_price, side, slippage_bps):
    ratio = float(slippage_bps) / 10000.0
    return (
        float(reference_price) * (1.0 + ratio)
        if side == "buy"
        else float(reference_price) * (1.0 - ratio)
    )


def trend_baseline_probabilities(data, spec):
    rows = []
    for _, row in data.iterrows():
        trend = derive_trend_context(
            row,
            interval="15m",
            fast_col="15m_ema_20",
            slow_col="15m_ema_60",
            min_gap=0.0,
        ).get("trend_bias")
        rows.append({
            "flat": 0.0 if trend in {"long", "short"} else 1.0,
            "long": 1.0 if trend == "long" else 0.0,
            "short": 1.0 if trend == "short" else 0.0,
        })
    return pd.DataFrame(rows, index=data.index)


def _performance_bucket(trades):
    pnls = [float(trade["net_pnl_after_costs"]) for trade in trades]
    wins = [value for value in pnls if value > 0]
    losses = [value for value in pnls if value < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "closed_trade_count": len(pnls),
        "winning_trade_count": len(wins),
        "losing_trade_count": len(losses),
        "win_rate_pct": len(wins) / len(pnls) * 100.0 if pnls else 0.0,
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": (
            float(gross_profit / gross_loss)
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        ),
        "net_pnl_after_costs": float(sum(pnls)),
    }


def run_directional_backtest(data, probabilities, spec):
    if len(data) < 2:
        raise ValueError("directional-v2 holdout data requires at least two rows")
    probabilities = probabilities.reindex(data.index)
    if probabilities[["flat", "long", "short"]].isna().any().any():
        raise ValueError("directional-v2 probabilities do not cover holdout data")

    execution = spec["execution"]
    signal_spec = spec["signal"]
    initial_balance = float(execution["initial_balance"])
    balance = initial_balance
    peak_equity = initial_balance
    max_drawdown = 0.0
    position = 0.0
    entry_price = 0.0
    entry_balance = initial_balance
    entry_time = None
    hold_bars = 0
    fees_paid = 0.0
    slippage_cost = 0.0
    trades = []
    max_hold_bars = int(execution["maximum_hold_bars"])
    fee_rate = float(execution["fee_rate_per_side"])
    slippage_bps = float(execution["slippage_bps_per_side"])
    take_profit = float(execution["take_profit_pct"])
    stop_loss = float(execution["stop_loss_pct"])

    def close_position(ts, reference_price, reason):
        nonlocal balance, position, entry_price, entry_balance, entry_time
        nonlocal hold_bars, fees_paid, slippage_cost
        side = "sell" if position > 0 else "buy"
        exit_price = _execution_price(reference_price, side, slippage_bps)
        qty = abs(position)
        pnl = (exit_price - entry_price) * position
        exit_fee = qty * exit_price * fee_rate
        balance += pnl - exit_fee
        fees_paid += exit_fee
        slippage_cost += qty * abs(exit_price - float(reference_price))
        trades.append({
            "entry_time": entry_time,
            "exit_time": ts,
            "direction": "long" if position > 0 else "short",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "quantity": float(qty),
            "hold_bars": int(hold_bars),
            "reason": reason,
            "net_pnl_after_costs": float(balance - entry_balance),
        })
        position = 0.0
        entry_price = 0.0
        entry_balance = balance
        entry_time = None
        hold_bars = 0

    for index in range(1, len(data)):
        signal_row = probabilities.iloc[index - 1]
        bar = data.iloc[index]
        bar_ts = data.index[index]
        bar_open = float(bar["5m_open"])
        bar_high = float(bar["5m_high"])
        bar_low = float(bar["5m_low"])
        bar_close = float(bar["5m_close"])
        closed_this_bar = False

        if position != 0:
            hit = resolve_intrabar_tp_sl(
                position,
                entry_price,
                bar_open,
                bar_high,
                bar_low,
                take_profit,
                stop_loss,
                worst_case=True,
            )
            if hit:
                close_position(bar_ts, hit["trigger_price"], hit["reason"])
                closed_this_bar = True

        if position == 0 and not closed_this_bar:
            signal = select_directional_signal(signal_row.to_dict(), signal_spec)
            if signal["direction"] in {"long", "short"}:
                side = "buy" if signal["direction"] == "long" else "sell"
                entry_balance = balance
                entry_price = _execution_price(bar_open, side, slippage_bps)
                notional = (
                    balance
                    * float(execution["position_notional_ratio"])
                    * float(execution["leverage"])
                )
                qty = notional / entry_price
                position = qty if signal["direction"] == "long" else -qty
                entry_fee = qty * entry_price * fee_rate
                balance -= entry_fee
                fees_paid += entry_fee
                slippage_cost += qty * abs(entry_price - bar_open)
                entry_time = bar_ts
                hold_bars = 0

                hit = resolve_intrabar_tp_sl(
                    position,
                    entry_price,
                    bar_open,
                    bar_high,
                    bar_low,
                    take_profit,
                    stop_loss,
                    worst_case=True,
                )
                if hit:
                    close_position(bar_ts, hit["trigger_price"], hit["reason"])
                    closed_this_bar = True

        if position != 0:
            hold_bars += 1
            if hold_bars >= max_hold_bars:
                close_position(bar_ts, bar_close, "TIMEOUT")

        equity = balance
        if position != 0:
            equity += (bar_close - entry_price) * position
        peak_equity = max(peak_equity, equity)
        if peak_equity > 0:
            max_drawdown = min(max_drawdown, (equity - peak_equity) / peak_equity)

    if position != 0:
        last = data.iloc[-1]
        close_position(data.index[-1], float(last["5m_close"]), "FINAL_CLOSE")
        peak_equity = max(peak_equity, balance)
        max_drawdown = min(max_drawdown, (balance - peak_equity) / peak_equity)

    by_direction = {}
    for direction in ("long", "short"):
        direction_trades = [trade for trade in trades if trade["direction"] == direction]
        if direction_trades:
            by_direction[direction] = _performance_bucket(direction_trades)

    weekly_pnl = defaultdict(float)
    for trade in trades:
        timestamp = pd.Timestamp(trade["exit_time"])
        iso = timestamp.isocalendar()
        weekly_pnl[f"{iso.year}-W{iso.week:02d}"] += float(trade["net_pnl_after_costs"])
    positive_weeks = sum(1 for value in weekly_pnl.values() if value > 0)
    performance = _performance_bucket(trades)
    performance.update({
        "initial_balance": initial_balance,
        "final_balance": float(balance),
        "return_pct": float((balance - initial_balance) / initial_balance * 100.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "fees_paid": float(fees_paid),
        "slippage_cost": float(slippage_cost),
        "positive_week_ratio": (
            float(positive_weeks / len(weekly_pnl))
            if weekly_pnl
            else 0.0
        ),
        "weekly_pnl": dict(sorted(weekly_pnl.items())),
        "by_direction": by_direction,
        "trades": trades,
    })
    return performance


def evaluate_forward_result(summary, baseline, spec):
    holdout = spec["holdout"]
    gates = spec["evaluation"]
    closed = int(summary["closed_trade_count"])
    if closed < int(holdout["minimum_closed_trades"]):
        return {
            "verdict": gates["result_if_insufficient_sample"],
            "reason": "insufficient_closed_trades",
            "failed": [],
        }

    checks = {
        "net_pnl_after_costs": summary["net_pnl_after_costs"] > float(
            gates["minimum_net_pnl_after_costs"]
        ),
        "profit_factor": summary["profit_factor"] >= float(gates["minimum_profit_factor"]),
        "max_drawdown_pct": summary["max_drawdown_pct"] >= float(gates["maximum_drawdown_pct"]),
        "positive_week_ratio": summary["positive_week_ratio"] >= float(
            gates["minimum_positive_week_ratio"]
        ),
        "beat_no_trade": summary["net_pnl_after_costs"] > 0,
        "beat_trend_baseline": summary["net_pnl_after_costs"] > baseline["net_pnl_after_costs"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "verdict": (
            gates["result_if_any_decisive_gate_fails"]
            if failed
            else gates["result_if_all_gates_pass"]
        ),
        "reason": "failed_forward_gates" if failed else "passed_all_forward_gates",
        "failed": failed,
        "checks": checks,
    }
