import hashlib
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from utils.utils import BASE_DIR


TARGET_FLAT = 0
TARGET_LONG = 1
TARGET_SHORT = 2
TARGET_NAMES = {
    TARGET_FLAT: "flat",
    TARGET_LONG: "long",
    TARGET_SHORT: "short",
}
DEFAULT_EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "research",
    "experiments",
    "directional_v2",
)
DEFAULT_SPEC_PATH = os.path.join(DEFAULT_EXPERIMENT_DIR, "spec.json")
DEFAULT_HASH_PATH = os.path.join(DEFAULT_EXPERIMENT_DIR, "spec.sha256")


def load_experiment_spec(path=DEFAULT_SPEC_PATH):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or not payload.get("experiment_id"):
        raise ValueError(f"invalid directional-v2 experiment spec: {path}")
    return payload


def spec_sha256(path=DEFAULT_SPEC_PATH):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_spec(spec_path=DEFAULT_SPEC_PATH, hash_path=DEFAULT_HASH_PATH):
    with open(hash_path, "r", encoding="utf-8") as file:
        expected = file.read().strip().split()[0]
    actual = spec_sha256(spec_path)
    if actual != expected:
        raise RuntimeError(
            "directional-v2 spec hash mismatch; create a new experiment id instead of "
            "changing the frozen holdout spec"
        )
    return actual


def _safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _empty_quality(outcome="INVALID"):
    return {
        "outcome": outcome,
        "exit_bars": 0,
        "gross_return": 0.0,
        "net_return": 0.0,
        "mfe": 0.0,
        "mae": 0.0,
        "score": 0.0,
    }


def simulate_direction_quality(entry_price, future_bars, direction, label_spec):
    entry_price = _safe_float(entry_price)
    if entry_price is None or entry_price <= 0 or direction not in {"long", "short"}:
        return _empty_quality()

    take_profit = max(0.0, float(label_spec["take_profit_pct"]))
    stop_loss = max(0.0, float(label_spec["stop_loss_pct"]))
    round_trip_cost = max(0.0, float(label_spec["round_trip_fee_rate"])) + max(
        0.0,
        float(label_spec["round_trip_slippage_rate"]),
    )
    mae_penalty = max(0.0, float(label_spec["mae_penalty"]))
    last_close = entry_price
    mfe = 0.0
    mae = 0.0
    outcome = "TIMEOUT"
    gross_return = 0.0
    exit_bars = len(future_bars)

    for bar_no, (_, bar) in enumerate(future_bars.iterrows(), start=1):
        high = _safe_float(bar.get("5m_high"))
        low = _safe_float(bar.get("5m_low"))
        close = _safe_float(bar.get("5m_close"))
        if high is None or low is None:
            continue
        if close is not None:
            last_close = close

        if direction == "long":
            mfe = max(mfe, (high - entry_price) / entry_price)
            mae = max(mae, (entry_price - low) / entry_price)
            hit_stop = low <= entry_price * (1.0 - stop_loss)
            hit_take = high >= entry_price * (1.0 + take_profit)
        else:
            mfe = max(mfe, (entry_price - low) / entry_price)
            mae = max(mae, (high - entry_price) / entry_price)
            hit_stop = high >= entry_price * (1.0 + stop_loss)
            hit_take = low <= entry_price * (1.0 - take_profit)

        if hit_stop:
            outcome = "SL"
            gross_return = -stop_loss
            exit_bars = bar_no
            break
        if hit_take:
            outcome = "TP"
            gross_return = take_profit
            exit_bars = bar_no
            break

    if outcome == "TIMEOUT":
        gross_return = (
            (last_close - entry_price) / entry_price
            if direction == "long"
            else (entry_price - last_close) / entry_price
        )

    net_return = gross_return - round_trip_cost
    return {
        "outcome": outcome,
        "exit_bars": int(exit_bars),
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "mfe": float(max(mfe, 0.0)),
        "mae": float(max(mae, 0.0)),
        "score": float(net_return - mae_penalty * max(mae, 0.0)),
    }


def choose_directional_target(long_quality, short_quality, label_spec):
    minimum_net_return = float(label_spec["minimum_net_return"])
    minimum_score_gap = float(label_spec["minimum_direction_score_gap"])
    eligible = []
    for target, direction, quality in (
        (TARGET_LONG, "long", long_quality),
        (TARGET_SHORT, "short", short_quality),
    ):
        if (
            quality.get("outcome") in {"TP", "TIMEOUT"}
            and float(quality.get("net_return", 0.0)) >= minimum_net_return
        ):
            eligible.append((target, direction, float(quality.get("score", 0.0))))

    if not eligible:
        return TARGET_FLAT, "flat", "no_positive_edge"
    eligible.sort(key=lambda item: item[2], reverse=True)
    best = eligible[0]
    runner_up_score = eligible[1][2] if len(eligible) > 1 else 0.0
    if best[2] - runner_up_score < minimum_score_gap:
        return TARGET_FLAT, "flat", "ambiguous_direction"
    return best[0], best[1], "accepted"


def build_directional_labels(data, spec):
    labeled = data.copy()
    label_spec = spec["label"]
    lookahead = int(label_spec["lookahead_bars"])
    records = []

    for index in range(len(labeled)):
        record = {
            "target_v2": np.nan,
            "label_v2_direction": "none",
            "label_v2_reason": "no_lookahead",
            "label_v2_entry_price": np.nan,
            "label_v2_long_outcome": "NO_LOOKAHEAD",
            "label_v2_short_outcome": "NO_LOOKAHEAD",
            "label_v2_long_net_return": np.nan,
            "label_v2_short_net_return": np.nan,
            "label_v2_long_score": np.nan,
            "label_v2_short_score": np.nan,
        }
        if index + lookahead >= len(labeled):
            records.append(record)
            continue

        entry_row = labeled.iloc[index + 1]
        entry_price = _safe_float(entry_row.get("5m_open"))
        future_bars = labeled.iloc[index + 1:index + lookahead + 1]
        long_quality = simulate_direction_quality(
            entry_price,
            future_bars,
            "long",
            label_spec,
        )
        short_quality = simulate_direction_quality(
            entry_price,
            future_bars,
            "short",
            label_spec,
        )
        target, direction, reason = choose_directional_target(
            long_quality,
            short_quality,
            label_spec,
        )
        record.update({
            "target_v2": target,
            "label_v2_direction": direction,
            "label_v2_reason": reason,
            "label_v2_entry_price": entry_price,
            "label_v2_long_outcome": long_quality["outcome"],
            "label_v2_short_outcome": short_quality["outcome"],
            "label_v2_long_net_return": long_quality["net_return"],
            "label_v2_short_net_return": short_quality["net_return"],
            "label_v2_long_score": long_quality["score"],
            "label_v2_short_score": short_quality["score"],
        })
        records.append(record)

    label_frame = pd.DataFrame(records, index=labeled.index)
    for column in label_frame:
        labeled[column] = label_frame[column]
    labeled = labeled[labeled["target_v2"].notna()].copy()
    labeled["target_v2"] = labeled["target_v2"].astype(int)
    return labeled


def select_directional_signal(probabilities, signal_spec):
    values = {
        "flat": max(0.0, float(probabilities.get("flat", 0.0))),
        "long": max(0.0, float(probabilities.get("long", 0.0))),
        "short": max(0.0, float(probabilities.get("short", 0.0))),
    }
    total = sum(values.values())
    if total <= 0 or not all(math.isfinite(value) for value in values.values()):
        return {"direction": "flat", "reason": "invalid_probabilities", **values}
    values = {key: value / total for key, value in values.items()}
    direction = "long" if values["long"] >= values["short"] else "short"
    other_direction = "short" if direction == "long" else "long"
    blocked = {str(item).lower() for item in signal_spec.get("hard_blocked_directions", [])}

    reason = "accepted"
    if direction in blocked:
        reason = "blocked_direction"
    elif values[direction] < float(signal_spec["minimum_direction_probability"]):
        reason = "probability_below_minimum"
    elif values[direction] - values[other_direction] < float(
        signal_spec["minimum_direction_probability_gap"]
    ):
        reason = "direction_gap_below_minimum"
    elif values[direction] - values["flat"] < float(signal_spec["minimum_advantage_over_flat"]):
        reason = "flat_advantage_below_minimum"

    return {
        "direction": direction if reason == "accepted" else "flat",
        "reason": reason,
        **values,
    }


def _utc_timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def forward_holdout_status(spec, *, now=None, closed_trades=0):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    holdout = spec["holdout"]
    start = _utc_timestamp(holdout["start_inclusive"])
    minimum_end = _utc_timestamp(holdout["minimum_end_exclusive"])
    minimum_trades = int(holdout["minimum_closed_trades"])

    if now < start:
        state = "FROZEN_WAITING_START"
    elif now < minimum_end:
        state = "COLLECTING_FORWARD_DATA"
    elif int(closed_trades) < minimum_trades:
        state = "WATCH_INSUFFICIENT_TRADES"
    else:
        state = "READY_FOR_FINAL_EVALUATION"
    return {
        "experiment_id": spec["experiment_id"],
        "state": state,
        "now": now.isoformat(),
        "holdout_start": start.isoformat(),
        "minimum_end": minimum_end.isoformat(),
        "closed_trades": int(closed_trades),
        "minimum_closed_trades": minimum_trades,
        "final_evaluation_allowed": state == "READY_FOR_FINAL_EVALUATION",
    }
