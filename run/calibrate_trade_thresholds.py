import argparse
import contextlib
import io
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.backtest import Backtester
from config import config
from run import training_diagnostics as td
from utils.utils import LOGS_DIR, log_info


PROBABILITY_DIRECTIONS = {
    "long": {
        "label": 1,
        "raw_col": "long_prob",
        "calibrated_col": "long_prob_calibrated",
    },
    "short": {
        "label": 0,
        "raw_col": "short_prob",
        "calibrated_col": "short_prob_calibrated",
    },
}


def parse_float_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        result.append(float(item))
    return result


def parse_int_list(value, default):
    if value is None or str(value).strip() == "":
        return list(default)
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


@contextlib.contextmanager
def temporary_env(overrides):
    originals = {}
    missing = set()
    for key, value in overrides.items():
        if key in os.environ:
            originals[key] = os.environ[key]
        else:
            missing.add(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key in overrides:
            if key in missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = originals[key]


def apply_overrides(overrides):
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)
    return originals


def restore_overrides(originals):
    for key, value in originals.items():
        setattr(config, key, value)


def load_diagnostic_data(model_root, split, rows, raw_labels=False):
    bundle, seed_bt, labeled = load_all_diagnostic_data(model_root, raw_labels=raw_labels)
    selected = td.select_split(labeled, bundle["metadata"], split, rows)
    if selected.empty:
        raise RuntimeError("校准样本为空，请调整 --split 或 --rows")
    return bundle, seed_bt, selected


def load_all_diagnostic_data(model_root, raw_labels=False):
    bundle = td.load_model_bundle(model_root)
    seed_bt = td.create_seed_backtester(bundle)
    labeled = td.create_labels(
        seed_bt.data.copy(),
        future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
        threshold=float(config.MODEL_LABEL_THRESHOLD),
        tradable_only=not bool(raw_labels),
    )
    labeled = td.enrich_regime_context(labeled)
    labeled = td.add_predictions(labeled, bundle)
    return bundle, seed_bt, labeled


def calibration_bins(data, direction, bins, prob_col=None):
    prob_col = prob_col or PROBABILITY_DIRECTIONS[direction]["raw_col"]
    target = direction_target_series(data, direction).astype(float)
    probs = data[prob_col].astype(float)
    brier = float(np.mean((probs.to_numpy() - target.to_numpy()) ** 2))

    rows = []
    ece = 0.0
    total = max(1, len(data))
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper >= 1.0:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        bucket = data[mask]
        if bucket.empty:
            continue
        bucket_target = target[mask]
        avg_prob = float(probs[mask].mean())
        hit_rate = float(bucket_target.mean())
        rows.append({
            "bin": f"[{lower:.2f},{upper:.2f}{']' if upper >= 1.0 else ')'}",
            "rows": int(len(bucket)),
            "avg_prob": avg_prob,
            "hit_rate": hit_rate,
            "error": float(avg_prob - hit_rate),
        })
        ece += len(bucket) / total * abs(avg_prob - hit_rate)

    return {
        "direction": direction,
        "rows": int(len(data)),
        "brier": brier,
        "ece": float(ece),
        "bins": rows,
    }


def build_probability_calibration_report(data, bins):
    return build_probability_calibration_report_for_columns(
        data,
        bins,
        {
            "long": PROBABILITY_DIRECTIONS["long"]["raw_col"],
            "short": PROBABILITY_DIRECTIONS["short"]["raw_col"],
        },
    )


def build_probability_calibration_report_for_columns(data, bins, prob_cols):
    report = {
        "all": {
            "long": calibration_bins(data, "long", bins, prob_cols["long"]),
            "short": calibration_bins(data, "short", bins, prob_cols["short"]),
        },
        "by_regime": {},
    }
    for regime, group in data.groupby("diag_regime", dropna=False):
        regime = str(regime or "unknown")
        report["by_regime"][regime] = {
            "long": calibration_bins(group, "long", bins, prob_cols["long"]),
            "short": calibration_bins(group, "short", bins, prob_cols["short"]),
        }
    return report


def directional_label_series(data):
    if "actual_label" in data.columns:
        return data["actual_label"].astype(int)
    return data["target"].astype(int)


def direction_target_series(data, direction):
    label = PROBABILITY_DIRECTIONS[direction]["label"]
    return (directional_label_series(data) == label).astype(int)


class ProbabilityCalibrator:
    def __init__(self, direction, method, model=None, fallback_reason=None, fitted_rows=0, positive_rows=0):
        self.direction = direction
        self.method = method
        self.model = model
        self.fallback_reason = fallback_reason
        self.fitted_rows = int(fitted_rows)
        self.positive_rows = int(positive_rows)

    @property
    def active(self):
        return self.model is not None and self.fallback_reason is None

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        if not self.active:
            return np.clip(values, 0.0, 1.0)
        if self.method == "sigmoid":
            calibrated = self.model.predict_proba(values.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.model.predict(values)
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)

    def summary(self):
        payload = {
            "direction": self.direction,
            "method": self.method,
            "active": bool(self.active),
            "fallback_reason": self.fallback_reason,
            "fitted_rows": int(self.fitted_rows),
            "positive_rows": int(self.positive_rows),
            "negative_rows": int(self.fitted_rows - self.positive_rows),
        }
        if self.active and self.method == "isotonic":
            payload["x_thresholds"] = [float(value) for value in self.model.X_thresholds_]
            payload["y_thresholds"] = [float(value) for value in self.model.y_thresholds_]
        elif self.active and self.method == "sigmoid":
            payload["coef"] = float(self.model.coef_[0][0])
            payload["intercept"] = float(self.model.intercept_[0])
        return payload


def fit_direction_probability_calibrator(data, direction, method):
    if method == "none":
        return ProbabilityCalibrator(direction, method, fallback_reason="disabled")

    prob_col = PROBABILITY_DIRECTIONS[direction]["raw_col"]
    label_col = "actual_label" if "actual_label" in data.columns else "target"
    cleaned = data[[prob_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return ProbabilityCalibrator(direction, method, fallback_reason="empty_calibration_data")

    y = direction_target_series(cleaned.rename(columns={label_col: "actual_label"}), direction).to_numpy()
    probs = cleaned[prob_col].astype(float).to_numpy()
    positive_rows = int(y.sum())
    fitted_rows = int(len(y))
    if positive_rows == 0 or positive_rows == fitted_rows:
        return ProbabilityCalibrator(
            direction,
            method,
            fallback_reason="single_class_calibration_data",
            fitted_rows=fitted_rows,
            positive_rows=positive_rows,
        )

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(probs, y)
    elif method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(solver="lbfgs")
        model.fit(probs.reshape(-1, 1), y)
    else:
        raise ValueError(f"不支持的概率校准方法: {method}")

    return ProbabilityCalibrator(
        direction,
        method,
        model=model,
        fitted_rows=fitted_rows,
        positive_rows=positive_rows,
    )


def fit_probability_calibrators(data, method):
    return {
        direction: fit_direction_probability_calibrator(data, direction, method)
        for direction in PROBABILITY_DIRECTIONS
    }


def apply_probability_calibrators(data, calibrators):
    calibrated = data.copy()
    for direction, spec in PROBABILITY_DIRECTIONS.items():
        raw_values = calibrated[spec["raw_col"]].astype(float).to_numpy()
        calibrated[spec["calibrated_col"]] = calibrators[direction].predict(raw_values)
    return calibrated


def use_calibrated_probability_columns(data):
    calibrated = data.copy()
    calibrated["long_prob_raw"] = calibrated["long_prob"]
    calibrated["short_prob_raw"] = calibrated["short_prob"]
    calibrated["long_prob"] = calibrated["long_prob_calibrated"]
    calibrated["short_prob"] = calibrated["short_prob_calibrated"]
    calibrated["prob_gap"] = (calibrated["long_prob"] - calibrated["short_prob"]).abs()
    calibrated["pred_label"] = np.where(
        calibrated["long_prob"] >= calibrated["short_prob"],
        1,
        0,
    )
    calibrated["pred_direction"] = calibrated["pred_label"].map(td.DIRECTION_LABELS)
    return calibrated


def split_metadata_available(metadata, split):
    if split == "all":
        return True
    if split == "validation":
        return bool(metadata.get("validation_start") and metadata.get("validation_end"))
    if split == "oos":
        return bool(metadata.get("oos_start"))
    return False


def select_probability_calibration_source(labeled, metadata, source, selected_data, rows=None):
    fallback_reason = None
    if source == "selected":
        calibration_data = selected_data.copy()
    elif not split_metadata_available(metadata, source):
        calibration_data = selected_data.copy()
        fallback_reason = f"{source}_metadata_missing_used_selected"
    else:
        calibration_data = td.select_split(labeled, metadata, source, rows)
        if calibration_data.empty:
            calibration_data = selected_data.copy()
            fallback_reason = f"{source}_empty_used_selected"

    return calibration_data, fallback_reason


def weak_signal_gate_counts(data, threshold_long, threshold_short, signal_min_prob_diff):
    long_prob = data["long_prob"].astype(float)
    short_prob = data["short_prob"].astype(float)
    gap = (long_prob - short_prob).abs()
    long_mask = (long_prob >= short_prob) & (long_prob > threshold_long) & (gap >= signal_min_prob_diff)
    short_mask = (short_prob > long_prob) & (short_prob > threshold_short) & (gap >= signal_min_prob_diff)
    return {
        "rows": int(len(data)),
        "long_gate_count": int(long_mask.sum()),
        "short_gate_count": int(short_mask.sum()),
        "gate_count": int((long_mask | short_mask).sum()),
        "long_gate_pct": float(long_mask.mean() * 100.0) if len(data) else 0.0,
        "short_gate_pct": float(short_mask.mean() * 100.0) if len(data) else 0.0,
        "gate_pct": float((long_mask | short_mask).mean() * 100.0) if len(data) else 0.0,
    }


def _probability_quantiles(values):
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {}
    return {
        "mean": float(series.mean()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def _direction_context_series(data):
    if "diag_trend_bias" in data:
        direction = data["diag_trend_bias"].fillna("unknown").astype(str).str.lower()
    else:
        direction = pd.Series(
            np.where(data["long_prob"].astype(float) >= data["short_prob"].astype(float), "long", "short"),
            index=data.index,
            dtype="object",
        )
    return direction.where(direction.isin(PROBABILITY_DIRECTIONS), "none")


def _direction_probability_series(data, direction):
    return pd.to_numeric(data[PROBABILITY_DIRECTIONS[direction]["raw_col"]], errors="coerce").fillna(0.0).astype(float)


def _direction_gap_series(data):
    long_prob = pd.to_numeric(data["long_prob"], errors="coerce").fillna(0.0).astype(float)
    short_prob = pd.to_numeric(data["short_prob"], errors="coerce").fillna(0.0).astype(float)
    return (long_prob - short_prob).abs()


def _gate_metrics_for_direction(data, direction, threshold, gap):
    rows = int(len(data))
    target = _trade_target_series_for_direction(data, direction)
    probs = _direction_probability_series(data, direction)
    gaps = _direction_gap_series(data)
    passed = (probs > float(threshold)) & (gaps >= float(gap))

    tp = int((passed & target).sum())
    fp = int((passed & ~target).sum())
    fn = int((~passed & target).sum())
    tn = int((~passed & ~target).sum())
    trade_rows = int(target.sum())
    no_trade_rows = int(rows - trade_rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(trade_rows, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "gap": float(gap),
        "rows": rows,
        "trade_rows": trade_rows,
        "no_trade_rows": no_trade_rows,
        "pass_rows": int(passed.sum()),
        "pass_pct": float(passed.mean() * 100.0) if rows else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / max(no_trade_rows, 1)),
    }


def _trade_target_series_for_direction(data, direction):
    if "actual_label" in data.columns:
        return direction_target_series(data, direction).astype(int) == 1
    if "target" in data.columns and "diag_trend_bias" in data.columns:
        trend_direction = data["diag_trend_bias"].fillna("unknown").astype(str).str.lower()
        return (data["target"].astype(int) == 1) & (trend_direction == str(direction).lower())
    return direction_target_series(data, direction).astype(int) == 1


def _rank_gate_metrics(item, min_trade_rows):
    enough_labels = 1 if int(item.get("trade_rows") or 0) >= int(min_trade_rows) else 0
    return (
        enough_labels,
        float(item.get("f1") or 0.0),
        float(item.get("precision") or 0.0),
        float(item.get("recall") or 0.0),
        -float(item.get("false_positive_rate") or 0.0),
        -float(item.get("pass_pct") or 0.0),
    )


def _regime_direction_action(base_gate, recommended_gate, probability_quantiles, *, min_group_rows, min_trade_rows):
    reasons = []
    if int(base_gate.get("rows") or 0) < int(min_group_rows):
        return "insufficient_rows", ["group_rows_below_minimum"]
    if int(base_gate.get("trade_rows") or 0) < int(min_trade_rows):
        return "insufficient_trade_labels", ["trade_labels_below_minimum"]

    true_trade = probability_quantiles.get("true_trade") or {}
    true_no_trade = probability_quantiles.get("true_no_trade") or {}
    false_positive = probability_quantiles.get("false_positive") or {}
    true_trade_p50 = true_trade.get("p50")
    true_no_trade_p50 = true_no_trade.get("p50")
    false_positive_p50 = false_positive.get("p50")
    base_threshold = float(base_gate.get("threshold") or 0.0)
    base_precision = float(base_gate.get("precision") or 0.0)
    base_recall = float(base_gate.get("recall") or 0.0)
    recommended_f1 = float(recommended_gate.get("f1") or 0.0)
    base_f1 = float(base_gate.get("f1") or 0.0)

    if base_recall == 0.0 and true_trade_p50 is not None and true_trade_p50 <= base_threshold:
        reasons.append("true_trade_prob_below_base_threshold")
    if (
        true_trade_p50 is not None
        and true_no_trade_p50 is not None
        and true_trade_p50 <= true_no_trade_p50
    ):
        reasons.append("true_trade_prob_not_above_no_trade")
    if (
        true_trade_p50 is not None
        and false_positive_p50 is not None
        and false_positive_p50 > true_trade_p50
    ):
        reasons.append("false_positive_prob_above_true_trade")

    if "true_trade_prob_not_above_no_trade" in reasons or "false_positive_prob_above_true_trade" in reasons:
        return "probability_inversion_retrain_or_group_calibrate", reasons
    if "true_trade_prob_below_base_threshold" in reasons:
        return "probability_scale_low_consider_group_threshold_or_calibration", reasons
    if recommended_f1 > base_f1 + 0.05:
        return "use_regime_direction_gate_candidate", ["candidate_improves_f1"]
    if base_precision < 0.20 and int(base_gate.get("fp") or 0) > int(base_gate.get("tp") or 0):
        return "tighten_or_retrain_false_positives", ["false_positives_exceed_true_positives"]
    return "keep_base_gate", ["base_gate_is_competitive"]


def build_regime_direction_probability_report(
    data,
    threshold_candidates,
    gap_candidates,
    *,
    min_group_rows=30,
    min_trade_rows=5,
    top_n=8,
):
    if data.empty:
        return {
            "enabled": True,
            "rows": 0,
            "groups": {},
            "recommended": [],
        }

    working = data.copy()
    working["_calibration_regime"] = (
        working.get("diag_regime", pd.Series("unknown", index=working.index))
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )
    working["_calibration_direction"] = _direction_context_series(working)
    working = working[working["_calibration_direction"].isin(PROBABILITY_DIRECTIONS)].copy()

    thresholds = unique_sorted(float(value) for value in threshold_candidates)
    gaps = unique_sorted(float(value) for value in gap_candidates)
    groups = {}
    recommended = []

    for (regime, direction), group in working.groupby(["_calibration_regime", "_calibration_direction"], sort=True):
        base_threshold = float(config.THRESHOLD_LONG if direction == "long" else config.THRESHOLD_SHORT)
        base_gap = float(config.SIGNAL_MIN_PROB_DIFF)
        local_thresholds = unique_sorted([*thresholds, base_threshold])
        local_gaps = unique_sorted([*gaps, base_gap])
        base_gate = _gate_metrics_for_direction(group, direction, base_threshold, base_gap)

        sweep = []
        for threshold in local_thresholds:
            for gap in local_gaps:
                sweep.append(_gate_metrics_for_direction(group, direction, threshold, gap))
        ranked = sorted(
            sweep,
            key=lambda item: _rank_gate_metrics(item, min_trade_rows),
            reverse=True,
        )
        recommended_gate = ranked[0] if ranked else base_gate

        target = _trade_target_series_for_direction(group, direction)
        probs = _direction_probability_series(group, direction)
        gaps_series = _direction_gap_series(group)
        base_passed = (probs > base_threshold) & (gaps_series >= base_gap)
        probability_quantiles = {
            "all": _probability_quantiles(probs),
            "true_trade": _probability_quantiles(probs[target]),
            "true_no_trade": _probability_quantiles(probs[~target]),
            "true_positive": _probability_quantiles(probs[target & base_passed]),
            "false_negative": _probability_quantiles(probs[target & ~base_passed]),
            "false_positive": _probability_quantiles(probs[~target & base_passed]),
            "true_negative": _probability_quantiles(probs[~target & ~base_passed]),
            "gap_all": _probability_quantiles(gaps_series),
        }
        action, action_reasons = _regime_direction_action(
            base_gate,
            recommended_gate,
            probability_quantiles,
            min_group_rows=min_group_rows,
            min_trade_rows=min_trade_rows,
        )
        key = f"{regime}:{direction}"
        summary = {
            "regime": str(regime),
            "direction": str(direction),
            "rows": int(len(group)),
            "base_gate": base_gate,
            "recommended_gate": recommended_gate,
            "recommended_action": action,
            "action_reasons": action_reasons,
            "probability_quantiles": probability_quantiles,
            "threshold_sweep_top": ranked[: int(top_n)],
        }
        groups[key] = summary
        if action not in {"insufficient_rows", "insufficient_trade_labels", "keep_base_gate"}:
            recommended.append({
                "group": key,
                "regime": str(regime),
                "direction": str(direction),
                "action": action,
                "reasons": action_reasons,
                "base_gate": base_gate,
                "recommended_gate": recommended_gate,
            })

    recommended = sorted(
        recommended,
        key=lambda item: (
            int(item["base_gate"].get("trade_rows") or 0),
            float(item["recommended_gate"].get("f1") or 0.0) - float(item["base_gate"].get("f1") or 0.0),
        ),
        reverse=True,
    )
    return {
        "enabled": True,
        "rows": int(len(working)),
        "min_group_rows": int(min_group_rows),
        "min_trade_rows": int(min_trade_rows),
        "threshold_candidates": [float(value) for value in thresholds],
        "gap_candidates": [float(value) for value in gaps],
        "base_gate": {
            "threshold_long": float(config.THRESHOLD_LONG),
            "threshold_short": float(config.THRESHOLD_SHORT),
            "signal_min_prob_diff": float(config.SIGNAL_MIN_PROB_DIFF),
        },
        "groups": groups,
        "recommended": recommended,
    }


def build_label_strength_candidates(lookaheads, take_profits, stop_losses):
    candidates = []
    for lookahead in lookaheads:
        for take_profit in take_profits:
            for stop_loss in stop_losses:
                candidates.append({
                    "name": f"lh{int(lookahead)}_tp{float(take_profit):.3f}_sl{float(stop_loss):.3f}",
                    "lookahead_bars": int(lookahead),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                })
    return candidates


def summarize_label_strength(data, candidate, *, target_trade_pct, min_trade_rows):
    rows = int(len(data))
    if rows == 0:
        return {
            **candidate,
            "rows": 0,
            "trade_rows": 0,
            "trade_pct": 0.0,
            "score": float("-inf"),
            "reason": "empty_label_sample",
        }

    target = data["target"].astype(int)
    trade_mask = target == 1
    trade_rows = int(trade_mask.sum())
    trade_pct = float(trade_rows / rows * 100.0)

    trend_bias = data.get("diag_trend_bias", pd.Series("unknown", index=data.index)).astype(str)
    regime = data.get("diag_regime", pd.Series("unknown", index=data.index)).astype(str)
    trade_direction_counts = {
        "long": int((trade_mask & (trend_bias == "long")).sum()),
        "short": int((trade_mask & (trend_bias == "short")).sum()),
        "neutral": int((trade_mask & (trend_bias == "neutral")).sum()),
        "unknown": int((trade_mask & (~trend_bias.isin(["long", "short", "neutral"]))).sum()),
    }
    directional_trade_rows = trade_direction_counts["long"] + trade_direction_counts["short"]
    if directional_trade_rows > 0:
        direction_imbalance_pct = abs(
            trade_direction_counts["long"] - trade_direction_counts["short"]
        ) / directional_trade_rows * 100.0
    else:
        direction_imbalance_pct = 100.0

    regime_rows = data.groupby(regime).size().sort_index()
    regime_trade_rows = data[trade_mask].groupby(regime[trade_mask]).size().sort_index()
    by_regime = {}
    for regime_name, regime_count in regime_rows.items():
        regime_trade_count = int(regime_trade_rows.get(regime_name, 0))
        by_regime[str(regime_name)] = {
            "rows": int(regime_count),
            "trade_rows": regime_trade_count,
            "trade_pct": float(regime_trade_count / max(int(regime_count), 1) * 100.0),
        }

    score = -abs(trade_pct - float(target_trade_pct)) - 0.25 * direction_imbalance_pct
    if trade_rows < int(min_trade_rows):
        score -= (int(min_trade_rows) - trade_rows) / max(int(min_trade_rows), 1) * 100.0

    return {
        **candidate,
        "rows": rows,
        "trade_rows": trade_rows,
        "no_trade_rows": int(rows - trade_rows),
        "trade_pct": trade_pct,
        "target_distribution": {
            str(k): int(v)
            for k, v in target.value_counts().sort_index().items()
        },
        "trade_direction_counts": trade_direction_counts,
        "direction_imbalance_pct": float(direction_imbalance_pct),
        "by_regime": by_regime,
        "score": float(score),
        "score_inputs": {
            "target_trade_pct": float(target_trade_pct),
            "min_trade_rows": int(min_trade_rows),
        },
    }


def build_label_strength_report(
    seed_data,
    metadata,
    split,
    rows,
    candidates,
    *,
    target_trade_pct,
    min_trade_rows,
):
    results = []
    for candidate in candidates:
        with temporary_env({
            "MODEL_LABEL_USE_REALISTIC": "1",
            "MODEL_LABEL_LOOKAHEAD_BARS": candidate["lookahead_bars"],
            "MODEL_LABEL_TAKE_PROFIT": candidate["take_profit"],
            "MODEL_LABEL_STOP_LOSS": candidate["stop_loss"],
        }):
            labeled = td.create_labels(
                seed_data.copy(),
                future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
                threshold=float(config.MODEL_LABEL_THRESHOLD),
                tradable_only=True,
            )
        labeled = td.enrich_regime_context(labeled)
        selected = td.select_split(labeled, metadata, split, rows)
        results.append(
            summarize_label_strength(
                selected,
                candidate,
                target_trade_pct=target_trade_pct,
                min_trade_rows=min_trade_rows,
            )
        )

    ranked = sorted(results, key=lambda item: item.get("score", float("-inf")), reverse=True)
    return {
        "enabled": True,
        "split": split,
        "rows_limit": rows,
        "candidate_count": int(len(candidates)),
        "target_trade_pct": float(target_trade_pct),
        "min_trade_rows": int(min_trade_rows),
        "candidates": results,
        "recommended": ranked[:10],
    }


def _label_strength_env(candidate):
    return {
        "MODEL_LABEL_USE_REALISTIC": "1",
        "MODEL_LABEL_LOOKAHEAD_BARS": candidate["lookahead_bars"],
        "MODEL_LABEL_TAKE_PROFIT": candidate["take_profit"],
        "MODEL_LABEL_STOP_LOSS": candidate["stop_loss"],
    }


def build_candidate_metadata(index, feature_cols, candidate, split_positions, final_train_end, purge_bars):
    train_end, validation_start, validation_end, oos_start = split_positions
    return {
        "schema_version": 2,
        "source": "run.calibrate_trade_thresholds",
        "target_schema": "binary_trade_quality",
        "label_mode": "binary_realistic",
        "label_use_realistic": True,
        "label_lookahead_bars": int(candidate["lookahead_bars"]),
        "label_take_profit": float(candidate["take_profit"]),
        "label_stop_loss": float(candidate["stop_loss"]),
        "feature_count": int(len(feature_cols)),
        "train_rows": int(train_end),
        "validation_rows": int(validation_end - validation_start),
        "oos_rows": int(len(index) - oos_start),
        "final_train_rows": int(final_train_end),
        "purge_bars": int(purge_bars),
        "train_start": index[0].isoformat(),
        "train_end": index[train_end - 1].isoformat(),
        "validation_start": index[validation_start].isoformat(),
        "validation_end": index[validation_end - 1].isoformat(),
        "oos_start": index[oos_start].isoformat(),
        "oos_end": index[-1].isoformat(),
    }


def fit_label_strength_candidate(seed_data, feature_cols, candidate, split):
    from train.train import build_time_splits, train_direction_quality_bundle

    with temporary_env(_label_strength_env(candidate)):
        labeled = td.create_labels(
            seed_data.copy(),
            future_window=int(config.MODEL_LABEL_FUTURE_WINDOW),
            threshold=float(config.MODEL_LABEL_THRESHOLD),
            tradable_only=True,
        )
    labeled = td.enrich_regime_context(labeled)
    missing_cols = [col for col in feature_cols if col not in labeled.columns]
    if missing_cols:
        raise RuntimeError(
            "标签强度模型训练缺少特征列: "
            + ",".join(str(col) for col in missing_cols[:12])
        )

    X = labeled[feature_cols].astype(float)
    y = labeled["target"].astype(int)
    purge_bars = max(int(config.MODEL_PURGE_BARS), int(candidate["lookahead_bars"]))
    originals = apply_overrides({"MODEL_PURGE_BARS": purge_bars})
    try:
        split_positions = build_time_splits(len(X))
    finally:
        restore_overrides(originals)
    train_end, _, validation_end, _ = split_positions
    if split == "validation":
        model_train_end = train_end
    else:
        model_train_end = validation_end if bool(config.MODEL_FINAL_TRAIN_ON_VALIDATION) else train_end

    models, _, _, sample_weight_summary, direction_quality_summary = train_direction_quality_bundle(
        X.iloc[:model_train_end].copy(),
        y.iloc[:model_train_end].copy(),
        sample_context=labeled.iloc[:model_train_end].copy(),
    )
    metadata = build_candidate_metadata(
        X.index,
        feature_cols,
        candidate,
        split_positions,
        final_train_end=model_train_end,
        purge_bars=purge_bars,
    )
    metadata["direction_quality_models"] = direction_quality_summary
    bundle = {
        "root_dir": PROJECT_ROOT,
        "models": models,
        "feature_cols": list(feature_cols),
        "metadata": metadata,
        "model_weights": dict(config.MODEL_WEIGHTS),
    }
    predicted = td.add_predictions(labeled.copy(), bundle)
    return predicted, metadata, sample_weight_summary


def _run_threshold_sweep_for_data(seed_bt, data, threshold_candidates, min_closed_trades):
    results = []
    for candidate in threshold_candidates:
        overrides = candidate["overrides"]
        summary = run_candidate(seed_bt, data.copy(), overrides)
        summary["name"] = candidate["name"]
        summary["overrides"] = overrides
        summary["weak_signal_gate_counts"] = weak_signal_gate_counts(
            data,
            overrides["THRESHOLD_LONG"],
            overrides["THRESHOLD_SHORT"],
            overrides["SIGNAL_MIN_PROB_DIFF"],
        )
        results.append(summary)
    return sorted(
        results,
        key=lambda item: score_candidate(item, int(min_closed_trades)),
        reverse=True,
    )


def build_label_strength_model_sweep(
    seed_bt,
    feature_cols,
    split,
    rows,
    label_summaries,
    threshold_candidates,
    *,
    top_n,
    probability_method,
    probability_source,
    probability_rows,
    bins,
    min_closed_trades,
    target_trade_pct,
    min_trade_rows,
):
    selected_label_summaries = list(label_summaries[:max(0, int(top_n))])
    if not selected_label_summaries:
        return {
            "enabled": False,
            "reason": "label_strength_model_sweep_top_n_is_zero",
        }

    report = {
        "enabled": True,
        "split": split,
        "rows_limit": rows,
        "label_candidate_count": int(len(selected_label_summaries)),
        "threshold_candidate_count": int(len(threshold_candidates)),
        "probability_calibration_method": probability_method,
        "probability_calibration_source": probability_source,
        "candidates": [],
        "recommended": [],
    }

    for label_summary in selected_label_summaries:
        candidate = {
            "name": label_summary["name"],
            "lookahead_bars": int(label_summary["lookahead_bars"]),
            "take_profit": float(label_summary["take_profit"]),
            "stop_loss": float(label_summary["stop_loss"]),
        }
        log_info(f"标签强度模型sweep候选: {candidate['name']}")
        predicted, metadata, sample_weight_summary = fit_label_strength_candidate(
            seed_bt.data.copy(),
            feature_cols,
            candidate,
            split,
        )
        selected = td.select_split(predicted, metadata, split, rows)
        if selected.empty:
            item = {
                **candidate,
                "skipped": True,
                "reason": "empty_selected_split",
                "metadata": metadata,
            }
            report["candidates"].append(item)
            continue

        probability_source_data, source_fallback = select_probability_calibration_source(
            predicted,
            metadata,
            probability_source,
            selected,
            rows=probability_rows,
        )
        calibrators = fit_probability_calibrators(probability_source_data, probability_method)
        probability_data = apply_probability_calibrators(selected, calibrators)
        threshold_data = (
            use_calibrated_probability_columns(probability_data)
            if probability_method != "none"
            else selected
        )
        ranked = _run_threshold_sweep_for_data(
            seed_bt,
            threshold_data,
            threshold_candidates,
            min_closed_trades,
        )
        label_summary_on_selected = summarize_label_strength(
            selected,
            candidate,
            target_trade_pct=target_trade_pct,
            min_trade_rows=min_trade_rows,
        )
        item = {
            **candidate,
            "skipped": False,
            "rows": int(len(selected)),
            "start": selected.index.min().isoformat(),
            "end": selected.index.max().isoformat(),
            "metadata": metadata,
            "sample_weight_summary": sample_weight_summary,
            "label_summary": label_summary_on_selected,
            "probability_calibration": {
                "method": probability_method,
                "source": probability_source,
                "source_rows": int(len(probability_source_data)),
                "source_start": (
                    probability_source_data.index.min().isoformat()
                    if not probability_source_data.empty
                    else None
                ),
                "source_end": (
                    probability_source_data.index.max().isoformat()
                    if not probability_source_data.empty
                    else None
                ),
                "source_fallback_reason": source_fallback,
                "used_for_threshold_sweep": probability_method != "none",
                "calibrators": {
                    direction: calibrator.summary()
                    for direction, calibrator in calibrators.items()
                },
                "raw": build_probability_calibration_report(selected, bins),
                "calibrated": (
                    build_probability_calibration_report_for_columns(
                        probability_data,
                        bins,
                        {
                            "long": PROBABILITY_DIRECTIONS["long"]["calibrated_col"],
                            "short": PROBABILITY_DIRECTIONS["short"]["calibrated_col"],
                        },
                    )
                    if probability_method != "none"
                    else None
                ),
            },
            "recommended_thresholds": ranked[:10],
            "best_threshold": ranked[0] if ranked else None,
        }
        report["candidates"].append(item)

    best_items = []
    for item in report["candidates"]:
        best = item.get("best_threshold")
        if not best:
            continue
        best_items.append({
            "label_name": item["name"],
            "lookahead_bars": int(item["lookahead_bars"]),
            "take_profit": float(item["take_profit"]),
            "stop_loss": float(item["stop_loss"]),
            "label_trade_pct": float(item["label_summary"].get("trade_pct", 0.0)),
            "threshold_name": best["name"],
            "threshold_overrides": best.get("overrides", {}),
            "closed_trade_count": int(best.get("closed_trade_count") or 0),
            "trade_count": int(best.get("trade_count") or 0),
            "net_pnl_after_costs": float(best.get("net_pnl_after_costs") or 0.0),
            "profit_factor": float(best.get("profit_factor") or 0.0),
            "win_rate_pct": float(best.get("win_rate_pct") or 0.0),
            "max_drawdown_pct": float(best.get("max_drawdown_pct") or 0.0),
            "_rank_score": score_candidate(best, min_closed_trades),
        })
    recommended = sorted(
        best_items,
        key=lambda item: item["_rank_score"],
        reverse=True,
    )[:10]
    for item in recommended:
        item.pop("_rank_score", None)
    report["recommended"] = recommended
    return report


def compact_summary(summary):
    keys = [
        "final_equity",
        "return_pct",
        "max_drawdown_pct",
        "trade_count",
        "closed_trade_count",
        "win_rate_pct",
        "profit_factor",
        "avg_win_loss_ratio",
        "net_pnl_after_costs",
        "net_return_pct_after_costs",
        "fees_paid",
        "slippage_cost",
        "funding_pnl",
        "take_profit_count",
        "stop_loss_count",
        "decision_action_counts",
        "decision_reason_top",
        "decision_direction_counts",
        "decision_regime_counts",
        "decision_regime_signal_summary",
        "decision_regime_reason_top",
        "decision_direction_reason_top",
        "decision_probability_quantiles",
        "decision_gate_config",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def run_candidate(seed_bt, data, overrides):
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
        original_predict_row = bt._predict_row
        bt._predict_row = lambda row: (row["long_prob"], row["short_prob"])
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                summary = bt.run_backtest()
        finally:
            bt._predict_row = original_predict_row
        return compact_summary(summary or {})
    finally:
        restore_overrides(originals)


def score_candidate(item, min_closed_trades):
    closed = int(item.get("closed_trade_count") or 0)
    net = float(item.get("net_pnl_after_costs") or 0.0)
    pf = float(item.get("profit_factor") or 0.0)
    drawdown = abs(float(item.get("max_drawdown_pct") or 0.0))
    enough_trades = 1 if closed >= min_closed_trades else 0
    return (enough_trades, net, pf, -drawdown, closed)


def unique_sorted(values):
    return sorted(set(values))


def build_candidates(
    long_thresholds,
    short_thresholds,
    gaps,
    min_target_ratios,
    position_probability_centers=None,
    asymmetric=False,
):
    candidates = []
    long_thresholds = unique_sorted(long_thresholds)
    short_thresholds = unique_sorted(short_thresholds)
    gaps = unique_sorted(gaps)
    min_target_ratios = unique_sorted(min_target_ratios)
    position_probability_centers = unique_sorted(
        position_probability_centers
        if position_probability_centers is not None
        else [float(config.POSITION_PROBABILITY_CENTER)]
    )
    if asymmetric:
        threshold_pairs = [(long, short) for long in long_thresholds for short in short_thresholds]
    else:
        shared = unique_sorted(long_thresholds + short_thresholds)
        threshold_pairs = [(value, value) for value in shared]

    for long_threshold, short_threshold in threshold_pairs:
        for gap in gaps:
            for min_target_ratio in min_target_ratios:
                for probability_center in position_probability_centers:
                    backtest_min_adjust = min(
                        float(config.MIN_ADJUST_AMOUNT),
                        float(config.INITIAL_BALANCE) * float(min_target_ratio),
                    )
                    candidates.append({
                        "name": (
                            f"tl{long_threshold:.2f}_ts{short_threshold:.2f}_"
                            f"gap{gap:.2f}_mt{min_target_ratio:.3f}_pc{probability_center:.2f}"
                        ),
                        "overrides": {
                            "THRESHOLD_LONG": float(long_threshold),
                            "THRESHOLD_SHORT": float(short_threshold),
                            "SIGNAL_MIN_PROB_DIFF": float(gap),
                            "MIN_SIGNAL_TARGET_RATIO": float(min_target_ratio),
                            "REGIME_RANGE_MIN_SIGNAL_TARGET_RATIO": float(min_target_ratio),
                            "REGIME_HIGH_VOL_MIN_SIGNAL_TARGET_RATIO": float(min_target_ratio),
                            "POSITION_PROBABILITY_CENTER": float(probability_center),
                            "BACKTEST_MIN_ADJUST_AMOUNT": float(backtest_min_adjust),
                        },
                    })
    return candidates


def write_report(report, output_path=None):
    if output_path is None:
        output_path = os.path.join(
            LOGS_DIR,
            f"trade_threshold_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(td.json_safe(report), file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, output_path)
    return output_path


def print_top_results(results, limit=12):
    headers = [
        "name",
        "closed",
        "trades",
        "net",
        "pf",
        "win%",
        "maxDD%",
        "gate",
        "top_reason",
    ]
    print(",".join(headers))
    for item in results[:limit]:
        top_reason = "-"
        if item.get("decision_reason_top"):
            top_reason = str(item["decision_reason_top"][0])
        print(
            f"{item['name']},"
            f"{int(item.get('closed_trade_count') or 0)},"
            f"{int(item.get('trade_count') or 0)},"
            f"{float(item.get('net_pnl_after_costs') or 0.0):.2f},"
            f"{float(item.get('profit_factor') or 0.0):.3f},"
            f"{float(item.get('win_rate_pct') or 0.0):.2f},"
            f"{float(item.get('max_drawdown_pct') or 0.0):.2f},"
            f"{int(item.get('weak_signal_gate_counts', {}).get('gate_count') or 0)},"
            f"\"{top_reason}\""
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="校准交易方向阈值、概率差阈值，并输出概率校准表")
    parser.add_argument("--model-root", default=td.BASE_DIR, help="模型产物根目录，默认项目根目录")
    parser.add_argument("--split", choices=["all", "validation", "oos"], default="all", help="校准样本切片")
    parser.add_argument("--rows", type=int, default=int(os.getenv("THRESHOLD_CALIBRATION_ROWS", "927")), help="使用切片尾部 N 行；<=0 表示全量")
    parser.add_argument("--long-thresholds", default=os.getenv("THRESHOLD_CALIBRATION_LONGS"), help="逗号分隔 long 阈值")
    parser.add_argument("--short-thresholds", default=os.getenv("THRESHOLD_CALIBRATION_SHORTS"), help="逗号分隔 short 阈值")
    parser.add_argument("--gaps", default=os.getenv("THRESHOLD_CALIBRATION_GAPS"), help="逗号分隔 SIGNAL_MIN_PROB_DIFF 值")
    parser.add_argument("--min-target-ratios", default=os.getenv("THRESHOLD_CALIBRATION_MIN_TARGET_RATIOS"), help="逗号分隔 MIN_SIGNAL_TARGET_RATIO 值")
    parser.add_argument("--position-probability-centers", default=os.getenv("THRESHOLD_CALIBRATION_POSITION_PROBABILITY_CENTERS"), help="逗号分隔仓位 sizing 概率中心")
    parser.add_argument("--bins", default=os.getenv("THRESHOLD_CALIBRATION_BINS"), help="逗号分隔概率校准 bin 边界")
    parser.add_argument("--asymmetric", action="store_true", help="跑 long/short 阈值笛卡尔积；默认使用对称阈值")
    parser.add_argument(
        "--probability-calibration",
        choices=["none", "isotonic", "sigmoid"],
        default=os.getenv("THRESHOLD_CALIBRATION_PROBABILITY_METHOD", "none"),
        help="是否先用校准集拟合概率校准器，再用校准后概率跑阈值 sweep",
    )
    parser.add_argument(
        "--probability-calibration-source",
        choices=["validation", "all", "selected"],
        default=os.getenv("THRESHOLD_CALIBRATION_PROBABILITY_SOURCE", "validation"),
        help="概率校准器拟合来源；默认 validation，避免用 OOS 拟合",
    )
    parser.add_argument(
        "--probability-calibration-rows",
        type=int,
        default=int(os.getenv("THRESHOLD_CALIBRATION_PROBABILITY_ROWS", "0")),
        help="概率校准拟合来源尾部 N 行；<=0 表示全量",
    )
    parser.add_argument(
        "--regime-direction-thresholds",
        default=os.getenv("REGIME_DIRECTION_CALIBRATION_THRESHOLDS"),
        help="逗号分隔 regime+direction 局部诊断阈值；默认包含低概率段",
    )
    parser.add_argument(
        "--regime-direction-gaps",
        default=os.getenv("REGIME_DIRECTION_CALIBRATION_GAPS"),
        help="逗号分隔 regime+direction 局部诊断概率差阈值",
    )
    parser.add_argument(
        "--regime-direction-min-rows",
        type=int,
        default=int(os.getenv("REGIME_DIRECTION_CALIBRATION_MIN_ROWS", "30")),
        help="regime+direction 分组诊断最低样本数",
    )
    parser.add_argument(
        "--regime-direction-min-trades",
        type=int,
        default=int(os.getenv("REGIME_DIRECTION_CALIBRATION_MIN_TRADES", "5")),
        help="regime+direction 分组推荐最低 trade 标签数",
    )
    parser.add_argument("--raw-labels", action="store_true", help="使用原始涨跌标签，不按交易门禁过滤")
    parser.add_argument("--min-closed-trades", type=int, default=int(os.getenv("THRESHOLD_CALIBRATION_MIN_CLOSED_TRADES", "5")), help="推荐排序最低平仓笔数")
    parser.add_argument("--skip-label-strength", action="store_true", help="跳过标签强度 sweep")
    parser.add_argument("--label-lookaheads", default=os.getenv("LABEL_STRENGTH_LOOKAHEADS"), help="逗号分隔 realistic 标签 lookahead bars")
    parser.add_argument("--label-take-profits", default=os.getenv("LABEL_STRENGTH_TAKE_PROFITS"), help="逗号分隔 realistic 标签 TP")
    parser.add_argument("--label-stop-losses", default=os.getenv("LABEL_STRENGTH_STOP_LOSSES"), help="逗号分隔 realistic 标签 SL")
    parser.add_argument(
        "--label-target-trade-pct",
        type=float,
        default=float(os.getenv("LABEL_STRENGTH_TARGET_TRADE_PCT", "8.0")),
        help="标签强度推荐目标 trade 占比百分数",
    )
    parser.add_argument(
        "--label-min-trade-rows",
        type=int,
        default=int(os.getenv("LABEL_STRENGTH_MIN_TRADE_ROWS", "80")),
        help="标签强度推荐最低 trade 样本数",
    )
    parser.add_argument(
        "--label-strength-model-top-n",
        type=int,
        default=int(os.getenv("LABEL_STRENGTH_MODEL_TOP_N", "0")),
        help="对标签强度推荐前 N 个候选临时训练模型并跑阈值回测；0 表示跳过重训练 sweep",
    )
    parser.add_argument("--output", default=None, help="报告 JSON 输出路径")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.rows is not None and args.rows <= 0:
        args.rows = None
    calibration_rows = None
    if args.probability_calibration_rows and args.probability_calibration_rows > 0:
        calibration_rows = int(args.probability_calibration_rows)

    default_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, float(config.THRESHOLD_LONG)]
    default_gaps = [0.08, 0.12, 0.16, 0.20, float(config.SIGNAL_MIN_PROB_DIFF)]
    default_min_target_ratios = [0.01, 0.02, 0.04, float(config.MIN_SIGNAL_TARGET_RATIO)]
    default_position_probability_centers = [0.35, 0.40, 0.45, 0.50, float(config.POSITION_PROBABILITY_CENTER)]
    default_regime_direction_thresholds = [
        0.02, 0.04, 0.06, 0.08, 0.10,
        0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.45, 0.50, 0.55, 0.60,
        0.65, 0.70, 0.75, 0.80,
        float(config.THRESHOLD_LONG),
        float(config.THRESHOLD_SHORT),
    ]
    default_regime_direction_gaps = [0.0, 0.04, 0.08, 0.12, float(config.SIGNAL_MIN_PROB_DIFF)]
    default_label_lookaheads = [24, 36, 48, 72]
    default_label_take_profits = [0.018, 0.022, 0.026, float(config.TAKE_PROFIT)]
    default_label_stop_losses = [0.010, 0.012, 0.014, float(config.STOP_LOSS)]
    long_thresholds = parse_float_list(args.long_thresholds, default_thresholds)
    short_thresholds = parse_float_list(args.short_thresholds, default_thresholds)
    gaps = parse_float_list(args.gaps, default_gaps)
    min_target_ratios = parse_float_list(args.min_target_ratios, default_min_target_ratios)
    position_probability_centers = parse_float_list(
        args.position_probability_centers,
        default_position_probability_centers,
    )
    regime_direction_thresholds = parse_float_list(
        args.regime_direction_thresholds,
        default_regime_direction_thresholds,
    )
    regime_direction_gaps = parse_float_list(
        args.regime_direction_gaps,
        default_regime_direction_gaps,
    )
    label_lookaheads = parse_int_list(args.label_lookaheads, default_label_lookaheads)
    label_take_profits = parse_float_list(args.label_take_profits, default_label_take_profits)
    label_stop_losses = parse_float_list(args.label_stop_losses, default_label_stop_losses)
    bins = parse_float_list(args.bins, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    label_candidates = build_label_strength_candidates(
        sorted(set(label_lookaheads)),
        sorted(set(label_take_profits)),
        sorted(set(label_stop_losses)),
    )

    bundle, seed_bt, labeled = load_all_diagnostic_data(args.model_root, raw_labels=args.raw_labels)
    raw_data = td.select_split(labeled, bundle["metadata"], args.split, args.rows)
    if raw_data.empty:
        raise RuntimeError("校准样本为空，请调整 --split 或 --rows")

    probability_source, probability_source_fallback = select_probability_calibration_source(
        labeled,
        bundle["metadata"],
        args.probability_calibration_source,
        raw_data,
        rows=calibration_rows,
    )
    calibrators = fit_probability_calibrators(probability_source, args.probability_calibration)
    probability_data = apply_probability_calibrators(raw_data, calibrators)
    using_calibrated_probabilities = args.probability_calibration != "none"
    data = (
        use_calibrated_probability_columns(probability_data)
        if using_calibrated_probabilities
        else raw_data
    )

    candidates = build_candidates(
        long_thresholds,
        short_thresholds,
        gaps,
        min_target_ratios,
        position_probability_centers,
        asymmetric=bool(args.asymmetric),
    )
    label_strength_report = (
        {"enabled": False, "reason": "skipped"}
        if args.skip_label_strength
        else build_label_strength_report(
            seed_bt.data.copy(),
            bundle["metadata"],
            args.split,
            args.rows,
            label_candidates,
            target_trade_pct=float(args.label_target_trade_pct),
            min_trade_rows=int(args.label_min_trade_rows),
        )
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_root": os.path.abspath(args.model_root),
        "split": args.split,
        "rows": int(len(data)),
        "start": data.index.min().isoformat(),
        "end": data.index.max().isoformat(),
        "metadata": {
            "created_at": bundle["metadata"].get("created_at"),
            "validation_start": bundle["metadata"].get("validation_start"),
            "validation_end": bundle["metadata"].get("validation_end"),
            "oos_start": bundle["metadata"].get("oos_start"),
            "oos_end": bundle["metadata"].get("oos_end"),
            "feature_count": bundle["metadata"].get("feature_count"),
            "target_schema": bundle["metadata"].get("target_schema"),
            "label_mode": bundle["metadata"].get("label_mode"),
            "label_take_profit": bundle["metadata"].get("label_take_profit"),
            "label_stop_loss": bundle["metadata"].get("label_stop_loss"),
            "label_lookahead_bars": bundle["metadata"].get("label_lookahead_bars"),
        },
        "base_gate_config": {
            "threshold_long": float(config.THRESHOLD_LONG),
            "threshold_short": float(config.THRESHOLD_SHORT),
            "signal_min_prob_diff": float(config.SIGNAL_MIN_PROB_DIFF),
            "min_signal_target_ratio": float(config.MIN_SIGNAL_TARGET_RATIO),
            "backtest_min_adjust_amount": float(config.BACKTEST_MIN_ADJUST_AMOUNT),
            "live_min_adjust_amount": float(config.MIN_ADJUST_AMOUNT),
            "min_expected_net_edge": float(config.MIN_EXPECTED_NET_EDGE),
            "position_probability_center": float(config.POSITION_PROBABILITY_CENTER),
            "position_probability_note": (
                "PositionManager.calculate_target_ratio treats prob<=POSITION_PROBABILITY_CENTER "
                "as zero signal strength. Binary trade-quality models often need this sizing center "
                "calibrated together with thresholds and min target ratio."
            ),
        },
        "probability_calibration": {
            "method": args.probability_calibration,
            "source": args.probability_calibration_source,
            "source_rows": int(len(probability_source)),
            "source_start": probability_source.index.min().isoformat() if not probability_source.empty else None,
            "source_end": probability_source.index.max().isoformat() if not probability_source.empty else None,
            "source_fallback_reason": probability_source_fallback,
            "used_for_threshold_sweep": bool(using_calibrated_probabilities),
            "calibrators": {
                direction: calibrator.summary()
                for direction, calibrator in calibrators.items()
            },
            "raw": build_probability_calibration_report(raw_data, bins),
            "calibrated": (
                build_probability_calibration_report_for_columns(
                    probability_data,
                    bins,
                    {
                        "long": PROBABILITY_DIRECTIONS["long"]["calibrated_col"],
                        "short": PROBABILITY_DIRECTIONS["short"]["calibrated_col"],
                    },
                )
                if using_calibrated_probabilities
                else None
            ),
        },
        "regime_direction_calibration": build_regime_direction_probability_report(
            data,
            regime_direction_thresholds,
            regime_direction_gaps,
            min_group_rows=int(args.regime_direction_min_rows),
            min_trade_rows=int(args.regime_direction_min_trades),
        ),
        "threshold_probability_mode": "calibrated" if using_calibrated_probabilities else "raw",
        "label_strength": label_strength_report,
        "label_strength_model_sweep": {"enabled": False, "reason": "not_requested"},
        "candidates": [],
    }
    if (
        label_strength_report.get("enabled")
        and int(args.label_strength_model_top_n) > 0
    ):
        log_info(
            "标签强度模型sweep开始: "
            f"top_n={int(args.label_strength_model_top_n)} "
            f"threshold_candidates={len(candidates)}"
        )
        report["label_strength_model_sweep"] = build_label_strength_model_sweep(
            seed_bt,
            bundle["feature_cols"],
            args.split,
            args.rows,
            label_strength_report.get("recommended", []),
            candidates,
            top_n=int(args.label_strength_model_top_n),
            probability_method=args.probability_calibration,
            probability_source=args.probability_calibration_source,
            probability_rows=calibration_rows,
            bins=bins,
            min_closed_trades=int(args.min_closed_trades),
            target_trade_pct=float(args.label_target_trade_pct),
            min_trade_rows=int(args.label_min_trade_rows),
        )

    path = write_report(report, args.output)
    log_info(f"阈值校准开始: candidates={len(candidates)} rows={len(data)}")
    for idx, candidate in enumerate(candidates, start=1):
        overrides = candidate["overrides"]
        summary = run_candidate(seed_bt, data.copy(), overrides)
        summary["name"] = candidate["name"]
        summary["overrides"] = overrides
        summary["weak_signal_gate_counts"] = weak_signal_gate_counts(
            data,
            overrides["THRESHOLD_LONG"],
            overrides["THRESHOLD_SHORT"],
            overrides["SIGNAL_MIN_PROB_DIFF"],
        )
        report["candidates"].append(summary)
        if idx % 10 == 0 or idx == len(candidates):
            write_report(report, path)
            log_info(f"阈值校准进度: {idx}/{len(candidates)}")

    ranked = sorted(
        report["candidates"],
        key=lambda item: score_candidate(item, int(args.min_closed_trades)),
        reverse=True,
    )
    report["recommended"] = ranked[:10]
    path = write_report(report, path)
    log_info(f"阈值校准报告: {path}")
    print_top_results(ranked)
    print(json.dumps(td.json_safe({"report_path": path, "recommended": ranked[:10]}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
