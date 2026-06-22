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
    label_lookaheads = parse_int_list(args.label_lookaheads, default_label_lookaheads)
    label_take_profits = parse_float_list(args.label_take_profits, default_label_take_profits)
    label_stop_losses = parse_float_list(args.label_stop_losses, default_label_stop_losses)
    bins = parse_float_list(args.bins, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

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
        "threshold_probability_mode": "calibrated" if using_calibrated_probabilities else "raw",
        "label_strength": (
            {"enabled": False, "reason": "skipped"}
            if args.skip_label_strength
            else build_label_strength_report(
                seed_bt.data.copy(),
                bundle["metadata"],
                args.split,
                args.rows,
                build_label_strength_candidates(
                    sorted(set(label_lookaheads)),
                    sorted(set(label_take_profits)),
                    sorted(set(label_stop_losses)),
                ),
                target_trade_pct=float(args.label_target_trade_pct),
                min_trade_rows=int(args.label_min_trade_rows),
            )
        ),
        "candidates": [],
    }

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
