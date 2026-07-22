import argparse
import json
import math
import os
from datetime import datetime, timezone

import pandas as pd

from core.ml_feature_engineering import add_advanced_features, merge_multi_period_features
from core.okx_api import OKXClient
from research.directional_v2 import (
    DEFAULT_SPEC_PATH,
    forward_holdout_status,
    load_experiment_spec,
    verify_frozen_spec,
)
from research.directional_v2_backtest import (
    evaluate_forward_result,
    run_directional_backtest,
    trend_baseline_probabilities,
)
from research.directional_v2_training import (
    load_frozen_artifacts,
    probability_frame,
)
from run.strict_oos_validation import required_data_windows, temporary_config
from run.train_directional_v2 import DEFAULT_OUTPUT_DIR
from utils.utils import LOGS_DIR


REPORT_DIR = os.path.join(LOGS_DIR, "directional_v2_forward")


def _utc_timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def write_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    payload = _json_safe(report)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join(output_dir, f"forward_{run_id}.json")
    latest_path = os.path.join(output_dir, "latest.json")
    for path in (run_path, latest_path):
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, path)
    return run_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen directional-v2 model on its forward holdout.",
    )
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", default=REPORT_DIR)
    return parser


def main():
    args = build_parser().parse_args()
    spec_hash = verify_frozen_spec(args.spec)
    spec = load_experiment_spec(args.spec)
    now = datetime.now(timezone.utc)
    time_status = forward_holdout_status(spec, now=now, closed_trades=0)
    minimum_end = _utc_timestamp(spec["holdout"]["minimum_end_exclusive"])
    if now < minimum_end:
        time_status.update({
            "spec_sha256": spec_hash,
            "message": "Forward holdout is still collecting. Final evaluation is locked.",
        })
        print(json.dumps(time_status, ensure_ascii=False, indent=2, sort_keys=True))
        return

    model, feature_cols, metadata = load_frozen_artifacts(
        os.path.abspath(args.model_dir),
        spec_hash,
    )
    holdout_start = _utc_timestamp(spec["holdout"]["start_inclusive"])
    holdout_days = max(1, int(math.ceil((now - holdout_start).total_seconds() / 86400.0)))
    windows = required_data_windows(holdout_days)
    with temporary_config(WINDOWS=windows):
        data_dict = OKXClient().fetch_data()
    data = add_advanced_features(merge_multi_period_features(data_dict)).dropna().copy()
    missing = [column for column in feature_cols if column not in data]
    if missing:
        raise RuntimeError(
            "directional-v2 forward features are missing: " + ",".join(missing[:10])
        )
    probabilities = probability_frame(model, data[feature_cols].astype(float))
    data_start = pd.Timestamp(holdout_start)
    if getattr(data.index, "tz", None) is None:
        data_start = data_start.tz_convert(None)
    else:
        data_start = data_start.tz_convert(data.index.tz)
    holdout_data = data.loc[data.index >= data_start].copy()
    if holdout_data.empty or holdout_data.index.max() < pd.Timestamp(minimum_end).tz_convert(data.index.tz):
        raise RuntimeError("directional-v2 forward market data has not reached minimum holdout end")
    holdout_probabilities = probabilities.reindex(holdout_data.index)
    summary = run_directional_backtest(holdout_data, holdout_probabilities, spec)
    baseline = run_directional_backtest(
        holdout_data,
        trend_baseline_probabilities(holdout_data, spec),
        spec,
    )
    decision = evaluate_forward_result(summary, baseline, spec)
    report = {
        "schema_version": 1,
        "generated_at": now.isoformat(),
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec_hash,
        "model_sha256": metadata["model_sha256"],
        "holdout_start": holdout_data.index.min().isoformat(),
        "holdout_end": holdout_data.index.max().isoformat(),
        "strategy": summary,
        "trend_baseline": baseline,
        "decision": decision,
    }
    report_path = write_report(report, os.path.abspath(args.output_dir))
    print(json.dumps({
        "verdict": decision["verdict"],
        "reason": decision["reason"],
        "closed_trades": summary["closed_trade_count"],
        "net_pnl_after_costs": summary["net_pnl_after_costs"],
        "profit_factor": summary["profit_factor"],
        "report_path": report_path,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
