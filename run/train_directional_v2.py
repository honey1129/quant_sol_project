import argparse
import json
import os
from datetime import datetime, timezone

import joblib

from core.ml_feature_engineering import add_advanced_features, merge_multi_period_features
from core.okx_api import OKXClient
from research.directional_v2 import (
    DEFAULT_SPEC_PATH,
    load_experiment_spec,
    verify_frozen_spec,
)
from research.directional_v2_training import (
    save_frozen_artifacts,
    train_directional_v2,
)
from run.strict_oos_validation import required_data_windows, temporary_config
from utils.utils import BASE_DIR


DEFAULT_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "experiments",
    "directional_v2",
)


def _utc_timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train and freeze the preregistered directional-v2 model.",
    )
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--history-days", type=int, default=180)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main():
    args = build_parser().parse_args()
    if args.history_days <= 0:
        raise ValueError("history-days must be positive")
    spec_hash = verify_frozen_spec(args.spec)
    spec = load_experiment_spec(args.spec)
    holdout_start = _utc_timestamp(spec["holdout"]["start_inclusive"])
    if datetime.now(timezone.utc) >= holdout_start:
        raise RuntimeError(
            "directional-v2 holdout already started; retraining would invalidate the experiment"
        )

    windows = required_data_windows(args.history_days)
    with temporary_config(WINDOWS=windows):
        data_dict = OKXClient().fetch_data()
    data = add_advanced_features(merge_multi_period_features(data_dict)).dropna().copy()
    feature_path = os.path.join(BASE_DIR, spec["training"]["feature_list"])
    feature_cols = joblib.load(feature_path)
    model, feature_cols, metadata = train_directional_v2(
        data,
        feature_cols,
        spec,
    )
    artifacts = save_frozen_artifacts(
        model,
        feature_cols,
        metadata,
        os.path.abspath(args.output_dir),
        spec_hash,
    )
    diagnostics = artifacts["metadata"]["development_diagnostics"]
    print(json.dumps({
        "experiment_id": spec["experiment_id"],
        "spec_sha256": spec_hash,
        "training_rows": artifacts["metadata"]["training_rows"],
        "label_counts": artifacts["metadata"]["label_counts"],
        "development_accepted_signal_count": diagnostics["accepted_signal_count"],
        "development_accepted_signal_precision": diagnostics["accepted_signal_precision"],
        "metadata_path": artifacts["metadata_path"],
        "status": "FROZEN_PENDING_FORWARD_HOLDOUT",
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
