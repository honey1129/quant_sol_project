import hashlib
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss

from research.directional_v2 import (
    TARGET_NAMES,
    build_directional_labels,
    select_directional_signal,
)


def class_weight_map(target, maximum_weight):
    counts = target.value_counts().to_dict()
    class_count = len(counts)
    rows = len(target)
    return {
        int(label): min(float(maximum_weight), rows / max(class_count * count, 1))
        for label, count in counts.items()
    }


def temporal_development_split(rows, validation_ratio, purge_bars):
    validation_start = int(rows * (1.0 - float(validation_ratio)))
    train_end = validation_start - int(purge_bars)
    if train_end <= 0 or validation_start >= rows:
        raise ValueError(
            f"directional-v2 development split is too small: rows={rows}, "
            f"validation_start={validation_start}, purge_bars={purge_bars}"
        )
    return train_end, validation_start


def _model_params(spec):
    configured = dict(spec["training"]["model"])
    configured.pop("type", None)
    configured.setdefault("verbosity", -1)
    configured.setdefault("n_jobs", -1)
    return configured


def _fit_model(X, y, spec):
    weights = class_weight_map(y, spec["training"]["maximum_class_weight"])
    sample_weight = y.map(weights).astype(float)
    model = lgb.LGBMClassifier(**_model_params(spec))
    model.fit(X, y, sample_weight=sample_weight)
    return model, weights


def probability_frame(model, X):
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = [int(value) for value in model.classes_]
    frame = pd.DataFrame(0.0, index=X.index, columns=["flat", "long", "short"])
    for index, target in enumerate(classes):
        frame[TARGET_NAMES[target]] = raw[:, index]
    return frame


def development_diagnostics(model, X, y, spec):
    probabilities = probability_frame(model, X)
    predictions = model.predict(X).astype(int)
    signals = [
        select_directional_signal(row.to_dict(), spec["signal"])
        for _, row in probabilities.iterrows()
    ]
    accepted = []
    for row_index, signal in enumerate(signals):
        if signal["direction"] == "long":
            accepted.append((row_index, 1))
        elif signal["direction"] == "short":
            accepted.append((row_index, 2))
    correct_signals = sum(
        1 for row_index, target in accepted if int(y.iloc[row_index]) == target
    )
    labels = [0, 1, 2]
    return {
        "rows": int(len(y)),
        "period_start": X.index.min().isoformat(),
        "period_end": X.index.max().isoformat(),
        "target_counts": {
            TARGET_NAMES[int(key)]: int(value)
            for key, value in y.value_counts().sort_index().items()
        },
        "prediction_counts": {
            TARGET_NAMES[int(key)]: int(value)
            for key, value in pd.Series(predictions).value_counts().sort_index().items()
        },
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=labels,
            target_names=[TARGET_NAMES[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
        "log_loss": float(log_loss(y, model.predict_proba(X), labels=model.classes_)),
        "accepted_signal_count": int(len(accepted)),
        "accepted_signal_pct": float(len(accepted) / len(y) * 100.0) if len(y) else 0.0,
        "accepted_signal_precision": (
            float(correct_signals / len(accepted))
            if accepted
            else 0.0
        ),
    }


def train_directional_v2(data, feature_cols, spec):
    cutoff = pd.Timestamp(spec["training"]["final_refit_end_exclusive"])
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    index_tz = getattr(data.index, "tz", None)
    if index_tz is None:
        cutoff = cutoff.tz_convert(None)
    else:
        cutoff = cutoff.tz_convert(index_tz)

    development_data = data.loc[data.index < cutoff].copy()
    missing = [column for column in feature_cols if column not in development_data]
    if missing:
        raise RuntimeError(
            "directional-v2 training features are missing: "
            + ",".join(missing[:10])
        )
    labeled = build_directional_labels(development_data, spec)
    if labeled.empty:
        raise RuntimeError("directional-v2 labels are empty")
    target = labeled["target_v2"].astype(int)
    if set(target.unique()) != {0, 1, 2}:
        raise RuntimeError(
            f"directional-v2 requires all three classes: classes={sorted(target.unique())}"
        )

    X = labeled[feature_cols].astype(float)
    train_end, validation_start = temporal_development_split(
        len(labeled),
        spec["training"]["development_validation_ratio"],
        spec["development"]["purge_bars"],
    )
    validation_model, validation_weights = _fit_model(
        X.iloc[:train_end],
        target.iloc[:train_end],
        spec,
    )
    diagnostics = development_diagnostics(
        validation_model,
        X.iloc[validation_start:],
        target.iloc[validation_start:],
        spec,
    )
    final_model, final_weights = _fit_model(X, target, spec)
    metadata = {
        "schema_version": 1,
        "experiment_id": spec["experiment_id"],
        "target_schema": "directional_multiclass_v2",
        "class_names": {str(key): value for key, value in TARGET_NAMES.items()},
        "feature_count": int(len(feature_cols)),
        "training_rows": int(len(labeled)),
        "training_period_start": labeled.index.min().isoformat(),
        "training_period_end": labeled.index.max().isoformat(),
        "development_train_end": labeled.index[train_end - 1].isoformat(),
        "development_validation_start": labeled.index[validation_start].isoformat(),
        "validation_class_weights": {
            str(key): float(value) for key, value in validation_weights.items()
        },
        "final_class_weights": {
            str(key): float(value) for key, value in final_weights.items()
        },
        "label_counts": {
            TARGET_NAMES[int(key)]: int(value)
            for key, value in target.value_counts().sort_index().items()
        },
        "development_diagnostics": diagnostics,
    }
    return final_model, list(feature_cols), metadata


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_frozen_artifacts(model, feature_cols, metadata, output_dir, spec_hash):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    feature_path = os.path.join(output_dir, "feature_columns.joblib")
    metadata_path = os.path.join(output_dir, "metadata.json")
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_path)
    metadata = dict(metadata)
    metadata.update({
        "spec_sha256": spec_hash,
        "model_sha256": file_sha256(model_path),
        "feature_columns_sha256": file_sha256(feature_path),
    })
    tmp_path = f"{metadata_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, metadata_path)
    return {
        "model_path": model_path,
        "feature_path": feature_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
    }


def load_frozen_artifacts(output_dir, expected_spec_hash):
    model_path = os.path.join(output_dir, "model.joblib")
    feature_path = os.path.join(output_dir, "feature_columns.joblib")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("spec_sha256") != expected_spec_hash:
        raise RuntimeError("directional-v2 model was trained against a different spec hash")
    if metadata.get("model_sha256") != file_sha256(model_path):
        raise RuntimeError("directional-v2 model artifact hash mismatch")
    if metadata.get("feature_columns_sha256") != file_sha256(feature_path):
        raise RuntimeError("directional-v2 feature artifact hash mismatch")
    return joblib.load(model_path), joblib.load(feature_path), metadata
