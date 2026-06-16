"""Regenerate identity-aware CDV reconstruction errors from frozen AE artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from autoencoder_helpers import (
    EXPECTED_CDV_FEATURE_COUNT,
    apply_frozen_median_scaled_feature_block,
    get_ordered_cdv_feature_columns,
    load_reconstruction_errors,
    reconstruction_errors,
    validate_reconstruction_error_lengths,
)
from causal_behavioral_features import transaction_id_checksum
from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_USE_SCALED_CLIPPING,
    BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
    ID_COL,
    SAMPLE_SIZE,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import ensure_dir, log, save_json


CDV_ERROR_COLUMN = "cdv_ae_reconstruction_mse"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_frozen_autoencoder_artifacts(ae_output_dir: Path) -> dict[str, object]:
    run_config = load_json(ae_output_dir / "run_config.json")
    reconstruction_metrics = load_json(ae_output_dir / "reconstruction_metrics.json")
    required_files = [
        "autoencoder_model.keras",
        "cdv_scaler.pkl",
        "reconstruction_error_train.csv",
        "reconstruction_error_valid.csv",
        "reconstruction_error_test.csv",
    ]
    missing = [name for name in required_files if not (ae_output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing frozen AE artifact(s) in {ae_output_dir}: " + ", ".join(missing)
        )

    feature_block = run_config.get("feature_block", {})
    cdv_feature_count = int(feature_block.get("cdv_feature_count", 0))
    if cdv_feature_count != EXPECTED_CDV_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CDV_FEATURE_COUNT} CDV features, found {cdv_feature_count}."
        )

    return {
        "source_autoencoder_path": str(ae_output_dir),
        "cdv_feature_count": cdv_feature_count,
        "autoencoder_retrained": False,
        "labels_used_in_ae_training": bool(run_config.get("labels_used_in_training", False)),
        "scaler_fitted_on_train_only": True,
        "reconstruction_metrics": reconstruction_metrics,
    }


def regenerate_id_aligned_errors(
    ae_output_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    artifact_validation = validate_frozen_autoencoder_artifacts(ae_output_dir)

    log("Loading labeled training data and chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    cdv_columns = get_ordered_cdv_feature_columns(train_df)
    if len(cdv_columns) != EXPECTED_CDV_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CDV_FEATURE_COUNT} CDV columns, found {len(cdv_columns)}."
        )

    scaler = joblib.load(ae_output_dir / "cdv_scaler.pkl")
    imputer_path = ae_output_dir / "cdv_imputer.pkl"
    if imputer_path.exists():
        imputer = joblib.load(imputer_path)
        X_train, X_valid, X_test = apply_frozen_median_scaled_feature_block(
            train_df,
            valid_df,
            test_df,
            cdv_columns,
            imputer,
            scaler,
            use_scaled_clipping=AE_USE_SCALED_CLIPPING,
            clip_min=AE_CLIP_MIN,
            clip_max=AE_CLIP_MAX,
        )
    else:
        from autoencoder_helpers import raw_float_feature_matrix

        X_train_raw = raw_float_feature_matrix(train_df, cdv_columns)
        X_valid_raw = raw_float_feature_matrix(valid_df, cdv_columns)
        X_test_raw = raw_float_feature_matrix(test_df, cdv_columns)
        X_train = scaler.transform(X_train_raw).astype("float32")
        X_valid = scaler.transform(X_valid_raw).astype("float32")
        X_test = scaler.transform(X_test_raw).astype("float32")
        if AE_USE_SCALED_CLIPPING:
            X_train = np.clip(X_train, AE_CLIP_MIN, AE_CLIP_MAX).astype("float32")
            X_valid = np.clip(X_valid, AE_CLIP_MIN, AE_CLIP_MAX).astype("float32")
            X_test = np.clip(X_test, AE_CLIP_MIN, AE_CLIP_MAX).astype("float32")

    autoencoder = tf.keras.models.load_model(ae_output_dir / "autoencoder_model.keras")
    regenerated = {
        "train": reconstruction_errors(autoencoder, X_train, AE_BATCH_SIZE),
        "validation": reconstruction_errors(autoencoder, X_valid, AE_BATCH_SIZE),
        "test": reconstruction_errors(autoencoder, X_test, AE_BATCH_SIZE),
    }
    validate_reconstruction_error_lengths(
        regenerated,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    split_frames = {
        "train": train_df,
        "validation": valid_df,
        "test": test_df,
    }
    output_frames: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_frames.items():
        frame = pd.DataFrame(
            {
                ID_COL: split_df[ID_COL].tolist(),
                CDV_ERROR_COLUMN: regenerated[split_name],
            }
        )
        if frame[ID_COL].duplicated().any():
            raise ValueError(f"{split_name} regenerated CDV errors contain duplicate IDs.")
        if not np.isfinite(frame[CDV_ERROR_COLUMN]).all():
            raise ValueError(f"{split_name} regenerated CDV errors are non-finite.")
        if (frame[CDV_ERROR_COLUMN] < 0).any():
            raise ValueError(f"{split_name} regenerated CDV errors are negative.")
        output_frames[split_name] = frame
        frame.to_csv(
            output_dir / f"cdv_reconstruction_error_{split_name}.csv",
            index=False,
        )

    legacy_errors = load_reconstruction_errors(ae_output_dir)
    positional_consistency: dict[str, object] = {}
    for split_name, split_df in split_frames.items():
        legacy = legacy_errors[split_name]
        regenerated_values = regenerated[split_name]
        max_abs_diff = float(np.max(np.abs(legacy - regenerated_values)))
        mean_abs_diff = float(np.mean(np.abs(legacy - regenerated_values)))
        positional_consistency[split_name] = {
            "row_count": int(len(split_df)),
            "transaction_id_checksum": transaction_id_checksum(split_df[ID_COL]),
            "max_abs_diff_vs_legacy_positional_arrays": max_abs_diff,
            "mean_abs_diff_vs_legacy_positional_arrays": mean_abs_diff,
            "numerically_consistent_with_legacy_by_position": bool(
                np.allclose(legacy, regenerated_values, rtol=1e-5, atol=1e-6)
            ),
        }

    summary = {
        **artifact_validation,
        "regeneration_policy": (
            "Recomputed CDV reconstruction errors from frozen autoencoder and scaler "
            "using the current frozen chronological split rows, keyed by TransactionID."
        ),
        "cdv_error_column": CDV_ERROR_COLUMN,
        "split_row_counts": {
            split_name: int(len(split_df))
            for split_name, split_df in split_frames.items()
        },
        "positional_consistency_vs_legacy_arrays": positional_consistency,
        "identity_status": "verified_one_to_one_transaction_id_coverage",
    }
    save_json(summary, output_dir / "cdv_reconstruction_error_regeneration.json")
    return {
        "output_dir": str(output_dir),
        "frames": output_frames,
        "summary": summary,
        "arrays": regenerated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate identity-aware CDV reconstruction errors."
    )
    parser.add_argument(
        "--ae-output-dir",
        type=Path,
        default=BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return regenerate_id_aligned_errors(args.ae_output_dir, args.output_dir)


if __name__ == "__main__":
    main()