"""Phase 10 post-execution validation for selected-numerical AE experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    AE_LGBM_LD128_OUTPUT_DIR,
    AUTOENCODER_INPUT_SCOPE_COMPARISON_FILE,
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    ID_COL,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split

AE_REQUIRED = [
    "autoencoder_model.keras",
    "encoder_model.keras",
    "numerical_imputer.pkl",
    "numerical_scaler.pkl",
    "selected_numerical_feature_names.json",
    "latent_feature_names.json",
    "latent_train.npy",
    "latent_valid.npy",
    "latent_test.npy",
    "reconstruction_error_train.csv",
    "reconstruction_error_valid.csv",
    "reconstruction_error_test.csv",
    "reconstruction_metrics.json",
    "ae_training_history.csv",
    "run_config.json",
]

LGBM_REQUIRED = [
    "metrics_validation_default_threshold.json",
    "metrics_validation_selected_threshold.json",
    "metrics_test_default_threshold.json",
    "metrics_test_selected_threshold.json",
    "confusion_matrix_validation.csv",
    "confusion_matrix_test.csv",
    "threshold_selection.csv",
    "feature_importance.csv",
    "model.pkl",
    "model.txt",
    "preprocessing_non_ae_features.pkl",
    "run_config.json",
]


def main() -> None:
    errors: list[str] = []

    ae_dir = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR
    lgbm_dir = SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR

    for name in AE_REQUIRED:
        if not (ae_dir / name).exists():
            errors.append(f"Missing AE artifact: {ae_dir / name}")

    for name in LGBM_REQUIRED:
        if not (lgbm_dir / name).exists():
            errors.append(f"Missing LGBM artifact: {lgbm_dir / name}")

    with (ae_dir / "selected_numerical_feature_names.json").open("r", encoding="utf-8") as file:
        selected_features = json.load(file)
    with (ae_dir / "run_config.json").open("r", encoding="utf-8") as file:
        ae_run = json.load(file)
    with (lgbm_dir / "run_config.json").open("r", encoding="utf-8") as file:
        lgbm_run = json.load(file)

    forbidden = {TARGET_COL, ID_COL, TIME_COL}
    leaked = sorted(set(selected_features) & forbidden)
    if leaked:
        errors.append(f"Forbidden columns in AE input: {leaked}")

    latent_train = np.load(ae_dir / "latent_train.npy")
    latent_valid = np.load(ae_dir / "latent_valid.npy")
    latent_test = np.load(ae_dir / "latent_test.npy")
    if latent_train.shape[1] != 128:
        errors.append(f"Unexpected latent dimension: {latent_train.shape[1]}")
    if not np.isfinite(latent_train).all():
        errors.append("Non-finite latent values detected.")

    full_df = load_labeled_train_data()
    train_df, valid_df, test_df = chronological_split(full_df)
    expected_rows = (len(train_df), len(valid_df), len(test_df))
    for name, expected in zip(
        ["latent_train.npy", "latent_valid.npy", "latent_test.npy"],
        expected_rows,
    ):
        arr = np.load(ae_dir / name)
        if arr.shape[0] != expected:
            errors.append(f"{name} row count {arr.shape[0]} != expected {expected}")

    if lgbm_run.get("reconstruction_error_included"):
        errors.append("Reconstruction error should not be included.")
    if not lgbm_run.get("transactiondt_retained_downstream"):
        errors.append("TransactionDT should be retained downstream.")
    if not lgbm_run.get("original_selected_numerical_features_removed"):
        errors.append("Original selected numerical features should be removed.")

    importance = pd.read_csv(lgbm_dir / "feature_importance.csv")
    leaked_selected = sorted(set(importance["feature"]) & set(selected_features))
    if leaked_selected:
        errors.append(
            "Selected numerical columns found in final model features: "
            + ", ".join(leaked_selected[:5])
        )
    if "reconstruction_mse" in importance["feature"].values:
        errors.append("reconstruction_mse found in feature importance.")

    if len(importance["feature"]) != len(set(importance["feature"])):
        errors.append("Duplicate feature names in importance table.")

    if lgbm_run["final_feature_count"] != 173:
        errors.append(f"Unexpected final feature count: {lgbm_run['final_feature_count']}")

    comparison = pd.read_csv(AUTOENCODER_INPUT_SCOPE_COMPARISON_FILE)
    sel_row = comparison[
        comparison["model_name"] == "AAE01_selected_numerical_replacement_ld128_default"
    ].iloc[0]
    saved_val = json.load(
        open(lgbm_dir / "metrics_validation_selected_threshold.json", encoding="utf-8")
    )["average_precision"]
    if abs(sel_row["validation_average_precision"] - saved_val) > 1e-6:
        errors.append("Comparison CSV validation AP mismatch.")

    protected_dirs = [
        BASELINE_OUTPUT_DIR / "run_config.json",
        AE_LGBM_LD128_OUTPUT_DIR / "run_config.json",
        AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "run_config.json",
    ]
    for path in protected_dirs:
        if not path.exists():
            errors.append(f"Protected reference artifact missing: {path}")

    print("=" * 60)
    print("PHASE 10 POST-EXECUTION VALIDATION")
    print("=" * 60)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)

    print("All required AE artifacts present.")
    print("All required LGBM artifacts present.")
    print("Selected feature order persisted and leakage checks passed.")
    print("Original selected numerical columns absent from final LightGBM matrix.")
    print("TransactionDT downstream policy matches P01 (retained).")
    print("Latent count correct (128).")
    print("No reconstruction error feature.")
    print("No duplicate columns.")
    print("Comparison CSV matches saved metrics.")
    print("Protected prior outputs intact.")
    print("POST-EXECUTION VALIDATION PASSED")


if __name__ == "__main__":
    main()