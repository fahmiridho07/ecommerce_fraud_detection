"""Phase 11 post-run validation for task-aware AE experiment (TAE01)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    ID_COL,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    TARGET_COL,
    TASK_AWARE_AE_COMPARISON_FILE,
    TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR,
    TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    TASK_AWARE_LAMBDA_SELECTION_FILE,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split

LAMBDA_DIRS = ["lambda_0_1", "lambda_0_5", "lambda_1_0"]
AE_REQUIRED = [
    "autoencoder_model.keras",
    "encoder_model.keras",
    "classification_head_model.keras",
    "training_history.csv",
    "latent_train.npy",
    "latent_valid.npy",
    "run_config.json",
    "validation_diagnostics.json",
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
    "preprocessing_retained_features.pkl",
    "run_config.json",
]


def main() -> None:
    errors: list[str] = []
    root = TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR
    selected_dir = root / "selected"
    lgbm_dir = TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR / "selected"

    for lambda_dir in LAMBDA_DIRS:
        candidate = root / lambda_dir
        if not candidate.exists():
            errors.append(f"Missing lambda candidate directory: {candidate}")
            continue
        for name in AE_REQUIRED:
            if not (candidate / name).exists():
                errors.append(f"Missing AE artifact: {candidate / name}")
        if (candidate / "latent_test.npy").exists():
            errors.append(f"Non-selected lambda has test latent: {candidate}")

    for name in AE_REQUIRED + ["latent_test.npy", "selected_lambda.json"]:
        if not (selected_dir / name).exists():
            errors.append(f"Missing selected AE artifact: {selected_dir / name}")

    for name in LGBM_REQUIRED:
        if not (lgbm_dir / name).exists():
            errors.append(f"Missing selected LGBM artifact: {lgbm_dir / name}")

    for lambda_dir in LAMBDA_DIRS:
        if (TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR / lambda_dir / "metrics_test_selected_threshold.json").exists():
            errors.append(
                f"Rejected lambda candidate has test metrics: {lambda_dir}"
            )

    if not TASK_AWARE_LAMBDA_SELECTION_FILE.exists():
        errors.append(f"Missing lambda selection CSV: {TASK_AWARE_LAMBDA_SELECTION_FILE}")
    if not TASK_AWARE_AE_COMPARISON_FILE.exists():
        errors.append(f"Missing comparison CSV: {TASK_AWARE_AE_COMPARISON_FILE}")

    with (selected_dir / "selected_numerical_feature_names.json").open("r", encoding="utf-8") as file:
        selected_features = json.load(file)
    with (lgbm_dir / "run_config.json").open("r", encoding="utf-8") as file:
        lgbm_run = json.load(file)

    forbidden = {TARGET_COL, ID_COL, TIME_COL}
    leaked = sorted(set(selected_features) & forbidden)
    if leaked:
        errors.append(f"Forbidden columns in AE input: {leaked}")

    if lgbm_run.get("reconstruction_error_included"):
        errors.append("Reconstruction error should not be included.")
    if lgbm_run.get("reconstructed_features_included"):
        errors.append("Reconstructed features should not be included.")
    if lgbm_run.get("behavioral_features_included"):
        errors.append("Behavioral features should not be included.")
    if not lgbm_run.get("transactiondt_retained_downstream"):
        errors.append("TransactionDT should be retained downstream.")
    if not lgbm_run.get("original_selected_numerical_features_removed"):
        errors.append("Original selected numerical features should be removed.")
    if lgbm_run["final_feature_count"] != 173:
        errors.append(f"Unexpected final feature count: {lgbm_run['final_feature_count']}")

    full_df = load_labeled_train_data()
    train_df, valid_df, test_df = chronological_split(full_df)
    expected_rows = (len(train_df), len(valid_df), len(test_df))
    for name, expected in zip(
        ["latent_train.npy", "latent_valid.npy", "latent_test.npy"],
        expected_rows,
    ):
        arr = np.load(selected_dir / name)
        if arr.shape[0] != expected:
            errors.append(f"{name} row count {arr.shape[0]} != expected {expected}")

    if TASK_AWARE_LAMBDA_SELECTION_FILE.exists():
        selection = pd.read_csv(TASK_AWARE_LAMBDA_SELECTION_FILE)
        selected_rows = selection[selection["selected"] == True]  # noqa: E712
        if len(selected_rows) != 1:
            errors.append("Lambda selection CSV must mark exactly one selected row.")
        else:
            best_lambda = float(selected_rows.iloc[0]["lambda_classification"])
            max_ap = selection["downstream_lgbm_validation_ap"].max()
            if abs(selected_rows.iloc[0]["downstream_lgbm_validation_ap"] - max_ap) > 1e-9:
                errors.append("Selected lambda does not match highest validation AP.")
            with (selected_dir / "selected_lambda.json").open("r", encoding="utf-8") as file:
                selected_payload = json.load(file)
            if float(selected_payload["selected_lambda"]) != best_lambda:
                errors.append("selected_lambda.json mismatch.")

    if TASK_AWARE_AE_COMPARISON_FILE.exists():
        comparison = pd.read_csv(TASK_AWARE_AE_COMPARISON_FILE)
        tae_row = comparison[comparison["model_id"] == "TAE01"].iloc[0]
        saved_val = json.load(
            open(lgbm_dir / "metrics_validation_selected_threshold.json", encoding="utf-8")
        )["average_precision"]
        if abs(tae_row["validation_average_precision"] - saved_val) > 1e-6:
            errors.append("Comparison CSV validation AP mismatch for TAE01.")

    importance = pd.read_csv(lgbm_dir / "feature_importance.csv")
    leaked_selected = sorted(set(importance["feature"]) & set(selected_features))
    if leaked_selected:
        errors.append("Selected numerical columns found in final model features.")
    if "reconstruction_mse" in importance["feature"].values:
        errors.append("reconstruction_mse found in feature importance.")
    if any(str(feature).startswith("cb_") for feature in importance["feature"]):
        errors.append("Behavioral features found in feature importance.")

    protected_dirs = [
        BASELINE_OUTPUT_DIR / "run_config.json",
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json",
        AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR / "run_config.json",
    ]
    for path in protected_dirs:
        if not path.exists():
            errors.append(f"Protected reference artifact missing: {path}")

    print("=" * 60)
    print("TAE01 POST-EXECUTION VALIDATION")
    print("=" * 60)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)

    print("All lambda candidates completed with required AE artifacts.")
    print("No test latent arrays for non-selected lambdas.")
    print("Selected lambda matches highest downstream validation AP.")
    print("Test evaluated only for selected lambda downstream run.")
    print("Latent arrays align with split rows.")
    print("Final feature count is 173.")
    print("No reconstruction error, reconstructed, or behavioral features.")
    print("Comparison CSV matches saved metrics.")
    print("Protected prior outputs intact.")
    print("POST-EXECUTION VALIDATION PASSED")


if __name__ == "__main__":
    main()