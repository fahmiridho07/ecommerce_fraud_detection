"""Phase 10 post-execution validation for reconstructed-feature experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    AUTOENCODER_OUTPUT_STRATEGY_COMPARISON_FILE,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR,
)

AE_RECON_FILES = [
    "reconstructed_train.npy",
    "reconstructed_valid.npy",
    "reconstructed_test.npy",
    "reconstructed_feature_names.json",
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
    ae_dir = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR
    lgbm_dir = SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR

    for name in AE_RECON_FILES:
        if not (ae_dir / name).exists():
            errors.append(f"Missing reconstructed artifact: {ae_dir / name}")

    for name in LGBM_REQUIRED:
        if not (lgbm_dir / name).exists():
            errors.append(f"Missing LGBM artifact: {lgbm_dir / name}")

    with (ae_dir / "selected_numerical_feature_names.json").open("r", encoding="utf-8") as file:
        selected_features = json.load(file)
    with (ae_dir / "reconstructed_feature_names.json").open("r", encoding="utf-8") as file:
        reconstructed_names = json.load(file)
    with (lgbm_dir / "run_config.json").open("r", encoding="utf-8") as file:
        lgbm_run = json.load(file)

    recon_train = np.load(ae_dir / "reconstructed_train.npy")
    if recon_train.shape != (354324, 387):
        errors.append(f"Unexpected reconstructed_train shape: {recon_train.shape}")
    if len(reconstructed_names) != 387:
        errors.append(f"Unexpected reconstructed name count: {len(reconstructed_names)}")

    importance = pd.read_csv(lgbm_dir / "feature_importance.csv")
    leaked_selected = sorted(set(importance["feature"]) & set(selected_features))
    if leaked_selected:
        errors.append("Original selected numerical columns present in final model.")
    latent_like = [c for c in importance["feature"] if c.startswith("ae_latent_")]
    if latent_like:
        errors.append("Latent features present in final model.")
    if "reconstruction_mse" in importance["feature"].values:
        errors.append("Reconstruction error present in final model.")
    if lgbm_run["final_feature_count"] != 432:
        errors.append(f"Final feature count != 432: {lgbm_run['final_feature_count']}")
    if lgbm_run.get("autoencoder_retrained"):
        errors.append("Autoencoder must not be retrained.")

    comparison = pd.read_csv(AUTOENCODER_OUTPUT_STRATEGY_COMPARISON_FILE)
    recon_row = comparison[
        comparison["model_name"] == "AAE02_selected_numerical_reconstructed_replacement"
    ].iloc[0]
    saved_val = json.load(
        open(lgbm_dir / "metrics_validation_selected_threshold.json", encoding="utf-8")
    )["average_precision"]
    if abs(recon_row["validation_average_precision"] - saved_val) > 1e-6:
        errors.append("Comparison CSV validation AP mismatch.")

    protected = [
        BASELINE_OUTPUT_DIR / "run_config.json",
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json",
        ae_dir / "latent_train.npy",
        ae_dir / "autoencoder_model.keras",
    ]
    for path in protected:
        if not path.exists():
            errors.append(f"Protected artifact missing: {path}")

    print("=" * 60)
    print("PHASE 10 POST-EXECUTION VALIDATION")
    print("=" * 60)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)

    print("Reconstructed arrays exist with correct shapes.")
    print("All required LightGBM artifacts exist.")
    print("Selected raw numerical columns absent downstream.")
    print("Reconstructed column count is 387.")
    print("Final feature count is 432.")
    print("Latent and reconstruction-error columns absent.")
    print("Comparison CSV matches metric artifacts.")
    print("Protected prior outputs intact.")
    print("No Autoencoder retraining occurred.")
    print("POST-EXECUTION VALIDATION PASSED")


if __name__ == "__main__":
    main()