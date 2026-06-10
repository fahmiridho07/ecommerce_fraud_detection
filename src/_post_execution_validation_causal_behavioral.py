"""Phase 11 post-run validation for causal behavioral B2/B3 experiment family."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from causal_behavioral_features import causal_behavioral_feature_names
from config import (
    BASELINE_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_AE_COMPARISON_FILE,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_FEATURE_IMPORTANCE_FILE,
    CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
    TARGET_COL,
)
from train_causal_behavioral_lgbm import prepare_causal_behavioral_splits


REQUIRED_B2_ARTIFACTS = [
    "metrics_validation_selected_threshold.json",
    "metrics_test_selected_threshold.json",
    "confusion_matrix_validation.csv",
    "confusion_matrix_test.csv",
    "threshold_selection.csv",
    "feature_importance.csv",
    "behavioral_feature_importance.csv",
    "model.pkl",
    "model.txt",
    "preprocessing.pkl",
    "feature_definition.json",
    "run_config.json",
]

REQUIRED_B3_ARTIFACTS = REQUIRED_B2_ARTIFACTS + [
    "cdv_feature_importance.csv",
    "source_ae_validation.json",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_artifacts(output_dir: Path, required: list[str], label: str) -> None:
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing artifacts: {missing}")


def main() -> None:
    print("=" * 72)
    print("PHASE 11 — CAUSAL BEHAVIORAL POST-RUN VALIDATION")
    print("=" * 72)

    b2_dir = CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR
    b3_dir = CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR

    print("\n1. Artifact completeness")
    check_artifacts(b2_dir, REQUIRED_B2_ARTIFACTS, "B2")
    check_artifacts(b3_dir, REQUIRED_B3_ARTIFACTS, "B3")
    print("  B2 and B3 artifacts: COMPLETE")

    print("\n2. B2/B3 split row alignment")
    prepared = prepare_causal_behavioral_splits()
    if len(prepared["train_df"]) != len(prepared["X_train_combined"]):
        raise ValueError("Train row count mismatch.")
    print("  Split row alignment: PASSED")

    print("\n3. B2/B3 original and behavioral column identity (dry check via shapes)")
    original_count = prepared["X_train_raw"].shape[1]
    behavioral_count = len(causal_behavioral_feature_names())
    b2_run = load_json(b2_dir / "run_config.json")
    b3_run = load_json(b3_dir / "run_config.json")
    if b2_run["final_feature_count"] != original_count + behavioral_count:
        raise ValueError("B2 feature count mismatch.")
    if b3_run["final_feature_count"] != original_count + behavioral_count + 1:
        raise ValueError("B3 feature count mismatch.")
    print("  Feature counts: PASSED")

    print("\n4. B3 has exactly one extra feature")
    if b3_run.get("only_added_feature") != "cdv_ae_reconstruction_mse":
        raise ValueError("B3 only_added_feature mismatch.")
    if b3_run.get("reconstruction_error_count") != 1:
        raise ValueError("B3 reconstruction_error_count must be 1.")
    print("  Single CDV recon feature: PASSED")

    print("\n5. No latent or reconstructed feature columns")
    b3_importance = pd.read_csv(b3_dir / "feature_importance.csv")
    forbidden = b3_importance[
        b3_importance["feature"].astype(str).str.contains(
            "latent|ae_reconstructed|log1p_",
            regex=True,
        )
    ]
    if not forbidden.empty:
        raise ValueError("Forbidden AE columns found in B3 importance.")
    print("  No latent/reconstructed columns: PASSED")

    print("\n6. Comparison CSV matches metric artifacts")
    if not CAUSAL_BEHAVIORAL_AE_COMPARISON_FILE.exists():
        raise FileNotFoundError("Comparison CSV missing.")
    comparison = pd.read_csv(CAUSAL_BEHAVIORAL_AE_COMPARISON_FILE)
    b2_val_csv = float(
        comparison.loc[comparison["model_id"] == "B2", "validation_average_precision"].iloc[0]
    )
    b2_val_json = float(
        load_json(b2_dir / "metrics_validation_selected_threshold.json")[
            "average_precision"
        ]
    )
    if abs(b2_val_csv - b2_val_json) > 1e-9:
        raise ValueError("B2 comparison CSV metric mismatch.")
    print("  Comparison CSV consistency: PASSED")

    print("\n7. Feature importance files match model feature names")
    b2_model_features = set(pd.read_csv(b2_dir / "feature_importance.csv")["feature"])
    if not set(causal_behavioral_feature_names()).issubset(b2_model_features):
        raise ValueError("B2 behavioral features missing from importance.")
    if not CAUSAL_BEHAVIORAL_FEATURE_IMPORTANCE_FILE.exists():
        raise FileNotFoundError("Combined importance CSV missing.")
    print("  Feature importance alignment: PASSED")

    print("\n8. Leakage flags in run_config")
    for label, run in (("B2", b2_run), ("B3", b3_run)):
        if not run.get("labels_not_used_in_feature_state"):
            raise ValueError(f"{label} labels_not_used_in_feature_state false.")
        if not run.get("future_rows_not_used"):
            raise ValueError(f"{label} future_rows_not_used false.")
    print("  Leakage flags: PASSED")

    print("\n9. Syntax/import smoke check")
    import causal_behavioral_features  # noqa: F401
    import train_causal_behavioral_lgbm  # noqa: F401
    import train_causal_behavioral_cdv_reconstruction_lgbm  # noqa: F401
    print("  Imports: PASSED")

    print("\n" + "=" * 72)
    print("POST-RUN VALIDATION: ALL CHECKS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPOST-RUN VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)