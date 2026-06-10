"""Phase 10 pre-run validation for causal behavioral B1/B2/B3 experiment family."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from autoencoder_helpers import (
    EXPECTED_CDV_FEATURE_COUNT,
    load_reconstruction_errors,
    reconstruction_error_file_paths,
)
from causal_behavioral_features import (
    CAUSAL_ENTITY_GROUPS,
    causal_behavioral_feature_names,
    generate_causal_behavioral_features,
    run_causal_behavioral_fixture_check,
    run_future_immutability_check,
    run_label_invariance_check,
    validate_causal_behavioral_features,
)
from config import (
    BASELINE_OUTPUT_DIR,
    BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR,
    ID_COL,
    SAMPLE_SIZE,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from train_causal_behavioral_lgbm import prepare_causal_behavioral_splits


RECONSTRUCTION_ERROR_FEATURE = "cdv_ae_reconstruction_mse"


def artifact_status(path: Path, label: str) -> dict[str, object]:
    metrics_val = path / "metrics_validation_selected_threshold.json"
    metrics_test = path / "metrics_test_selected_threshold.json"
    run_config = path / "run_config.json"
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "metrics_validation": metrics_val.exists(),
        "metrics_test": metrics_test.exists(),
        "run_config": run_config.exists(),
        "complete": all(
            [
                metrics_val.exists(),
                metrics_test.exists(),
                run_config.exists(),
                (path / "model.pkl").exists(),
            ]
        ),
    }


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def estimate_memory_mb(row_count: int, feature_count: int) -> float:
    bytes_per_value = 4.0
    return row_count * feature_count * bytes_per_value / (1024.0 * 1024.0)


def main() -> None:
    print("=" * 72)
    print("PHASE 10 — CAUSAL BEHAVIORAL PRE-RUN VALIDATION")
    print("=" * 72)

    print("\n1. Existing B1/B2/B3 artifact status")
    statuses = [
        artifact_status(BASELINE_OUTPUT_DIR, "B1 P01 baseline"),
        artifact_status(CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR, "B2 causal behavioral"),
        artifact_status(
            CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
            "B3 causal behavioral + CDV recon",
        ),
        artifact_status(
            FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
            "EX01 FE (not directly comparable)",
        ),
        artifact_status(
            FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR,
            "AE15 FE+CDV (not directly comparable)",
        ),
    ]
    for status in statuses:
        print(
            f"  {status['label']}: exists={status['exists']} "
            f"complete={status['complete']} path={status['path']}"
        )

    print("\n2. Selected entity definitions")
    for entity, columns in CAUSAL_ENTITY_GROUPS.items():
        print(f"  {entity}: {columns}")

    behavioral_names = causal_behavioral_feature_names()
    print("\n3. Final behavioral feature list")
    for feature in behavioral_names:
        print(f"  {feature}")

    print("\n4. Verify every feature is past-only (synthetic tests)")
    run_causal_behavioral_fixture_check()
    run_future_immutability_check()
    run_label_invariance_check()
    print("  Synthetic causal tests: PASSED")

    print("\n5. Row alignment and split preparation")
    prepared = prepare_causal_behavioral_splits()
    train_rows = len(prepared["train_df"])
    valid_rows = len(prepared["valid_df"])
    test_rows = len(prepared["test_df"])
    print(f"  Train rows: {train_rows}")
    print(f"  Validation rows: {valid_rows}")
    print(f"  Test rows: {test_rows}")
    print(
        f"  Combined behavioral shape: {prepared['X_train_combined'].shape} "
        f"(train)"
    )

    print("\n6. No target usage in behavioral state")
    print("  labels_do_not_affect_features: PASSED (fixture)")

    print("\n7. CDV reconstruction error alignment")
    ae_dir = BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR
    for split_name, path in reconstruction_error_file_paths(ae_dir).items():
        if not path.exists():
            raise FileNotFoundError(f"Missing reconstruction error file: {path}")
    recon = load_reconstruction_errors(ae_dir)
    expected_rows = {
        "train": train_rows,
        "validation": valid_rows,
        "test": test_rows,
    }
    for split_name, expected in expected_rows.items():
        actual = recon[split_name].shape[0]
        if actual != expected:
            raise ValueError(
                f"{split_name} reconstruction errors {actual} != rows {expected}"
            )
        if not np.isfinite(recon[split_name]).all():
            raise ValueError(f"{split_name} reconstruction errors non-finite")
    ae_run = load_json(ae_dir / "run_config.json")
    cdv_count = ae_run["feature_block"]["cdv_feature_count"]
    if cdv_count != EXPECTED_CDV_FEATURE_COUNT:
        raise ValueError(f"CDV feature count {cdv_count} != {EXPECTED_CDV_FEATURE_COUNT}")
    print(f"  CDV AE alignment: PASSED ({EXPECTED_CDV_FEATURE_COUNT} inputs)")

    print("\n8. B2 vs B3 column difference check (dry-run)")
    original_count = prepared["X_train_raw"].shape[1]
    behavioral_count = len(behavioral_names)
    b2_count = original_count + behavioral_count
    b3_count = b2_count + 1
    print(f"  B2 expected features: {b2_count}")
    print(f"  B3 expected features: {b3_count} (B2 + {RECONSTRUCTION_ERROR_FEATURE})")

    print("\n9. Expected feature counts")
    print(f"  B1 original features: 432 (from P01 reference)")
    print(f"  B2 original + behavioral: {b2_count}")
    print(f"  B3 B2 + CDV recon: {b3_count}")

    print("\n10. Memory estimate (float32 matrices)")
    total_rows = train_rows + valid_rows + test_rows
    print(f"  Behavioral block (~{behavioral_count} cols): "
          f"{estimate_memory_mb(total_rows, behavioral_count):.1f} MB")
    print(f"  B2 combined (~{b2_count} cols): "
          f"{estimate_memory_mb(total_rows, b2_count):.1f} MB")

    print("\n11. B1 reference metric availability")
    b1_val = load_json(
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    print(f"  B1 validation AP reference: {b1_val:.6f}")

    print("\n" + "=" * 72)
    print("PRE-RUN VALIDATION: ALL CHECKS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPRE-RUN VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)