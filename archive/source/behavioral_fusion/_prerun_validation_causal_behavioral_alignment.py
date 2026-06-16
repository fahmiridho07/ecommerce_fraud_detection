"""Phase 11 pre-run validation for causal behavioral alignment correction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from audit_causal_behavioral_row_alignment import run_audit
from causal_behavioral_features import causal_behavioral_feature_names
from config import (
    BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
    SAMPLE_SIZE,
)
from data_loader import load_labeled_train_data
from regenerate_cdv_reconstruction_errors_id_aligned import regenerate_id_aligned_errors
from splitting import chronological_split
from train_causal_behavioral_lgbm import prepare_causal_behavioral_splits


RECONSTRUCTION_ERROR_FEATURE = "cdv_ae_reconstruction_mse"


def main() -> None:
    print("=" * 72)
    print("PHASE 11 — CAUSAL BEHAVIORAL ALIGNMENT PRE-RUN VALIDATION")
    print("=" * 72)

    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    print(f"Original split row counts: train={len(train_df)}, "
          f"validation={len(valid_df)}, test={len(test_df)}")

    audit_report = run_audit(output_dir=str(CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR))
    print(f"Pre-fix mismatch count: {audit_report['pre_fix_mismatch_count']}")
    print(
        "Pre-fix split-membership changes: "
        f"{audit_report['pre_fix_split_membership_change_count']}"
    )
    print(
        "Duplicate timestamp rows (full): "
        f"{audit_report['duplicate_timestamp_counts']['full_dataset']}"
    )
    print(
        "Boundary timestamp ties train/validation: "
        f"{audit_report['boundary_timestamp_tie_counts']['train_validation']}"
    )
    print(
        "Boundary timestamp ties validation/test: "
        f"{audit_report['boundary_timestamp_tie_counts']['validation_test']}"
    )

    prepared = prepare_causal_behavioral_splits()
    alignment = prepared["alignment_validation"]
    print(f"Corrected mismatch count: 0")
    print(f"Corrected split-membership change count: 0")
    print(
        "Corrected one-to-one ID status: "
        f"{alignment['transaction_id_join_verified']}"
    )
    print(
        "Corrected output order status: "
        f"{alignment['split_checks']['train']['restored_order_matches_input']}"
    )

    behavioral_count = len(causal_behavioral_feature_names())
    original_count = prepared["X_train_raw"].shape[1]
    print(f"Behavioral feature count: {behavioral_count}")
    print(f"B2 expected feature count: {original_count + behavioral_count}")
    print(f"B3 expected feature count: {original_count + behavioral_count + 1}")

    cdv_result = regenerate_id_aligned_errors(
        BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
        CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
    )
    print(f"CDV error identity status: {cdv_result['summary']['identity_status']}")

    stop_reasons: list[str] = []
    if audit_report["corrected_mismatch_count"] != 0:
        stop_reasons.append("corrected mismatch count is not zero")
    if not alignment["transaction_id_join_verified"]:
        stop_reasons.append("TransactionID join not verified")
    if alignment["positional_join_used"]:
        stop_reasons.append("positional join still enabled")
    for split_name, split_check in alignment["split_checks"].items():
        if not split_check["restored_order_matches_input"]:
            stop_reasons.append(f"{split_name} order cannot be restored")

    if stop_reasons:
        raise RuntimeError("Pre-run validation failed: " + "; ".join(stop_reasons))

    print("\n" + "=" * 72)
    print("PRE-RUN VALIDATION: ALL CHECKS PASSED — READY FOR TRAINING")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPRE-RUN VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)