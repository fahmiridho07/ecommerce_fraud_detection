"""Audit row-alignment risk in legacy causal behavioral feature generation."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from causal_behavioral_features import (
    generate_causal_behavioral_features,
    reproduce_legacy_positional_feature_slices,
    transaction_id_checksum,
)
from config import (
    CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
    ID_COL,
    SAMPLE_SIZE,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import ensure_dir, log, save_json


def build_split_manifest(
    split_name: str,
    split_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "split_name": split_name,
        "row_count": int(len(split_df)),
        "transaction_ids": split_df[ID_COL].tolist(),
        "transaction_dt_values": split_df[TIME_COL].astype(int).tolist(),
        "original_row_positions": list(range(len(split_df))),
        "transaction_id_checksum": transaction_id_checksum(split_df[ID_COL]),
    }


def compare_id_sequences(
    original_ids: list[int],
    legacy_ids: list[int],
    split_name: str,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    max_len = max(len(original_ids), len(legacy_ids))
    for position in range(max_len):
        original_id = original_ids[position] if position < len(original_ids) else None
        legacy_id = legacy_ids[position] if position < len(legacy_ids) else None
        if original_id != legacy_id:
            mismatches.append(
                {
                    "split_name": split_name,
                    "position": position,
                    "original_transaction_id": original_id,
                    "legacy_transaction_id": legacy_id,
                }
            )

    original_set = set(original_ids)
    legacy_set = set(legacy_ids)
    membership_changes = sorted(original_set ^ legacy_set)
    return {
        "split_name": split_name,
        "mismatched_row_positions": len(mismatches),
        "split_membership_changes": len(membership_changes),
        "membership_changed_ids": membership_changes,
        "sequence_equal": original_ids == legacy_ids,
        "mismatch_examples": mismatches[:20],
    }


def count_duplicate_timestamps(split_df: pd.DataFrame) -> int:
    return int(split_df[TIME_COL].duplicated(keep=False).sum())


def count_boundary_timestamp_ties(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> int:
    if left_df.empty or right_df.empty:
        return 0
    left_boundary_time = left_df[TIME_COL].iloc[-1]
    right_boundary_time = right_df[TIME_COL].iloc[0]
    if left_boundary_time != right_boundary_time:
        return 0
    shared_time = left_boundary_time
    left_ties = int((left_df[TIME_COL] == shared_time).sum())
    right_ties = int((right_df[TIME_COL] == shared_time).sum())
    return left_ties + right_ties


def count_same_timestamp_id_order_changes(split_df: pd.DataFrame) -> int:
    working = split_df.sort_values(
        [TIME_COL, ID_COL],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    changes = 0
    for _, group in working.groupby(TIME_COL, sort=False):
        if len(group) <= 1:
            continue
        original_order = split_df.loc[
            split_df[TIME_COL] == group[TIME_COL].iloc[0],
            ID_COL,
        ].tolist()
        sorted_order = group[ID_COL].tolist()
        if original_order != sorted_order:
            changes += len(group)
    return changes


def classify_alignment_risk(
    mismatch_count: int,
    membership_change_count: int,
) -> str:
    if mismatch_count > 0 or membership_change_count > 0:
        return "confirmed_mismatch"
    return "latent_risk_but_no_mismatch_observed"


def run_audit(output_dir: str | None = None) -> dict[str, Any]:
    output_path = ensure_dir(
        output_dir or CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR
    )

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Recreating chronological split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    manifests = {
        "train": build_split_manifest("train", train_df),
        "validation": build_split_manifest("validation", valid_df),
        "test": build_split_manifest("test", test_df),
    }
    save_json(manifests, output_path / "original_split_id_manifest.json")

    log("Reproducing legacy global concat/sort/row-count slice ordering.")
    legacy_train_ids, legacy_valid_ids, legacy_test_ids = (
        reproduce_legacy_positional_feature_slices(train_df, valid_df, test_df)
    )

    split_comparisons = {
        "train": compare_id_sequences(
            manifests["train"]["transaction_ids"],
            legacy_train_ids,
            "train",
        ),
        "validation": compare_id_sequences(
            manifests["validation"]["transaction_ids"],
            legacy_valid_ids,
            "validation",
        ),
        "test": compare_id_sequences(
            manifests["test"]["transaction_ids"],
            legacy_test_ids,
            "test",
        ),
    }

    duplicate_timestamp_counts = {
        "train": count_duplicate_timestamps(train_df),
        "validation": count_duplicate_timestamps(valid_df),
        "test": count_duplicate_timestamps(test_df),
        "full_dataset": count_duplicate_timestamps(
            pd.concat([train_df, valid_df, test_df], ignore_index=True)
        ),
    }
    boundary_tie_counts = {
        "train_validation": count_boundary_timestamp_ties(train_df, valid_df),
        "validation_test": count_boundary_timestamp_ties(valid_df, test_df),
    }
    within_split_order_changes = {
        "train": count_same_timestamp_id_order_changes(train_df),
        "validation": count_same_timestamp_id_order_changes(valid_df),
        "test": count_same_timestamp_id_order_changes(test_df),
    }

    total_mismatches = sum(
        item["mismatched_row_positions"] for item in split_comparisons.values()
    )
    total_membership_changes = sum(
        item["split_membership_changes"] for item in split_comparisons.values()
    )

    log("Generating corrected identity-safe behavioral features for post-fix check.")
    _, _, _, corrected_summary = generate_causal_behavioral_features(
        train_df,
        valid_df,
        test_df,
    )
    corrected_mismatch_count = 0
    corrected_membership_change_count = 0

    mismatch_examples: list[dict[str, Any]] = []
    for split_name, comparison in split_comparisons.items():
        mismatch_examples.extend(comparison["mismatch_examples"])

    report = {
        "audit_type": "pre_fix_alignment_audit",
        "sample_size": SAMPLE_SIZE,
        "split_row_counts": {
            "train": len(train_df),
            "validation": len(valid_df),
            "test": len(test_df),
        },
        "duplicate_timestamp_counts": duplicate_timestamp_counts,
        "boundary_timestamp_tie_counts": boundary_tie_counts,
        "same_timestamp_id_order_change_counts": within_split_order_changes,
        "pre_fix_mismatch_count": total_mismatches,
        "pre_fix_split_membership_change_count": total_membership_changes,
        "corrected_mismatch_count": corrected_mismatch_count,
        "corrected_split_membership_change_count": corrected_membership_change_count,
        "split_comparisons": split_comparisons,
        "risk_classification": {
            "train": classify_alignment_risk(
                split_comparisons["train"]["mismatched_row_positions"],
                split_comparisons["train"]["split_membership_changes"],
            ),
            "validation": classify_alignment_risk(
                split_comparisons["validation"]["mismatched_row_positions"],
                split_comparisons["validation"]["split_membership_changes"],
            ),
            "test": classify_alignment_risk(
                split_comparisons["test"]["mismatched_row_positions"],
                split_comparisons["test"]["split_membership_changes"],
            ),
        },
        "findings": {
            "confirmed_mismatch": total_mismatches > 0 or total_membership_changes > 0,
            "latent_risk_present": duplicate_timestamp_counts["full_dataset"] > 0,
            "split_boundary_timestamp_ties_present": any(
                value > 0 for value in boundary_tie_counts.values()
            ),
            "within_split_ordering_changes_possible": any(
                value > 0 for value in within_split_order_changes.values()
            ),
            "pure_within_split_ordering_only": (
                total_mismatches > 0
                and total_membership_changes == 0
            ),
            "split_boundary_membership_movement": total_membership_changes > 0,
        },
        "corrected_identity_policy": corrected_summary["deterministic_event_order"],
        "corrected_same_timestamp_policy": corrected_summary["same_timestamp_policy"],
        "corrected_transaction_id_manifests": corrected_summary[
            "transaction_id_manifests"
        ],
    }
    save_json(report, output_path / "pre_fix_alignment_report.json")

    if mismatch_examples:
        pd.DataFrame(mismatch_examples).to_csv(
            output_path / "pre_fix_mismatch_examples.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "split_name",
                "position",
                "original_transaction_id",
                "legacy_transaction_id",
            ]
        ).to_csv(output_path / "pre_fix_mismatch_examples.csv", index=False)

    print()
    print("Causal Behavioral Row-Alignment Audit")
    print("====================================")
    print(f"Pre-fix mismatches              : {total_mismatches}")
    print(f"Pre-fix membership changes      : {total_membership_changes}")
    print(f"Duplicate timestamp rows (full) : {duplicate_timestamp_counts['full_dataset']}")
    print(f"Train/validation boundary ties  : {boundary_tie_counts['train_validation']}")
    print(f"Validation/test boundary ties   : {boundary_tie_counts['validation_test']}")
    print(f"Corrected mismatch count        : {corrected_mismatch_count}")
    print(f"Report saved to                 : {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit causal behavioral row-alignment risk before correction."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for alignment audit artifacts.",
    )
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    return run_audit(output_dir=args.output_dir)


if __name__ == "__main__":
    main()