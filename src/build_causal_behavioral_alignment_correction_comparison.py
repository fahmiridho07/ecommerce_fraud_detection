"""Build corrected causal behavioral alignment comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    BASELINE_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
    RESULTS_DIR,
)
from utils import ensure_dir, save_json


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_metric(path: Path) -> dict[str, object]:
    return load_json(path)


def build_row(
    model_id: str,
    model_name: str,
    result_status: str,
    output_dir: Path,
    alignment_policy: str,
    positional_join_used: bool,
    transaction_id_join_verified: bool,
    behavioral_feature_count: int,
    cdv_reconstruction_error_used: bool,
    supersedes: str | None = None,
    caveat: str | None = None,
    validation_delta_vs_corrected_b2: float | None = None,
) -> dict[str, object]:
    metrics_val = load_metric(output_dir / "metrics_validation_selected_threshold.json")
    metrics_test = load_metric(output_dir / "metrics_test_selected_threshold.json")
    run_config = load_metric(output_dir / "run_config.json")
    p01_val = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    validation_ap = float(metrics_val["average_precision"])
    test_ap = float(metrics_test["average_precision"])
    return {
        "model_id": model_id,
        "model_name": model_name,
        "result_status": result_status,
        "alignment_policy": alignment_policy,
        "positional_join_used": positional_join_used,
        "transaction_id_join_verified": transaction_id_join_verified,
        "behavioral_feature_count": behavioral_feature_count,
        "cdv_reconstruction_error_used": cdv_reconstruction_error_used,
        "validation_average_precision": validation_ap,
        "test_average_precision": test_ap,
        "validation_delta_vs_p01": validation_ap - float(p01_val),
        "validation_delta_vs_corrected_b2": validation_delta_vs_corrected_b2,
        "test_delta_vs_p01": test_ap
        - float(
            load_metric(
                BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
            )["average_precision"]
        ),
        "selected_threshold": float(run_config.get("threshold_selection", {}).get(
            "selected_threshold",
            metrics_val.get("threshold", 0.0),
        )),
        "best_iteration": int(
            run_config.get("early_stopping", {}).get("best_iteration", 0)
        ),
        "feature_count": int(
            run_config.get(
                "final_feature_count",
                run_config.get("model_features_count", 0),
            )
        ),
        "metric_source": str(output_dir / "metrics_validation_selected_threshold.json"),
        "run_config_source": str(output_dir / "run_config.json"),
        "supersedes": supersedes,
        "caveat": caveat,
    }


def main() -> pd.DataFrame:
    cba01r_val = float(
        load_metric(
            CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )
    rows = [
        build_row(
            model_id="P01",
            model_name="Baseline LightGBM default",
            result_status="original_reference",
            output_dir=BASELINE_OUTPUT_DIR,
            alignment_policy="chronological_split_transactiondt_only",
            positional_join_used=False,
            transaction_id_join_verified=True,
            behavioral_feature_count=0,
            cdv_reconstruction_error_used=False,
            caveat="Primary chronological baseline reference.",
        ),
        build_row(
            model_id="CBA01",
            model_name="Causal behavioral LightGBM default (B2)",
            result_status="provisional_alignment_risk",
            output_dir=CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
            alignment_policy="legacy_global_resort_positional_slice",
            positional_join_used=True,
            transaction_id_join_verified=False,
            behavioral_feature_count=19,
            cdv_reconstruction_error_used=False,
            supersedes=None,
            caveat=(
                "Provisional result superseded by CBA01R due to possible "
                "TransactionID misalignment under duplicate TransactionDT values."
            ),
        ),
        build_row(
            model_id="CBA01R",
            model_name="Causal behavioral LightGBM ID-aligned (B2 corrected)",
            result_status="corrected_authoritative",
            output_dir=CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
            alignment_policy="split_precedence_transactiondt_transactionid_restore",
            positional_join_used=False,
            transaction_id_join_verified=True,
            behavioral_feature_count=19,
            cdv_reconstruction_error_used=False,
            supersedes="CBA01",
            caveat="Corrected identity-safe behavioral feature alignment.",
        ),
        build_row(
            model_id="CBA02",
            model_name="Causal behavioral + CDV recon (B3)",
            result_status="provisional_alignment_risk",
            output_dir=CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
            alignment_policy="legacy_global_resort_positional_slice",
            positional_join_used=True,
            transaction_id_join_verified=False,
            behavioral_feature_count=19,
            cdv_reconstruction_error_used=True,
            validation_delta_vs_corrected_b2=None,
            caveat=(
                "Provisional result superseded by CBA02R due to possible "
                "behavioral and CDV reconstruction-error misalignment."
            ),
        ),
        build_row(
            model_id="CBA02R",
            model_name="Causal behavioral + CDV recon ID-aligned (B3 corrected)",
            result_status="corrected_authoritative",
            output_dir=CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR,
            alignment_policy="split_precedence_transactiondt_transactionid_restore",
            positional_join_used=False,
            transaction_id_join_verified=True,
            behavioral_feature_count=19,
            cdv_reconstruction_error_used=True,
            validation_delta_vs_corrected_b2=float(
                load_metric(
                    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR
                    / "metrics_validation_selected_threshold.json"
                )["average_precision"]
            )
            - cba01r_val,
            supersedes="CBA02",
            caveat="Corrected B2 matrix plus one TransactionID-joined CDV error.",
        ),
    ]
    comparison = pd.DataFrame(rows)
    ensure_dir(CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE.parent)
    comparison.to_csv(CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE, index=False)

    results_dir = ensure_dir(RESULTS_DIR)
    comparison.to_csv(
        results_dir / "causal_behavioral_alignment_correction.csv",
        index=False,
    )
    manifest = {
        "comparison_csv": str(CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE),
        "tracked_summary_csv": str(
            results_dir / "causal_behavioral_alignment_correction.csv"
        ),
        "models": comparison[
            [
                "model_id",
                "result_status",
                "validation_average_precision",
                "test_average_precision",
            ]
        ].to_dict(orient="records"),
    }
    save_json(manifest, results_dir / "causal_behavioral_alignment_manifest.json")
    print(f"Saved comparison to {CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE}")
    return comparison


if __name__ == "__main__":
    main()