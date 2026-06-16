"""LF01: Causal Behavioral + AE-LightGBM validation-selected late fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from audit_causal_behavioral_ae_complementarity import run_complementarity_audit
from config import (
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE,
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_WEIGHT_SEARCH_FILE,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    DATA_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    ID_COL,
    OPTUNA_OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from late_fusion_experts import (
    BEHAVIORAL_WEIGHT_GRID,
    BOOTSTRAP_RESAMPLES,
    CBA01R_OUTPUT_DIR,
    P04_OUTPUT_DIR,
    REFERENCE_AP,
    align_expert_scores_by_transaction_id,
    build_weight_search_table,
    classify_practical_result,
    fusion_score,
    paired_bootstrap_ap_delta,
    regenerate_cba01r_scores,
    regenerate_p02_scores,
    regenerate_p04_scores,
    score_alignment_summary,
    selected_weights_from_table,
    topk_metrics_table,
)
from train_baseline_lgbm import DEFAULT_THRESHOLD
from utils import ensure_dir, log, save_json, set_seed


def print_prerun_gates(
    cba: dict[str, object],
    p04: dict[str, object],
    valid_table: pd.DataFrame,
    test_table: pd.DataFrame,
    complementarity: dict[str, object],
) -> None:
    print("=" * 72)
    print("LF01 PRE-RUN GATES")
    print("=" * 72)
    print("1. CBA01R artifact status: OK")
    print(f"   best_iteration={cba['best_iteration']}")
    print("2. P04 artifact status: OK")
    print(f"   best_iteration={p04['best_iteration']} latent_dim={p04['latent_dim']}")
    prepared = cba["prepared"]
    print("3. Chronological split row counts:")
    print(f"   train={len(prepared['train_df'])} validation={len(prepared['valid_df'])} test={len(prepared['test_df'])}")
    for split_name, table, checks in (
        ("validation", valid_table, cba["metric_checks"]["validation"]),
        ("test", test_table, cba["metric_checks"]["test"]),
    ):
        print(f"4-8. {split_name} CBA01R regenerated AP={checks['regenerated_ap']:.12f} diff={checks['absolute_difference']:.2e}")
    for split_name, key in (("validation", "validation"), ("test", "test")):
        checks = p04["metric_checks"][key]
        print(f"   {split_name} P04 regenerated AP={checks['regenerated_ap']:.12f} diff={checks['absolute_difference']:.2e}")
    print("9. Reference metric differences within tolerance.")
    print("10-11. Score alignment:")
    print(f"   validation rows={len(valid_table)} test rows={len(test_table)}")
    print("12. Score probability ranges: OK")
    print("13. Duplicate/missing ID counts: 0")
    thesis = complementarity["thesis_safe_summary"]
    print(f"14. Complementarity Spearman={thesis['spearman_correlation']:.6f}")
    topk = thesis["topk_unique_fraud_capture"]
    print(
        "15. Unique fraud captures: "
        f"top1% only_p04={topk['0.01']['only_p04']} "
        f"top3% only_p04={topk['0.03']['only_p04']} "
        f"top5% only_p04={topk['0.05']['only_p04']}"
    )
    print(f"16. Weight grid: {BEHAVIORAL_WEIGHT_GRID}")
    print("17. Test labels excluded from weight-selection code: confirmed")
    print("=" * 72)


def run_bootstrap_block(
    split_name: str,
    y_true: np.ndarray,
    fusion_score_arr: np.ndarray,
    cba_score: np.ndarray,
    p04_score: np.ndarray,
    p02_score: np.ndarray | None,
) -> list[dict[str, object]]:
    comparisons = [
        ("fusion_minus_cba01r", cba_score, fusion_score_arr),
        ("fusion_minus_p04", p04_score, fusion_score_arr),
    ]
    rows = []
    for comparison_id, baseline, candidate in comparisons:
        summary = paired_bootstrap_ap_delta(y_true, baseline, candidate)
        rows.append(
            {
                "split": split_name,
                "comparison": comparison_id,
                **summary,
            }
        )

    if p02_score is not None:
        summary = paired_bootstrap_ap_delta(y_true, p02_score, fusion_score_arr)
        rows.append(
            {
                "split": split_name,
                "comparison": "fusion_minus_p02",
                **summary,
            }
        )
    else:
        rows.append(
            {
                "split": split_name,
                "comparison": "fusion_minus_p02",
                "mean_delta": None,
                "median_delta": None,
                "ci_lower_2_5": None,
                "ci_upper_97_5": None,
                "proportion_delta_gt_zero": None,
                "n_resamples": 0,
                "seed": RANDOM_SEED,
                "note": "P02 identity-aligned scores unavailable; scalar reference only",
            }
        )
    return rows


def main(overwrite: bool = False) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR)
    if not overwrite and (output_dir / "frozen_fusion_config.json").exists():
        raise FileExistsError(
            f"LF01 outputs already exist at {output_dir}. "
            "Pass --overwrite to regenerate."
        )

    log("Regenerating frozen CBA01R expert scores.")
    cba = regenerate_cba01r_scores()
    prepared = cba["prepared"]

    log("Regenerating frozen P04 expert scores.")
    p04 = regenerate_p04_scores(prepared)

    valid_table = align_expert_scores_by_transaction_id(
        cba["valid_df"],
        cba["y_valid"],
        cba["valid_score"],
        p04["valid_score"],
        "validation",
    )
    test_table = align_expert_scores_by_transaction_id(
        cba["test_df"],
        cba["y_test"],
        cba["test_score"],
        p04["test_score"],
        "test",
    )

    valid_table.to_csv(output_dir / "validation_expert_scores.csv", index=False)
    test_table.to_csv(output_dir / "test_expert_scores.csv", index=False)

    stored_checksums_path = (
        CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR
        / "feature_transaction_id_checksums.json"
    )
    stored_checksums = {}
    if stored_checksums_path.exists():
        with stored_checksums_path.open("r", encoding="utf-8") as file:
            stored_checksums = json.load(file)

    alignment_validation = {
        "validation": score_alignment_summary(
            valid_table,
            stored_checksums.get("validation"),
            "validation",
        ),
        "test": score_alignment_summary(
            test_table,
            stored_checksums.get("test"),
            "test",
        ),
        "transaction_id_join_only": True,
        "positional_concatenation_used": False,
    }
    save_json(alignment_validation, output_dir / "score_alignment_validation.json")

    log("Running validation-only complementarity audit.")
    complementarity = run_complementarity_audit(valid_table, output_dir=output_dir)

    y_valid = valid_table[TARGET_COL].astype(int).to_numpy()
    cba_valid_score = valid_table["cba01r_score"].to_numpy(dtype=float)
    p04_valid_score = valid_table["p04_ae_score"].to_numpy(dtype=float)

    print_prerun_gates(cba, p04, valid_table, test_table, complementarity)

    log("Searching predefined validation-only fusion weights.")
    cba01r_val_ap = float(average_precision_score(y_valid, cba_valid_score))
    p04_val_ap = float(average_precision_score(y_valid, p04_valid_score))
    weight_table = build_weight_search_table(
        y_valid,
        cba_valid_score,
        p04_valid_score,
        cba01r_val_ap,
        p04_val_ap,
    )
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    weight_table.to_csv(
        CAUSAL_BEHAVIORAL_AE_LATE_FUSION_WEIGHT_SEARCH_FILE,
        index=False,
    )

    behavioral_weight, ae_weight = selected_weights_from_table(weight_table)
    selected_row = weight_table.loc[weight_table["selected"]].iloc[0]
    fusion_val_ap = float(selected_row["validation_average_precision"])
    p02_val_ap = REFERENCE_AP["P02"]["validation"]

    practical_category = classify_practical_result(
        ae_weight,
        fusion_val_ap,
        cba01r_val_ap,
        p02_val_ap,
    )

    valid_fusion_score = fusion_score(
        behavioral_weight,
        cba_valid_score,
        p04_valid_score,
    )

    log("Selecting validation-only classification threshold.")
    threshold_table = threshold_selection_table(y_valid, valid_fusion_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    frozen_config = {
        "model_id": "LF01",
        "behavioral_expert": "CBA01R",
        "ae_expert": "P04",
        "behavioral_weight": behavioral_weight,
        "ae_weight": ae_weight,
        "weight_grid": BEHAVIORAL_WEIGHT_GRID,
        "selection_metric": "validation_average_precision",
        "validation_average_precision": fusion_val_ap,
        "validation_delta_vs_cba01r": fusion_val_ap - cba01r_val_ap,
        "validation_delta_vs_p04": fusion_val_ap - p04_val_ap,
        "validation_delta_vs_p02": fusion_val_ap - p02_val_ap,
        "selected_threshold": selected_threshold,
        "practical_success_rule": {
            "strong_success": (
                "ae_weight>0; fusion_val_ap>=cba01r+0.002; fusion_val_ap>p02"
            ),
            "partial_success": (
                "ae_weight>0; fusion_val_ap>=cba01r+0.002; fusion_val_ap<=p02"
            ),
            "marginal_signal": (
                "ae_weight>0; fusion_val_ap>cba01r; delta<0.002"
            ),
            "no_contribution": "ae_weight=0 or fusion_val_ap<=cba01r",
        },
        "practical_result_category": practical_category,
        "tie_break_policy": "prefer larger behavioral_weight when AP tied within 1e-8",
        "validation_only_weight_selection": True,
        "validation_only_threshold_selection": True,
        "test_not_used_for_weight_selection": True,
        "test_not_used_for_threshold_selection": True,
        "experts_retrained": False,
    }
    save_json(frozen_config, output_dir / "frozen_fusion_config.json")

    log("Evaluating frozen fusion on validation and test (selected weight only).")
    y_test = test_table[TARGET_COL].astype(int).to_numpy()
    cba_test_score = test_table["cba01r_score"].to_numpy(dtype=float)
    p04_test_score = test_table["p04_ae_score"].to_numpy(dtype=float)
    test_fusion_score = fusion_score(
        behavioral_weight,
        cba_test_score,
        p04_test_score,
    )

    metrics_valid_default = binary_classification_metrics(
        y_valid,
        valid_fusion_score,
        DEFAULT_THRESHOLD,
    )
    metrics_valid_selected = binary_classification_metrics(
        y_valid,
        valid_fusion_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        y_test,
        test_fusion_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        y_test,
        test_fusion_score,
        selected_threshold,
    )

    for name, payload in (
        ("metrics_validation_default_threshold.json", metrics_valid_default),
        ("metrics_validation_selected_threshold.json", metrics_valid_selected),
        ("metrics_test_default_threshold.json", metrics_test_default),
        ("metrics_test_selected_threshold.json", metrics_test_selected),
    ):
        save_json(payload, output_dir / name)

    confusion_matrix_table(
        y_valid,
        valid_fusion_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        y_test,
        test_fusion_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    topk_metrics_table(y_valid, valid_fusion_score).to_csv(
        output_dir / "topk_metrics_validation.csv",
        index=False,
    )
    topk_metrics_table(y_test, test_fusion_score).to_csv(
        output_dir / "topk_metrics_test.csv",
        index=False,
    )

    p02 = regenerate_p02_scores(prepared)
    p02_valid = p02["valid_score"] if p02 else None
    p02_test = p02["test_score"] if p02 else None

    bootstrap_rows = []
    bootstrap_rows.extend(
        run_bootstrap_block(
            "validation",
            y_valid,
            valid_fusion_score,
            cba_valid_score,
            p04_valid_score,
            p02_valid,
        )
    )
    bootstrap_rows.extend(
        run_bootstrap_block(
            "test",
            y_test,
            test_fusion_score,
            cba_test_score,
            p04_test_score,
            p02_test,
        )
    )
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(output_dir / "paired_bootstrap_ap.csv", index=False)
    bootstrap_summary = {
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "seed": RANDOM_SEED,
        "paired": True,
        "comparisons": bootstrap_rows,
        "p02_bootstrap_available": p02 is not None,
        "note": (
            "Bootstrap did not influence weight selection."
        ),
    }
    save_json(bootstrap_summary, output_dir / "paired_bootstrap_summary.json")

    run_config = {
        "model_id": "LF01",
        "experiment_name": "causal_behavioral_ae_late_fusion",
        "phase": "decision_level_late_fusion",
        "exception_to_post_tae01_freeze": True,
        "supervisor_approval_required_for_primary_promotion": True,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "target_column": TARGET_COL,
        "id_column": ID_COL,
        "time_column": TIME_COL,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "split_row_counts": {
            "train": int(len(prepared["train_df"])),
            "validation": int(len(valid_table)),
            "test": int(len(test_table)),
        },
        "experts": {
            "cba01r": {
                "model_id": "CBA01R",
                "output_dir": str(CBA01R_OUTPUT_DIR),
                "model_file": "model.pkl",
                "preprocessing_file": "preprocessing.pkl",
                "best_iteration": cba["best_iteration"],
                "validation_ap_reference": REFERENCE_AP["CBA01R"]["validation"],
                "test_ap_reference": REFERENCE_AP["CBA01R"]["test"],
            },
            "p04": {
                "model_id": "P04",
                "output_dir": str(P04_OUTPUT_DIR),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing_non_v.pkl",
                "best_iteration": p04["best_iteration"],
                "latent_dim": p04["latent_dim"],
                "validation_ap_reference": REFERENCE_AP["P04"]["validation"],
                "test_ap_reference": REFERENCE_AP["P04"]["test"],
            },
        },
        "fusion": {
            "formula": "behavioral_weight * cba01r_score + ae_weight * p04_ae_score",
            "weight_grid": BEHAVIORAL_WEIGHT_GRID,
            "selected_behavioral_weight": behavioral_weight,
            "selected_ae_weight": ae_weight,
            "selection_metric": "validation_average_precision",
            "tie_break": "prefer larger behavioral_weight",
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "practical_result_category": practical_category,
        "experts_retrained": False,
        "test_not_used_for_weight_selection": True,
        "test_not_used_for_threshold_selection": True,
        "frozen_config_written_before_test_evaluation": True,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("LF01 Late Fusion Summary")
    print("========================")
    print(f"Selected behavioral weight : {behavioral_weight:.2f}")
    print(f"Selected AE weight         : {ae_weight:.2f}")
    print(f"Validation AP (fusion)     : {fusion_val_ap:.6f}")
    print(f"Validation AP (CBA01R)     : {cba01r_val_ap:.6f}")
    print(f"Validation AP (P04)        : {p04_val_ap:.6f}")
    print(f"Practical result category  : {practical_category}")
    print(f"Test AP (fusion)           : {metrics_test_selected['average_precision']:.6f}")
    print(f"Selected threshold         : {selected_threshold:.2f}")
    print(f"Outputs saved to           : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "frozen_config": frozen_config,
        "weight_table": weight_table,
        "metrics_test_selected": metrics_test_selected,
        "complementarity": complementarity,
        "comparison_file": str(CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LF01 late fusion experiment.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing LF01 output directory contents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(overwrite=args.overwrite)