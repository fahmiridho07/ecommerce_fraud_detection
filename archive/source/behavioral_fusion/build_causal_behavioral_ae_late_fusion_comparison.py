"""Build LF01 controlled comparison table against P01-P04 and CBA01R."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    BASELINE_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE,
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    OPTUNA_OUTPUT_DIR,
)
from late_fusion_experts import REFERENCE_AP
from utils import ensure_dir, save_json


def load_metric(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def primary_row(
    model_id: str,
    model_name: str,
    model_type: str,
    output_dir: Path,
    behavioral_expert: str,
    ae_expert: str,
    behavioral_weight: float | None,
    ae_weight: float | None,
    practical_category: str | None,
    caveat: str,
) -> dict[str, object]:
    val_metrics = load_metric(
        output_dir / "metrics_validation_selected_threshold.json"
    )
    test_metrics = load_metric(
        output_dir / "metrics_test_selected_threshold.json"
    )
    run_config = load_json(output_dir / "run_config.json")
    val_ap = float(val_metrics["average_precision"])
    test_ap = float(test_metrics["average_precision"])
    cba01r_val = REFERENCE_AP["CBA01R"]["validation"]
    cba01r_test = REFERENCE_AP["CBA01R"]["test"]
    p04_val = REFERENCE_AP["P04"]["validation"]
    p04_test = REFERENCE_AP["P04"]["test"]
    p02_val = REFERENCE_AP["P02"]["validation"]
    p02_test = REFERENCE_AP["P02"]["test"]
    threshold = run_config.get("threshold_selection", {}).get("selected_threshold")
    if threshold is None and "selected_threshold" in run_config:
        threshold = run_config["selected_threshold"]

    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "behavioral_expert_used": behavioral_expert,
        "ae_expert_used": ae_expert,
        "behavioral_weight": behavioral_weight,
        "ae_weight": ae_weight,
        "validation_average_precision": val_ap,
        "test_average_precision": test_ap,
        "validation_delta_vs_cba01r": val_ap - cba01r_val,
        "test_delta_vs_cba01r": test_ap - cba01r_test,
        "validation_delta_vs_p04": val_ap - p04_val,
        "test_delta_vs_p04": test_ap - p04_test,
        "validation_delta_vs_p02": val_ap - p02_val,
        "test_delta_vs_p02": test_ap - p02_test,
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_mcc": float(test_metrics["mcc"]),
        "selected_threshold": threshold,
        "practical_result_category": practical_category,
        "metric_source": "metrics_validation_selected_threshold.json",
        "run_config_source": str(output_dir / "run_config.json"),
        "comparability": "chronological 60/20/20; identity-safe score regeneration",
        "caveat": caveat,
    }


def main() -> pd.DataFrame:
    fusion_dir = CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR
    frozen = load_json(fusion_dir / "frozen_fusion_config.json")
    practical = frozen.get("practical_result_category")

    rows = [
        primary_row(
            "P01",
            "original-feature LightGBM default",
            "primary_baseline_default",
            BASELINE_OUTPUT_DIR,
            "none",
            "none",
            None,
            None,
            None,
            "Default LightGBM; original 432 features.",
        ),
        primary_row(
            "P02",
            "original-feature LightGBM tuned",
            "primary_baseline_tuned",
            OPTUNA_OUTPUT_DIR / "baseline_lgbm",
            "none",
            "none",
            None,
            None,
            None,
            "Optuna-tuned LightGBM; original 432 features.",
        ),
        primary_row(
            "P04",
            "tuned V-only AE-LightGBM LD128",
            "primary_ae_replacement_tuned",
            OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128",
            "none",
            "none",
            None,
            None,
            None,
            "P04 uses tuned LightGBM and LD128 while thesis-original P03 uses LD32.",
        ),
        primary_row(
            "CBA01R",
            "identity-aligned causal behavioral LightGBM",
            "causal_behavioral_expert",
            CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
            "CBA01R",
            "none",
            1.0,
            0.0,
            None,
            "CBA01R uses default LightGBM, not Optuna tuning.",
        ),
        primary_row(
            "LF01",
            "Causal Behavioral + AE-LightGBM Late Fusion",
            "decision_level_late_fusion",
            fusion_dir,
            "CBA01R",
            "P04",
            float(frozen["behavioral_weight"]),
            float(frozen["ae_weight"]),
            practical,
            (
                "Fusion weight selected on validation only; simple linear probability "
                "fusion without calibration; mixed tuning status between experts; "
                "decision-level integration differs from feature-level integration; "
                "historical test inspection exists in earlier exploratory branches."
            ),
        ),
    ]

    table = pd.DataFrame(rows)
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table.to_csv(CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE, index=False)
    save_json(
        {
            "rows": len(table),
            "output_file": str(CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE),
        },
        fusion_dir / "comparison_build_summary.json",
    )
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()