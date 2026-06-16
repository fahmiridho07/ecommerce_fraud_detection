"""Build controlled Autoencoder input-scope comparison CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    AE_LGBM_LD128_OUTPUT_DIR,
    AE_LGBM_OUTPUT_DIR,
    AUTOENCODER_INPUT_SCOPE_COMPARISON_FILE,
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
)
from utils import ensure_dir, save_json


def load_metric(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> pd.DataFrame:
    p01_val = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    p01_test = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )["average_precision"]
    p01_run = load_json(BASELINE_OUTPUT_DIR / "run_config.json")

    v_ld128_val = load_metric(
        AE_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    v_ld128_test = load_metric(
        AE_LGBM_LD128_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )["average_precision"]
    v_ld128_run = load_json(AE_LGBM_LD128_OUTPUT_DIR / "run_config.json")
    v_ae_run = load_json(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "run_config.json")

    sel_val = load_metric(
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
        / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    sel_test = load_metric(
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
        / "metrics_test_selected_threshold.json"
    )["average_precision"]
    sel_lgbm_run = load_json(SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json")
    sel_ae_run = load_json(AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR / "run_config.json")

    rows = [
        {
            "model_name": "P01_baseline_lgbm_default",
            "autoencoder_input_scope": "none",
            "ae_input_feature_count": 0,
            "v_feature_count": 339,
            "additional_numerical_feature_count": 0,
            "replacement_strategy": "none",
            "original_ae_input_features_retained": True,
            "latent_dimension": 0,
            "final_lightgbm_feature_count": p01_run["model_features_count"],
            "validation_average_precision": p01_val,
            "test_average_precision": p01_test,
            "validation_delta_vs_p01": 0.0,
            "test_delta_vs_p01": 0.0,
            "validation_delta_vs_v_only_ld128": p01_val - v_ld128_val,
            "test_delta_vs_v_only_ld128": p01_test - v_ld128_test,
            "selected_threshold": p01_run["threshold_selection"]["selected_threshold"],
            "best_iteration": p01_run["early_stopping"]["best_iteration"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(BASELINE_OUTPUT_DIR / "run_config.json"),
            "comparison_status": "primary_control",
            "caveat": "Original-feature chronological baseline.",
        },
        {
            "model_name": "AE05_v_only_replacement_ld128_default",
            "autoencoder_input_scope": "V1-V339_only",
            "ae_input_feature_count": v_ae_run["v_feature_count"],
            "v_feature_count": v_ae_run["v_feature_count"],
            "additional_numerical_feature_count": 0,
            "replacement_strategy": "latent_replacement",
            "original_ae_input_features_retained": False,
            "latent_dimension": v_ld128_run["feature_construction"]["latent_feature_count"],
            "final_lightgbm_feature_count": v_ld128_run["feature_construction"]["total_feature_count"],
            "validation_average_precision": v_ld128_val,
            "test_average_precision": v_ld128_test,
            "validation_delta_vs_p01": v_ld128_val - p01_val,
            "test_delta_vs_p01": v_ld128_test - p01_test,
            "validation_delta_vs_v_only_ld128": 0.0,
            "test_delta_vs_v_only_ld128": 0.0,
            "selected_threshold": v_ld128_run["threshold_selection"]["selected_threshold"],
            "best_iteration": v_ld128_run["early_stopping"]["best_iteration"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(AE_LGBM_LD128_OUTPUT_DIR / "run_config.json"),
            "comparison_status": "primary_control",
            "caveat": "Thesis V-only AE replacement LD128 default.",
        },
        {
            "model_name": "AAE01_selected_numerical_replacement_ld128_default",
            "autoencoder_input_scope": "selected_numerical_predictors",
            "ae_input_feature_count": sel_ae_run["selected_feature_count"],
            "v_feature_count": sel_ae_run["v_feature_count"],
            "additional_numerical_feature_count": sel_ae_run["additional_numerical_feature_count"],
            "replacement_strategy": "latent_replacement",
            "original_ae_input_features_retained": False,
            "latent_dimension": sel_ae_run["latent_dimension"],
            "final_lightgbm_feature_count": sel_lgbm_run["final_feature_count"],
            "validation_average_precision": sel_val,
            "test_average_precision": sel_test,
            "validation_delta_vs_p01": sel_val - p01_val,
            "test_delta_vs_p01": sel_test - p01_test,
            "validation_delta_vs_v_only_ld128": sel_val - v_ld128_val,
            "test_delta_vs_v_only_ld128": sel_test - v_ld128_test,
            "selected_threshold": sel_lgbm_run["selected_threshold"],
            "best_iteration": sel_lgbm_run["best_iteration"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json"),
            "comparison_status": "anchor_alignment_diagnostic",
            "caveat": (
                "Broadened AE input scope; validation AP determines interpretation; "
                "test AP descriptive only."
            ),
        },
    ]

    if (AE_LGBM_OUTPUT_DIR / "metrics_validation_selected_threshold.json").exists():
        v_ld32_val = load_metric(
            AE_LGBM_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
        )["average_precision"]
        v_ld32_test = load_metric(
            AE_LGBM_OUTPUT_DIR / "metrics_test_selected_threshold.json"
        )["average_precision"]
        v_ld32_run = load_json(AE_LGBM_OUTPUT_DIR / "run_config.json")
        rows.append(
            {
                "model_name": "P03_v_only_replacement_ld32_default",
                "autoencoder_input_scope": "V1-V339_only",
                "ae_input_feature_count": 339,
                "v_feature_count": 339,
                "additional_numerical_feature_count": 0,
                "replacement_strategy": "latent_replacement",
                "original_ae_input_features_retained": False,
                "latent_dimension": v_ld32_run["feature_construction"]["latent_feature_count"],
                "final_lightgbm_feature_count": v_ld32_run["feature_construction"]["total_feature_count"],
                "validation_average_precision": v_ld32_val,
                "test_average_precision": v_ld32_test,
                "validation_delta_vs_p01": v_ld32_val - p01_val,
                "test_delta_vs_p01": v_ld32_test - p01_test,
                "validation_delta_vs_v_only_ld128": v_ld32_val - v_ld128_val,
                "test_delta_vs_v_only_ld128": v_ld32_test - v_ld128_test,
                "selected_threshold": v_ld32_run["threshold_selection"]["selected_threshold"],
                "best_iteration": v_ld32_run["early_stopping"]["best_iteration"],
                "metric_source": "metrics_validation_selected_threshold.json",
                "run_config_source": str(AE_LGBM_OUTPUT_DIR / "run_config.json"),
                "comparison_status": "historical_context_only",
                "caveat": "Thesis-original LD32 replacement; separated from primary LD128 comparison.",
            }
        )

    comparison_df = pd.DataFrame(rows)
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    comparison_df.to_csv(AUTOENCODER_INPUT_SCOPE_COMPARISON_FILE, index=False)
    print(f"Saved comparison to {AUTOENCODER_INPUT_SCOPE_COMPARISON_FILE}")
    return comparison_df


if __name__ == "__main__":
    main()