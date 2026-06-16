"""Build TAE01 controlled comparison CSV (P01 vs AAE01 vs TAE01)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    BASELINE_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    TASK_AWARE_AE_COMPARISON_FILE,
    TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR,
    TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
)
from utils import ensure_dir


def load_metric(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def interpretation_rule(
    tae_val: float,
    p01_val: float,
    aae01_val: float,
) -> str:
    if tae_val > p01_val and tae_val > aae01_val:
        return (
            "Task-aware latent learning provides evidence that jointly optimizing "
            "reconstruction and fraud discrimination improves AE–LightGBM effectiveness "
            "under the executed chronological protocol."
        )
    if tae_val > aae01_val and tae_val <= p01_val:
        return (
            "Task-aware supervision improves the latent representation relative to the "
            "unsupervised Autoencoder, but does not outperform original-feature LightGBM."
        )
    if tae_val <= aae01_val:
        return (
            "Adding the supervised classification objective does not improve the "
            "selected-numerical latent representation for downstream LightGBM under "
            "the executed protocol."
        )
    return (
        "The task-aware experiment is inconclusive because a verified implementation "
        "or comparison mismatch prevents fair interpretation."
    )


def main() -> pd.DataFrame:
    p01_val = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    p01_test = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )["average_precision"]
    p01_run = load_json(BASELINE_OUTPUT_DIR / "run_config.json")

    aae01_val = load_metric(
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
        / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    aae01_test = load_metric(
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
        / "metrics_test_selected_threshold.json"
    )["average_precision"]
    aae01_run = load_json(SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json")

    tae_lgbm_dir = TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR / "selected"
    tae_val = load_metric(
        tae_lgbm_dir / "metrics_validation_selected_threshold.json"
    )["average_precision"]
    tae_test_metrics = load_metric(
        tae_lgbm_dir / "metrics_test_selected_threshold.json"
    )
    tae_test = tae_test_metrics["average_precision"]
    tae_run = load_json(tae_lgbm_dir / "run_config.json")
    tae_ae_run = load_json(
        TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR
        / "selected"
        / "run_config.json"
    )

    rows = [
        {
            "model_id": "P01",
            "model_name": "P01_original_feature_lgbm_default",
            "representation_type": "original_features",
            "supervision_type": "supervised_downstream_only",
            "ae_input_scope": "none",
            "selected_numerical_feature_count": 432,
            "latent_dimension": 0,
            "lambda_classification": None,
            "original_selected_features_retained": True,
            "retained_raw_feature_count": 432,
            "final_feature_count": p01_run["model_features_count"],
            "validation_average_precision": p01_val,
            "test_average_precision": p01_test,
            "validation_delta_vs_p01": 0.0,
            "test_delta_vs_p01": 0.0,
            "validation_delta_vs_aae01": p01_val - aae01_val,
            "test_delta_vs_aae01": p01_test - aae01_test,
            "test_roc_auc": load_metric(
                BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
            )["roc_auc"],
            "test_precision": load_metric(
                BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
            )["precision"],
            "test_recall": load_metric(
                BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
            )["recall"],
            "test_f1": load_metric(
                BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
            )["f1"],
            "test_mcc": load_metric(
                BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
            )["mcc"],
            "selected_threshold": p01_run["threshold_selection"]["selected_threshold"],
            "best_iteration": p01_run["early_stopping"]["best_iteration"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(BASELINE_OUTPUT_DIR / "run_config.json"),
            "caveat": "Primary chronological baseline control.",
        },
        {
            "model_id": "AAE01",
            "model_name": "AAE01_selected_numerical_unsupervised_latent_replacement_ld128",
            "representation_type": "unsupervised_latent_replacement",
            "supervision_type": "unsupervised_ae",
            "ae_input_scope": "selected_numerical_predictors",
            "selected_numerical_feature_count": aae01_run["removed_numerical_feature_count"],
            "latent_dimension": aae01_run["latent_feature_count"],
            "lambda_classification": None,
            "original_selected_features_retained": False,
            "retained_raw_feature_count": aae01_run["retained_raw_feature_count"],
            "final_feature_count": aae01_run["final_feature_count"],
            "validation_average_precision": aae01_val,
            "test_average_precision": aae01_test,
            "validation_delta_vs_p01": aae01_val - p01_val,
            "test_delta_vs_p01": aae01_test - p01_test,
            "validation_delta_vs_aae01": 0.0,
            "test_delta_vs_aae01": 0.0,
            "test_roc_auc": load_metric(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
                / "metrics_test_selected_threshold.json"
            )["roc_auc"],
            "test_precision": load_metric(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
                / "metrics_test_selected_threshold.json"
            )["precision"],
            "test_recall": load_metric(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
                / "metrics_test_selected_threshold.json"
            )["recall"],
            "test_f1": load_metric(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
                / "metrics_test_selected_threshold.json"
            )["f1"],
            "test_mcc": load_metric(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
                / "metrics_test_selected_threshold.json"
            )["mcc"],
            "selected_threshold": aae01_run["selected_threshold"],
            "best_iteration": aae01_run["best_iteration"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json"
            ),
            "caveat": (
                "Unsupervised selected-numerical latent replacement; validation AP "
                "determines interpretation."
            ),
        },
        {
            "model_id": "TAE01",
            "model_name": "TAE01_selected_numerical_task_aware_latent_replacement_ld128",
            "representation_type": "task_aware_latent_replacement",
            "supervision_type": "joint_reconstruction_classification_ae",
            "ae_input_scope": "selected_numerical_predictors",
            "selected_numerical_feature_count": tae_ae_run["selected_numerical_feature_count"],
            "latent_dimension": tae_ae_run["latent_dimension"],
            "lambda_classification": tae_ae_run["lambda_classification"],
            "original_selected_features_retained": False,
            "retained_raw_feature_count": tae_run["retained_raw_feature_count"],
            "final_feature_count": tae_run["final_feature_count"],
            "validation_average_precision": tae_val,
            "test_average_precision": tae_test,
            "validation_delta_vs_p01": tae_val - p01_val,
            "test_delta_vs_p01": tae_test - p01_test,
            "validation_delta_vs_aae01": tae_val - aae01_val,
            "test_delta_vs_aae01": tae_test - aae01_test,
            "test_roc_auc": tae_test_metrics["roc_auc"],
            "test_precision": tae_test_metrics["precision"],
            "test_recall": tae_test_metrics["recall"],
            "test_f1": tae_test_metrics["f1"],
            "test_mcc": tae_test_metrics["mcc"],
            "selected_threshold": tae_run["selected_threshold"],
            "best_iteration": tae_run["best_iteration"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(tae_lgbm_dir / "run_config.json"),
            "caveat": (
                "Selected lambda from validation-only downstream LightGBM AP; "
                "test AP descriptive only."
            ),
        },
    ]

    comparison_df = pd.DataFrame(rows)
    comparison_df["interpretation_rule"] = interpretation_rule(
        float(tae_val),
        float(p01_val),
        float(aae01_val),
    )
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    comparison_df.to_csv(TASK_AWARE_AE_COMPARISON_FILE, index=False)
    print(f"Saved comparison to {TASK_AWARE_AE_COMPARISON_FILE}")
    print(f"Interpretation: {comparison_df.loc[2, 'interpretation_rule']}")
    return comparison_df


if __name__ == "__main__":
    main()