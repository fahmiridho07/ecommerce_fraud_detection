"""Build a compact comparison table for the next controlled experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    AE_LGBM_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR,
    NEXT_CONTROLLED_EXPERIMENTS_COMPARISON_FILE,
    OPTUNA_OUTPUT_DIR,
    RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR,
    RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR,
    RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR,
    RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR,
    SCORE_ENSEMBLE_TUNED_OUTPUT_DIR,
    UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
)
from utils import ensure_dir


COMPARISON_COLUMNS = [
    "model_name",
    "validation_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "selected_threshold",
    "best_iteration",
    "total_features",
    "output_dir",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_value(metrics: dict[str, object], key: str) -> object:
    return metrics.get(key)


def total_features_from_run_config(run_config: dict[str, object]) -> object:
    if "model_features_count" in run_config:
        return run_config["model_features_count"]

    feature_construction = run_config.get("feature_construction", {})
    if isinstance(feature_construction, dict):
        for key in (
            "total_feature_count",
            "total_final_features",
            "model_features_count",
        ):
            if key in feature_construction:
                return feature_construction[key]

    feature_set_summary = run_config.get("feature_set_summary", {})
    if isinstance(feature_set_summary, dict):
        for key in ("total_feature_count", "total_final_features"):
            if key in feature_set_summary:
                return feature_set_summary[key]

    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def comparison_row(model_name: str, output_dir: Path) -> dict[str, object] | None:
    valid_metrics_path = output_dir / "metrics_validation_selected_threshold.json"
    test_metrics_path = output_dir / "metrics_test_selected_threshold.json"
    run_config_path = output_dir / "run_config.json"
    if (
        not valid_metrics_path.exists()
        or not test_metrics_path.exists()
        or not run_config_path.exists()
    ):
        return None

    run_config = load_json(run_config_path)
    if run_config.get("final_training_completed") is False:
        return None

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)
    return {
        "model_name": model_name,
        "validation_pr_auc": metric_value(valid_metrics, "average_precision"),
        "test_pr_auc": metric_value(test_metrics, "average_precision"),
        "test_roc_auc": metric_value(test_metrics, "roc_auc"),
        "test_precision": metric_value(test_metrics, "precision"),
        "test_recall": metric_value(test_metrics, "recall"),
        "test_f1": metric_value(test_metrics, "f1"),
        "test_mcc": metric_value(test_metrics, "mcc"),
        "selected_threshold": metric_value(test_metrics, "threshold"),
        "best_iteration": best_iteration_from_run_config(run_config),
        "total_features": total_features_from_run_config(run_config),
        "output_dir": str(output_dir),
    }


def build_comparison_table() -> pd.DataFrame:
    candidates = [
        ("baseline_lgbm_default", BASELINE_OUTPUT_DIR),
        ("baseline_lgbm_tuned", OPTUNA_OUTPUT_DIR / "baseline_lgbm"),
        (
            "baseline_lgbm_entity_time_amount_features_default",
            FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
        ),
        (
            "baseline_lgbm_entity_time_amount_features_tuned",
            OPTUNA_OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features",
        ),
        (
            "baseline_lgbm_entity_time_amount_uid_features_default",
            UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
        ),
        (
            "baseline_lgbm_entity_time_amount_historical_velocity_features_default",
            HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR,
        ),
        ("ae_lgbm_ld128_default", AE_LGBM_LD128_OUTPUT_DIR),
        ("ae_lgbm_ld128_tuned", OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128"),
        (
            "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned",
            SCORE_ENSEMBLE_TUNED_OUTPUT_DIR,
        ),
        (
            "baseline_lgbm_plus_ae_reconstruction_mse_default",
            RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR,
        ),
        (
            "baseline_lgbm_plus_log1p_ae_reconstruction_mse_default",
            RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR,
        ),
        (
            "baseline_lgbm_plus_raw_log1p_ae_reconstruction_mse_default",
            RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR,
        ),
        (
            "baseline_lgbm_plus_normal_only_ae_reconstruction_mse_default",
            RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR,
        ),
    ]

    rows = []
    for model_name, output_dir in candidates:
        row = comparison_row(model_name, output_dir)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if not table.empty:
        table = table.sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
    return table


def main() -> pd.DataFrame:
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table = build_comparison_table()
    table.to_csv(NEXT_CONTROLLED_EXPERIMENTS_COMPARISON_FILE, index=False)

    print()
    print("Next Controlled Experiments Comparison")
    print("======================================")
    if table.empty:
        print("No completed comparison rows are available yet.")
    else:
        print(table.to_string(index=False))
    print(f"\nSaved comparison to: {NEXT_CONTROLLED_EXPERIMENTS_COMPARISON_FILE}")
    return table


if __name__ == "__main__":
    main()
