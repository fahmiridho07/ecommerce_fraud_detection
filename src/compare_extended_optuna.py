"""Compare original and extended Optuna tuning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
    FE_AE_CONTROLLED_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    OPTUNA_OUTPUT_DIR,
    OUTPUT_DIR,
)
from utils import ensure_dir


EXTENDED_OPTUNA_DIR = OUTPUT_DIR / "optuna_extended"
COMPARISON_FILE = FINAL_COMPARISON_OUTPUT_DIR / "extended_optuna_comparison.csv"

COMPARISON_COLUMNS = [
    "model_name",
    "validation_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_f1",
    "test_mcc",
    "selected_threshold",
    "best_iteration",
    "total_completed_trials",
    "best_trial_number",
    "output_dir",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_value(metrics: dict[str, object], key: str) -> object:
    return metrics.get(key)


def n_trials_from_run_config(run_config: dict[str, object]) -> int | None:
    optuna_config = run_config.get("optuna", {})
    if isinstance(optuna_config, dict) and optuna_config.get("n_trials_completed") is not None:
        return int(optuna_config["n_trials_completed"])
    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def best_trial_number_from_file(output_dir: Path) -> object:
    best_params_path = output_dir / "best_params.json"
    if not best_params_path.exists():
        return None
    best_params = load_json(best_params_path)
    return best_params.get("best_trial_number")


def tuned_model_row(model_name: str, output_dir: Path) -> dict[str, object] | None:
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
        "test_f1": metric_value(test_metrics, "f1"),
        "test_mcc": metric_value(test_metrics, "mcc"),
        "selected_threshold": metric_value(test_metrics, "threshold"),
        "best_iteration": best_iteration_from_run_config(run_config),
        "total_completed_trials": n_trials_from_run_config(run_config),
        "best_trial_number": best_trial_number_from_file(output_dir),
        "output_dir": str(output_dir),
    }


def input_model_trial_sum(run_config: dict[str, object]) -> int | None:
    input_models = run_config.get("input_models", {})
    if not isinstance(input_models, dict):
        return None

    total = 0
    found = False
    for model_info in input_models.values():
        if not isinstance(model_info, dict):
            continue
        output_dir_value = model_info.get("output_dir")
        if not output_dir_value:
            continue
        input_run_config_path = Path(str(output_dir_value)) / "run_config.json"
        if not input_run_config_path.exists():
            continue
        input_run_config = load_json(input_run_config_path)
        n_trials = n_trials_from_run_config(input_run_config)
        if n_trials is None:
            continue
        total += n_trials
        found = True

    return total if found else None


def ensemble_row(model_name: str, output_dir: Path) -> dict[str, object] | None:
    valid_metrics_path = output_dir / "metrics_validation_selected_threshold.json"
    test_metrics_path = output_dir / "metrics_test_selected_threshold.json"
    run_config_path = output_dir / "run_config.json"
    if (
        not valid_metrics_path.exists()
        or not test_metrics_path.exists()
        or not run_config_path.exists()
    ):
        return None

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)
    run_config = load_json(run_config_path)
    return {
        "model_name": model_name,
        "validation_pr_auc": metric_value(valid_metrics, "average_precision"),
        "test_pr_auc": metric_value(test_metrics, "average_precision"),
        "test_roc_auc": metric_value(test_metrics, "roc_auc"),
        "test_f1": metric_value(test_metrics, "f1"),
        "test_mcc": metric_value(test_metrics, "mcc"),
        "selected_threshold": metric_value(test_metrics, "threshold"),
        "best_iteration": None,
        "total_completed_trials": input_model_trial_sum(run_config),
        "best_trial_number": None,
        "output_dir": str(output_dir),
    }


def build_comparison_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    candidates = [
        (
            tuned_model_row,
            "fe_lgbm_tuned_15_trials",
            OPTUNA_OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features",
        ),
        (
            tuned_model_row,
            "fe_lgbm_tuned_100_trials_extended",
            EXTENDED_OPTUNA_DIR / "baseline_lgbm_entity_time_amount_features_100_trials",
        ),
        (
            tuned_model_row,
            "ae_lgbm_ld128_tuned_15_trials",
            OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128",
        ),
        (
            tuned_model_row,
            "ae_lgbm_ld128_tuned_50_trials_extended",
            EXTENDED_OPTUNA_DIR / "ae_lgbm_ld128_50_trials",
        ),
        (
            tuned_model_row,
            "ae_lgbm_ld128_tuned_100_trials_extended",
            EXTENDED_OPTUNA_DIR / "ae_lgbm_ld128_100_trials",
        ),
        (
            ensemble_row,
            "fe_ae_score_ensemble_15_trial_inputs",
            FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
        ),
        (
            ensemble_row,
            "fe_ae_score_ensemble_fe100_ae50_extended",
            FE_AE_CONTROLLED_OUTPUT_DIR / "A_score_ensemble_fe100_ae50",
        ),
        (
            ensemble_row,
            "fe_ae_score_ensemble_fe100_ae100_extended",
            FE_AE_CONTROLLED_OUTPUT_DIR / "A_score_ensemble_fe100_ae100",
        ),
    ]

    for row_builder, model_name, output_dir in candidates:
        row = row_builder(model_name, output_dir)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if not table.empty:
        table = table.sort_values(
            ["test_pr_auc", "validation_pr_auc"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
    return table


def main() -> pd.DataFrame:
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table = build_comparison_table()
    table.to_csv(COMPARISON_FILE, index=False)

    print()
    print("Extended Optuna Comparison")
    print("==========================")
    if table.empty:
        print("No completed original or extended Optuna outputs are available yet.")
    else:
        print(table.to_string(index=False))
    print(f"\nSaved comparison to: {COMPARISON_FILE}")
    return table


if __name__ == "__main__":
    main()
