"""Summarize FE + AE controlled experiments and fixed reference metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    FE_AE_AUGMENTED_LGBM_OUTPUT_DIR,
    FE_AE_CONTROLLED_COMPARISON_FILE,
    FE_AE_CONTROLLED_OUTPUT_DIR,
    FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
    FE_RECON_ERROR_LGBM_OUTPUT_DIR,
)
from utils import ensure_dir


COMPARISON_COLUMNS = [
    "model_name",
    "stage",
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
    "selected_fe_weight",
    "selected_ae_weight",
    "stop_after_a",
    "run_b_next",
    "run_c_next",
    "consider_optuna_later",
    "output_dir",
]


REFERENCE_ROWS = [
    {
        "model_name": "reference_fe_lgbm_default",
        "stage": "reference",
        "validation_pr_auc": 0.6277932473974428,
        "test_pr_auc": 0.5091169248916745,
        "test_roc_auc": 0.8884105927343418,
        "test_precision": 0.7064056939501779,
        "test_recall": 0.390748031496063,
        "test_f1": 0.5031685678073511,
        "test_mcc": 0.5135286207915644,
        "selected_threshold": 0.65,
        "best_iteration": 1162,
        "total_features": 519,
        "selected_fe_weight": None,
        "selected_ae_weight": None,
        "stop_after_a": None,
        "run_b_next": None,
        "run_c_next": None,
        "consider_optuna_later": None,
        "output_dir": "fixed_reference",
    },
    {
        "model_name": "reference_fe_lgbm_tuned",
        "stage": "reference",
        "validation_pr_auc": 0.6543163969719032,
        "test_pr_auc": 0.529856621916188,
        "test_roc_auc": 0.8946015957855195,
        "test_precision": 0.6511371973587674,
        "test_recall": 0.43676181102362205,
        "test_f1": 0.5228276877761414,
        "test_mcc": 0.5200604345432417,
        "selected_threshold": 0.31,
        "best_iteration": 1734,
        "total_features": 519,
        "selected_fe_weight": None,
        "selected_ae_weight": None,
        "stop_after_a": None,
        "run_b_next": None,
        "run_c_next": None,
        "consider_optuna_later": None,
        "output_dir": "fixed_reference",
    },
]


EXPERIMENTS = [
    (
        "A_score_ensemble_fe_tuned_ae_tuned",
        "A",
        FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
    ),
    (
        "B_fe_lgbm_reconstruction_mse_default",
        "B",
        FE_RECON_ERROR_LGBM_OUTPUT_DIR,
    ),
    (
        "C_fe_lgbm_latent128_reconstruction_mse_default",
        "C",
        FE_AE_AUGMENTED_LGBM_OUTPUT_DIR,
    ),
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
        for key in ("total_feature_count", "total_final_features"):
            if key in feature_construction:
                return feature_construction[key]

    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def weights_from_run_config(run_config: dict[str, object]) -> tuple[object, object]:
    ensemble = run_config.get("ensemble", {})
    if not isinstance(ensemble, dict):
        return None, None
    return (
        ensemble.get("selected_fe_lgbm_tuned_weight"),
        ensemble.get("selected_ae_lgbm_ld128_tuned_weight"),
    )


def stopping_values(run_config: dict[str, object]) -> dict[str, object]:
    stopping = run_config.get("stopping_criteria", {})
    if not isinstance(stopping, dict):
        return {
            "stop_after_a": None,
            "run_b_next": None,
            "run_c_next": None,
            "consider_optuna_later": None,
        }
    return {
        "stop_after_a": stopping.get("stop_after_a"),
        "run_b_next": stopping.get("run_b_next"),
        "run_c_next": stopping.get("run_c_next"),
        "consider_optuna_later": stopping.get("consider_optuna_later"),
    }


def experiment_row(
    model_name: str,
    stage: str,
    output_dir: Path,
) -> dict[str, object] | None:
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
    selected_fe_weight, selected_ae_weight = weights_from_run_config(run_config)
    stopping = stopping_values(run_config)

    row = {
        "model_name": model_name,
        "stage": stage,
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
        "selected_fe_weight": selected_fe_weight,
        "selected_ae_weight": selected_ae_weight,
        "output_dir": str(output_dir),
    }
    row.update(stopping)
    return row


def build_comparison_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = list(REFERENCE_ROWS)
    for model_name, stage, output_dir in EXPERIMENTS:
        row = experiment_row(model_name, stage, output_dir)
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
    ensure_dir(FE_AE_CONTROLLED_OUTPUT_DIR)
    table = build_comparison_table()
    table.to_csv(FE_AE_CONTROLLED_COMPARISON_FILE, index=False)

    print()
    print("FE + AE Controlled Experiments Comparison")
    print("=========================================")
    print(table.to_string(index=False))
    print(f"\nSaved comparison to: {FE_AE_CONTROLLED_COMPARISON_FILE}")
    return table


if __name__ == "__main__":
    main()
