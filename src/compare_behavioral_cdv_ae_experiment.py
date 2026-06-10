"""Compare behavioral CDV AE Experiment A against fixed FE references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    BEHAVIORAL_CDV_AE_COMPARISON_FILE,
    BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR,
)
from utils import ensure_dir


EXPERIMENT_A_DIR_NAME = "A_fe_lgbm_cdv_reconstruction_mse_default"

REFERENCE_FE_DEFAULT_VALIDATION_PR_AUC = 0.6277932473974428
REFERENCE_FE_DEFAULT_TEST_PR_AUC = 0.5091169248916745
REFERENCE_FE_TUNED_VALIDATION_PR_AUC = 0.6543163969719032
REFERENCE_FE_TUNED_TEST_PR_AUC = 0.529856621916188
REFERENCE_CURRENT_FE_AE_VALIDATION_PR_AUC = 0.6599352534246169
REFERENCE_CURRENT_FE_AE_TEST_PR_AUC = 0.5339351404285598

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
    "delta_validation_pr_auc_vs_fe_default",
    "delta_test_pr_auc_vs_fe_default",
    "delta_validation_pr_auc_vs_fe_tuned",
    "delta_test_pr_auc_vs_fe_tuned",
    "delta_validation_pr_auc_vs_current_fe_ae",
    "delta_test_pr_auc_vs_current_fe_ae",
    "validation_promising",
    "latent_followup_allowed",
    "stop_after_a",
    "interpretation",
    "output_dir",
]

REFERENCE_ROWS = [
    {
        "model_name": "reference_fe_lgbm_default",
        "stage": "reference",
        "validation_pr_auc": REFERENCE_FE_DEFAULT_VALIDATION_PR_AUC,
        "test_pr_auc": REFERENCE_FE_DEFAULT_TEST_PR_AUC,
        "test_roc_auc": 0.8884105927343418,
        "test_precision": 0.7064056939501779,
        "test_recall": 0.390748031496063,
        "test_f1": 0.5031685678073511,
        "test_mcc": 0.5135286207915644,
        "selected_threshold": 0.65,
        "best_iteration": 1162,
        "total_features": 519,
        "validation_promising": None,
        "latent_followup_allowed": None,
        "stop_after_a": None,
        "interpretation": "Fixed FE-LGBM default reference.",
        "output_dir": "fixed_reference",
    },
    {
        "model_name": "reference_fe_lgbm_tuned",
        "stage": "reference",
        "validation_pr_auc": REFERENCE_FE_TUNED_VALIDATION_PR_AUC,
        "test_pr_auc": REFERENCE_FE_TUNED_TEST_PR_AUC,
        "test_roc_auc": 0.8946015957855195,
        "test_precision": 0.6511371973587674,
        "test_recall": 0.43676181102362205,
        "test_f1": 0.5228276877761414,
        "test_mcc": 0.5200604345432417,
        "selected_threshold": 0.31,
        "best_iteration": 1734,
        "total_features": 519,
        "validation_promising": None,
        "latent_followup_allowed": None,
        "stop_after_a": None,
        "interpretation": "Fixed tuned FE-LGBM reference.",
        "output_dir": "fixed_reference",
    },
    {
        "model_name": "reference_current_fe_ae_score_ensemble",
        "stage": "reference",
        "validation_pr_auc": REFERENCE_CURRENT_FE_AE_VALIDATION_PR_AUC,
        "test_pr_auc": REFERENCE_CURRENT_FE_AE_TEST_PR_AUC,
        "test_roc_auc": 0.9042153436013232,
        "test_precision": 0.6825079872204473,
        "test_recall": 0.42052165354330706,
        "test_f1": 0.5204019488428745,
        "test_mcc": 0.5232848912474045,
        "selected_threshold": 0.34,
        "best_iteration": None,
        "total_features": None,
        "validation_promising": None,
        "latent_followup_allowed": None,
        "stop_after_a": None,
        "interpretation": "Fixed current FE+AE score ensemble reference.",
        "output_dir": "fixed_reference",
    },
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
        return feature_construction.get("total_feature_count")

    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def stopping_values(run_config: dict[str, object]) -> dict[str, object]:
    stopping = run_config.get("stopping_criteria", {})
    if not isinstance(stopping, dict):
        return {
            "validation_promising": None,
            "latent_followup_allowed": None,
            "stop_after_a": None,
        }
    validation_delta = stopping.get("validation_pr_auc_delta_vs_default_fe")
    return {
        "validation_promising": (
            bool(validation_delta >= 0.005)
            if validation_delta is not None
            else None
        ),
        "latent_followup_allowed": stopping.get("latent_feature_followup_allowed"),
        "stop_after_a": stopping.get("stop_after_a"),
    }


def interpretation_for_experiment(
    validation_pr_auc: float,
    test_pr_auc: float,
    latent_followup_allowed: object,
) -> str:
    if validation_pr_auc <= REFERENCE_FE_DEFAULT_VALIDATION_PR_AUC:
        return (
            "CDV reconstruction error is flat or worse on validation; AE still "
            "looks more useful as score-level ensemble signal."
        )
    if test_pr_auc <= REFERENCE_FE_DEFAULT_TEST_PR_AUC:
        return (
            "Validation improved but test did not beat FE default; treat as "
            "temporal-risk signal, not a latent-feature trigger."
        )
    if latent_followup_allowed and test_pr_auc >= REFERENCE_FE_TUNED_TEST_PR_AUC:
        return (
            "Raw CDV reconstruction error is strong enough to justify a "
            "separate latent-feature follow-up."
        )
    return (
        "CDV reconstruction error is a weak additive behavioral feature, not "
        "a final-model replacement."
    )


def experiment_row(output_dir: Path) -> dict[str, object] | None:
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
    stopping = stopping_values(run_config)

    validation_pr_auc = float(valid_metrics["average_precision"])
    test_pr_auc = float(test_metrics["average_precision"])
    row = {
        "model_name": "A_fe_lgbm_cdv_reconstruction_mse_default",
        "stage": "A",
        "validation_pr_auc": validation_pr_auc,
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": metric_value(test_metrics, "roc_auc"),
        "test_precision": metric_value(test_metrics, "precision"),
        "test_recall": metric_value(test_metrics, "recall"),
        "test_f1": metric_value(test_metrics, "f1"),
        "test_mcc": metric_value(test_metrics, "mcc"),
        "selected_threshold": metric_value(test_metrics, "threshold"),
        "best_iteration": best_iteration_from_run_config(run_config),
        "total_features": total_features_from_run_config(run_config),
        "interpretation": interpretation_for_experiment(
            validation_pr_auc,
            test_pr_auc,
            stopping["latent_followup_allowed"],
        ),
        "output_dir": str(output_dir),
    }
    row.update(stopping)
    return row


def add_reference_deltas(row: dict[str, object]) -> dict[str, object]:
    validation_pr_auc = row.get("validation_pr_auc")
    test_pr_auc = row.get("test_pr_auc")
    row["delta_validation_pr_auc_vs_fe_default"] = (
        float(validation_pr_auc) - REFERENCE_FE_DEFAULT_VALIDATION_PR_AUC
        if validation_pr_auc is not None
        else None
    )
    row["delta_test_pr_auc_vs_fe_default"] = (
        float(test_pr_auc) - REFERENCE_FE_DEFAULT_TEST_PR_AUC
        if test_pr_auc is not None
        else None
    )
    row["delta_validation_pr_auc_vs_fe_tuned"] = (
        float(validation_pr_auc) - REFERENCE_FE_TUNED_VALIDATION_PR_AUC
        if validation_pr_auc is not None
        else None
    )
    row["delta_test_pr_auc_vs_fe_tuned"] = (
        float(test_pr_auc) - REFERENCE_FE_TUNED_TEST_PR_AUC
        if test_pr_auc is not None
        else None
    )
    row["delta_validation_pr_auc_vs_current_fe_ae"] = (
        float(validation_pr_auc) - REFERENCE_CURRENT_FE_AE_VALIDATION_PR_AUC
        if validation_pr_auc is not None
        else None
    )
    row["delta_test_pr_auc_vs_current_fe_ae"] = (
        float(test_pr_auc) - REFERENCE_CURRENT_FE_AE_TEST_PR_AUC
        if test_pr_auc is not None
        else None
    )
    return row


def build_comparison_table(experiment_dir: Path) -> pd.DataFrame:
    rows = [add_reference_deltas(dict(row)) for row in REFERENCE_ROWS]
    row = experiment_row(experiment_dir / EXPERIMENT_A_DIR_NAME)
    if row is not None:
        rows.append(add_reference_deltas(row))
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def comparison_output_file(experiment_dir: Path, output_file: Path | None) -> Path:
    if output_file is not None:
        return output_file
    if experiment_dir == BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR:
        return BEHAVIORAL_CDV_AE_COMPARISON_FILE
    return experiment_dir / "comparison.csv"


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(
        description="Compare behavioral CDV AE Experiment A against FE references."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR,
        help="Root directory for behavioral CDV AE experiment outputs.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional comparison CSV path. Defaults to <experiment-dir>/comparison.csv.",
    )
    args = parser.parse_args()

    table = build_comparison_table(args.experiment_dir)
    output_file = comparison_output_file(args.experiment_dir, args.output_file)
    ensure_dir(output_file.parent)
    table.to_csv(output_file, index=False)

    print()
    print("Behavioral CDV AE Experiment Comparison")
    print("=======================================")
    print(table.to_string(index=False))
    print(f"\nSaved comparison to: {output_file}")
    return table


if __name__ == "__main__":
    main()
