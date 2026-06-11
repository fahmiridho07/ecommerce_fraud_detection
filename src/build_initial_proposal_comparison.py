"""Build the initial thesis proposal comparison table (BASE-01..AE-02 only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    AE_LGBM_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    OPTUNA_OUTPUT_DIR,
)
from utils import ensure_dir, save_json

INITIAL_PROPOSAL_COMPARISON_FILENAME = "initial_proposal_comparison.csv"
INITIAL_PROPOSAL_MISSING_ARTIFACTS_FILENAME = "initial_proposal_missing_artifacts.json"

INITIAL_PROPOSAL_MODEL_NAMES = (
    "baseline_lgbm_default",
    "baseline_lgbm_tuned",
    "ae_lgbm_default",
    "ae_lgbm_ld128_tuned",
)

COMPARISON_COLUMNS = [
    "canonical_id",
    "legacy_id",
    "model_name",
    "tuned",
    "feature_setup",
    "validation_average_precision",
    "test_average_precision",
    "validation_roc_auc",
    "test_roc_auc",
    "selected_threshold",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "best_iteration",
    "n_trials",
    "total_features",
    "output_dir",
    "run_config_path",
    "metrics_path",
]

INITIAL_PROPOSAL_CANDIDATES = [
    {
        "canonical_id": "BASE-01",
        "legacy_id": "P01",
        "model_name": "baseline_lgbm_default",
        "tuned": False,
        "output_dir_key": "baseline_default",
    },
    {
        "canonical_id": "BASE-02",
        "legacy_id": "P02",
        "model_name": "baseline_lgbm_tuned",
        "tuned": True,
        "output_dir_key": "baseline_tuned",
    },
    {
        "canonical_id": "AE-01",
        "legacy_id": "P03",
        "model_name": "ae_lgbm_default",
        "tuned": False,
        "output_dir_key": "ae_default",
    },
    {
        "canonical_id": "AE-02",
        "legacy_id": "P04",
        "model_name": "ae_lgbm_ld128_tuned",
        "tuned": True,
        "output_dir_key": "ae_ld128_tuned",
    },
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


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


def feature_setup_from_run_config(run_config: dict[str, object]) -> object:
    for key in ("feature_setup", "experiment_type", "phase"):
        if key in run_config:
            return run_config[key]
    feature_info = run_config.get("feature_info", {})
    if isinstance(feature_info, dict) and "feature_setup" in feature_info:
        return feature_info["feature_setup"]
    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def n_trials_from_run_config(run_config: dict[str, object], tuned: bool) -> object:
    if not tuned:
        return None
    optuna_config = run_config.get("optuna", {})
    if isinstance(optuna_config, dict):
        for key in ("n_trials_completed", "n_trials", "completed_trials"):
            if key in optuna_config:
                return optuna_config[key]
    for key in ("n_trials", "completed_trials", "study_n_trials"):
        if key in run_config:
            return run_config[key]
    return None


def required_artifact_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / "metrics_validation_selected_threshold.json",
        output_dir / "metrics_test_selected_threshold.json",
        output_dir / "run_config.json",
    ]


def comparison_row(
    candidate: dict[str, object],
    output_dir: Path,
) -> tuple[dict[str, object] | None, list[str]]:
    required_paths = required_artifact_paths(output_dir)
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        return None, missing_paths

    run_config_path = output_dir / "run_config.json"
    valid_metrics_path = output_dir / "metrics_validation_selected_threshold.json"
    test_metrics_path = output_dir / "metrics_test_selected_threshold.json"

    run_config = load_json(run_config_path)
    if candidate["tuned"] and run_config.get("final_training_completed") is False:
        return None, [str(run_config_path) + " (final_training_completed=false)"]

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)
    tuned = bool(candidate["tuned"])

    return (
        {
            "canonical_id": candidate["canonical_id"],
            "legacy_id": candidate["legacy_id"],
            "model_name": candidate["model_name"],
            "tuned": tuned,
            "feature_setup": feature_setup_from_run_config(run_config),
            "validation_average_precision": metric_value(
                valid_metrics,
                "average_precision",
            ),
            "test_average_precision": metric_value(test_metrics, "average_precision"),
            "validation_roc_auc": metric_value(valid_metrics, "roc_auc"),
            "test_roc_auc": metric_value(test_metrics, "roc_auc"),
            "selected_threshold": metric_value(test_metrics, "threshold"),
            "test_precision": metric_value(test_metrics, "precision"),
            "test_recall": metric_value(test_metrics, "recall"),
            "test_f1": metric_value(test_metrics, "f1"),
            "test_mcc": metric_value(test_metrics, "mcc"),
            "best_iteration": best_iteration_from_run_config(run_config),
            "n_trials": n_trials_from_run_config(run_config, tuned=tuned),
            "total_features": total_features_from_run_config(run_config),
            "output_dir": str(output_dir),
            "run_config_path": str(run_config_path),
            "metrics_path": str(test_metrics_path),
        },
        [],
    )


def resolve_output_dirs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "baseline_default": args.baseline_default_dir,
        "baseline_tuned": args.baseline_tuned_dir,
        "ae_default": args.ae_lgbm_default_dir,
        "ae_ld128_tuned": args.ae_lgbm_ld128_tuned_dir,
    }


def comparison_output_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        output_dir / INITIAL_PROPOSAL_COMPARISON_FILENAME,
        output_dir / INITIAL_PROPOSAL_MISSING_ARTIFACTS_FILENAME,
    )


def build_initial_proposal_comparison_table(
    output_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    missing_artifacts: dict[str, list[str]] = {}

    for candidate in INITIAL_PROPOSAL_CANDIDATES:
        output_dir = output_dirs[candidate["output_dir_key"]]
        row, missing_paths = comparison_row(candidate, output_dir)
        if row is not None:
            rows.append(row)
        elif missing_paths:
            missing_artifacts[candidate["model_name"]] = missing_paths

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return table, missing_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the initial thesis proposal comparison table for "
            "BASE-01, BASE-02, AE-01, and AE-02 only."
        )
    )
    parser.add_argument(
        "--baseline-default-dir",
        type=Path,
        default=BASELINE_OUTPUT_DIR,
    )
    parser.add_argument(
        "--baseline-tuned-dir",
        type=Path,
        default=OPTUNA_OUTPUT_DIR / "baseline_lgbm",
    )
    parser.add_argument(
        "--ae-lgbm-default-dir",
        type=Path,
        default=AE_LGBM_OUTPUT_DIR,
    )
    parser.add_argument(
        "--ae-lgbm-ld128-tuned-dir",
        type=Path,
        default=OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINAL_COMPARISON_OUTPUT_DIR,
        help=(
            "Directory for initial_proposal_comparison.csv and "
            "initial_proposal_missing_artifacts.json."
        ),
    )
    return parser.parse_args()


def main() -> pd.DataFrame:
    args = parse_args()
    comparison_output_dir = ensure_dir(args.output_dir)
    comparison_file, missing_artifacts_file = comparison_output_paths(
        comparison_output_dir
    )
    output_dirs = resolve_output_dirs(args)
    table, missing_artifacts = build_initial_proposal_comparison_table(output_dirs)
    table.to_csv(comparison_file, index=False)
    save_json(missing_artifacts, missing_artifacts_file)

    print()
    print("Initial Proposal Comparison")
    print("===========================")
    if table.empty:
        print("No completed initial proposal rows are available yet.")
    else:
        print(table.to_string(index=False))
    if missing_artifacts:
        print()
        print("Missing artifacts recorded for:")
        for model_name, paths in missing_artifacts.items():
            print(f"- {model_name}: {len(paths)} path(s)")
    print(f"\nSaved comparison to: {comparison_file}")
    print(f"Saved missing-artifact log to: {missing_artifacts_file}")
    return table


if __name__ == "__main__":
    main()