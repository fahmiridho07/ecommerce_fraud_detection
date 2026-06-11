"""Build tuned AE strategy comparison table (TUNE-B0, TUNE-AE3 only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from utils import ensure_dir, save_json


TUNING_BASE_OUTPUT_DIR = (
    OUTPUT_DIR / "ae_integration_strategy_ablation_ld128" / "optuna"
)
COMPARISON_FILENAME = "comparison.csv"
MISSING_ARTIFACTS_FILENAME = "missing_artifacts.json"
EXPECTED_EXPERIMENT_FAMILY = "ae_strategy_tuning"

COMPARISON_COLUMNS = [
    "strategy_id",
    "model_name",
    "tuned",
    "original_v_features_retained",
    "reconstruction_error_used",
    "latent_features_used",
    "reconstructed_features_used",
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

TUNED_CANDIDATES = [
    {
        "strategy_id": "TUNE-B0",
        "variant": "baseline_lgbm_tuned",
        "model_name": "baseline_lgbm_tuned",
        "tuned": True,
        "original_v_features_retained": True,
        "reconstruction_error_used": False,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "output_subdir": "baseline_lgbm_tuned",
    },
    {
        "strategy_id": "TUNE-AE3",
        "variant": "ae3_reconstruction_error_lgbm_ld128_tuned",
        "model_name": "ae3_reconstruction_error_lgbm_ld128_tuned",
        "tuned": True,
        "original_v_features_retained": True,
        "reconstruction_error_used": True,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "output_subdir": "AE3_reconstruction_error_tuned",
    },
]

EXPECTED_STRATEGY_IDS = tuple(candidate["strategy_id"] for candidate in TUNED_CANDIDATES)
EXPECTED_MODEL_NAMES = tuple(candidate["model_name"] for candidate in TUNED_CANDIDATES)

OUT_OF_SCOPE_MODEL_NAMES = {
    "baseline_lgbm_default",
    "ae_lgbm_ld128_tuned",
    "ae_augmented_lgbm_ld128_tuned",
    "du_latent_replacement",
    "ding_reconstructed_replacement",
    "reconstruction_error_augmentation",
    "causal_behavioral_lgbm_id_aligned",
    "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned",
    "selected_numerical_reconstructed_lgbm",
    "fe_lgbm_plus_cdv_ae_reconstruction_mse_default",
    "task_aware_ae_lgbm_ld128",
}


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

    feature_set_summary = run_config.get("feature_set_summary", {})
    if isinstance(feature_set_summary, dict):
        for key in ("total_final_features", "total_feature_count"):
            if key in feature_set_summary:
                return feature_set_summary[key]
    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def n_trials_from_run_config(run_config: dict[str, object]) -> object:
    optuna_config = run_config.get("optuna", {})
    if isinstance(optuna_config, dict):
        for key in ("n_trials_completed", "n_trials", "completed_trials"):
            if key in optuna_config:
                return optuna_config[key]
    return None


def validate_tuned_run_config(
    run_config: dict[str, object],
    candidate: dict[str, object],
    run_config_path: Path,
) -> str | None:
    if run_config.get("final_training_completed") is False:
        return (
            f"{run_config_path} (final_training_completed=false)"
        )

    if run_config.get("experiment_family") != EXPECTED_EXPERIMENT_FAMILY:
        return (
            "run_config variant/family mismatch: "
            f"expected experiment_family={EXPECTED_EXPERIMENT_FAMILY!r}, "
            f"found {run_config.get('experiment_family')!r} ({run_config_path})"
        )

    expected_variant = candidate["variant"]
    actual_variant = run_config.get("variant")
    if actual_variant != expected_variant:
        return (
            "run_config variant/family mismatch: "
            f"expected variant={expected_variant!r}, "
            f"found {actual_variant!r} ({run_config_path})"
        )
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
    mismatch_message = validate_tuned_run_config(
        run_config,
        candidate,
        run_config_path,
    )
    if mismatch_message:
        return None, [mismatch_message]

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)

    return (
        {
            "strategy_id": candidate["strategy_id"],
            "model_name": candidate["model_name"],
            "tuned": candidate["tuned"],
            "original_v_features_retained": candidate["original_v_features_retained"],
            "reconstruction_error_used": candidate["reconstruction_error_used"],
            "latent_features_used": candidate["latent_features_used"],
            "reconstructed_features_used": candidate["reconstructed_features_used"],
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
            "n_trials": n_trials_from_run_config(run_config),
            "total_features": total_features_from_run_config(run_config),
            "output_dir": str(output_dir),
            "run_config_path": str(run_config_path),
            "metrics_path": str(test_metrics_path),
        },
        [],
    )


def resolve_output_dirs(
    base_output_dir: Path,
    overrides: dict[str, Path | None],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for candidate in TUNED_CANDIDATES:
        variant = candidate["variant"]
        override = overrides.get(variant)
        resolved[variant] = override or (base_output_dir / candidate["output_subdir"])
    return resolved


def build_ae_strategy_tuned_comparison_table(
    output_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    missing_artifacts: dict[str, list[str]] = {}

    for candidate in TUNED_CANDIDATES:
        output_dir = output_dirs[candidate["variant"]]
        row, missing_paths = comparison_row(candidate, output_dir)
        if row is not None:
            rows.append(row)
        elif missing_paths:
            missing_artifacts[candidate["model_name"]] = missing_paths

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return table, missing_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tuned AE strategy comparison for TUNE-B0 and TUNE-AE3."
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=TUNING_BASE_OUTPUT_DIR,
    )
    parser.add_argument("--baseline-lgbm-tuned-dir", type=Path, default=None)
    parser.add_argument(
        "--ae3-reconstruction-error-tuned-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> pd.DataFrame:
    args = parse_args()
    base_output_dir = ensure_dir(args.base_output_dir)
    output_dirs = resolve_output_dirs(
        base_output_dir,
        {
            "baseline_lgbm_tuned": args.baseline_lgbm_tuned_dir,
            "ae3_reconstruction_error_lgbm_ld128_tuned": (
                args.ae3_reconstruction_error_tuned_dir
            ),
        },
    )
    table, missing_artifacts = build_ae_strategy_tuned_comparison_table(output_dirs)

    comparison_file = base_output_dir / COMPARISON_FILENAME
    missing_artifacts_file = base_output_dir / MISSING_ARTIFACTS_FILENAME
    table.to_csv(comparison_file, index=False)
    save_json(missing_artifacts, missing_artifacts_file)

    print()
    print("AE Strategy Tuned Comparison")
    print("============================")
    if table.empty:
        print("No completed tuned rows are available yet.")
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