"""Build comparison table for AE integration strategy ablation (STR-B0..STR-AE3)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from utils import ensure_dir, save_json


ABLATION_BASE_OUTPUT_DIR = OUTPUT_DIR / "ae_integration_strategy_ablation"
COMPARISON_FILENAME = "comparison.csv"
MISSING_ARTIFACTS_FILENAME = "missing_artifacts.json"

COMPARISON_COLUMNS = [
    "strategy_id",
    "model_name",
    "paper_anchor",
    "original_v_features_retained",
    "latent_features_used",
    "reconstructed_features_used",
    "reconstruction_error_used",
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
    "total_features",
    "output_dir",
    "run_config_path",
    "metrics_path",
]

STRATEGY_CANDIDATES = [
    {
        "strategy_id": "STR-B0",
        "variant": "baseline_fixed",
        "model_name": "baseline_fixed",
        "paper_anchor": "Raw LightGBM fixed/default baseline",
        "original_v_features_retained": True,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "reconstruction_error_used": False,
        "output_subdir": "B0_baseline_fixed",
    },
    {
        "strategy_id": "STR-AE1",
        "variant": "du_latent_replacement",
        "model_name": "du_latent_replacement",
        "paper_anchor": "Du et al. latent representation",
        "original_v_features_retained": False,
        "latent_features_used": True,
        "reconstructed_features_used": False,
        "reconstruction_error_used": False,
        "output_subdir": "AE1_du_latent_replacement",
    },
    {
        "strategy_id": "STR-AE2",
        "variant": "ding_reconstructed_replacement",
        "model_name": "ding_reconstructed_replacement",
        "paper_anchor": "Ding et al. reconstructed features",
        "original_v_features_retained": False,
        "latent_features_used": False,
        "reconstructed_features_used": True,
        "reconstruction_error_used": False,
        "output_subdir": "AE2_ding_reconstructed_replacement",
    },
    {
        "strategy_id": "STR-AE3",
        "variant": "reconstruction_error_augmentation",
        "model_name": "reconstruction_error_augmentation",
        "paper_anchor": "Autoencoder anomaly detection reconstruction error",
        "original_v_features_retained": True,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "reconstruction_error_used": True,
        "output_subdir": "AE3_reconstruction_error_augmentation",
    },
]

EXPECTED_STRATEGY_IDS = tuple(candidate["strategy_id"] for candidate in STRATEGY_CANDIDATES)
EXPECTED_MODEL_NAMES = tuple(candidate["model_name"] for candidate in STRATEGY_CANDIDATES)

OUT_OF_SCOPE_MODEL_NAMES = {
    "baseline_lgbm_default",
    "baseline_lgbm_tuned",
    "ae_lgbm_default",
    "ae_lgbm_ld128_tuned",
    "causal_behavioral_lgbm_id_aligned",
    "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned",
    "selected_numerical_reconstructed_lgbm",
    "fe_lgbm_plus_cdv_ae_reconstruction_mse_default",
    "task_aware_ae_lgbm_ld128",
    "ae_augmented_lgbm_ld128_tuned",
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
    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)

    return (
        {
            "strategy_id": candidate["strategy_id"],
            "model_name": candidate["model_name"],
            "paper_anchor": candidate["paper_anchor"],
            "original_v_features_retained": candidate["original_v_features_retained"],
            "latent_features_used": candidate["latent_features_used"],
            "reconstructed_features_used": candidate["reconstructed_features_used"],
            "reconstruction_error_used": candidate["reconstruction_error_used"],
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
    for candidate in STRATEGY_CANDIDATES:
        variant = candidate["variant"]
        override = overrides.get(variant)
        resolved[variant] = override or (base_output_dir / candidate["output_subdir"])
    return resolved


def build_ae_strategy_ablation_comparison_table(
    output_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, object]] = []
    missing_artifacts: dict[str, list[str]] = {}

    for candidate in STRATEGY_CANDIDATES:
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
        description=(
            "Build AE integration strategy ablation comparison for "
            "STR-B0, STR-AE1, STR-AE2, and STR-AE3 only."
        )
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=ABLATION_BASE_OUTPUT_DIR,
        help="Root directory containing ablation variant outputs.",
    )
    parser.add_argument(
        "--baseline-fixed-dir",
        type=Path,
        default=None,
        help="Override output directory for STR-B0 / baseline_fixed.",
    )
    parser.add_argument(
        "--du-latent-replacement-dir",
        type=Path,
        default=None,
        help="Override output directory for STR-AE1 / du_latent_replacement.",
    )
    parser.add_argument(
        "--ding-reconstructed-replacement-dir",
        type=Path,
        default=None,
        help="Override output directory for STR-AE2 / ding_reconstructed_replacement.",
    )
    parser.add_argument(
        "--reconstruction-error-augmentation-dir",
        type=Path,
        default=None,
        help="Override output directory for STR-AE3 / reconstruction_error_augmentation.",
    )
    return parser.parse_args()


def main() -> pd.DataFrame:
    args = parse_args()
    base_output_dir = ensure_dir(args.base_output_dir)
    output_dirs = resolve_output_dirs(
        base_output_dir,
        {
            "baseline_fixed": args.baseline_fixed_dir,
            "du_latent_replacement": args.du_latent_replacement_dir,
            "ding_reconstructed_replacement": args.ding_reconstructed_replacement_dir,
            "reconstruction_error_augmentation": args.reconstruction_error_augmentation_dir,
        },
    )
    table, missing_artifacts = build_ae_strategy_ablation_comparison_table(output_dirs)

    comparison_file = base_output_dir / COMPARISON_FILENAME
    missing_artifacts_file = base_output_dir / MISSING_ARTIFACTS_FILENAME
    table.to_csv(comparison_file, index=False)
    save_json(missing_artifacts, missing_artifacts_file)

    print()
    print("AE Integration Strategy Ablation Comparison")
    print("===========================================")
    if table.empty:
        print("No completed ablation rows are available yet.")
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