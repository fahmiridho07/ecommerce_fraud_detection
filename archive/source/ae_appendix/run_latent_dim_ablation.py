"""Run Phase 4B latent-dimension ablation for robust AE-LightGBM.

This ablation tests whether the 32-dimensional Autoencoder bottleneck was too
aggressive by repeating the robust AE + LightGBM pipeline with latent_dim 64
and 128. It does not tune hyperparameters or change the temporal split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import (
    AE_LGBM_LD128_OUTPUT_DIR,
    AE_LGBM_LD64_OUTPUT_DIR,
    AE_LGBM_OUTPUT_DIR,
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    AUTOENCODER_ROBUST_LD64_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    LATENT_DIM_ABLATION_DIMS,
    LATENT_DIM_ABLATION_FILE,
)
from train_ae_lgbm import main as train_ae_lgbm
from train_autoencoder_robust import main as train_autoencoder_robust
from utils import ensure_dir, log


VARIANT_DIRS = {
    64: {
        "autoencoder": AUTOENCODER_ROBUST_LD64_OUTPUT_DIR,
        "ae_lgbm": AE_LGBM_LD64_OUTPUT_DIR,
    },
    128: {
        "autoencoder": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
        "ae_lgbm": AE_LGBM_LD128_OUTPUT_DIR,
    },
}


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def output_complete(output_dir: Path, required_files: list[str]) -> bool:
    return all((output_dir / file_name).exists() for file_name in required_files)


def train_variant(latent_dim: int, skip_existing: bool = False) -> None:
    if latent_dim not in VARIANT_DIRS:
        raise ValueError(f"Unsupported latent_dim for this ablation: {latent_dim}")

    ae_output_dir = VARIANT_DIRS[latent_dim]["autoencoder"]
    lgbm_output_dir = VARIANT_DIRS[latent_dim]["ae_lgbm"]

    ae_required = [
        "autoencoder_model.keras",
        "encoder_model.keras",
        "v_scaler.pkl",
        "latent_train.npy",
        "latent_valid.npy",
        "latent_test.npy",
        "reconstruction_metrics.json",
        "run_config.json",
    ]
    lgbm_required = [
        "metrics_test_selected_threshold.json",
        "threshold_selection.csv",
        "feature_importance.csv",
        "model.pkl",
        "feature_set_summary.json",
        "comparison_against_baseline.json",
        "run_config.json",
    ]

    if skip_existing and output_complete(ae_output_dir, ae_required):
        log(f"Skipping robust Autoencoder latent_dim={latent_dim}; outputs already exist.")
    else:
        log(f"Training robust Autoencoder latent_dim={latent_dim}.")
        train_autoencoder_robust(
            latent_dim=latent_dim,
            output_dir=ae_output_dir,
            phase_name=f"4B_robust_autoencoder_ld{latent_dim}",
            print_old_comparison=False,
        )

    if skip_existing and output_complete(lgbm_output_dir, lgbm_required):
        log(f"Skipping AE-LightGBM latent_dim={latent_dim}; outputs already exist.")
    else:
        log(f"Training AE-LightGBM latent_dim={latent_dim}.")
        train_ae_lgbm(
            autoencoder_output_dir=ae_output_dir,
            output_dir=lgbm_output_dir,
            phase_name=f"4B_ae_lgbm_ld{latent_dim}",
        )


def build_result_row(
    model_name: str,
    latent_dim: int | None,
    metrics_path: Path,
    run_config_path: Path,
    feature_summary_path: Path | None = None,
) -> dict[str, object]:
    metrics = load_json(metrics_path)
    run_config = load_json(run_config_path)
    feature_summary = load_json(feature_summary_path) if feature_summary_path else {}

    total_features = feature_summary.get("total_final_features")
    if total_features is None:
        total_features = run_config.get("model_features_count")

    return {
        "model_name": model_name,
        "latent_dim": "" if latent_dim is None else str(latent_dim),
        "test_pr_auc": metrics["average_precision"],
        "test_roc_auc": metrics["roc_auc"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_mcc": metrics["mcc"],
        "selected_threshold": metrics["threshold"],
        "best_iteration": run_config["early_stopping"]["best_iteration"],
        "total_features": total_features,
    }


def build_ablation_table() -> pd.DataFrame:
    rows = [
        build_result_row(
            model_name="baseline_lgbm",
            latent_dim=None,
            metrics_path=BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json",
            run_config_path=BASELINE_OUTPUT_DIR / "run_config.json",
        ),
        build_result_row(
            model_name="ae_lgbm_ld32",
            latent_dim=32,
            metrics_path=AE_LGBM_OUTPUT_DIR / "metrics_test_selected_threshold.json",
            run_config_path=AE_LGBM_OUTPUT_DIR / "run_config.json",
            feature_summary_path=AE_LGBM_OUTPUT_DIR / "feature_set_summary.json",
        ),
    ]

    for latent_dim in LATENT_DIM_ABLATION_DIMS:
        lgbm_output_dir = VARIANT_DIRS[latent_dim]["ae_lgbm"]
        rows.append(
            build_result_row(
                model_name=f"ae_lgbm_ld{latent_dim}",
                latent_dim=latent_dim,
                metrics_path=lgbm_output_dir / "metrics_test_selected_threshold.json",
                run_config_path=lgbm_output_dir / "run_config.json",
                feature_summary_path=lgbm_output_dir / "feature_set_summary.json",
            )
        )

    return pd.DataFrame(rows)


def save_and_print_ablation_table(table: pd.DataFrame) -> None:
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table.to_csv(LATENT_DIM_ABLATION_FILE, index=False)

    ranked = table.sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
    print()
    print("Latent Dimension Ablation Ranking by Test PR-AUC")
    print("================================================")
    for rank, row in ranked.iterrows():
        latent_dim = row["latent_dim"] if row["latent_dim"] else "baseline"
        print(
            f"{rank + 1}. {row['model_name']} "
            f"(latent_dim={latent_dim}) "
            f"PR-AUC={row['test_pr_auc']:.6f}, "
            f"ROC-AUC={row['test_roc_auc']:.6f}, "
            f"F1={row['test_f1']:.6f}, "
            f"MCC={row['test_mcc']:.6f}"
        )
    print(f"\nSaved comparison to: {LATENT_DIM_ABLATION_FILE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4B latent-dim ablation.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed ld64/ld128 outputs when all required files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for latent_dim in LATENT_DIM_ABLATION_DIMS:
        train_variant(latent_dim, skip_existing=args.skip_existing)

    table = build_ablation_table()
    save_and_print_ablation_table(table)


if __name__ == "__main__":
    main()
