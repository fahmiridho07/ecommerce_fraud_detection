"""Build controlled Autoencoder output-strategy comparison CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    AUTOENCODER_OUTPUT_STRATEGY_COMPARISON_FILE,
    BASELINE_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR,
)
from utils import ensure_dir


def load_metric(path: Path) -> float:
    with path.open("r", encoding="utf-8") as file:
        return float(json.load(file)["average_precision"])


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> pd.DataFrame:
    p01_val = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )
    p01_test = load_metric(
        BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )
    p01_run = load_json(BASELINE_OUTPUT_DIR / "run_config.json")

    latent_val = load_metric(
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
        / "metrics_validation_selected_threshold.json"
    )
    latent_test = load_metric(
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
        / "metrics_test_selected_threshold.json"
    )
    latent_run = load_json(SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json")

    recon_val = load_metric(
        SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR
        / "metrics_validation_selected_threshold.json"
    )
    recon_test = load_metric(
        SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR
        / "metrics_test_selected_threshold.json"
    )
    recon_run = load_json(SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR / "run_config.json")

    rows = [
        {
            "model_name": "P01_baseline_lgbm_default",
            "autoencoder_input_scope": "none",
            "autoencoder_output_used": "none",
            "selected_numerical_feature_count": 0,
            "original_selected_features_retained": True,
            "latent_feature_count": 0,
            "reconstructed_feature_count": 0,
            "retained_raw_feature_count": 432,
            "final_feature_count": 432,
            "validation_average_precision": p01_val,
            "test_average_precision": p01_test,
            "validation_delta_vs_p01": 0.0,
            "test_delta_vs_p01": 0.0,
            "validation_delta_vs_latent_replacement": p01_val - latent_val,
            "test_delta_vs_latent_replacement": p01_test - latent_test,
            "best_iteration": p01_run["early_stopping"]["best_iteration"],
            "selected_threshold": p01_run["threshold_selection"]["selected_threshold"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(BASELINE_OUTPUT_DIR / "run_config.json"),
            "comparison_status": "primary_control",
            "caveat": "Original-feature chronological baseline.",
        },
        {
            "model_name": "AAE01_selected_numerical_latent_replacement_ld128",
            "autoencoder_input_scope": "selected_numerical_predictors",
            "autoencoder_output_used": "encoder_latent",
            "selected_numerical_feature_count": latent_run["removed_numerical_feature_count"],
            "original_selected_features_retained": False,
            "latent_feature_count": latent_run["latent_feature_count"],
            "reconstructed_feature_count": 0,
            "retained_raw_feature_count": latent_run["retained_raw_feature_count"],
            "final_feature_count": latent_run["final_feature_count"],
            "validation_average_precision": latent_val,
            "test_average_precision": latent_test,
            "validation_delta_vs_p01": latent_val - p01_val,
            "test_delta_vs_p01": latent_test - p01_test,
            "validation_delta_vs_latent_replacement": 0.0,
            "test_delta_vs_latent_replacement": 0.0,
            "best_iteration": latent_run["best_iteration"],
            "selected_threshold": latent_run["selected_threshold"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(
                SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "run_config.json"
            ),
            "comparison_status": "primary_control",
            "caveat": "Selected-numerical latent replacement LD128.",
        },
        {
            "model_name": "AAE02_selected_numerical_reconstructed_replacement",
            "autoencoder_input_scope": "selected_numerical_predictors",
            "autoencoder_output_used": "decoder_reconstruction_scaled",
            "selected_numerical_feature_count": recon_run["selected_numerical_feature_count"],
            "original_selected_features_retained": False,
            "latent_feature_count": 0,
            "reconstructed_feature_count": recon_run["reconstructed_feature_count"],
            "retained_raw_feature_count": recon_run["retained_raw_feature_count"],
            "final_feature_count": recon_run["final_feature_count"],
            "validation_average_precision": recon_val,
            "test_average_precision": recon_test,
            "validation_delta_vs_p01": recon_val - p01_val,
            "test_delta_vs_p01": recon_test - p01_test,
            "validation_delta_vs_latent_replacement": recon_val - latent_val,
            "test_delta_vs_latent_replacement": recon_test - latent_test,
            "best_iteration": recon_run["best_iteration"],
            "selected_threshold": recon_run["selected_threshold"],
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(
                SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR / "run_config.json"
            ),
            "comparison_status": "ding_alignment_diagnostic",
            "caveat": (
                "Decoder-reconstructed scaled features; validation AP determines "
                "interpretation; test AP descriptive only."
            ),
        },
    ]

    comparison_df = pd.DataFrame(rows)
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    comparison_df.to_csv(AUTOENCODER_OUTPUT_STRATEGY_COMPARISON_FILE, index=False)
    print(f"Saved comparison to {AUTOENCODER_OUTPUT_STRATEGY_COMPARISON_FILE}")
    return comparison_df


if __name__ == "__main__":
    main()