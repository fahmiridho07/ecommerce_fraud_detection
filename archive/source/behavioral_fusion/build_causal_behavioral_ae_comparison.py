"""Build controlled B1/B2/B3 causal behavioral AE comparison CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    BASELINE_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_AE_COMPARISON_FILE,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_FEATURE_IMPORTANCE_FILE,
    CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
)
from utils import ensure_dir, save_json


def load_metric(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def model_row(
    model_id: str,
    model_name: str,
    output_dir: Path,
    feature_strategy: str,
    causal_behavioral: bool,
    behavioral_count: int,
    cdv_recon: bool,
    latent: bool,
    tuned: bool,
    feature_protocol: str,
    comparison_status: str,
    caveat: str,
    b1_val: float,
    b2_val: float | None = None,
) -> dict[str, object]:
    val_metrics = load_metric(
        output_dir / "metrics_validation_selected_threshold.json"
    )
    test_metrics = load_metric(
        output_dir / "metrics_test_selected_threshold.json"
    )
    run_config = load_json(output_dir / "run_config.json")
    val_ap = float(val_metrics["average_precision"])
    test_ap = float(test_metrics["average_precision"])
    row = {
        "model_id": model_id,
        "model_name": model_name,
        "feature_strategy": feature_strategy,
        "causal_behavioral_features_used": causal_behavioral,
        "behavioral_feature_count": behavioral_count,
        "cdv_reconstruction_error_used": cdv_recon,
        "latent_features_used": latent,
        "tuned": tuned,
        "validation_average_precision": val_ap,
        "test_average_precision": test_ap,
        "validation_delta_vs_b1": val_ap - b1_val,
        "test_delta_vs_b1": test_ap - float(
            load_metric(BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json")[
                "average_precision"
            ]
        ),
        "validation_delta_vs_b2": None,
        "test_delta_vs_b2": None,
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_mcc": float(test_metrics["mcc"]),
        "selected_threshold": run_config["threshold_selection"]["selected_threshold"],
        "best_iteration": run_config["early_stopping"]["best_iteration"],
        "total_features": run_config.get(
            "model_features_count",
            run_config.get("final_feature_count"),
        ),
        "feature_protocol": feature_protocol,
        "metric_source": "metrics_validation_selected_threshold.json",
        "run_config_source": str(output_dir / "run_config.json"),
        "comparison_status": comparison_status,
        "caveat": caveat,
    }
    if b2_val is not None:
        row["validation_delta_vs_b2"] = val_ap - b2_val
        b2_test = float(
            load_metric(
                CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR
                / "metrics_test_selected_threshold.json"
            )["average_precision"]
        )
        row["test_delta_vs_b2"] = test_ap - b2_test
    return row


def build_feature_importance_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id, output_dir in (
        ("B2", CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR),
        ("B3", CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR),
    ):
        importance_path = output_dir / "feature_importance.csv"
        importance = pd.read_csv(importance_path)
        importance = importance.sort_values(
            "importance_gain",
            ascending=False,
        ).reset_index(drop=True)
        importance["rank_gain"] = importance.index + 1
        for _, record in importance.iterrows():
            feature = str(record["feature"])
            if feature.startswith("cb_"):
                group = "causal_behavioral"
            elif feature == "cdv_ae_reconstruction_mse":
                group = "cdv_reconstruction_error"
            else:
                group = "original"
            rows.append(
                {
                    "feature": feature,
                    "feature_group": group,
                    "importance_gain": float(record["importance_gain"]),
                    "importance_split": int(record["importance_split"]),
                    "model_id": model_id,
                    "rank_gain": int(record["rank_gain"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    b1_val = float(
        load_metric(
            BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )
    b2_val = float(
        load_metric(
            CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )

    rows = [
        model_row(
            model_id="B1",
            model_name="P01_original_feature_lgbm_default",
            output_dir=BASELINE_OUTPUT_DIR,
            feature_strategy="original_features_only",
            causal_behavioral=False,
            behavioral_count=0,
            cdv_recon=False,
            latent=False,
            tuned=False,
            feature_protocol="P01 chronological baseline",
            comparison_status="primary_control",
            caveat="No behavioral or AE-derived features.",
            b1_val=b1_val,
        ),
        model_row(
            model_id="B2",
            model_name="causal_behavioral_lgbm_default",
            output_dir=CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
            feature_strategy="original_plus_causal_behavioral",
            causal_behavioral=True,
            behavioral_count=19,
            cdv_recon=False,
            latent=False,
            tuned=False,
            feature_protocol="online causal state continuation",
            comparison_status="primary_comparison_arm",
            caveat="Controlled causal behavioral experiment arm.",
            b1_val=b1_val,
        ),
        model_row(
            model_id="B3",
            model_name="causal_behavioral_cdv_reconstruction_lgbm_default",
            output_dir=CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
            feature_strategy="B2_plus_cdv_reconstruction_error",
            causal_behavioral=True,
            behavioral_count=19,
            cdv_recon=True,
            latent=False,
            tuned=False,
            feature_protocol="B2 + exactly one CDV recon error",
            comparison_status="primary_comparison_arm",
            caveat="Isolates CDV reconstruction error beyond causal behavioral context.",
            b1_val=b1_val,
            b2_val=b2_val,
        ),
    ]

    ex01_val = float(
        load_metric(
            FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )
    ex01_test = float(
        load_metric(
            FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
            / "metrics_test_selected_threshold.json"
        )["average_precision"]
    )
    ex01_run = load_json(FEATURE_ENGINEERED_LGBM_OUTPUT_DIR / "run_config.json")
    rows.append(
        {
            "model_id": "EX01",
            "model_name": "baseline_lgbm_entity_time_amount_features",
            "feature_strategy": "train_static_entity_time_amount_fe",
            "causal_behavioral_features_used": False,
            "behavioral_feature_count": 0,
            "cdv_reconstruction_error_used": False,
            "latent_features_used": False,
            "tuned": False,
            "validation_average_precision": ex01_val,
            "test_average_precision": ex01_test,
            "validation_delta_vs_b1": ex01_val - b1_val,
            "test_delta_vs_b1": ex01_test - float(
                load_metric(
                    BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
                )["average_precision"]
            ),
            "validation_delta_vs_b2": ex01_val - b2_val,
            "test_delta_vs_b2": None,
            "test_roc_auc": float(
                load_metric(
                    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["roc_auc"]
            ),
            "test_precision": float(
                load_metric(
                    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["precision"]
            ),
            "test_recall": float(
                load_metric(
                    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["recall"]
            ),
            "test_f1": float(
                load_metric(
                    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["f1"]
            ),
            "test_mcc": float(
                load_metric(
                    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["mcc"]
            ),
            "selected_threshold": ex01_run["threshold_selection"]["selected_threshold"],
            "best_iteration": ex01_run["early_stopping"]["best_iteration"],
            "total_features": ex01_run["model_features_count"],
            "feature_protocol": "train-fitted static FE mappings",
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(
                FEATURE_ENGINEERED_LGBM_OUTPUT_DIR / "run_config.json"
            ),
            "comparison_status": "historical_reference_only",
            "caveat": "Not directly comparable — uses train-static aggregates, not causal behavioral features.",
        }
    )

    ae15_val = float(
        load_metric(
            FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )
    ae15_test = float(
        load_metric(
            FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
            / "metrics_test_selected_threshold.json"
        )["average_precision"]
    )
    ae15_run = load_json(FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR / "run_config.json")
    rows.append(
        {
            "model_id": "AE15",
            "model_name": "fe_lgbm_cdv_reconstruction_mse_default",
            "feature_strategy": "train_static_fe_plus_cdv_recon",
            "causal_behavioral_features_used": False,
            "behavioral_feature_count": 0,
            "cdv_reconstruction_error_used": True,
            "latent_features_used": False,
            "tuned": False,
            "validation_average_precision": ae15_val,
            "test_average_precision": ae15_test,
            "validation_delta_vs_b1": ae15_val - b1_val,
            "test_delta_vs_b1": ae15_test - float(
                load_metric(
                    BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
                )["average_precision"]
            ),
            "validation_delta_vs_b2": ae15_val - b2_val,
            "test_delta_vs_b2": None,
            "test_roc_auc": float(
                load_metric(
                    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["roc_auc"]
            ),
            "test_precision": float(
                load_metric(
                    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["precision"]
            ),
            "test_recall": float(
                load_metric(
                    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["recall"]
            ),
            "test_f1": float(
                load_metric(
                    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["f1"]
            ),
            "test_mcc": float(
                load_metric(
                    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR
                    / "metrics_test_selected_threshold.json"
                )["mcc"]
            ),
            "selected_threshold": ae15_run["threshold_selection"]["selected_threshold"],
            "best_iteration": ae15_run["early_stopping"]["best_iteration"],
            "total_features": ae15_run["model_features_count"],
            "feature_protocol": "EX01 FE + CDV recon error",
            "metric_source": "metrics_validation_selected_threshold.json",
            "run_config_source": str(
                FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR / "run_config.json"
            ),
            "comparison_status": "historical_reference_only",
            "caveat": "Not directly comparable — FE-space CDV recon, not B2 + one recon feature.",
        }
    )

    comparison = pd.DataFrame(rows)
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    comparison.to_csv(CAUSAL_BEHAVIORAL_AE_COMPARISON_FILE, index=False)

    importance_table = build_feature_importance_table()
    importance_table.to_csv(CAUSAL_BEHAVIORAL_FEATURE_IMPORTANCE_FILE, index=False)

    summary = {
        "b1_validation_ap": b1_val,
        "b2_validation_ap": b2_val,
        "b3_validation_ap": float(rows[2]["validation_average_precision"]),
        "b1_vs_b2_validation_delta": b2_val - b1_val,
        "b2_vs_b3_validation_delta": float(rows[2]["validation_average_precision"])
        - b2_val,
        "comparison_csv": str(CAUSAL_BEHAVIORAL_AE_COMPARISON_FILE),
        "importance_csv": str(CAUSAL_BEHAVIORAL_FEATURE_IMPORTANCE_FILE),
    }
    save_json(
        summary,
        FINAL_COMPARISON_OUTPUT_DIR / "causal_behavioral_ae_comparison_summary.json",
    )
    print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    main()