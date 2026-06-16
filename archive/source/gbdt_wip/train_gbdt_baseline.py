"""Train fixed/default raw-feature GBDT baselines (LightGBM, XGBoost, CatBoost)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from config import (
    DATA_DIR,
    ID_COL,
    OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from gbdt_backends import (
    DEFAULT_THRESHOLD,
    PreparedMatrices,
    SUPPORTED_BACKENDS,
    SUPPORTED_PREPROCESSING_MODES,
    best_iteration_for,
    build_default_params,
    fit_model,
    predict_positive_proba,
    prepare_matrices_from_raw_splits,
    save_feature_importance,
    save_model_artifacts,
)
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed


GBDT_BASE_OUTPUT_DIR = OUTPUT_DIR / "gbdt_baseline_comparison"
EXPERIMENT_FAMILY = "gbdt_baseline_comparison"

BACKEND_DEFAULT_SUBDIRS = {
    "lightgbm": "LGBM_fixed",
    "xgboost": "XGB_fixed",
    "catboost": "CAT_fixed",
}

BACKEND_PHASE_NAMES = {
    "lightgbm": "GBDT-LGBM-FIX",
    "xgboost": "GBDT-XGB-FIX",
    "catboost": "GBDT-CAT-FIX",
}


def default_output_dir(backend: str) -> Path:
    return GBDT_BASE_OUTPUT_DIR / BACKEND_DEFAULT_SUBDIRS[backend]


def train_and_save(
    prepared: PreparedMatrices,
    output_dir: Path,
    phase_name: str,
    n_jobs: int,
) -> dict[str, object]:
    model_params = build_default_params(
        prepared.backend,
        prepared.y_train,
        preprocessing_mode=prepared.preprocessing_mode,
        n_jobs=n_jobs,
    )
    log(f"Training fixed/default {prepared.backend} with validation early stopping.")
    model = fit_model(prepared, model_params, log_period=50)
    best_iteration = best_iteration_for(model, prepared.backend, model_params)

    valid_score = predict_positive_proba(
        model,
        prepared.X_valid,
        prepared.backend,
        best_iteration,
        prepared.cat_feature_indices,
    )
    test_score = predict_positive_proba(
        model,
        prepared.X_test,
        prepared.backend,
        best_iteration,
        prepared.cat_feature_indices,
    )

    threshold_table = threshold_selection_table(
        prepared.y_valid.to_numpy(),
        valid_score,
    )
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_valid_default = binary_classification_metrics(
        prepared.y_valid.to_numpy(),
        valid_score,
        DEFAULT_THRESHOLD,
    )
    metrics_valid_selected = binary_classification_metrics(
        prepared.y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        prepared.y_test.to_numpy(),
        test_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        prepared.y_test.to_numpy(),
        test_score,
        selected_threshold,
    )

    save_json(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_json(metrics_test_default, output_dir / "metrics_test_default_threshold.json")
    save_json(
        metrics_test_selected,
        output_dir / "metrics_test_selected_threshold.json",
    )

    confusion_matrix_table(
        prepared.y_valid.to_numpy(),
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        prepared.y_test.to_numpy(),
        test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    save_feature_importance(
        model,
        prepared.backend,
        output_dir / "feature_importance.csv",
        prepared.X_train.columns.tolist(),
    )
    save_model_artifacts(model, prepared.backend, output_dir, model_stem="model")
    joblib.dump(prepared.preprocessing, output_dir / "preprocessing.pkl")

    run_config = {
        "experiment_family": EXPERIMENT_FAMILY,
        "experiment_id": BACKEND_PHASE_NAMES[prepared.backend],
        "backend": prepared.backend,
        "tuned": False,
        "phase": phase_name,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "feature_set_summary": {
            "feature_setup": "Raw IEEE-CIS baseline features (432).",
            "original_v_features_retained": True,
            "reconstruction_error_used": False,
            "preprocessing_mode": prepared.preprocessing_mode,
            "backend": prepared.backend,
            "total_final_features": prepared.total_features,
        },
        "model_features_count": prepared.total_features,
        "preprocessing": {
            "fit_split": "train only",
            "mode": prepared.preprocessing_mode,
            "categorical_columns": prepared.categorical_columns,
            "categorical_columns_count": len(prepared.categorical_columns),
            "native_categorical_indices_used": prepared.cat_feature_indices is not None,
        },
        "leakage_prevention": {
            "train": "Preprocessing fit and model fitting.",
            "validation": "Early stopping and threshold selection.",
            "test": "Final evaluation only.",
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": 100,
            "best_iteration": best_iteration,
        },
        "class_imbalance": {
            "method": "scale_pos_weight or class_weights",
            "computed_from": "training labels only",
            "value": model_params.get("scale_pos_weight")
            or model_params.get("class_weights"),
        },
        "model_params": model_params,
    }
    save_json(run_config, output_dir / "run_config.json")

    return {
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
    }


def main(
    backend: str,
    output_dir: Path | None = None,
    phase_name: str | None = None,
    preprocessing_mode: str = "native",
    n_jobs: int = -1,
) -> dict[str, object]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")
    if preprocessing_mode not in SUPPORTED_PREPROCESSING_MODES:
        raise ValueError(f"Unsupported preprocessing_mode: {preprocessing_mode}")

    set_seed(RANDOM_SEED)
    resolved_output_dir = ensure_dir(output_dir or default_output_dir(backend))
    resolved_phase_name = phase_name or BACKEND_PHASE_NAMES[backend]

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log(f"Building raw feature matrices for {backend} ({preprocessing_mode}).")
    prepared = prepare_matrices_from_raw_splits(
        train_df,
        valid_df,
        test_df,
        backend=backend,
        preprocessing_mode=preprocessing_mode,
    )

    result = train_and_save(
        prepared,
        resolved_output_dir,
        resolved_phase_name,
        n_jobs=n_jobs,
    )

    print()
    print(f"GBDT Baseline Summary ({backend})")
    print("=" * (24 + len(backend)))
    print(
        "Validation PR-AUC : "
        f"{result['metrics_validation_selected']['average_precision']:.6f}"
    )
    print(
        f"Test PR-AUC       : "
        f"{result['metrics_test_selected']['average_precision']:.6f}"
    )
    print(f"Selected threshold: {result['selected_threshold']:.2f}")
    print(f"Best iteration    : {result['best_iteration']}")
    print(f"Outputs saved to  : {resolved_output_dir}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed/default raw-feature GBDT baselines "
            "(LightGBM, XGBoost, CatBoost)."
        )
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=SUPPORTED_BACKENDS,
        help="GBDT backend to train.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--phase-name", default=None)
    parser.add_argument(
        "--preprocessing-mode",
        choices=SUPPORTED_PREPROCESSING_MODES,
        default="native",
        help="native = library-recommended preprocessing; shared_lgbm = sensitivity run.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        backend=args.backend,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        preprocessing_mode=args.preprocessing_mode,
        n_jobs=args.n_jobs,
    )