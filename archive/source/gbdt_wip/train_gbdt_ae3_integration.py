"""Train AE3 reconstruction-error integration on a selected GBDT backend."""

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
    apply_preprocessing_for_backend,
    best_iteration_for,
    build_default_params,
    cat_feature_indices_for_matrix,
    fit_model,
    fit_preprocessing_for_backend,
    predict_positive_proba,
    save_feature_importance,
    save_model_artifacts,
)
from preprocessing import get_v_feature_columns, split_features_target
from splitting import chronological_split
from train_ae_integration_strategy_ablation import (
    RECONSTRUCTION_ERROR_FEATURES,
    combine_with_reconstruction_errors,
    load_or_compute_reconstruction_errors,
    validate_latent_split_manifest_alignment,
    validate_reconstruction_error_feature_alignment,
    validate_reconstruction_error_lengths,
)
from utils import ensure_dir, log, save_json, set_seed


GBDT_BASE_OUTPUT_DIR = OUTPUT_DIR / "gbdt_baseline_comparison"
EXPERIMENT_FAMILY = "gbdt_baseline_comparison"
DEFAULT_AE3_AUTOENCODER_DIR = (
    OUTPUT_DIR / "ae_integration_strategy_ablation_ld128" / "autoencoder_robust_ld128"
)

AE3_FIXED_SUBDIR = "AE3_fixed"
AE3_TUNED_SUBDIR = "optuna/AE3_tuned"


def build_ae3_matrices(
    train_df,
    valid_df,
    test_df,
    autoencoder_output_dir: Path,
    backend: str,
    preprocessing_mode: str,
) -> tuple[PreparedMatrices, dict[str, object]]:
    v_columns = get_v_feature_columns(train_df)
    validate_latent_split_manifest_alignment(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
    )

    errors = load_or_compute_reconstruction_errors(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
        v_columns,
    )
    validate_reconstruction_error_lengths(
        errors,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessing = fit_preprocessing_for_backend(
        X_train_raw,
        backend,
        preprocessing_mode,
    )
    X_train_base = apply_preprocessing_for_backend(X_train_raw, preprocessing, backend)
    X_valid_base = apply_preprocessing_for_backend(X_valid_raw, preprocessing, backend)
    X_test_base = apply_preprocessing_for_backend(X_test_raw, preprocessing, backend)

    X_train = combine_with_reconstruction_errors(X_train_base, errors["train"])
    X_valid = combine_with_reconstruction_errors(X_valid_base, errors["validation"])
    X_test = combine_with_reconstruction_errors(X_test_base, errors["test"])
    validate_reconstruction_error_feature_alignment(
        X_train,
        X_valid,
        X_test,
        v_columns,
    )

    categorical_columns = list(preprocessing.get("categorical_columns", []))
    use_native_categoricals = (
        preprocessing_mode == "native"
        and backend in {"catboost", "xgboost"}
        and preprocessing.get("preprocessing_mode") != "shared_lgbm"
    )
    cat_feature_indices = (
        cat_feature_indices_for_matrix(X_train, categorical_columns)
        if use_native_categoricals
        else None
    )

    feature_set_summary = {
        "feature_setup": "STR-AE3 reconstruction-error augmentation on GBDT backend.",
        "strategy": "reconstruction_error_augmentation",
        "original_v_features_retained": True,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "reconstruction_error_used": True,
        "reconstruction_error_features": list(RECONSTRUCTION_ERROR_FEATURES),
        "preprocessing_mode": preprocessing_mode,
        "backend": backend,
        "autoencoder_output_dir": str(autoencoder_output_dir),
        "total_final_features": int(X_train.shape[1]),
    }

    prepared = PreparedMatrices(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        preprocessing=preprocessing,
        categorical_columns=categorical_columns,
        cat_feature_indices=cat_feature_indices,
        preprocessing_mode=preprocessing_mode,
        backend=backend,
    )
    return prepared, feature_set_summary


def train_and_save(
    prepared: PreparedMatrices,
    feature_set_summary: dict[str, object],
    output_dir: Path,
    phase_name: str,
    autoencoder_output_dir: Path,
    n_jobs: int,
) -> dict[str, object]:
    model_params = build_default_params(
        prepared.backend,
        prepared.y_train,
        preprocessing_mode=prepared.preprocessing_mode,
        n_jobs=n_jobs,
    )
    log(f"Training AE3 fixed/default {prepared.backend} with validation early stopping.")
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

    metrics_valid_selected = binary_classification_metrics(
        prepared.y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_selected = binary_classification_metrics(
        prepared.y_test.to_numpy(),
        test_score,
        selected_threshold,
    )

    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
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
        "experiment_id": phase_name,
        "backend": prepared.backend,
        "tuned": False,
        "phase": phase_name,
        "autoencoder_output_dir": str(autoencoder_output_dir),
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "feature_set_summary": feature_set_summary,
        "model_features_count": prepared.total_features,
        "preprocessing": {
            "fit_split": "train only",
            "mode": prepared.preprocessing_mode,
            "categorical_columns": prepared.categorical_columns,
            "native_categorical_indices_used": prepared.cat_feature_indices is not None,
        },
        "leakage_prevention": {
            "train": "Preprocessing fit and model fitting.",
            "validation": "Early stopping and threshold selection.",
            "test": "Final evaluation only.",
            "ae_artifacts": "Frozen LD128 AE; reconstruction errors not refit on val/test.",
        },
        "threshold_selection": {
            "source_split": "validation",
            "selected_threshold": selected_threshold,
        },
        "early_stopping": {
            "best_iteration": best_iteration,
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
    autoencoder_output_dir: Path = DEFAULT_AE3_AUTOENCODER_DIR,
    output_dir: Path | None = None,
    phase_name: str = "GBDT-WIN-AE3-FIX",
    preprocessing_mode: str = "native",
    n_jobs: int = -1,
) -> dict[str, object]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")

    set_seed(RANDOM_SEED)
    resolved_output_dir = ensure_dir(
        output_dir or (GBDT_BASE_OUTPUT_DIR / AE3_FIXED_SUBDIR / backend)
    )

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)

    log(f"Building AE3 feature matrices for {backend} ({preprocessing_mode}).")
    prepared, feature_set_summary = build_ae3_matrices(
        train_df,
        valid_df,
        test_df,
        autoencoder_output_dir,
        backend,
        preprocessing_mode,
    )

    result = train_and_save(
        prepared,
        feature_set_summary,
        resolved_output_dir,
        phase_name,
        autoencoder_output_dir,
        n_jobs=n_jobs,
    )

    print()
    print(f"GBDT AE3 Integration Summary ({backend})")
    print("=" * (30 + len(backend)))
    print(
        "Validation PR-AUC : "
        f"{result['metrics_validation_selected']['average_precision']:.6f}"
    )
    print(
        f"Test PR-AUC       : "
        f"{result['metrics_test_selected']['average_precision']:.6f}"
    )
    print(f"Outputs saved to  : {resolved_output_dir}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AE3 reconstruction-error integration on a GBDT backend."
    )
    parser.add_argument("--backend", required=True, choices=SUPPORTED_BACKENDS)
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=DEFAULT_AE3_AUTOENCODER_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--phase-name", default="GBDT-WIN-AE3-FIX")
    parser.add_argument(
        "--preprocessing-mode",
        choices=SUPPORTED_PREPROCESSING_MODES,
        default="native",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        backend=args.backend,
        autoencoder_output_dir=args.autoencoder_output_dir,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        preprocessing_mode=args.preprocessing_mode,
        n_jobs=args.n_jobs,
    )