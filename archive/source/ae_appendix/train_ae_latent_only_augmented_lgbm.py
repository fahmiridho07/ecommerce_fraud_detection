"""Train clean latent-only AE augmentation: original features + LD128 latents.

This experiment retains all original V-features and appends robust Autoencoder
LD128 latent features only. Reconstruction error is intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR,
    AE_LGBM_LD128_OUTPUT_DIR,
    AE_LATENT_ONLY_AUGMENTED_LGBM_LD128_OUTPUT_DIR,
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    DATA_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    ID_COL,
    LATENT_INTEGRATION_STRATEGY_COMPARISON_FILE,
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
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from splitting import chronological_split
from train_ae_lgbm import load_robust_latent_outputs, validate_latent_outputs
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
    save_metrics,
)
from utils import ensure_dir, log, save_json, set_seed


EXPECTED_LATENT_DIM = 128
RECONSTRUCTION_ERROR_FEATURE = "ae_reconstruction_mse"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def combine_original_and_latent(
    X_original: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
) -> pd.DataFrame:
    """Append latent features to the baseline feature matrix."""
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    combined = pd.concat(
        [X_original.reset_index(drop=True), latent_df.reset_index(drop=True)],
        axis=1,
    )
    if combined.columns.duplicated().any():
        duplicates = combined.columns[combined.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature names after latent append: {duplicates}")
    return combined


def validate_latent_only_feature_matrix(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
    latent_feature_names: list[str],
    expected_original_feature_count: int,
) -> dict[str, int]:
    """Assert clean latent-only augmentation invariants."""
    for split_name, matrix in (
        ("train", X_train),
        ("validation", X_valid),
        ("test", X_test),
    ):
        if matrix.columns.tolist() != X_train.columns.tolist():
            raise ValueError(f"{split_name} feature columns do not align with train.")

    if RECONSTRUCTION_ERROR_FEATURE in X_train.columns:
        raise ValueError("Reconstruction error must not be included in this experiment.")

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        raise ValueError(
            "All original V-features must be retained; "
            f"expected {len(v_columns)}, found {len(retained_v_columns)}."
        )

    latent_in_matrix = [name for name in latent_feature_names if name in X_train.columns]
    if len(latent_in_matrix) != EXPECTED_LATENT_DIM:
        raise ValueError(
            f"Expected {EXPECTED_LATENT_DIM} latent features in matrix, "
            f"found {len(latent_in_matrix)}."
        )

    original_feature_count = int(X_train.shape[1] - EXPECTED_LATENT_DIM)
    if original_feature_count != expected_original_feature_count:
        raise ValueError(
            "Original feature count mismatch: "
            f"expected {expected_original_feature_count}, "
            f"found {original_feature_count}."
        )

    if not np.isfinite(X_train[latent_feature_names].to_numpy(dtype="float64")).all():
        raise ValueError("Latent feature values must be finite.")

    return {
        "original_feature_count": original_feature_count,
        "original_v_feature_count": len(v_columns),
        "latent_feature_count": EXPECTED_LATENT_DIM,
        "total_feature_count": int(X_train.shape[1]),
    }


def prepare_feature_matrices(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    dict[str, object],
    list[str],
    list[str],
    dict[str, int],
    dict[str, object],
]:
    """Load data, preprocess baseline features, and append LD128 latents."""
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Separating target and fitting train-only baseline preprocessing.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train_original = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid_original = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test_original = apply_baseline_preprocessing(X_test_raw, preprocessing)

    v_columns = get_v_feature_columns(X_train_original)
    original_feature_count = int(X_train_original.shape[1])

    log(f"Loading LD128 latent arrays from {autoencoder_output_dir}.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        autoencoder_run_config,
    ) = load_robust_latent_outputs(autoencoder_output_dir)

    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    if latent_train.shape[1] != EXPECTED_LATENT_DIM:
        raise ValueError(
            f"Expected latent_dim={EXPECTED_LATENT_DIM}, "
            f"found {latent_train.shape[1]}."
        )

    X_train = combine_original_and_latent(
        X_train_original,
        latent_train,
        latent_feature_names,
    )
    X_valid = combine_original_and_latent(
        X_valid_original,
        latent_valid,
        latent_feature_names,
    )
    X_test = combine_original_and_latent(
        X_test_original,
        latent_test,
        latent_feature_names,
    )

    feature_counts = validate_latent_only_feature_matrix(
        X_train,
        X_valid,
        X_test,
        v_columns,
        latent_feature_names,
        original_feature_count,
    )

    return (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        preprocessing,
        v_columns,
        latent_feature_names,
        feature_counts,
        autoencoder_run_config,
    )


def run_pre_training_validation() -> dict[str, object]:
    """Validate artifacts and perform a lightweight feature-construction smoke test."""
    required_paths = {
        "baseline_validation_metrics": (
            BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
        ),
        "baseline_test_metrics": (
            BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
        ),
        "replacement_validation_metrics": (
            AE_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
        ),
        "latent_train": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_train.npy",
        "latent_valid": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_valid.npy",
        "latent_test": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_test.npy",
        "latent_feature_names": (
            AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_feature_names.json"
        ),
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Pre-run validation failed. Missing required artifacts: "
            + ", ".join(missing)
        )

    (
        X_train,
        X_valid,
        X_test,
        _y_train,
        _y_valid,
        _y_test,
        _preprocessing,
        v_columns,
        latent_feature_names,
        feature_counts,
        _autoencoder_run_config,
    ) = prepare_feature_matrices()

    smoke_sample = min(256, len(X_train))
    smoke_frame = X_train.iloc[:smoke_sample].copy()
    assert RECONSTRUCTION_ERROR_FEATURE not in smoke_frame.columns
    assert smoke_frame.shape[1] == feature_counts["total_feature_count"]

    return {
        "status": "ok",
        "feature_counts": feature_counts,
        "v_columns": len(v_columns),
        "latent_feature_names": len(latent_feature_names),
        "reconstruction_error_present": False,
        "smoke_rows": smoke_sample,
        "smoke_columns": int(smoke_frame.shape[1]),
    }


def build_integration_comparison(
    output_dir: Path,
    feature_counts: dict[str, int],
    metrics_valid_selected: dict[str, object],
    metrics_test_selected: dict[str, object],
    best_iteration: int,
    selected_threshold: float,
) -> pd.DataFrame:
    """Write the controlled integration-strategy comparison CSV."""
    baseline_valid = load_json(
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )
    baseline_test = load_json(
        BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )
    replacement_valid = load_json(
        AE_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )
    replacement_test = load_json(
        AE_LGBM_LD128_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )
    confounded_valid = load_json(
        AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
    )
    confounded_test = load_json(
        AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )

    baseline_valid_ap = float(baseline_valid["average_precision"])
    baseline_test_ap = float(baseline_test["average_precision"])
    baseline_run_config = load_json(BASELINE_OUTPUT_DIR / "run_config.json")

    rows = [
        {
            "model_name": "baseline_lgbm_default",
            "integration_strategy": "original_features_only",
            "original_v_retained": True,
            "latent_used": False,
            "reconstruction_error_used": False,
            "latent_dimension": "",
            "validation_average_precision": baseline_valid_ap,
            "test_average_precision": baseline_test_ap,
            "validation_delta_vs_baseline": 0.0,
            "test_delta_vs_baseline": 0.0,
            "total_features": int(baseline_run_config["model_features_count"]),
            "best_iteration": int(
                baseline_run_config["early_stopping"]["best_iteration"]
            ),
            "selected_threshold": float(baseline_valid["threshold"]),
            "comparison_status": "primary_control",
            "metric_source": str(
                BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
            ),
        },
        {
            "model_name": "ae_lgbm_ld128_default",
            "integration_strategy": "latent_replacement",
            "original_v_retained": False,
            "latent_used": True,
            "reconstruction_error_used": False,
            "latent_dimension": EXPECTED_LATENT_DIM,
            "validation_average_precision": float(
                replacement_valid["average_precision"]
            ),
            "test_average_precision": float(replacement_test["average_precision"]),
            "validation_delta_vs_baseline": float(
                replacement_valid["average_precision"]
            )
            - baseline_valid_ap,
            "test_delta_vs_baseline": float(replacement_test["average_precision"])
            - baseline_test_ap,
            "total_features": 221,
            "best_iteration": int(
                load_json(AE_LGBM_LD128_OUTPUT_DIR / "run_config.json")[
                    "early_stopping"
                ]["best_iteration"]
            ),
            "selected_threshold": float(replacement_valid["threshold"]),
            "comparison_status": "primary_control",
            "metric_source": str(
                AE_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
            ),
        },
        {
            "model_name": "ae_latent_only_augmented_lgbm_ld128_default",
            "integration_strategy": "latent_only_augmentation",
            "original_v_retained": True,
            "latent_used": True,
            "reconstruction_error_used": False,
            "latent_dimension": EXPECTED_LATENT_DIM,
            "validation_average_precision": float(
                metrics_valid_selected["average_precision"]
            ),
            "test_average_precision": float(metrics_test_selected["average_precision"]),
            "validation_delta_vs_baseline": float(
                metrics_valid_selected["average_precision"]
            )
            - baseline_valid_ap,
            "test_delta_vs_baseline": float(metrics_test_selected["average_precision"])
            - baseline_test_ap,
            "total_features": feature_counts["total_feature_count"],
            "best_iteration": best_iteration,
            "selected_threshold": selected_threshold,
            "comparison_status": "primary_control",
            "metric_source": str(
                output_dir / "metrics_validation_selected_threshold.json"
            ),
        },
        {
            "model_name": "ae_augmented_lgbm_ld128_default_confounded",
            "integration_strategy": "latent_plus_reconstruction_augmentation",
            "original_v_retained": True,
            "latent_used": True,
            "reconstruction_error_used": True,
            "latent_dimension": EXPECTED_LATENT_DIM,
            "validation_average_precision": float(confounded_valid["average_precision"]),
            "test_average_precision": float(confounded_test["average_precision"]),
            "validation_delta_vs_baseline": float(
                confounded_valid["average_precision"]
            )
            - baseline_valid_ap,
            "test_delta_vs_baseline": float(confounded_test["average_precision"])
            - baseline_test_ap,
            "total_features": 561,
            "best_iteration": int(
                load_json(AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR / "run_config.json")[
                    "early_stopping"
                ]["best_iteration"]
            ),
            "selected_threshold": float(confounded_valid["threshold"]),
            "comparison_status": "confounded_reference_only",
            "metric_source": str(
                AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR
                / "metrics_validation_selected_threshold.json"
            ),
        },
    ]

    table = pd.DataFrame(rows)
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table.to_csv(LATENT_INTEGRATION_STRATEGY_COMPARISON_FILE, index=False)
    return table


def main(skip_training: bool = False) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(AE_LATENT_ONLY_AUGMENTED_LGBM_LD128_OUTPUT_DIR)

    precheck = run_pre_training_validation()
    print("Pre-run validation passed.")
    print(f"Expected final feature count: {precheck['feature_counts']['total_feature_count']}")
    print(
        "Feature breakdown: "
        f"original={precheck['feature_counts']['original_feature_count']}, "
        f"v={precheck['feature_counts']['original_v_feature_count']}, "
        f"latent={precheck['feature_counts']['latent_feature_count']}"
    )
    print(f"Reconstruction error present: {precheck['reconstruction_error_present']}")

    if skip_training:
        return precheck

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        preprocessing,
        v_columns,
        latent_feature_names,
        feature_counts,
        autoencoder_run_config,
    ) = prepare_feature_matrices()

    categorical_columns = preprocessing["categorical_columns"]
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training latent-only augmented LightGBM with validation early stopping.")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=categorical_columns,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iteration = int(model.best_iteration_ or model.n_estimators)

    log("Generating validation and test probabilities.")
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]

    log("Selecting classification threshold on validation only.")
    threshold_table = threshold_selection_table(y_valid.to_numpy(), valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_valid_default = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        DEFAULT_THRESHOLD,
    )
    metrics_valid_selected = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        y_test.to_numpy(),
        test_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        y_test.to_numpy(),
        test_score,
        selected_threshold,
    )

    log("Saving latent-only augmentation outputs.")
    save_metrics(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_metrics(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_metrics(
        metrics_test_default,
        output_dir / "metrics_test_default_threshold.json",
    )
    save_metrics(
        metrics_test_selected,
        output_dir / "metrics_test_selected_threshold.json",
    )

    confusion_matrix_table(
        y_valid.to_numpy(),
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        y_test.to_numpy(),
        test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    save_feature_importance(model, output_dir / "feature_importance.csv")
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")

    run_config = {
        "phase": "ae_latent_only_augmented_lgbm_ld128_default",
        "experiment_purpose": (
            "Test whether LD128 Autoencoder latent features improve validation "
            "Average Precision when appended to original features while retaining "
            "all original V-features, without reconstruction error."
        ),
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "split_strategy": "chronological",
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "leakage_prevention": {
            "preprocessing_fit": "Baseline categorical mappings fit on train only.",
            "autoencoder_latents": (
                "Loaded from saved robust Autoencoder LD128 outputs; no refitting."
            ),
            "threshold_selection": "Validation split only.",
            "test_usage": "Final descriptive evaluation only.",
            "test_not_used_for_model_selection": True,
            "kaggle_competition_test_files_used": False,
        },
        "feature_construction": {
            "integration_strategy": "latent_only_augmentation",
            "original_v_features_retained": True,
            "latent_features_added": True,
            "reconstruction_error_included": False,
            "feature_engineering_included": False,
            "original_feature_count": feature_counts["original_feature_count"],
            "original_v_feature_count": feature_counts["original_v_feature_count"],
            "latent_feature_count": feature_counts["latent_feature_count"],
            "total_feature_count": feature_counts["total_feature_count"],
            "latent_dimension": EXPECTED_LATENT_DIM,
            "latent_source_directory": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
            "latent_feature_names": latent_feature_names,
            "v_columns": v_columns,
        },
        "preprocessing": {
            "baseline_categorical_fit": "Categorical mappings fit on train original features only.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "categorical_missing_value": preprocessing["missing_category"],
            "unknown_category_value": preprocessing["unknown_category_value"],
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
        },
        "autoencoder_source_run_config": autoencoder_run_config,
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
            "stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": best_iteration,
        },
        "class_imbalance": {
            "method": "scale_pos_weight",
            "computed_from": "training labels only",
            "value": model_params["scale_pos_weight"],
        },
        "model_params": model_params,
    }
    save_json(run_config, output_dir / "run_config.json")

    comparison_table = build_integration_comparison(
        output_dir,
        feature_counts,
        metrics_valid_selected,
        metrics_test_selected,
        best_iteration,
        selected_threshold,
    )

    baseline_valid_ap = float(
        load_json(BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json")[
            "average_precision"
        ]
    )
    aug_valid_ap = float(metrics_valid_selected["average_precision"])
    valid_delta = aug_valid_ap - baseline_valid_ap

    print()
    print("Latent-Only Augmentation Summary")
    print("==============================")
    print(f"Validation PR-AUC : {aug_valid_ap:.6f}")
    print(f"Baseline valid AP : {baseline_valid_ap:.6f}")
    print(f"Validation delta  : {valid_delta:+.6f}")
    print(f"Test PR-AUC       : {metrics_test_selected['average_precision']:.6f}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Best iteration    : {best_iteration}")
    print(f"Total features    : {feature_counts['total_feature_count']}")
    print(f"Outputs saved to  : {output_dir}")
    print(f"Comparison saved  : {LATENT_INTEGRATION_STRATEGY_COMPARISON_FILE}")
    print()
    print(comparison_table.to_string(index=False))

    return {
        "precheck": precheck,
        "validation_average_precision": aug_valid_ap,
        "validation_delta_vs_baseline": valid_delta,
        "test_average_precision": float(metrics_test_selected["average_precision"]),
        "feature_counts": feature_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clean latent-only AE augmentation (LD128)."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run pre-training validation and smoke test without training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(skip_training=args.validate_only)