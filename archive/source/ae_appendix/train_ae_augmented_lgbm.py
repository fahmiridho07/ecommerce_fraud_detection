"""Train AE-augmented LightGBM with original features plus robust LD128 AE features.

This experiment is intentionally different from the AE-LightGBM replacement
model. Original V1-V339 features are retained, then robust Autoencoder LD128
latent features and reconstruction error are appended as additional signals.
"""

from __future__ import annotations

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
    AE_AUGMENTED_COMPARISON_FILE,
    AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR,
    AE_LGBM_LD128_OUTPUT_DIR,
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    DATA_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    ID_COL,
    OPTUNA_OUTPUT_DIR,
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
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_THRESHOLD = 0.5
EXPECTED_LATENT_DIM = 128
RECONSTRUCTION_ERROR_FEATURE = "ae_reconstruction_mse"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_reconstruction_error_csv(path: Path) -> np.ndarray:
    """Load one saved reconstruction-error file."""
    if not path.exists():
        raise FileNotFoundError(str(path))

    values = pd.read_csv(path)
    if "reconstruction_mse" not in values.columns:
        raise KeyError(f"{path} is missing reconstruction_mse column.")
    return values["reconstruction_mse"].to_numpy(dtype="float32")


def reconstruction_error_files(autoencoder_output_dir: Path) -> dict[str, Path]:
    return {
        "train": autoencoder_output_dir / "reconstruction_error_train.csv",
        "validation": autoencoder_output_dir / "reconstruction_error_valid.csv",
        "test": autoencoder_output_dir / "reconstruction_error_test.csv",
    }


def validate_reconstruction_errors(
    errors: dict[str, np.ndarray],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    expected = {
        "train": train_rows,
        "validation": valid_rows,
        "test": test_rows,
    }
    for split_name, row_count in expected.items():
        if errors[split_name].shape[0] != row_count:
            raise ValueError(
                f"{split_name} reconstruction error row count "
                f"{errors[split_name].shape[0]} does not match split row count "
                f"{row_count}."
            )


def v_columns_from_autoencoder_config(
    autoencoder_run_config: dict[str, object],
    train_df: pd.DataFrame,
) -> list[str]:
    """Prefer saved AE column order; fall back to the current split columns."""
    saved_columns = autoencoder_run_config.get("v_columns")
    if isinstance(saved_columns, list) and saved_columns:
        return [str(column) for column in saved_columns]
    return get_v_feature_columns(train_df)


def transform_v_features_for_saved_autoencoder(
    df: pd.DataFrame,
    v_columns: list[str],
    scaler,
    autoencoder_run_config: dict[str, object],
) -> np.ndarray:
    """Apply the saved robust-AE preprocessing without fitting anything."""
    X_raw = df.loc[:, v_columns].fillna(0).astype("float32")
    X_scaled = scaler.transform(X_raw).astype("float32")

    preprocessing = autoencoder_run_config.get("preprocessing", {})
    if isinstance(preprocessing, dict) and preprocessing.get("scaled_clipping_enabled"):
        clip_min = float(preprocessing.get("clip_min", -10.0))
        clip_max = float(preprocessing.get("clip_max", 10.0))
        X_scaled = np.clip(X_scaled, clip_min, clip_max)

    return X_scaled


def compute_reconstruction_errors_from_artifacts(
    autoencoder_output_dir: Path,
    autoencoder_run_config: dict[str, object],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compute reconstruction MSE from saved robust AE artifacts only.

    This fallback never fits on validation or test. It only loads the saved
    train-fitted scaler and Autoencoder model.
    """
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "TensorFlow is required to compute missing reconstruction-error "
            "files from saved Autoencoder artifacts."
        ) from exc

    scaler_path = autoencoder_output_dir / "v_scaler.pkl"
    model_path = autoencoder_output_dir / "autoencoder_model.keras"
    if not scaler_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Cannot compute reconstruction errors because saved robust AE "
            f"artifacts are missing: {scaler_path} or {model_path}"
        )

    v_columns = v_columns_from_autoencoder_config(autoencoder_run_config, train_df)
    scaler = joblib.load(scaler_path)
    autoencoder = keras.models.load_model(model_path, compile=False)
    batch_size = int(autoencoder_run_config.get("training", {}).get("batch_size", 1024))

    errors: dict[str, np.ndarray] = {}
    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        X = transform_v_features_for_saved_autoencoder(
            split_df,
            v_columns,
            scaler,
            autoencoder_run_config,
        )
        reconstructed = autoencoder.predict(X, batch_size=batch_size, verbose=0)
        errors[split_name] = np.mean(np.square(X - reconstructed), axis=1).astype("float32")
    return errors


def load_or_compute_reconstruction_errors(
    autoencoder_output_dir: Path,
    autoencoder_run_config: dict[str, object],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], str]:
    """Load saved reconstruction errors, or compute them from saved AE artifacts."""
    paths = reconstruction_error_files(autoencoder_output_dir)
    if all(path.exists() for path in paths.values()):
        errors = {
            split_name: load_reconstruction_error_csv(path)
            for split_name, path in paths.items()
        }
        source = "saved_reconstruction_error_csv"
    else:
        missing = [str(path) for path in paths.values() if not path.exists()]
        log(
            "Reconstruction error CSV file(s) missing; computing from saved "
            "robust Autoencoder artifacts: " + "; ".join(missing)
        )
        errors = compute_reconstruction_errors_from_artifacts(
            autoencoder_output_dir,
            autoencoder_run_config,
            train_df,
            valid_df,
            test_df,
        )
        source = "computed_from_saved_autoencoder_artifacts"

    validate_reconstruction_errors(
        errors,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    return errors, source


def combine_original_latent_and_error(
    X_original: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
    reconstruction_error: np.ndarray,
) -> pd.DataFrame:
    """Append AE features to the original baseline feature matrix."""
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    error_df = pd.DataFrame(
        {RECONSTRUCTION_ERROR_FEATURE: reconstruction_error.astype("float32")}
    )
    return pd.concat(
        [
            X_original.reset_index(drop=True),
            latent_df.reset_index(drop=True),
            error_df.reset_index(drop=True),
        ],
        axis=1,
    )


def validate_augmented_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
) -> None:
    """Ensure aligned matrices and confirm original V-features are retained."""
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        missing_count = len(v_columns) - len(retained_v_columns)
        raise ValueError(
            "AE-augmented experiment must retain all original V-features; "
            f"missing count: {missing_count}."
        )
    if RECONSTRUCTION_ERROR_FEATURE not in X_train.columns:
        raise ValueError("Missing AE reconstruction error feature.")


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    save_json(metrics, path)


def metric_delta(
    augmented_metrics: dict[str, object],
    baseline_metrics: dict[str, object],
    metric_name: str,
) -> float:
    return float(augmented_metrics[metric_name] - baseline_metrics[metric_name])


def load_metrics_if_available(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return load_json(path)


def build_baseline_comparison(
    augmented_metrics: dict[str, object],
) -> dict[str, object] | None:
    """Compare augmented results against available baseline metrics."""
    baseline_default = load_metrics_if_available(
        BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    )
    baseline_tuned = load_metrics_if_available(
        OPTUNA_OUTPUT_DIR / "baseline_lgbm" / "metrics_test_selected_threshold.json"
    )

    if baseline_default is None and baseline_tuned is None:
        return None

    comparison: dict[str, object] = {
        "ae_augmented_test_pr_auc": augmented_metrics["average_precision"],
        "ae_augmented_test_roc_auc": augmented_metrics["roc_auc"],
        "ae_augmented_test_f1": augmented_metrics["f1"],
        "ae_augmented_test_mcc": augmented_metrics["mcc"],
    }

    if baseline_default is not None:
        comparison.update(
            {
                "baseline_default_test_pr_auc": baseline_default["average_precision"],
                "delta_vs_baseline_default_pr_auc": metric_delta(
                    augmented_metrics,
                    baseline_default,
                    "average_precision",
                ),
                "baseline_default_test_roc_auc": baseline_default["roc_auc"],
                "delta_vs_baseline_default_roc_auc": metric_delta(
                    augmented_metrics,
                    baseline_default,
                    "roc_auc",
                ),
                "baseline_default_test_f1": baseline_default["f1"],
                "delta_vs_baseline_default_f1": metric_delta(
                    augmented_metrics,
                    baseline_default,
                    "f1",
                ),
                "baseline_default_test_mcc": baseline_default["mcc"],
                "delta_vs_baseline_default_mcc": metric_delta(
                    augmented_metrics,
                    baseline_default,
                    "mcc",
                ),
            }
        )

    if baseline_tuned is not None:
        comparison.update(
            {
                "baseline_tuned_test_pr_auc": baseline_tuned["average_precision"],
                "delta_vs_baseline_tuned_pr_auc": metric_delta(
                    augmented_metrics,
                    baseline_tuned,
                    "average_precision",
                ),
                "baseline_tuned_test_roc_auc": baseline_tuned["roc_auc"],
                "delta_vs_baseline_tuned_roc_auc": metric_delta(
                    augmented_metrics,
                    baseline_tuned,
                    "roc_auc",
                ),
                "baseline_tuned_test_f1": baseline_tuned["f1"],
                "delta_vs_baseline_tuned_f1": metric_delta(
                    augmented_metrics,
                    baseline_tuned,
                    "f1",
                ),
                "baseline_tuned_test_mcc": baseline_tuned["mcc"],
                "delta_vs_baseline_tuned_mcc": metric_delta(
                    augmented_metrics,
                    baseline_tuned,
                    "mcc",
                ),
            }
        )

    return comparison


def total_features_from_config(
    run_config: dict[str, object],
    feature_summary: dict[str, object] | None = None,
) -> object:
    if feature_summary and "total_final_features" in feature_summary:
        return feature_summary["total_final_features"]
    if "model_features_count" in run_config:
        return run_config["model_features_count"]

    feature_construction = run_config.get("feature_construction", {})
    if isinstance(feature_construction, dict):
        for key in ("total_feature_count", "total_final_features"):
            if key in feature_construction:
                return feature_construction[key]
    return None


def comparison_row(
    model_name: str,
    metrics_path: Path,
    run_config_path: Path,
    feature_summary_path: Path | None = None,
) -> dict[str, object] | None:
    if not metrics_path.exists() or not run_config_path.exists():
        return None

    metrics = load_json(metrics_path)
    run_config = load_json(run_config_path)
    feature_summary = (
        load_json(feature_summary_path)
        if feature_summary_path and feature_summary_path.exists()
        else None
    )

    return {
        "model_name": model_name,
        "test_pr_auc": metrics.get("average_precision"),
        "test_roc_auc": metrics.get("roc_auc"),
        "test_precision": metrics.get("precision"),
        "test_recall": metrics.get("recall"),
        "test_f1": metrics.get("f1"),
        "test_mcc": metrics.get("mcc"),
        "selected_threshold": metrics.get("threshold"),
        "best_iteration": run_config.get("early_stopping", {}).get("best_iteration"),
        "total_features": total_features_from_config(run_config, feature_summary),
    }


def build_augmented_comparison_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = [
        (
            "baseline_lgbm_default",
            BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json",
            BASELINE_OUTPUT_DIR / "run_config.json",
            None,
        ),
        (
            "baseline_lgbm_tuned",
            OPTUNA_OUTPUT_DIR / "baseline_lgbm" / "metrics_test_selected_threshold.json",
            OPTUNA_OUTPUT_DIR / "baseline_lgbm" / "run_config.json",
            None,
        ),
        (
            "ae_lgbm_ld128_default",
            AE_LGBM_LD128_OUTPUT_DIR / "metrics_test_selected_threshold.json",
            AE_LGBM_LD128_OUTPUT_DIR / "run_config.json",
            AE_LGBM_LD128_OUTPUT_DIR / "feature_set_summary.json",
        ),
        (
            "ae_lgbm_ld128_tuned",
            OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128" / "metrics_test_selected_threshold.json",
            OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128" / "run_config.json",
            OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128" / "feature_set_summary.json",
        ),
        (
            "ae_augmented_lgbm_ld128_default",
            AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR
            / "metrics_test_selected_threshold.json",
            AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR / "run_config.json",
            AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR / "feature_set_summary.json",
        ),
    ]

    for model_name, metrics_path, run_config_path, feature_summary_path in candidates:
        row = comparison_row(
            model_name,
            metrics_path,
            run_config_path,
            feature_summary_path,
        )
        if row is not None:
            rows.append(row)

    columns = [
        "model_name",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_mcc",
        "selected_threshold",
        "best_iteration",
        "total_features",
    ]
    return pd.DataFrame(rows, columns=columns)


def save_augmented_comparison_table() -> pd.DataFrame:
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table = build_augmented_comparison_table()
    table.to_csv(AE_AUGMENTED_COMPARISON_FILE, index=False)
    return table


def main(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    output_dir: Path = AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR,
    phase_name: str = "6_ae_augmented_lgbm_ld128_default",
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)

    log("Building original baseline feature matrices.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    # Baseline preprocessing is fit only on train. Numeric NaNs, including
    # original V-feature NaNs, are preserved for LightGBM native handling.
    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train_original = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid_original = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test_original = apply_baseline_preprocessing(X_test_raw, preprocessing)

    log("Loading robust Autoencoder LD128 latent features.")
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
            f"Expected LD128 latent features, found {latent_train.shape[1]}."
        )

    log("Loading or computing robust Autoencoder reconstruction errors.")
    reconstruction_errors, reconstruction_error_source = load_or_compute_reconstruction_errors(
        autoencoder_output_dir,
        autoencoder_run_config,
        train_df,
        valid_df,
        test_df,
    )

    log("Combining original features with AE latent features and reconstruction error.")
    # This is an augmentation experiment: original V1-V339 features are kept.
    X_train = combine_original_latent_and_error(
        X_train_original,
        latent_train,
        latent_feature_names,
        reconstruction_errors["train"],
    )
    X_valid = combine_original_latent_and_error(
        X_valid_original,
        latent_valid,
        latent_feature_names,
        reconstruction_errors["validation"],
    )
    X_test = combine_original_latent_and_error(
        X_test_original,
        latent_test,
        latent_feature_names,
        reconstruction_errors["test"],
    )
    validate_augmented_feature_alignment(X_train, X_valid, X_test, v_columns)

    categorical_columns = preprocessing["categorical_columns"]
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training AE-augmented LightGBM with validation early stopping.")
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

    log("Saving AE-augmented LightGBM outputs.")
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

    feature_set_summary = {
        "number_of_original_features": int(X_train_original.shape[1]),
        "number_of_ae_latent_features": int(len(latent_feature_names)),
        "reconstruction_error_included": True,
        "reconstruction_error_feature": RECONSTRUCTION_ERROR_FEATURE,
        "total_final_features": int(X_train.shape[1]),
        "original_v_features_retained": True,
        "number_of_original_v_features_retained": int(len(v_columns)),
        "ae_output_directory_used": str(autoencoder_output_dir),
    }
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")

    comparison = build_baseline_comparison(metrics_test_selected)
    if comparison is not None:
        save_json(comparison, output_dir / "comparison_against_baseline.json")

    run_config = {
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
        "leakage_prevention": {
            "preprocessing_fit": "Categorical mappings fit on train only.",
            "autoencoder_features": (
                "Saved robust Autoencoder LD128 outputs are loaded without "
                "refitting on validation or test."
            ),
            "reconstruction_error": (
                "Loaded from saved AE outputs when available; fallback computes "
                "from saved train-fitted AE artifacts without fitting anything."
            ),
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split is used only once for final evaluation.",
            "kaggle_competition_test_files_used": False,
        },
        "feature_construction": {
            "experiment_type": "augmentation_not_replacement",
            "original_features_retained": True,
            "original_v_features_retained": True,
            "original_feature_count": int(X_train_original.shape[1]),
            "original_v_feature_count": int(len(v_columns)),
            "ae_latent_feature_count": int(len(latent_feature_names)),
            "reconstruction_error_included": True,
            "reconstruction_error_feature": RECONSTRUCTION_ERROR_FEATURE,
            "reconstruction_error_source": reconstruction_error_source,
            "total_feature_count": int(X_train.shape[1]),
            "robust_autoencoder_output_dir": str(autoencoder_output_dir),
        },
        "preprocessing": {
            "baseline_categorical_fit": "Categorical mappings fit on train original features only.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "categorical_missing_value": preprocessing["missing_category"],
            "unknown_category_value": preprocessing["unknown_category_value"],
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
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

    comparison_table = save_augmented_comparison_table()

    print()
    print("AE-Augmented LightGBM LD128 Summary")
    print("===================================")
    print(f"Validation PR-AUC : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test PR-AUC       : {metrics_test_selected['average_precision']:.6f}")
    print(f"Validation ROC-AUC: {metrics_valid_selected['roc_auc']:.6f}")
    print(f"Test ROC-AUC      : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Test precision    : {metrics_test_selected['precision']:.6f}")
    print(f"Test recall       : {metrics_test_selected['recall']:.6f}")
    print(f"Test F1           : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC          : {metrics_test_selected['mcc']:.6f}")
    print(f"Best iteration    : {best_iteration}")
    print(f"Total features    : {X_train.shape[1]}")
    print(f"Outputs saved to  : {output_dir}")
    print(f"Comparison saved  : {AE_AUGMENTED_COMPARISON_FILE}")

    if comparison is not None:
        print()
        print("Comparison Against Baseline")
        print("===========================")
        if "delta_vs_baseline_default_pr_auc" in comparison:
            print(
                "Delta vs default baseline PR-AUC: "
                f"{comparison['delta_vs_baseline_default_pr_auc']:+.6f}"
            )
        if "delta_vs_baseline_tuned_pr_auc" in comparison:
            print(
                "Delta vs tuned baseline PR-AUC  : "
                f"{comparison['delta_vs_baseline_tuned_pr_auc']:+.6f}"
            )

    print()
    print("Comparison Table")
    print("================")
    print(comparison_table.to_string(index=False))

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
        "comparison_against_baseline": comparison,
    }


if __name__ == "__main__":
    main()
