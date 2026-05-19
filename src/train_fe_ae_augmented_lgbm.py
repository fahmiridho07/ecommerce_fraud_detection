"""Train FE-LightGBM with robust LD128 AE latent features and reconstruction error."""

from __future__ import annotations

import argparse
import gc
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
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    DATA_DIR,
    FE_AE_AUGMENTED_LGBM_OUTPUT_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from feature_engineering import feature_engineering_summary
from preprocessing import apply_baseline_preprocessing, fit_baseline_preprocessing, get_v_feature_columns
from train_ae_lgbm import load_robust_latent_outputs, validate_latent_outputs
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from train_feature_engineered_lgbm import (
    prepare_engineered_splits,
    save_engineered_feature_importance,
)
from train_reconstruction_error_lgbm import (
    load_reconstruction_errors,
    validate_reconstruction_error_lengths,
)
from utils import ensure_dir, log, save_json, set_seed


RECONSTRUCTION_ERROR_FEATURE = "ae_reconstruction_mse"
EXPECTED_LATENT_DIM = 128
DEFAULT_FE_VALIDATION_PR_AUC_REFERENCE = 0.627793
TUNED_FE_VALIDATION_PR_AUC_REFERENCE = 0.654316
C_PROMISING_DELTA_FOR_OPTUNA = 0.005
MEMORY_SAFE_N_JOBS = 4


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def output_dir_is_non_empty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir_is_non_empty(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Pass --overwrite only when you intentionally want to replace this "
            "controlled experiment output."
        )
    return ensure_dir(output_dir)


def combine_fe_latent_and_error(
    X_fe: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
    reconstruction_error: np.ndarray,
) -> pd.DataFrame:
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    error_df = pd.DataFrame(
        {RECONSTRUCTION_ERROR_FEATURE: reconstruction_error.astype("float32")}
    )
    return pd.concat(
        [
            X_fe.reset_index(drop=True),
            latent_df.reset_index(drop=True),
            error_df.reset_index(drop=True),
        ],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    engineered_features: list[str],
    v_columns: list[str],
    latent_feature_names: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train.")

    missing_engineered = [
        feature for feature in engineered_features if feature not in X_train.columns
    ]
    if missing_engineered:
        raise ValueError(
            "Missing engineered feature(s): " + ", ".join(missing_engineered[:20])
        )

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        raise ValueError(
            "Original V-features must be retained; "
            f"retained {len(retained_v_columns)} of {len(v_columns)}."
        )

    missing_latent = [
        feature for feature in latent_feature_names if feature not in X_train.columns
    ]
    if missing_latent:
        raise ValueError("Missing latent feature(s): " + ", ".join(missing_latent[:10]))
    if len(latent_feature_names) != EXPECTED_LATENT_DIM:
        raise ValueError(
            f"Expected {EXPECTED_LATENT_DIM} latent features, found "
            f"{len(latent_feature_names)}."
        )

    if RECONSTRUCTION_ERROR_FEATURE not in X_train.columns:
        raise ValueError(f"Missing {RECONSTRUCTION_ERROR_FEATURE}.")
    if not np.isfinite(X_train[RECONSTRUCTION_ERROR_FEATURE].to_numpy()).all():
        raise ValueError(f"{RECONSTRUCTION_ERROR_FEATURE} contains non-finite values.")
    if (X_train[RECONSTRUCTION_ERROR_FEATURE] < 0).any():
        raise ValueError(f"{RECONSTRUCTION_ERROR_FEATURE} contains negative values.")


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
) -> tuple[lgb.LGBMClassifier, dict[str, object], int]:
    model_params = build_model_params(y_train)
    model_params["force_col_wise"] = True
    model_params["n_jobs"] = MEMORY_SAFE_N_JOBS
    model = lgb.LGBMClassifier(**model_params)
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
    return model, model_params, best_iteration


def compact_model_matrix(
    X: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Use compact numeric dtypes before LightGBM constructs its Dataset."""
    categorical_set = set(categorical_columns)
    dtype_map = {
        column: ("int32" if column in categorical_set else "float32")
        for column in X.columns
    }
    return X.astype(dtype_map, copy=False)


def save_scores(
    path: Path,
    transaction_ids: np.ndarray,
    y: pd.Series,
    score: np.ndarray,
    X: pd.DataFrame,
) -> None:
    pd.DataFrame(
        {
            ID_COL: transaction_ids,
            TARGET_COL: y.to_numpy(),
            "score": score,
            RECONSTRUCTION_ERROR_FEATURE: X[RECONSTRUCTION_ERROR_FEATURE].to_numpy(),
        }
    ).to_csv(path, index=False)


def save_ae_feature_importance(
    model: lgb.LGBMClassifier,
    ae_features: list[str],
    output_path: Path,
) -> None:
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    )
    importance = importance.loc[importance["feature"].isin(ae_features)]
    importance = importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)
    importance.to_csv(output_path, index=False)


def latent_dim_from_run_config(run_config: dict[str, object]) -> int | None:
    architecture = run_config.get("architecture", {})
    if not isinstance(architecture, dict):
        return None
    encoder = architecture.get("encoder", [])
    if not isinstance(encoder, list) or not encoder:
        return None
    return int(encoder[-1])


def stopping_decision(validation_pr_auc: float) -> dict[str, object]:
    delta_vs_default = float(validation_pr_auc - DEFAULT_FE_VALIDATION_PR_AUC_REFERENCE)
    delta_vs_tuned = float(validation_pr_auc - TUNED_FE_VALIDATION_PR_AUC_REFERENCE)
    return {
        "default_fe_validation_pr_auc_reference": DEFAULT_FE_VALIDATION_PR_AUC_REFERENCE,
        "tuned_fe_validation_pr_auc_reference": TUNED_FE_VALIDATION_PR_AUC_REFERENCE,
        "min_default_fe_delta_for_future_optuna": C_PROMISING_DELTA_FOR_OPTUNA,
        "validation_pr_auc_delta_vs_default_fe": delta_vs_default,
        "validation_pr_auc_delta_vs_tuned_fe": delta_vs_tuned,
        "consider_optuna_later": bool(delta_vs_default >= C_PROMISING_DELTA_FOR_OPTUNA),
        "rule": (
            "Consider Optuna later only if B or C default validation PR-AUC beats "
            "default FE-LGBM by at least 0.005."
        ),
    }


def run_experiment(output_dir: Path, overwrite: bool) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)

    prepared = prepare_engineered_splits()
    feature_artifacts = prepared["feature_artifacts"]
    engineered_features = feature_artifacts["engineered_feature_names"]
    v_columns = get_v_feature_columns(prepared["train_df"])
    y_train = prepared["y_train"]
    y_valid = prepared["y_valid"]
    y_test = prepared["y_test"]
    validation_ids = prepared["valid_df"][ID_COL].to_numpy(copy=True)
    test_ids = prepared["test_df"][ID_COL].to_numpy(copy=True)
    split_row_counts = {
        "train": int(len(prepared["train_df"])),
        "validation": int(len(prepared["valid_df"])),
        "test": int(len(prepared["test_df"])),
    }
    original_feature_count = int(prepared["X_train_raw"].shape[1])
    unknown_rates = prepared["unknown_rates"]

    log("Fitting train-only categorical preprocessing on FE features.")
    preprocessing = fit_baseline_preprocessing(prepared["X_train_engineered"])
    X_train_fe = apply_baseline_preprocessing(
        prepared["X_train_engineered"],
        preprocessing,
    )
    X_valid_fe = apply_baseline_preprocessing(
        prepared["X_valid_engineered"],
        preprocessing,
    )
    X_test_fe = apply_baseline_preprocessing(
        prepared["X_test_engineered"],
        preprocessing,
    )

    log("Loading robust LD128 latent features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        autoencoder_run_config,
    ) = load_robust_latent_outputs(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR)
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(prepared["train_df"]),
        len(prepared["valid_df"]),
        len(prepared["test_df"]),
    )
    if latent_train.shape[1] != EXPECTED_LATENT_DIM:
        raise ValueError(
            f"Expected LD128 latent features, found {latent_train.shape[1]}."
        )

    log("Loading robust LD128 reconstruction errors.")
    reconstruction_errors, reconstruction_error_source_dir = load_reconstruction_errors(
        "robust_ld128"
    )
    validate_reconstruction_error_lengths(
        reconstruction_errors,
        len(prepared["train_df"]),
        len(prepared["valid_df"]),
        len(prepared["test_df"]),
    )

    log("Appending LD128 latent features and raw AE reconstruction error to FE matrices.")
    X_train = combine_fe_latent_and_error(
        X_train_fe,
        latent_train,
        latent_feature_names,
        reconstruction_errors["train"],
    )
    X_valid = combine_fe_latent_and_error(
        X_valid_fe,
        latent_valid,
        latent_feature_names,
        reconstruction_errors["validation"],
    )
    X_test = combine_fe_latent_and_error(
        X_test_fe,
        latent_test,
        latent_feature_names,
        reconstruction_errors["test"],
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        engineered_features,
        v_columns,
        latent_feature_names,
    )

    total_feature_count = int(X_train.shape[1])
    fe_feature_count_after_preprocessing = int(X_train_fe.shape[1])
    log("Compacting FE + AE matrices to memory-safe numeric dtypes.")
    categorical_columns = preprocessing["categorical_columns"]
    X_train = compact_model_matrix(X_train, categorical_columns)
    X_valid = compact_model_matrix(X_valid, categorical_columns)
    X_test = compact_model_matrix(X_test, categorical_columns)

    del X_train_fe, X_valid_fe, X_test_fe
    del latent_train, latent_valid, latent_test
    del reconstruction_errors
    del prepared
    gc.collect()

    log("Training default LightGBM with validation early stopping.")
    model, model_params, best_iteration = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_columns,
    )

    log("Generating validation and test probabilities.")
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]

    log("Selecting classification threshold on validation only.")
    threshold_table = threshold_selection_table(
        y_valid.to_numpy(),
        valid_score,
    )
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
    decision = stopping_decision(float(metrics_valid_selected["average_precision"]))

    log("Saving FE AE-augmented LightGBM outputs.")
    save_json(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_json(
        metrics_test_default,
        output_dir / "metrics_test_default_threshold.json",
    )
    save_json(
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
    save_engineered_feature_importance(
        model,
        engineered_features,
        output_dir / "engineered_feature_importance.csv",
    )
    save_ae_feature_importance(
        model,
        [*latent_feature_names, RECONSTRUCTION_ERROR_FEATURE],
        output_dir / "ae_feature_importance.csv",
    )
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")
    joblib.dump(feature_artifacts, output_dir / "feature_engineering.pkl")

    save_scores(
        output_dir / "scores_validation.csv",
        validation_ids,
        y_valid,
        valid_score,
        X_valid,
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_ids,
        y_test,
        test_score,
        X_test,
    )

    feature_summary = feature_engineering_summary(feature_artifacts)
    feature_set_summary = {
        "experiment_type": "fe_lgbm_plus_ld128_latent_and_ae_reconstruction_mse_default",
        "original_feature_count": original_feature_count,
        "engineered_feature_count": int(feature_summary["engineered_feature_count"]),
        "fe_feature_count_after_preprocessing": fe_feature_count_after_preprocessing,
        "ae_latent_features_included": True,
        "ae_latent_feature_count": int(len(latent_feature_names)),
        "ae_latent_output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
        "reconstruction_error_included": True,
        "reconstruction_error_feature": RECONSTRUCTION_ERROR_FEATURE,
        "reconstruction_error_source_dir": str(reconstruction_error_source_dir),
        "total_feature_count": total_feature_count,
        "original_features_retained": True,
        "original_v_features_retained": True,
        "original_v_feature_count": int(len(v_columns)),
        "engineered_features": feature_summary["engineered_feature_names"],
        "latent_features": latent_feature_names,
        "feature_engineering": feature_summary,
    }
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")

    source_reconstruction_metrics_path = (
        reconstruction_error_source_dir / "reconstruction_metrics.json"
    )
    source_reconstruction_metrics = (
        load_json(source_reconstruction_metrics_path)
        if source_reconstruction_metrics_path.exists()
        else None
    )

    run_config = {
        "phase": "fe_ae_controlled_C_fe_latent128_reconstruction_error_default",
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
        "split_row_counts": {
            "train": split_row_counts["train"],
            "validation": split_row_counts["validation"],
            "test": split_row_counts["test"],
        },
        "leakage_prevention": {
            "split": "Existing chronological 60/20/20 labeled-train split.",
            "feature_engineering_fit": "FE mappings fit on train only.",
            "feature_engineering_apply": "Validation/test transformed with train-fitted FE mappings.",
            "autoencoder_features": (
                "Loaded saved robust LD128 AE latent arrays and reconstruction "
                "errors; no AE fitting is performed here."
            ),
            "categorical_preprocessing_fit": "Categorical mappings fit on train FE features only.",
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split used once after model and threshold selection.",
            "kaggle_competition_test_files_used": False,
            "optuna_used": False,
        },
        "feature_construction": feature_set_summary,
        "unknown_rate_summary": unknown_rates,
        "preprocessing": {
            "categorical_fit": "Categorical mappings fit on engineered train features only.",
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
        "model_features_count": total_feature_count,
        "source_autoencoder": {
            "output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
            "latent_dim": latent_dim_from_run_config(autoencoder_run_config),
            "run_config": autoencoder_run_config,
            "reconstruction_metrics": source_reconstruction_metrics,
        },
        "stopping_criteria": decision,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("FE + LD128 AE-Augmented LightGBM Summary")
    print("========================================")
    print(f"Validation PR-AUC       : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Delta vs default FE     : {decision['validation_pr_auc_delta_vs_default_fe']:+.6f}")
    print(f"Delta vs tuned FE       : {decision['validation_pr_auc_delta_vs_tuned_fe']:+.6f}")
    print(f"Test PR-AUC             : {metrics_test_selected['average_precision']:.6f}")
    print(f"Test ROC-AUC            : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold      : {selected_threshold:.2f}")
    print(f"Test F1                 : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC                : {metrics_test_selected['mcc']:.6f}")
    print(f"Best iteration          : {best_iteration}")
    print(f"Total features          : {X_train.shape[1]}")
    print(f"Consider Optuna later   : {decision['consider_optuna_later']}")
    print(f"Outputs saved to        : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
        "stopping_criteria": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train default FE-LGBM with robust LD128 AE latent features and raw "
            "reconstruction MSE."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FE_AE_AUGMENTED_LGBM_OUTPUT_DIR,
        help="Output directory for Experiment C.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a non-empty Experiment C output directory.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return run_experiment(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
