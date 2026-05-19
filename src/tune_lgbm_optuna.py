"""Phase 5 Optuna/TPE tuning for the main LightGBM models.

This script tunes only:
- baseline_lgbm
- baseline_lgbm_entity_time_amount_features
- ae_lgbm_ld128
- ae_augmented_lgbm_ld128

Leakage prevention is intentionally strict:
- train split: preprocessing fit and LightGBM fitting
- validation split: early stopping, Optuna objective, threshold selection
- test split: final evaluation only after the study has selected parameters
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import optuna
    from optuna.trial import TrialState
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Optuna is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

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
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    DATA_DIR,
    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
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
from feature_engineering import (
    apply_entity_time_amount_features,
    feature_engineering_summary,
    fit_entity_time_amount_features,
    unknown_rate_summary,
    validate_engineered_features,
)
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from splitting import chronological_split
from train_ae_augmented_lgbm import (
    RECONSTRUCTION_ERROR_FEATURE,
    combine_original_latent_and_error,
    load_or_compute_reconstruction_errors,
    validate_augmented_feature_alignment,
)
from train_ae_lgbm import (
    apply_non_v_preprocessing,
    combine_non_v_and_latent,
    fit_non_v_preprocessing,
    load_robust_latent_outputs,
    split_non_v_features_target,
    validate_feature_alignment,
    validate_latent_outputs,
)
from train_baseline_lgbm import (
    average_precision_eval,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_THRESHOLD = 0.5
EARLY_STOPPING_ROUNDS = 100
EXPECTED_LD128_LATENT_DIM = 128
DEFAULT_N_JOBS = 4
FEATURE_ENGINEERED_MODEL_TYPE = "baseline_lgbm_entity_time_amount_features"

SUPPORTED_MODEL_TYPES = (
    "baseline_lgbm",
    FEATURE_ENGINEERED_MODEL_TYPE,
    "ae_lgbm_ld128",
    "ae_augmented_lgbm_ld128",
)
SUPPORTED_TUNING_PROFILES = ("quick", "final")

TUNING_PROFILES = {
    "quick": {
        "num_leaves": {"kind": "int", "low": 31, "high": 128},
        "max_depth": {"kind": "categorical", "choices": [-1, 4, 6, 8, 10]},
        "learning_rate": {"kind": "float", "low": 0.02, "high": 0.08, "log": True},
        "n_estimators": {"kind": "int", "low": 500, "high": 1200},
        "min_child_samples": {"kind": "int", "low": 30, "high": 200},
        "subsample": {"kind": "float", "low": 0.7, "high": 1.0},
        "subsample_freq": {"kind": "int", "low": 1, "high": 5},
        "colsample_bytree": {"kind": "float", "low": 0.7, "high": 1.0},
        "reg_alpha": {"kind": "float", "low": 0.0, "high": 5.0},
        "reg_lambda": {"kind": "float", "low": 0.0, "high": 10.0},
        "scale_pos_weight": {"kind": "float", "low": 1.0, "high": 35.0},
    },
    "final": {
        "num_leaves": {"kind": "int", "low": 16, "high": 256},
        "max_depth": {
            "kind": "categorical",
            "choices": [-1, 3, 4, 5, 6, 7, 8, 10, 12],
        },
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.10, "log": True},
        "n_estimators": {"kind": "int", "low": 700, "high": 2000},
        "min_child_samples": {"kind": "int", "low": 20, "high": 300},
        "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
        "subsample_freq": {"kind": "int", "low": 1, "high": 10},
        "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"kind": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"kind": "float", "low": 0.0, "high": 10.0},
        "scale_pos_weight": {"kind": "float", "low": 1.0, "high": 50.0},
    },
}

TUNED_OUTPUT_DIRS = {
    "baseline_lgbm": OPTUNA_OUTPUT_DIR / "baseline_lgbm",
    FEATURE_ENGINEERED_MODEL_TYPE: (
        OPTUNA_OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features"
    ),
    "ae_lgbm_ld128": OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128",
    "ae_augmented_lgbm_ld128": OPTUNA_OUTPUT_DIR / "ae_augmented_lgbm_ld128",
}

COMPARISON_FILE = FINAL_COMPARISON_OUTPUT_DIR / "optuna_comparison.csv"

REQUIRED_OUTPUT_FILES = [
    "optuna_study.pkl",
    "trials.csv",
    "best_params.json",
    "metrics_validation_default_threshold.json",
    "metrics_validation_selected_threshold.json",
    "metrics_test_default_threshold.json",
    "metrics_test_selected_threshold.json",
    "confusion_matrix_validation.csv",
    "confusion_matrix_test.csv",
    "threshold_selection.csv",
    "feature_importance.csv",
    "final_model.pkl",
    "final_model.txt",
    "run_config.json",
]

COMPARISON_COLUMNS = [
    "model_name",
    "tuned",
    "test_pr_auc",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "selected_threshold",
    "best_iteration",
    "n_trials",
    "total_features",
]


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    categorical_columns: list[str]
    preprocessing: dict[str, object]
    preprocessing_filename: str
    feature_info: dict[str, object]
    extra_artifacts: dict[str, object] | None = None

    @property
    def total_features(self) -> int:
        return int(self.X_train.shape[1])


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def output_complete(output_dir: Path, model_type: str) -> bool:
    required_files = list(REQUIRED_OUTPUT_FILES)
    preprocessing_file = (
        "preprocessing_non_v.pkl"
        if model_type == "ae_lgbm_ld128"
        else "preprocessing.pkl"
    )
    required_files.append(preprocessing_file)
    if model_type == FEATURE_ENGINEERED_MODEL_TYPE:
        required_files.append("feature_engineering.pkl")
    if not all((output_dir / file_name).exists() for file_name in required_files):
        return False

    run_config = load_json(output_dir / "run_config.json")
    return bool(run_config.get("final_training_completed", True))


def prepare_baseline_data() -> PreparedData:
    """Build Phase 2 baseline matrices using train-only preprocessing."""
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Building baseline feature matrices.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    # Leakage prevention: categorical mappings are fit only on train.
    # Numeric NaNs are left in place for LightGBM's native missing handling.
    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test = apply_baseline_preprocessing(X_test_raw, preprocessing)

    feature_info = {
        "feature_setup": "Phase 2 baseline original features.",
        "original_v_features_kept": True,
        "model_features_count": int(X_train.shape[1]),
        "categorical_columns": preprocessing["categorical_columns"],
        "categorical_columns_count": len(preprocessing["categorical_columns"]),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }

    return PreparedData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=preprocessing["categorical_columns"],
        preprocessing=preprocessing,
        preprocessing_filename="preprocessing.pkl",
        feature_info=feature_info,
    )


def validate_feature_engineered_final_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    engineered_features: list[str],
) -> None:
    """Validate final model matrices for the feature-engineered branch."""
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError(
            "Feature-engineered validation columns do not align with train."
        )
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Feature-engineered test columns do not align with train.")

    missing_engineered = [
        feature for feature in engineered_features if feature not in X_train.columns
    ]
    if missing_engineered:
        raise ValueError(
            "Feature-engineered model matrix is missing engineered feature(s): "
            + ", ".join(missing_engineered[:20])
        )

    retained_internal_keys = [
        column
        for column in X_train.columns
        if str(column).startswith("__fe_key_") or str(column).startswith("uid_")
    ]
    if retained_internal_keys:
        raise ValueError(
            "Internal UID/combo key columns leaked into feature-engineered model: "
            + ", ".join(retained_internal_keys)
        )


def prepare_feature_engineered_lgbm_data() -> PreparedData:
    """Build leakage-safe entity/time/amount feature-engineered matrices."""
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Separating target before feature engineering.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    log("Fitting entity/time/amount feature artifacts on train only.")
    feature_artifacts = fit_entity_time_amount_features(X_train_raw)

    log("Applying train-fitted feature artifacts to all splits.")
    X_train_engineered = apply_entity_time_amount_features(
        X_train_raw,
        feature_artifacts,
    )
    X_valid_engineered = apply_entity_time_amount_features(
        X_valid_raw,
        feature_artifacts,
    )
    X_test_engineered = apply_entity_time_amount_features(
        X_test_raw,
        feature_artifacts,
    )
    validate_engineered_features(
        X_train_engineered,
        X_valid_engineered,
        X_test_engineered,
        feature_artifacts,
    )

    log("Computing unknown-rate summaries against train-fitted mappings.")
    unknown_rates = {
        "train": unknown_rate_summary(X_train_raw, feature_artifacts),
        "validation": unknown_rate_summary(X_valid_raw, feature_artifacts),
        "test": unknown_rate_summary(X_test_raw, feature_artifacts),
    }

    log("Fitting train-only categorical preprocessing on engineered features.")
    preprocessing = fit_baseline_preprocessing(X_train_engineered)
    X_train = apply_baseline_preprocessing(X_train_engineered, preprocessing)
    X_valid = apply_baseline_preprocessing(X_valid_engineered, preprocessing)
    X_test = apply_baseline_preprocessing(X_test_engineered, preprocessing)
    validate_feature_engineered_final_alignment(
        X_train,
        X_valid,
        X_test,
        feature_artifacts["engineered_feature_names"],
    )

    feature_summary = feature_engineering_summary(feature_artifacts)
    feature_info = {
        "feature_setup": "Leakage-safe entity/time/amount LightGBM features.",
        "experiment_type": FEATURE_ENGINEERED_MODEL_TYPE,
        "original_features_retained": True,
        "original_feature_count": int(X_train_raw.shape[1]),
        "engineered_feature_count": feature_summary["engineered_feature_count"],
        "total_feature_count": int(X_train.shape[1]),
        "engineered_features": feature_summary["engineered_feature_names"],
        "feature_engineering": feature_summary,
        "unknown_rate_summary": unknown_rates,
        "leakage_prevention": {
            "feature_engineering_fit": (
                "Count, frequency, and amount-stat mappings are fit on train only."
            ),
            "feature_engineering_apply": (
                "Validation/test are transformed only with train-fitted mappings."
            ),
            "target_encoding_used": False,
            "fraud_labels_used_for_features": False,
        },
        "categorical_columns": preprocessing["categorical_columns"],
        "categorical_columns_count": len(preprocessing["categorical_columns"]),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }

    return PreparedData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=preprocessing["categorical_columns"],
        preprocessing=preprocessing,
        preprocessing_filename="preprocessing.pkl",
        feature_info=feature_info,
        extra_artifacts={"feature_engineering.pkl": feature_artifacts},
    )


def prepare_ae_lgbm_ld128_data() -> PreparedData:
    """Build Phase 4B ld128 AE-LightGBM matrices."""
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)

    log("Loading robust Autoencoder latent_dim=128 features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        robust_ae_run_config,
    ) = load_robust_latent_outputs(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR)

    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    if latent_train.shape[1] != EXPECTED_LD128_LATENT_DIM:
        raise ValueError(
            "Expected latent_dim=128 for ae_lgbm_ld128, but found "
            f"{latent_train.shape[1]} columns."
        )

    log("Building non-V feature matrices.")
    X_train_non_v_raw, y_train = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, y_valid = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test = split_non_v_features_target(test_df, v_columns)

    # Leakage prevention: non-V categorical mappings are fit only on train.
    # Original V1-V339 columns are excluded and replaced by train-aligned AE latents.
    preprocessing_non_v = fit_non_v_preprocessing(X_train_non_v_raw, v_columns)
    X_train_non_v = apply_non_v_preprocessing(X_train_non_v_raw, preprocessing_non_v)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing_non_v)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing_non_v)

    log("Combining non-V features with robust latent V features.")
    X_train = combine_non_v_and_latent(X_train_non_v, latent_train, latent_feature_names)
    X_valid = combine_non_v_and_latent(X_valid_non_v, latent_valid, latent_feature_names)
    X_test = combine_non_v_and_latent(X_test_non_v, latent_test, latent_feature_names)
    validate_feature_alignment(X_train, X_valid, X_test, v_columns)

    robust_preprocessing = robust_ae_run_config.get("preprocessing", {})
    feature_info = {
        "feature_setup": "Phase 4B AE-LightGBM latent_dim=128.",
        "original_v_features_excluded": True,
        "original_v_feature_count": len(v_columns),
        "non_v_feature_count": int(X_train_non_v.shape[1]),
        "latent_feature_count": len(latent_feature_names),
        "total_feature_count": int(X_train.shape[1]),
        "robust_autoencoder_output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
        "robust_autoencoder_clipping": {
            "enabled": robust_preprocessing.get("scaled_clipping_enabled"),
            "clip_min": robust_preprocessing.get("clip_min"),
            "clip_max": robust_preprocessing.get("clip_max"),
        },
        "categorical_columns": preprocessing_non_v["categorical_columns"],
        "categorical_columns_count": len(preprocessing_non_v["categorical_columns"]),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }

    return PreparedData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=preprocessing_non_v["categorical_columns"],
        preprocessing=preprocessing_non_v,
        preprocessing_filename="preprocessing_non_v.pkl",
        feature_info=feature_info,
    )


def prepare_ae_augmented_lgbm_ld128_data() -> PreparedData:
    """Build AE-augmented LD128 matrices with original V-features retained."""
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)

    log("Building original baseline feature matrices.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    # Leakage prevention: baseline categorical mappings are fit only on train.
    # Numeric NaNs, including original V-feature NaNs, stay available to LightGBM.
    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train_original = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid_original = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test_original = apply_baseline_preprocessing(X_test_raw, preprocessing)

    log("Loading robust Autoencoder latent_dim=128 features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        robust_ae_run_config,
    ) = load_robust_latent_outputs(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR)

    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    if latent_train.shape[1] != EXPECTED_LD128_LATENT_DIM:
        raise ValueError(
            "Expected latent_dim=128 for ae_augmented_lgbm_ld128, but found "
            f"{latent_train.shape[1]} columns."
        )

    log("Loading or computing robust Autoencoder reconstruction errors.")
    reconstruction_errors, reconstruction_error_source = load_or_compute_reconstruction_errors(
        AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
        robust_ae_run_config,
        train_df,
        valid_df,
        test_df,
    )

    log("Combining original features with robust AE latent features and error.")
    # AE-Augmented is an augmentation experiment: original V1-V339 features
    # are retained, while LD128 latent features and reconstruction MSE are added.
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

    robust_preprocessing = robust_ae_run_config.get("preprocessing", {})
    feature_info = {
        "feature_setup": "AE-Augmented LightGBM latent_dim=128.",
        "experiment_type": "augmentation_not_replacement",
        "original_features_retained": True,
        "original_v_features_retained": True,
        "original_feature_count": int(X_train_original.shape[1]),
        "original_v_feature_count": len(v_columns),
        "ae_latent_feature_count": len(latent_feature_names),
        "reconstruction_error_included": True,
        "reconstruction_error_feature": RECONSTRUCTION_ERROR_FEATURE,
        "reconstruction_error_source": reconstruction_error_source,
        "total_feature_count": int(X_train.shape[1]),
        "robust_autoencoder_output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
        "robust_autoencoder_clipping": {
            "enabled": robust_preprocessing.get("scaled_clipping_enabled"),
            "clip_min": robust_preprocessing.get("clip_min"),
            "clip_max": robust_preprocessing.get("clip_max"),
        },
        "categorical_columns": preprocessing["categorical_columns"],
        "categorical_columns_count": len(preprocessing["categorical_columns"]),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }

    return PreparedData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=preprocessing["categorical_columns"],
        preprocessing=preprocessing,
        preprocessing_filename="preprocessing.pkl",
        feature_info=feature_info,
    )


def prepare_data(model_type: str) -> PreparedData:
    if model_type == "baseline_lgbm":
        return prepare_baseline_data()
    if model_type == FEATURE_ENGINEERED_MODEL_TYPE:
        return prepare_feature_engineered_lgbm_data()
    if model_type == "ae_lgbm_ld128":
        return prepare_ae_lgbm_ld128_data()
    if model_type == "ae_augmented_lgbm_ld128":
        return prepare_ae_augmented_lgbm_ld128_data()
    raise ValueError(f"Unsupported model_type: {model_type}")


def fixed_lgbm_params(
    n_jobs: int,
    device_type: str = "cpu",
    max_bin: int | None = None,
) -> dict[str, object]:
    """LightGBM parameters that are fixed for every trial."""
    params: dict[str, object] = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": n_jobs,
        "random_state": RANDOM_SEED,
        "metric": "None",
        "verbosity": -1,
    }
    if device_type != "cpu":
        params["device_type"] = device_type
    if max_bin is not None:
        params["max_bin"] = max_bin
    return params


def get_tuning_profile_space(tuning_profile: str) -> dict[str, dict[str, object]]:
    if tuning_profile not in TUNING_PROFILES:
        raise ValueError(f"Unsupported tuning_profile: {tuning_profile}")
    return TUNING_PROFILES[tuning_profile]


def _suggest_from_spec(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, object],
) -> object:
    kind = spec["kind"]
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported Optuna parameter kind for {name}: {kind}")


def suggest_lgbm_params(
    trial: optuna.Trial,
    tuning_profile: str,
) -> dict[str, object]:
    """Optuna search space using sklearn LightGBM parameter names."""
    space = get_tuning_profile_space(tuning_profile)
    return {
        name: _suggest_from_spec(trial, name, spec)
        for name, spec in space.items()
    }


def search_space_summary(tuning_profile: str) -> dict[str, object]:
    return get_tuning_profile_space(tuning_profile)


def fit_lgbm(
    prepared: PreparedData,
    params: dict[str, object],
    log_period: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        prepared.X_train,
        prepared.y_train,
        eval_set=[(prepared.X_valid, prepared.y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=prepared.categorical_columns,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=log_period),
        ],
    )
    return model


def best_iteration_for(model: lgb.LGBMClassifier, params: dict[str, object]) -> int:
    return int(model.best_iteration_ or params["n_estimators"])


def validation_average_precision(
    model: lgb.LGBMClassifier,
    prepared: PreparedData,
    params: dict[str, object],
) -> tuple[float, int]:
    best_iteration = best_iteration_for(model, params)
    valid_score = model.predict_proba(
        prepared.X_valid,
        num_iteration=best_iteration,
    )[:, 1]
    score = average_precision_score(prepared.y_valid.to_numpy(), valid_score)
    return float(score), best_iteration


def make_objective(
    prepared: PreparedData,
    tuning_profile: str,
    n_jobs: int,
    device_type: str,
    max_bin: int | None,
):
    def objective(trial: optuna.Trial) -> float:
        params = (
            fixed_lgbm_params(n_jobs, device_type, max_bin)
            | suggest_lgbm_params(
                trial,
                tuning_profile,
            )
        )
        model = fit_lgbm(prepared, params, log_period=0)
        score, best_iteration = validation_average_precision(model, prepared, params)
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("validation_average_precision", score)
        return score

    return objective


def create_or_load_study(args: argparse.Namespace) -> optuna.Study:
    ensure_sqlite_storage_parent_dir(args.storage)
    study_name = args.study_name or f"phase5_{args.model_type}"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    return optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
    )


def ensure_sqlite_storage_parent_dir(storage: str | None) -> None:
    """Create the parent directory for simple sqlite:///path.db storage URLs."""
    if not storage or not storage.startswith("sqlite:///"):
        return

    db_path = storage.removeprefix("sqlite:///")
    if not db_path or db_path == ":memory:":
        return

    ensure_dir(Path(db_path).parent)


def completed_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)


def save_study_outputs(study: optuna.Study, output_dir: Path) -> None:
    joblib.dump(study, output_dir / "optuna_study.pkl")
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)


def train_final_model(
    prepared: PreparedData,
    best_params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    log("Training final model with the best validation PR-AUC parameters.")
    model = fit_lgbm(prepared, best_params, log_period=50)
    best_iteration = best_iteration_for(model, best_params)

    log("Generating validation and test probabilities.")
    valid_score = model.predict_proba(
        prepared.X_valid,
        num_iteration=best_iteration,
    )[:, 1]
    test_score = model.predict_proba(
        prepared.X_test,
        num_iteration=best_iteration,
    )[:, 1]

    log("Selecting classification threshold on validation only.")
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

    log("Saving final tuned model outputs.")
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

    save_feature_importance(model, output_dir / "feature_importance.csv")
    joblib.dump(model, output_dir / "final_model.pkl")
    model.booster_.save_model(str(output_dir / "final_model.txt"))
    joblib.dump(prepared.preprocessing, output_dir / prepared.preprocessing_filename)
    if prepared.extra_artifacts:
        for filename, artifact in prepared.extra_artifacts.items():
            joblib.dump(artifact, output_dir / filename)

    return {
        "model": model,
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "metrics_validation_default": metrics_valid_default,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_default": metrics_test_default,
        "metrics_test_selected": metrics_test_selected,
    }


def leakage_prevention_summary(model_type: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "train": "Preprocessing fit and LightGBM model fitting.",
        "validation": "Early stopping, Optuna objective, and threshold selection.",
        "test": "Final evaluation only after best hyperparameters and threshold are selected.",
        "kaggle_competition_test_files_used": False,
    }
    if model_type == FEATURE_ENGINEERED_MODEL_TYPE:
        summary["feature_engineering"] = (
            "Entity/time/amount mappings are fit on train only and applied to "
            "validation/test."
        )
    return summary


def build_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    prepared: PreparedData,
    study: optuna.Study,
    best_params: dict[str, object],
    final_result: dict[str, object] | None,
) -> dict[str, object]:
    final_training_completed = final_result is not None
    selected_threshold = (
        final_result["selected_threshold"] if final_training_completed else None
    )
    final_best_iteration = (
        final_result["best_iteration"] if final_training_completed else None
    )

    return {
        "phase": "5_optuna_tpe_lgbm",
        "model_type": args.model_type,
        "tuning_profile": args.tuning_profile,
        "n_jobs": args.n_jobs,
        "device_type": args.device_type,
        "max_bin": args.max_bin,
        "skip_final_training": args.skip_final_training,
        "skip_global_comparison_update": args.skip_global_comparison_update,
        "final_training_completed": final_training_completed,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "output_dir_override": str(args.output_dir) if args.output_dir else None,
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
        "leakage_prevention": leakage_prevention_summary(args.model_type),
        "feature_construction": prepared.feature_info,
        "model_features_count": prepared.total_features,
        "preprocessing_file": prepared.preprocessing_filename,
        "optuna": {
            "sampler": "TPESampler",
            "sampler_seed": RANDOM_SEED,
            "direction": "maximize",
            "objective": "validation average_precision / PR-AUC",
            "study_name": study.study_name,
            "storage": args.storage,
            "n_trials_requested_this_run": args.n_trials,
            "n_trials_total": len(study.trials),
            "n_trials_completed": completed_trial_count(study),
            "timeout_seconds": args.timeout,
            "pruning_used": False,
            "tuning_profile": args.tuning_profile,
            "search_space": search_space_summary(args.tuning_profile),
            "best_trial_validation_best_iteration": (
                study.best_trial.user_attrs.get("best_iteration")
            ),
        },
        "final_training": {
            "completed": final_training_completed,
            "skipped_by_flag": args.skip_final_training,
            "note": (
                "Final model training, threshold selection, and test evaluation completed."
                if final_training_completed
                else (
                    "Skipped by --skip_final_training; test set was not evaluated "
                    "and optuna_comparison.csv was not updated."
                )
            ),
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "completed": final_training_completed,
            "selected_threshold": selected_threshold,
        },
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": final_best_iteration,
        },
        "model_params": best_params,
    }


def save_best_params(
    study: optuna.Study,
    best_params: dict[str, object],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_type": args.model_type,
        "tuning_profile": args.tuning_profile,
        "n_jobs": args.n_jobs,
        "device_type": args.device_type,
        "max_bin": args.max_bin,
        "skip_final_training": args.skip_final_training,
        "best_trial_number": study.best_trial.number,
        "best_validation_average_precision": float(study.best_value),
        "best_trial_user_attrs": study.best_trial.user_attrs,
        "best_params": study.best_params,
        "fixed_params": fixed_lgbm_params(
            args.n_jobs,
            args.device_type,
            args.max_bin,
        ),
        "search_space": search_space_summary(args.tuning_profile),
        "final_model_params": best_params,
    }
    save_json(payload, output_dir / "best_params.json")


def metric_value(metrics: dict[str, object], key: str) -> object:
    value = metrics.get(key)
    return value


def total_features_from_run_config(run_config: dict[str, object]) -> object:
    if "model_features_count" in run_config:
        return run_config["model_features_count"]
    feature_construction = run_config.get("feature_construction", {})
    if isinstance(feature_construction, dict):
        for key in ("total_feature_count", "total_final_features"):
            if key in feature_construction:
                return feature_construction[key]
    return None


def n_trials_from_run_config(run_config: dict[str, object], tuned: bool) -> int:
    if not tuned:
        return 0
    optuna_config = run_config.get("optuna", {})
    if isinstance(optuna_config, dict):
        return int(optuna_config.get("n_trials_completed", 0))
    return 0


def comparison_row(
    model_name: str,
    tuned: bool,
    metrics_path: Path,
    run_config_path: Path,
) -> dict[str, object] | None:
    if not metrics_path.exists() or not run_config_path.exists():
        return None

    run_config = load_json(run_config_path)
    if tuned and run_config.get("final_training_completed") is False:
        return None

    metrics = load_json(metrics_path)
    return {
        "model_name": model_name,
        "tuned": tuned,
        "test_pr_auc": metric_value(metrics, "average_precision"),
        "test_roc_auc": metric_value(metrics, "roc_auc"),
        "test_precision": metric_value(metrics, "precision"),
        "test_recall": metric_value(metrics, "recall"),
        "test_f1": metric_value(metrics, "f1"),
        "test_mcc": metric_value(metrics, "mcc"),
        "selected_threshold": metric_value(metrics, "threshold"),
        "best_iteration": run_config.get("early_stopping", {}).get("best_iteration"),
        "n_trials": n_trials_from_run_config(run_config, tuned),
        "total_features": total_features_from_run_config(run_config),
    }


def build_optuna_comparison_table() -> pd.DataFrame:
    candidates = [
        (
            "baseline_lgbm_default",
            False,
            BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json",
            BASELINE_OUTPUT_DIR / "run_config.json",
        ),
        (
            "baseline_lgbm_tuned",
            True,
            TUNED_OUTPUT_DIRS["baseline_lgbm"] / "metrics_test_selected_threshold.json",
            TUNED_OUTPUT_DIRS["baseline_lgbm"] / "run_config.json",
        ),
        (
            "baseline_lgbm_entity_time_amount_features_default",
            False,
            FEATURE_ENGINEERED_LGBM_OUTPUT_DIR
            / "metrics_test_selected_threshold.json",
            FEATURE_ENGINEERED_LGBM_OUTPUT_DIR / "run_config.json",
        ),
        (
            "baseline_lgbm_entity_time_amount_features_tuned",
            True,
            TUNED_OUTPUT_DIRS[FEATURE_ENGINEERED_MODEL_TYPE]
            / "metrics_test_selected_threshold.json",
            TUNED_OUTPUT_DIRS[FEATURE_ENGINEERED_MODEL_TYPE] / "run_config.json",
        ),
        (
            "ae_lgbm_ld128_default",
            False,
            AE_LGBM_LD128_OUTPUT_DIR / "metrics_test_selected_threshold.json",
            AE_LGBM_LD128_OUTPUT_DIR / "run_config.json",
        ),
        (
            "ae_lgbm_ld128_tuned",
            True,
            TUNED_OUTPUT_DIRS["ae_lgbm_ld128"] / "metrics_test_selected_threshold.json",
            TUNED_OUTPUT_DIRS["ae_lgbm_ld128"] / "run_config.json",
        ),
        (
            "ae_augmented_lgbm_ld128_default",
            False,
            AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR
            / "metrics_test_selected_threshold.json",
            AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR / "run_config.json",
        ),
        (
            "ae_augmented_lgbm_ld128_tuned",
            True,
            TUNED_OUTPUT_DIRS["ae_augmented_lgbm_ld128"]
            / "metrics_test_selected_threshold.json",
            TUNED_OUTPUT_DIRS["ae_augmented_lgbm_ld128"] / "run_config.json",
        ),
    ]
    rows: list[dict[str, object]] = []
    for model_name, tuned, metrics_path, run_config_path in candidates:
        row = comparison_row(model_name, tuned, metrics_path, run_config_path)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def save_optuna_comparison_table() -> pd.DataFrame:
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    table = build_optuna_comparison_table()
    table.to_csv(COMPARISON_FILE, index=False)
    return table


def tuned_summary(model_type: str) -> dict[str, object] | None:
    output_dir = TUNED_OUTPUT_DIRS[model_type]
    best_params_path = output_dir / "best_params.json"
    metrics_path = output_dir / "metrics_test_selected_threshold.json"
    run_config_path = output_dir / "run_config.json"
    if not best_params_path.exists() or not metrics_path.exists() or not run_config_path.exists():
        return None

    run_config = load_json(run_config_path)
    if run_config.get("final_training_completed") is False:
        return None

    best_params = load_json(best_params_path)
    metrics = load_json(metrics_path)
    return {
        "model_type": model_type,
        "best_validation_average_precision": best_params["best_validation_average_precision"],
        "test_metrics": metrics,
        "best_iteration": run_config["early_stopping"]["best_iteration"],
        "n_trials": n_trials_from_run_config(run_config, tuned=True),
    }


def print_available_tuned_summaries() -> None:
    print()
    print("Available Tuned Model Summaries")
    print("================================")
    for model_type in SUPPORTED_MODEL_TYPES:
        summary = tuned_summary(model_type)
        if summary is None:
            print(f"{model_type}: tuned outputs not available yet.")
            continue

        metrics = summary["test_metrics"]
        print(
            f"{model_type}: "
            f"best validation PR-AUC={summary['best_validation_average_precision']:.6f}, "
            f"test PR-AUC={metrics['average_precision']:.6f}, "
            f"test ROC-AUC={metrics['roc_auc']:.6f}, "
            f"precision={metrics['precision']:.6f}, "
            f"recall={metrics['recall']:.6f}, "
            f"F1={metrics['f1']:.6f}, "
            f"MCC={metrics['mcc']:.6f}, "
            f"best_iteration={summary['best_iteration']}, "
            f"n_trials={summary['n_trials']}"
        )


def _row_by_name(table: pd.DataFrame, model_name: str) -> pd.Series | None:
    rows = table.loc[table["model_name"] == model_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def print_deltas(table: pd.DataFrame) -> None:
    print()
    print("Delta Tuned vs Default")
    print("======================")
    pairs = [
        ("baseline_lgbm", "baseline_lgbm_default", "baseline_lgbm_tuned"),
        (
            "baseline_lgbm_entity_time_amount_features",
            "baseline_lgbm_entity_time_amount_features_default",
            "baseline_lgbm_entity_time_amount_features_tuned",
        ),
        ("ae_lgbm_ld128", "ae_lgbm_ld128_default", "ae_lgbm_ld128_tuned"),
        (
            "ae_augmented_lgbm_ld128",
            "ae_augmented_lgbm_ld128_default",
            "ae_augmented_lgbm_ld128_tuned",
        ),
    ]
    for label, default_name, tuned_name in pairs:
        default_row = _row_by_name(table, default_name)
        tuned_row = _row_by_name(table, tuned_name)
        if (
            default_row is None
            or tuned_row is None
            or pd.isna(default_row["test_pr_auc"])
            or pd.isna(tuned_row["test_pr_auc"])
        ):
            print(f"{label}: tuned result not available yet.")
            continue

        print(
            f"{label}: "
            f"Delta PR-AUC={tuned_row['test_pr_auc'] - default_row['test_pr_auc']:+.6f}, "
            f"Delta ROC-AUC={tuned_row['test_roc_auc'] - default_row['test_roc_auc']:+.6f}, "
            f"Delta F1={tuned_row['test_f1'] - default_row['test_f1']:+.6f}, "
            f"Delta MCC={tuned_row['test_mcc'] - default_row['test_mcc']:+.6f}"
        )


def print_pr_auc_ranking(table: pd.DataFrame) -> None:
    ranked = (
        table.dropna(subset=["test_pr_auc"])
        .sort_values("test_pr_auc", ascending=False)
        .reset_index(drop=True)
    )
    print()
    print("Ranking by Test PR-AUC")
    print("======================")
    if ranked.empty:
        print("No comparison rows with test PR-AUC are available yet.")
        return
    for index, row in ranked.iterrows():
        print(
            f"{index + 1}. {row['model_name']} "
            f"PR-AUC={row['test_pr_auc']:.6f}, "
            f"ROC-AUC={row['test_roc_auc']:.6f}, "
            f"F1={row['test_f1']:.6f}, "
            f"MCC={row['test_mcc']:.6f}"
        )


def print_final_summary(table: pd.DataFrame) -> None:
    print_available_tuned_summaries()
    print_deltas(table)
    print_pr_auc_ranking(table)
    print(f"\nSaved comparison to: {COMPARISON_FILE}")


def print_study_only_summary(
    args: argparse.Namespace,
    study: optuna.Study,
    output_dir: Path,
) -> None:
    print()
    print("Optuna Study-Only Summary")
    print("=========================")
    print(f"Model type             : {args.model_type}")
    print(f"Tuning profile         : {args.tuning_profile}")
    print(f"Completed trials       : {completed_trial_count(study)}")
    print(f"Best validation PR-AUC : {study.best_value:.6f}")
    print(f"Best trial             : {study.best_trial.number}")
    print("Final training skipped : True")
    print("Test evaluation skipped: True")
    print(f"Outputs saved to       : {output_dir}")


def resolved_tuned_output_dir(args: argparse.Namespace) -> Path:
    return args.output_dir or TUNED_OUTPUT_DIRS[args.model_type]


def run_tuning(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(resolved_tuned_output_dir(args))

    if args.skip_existing and output_complete(output_dir, args.model_type):
        log(f"Skipping {args.model_type}; complete Optuna outputs already exist.")
        if args.skip_global_comparison_update:
            log("Skipping global Optuna comparison update by request.")
            return
        table = save_optuna_comparison_table()
        print_final_summary(table)
        return

    prepared = prepare_data(args.model_type)

    log("Creating/loading Optuna study.")
    study = create_or_load_study(args)

    log(
        "Running Optuna/TPE optimization. Objective is validation PR-AUC; "
        "test data remains untouched."
    )
    study.optimize(
        make_objective(
            prepared,
            args.tuning_profile,
            args.n_jobs,
            args.device_type,
            args.max_bin,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    if completed_trial_count(study) == 0:
        raise RuntimeError("Optuna finished without any completed trials.")

    save_study_outputs(study, output_dir)

    best_params = (
        fixed_lgbm_params(args.n_jobs, args.device_type, args.max_bin)
        | study.best_params
    )
    save_best_params(study, best_params, output_dir, args)

    if args.skip_final_training:
        log(
            "Skipping final model training and test evaluation because "
            "--skip_final_training is active."
        )
        run_config = build_run_config(
            args,
            output_dir,
            prepared,
            study,
            best_params,
            final_result=None,
        )
        save_json(run_config, output_dir / "run_config.json")
        print_study_only_summary(args, study, output_dir)
        return

    final_result = train_final_model(prepared, best_params, output_dir)
    run_config = build_run_config(
        args,
        output_dir,
        prepared,
        study,
        best_params,
        final_result,
    )
    save_json(run_config, output_dir / "run_config.json")

    if args.skip_global_comparison_update:
        log("Skipping global Optuna comparison update by request.")
        print()
        print("Optuna Tuning Summary")
        print("=====================")
        print(f"Model type             : {args.model_type}")
        print(f"Tuning profile         : {args.tuning_profile}")
        print(f"Completed trials       : {completed_trial_count(study)}")
        print(f"Best validation PR-AUC : {study.best_value:.6f}")
        print(f"Best trial             : {study.best_trial.number}")
        print(
            "Test PR-AUC             : "
            f"{final_result['metrics_test_selected']['average_precision']:.6f}"
        )
        print(f"Outputs saved to       : {output_dir}")
        return

    table = save_optuna_comparison_table()
    print_final_summary(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 5 Optuna/TPE tuning for baseline_lgbm, "
            "baseline_lgbm_entity_time_amount_features, ae_lgbm_ld128, "
            "and ae_augmented_lgbm_ld128."
        )
    )
    parser.add_argument(
        "--model_type",
        required=True,
        choices=SUPPORTED_MODEL_TYPES,
        help=(
            "Model to tune: baseline_lgbm, "
            "baseline_lgbm_entity_time_amount_features, ae_lgbm_ld128, "
            "or ae_augmented_lgbm_ld128."
        ),
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run in this invocation.",
    )
    parser.add_argument(
        "--tuning_profile",
        choices=SUPPORTED_TUNING_PROFILES,
        default="quick",
        help="Search-space profile. Use quick for smoke tests and final for Kaggle runs.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="Number of LightGBM worker threads. Use 4 by default to limit CPU usage.",
    )
    parser.add_argument(
        "--device-type",
        choices=("cpu", "gpu", "cuda"),
        default="cpu",
        help=(
            "LightGBM device_type. Defaults to cpu; gpu/cuda are passed through "
            "to LightGBM for GPU training."
        ),
    )
    parser.add_argument(
        "--max-bin",
        type=int,
        default=None,
        help="Optional LightGBM max_bin value, commonly tuned lower for GPU training.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--study_name",
        default=None,
        help="Optional Optuna study name. Defaults to phase5_<model_type>.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///outputs/optuna/study.db.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory override for all tuned artifacts. "
            "Defaults to outputs/optuna/<model_type>."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse complete existing outputs for the requested model_type.",
    )
    parser.add_argument(
        "--skip_final_training",
        action="store_true",
        help=(
            "Run Optuna and save tuning artifacts, but skip final model training, "
            "threshold selection, test evaluation, and optuna_comparison.csv update."
        ),
    )
    parser.add_argument(
        "--skip-global-comparison-update",
        action="store_true",
        help=(
            "Do not rewrite the hardcoded global outputs/final_comparison/"
            "optuna_comparison.csv table."
        ),
    )
    args = parser.parse_args()
    if args.n_trials < 0:
        raise SystemExit("--n_trials must be zero or a positive integer.")
    if args.n_jobs == 0:
        raise SystemExit("--n_jobs must be non-zero.")
    return args


def main() -> None:
    args = parse_args()
    run_tuning(args)


if __name__ == "__main__":
    main()
