"""Train selected-numerical reconstructed-feature replacement LightGBM (Ding alignment)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    DATA_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR,
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
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_VALUE,
    fit_categorical_mappings,
    get_categorical_columns,
    transform_categorical_columns,
)
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed

DEFAULT_THRESHOLD = 0.5
EARLY_STOPPING_ROUNDS = 100
EXPECTED_RETAINED_RAW_COUNT = 45
EXPECTED_RECONSTRUCTED_COUNT = 387
EXPECTED_FINAL_FEATURE_COUNT = 432


def average_precision_eval(y_true, y_pred):
    return "average_precision", average_precision_score(y_true, y_pred), True


def roc_auc_eval(y_true, y_pred):
    if len(set(y_true)) < 2:
        return "roc_auc", 0.0, True
    return "roc_auc", roc_auc_score(y_true, y_pred), True


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_reconstructed_outputs(
    autoencoder_output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], dict[str, object]]:
    required_files = {
        "reconstructed_train": autoencoder_output_dir / "reconstructed_train.npy",
        "reconstructed_valid": autoencoder_output_dir / "reconstructed_valid.npy",
        "reconstructed_test": autoencoder_output_dir / "reconstructed_test.npy",
        "reconstructed_feature_names": (
            autoencoder_output_dir / "reconstructed_feature_names.json"
        ),
        "selected_numerical_feature_names": (
            autoencoder_output_dir / "selected_numerical_feature_names.json"
        ),
        "run_config": autoencoder_output_dir / "run_config.json",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing reconstructed feature output(s):\n"
            + "\n".join(missing)
            + "\nRun `python src/generate_selected_numerical_reconstructed_features.py` first."
        )

    reconstructed_train = np.load(required_files["reconstructed_train"])
    reconstructed_valid = np.load(required_files["reconstructed_valid"])
    reconstructed_test = np.load(required_files["reconstructed_test"])
    reconstructed_feature_names = load_json(required_files["reconstructed_feature_names"])
    selected_numerical_features = load_json(required_files["selected_numerical_feature_names"])
    run_config = load_json(required_files["run_config"])

    if not isinstance(reconstructed_feature_names, list):
        raise TypeError("reconstructed_feature_names.json must contain a list.")
    if not isinstance(selected_numerical_features, list):
        raise TypeError("selected_numerical_feature_names.json must contain a list.")

    return (
        reconstructed_train,
        reconstructed_valid,
        reconstructed_test,
        reconstructed_feature_names,
        selected_numerical_features,
        run_config,
    )


def validate_reconstructed_outputs(
    reconstructed_train: np.ndarray,
    reconstructed_valid: np.ndarray,
    reconstructed_test: np.ndarray,
    reconstructed_feature_names: list[str],
    expected_feature_count: int,
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    expected = {
        "train": (reconstructed_train, train_rows),
        "validation": (reconstructed_valid, valid_rows),
        "test": (reconstructed_test, test_rows),
    }
    for split_name, (array, row_count) in expected.items():
        if array.shape[0] != row_count:
            raise ValueError(
                f"{split_name} reconstructed row count {array.shape[0]} does not match "
                f"split row count {row_count}."
            )
        if array.shape[1] != expected_feature_count:
            raise ValueError(
                f"{split_name} reconstructed feature count {array.shape[1]} != "
                f"expected {expected_feature_count}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{split_name} reconstructed array contains non-finite values.")

    if len(reconstructed_feature_names) != expected_feature_count:
        raise ValueError("Reconstructed feature name count mismatch.")
    if len(set(reconstructed_feature_names)) != len(reconstructed_feature_names):
        raise ValueError("Duplicate reconstructed feature names found.")


def split_non_selected_features_target(
    df: pd.DataFrame,
    selected_numerical_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].astype(int).copy()
    excluded = set(selected_numerical_features + [TARGET_COL, ID_COL])
    feature_columns = [column for column in df.columns if column not in excluded]
    return df.loc[:, feature_columns].copy(), y


def fit_retained_preprocessing(
    X_train: pd.DataFrame,
    selected_numerical_features: list[str],
) -> dict[str, object]:
    categorical_columns = get_categorical_columns(X_train)
    categorical_mappings = fit_categorical_mappings(X_train, categorical_columns)
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_mappings": categorical_mappings,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "dropped_columns": [ID_COL],
        "excluded_selected_numerical_features": selected_numerical_features,
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }


def apply_retained_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    feature_columns = preprocessing["feature_columns"]
    categorical_mappings = preprocessing["categorical_mappings"]
    X = X.loc[:, feature_columns].copy()
    return transform_categorical_columns(X, categorical_mappings)


def combine_retained_and_reconstructed(
    X_retained: pd.DataFrame,
    reconstructed: np.ndarray,
    reconstructed_feature_names: list[str],
) -> pd.DataFrame:
    reconstructed_df = pd.DataFrame(reconstructed, columns=reconstructed_feature_names)
    return pd.concat(
        [
            X_retained.reset_index(drop=True),
            reconstructed_df.reset_index(drop=True),
        ],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_numerical_features: list[str],
    reconstructed_feature_names: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    leaked_selected = sorted(set(X_train.columns) & set(selected_numerical_features))
    if leaked_selected:
        raise ValueError(
            "Original selected numerical columns remain in final matrix: "
            + ", ".join(leaked_selected[:10])
        )

    missing_reconstructed = sorted(
        set(reconstructed_feature_names) - set(X_train.columns)
    )
    if missing_reconstructed:
        raise ValueError("Missing reconstructed columns in final matrix.")

    latent_like = [column for column in X_train.columns if column.startswith("ae_latent_")]
    if latent_like:
        raise ValueError("Latent features must not be present.")

    if "reconstruction_mse" in X_train.columns:
        raise ValueError("Reconstruction error must not be present.")

    if len(X_train.columns) != len(set(X_train.columns)):
        raise ValueError("Duplicate columns found in final feature matrix.")

    if X_train.shape[1] != EXPECTED_FINAL_FEATURE_COUNT:
        raise ValueError(
            f"Final feature count {X_train.shape[1]} != expected "
            f"{EXPECTED_FINAL_FEATURE_COUNT}."
        )


def build_model_params(y_train: pd.Series) -> dict[str, object]:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "metric": "None",
        "verbosity": -1,
    }


def save_feature_importance(model: lgb.LGBMClassifier, output_path: Path) -> None:
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    )
    importance = importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)
    importance.to_csv(output_path, index=False)


def main(
    autoencoder_output_dir: Path = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    output_dir: Path = SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Loading decoder-reconstructed selected-numerical features.")
    (
        reconstructed_train,
        reconstructed_valid,
        reconstructed_test,
        reconstructed_feature_names,
        selected_numerical_features,
        ae_run_config,
    ) = load_reconstructed_outputs(autoencoder_output_dir)
    validate_reconstructed_outputs(
        reconstructed_train,
        reconstructed_valid,
        reconstructed_test,
        reconstructed_feature_names,
        EXPECTED_RECONSTRUCTED_COUNT,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    log("Building retained raw feature matrices and fitting train-only preprocessing.")
    X_train_raw, y_train = split_non_selected_features_target(
        train_df,
        selected_numerical_features,
    )
    X_valid_raw, y_valid = split_non_selected_features_target(
        valid_df,
        selected_numerical_features,
    )
    X_test_raw, y_test = split_non_selected_features_target(
        test_df,
        selected_numerical_features,
    )

    if X_train_raw.shape[1] != EXPECTED_RETAINED_RAW_COUNT:
        raise ValueError(
            f"Retained raw feature count {X_train_raw.shape[1]} != "
            f"expected {EXPECTED_RETAINED_RAW_COUNT}."
        )

    preprocessing_retained = fit_retained_preprocessing(
        X_train_raw,
        selected_numerical_features,
    )
    X_train_retained = apply_retained_preprocessing(X_train_raw, preprocessing_retained)
    X_valid_retained = apply_retained_preprocessing(X_valid_raw, preprocessing_retained)
    X_test_retained = apply_retained_preprocessing(X_test_raw, preprocessing_retained)

    log("Combining retained raw features with reconstructed numerical features.")
    X_train = combine_retained_and_reconstructed(
        X_train_retained,
        reconstructed_train,
        reconstructed_feature_names,
    )
    X_valid = combine_retained_and_reconstructed(
        X_valid_retained,
        reconstructed_valid,
        reconstructed_feature_names,
    )
    X_test = combine_retained_and_reconstructed(
        X_test_retained,
        reconstructed_test,
        reconstructed_feature_names,
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        selected_numerical_features,
        reconstructed_feature_names,
    )

    transactiondt_retained = TIME_COL in X_train.columns
    categorical_columns = preprocessing_retained["categorical_columns"]
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training reconstructed-feature LightGBM with validation early stopping.")
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

    log("Saving reconstructed-feature LightGBM outputs.")
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
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing_retained, output_dir / "preprocessing_retained_features.pkl")

    run_config = {
        "experiment_name": "selected_numerical_reconstructed_lgbm",
        "experiment_purpose": (
            "Ding-alignment diagnostic: replace selected numerical predictors "
            "with decoder-reconstructed features instead of bottleneck latent "
            "features in downstream LightGBM."
        ),
        "anchor_alignment_target": "Ding et al. reconstructed-feature integration",
        "autoencoder_source_directory": str(autoencoder_output_dir),
        "autoencoder_retrained": False,
        "selected_numerical_feature_count": len(selected_numerical_features),
        "reconstructed_feature_count": len(reconstructed_feature_names),
        "retained_raw_feature_count": int(X_train_retained.shape[1]),
        "final_feature_count": int(X_train.shape[1]),
        "original_selected_numerical_features_removed": True,
        "reconstructed_features_added": True,
        "latent_features_added": False,
        "reconstruction_error_added": False,
        "reconstructed_representation_space": "scaled",
        "split_strategy": "chronological TransactionDT holdout",
        "train_only_preprocessing": True,
        "validation_only_threshold_selection": True,
        "test_not_used_for_model_selection": True,
        "transactiondt_retained_downstream": transactiondt_retained,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "model_params": model_params,
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "autoencoder_run_config": ae_run_config,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Selected-Numerical Reconstructed LightGBM Summary")
    print("=================================================")
    print(f"Validation AP        : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test AP (descriptive): {metrics_test_selected['average_precision']:.6f}")
    print(f"Retained raw feats   : {X_train_retained.shape[1]}")
    print(f"Reconstructed feats  : {len(reconstructed_feature_names)}")
    print(f"Final feature count  : {X_train.shape[1]}")
    print(f"TransactionDT kept   : {transactiondt_retained}")
    print(f"Selected threshold   : {selected_threshold:.2f}")
    print(f"Best iteration       : {best_iteration}")
    print(f"Outputs saved to     : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
    }


if __name__ == "__main__":
    main()