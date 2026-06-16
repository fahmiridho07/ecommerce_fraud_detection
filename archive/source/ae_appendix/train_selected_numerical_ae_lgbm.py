"""Train selected-numerical AE latent replacement LightGBM (anchor-alignment experiment)."""

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
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
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


def average_precision_eval(y_true, y_pred):
    return "average_precision", average_precision_score(y_true, y_pred), True


def roc_auc_eval(y_true, y_pred):
    if len(set(y_true)) < 2:
        return "roc_auc", 0.0, True
    return "roc_auc", roc_auc_score(y_true, y_pred), True


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_selected_numerical_latent_outputs(
    autoencoder_output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], dict[str, object]]:
    required_files = {
        "latent_train": autoencoder_output_dir / "latent_train.npy",
        "latent_valid": autoencoder_output_dir / "latent_valid.npy",
        "latent_test": autoencoder_output_dir / "latent_test.npy",
        "latent_feature_names": autoencoder_output_dir / "latent_feature_names.json",
        "selected_numerical_feature_names": (
            autoencoder_output_dir / "selected_numerical_feature_names.json"
        ),
        "run_config": autoencoder_output_dir / "run_config.json",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing selected-numerical Autoencoder output(s):\n"
            + "\n".join(missing)
            + "\nRun `python src/train_autoencoder_selected_numerical.py` first."
        )

    latent_train = np.load(required_files["latent_train"])
    latent_valid = np.load(required_files["latent_valid"])
    latent_test = np.load(required_files["latent_test"])
    latent_feature_names = load_json(required_files["latent_feature_names"])
    selected_numerical_features = load_json(required_files["selected_numerical_feature_names"])
    run_config = load_json(required_files["run_config"])

    if not isinstance(latent_feature_names, list):
        raise TypeError("latent_feature_names.json must contain a list.")
    if not isinstance(selected_numerical_features, list):
        raise TypeError("selected_numerical_feature_names.json must contain a list.")

    return (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        selected_numerical_features,
        run_config,
    )


def validate_latent_outputs(
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray,
    latent_feature_names: list[str],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    expected = {
        "train": (latent_train, train_rows),
        "validation": (latent_valid, valid_rows),
        "test": (latent_test, test_rows),
    }
    for split_name, (latent, row_count) in expected.items():
        if latent.shape[0] != row_count:
            raise ValueError(
                f"{split_name} latent row count {latent.shape[0]} does not match "
                f"split row count {row_count}."
            )

    latent_dim = latent_train.shape[1]
    if latent_valid.shape[1] != latent_dim or latent_test.shape[1] != latent_dim:
        raise ValueError("Latent arrays do not have the same number of columns.")
    if len(latent_feature_names) != latent_dim:
        raise ValueError("Latent feature name count does not match latent dimension.")
    if len(set(latent_feature_names)) != len(latent_feature_names):
        raise ValueError("Duplicate latent feature names found.")
    if not np.isfinite(latent_train).all():
        raise ValueError("latent_train contains non-finite values.")


def split_non_selected_features_target(
    df: pd.DataFrame,
    selected_numerical_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].astype(int).copy()
    excluded = set(selected_numerical_features + [TARGET_COL, ID_COL])
    feature_columns = [column for column in df.columns if column not in excluded]
    return df.loc[:, feature_columns].copy(), y


def fit_non_ae_preprocessing(
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


def apply_non_ae_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    feature_columns = preprocessing["feature_columns"]
    categorical_mappings = preprocessing["categorical_mappings"]
    X = X.loc[:, feature_columns].copy()
    return transform_categorical_columns(X, categorical_mappings)


def combine_non_selected_and_latent(
    X_non_selected: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
) -> pd.DataFrame:
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    return pd.concat(
        [
            X_non_selected.reset_index(drop=True),
            latent_df.reset_index(drop=True),
        ],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_numerical_features: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    leaked = sorted(set(X_train.columns) & set(selected_numerical_features))
    if leaked:
        raise ValueError(
            "Original selected numerical AE columns remain in final LightGBM matrix: "
            + ", ".join(leaked[:10])
        )

    if len(X_train.columns) != len(set(X_train.columns)):
        raise ValueError("Duplicate columns found in final LightGBM feature matrix.")

    if "reconstruction_mse" in X_train.columns or any(
        column.startswith("ae_reconstruction") for column in X_train.columns
    ):
        raise ValueError("Reconstruction error features are not allowed in this experiment.")


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
    output_dir: Path = SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Loading selected-numerical Autoencoder latent features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        selected_numerical_features,
        ae_run_config,
    ) = load_selected_numerical_latent_outputs(autoencoder_output_dir)
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    log("Building non-selected feature matrices and fitting train-only preprocessing.")
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

    preprocessing_non_ae = fit_non_ae_preprocessing(
        X_train_raw,
        selected_numerical_features,
    )
    X_train_non_selected = apply_non_ae_preprocessing(X_train_raw, preprocessing_non_ae)
    X_valid_non_selected = apply_non_ae_preprocessing(X_valid_raw, preprocessing_non_ae)
    X_test_non_selected = apply_non_ae_preprocessing(X_test_raw, preprocessing_non_ae)

    log("Combining retained raw features with selected-numerical latent features.")
    X_train = combine_non_selected_and_latent(
        X_train_non_selected,
        latent_train,
        latent_feature_names,
    )
    X_valid = combine_non_selected_and_latent(
        X_valid_non_selected,
        latent_valid,
        latent_feature_names,
    )
    X_test = combine_non_selected_and_latent(
        X_test_non_selected,
        latent_test,
        latent_feature_names,
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        selected_numerical_features,
    )

    transactiondt_retained = TIME_COL in X_train.columns
    categorical_columns = preprocessing_non_ae["categorical_columns"]
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training selected-numerical AE-LightGBM with validation early stopping.")
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

    log("Saving selected-numerical AE-LightGBM outputs.")
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
    joblib.dump(preprocessing_non_ae, output_dir / "preprocessing_non_ae_features.pkl")

    run_config = {
        "experiment_name": "selected_numerical_ae_lgbm_ld128",
        "selected_numerical_ae_source_directory": str(autoencoder_output_dir),
        "original_selected_numerical_features_removed": True,
        "latent_features_added": True,
        "reconstruction_error_included": False,
        "original_v_feature_policy": (
            "All selected numerical features including V1-V339 are removed and "
            "replaced by LD128 latent features."
        ),
        "transactiondt_retained_downstream": transactiondt_retained,
        "final_feature_count": int(X_train.shape[1]),
        "removed_numerical_feature_count": len(selected_numerical_features),
        "retained_raw_feature_count": int(X_train_non_selected.shape[1]),
        "latent_feature_count": len(latent_feature_names),
        "categorical_columns": categorical_columns,
        "train_only_preprocessing": True,
        "validation_only_threshold_selection": True,
        "test_not_used_for_model_selection": True,
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
    print("Selected-Numerical AE-LightGBM Summary")
    print("======================================")
    print(f"Validation AP      : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test AP (descriptive): {metrics_test_selected['average_precision']:.6f}")
    print(f"Retained raw feats : {X_train_non_selected.shape[1]}")
    print(f"Latent features    : {len(latent_feature_names)}")
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"TransactionDT kept : {transactiondt_retained}")
    print(f"Selected threshold : {selected_threshold:.2f}")
    print(f"Best iteration     : {best_iteration}")
    print(f"Outputs saved to   : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
    }


if __name__ == "__main__":
    main()