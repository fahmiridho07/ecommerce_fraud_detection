"""Train task-aware AE latent replacement LightGBM (TAE01 selected lambda)."""

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
    DATA_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR,
    TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
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
SELECTED_AE_SUBDIR = "selected"


def average_precision_eval(y_true, y_pred):
    from sklearn.metrics import average_precision_score

    return "average_precision", average_precision_score(y_true, y_pred), True


def roc_auc_eval(y_true, y_pred):
    from sklearn.metrics import roc_auc_score

    if len(set(y_true)) < 2:
        return "roc_auc", 0.0, True
    return "roc_auc", roc_auc_score(y_true, y_pred), True


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_task_aware_latent_outputs(
    autoencoder_output_dir: Path,
    require_test: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str], list[str], dict[str, object]]:
    required_files = {
        "latent_train": autoencoder_output_dir / "latent_train.npy",
        "latent_valid": autoencoder_output_dir / "latent_valid.npy",
        "latent_feature_names": autoencoder_output_dir / "latent_feature_names.json",
        "selected_numerical_feature_names": (
            autoencoder_output_dir / "selected_numerical_feature_names.json"
        ),
        "run_config": autoencoder_output_dir / "run_config.json",
    }
    if require_test:
        required_files["latent_test"] = autoencoder_output_dir / "latent_test.npy"

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing task-aware Autoencoder output(s):\n"
            + "\n".join(missing)
            + "\nRun `python src/train_task_aware_autoencoder_selected_numerical.py` first."
        )

    latent_train = np.load(required_files["latent_train"])
    latent_valid = np.load(required_files["latent_valid"])
    latent_test = (
        np.load(required_files["latent_test"])
        if require_test
        else None
    )
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
    latent_test: np.ndarray | None,
    latent_feature_names: list[str],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    expected = {
        "train": (latent_train, train_rows),
        "validation": (latent_valid, valid_rows),
    }
    if latent_test is not None:
        expected["test"] = (latent_test, test_rows)

    for split_name, (latent, row_count) in expected.items():
        if latent.shape[0] != row_count:
            raise ValueError(
                f"{split_name} latent row count {latent.shape[0]} does not match "
                f"split row count {row_count}."
            )

    latent_dim = latent_train.shape[1]
    if latent_valid.shape[1] != latent_dim:
        raise ValueError("Latent arrays do not have the same number of columns.")
    if latent_test is not None and latent_test.shape[1] != latent_dim:
        raise ValueError("Test latent array does not match latent dimension.")
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


def combine_retained_and_latent(
    X_retained: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
) -> pd.DataFrame:
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    return pd.concat(
        [
            X_retained.reset_index(drop=True),
            latent_df.reset_index(drop=True),
        ],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame | None,
    selected_numerical_features: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test is not None and X_test.columns.tolist() != X_train.columns.tolist():
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

    if any(column.startswith("cb_") for column in X_train.columns):
        raise ValueError("Behavioral features are not allowed in this experiment.")


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


def train_task_aware_downstream_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray | None,
    latent_feature_names: list[str],
    selected_numerical_features: list[str],
    output_dir: Path | None = None,
    save_artifacts: bool = True,
) -> dict[str, object]:
    X_train_raw, y_train = split_non_selected_features_target(
        train_df,
        selected_numerical_features,
    )
    X_valid_raw, y_valid = split_non_selected_features_target(
        valid_df,
        selected_numerical_features,
    )
    X_test_raw = None
    y_test = None
    if test_df is not None:
        X_test_raw, y_test = split_non_selected_features_target(
            test_df,
            selected_numerical_features,
        )

    preprocessing_retained = fit_retained_preprocessing(
        X_train_raw,
        selected_numerical_features,
    )
    X_train_retained = apply_retained_preprocessing(X_train_raw, preprocessing_retained)
    X_valid_retained = apply_retained_preprocessing(X_valid_raw, preprocessing_retained)
    X_test_retained = (
        apply_retained_preprocessing(X_test_raw, preprocessing_retained)
        if X_test_raw is not None
        else None
    )

    X_train = combine_retained_and_latent(
        X_train_retained,
        latent_train,
        latent_feature_names,
    )
    X_valid = combine_retained_and_latent(
        X_valid_retained,
        latent_valid,
        latent_feature_names,
    )
    X_test = (
        combine_retained_and_latent(
            X_test_retained,
            latent_test,
            latent_feature_names,
        )
        if X_test_retained is not None and latent_test is not None
        else None
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        selected_numerical_features,
    )

    categorical_columns = preprocessing_retained["categorical_columns"]
    model_params = build_model_params(y_train)
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
            lgb.log_evaluation(period=50) if save_artifacts else lgb.log_evaluation(0),
        ],
    )

    best_iteration = int(model.best_iteration_ or model.n_estimators)
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    threshold_table = threshold_selection_table(y_valid.to_numpy(), valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)

    metrics_valid_selected = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )

    result: dict[str, object] = {
        "validation_average_precision": metrics_valid_selected["average_precision"],
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "final_feature_count": int(X_train.shape[1]),
        "retained_raw_feature_count": int(X_train_retained.shape[1]),
        "latent_feature_count": len(latent_feature_names),
        "transactiondt_retained": TIME_COL in X_train.columns,
        "model": model,
        "metrics_valid_selected": metrics_valid_selected,
        "preprocessing_retained": preprocessing_retained,
        "threshold_table": threshold_table,
    }

    if not save_artifacts or output_dir is None:
        return result

    output_dir = ensure_dir(output_dir)
    test_score = None
    if X_test is not None and y_test is not None:
        test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]

    metrics_valid_default = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        DEFAULT_THRESHOLD,
    )
    save_json(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )

    if test_score is not None and y_test is not None:
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
        save_json(
            metrics_test_default,
            output_dir / "metrics_test_default_threshold.json",
        )
        save_json(
            metrics_test_selected,
            output_dir / "metrics_test_selected_threshold.json",
        )
        confusion_matrix_table(
            y_test.to_numpy(),
            test_score,
            {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
            "test",
        ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)
        result["metrics_test_selected"] = metrics_test_selected

    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)
    confusion_matrix_table(
        y_valid.to_numpy(),
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)

    save_feature_importance(model, output_dir / "feature_importance.csv")
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing_retained, output_dir / "preprocessing_retained_features.pkl")

    return result


def main(
    autoencoder_output_dir: Path = (
        TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR / SELECTED_AE_SUBDIR
    ),
    output_dir: Path = TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR / SELECTED_AE_SUBDIR,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Loading task-aware Autoencoder latent features (selected lambda).")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        selected_numerical_features,
        ae_run_config,
    ) = load_task_aware_latent_outputs(autoencoder_output_dir, require_test=True)
    if latent_test is None:
        raise ValueError("Selected lambda must include latent_test.npy.")

    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    log("Training task-aware AE-LightGBM with validation early stopping.")
    result = train_task_aware_downstream_lgbm(
        train_df,
        valid_df,
        test_df,
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        selected_numerical_features,
        output_dir=output_dir,
        save_artifacts=True,
    )

    run_config = {
        "experiment_name": "task_aware_ae_lgbm_ld128",
        "source_task_aware_encoder": str(autoencoder_output_dir),
        "selected_lambda": ae_run_config.get("lambda_classification"),
        "original_selected_numerical_features_removed": True,
        "task_aware_latent_features_added": True,
        "reconstruction_error_included": False,
        "reconstructed_features_included": False,
        "behavioral_features_included": False,
        "retained_raw_feature_count": result["retained_raw_feature_count"],
        "latent_feature_count": result["latent_feature_count"],
        "final_feature_count": result["final_feature_count"],
        "transactiondt_retained_downstream": result["transactiondt_retained"],
        "train_only_categorical_preprocessing": True,
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
        "model_params": build_model_params(
            train_df[TARGET_COL].astype(int)
        ),
        "best_iteration": result["best_iteration"],
        "selected_threshold": result["selected_threshold"],
        "autoencoder_run_config": ae_run_config,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Task-Aware AE-LightGBM Summary")
    print("==============================")
    print(f"Selected lambda     : {ae_run_config.get('lambda_classification')}")
    print(f"Validation AP       : {result['validation_average_precision']:.6f}")
    if "metrics_test_selected" in result:
        print(
            "Test AP (descriptive): "
            f"{result['metrics_test_selected']['average_precision']:.6f}"
        )
    print(f"Retained raw feats  : {result['retained_raw_feature_count']}")
    print(f"Latent features     : {result['latent_feature_count']}")
    print(f"Final feature count : {result['final_feature_count']}")
    print(f"TransactionDT kept  : {result['transactiondt_retained']}")
    print(f"Selected threshold  : {result['selected_threshold']:.2f}")
    print(f"Best iteration      : {result['best_iteration']}")
    print(f"Outputs saved to    : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": result["metrics_valid_selected"],
        "metrics_test_selected": result.get("metrics_test_selected"),
        "selected_threshold": result["selected_threshold"],
        "best_iteration": result["best_iteration"],
    }


if __name__ == "__main__":
    main()