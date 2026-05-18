"""Train baseline LightGBM with only AE reconstruction-error features added."""

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
    AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR,
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    DATA_DIR,
    ID_COL,
    RANDOM_SEED,
    RECON_ERROR_LGBM_NORMAL_ONLY_LOG1P_OUTPUT_DIR,
    RECON_ERROR_LGBM_NORMAL_ONLY_RAW_LOG1P_OUTPUT_DIR,
    RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR,
    RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR,
    RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR,
    RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR,
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
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


SUPPORTED_ERROR_SOURCES = ("robust_ld128", "normal_only_ld128")
SUPPORTED_FEATURE_MODES = ("raw", "log1p", "raw_log1p")

OUTPUT_DIRS = {
    ("robust_ld128", "raw"): RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR,
    ("robust_ld128", "log1p"): RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR,
    ("robust_ld128", "raw_log1p"): RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR,
    ("normal_only_ld128", "raw"): RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR,
    ("normal_only_ld128", "log1p"): RECON_ERROR_LGBM_NORMAL_ONLY_LOG1P_OUTPUT_DIR,
    ("normal_only_ld128", "raw_log1p"): (
        RECON_ERROR_LGBM_NORMAL_ONLY_RAW_LOG1P_OUTPUT_DIR
    ),
}


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def error_source_dir(error_source: str) -> Path:
    if error_source == "robust_ld128":
        return AUTOENCODER_ROBUST_LD128_OUTPUT_DIR
    if error_source == "normal_only_ld128":
        return AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR
    raise ValueError(f"Unsupported error source: {error_source}")


def default_output_dir(error_source: str, feature_mode: str) -> Path:
    try:
        return OUTPUT_DIRS[(error_source, feature_mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported output combination: {error_source}, {feature_mode}"
        ) from exc


def error_feature_names(error_source: str) -> tuple[str, str]:
    if error_source == "robust_ld128":
        raw_name = "ae_reconstruction_mse"
    elif error_source == "normal_only_ld128":
        raw_name = "normal_only_ae_reconstruction_mse"
    else:
        raise ValueError(f"Unsupported error source: {error_source}")
    return raw_name, f"log1p_{raw_name}"


def reconstruction_error_file_paths(source_dir: Path) -> dict[str, Path]:
    return {
        "train": source_dir / "reconstruction_error_train.csv",
        "validation": source_dir / "reconstruction_error_valid.csv",
        "test": source_dir / "reconstruction_error_test.csv",
    }


def load_reconstruction_error_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    if "reconstruction_mse" not in df.columns:
        raise KeyError(f"{path} is missing reconstruction_mse column.")

    values = df["reconstruction_mse"].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite reconstruction errors.")
    if np.any(values < 0):
        raise ValueError(f"{path} contains negative reconstruction errors.")
    return values


def load_reconstruction_errors(error_source: str) -> tuple[dict[str, np.ndarray], Path]:
    source_dir = error_source_dir(error_source)
    paths = reconstruction_error_file_paths(source_dir)
    errors = {
        split_name: load_reconstruction_error_csv(path)
        for split_name, path in paths.items()
    }
    return errors, source_dir


def validate_reconstruction_error_lengths(
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
                f"{split_name} reconstruction-error length "
                f"{errors[split_name].shape[0]} does not match split rows "
                f"{row_count}."
            )


def reconstruction_features(
    errors: np.ndarray,
    feature_mode: str,
    error_source: str,
) -> pd.DataFrame:
    raw_name, log_name = error_feature_names(error_source)
    data: dict[str, np.ndarray] = {}
    if feature_mode in ("raw", "raw_log1p"):
        data[raw_name] = errors.astype("float32")
    if feature_mode in ("log1p", "raw_log1p"):
        data[log_name] = np.log1p(errors).astype("float32")
    if not data:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")
    return pd.DataFrame(data)


def add_reconstruction_features(
    X: pd.DataFrame,
    errors: np.ndarray,
    feature_mode: str,
    error_source: str,
) -> pd.DataFrame:
    error_df = reconstruction_features(errors, feature_mode, error_source)
    return pd.concat(
        [X.reset_index(drop=True), error_df.reset_index(drop=True)],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
    added_features: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    latent_columns = [
        column for column in X_train.columns if str(column).startswith("ae_latent_")
    ]
    if latent_columns:
        raise ValueError(
            "Latent AE features are not allowed in this experiment: "
            + ", ".join(latent_columns[:10])
        )

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        raise ValueError(
            "Original baseline V-features must be retained; "
            f"retained {len(retained_v_columns)} of {len(v_columns)}."
        )

    missing_added = [feature for feature in added_features if feature not in X_train.columns]
    if missing_added:
        raise ValueError("Missing reconstruction feature(s): " + ", ".join(missing_added))


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score: np.ndarray,
    X: pd.DataFrame,
    added_features: list[str],
) -> None:
    payload = {
        ID_COL: split_df[ID_COL].to_numpy(),
        TARGET_COL: y.to_numpy(),
        "score": score,
    }
    for feature in added_features:
        payload[feature] = X[feature].to_numpy()
    pd.DataFrame(payload).to_csv(path, index=False)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
) -> tuple[lgb.LGBMClassifier, dict[str, object], int]:
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
            lgb.log_evaluation(period=50),
        ],
    )
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    return model, model_params, best_iteration


def run_experiment(
    error_source: str,
    feature_mode: str,
    output_dir: Path,
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

    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train_original = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid_original = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test_original = apply_baseline_preprocessing(X_test_raw, preprocessing)

    log(f"Loading reconstruction errors from {error_source}.")
    reconstruction_errors, reconstruction_error_source_dir = load_reconstruction_errors(
        error_source
    )
    validate_reconstruction_error_lengths(
        reconstruction_errors,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    log(f"Adding reconstruction-error feature mode: {feature_mode}.")
    X_train = add_reconstruction_features(
        X_train_original,
        reconstruction_errors["train"],
        feature_mode,
        error_source,
    )
    X_valid = add_reconstruction_features(
        X_valid_original,
        reconstruction_errors["validation"],
        feature_mode,
        error_source,
    )
    X_test = add_reconstruction_features(
        X_test_original,
        reconstruction_errors["test"],
        feature_mode,
        error_source,
    )
    added_features = [
        column
        for column in X_train.columns
        if column not in X_train_original.columns
    ]
    validate_feature_alignment(X_train, X_valid, X_test, v_columns, added_features)

    log("Training default LightGBM with validation early stopping.")
    categorical_columns = preprocessing["categorical_columns"]
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

    log("Saving reconstruction-error LightGBM outputs.")
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
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")

    save_scores(
        output_dir / "scores_validation.csv",
        valid_df,
        y_valid,
        valid_score,
        X_valid,
        added_features,
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_df,
        y_test,
        test_score,
        X_test,
        added_features,
    )

    feature_set_summary = {
        "experiment_type": "baseline_features_plus_reconstruction_error_only",
        "original_feature_count": int(X_train_original.shape[1]),
        "original_features_retained": True,
        "original_v_features_retained": True,
        "original_v_feature_count": int(len(v_columns)),
        "ae_latent_features_included": False,
        "added_reconstruction_features": added_features,
        "added_reconstruction_feature_count": int(len(added_features)),
        "total_feature_count": int(X_train.shape[1]),
        "reconstruction_error_source": error_source,
        "reconstruction_error_source_dir": str(reconstruction_error_source_dir),
        "feature_mode": feature_mode,
    }
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")

    source_run_config_path = reconstruction_error_source_dir / "run_config.json"
    source_reconstruction_metrics_path = (
        reconstruction_error_source_dir / "reconstruction_metrics.json"
    )
    source_run_config = (
        load_json(source_run_config_path)
        if source_run_config_path.exists()
        else None
    )
    source_reconstruction_metrics = (
        load_json(source_reconstruction_metrics_path)
        if source_reconstruction_metrics_path.exists()
        else None
    )

    run_config = {
        "phase": "next_reconstruction_error_lgbm_default",
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
            "preprocessing_fit": "Categorical mappings fit on train original features only.",
            "reconstruction_error_source": (
                "Loaded from saved train-fitted Autoencoder artifacts; no "
                "fitting is performed in this LightGBM stage."
            ),
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split is used only once for final evaluation.",
            "kaggle_competition_test_files_used": False,
            "optuna_used": False,
        },
        "feature_construction": feature_set_summary,
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
        "model_features_count": int(X_train.shape[1]),
        "source_autoencoder_run_config": source_run_config,
        "source_autoencoder_reconstruction_metrics": source_reconstruction_metrics,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Reconstruction-Error LightGBM Summary")
    print("=====================================")
    print(f"Error source      : {error_source}")
    print(f"Feature mode      : {feature_mode}")
    print(f"Added features    : {', '.join(added_features)}")
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

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train default baseline LightGBM with only AE reconstruction-error "
            "features added."
        )
    )
    parser.add_argument(
        "--error-source",
        choices=SUPPORTED_ERROR_SOURCES,
        required=True,
        help="Saved reconstruction-error source to use.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=SUPPORTED_FEATURE_MODES,
        required=True,
        help="Use raw MSE, log1p(MSE), or both reconstruction-error features.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the output directory.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir(
        args.error_source,
        args.feature_mode,
    )
    return run_experiment(
        error_source=args.error_source,
        feature_mode=args.feature_mode,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
