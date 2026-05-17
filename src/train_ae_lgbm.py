"""Train Phase 4 AE-LightGBM using robust Autoencoder latent V-features."""

from __future__ import annotations

import argparse
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
    AE_LGBM_OUTPUT_DIR,
    AUTOENCODER_ROBUST_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    DATA_DIR,
    ID_COL,
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
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_VALUE,
    fit_categorical_mappings,
    get_categorical_columns,
    get_v_feature_columns,
    transform_categorical_columns,
)
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_THRESHOLD = 0.5
EARLY_STOPPING_ROUNDS = 100


def average_precision_eval(y_true, y_pred):
    """LightGBM custom validation metric for PR-AUC / Average Precision."""
    return "average_precision", average_precision_score(y_true, y_pred), True


def roc_auc_eval(y_true, y_pred):
    """LightGBM custom validation metric for ROC-AUC."""
    if len(set(y_true)) < 2:
        return "roc_auc", 0.0, True
    return "roc_auc", roc_auc_score(y_true, y_pred), True


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_robust_latent_outputs(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_OUTPUT_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    """Load robust Autoencoder latent arrays and metadata."""
    required_files = {
        "latent_train": autoencoder_output_dir / "latent_train.npy",
        "latent_valid": autoencoder_output_dir / "latent_valid.npy",
        "latent_test": autoencoder_output_dir / "latent_test.npy",
        "latent_feature_names": autoencoder_output_dir / "latent_feature_names.json",
        "run_config": autoencoder_output_dir / "run_config.json",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing robust Autoencoder output(s):\n"
            + "\n".join(missing)
            + "\nRun `python src/train_autoencoder_robust.py` first."
        )

    latent_train = np.load(required_files["latent_train"])
    latent_valid = np.load(required_files["latent_valid"])
    latent_test = np.load(required_files["latent_test"])
    latent_feature_names = load_json(required_files["latent_feature_names"])
    run_config = load_json(required_files["run_config"])

    if not isinstance(latent_feature_names, list):
        raise TypeError("latent_feature_names.json must contain a list of feature names.")

    return latent_train, latent_valid, latent_test, latent_feature_names, run_config


def validate_latent_outputs(
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray,
    latent_feature_names: list[str],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    """Validate robust latent arrays align to the Phase 1 temporal split."""
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
        raise ValueError(
            "Latent feature name count does not match latent array dimension: "
            f"{len(latent_feature_names)} vs {latent_dim}."
        )
    if len(set(latent_feature_names)) != len(latent_feature_names):
        raise ValueError("Duplicate latent feature names found.")


def split_non_v_features_target(
    df: pd.DataFrame,
    v_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return non-V model features and target labels.

    Original V-features are intentionally excluded because they are replaced by
    robust Autoencoder latent features in the proposed model.
    """
    y = df[TARGET_COL].astype(int).copy()
    excluded = set(v_columns + [TARGET_COL, ID_COL])
    feature_columns = [column for column in df.columns if column not in excluded]
    return df.loc[:, feature_columns].copy(), y


def fit_non_v_preprocessing(X_train: pd.DataFrame, v_columns: list[str]) -> dict[str, object]:
    """Fit categorical mappings on train non-V features only."""
    categorical_columns = get_categorical_columns(X_train)
    categorical_mappings = fit_categorical_mappings(X_train, categorical_columns)
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_mappings": categorical_mappings,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "dropped_columns": [ID_COL],
        "excluded_original_v_features": v_columns,
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }


def apply_non_v_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    """Apply train-fitted non-V categorical mappings."""
    feature_columns = preprocessing["feature_columns"]
    categorical_mappings = preprocessing["categorical_mappings"]
    X = X.loc[:, feature_columns].copy()
    return transform_categorical_columns(X, categorical_mappings)


def combine_non_v_and_latent(
    X_non_v: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
) -> pd.DataFrame:
    """Concatenate processed non-V features with robust latent V features."""
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    return pd.concat(
        [
            X_non_v.reset_index(drop=True),
            latent_df.reset_index(drop=True),
        ],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
) -> None:
    """Ensure final model matrices are aligned and original V-features are absent."""
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    leaked_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if leaked_v_columns:
        raise ValueError(
            "Original V-features were found in final AE-LightGBM features: "
            + ", ".join(leaked_v_columns[:10])
        )


def build_model_params(y_train: pd.Series) -> dict[str, object]:
    """Build fixed LightGBM parameters consistent with the Phase 2 baseline."""
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


def load_baseline_selected_metrics() -> dict[str, float] | None:
    baseline_path = BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    if not baseline_path.exists():
        return None
    return load_json(baseline_path)


def build_baseline_comparison(
    ae_metrics: dict[str, object],
) -> dict[str, float] | None:
    baseline_metrics = load_baseline_selected_metrics()
    if baseline_metrics is None:
        return None

    return {
        "baseline_test_pr_auc": baseline_metrics["average_precision"],
        "ae_lgbm_test_pr_auc": ae_metrics["average_precision"],
        "delta_pr_auc": ae_metrics["average_precision"] - baseline_metrics["average_precision"],
        "baseline_test_roc_auc": baseline_metrics["roc_auc"],
        "ae_lgbm_test_roc_auc": ae_metrics["roc_auc"],
        "delta_roc_auc": ae_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        "baseline_test_f1": baseline_metrics["f1"],
        "ae_lgbm_test_f1": ae_metrics["f1"],
        "delta_f1": ae_metrics["f1"] - baseline_metrics["f1"],
        "baseline_test_mcc": baseline_metrics["mcc"],
        "ae_lgbm_test_mcc": ae_metrics["mcc"],
        "delta_mcc": ae_metrics["mcc"] - baseline_metrics["mcc"],
    }


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    save_json(metrics, path)


def main(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_OUTPUT_DIR,
    output_dir: Path = AE_LGBM_OUTPUT_DIR,
    phase_name: str = "4_ae_lgbm",
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)

    log("Loading robust Autoencoder latent features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        robust_ae_run_config,
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

    log("Building non-V feature matrices and fitting train-only preprocessing.")
    X_train_non_v_raw, y_train = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, y_valid = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test = split_non_v_features_target(test_df, v_columns)

    # Leakage prevention: non-V categorical mappings are fit on train only.
    # Original V-features are excluded here and replaced by robust AE latents.
    preprocessing_non_v = fit_non_v_preprocessing(X_train_non_v_raw, v_columns)
    X_train_non_v = apply_non_v_preprocessing(X_train_non_v_raw, preprocessing_non_v)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing_non_v)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing_non_v)

    log("Combining processed non-V features with robust latent V features.")
    X_train = combine_non_v_and_latent(X_train_non_v, latent_train, latent_feature_names)
    X_valid = combine_non_v_and_latent(X_valid_non_v, latent_valid, latent_feature_names)
    X_test = combine_non_v_and_latent(X_test_non_v, latent_test, latent_feature_names)
    validate_feature_alignment(X_train, X_valid, X_test, v_columns)

    categorical_columns = preprocessing_non_v["categorical_columns"]
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training AE-LightGBM with validation early stopping.")
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

    log("Saving AE-LightGBM outputs.")
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
    joblib.dump(preprocessing_non_v, output_dir / "preprocessing_non_v.pkl")

    robust_preprocessing = robust_ae_run_config.get("preprocessing", {})
    feature_set_summary = {
        "number_of_non_v_features": int(X_train_non_v.shape[1]),
        "number_of_latent_v_features": int(len(latent_feature_names)),
        "total_final_features": int(X_train.shape[1]),
        "original_v_features_excluded": True,
        "number_of_original_v_features_excluded": int(len(v_columns)),
        "robust_autoencoder_output_path_used": str(autoencoder_output_dir),
        "robust_autoencoder_clipping": {
            "enabled": robust_preprocessing.get("scaled_clipping_enabled"),
            "clip_min": robust_preprocessing.get("clip_min"),
            "clip_max": robust_preprocessing.get("clip_max"),
        },
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
        "transactiondt_note": (
            "TransactionDT is kept as a non-V model feature and was also used "
            "to create the chronological split."
        ),
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "feature_construction": {
            "original_v_features_replaced_by_robust_latents": True,
            "original_v_feature_count": len(v_columns),
            "non_v_feature_count": int(X_train_non_v.shape[1]),
            "latent_feature_count": len(latent_feature_names),
            "total_feature_count": int(X_train.shape[1]),
            "robust_autoencoder_output_dir": str(autoencoder_output_dir),
        },
        "preprocessing": {
            "non_v_categorical_fit": "Categorical mappings fit on train non-V features only.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "categorical_missing_value": MISSING_CATEGORY,
            "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
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

    print()
    print("AE-LightGBM Summary")
    print("===================")
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
    print(f"Outputs saved to  : {output_dir}")

    if comparison is not None:
        print()
        print("Comparison Against Baseline")
        print("===========================")
        print(f"Delta test PR-AUC : {comparison['delta_pr_auc']:+.6f}")
        print(f"Delta test ROC-AUC: {comparison['delta_roc_auc']:+.6f}")
        print(f"Delta test F1     : {comparison['delta_f1']:+.6f}")
        print(f"Delta test MCC    : {comparison['delta_mcc']:+.6f}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
        "comparison_against_baseline": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AE-LightGBM using robust Autoencoder latent V-features."
    )
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=AUTOENCODER_ROBUST_OUTPUT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=AE_LGBM_OUTPUT_DIR)
    parser.add_argument("--phase-name", default="4_ae_lgbm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        autoencoder_output_dir=args.autoencoder_output_dir,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
    )
