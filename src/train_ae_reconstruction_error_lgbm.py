"""Train LightGBM with original features plus post-fix AE reconstruction errors."""

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
    DATA_DIR,
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
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
from splitting import create_holdout_split
from train_ae_lgbm import (
    validate_autoencoder_preprocessing_contract,
    validate_latent_split_manifest_alignment,
)
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_AUTOENCODER_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "autoencoder_robust_ld128"
DEFAULT_OUTPUT_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "ae_reconstruction_error_ld128_default"
RECONSTRUCTION_ERROR_FEATURE = "v_ae_reconstruction_mse"
LOG_RECONSTRUCTION_ERROR_FEATURE = "v_ae_reconstruction_log1p_mse"
ADDED_FEATURES = [
    RECONSTRUCTION_ERROR_FEATURE,
    LOG_RECONSTRUCTION_ERROR_FEATURE,
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_postfix_autoencoder_config(autoencoder_dir: Path) -> dict[str, object]:
    run_config_path = autoencoder_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing Autoencoder run_config: {run_config_path}")
    run_config = load_json(run_config_path)
    validate_autoencoder_preprocessing_contract(run_config, autoencoder_dir)
    return run_config


def reconstruction_error_paths(autoencoder_dir: Path) -> dict[str, Path]:
    return {
        "train": autoencoder_dir / "reconstruction_error_train.csv",
        "validation": autoencoder_dir / "reconstruction_error_valid.csv",
        "test": autoencoder_dir / "reconstruction_error_test.csv",
    }


def load_reconstruction_error_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstruction-error file: {path}")
    frame = pd.read_csv(path)
    if "reconstruction_mse" not in frame.columns:
        raise KeyError(f"{path} is missing reconstruction_mse column.")
    values = frame["reconstruction_mse"].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite reconstruction errors.")
    if np.any(values < 0):
        raise ValueError(f"{path} contains negative reconstruction errors.")
    return values


def load_reconstruction_errors(autoencoder_dir: Path) -> dict[str, np.ndarray]:
    return {
        split_name: load_reconstruction_error_csv(path)
        for split_name, path in reconstruction_error_paths(autoencoder_dir).items()
    }


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
        actual = int(errors[split_name].shape[0])
        if actual != row_count:
            raise ValueError(
                f"{split_name} reconstruction-error length {actual} does not "
                f"match split rows {row_count}."
            )


def reconstruction_error_features(errors: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            RECONSTRUCTION_ERROR_FEATURE: errors.astype("float32"),
            LOG_RECONSTRUCTION_ERROR_FEATURE: np.log1p(errors).astype("float32"),
        }
    )


def add_reconstruction_error_features(
    X: pd.DataFrame,
    errors: np.ndarray,
) -> pd.DataFrame:
    error_frame = reconstruction_error_features(errors)
    return pd.concat(
        [X.reset_index(drop=True), error_frame.reset_index(drop=True)],
        axis=1,
    )


def validate_feature_matrix(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
    original_feature_count: int,
) -> dict[str, int]:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")
    if X_train.columns.duplicated().any():
        duplicates = X_train.columns[X_train.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature columns found: {duplicates[:10]}")

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        raise ValueError(
            "AE reconstruction-error augmentation must retain all original V-features; "
            f"retained {len(retained_v_columns)} of {len(v_columns)}."
        )

    missing_added = [feature for feature in ADDED_FEATURES if feature not in X_train.columns]
    if missing_added:
        raise ValueError("Missing added AE feature(s): " + ", ".join(missing_added))

    latent_columns = [
        column for column in X_train.columns if str(column).startswith("ae_latent_")
    ]
    if latent_columns:
        raise ValueError(
            "Latent AE features are not part of this controlled experiment: "
            + ", ".join(latent_columns[:10])
        )

    if X_train.shape[1] != original_feature_count + len(ADDED_FEATURES):
        raise ValueError(
            "Unexpected feature count after reconstruction-error augmentation: "
            f"{X_train.shape[1]} vs {original_feature_count + len(ADDED_FEATURES)}."
        )

    return {
        "original_feature_count": int(original_feature_count),
        "original_v_feature_count": int(len(v_columns)),
        "reconstruction_error_feature_count": len(ADDED_FEATURES),
        "total_feature_count": int(X_train.shape[1]),
    }


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score: np.ndarray,
    X: pd.DataFrame,
) -> None:
    payload = {
        ID_COL: split_df[ID_COL].to_numpy(),
        TARGET_COL: y.to_numpy(),
        "score": score,
    }
    for feature in ADDED_FEATURES:
        payload[feature] = X[feature].to_numpy()
    pd.DataFrame(payload).to_csv(path, index=False)


def load_metrics_if_available(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return load_json(path)


def metric_delta(
    metrics: dict[str, object],
    reference: dict[str, object],
    key: str,
) -> float:
    return float(metrics[key]) - float(reference[key])


def build_reference_comparison(
    metrics_test_selected: dict[str, object],
    initial_proposal_dir: Path,
) -> dict[str, object]:
    references = {
        "p01_baseline_default": (
            initial_proposal_dir / "baseline_lgbm_default" / "metrics_test_selected_threshold.json"
        ),
        "p02_baseline_tuned": (
            initial_proposal_dir
            / "optuna"
            / "baseline_lgbm_tuned"
            / "metrics_test_selected_threshold.json"
        ),
        "p03_ae_lgbm_ld32": (
            initial_proposal_dir / "ae_lgbm_ld32_default" / "metrics_test_selected_threshold.json"
        ),
        "p04_ae_lgbm_ld128_tuned": (
            initial_proposal_dir
            / "optuna"
            / "ae_lgbm_ld128_tuned"
            / "metrics_test_selected_threshold.json"
        ),
    }

    comparison: dict[str, object] = {
        "model_test_average_precision": metrics_test_selected["average_precision"],
        "model_test_roc_auc": metrics_test_selected["roc_auc"],
        "model_test_f1": metrics_test_selected["f1"],
        "model_test_mcc": metrics_test_selected["mcc"],
    }
    for name, path in references.items():
        reference = load_metrics_if_available(path)
        if reference is None:
            comparison[f"{name}_metrics_missing"] = str(path)
            continue
        comparison[f"{name}_test_average_precision"] = reference["average_precision"]
        comparison[f"delta_pr_auc_vs_{name}"] = metric_delta(
            metrics_test_selected,
            reference,
            "average_precision",
        )
        comparison[f"{name}_test_roc_auc"] = reference["roc_auc"]
        comparison[f"delta_roc_auc_vs_{name}"] = metric_delta(
            metrics_test_selected,
            reference,
            "roc_auc",
        )
        comparison[f"{name}_test_mcc"] = reference["mcc"]
        comparison[f"delta_mcc_vs_{name}"] = metric_delta(
            metrics_test_selected,
            reference,
            "mcc",
        )
    return comparison


def main(
    autoencoder_output_dir: Path = DEFAULT_AUTOENCODER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    initial_proposal_dir: Path = DEFAULT_INITIAL_PROPOSAL_DIR,
    phase_name: str = "AE_RECON_LD128_original_features_plus_reconstruction_error",
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Validating post-fix Autoencoder reconstruction-error source.")
    autoencoder_run_config = load_postfix_autoencoder_config(autoencoder_output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log(f"Creating {split_strategy} train/validation/test split.")
    train_df, valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=split_strategy,
    )
    validate_latent_split_manifest_alignment(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
    )
    v_columns = get_v_feature_columns(train_df)

    log("Separating target and fitting train-only baseline preprocessing.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train_original = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid_original = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test_original = apply_baseline_preprocessing(X_test_raw, preprocessing)
    original_feature_count = int(X_train_original.shape[1])

    log("Loading saved AE reconstruction errors.")
    reconstruction_errors = load_reconstruction_errors(autoencoder_output_dir)
    validate_reconstruction_error_lengths(
        reconstruction_errors,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    log("Appending AE reconstruction-error features to original feature matrices.")
    X_train = add_reconstruction_error_features(
        X_train_original,
        reconstruction_errors["train"],
    )
    X_valid = add_reconstruction_error_features(
        X_valid_original,
        reconstruction_errors["validation"],
    )
    X_test = add_reconstruction_error_features(
        X_test_original,
        reconstruction_errors["test"],
    )
    feature_counts = validate_feature_matrix(
        X_train,
        X_valid,
        X_test,
        v_columns,
        original_feature_count,
    )

    categorical_columns = preprocessing["categorical_columns"]
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training LightGBM with validation early stopping.")
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

    log("Saving outputs.")
    save_json(metrics_valid_default, output_dir / "metrics_validation_default_threshold.json")
    save_json(metrics_valid_selected, output_dir / "metrics_validation_selected_threshold.json")
    save_json(metrics_test_default, output_dir / "metrics_test_default_threshold.json")
    save_json(metrics_test_selected, output_dir / "metrics_test_selected_threshold.json")
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
    save_scores(output_dir / "scores_validation.csv", valid_df, y_valid, valid_score, X_valid)
    save_scores(output_dir / "scores_test.csv", test_df, y_test, test_score, X_test)
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")

    comparison = build_reference_comparison(metrics_test_selected, initial_proposal_dir)
    save_json(comparison, output_dir / "comparison_against_initial_proposal.json")

    run_config = {
        "phase": phase_name,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "split_strategy": split_strategy,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "feature_construction": {
            "representation_mode": "original_features_plus_ae_reconstruction_error",
            "original_features_retained": True,
            "original_feature_count": feature_counts["original_feature_count"],
            "original_v_feature_count": feature_counts["original_v_feature_count"],
            "added_reconstruction_error_features": ADDED_FEATURES,
            "reconstruction_error_feature_count": feature_counts[
                "reconstruction_error_feature_count"
            ],
            "latent_features_used": False,
            "reconstructed_features_used": False,
            "total_feature_count": feature_counts["total_feature_count"],
            "autoencoder_output_dir": str(autoencoder_output_dir),
            "autoencoder_latent_dim": autoencoder_run_config.get("architecture", {}).get(
                "encoder",
                [None],
            )[-1],
            "autoencoder_loss": autoencoder_run_config.get("training", {}).get("loss"),
            "autoencoder_missing_value_strategy": autoencoder_run_config.get(
                "preprocessing",
                {},
            ).get("missing_value_strategy"),
        },
        "preprocessing": {
            "categorical_mappings_fit": "Train split only.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
            "reconstruction_error_source": "Saved post-fix AE reconstruction_error_*.csv.",
            "reconstruction_error_transform": "raw MSE plus log1p(MSE).",
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
        "comparison_against_initial_proposal": comparison,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("AE Reconstruction-Error LightGBM Summary")
    print("========================================")
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
    print(f"Total features    : {feature_counts['total_feature_count']}")
    print(f"Outputs saved to  : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_counts": feature_counts,
        "comparison_against_initial_proposal": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM with original features plus post-fix LD128 "
            "Autoencoder reconstruction-error features."
        )
    )
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=DEFAULT_AUTOENCODER_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=DEFAULT_INITIAL_PROPOSAL_DIR,
    )
    parser.add_argument(
        "--phase-name",
        default="AE_RECON_LD128_original_features_plus_reconstruction_error",
    )
    parser.add_argument(
        "--split-strategy",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help="Holdout split strategy. Default is the active thesis stratified reset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        autoencoder_output_dir=args.autoencoder_output_dir,
        output_dir=args.output_dir,
        initial_proposal_dir=args.initial_proposal_dir,
        phase_name=args.phase_name,
        split_strategy=args.split_strategy,
    )
