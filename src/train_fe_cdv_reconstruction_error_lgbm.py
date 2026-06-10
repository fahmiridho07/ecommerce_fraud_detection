"""Train FE-LightGBM with behavioral CDV AE reconstruction error appended."""

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

from autoencoder_helpers import (
    EXPECTED_CDV_FEATURE_COUNT,
    load_reconstruction_errors,
    prepare_output_dir,
    validate_reconstruction_error_lengths,
)
from config import (
    BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    DATA_DIR,
    FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR,
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
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
)
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
from utils import log, save_json, set_seed


RECONSTRUCTION_ERROR_FEATURE = "cdv_ae_reconstruction_mse"
DEFAULT_FE_VALIDATION_PR_AUC_REFERENCE = 0.6277932473974428
TUNED_FE_VALIDATION_PR_AUC_REFERENCE = 0.6543163969719032
CURRENT_FE_AE_ENSEMBLE_VALIDATION_PR_AUC_REFERENCE = 0.6599352534246169
DEFAULT_FE_TEST_PR_AUC_REFERENCE = 0.5091169248916745
TUNED_FE_TEST_PR_AUC_REFERENCE = 0.529856621916188
CURRENT_FE_AE_ENSEMBLE_TEST_PR_AUC_REFERENCE = 0.5339351404285598
MIN_VALIDATION_DELTA_FOR_LATENT_FOLLOWUP = 0.005


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def optional_json(path: Path) -> dict[str, object] | None:
    return load_json(path) if path.exists() else None


def source_cdv_feature_count(
    source_run_config: dict[str, object] | None,
    source_reconstruction_metrics: dict[str, object] | None,
) -> int | None:
    if source_run_config:
        feature_block = source_run_config.get("feature_block", {})
        if isinstance(feature_block, dict):
            value = feature_block.get("cdv_feature_count")
            if value is not None:
                return int(value)

    if source_reconstruction_metrics:
        for key in ("cdv_feature_count", "input_dim"):
            value = source_reconstruction_metrics.get(key)
            if value is not None:
                return int(value)

    return None


def validate_source_autoencoder(
    source_dir: Path,
    source_run_config: dict[str, object] | None,
    source_reconstruction_metrics: dict[str, object] | None,
) -> None:
    cdv_feature_count = source_cdv_feature_count(
        source_run_config,
        source_reconstruction_metrics,
    )
    if cdv_feature_count is None:
        raise ValueError(
            f"Could not validate CDV feature count from AE source: {source_dir}"
        )
    if cdv_feature_count != EXPECTED_CDV_FEATURE_COUNT:
        raise ValueError(
            f"Expected AE source with {EXPECTED_CDV_FEATURE_COUNT} CDV features, "
            f"found {cdv_feature_count}."
        )


def add_cdv_reconstruction_error_feature(
    X: pd.DataFrame,
    reconstruction_error: np.ndarray,
) -> pd.DataFrame:
    error_df = pd.DataFrame(
        {RECONSTRUCTION_ERROR_FEATURE: reconstruction_error.astype("float32")}
    )
    return pd.concat(
        [X.reset_index(drop=True), error_df.reset_index(drop=True)],
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    engineered_features: list[str],
    v_columns: list[str],
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

    latent_columns = [
        column
        for column in X_train.columns
        if str(column).startswith("ae_latent_")
        or str(column).startswith("cdv_ae_latent_")
    ]
    if latent_columns:
        raise ValueError(
            "Latent AE features are not allowed in this experiment: "
            + ", ".join(latent_columns[:10])
        )

    if RECONSTRUCTION_ERROR_FEATURE not in X_train.columns:
        raise ValueError(f"Missing {RECONSTRUCTION_ERROR_FEATURE}.")
    values = X_train[RECONSTRUCTION_ERROR_FEATURE].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{RECONSTRUCTION_ERROR_FEATURE} contains non-finite values.")
    if np.any(values < 0):
        raise ValueError(f"{RECONSTRUCTION_ERROR_FEATURE} contains negative values.")


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


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score: np.ndarray,
    X: pd.DataFrame,
) -> None:
    pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "score": score,
            RECONSTRUCTION_ERROR_FEATURE: X[RECONSTRUCTION_ERROR_FEATURE].to_numpy(),
        }
    ).to_csv(path, index=False)


def save_cdv_ae_feature_importance(
    model: lgb.LGBMClassifier,
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
    importance = importance.loc[
        importance["feature"] == RECONSTRUCTION_ERROR_FEATURE
    ].reset_index(drop=True)
    importance.to_csv(output_path, index=False)


def stopping_decision(validation_pr_auc: float, test_pr_auc: float) -> dict[str, object]:
    validation_delta_vs_default = float(
        validation_pr_auc - DEFAULT_FE_VALIDATION_PR_AUC_REFERENCE
    )
    test_delta_vs_default = float(test_pr_auc - DEFAULT_FE_TEST_PR_AUC_REFERENCE)
    test_regression_vs_default = bool(test_delta_vs_default < 0.0)
    latent_followup_allowed = bool(
        validation_delta_vs_default >= MIN_VALIDATION_DELTA_FOR_LATENT_FOLLOWUP
        and not test_regression_vs_default
    )
    return {
        "default_fe_validation_pr_auc_reference": DEFAULT_FE_VALIDATION_PR_AUC_REFERENCE,
        "tuned_fe_validation_pr_auc_reference": TUNED_FE_VALIDATION_PR_AUC_REFERENCE,
        "current_fe_ae_ensemble_validation_pr_auc_reference": (
            CURRENT_FE_AE_ENSEMBLE_VALIDATION_PR_AUC_REFERENCE
        ),
        "default_fe_test_pr_auc_reference": DEFAULT_FE_TEST_PR_AUC_REFERENCE,
        "tuned_fe_test_pr_auc_reference": TUNED_FE_TEST_PR_AUC_REFERENCE,
        "current_fe_ae_ensemble_test_pr_auc_reference": (
            CURRENT_FE_AE_ENSEMBLE_TEST_PR_AUC_REFERENCE
        ),
        "min_validation_delta_for_latent_followup": (
            MIN_VALIDATION_DELTA_FOR_LATENT_FOLLOWUP
        ),
        "validation_pr_auc_delta_vs_default_fe": validation_delta_vs_default,
        "validation_pr_auc_delta_vs_tuned_fe": float(
            validation_pr_auc - TUNED_FE_VALIDATION_PR_AUC_REFERENCE
        ),
        "validation_pr_auc_delta_vs_current_fe_ae_ensemble": float(
            validation_pr_auc - CURRENT_FE_AE_ENSEMBLE_VALIDATION_PR_AUC_REFERENCE
        ),
        "test_pr_auc_delta_vs_default_fe": test_delta_vs_default,
        "test_pr_auc_delta_vs_tuned_fe": float(
            test_pr_auc - TUNED_FE_TEST_PR_AUC_REFERENCE
        ),
        "test_pr_auc_delta_vs_current_fe_ae_ensemble": float(
            test_pr_auc - CURRENT_FE_AE_ENSEMBLE_TEST_PR_AUC_REFERENCE
        ),
        "stop_after_a": not latent_followup_allowed,
        "latent_feature_followup_allowed": latent_followup_allowed,
        "rule": (
            "Stop after Experiment A unless validation PR-AUC beats default "
            "FE-LGBM by at least 0.005 and test does not regress against "
            "default FE-LGBM."
        ),
    }


def run_experiment(
    ae_output_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    ae_output_dir = Path(ae_output_dir)

    source_run_config = optional_json(ae_output_dir / "run_config.json")
    source_reconstruction_metrics = optional_json(
        ae_output_dir / "reconstruction_metrics.json"
    )
    validate_source_autoencoder(
        ae_output_dir,
        source_run_config,
        source_reconstruction_metrics,
    )

    prepared = prepare_engineered_splits()
    feature_artifacts = prepared["feature_artifacts"]
    engineered_features = feature_artifacts["engineered_feature_names"]
    v_columns = get_v_feature_columns(prepared["train_df"])

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

    log("Loading behavioral CDV AE reconstruction errors.")
    reconstruction_errors = load_reconstruction_errors(ae_output_dir)
    validate_reconstruction_error_lengths(
        reconstruction_errors,
        len(prepared["train_df"]),
        len(prepared["valid_df"]),
        len(prepared["test_df"]),
    )

    log("Appending raw CDV AE reconstruction error to FE matrices.")
    X_train = add_cdv_reconstruction_error_feature(
        X_train_fe,
        reconstruction_errors["train"],
    )
    X_valid = add_cdv_reconstruction_error_feature(
        X_valid_fe,
        reconstruction_errors["validation"],
    )
    X_test = add_cdv_reconstruction_error_feature(
        X_test_fe,
        reconstruction_errors["test"],
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        engineered_features,
        v_columns,
    )

    log("Training default FE-LightGBM with validation early stopping.")
    categorical_columns = preprocessing["categorical_columns"]
    model, model_params, best_iteration = train_model(
        X_train,
        prepared["y_train"],
        X_valid,
        prepared["y_valid"],
        categorical_columns,
    )

    log("Generating validation and test probabilities.")
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]

    log("Selecting classification threshold on validation only.")
    threshold_table = threshold_selection_table(
        prepared["y_valid"].to_numpy(),
        valid_score,
    )
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_valid_default = binary_classification_metrics(
        prepared["y_valid"].to_numpy(),
        valid_score,
        DEFAULT_THRESHOLD,
    )
    metrics_valid_selected = binary_classification_metrics(
        prepared["y_valid"].to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        prepared["y_test"].to_numpy(),
        test_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        prepared["y_test"].to_numpy(),
        test_score,
        selected_threshold,
    )
    decision = stopping_decision(
        float(metrics_valid_selected["average_precision"]),
        float(metrics_test_selected["average_precision"]),
    )

    log("Saving FE + CDV reconstruction-error LightGBM outputs.")
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
        prepared["y_valid"].to_numpy(),
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        prepared["y_test"].to_numpy(),
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
    save_cdv_ae_feature_importance(
        model,
        output_dir / "cdv_ae_feature_importance.csv",
    )
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")
    joblib.dump(feature_artifacts, output_dir / "feature_engineering.pkl")

    save_scores(
        output_dir / "scores_validation.csv",
        prepared["valid_df"],
        prepared["y_valid"],
        valid_score,
        X_valid,
    )
    save_scores(
        output_dir / "scores_test.csv",
        prepared["test_df"],
        prepared["y_test"],
        test_score,
        X_test,
    )

    feature_summary = feature_engineering_summary(feature_artifacts)
    feature_set_summary = {
        "experiment_type": "fe_lgbm_plus_cdv_ae_reconstruction_mse_default",
        "original_feature_count": int(prepared["X_train_raw"].shape[1]),
        "engineered_feature_count": int(feature_summary["engineered_feature_count"]),
        "fe_feature_count_after_preprocessing": int(X_train_fe.shape[1]),
        "cdv_ae_latent_features_included": False,
        "reconstruction_error_included": True,
        "reconstruction_error_feature": RECONSTRUCTION_ERROR_FEATURE,
        "reconstruction_error_source_dir": str(ae_output_dir),
        "cdv_autoencoder_feature_count": EXPECTED_CDV_FEATURE_COUNT,
        "total_feature_count": int(X_train.shape[1]),
        "original_features_retained": True,
        "original_v_features_retained": True,
        "original_v_feature_count": int(len(v_columns)),
        "engineered_features": feature_summary["engineered_feature_names"],
        "feature_engineering": feature_summary,
    }
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")

    run_config = {
        "phase": "behavioral_cdv_A_fe_reconstruction_error_default",
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
            "train": int(len(prepared["train_df"])),
            "validation": int(len(prepared["valid_df"])),
            "test": int(len(prepared["test_df"])),
        },
        "leakage_prevention": {
            "split": "Existing chronological 60/20/20 labeled-train split.",
            "feature_engineering_fit": "FE mappings fit on train only.",
            "feature_engineering_apply": "Validation/test transformed with train-fitted FE mappings.",
            "reconstruction_error_source": (
                "Loaded from saved train-fitted CDV AE outputs; no AE fitting "
                "is performed in this LightGBM stage."
            ),
            "categorical_preprocessing_fit": "Categorical mappings fit on train FE features only.",
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split used once after model and threshold selection.",
            "kaggle_competition_test_files_used": False,
            "optuna_used": False,
            "latent_features_used": False,
            "log_reconstruction_error_used": False,
        },
        "feature_construction": feature_set_summary,
        "unknown_rate_summary": prepared["unknown_rates"],
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
        "model_features_count": int(X_train.shape[1]),
        "source_autoencoder": {
            "output_dir": str(ae_output_dir),
            "run_config": source_run_config,
            "reconstruction_metrics": source_reconstruction_metrics,
        },
        "stopping_criteria": decision,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("FE + CDV AE Reconstruction Error LightGBM Summary")
    print("=================================================")
    print(f"Validation PR-AUC       : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Delta vs default FE     : {decision['validation_pr_auc_delta_vs_default_fe']:+.6f}")
    print(f"Delta vs tuned FE       : {decision['validation_pr_auc_delta_vs_tuned_fe']:+.6f}")
    print(f"Test PR-AUC             : {metrics_test_selected['average_precision']:.6f}")
    print(f"Test delta vs default FE: {decision['test_pr_auc_delta_vs_default_fe']:+.6f}")
    print(f"Selected threshold      : {selected_threshold:.2f}")
    print(f"Test F1                 : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC                : {metrics_test_selected['mcc']:.6f}")
    print(f"Best iteration          : {best_iteration}")
    print(f"Total features          : {X_train.shape[1]}")
    print(f"Stop after A            : {decision['stop_after_a']}")
    print(f"Latent follow-up allowed: {decision['latent_feature_followup_allowed']}")
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
            "Train default FE-LGBM with raw behavioral CDV AE reconstruction MSE."
        )
    )
    parser.add_argument(
        "--ae-output-dir",
        type=Path,
        default=BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
        help="Directory containing behavioral CDV AE reconstruction errors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FE_CDV_RECON_ERROR_LGBM_OUTPUT_DIR,
        help="Output directory for Experiment A.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a non-empty Experiment A output directory.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return run_experiment(
        ae_output_dir=args.ae_output_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
