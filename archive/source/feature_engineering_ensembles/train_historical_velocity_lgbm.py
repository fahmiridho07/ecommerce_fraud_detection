"""Train default LightGBM with leakage-safe historical velocity features."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
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
    HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR,
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
from feature_engineering import (
    apply_entity_time_amount_features,
    feature_engineering_summary,
    fit_entity_time_amount_features,
    unknown_rate_summary,
    validate_engineered_features,
)
from historical_velocity_features import (
    generate_historical_velocity_features,
    historical_feature_names,
    validate_historical_velocity_features,
)
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    split_features_target,
)
from splitting import chronological_split, validate_split_integrity
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


EXPERIMENT_NAME = "baseline_lgbm_entity_time_amount_historical_velocity_features"
FE_DEFAULT_VALIDATION_PR_AUC = 0.6277932473974428
FE_TUNED_VALIDATION_PR_AUC = 0.6543163969719032
FE_AE_ENSEMBLE_VALIDATION_PR_AUC = 0.6599352534246169
MIN_MEANINGFUL_VALIDATION_DELTA = 0.005
CURRENT_BEST_TEST_PR_AUC = 0.5339351404285598
BEST_STANDALONE_TUNED_FE_TEST_PR_AUC = 0.529856621916188


def output_dir_is_non_empty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir_is_non_empty(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Pass --overwrite only when you intentionally want to replace files "
            "from this historical velocity experiment."
        )
    return ensure_dir(output_dir)


def feature_importance_frame(model: lgb.LGBMClassifier) -> pd.DataFrame:
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    )
    return importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)


def save_selected_feature_importance(
    model: lgb.LGBMClassifier,
    selected_features: list[str],
    output_path: Path,
) -> pd.DataFrame:
    importance = feature_importance_frame(model)
    selected_importance = importance.loc[
        importance["feature"].isin(selected_features)
    ].reset_index(drop=True)
    selected_importance.to_csv(output_path, index=False)
    return selected_importance


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score,
) -> None:
    pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "score": score,
        }
    ).to_csv(path, index=False)


def validate_final_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    required_features: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation final columns do not align with train.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test final columns do not align with train.")

    missing_features = [
        feature for feature in required_features if feature not in X_train.columns
    ]
    if missing_features:
        raise ValueError(
            "Final model matrix is missing required feature(s): "
            + ", ".join(missing_features[:20])
        )

    retained_internal_keys = [
        column
        for column in X_train.columns
        if str(column).startswith("__fe_key_") or str(column).startswith("uid_")
    ]
    if retained_internal_keys:
        raise ValueError(
            "Internal key columns leaked into final model matrix: "
            + ", ".join(retained_internal_keys)
        )


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


def prepare_historical_velocity_splits() -> dict[str, object]:
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating unchanged chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_split_integrity(full_df, train_df, valid_df, test_df)

    log("Separating target before feature engineering.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    log("Fitting existing entity/time/amount artifacts on train only.")
    feature_artifacts = fit_entity_time_amount_features(X_train_raw)

    log("Applying existing train-fitted FE artifacts to all splits.")
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

    log("Generating historical velocity features without labels.")
    (
        X_train_historical,
        X_valid_historical,
        X_test_historical,
        historical_summary,
    ) = generate_historical_velocity_features(
        X_train_raw,
        X_valid_raw,
        X_test_raw,
    )
    historical_leakage_checks = validate_historical_velocity_features(
        X_train_historical,
        X_valid_historical,
        X_test_historical,
    )
    historical_leakage_checks.update(
        {
            "split_integrity_checked": True,
            "target_column_absent_from_historical_inputs": all(
                TARGET_COL not in X.columns
                for X in (X_train_raw, X_valid_raw, X_test_raw)
            ),
            "competition_test_files_used": False,
            "history_state_update_policy": (
                "For each TransactionDT, all features are computed before any row "
                "with that same timestamp updates historical state."
            ),
            "validation_can_use_prior_train_period": (
                "Validation rows are processed after train rows but can only see "
                "state from strictly earlier TransactionDT values."
            ),
            "test_can_use_prior_train_validation_period": (
                "Test rows are processed after train/validation rows but can only "
                "see state from strictly earlier TransactionDT values."
            ),
            "train_time_max": int(train_df[TIME_COL].max()),
            "validation_time_min": int(valid_df[TIME_COL].min()),
            "validation_time_max": int(valid_df[TIME_COL].max()),
            "test_time_min": int(test_df[TIME_COL].min()),
        }
    )

    log("Computing unknown-rate summaries against train-fitted FE mappings.")
    train_unknown_rates = unknown_rate_summary(X_train_raw, feature_artifacts)
    valid_unknown_rates = unknown_rate_summary(X_valid_raw, feature_artifacts)
    test_unknown_rates = unknown_rate_summary(X_test_raw, feature_artifacts)

    X_train_combined = pd.concat(
        [
            X_train_engineered.reset_index(drop=True),
            X_train_historical.reset_index(drop=True),
        ],
        axis=1,
    )
    X_valid_combined = pd.concat(
        [
            X_valid_engineered.reset_index(drop=True),
            X_valid_historical.reset_index(drop=True),
        ],
        axis=1,
    )
    X_test_combined = pd.concat(
        [
            X_test_engineered.reset_index(drop=True),
            X_test_historical.reset_index(drop=True),
        ],
        axis=1,
    )

    return {
        "full_df": full_df,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "X_train_raw": X_train_raw,
        "X_valid_raw": X_valid_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "feature_artifacts": feature_artifacts,
        "X_train_engineered": X_train_engineered,
        "X_valid_engineered": X_valid_engineered,
        "X_test_engineered": X_test_engineered,
        "X_train_historical": X_train_historical,
        "X_valid_historical": X_valid_historical,
        "X_test_historical": X_test_historical,
        "X_train_combined": X_train_combined,
        "X_valid_combined": X_valid_combined,
        "X_test_combined": X_test_combined,
        "historical_summary": historical_summary,
        "historical_leakage_checks": historical_leakage_checks,
        "unknown_rates": {
            "train": train_unknown_rates,
            "validation": valid_unknown_rates,
            "test": test_unknown_rates,
        },
    }


def build_feature_set_summary(
    prepared: dict[str, object],
    X_train_final: pd.DataFrame | None = None,
) -> dict[str, object]:
    feature_artifacts = prepared["feature_artifacts"]
    X_train_raw = prepared["X_train_raw"]
    feature_summary = feature_engineering_summary(feature_artifacts)
    historical_summary = prepared["historical_summary"]
    total_feature_count = (
        int(X_train_final.shape[1])
        if X_train_final is not None
        else int(prepared["X_train_combined"].shape[1])
    )
    return {
        "experiment_type": EXPERIMENT_NAME,
        "original_feature_count": int(X_train_raw.shape[1]),
        "base_engineered_feature_count": feature_summary["engineered_feature_count"],
        "historical_feature_count": historical_summary["feature_count"],
        "total_feature_count": total_feature_count,
        "original_features_retained": True,
        "base_engineered_features": feature_summary["engineered_feature_names"],
        "historical_features": historical_summary["feature_names"],
        "feature_engineering": feature_summary,
        "historical_velocity": historical_summary,
    }


def stopping_decision(
    metrics_valid_selected: dict[str, object],
    historical_importance: pd.DataFrame,
) -> dict[str, object]:
    validation_pr_auc = float(metrics_valid_selected["average_precision"])
    validation_delta_vs_fe_default = validation_pr_auc - FE_DEFAULT_VALIDATION_PR_AUC
    historical_gain = float(historical_importance["importance_gain"].sum())
    historical_split_count = int(historical_importance["importance_split"].sum())
    historical_features_have_gain = historical_gain > 0.0
    promising = (
        validation_pr_auc >= FE_TUNED_VALIDATION_PR_AUC
        or (
            validation_delta_vs_fe_default >= MIN_MEANINGFUL_VALIDATION_DELTA
            and historical_features_have_gain
        )
    )
    stop = (
        validation_pr_auc
        < FE_DEFAULT_VALIDATION_PR_AUC + MIN_MEANINGFUL_VALIDATION_DELTA
        or not historical_features_have_gain
    )
    return {
        "fe_default_validation_pr_auc_reference": FE_DEFAULT_VALIDATION_PR_AUC,
        "fe_tuned_validation_pr_auc_reference": FE_TUNED_VALIDATION_PR_AUC,
        "fe_ae_ensemble_validation_pr_auc_reference": FE_AE_ENSEMBLE_VALIDATION_PR_AUC,
        "minimum_meaningful_validation_delta": MIN_MEANINGFUL_VALIDATION_DELTA,
        "validation_pr_auc_delta_vs_fe_default": validation_delta_vs_fe_default,
        "historical_importance_gain_sum": historical_gain,
        "historical_importance_split_sum": historical_split_count,
        "historical_features_have_gain": historical_features_have_gain,
        "stop_after_default_run": bool(stop),
        "promising_for_later_tuning_or_ensemble": bool(promising),
        "rule": (
            "Stop if validation PR-AUC is below FE default + 0.005 or historical "
            "features have zero total gain. Mark promising if validation PR-AUC "
            "reaches tuned FE or improves over FE default by at least 0.005 with "
            "nonzero historical gain."
        ),
    }


def run_validate_only() -> dict[str, object]:
    prepared = prepare_historical_velocity_splits()
    feature_set_summary = build_feature_set_summary(prepared)

    print()
    print("Historical Velocity FE LightGBM Validate-Only Summary")
    print("=====================================================")
    print(f"Original features    : {feature_set_summary['original_feature_count']}")
    print(f"Base FE features     : {feature_set_summary['base_engineered_feature_count']}")
    print(f"Historical features  : {feature_set_summary['historical_feature_count']}")
    print(f"Total features       : {feature_set_summary['total_feature_count']}")
    print("Training skipped     : True")
    print("Output writing       : False")
    return {
        "feature_set_summary": feature_set_summary,
        "historical_leakage_checks": prepared["historical_leakage_checks"],
        "unknown_rates": prepared["unknown_rates"],
    }


def run_experiment(output_dir: Path, overwrite: bool) -> dict[str, object]:
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    prepared = prepare_historical_velocity_splits()
    base_engineered_features = prepared["feature_artifacts"]["engineered_feature_names"]
    historical_features = historical_feature_names()
    required_features = base_engineered_features + historical_features

    log("Fitting train-only categorical preprocessing on combined features.")
    preprocessing = fit_baseline_preprocessing(prepared["X_train_combined"])
    X_train = apply_baseline_preprocessing(
        prepared["X_train_combined"],
        preprocessing,
    )
    X_valid = apply_baseline_preprocessing(
        prepared["X_valid_combined"],
        preprocessing,
    )
    X_test = apply_baseline_preprocessing(
        prepared["X_test_combined"],
        preprocessing,
    )
    validate_final_feature_alignment(X_train, X_valid, X_test, required_features)

    log("Training default LightGBM with validation early stopping.")
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

    log("Saving historical velocity LightGBM outputs.")
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
    save_selected_feature_importance(
        model,
        base_engineered_features,
        output_dir / "engineered_feature_importance.csv",
    )
    historical_importance = save_selected_feature_importance(
        model,
        historical_features,
        output_dir / "historical_feature_importance.csv",
    )
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")
    joblib.dump(prepared["feature_artifacts"], output_dir / "feature_engineering.pkl")

    save_scores(
        output_dir / "scores_validation.csv",
        prepared["valid_df"],
        prepared["y_valid"],
        valid_score,
    )
    save_scores(
        output_dir / "scores_test.csv",
        prepared["test_df"],
        prepared["y_test"],
        test_score,
    )

    feature_set_summary = build_feature_set_summary(prepared, X_train_final=X_train)
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")
    save_json(
        prepared["historical_leakage_checks"],
        output_dir / "historical_leakage_checks.json",
    )
    decision = stopping_decision(metrics_valid_selected, historical_importance)

    run_config = {
        "phase": "historical_velocity_lgbm_default",
        "experiment_name": EXPERIMENT_NAME,
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
            "split": "Chronological split is unchanged: first 60%, next 20%, final 20%.",
            "base_feature_engineering_fit": (
                "Existing count/frequency and amount-stat mappings are fit on train only."
            ),
            "historical_features": (
                "Historical features are computed in TransactionDT order; each row "
                "uses only transactions with earlier TransactionDT values."
            ),
            "same_timestamp_policy": (
                "Rows sharing the same TransactionDT do not update state until all "
                "features for that timestamp have been computed."
            ),
            "target_encoding_used": False,
            "fraud_labels_used_for_features": False,
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split used only once after model and threshold selection.",
            "kaggle_competition_test_files_used": False,
            "optuna_used": False,
        },
        "feature_construction": feature_set_summary,
        "historical_leakage_checks": prepared["historical_leakage_checks"],
        "unknown_rate_summary": prepared["unknown_rates"],
        "preprocessing": {
            "categorical_fit": "Categorical mappings fit on combined train features only.",
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
        "reference_test_pr_auc": {
            "current_best_fe_ae_score_ensemble": CURRENT_BEST_TEST_PR_AUC,
            "best_standalone_tuned_fe_lgbm": BEST_STANDALONE_TUNED_FE_TEST_PR_AUC,
        },
        "stopping_criteria": decision,
        "final_training_completed": True,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Historical Velocity FE LightGBM Summary")
    print("=======================================")
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
    print(f"Historical gain   : {decision['historical_importance_gain_sum']:.6f}")
    print(f"Stop after default: {decision['stop_after_default_run']}")
    print(f"Promising later   : {decision['promising_for_later_tuning_or_ensemble']}")
    print(f"Outputs saved to  : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
        "historical_leakage_checks": prepared["historical_leakage_checks"],
        "stopping_criteria": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train default LightGBM with leakage-safe entity/time/amount and "
            "historical velocity features."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR,
        help="Output directory for this experiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and validate features, then exit before training or output writing.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    set_seed(RANDOM_SEED)
    if args.validate_only:
        return run_validate_only()
    return run_experiment(output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
