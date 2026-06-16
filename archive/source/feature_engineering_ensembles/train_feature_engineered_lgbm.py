"""Train default LightGBM with leakage-safe entity/time/amount features."""

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
    FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
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
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
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


EXPERIMENT_NAME = "baseline_lgbm_entity_time_amount_features"
VALIDATION_TUNING_GATE = {
    "default_baseline_validation_pr_auc": 0.602433,
    "meaningful_improvement_margin": 0.005,
    "tuned_baseline_validation_pr_auc": 0.624072,
}
REFERENCE_TEST_PR_AUC = {
    "baseline_lgbm_default": 0.485756,
    "baseline_lgbm_tuned": 0.501438,
    "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned": 0.508124,
}


def output_dir_is_non_empty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir_is_non_empty(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Pass --overwrite only when you intentionally want to replace files "
            "from this experiment."
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


def save_engineered_feature_importance(
    model: lgb.LGBMClassifier,
    engineered_features: list[str],
    output_path: Path,
) -> None:
    importance = feature_importance_frame(model)
    engineered_importance = importance.loc[
        importance["feature"].isin(engineered_features)
    ].reset_index(drop=True)
    engineered_importance.to_csv(output_path, index=False)


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score: pd.Series,
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
    engineered_features: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation final columns do not align with train.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test final columns do not align with train.")

    missing_engineered = [
        feature for feature in engineered_features if feature not in X_train.columns
    ]
    if missing_engineered:
        raise ValueError(
            "Final model matrix is missing engineered feature(s): "
            + ", ".join(missing_engineered[:20])
        )

    retained_internal_keys = [
        column
        for column in X_train.columns
        if str(column).startswith("__fe_key_") or str(column).startswith("uid_")
    ]
    if retained_internal_keys:
        raise ValueError(
            "Internal UID/combo key columns leaked into final model matrix: "
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


def prepare_engineered_splits() -> dict[str, object]:
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
    train_unknown_rates = unknown_rate_summary(X_train_raw, feature_artifacts)
    valid_unknown_rates = unknown_rate_summary(X_valid_raw, feature_artifacts)
    test_unknown_rates = unknown_rate_summary(X_test_raw, feature_artifacts)
    log(f"Validation count/frequency unknown rates: {valid_unknown_rates['count_frequency']}")
    log(f"Validation amount-stat unknown rates: {valid_unknown_rates['amount_stats']}")
    log(f"Test count/frequency unknown rates: {test_unknown_rates['count_frequency']}")
    log(f"Test amount-stat unknown rates: {test_unknown_rates['amount_stats']}")

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
    X_train_engineered = prepared["X_train_engineered"]
    feature_summary = feature_engineering_summary(feature_artifacts)
    total_feature_count = (
        int(X_train_final.shape[1])
        if X_train_final is not None
        else int(X_train_engineered.shape[1])
    )
    return {
        "experiment_type": EXPERIMENT_NAME,
        "original_feature_count": int(X_train_raw.shape[1]),
        "engineered_feature_count": feature_summary["engineered_feature_count"],
        "total_feature_count": total_feature_count,
        "original_features_retained": True,
        "engineered_features": feature_summary["engineered_feature_names"],
        "feature_engineering": feature_summary,
    }


def run_validate_only() -> dict[str, object]:
    prepared = prepare_engineered_splits()
    feature_set_summary = build_feature_set_summary(prepared)

    print()
    print("Feature-Engineered LightGBM Validate-Only Summary")
    print("=================================================")
    print(f"Original features  : {feature_set_summary['original_feature_count']}")
    print(f"Engineered features: {feature_set_summary['engineered_feature_count']}")
    print(f"Total features     : {feature_set_summary['total_feature_count']}")
    print("Training skipped   : True")
    print("Output writing     : False")
    return {
        "feature_set_summary": feature_set_summary,
        "unknown_rates": prepared["unknown_rates"],
    }


def run_experiment(output_dir: Path, overwrite: bool) -> dict[str, object]:
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    prepared = prepare_engineered_splits()
    feature_artifacts = prepared["feature_artifacts"]
    engineered_features = feature_artifacts["engineered_feature_names"]

    log("Fitting train-only categorical preprocessing on engineered features.")
    preprocessing = fit_baseline_preprocessing(prepared["X_train_engineered"])
    X_train = apply_baseline_preprocessing(
        prepared["X_train_engineered"],
        preprocessing,
    )
    X_valid = apply_baseline_preprocessing(
        prepared["X_valid_engineered"],
        preprocessing,
    )
    X_test = apply_baseline_preprocessing(
        prepared["X_test_engineered"],
        preprocessing,
    )
    validate_final_feature_alignment(X_train, X_valid, X_test, engineered_features)

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

    log("Saving feature-engineered LightGBM outputs.")
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
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")
    joblib.dump(feature_artifacts, output_dir / "feature_engineering.pkl")

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

    run_config = {
        "phase": "feature_engineered_lgbm_default",
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
            "feature_engineering_fit": (
                "Count, frequency, and amount-stat mappings are fit on train only."
            ),
            "feature_engineering_apply": (
                "Validation/test are transformed only with train-fitted mappings."
            ),
            "unknown_categories": (
                "Unknown count/frequency keys map to 0; unknown amount-stat keys "
                "use global train TransactionAmt fallbacks."
            ),
            "target_encoding_used": False,
            "fraud_labels_used_for_features": False,
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split used only once after model and threshold selection.",
            "kaggle_competition_test_files_used": False,
            "optuna_used": False,
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
        "stopping_criteria_for_future_tuning": VALIDATION_TUNING_GATE,
        "reference_test_pr_auc": REFERENCE_TEST_PR_AUC,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Feature-Engineered LightGBM Summary")
    print("====================================")
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
            "Train default LightGBM with leakage-safe entity/time/amount "
            "features for IEEE-CIS fraud detection."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
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
        help="Build and validate engineered features, then exit before training.",
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
