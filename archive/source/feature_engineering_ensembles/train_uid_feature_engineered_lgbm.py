"""Train default LightGBM with UID-inspired entity/time/amount features."""

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
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
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
    fit_uid_entity_time_amount_features,
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
    save_feature_importance,
)
from train_feature_engineered_lgbm import (
    prepare_output_dir,
    save_engineered_feature_importance,
    save_scores,
    train_model,
    validate_final_feature_alignment,
)
from utils import log, save_json, set_seed


EXPERIMENT_NAME = "baseline_lgbm_entity_time_amount_uid_features"
BASE_FE_EXPERIMENT_NAME = "baseline_lgbm_entity_time_amount_features"
REFERENCE_VALIDATION_PR_AUC = {
    "baseline_lgbm_entity_time_amount_features_default": 0.627793,
    "baseline_lgbm_entity_time_amount_features_tuned": 0.654316,
    "uid_default_tuning_gate": 0.631000,
}
REFERENCE_TEST_PR_AUC = {
    "baseline_lgbm_entity_time_amount_features_tuned": 0.529857,
}


def uid_sparsity_summary(feature_artifacts: dict[str, object]) -> dict[str, object]:
    rows = []
    for group_name, mapping in feature_artifacts["amount_stat_mappings"].items():
        if not str(group_name).startswith("uid_"):
            continue
        counts = mapping["stats"]["count"]
        rows.append(
            {
                "group": group_name,
                "columns": mapping["columns"],
                "mapping_size": int(mapping["mapping_size"]),
                "singleton_group_share": float((counts == 1).mean()),
                "train_row_share_in_singletons": float(
                    counts[counts == 1].sum() / feature_artifacts["fit_row_count"]
                ),
                "median_count": float(counts.median()),
                "p95_count": float(counts.quantile(0.95)),
                "min_stat_count": int(mapping["min_stat_count"]),
            }
        )

    return {
        "uid_amount_stat_groups": rows,
        "skipped_uid_groups_by_design": feature_artifacts.get(
            "skipped_uid_groups_by_design",
            [],
        ),
        "uid_alias_policy": feature_artifacts.get("uid_alias_policy", {}),
    }


def validate_base_features_retained(
    X_train: pd.DataFrame,
    feature_artifacts: dict[str, object],
) -> None:
    base_features = feature_artifacts.get("base_engineered_feature_names", [])
    missing_base = [feature for feature in base_features if feature not in X_train.columns]
    if missing_base:
        raise ValueError(
            "UID feature experiment dropped base engineered feature(s): "
            + ", ".join(missing_base[:20])
        )


def prepare_uid_engineered_splits() -> dict[str, object]:
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Separating target before UID feature engineering.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    log("Fitting base plus UID feature artifacts on train only.")
    feature_artifacts = fit_uid_entity_time_amount_features(X_train_raw)

    log("Applying train-fitted UID feature artifacts to all splits.")
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
    validate_base_features_retained(X_train_engineered, feature_artifacts)

    log("Computing unknown-rate summaries against train-fitted mappings.")
    train_unknown_rates = unknown_rate_summary(X_train_raw, feature_artifacts)
    valid_unknown_rates = unknown_rate_summary(X_valid_raw, feature_artifacts)
    test_unknown_rates = unknown_rate_summary(X_test_raw, feature_artifacts)
    log(f"Validation UID amount-stat unknown rates: {valid_unknown_rates['amount_stats']}")
    log(f"Validation nunique unknown rates: {valid_unknown_rates['nunique']}")
    log(f"Test UID amount-stat unknown rates: {test_unknown_rates['amount_stats']}")
    log(f"Test nunique unknown rates: {test_unknown_rates['nunique']}")

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
        "base_experiment_type": BASE_FE_EXPERIMENT_NAME,
        "original_feature_count": int(X_train_raw.shape[1]),
        "base_engineered_feature_count": feature_summary[
            "base_engineered_feature_count"
        ],
        "uid_engineered_feature_count": feature_summary[
            "uid_engineered_feature_count"
        ],
        "engineered_feature_count": feature_summary["engineered_feature_count"],
        "total_feature_count": total_feature_count,
        "original_features_retained": True,
        "base_engineered_features_retained": True,
        "engineered_features": feature_summary["engineered_feature_names"],
        "feature_engineering": feature_summary,
    }


def run_validate_only() -> dict[str, object]:
    prepared = prepare_uid_engineered_splits()
    feature_set_summary = build_feature_set_summary(prepared)
    sparsity_summary = uid_sparsity_summary(prepared["feature_artifacts"])

    print()
    print("UID Feature-Engineered LightGBM Validate-Only Summary")
    print("=====================================================")
    print(f"Original features       : {feature_set_summary['original_feature_count']}")
    print(f"Base engineered features: {feature_set_summary['base_engineered_feature_count']}")
    print(f"UID engineered features : {feature_set_summary['uid_engineered_feature_count']}")
    print(f"Total features          : {feature_set_summary['total_feature_count']}")
    print("Training skipped        : True")
    print("Output writing          : False")
    return {
        "feature_set_summary": feature_set_summary,
        "uid_sparsity_summary": sparsity_summary,
        "unknown_rates": prepared["unknown_rates"],
    }


def run_experiment(output_dir: Path, overwrite: bool) -> dict[str, object]:
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    prepared = prepare_uid_engineered_splits()
    feature_artifacts = prepared["feature_artifacts"]
    engineered_features = feature_artifacts["engineered_feature_names"]
    uid_engineered_features = feature_artifacts["uid_engineered_feature_names"]

    log("Fitting train-only categorical preprocessing on UID engineered features.")
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
    validate_base_features_retained(X_train, feature_artifacts)

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

    log("Saving UID feature-engineered LightGBM outputs.")
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
    save_engineered_feature_importance(
        model,
        uid_engineered_features,
        output_dir / "uid_engineered_feature_importance.csv",
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
    sparsity_summary = uid_sparsity_summary(feature_artifacts)
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")
    save_json(sparsity_summary, output_dir / "uid_sparsity_summary.json")
    save_json(prepared["unknown_rates"], output_dir / "unknown_rate_summary.json")

    run_config = {
        "phase": "uid_feature_engineered_lgbm_default",
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment_name": BASE_FE_EXPERIMENT_NAME,
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
                "Count, frequency, amount-stat, and nunique mappings are fit on train only."
            ),
            "feature_engineering_apply": (
                "Validation/test are transformed only with train-fitted mappings."
            ),
            "unknown_categories": (
                "Unknown count/frequency/nunique keys map to 0; unknown amount-stat "
                "keys use global train TransactionAmt fallbacks."
            ),
            "target_encoding_used": False,
            "fraud_labels_used_for_features": False,
            "threshold_selection": "Classification threshold selected on validation only.",
            "test_usage": "Test split used only once after model and threshold selection.",
            "kaggle_competition_test_files_used": False,
            "optuna_used": False,
        },
        "feature_construction": feature_set_summary,
        "uid_sparsity_summary": sparsity_summary,
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
        "stopping_criteria_for_future_tuning": {
            "primary_gate": "validation average_precision / PR-AUC",
            "base_fe_default_validation_pr_auc": REFERENCE_VALIDATION_PR_AUC[
                "baseline_lgbm_entity_time_amount_features_default"
            ],
            "recommended_tuning_gate": REFERENCE_VALIDATION_PR_AUC[
                "uid_default_tuning_gate"
            ],
            "base_fe_tuned_validation_pr_auc": REFERENCE_VALIDATION_PR_AUC[
                "baseline_lgbm_entity_time_amount_features_tuned"
            ],
        },
        "reference_test_pr_auc": REFERENCE_TEST_PR_AUC,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("UID Feature-Engineered LightGBM Summary")
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
            "Train default LightGBM with leakage-safe UID-inspired "
            "entity/time/amount features for IEEE-CIS fraud detection."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR,
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
