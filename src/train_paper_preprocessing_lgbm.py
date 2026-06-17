"""Train LightGBM with the active paper-anchored A1 preprocessing branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    ALHARBI_STYLE_OUTPUT_DIR,
    DATA_DIR,
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
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
from paper_preprocessing import (
    apply_alharbi_style_preprocessing,
    fit_alharbi_style_preprocessing,
)
from preprocessing import split_features_target
from splitting import create_holdout_split
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


def main(
    output_dir: Path = ALHARBI_STYLE_OUTPUT_DIR,
    phase_name: str = "A1_alharbi_style_lgbm_default",
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log(f"Creating {split_strategy} train/validation/test split.")
    train_df, valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=split_strategy,
    )

    log("Separating target and fitting A1 train-only preprocessing.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessing = fit_alharbi_style_preprocessing(X_train_raw)
    X_train = apply_alharbi_style_preprocessing(X_train_raw, preprocessing)
    X_valid = apply_alharbi_style_preprocessing(X_valid_raw, preprocessing)
    X_test = apply_alharbi_style_preprocessing(X_test_raw, preprocessing)

    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training A1 LightGBM with validation early stopping.")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
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

    log("Saving A1 outputs.")
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
    joblib.dump(preprocessing, output_dir / "paper_preprocessing.pkl")

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
        "preprocessing": {
            "kind": preprocessing["kind"],
            "anchor": preprocessing["anchor"],
            "fit_scope": "train split only",
            "numeric_missing_values": (
                "Median imputed from train, then z-score scaled with train "
                "mean and standard deviation."
            ),
            "categorical_missing_value": preprocessing["missing_category"],
            "categorical_encoding": (
                "Train-frequency encoding with unseen validation/test "
                "categories mapped to 0 frequency."
            ),
            "source_cards": [
                "docs/literature/cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md",
                "docs/literature/cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md",
            ],
        },
        "feature_construction": {
            "branch_id": "A1",
            "feature_setup": "Alharbi-style frequency/median/z-score preprocessing.",
            "numeric_columns_count": len(preprocessing["numeric_columns"]),
            "categorical_columns_count": len(preprocessing["categorical_columns"]),
            "model_features_count": int(X_train.shape[1]),
            "original_categorical_columns_replaced": True,
            "original_numeric_columns_kept_after_scaling": True,
        },
        "model_params": model_params,
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
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("A1 Alharbi-Style LightGBM Summary")
    print("=================================")
    print(f"Validation PR-AUC : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test PR-AUC       : {metrics_test_selected['average_precision']:.6f}")
    print(f"Validation ROC-AUC: {metrics_valid_selected['roc_auc']:.6f}")
    print(f"Test ROC-AUC      : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Test F1           : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC          : {metrics_test_selected['mcc']:.6f}")
    print(f"Features          : {X_train.shape[1]}")
    print(f"Best iteration    : {best_iteration}")
    print(f"Outputs saved to  : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the A1 Alharbi-style paper-anchored LightGBM baseline."
    )
    parser.add_argument("--output-dir", type=Path, default=ALHARBI_STYLE_OUTPUT_DIR)
    parser.add_argument("--phase-name", default="A1_alharbi_style_lgbm_default")
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
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        split_strategy=args.split_strategy,
    )
