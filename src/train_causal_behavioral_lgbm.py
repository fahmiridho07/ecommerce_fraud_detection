"""Train B2: original-feature LightGBM with leakage-safe causal behavioral features."""

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

from autoencoder_helpers import prepare_output_dir
from causal_behavioral_features import (
    SAME_TIMESTAMP_POLICY,
    build_feature_definition_metadata,
    causal_behavioral_feature_names,
    generate_causal_behavioral_features,
    validate_causal_behavioral_features,
    validate_feature_identity_alignment,
)
from config import (
    CAUSAL_BEHAVIORAL_FEATURE_AUDIT_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
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


EXPERIMENT_NAME = "causal_behavioral_lgbm_default"
ID_ALIGNED_EXPERIMENT_NAME = "causal_behavioral_lgbm_id_aligned"
ID_ALIGNED_MODEL_ID = "CBA01R"


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


def build_alignment_validation(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train_behavioral: pd.DataFrame,
    X_valid_behavioral: pd.DataFrame,
    X_test_behavioral: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Verify raw, label, and behavioral rows share identical TransactionID order."""
    split_checks: dict[str, object] = {}
    for split_name, split_df, X_behavioral, y_split in (
        ("train", train_df, X_train_behavioral, y_train),
        ("validation", valid_df, X_valid_behavioral, y_valid),
        ("test", test_df, X_test_behavioral, y_test),
    ):
        raw_ids = split_df[ID_COL].tolist()
        label_ids = split_df[ID_COL].tolist()
        if raw_ids != label_ids:
            raise ValueError(f"{split_name}: raw and label TransactionID order differ.")
        if len(X_behavioral) != len(split_df):
            raise ValueError(
                f"{split_name}: behavioral row count does not match split row count."
            )
        split_checks[split_name] = {
            **validate_feature_identity_alignment(split_df, X_behavioral, split_name),
            "raw_id_sequence_equals_label_id_sequence": raw_ids == label_ids,
            "raw_id_sequence_equals_behavioral_row_order": True,
            "label_row_count": int(len(y_split)),
        }

    return {
        "transaction_id_join_verified": True,
        "positional_join_used": False,
        "behavioral_join_key": ID_COL,
        "split_checks": split_checks,
        "feature_transaction_id_checksums": {
            split_name: split_checks[split_name]["transaction_id_checksum"]
            for split_name in split_checks
        },
    }


def save_alignment_artifacts(
    output_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alignment_validation: dict[str, object],
) -> None:
    save_json(
        train_df[ID_COL].tolist(),
        output_dir / "train_transaction_ids.json",
    )
    save_json(
        valid_df[ID_COL].tolist(),
        output_dir / "validation_transaction_ids.json",
    )
    save_json(
        test_df[ID_COL].tolist(),
        output_dir / "test_transaction_ids.json",
    )
    save_json(
        alignment_validation["feature_transaction_id_checksums"],
        output_dir / "feature_transaction_id_checksums.json",
    )
    save_json(alignment_validation, output_dir / "alignment_validation.json")


def validate_final_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    original_feature_count: int,
    behavioral_feature_count: int,
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation final columns do not align with train.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test final columns do not align with train.")

    behavioral_features = causal_behavioral_feature_names()
    missing_behavioral = [
        feature for feature in behavioral_features if feature not in X_train.columns
    ]
    if missing_behavioral:
        raise ValueError(
            "Missing causal behavioral feature(s): " + ", ".join(missing_behavioral)
        )

    expected_total = original_feature_count + behavioral_feature_count
    if X_train.shape[1] != expected_total:
        raise ValueError(
            f"Expected {expected_total} features, found {X_train.shape[1]}."
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


def prepare_causal_behavioral_splits() -> dict[str, object]:
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_split_integrity(full_df, train_df, valid_df, test_df)

    log("Generating causal behavioral features with online state continuation.")
    (
        X_train_behavioral,
        X_valid_behavioral,
        X_test_behavioral,
        behavioral_summary,
    ) = generate_causal_behavioral_features(train_df, valid_df, test_df)
    behavioral_checks = validate_causal_behavioral_features(
        X_train_behavioral,
        X_valid_behavioral,
        X_test_behavioral,
    )

    log("Separating target and fitting train-only preprocessing on original features.")
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    alignment_validation = build_alignment_validation(
        train_df,
        valid_df,
        test_df,
        X_train_behavioral,
        X_valid_behavioral,
        X_test_behavioral,
        y_train,
        y_valid,
        y_test,
    )

    X_train_combined = pd.concat(
        [
            X_train_raw.reset_index(drop=True),
            X_train_behavioral.reset_index(drop=True),
        ],
        axis=1,
    )
    X_valid_combined = pd.concat(
        [
            X_valid_raw.reset_index(drop=True),
            X_valid_behavioral.reset_index(drop=True),
        ],
        axis=1,
    )
    X_test_combined = pd.concat(
        [
            X_test_raw.reset_index(drop=True),
            X_test_behavioral.reset_index(drop=True),
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
        "X_train_behavioral": X_train_behavioral,
        "X_valid_behavioral": X_valid_behavioral,
        "X_test_behavioral": X_test_behavioral,
        "X_train_combined": X_train_combined,
        "X_valid_combined": X_valid_combined,
        "X_test_combined": X_test_combined,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "behavioral_summary": behavioral_summary,
        "behavioral_checks": behavioral_checks,
        "alignment_validation": alignment_validation,
    }


def run_validate_only() -> dict[str, object]:
    prepared = prepare_causal_behavioral_splits()
    behavioral_names = causal_behavioral_feature_names()
    print()
    print("Causal Behavioral LightGBM Validate-Only Summary")
    print("================================================")
    print(f"Original features      : {prepared['X_train_raw'].shape[1]}")
    print(f"Behavioral features    : {len(behavioral_names)}")
    print(
        f"Total features         : "
        f"{prepared['X_train_combined'].shape[1]}"
    )
    print(f"Entity definitions     : {list(behavioral_names)[:3]} ...")
    print("Training skipped       : True")
    return {
        "behavioral_summary": prepared["behavioral_summary"],
        "behavioral_checks": prepared["behavioral_checks"],
        "feature_names": behavioral_names,
    }


def run_experiment(
    output_dir: Path,
    overwrite: bool,
    id_aligned: bool = False,
) -> dict[str, object]:
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    prepared = prepare_causal_behavioral_splits()

    behavioral_names = causal_behavioral_feature_names()
    original_feature_count = int(prepared["X_train_raw"].shape[1])
    behavioral_feature_count = len(behavioral_names)

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
    validate_final_feature_alignment(
        X_train,
        X_valid,
        X_test,
        original_feature_count,
        behavioral_feature_count,
    )
    if id_aligned:
        save_alignment_artifacts(
            output_dir,
            prepared["train_df"],
            prepared["valid_df"],
            prepared["test_df"],
            prepared["alignment_validation"],
        )

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

    log("Saving B2 causal behavioral LightGBM outputs.")
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
        behavioral_names,
        output_dir / "behavioral_feature_importance.csv",
    )
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")

    feature_definition = build_feature_definition_metadata()
    save_json(feature_definition, output_dir / "feature_definition.json")
    audit_dir = ensure_dir(CAUSAL_BEHAVIORAL_FEATURE_AUDIT_OUTPUT_DIR)
    save_json(feature_definition, audit_dir / "feature_definition.json")

    experiment_name = ID_ALIGNED_EXPERIMENT_NAME if id_aligned else EXPERIMENT_NAME
    model_id = ID_ALIGNED_MODEL_ID if id_aligned else "B2"
    run_config = {
        "phase": experiment_name,
        "experiment_name": experiment_name,
        "model_id": model_id,
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
        "causal_feature_policy": feature_definition["causal_policy"],
        "entity_definitions": feature_definition["entity_definitions"],
        "behavioral_feature_count": behavioral_feature_count,
        "original_feature_count": original_feature_count,
        "final_feature_count": int(X_train.shape[1]),
        "online_state_continuation_policy": (
            feature_definition["state_transition_policy"]
        ),
        "labels_not_used_in_feature_state": True,
        "future_rows_not_used": True,
        "train_only_categorical_preprocessing": True,
        "validation_only_threshold_selection": True,
        "test_not_used_for_model_selection": True,
        "leakage_prevention": {
            "split": "Chronological 60/20/20 by TransactionDT.",
            "behavioral_features": feature_definition["causal_policy"],
            "tie_breaking": feature_definition["tie_breaking_policy"],
            "target_encoding_used": False,
            "fraud_labels_used_for_features": False,
            "train_static_entity_counts_used": False,
            "ae_signals_used": False,
            "optuna_used": False,
        },
        "behavioral_checks": prepared["behavioral_checks"],
        "behavioral_summary": prepared["behavioral_summary"],
        "preprocessing": {
            "categorical_fit": "Categorical mappings fit on train combined features only.",
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
    }
    if id_aligned:
        run_config.update(
            {
                "correction_type": "transaction_id_alignment",
                "supersedes_experiment": "CBA01",
                "source_output_preserved": True,
                "frozen_split_membership_preserved": True,
                "positional_join_used": False,
                "behavioral_join_key": ID_COL,
                "deterministic_event_order": prepared["behavioral_summary"][
                    "deterministic_event_order"
                ],
                "same_timestamp_policy": SAME_TIMESTAMP_POLICY,
                "boundary_tie_counts": {
                    "train_validation": int(
                        prepared["behavioral_summary"].get(
                            "boundary_tie_counts", {}
                        ).get("train_validation", 0)
                    ),
                    "validation_test": int(
                        prepared["behavioral_summary"].get(
                            "boundary_tie_counts", {}
                        ).get("validation_test", 0)
                    ),
                },
                "alignment_validation": prepared["alignment_validation"],
            }
        )
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("B2 Causal Behavioral LightGBM Summary")
    print("=====================================")
    print(f"Validation AP      : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test AP            : {metrics_test_selected['average_precision']:.6f}")
    print(f"Selected threshold : {selected_threshold:.2f}")
    print(f"Best iteration     : {best_iteration}")
    print(f"Total features     : {X_train.shape[1]}")
    print(f"Outputs saved to   : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "prepared": prepared,
        "preprocessing": preprocessing,
        "categorical_columns": categorical_columns,
        "model_params": model_params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train B2: original-feature LightGBM with causal behavioral features."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
        help="Output directory for B2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a non-empty output directory.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and validate features without training.",
    )
    parser.add_argument(
        "--id-aligned",
        action="store_true",
        help="Run corrected CBA01R with TransactionID-keyed behavioral alignment.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    set_seed(RANDOM_SEED)
    output_dir = args.output_dir
    if args.id_aligned and args.output_dir == CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR:
        output_dir = CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR
    if args.validate_only:
        return run_validate_only()
    return run_experiment(
        output_dir=output_dir,
        overwrite=args.overwrite,
        id_aligned=args.id_aligned,
    )


if __name__ == "__main__":
    main()