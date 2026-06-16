"""Train B3: B2 feature matrix plus exactly one CDV reconstruction-error feature."""

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
from causal_behavioral_features import causal_behavioral_feature_names
from config import (
    BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
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
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from preprocessing import apply_baseline_preprocessing, fit_baseline_preprocessing
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from train_causal_behavioral_lgbm import (
    prepare_causal_behavioral_splits,
    save_alignment_artifacts,
    save_selected_feature_importance,
    train_model,
    validate_final_feature_alignment,
)
from utils import log, save_json, set_seed


RECONSTRUCTION_ERROR_FEATURE = "cdv_ae_reconstruction_mse"
ID_ALIGNED_MODEL_ID = "CBA02R"


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


def load_id_aligned_cdv_errors(
    audit_dir: Path,
    split_df: pd.DataFrame,
    split_name: str,
) -> np.ndarray:
    path = audit_dir / f"cdv_reconstruction_error_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing identity-aware CDV reconstruction error file: {path}"
        )
    frame = pd.read_csv(path)
    if ID_COL not in frame.columns:
        raise KeyError(f"{path} is missing {ID_COL}.")
    if RECONSTRUCTION_ERROR_FEATURE not in frame.columns:
        raise KeyError(f"{path} is missing {RECONSTRUCTION_ERROR_FEATURE}.")
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{path} contains duplicate {ID_COL} values.")
    expected_ids = split_df[ID_COL].tolist()
    indexed = frame.set_index(ID_COL)
    missing_ids = [value for value in expected_ids if value not in indexed.index]
    if missing_ids:
        raise ValueError(
            f"{split_name} CDV errors missing TransactionID value(s): "
            + ", ".join(str(value) for value in missing_ids[:20])
        )
    values = indexed.loc[expected_ids, RECONSTRUCTION_ERROR_FEATURE].to_numpy(
        dtype="float32"
    )
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} CDV reconstruction errors are non-finite.")
    if np.any(values < 0):
        raise ValueError(f"{split_name} CDV reconstruction errors are negative.")
    return values


def assert_b3_matches_b2_except_cdv_error(
    X_train_b2: pd.DataFrame,
    X_valid_b2: pd.DataFrame,
    X_test_b2: pd.DataFrame,
    X_train_b3: pd.DataFrame,
    X_valid_b3: pd.DataFrame,
    X_test_b3: pd.DataFrame,
) -> None:
    for split_name, b2_frame, b3_frame in (
        ("train", X_train_b2, X_train_b3),
        ("validation", X_valid_b2, X_valid_b3),
        ("test", X_test_b2, X_test_b3),
    ):
        b3_without_error = b3_frame.drop(columns=[RECONSTRUCTION_ERROR_FEATURE])
        if not b2_frame.equals(b3_without_error):
            raise ValueError(
                f"{split_name}: B3 matrix without CDV error does not match B2."
            )


def validate_b3_feature_alignment(
    X_train_b2: pd.DataFrame,
    X_valid_b2: pd.DataFrame,
    X_test_b2: pd.DataFrame,
    X_train_b3: pd.DataFrame,
    X_valid_b3: pd.DataFrame,
    X_test_b3: pd.DataFrame,
) -> None:
    if X_valid_b3.columns.tolist() != X_train_b3.columns.tolist():
        raise ValueError("B3 validation columns do not align with train.")
    if X_test_b3.columns.tolist() != X_train_b3.columns.tolist():
        raise ValueError("B3 test columns do not align with train.")

    b2_columns = X_train_b2.columns.tolist()
    b3_columns = X_train_b3.columns.tolist()
    if b3_columns[:-1] != b2_columns:
        raise ValueError("B3 must differ from B2 by exactly one appended column.")
    if b3_columns[-1] != RECONSTRUCTION_ERROR_FEATURE:
        raise ValueError(f"B3 extra column must be {RECONSTRUCTION_ERROR_FEATURE}.")

    latent_columns = [
        column
        for column in X_train_b3.columns
        if str(column).startswith("ae_latent_")
        or str(column).startswith("cdv_ae_latent_")
    ]
    if latent_columns:
        raise ValueError("Latent features are not allowed: " + ", ".join(latent_columns))

    recon_values = X_train_b3[RECONSTRUCTION_ERROR_FEATURE].to_numpy()
    if not np.isfinite(recon_values).all():
        raise ValueError(f"{RECONSTRUCTION_ERROR_FEATURE} contains non-finite values.")
    if np.any(recon_values < 0):
        raise ValueError(f"{RECONSTRUCTION_ERROR_FEATURE} contains negative values.")


def save_cdv_feature_importance(
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


def build_source_ae_validation(
    ae_output_dir: Path,
    source_run_config: dict[str, object] | None,
    source_reconstruction_metrics: dict[str, object] | None,
    reconstruction_errors: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "source_autoencoder_path": str(ae_output_dir),
        "cdv_feature_count": EXPECTED_CDV_FEATURE_COUNT,
        "feature_block": (
            source_run_config.get("feature_block") if source_run_config else None
        ),
        "autoencoder_retrained": False,
        "reconstruction_error_finite": all(
            np.isfinite(errors).all() for errors in reconstruction_errors.values()
        ),
        "reconstruction_error_lengths": {
            split_name: int(errors.shape[0])
            for split_name, errors in reconstruction_errors.items()
        },
        "reconstruction_metrics": source_reconstruction_metrics,
        "labels_used_in_ae_training": False,
        "latent_features_saved": False,
    }


def run_experiment(
    ae_output_dir: Path,
    output_dir: Path,
    overwrite: bool,
    id_aligned: bool = False,
    cdv_error_audit_dir: Path | None = None,
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

    prepared = prepare_causal_behavioral_splits()
    behavioral_names = causal_behavioral_feature_names()
    original_feature_count = int(prepared["X_train_raw"].shape[1])
    behavioral_feature_count = len(behavioral_names)

    log("Fitting train-only preprocessing on B2 combined features.")
    preprocessing = fit_baseline_preprocessing(prepared["X_train_combined"])
    X_train_b2 = apply_baseline_preprocessing(
        prepared["X_train_combined"],
        preprocessing,
    )
    X_valid_b2 = apply_baseline_preprocessing(
        prepared["X_valid_combined"],
        preprocessing,
    )
    X_test_b2 = apply_baseline_preprocessing(
        prepared["X_test_combined"],
        preprocessing,
    )
    validate_final_feature_alignment(
        X_train_b2,
        X_valid_b2,
        X_test_b2,
        original_feature_count,
        behavioral_feature_count,
    )

    if id_aligned:
        audit_dir = Path(cdv_error_audit_dir or CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR)
        log("Loading identity-aware CDV reconstruction errors by TransactionID.")
        reconstruction_errors = {
            "train": load_id_aligned_cdv_errors(
                audit_dir,
                prepared["train_df"],
                "train",
            ),
            "validation": load_id_aligned_cdv_errors(
                audit_dir,
                prepared["valid_df"],
                "validation",
            ),
            "test": load_id_aligned_cdv_errors(
                audit_dir,
                prepared["test_df"],
                "test",
            ),
        }
    else:
        log("Loading behavioral CDV AE reconstruction errors.")
        reconstruction_errors = load_reconstruction_errors(ae_output_dir)
        validate_reconstruction_error_lengths(
            reconstruction_errors,
            len(prepared["train_df"]),
            len(prepared["valid_df"]),
            len(prepared["test_df"]),
        )

    log("Appending exactly one CDV reconstruction-error feature to B2 matrices.")
    X_train = add_cdv_reconstruction_error_feature(
        X_train_b2,
        reconstruction_errors["train"],
    )
    X_valid = add_cdv_reconstruction_error_feature(
        X_valid_b2,
        reconstruction_errors["validation"],
    )
    X_test = add_cdv_reconstruction_error_feature(
        X_test_b2,
        reconstruction_errors["test"],
    )
    validate_b3_feature_alignment(
        X_train_b2,
        X_valid_b2,
        X_test_b2,
        X_train,
        X_valid,
        X_test,
    )
    if id_aligned:
        assert_b3_matches_b2_except_cdv_error(
            X_train_b2,
            X_valid_b2,
            X_test_b2,
            X_train,
            X_valid,
            X_test,
        )
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

    log("Saving B3 causal behavioral + CDV reconstruction LightGBM outputs.")
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
    save_cdv_feature_importance(model, output_dir / "cdv_feature_importance.csv")
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")

    source_ae_validation = build_source_ae_validation(
        ae_output_dir,
        source_run_config,
        source_reconstruction_metrics,
        reconstruction_errors,
    )
    save_json(source_ae_validation, output_dir / "source_ae_validation.json")

    b2_feature_definition_path = (
        CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR.parent
        / "causal_behavioral_lgbm_default"
        / "feature_definition.json"
    )
    if b2_feature_definition_path.exists():
        save_json(load_json(b2_feature_definition_path), output_dir / "feature_definition.json")
    else:
        from causal_behavioral_features import build_feature_definition_metadata

        save_json(
            build_feature_definition_metadata(),
            output_dir / "feature_definition.json",
        )

    model_id = ID_ALIGNED_MODEL_ID if id_aligned else "B3"
    phase_name = (
        "causal_behavioral_cdv_reconstruction_lgbm_id_aligned"
        if id_aligned
        else "causal_behavioral_cdv_reconstruction_lgbm_default"
    )
    run_config = {
        "phase": phase_name,
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
        "B2_feature_set_reused": True,
        "only_added_feature": RECONSTRUCTION_ERROR_FEATURE,
        "source_autoencoder_path": str(ae_output_dir),
        "source_feature_block": "C1-C14 + D1-D15 + V1-V339",
        "autoencoder_retrained": False,
        "reconstruction_error_count": 1,
        "latent_features_added": False,
        "reconstructed_features_added": False,
        "test_not_used_for_model_selection": True,
        "causal_feature_policy": prepared["behavioral_summary"]["causal_policy"],
        "entity_definitions": prepared["behavioral_summary"]["entity_definitions"],
        "behavioral_feature_count": behavioral_feature_count,
        "original_feature_count": original_feature_count,
        "final_feature_count": int(X_train.shape[1]),
        "labels_not_used_in_feature_state": True,
        "future_rows_not_used": True,
        "leakage_prevention": {
            "split": "Chronological 60/20/20 by TransactionDT.",
            "behavioral_features": prepared["behavioral_summary"]["causal_policy"],
            "reconstruction_error_source": (
                "Loaded from saved train-fitted CDV AE; no AE retraining in B3."
            ),
            "ae_signals_used": RECONSTRUCTION_ERROR_FEATURE,
            "latent_features_used": False,
            "optuna_used": False,
        },
        "preprocessing": {
            "categorical_fit": "Same train-fitted preprocessing as B2.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "categorical_missing_value": preprocessing["missing_category"],
            "unknown_category_value": preprocessing["unknown_category_value"],
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
        "source_ae_validation": source_ae_validation,
    }
    if id_aligned:
        run_config.update(
            {
                "correction_type": "transaction_id_alignment",
                "supersedes_experiment": "CBA02",
                "source_corrected_behavioral_model": "CBA01R",
                "source_output_preserved": True,
                "frozen_split_membership_preserved": True,
                "positional_join_used": False,
                "reconstruction_error_join_key": ID_COL,
                "alignment_validation": prepared["alignment_validation"],
            }
        )
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("B3 Causal Behavioral + CDV Reconstruction LightGBM Summary")
    print("==========================================================")
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
        "X_train_b2": X_train_b2,
        "X_train_b3": X_train,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train B3: B2 plus exactly one CDV reconstruction-error feature."
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
        default=CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
        help="Output directory for B3.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a non-empty output directory.",
    )
    parser.add_argument(
        "--id-aligned",
        action="store_true",
        help="Run corrected CBA02R with TransactionID-keyed CDV error join.",
    )
    parser.add_argument(
        "--cdv-error-audit-dir",
        type=Path,
        default=CAUSAL_BEHAVIORAL_ALIGNMENT_AUDIT_OUTPUT_DIR,
        help="Directory containing identity-aware CDV reconstruction error CSVs.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    output_dir = args.output_dir
    if (
        args.id_aligned
        and args.output_dir == CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR
    ):
        output_dir = CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR
    return run_experiment(
        ae_output_dir=args.ae_output_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        id_aligned=args.id_aligned,
        cdv_error_audit_dir=args.cdv_error_audit_dir,
    )


if __name__ == "__main__":
    main()