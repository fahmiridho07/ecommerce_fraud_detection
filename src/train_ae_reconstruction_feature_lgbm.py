"""Train LightGBM with original features plus grouped AE reconstruction features."""

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
from train_ae_lgbm import validate_latent_split_manifest_alignment
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
DEFAULT_AE_FEATURE_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "normal_masked_autoencoder_ld128"
DEFAULT_OUTPUT_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "ae_normal_masked_error_ld128_default"
RECONSTRUCTION_FEATURE_FILES = {
    "train": "reconstruction_features_train.csv",
    "validation": "reconstruction_features_valid.csv",
    "test": "reconstruction_features_test.csv",
}


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def validate_ae_feature_source(
    ae_feature_dir: Path,
    require_normal_only: bool,
) -> dict[str, object]:
    run_config_path = ae_feature_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing AE run_config: {run_config_path}")
    run_config = load_json(run_config_path)
    training = run_config.get("training", {})
    preprocessing = run_config.get("preprocessing", {})
    if (
        not isinstance(training, dict)
        or not isinstance(preprocessing, dict)
        or training.get("loss") != "masked_mse_loss"
    ):
        raise ValueError(f"{ae_feature_dir} does not report masked-loss AE features.")
    target_usage = str(run_config.get("target_usage", ""))
    if require_normal_only and "normal train rows" not in target_usage:
        raise ValueError(
            f"{ae_feature_dir} does not report normal-only AE training."
        )
    return run_config


def load_reconstruction_feature_frame(path: Path, expected_prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing reconstruction feature file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate reconstruction feature columns: {duplicates[:10]}")
    if not all(str(column).startswith(expected_prefix) for column in frame.columns):
        raise ValueError(
            f"All reconstruction feature columns in {path} must start with "
            f"{expected_prefix}."
        )
    values = frame.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite reconstruction feature values.")
    if (values < 0).any():
        negative_columns = frame.columns[(frame < 0).any(axis=0)].tolist()
        raise ValueError(
            "Reconstruction features must be non-negative. Negative columns: "
            + ", ".join(negative_columns[:10])
        )
    return frame.astype("float32")


def load_reconstruction_features(
    ae_feature_dir: Path,
    expected_prefix: str,
) -> dict[str, pd.DataFrame]:
    return {
        split_name: load_reconstruction_feature_frame(
            ae_feature_dir / filename,
            expected_prefix,
        )
        for split_name, filename in RECONSTRUCTION_FEATURE_FILES.items()
    }


def validate_reconstruction_feature_frames(
    frames: dict[str, pd.DataFrame],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> list[str]:
    expected_rows = {
        "train": train_rows,
        "validation": valid_rows,
        "test": test_rows,
    }
    reference_columns = frames["train"].columns.tolist()
    for split_name, row_count in expected_rows.items():
        frame = frames[split_name]
        if len(frame) != row_count:
            raise ValueError(
                f"{split_name} reconstruction feature rows {len(frame)} do not "
                f"match split rows {row_count}."
            )
        if frame.columns.tolist() != reference_columns:
            raise ValueError(
                f"{split_name} reconstruction feature columns do not align with train."
            )
    return reference_columns


def add_reconstruction_features(
    X: pd.DataFrame,
    reconstruction_features: pd.DataFrame,
) -> pd.DataFrame:
    return pd.concat(
        [X.reset_index(drop=True), reconstruction_features.reset_index(drop=True)],
        axis=1,
    )


def validate_feature_matrix(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
    original_feature_count: int,
    reconstruction_feature_names: list[str],
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
            "Grouped AE reconstruction-feature augmentation must retain all "
            f"original V-features; retained {len(retained_v_columns)} of {len(v_columns)}."
        )
    missing_added = [
        feature for feature in reconstruction_feature_names if feature not in X_train.columns
    ]
    if missing_added:
        raise ValueError("Missing AE reconstruction feature(s): " + ", ".join(missing_added))
    latent_columns = [
        column for column in X_train.columns if str(column).startswith("ae_latent_")
    ]
    if latent_columns:
        raise ValueError(
            "Latent AE features are not part of this experiment: "
            + ", ".join(latent_columns[:10])
        )
    expected_total = original_feature_count + len(reconstruction_feature_names)
    if X_train.shape[1] != expected_total:
        raise ValueError(
            f"Unexpected feature count {X_train.shape[1]} vs {expected_total}."
        )
    return {
        "original_feature_count": int(original_feature_count),
        "original_v_feature_count": int(len(v_columns)),
        "reconstruction_feature_count": len(reconstruction_feature_names),
        "total_feature_count": int(X_train.shape[1]),
    }


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
        "ae04_reconstruction_error_ld128": (
            initial_proposal_dir
            / "ae_reconstruction_error_ld128_default"
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
        comparison[f"{name}_test_mcc"] = reference["mcc"]
        comparison[f"delta_mcc_vs_{name}"] = metric_delta(
            metrics_test_selected,
            reference,
            "mcc",
        )
    return comparison


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score: np.ndarray,
    X: pd.DataFrame,
    reconstruction_feature_names: list[str],
) -> None:
    payload = {
        ID_COL: split_df[ID_COL].to_numpy(),
        TARGET_COL: y.to_numpy(),
        "score": score,
    }
    for feature in reconstruction_feature_names:
        payload[feature] = X[feature].to_numpy()
    pd.DataFrame(payload).to_csv(path, index=False)


def main(
    ae_feature_dir: Path = DEFAULT_AE_FEATURE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    initial_proposal_dir: Path = DEFAULT_INITIAL_PROPOSAL_DIR,
    phase_name: str = "AE_NORMAL_MASKED_GROUPED_ERRORS_LD128_default_lgbm",
    expected_feature_prefix: str = "normal_masked_ae_",
    require_normal_only: bool = True,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Validating normal-only mask-aware AE feature source.")
    ae_run_config = validate_ae_feature_source(
        ae_feature_dir,
        require_normal_only=require_normal_only,
    )

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log(f"Creating {split_strategy} train/validation/test split.")
    train_df, valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=split_strategy,
    )
    validate_latent_split_manifest_alignment(
        ae_feature_dir,
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

    log("Loading grouped AE reconstruction features.")
    reconstruction_frames = load_reconstruction_features(
        ae_feature_dir,
        expected_feature_prefix,
    )
    reconstruction_feature_names = validate_reconstruction_feature_frames(
        reconstruction_frames,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    log("Appending AE reconstruction features to original feature matrices.")
    X_train = add_reconstruction_features(
        X_train_original,
        reconstruction_frames["train"],
    )
    X_valid = add_reconstruction_features(
        X_valid_original,
        reconstruction_frames["validation"],
    )
    X_test = add_reconstruction_features(
        X_test_original,
        reconstruction_frames["test"],
    )
    feature_counts = validate_feature_matrix(
        X_train,
        X_valid,
        X_test,
        v_columns,
        original_feature_count,
        reconstruction_feature_names,
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
    save_scores(
        output_dir / "scores_validation.csv",
        valid_df,
        y_valid,
        valid_score,
        X_valid,
        reconstruction_feature_names,
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_df,
        y_test,
        test_score,
        X_test,
        reconstruction_feature_names,
    )
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
            "representation_mode": "original_features_plus_normal_masked_ae_grouped_errors",
            "original_features_retained": True,
            "original_feature_count": feature_counts["original_feature_count"],
            "original_v_feature_count": feature_counts["original_v_feature_count"],
            "reconstruction_feature_names": reconstruction_feature_names,
            "reconstruction_feature_count": feature_counts["reconstruction_feature_count"],
            "latent_features_used": False,
            "reconstructed_features_used": False,
            "total_feature_count": feature_counts["total_feature_count"],
            "ae_feature_dir": str(ae_feature_dir),
            "ae_training_phase": ae_run_config.get("phase"),
            "expected_feature_prefix": expected_feature_prefix,
            "normal_only_source_required": require_normal_only,
        },
        "literature_alignment": {
            "anomaly_detection": (
                "Normal-only AE reconstruction features are used as supervised "
                "LightGBM augmentation, not as replacement for raw V-features."
            ),
            "mask_aware_tabular": (
                "The upstream AE consumed observed masks and reports grouped "
                "reconstruction errors for missing-heavy V blocks."
            ),
        },
        "preprocessing": {
            "categorical_mappings_fit": "Train split only.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
            "reconstruction_feature_source": "Saved normal-only mask-aware AE feature CSVs.",
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
    print("Normal Masked AE Reconstruction-Feature LightGBM Summary")
    print("=========================================================")
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
            "Train LightGBM with original features plus grouped reconstruction "
            "features from a normal-only mask-aware AE."
        )
    )
    parser.add_argument("--ae-feature-dir", type=Path, default=DEFAULT_AE_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=DEFAULT_INITIAL_PROPOSAL_DIR,
    )
    parser.add_argument(
        "--phase-name",
        default="AE_NORMAL_MASKED_GROUPED_ERRORS_LD128_default_lgbm",
    )
    parser.add_argument("--expected-feature-prefix", default="normal_masked_ae_")
    parser.add_argument(
        "--allow-non-normal-source",
        action="store_true",
        help="Allow grouped features generated from a non-normal-only masked-loss AE.",
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
        ae_feature_dir=args.ae_feature_dir,
        output_dir=args.output_dir,
        initial_proposal_dir=args.initial_proposal_dir,
        phase_name=args.phase_name,
        expected_feature_prefix=args.expected_feature_prefix,
        require_normal_only=not args.allow_non_normal_source,
        split_strategy=args.split_strategy,
    )
