"""Train fixed-parameter LightGBM with enhanced preprocessing ablations.

This script is separate from the canonical thesis pipeline. It tests whether
normalizing drifting identity/device categoricals and train-only rare bucketing
improves the tuned baseline and AE-05 candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    DATA_DIR,
    ID_COL,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from enhanced_preprocessing import (
    DEFAULT_RARE_MIN_COUNT,
    apply_enhanced_preprocessing,
    fit_enhanced_preprocessing,
)
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from preprocessing import get_v_feature_columns, split_features_target
from splitting import chronological_split
from train_ae_lgbm import (
    build_retained_v_features,
    build_v_missing_indicators,
    combine_non_v_and_latent,
    load_robust_latent_outputs,
    load_top_v_features_from_importance,
    resolve_replaced_v_columns,
    split_non_v_features_target,
    validate_feature_alignment,
    validate_latent_outputs,
    validate_latent_split_manifest_alignment,
)
from train_ae_reconstruction_error_lgbm import (
    LOG_RECONSTRUCTION_ERROR_FEATURE,
    RECONSTRUCTION_ERROR_FEATURE,
    load_reconstruction_errors,
    reconstruction_error_features,
    validate_reconstruction_error_lengths,
)
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_BASELINE_PARAMS = (
    DEFAULT_INITIAL_PROPOSAL_DIR / "optuna" / "baseline_lgbm_tuned" / "best_params.json"
)
DEFAULT_AE05_PARAMS = (
    DEFAULT_INITIAL_PROPOSAL_DIR
    / "ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned"
    / "best_params.json"
)
DEFAULT_AUTOENCODER_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "autoencoder_robust_ld32"
DEFAULT_BASELINE_IMPORTANCE = (
    DEFAULT_INITIAL_PROPOSAL_DIR / "baseline_lgbm_default" / "feature_importance.csv"
)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_params(path: Path, n_jobs: int) -> dict[str, object]:
    payload = load_json(path)
    if "final_model_params" in payload:
        params = dict(payload["final_model_params"])
    elif "best_params" in payload:
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "None",
            "random_state": RANDOM_SEED,
            "verbosity": -1,
        } | dict(payload["best_params"])
    else:
        raise KeyError(f"{path} must contain final_model_params or best_params.")
    params["n_jobs"] = n_jobs
    params["metric"] = "None"
    params["random_state"] = RANDOM_SEED
    return params


def add_reconstruction_features(X: pd.DataFrame, errors: np.ndarray) -> pd.DataFrame:
    error_frame = reconstruction_error_features(errors)
    return pd.concat(
        [X.reset_index(drop=True), error_frame.reset_index(drop=True)],
        axis=1,
    )


def prepare_baseline_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rare_min_count: int,
) -> dict[str, object]:
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessing = fit_enhanced_preprocessing(
        X_train_raw,
        rare_min_count=rare_min_count,
    )
    X_train = apply_enhanced_preprocessing(X_train_raw, preprocessing)
    X_valid = apply_enhanced_preprocessing(X_valid_raw, preprocessing)
    X_test = apply_enhanced_preprocessing(X_test_raw, preprocessing)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "categorical_columns": preprocessing["categorical_columns"],
        "preprocessing": preprocessing,
        "feature_info": {
            "feature_setup": "Enhanced preprocessing baseline.",
            "model_family": "baseline_lgbm",
            "original_feature_count": int(X_train_raw.shape[1]),
            "final_feature_count": int(X_train.shape[1]),
            "categorical_columns_count": len(preprocessing["categorical_columns"]),
            "rare_min_count": rare_min_count,
            "normalization": preprocessing["normalization"],
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
        },
    }


def prepare_ae05_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autoencoder_dir: Path,
    reconstruction_error_dir: Path,
    baseline_importance_path: Path,
    retain_top_v_features: int,
    rare_min_count: int,
) -> dict[str, object]:
    v_columns = get_v_feature_columns(train_df)
    retained_v_columns = load_top_v_features_from_importance(
        baseline_importance_path,
        retain_top_v_features,
        v_columns,
    )
    replaced_v_columns, retained_v_columns = resolve_replaced_v_columns(
        v_columns,
        retained_v_columns,
    )

    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        robust_ae_run_config,
    ) = load_robust_latent_outputs(autoencoder_dir)
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    validate_latent_split_manifest_alignment(
        autoencoder_dir,
        train_df,
        valid_df,
        test_df,
    )

    X_train_non_v_raw, y_train = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, y_valid = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test = split_non_v_features_target(test_df, v_columns)

    preprocessing = fit_enhanced_preprocessing(
        X_train_non_v_raw,
        rare_min_count=rare_min_count,
    )
    X_train_non_v = apply_enhanced_preprocessing(X_train_non_v_raw, preprocessing)
    X_valid_non_v = apply_enhanced_preprocessing(X_valid_non_v_raw, preprocessing)
    X_test_non_v = apply_enhanced_preprocessing(X_test_non_v_raw, preprocessing)

    missing_train = build_v_missing_indicators(train_df, replaced_v_columns)
    missing_valid = build_v_missing_indicators(valid_df, replaced_v_columns)
    missing_test = build_v_missing_indicators(test_df, replaced_v_columns)
    retained_train = build_retained_v_features(train_df, retained_v_columns)
    retained_valid = build_retained_v_features(valid_df, retained_v_columns)
    retained_test = build_retained_v_features(test_df, retained_v_columns)

    X_train = combine_non_v_and_latent(
        X_train_non_v,
        latent_train,
        latent_feature_names,
        missing_train,
        retained_train,
    )
    X_valid = combine_non_v_and_latent(
        X_valid_non_v,
        latent_valid,
        latent_feature_names,
        missing_valid,
        retained_valid,
    )
    X_test = combine_non_v_and_latent(
        X_test_non_v,
        latent_test,
        latent_feature_names,
        missing_test,
        retained_test,
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        replaced_v_columns,
        retained_v_columns,
    )

    errors = load_reconstruction_errors(reconstruction_error_dir)
    validate_reconstruction_error_lengths(
        errors,
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        test_rows=len(test_df),
    )
    base_feature_count = int(X_train.shape[1])
    X_train = add_reconstruction_features(X_train, errors["train"])
    X_valid = add_reconstruction_features(X_valid, errors["validation"])
    X_test = add_reconstruction_features(X_test, errors["test"])

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "categorical_columns": preprocessing["categorical_columns"],
        "preprocessing": preprocessing,
        "feature_info": {
            "feature_setup": "Enhanced preprocessing AE-05 hybrid reconstruction.",
            "model_family": "ae05_hybrid_reconstruction",
            "autoencoder_dir": str(autoencoder_dir),
            "reconstruction_error_dir": str(reconstruction_error_dir),
            "retain_top_v_features": retain_top_v_features,
            "retained_original_v_features": retained_v_columns,
            "replaced_original_v_feature_count": len(replaced_v_columns),
            "latent_feature_count": len(latent_feature_names),
            "v_missing_indicator_count": int(missing_train.shape[1]),
            "feature_count_before_reconstruction_error": base_feature_count,
            "final_feature_count": int(X_train.shape[1]),
            "categorical_columns_count": len(preprocessing["categorical_columns"]),
            "rare_min_count": rare_min_count,
            "normalization": preprocessing["normalization"],
            "reconstruction_error_features": [
                RECONSTRUCTION_ERROR_FEATURE,
                LOG_RECONSTRUCTION_ERROR_FEATURE,
            ],
            "autoencoder_missing_value_strategy": robust_ae_run_config.get(
                "preprocessing",
                {},
            ).get("missing_value_strategy"),
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
        },
    }


def fit_and_evaluate(
    prepared: dict[str, object],
    params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        prepared["X_train"],
        prepared["y_train"],
        eval_set=[(prepared["X_valid"], prepared["y_valid"])],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=prepared["categorical_columns"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )
    best_iteration = int(model.best_iteration_ or params["n_estimators"])
    valid_score = model.predict_proba(
        prepared["X_valid"],
        num_iteration=best_iteration,
    )[:, 1]
    test_score = model.predict_proba(
        prepared["X_test"],
        num_iteration=best_iteration,
    )[:, 1]

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

    save_json(metrics_valid_default, output_dir / "metrics_validation_default_threshold.json")
    save_json(metrics_valid_selected, output_dir / "metrics_validation_selected_threshold.json")
    save_json(metrics_test_default, output_dir / "metrics_test_default_threshold.json")
    save_json(metrics_test_selected, output_dir / "metrics_test_selected_threshold.json")
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
    joblib.dump(model, output_dir / "final_model.pkl")
    model.booster_.save_model(str(output_dir / "final_model.txt"))
    joblib.dump(prepared["preprocessing"], output_dir / "enhanced_preprocessing.pkl")
    return {
        "model": model,
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
    }


def run(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(args.output_dir)

    log("Loading data and creating chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)

    if args.model_type == "baseline":
        prepared = prepare_baseline_data(
            train_df,
            valid_df,
            test_df,
            rare_min_count=args.rare_min_count,
        )
        params = load_params(args.params_json or DEFAULT_BASELINE_PARAMS, args.n_jobs)
    elif args.model_type == "ae05":
        prepared = prepare_ae05_data(
            train_df,
            valid_df,
            test_df,
            autoencoder_dir=args.autoencoder_dir,
            reconstruction_error_dir=args.reconstruction_error_dir,
            baseline_importance_path=args.baseline_importance_path,
            retain_top_v_features=args.retain_top_v_features,
            rare_min_count=args.rare_min_count,
        )
        params = load_params(args.params_json or DEFAULT_AE05_PARAMS, args.n_jobs)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    log(f"Training {args.model_type} with enhanced preprocessing.")
    result = fit_and_evaluate(prepared, params, output_dir)

    run_config = {
        "phase": "enhanced_preprocessing_ablation",
        "model_type": args.model_type,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "preprocessing": {
            "kind": "enhanced_identity_device_rare_bucket",
            "rare_min_count": args.rare_min_count,
            "fit_scope": "train split only",
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
        },
        "feature_construction": prepared["feature_info"],
        "model_features_count": int(prepared["X_train"].shape[1]),
        "params_json": str(args.params_json),
        "model_params": params,
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": result["selected_threshold"],
        },
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": result["best_iteration"],
        },
    }
    save_json(run_config, output_dir / "run_config.json")

    metrics = result["metrics_test_selected"]
    print()
    print("Enhanced Preprocessing LGBM Summary")
    print("===================================")
    print(f"Model type       : {args.model_type}")
    print(f"Validation PR-AUC: {result['metrics_validation_selected']['average_precision']:.6f}")
    print(f"Test PR-AUC      : {metrics['average_precision']:.6f}")
    print(f"Test ROC-AUC     : {metrics['roc_auc']:.6f}")
    print(f"Selected threshold: {result['selected_threshold']:.2f}")
    print(f"Test F1          : {metrics['f1']:.6f}")
    print(f"Test MCC         : {metrics['mcc']:.6f}")
    print(f"Features         : {prepared['X_train'].shape[1]}")
    print(f"Outputs saved to : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run enhanced preprocessing ablations for baseline or AE-05."
    )
    parser.add_argument("--model-type", choices=("baseline", "ae05"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--rare-min-count", type=int, default=DEFAULT_RARE_MIN_COUNT)
    parser.add_argument("--autoencoder-dir", type=Path, default=DEFAULT_AUTOENCODER_DIR)
    parser.add_argument("--reconstruction-error-dir", type=Path, default=DEFAULT_AUTOENCODER_DIR)
    parser.add_argument("--baseline-importance-path", type=Path, default=DEFAULT_BASELINE_IMPORTANCE)
    parser.add_argument("--retain-top-v-features", type=int, default=25)
    args = parser.parse_args()
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs must be non-zero.")
    if args.rare_min_count <= 0:
        raise SystemExit("--rare-min-count must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
