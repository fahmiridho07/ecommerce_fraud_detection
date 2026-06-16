"""Appendix experiment: compare chronological vs stratified split strategies.

This script is intentionally separate from the main thesis pipeline. It reuses
the same LightGBM training recipe to show how evaluation changes when:

1. Chronological holdout (main thesis design)
2. Stratified holdout (same 60/20/20 ratios, shuffled labels)
3. Stratified K-fold CV (out-of-fold scores across the full labeled dataset)

Leakage prevention is preserved inside each strategy:
- preprocessing / feature engineering are fit on train only
- validation is used for early stopping and threshold selection in holdout runs
- each CV fold fits preprocessing on that fold's train indices only
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    DATA_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SPLIT_STRATEGY_APPENDIX_COMPARISON_FILE,
    SPLIT_STRATEGY_APPENDIX_CV_FILE,
    SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from evaluation import (
    binary_classification_metrics,
    selected_threshold_from_table,
    threshold_selection_table,
)
from feature_engineering import (
    apply_entity_time_amount_features,
    fit_entity_time_amount_features,
    validate_engineered_features,
)
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    split_features_target,
)
from splitting import (
    build_holdout_split_summary,
    chronological_split,
    stratified_holdout_split,
    stratified_kfold_splits,
)
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
)
from utils import ensure_dir, log, save_json, set_seed


SUPPORTED_MODEL_TYPES = ("baseline_lgbm", "feature_engineered_lgbm")
HOLDOUT_STRATEGIES = ("chronological", "stratified_holdout")
DEFAULT_APPENDIX_N_JOBS = 2

HOLDOUT_REQUIRED_FILES = (
    "metrics_test_selected_threshold.json",
    "run_config.json",
    "split_summary.json",
)
CV_REQUIRED_FILES = ("cv_summary.json", "fold_metrics.csv", "run_config.json")

HOLDOUT_COMPARISON_COLUMNS = [
    "model_type",
    "split_strategy",
    "evaluation_scope",
    "temporal_order_preserved",
    "validation_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "selected_threshold",
    "best_iteration",
    "train_fraud_rate",
    "validation_fraud_rate",
    "test_fraud_rate",
    "total_features",
    "output_dir",
]

CV_SUMMARY_COLUMNS = [
    "model_type",
    "split_strategy",
    "evaluation_scope",
    "n_folds",
    "mean_fold_pr_auc",
    "std_fold_pr_auc",
    "mean_fold_roc_auc",
    "std_fold_roc_auc",
    "oof_pr_auc",
    "oof_roc_auc",
    "output_dir",
]


def resolve_sample_size(cli_sample_size: int | None) -> int | None:
    return cli_sample_size if cli_sample_size is not None else SAMPLE_SIZE


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def output_has_required_files(output_dir: Path, required_files: tuple[str, ...]) -> bool:
    return all((output_dir / file_name).exists() for file_name in required_files)


def release_memory(*objects: object) -> None:
    for obj in objects:
        del obj
    gc.collect()


def holdout_experiment_complete(output_dir: Path) -> bool:
    return output_has_required_files(output_dir, HOLDOUT_REQUIRED_FILES)


def cv_experiment_complete(output_dir: Path, n_folds: int) -> bool:
    if not output_has_required_files(output_dir, CV_REQUIRED_FILES):
        return False
    summary = load_json(output_dir / "cv_summary.json")
    return int(summary.get("n_folds", 0)) == n_folds


def holdout_split(
    full_df: pd.DataFrame,
    strategy: str,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strategy == "chronological":
        return chronological_split(full_df)
    if strategy == "stratified_holdout":
        return stratified_holdout_split(full_df, random_seed=random_seed)
    raise ValueError(f"Unsupported holdout strategy: {strategy}")


def prepare_baseline_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    preprocessing = fit_baseline_preprocessing(X_train_raw)
    return {
        "X_train": apply_baseline_preprocessing(X_train_raw, preprocessing),
        "X_valid": apply_baseline_preprocessing(X_valid_raw, preprocessing),
        "X_test": apply_baseline_preprocessing(X_test_raw, preprocessing),
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "categorical_columns": preprocessing["categorical_columns"],
        "preprocessing": preprocessing,
        "feature_engineering": None,
    }


def prepare_feature_engineered_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    feature_artifacts = fit_entity_time_amount_features(X_train_raw)
    X_train_engineered = apply_entity_time_amount_features(X_train_raw, feature_artifacts)
    X_valid_engineered = apply_entity_time_amount_features(X_valid_raw, feature_artifacts)
    X_test_engineered = apply_entity_time_amount_features(X_test_raw, feature_artifacts)
    validate_engineered_features(
        X_train_engineered,
        X_valid_engineered,
        X_test_engineered,
        feature_artifacts,
    )

    preprocessing = fit_baseline_preprocessing(X_train_engineered)
    return {
        "X_train": apply_baseline_preprocessing(X_train_engineered, preprocessing),
        "X_valid": apply_baseline_preprocessing(X_valid_engineered, preprocessing),
        "X_test": apply_baseline_preprocessing(X_test_engineered, preprocessing),
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test,
        "categorical_columns": preprocessing["categorical_columns"],
        "preprocessing": preprocessing,
        "feature_engineering": feature_artifacts,
    }


def prepare_model_matrices(
    model_type: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    if model_type == "baseline_lgbm":
        return prepare_baseline_matrices(train_df, valid_df, test_df)
    if model_type == "feature_engineered_lgbm":
        return prepare_feature_engineered_matrices(train_df, valid_df, test_df)
    raise ValueError(f"Unsupported model_type: {model_type}")


def prepare_train_valid_matrices(
    model_type: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> dict[str, object]:
    """Prepare train/validation matrices without materializing a test split."""
    if model_type == "baseline_lgbm":
        X_train_raw, y_train = split_features_target(train_df)
        X_valid_raw, y_valid = split_features_target(valid_df)
        preprocessing = fit_baseline_preprocessing(X_train_raw)
        return {
            "X_train": apply_baseline_preprocessing(X_train_raw, preprocessing),
            "X_valid": apply_baseline_preprocessing(X_valid_raw, preprocessing),
            "y_train": y_train,
            "y_valid": y_valid,
            "categorical_columns": preprocessing["categorical_columns"],
            "preprocessing": preprocessing,
            "feature_engineering": None,
        }

    if model_type == "feature_engineered_lgbm":
        X_train_raw, y_train = split_features_target(train_df)
        X_valid_raw, y_valid = split_features_target(valid_df)
        feature_artifacts = fit_entity_time_amount_features(X_train_raw)
        X_train_engineered = apply_entity_time_amount_features(X_train_raw, feature_artifacts)
        X_valid_engineered = apply_entity_time_amount_features(X_valid_raw, feature_artifacts)
        validate_engineered_features(
            X_train_engineered,
            X_valid_engineered,
            X_train_engineered.iloc[:0],
            feature_artifacts,
        )
        preprocessing = fit_baseline_preprocessing(X_train_engineered)
        return {
            "X_train": apply_baseline_preprocessing(X_train_engineered, preprocessing),
            "X_valid": apply_baseline_preprocessing(X_valid_engineered, preprocessing),
            "y_train": y_train,
            "y_valid": y_valid,
            "categorical_columns": preprocessing["categorical_columns"],
            "preprocessing": preprocessing,
            "feature_engineering": feature_artifacts,
        }

    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_lgbm_classifier(
    prepared: dict[str, object],
    log_period: int = 0,
    n_jobs: int = DEFAULT_APPENDIX_N_JOBS,
) -> tuple[lgb.LGBMClassifier, dict[str, object], int]:
    model_params = build_model_params(prepared["y_train"])
    model_params["n_jobs"] = n_jobs
    model = lgb.LGBMClassifier(**model_params)
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
                verbose=False,
            ),
            lgb.log_evaluation(period=log_period),
        ],
    )
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    return model, model_params, best_iteration


def evaluate_holdout(
    prepared: dict[str, object],
    model: lgb.LGBMClassifier,
    best_iteration: int,
) -> dict[str, object]:
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

    metrics_valid = binary_classification_metrics(
        prepared["y_valid"].to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test = binary_classification_metrics(
        prepared["y_test"].to_numpy(),
        test_score,
        selected_threshold,
    )
    return {
        "metrics_validation_selected": metrics_valid,
        "metrics_test_selected": metrics_test,
        "selected_threshold": selected_threshold,
        "threshold_table": threshold_table,
        "valid_score": valid_score,
        "test_score": test_score,
    }


def run_holdout_experiment(
    full_df: pd.DataFrame,
    model_type: str,
    strategy: str,
    output_dir: Path,
    sample_size: int | None,
    random_seed: int,
    n_jobs: int = DEFAULT_APPENDIX_N_JOBS,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    train_df, valid_df, test_df = holdout_split(full_df, strategy, random_seed)
    split_summary = build_holdout_split_summary(
        full_df,
        train_df,
        valid_df,
        test_df,
        split_strategy=strategy,
        sample_size=sample_size,
        random_seed=random_seed if strategy == "stratified_holdout" else None,
    )
    save_json(split_summary, output_dir / "split_summary.json")

    prepared = prepare_model_matrices(model_type, train_df, valid_df, test_df)
    model, model_params, best_iteration = fit_lgbm_classifier(
        prepared,
        log_period=50,
        n_jobs=n_jobs,
    )
    evaluation = evaluate_holdout(prepared, model, best_iteration)

    save_json(
        evaluation["metrics_validation_selected"],
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_json(
        evaluation["metrics_test_selected"],
        output_dir / "metrics_test_selected_threshold.json",
    )
    evaluation["threshold_table"].to_csv(output_dir / "threshold_selection.csv", index=False)

    run_config = {
        "appendix": "split_strategy_comparison",
        "model_type": model_type,
        "split_strategy": strategy,
        "evaluation_scope": "holdout_test",
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": sample_size,
        "random_seed": random_seed,
        "split_summary": split_summary,
        "model_params": model_params,
        "best_iteration": best_iteration,
        "selected_threshold": evaluation["selected_threshold"],
        "model_features_count": int(prepared["X_train"].shape[1]),
        "leakage_prevention": {
            "preprocessing_fit": "Train split only.",
            "feature_engineering_fit": (
                "Train split only."
                if model_type == "feature_engineered_lgbm"
                else "Not used."
            ),
            "early_stopping": "Validation split only.",
            "threshold_selection": "Validation split only.",
            "test_usage": "Final holdout evaluation only.",
        },
        "interpretation_note": (
            "Stratified holdout and CV shuffle time and usually produce more "
            "optimistic scores than the chronological thesis split."
        ),
    }
    save_json(run_config, output_dir / "run_config.json")

    test_metrics = evaluation["metrics_test_selected"]
    row = {
        "model_type": model_type,
        "split_strategy": strategy,
        "evaluation_scope": "holdout_test",
        "temporal_order_preserved": bool(split_summary["temporal_order_preserved"]),
        "validation_pr_auc": evaluation["metrics_validation_selected"]["average_precision"],
        "test_pr_auc": test_metrics["average_precision"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_mcc": test_metrics["mcc"],
        "selected_threshold": evaluation["selected_threshold"],
        "best_iteration": best_iteration,
        "train_fraud_rate": split_summary["train_fraud_rate"],
        "validation_fraud_rate": split_summary["validation_fraud_rate"],
        "test_fraud_rate": split_summary["test_fraud_rate"],
        "total_features": int(prepared["X_train"].shape[1]),
        "output_dir": str(output_dir),
    }
    release_memory(prepared, model, evaluation)
    return row


def load_holdout_result_row(
    output_dir: Path,
    model_type: str,
    strategy: str,
) -> dict[str, object]:
    metrics_valid = load_json(output_dir / "metrics_validation_selected_threshold.json")
    metrics_test = load_json(output_dir / "metrics_test_selected_threshold.json")
    split_summary = load_json(output_dir / "split_summary.json")
    run_config = load_json(output_dir / "run_config.json")
    return {
        "model_type": model_type,
        "split_strategy": strategy,
        "evaluation_scope": "holdout_test",
        "temporal_order_preserved": bool(split_summary["temporal_order_preserved"]),
        "validation_pr_auc": metrics_valid["average_precision"],
        "test_pr_auc": metrics_test["average_precision"],
        "test_roc_auc": metrics_test["roc_auc"],
        "test_precision": metrics_test["precision"],
        "test_recall": metrics_test["recall"],
        "test_f1": metrics_test["f1"],
        "test_mcc": metrics_test["mcc"],
        "selected_threshold": run_config["selected_threshold"],
        "best_iteration": run_config["best_iteration"],
        "train_fraud_rate": split_summary["train_fraud_rate"],
        "validation_fraud_rate": split_summary["validation_fraud_rate"],
        "test_fraud_rate": split_summary["test_fraud_rate"],
        "total_features": run_config.get("model_features_count"),
        "output_dir": str(output_dir),
    }


def load_cv_result_row(output_dir: Path, model_type: str) -> dict[str, object]:
    summary = load_json(output_dir / "cv_summary.json")
    summary["model_type"] = model_type
    return summary


def run_stratified_cv_experiment(
    full_df: pd.DataFrame,
    model_type: str,
    output_dir: Path,
    n_folds: int,
    random_seed: int,
    n_jobs: int = DEFAULT_APPENDIX_N_JOBS,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)
    splitter = stratified_kfold_splits(
        full_df,
        n_splits=n_folds,
        random_seed=random_seed,
    )

    labels = full_df[TARGET_COL].to_numpy()
    oof_scores = np.zeros(len(full_df), dtype="float32")
    fold_rows: list[dict[str, object]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(np.arange(len(full_df)), labels),
        start=1,
    ):
        log(f"Running stratified CV fold {fold_number}/{n_folds} for {model_type}.")
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        valid_df = full_df.iloc[valid_idx].reset_index(drop=True)
        prepared = prepare_train_valid_matrices(model_type, train_df, valid_df)
        model, model_params, best_iteration = fit_lgbm_classifier(
            prepared,
            log_period=0,
            n_jobs=n_jobs,
        )
        valid_score = model.predict_proba(
            prepared["X_valid"],
            num_iteration=best_iteration,
        )[:, 1]
        y_valid = prepared["y_valid"].to_numpy()
        oof_scores[valid_idx] = valid_score

        fold_rows.append(
            {
                "fold": fold_number,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(valid_idx)),
                "train_fraud_rate": float(labels[train_idx].mean()),
                "validation_fraud_rate": float(labels[valid_idx].mean()),
                "validation_pr_auc": float(average_precision_score(y_valid, valid_score)),
                "validation_roc_auc": float(roc_auc_score(y_valid, valid_score)),
                "best_iteration": best_iteration,
                "scale_pos_weight": model_params["scale_pos_weight"],
            }
        )
        release_memory(train_df, valid_df, prepared, model)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    oof_pr_auc = float(average_precision_score(labels, oof_scores))
    oof_roc_auc = float(roc_auc_score(labels, oof_scores))
    cv_summary = {
        "model_type": model_type,
        "split_strategy": "stratified_kfold",
        "evaluation_scope": "out_of_fold",
        "n_folds": n_folds,
        "mean_fold_pr_auc": float(fold_df["validation_pr_auc"].mean()),
        "std_fold_pr_auc": float(fold_df["validation_pr_auc"].std(ddof=0)),
        "mean_fold_roc_auc": float(fold_df["validation_roc_auc"].mean()),
        "std_fold_roc_auc": float(fold_df["validation_roc_auc"].std(ddof=0)),
        "oof_pr_auc": oof_pr_auc,
        "oof_roc_auc": oof_roc_auc,
        "output_dir": str(output_dir),
    }
    save_json(cv_summary, output_dir / "cv_summary.json")
    pd.DataFrame({"oof_score": oof_scores, TARGET_COL: labels}).to_csv(
        output_dir / "oof_scores.csv",
        index=False,
    )

    run_config = {
        "appendix": "split_strategy_comparison",
        "model_type": model_type,
        "split_strategy": "stratified_kfold",
        "evaluation_scope": "out_of_fold",
        "n_folds": n_folds,
        "random_seed": random_seed,
        "n_jobs": n_jobs,
        "cv_summary": cv_summary,
        "leakage_prevention": {
            "preprocessing_fit": "Train fold only for each split.",
            "feature_engineering_fit": (
                "Train fold only for each split."
                if model_type == "feature_engineered_lgbm"
                else "Not used."
            ),
            "early_stopping": "Held-out fold only.",
            "threshold_selection": "Not used; PR-AUC and ROC-AUC are threshold-free.",
        },
        "interpretation_note": (
            "CV scores are usually higher than chronological test scores because "
            "folds mix transactions from different time periods."
        ),
    }
    save_json(run_config, output_dir / "run_config.json")
    return cv_summary


def print_holdout_comparison(table: pd.DataFrame) -> None:
    print()
    print("Holdout Split Strategy Comparison")
    print("=================================")
    if table.empty:
        print("No holdout comparison rows were produced.")
        return
    for _, row in table.iterrows():
        print(
            f"{row['model_type']} | {row['split_strategy']}: "
            f"test PR-AUC={row['test_pr_auc']:.6f}, "
            f"test ROC-AUC={row['test_roc_auc']:.6f}, "
            f"temporal_order_preserved={row['temporal_order_preserved']}"
        )


def print_cv_summary(table: pd.DataFrame) -> None:
    print()
    print("Stratified CV Summary")
    print("=====================")
    if table.empty:
        print("No CV summary rows were produced.")
        return
    for _, row in table.iterrows():
        print(
            f"{row['model_type']}: "
            f"OOF PR-AUC={row['oof_pr_auc']:.6f}, "
            f"mean fold PR-AUC={row['mean_fold_pr_auc']:.6f} "
            f"+/- {row['std_fold_pr_auc']:.6f}"
        )


def print_deltas(holdout_table: pd.DataFrame, cv_table: pd.DataFrame) -> None:
    print()
    print("Appendix Deltas vs Chronological Holdout")
    print("========================================")
    for model_type in holdout_table["model_type"].drop_duplicates():
        model_rows = holdout_table.loc[holdout_table["model_type"] == model_type]
        chronological = model_rows.loc[model_rows["split_strategy"] == "chronological"]
        stratified = model_rows.loc[model_rows["split_strategy"] == "stratified_holdout"]
        if chronological.empty or stratified.empty:
            print(f"{model_type}: holdout delta not available.")
            continue
        chrono_pr = float(chronological.iloc[0]["test_pr_auc"])
        strat_pr = float(stratified.iloc[0]["test_pr_auc"])
        print(
            f"{model_type}: "
            f"stratified_holdout - chronological = {strat_pr - chrono_pr:+.6f} PR-AUC"
        )

        cv_rows = cv_table.loc[cv_table["model_type"] == model_type]
        if cv_rows.empty:
            continue
        oof_pr = float(cv_rows.iloc[0]["oof_pr_auc"])
        print(
            f"{model_type}: "
            f"stratified_cv_oof - chronological = {oof_pr - chrono_pr:+.6f} PR-AUC"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Appendix experiment comparing chronological, stratified holdout, "
            "and stratified K-fold evaluation strategies."
        )
    )
    parser.add_argument(
        "--model-types",
        nargs="+",
        choices=SUPPORTED_MODEL_TYPES,
        default=["baseline_lgbm"],
        help="Models to compare. Add feature_engineered_lgbm for the FE branch.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row sample for quick local smoke tests.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for stratified holdout and CV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR,
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Run only chronological and stratified holdout comparisons.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed holdout/CV outputs when present.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_APPENDIX_N_JOBS,
        help="LightGBM worker threads for appendix runs. Lower values reduce memory use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_seed)
    output_dir = ensure_dir(args.output_dir)
    sample_size = resolve_sample_size(args.sample_size)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=sample_size)

    holdout_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []

    for model_type in args.model_types:
        for strategy in HOLDOUT_STRATEGIES:
            strategy_dir = output_dir / "holdout" / model_type / strategy
            if args.skip_existing and holdout_experiment_complete(strategy_dir):
                log(f"Skipping completed holdout experiment: {model_type} / {strategy}.")
                holdout_rows.append(
                    load_holdout_result_row(strategy_dir, model_type, strategy)
                )
                continue

            log(f"Running holdout experiment: {model_type} / {strategy}.")
            holdout_rows.append(
                run_holdout_experiment(
                    full_df=full_df,
                    model_type=model_type,
                    strategy=strategy,
                    output_dir=strategy_dir,
                    sample_size=sample_size,
                    random_seed=args.random_seed,
                    n_jobs=args.n_jobs,
                )
            )

        if not args.skip_cv:
            cv_dir = output_dir / "stratified_cv" / model_type
            if args.skip_existing and cv_experiment_complete(cv_dir, args.n_folds):
                log(f"Skipping completed stratified CV experiment: {model_type}.")
                cv_rows.append(load_cv_result_row(cv_dir, model_type))
            else:
                log(f"Running stratified CV experiment: {model_type}.")
                cv_rows.append(
                    run_stratified_cv_experiment(
                        full_df=full_df,
                        model_type=model_type,
                        output_dir=cv_dir,
                        n_folds=args.n_folds,
                        random_seed=args.random_seed,
                        n_jobs=args.n_jobs,
                    )
                )

    holdout_table = pd.DataFrame(holdout_rows, columns=HOLDOUT_COMPARISON_COLUMNS)
    holdout_table.to_csv(SPLIT_STRATEGY_APPENDIX_COMPARISON_FILE, index=False)

    cv_table = pd.DataFrame(cv_rows, columns=CV_SUMMARY_COLUMNS)
    if not cv_table.empty:
        cv_table.to_csv(SPLIT_STRATEGY_APPENDIX_CV_FILE, index=False)

    appendix_summary = {
        "appendix": "split_strategy_comparison",
        "sample_size": sample_size,
        "random_seed": args.random_seed,
        "model_types": args.model_types,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "skip_cv": args.skip_cv,
        "skip_existing": args.skip_existing,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "holdout_comparison_file": str(SPLIT_STRATEGY_APPENDIX_COMPARISON_FILE),
        "cv_summary_file": str(SPLIT_STRATEGY_APPENDIX_CV_FILE) if not cv_table.empty else None,
        "thesis_interpretation": (
            "If stratified holdout and CV scores are materially higher than the "
            "chronological test score, the temporal split is likely reducing "
            "reported performance because later transactions are harder to predict."
        ),
    }
    save_json(appendix_summary, output_dir / "appendix_summary.json")

    print_holdout_comparison(holdout_table)
    print_cv_summary(cv_table)
    print_deltas(holdout_table, cv_table)
    print(f"\nSaved holdout comparison to: {SPLIT_STRATEGY_APPENDIX_COMPARISON_FILE}")
    if not cv_table.empty:
        print(f"Saved CV summary to: {SPLIT_STRATEGY_APPENDIX_CV_FILE}")
    print(f"Appendix outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()