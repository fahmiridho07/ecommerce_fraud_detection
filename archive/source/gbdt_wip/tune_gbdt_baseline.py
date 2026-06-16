"""Optuna tuning for raw-feature GBDT baselines (LightGBM, XGBoost, CatBoost)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

try:
    import optuna
    from optuna.trial import TrialState
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Optuna is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    DATA_DIR,
    ID_COL,
    OUTPUT_DIR,
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
from gbdt_backends import (
    DEFAULT_THRESHOLD,
    PreparedMatrices,
    SUPPORTED_BACKENDS,
    SUPPORTED_PREPROCESSING_MODES,
    best_iteration_for,
    fit_model,
    fixed_trial_params,
    map_trial_params_to_backend,
    predict_positive_proba,
    prepare_matrices_from_raw_splits,
    save_feature_importance,
    save_model_artifacts,
    suggest_unified_trial_params,
    validation_average_precision,
)
from splitting import chronological_split
from train_gbdt_ae3_integration import (
    DEFAULT_AE3_AUTOENCODER_DIR,
    build_ae3_matrices,
)
from tune_lgbm_optuna import (
    completed_trial_count,
    ensure_sqlite_storage_parent_dir,
    search_space_summary,
)
from utils import ensure_dir, log, save_json, set_seed


GBDT_BASE_OUTPUT_DIR = OUTPUT_DIR / "gbdt_baseline_comparison"
EXPERIMENT_FAMILY = "gbdt_baseline_comparison"
SUPPORTED_TUNING_PROFILES = ("quick", "final")

BACKEND_TUNED_SUBDIRS = {
    "lightgbm": "optuna/LGBM_tuned",
    "xgboost": "optuna/XGB_tuned",
    "catboost": "optuna/CAT_tuned",
}

BACKEND_PHASE_NAMES = {
    "lightgbm": "GBDT-LGBM-TUNE",
    "xgboost": "GBDT-XGB-TUNE",
    "catboost": "GBDT-CAT-TUNE",
}

AE3_TUNED_SUBDIR = "optuna/AE3_tuned"
AE3_PHASE_NAME = "GBDT-WIN-AE3-TUNE"
SUPPORTED_FEATURE_SETS = ("raw", "ae3")

DEFAULT_N_TRIALS = 50
DEFAULT_N_JOBS = 4

REQUIRED_OUTPUT_FILES = [
    "optuna_study.pkl",
    "trials.csv",
    "best_params.json",
    "metrics_validation_default_threshold.json",
    "metrics_validation_selected_threshold.json",
    "metrics_test_default_threshold.json",
    "metrics_test_selected_threshold.json",
    "confusion_matrix_validation.csv",
    "confusion_matrix_test.csv",
    "threshold_selection.csv",
    "feature_importance.csv",
    "final_model.pkl",
    "preprocessing.pkl",
    "run_config.json",
]


def default_output_dir(backend: str, feature_set: str) -> Path:
    if feature_set == "ae3":
        return GBDT_BASE_OUTPUT_DIR / AE3_TUNED_SUBDIR / backend
    return GBDT_BASE_OUTPUT_DIR / BACKEND_TUNED_SUBDIRS[backend]


def prepare_data(
    args: argparse.Namespace,
    train_df,
    valid_df,
    test_df,
) -> tuple[PreparedMatrices, dict[str, object] | None]:
    if args.feature_set == "raw":
        prepared = prepare_matrices_from_raw_splits(
            train_df,
            valid_df,
            test_df,
            backend=args.backend,
            preprocessing_mode=args.preprocessing_mode,
        )
        return prepared, None

    if args.autoencoder_output_dir is None:
        raise ValueError(
            "--autoencoder-output-dir is required when --feature-set=ae3."
        )
    prepared, feature_set_summary = build_ae3_matrices(
        train_df,
        valid_df,
        test_df,
        args.autoencoder_output_dir,
        args.backend,
        args.preprocessing_mode,
    )
    return prepared, feature_set_summary


def output_complete(output_dir: Path, backend: str) -> bool:
    required = list(REQUIRED_OUTPUT_FILES)
    if backend == "lightgbm":
        required.append("final_model.txt")
    elif backend == "xgboost":
        required.append("final_model.json")
    elif backend == "catboost":
        required.append("final_model.cbm")
    if not all((output_dir / file_name).exists() for file_name in required):
        return False

    from gbdt_backends import load_json

    run_config = load_json(output_dir / "run_config.json")
    return bool(run_config.get("final_training_completed", True))


def make_objective(
    prepared: PreparedMatrices,
    tuning_profile: str,
    n_jobs: int,
):
    def objective(trial: optuna.Trial) -> float:
        trial_params = suggest_unified_trial_params(trial, tuning_profile)
        backend_params = map_trial_params_to_backend(
            prepared.backend,
            trial_params,
        )
        params = (
            fixed_trial_params(
                prepared.backend,
                n_jobs,
                preprocessing_mode=prepared.preprocessing_mode,
            )
            | backend_params
        )
        model = fit_model(prepared, params, log_period=0)
        score, best_iteration = validation_average_precision(
            model,
            prepared,
            params,
        )
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("validation_average_precision", score)
        return score

    return objective


def create_or_load_study(args: argparse.Namespace) -> optuna.Study:
    ensure_sqlite_storage_parent_dir(args.storage)
    study_name = args.study_name or f"gbdt_baseline_{args.backend}"
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    return optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
    )


def save_study_outputs(study: optuna.Study, output_dir: Path) -> None:
    joblib.dump(study, output_dir / "optuna_study.pkl")
    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)


def save_best_params(
    study: optuna.Study,
    best_params: dict[str, object],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    payload = {
        "experiment_family": EXPERIMENT_FAMILY,
        "backend": args.backend,
        "tuning_profile": args.tuning_profile,
        "best_trial_number": study.best_trial.number,
        "best_validation_average_precision": float(study.best_value),
        "best_params": study.best_params,
        "final_model_params": best_params,
        "n_trials_completed": completed_trial_count(study),
        "study_name": study.study_name,
        "storage": args.storage,
        "fixed_params": fixed_trial_params(
            args.backend,
            args.n_jobs,
            preprocessing_mode=args.preprocessing_mode,
        ),
        "search_space": search_space_summary(args.tuning_profile),
    }
    save_json(payload, output_dir / "best_params.json")


def train_final_model(
    prepared: PreparedMatrices,
    best_params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    log("Training final model with the best validation PR-AUC parameters.")
    model = fit_model(prepared, best_params, log_period=50)
    best_iteration = best_iteration_for(model, prepared.backend, best_params)

    valid_score = predict_positive_proba(
        model,
        prepared.X_valid,
        prepared.backend,
        best_iteration,
        prepared.cat_feature_indices,
    )
    test_score = predict_positive_proba(
        model,
        prepared.X_test,
        prepared.backend,
        best_iteration,
        prepared.cat_feature_indices,
    )

    threshold_table = threshold_selection_table(
        prepared.y_valid.to_numpy(),
        valid_score,
    )
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_valid_default = binary_classification_metrics(
        prepared.y_valid.to_numpy(),
        valid_score,
        DEFAULT_THRESHOLD,
    )
    metrics_valid_selected = binary_classification_metrics(
        prepared.y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        prepared.y_test.to_numpy(),
        test_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        prepared.y_test.to_numpy(),
        test_score,
        selected_threshold,
    )

    save_json(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_json(metrics_test_default, output_dir / "metrics_test_default_threshold.json")
    save_json(
        metrics_test_selected,
        output_dir / "metrics_test_selected_threshold.json",
    )

    confusion_matrix_table(
        prepared.y_valid.to_numpy(),
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        prepared.y_test.to_numpy(),
        test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    save_feature_importance(
        model,
        prepared.backend,
        output_dir / "feature_importance.csv",
        prepared.X_train.columns.tolist(),
    )
    save_model_artifacts(model, prepared.backend, output_dir, model_stem="final_model")
    joblib.dump(prepared.preprocessing, output_dir / "preprocessing.pkl")

    return {
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
    }


def build_run_config(
    args: argparse.Namespace,
    output_dir: Path,
    prepared: PreparedMatrices,
    study: optuna.Study,
    best_params: dict[str, object],
    final_result: dict[str, object] | None,
    feature_set_summary: dict[str, object] | None,
) -> dict[str, object]:
    final_training_completed = final_result is not None
    experiment_id = (
        AE3_PHASE_NAME if args.feature_set == "ae3" else BACKEND_PHASE_NAMES[args.backend]
    )
    feature_summary = feature_set_summary or {
        "feature_setup": "Raw IEEE-CIS baseline features (432).",
        "original_v_features_retained": True,
        "reconstruction_error_used": False,
        "preprocessing_mode": prepared.preprocessing_mode,
        "backend": prepared.backend,
        "total_final_features": prepared.total_features,
    }
    return {
        "experiment_family": EXPERIMENT_FAMILY,
        "experiment_id": experiment_id,
        "backend": args.backend,
        "feature_set": args.feature_set,
        "tuned": True,
        "phase": args.study_name or experiment_id,
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
        "feature_set_summary": feature_summary,
        "autoencoder_output_dir": (
            str(args.autoencoder_output_dir) if args.feature_set == "ae3" else None
        ),
        "model_features_count": prepared.total_features,
        "preprocessing": {
            "fit_split": "train only",
            "mode": prepared.preprocessing_mode,
            "categorical_columns": prepared.categorical_columns,
            "categorical_columns_count": len(prepared.categorical_columns),
            "native_categorical_indices_used": prepared.cat_feature_indices is not None,
        },
        "leakage_prevention": {
            "train": "Preprocessing fit and model fitting.",
            "validation": "Early stopping, Optuna objective, and threshold selection.",
            "test": "Final evaluation only after best hyperparameters are selected.",
        },
        "optuna": {
            "sampler": "TPESampler",
            "sampler_seed": RANDOM_SEED,
            "direction": "maximize",
            "objective": "validation average_precision",
            "study_name": study.study_name,
            "storage": args.storage,
            "n_trials_requested": args.n_trials,
            "n_trials_total": len(study.trials),
            "n_trials_completed": completed_trial_count(study),
            "tuning_profile": args.tuning_profile,
            "search_space": search_space_summary(args.tuning_profile),
            "best_trial_number": study.best_trial.number,
            "best_validation_average_precision": float(study.best_value),
        },
        "final_training": {
            "completed": final_training_completed,
            "skipped_by_flag": args.skip_final_training,
        },
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": 100,
            "best_iteration": (
                final_result["best_iteration"] if final_training_completed else None
            ),
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "completed": final_training_completed,
            "selected_threshold": (
                final_result["selected_threshold"] if final_training_completed else None
            ),
        },
        "model_params": best_params,
        "final_training_completed": final_training_completed,
    }


def run_tuning(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(
        args.output_dir or default_output_dir(args.backend, args.feature_set)
    )

    if args.skip_existing and output_complete(output_dir, args.backend):
        log(f"Skipping {args.backend}; complete tuned outputs already exist.")
        return

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)

    log(
        f"Building {args.feature_set} feature matrices for "
        f"{args.backend} ({args.preprocessing_mode})."
    )
    prepared, feature_set_summary = prepare_data(args, train_df, valid_df, test_df)

    log("Creating/loading Optuna study.")
    study = create_or_load_study(args)

    log(
        "Running Optuna/TPE optimization on validation average precision; "
        "test split remains untouched."
    )
    study.optimize(
        make_objective(prepared, args.tuning_profile, args.n_jobs),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    if completed_trial_count(study) == 0:
        raise RuntimeError("Optuna finished without any completed trials.")

    save_study_outputs(study, output_dir)

    backend_params = map_trial_params_to_backend(args.backend, study.best_params)
    best_params = (
        fixed_trial_params(
            args.backend,
            args.n_jobs,
            preprocessing_mode=args.preprocessing_mode,
        )
        | backend_params
    )
    save_best_params(study, best_params, output_dir, args)

    if args.skip_final_training:
        run_config = build_run_config(
            args,
            output_dir,
            prepared,
            study,
            best_params,
            final_result=None,
            feature_set_summary=feature_set_summary,
        )
        save_json(run_config, output_dir / "run_config.json")
        print()
        print("Optuna Study-Only Summary")
        print("=========================")
        print(f"Backend                : {args.backend}")
        print(f"Completed trials       : {completed_trial_count(study)}")
        print(f"Best validation PR-AUC : {study.best_value:.6f}")
        print(f"Outputs saved to       : {output_dir}")
        return

    final_result = train_final_model(prepared, best_params, output_dir)
    run_config = build_run_config(
        args,
        output_dir,
        prepared,
        study,
        best_params,
        final_result,
        feature_set_summary=feature_set_summary,
    )
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("GBDT Tuning Summary")
    print("===================")
    print(f"Backend                : {args.backend}")
    print(f"Completed trials       : {completed_trial_count(study)}")
    print(f"Best validation PR-AUC : {study.best_value:.6f}")
    print(
        "Test PR-AUC             : "
        f"{final_result['metrics_test_selected']['average_precision']:.6f}"
    )
    print(f"Outputs saved to       : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna tuning for raw-feature GBDT baselines."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=SUPPORTED_BACKENDS,
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--tuning-profile",
        choices=SUPPORTED_TUNING_PROFILES,
        default="final",
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--preprocessing-mode",
        choices=SUPPORTED_PREPROCESSING_MODES,
        default="native",
    )
    parser.add_argument(
        "--feature-set",
        choices=SUPPORTED_FEATURE_SETS,
        default="raw",
        help="raw = 432 baseline features; ae3 = reconstruction-error augmentation.",
    )
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=DEFAULT_AE3_AUTOENCODER_DIR,
        help="LD128 AE output directory (required semantics for --feature-set=ae3).",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-final-training", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_tuning(parse_args())