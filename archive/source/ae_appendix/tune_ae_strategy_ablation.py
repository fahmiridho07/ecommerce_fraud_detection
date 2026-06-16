"""Optuna tuning for the LD128 AE strategy ablation winners (TUNE-B0, TUNE-AE3 only)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import optuna
    from optuna.trial import TrialState
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Optuna is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

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
from preprocessing import get_v_feature_columns, split_features_target
from splitting import chronological_split
from train_ae_integration_strategy_ablation import (
    RECONSTRUCTED_V_NAME_PREFIX,
    RECONSTRUCTION_ERROR_FEATURES,
    build_baseline_features,
    build_reconstruction_error_features,
)
from train_baseline_lgbm import (
    average_precision_eval,
    roc_auc_eval,
    save_feature_importance,
)
from tune_lgbm_optuna import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    completed_trial_count,
    ensure_sqlite_storage_parent_dir,
    fixed_lgbm_params,
    search_space_summary,
    suggest_lgbm_params,
)
from utils import ensure_dir, log, save_json, set_seed


TUNING_BASE_OUTPUT_DIR = (
    OUTPUT_DIR / "ae_integration_strategy_ablation_ld128" / "optuna"
)
EXPERIMENT_FAMILY = "ae_strategy_tuning"

SUPPORTED_VARIANTS = (
    "baseline_lgbm_tuned",
    "ae3_reconstruction_error_lgbm_ld128_tuned",
)
SUPPORTED_TUNING_PROFILES = ("quick", "final")

VARIANT_DEFAULT_SUBDIRS = {
    "baseline_lgbm_tuned": "baseline_lgbm_tuned",
    "ae3_reconstruction_error_lgbm_ld128_tuned": "AE3_reconstruction_error_tuned",
}

DEFAULT_N_TRIALS = 50
DEFAULT_N_JOBS = 4


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    categorical_columns: list[str]
    preprocessing: dict[str, object]
    feature_set_summary: dict[str, object]

    @property
    def total_features(self) -> int:
        return int(self.X_train.shape[1])


def requires_autoencoder_output_dir(variant: str) -> bool:
    return variant == "ae3_reconstruction_error_lgbm_ld128_tuned"


def default_output_dir(variant: str) -> Path:
    return TUNING_BASE_OUTPUT_DIR / VARIANT_DEFAULT_SUBDIRS[variant]


def validate_ae3_tuned_feature_matrix(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Ensure AE3 tuned matrices contain only original features + recon-error columns."""
    for split_name, X in (
        ("train", X_train),
        ("validation", X_valid),
        ("test", X_test),
    ):
        latent_columns = [
            column for column in X.columns if str(column).startswith("ae_latent_")
        ]
        if latent_columns:
            raise ValueError(
                f"{split_name} tuned AE3 matrix must not include latent features: "
                + ", ".join(latent_columns[:10])
            )

        reconstructed_columns = [
            column
            for column in X.columns
            if str(column).startswith(RECONSTRUCTED_V_NAME_PREFIX)
        ]
        if reconstructed_columns:
            raise ValueError(
                f"{split_name} tuned AE3 matrix must not include reconstructed V "
                f"features: " + ", ".join(reconstructed_columns[:10])
            )

        missing_error_features = [
            feature
            for feature in RECONSTRUCTION_ERROR_FEATURES
            if feature not in X.columns
        ]
        if missing_error_features:
            raise ValueError(
                f"{split_name} tuned AE3 matrix is missing reconstruction-error "
                f"feature(s): " + ", ".join(missing_error_features)
            )

    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")


def prepare_baseline_lgbm_tuned_data() -> PreparedData:
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Building baseline feature matrices with train-only preprocessing.")
    X_train, X_valid, X_test, preprocessing, categorical_columns = build_baseline_features(
        train_df,
        valid_df,
        test_df,
    )

    _, y_train = split_features_target(train_df)
    _, y_valid = split_features_target(valid_df)
    _, y_test = split_features_target(test_df)

    feature_set_summary = {
        "experiment_family": EXPERIMENT_FAMILY,
        "variant": "baseline_lgbm_tuned",
        "strategy": "baseline_lgbm_tuned",
        "original_v_features_retained": True,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "reconstruction_error_used": False,
        "total_final_features": int(X_train.shape[1]),
    }

    return PreparedData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=categorical_columns,
        preprocessing=preprocessing,
        feature_set_summary=feature_set_summary,
    )


def prepare_ae3_reconstruction_error_tuned_data(
    autoencoder_output_dir: Path,
) -> PreparedData:
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)

    log("Building STR-AE3 reconstruction-error feature matrices.")
    (
        X_train,
        X_valid,
        X_test,
        preprocessing,
        categorical_columns,
        feature_set_summary,
    ) = build_reconstruction_error_features(
        train_df,
        valid_df,
        test_df,
        autoencoder_output_dir,
    )
    validate_ae3_tuned_feature_matrix(X_train, X_valid, X_test)

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        raise ValueError(
            "AE3 tuned matrix must retain all original V-features; "
            f"retained {len(retained_v_columns)} of {len(v_columns)}."
        )

    _, y_train = split_features_target(train_df)
    _, y_valid = split_features_target(valid_df)
    _, y_test = split_features_target(test_df)

    feature_set_summary = dict(feature_set_summary)
    feature_set_summary.update(
        {
            "experiment_family": EXPERIMENT_FAMILY,
            "variant": "ae3_reconstruction_error_lgbm_ld128_tuned",
            "strategy": "ae3_reconstruction_error_lgbm_ld128_tuned",
            "ablation_source_variant": "reconstruction_error_augmentation",
            "latent_features_used": False,
            "reconstructed_features_used": False,
            "reconstruction_error_used": True,
            "reconstruction_error_features": list(RECONSTRUCTION_ERROR_FEATURES),
            "autoencoder_output_dir": str(autoencoder_output_dir),
        }
    )

    return PreparedData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=categorical_columns,
        preprocessing=preprocessing,
        feature_set_summary=feature_set_summary,
    )


def prepare_data(
    variant: str,
    autoencoder_output_dir: Path | None,
) -> PreparedData:
    if variant == "baseline_lgbm_tuned":
        return prepare_baseline_lgbm_tuned_data()
    if variant == "ae3_reconstruction_error_lgbm_ld128_tuned":
        if autoencoder_output_dir is None:
            raise ValueError(
                "--autoencoder-output-dir is required for variant "
                "'ae3_reconstruction_error_lgbm_ld128_tuned'."
            )
        return prepare_ae3_reconstruction_error_tuned_data(autoencoder_output_dir)
    raise ValueError(f"Unsupported variant: {variant}")


def fit_lgbm(
    prepared: PreparedData,
    params: dict[str, object],
    log_period: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        prepared.X_train,
        prepared.y_train,
        eval_set=[(prepared.X_valid, prepared.y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=prepared.categorical_columns,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.log_evaluation(period=log_period),
        ],
    )
    return model


def best_iteration_for(model: lgb.LGBMClassifier, params: dict[str, object]) -> int:
    return int(model.best_iteration_ or params["n_estimators"])


def validation_average_precision(
    model: lgb.LGBMClassifier,
    prepared: PreparedData,
    params: dict[str, object],
) -> tuple[float, int]:
    best_iteration = best_iteration_for(model, params)
    valid_score = model.predict_proba(
        prepared.X_valid,
        num_iteration=best_iteration,
    )[:, 1]
    score = average_precision_score(prepared.y_valid.to_numpy(), valid_score)
    return float(score), best_iteration


def make_objective(
    prepared: PreparedData,
    tuning_profile: str,
    n_jobs: int,
    device_type: str,
    max_bin: int | None,
):
    def objective(trial: optuna.Trial) -> float:
        params = fixed_lgbm_params(n_jobs, device_type, max_bin) | suggest_lgbm_params(
            trial,
            tuning_profile,
        )
        model = fit_lgbm(prepared, params, log_period=0)
        score, best_iteration = validation_average_precision(model, prepared, params)
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("validation_average_precision", score)
        return score

    return objective


def create_or_load_study(args: argparse.Namespace) -> optuna.Study:
    ensure_sqlite_storage_parent_dir(args.storage)
    study_name = args.study_name or f"ae_strategy_{args.variant}"
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
        "best_trial_number": study.best_trial.number,
        "best_validation_average_precision": float(study.best_value),
        "best_params": study.best_params,
        "final_model_params": best_params,
        "n_trials_completed": completed_trial_count(study),
        "tuning_profile": args.tuning_profile,
        "study_name": study.study_name,
        "storage": args.storage,
        "variant": args.variant,
        "fixed_params": fixed_lgbm_params(
            args.n_jobs,
            args.device_type,
            args.max_bin,
        ),
        "search_space": search_space_summary(args.tuning_profile),
    }
    save_json(payload, output_dir / "best_params.json")


def train_final_model(
    prepared: PreparedData,
    best_params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    log("Training final model with the best validation PR-AUC parameters.")
    model = fit_lgbm(prepared, best_params, log_period=50)
    best_iteration = best_iteration_for(model, best_params)

    valid_score = model.predict_proba(
        prepared.X_valid,
        num_iteration=best_iteration,
    )[:, 1]
    test_score = model.predict_proba(
        prepared.X_test,
        num_iteration=best_iteration,
    )[:, 1]

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

    save_feature_importance(model, output_dir / "feature_importance.csv")
    joblib.dump(model, output_dir / "final_model.pkl")
    model.booster_.save_model(str(output_dir / "final_model.txt"))
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
    prepared: PreparedData,
    study: optuna.Study,
    best_params: dict[str, object],
    final_result: dict[str, object] | None,
) -> dict[str, object]:
    final_training_completed = final_result is not None
    positive_count = int(prepared.y_train.sum())
    negative_count = int(len(prepared.y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    run_config: dict[str, object] = {
        "experiment_family": EXPERIMENT_FAMILY,
        "variant": args.variant,
        "phase": args.study_name or f"ae_strategy_{args.variant}",
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
        "feature_set_summary": prepared.feature_set_summary,
        "model_features_count": prepared.total_features,
        "preprocessing": {
            "fit_split": "train only",
            "categorical_columns": prepared.categorical_columns,
            "categorical_columns_count": len(prepared.categorical_columns),
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
        },
        "class_imbalance": {
            "method": "scale_pos_weight in search space",
            "computed_from": "training labels only for documentation",
            "train_only_value": scale_pos_weight,
        },
        "leakage_prevention": {
            "train": "Preprocessing fit and LightGBM fitting.",
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
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": (
                final_result["best_iteration"] if final_training_completed else None
            ),
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": (
                final_result["selected_threshold"] if final_training_completed else None
            ),
        },
        "final_training_completed": final_training_completed,
        "model_params": best_params,
        "tuning_profile": args.tuning_profile,
        "n_jobs": args.n_jobs,
        "device_type": args.device_type,
        "max_bin": args.max_bin,
    }

    if args.autoencoder_output_dir is not None:
        run_config["autoencoder_output_dir"] = str(args.autoencoder_output_dir)

    return run_config


def run_tuning(args: argparse.Namespace) -> None:
    if args.variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant '{args.variant}'. Supported: {SUPPORTED_VARIANTS}"
        )
    if requires_autoencoder_output_dir(args.variant) and args.autoencoder_output_dir is None:
        raise ValueError(
            f"--autoencoder-output-dir is required for variant '{args.variant}'."
        )

    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(args.output_dir or default_output_dir(args.variant))
    prepared = prepare_data(args.variant, args.autoencoder_output_dir)

    log("Creating/loading Optuna study.")
    study = create_or_load_study(args)

    log(
        "Running Optuna/TPE optimization on validation average precision; "
        "test split remains untouched."
    )
    study.optimize(
        make_objective(
            prepared,
            args.tuning_profile,
            args.n_jobs,
            args.device_type,
            args.max_bin,
        ),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    if completed_trial_count(study) == 0:
        raise RuntimeError("Optuna finished without any completed trials.")

    save_study_outputs(study, output_dir)

    best_params = fixed_lgbm_params(
        args.n_jobs,
        args.device_type,
        args.max_bin,
    ) | study.best_params
    save_best_params(study, best_params, output_dir, args)

    if args.skip_final_training:
        log("Skipping final model training because --skip-final-training is active.")
        run_config = build_run_config(
            args,
            output_dir,
            prepared,
            study,
            best_params,
            final_result=None,
        )
        save_json(run_config, output_dir / "run_config.json")
        print()
        print("AE Strategy Tuning — study only")
        print("==============================")
        print(f"Variant                : {args.variant}")
        print(f"Completed trials       : {completed_trial_count(study)}")
        print(f"Best validation PR-AUC : {study.best_value:.6f}")
        print(f"Best trial             : {study.best_trial.number}")
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
    )
    save_json(run_config, output_dir / "run_config.json")

    metrics = final_result["metrics_test_selected"]
    print()
    print(f"AE Strategy Tuning — {args.variant}")
    print("=" * (24 + len(args.variant)))
    print(f"Best validation PR-AUC : {study.best_value:.6f}")
    print(f"Test PR-AUC            : {metrics['average_precision']:.6f}")
    print(f"Test ROC-AUC           : {metrics['roc_auc']:.6f}")
    print(f"Selected threshold     : {final_result['selected_threshold']:.2f}")
    print(f"Test MCC               : {metrics['mcc']:.6f}")
    print(f"Best iteration         : {final_result['best_iteration']}")
    print(f"Outputs saved to       : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optuna tuning for AE strategy ablation winners: "
            "baseline_lgbm_tuned and ae3_reconstruction_error_lgbm_ld128_tuned."
        )
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=SUPPORTED_VARIANTS,
        help="Tuned variant to optimize.",
    )
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=None,
        help=(
            "Frozen V-only AE LD128 output directory. Required for "
            "ae3_reconstruction_error_lgbm_ld128_tuned."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to "
            "outputs/ae_integration_strategy_ablation_ld128/optuna/<variant>."
        ),
    )
    parser.add_argument("--study-name", default=None)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///path/to/study.db.",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument(
        "--tuning-profile",
        choices=SUPPORTED_TUNING_PROFILES,
        default="quick",
    )
    parser.add_argument("--skip-final-training", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument(
        "--device-type",
        choices=("cpu", "gpu", "cuda"),
        default="cpu",
    )
    parser.add_argument("--max-bin", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials < 1:
        raise SystemExit("--n-trials must be a positive integer.")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs must be non-zero.")
    run_tuning(args)


if __name__ == "__main__":
    main()