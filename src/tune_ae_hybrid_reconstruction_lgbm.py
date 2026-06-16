"""Tune hybrid AE-LightGBM with global AE reconstruction-error features.

This experiment keeps the strongest supervised tabular signals, adds LD32 AE
latent features for the replaced V columns, and appends the Autoencoder's global
reconstruction error as an anomaly score. It is intentionally kept separate from
the canonical P01-P04 thesis pipeline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

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
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from splitting import chronological_split
from train_ae_lgbm import prepare_ae_lgbm_training_data
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
DEFAULT_OUTPUT_DIR = (
    DEFAULT_INITIAL_PROPOSAL_DIR / "optuna" / "ae_lgbm_ld32_top25v_recon_tuned"
)
DEFAULT_AUTOENCODER_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "autoencoder_robust_ld32"
DEFAULT_BASELINE_IMPORTANCE_PATH = (
    DEFAULT_INITIAL_PROPOSAL_DIR / "baseline_lgbm_default" / "feature_importance.csv"
)
DEFAULT_PARAMS_JSON = (
    DEFAULT_INITIAL_PROPOSAL_DIR
    / "optuna"
    / "ae_lgbm_ld32_top25v_tuned"
    / "best_params.json"
)
SUPPORTED_TUNING_PROFILES = ("quick", "final")
TUNING_PROFILES = {
    "quick": {
        "num_leaves": {"kind": "int", "low": 31, "high": 128},
        "max_depth": {"kind": "categorical", "choices": [-1, 4, 6, 8, 10]},
        "learning_rate": {"kind": "float", "low": 0.02, "high": 0.08, "log": True},
        "n_estimators": {"kind": "int", "low": 500, "high": 1200},
        "min_child_samples": {"kind": "int", "low": 30, "high": 200},
        "subsample": {"kind": "float", "low": 0.7, "high": 1.0},
        "subsample_freq": {"kind": "int", "low": 1, "high": 5},
        "colsample_bytree": {"kind": "float", "low": 0.7, "high": 1.0},
        "reg_alpha": {"kind": "float", "low": 0.0, "high": 5.0},
        "reg_lambda": {"kind": "float", "low": 0.0, "high": 10.0},
        "scale_pos_weight": {"kind": "float", "low": 1.0, "high": 35.0},
    },
    "final": {
        "num_leaves": {"kind": "int", "low": 16, "high": 256},
        "max_depth": {
            "kind": "categorical",
            "choices": [-1, 3, 4, 5, 6, 7, 8, 10, 12],
        },
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.10, "log": True},
        "n_estimators": {"kind": "int", "low": 700, "high": 2000},
        "min_child_samples": {"kind": "int", "low": 20, "high": 300},
        "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
        "subsample_freq": {"kind": "int", "low": 1, "high": 10},
        "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"kind": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"kind": "float", "low": 0.0, "high": 10.0},
        "scale_pos_weight": {"kind": "float", "low": 1.0, "high": 50.0},
    },
}


@dataclass
class PreparedHybridReconData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    categorical_columns: list[str]
    preprocessing: dict[str, object]
    preprocessing_filename: str
    feature_info: dict[str, object]

    @property
    def total_features(self) -> int:
        return int(self.X_train.shape[1])


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def add_reconstruction_features(X: pd.DataFrame, errors: np.ndarray) -> pd.DataFrame:
    error_frame = reconstruction_error_features(errors)
    return pd.concat(
        [X.reset_index(drop=True), error_frame.reset_index(drop=True)],
        axis=1,
    )


def validate_hybrid_reconstruction_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    base_feature_count: int,
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test columns do not align with train columns.")
    if X_train.columns.duplicated().any():
        duplicates = X_train.columns[X_train.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns found: {duplicates[:10]}")
    required = [RECONSTRUCTION_ERROR_FEATURE, LOG_RECONSTRUCTION_ERROR_FEATURE]
    missing = [column for column in required if column not in X_train.columns]
    if missing:
        raise ValueError("Missing reconstruction feature(s): " + ", ".join(missing))
    expected_count = base_feature_count + len(required)
    if X_train.shape[1] != expected_count:
        raise ValueError(
            "Unexpected feature count after adding reconstruction features: "
            f"{X_train.shape[1]} vs {expected_count}."
        )


def prepare_data(
    autoencoder_output_dir: Path,
    reconstruction_error_dir: Path,
    retain_top_v_features: int,
    baseline_importance_path: Path,
) -> PreparedHybridReconData:
    log("Preparing hybrid LD32 AE-LightGBM matrices.")
    prepared = prepare_ae_lgbm_training_data(
        autoencoder_output_dir=autoencoder_output_dir,
        retain_top_v_features=retain_top_v_features,
        baseline_importance_path=baseline_importance_path,
    )

    log("Loading labeled splits for row IDs and reconstruction-error validation.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Appending global AE reconstruction-error features.")
    errors = load_reconstruction_errors(reconstruction_error_dir)
    validate_reconstruction_error_lengths(
        errors,
        train_rows=len(train_df),
        valid_rows=len(valid_df),
        test_rows=len(test_df),
    )

    base_feature_count = prepared.total_features
    X_train = add_reconstruction_features(prepared.X_train, errors["train"])
    X_valid = add_reconstruction_features(prepared.X_valid, errors["validation"])
    X_test = add_reconstruction_features(prepared.X_test, errors["test"])
    validate_hybrid_reconstruction_features(
        X_train,
        X_valid,
        X_test,
        base_feature_count=base_feature_count,
    )

    robust_preprocessing = prepared.robust_ae_run_config.get("preprocessing", {})
    feature_info = {
        "feature_setup": (
            "Hybrid AE-LightGBM LD32 top-V retention plus global AE "
            "reconstruction-error anomaly score."
        ),
        "representation_mode": prepared.representation_mode,
        "autoencoder_output_dir": str(autoencoder_output_dir),
        "reconstruction_error_dir": str(reconstruction_error_dir),
        "retain_top_v_features": retain_top_v_features,
        "baseline_importance_path": str(baseline_importance_path),
        "retained_original_v_features": prepared.retained_v_columns,
        "replaced_original_v_feature_count": len(prepared.replaced_v_columns),
        "original_v_feature_count": len(prepared.v_columns),
        "non_v_feature_count": len(prepared.preprocessing_non_v["feature_columns"]),
        "latent_feature_count": len(prepared.latent_feature_names),
        "v_missing_indicator_count": len(prepared.missing_indicator_names),
        "global_reconstruction_error_features": [
            RECONSTRUCTION_ERROR_FEATURE,
            LOG_RECONSTRUCTION_ERROR_FEATURE,
        ],
        "total_feature_count_before_reconstruction_error": base_feature_count,
        "total_feature_count": int(X_train.shape[1]),
        "robust_autoencoder_clipping": {
            "enabled": robust_preprocessing.get("scaled_clipping_enabled"),
            "clip_min": robust_preprocessing.get("clip_min"),
            "clip_max": robust_preprocessing.get("clip_max"),
        },
        "categorical_columns": prepared.categorical_columns,
        "categorical_columns_count": len(prepared.categorical_columns),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }

    return PreparedHybridReconData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=prepared.y_train,
        y_valid=prepared.y_valid,
        y_test=prepared.y_test,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        categorical_columns=prepared.categorical_columns,
        preprocessing=prepared.preprocessing_non_v,
        preprocessing_filename="preprocessing_non_v.pkl",
        feature_info=feature_info,
    )


def _suggest_from_spec(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, object],
) -> object:
    kind = spec["kind"]
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
        )
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported Optuna spec kind for {name}: {kind}")


def fixed_lgbm_params(n_jobs: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_jobs": n_jobs,
        "random_state": RANDOM_SEED,
        "metric": "None",
        "verbosity": -1,
    }


def suggest_lgbm_params(
    trial: optuna.Trial,
    tuning_profile: str,
) -> dict[str, object]:
    space = TUNING_PROFILES[tuning_profile]
    return {
        name: _suggest_from_spec(trial, name, spec)
        for name, spec in space.items()
    }


def fit_lgbm(
    prepared: PreparedHybridReconData,
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
                verbose=log_period > 0,
            ),
            lgb.log_evaluation(period=log_period),
        ],
    )
    return model


def best_iteration_for(model: lgb.LGBMClassifier, params: dict[str, object]) -> int:
    return int(model.best_iteration_ or params["n_estimators"])


def validation_average_precision(
    model: lgb.LGBMClassifier,
    prepared: PreparedHybridReconData,
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
    prepared: PreparedHybridReconData,
    tuning_profile: str,
    n_jobs: int,
):
    def objective(trial: optuna.Trial) -> float:
        params = fixed_lgbm_params(n_jobs) | suggest_lgbm_params(
            trial,
            tuning_profile,
        )
        model = fit_lgbm(prepared, params, log_period=0)
        score, best_iteration = validation_average_precision(model, prepared, params)
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("validation_average_precision", score)
        return score

    return objective


def completed_trial_count(study: optuna.Study) -> int:
    return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)


def load_params_from_json(path: Path, n_jobs: int) -> dict[str, object]:
    payload = load_json(path)
    if "final_model_params" in payload:
        params = dict(payload["final_model_params"])
    elif "best_params" in payload:
        params = fixed_lgbm_params(n_jobs) | dict(payload["best_params"])
    else:
        raise KeyError(f"{path} must contain final_model_params or best_params.")
    params["n_jobs"] = n_jobs
    params["metric"] = "None"
    return params


def create_or_load_study(args: argparse.Namespace) -> optuna.Study:
    storage = args.storage
    if storage and storage.startswith("sqlite:///"):
        db_path = Path(storage.removeprefix("sqlite:///"))
        if str(db_path) != ":memory:":
            ensure_dir(db_path.parent)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    return optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=bool(storage),
        direction="maximize",
        sampler=sampler,
    )


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    score: np.ndarray,
    X: pd.DataFrame,
) -> None:
    payload = {
        ID_COL: split_df[ID_COL].to_numpy(),
        TARGET_COL: y.to_numpy(),
        "score": score,
        RECONSTRUCTION_ERROR_FEATURE: X[RECONSTRUCTION_ERROR_FEATURE].to_numpy(),
        LOG_RECONSTRUCTION_ERROR_FEATURE: X[
            LOG_RECONSTRUCTION_ERROR_FEATURE
        ].to_numpy(),
    }
    pd.DataFrame(payload).to_csv(path, index=False)


def train_final_model(
    prepared: PreparedHybridReconData,
    params: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    log("Training final hybrid AE reconstruction model.")
    model = fit_lgbm(prepared, params, log_period=50)
    best_iteration = best_iteration_for(model, params)

    log("Generating validation and test probabilities.")
    valid_score = model.predict_proba(
        prepared.X_valid,
        num_iteration=best_iteration,
    )[:, 1]
    test_score = model.predict_proba(
        prepared.X_test,
        num_iteration=best_iteration,
    )[:, 1]

    log("Selecting classification threshold on validation only.")
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

    log("Saving final model artifacts.")
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

    save_scores(
        output_dir / "scores_validation.csv",
        prepared.valid_df,
        prepared.y_valid,
        valid_score,
        prepared.X_valid,
    )
    save_scores(
        output_dir / "scores_test.csv",
        prepared.test_df,
        prepared.y_test,
        test_score,
        prepared.X_test,
    )
    save_feature_importance(model, output_dir / "feature_importance.csv")
    joblib.dump(model, output_dir / "final_model.pkl")
    model.booster_.save_model(str(output_dir / "final_model.txt"))
    joblib.dump(prepared.preprocessing, output_dir / prepared.preprocessing_filename)

    return {
        "model": model,
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
    }


def reference_metrics(initial_proposal_dir: Path) -> dict[str, dict[str, object]]:
    references = {
        "p01_baseline_default": (
            initial_proposal_dir
            / "baseline_lgbm_default"
            / "metrics_test_selected_threshold.json"
        ),
        "p02_baseline_tuned": (
            initial_proposal_dir
            / "optuna"
            / "baseline_lgbm_tuned"
            / "metrics_test_selected_threshold.json"
        ),
        "ae_hybrid_top25_tuned": (
            initial_proposal_dir
            / "optuna"
            / "ae_lgbm_ld32_top25v_tuned"
            / "metrics_test_selected_threshold.json"
        ),
        "ae_reconstruction_error_ld128_default": (
            initial_proposal_dir
            / "ae_reconstruction_error_ld128_default"
            / "metrics_test_selected_threshold.json"
        ),
    }
    loaded: dict[str, dict[str, object]] = {}
    for name, path in references.items():
        if path.exists():
            loaded[name] = load_json(path)
    return loaded


def build_reference_comparison(
    metrics_test_selected: dict[str, object],
    references: dict[str, dict[str, object]],
) -> dict[str, object]:
    rows = {}
    for name, metrics in references.items():
        rows[name] = {
            "reference_average_precision": metrics.get("average_precision"),
            "delta_average_precision": (
                float(metrics_test_selected["average_precision"])
                - float(metrics["average_precision"])
            ),
            "reference_roc_auc": metrics.get("roc_auc"),
            "delta_roc_auc": (
                float(metrics_test_selected["roc_auc"]) - float(metrics["roc_auc"])
            ),
            "reference_f1": metrics.get("f1"),
            "delta_f1": float(metrics_test_selected["f1"]) - float(metrics["f1"]),
            "reference_mcc": metrics.get("mcc"),
            "delta_mcc": float(metrics_test_selected["mcc"]) - float(metrics["mcc"]),
        }
    return rows


def build_run_config(
    args: argparse.Namespace,
    prepared: PreparedHybridReconData,
    output_dir: Path,
    params: dict[str, object],
    final_result: dict[str, object],
    study: optuna.Study | None,
) -> dict[str, object]:
    return {
        "phase": "ae_hybrid_reconstruction_lgbm",
        "model_type": "ae_lgbm_ld32_hybrid_reconstruction_error",
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
        "leakage_prevention": {
            "train": "Preprocessing fit, Autoencoder training artifacts, and LightGBM fitting.",
            "validation": "Early stopping, Optuna objective, and threshold selection.",
            "test": "Final evaluation only after selected params and validation threshold.",
        },
        "literature_alignment": {
            "hybrid_ae_boosting": (
                "Follows AE plus gradient boosting precedent: AE contributes "
                "learned representation/anomaly score while LightGBM handles "
                "tabular supervised classification."
            ),
            "reconstruction_error": (
                "Uses global reconstruction error as an anomaly score instead "
                "of replacing all original V-features."
            ),
            "primary_metric": (
                "Optimizes validation average precision / PR-AUC for the "
                "imbalanced fraud setting."
            ),
        },
        "feature_construction": prepared.feature_info,
        "model_features_count": prepared.total_features,
        "preprocessing_file": prepared.preprocessing_filename,
        "training_mode": "optuna" if study is not None else "fixed_params",
        "params_json": str(args.params_json) if args.params_json else None,
        "optuna": (
            {
                "sampler": "TPESampler",
                "sampler_seed": RANDOM_SEED,
                "direction": "maximize",
                "objective": "validation average_precision / PR-AUC",
                "study_name": study.study_name,
                "storage": args.storage,
                "n_trials_requested_this_run": args.n_trials,
                "n_trials_total": len(study.trials),
                "n_trials_completed": completed_trial_count(study),
                "timeout_seconds": args.timeout,
                "tuning_profile": args.tuning_profile,
                "search_space": TUNING_PROFILES[args.tuning_profile],
                "best_trial_validation_best_iteration": (
                    study.best_trial.user_attrs.get("best_iteration")
                ),
                "best_validation_average_precision": float(study.best_value),
            }
            if study is not None
            else None
        ),
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": final_result["selected_threshold"],
        },
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": final_result["best_iteration"],
        },
        "model_params": params,
    }


def run(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(args.output_dir)
    prepared = prepare_data(
        autoencoder_output_dir=args.autoencoder_output_dir,
        reconstruction_error_dir=args.reconstruction_error_dir,
        retain_top_v_features=args.retain_top_v_features,
        baseline_importance_path=args.baseline_importance_path,
    )

    study: optuna.Study | None = None
    if args.n_trials > 0:
        log("Running Optuna/TPE search for hybrid + reconstruction features.")
        study = create_or_load_study(args)
        study.optimize(
            make_objective(prepared, args.tuning_profile, args.n_jobs),
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
        )
        if completed_trial_count(study) == 0:
            raise RuntimeError("Optuna finished without any completed trials.")
        joblib.dump(study, output_dir / "optuna_study.pkl")
        study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)
        params = fixed_lgbm_params(args.n_jobs) | study.best_params
        save_json(
            {
                "best_params": study.best_params,
                "fixed_params": fixed_lgbm_params(args.n_jobs),
                "final_model_params": params,
                "best_trial_number": study.best_trial.number,
                "best_trial_user_attrs": study.best_trial.user_attrs,
                "best_validation_average_precision": float(study.best_value),
                "tuning_profile": args.tuning_profile,
                "search_space": TUNING_PROFILES[args.tuning_profile],
            },
            output_dir / "best_params.json",
        )
    else:
        if args.params_json is None:
            raise ValueError("--params-json is required when --n_trials is 0.")
        log(f"Using fixed LightGBM params from {args.params_json}.")
        params = load_params_from_json(args.params_json, n_jobs=args.n_jobs)
        save_json(
            {
                "source_params_json": str(args.params_json),
                "final_model_params": params,
                "note": (
                    "Fixed-parameter diagnostic using existing tuned hybrid "
                    "parameters with two extra reconstruction-error features."
                ),
            },
            output_dir / "best_params.json",
        )

    final_result = train_final_model(prepared, params, output_dir)
    references = reference_metrics(args.initial_proposal_dir)
    comparison = build_reference_comparison(
        final_result["metrics_test_selected"],
        references,
    )
    save_json(comparison, output_dir / "reference_comparison.json")
    run_config = build_run_config(
        args,
        prepared,
        output_dir,
        params,
        final_result,
        study,
    )
    save_json(run_config, output_dir / "run_config.json")

    metrics = final_result["metrics_test_selected"]
    print()
    print("Hybrid AE + Reconstruction LightGBM Summary")
    print("===========================================")
    print(f"Validation PR-AUC : {final_result['metrics_validation_selected']['average_precision']:.6f}")
    print(f"Test PR-AUC       : {metrics['average_precision']:.6f}")
    print(f"Test ROC-AUC      : {metrics['roc_auc']:.6f}")
    print(f"Selected threshold: {final_result['selected_threshold']:.2f}")
    print(f"Test F1           : {metrics['f1']:.6f}")
    print(f"Test MCC          : {metrics['mcc']:.6f}")
    if "p02_baseline_tuned" in comparison:
        print(
            "Delta vs P02 AP    : "
            f"{comparison['p02_baseline_tuned']['delta_average_precision']:+.6f}"
        )
    print(f"Outputs saved to  : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune hybrid LD32 AE-LightGBM with global reconstruction error."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=DEFAULT_AUTOENCODER_DIR,
    )
    parser.add_argument(
        "--reconstruction-error-dir",
        type=Path,
        default=DEFAULT_AUTOENCODER_DIR,
    )
    parser.add_argument("--retain-top-v-features", type=int, default=25)
    parser.add_argument(
        "--baseline-importance-path",
        type=Path,
        default=DEFAULT_BASELINE_IMPORTANCE_PATH,
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_PARAMS_JSON,
        help="Existing tuned parameter JSON used when --n_trials is 0.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=0,
        help="Optuna trials to run. Use 0 for fixed-parameter diagnostic.",
    )
    parser.add_argument(
        "--tuning_profile",
        choices=SUPPORTED_TUNING_PROFILES,
        default="final",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument(
        "--study-name",
        default="phase5_ae_lgbm_ld32_hybrid_reconstruction_error",
    )
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=DEFAULT_INITIAL_PROPOSAL_DIR,
    )
    args = parser.parse_args()
    if args.n_trials < 0:
        raise SystemExit("--n_trials must be zero or a positive integer.")
    if args.n_jobs == 0:
        raise SystemExit("--n-jobs must be non-zero.")
    if args.retain_top_v_features <= 0:
        raise SystemExit("--retain-top-v-features must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
