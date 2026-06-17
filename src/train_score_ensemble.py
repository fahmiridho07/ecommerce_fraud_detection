"""Train/evaluate score-level ensembles from fixed experiment components.

This script does not refit LightGBM components. It combines validation/test
probability scores from an existing baseline model and an existing AE-augmented
model using a validation-selected or fixed alpha:

    ensemble_score = (1 - alpha) * baseline_score + alpha * ae_score
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from config import RANDOM_SEED, SAMPLE_SIZE
from data_loader import load_labeled_train_data
from enhanced_preprocessing import apply_enhanced_preprocessing
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from preprocessing import split_features_target
from splitting import chronological_split
from train_ae_lgbm import validate_latent_split_manifest_alignment
from train_baseline_lgbm import DEFAULT_THRESHOLD
from train_enhanced_preprocessing_lgbm import add_latent_features
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_INITIAL_PROPOSAL_DIR = Path("outputs") / "initial_proposal"
DEFAULT_PREPROCESSING_ABLATION_DIR = (
    DEFAULT_INITIAL_PROPOSAL_DIR / "preprocessing_ablation"
)
DEFAULT_BASELINE_DIR = (
    DEFAULT_PREPROCESSING_ABLATION_DIR
    / "baseline_frequency_missingness_time_amount_fixed_p02"
)
DEFAULT_AE_DIR = (
    DEFAULT_PREPROCESSING_ABLATION_DIR
    / "baseline_latent_all_masked_ld32_frequency_missingness_time_amount_fixed_p02"
)
DEFAULT_AUTOENCODER_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "all_masked_autoencoder_ld32"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PREPROCESSING_ABLATION_DIR
    / "score_ensemble_baseline_all_masked_ld32_canonical"
)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_component_scores(
    baseline_dir: Path,
    ae_dir: Path,
    autoencoder_dir: Path,
) -> dict[str, object]:
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    log("Loading baseline component.")
    baseline_preprocessing = joblib.load(baseline_dir / "enhanced_preprocessing.pkl")
    baseline_model = joblib.load(baseline_dir / "final_model.pkl")
    X_valid_baseline = apply_enhanced_preprocessing(X_valid_raw, baseline_preprocessing)
    X_test_baseline = apply_enhanced_preprocessing(X_test_raw, baseline_preprocessing)
    baseline_valid_score = baseline_model.predict_proba(X_valid_baseline)[:, 1]
    baseline_test_score = baseline_model.predict_proba(X_test_baseline)[:, 1]

    log("Loading AE latent component.")
    validate_latent_split_manifest_alignment(
        autoencoder_dir,
        train_df,
        valid_df,
        test_df,
    )
    ae_preprocessing = joblib.load(ae_dir / "enhanced_preprocessing.pkl")
    ae_model = joblib.load(ae_dir / "final_model.pkl")
    latent_valid = np.load(autoencoder_dir / "latent_valid.npy")
    latent_test = np.load(autoencoder_dir / "latent_test.npy")
    latent_feature_names = load_json(autoencoder_dir / "latent_feature_names.json")
    if not isinstance(latent_feature_names, list):
        raise TypeError("latent_feature_names.json must contain a list.")
    if latent_valid.shape[0] != len(valid_df) or latent_test.shape[0] != len(test_df):
        raise ValueError("AE latent row counts do not match chronological split rows.")

    X_valid_ae = apply_enhanced_preprocessing(X_valid_raw, ae_preprocessing)
    X_test_ae = apply_enhanced_preprocessing(X_test_raw, ae_preprocessing)
    X_valid_ae = add_latent_features(X_valid_ae, latent_valid, latent_feature_names)
    X_test_ae = add_latent_features(X_test_ae, latent_test, latent_feature_names)
    ae_valid_score = ae_model.predict_proba(X_valid_ae)[:, 1]
    ae_test_score = ae_model.predict_proba(X_test_ae)[:, 1]

    return {
        "y_valid": y_valid.to_numpy(),
        "y_test": y_test.to_numpy(),
        "baseline_valid_score": baseline_valid_score,
        "baseline_test_score": baseline_test_score,
        "ae_valid_score": ae_valid_score,
        "ae_test_score": ae_test_score,
        "rows": {
            "train": len(train_df),
            "validation": len(valid_df),
            "test": len(test_df),
        },
    }


def combine_scores(
    baseline_score: np.ndarray,
    ae_score: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return ((1.0 - alpha) * baseline_score) + (alpha * ae_score)


def tune_alpha(
    y_valid: np.ndarray,
    baseline_valid_score: np.ndarray,
    ae_valid_score: np.ndarray,
    n_trials: int,
) -> tuple[float, optuna.study.Study]:
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 0.0, 1.0)
        score = combine_scores(baseline_valid_score, ae_valid_score, alpha)
        return float(average_precision_score(y_valid, score))

    study.optimize(objective, n_trials=n_trials)
    return float(study.best_params["alpha"]), study


def paired_bootstrap_delta_ap(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    n_bootstrap: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(RANDOM_SEED)
    row_count = len(y_true)
    deltas = np.empty(n_bootstrap, dtype="float64")
    for index in range(n_bootstrap):
        sample_indices = rng.integers(0, row_count, row_count)
        if np.unique(y_true[sample_indices]).size < 2:
            deltas[index] = np.nan
            continue
        deltas[index] = average_precision_score(
            y_true[sample_indices],
            candidate_score[sample_indices],
        ) - average_precision_score(
            y_true[sample_indices],
            reference_score[sample_indices],
        )
    deltas = deltas[np.isfinite(deltas)]
    return {
        "n_bootstrap": int(len(deltas)),
        "ci_2_5": float(np.percentile(deltas, 2.5)),
        "ci_50": float(np.percentile(deltas, 50)),
        "ci_97_5": float(np.percentile(deltas, 97.5)),
        "p_delta_le_0": float(np.mean(deltas <= 0.0)),
        "deltas": deltas,
    }


def run(args: argparse.Namespace) -> None:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(args.output_dir)

    scores = load_component_scores(
        baseline_dir=args.baseline_dir,
        ae_dir=args.ae_dir,
        autoencoder_dir=args.autoencoder_dir,
    )

    if args.tune_trials > 0:
        log(f"Tuning score ensemble alpha for {args.tune_trials} validation trials.")
        selected_alpha, study = tune_alpha(
            scores["y_valid"],
            scores["baseline_valid_score"],
            scores["ae_valid_score"],
            args.tune_trials,
        )
        trials = study.trials_dataframe()
        trials.to_csv(output_dir / "optuna_alpha_trials.csv", index=False)
        alpha_selection = {
            "mode": "optuna_validation_ap",
            "n_trials": args.tune_trials,
            "best_validation_ap": float(study.best_value),
        }
    else:
        selected_alpha = args.alpha
        alpha_selection = {"mode": "fixed", "n_trials": 0}

    ensemble_valid_score = combine_scores(
        scores["baseline_valid_score"],
        scores["ae_valid_score"],
        selected_alpha,
    )
    ensemble_test_score = combine_scores(
        scores["baseline_test_score"],
        scores["ae_test_score"],
        selected_alpha,
    )

    threshold_table = threshold_selection_table(scores["y_valid"], ensemble_valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_validation_default = binary_classification_metrics(
        scores["y_valid"],
        ensemble_valid_score,
        DEFAULT_THRESHOLD,
    )
    metrics_validation_selected = binary_classification_metrics(
        scores["y_valid"],
        ensemble_valid_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        scores["y_test"],
        ensemble_test_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        scores["y_test"],
        ensemble_test_score,
        selected_threshold,
    )
    save_json(metrics_validation_default, output_dir / "metrics_validation_default_threshold.json")
    save_json(metrics_validation_selected, output_dir / "metrics_validation_selected_threshold.json")
    save_json(metrics_test_default, output_dir / "metrics_test_default_threshold.json")
    save_json(metrics_test_selected, output_dir / "metrics_test_selected_threshold.json")
    confusion_matrix_table(
        scores["y_valid"],
        ensemble_valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        scores["y_test"],
        ensemble_test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    pd.DataFrame(
        {
            "baseline_score": scores["baseline_test_score"],
            "ae_score": scores["ae_test_score"],
            "ensemble_score": ensemble_test_score,
            "isFraud": scores["y_test"],
        }
    ).to_csv(output_dir / "scores_test.csv", index=False)
    pd.DataFrame(
        {
            "baseline_score": scores["baseline_valid_score"],
            "ae_score": scores["ae_valid_score"],
            "ensemble_score": ensemble_valid_score,
            "isFraud": scores["y_valid"],
        }
    ).to_csv(output_dir / "scores_validation.csv", index=False)

    bootstrap_summary: dict[str, object] | None = None
    if args.bootstrap_samples > 0:
        log(f"Running paired bootstrap with {args.bootstrap_samples} samples.")
        bootstrap = paired_bootstrap_delta_ap(
            scores["y_test"],
            scores["baseline_test_score"],
            ensemble_test_score,
            args.bootstrap_samples,
        )
        pd.DataFrame({"delta_ap": bootstrap.pop("deltas")}).to_csv(
            output_dir / "paired_bootstrap_pr_auc_delta.csv",
            index=False,
        )
        bootstrap_summary = {
            "reference": "baseline_component",
            "candidate": "score_ensemble",
            "metric": "average_precision",
            "reference_ap": float(
                average_precision_score(scores["y_test"], scores["baseline_test_score"])
            ),
            "candidate_ap": metrics_test_selected["average_precision"],
            "observed_delta_ap": float(
                metrics_test_selected["average_precision"]
                - average_precision_score(scores["y_test"], scores["baseline_test_score"])
            ),
            **bootstrap,
        }
        save_json(bootstrap_summary, output_dir / "paired_bootstrap_summary.json")

    component_metrics = {
        "baseline_validation_ap": float(
            average_precision_score(scores["y_valid"], scores["baseline_valid_score"])
        ),
        "baseline_test_ap": float(
            average_precision_score(scores["y_test"], scores["baseline_test_score"])
        ),
        "ae_validation_ap": float(
            average_precision_score(scores["y_valid"], scores["ae_valid_score"])
        ),
        "ae_test_ap": float(average_precision_score(scores["y_test"], scores["ae_test_score"])),
    }
    run_config = {
        "phase": "score_level_ensemble",
        "baseline_dir": str(args.baseline_dir),
        "ae_dir": str(args.ae_dir),
        "autoencoder_dir": str(args.autoencoder_dir),
        "output_dir": str(output_dir),
        "score_formula": "ensemble = (1 - alpha) * baseline_score + alpha * ae_score",
        "selected_alpha": selected_alpha,
        "alpha_selection": alpha_selection,
        "threshold_selection": {
            "source_split": "validation",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "component_metrics": component_metrics,
        "metrics_validation_selected": metrics_validation_selected,
        "metrics_test_selected": metrics_test_selected,
        "bootstrap": bootstrap_summary,
        "random_seed": RANDOM_SEED,
        "sample_size": SAMPLE_SIZE,
        "rows": scores["rows"],
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Score Ensemble Summary")
    print("======================")
    print(f"Alpha (AE weight) : {selected_alpha:.6f}")
    print(f"Validation PR-AUC : {metrics_validation_selected['average_precision']:.6f}")
    print(f"Test PR-AUC       : {metrics_test_selected['average_precision']:.6f}")
    print(f"Test ROC-AUC      : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Test F1           : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC          : {metrics_test_selected['mcc']:.6f}")
    if bootstrap_summary is not None:
        print(f"Delta AP vs base  : {bootstrap_summary['observed_delta_ap']:.6f}")
        print(
            "Bootstrap 95% CI  : "
            f"[{bootstrap_summary['ci_2_5']:.6f}, {bootstrap_summary['ci_97_5']:.6f}]"
        )
    print(f"Outputs saved to  : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed or validation-tuned score ensembles."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--ae-dir", type=Path, default=DEFAULT_AE_DIR)
    parser.add_argument("--autoencoder-dir", type=Path, default=DEFAULT_AUTOENCODER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Fixed AE score weight used when --tune-trials is 0.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="Optuna trials for validation-only alpha tuning. Use 0 for fixed alpha.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1].")
    if args.tune_trials < 0:
        raise SystemExit("--tune-trials must be non-negative.")
    if args.bootstrap_samples < 0:
        raise SystemExit("--bootstrap-samples must be non-negative.")
    return args


if __name__ == "__main__":
    run(parse_args())
