"""Run controlled 3-model score ensemble using saved tuned artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import run_fe_ae_score_ensemble as fe_ae_helpers
from config import (
    DATA_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    ID_COL,
    OPTUNA_OUTPUT_DIR,
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
from preprocessing import apply_baseline_preprocessing, split_features_target
from run_fe_ae_score_ensemble import (
    AE_LGBM_LD128_TUNED_DIR,
    EXPECTED_LATENT_DIM,
    FE_TUNED_DIR,
    best_iteration_from_config,
    latent_dim_from_run_config,
    load_json,
    prepare_ae_scores,
    prepare_fe_scores,
    predict_scores,
    require_files,
    validate_model_features,
    validate_same_labels,
    validate_score_length,
    validate_transaction_id_alignment,
)
from splitting import chronological_split
from train_baseline_lgbm import DEFAULT_THRESHOLD
from utils import ensure_dir, log, save_json, set_seed


BASELINE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "baseline_lgbm"
OUTPUT_SUBDIR = OUTPUT_DIR / "three_model_score_ensemble"
COMPARISON_FILE = FINAL_COMPARISON_OUTPUT_DIR / "three_model_ensemble_comparison.csv"
CURRENT_FE_AE_ENSEMBLE_DIR = (
    OUTPUT_DIR / "fe_ae_controlled_experiments" / "A_score_ensemble_fe_tuned_ae_tuned"
)
WEIGHT_STEP = 0.02
WEIGHT_UNITS = int(round(1.0 / WEIGHT_STEP))
METRIC_TOLERANCE = 1e-6
METRIC_CONSISTENCY_RECORDS: list[dict[str, object]] = []


def load_tuned_baseline_artifacts():
    """Load the tuned baseline LightGBM and train-fitted preprocessing."""
    require_files(
        [
            BASELINE_TUNED_DIR / "final_model.pkl",
            BASELINE_TUNED_DIR / "preprocessing.pkl",
            BASELINE_TUNED_DIR / "run_config.json",
        ]
    )
    return (
        joblib.load(BASELINE_TUNED_DIR / "final_model.pkl"),
        joblib.load(BASELINE_TUNED_DIR / "preprocessing.pkl"),
        load_json(BASELINE_TUNED_DIR / "run_config.json"),
    )


def reset_metric_consistency_records() -> None:
    """Clear metric consistency records for one script run."""
    METRIC_CONSISTENCY_RECORDS.clear()


def validate_metric_consistency_record(
    metrics_path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
    strict: bool = False,
) -> None:
    """Validate regenerated AP against stored metrics and record the result."""
    record: dict[str, object] = {
        "model_name": model_name,
        "split": split_name,
        "metrics_path": str(metrics_path),
        "stored_average_precision": None,
        "regenerated_average_precision": None,
        "absolute_difference": None,
        "tolerance": METRIC_TOLERANCE,
        "status": "not_checked",
    }

    if not metrics_path.exists():
        record["status"] = "missing_metrics_file"
        METRIC_CONSISTENCY_RECORDS.append(record)
        return

    stored = load_json(metrics_path)
    stored_ap = stored.get("average_precision")
    if stored_ap is None:
        record["status"] = "missing_average_precision"
        METRIC_CONSISTENCY_RECORDS.append(record)
        return

    regenerated_ap = float(average_precision_score(y_true, y_score))
    difference = abs(regenerated_ap - float(stored_ap))
    record.update(
        {
            "stored_average_precision": float(stored_ap),
            "regenerated_average_precision": regenerated_ap,
            "absolute_difference": difference,
            "status": "match" if difference <= METRIC_TOLERANCE else "mismatch",
        }
    )
    METRIC_CONSISTENCY_RECORDS.append(record)

    if difference > METRIC_TOLERANCE:
        message = (
            f"{model_name} {split_name} regenerated AP {regenerated_ap:.12f} "
            f"does not match stored AP {float(stored_ap):.12f}."
        )
        if strict:
            raise ValueError(message)
        log("WARNING: " + message)


def patch_fe_ae_metric_validator(strict: bool) -> None:
    """Route imported FE/AE helper validation into this run's records."""

    def _patched_validator(
        metrics_path: Path,
        y_true: np.ndarray,
        y_score: np.ndarray,
        model_name: str,
        split_name: str,
    ) -> None:
        validate_metric_consistency_record(
            metrics_path,
            y_true,
            y_score,
            model_name,
            split_name,
            strict=strict,
        )

    fe_ae_helpers.validate_metric_consistency = _patched_validator


def prepare_baseline_scores(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_valid: pd.Series,
    y_test: pd.Series,
    strict_metric_consistency: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Regenerate tuned baseline validation/test probabilities."""
    log("Loading tuned baseline LightGBM artifacts.")
    baseline_model, baseline_preprocessing, baseline_run_config = (
        load_tuned_baseline_artifacts()
    )

    log("Preparing tuned baseline validation/test matrices from saved preprocessing.")
    X_valid_raw, y_valid_baseline = split_features_target(valid_df)
    X_test_raw, y_test_baseline = split_features_target(test_df)
    validate_same_labels(y_valid, y_valid_baseline, "FE validation vs baseline validation")
    validate_same_labels(y_test, y_test_baseline, "FE test vs baseline test")

    X_valid_baseline = apply_baseline_preprocessing(
        X_valid_raw,
        baseline_preprocessing,
    )
    X_test_baseline = apply_baseline_preprocessing(
        X_test_raw,
        baseline_preprocessing,
    )
    validate_model_features(baseline_model, X_valid_baseline, "baseline_lgbm_tuned")
    validate_model_features(baseline_model, X_test_baseline, "baseline_lgbm_tuned")

    best_iteration = best_iteration_from_config(baseline_model, baseline_run_config)
    valid_score = predict_scores(baseline_model, X_valid_baseline, best_iteration)
    test_score = predict_scores(baseline_model, X_test_baseline, best_iteration)
    validate_score_length(valid_score, y_valid, "validation tuned baseline")
    validate_score_length(test_score, y_test, "test tuned baseline")
    validate_metric_consistency_record(
        BASELINE_TUNED_DIR / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        valid_score,
        "baseline_lgbm_tuned",
        "validation",
        strict=strict_metric_consistency,
    )
    validate_metric_consistency_record(
        BASELINE_TUNED_DIR / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        test_score,
        "baseline_lgbm_tuned",
        "test",
        strict=strict_metric_consistency,
    )
    return valid_score, test_score, best_iteration


def build_weight_selection_table(
    y_valid: np.ndarray,
    fe_valid_score: np.ndarray,
    ae_valid_score: np.ndarray,
    baseline_valid_score: np.ndarray,
) -> pd.DataFrame:
    """Search 0.02-spaced 3-model weights on validation PR-AUC only."""
    rows = []
    for fe_units in range(WEIGHT_UNITS + 1):
        for ae_units in range(WEIGHT_UNITS - fe_units + 1):
            baseline_units = WEIGHT_UNITS - fe_units - ae_units
            w_fe = round(fe_units * WEIGHT_STEP, 2)
            w_ae = round(ae_units * WEIGHT_STEP, 2)
            w_base = round(baseline_units * WEIGHT_STEP, 2)
            ensemble_score = (
                w_fe * fe_valid_score
                + w_ae * ae_valid_score
                + w_base * baseline_valid_score
            )
            rows.append(
                {
                    "fe_lgbm_tuned_weight": float(w_fe),
                    "ae_lgbm_ld128_tuned_weight": float(w_ae),
                    "baseline_lgbm_tuned_weight": float(w_base),
                    "validation_average_precision": float(
                        average_precision_score(y_valid, ensemble_score)
                    ),
                }
            )

    table = pd.DataFrame(rows)
    best_index = table.sort_values(
        [
            "validation_average_precision",
            "fe_lgbm_tuned_weight",
            "ae_lgbm_ld128_tuned_weight",
        ],
        ascending=[False, False, False],
    ).index[0]
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    return table


def selected_weights_from_table(weight_table: pd.DataFrame) -> dict[str, float]:
    """Return the validation-selected ensemble weights."""
    selected = weight_table.loc[weight_table["selected"]]
    if selected.empty:
        raise ValueError("No selected ensemble weights found.")

    row = selected.iloc[0]
    return {
        "fe_lgbm_tuned": float(row["fe_lgbm_tuned_weight"]),
        "ae_lgbm_ld128_tuned": float(row["ae_lgbm_ld128_tuned_weight"]),
        "baseline_lgbm_tuned": float(row["baseline_lgbm_tuned_weight"]),
    }


def weighted_score(
    weights: dict[str, float],
    fe_score: np.ndarray,
    ae_score: np.ndarray,
    baseline_score: np.ndarray,
) -> np.ndarray:
    """Apply selected weights to one split."""
    return (
        weights["fe_lgbm_tuned"] * fe_score
        + weights["ae_lgbm_ld128_tuned"] * ae_score
        + weights["baseline_lgbm_tuned"] * baseline_score
    )


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    fe_score: np.ndarray,
    ae_score: np.ndarray,
    baseline_score: np.ndarray,
    ensemble_score: np.ndarray,
) -> None:
    """Save per-row component and ensemble scores for auditability."""
    pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "fe_lgbm_tuned_score": fe_score,
            "ae_lgbm_ld128_tuned_score": ae_score,
            "baseline_lgbm_tuned_score": baseline_score,
            "ensemble_score": ensemble_score,
        }
    ).to_csv(path, index=False)


def selected_metrics_row(model_name: str, output_dir: Path) -> dict[str, object] | None:
    """Build a compact comparison row from an experiment output directory."""
    valid_metrics_path = output_dir / "metrics_validation_selected_threshold.json"
    test_metrics_path = output_dir / "metrics_test_selected_threshold.json"
    run_config_path = output_dir / "run_config.json"
    if (
        not valid_metrics_path.exists()
        or not test_metrics_path.exists()
        or not run_config_path.exists()
    ):
        return None

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)
    run_config = load_json(run_config_path)
    ensemble = run_config.get("ensemble", {})
    if not isinstance(ensemble, dict):
        ensemble = {}

    return {
        "model_name": model_name,
        "validation_pr_auc": valid_metrics.get("average_precision"),
        "test_pr_auc": test_metrics.get("average_precision"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_mcc": test_metrics.get("mcc"),
        "selected_threshold": test_metrics.get("threshold"),
        "selected_fe_weight": ensemble.get("selected_fe_lgbm_tuned_weight"),
        "selected_ae_weight": ensemble.get("selected_ae_lgbm_ld128_tuned_weight"),
        "selected_baseline_weight": ensemble.get(
            "selected_baseline_lgbm_tuned_weight",
            0.0,
        ),
        "output_dir": str(output_dir),
    }


def save_comparison(output_dir: Path) -> pd.DataFrame:
    """Create the final comparison focused on whether baseline helps FE+AE."""
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    rows = [
        selected_metrics_row(
            "current_best_fe_ae_tuned_score_ensemble",
            CURRENT_FE_AE_ENSEMBLE_DIR,
        ),
        selected_metrics_row("three_model_score_ensemble", output_dir),
    ]
    rows = [row for row in rows if row is not None]
    table = pd.DataFrame(rows)

    if not table.empty:
        reference = table.loc[
            table["model_name"] == "current_best_fe_ae_tuned_score_ensemble"
        ]
        if not reference.empty:
            ref_row = reference.iloc[0]
            for metric in ("test_pr_auc", "test_roc_auc", "test_f1", "test_mcc"):
                table[f"delta_{metric}_vs_current_fe_ae"] = (
                    table[metric].astype(float) - float(ref_row[metric])
                )
        table = table.sort_values(
            ["test_pr_auc", "validation_pr_auc"],
            ascending=[False, False],
        ).reset_index(drop=True)

    table.to_csv(COMPARISON_FILE, index=False)
    return table


def run_experiment(
    output_dir: Path,
    strict_metric_consistency: bool = False,
) -> dict[str, object]:
    """Run the controlled 3-model score-level ensemble."""
    set_seed(RANDOM_SEED)
    reset_metric_consistency_records()
    patch_fe_ae_metric_validator(strict=strict_metric_consistency)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data and recreating chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_transaction_id_alignment(valid_df, test_df)

    fe_valid_score, fe_test_score, y_valid, y_test, fe_best_iteration = prepare_fe_scores(
        train_df,
        valid_df,
        test_df,
    )
    ae_valid_score, ae_test_score, ae_best_iteration, autoencoder_run_config = (
        prepare_ae_scores(train_df, valid_df, test_df, y_valid, y_test)
    )
    baseline_valid_score, baseline_test_score, baseline_best_iteration = (
        prepare_baseline_scores(
            valid_df,
            test_df,
            y_valid,
            y_test,
            strict_metric_consistency=strict_metric_consistency,
        )
    )

    log("Selecting 3-model weights on validation PR-AUC only.")
    weight_table = build_weight_selection_table(
        y_valid.to_numpy(),
        fe_valid_score,
        ae_valid_score,
        baseline_valid_score,
    )
    selected_weights = selected_weights_from_table(weight_table)
    weight_table.to_csv(output_dir / "weight_selection.csv", index=False)

    valid_score = weighted_score(
        selected_weights,
        fe_valid_score,
        ae_valid_score,
        baseline_valid_score,
    )
    test_score = weighted_score(
        selected_weights,
        fe_test_score,
        ae_test_score,
        baseline_test_score,
    )

    log("Selecting classification threshold on validation ensemble score only.")
    threshold_table = threshold_selection_table(y_valid.to_numpy(), valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_valid_selected = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_selected = binary_classification_metrics(
        y_test.to_numpy(),
        test_score,
        selected_threshold,
    )

    log("Saving controlled 3-model ensemble outputs.")
    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_json(
        metrics_test_selected,
        output_dir / "metrics_test_selected_threshold.json",
    )
    confusion_matrix_table(
        y_valid.to_numpy(),
        valid_score,
        {"selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        y_test.to_numpy(),
        test_score,
        {"selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)
    save_scores(
        output_dir / "scores_validation.csv",
        valid_df,
        y_valid,
        fe_valid_score,
        ae_valid_score,
        baseline_valid_score,
        valid_score,
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_df,
        y_test,
        fe_test_score,
        ae_test_score,
        baseline_test_score,
        test_score,
    )

    run_config = {
        "phase": "controlled_three_model_score_ensemble",
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "target_column": TARGET_COL,
        "id_column": ID_COL,
        "time_column": TIME_COL,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "split_row_counts": {
            "train": int(len(train_df)),
            "validation": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "leakage_prevention": {
            "split": "Existing chronological 60/20/20 labeled-train split.",
            "feature_engineering": "Loaded saved train-fitted FE artifacts.",
            "preprocessing": "Loaded saved train-fitted preprocessing artifacts.",
            "weight_selection": "Selected on validation PR-AUC only.",
            "threshold_selection": "Selected on validation ensemble scores only.",
            "test_usage": "Test split used only after weights and threshold were fixed.",
            "kaggle_competition_test_files_used": False,
            "training_performed": False,
        },
        "regenerated_score_validation": {
            "strict_metric_consistency": strict_metric_consistency,
            "stored_metric_checked": (
                "Regenerated validation/test average_precision was checked against "
                "stored metric JSON files for each component model when present."
            ),
            "metric_tolerance": METRIC_TOLERANCE,
            "checks": METRIC_CONSISTENCY_RECORDS,
        },
        "ensemble": {
            "formula": (
                "score = w_fe * fe_score + w_ae * ae_score + "
                "w_base * baseline_score"
            ),
            "weight_grid": "0.00 to 1.00 step 0.02 for each weight, constrained to sum to 1.",
            "objective": "validation average_precision / PR-AUC",
            "tie_break": "Higher FE weight, then higher AE weight.",
            "selected_fe_lgbm_tuned_weight": selected_weights["fe_lgbm_tuned"],
            "selected_ae_lgbm_ld128_tuned_weight": selected_weights[
                "ae_lgbm_ld128_tuned"
            ],
            "selected_baseline_lgbm_tuned_weight": selected_weights[
                "baseline_lgbm_tuned"
            ],
        },
        "input_models": {
            "fe_lgbm_tuned": {
                "output_dir": str(FE_TUNED_DIR),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing.pkl",
                "feature_engineering_file": "feature_engineering.pkl",
                "best_iteration": fe_best_iteration,
            },
            "ae_lgbm_ld128_tuned": {
                "output_dir": str(AE_LGBM_LD128_TUNED_DIR),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing_non_v.pkl",
                "best_iteration": ae_best_iteration,
                "latent_dim_expected": EXPECTED_LATENT_DIM,
                "latent_dim": latent_dim_from_run_config(autoencoder_run_config),
            },
            "baseline_lgbm_tuned": {
                "output_dir": str(BASELINE_TUNED_DIR),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing.pkl",
                "best_iteration": baseline_best_iteration,
            },
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold_reference": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "scores_saved": {
            "validation": "scores_validation.csv",
            "test": "scores_test.csv",
        },
        "comparison_file": str(COMPARISON_FILE),
    }
    save_json(run_config, output_dir / "run_config.json")
    comparison = save_comparison(output_dir)

    print()
    print("Three-Model Score Ensemble Summary")
    print("==================================")
    print(f"Selected FE weight       : {selected_weights['fe_lgbm_tuned']:.2f}")
    print(f"Selected AE weight       : {selected_weights['ae_lgbm_ld128_tuned']:.2f}")
    print(f"Selected baseline weight : {selected_weights['baseline_lgbm_tuned']:.2f}")
    print(f"Validation PR-AUC        : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test PR-AUC              : {metrics_test_selected['average_precision']:.6f}")
    print(f"Test ROC-AUC             : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold       : {selected_threshold:.2f}")
    print(f"Test F1                  : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC                 : {metrics_test_selected['mcc']:.6f}")
    print(f"Outputs saved to         : {output_dir}")
    print(f"Comparison saved to      : {COMPARISON_FILE}")

    return {
        "output_dir": str(output_dir),
        "selected_weights": selected_weights,
        "selected_threshold": selected_threshold,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "comparison": comparison.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controlled score-level ensemble of tuned FE, AE, and baseline models."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_SUBDIR,
        help="Output directory for the controlled 3-model ensemble.",
    )
    parser.add_argument(
        "--strict-metric-consistency",
        action="store_true",
        help=(
            "Fail if regenerated component AP differs from stored metric JSON. "
            "By default mismatches are recorded and the controlled run continues."
        ),
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return run_experiment(
        output_dir=args.output_dir,
        strict_metric_consistency=args.strict_metric_consistency,
    )


if __name__ == "__main__":
    main()
