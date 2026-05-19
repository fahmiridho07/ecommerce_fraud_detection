"""Run fine-grained FE-LGBM + AE-LightGBM score ensemble calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import (
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    DATA_DIR,
    FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
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
from feature_engineering import apply_entity_time_amount_features, validate_engineered_features
from preprocessing import apply_baseline_preprocessing, get_v_feature_columns, split_features_target
from splitting import chronological_split
from train_ae_lgbm import (
    apply_non_v_preprocessing,
    combine_non_v_and_latent,
    load_robust_latent_outputs,
    split_non_v_features_target,
    validate_feature_alignment,
    validate_latent_outputs,
)
from train_baseline_lgbm import DEFAULT_THRESHOLD
from utils import ensure_dir, log, save_json, set_seed


FE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features"
AE_LGBM_LD128_TUNED_DIR = OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128"
FINE_ENSEMBLE_OUTPUT_DIR = OUTPUT_DIR / "fe_ae_fine_ensemble"
FINE_COMPARISON_FILE = (
    FINAL_COMPARISON_OUTPUT_DIR / "fe_ae_fine_ensemble_comparison.csv"
)

WEIGHT_GRID = np.arange(600, 951, 5, dtype=float) / 1000.0
METRIC_TOLERANCE = 1e-6
EXPECTED_LATENT_DIM = 128

COMPARISON_COLUMNS = [
    "model_name",
    "score_type",
    "validation_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "selected_threshold",
    "selected_fe_weight",
    "selected_ae_weight",
    "output_dir",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s):\n" + "\n".join(missing))


def best_iteration_from_config(model, run_config: dict[str, object]) -> int:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict) and early_stopping.get("best_iteration"):
        return int(early_stopping["best_iteration"])

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration:
        return int(best_iteration)

    model_params = run_config.get("model_params", {})
    if isinstance(model_params, dict) and model_params.get("n_estimators"):
        return int(model_params["n_estimators"])

    n_estimators = getattr(model, "n_estimators", None)
    if n_estimators:
        return int(n_estimators)

    raise ValueError("Could not determine model best_iteration.")


def model_feature_names(model) -> list[str]:
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise ValueError("Loaded model does not expose LightGBM feature names.")


def validate_model_features(model, X: pd.DataFrame, model_name: str) -> None:
    expected = model_feature_names(model)
    observed = X.columns.tolist()
    if expected != observed:
        expected_only = [column for column in expected if column not in observed][:10]
        observed_only = [column for column in observed if column not in expected][:10]
        raise ValueError(
            f"{model_name} feature-name mismatch. "
            f"expected_count={len(expected)}, observed_count={len(observed)}, "
            f"expected_only={expected_only}, observed_only={observed_only}"
        )


def predict_scores(model, X: pd.DataFrame, best_iteration: int) -> np.ndarray:
    return model.predict_proba(X, num_iteration=best_iteration)[:, 1]


def validate_score_length(score: np.ndarray, y: pd.Series, split_name: str) -> None:
    if score.shape[0] != len(y):
        raise ValueError(
            f"{split_name} score length {score.shape[0]} does not match "
            f"label length {len(y)}."
        )


def metric_consistency_warning(
    metrics_path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
) -> dict[str, object] | None:
    if not metrics_path.exists():
        return {
            "model_name": model_name,
            "split": split_name,
            "metric": "average_precision",
            "warning": "stored_metrics_file_missing",
            "path": str(metrics_path),
        }

    stored = load_json(metrics_path)
    stored_ap = stored.get("average_precision")
    if stored_ap is None:
        return {
            "model_name": model_name,
            "split": split_name,
            "metric": "average_precision",
            "warning": "stored_metric_missing",
            "path": str(metrics_path),
        }

    regenerated_ap = float(average_precision_score(y_true, y_score))
    delta = regenerated_ap - float(stored_ap)
    if abs(delta) <= METRIC_TOLERANCE:
        return None

    return {
        "model_name": model_name,
        "split": split_name,
        "metric": "average_precision",
        "warning": "regenerated_metric_mismatch",
        "stored_value": float(stored_ap),
        "regenerated_value": regenerated_ap,
        "absolute_delta": abs(delta),
        "tolerance": METRIC_TOLERANCE,
        "path": str(metrics_path),
    }


def append_metric_warning(
    warnings: list[dict[str, object]],
    metrics_path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
) -> None:
    warning = metric_consistency_warning(
        metrics_path,
        y_true,
        y_score,
        model_name,
        split_name,
    )
    if warning is not None:
        warnings.append(warning)


def validate_same_labels(left: pd.Series, right: pd.Series, label: str) -> None:
    if not left.equals(right):
        raise ValueError(f"Label alignment mismatch: {label}.")


def validate_transaction_id_alignment(valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    if valid_df[ID_COL].duplicated().any():
        raise ValueError("Validation TransactionID values are not unique.")
    if test_df[ID_COL].duplicated().any():
        raise ValueError("Test TransactionID values are not unique.")
    if set(valid_df[ID_COL]) & set(test_df[ID_COL]):
        raise ValueError("Validation and test TransactionID values overlap.")


def load_tuned_fe_artifacts():
    require_files(
        [
            FE_TUNED_DIR / "final_model.pkl",
            FE_TUNED_DIR / "preprocessing.pkl",
            FE_TUNED_DIR / "feature_engineering.pkl",
            FE_TUNED_DIR / "run_config.json",
        ]
    )
    return (
        joblib.load(FE_TUNED_DIR / "final_model.pkl"),
        joblib.load(FE_TUNED_DIR / "preprocessing.pkl"),
        joblib.load(FE_TUNED_DIR / "feature_engineering.pkl"),
        load_json(FE_TUNED_DIR / "run_config.json"),
    )


def load_tuned_ae_lgbm_artifacts():
    require_files(
        [
            AE_LGBM_LD128_TUNED_DIR / "final_model.pkl",
            AE_LGBM_LD128_TUNED_DIR / "preprocessing_non_v.pkl",
            AE_LGBM_LD128_TUNED_DIR / "run_config.json",
        ]
    )
    return (
        joblib.load(AE_LGBM_LD128_TUNED_DIR / "final_model.pkl"),
        joblib.load(AE_LGBM_LD128_TUNED_DIR / "preprocessing_non_v.pkl"),
        load_json(AE_LGBM_LD128_TUNED_DIR / "run_config.json"),
    )


def prepare_fe_scores(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    warnings: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, int]:
    log("Loading tuned FE-LGBM artifacts.")
    fe_model, fe_preprocessing, feature_artifacts, fe_run_config = load_tuned_fe_artifacts()

    log("Preparing FE-LGBM validation/test matrices from saved artifacts.")
    X_train_raw, _ = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    X_train_engineered = apply_entity_time_amount_features(X_train_raw, feature_artifacts)
    X_valid_engineered = apply_entity_time_amount_features(X_valid_raw, feature_artifacts)
    X_test_engineered = apply_entity_time_amount_features(X_test_raw, feature_artifacts)
    validate_engineered_features(
        X_train_engineered,
        X_valid_engineered,
        X_test_engineered,
        feature_artifacts,
    )

    X_valid_fe = apply_baseline_preprocessing(X_valid_engineered, fe_preprocessing)
    X_test_fe = apply_baseline_preprocessing(X_test_engineered, fe_preprocessing)
    validate_model_features(fe_model, X_valid_fe, "fe_lgbm_tuned")
    validate_model_features(fe_model, X_test_fe, "fe_lgbm_tuned")

    best_iteration = best_iteration_from_config(fe_model, fe_run_config)
    valid_score = predict_scores(fe_model, X_valid_fe, best_iteration)
    test_score = predict_scores(fe_model, X_test_fe, best_iteration)
    validate_score_length(valid_score, y_valid, "validation FE-LGBM")
    validate_score_length(test_score, y_test, "test FE-LGBM")
    append_metric_warning(
        warnings,
        FE_TUNED_DIR / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        valid_score,
        "fe_lgbm_tuned",
        "validation",
    )
    append_metric_warning(
        warnings,
        FE_TUNED_DIR / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        test_score,
        "fe_lgbm_tuned",
        "test",
    )
    return valid_score, test_score, y_valid, y_test, best_iteration


def prepare_ae_scores(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_valid: pd.Series,
    y_test: pd.Series,
    warnings: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, int, dict[str, object], int]:
    log("Loading tuned AE-LightGBM LD128 artifacts.")
    ae_model, ae_preprocessing, ae_run_config = load_tuned_ae_lgbm_artifacts()

    log("Loading robust AE LD128 latent features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        autoencoder_run_config,
    ) = load_robust_latent_outputs(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR)
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    if latent_valid.shape[1] != EXPECTED_LATENT_DIM:
        raise ValueError(
            f"Expected LD128 latent features, found {latent_valid.shape[1]}."
        )

    v_columns = get_v_feature_columns(train_df)
    X_train_non_v_raw, _ = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, y_valid_ae = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test_ae = split_non_v_features_target(test_df, v_columns)
    validate_same_labels(y_valid, y_valid_ae, "FE validation vs AE validation")
    validate_same_labels(y_test, y_test_ae, "FE test vs AE test")

    X_train_non_v = apply_non_v_preprocessing(X_train_non_v_raw, ae_preprocessing)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, ae_preprocessing)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, ae_preprocessing)
    X_train_ae = combine_non_v_and_latent(
        X_train_non_v,
        latent_train,
        latent_feature_names,
    )
    X_valid_ae = combine_non_v_and_latent(
        X_valid_non_v,
        latent_valid,
        latent_feature_names,
    )
    X_test_ae = combine_non_v_and_latent(
        X_test_non_v,
        latent_test,
        latent_feature_names,
    )
    validate_feature_alignment(X_train_ae, X_valid_ae, X_test_ae, v_columns)
    validate_model_features(ae_model, X_valid_ae, "ae_lgbm_ld128_tuned")
    validate_model_features(ae_model, X_test_ae, "ae_lgbm_ld128_tuned")

    best_iteration = best_iteration_from_config(ae_model, ae_run_config)
    valid_score = predict_scores(ae_model, X_valid_ae, best_iteration)
    test_score = predict_scores(ae_model, X_test_ae, best_iteration)
    validate_score_length(valid_score, y_valid, "validation AE-LightGBM")
    validate_score_length(test_score, y_test, "test AE-LightGBM")
    append_metric_warning(
        warnings,
        AE_LGBM_LD128_TUNED_DIR / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        valid_score,
        "ae_lgbm_ld128_tuned",
        "validation",
    )
    append_metric_warning(
        warnings,
        AE_LGBM_LD128_TUNED_DIR / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        test_score,
        "ae_lgbm_ld128_tuned",
        "test",
    )
    return (
        valid_score,
        test_score,
        best_iteration,
        autoencoder_run_config,
        len(latent_feature_names),
    )


def build_weight_selection_table(
    y_valid: np.ndarray,
    fe_valid_score: np.ndarray,
    ae_valid_score: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for weight in WEIGHT_GRID:
        ae_weight = 1.0 - weight
        ensemble_score = weight * fe_valid_score + ae_weight * ae_valid_score
        rows.append(
            {
                "fe_lgbm_tuned_weight": round(float(weight), 3),
                "ae_lgbm_ld128_tuned_weight": round(float(ae_weight), 3),
                "validation_average_precision": float(
                    average_precision_score(y_valid, ensemble_score)
                ),
            }
        )

    table = pd.DataFrame(rows)
    best_index = table.sort_values(
        ["validation_average_precision", "fe_lgbm_tuned_weight"],
        ascending=[False, False],
    ).index[0]
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    return table


def selected_fe_weight_from_table(weight_table: pd.DataFrame) -> float:
    selected = weight_table.loc[weight_table["selected"], "fe_lgbm_tuned_weight"]
    if selected.empty:
        raise ValueError("No selected ensemble weight found.")
    return float(selected.iloc[0])


def percentile_rank(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy()


def latent_dim_from_run_config(run_config: dict[str, object]) -> int | None:
    architecture = run_config.get("architecture", {})
    if not isinstance(architecture, dict):
        return None
    encoder = architecture.get("encoder", [])
    if not isinstance(encoder, list) or not encoder:
        return None
    return int(encoder[-1])


def evaluate_ensemble_variant(
    variant_name: str,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    fe_valid_score: np.ndarray,
    ae_valid_score: np.ndarray,
    fe_test_score: np.ndarray,
    ae_test_score: np.ndarray,
    output_dir: Path,
) -> dict[str, object]:
    weight_table = build_weight_selection_table(
        y_valid,
        fe_valid_score,
        ae_valid_score,
    )
    selected_fe_weight = selected_fe_weight_from_table(weight_table)
    selected_ae_weight = 1.0 - selected_fe_weight
    weight_table.to_csv(output_dir / f"{variant_name}_weight_selection.csv", index=False)

    valid_score = selected_fe_weight * fe_valid_score + selected_ae_weight * ae_valid_score
    test_score = selected_fe_weight * fe_test_score + selected_ae_weight * ae_test_score

    threshold_table = threshold_selection_table(y_valid, valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(
        output_dir / f"threshold_selection_{variant_name}.csv",
        index=False,
    )

    metrics_valid = binary_classification_metrics(
        y_valid,
        valid_score,
        selected_threshold,
    )
    metrics_test = binary_classification_metrics(
        y_test,
        test_score,
        selected_threshold,
    )
    save_json(metrics_valid, output_dir / f"metrics_validation_{variant_name}.json")
    save_json(metrics_test, output_dir / f"metrics_test_{variant_name}.json")

    confusion_matrix_table(
        y_valid,
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / f"confusion_matrix_validation_{variant_name}.csv", index=False)
    confusion_matrix_table(
        y_test,
        test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / f"confusion_matrix_test_{variant_name}.csv", index=False)

    return {
        "variant_name": variant_name,
        "selected_fe_weight": selected_fe_weight,
        "selected_ae_weight": selected_ae_weight,
        "selected_threshold": selected_threshold,
        "validation_score": valid_score,
        "test_score": test_score,
        "metrics_validation": metrics_valid,
        "metrics_test": metrics_test,
    }


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    fe_probability: np.ndarray,
    ae_probability: np.ndarray,
    probability_score: np.ndarray,
    fe_rank: np.ndarray,
    ae_rank: np.ndarray,
    rank_score: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "fe_lgbm_tuned_probability": fe_probability,
            "ae_lgbm_ld128_tuned_probability": ae_probability,
            "probability_ensemble_score": probability_score,
            "fe_lgbm_tuned_rank": fe_rank,
            "ae_lgbm_ld128_tuned_rank": ae_rank,
            "rank_ensemble_score": rank_score,
        }
    ).to_csv(path, index=False)


def metric_value(metrics: dict[str, object], key: str) -> object:
    return metrics.get(key)


def weights_from_run_config(run_config: dict[str, object]) -> tuple[object, object]:
    ensemble = run_config.get("ensemble", {})
    if not isinstance(ensemble, dict):
        return None, None
    return (
        ensemble.get("selected_fe_lgbm_tuned_weight"),
        ensemble.get("selected_ae_lgbm_ld128_tuned_weight"),
    )


def comparison_row_from_files(
    model_name: str,
    score_type: str,
    output_dir: Path,
    validation_metrics_file: str,
    test_metrics_file: str,
    run_config_file: str = "run_config.json",
) -> dict[str, object]:
    valid_metrics_path = output_dir / validation_metrics_file
    test_metrics_path = output_dir / test_metrics_file
    run_config_path = output_dir / run_config_file
    require_files([valid_metrics_path, test_metrics_path])

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)
    selected_fe_weight = None
    selected_ae_weight = None
    if run_config_path.exists():
        selected_fe_weight, selected_ae_weight = weights_from_run_config(
            load_json(run_config_path)
        )

    return comparison_row_from_metrics(
        model_name,
        score_type,
        valid_metrics,
        test_metrics,
        selected_fe_weight,
        selected_ae_weight,
        output_dir,
    )


def comparison_row_from_metrics(
    model_name: str,
    score_type: str,
    valid_metrics: dict[str, object],
    test_metrics: dict[str, object],
    selected_fe_weight: object,
    selected_ae_weight: object,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "score_type": score_type,
        "validation_pr_auc": metric_value(valid_metrics, "average_precision"),
        "test_pr_auc": metric_value(test_metrics, "average_precision"),
        "test_roc_auc": metric_value(test_metrics, "roc_auc"),
        "test_precision": metric_value(test_metrics, "precision"),
        "test_recall": metric_value(test_metrics, "recall"),
        "test_f1": metric_value(test_metrics, "f1"),
        "test_mcc": metric_value(test_metrics, "mcc"),
        "selected_threshold": metric_value(test_metrics, "threshold"),
        "selected_fe_weight": selected_fe_weight,
        "selected_ae_weight": selected_ae_weight,
        "output_dir": str(output_dir),
    }


def build_comparison_table(
    probability_result: dict[str, object],
    rank_result: dict[str, object],
    output_dir: Path,
) -> pd.DataFrame:
    rows = [
        comparison_row_from_files(
            "fe_lgbm_tuned",
            "probability",
            FE_TUNED_DIR,
            "metrics_validation_selected_threshold.json",
            "metrics_test_selected_threshold.json",
        ),
        comparison_row_from_files(
            "ae_lgbm_ld128_tuned",
            "probability",
            AE_LGBM_LD128_TUNED_DIR,
            "metrics_validation_selected_threshold.json",
            "metrics_test_selected_threshold.json",
        ),
        comparison_row_from_files(
            "fe_ae_ensemble_current",
            "probability",
            FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
            "metrics_validation_selected_threshold.json",
            "metrics_test_selected_threshold.json",
        ),
        comparison_row_from_metrics(
            "fe_ae_fine_probability_ensemble",
            "probability",
            probability_result["metrics_validation"],
            probability_result["metrics_test"],
            probability_result["selected_fe_weight"],
            probability_result["selected_ae_weight"],
            output_dir,
        ),
        comparison_row_from_metrics(
            "fe_ae_fine_rank_ensemble",
            "rank",
            rank_result["metrics_validation"],
            rank_result["metrics_test"],
            rank_result["selected_fe_weight"],
            rank_result["selected_ae_weight"],
            output_dir,
        ),
    ]
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def run_experiment(output_dir: Path) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)
    warnings: list[dict[str, object]] = []

    log("Loading labeled training data and recreating chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_transaction_id_alignment(valid_df, test_df)

    fe_valid_score, fe_test_score, y_valid, y_test, fe_best_iteration = prepare_fe_scores(
        train_df,
        valid_df,
        test_df,
        warnings,
    )
    (
        ae_valid_score,
        ae_test_score,
        ae_best_iteration,
        autoencoder_run_config,
        latent_feature_count,
    ) = prepare_ae_scores(train_df, valid_df, test_df, y_valid, y_test, warnings)

    log("Selecting fine probability ensemble weight on validation PR-AUC only.")
    probability_result = evaluate_ensemble_variant(
        "probability",
        y_valid.to_numpy(),
        y_test.to_numpy(),
        fe_valid_score,
        ae_valid_score,
        fe_test_score,
        ae_test_score,
        output_dir,
    )

    log("Selecting fine rank ensemble weight on validation PR-AUC only.")
    fe_valid_rank = percentile_rank(fe_valid_score)
    ae_valid_rank = percentile_rank(ae_valid_score)
    fe_test_rank = percentile_rank(fe_test_score)
    ae_test_rank = percentile_rank(ae_test_score)
    rank_result = evaluate_ensemble_variant(
        "rank",
        y_valid.to_numpy(),
        y_test.to_numpy(),
        fe_valid_rank,
        ae_valid_rank,
        fe_test_rank,
        ae_test_rank,
        output_dir,
    )

    log("Saving validation/test score files.")
    save_scores(
        output_dir / "scores_validation.csv",
        valid_df,
        y_valid,
        fe_valid_score,
        ae_valid_score,
        probability_result["validation_score"],
        fe_valid_rank,
        ae_valid_rank,
        rank_result["validation_score"],
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_df,
        y_test,
        fe_test_score,
        ae_test_score,
        probability_result["test_score"],
        fe_test_rank,
        ae_test_rank,
        rank_result["test_score"],
    )

    run_config = {
        "phase": "fe_ae_fine_ensemble_calibration",
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
            "threshold_selection": "Selected on validation scores only using MCC.",
            "test_usage": "Test split used only after weights and thresholds were fixed.",
            "kaggle_competition_test_files_used": False,
            "training_performed": False,
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
                "latent_output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
                "latent_dim": latent_dim_from_run_config(autoencoder_run_config),
                "latent_feature_count": latent_feature_count,
            },
        },
        "weight_search": {
            "source_split": "validation",
            "objective": "average_precision / PR-AUC",
            "fe_weight_min": float(WEIGHT_GRID.min()),
            "fe_weight_max": float(WEIGHT_GRID.max()),
            "fe_weight_step": 0.005,
            "tie_break": "Higher FE-LGBM weight is selected when validation PR-AUC ties.",
        },
        "ensembles": {
            "probability": {
                "formula": "score = w * fe_probability + (1 - w) * ae_probability",
                "selected_fe_lgbm_tuned_weight": probability_result[
                    "selected_fe_weight"
                ],
                "selected_ae_lgbm_ld128_tuned_weight": probability_result[
                    "selected_ae_weight"
                ],
                "selected_threshold": probability_result["selected_threshold"],
                "validation_average_precision": probability_result[
                    "metrics_validation"
                ]["average_precision"],
                "test_average_precision": probability_result["metrics_test"][
                    "average_precision"
                ],
            },
            "rank": {
                "formula": "score = w * percentile_rank(fe_probability) + "
                "(1 - w) * percentile_rank(ae_probability)",
                "rank_method": "pandas average tie handling, percentile ranks per split",
                "selected_fe_lgbm_tuned_weight": rank_result["selected_fe_weight"],
                "selected_ae_lgbm_ld128_tuned_weight": rank_result[
                    "selected_ae_weight"
                ],
                "selected_threshold": rank_result["selected_threshold"],
                "validation_average_precision": rank_result["metrics_validation"][
                    "average_precision"
                ],
                "test_average_precision": rank_result["metrics_test"][
                    "average_precision"
                ],
            },
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
        },
        "regenerated_metric_warnings": warnings,
        "scores_saved": {
            "validation": "scores_validation.csv",
            "test": "scores_test.csv",
        },
        "comparison_file": str(FINE_COMPARISON_FILE),
    }
    save_json(run_config, output_dir / "run_config.json")

    comparison_table = build_comparison_table(
        probability_result,
        rank_result,
        output_dir,
    )
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    comparison_table.to_csv(FINE_COMPARISON_FILE, index=False)

    print()
    print("FE + AE Fine Ensemble Summary")
    print("=============================")
    print(
        "Probability selected FE weight : "
        f"{probability_result['selected_fe_weight']:.3f}"
    )
    print(
        "Probability validation PR-AUC  : "
        f"{probability_result['metrics_validation']['average_precision']:.6f}"
    )
    print(
        "Probability test PR-AUC        : "
        f"{probability_result['metrics_test']['average_precision']:.6f}"
    )
    print(
        "Probability test ROC-AUC       : "
        f"{probability_result['metrics_test']['roc_auc']:.6f}"
    )
    print(
        "Probability threshold          : "
        f"{probability_result['selected_threshold']:.2f}"
    )
    print(
        "Probability test F1 / MCC      : "
        f"{probability_result['metrics_test']['f1']:.6f} / "
        f"{probability_result['metrics_test']['mcc']:.6f}"
    )
    print(f"Rank selected FE weight        : {rank_result['selected_fe_weight']:.3f}")
    print(
        "Rank validation PR-AUC         : "
        f"{rank_result['metrics_validation']['average_precision']:.6f}"
    )
    print(
        "Rank test PR-AUC               : "
        f"{rank_result['metrics_test']['average_precision']:.6f}"
    )
    print(
        "Rank test ROC-AUC              : "
        f"{rank_result['metrics_test']['roc_auc']:.6f}"
    )
    print(f"Rank threshold                 : {rank_result['selected_threshold']:.2f}")
    print(
        "Rank test F1 / MCC             : "
        f"{rank_result['metrics_test']['f1']:.6f} / "
        f"{rank_result['metrics_test']['mcc']:.6f}"
    )
    print(f"Metric warnings recorded       : {len(warnings)}")
    print(f"Outputs saved to               : {output_dir}")
    print(f"Comparison saved to            : {FINE_COMPARISON_FILE}")

    return {
        "output_dir": str(output_dir),
        "comparison_file": str(FINE_COMPARISON_FILE),
        "probability": probability_result,
        "rank": rank_result,
        "regenerated_metric_warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fine-grid FE-LGBM + AE-LightGBM ensemble calibration."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINE_ENSEMBLE_OUTPUT_DIR,
        help="Output directory for the fine ensemble run.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return run_experiment(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
