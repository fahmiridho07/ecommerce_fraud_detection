"""Run controlled FE-LGBM tuned + AE-LightGBM LD128 tuned score ensemble."""

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
    ID_COL,
    OPTUNA_OUTPUT_DIR,
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
WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
METRIC_TOLERANCE = 1e-6
EXPECTED_LATENT_DIM = 128
FE_TUNED_VALIDATION_PR_AUC = 0.654316
ENSEMBLE_MIN_VALIDATION_DELTA = 0.002


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifact(s):\n" + "\n".join(missing))


def output_dir_is_non_empty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    if output_dir_is_non_empty(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Pass --overwrite only when you intentionally want to replace this "
            "controlled experiment output."
        )
    return ensure_dir(output_dir)


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


def validate_metric_consistency(
    metrics_path: Path,
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
) -> None:
    if not metrics_path.exists():
        return

    stored = load_json(metrics_path)
    stored_ap = stored.get("average_precision")
    if stored_ap is None:
        return

    regenerated_ap = float(average_precision_score(y_true, y_score))
    if abs(regenerated_ap - float(stored_ap)) > METRIC_TOLERANCE:
        raise ValueError(
            f"{model_name} {split_name} regenerated AP {regenerated_ap:.12f} "
            f"does not match stored AP {float(stored_ap):.12f}."
        )


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


def load_tuned_fe_artifacts(fe_tuned_dir: Path):
    require_files(
        [
            fe_tuned_dir / "final_model.pkl",
            fe_tuned_dir / "preprocessing.pkl",
            fe_tuned_dir / "feature_engineering.pkl",
            fe_tuned_dir / "run_config.json",
        ]
    )
    return (
        joblib.load(fe_tuned_dir / "final_model.pkl"),
        joblib.load(fe_tuned_dir / "preprocessing.pkl"),
        joblib.load(fe_tuned_dir / "feature_engineering.pkl"),
        load_json(fe_tuned_dir / "run_config.json"),
    )


def load_tuned_ae_lgbm_artifacts(ae_tuned_dir: Path):
    require_files(
        [
            ae_tuned_dir / "final_model.pkl",
            ae_tuned_dir / "preprocessing_non_v.pkl",
            ae_tuned_dir / "run_config.json",
        ]
    )
    return (
        joblib.load(ae_tuned_dir / "final_model.pkl"),
        joblib.load(ae_tuned_dir / "preprocessing_non_v.pkl"),
        load_json(ae_tuned_dir / "run_config.json"),
    )


def prepare_fe_scores(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fe_tuned_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, int, float]:
    log("Loading tuned FE-LGBM artifacts.")
    fe_model, fe_preprocessing, feature_artifacts, fe_run_config = (
        load_tuned_fe_artifacts(fe_tuned_dir)
    )

    log("Preparing FE-LGBM validation/test matrices from saved train-fitted artifacts.")
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
    validate_metric_consistency(
        fe_tuned_dir / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        valid_score,
        "fe_lgbm_tuned",
        "validation",
    )
    validate_metric_consistency(
        fe_tuned_dir / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        test_score,
        "fe_lgbm_tuned",
        "test",
    )
    validation_pr_auc = float(average_precision_score(y_valid.to_numpy(), valid_score))
    return valid_score, test_score, y_valid, y_test, best_iteration, validation_pr_auc


def prepare_ae_scores(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_valid: pd.Series,
    y_test: pd.Series,
    ae_tuned_dir: Path,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, object]]:
    log("Loading tuned AE-LightGBM LD128 artifacts.")
    ae_model, ae_preprocessing, ae_run_config = load_tuned_ae_lgbm_artifacts(
        ae_tuned_dir
    )

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
    validate_metric_consistency(
        ae_tuned_dir / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        valid_score,
        "ae_lgbm_ld128_tuned",
        "validation",
    )
    validate_metric_consistency(
        ae_tuned_dir / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        test_score,
        "ae_lgbm_ld128_tuned",
        "test",
    )
    return valid_score, test_score, best_iteration, autoencoder_run_config


def build_weight_selection_table(
    y_valid: np.ndarray,
    fe_valid_score: np.ndarray,
    ae_valid_score: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for weight in WEIGHT_GRID:
        ensemble_score = weight * fe_valid_score + (1.0 - weight) * ae_valid_score
        rows.append(
            {
                "fe_lgbm_tuned_weight": float(weight),
                "ae_lgbm_ld128_tuned_weight": float(1.0 - weight),
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


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    fe_score: np.ndarray,
    ae_score: np.ndarray,
    ensemble_score: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "fe_lgbm_tuned_score": fe_score,
            "ae_lgbm_ld128_tuned_score": ae_score,
            "ensemble_score": ensemble_score,
        }
    ).to_csv(path, index=False)


def latent_dim_from_run_config(run_config: dict[str, object]) -> int | None:
    architecture = run_config.get("architecture", {})
    if not isinstance(architecture, dict):
        return None
    encoder = architecture.get("encoder", [])
    if not isinstance(encoder, list) or not encoder:
        return None
    return int(encoder[-1])


def stopping_decision(
    selected_fe_weight: float,
    validation_pr_auc: float,
    fe_reference_validation_pr_auc: float,
) -> dict[str, object]:
    validation_delta = float(validation_pr_auc - fe_reference_validation_pr_auc)
    selected_ae_weight = float(1.0 - selected_fe_weight)
    should_run_b = selected_ae_weight > 0.0 and validation_delta >= ENSEMBLE_MIN_VALIDATION_DELTA
    return {
        "fe_tuned_validation_pr_auc_reference": fe_reference_validation_pr_auc,
        "ensemble_min_validation_delta": ENSEMBLE_MIN_VALIDATION_DELTA,
        "validation_pr_auc_delta_vs_fe_tuned": validation_delta,
        "selected_ae_weight": selected_ae_weight,
        "stop_after_a": bool(
            selected_fe_weight == 1.0 or validation_delta < ENSEMBLE_MIN_VALIDATION_DELTA
        ),
        "run_b_next": bool(should_run_b),
        "rule": (
            "Run B only if AE weight > 0 and ensemble validation PR-AUC improves "
            "over tuned FE-LGBM by at least 0.002."
        ),
    }


def run_experiment(
    output_dir: Path,
    overwrite: bool,
    fe_tuned_dir: Path = FE_TUNED_DIR,
    ae_tuned_dir: Path = AE_LGBM_LD128_TUNED_DIR,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)

    log("Loading labeled training data and recreating chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_transaction_id_alignment(valid_df, test_df)

    (
        fe_valid_score,
        fe_test_score,
        y_valid,
        y_test,
        fe_best_iteration,
        fe_validation_pr_auc,
    ) = prepare_fe_scores(
        train_df,
        valid_df,
        test_df,
        fe_tuned_dir,
    )
    ae_valid_score, ae_test_score, ae_best_iteration, autoencoder_run_config = (
        prepare_ae_scores(train_df, valid_df, test_df, y_valid, y_test, ae_tuned_dir)
    )

    log("Selecting ensemble weight on validation PR-AUC only.")
    weight_table = build_weight_selection_table(
        y_valid.to_numpy(),
        fe_valid_score,
        ae_valid_score,
    )
    selected_fe_weight = selected_fe_weight_from_table(weight_table)
    selected_ae_weight = float(1.0 - selected_fe_weight)
    weight_table.to_csv(output_dir / "weight_selection.csv", index=False)

    valid_score = selected_fe_weight * fe_valid_score + selected_ae_weight * ae_valid_score
    test_score = selected_fe_weight * fe_test_score + selected_ae_weight * ae_test_score

    log("Selecting classification threshold on validation ensemble score only.")
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
    decision = stopping_decision(
        selected_fe_weight,
        float(metrics_valid_selected["average_precision"]),
        fe_validation_pr_auc,
    )

    log("Saving controlled score ensemble outputs.")
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

    save_scores(
        output_dir / "scores_validation.csv",
        valid_df,
        y_valid,
        fe_valid_score,
        ae_valid_score,
        valid_score,
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_df,
        y_test,
        fe_test_score,
        ae_test_score,
        test_score,
    )

    run_config = {
        "phase": "fe_ae_controlled_A_score_ensemble",
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
            "test_usage": "Test split used only after weight and threshold were fixed.",
            "kaggle_competition_test_files_used": False,
            "training_performed": False,
        },
        "ensemble": {
            "formula": "score = w * fe_score + (1 - w) * ae_score",
            "weight_grid": "0.00 to 1.00 step 0.01",
            "objective": "validation average_precision / PR-AUC",
            "tie_break": "Higher FE-LGBM weight is selected when validation PR-AUC ties.",
            "selected_fe_lgbm_tuned_weight": selected_fe_weight,
            "selected_ae_lgbm_ld128_tuned_weight": selected_ae_weight,
        },
        "input_models": {
            "fe_lgbm_tuned": {
                "output_dir": str(fe_tuned_dir),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing.pkl",
                "feature_engineering_file": "feature_engineering.pkl",
                "best_iteration": fe_best_iteration,
            },
            "ae_lgbm_ld128_tuned": {
                "output_dir": str(ae_tuned_dir),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing_non_v.pkl",
                "best_iteration": ae_best_iteration,
                "latent_output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
                "latent_dim": latent_dim_from_run_config(autoencoder_run_config),
            },
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "stopping_criteria": decision,
        "scores_saved": {
            "validation": "scores_validation.csv",
            "test": "scores_test.csv",
        },
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("FE + AE Score Ensemble Summary")
    print("==============================")
    print(f"Selected FE weight      : {selected_fe_weight:.2f}")
    print(f"Selected AE weight      : {selected_ae_weight:.2f}")
    print(f"Validation PR-AUC       : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Delta vs tuned FE valid : {decision['validation_pr_auc_delta_vs_fe_tuned']:+.6f}")
    print(f"Test PR-AUC             : {metrics_test_selected['average_precision']:.6f}")
    print(f"Test ROC-AUC            : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold      : {selected_threshold:.2f}")
    print(f"Test F1                 : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC                : {metrics_test_selected['mcc']:.6f}")
    print(f"Run B next              : {decision['run_b_next']}")
    print(f"Outputs saved to        : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "selected_fe_weight": selected_fe_weight,
        "selected_threshold": selected_threshold,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "stopping_criteria": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled score-level ensemble of tuned FE-LGBM and tuned "
            "AE-LightGBM LD128."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
        help="Output directory for Experiment A.",
    )
    parser.add_argument(
        "--fe-tuned-dir",
        type=Path,
        default=FE_TUNED_DIR,
        help="Directory containing tuned FE-LGBM artifacts.",
    )
    parser.add_argument(
        "--ae-tuned-dir",
        type=Path,
        default=AE_LGBM_LD128_TUNED_DIR,
        help="Directory containing tuned AE-LightGBM LD128 artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a non-empty Experiment A output directory.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return run_experiment(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        fe_tuned_dir=args.fe_tuned_dir,
        ae_tuned_dir=args.ae_tuned_dir,
    )


if __name__ == "__main__":
    main()
