"""Run a validation-selected score ensemble of tuned baseline and AE-LightGBM."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import (
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    DATA_DIR,
    ID_COL,
    OPTUNA_OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SCORE_ENSEMBLE_TUNED_OUTPUT_DIR,
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
    get_v_feature_columns,
    split_features_target,
)
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


BASELINE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "baseline_lgbm"
AE_LGBM_LD128_TUNED_DIR = OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128"
WEIGHT_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
METRIC_TOLERANCE = 1e-6


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


def load_tuned_baseline_artifacts():
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


def build_weight_selection_table(
    y_valid: np.ndarray,
    baseline_valid_score: np.ndarray,
    ae_valid_score: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for weight in WEIGHT_GRID:
        ensemble_score = weight * baseline_valid_score + (1.0 - weight) * ae_valid_score
        rows.append(
            {
                "baseline_weight": float(weight),
                "ae_lgbm_ld128_weight": float(1.0 - weight),
                "validation_average_precision": float(
                    average_precision_score(y_valid, ensemble_score)
                ),
            }
        )

    table = pd.DataFrame(rows)
    best_index = table.sort_values(
        ["validation_average_precision", "baseline_weight"],
        ascending=[False, False],
    ).index[0]
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    return table


def selected_weight_from_table(weight_table: pd.DataFrame) -> float:
    selected = weight_table.loc[weight_table["selected"], "baseline_weight"]
    if selected.empty:
        raise ValueError("No selected ensemble weight found.")
    return float(selected.iloc[0])


def latent_dim_from_run_config(run_config: dict[str, object]) -> int | None:
    architecture = run_config.get("architecture", {})
    if not isinstance(architecture, dict):
        return None
    encoder = architecture.get("encoder", [])
    if not isinstance(encoder, list) or not encoder:
        return None
    return int(encoder[-1])


def save_scores(
    path: Path,
    split_df: pd.DataFrame,
    y: pd.Series,
    baseline_score: np.ndarray,
    ae_score: np.ndarray,
    ensemble_score: np.ndarray,
) -> None:
    pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "baseline_tuned_score": baseline_score,
            "ae_lgbm_ld128_tuned_score": ae_score,
            "ensemble_score": ensemble_score,
        }
    ).to_csv(path, index=False)


def main() -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(SCORE_ENSEMBLE_TUNED_OUTPUT_DIR)

    log("Loading labeled training data and recreating chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)

    log("Loading tuned baseline artifacts.")
    baseline_model, baseline_preprocessing, baseline_run_config = (
        load_tuned_baseline_artifacts()
    )

    log("Preparing baseline validation/test matrices from saved preprocessing.")
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    X_valid_baseline = apply_baseline_preprocessing(X_valid_raw, baseline_preprocessing)
    X_test_baseline = apply_baseline_preprocessing(X_test_raw, baseline_preprocessing)
    validate_model_features(baseline_model, X_valid_baseline, "baseline_lgbm_tuned")
    validate_model_features(baseline_model, X_test_baseline, "baseline_lgbm_tuned")

    baseline_best_iteration = best_iteration_from_config(
        baseline_model,
        baseline_run_config,
    )
    baseline_valid_score = predict_scores(
        baseline_model,
        X_valid_baseline,
        baseline_best_iteration,
    )
    baseline_test_score = predict_scores(
        baseline_model,
        X_test_baseline,
        baseline_best_iteration,
    )
    validate_score_length(baseline_valid_score, y_valid, "validation baseline")
    validate_score_length(baseline_test_score, y_test, "test baseline")
    validate_metric_consistency(
        BASELINE_TUNED_DIR / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        baseline_valid_score,
        "baseline_lgbm_tuned",
        "validation",
    )
    validate_metric_consistency(
        BASELINE_TUNED_DIR / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        baseline_test_score,
        "baseline_lgbm_tuned",
        "test",
    )

    log("Loading tuned AE-LightGBM LD128 artifacts.")
    ae_model, ae_preprocessing, ae_run_config = load_tuned_ae_lgbm_artifacts()

    log("Loading robust AE LD128 latent features.")
    latent_train, latent_valid, latent_test, latent_feature_names, ae_latent_config = (
        load_robust_latent_outputs(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR)
    )
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    if latent_valid.shape[1] != 128:
        raise ValueError(f"Expected LD128 latent features, found {latent_valid.shape[1]}.")

    log("Preparing AE-LightGBM validation/test matrices from saved preprocessing.")
    X_valid_non_v_raw, y_valid_ae = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test_ae = split_non_v_features_target(test_df, v_columns)
    if not y_valid.equals(y_valid_ae) or not y_test.equals(y_test_ae):
        raise ValueError("Baseline and AE-LightGBM labels are not aligned.")

    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, ae_preprocessing)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, ae_preprocessing)
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
    validate_feature_alignment(X_valid_ae, X_valid_ae, X_test_ae, v_columns)
    validate_model_features(ae_model, X_valid_ae, "ae_lgbm_ld128_tuned")
    validate_model_features(ae_model, X_test_ae, "ae_lgbm_ld128_tuned")

    ae_best_iteration = best_iteration_from_config(ae_model, ae_run_config)
    ae_valid_score = predict_scores(ae_model, X_valid_ae, ae_best_iteration)
    ae_test_score = predict_scores(ae_model, X_test_ae, ae_best_iteration)
    validate_score_length(ae_valid_score, y_valid, "validation AE-LightGBM")
    validate_score_length(ae_test_score, y_test, "test AE-LightGBM")
    validate_metric_consistency(
        AE_LGBM_LD128_TUNED_DIR / "metrics_validation_selected_threshold.json",
        y_valid.to_numpy(),
        ae_valid_score,
        "ae_lgbm_ld128_tuned",
        "validation",
    )
    validate_metric_consistency(
        AE_LGBM_LD128_TUNED_DIR / "metrics_test_selected_threshold.json",
        y_test.to_numpy(),
        ae_test_score,
        "ae_lgbm_ld128_tuned",
        "test",
    )

    log("Selecting ensemble weight on validation PR-AUC only.")
    weight_table = build_weight_selection_table(
        y_valid.to_numpy(),
        baseline_valid_score,
        ae_valid_score,
    )
    selected_weight = selected_weight_from_table(weight_table)
    weight_table.to_csv(output_dir / "weight_selection.csv", index=False)

    valid_score = selected_weight * baseline_valid_score + (
        1.0 - selected_weight
    ) * ae_valid_score
    test_score = selected_weight * baseline_test_score + (
        1.0 - selected_weight
    ) * ae_test_score

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

    log("Saving score ensemble outputs.")
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
        baseline_valid_score,
        ae_valid_score,
        valid_score,
    )
    save_scores(
        output_dir / "scores_test.csv",
        test_df,
        y_test,
        baseline_test_score,
        ae_test_score,
        test_score,
    )

    run_config = {
        "phase": "next_A_score_level_ensemble",
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
            "weight_selection": "Selected on validation PR-AUC only.",
            "threshold_selection": "Selected on validation ensemble scores only.",
            "test_usage": "Test split used only after weight and threshold were fixed.",
            "kaggle_competition_test_files_used": False,
            "preprocessing": "Loaded saved train-fitted preprocessing artifacts.",
        },
        "ensemble": {
            "formula": "score = w * baseline_score + (1 - w) * ae_score",
            "weight_grid": "0.00 to 1.00 step 0.01",
            "objective": "validation average_precision / PR-AUC",
            "tie_break": "Higher baseline weight is selected when validation PR-AUC ties.",
            "selected_baseline_weight": selected_weight,
            "selected_ae_lgbm_ld128_weight": 1.0 - selected_weight,
        },
        "input_models": {
            "baseline_lgbm_tuned": {
                "output_dir": str(BASELINE_TUNED_DIR),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing.pkl",
                "best_iteration": baseline_best_iteration,
            },
            "ae_lgbm_ld128_tuned": {
                "output_dir": str(AE_LGBM_LD128_TUNED_DIR),
                "model_file": "final_model.pkl",
                "preprocessing_file": "preprocessing_non_v.pkl",
                "best_iteration": ae_best_iteration,
                "latent_output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
                "latent_dim": len(latent_feature_names),
            },
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "scores_saved": {
            "validation": "scores_validation.csv",
            "test": "scores_test.csv",
        },
        "robust_autoencoder_run_config": {
            "latent_dim": latent_dim_from_run_config(ae_latent_config),
            "output_dir": str(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR),
        },
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Score Ensemble Summary")
    print("======================")
    print(f"Selected baseline weight: {selected_weight:.2f}")
    print(f"Validation PR-AUC       : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test PR-AUC             : {metrics_test_selected['average_precision']:.6f}")
    print(f"Validation ROC-AUC      : {metrics_valid_selected['roc_auc']:.6f}")
    print(f"Test ROC-AUC            : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold      : {selected_threshold:.2f}")
    print(f"Test precision          : {metrics_test_selected['precision']:.6f}")
    print(f"Test recall             : {metrics_test_selected['recall']:.6f}")
    print(f"Test F1                 : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC                : {metrics_test_selected['mcc']:.6f}")
    print(f"Outputs saved to        : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "selected_weight": selected_weight,
        "selected_threshold": selected_threshold,
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
    }


if __name__ == "__main__":
    main()
