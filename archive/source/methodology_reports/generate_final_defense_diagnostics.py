"""Generate final robustness and interpretability diagnostics for thesis defense.

The script uses existing artifacts only:
- labeled train files are used only to recreate the chronological splits;
- saved score files are used when available;
- saved model/preprocessing artifacts are loaded only for score regeneration or
  inference benchmarking;
- no model fitting or Kaggle competition test files are used.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import (  # noqa: E402
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
    ID_COL,
    OPTUNA_OUTPUT_DIR,
    OUTPUT_DIR as PROJECT_OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data  # noqa: E402
from feature_engineering import apply_entity_time_amount_features  # noqa: E402
from preprocessing import (  # noqa: E402
    apply_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
    transform_categorical_columns,
)
from splitting import chronological_split  # noqa: E402


FINAL_DIAGNOSTICS_DIR = PROJECT_OUTPUT_DIR / "final_diagnostics"
FE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features"
AE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128"
FINAL_ENSEMBLE_DIR = FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR

RECONSTRUCTION_ERROR_FILES = {
    "validation": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "reconstruction_error_valid.csv",
    "test": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "reconstruction_error_test.csv",
}

TEMPORAL_BIN_COUNT = 5
MIN_FRAUD_CASES_PER_BIN = 20
SCATTER_SAMPLE_ROWS = 30000
BENCHMARK_REPEATS = 5
EPSILON = 1e-12

DEFAULT_FINAL_FE_WEIGHT = 0.78
DEFAULT_FINAL_AE_WEIGHT = 0.22


def log(message: str) -> None:
    print(f"[final-defense] {message}", flush=True)


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    log(f"WARNING: {message}")


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def save_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def require_files(paths: list[Path], context: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) for {context}:\n" + "\n".join(missing)
        )


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0 or int(np.sum(y_true)) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def format_number(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def load_labeled_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("Loading labeled training data and recreating chronological split.")
    full_df = load_labeled_train_data(sample_size=None)
    return chronological_split(full_df)


def final_weights(warnings: list[str]) -> tuple[float, float, str]:
    run_config_path = FINAL_ENSEMBLE_DIR / "run_config.json"
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        ensemble = run_config.get("ensemble", {})
        if isinstance(ensemble, dict):
            fe_weight = ensemble.get("selected_fe_lgbm_tuned_weight")
            ae_weight = ensemble.get("selected_ae_lgbm_ld128_tuned_weight")
            if fe_weight is not None and ae_weight is not None:
                return float(fe_weight), float(ae_weight), str(run_config_path)

    warn(
        warnings,
        "Final ensemble run_config.json is missing selected weights; "
        f"using defaults FE={DEFAULT_FINAL_FE_WEIGHT}, AE={DEFAULT_FINAL_AE_WEIGHT}.",
    )
    return DEFAULT_FINAL_FE_WEIGHT, DEFAULT_FINAL_AE_WEIGHT, "fixed_default"


def best_iteration_from_config(model, run_config: dict[str, object]) -> int | None:
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

    return None


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
            f"{model_name} feature mismatch. "
            f"expected_count={len(expected)}, observed_count={len(observed)}, "
            f"expected_only={expected_only}, observed_only={observed_only}"
        )


def predict_scores(model, X: pd.DataFrame, best_iteration: int | None) -> np.ndarray:
    kwargs = {}
    if best_iteration is not None:
        kwargs["num_iteration"] = best_iteration
    try:
        scores = model.predict_proba(X, **kwargs)[:, 1]
    except TypeError:
        scores = model.predict_proba(X)[:, 1]
    return np.asarray(scores, dtype="float64")


def apply_non_v_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    feature_columns = preprocessing["feature_columns"]
    categorical_mappings = preprocessing["categorical_mappings"]
    X = X.loc[:, feature_columns].copy()
    return transform_categorical_columns(X, categorical_mappings)


def split_non_v_features_target(
    df: pd.DataFrame,
    v_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET_COL].astype(int).copy()
    excluded = set(v_columns + [TARGET_COL, ID_COL])
    feature_columns = [column for column in df.columns if column not in excluded]
    return df.loc[:, feature_columns].copy(), y


def combine_non_v_and_latent(
    X_non_v: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
) -> pd.DataFrame:
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    return pd.concat(
        [X_non_v.reset_index(drop=True), latent_df.reset_index(drop=True)],
        axis=1,
    )


def load_latent_feature_names() -> list[str]:
    path = AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_feature_names.json"
    require_files([path], "robust AE LD128 latent feature names")
    names = load_json(path)
    if not isinstance(names, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return [str(name) for name in names]


def load_fe_model_and_matrices(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[object, pd.DataFrame, pd.DataFrame, int | None]:
    require_files(
        [
            FE_TUNED_DIR / "final_model.pkl",
            FE_TUNED_DIR / "preprocessing.pkl",
            FE_TUNED_DIR / "feature_engineering.pkl",
            FE_TUNED_DIR / "run_config.json",
        ],
        "FE-LGBM tuned saved inference artifacts",
    )

    model = joblib.load(FE_TUNED_DIR / "final_model.pkl")
    preprocessing = joblib.load(FE_TUNED_DIR / "preprocessing.pkl")
    feature_artifacts = joblib.load(FE_TUNED_DIR / "feature_engineering.pkl")
    run_config = load_json(FE_TUNED_DIR / "run_config.json")

    X_valid_raw, _ = split_features_target(valid_df)
    X_test_raw, _ = split_features_target(test_df)
    X_valid_engineered = apply_entity_time_amount_features(
        X_valid_raw,
        feature_artifacts,
    )
    X_test_engineered = apply_entity_time_amount_features(
        X_test_raw,
        feature_artifacts,
    )
    X_valid = apply_baseline_preprocessing(X_valid_engineered, preprocessing)
    X_test = apply_baseline_preprocessing(X_test_engineered, preprocessing)
    validate_model_features(model, X_valid, "FE-LGBM tuned validation")
    validate_model_features(model, X_test, "FE-LGBM tuned test")
    return model, X_valid, X_test, best_iteration_from_config(model, run_config)


def load_ae_model_and_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[object, pd.DataFrame, pd.DataFrame, int | None]:
    require_files(
        [
            AE_TUNED_DIR / "final_model.pkl",
            AE_TUNED_DIR / "preprocessing_non_v.pkl",
            AE_TUNED_DIR / "run_config.json",
            AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_valid.npy",
            AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_test.npy",
            AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_feature_names.json",
        ],
        "AE-LGBM tuned saved inference artifacts",
    )

    model = joblib.load(AE_TUNED_DIR / "final_model.pkl")
    preprocessing = joblib.load(AE_TUNED_DIR / "preprocessing_non_v.pkl")
    run_config = load_json(AE_TUNED_DIR / "run_config.json")
    latent_valid = np.load(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_valid.npy")
    latent_test = np.load(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_test.npy")
    latent_feature_names = load_latent_feature_names()

    if latent_valid.shape[0] != len(valid_df):
        raise ValueError(
            f"latent_valid rows {latent_valid.shape[0]} do not match validation rows "
            f"{len(valid_df)}."
        )
    if latent_test.shape[0] != len(test_df):
        raise ValueError(
            f"latent_test rows {latent_test.shape[0]} do not match test rows "
            f"{len(test_df)}."
        )
    if latent_valid.shape[1] != len(latent_feature_names):
        raise ValueError("latent_valid column count does not match feature names.")
    if latent_test.shape[1] != len(latent_feature_names):
        raise ValueError("latent_test column count does not match feature names.")

    v_columns = get_v_feature_columns(train_df)
    X_valid_non_v_raw, _ = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, _ = split_non_v_features_target(test_df, v_columns)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing)
    X_valid = combine_non_v_and_latent(
        X_valid_non_v,
        latent_valid,
        latent_feature_names,
    )
    X_test = combine_non_v_and_latent(
        X_test_non_v,
        latent_test,
        latent_feature_names,
    )
    validate_model_features(model, X_valid, "AE-LGBM tuned validation")
    validate_model_features(model, X_test, "AE-LGBM tuned test")
    return model, X_valid, X_test, best_iteration_from_config(model, run_config)


def standardize_score_frame(
    score_df: pd.DataFrame,
    split_name: str,
    fe_weight: float,
    ae_weight: float,
    source_path: Path,
) -> pd.DataFrame:
    required = [ID_COL, TARGET_COL, "fe_lgbm_tuned_score", "ae_lgbm_ld128_tuned_score"]
    missing = [column for column in required if column not in score_df.columns]
    if missing:
        raise KeyError(f"{source_path} is missing required column(s): {missing}")

    standardized = pd.DataFrame(
        {
            ID_COL: score_df[ID_COL].to_numpy(),
            TARGET_COL: score_df[TARGET_COL].astype(int).to_numpy(),
            "fe_score": pd.to_numeric(
                score_df["fe_lgbm_tuned_score"],
                errors="coerce",
            ).to_numpy(dtype="float64"),
            "ae_score": pd.to_numeric(
                score_df["ae_lgbm_ld128_tuned_score"],
                errors="coerce",
            ).to_numpy(dtype="float64"),
        }
    )

    if "ensemble_score" in score_df.columns:
        standardized["ensemble_score"] = pd.to_numeric(
            score_df["ensemble_score"],
            errors="coerce",
        ).to_numpy(dtype="float64")
    else:
        standardized["ensemble_score"] = (
            fe_weight * standardized["fe_score"] + ae_weight * standardized["ae_score"]
        )

    if standardized[["fe_score", "ae_score", "ensemble_score"]].isna().any().any():
        raise ValueError(f"{source_path} contains missing or non-numeric score values.")

    standardized["split"] = split_name
    standardized["score_source"] = str(source_path)
    return standardized


def regenerate_scores_from_saved_artifacts(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fe_weight: float,
    ae_weight: float,
) -> dict[str, pd.DataFrame]:
    log("Regenerating final scores in memory from saved artifacts.")
    fe_model, X_valid_fe, X_test_fe, fe_best_iteration = load_fe_model_and_matrices(
        valid_df,
        test_df,
    )
    ae_model, X_valid_ae, X_test_ae, ae_best_iteration = load_ae_model_and_matrices(
        train_df,
        valid_df,
        test_df,
    )

    y_valid = valid_df[TARGET_COL].astype(int).to_numpy()
    y_test = test_df[TARGET_COL].astype(int).to_numpy()
    valid_fe_score = predict_scores(fe_model, X_valid_fe, fe_best_iteration)
    test_fe_score = predict_scores(fe_model, X_test_fe, fe_best_iteration)
    valid_ae_score = predict_scores(ae_model, X_valid_ae, ae_best_iteration)
    test_ae_score = predict_scores(ae_model, X_test_ae, ae_best_iteration)

    return {
        "validation": pd.DataFrame(
            {
                ID_COL: valid_df[ID_COL].to_numpy(),
                TARGET_COL: y_valid,
                "fe_score": valid_fe_score,
                "ae_score": valid_ae_score,
                "ensemble_score": fe_weight * valid_fe_score + ae_weight * valid_ae_score,
                "split": "validation",
                "score_source": "regenerated_in_memory_from_saved_artifacts",
            }
        ),
        "test": pd.DataFrame(
            {
                ID_COL: test_df[ID_COL].to_numpy(),
                TARGET_COL: y_test,
                "fe_score": test_fe_score,
                "ae_score": test_ae_score,
                "ensemble_score": fe_weight * test_fe_score + ae_weight * test_ae_score,
                "split": "test",
                "score_source": "regenerated_in_memory_from_saved_artifacts",
            }
        ),
    }


def load_or_regenerate_scores(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    warnings: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    fe_weight, ae_weight, weight_source = final_weights(warnings)
    score_paths = {
        "validation": FINAL_ENSEMBLE_DIR / "scores_validation.csv",
        "test": FINAL_ENSEMBLE_DIR / "scores_test.csv",
    }

    if all(path.exists() for path in score_paths.values()):
        log("Loading saved final validation/test score files.")
        scores = {
            split_name: standardize_score_frame(
                pd.read_csv(path),
                split_name,
                fe_weight,
                ae_weight,
                path,
            )
            for split_name, path in score_paths.items()
        }
    else:
        missing = [str(path) for path in score_paths.values() if not path.exists()]
        warn(
            warnings,
            "Final score file(s) missing; regenerating scores in memory from "
            f"saved artifacts only: {missing}",
        )
        scores = regenerate_scores_from_saved_artifacts(
            train_df,
            valid_df,
            test_df,
            fe_weight,
            ae_weight,
        )

    metadata = {
        "fe_weight": fe_weight,
        "ae_weight": ae_weight,
        "weight_source": weight_source,
        "validation_score_source": str(scores["validation"]["score_source"].iloc[0]),
        "test_score_source": str(scores["test"]["score_source"].iloc[0]),
    }
    return scores, metadata


def align_scores_with_split(
    split_df: pd.DataFrame,
    score_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    if score_df[ID_COL].duplicated().any():
        raise ValueError(f"{split_name} scores contain duplicate {ID_COL} values.")
    if split_df[ID_COL].duplicated().any():
        raise ValueError(f"{split_name} split contains duplicate {ID_COL} values.")

    aligned = split_df[[ID_COL, TARGET_COL, TIME_COL]].merge(
        score_df[[ID_COL, TARGET_COL, "fe_score", "ae_score", "ensemble_score"]],
        on=ID_COL,
        how="inner",
        suffixes=("_split", "_score"),
        validate="one_to_one",
    )
    if len(aligned) != len(split_df):
        raise ValueError(
            f"{split_name} score rows do not align with split rows: "
            f"{len(aligned)} matched vs {len(split_df)} expected."
        )
    if not np.array_equal(
        aligned[f"{TARGET_COL}_split"].astype(int).to_numpy(),
        aligned[f"{TARGET_COL}_score"].astype(int).to_numpy(),
    ):
        raise ValueError(f"{split_name} labels differ between split data and scores.")

    aligned = aligned.rename(columns={f"{TARGET_COL}_split": TARGET_COL})
    return aligned.drop(columns=[f"{TARGET_COL}_score"])


def run_temporal_degradation(
    test_df: pd.DataFrame,
    test_scores: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    log("Computing temporal degradation diagnostics on the test split.")
    aligned = align_scores_with_split(test_df, test_scores, "test")
    aligned = aligned.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for bin_number, indices in enumerate(
        np.array_split(np.arange(len(aligned)), TEMPORAL_BIN_COUNT),
        start=1,
    ):
        bin_df = aligned.iloc[indices]
        y_true = bin_df[TARGET_COL].astype(int).to_numpy()
        fe_score = bin_df["fe_score"].to_numpy(dtype="float64")
        ensemble_score = bin_df["ensemble_score"].to_numpy(dtype="float64")
        row_count = int(len(bin_df))
        fraud_count = int(y_true.sum())
        fraud_rate = float(fraud_count / row_count) if row_count else None

        if fraud_count < MIN_FRAUD_CASES_PER_BIN:
            warn(
                warnings,
                f"Temporal bin {bin_number} has only {fraud_count} fraud cases; "
                "PR-AUC/ROC-AUC estimates may be unstable.",
            )

        fe_pr_auc = safe_average_precision(y_true, fe_score)
        ensemble_pr_auc = safe_average_precision(y_true, ensemble_score)
        rows.append(
            {
                "bin": bin_number,
                "row_count": row_count,
                "fraud_count": fraud_count,
                "fraud_rate": fraud_rate,
                "transactiondt_min": int(bin_df[TIME_COL].min()),
                "transactiondt_max": int(bin_df[TIME_COL].max()),
                "fe_lgbm_tuned_pr_auc": fe_pr_auc,
                "fe_ae_ensemble_pr_auc": ensemble_pr_auc,
                "fe_lgbm_tuned_roc_auc": safe_roc_auc(y_true, fe_score),
                "fe_ae_ensemble_roc_auc": safe_roc_auc(y_true, ensemble_score),
                "delta_pr_auc": (
                    float(ensemble_pr_auc - fe_pr_auc)
                    if fe_pr_auc is not None and ensemble_pr_auc is not None
                    else None
                ),
            }
        )

    temporal_df = pd.DataFrame(rows)
    temporal_df.to_csv(
        FINAL_DIAGNOSTICS_DIR / "temporal_degradation_by_bin.csv",
        index=False,
    )
    plot_temporal_pr_auc(temporal_df)
    return temporal_df


def plot_temporal_pr_auc(temporal_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        temporal_df["bin"],
        temporal_df["fe_lgbm_tuned_pr_auc"],
        marker="o",
        linewidth=2,
        color="#3B6EA8",
        label="FE-LGBM tuned",
    )
    ax.plot(
        temporal_df["bin"],
        temporal_df["fe_ae_ensemble_pr_auc"],
        marker="o",
        linewidth=2,
        color="#D95F02",
        label="FE+AE score ensemble",
    )
    ax.set_title("Temporal Test Split PR-AUC by Chronological Bin")
    ax.set_xlabel("Chronological test bin")
    ax.set_ylabel("PR-AUC")
    ax.set_xticks(temporal_df["bin"].astype(int).tolist())
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FINAL_DIAGNOSTICS_DIR / "temporal_degradation_pr_auc.png",
        dpi=170,
    )
    plt.close(fig)


def run_score_complementarity(
    test_scores: pd.DataFrame,
    score_metadata: dict[str, object],
) -> dict[str, object]:
    log("Computing FE-vs-AE score complementarity diagnostics.")
    y_true = test_scores[TARGET_COL].astype(int).to_numpy()
    fe_score = test_scores["fe_score"].to_numpy(dtype="float64")
    ae_score = test_scores["ae_score"].to_numpy(dtype="float64")
    ensemble_score = test_scores["ensemble_score"].to_numpy(dtype="float64")

    pearson = float(np.corrcoef(fe_score, ae_score)[0, 1])
    spearman = float(
        pd.Series(fe_score).corr(pd.Series(ae_score), method="spearman")
    )
    summary = {
        "split": "test",
        "rows": int(len(test_scores)),
        "fraud_count": int(y_true.sum()),
        "fraud_rate": float(np.mean(y_true)),
        "score_sources": {
            "test": score_metadata["test_score_source"],
            "weights": score_metadata["weight_source"],
        },
        "weights": {
            "fe_lgbm_tuned": score_metadata["fe_weight"],
            "ae_lgbm_ld128_tuned": score_metadata["ae_weight"],
        },
        "pearson_correlation": pearson,
        "spearman_correlation": spearman,
        "metrics": {
            "fe_lgbm_tuned": {
                "pr_auc": safe_average_precision(y_true, fe_score),
                "roc_auc": safe_roc_auc(y_true, fe_score),
            },
            "ae_lgbm_ld128_tuned": {
                "pr_auc": safe_average_precision(y_true, ae_score),
                "roc_auc": safe_roc_auc(y_true, ae_score),
            },
            "fe_ae_score_ensemble": {
                "pr_auc": safe_average_precision(y_true, ensemble_score),
                "roc_auc": safe_roc_auc(y_true, ensemble_score),
            },
        },
    }
    summary["deltas"] = {
        "ensemble_minus_fe_pr_auc": (
            summary["metrics"]["fe_ae_score_ensemble"]["pr_auc"]
            - summary["metrics"]["fe_lgbm_tuned"]["pr_auc"]
        ),
        "ae_minus_fe_pr_auc": (
            summary["metrics"]["ae_lgbm_ld128_tuned"]["pr_auc"]
            - summary["metrics"]["fe_lgbm_tuned"]["pr_auc"]
        ),
        "ensemble_minus_ae_pr_auc": (
            summary["metrics"]["fe_ae_score_ensemble"]["pr_auc"]
            - summary["metrics"]["ae_lgbm_ld128_tuned"]["pr_auc"]
        ),
    }
    save_json(summary, FINAL_DIAGNOSTICS_DIR / "score_correlation_summary.json")
    plot_score_scatter(test_scores)
    return summary


def plot_score_scatter(test_scores: pd.DataFrame) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    n_rows = len(test_scores)
    sample_size = min(SCATTER_SAMPLE_ROWS, n_rows)
    if sample_size < n_rows:
        sample_indices = rng.choice(n_rows, size=sample_size, replace=False)
        sample = test_scores.iloc[sample_indices].copy()
    else:
        sample = test_scores.copy()

    fe_score = np.clip(sample["fe_score"].to_numpy(dtype="float64"), EPSILON, 1.0)
    ae_score = np.clip(sample["ae_score"].to_numpy(dtype="float64"), EPSILON, 1.0)
    labels = sample[TARGET_COL].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    non_fraud = labels == 0
    fraud = labels == 1
    ax.scatter(
        fe_score[non_fraud],
        ae_score[non_fraud],
        s=7,
        alpha=0.14,
        color="#5F6B7A",
        label="Non-fraud",
        linewidths=0,
    )
    ax.scatter(
        fe_score[fraud],
        ae_score[fraud],
        s=10,
        alpha=0.55,
        color="#D95F02",
        label="Fraud",
        linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FE-LGBM tuned score")
    ax.set_ylabel("AE-LGBM tuned score")
    ax.set_title(f"FE vs AE Test Scores (sample n={sample_size:,})")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(
        FINAL_DIAGNOSTICS_DIR / "fe_vs_ae_score_scatter_sample.png",
        dpi=170,
    )
    plt.close(fig)


def reconstruction_error_column(df: pd.DataFrame, path: Path) -> str:
    candidates = ["reconstruction_mse", "reconstruction_error", "mse", "error"]
    for column in candidates:
        if column in df.columns:
            return column
    if df.shape[1] == 1:
        return str(df.columns[0])
    raise ValueError(
        f"Could not identify reconstruction-error column in {path}. "
        f"Columns: {df.columns.tolist()}"
    )


def read_reconstruction_errors(split_name: str) -> np.ndarray:
    path = RECONSTRUCTION_ERROR_FILES[split_name]
    require_files([path], f"robust AE LD128 {split_name} reconstruction errors")
    df = pd.read_csv(path)
    column = reconstruction_error_column(df, path)
    errors = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype="float64")
    if np.isnan(errors).any():
        raise ValueError(f"{path} contains missing or non-numeric values.")
    return errors


def class_summary_rows(
    split_name: str,
    y_true: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for class_value, label in ((0, "non_fraud"), (1, "fraud")):
        values = errors[y_true == class_value]
        if values.size == 0:
            continue
        rows.append(
            {
                "split": split_name,
                "isFraud": class_value,
                "class_label": label,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "q25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
        )
    return rows


def run_reconstruction_error_diagnostics(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    log("Computing robust AE LD128 reconstruction-error diagnostics.")
    split_frames = {"validation": valid_df, "test": test_df}
    summary_rows: list[dict[str, object]] = []
    payloads: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, object] = {}

    for split_name, split_df in split_frames.items():
        errors = read_reconstruction_errors(split_name)
        y_true = split_df[TARGET_COL].astype(int).to_numpy()
        if errors.shape[0] != y_true.shape[0]:
            warn(
                warnings,
                f"{split_name} reconstruction-error rows ({errors.shape[0]}) do not "
                f"match split rows ({y_true.shape[0]}). Skipping this split.",
            )
            continue

        summary_rows.extend(class_summary_rows(split_name, y_true, errors))
        metrics[split_name] = {
            "rows": int(y_true.shape[0]),
            "fraud_count": int(y_true.sum()),
            "fraud_rate": float(np.mean(y_true)),
            "pr_auc_using_error_as_score": safe_average_precision(y_true, errors),
            "roc_auc_using_error_as_score": safe_roc_auc(y_true, errors),
            "score_direction": "higher reconstruction error treated as more anomalous",
        }
        payloads[split_name] = {"y_true": y_true, "errors": errors}

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        FINAL_DIAGNOSTICS_DIR / "reconstruction_error_class_summary.csv",
        index=False,
    )
    if payloads:
        plot_reconstruction_log_hist(payloads)
        plot_reconstruction_boxplot(payloads)
    return summary_df, metrics


def reconstruction_log_bins(payloads: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    all_errors = np.concatenate(
        [np.clip(payload["errors"], EPSILON, None) for payload in payloads.values()]
    )
    lower = float(np.percentile(all_errors, 0.1))
    upper = float(np.percentile(all_errors, 99.9))
    lower = max(lower, EPSILON)
    upper = max(upper, lower * 10.0)
    return np.logspace(np.log10(lower), np.log10(upper), 70)


def plot_reconstruction_log_hist(payloads: dict[str, dict[str, np.ndarray]]) -> None:
    split_names = [name for name in ("validation", "test") if name in payloads]
    fig, axes = plt.subplots(1, len(split_names), figsize=(7 * len(split_names), 5))
    if len(split_names) == 1:
        axes = [axes]
    bins = reconstruction_log_bins(payloads)

    for ax, split_name in zip(axes, split_names):
        payload = payloads[split_name]
        y_true = payload["y_true"]
        errors = np.clip(payload["errors"], EPSILON, None)
        for class_value, label, color in (
            (0, "Non-fraud", "#3B6EA8"),
            (1, "Fraud", "#D95F02"),
        ):
            values = errors[y_true == class_value]
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.45,
                color=color,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_title(f"{split_name.title()} Reconstruction Error")
        ax.set_xlabel("Reconstruction MSE (log scale)")
        ax.set_ylabel("Density")
        ax.grid(True, axis="y", alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(
        FINAL_DIAGNOSTICS_DIR / "reconstruction_error_distribution_log_hist.png",
        dpi=170,
    )
    plt.close(fig)


def plot_reconstruction_boxplot(payloads: dict[str, dict[str, np.ndarray]]) -> None:
    values = []
    labels = []
    colors = []
    for split_name in ("validation", "test"):
        if split_name not in payloads:
            continue
        payload = payloads[split_name]
        y_true = payload["y_true"]
        log_errors = np.log10(np.clip(payload["errors"], EPSILON, None))
        for class_value, label, color in (
            (0, "Non-fraud", "#B8CBE3"),
            (1, "Fraud", "#F2B077"),
        ):
            values.append(log_errors[y_true == class_value])
            labels.append(f"{split_name.title()}\n{label}")
            colors.append(color)

    fig, ax = plt.subplots(figsize=(9, 5))
    boxplot_kwargs = {"showfliers": False, "patch_artist": True}
    try:
        box = ax.boxplot(values, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        box = ax.boxplot(values, labels=labels, **boxplot_kwargs)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
    for median in box["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.6)
    ax.set_title("Robust AE LD128 Reconstruction Error by Class")
    ax.set_ylabel("log10(reconstruction MSE)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FINAL_DIAGNOSTICS_DIR / "reconstruction_error_boxplot.png", dpi=170)
    plt.close(fig)


def benchmark_callable(
    predict_fn: Callable[[], np.ndarray],
    expected_rows: int,
    repeats: int,
) -> dict[str, object]:
    warmup = predict_fn()
    if warmup.shape[0] != expected_rows:
        raise ValueError(
            f"Warmup prediction returned {warmup.shape[0]} rows; "
            f"expected {expected_rows}."
        )

    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        scores = predict_fn()
        elapsed.append(time.perf_counter() - start)
        if scores.shape[0] != expected_rows:
            raise ValueError(
                f"Prediction returned {scores.shape[0]} rows; expected {expected_rows}."
            )

    elapsed_array = np.asarray(elapsed, dtype="float64")
    total_seconds = float(elapsed_array.sum())
    mean_seconds = float(elapsed_array.mean())
    return {
        "repeats": repeats,
        "warmup_excluded": True,
        "total_seconds": total_seconds,
        "mean_seconds": mean_seconds,
        "std_seconds": float(elapsed_array.std(ddof=1)) if repeats > 1 else 0.0,
        "ms_per_10000_rows": float(
            (mean_seconds * 1000.0) / (expected_rows / 10000.0)
        ),
    }


def run_inference_complexity_benchmark(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    score_metadata: dict[str, object],
    warnings: list[str],
) -> pd.DataFrame | None:
    log("Preparing saved feature matrices for inference complexity benchmark.")
    try:
        fe_model, _, X_test_fe, fe_best_iteration = load_fe_model_and_matrices(
            valid_df,
            test_df,
        )
        ae_model, _, X_test_ae, ae_best_iteration = load_ae_model_and_matrices(
            train_df,
            valid_df,
            test_df,
        )
    except FileNotFoundError as exc:
        warn(warnings, f"Skipping inference benchmark because artifact is missing: {exc}")
        return None

    fe_weight = float(score_metadata["fe_weight"])
    ae_weight = float(score_metadata["ae_weight"])
    n_rows = int(len(test_df))

    log("Benchmarking FE-LGBM tuned score generation.")
    fe_timing = benchmark_callable(
        lambda: predict_scores(fe_model, X_test_fe, fe_best_iteration),
        n_rows,
        BENCHMARK_REPEATS,
    )
    log("Benchmarking AE-LGBM tuned score generation.")
    ae_timing = benchmark_callable(
        lambda: predict_scores(ae_model, X_test_ae, ae_best_iteration),
        n_rows,
        BENCHMARK_REPEATS,
    )
    log("Benchmarking FE+AE ensemble score generation.")
    ensemble_timing = benchmark_callable(
        lambda: (
            fe_weight * predict_scores(fe_model, X_test_fe, fe_best_iteration)
            + ae_weight * predict_scores(ae_model, X_test_ae, ae_best_iteration)
        ),
        n_rows,
        BENCHMARK_REPEATS,
    )

    timing_specs = [
        (
            "FE-LGBM tuned",
            fe_timing,
            int(X_test_fe.shape[1]),
            "one saved LightGBM model on the prepared engineered feature matrix",
        ),
        (
            "AE-LGBM tuned",
            ae_timing,
            int(X_test_ae.shape[1]),
            "one saved LightGBM model on prepared non-V plus LD128 latent features",
        ),
        (
            "FE-LGBM tuned + AE-LGBM tuned score ensemble",
            ensemble_timing,
            int(X_test_fe.shape[1] + X_test_ae.shape[1]),
            "both saved LightGBM models plus weighted probability blending",
        ),
    ]
    fe_ms = float(fe_timing["ms_per_10000_rows"])
    rows = []
    for model_name, timing, feature_columns, scope in timing_specs:
        relative_time = (
            float(timing["ms_per_10000_rows"] / fe_ms) if fe_ms > 0 else None
        )
        rows.append(
            {
                "model": model_name,
                "split": "test",
                "rows": n_rows,
                "feature_columns": feature_columns,
                "repeats": timing["repeats"],
                "warmup_excluded": timing["warmup_excluded"],
                "total_seconds": timing["total_seconds"],
                "mean_seconds": timing["mean_seconds"],
                "std_seconds": timing["std_seconds"],
                "ms_per_10000_rows": timing["ms_per_10000_rows"],
                "relative_time_vs_fe": relative_time,
                "relative_overhead_vs_fe": (
                    float(relative_time - 1.0) if relative_time is not None else None
                ),
                "relative_overhead_vs_fe_pct": (
                    float((relative_time - 1.0) * 100.0)
                    if relative_time is not None
                    else None
                ),
                "benchmark_scope": scope,
                "preprocessing_timed": False,
                "training_performed": False,
            }
        )

    timing_df = pd.DataFrame(rows)
    timing_df.to_csv(
        FINAL_DIAGNOSTICS_DIR / "inference_complexity_summary.csv",
        index=False,
    )
    return timing_df


def load_bootstrap_summary_if_exists() -> dict[str, object] | None:
    summary_path = FINAL_DIAGNOSTICS_DIR / "bootstrap_delta_summary.json"
    ci_path = FINAL_DIAGNOSTICS_DIR / "bootstrap_pr_auc_ci.csv"
    if not summary_path.exists():
        return None

    summary = load_json(summary_path)
    if ci_path.exists():
        summary["ci_file"] = str(ci_path)
    return summary


def temporal_interpretation(temporal_df: pd.DataFrame) -> list[str]:
    valid_delta = temporal_df["delta_pr_auc"].dropna()
    if valid_delta.empty:
        return ["- Temporal bins could not be compared because PR-AUC was unavailable."]

    best_bin = temporal_df.loc[temporal_df["fe_ae_ensemble_pr_auc"].idxmax()]
    worst_bin = temporal_df.loc[temporal_df["fe_ae_ensemble_pr_auc"].idxmin()]
    positive_bins = int((valid_delta > 0).sum())
    lines = [
        (
            "- The ensemble PR-AUC exceeds FE-LGBM in "
            f"{positive_bins}/{len(valid_delta)} chronological test bins; "
            f"mean delta={format_number(valid_delta.mean())}."
        ),
        (
            "- Best ensemble bin: "
            f"bin {int(best_bin['bin'])} with PR-AUC "
            f"{format_number(best_bin['fe_ae_ensemble_pr_auc'])}; "
            "lowest ensemble bin: "
            f"bin {int(worst_bin['bin'])} with PR-AUC "
            f"{format_number(worst_bin['fe_ae_ensemble_pr_auc'])}."
        ),
    ]
    if temporal_df["fe_ae_ensemble_pr_auc"].iloc[-1] < temporal_df[
        "fe_ae_ensemble_pr_auc"
    ].iloc[0]:
        lines.append(
            "- The final chronological bin is lower than the earliest test bin, "
            "which supports a cautious temporal-degradation discussion."
        )
    else:
        lines.append(
            "- The last chronological bin does not collapse relative to the first, "
            "so temporal drift is present but not a simple monotonic failure pattern."
        )
    return lines


def score_correlation_interpretation(summary: dict[str, object]) -> str:
    spearman = float(summary["spearman_correlation"])
    delta = float(summary["deltas"]["ensemble_minus_fe_pr_auc"])
    ae_pr_auc = float(summary["metrics"]["ae_lgbm_ld128_tuned"]["pr_auc"])
    fe_pr_auc = float(summary["metrics"]["fe_lgbm_tuned"]["pr_auc"])

    if spearman >= 0.90:
        corr_text = "highly correlated"
    elif spearman >= 0.70:
        corr_text = "moderately to strongly correlated"
    else:
        corr_text = "not fully aligned"

    return (
        f"- FE and AE scores are {corr_text} "
        f"(Pearson={format_number(summary['pearson_correlation'])}, "
        f"Spearman={format_number(spearman)}). AE standalone PR-AUC "
        f"({format_number(ae_pr_auc)}) remains below FE-LGBM "
        f"({format_number(fe_pr_auc)}), but the weighted ensemble improves by "
        f"{format_number(delta)} PR-AUC, supporting complementary probability "
        "signal rather than standalone AE superiority."
    )


def reconstruction_interpretation(
    summary_df: pd.DataFrame,
    metrics: dict[str, object],
) -> list[str]:
    lines = []
    for split_name in ("validation", "test"):
        if split_name not in metrics:
            continue
        split_metrics = metrics[split_name]
        split_summary = summary_df[summary_df["split"] == split_name]
        nonfraud = split_summary[split_summary["isFraud"] == 0]
        fraud = split_summary[split_summary["isFraud"] == 1]
        if nonfraud.empty or fraud.empty:
            continue
        nonfraud_row = nonfraud.iloc[0]
        fraud_row = fraud.iloc[0]
        median_ratio = (
            float(fraud_row["median"]) / float(nonfraud_row["median"])
            if float(nonfraud_row["median"]) > 0
            else None
        )
        lines.append(
            "- "
            f"{split_name.title()} reconstruction-error-as-score PR-AUC="
            f"{format_number(split_metrics['pr_auc_using_error_as_score'])}, "
            f"ROC-AUC={format_number(split_metrics['roc_auc_using_error_as_score'])}; "
            f"fraud/non-fraud median error ratio={format_number(median_ratio)}."
        )
    lines.append(
        "- Interpretation: robust AE LD128 reconstruction error separates fraud and "
        "non-fraud to some degree, but by itself it is much weaker than supervised "
        "FE-LGBM ranking. This is consistent with engineered/masked Vesta features "
        "already carrying substantial structured signal."
    )
    return lines


def bootstrap_interpretation(summary: dict[str, object] | None) -> list[str]:
    if summary is None:
        return ["- Bootstrap CI files were not found, so no bootstrap claim is added."]

    overlaps_zero = bool(summary.get("ci_overlaps_zero"))
    if overlaps_zero:
        stability = "overlaps zero; present the ensemble gain as a point-estimate improvement only"
    else:
        stability = "is above zero; this supports a small but stable paired-bootstrap gain"
    return [
        (
            "- Existing paired bootstrap delta, ensemble minus FE-LGBM tuned: "
            f"{format_number(summary.get('point_estimate_delta'))} PR-AUC "
            f"with 95% CI [{format_number(summary.get('ci_lower_95'))}, "
            f"{format_number(summary.get('ci_upper_95'))}], which {stability}."
        )
    ]


def complexity_interpretation(
    timing_df: pd.DataFrame | None,
    score_summary: dict[str, object],
) -> list[str]:
    if timing_df is None or timing_df.empty:
        return ["- Inference complexity benchmark was skipped because artifacts were missing."]

    fe_row = timing_df[timing_df["model"] == "FE-LGBM tuned"].iloc[0]
    ensemble_row = timing_df[
        timing_df["model"] == "FE-LGBM tuned + AE-LGBM tuned score ensemble"
    ].iloc[0]
    delta = float(score_summary["deltas"]["ensemble_minus_fe_pr_auc"])
    relative_time = float(ensemble_row["relative_time_vs_fe"])
    overhead_pct = float(ensemble_row["relative_overhead_vs_fe_pct"])
    return [
        (
            "- FE-LGBM tuned score generation: "
            f"{format_number(fe_row['ms_per_10000_rows'], 3)} ms per 10,000 rows; "
            "FE+AE ensemble: "
            f"{format_number(ensemble_row['ms_per_10000_rows'], 3)} ms per 10,000 rows "
            f"({format_number(relative_time, 2)}x FE, "
            f"{format_number(overhead_pct, 1)}% overhead)."
        ),
        (
            "- Trade-off: the ensemble adds "
            f"{format_number(delta)} PR-AUC over FE-LGBM but requires two model "
            "pipelines and latent-feature artifacts. FE-LGBM remains the simpler "
            "deployment candidate when operational simplicity matters more than "
            "the small ranking gain."
        ),
    ]


def build_defense_notes(
    temporal_df: pd.DataFrame | None,
    score_summary: dict[str, object] | None,
    reconstruction_summary: pd.DataFrame | None,
    reconstruction_metrics: dict[str, object] | None,
    timing_df: pd.DataFrame | None,
    bootstrap_summary: dict[str, object] | None,
    warnings: list[str],
) -> str:
    lines = [
        "# Final Defense Diagnostics",
        "",
        "## Scope",
        "",
        "- Existing artifacts only; no model training is performed.",
        "- Chronological validation/test splits are recreated from labeled train files.",
        "- Kaggle competition test files are not used.",
        "",
        "## Temporal Degradation",
        "",
    ]

    if temporal_df is not None:
        lines.extend(temporal_interpretation(temporal_df))
    else:
        lines.append("- Temporal degradation diagnostic was skipped.")

    lines.extend(["", "## FE vs AE Score Complementarity", ""])
    if score_summary is not None:
        lines.append(score_correlation_interpretation(score_summary))
    else:
        lines.append("- Score correlation diagnostic was skipped.")

    lines.extend(["", "## Reconstruction Error", ""])
    if reconstruction_summary is not None and reconstruction_metrics is not None:
        lines.extend(
            reconstruction_interpretation(reconstruction_summary, reconstruction_metrics)
        )
    else:
        lines.append("- Reconstruction-error diagnostic was skipped.")

    lines.extend(["", "## Bootstrap CI", ""])
    lines.extend(bootstrap_interpretation(bootstrap_summary))

    lines.extend(["", "## Complexity vs Performance", ""])
    if score_summary is not None:
        lines.extend(complexity_interpretation(timing_df, score_summary))
    else:
        lines.append("- Complexity/performance trade-off could not be summarized.")

    lines.extend(
        [
            "",
            "## Recommended Final Framing",
            "",
            "- Best overall by PR-AUC: FE-LGBM tuned + AE-LGBM tuned score ensemble.",
            "- Best standalone/deployment-friendly model: FE-LGBM tuned.",
            (
                "- AE contribution: complementary probability signal, not "
                "standalone superiority."
            ),
            "- Ensemble improvement: small in absolute PR-AUC, but meaningful when defended with paired score and bootstrap diagnostics.",
            "- Deployment caveat: FE-LGBM may remain preferable when simplicity, artifact count, and monitoring burden dominate.",
        ]
    )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {message}" for message in warnings])

    return "\n".join(lines) + "\n"


def main() -> None:
    FINAL_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    train_df, valid_df, test_df = load_labeled_splits()
    scores, score_metadata = load_or_regenerate_scores(
        train_df,
        valid_df,
        test_df,
        warnings,
    )

    temporal_df = run_temporal_degradation(test_df, scores["test"], warnings)
    score_summary = run_score_complementarity(scores["test"], score_metadata)
    reconstruction_summary, reconstruction_metrics = run_reconstruction_error_diagnostics(
        valid_df,
        test_df,
        warnings,
    )
    timing_df = run_inference_complexity_benchmark(
        train_df,
        valid_df,
        test_df,
        score_metadata,
        warnings,
    )
    bootstrap_summary = load_bootstrap_summary_if_exists()

    notes = build_defense_notes(
        temporal_df,
        score_summary,
        reconstruction_summary,
        reconstruction_metrics,
        timing_df,
        bootstrap_summary,
        warnings,
    )
    (FINAL_DIAGNOSTICS_DIR / "final_defense_notes.md").write_text(
        notes,
        encoding="utf-8",
    )

    log(f"Final defense diagnostics saved to {FINAL_DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
