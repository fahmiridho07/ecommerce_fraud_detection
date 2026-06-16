"""Generate thesis diagnostic analyses from existing experiment artifacts.

This script intentionally performs no model training. It uses saved
reconstruction errors, saved model score files, and saved LightGBM artifacts to
support discussion of why the Autoencoder representation did not outperform the
feature-engineered LightGBM standalone model.
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
    PROJECT_ROOT,
    TARGET_COL,
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


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_diagnostics"
FINAL_REPORT_DIR = PROJECT_ROOT / "outputs" / "final_report"
FINAL_REPORT_NOTES_PATH = FINAL_REPORT_DIR / "diagnostic_notes.md"

FE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features"
AE_TUNED_DIR = OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128"

RECONSTRUCTION_ERROR_FILES = {
    "train": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "reconstruction_error_train.csv",
    "validation": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "reconstruction_error_valid.csv",
    "test": AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "reconstruction_error_test.csv",
}

N_BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42
BENCHMARK_REPEATS = 5
EPSILON = 1e-12


def log(message: str) -> None:
    print(f"[diagnostics] {message}", flush=True)


def ensure_required_files(paths: list[Path], context: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s) for {context}:\n"
            + "\n".join(missing)
            + "\nThis diagnostic script does not retrain models or recreate old outputs."
        )


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload: dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def load_labeled_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("Loading labeled train files and recreating the chronological split.")
    full_df = load_labeled_train_data(sample_size=None)
    return chronological_split(full_df)


def reconstruction_error_column(df: pd.DataFrame, path: Path) -> str:
    candidates = ["reconstruction_mse", "reconstruction_error", "mse", "error"]
    for column in candidates:
        if column in df.columns:
            return column
    if df.shape[1] == 1:
        return str(df.columns[0])
    raise ValueError(
        f"Could not identify reconstruction-error column in {path}. "
        f"Columns found: {df.columns.tolist()}"
    )


def read_reconstruction_errors(split_name: str) -> np.ndarray:
    path = RECONSTRUCTION_ERROR_FILES[split_name]
    df = pd.read_csv(path)
    column = reconstruction_error_column(df, path)
    errors = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype="float64")
    if np.isnan(errors).any():
        raise ValueError(f"{path} contains non-numeric or missing reconstruction errors.")
    return errors


def class_distribution_summary(
    split_name: str,
    y_true: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_value in (0, 1):
        values = errors[y_true == class_value]
        if values.size == 0:
            raise ValueError(f"No rows for class {class_value} in {split_name}.")
        rows.append(
            {
                "split": split_name,
                "isFraud": class_value,
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }
        )
    return rows


def safe_roc_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, score))


def plot_reconstruction_distributions(
    split_payloads: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Robust AE LD128 Reconstruction Error by Fraud Label", fontsize=14)

    all_log_errors = np.concatenate(
        [
            np.log10(np.maximum(payload["errors"], 0.0) + EPSILON)
            for payload in split_payloads.values()
        ]
    )
    bins = np.linspace(
        float(np.nanpercentile(all_log_errors, 0.5)),
        float(np.nanpercentile(all_log_errors, 99.5)),
        60,
    )

    for column_index, split_name in enumerate(("validation", "test")):
        payload = split_payloads[split_name]
        y_true = payload["y_true"]
        log_errors = np.log10(np.maximum(payload["errors"], 0.0) + EPSILON)

        ax_hist = axes[0, column_index]
        for class_value, label, color in (
            (0, "Non-fraud", "#4C78A8"),
            (1, "Fraud", "#F58518"),
        ):
            values = log_errors[y_true == class_value]
            ax_hist.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.55,
                color=color,
                label=label,
            )
        ax_hist.set_title(f"{split_name.title()} histogram")
        ax_hist.set_xlabel("log10(reconstruction MSE)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend()

        ax_box = axes[1, column_index]
        boxplot_values = [log_errors[y_true == 0], log_errors[y_true == 1]]
        boxplot_kwargs = {
            "showfliers": False,
            "patch_artist": True,
            "boxprops": {"facecolor": "#DDE8F4", "color": "#3C5B76"},
            "medianprops": {"color": "#222222"},
        }
        try:
            ax_box.boxplot(
                boxplot_values,
                tick_labels=["Non-fraud", "Fraud"],
                **boxplot_kwargs,
            )
        except TypeError:
            ax_box.boxplot(
                boxplot_values,
                labels=["Non-fraud", "Fraud"],
                **boxplot_kwargs,
            )
        ax_box.set_title(f"{split_name.title()} boxplot")
        ax_box.set_ylabel("log10(reconstruction MSE)")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_reconstruction_diagnostics(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    log("Running reconstruction-error diagnostics.")
    ensure_required_files(
        list(RECONSTRUCTION_ERROR_FILES.values()),
        "robust AE LD128 reconstruction-error analysis",
    )

    split_frames = {"validation": valid_df, "test": test_df}
    summary_rows: list[dict[str, object]] = []
    metrics: dict[str, dict[str, object]] = {}
    plot_payloads: dict[str, dict[str, np.ndarray]] = {}

    for split_name, split_df in split_frames.items():
        errors = read_reconstruction_errors(split_name)
        y_true = split_df[TARGET_COL].astype(int).to_numpy()
        if errors.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"{split_name} reconstruction error row count {errors.shape[0]} "
                f"does not match split row count {y_true.shape[0]}."
            )

        summary_rows.extend(class_distribution_summary(split_name, y_true, errors))
        base_rate = float(np.mean(y_true))
        metrics[split_name] = {
            "rows": int(y_true.shape[0]),
            "fraud_count": int(y_true.sum()),
            "fraud_rate": base_rate,
            "average_precision": float(average_precision_score(y_true, errors)),
            "roc_auc": safe_roc_auc(y_true, errors),
            "score_direction": "higher reconstruction_mse treated as more anomalous",
            "source_file": str(RECONSTRUCTION_ERROR_FILES[split_name]),
        }
        plot_payloads[split_name] = {"y_true": y_true, "errors": errors}

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "reconstruction_error_class_summary.csv", index=False)
    save_json(metrics, OUTPUT_DIR / "reconstruction_error_anomaly_metrics.json")
    plot_reconstruction_distributions(
        plot_payloads,
        OUTPUT_DIR / "reconstruction_error_distribution.png",
    )
    return {"summary": summary_df, "metrics": metrics}


def score_column(df: pd.DataFrame, preferred: list[str], path: Path) -> str:
    for column in preferred:
        if column in df.columns:
            return column
    numeric_columns = [
        column
        for column in df.columns
        if column not in {ID_COL, TARGET_COL} and pd.api.types.is_numeric_dtype(df[column])
    ]
    if len(numeric_columns) == 1:
        return numeric_columns[0]
    raise ValueError(
        f"Could not identify score column in {path}. "
        f"Preferred={preferred}, columns={df.columns.tolist()}"
    )


def load_bootstrap_scores() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    ensemble_path = FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR / "scores_test.csv"
    ensure_required_files([ensemble_path], "bootstrap comparison score loading")
    ensemble_df = pd.read_csv(ensemble_path)
    ensure_columns = [ID_COL, TARGET_COL, "ensemble_score"]
    missing = [column for column in ensure_columns if column not in ensemble_df.columns]
    if missing:
        raise KeyError(f"{ensemble_path} is missing required column(s): {missing}")

    fe_source = "ensemble_score_file_fe_lgbm_tuned_score_column"
    fe_path = FE_TUNED_DIR / "scores_test.csv"
    if fe_path.exists():
        fe_df = pd.read_csv(fe_path)
        fe_score_col = score_column(fe_df, ["score", "fe_lgbm_tuned_score"], fe_path)
        required = [ID_COL, TARGET_COL, fe_score_col]
        missing = [column for column in required if column not in fe_df.columns]
        if missing:
            raise KeyError(f"{fe_path} is missing required column(s): {missing}")
        merged = ensemble_df[[ID_COL, TARGET_COL, "ensemble_score"]].merge(
            fe_df[[ID_COL, TARGET_COL, fe_score_col]],
            on=ID_COL,
            suffixes=("_ensemble", "_fe"),
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(ensemble_df):
            raise ValueError(
                "Optuna FE score file does not align one-to-one with ensemble test scores."
            )
        if not np.array_equal(
            merged[f"{TARGET_COL}_ensemble"].to_numpy(),
            merged[f"{TARGET_COL}_fe"].to_numpy(),
        ):
            raise ValueError("FE and ensemble score files have mismatched labels.")
        y_true = merged[f"{TARGET_COL}_ensemble"].astype(int).to_numpy()
        fe_score = merged[fe_score_col].to_numpy(dtype="float64")
        ensemble_score = merged["ensemble_score"].to_numpy(dtype="float64")
        fe_source = str(fe_path)
    elif "fe_lgbm_tuned_score" in ensemble_df.columns:
        y_true = ensemble_df[TARGET_COL].astype(int).to_numpy()
        fe_score = ensemble_df["fe_lgbm_tuned_score"].to_numpy(dtype="float64")
        ensemble_score = ensemble_df["ensemble_score"].to_numpy(dtype="float64")
    else:
        raise FileNotFoundError(
            "Missing FE tuned scores. Expected either:\n"
            f"{fe_path}\n"
            f"or column 'fe_lgbm_tuned_score' in {ensemble_path}."
        )

    metadata = {
        "split": "test",
        "fe_score_source": fe_source,
        "ensemble_score_source": str(ensemble_path),
        "rows": int(y_true.shape[0]),
    }
    return y_true, fe_score, ensemble_score, metadata


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lower, upper = np.nanpercentile(values, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lower), float(upper)


def run_bootstrap_pr_auc() -> dict[str, object]:
    log("Running paired bootstrap PR-AUC comparison.")
    y_true, fe_score, ensemble_score, metadata = load_bootstrap_scores()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_rows = int(y_true.shape[0])

    fe_boot = np.empty(N_BOOTSTRAPS, dtype="float64")
    ensemble_boot = np.empty(N_BOOTSTRAPS, dtype="float64")
    delta_boot = np.empty(N_BOOTSTRAPS, dtype="float64")

    for iteration in range(N_BOOTSTRAPS):
        indices = rng.integers(0, n_rows, size=n_rows)
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            fe_boot[iteration] = np.nan
            ensemble_boot[iteration] = np.nan
            delta_boot[iteration] = np.nan
            continue
        fe_ap = average_precision_score(sampled_y, fe_score[indices])
        ensemble_ap = average_precision_score(sampled_y, ensemble_score[indices])
        fe_boot[iteration] = fe_ap
        ensemble_boot[iteration] = ensemble_ap
        delta_boot[iteration] = ensemble_ap - fe_ap
        if (iteration + 1) % 100 == 0:
            log(f"Bootstrap iteration {iteration + 1}/{N_BOOTSTRAPS}.")

    fe_point = float(average_precision_score(y_true, fe_score))
    ensemble_point = float(average_precision_score(y_true, ensemble_score))
    delta_point = float(ensemble_point - fe_point)

    fe_ci = percentile_ci(fe_boot)
    ensemble_ci = percentile_ci(ensemble_boot)
    delta_ci = percentile_ci(delta_boot)

    ci_df = pd.DataFrame(
        [
            {
                "model": "FE-LGBM tuned",
                "split": metadata["split"],
                "point_estimate_pr_auc": fe_point,
                "bootstrap_mean_pr_auc": float(np.nanmean(fe_boot)),
                "ci_lower_95": fe_ci[0],
                "ci_upper_95": fe_ci[1],
                "n_bootstraps": N_BOOTSTRAPS,
                "seed": BOOTSTRAP_SEED,
                "score_source": metadata["fe_score_source"],
            },
            {
                "model": "FE-LGBM tuned + AE-LGBM tuned score ensemble",
                "split": metadata["split"],
                "point_estimate_pr_auc": ensemble_point,
                "bootstrap_mean_pr_auc": float(np.nanmean(ensemble_boot)),
                "ci_lower_95": ensemble_ci[0],
                "ci_upper_95": ensemble_ci[1],
                "n_bootstraps": N_BOOTSTRAPS,
                "seed": BOOTSTRAP_SEED,
                "score_source": metadata["ensemble_score_source"],
            },
        ]
    )
    ci_df.to_csv(OUTPUT_DIR / "bootstrap_pr_auc_ci.csv", index=False)

    delta_summary = {
        "comparison": "FE+AE score ensemble minus FE-LGBM tuned",
        "metric": "average_precision / PR-AUC",
        "split": metadata["split"],
        "point_estimate_delta": delta_point,
        "bootstrap_mean_delta": float(np.nanmean(delta_boot)),
        "bootstrap_std_delta": float(np.nanstd(delta_boot, ddof=1)),
        "ci_lower_95": delta_ci[0],
        "ci_upper_95": delta_ci[1],
        "ci_overlaps_zero": bool(delta_ci[0] <= 0.0 <= delta_ci[1]),
        "probability_delta_gt_zero": float(np.nanmean(delta_boot > 0.0)),
        "n_bootstraps": N_BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "rows": metadata["rows"],
        "fe_score_source": metadata["fe_score_source"],
        "ensemble_score_source": metadata["ensemble_score_source"],
        "paired_bootstrap": True,
    }
    save_json(delta_summary, OUTPUT_DIR / "bootstrap_delta_summary.json")
    plot_bootstrap_delta(delta_boot, delta_point, delta_ci)
    return {
        "ci": ci_df,
        "delta_summary": delta_summary,
        "delta_distribution": delta_boot,
    }


def plot_bootstrap_delta(
    delta_boot: np.ndarray,
    delta_point: float,
    delta_ci: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(delta_boot[np.isfinite(delta_boot)], bins=50, color="#4C78A8", alpha=0.8)
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="Zero delta")
    ax.axvline(delta_point, color="#F58518", linewidth=1.8, label="Point delta")
    ax.axvline(delta_ci[0], color="#54A24B", linestyle=":", linewidth=1.6, label="95% CI")
    ax.axvline(delta_ci[1], color="#54A24B", linestyle=":", linewidth=1.6)
    ax.set_title("Paired Bootstrap Delta in Test PR-AUC")
    ax.set_xlabel("Delta PR-AUC: ensemble - FE-LGBM tuned")
    ax.set_ylabel("Bootstrap samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bootstrap_delta_distribution.png", dpi=160)
    plt.close(fig)


def load_latent_feature_names() -> list[str]:
    path = AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_feature_names.json"
    ensure_required_files([path], "AE latent feature names")
    names = load_json(path)
    if not isinstance(names, list):
        raise TypeError(f"{path} must contain a list.")
    return [str(name) for name in names]


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
            f"{model_name} feature-name mismatch. "
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


def load_fe_model_and_matrix(test_df: pd.DataFrame) -> tuple[object, pd.DataFrame, int | None]:
    ensure_required_files(
        [
            FE_TUNED_DIR / "final_model.pkl",
            FE_TUNED_DIR / "preprocessing.pkl",
            FE_TUNED_DIR / "feature_engineering.pkl",
            FE_TUNED_DIR / "run_config.json",
        ],
        "FE-LGBM tuned inference benchmark",
    )
    model = joblib.load(FE_TUNED_DIR / "final_model.pkl")
    preprocessing = joblib.load(FE_TUNED_DIR / "preprocessing.pkl")
    feature_artifacts = joblib.load(FE_TUNED_DIR / "feature_engineering.pkl")
    run_config = load_json(FE_TUNED_DIR / "run_config.json")

    X_test_raw, _ = split_features_target(test_df)
    X_test_engineered = apply_entity_time_amount_features(X_test_raw, feature_artifacts)
    X_test = apply_baseline_preprocessing(X_test_engineered, preprocessing)
    validate_model_features(model, X_test, "FE-LGBM tuned")
    return model, X_test, best_iteration_from_config(model, run_config)


def load_ae_model_and_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[object, pd.DataFrame, int | None]:
    ensure_required_files(
        [
            AE_TUNED_DIR / "final_model.pkl",
            AE_TUNED_DIR / "preprocessing_non_v.pkl",
            AE_TUNED_DIR / "run_config.json",
            AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_test.npy",
            AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_feature_names.json",
        ],
        "AE-LGBM tuned inference benchmark",
    )
    model = joblib.load(AE_TUNED_DIR / "final_model.pkl")
    preprocessing = joblib.load(AE_TUNED_DIR / "preprocessing_non_v.pkl")
    run_config = load_json(AE_TUNED_DIR / "run_config.json")

    latent_test = np.load(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR / "latent_test.npy")
    latent_feature_names = load_latent_feature_names()
    if latent_test.shape[0] != len(test_df):
        raise ValueError(
            f"latent_test rows {latent_test.shape[0]} do not match test rows {len(test_df)}."
        )
    if latent_test.shape[1] != len(latent_feature_names):
        raise ValueError("latent_test column count does not match latent feature names.")

    v_columns = get_v_feature_columns(train_df)
    X_test_non_v_raw, _ = split_non_v_features_target(test_df, v_columns)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing)
    X_test = combine_non_v_and_latent(X_test_non_v, latent_test, latent_feature_names)
    validate_model_features(model, X_test, "AE-LGBM tuned")
    return model, X_test, best_iteration_from_config(model, run_config)


def benchmark_callable(
    predict_fn: Callable[[], np.ndarray],
    expected_rows: int,
    repeats: int,
) -> dict[str, object]:
    warmup_scores = predict_fn()
    if warmup_scores.shape[0] != expected_rows:
        raise ValueError(
            f"Warmup prediction returned {warmup_scores.shape[0]} rows; "
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
        "total_prediction_time_seconds": total_seconds,
        "mean_prediction_time_seconds": mean_seconds,
        "std_prediction_time_seconds": float(elapsed_array.std(ddof=1)) if repeats > 1 else 0.0,
        "ms_per_10000_rows": float((mean_seconds * 1000.0) / (expected_rows / 10000.0)),
    }


def run_inference_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    log("Preparing saved feature matrices for inference-time benchmark.")
    fe_model, X_test_fe, fe_best_iteration = load_fe_model_and_matrix(test_df)
    ae_model, X_test_ae, ae_best_iteration = load_ae_model_and_matrix(train_df, test_df)

    ensemble_config = load_json(FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR / "run_config.json")
    ensemble_settings = ensemble_config.get("ensemble", {})
    fe_weight = float(ensemble_settings.get("selected_fe_lgbm_tuned_weight", 0.78))
    ae_weight = float(ensemble_settings.get("selected_ae_lgbm_ld128_tuned_weight", 1.0 - fe_weight))
    n_rows = int(len(test_df))

    log("Timing FE-LGBM tuned predictions.")
    fe_timing = benchmark_callable(
        lambda: predict_scores(fe_model, X_test_fe, fe_best_iteration),
        n_rows,
        BENCHMARK_REPEATS,
    )
    log("Timing AE-LGBM tuned predictions.")
    ae_timing = benchmark_callable(
        lambda: predict_scores(ae_model, X_test_ae, ae_best_iteration),
        n_rows,
        BENCHMARK_REPEATS,
    )
    log("Timing FE+AE score ensemble predictions.")
    ensemble_timing = benchmark_callable(
        lambda: (
            fe_weight * predict_scores(fe_model, X_test_fe, fe_best_iteration)
            + ae_weight * predict_scores(ae_model, X_test_ae, ae_best_iteration)
        ),
        n_rows,
        BENCHMARK_REPEATS,
    )

    rows = []
    timing_payload = [
        (
            "FE-LGBM tuned",
            fe_timing,
            int(X_test_fe.shape[1]),
            "one LightGBM prediction on engineered feature matrix",
        ),
        (
            "AE-LGBM tuned",
            ae_timing,
            int(X_test_ae.shape[1]),
            "one LightGBM prediction on non-V plus LD128 latent feature matrix",
        ),
        (
            "FE-LGBM tuned + AE-LGBM tuned score ensemble",
            ensemble_timing,
            int(X_test_fe.shape[1] + X_test_ae.shape[1]),
            "two LightGBM predictions plus weighted score blending",
        ),
    ]
    fe_ms = float(fe_timing["ms_per_10000_rows"])
    for model_name, timing, feature_columns, complexity_note in timing_payload:
        row = {
            "model": model_name,
            "split": "test",
            "rows": n_rows,
            "feature_columns": feature_columns,
            "repeats": timing["repeats"],
            "warmup_excluded": timing["warmup_excluded"],
            "total_prediction_time_seconds": timing["total_prediction_time_seconds"],
            "mean_prediction_time_seconds": timing["mean_prediction_time_seconds"],
            "std_prediction_time_seconds": timing["std_prediction_time_seconds"],
            "ms_per_10000_rows": timing["ms_per_10000_rows"],
            "relative_time_vs_fe": float(timing["ms_per_10000_rows"] / fe_ms) if fe_ms > 0 else None,
            "complexity_note": complexity_note,
        }
        rows.append(row)

    timing_df = pd.DataFrame(rows)
    timing_df.to_csv(OUTPUT_DIR / "inference_time_summary.csv", index=False)
    return {
        "timing": timing_df,
        "fe_weight": fe_weight,
        "ae_weight": ae_weight,
    }


def format_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build_notes(
    reconstruction_results: dict[str, object],
    bootstrap_results: dict[str, object],
    benchmark_results: dict[str, object],
) -> str:
    reconstruction_summary = reconstruction_results["summary"]
    reconstruction_metrics = reconstruction_results["metrics"]
    delta_summary = bootstrap_results["delta_summary"]
    timing_df = benchmark_results["timing"]

    test_metrics = reconstruction_metrics["test"]
    validation_metrics = reconstruction_metrics["validation"]

    test_summary = reconstruction_summary[reconstruction_summary["split"] == "test"]
    test_nonfraud = test_summary[test_summary[TARGET_COL] == 0].iloc[0]
    test_fraud = test_summary[test_summary[TARGET_COL] == 1].iloc[0]
    median_ratio = (
        float(test_fraud["median"]) / float(test_nonfraud["median"])
        if float(test_nonfraud["median"]) > 0
        else np.nan
    )

    delta_overlaps_zero = bool(delta_summary["ci_overlaps_zero"])
    if delta_overlaps_zero:
        stability_text = (
            "The ensemble has the better point estimate, but the paired bootstrap "
            "95% CI for the PR-AUC delta overlaps zero, so the gain should be "
            "described as not statistically stable in this diagnostic."
        )
    else:
        stability_text = (
            "The paired bootstrap 95% CI for the PR-AUC delta is entirely above "
            "zero, so the ensemble improvement is statistically stable under "
            "this bootstrap diagnostic."
        )

    fe_row = timing_df[timing_df["model"] == "FE-LGBM tuned"].iloc[0]
    ensemble_row = timing_df[
        timing_df["model"] == "FE-LGBM tuned + AE-LGBM tuned score ensemble"
    ].iloc[0]
    relative_time = float(ensemble_row["relative_time_vs_fe"])

    if relative_time <= 1.25:
        complexity_text = (
            "The measured prediction overhead is small, but the ensemble still "
            "requires maintaining two saved model pipelines."
        )
    elif relative_time <= 2.25:
        complexity_text = (
            "The ensemble is roughly the cost of running both component models, "
            "so its small PR-AUC gain should be weighed against extra pipeline "
            "maintenance."
        )
    else:
        complexity_text = (
            "The ensemble is substantially slower than FE standalone, making the "
            "small point-estimate gain harder to justify unless recall ranking "
            "performance is the priority."
        )

    return "\n".join(
        [
            "# Diagnostic Notes",
            "",
            "## Scope",
            "",
            "- These diagnostics use existing artifacts only; no model training is performed.",
            "- Data labels come from the chronological validation/test split of the labeled training files only.",
            "- Kaggle competition test files are not used.",
            "",
            "## Reconstruction Error",
            "",
            (
                "- Validation reconstruction-error anomaly metrics: "
                f"PR-AUC={format_float(validation_metrics['average_precision'])}, "
                f"ROC-AUC={format_float(validation_metrics['roc_auc'])}, "
                f"fraud rate={format_float(validation_metrics['fraud_rate'])}."
            ),
            (
                "- Test reconstruction-error anomaly metrics: "
                f"PR-AUC={format_float(test_metrics['average_precision'])}, "
                f"ROC-AUC={format_float(test_metrics['roc_auc'])}, "
                f"fraud rate={format_float(test_metrics['fraud_rate'])}."
            ),
            (
                "- On the test split, median reconstruction MSE is "
                f"{format_float(test_nonfraud['median'])} for non-fraud and "
                f"{format_float(test_fraud['median'])} for fraud "
                f"(fraud/non-fraud median ratio={format_float(median_ratio)})."
            ),
            (
                "- Interpretation: reconstruction error alone "
                + reconstruction_separation_sentence(test_metrics)
            ),
            "",
            "## Bootstrap PR-AUC",
            "",
            (
                "- Point delta, ensemble minus FE-LGBM tuned: "
                f"{format_float(delta_summary['point_estimate_delta'])} PR-AUC."
            ),
            (
                "- Paired bootstrap 95% CI for delta: "
                f"[{format_float(delta_summary['ci_lower_95'])}, "
                f"{format_float(delta_summary['ci_upper_95'])}]."
            ),
            f"- {stability_text}",
            "",
            "## Inference-Time Complexity",
            "",
            (
                "- FE-LGBM tuned: "
                f"{format_float(fe_row['ms_per_10000_rows'], 3)} ms per 10,000 rows."
            ),
            (
                "- FE+AE score ensemble: "
                f"{format_float(ensemble_row['ms_per_10000_rows'], 3)} ms per 10,000 rows "
                f"({format_float(relative_time, 2)}x FE standalone)."
            ),
            f"- {complexity_text}",
            "",
            "## Thesis Discussion Takeaway",
            "",
            (
                "- The Autoencoder representation is better framed as a source of "
                "limited complementary score signal than as a standalone replacement "
                "for feature-engineered supervised features."
            ),
            (
                "- If the bootstrap CI overlaps zero, report the ensemble as best by "
                "point estimate but avoid claiming a statistically significant gain."
            ),
            "",
        ]
    )


def reconstruction_separation_sentence(test_metrics: dict[str, object]) -> str:
    ap = float(test_metrics["average_precision"])
    base_rate = float(test_metrics["fraud_rate"])
    roc_auc = float(test_metrics["roc_auc"]) if test_metrics["roc_auc"] is not None else 0.5
    if roc_auc >= 0.70 and ap >= 2.0 * base_rate:
        return (
            "shows useful fraud/non-fraud separation, but it remains far below "
            "the tuned supervised LightGBM ranking performance."
        )
    if roc_auc >= 0.60 or ap >= 1.5 * base_rate:
        return (
            "shows weak to moderate separation. This supports the explanation "
            "that reconstruction error captures some anomaly signal but not enough "
            "to dominate supervised FE-LightGBM features."
        )
    return (
        "does not separate fraud from non-fraud well. This supports the thesis "
        "discussion that reconstruction fidelity is not aligned strongly enough "
        "with the fraud label for this dataset."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = load_labeled_splits()
    reconstruction_results = run_reconstruction_diagnostics(valid_df, test_df)
    bootstrap_results = run_bootstrap_pr_auc()
    benchmark_results = run_inference_benchmark(train_df, test_df)

    notes = build_notes(reconstruction_results, bootstrap_results, benchmark_results)
    (OUTPUT_DIR / "diagnostic_notes.md").write_text(notes, encoding="utf-8")
    FINAL_REPORT_NOTES_PATH.write_text(notes, encoding="utf-8")

    log(f"Diagnostics saved to {OUTPUT_DIR}")
    log(f"Final-report diagnostic notes saved to {FINAL_REPORT_NOTES_PATH}")


if __name__ == "__main__":
    main()
