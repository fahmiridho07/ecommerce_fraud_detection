"""Audit Phase 3 Autoencoder reconstruction behavior without refitting."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AE_BATCH_SIZE,
    AUTOENCODER_OUTPUT_DIR,
    ID_COL,
    OUTPUT_DIR,
    SAMPLE_SIZE,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns
from splitting import chronological_split
from utils import ensure_dir, log, save_json


AUDIT_OUTPUT_DIR = OUTPUT_DIR / "autoencoder_audit"


def load_autoencoder_run_config() -> dict[str, object]:
    config_path = AUTOENCODER_OUTPUT_DIR / "run_config.json"
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_v_column_consistency(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler,
    autoencoder: keras.Model,
    output_dir: Path,
) -> tuple[list[str], dict[str, object]]:
    """Verify V-feature names and order across splits, scaler, and model."""
    split_columns = {
        "train": get_v_feature_columns(train_df),
        "validation": get_v_feature_columns(valid_df),
        "test": get_v_feature_columns(test_df),
    }
    train_columns = split_columns["train"]
    run_config_columns = load_autoencoder_run_config().get("v_columns", [])
    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    model_input_dim = int(autoencoder.input_shape[-1])

    comparisons = {
        "validation_matches_train_order": split_columns["validation"] == train_columns,
        "test_matches_train_order": split_columns["test"] == train_columns,
        "scaler_matches_train_order": scaler_columns == train_columns,
        "run_config_matches_train_order": run_config_columns == train_columns,
        "model_input_dim_matches_v_count": model_input_dim == len(train_columns),
    }
    consistency_ok = all(comparisons.values())

    missing_extra = {}
    for split_name, columns in split_columns.items():
        missing_extra[split_name] = {
            "missing_vs_train": sorted(set(train_columns) - set(columns)),
            "extra_vs_train": sorted(set(columns) - set(train_columns)),
        }

    report = {
        "v_column_order_correct": consistency_ok,
        "comparisons": comparisons,
        "v_feature_count": len(train_columns),
        "model_input_dim": model_input_dim,
        "missing_extra_by_split": missing_extra,
    }
    save_json(train_columns, output_dir / "ordered_v_columns.json")
    save_json(report, output_dir / "v_column_consistency.json")
    return train_columns, report


def raw_distribution_by_split(
    split_name: str,
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Summarize raw V-feature distributions for one split."""
    values = df.loc[:, columns]
    summary = pd.DataFrame(
        {
            "split": split_name,
            "feature": columns,
            "missing_rate": values.isna().mean().to_numpy(),
            "mean": values.mean(skipna=True).to_numpy(),
            "std": values.std(skipna=True).to_numpy(),
            "min": values.min(skipna=True).to_numpy(),
            "max": values.max(skipna=True).to_numpy(),
            "p01": values.quantile(0.01).to_numpy(),
            "p99": values.quantile(0.99).to_numpy(),
        }
    )
    return summary


def transform_with_saved_scaler(
    df: pd.DataFrame,
    columns: list[str],
    scaler,
) -> np.ndarray:
    """Apply the existing train-fitted scaler; do not refit anything."""
    values = df.loc[:, columns].fillna(0).astype("float32")
    return scaler.transform(values).astype("float32")


def scaled_distribution_by_split(
    split_name: str,
    scaled_values: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Summarize scaled V-feature distributions for one split."""
    abs_values = np.abs(scaled_values)
    return pd.DataFrame(
        {
            "split": split_name,
            "feature": columns,
            "mean": np.mean(scaled_values, axis=0),
            "std": np.std(scaled_values, axis=0),
            "min": np.min(scaled_values, axis=0),
            "max": np.max(scaled_values, axis=0),
            "p01": np.percentile(scaled_values, 1, axis=0),
            "p99": np.percentile(scaled_values, 99, axis=0),
            "pct_abs_gt_5": np.mean(abs_values > 5, axis=0),
            "pct_abs_gt_10": np.mean(abs_values > 10, axis=0),
            "pct_abs_gt_20": np.mean(abs_values > 20, axis=0),
            "max_abs": np.max(abs_values, axis=0),
        }
    )


def build_problematic_feature_table(
    raw_summary: pd.DataFrame,
    scaled_summary: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """Rank features by distribution shift and scaled outlier behavior."""
    raw_train = raw_summary[raw_summary["split"] == "train"].set_index("feature")
    raw_test = raw_summary[raw_summary["split"] == "test"].set_index("feature")
    scaled_train = scaled_summary[scaled_summary["split"] == "train"].set_index("feature")
    scaled_test = scaled_summary[scaled_summary["split"] == "test"].set_index("feature")

    feature_table = pd.DataFrame(index=scaled_test.index)
    feature_table["raw_missing_rate_shift"] = (
        raw_test["missing_rate"] - raw_train["missing_rate"]
    ).abs()
    feature_table["scaled_mean_abs_diff"] = (
        scaled_test["mean"] - scaled_train["mean"]
    ).abs()
    feature_table["scaled_std_abs_diff"] = (
        scaled_test["std"] - scaled_train["std"]
    ).abs()
    feature_table["scaled_p01_abs_diff"] = (
        scaled_test["p01"] - scaled_train["p01"]
    ).abs()
    feature_table["scaled_p99_abs_diff"] = (
        scaled_test["p99"] - scaled_train["p99"]
    ).abs()
    feature_table["distribution_shift_score"] = feature_table[
        [
            "scaled_mean_abs_diff",
            "scaled_std_abs_diff",
            "scaled_p01_abs_diff",
            "scaled_p99_abs_diff",
            "raw_missing_rate_shift",
        ]
    ].sum(axis=1)
    feature_table["test_pct_abs_gt_5"] = scaled_test["pct_abs_gt_5"]
    feature_table["test_pct_abs_gt_10"] = scaled_test["pct_abs_gt_10"]
    feature_table["test_pct_abs_gt_20"] = scaled_test["pct_abs_gt_20"]
    feature_table["test_max_abs_scaled"] = scaled_test["max_abs"]

    rankings = [
        (
            "test_train_distribution_shift",
            feature_table.sort_values("distribution_shift_score", ascending=False),
        ),
        (
            "test_scaled_outlier_rate",
            feature_table.sort_values(
                ["test_pct_abs_gt_20", "test_pct_abs_gt_10", "test_pct_abs_gt_5"],
                ascending=False,
            ),
        ),
        (
            "test_max_abs_scaled_value",
            feature_table.sort_values("test_max_abs_scaled", ascending=False),
        ),
    ]

    rows = []
    for ranking_name, ranking_df in rankings:
        top_features = ranking_df.head(top_n).reset_index()
        top_features.insert(0, "rank", np.arange(1, len(top_features) + 1))
        top_features.insert(0, "ranking_type", ranking_name)
        rows.append(top_features)
    return pd.concat(rows, ignore_index=True)


def reconstruction_errors(
    autoencoder: keras.Model,
    scaled_values: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    reconstructed = autoencoder.predict(scaled_values, batch_size=batch_size, verbose=0)
    return np.mean(np.square(scaled_values - reconstructed), axis=1)


def error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "p99_9": float(np.percentile(errors, 99.9)),
    }


def count_threshold_exceedances(errors: np.ndarray) -> dict[str, int]:
    thresholds = [1, 5, 10, 50, 100]
    return {f"gt_{threshold}": int(np.sum(errors > threshold)) for threshold in thresholds}


def plot_reconstruction_errors(
    train_errors: np.ndarray,
    valid_errors: np.ndarray,
    test_errors: np.ndarray,
    output_dir: Path,
) -> None:
    """Save optional diagnostic plots when matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed; skipping optional plots.")
        return

    plt.figure(figsize=(10, 6))
    for split_name, errors in (
        ("train", train_errors),
        ("validation", valid_errors),
        ("test", test_errors),
    ):
        plt.hist(
            np.log1p(errors),
            bins=80,
            alpha=0.45,
            density=True,
            label=split_name,
        )
    plt.xlabel("log1p(reconstruction MSE)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_error_histogram_log1p.png", dpi=150)
    plt.close()

    percentiles = np.array([50, 90, 95, 99, 99.9, 100])
    percentile_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "train": np.percentile(train_errors, percentiles),
            "validation": np.percentile(valid_errors, percentiles),
            "test": np.percentile(test_errors, percentiles),
        }
    )
    percentile_df.to_csv(output_dir / "reconstruction_error_percentiles.csv", index=False)

    plt.figure(figsize=(10, 6))
    for split_name in ("train", "validation", "test"):
        plt.plot(percentile_df["percentile"], percentile_df[split_name], marker="o", label=split_name)
    plt.yscale("log")
    plt.xlabel("percentile")
    plt.ylabel("reconstruction MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reconstruction_error_percentiles.png", dpi=150)
    plt.close()


def diagnose(
    column_report: dict[str, object],
    error_summary_payload: dict[str, object],
    top_features: pd.DataFrame,
) -> dict[str, object]:
    test_summary = error_summary_payload["reconstruction_error_summary"]["test"]
    test_counts = error_summary_payload["test_error_counts"]
    test_rows = error_summary_payload["row_counts"]["test"]
    high_error_rate_gt_100 = test_counts["gt_100"] / test_rows
    broad_shift = test_summary["median"] > 1 or test_summary["p90"] > 5

    top_shift = top_features[
        top_features["ranking_type"] == "test_train_distribution_shift"
    ].head(10)["feature"].tolist()
    top_outlier = top_features[
        top_features["ranking_type"] == "test_scaled_outlier_rate"
    ].head(10)["feature"].tolist()
    top_max_abs = top_features[
        top_features["ranking_type"] == "test_max_abs_scaled_value"
    ].head(10)["feature"].tolist()

    likely_cause = (
        "likely temporal distribution drift / extreme outliers"
        if column_report["v_column_order_correct"]
        else "possible preprocessing or column-order issue"
    )
    concentration = (
        "broad distribution shift"
        if broad_shift
        else "mostly concentrated in a minority of high-error rows"
    )

    return {
        "v_column_order_correct": column_report["v_column_order_correct"],
        "test_error_concentration": concentration,
        "likely_cause": likely_cause,
        "test_error_rate_gt_100": high_error_rate_gt_100,
        "top_distribution_shift_features": top_shift,
        "top_scaled_outlier_rate_features": top_outlier,
        "top_max_abs_scaled_features": top_max_abs,
    }


def main() -> None:
    output_dir = ensure_dir(AUDIT_OUTPUT_DIR)

    log("Loading Phase 1 data split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)

    log("Loading saved Phase 3 Autoencoder artifacts.")
    scaler = joblib.load(AUTOENCODER_OUTPUT_DIR / "v_scaler.pkl")
    autoencoder = keras.models.load_model(
        AUTOENCODER_OUTPUT_DIR / "autoencoder_model.keras",
        compile=False,
    )

    log("Checking V-feature column consistency.")
    v_columns, column_report = check_v_column_consistency(
        train_df,
        valid_df,
        test_df,
        scaler,
        autoencoder,
        output_dir,
    )

    log("Summarizing raw V-feature distributions.")
    raw_summary = pd.concat(
        [
            raw_distribution_by_split("train", train_df, v_columns),
            raw_distribution_by_split("validation", valid_df, v_columns),
            raw_distribution_by_split("test", test_df, v_columns),
        ],
        ignore_index=True,
    )
    raw_summary.to_csv(
        output_dir / "v_feature_distribution_by_split.csv",
        index=False,
    )

    log("Applying saved scaler and summarizing scaled V-feature distributions.")
    X_train = transform_with_saved_scaler(train_df, v_columns, scaler)
    X_valid = transform_with_saved_scaler(valid_df, v_columns, scaler)
    X_test = transform_with_saved_scaler(test_df, v_columns, scaler)
    scaled_summary = pd.concat(
        [
            scaled_distribution_by_split("train", X_train, v_columns),
            scaled_distribution_by_split("validation", X_valid, v_columns),
            scaled_distribution_by_split("test", X_test, v_columns),
        ],
        ignore_index=True,
    )
    scaled_summary.to_csv(
        output_dir / "scaled_v_feature_distribution_by_split.csv",
        index=False,
    )

    log("Ranking problematic V-features.")
    problematic_features = build_problematic_feature_table(raw_summary, scaled_summary)
    problematic_features.to_csv(
        output_dir / "top_problematic_v_features.csv",
        index=False,
    )

    log("Recomputing reconstruction errors with the saved Autoencoder.")
    train_errors = reconstruction_errors(autoencoder, X_train, AE_BATCH_SIZE)
    valid_errors = reconstruction_errors(autoencoder, X_valid, AE_BATCH_SIZE)
    test_errors = reconstruction_errors(autoencoder, X_test, AE_BATCH_SIZE)

    error_summary_payload = {
        "reconstruction_error_summary": {
            "train": error_summary(train_errors),
            "validation": error_summary(valid_errors),
            "test": error_summary(test_errors),
        },
        "test_error_counts": count_threshold_exceedances(test_errors),
        "row_counts": {
            "train": int(len(train_errors)),
            "validation": int(len(valid_errors)),
            "test": int(len(test_errors)),
        },
    }
    save_json(error_summary_payload, output_dir / "reconstruction_error_summary.json")

    top_test_errors = test_df[[ID_COL, TIME_COL, TARGET_COL]].copy()
    top_test_errors["reconstruction_error"] = test_errors
    top_test_errors = top_test_errors.sort_values(
        "reconstruction_error",
        ascending=False,
    ).head(100)
    top_test_errors.to_csv(
        output_dir / "top_test_reconstruction_errors.csv",
        index=False,
    )

    log("Saving optional reconstruction plots.")
    plot_reconstruction_errors(train_errors, valid_errors, test_errors, output_dir)

    diagnosis = diagnose(column_report, error_summary_payload, problematic_features)
    save_json(diagnosis, output_dir / "diagnosis_summary.json")

    test_counts = error_summary_payload["test_error_counts"]
    test_summary = error_summary_payload["reconstruction_error_summary"]["test"]
    top_shift_features = ", ".join(diagnosis["top_distribution_shift_features"][:5])
    top_outlier_features = ", ".join(diagnosis["top_scaled_outlier_rate_features"][:5])
    top_max_abs_features = ", ".join(diagnosis["top_max_abs_scaled_features"][:5])

    print()
    print("Autoencoder Audit Summary")
    print("=========================")
    print(f"V-column order correct        : {diagnosis['v_column_order_correct']}")
    print(f"Test reconstruction mean MSE  : {test_summary['mean']:.6f}")
    print(f"Test reconstruction median MSE: {test_summary['median']:.6f}")
    print(f"Test reconstruction p99 MSE   : {test_summary['p99']:.6f}")
    print(f"Test rows with MSE > 1        : {test_counts['gt_1']:,}")
    print(f"Test rows with MSE > 5        : {test_counts['gt_5']:,}")
    print(f"Test rows with MSE > 10       : {test_counts['gt_10']:,}")
    print(f"Test rows with MSE > 50       : {test_counts['gt_50']:,}")
    print(f"Test rows with MSE > 100      : {test_counts['gt_100']:,}")
    print(f"Error concentration           : {diagnosis['test_error_concentration']}")
    print(f"Likely cause                  : {diagnosis['likely_cause']}")
    print(f"Top shift features            : {top_shift_features}")
    print(f"Top outlier-rate features     : {top_outlier_features}")
    print(f"Top max-abs features          : {top_max_abs_features}")
    print(f"Audit outputs saved to        : {output_dir}")


if __name__ == "__main__":
    main()
