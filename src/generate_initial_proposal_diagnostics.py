"""Generate in-depth diagnostics for the initial proposal P01-P04 path.

This script performs no model training. It analyzes:
- V-feature missingness as a fraud signal
- V-feature gain share in the baseline LightGBM model
- Feature-group gain decomposition for baseline and AE-LightGBM models
- Autoencoder reconstruction drift and fraud-class separation
- Cross-links between baseline V importance and AE missing indicators / latents

Outputs are written under outputs/initial_proposal/diagnostics/ by default.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, SAMPLE_SIZE, TARGET_COL, V_FEATURE_PATTERN
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns
from splitting import chronological_split
from utils import ensure_dir, log, save_json

DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_OUTPUT_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "diagnostics"

V_FEATURE_REGEX = re.compile(V_FEATURE_PATTERN)
LATENT_FEATURE_REGEX = re.compile(r"^ae_latent_\d+$")
MISSING_INDICATOR_REGEX = re.compile(r"^v_missing_V\d+$")

RECONSTRUCTION_SPLITS = ("train", "validation", "test")
RECONSTRUCTION_ERROR_FILENAMES = {
    "train": "reconstruction_error_train.csv",
    "validation": "reconstruction_error_valid.csv",
    "test": "reconstruction_error_test.csv",
}
ROW_MISSING_COUNT_BINS = (
    0,
    1,
    10,
    50,
    100,
    150,
    200,
    250,
    300,
    339,
)
ROW_MISSING_COUNT_LABELS = (
    "000_000",
    "001_009",
    "010_049",
    "050_099",
    "100_149",
    "150_199",
    "200_249",
    "250_299",
    "300_338",
    "339_339",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_feature_importance(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature importance file: {path}")
    importance = pd.read_csv(path)
    required = {"feature", "importance_gain"}
    missing = required - set(importance.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(sorted(missing))}")
    return importance


def classify_feature_group(feature_name: str) -> str:
    if LATENT_FEATURE_REGEX.match(feature_name):
        return "ae_latent"
    if MISSING_INDICATOR_REGEX.match(feature_name):
        return "v_missing_indicator"
    if V_FEATURE_REGEX.match(feature_name):
        return "v_value"
    return "non_v"


def feature_group_gain_summary(
    importance: pd.DataFrame,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = importance.copy()
    working["feature_group"] = working["feature"].map(classify_feature_group)
    grouped = (
        working.groupby("feature_group", as_index=False)
        .agg(
            feature_count=("feature", "count"),
            total_gain=("importance_gain", "sum"),
            total_split=("importance_split", "sum"),
        )
        .sort_values("total_gain", ascending=False)
    )
    total_gain = float(grouped["total_gain"].sum())
    total_split = float(grouped["total_split"].sum())
    grouped["gain_share_pct"] = np.where(
        total_gain > 0,
        grouped["total_gain"] / total_gain * 100.0,
        0.0,
    )
    grouped["split_share_pct"] = np.where(
        total_split > 0,
        grouped["total_split"] / total_split * 100.0,
        0.0,
    )
    grouped.insert(0, "model_name", model_name)

    summary = {
        "model_name": model_name,
        "total_features": int(len(working)),
        "total_gain": total_gain,
        "total_split": total_split,
        "groups": {
            row["feature_group"]: {
                "feature_count": int(row["feature_count"]),
                "total_gain": float(row["total_gain"]),
                "gain_share_pct": float(row["gain_share_pct"]),
                "total_split": float(row["total_split"]),
                "split_share_pct": float(row["split_share_pct"]),
            }
            for _, row in grouped.iterrows()
        },
    }
    return grouped, summary


def top_features_by_gain(
    importance: pd.DataFrame,
    model_name: str,
    feature_group: str | None = None,
    top_n: int = 25,
) -> pd.DataFrame:
    working = importance.copy()
    working["feature_group"] = working["feature"].map(classify_feature_group)
    if feature_group is not None:
        working = working.loc[working["feature_group"] == feature_group]
    working = working.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).head(top_n)
    working.insert(0, "model_name", model_name)
    if feature_group is not None:
        working.insert(1, "filtered_group", feature_group)
    return working.reset_index(drop=True)


def cell_level_missingness_summary(
    split_frames: dict[str, pd.DataFrame],
    v_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, split_df in split_frames.items():
        values = split_df.loc[:, v_columns]
        total_cells = int(values.size)
        missing_cells = int(values.isna().sum().sum())
        rows.append(
            {
                "split": split_name,
                "rows": int(len(split_df)),
                "v_feature_count": len(v_columns),
                "total_v_cells": total_cells,
                "missing_v_cells": missing_cells,
                "observed_v_cells": total_cells - missing_cells,
                "missing_cell_rate": missing_cells / total_cells if total_cells else 0.0,
                "observed_cell_rate": (total_cells - missing_cells) / total_cells
                if total_cells
                else 0.0,
                "fraud_rate": float(split_df[TARGET_COL].mean()),
            }
        )
    return pd.DataFrame(rows)


def assign_missing_count_bin(missing_count: int) -> str:
    if missing_count == 0:
        return ROW_MISSING_COUNT_LABELS[0]
    if missing_count <= 9:
        return ROW_MISSING_COUNT_LABELS[1]
    if missing_count <= 49:
        return ROW_MISSING_COUNT_LABELS[2]
    if missing_count <= 99:
        return ROW_MISSING_COUNT_LABELS[3]
    if missing_count <= 149:
        return ROW_MISSING_COUNT_LABELS[4]
    if missing_count <= 199:
        return ROW_MISSING_COUNT_LABELS[5]
    if missing_count <= 249:
        return ROW_MISSING_COUNT_LABELS[6]
    if missing_count <= 299:
        return ROW_MISSING_COUNT_LABELS[7]
    if missing_count <= 338:
        return ROW_MISSING_COUNT_LABELS[8]
    return ROW_MISSING_COUNT_LABELS[9]


def row_level_missingness_fraud_table(
    split_frames: dict[str, pd.DataFrame],
    v_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, split_df in split_frames.items():
        missing_count = split_df.loc[:, v_columns].isna().sum(axis=1)
        frame = pd.DataFrame(
            {
                "split": split_name,
                "missing_v_count": missing_count.astype(int),
                TARGET_COL: split_df[TARGET_COL].astype(int),
            }
        )
        frame["missing_v_fraction"] = frame["missing_v_count"] / len(v_columns)
        frame["missing_count_bin"] = frame["missing_v_count"].map(assign_missing_count_bin)
        grouped = (
            frame.groupby(["split", "missing_count_bin"], as_index=False)
            .agg(
                rows=(TARGET_COL, "size"),
                missing_v_count_min=("missing_v_count", "min"),
                missing_v_count_max=("missing_v_count", "max"),
                missing_v_fraction_mean=("missing_v_fraction", "mean"),
                fraud_count=(TARGET_COL, "sum"),
                fraud_rate=(TARGET_COL, "mean"),
            )
            .sort_values(["split", "missing_v_count_min"])
        )
        split_total = len(frame)
        grouped["row_share_pct"] = grouped["rows"] / split_total * 100.0
        rows.extend(grouped.to_dict(orient="records"))
    return pd.DataFrame(rows)


def row_missing_count_deciles_train(
    train_df: pd.DataFrame,
    v_columns: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    missing_count = train_df.loc[:, v_columns].isna().sum(axis=1)
    frame = pd.DataFrame(
        {
            "missing_v_count": missing_count.astype(int),
            TARGET_COL: train_df[TARGET_COL].astype(int),
        }
    )
    frame["missing_v_fraction"] = frame["missing_v_count"] / len(v_columns)
    try:
        frame["decile"] = pd.qcut(
            frame["missing_v_count"],
            q=n_bins,
            duplicates="drop",
        )
    except ValueError:
        frame["decile"] = frame["missing_v_count"].astype(str)
    grouped = (
        frame.groupby("decile", observed=False)
        .agg(
            rows=(TARGET_COL, "size"),
            missing_v_count_min=("missing_v_count", "min"),
            missing_v_count_max=("missing_v_count", "max"),
            missing_v_fraction_mean=("missing_v_fraction", "mean"),
            fraud_count=(TARGET_COL, "sum"),
            fraud_rate=(TARGET_COL, "mean"),
        )
        .reset_index()
        .sort_values("missing_v_count_min")
    )
    grouped["row_share_pct"] = grouped["rows"] / len(frame) * 100.0
    return grouped


def per_column_missing_fraud_lift(
    train_df: pd.DataFrame,
    v_columns: list[str],
    baseline_fraud_rate: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = train_df[TARGET_COL].astype(int)
    for column in v_columns:
        missing_mask = train_df[column].isna()
        missing_rows = int(missing_mask.sum())
        observed_rows = int((~missing_mask).sum())
        if missing_rows == 0 or observed_rows == 0:
            continue
        fraud_rate_missing = float(labels.loc[missing_mask].mean())
        fraud_rate_observed = float(labels.loc[~missing_mask].mean())
        rows.append(
            {
                "v_feature": column,
                "missing_rows": missing_rows,
                "observed_rows": observed_rows,
                "missing_rate": missing_rows / len(train_df),
                "fraud_rate_when_missing": fraud_rate_missing,
                "fraud_rate_when_observed": fraud_rate_observed,
                "fraud_rate_lift_missing_vs_observed": (
                    fraud_rate_missing / fraud_rate_observed
                    if fraud_rate_observed > 0
                    else np.nan
                ),
                "fraud_rate_delta_missing_minus_observed": (
                    fraud_rate_missing - fraud_rate_observed
                ),
                "fraud_rate_lift_missing_vs_baseline": (
                    fraud_rate_missing / baseline_fraud_rate
                    if baseline_fraud_rate > 0
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        "fraud_rate_lift_missing_vs_observed",
        ascending=False,
        na_position="last",
    )
    return result.reset_index(drop=True)


def reconstruction_metrics_table(
    autoencoder_dir: Path,
    latent_dim: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics_path = autoencoder_dir / "reconstruction_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing reconstruction metrics: {metrics_path}")
    metrics = load_json(metrics_path)
    rows: list[dict[str, Any]] = []
    for split_name in RECONSTRUCTION_SPLITS:
        split_stats = metrics.get("splits", {}).get(split_name, {})
        if not isinstance(split_stats, dict):
            continue
        rows.append(
            {
                "latent_dim": latent_dim,
                "autoencoder_dir": str(autoencoder_dir),
                "split": split_name,
                "mean_mse": split_stats.get("mean"),
                "median_mse": split_stats.get("median"),
                "p95_mse": split_stats.get("p95"),
                "p99_mse": split_stats.get("p99"),
                "max_mse": split_stats.get("max"),
                "rows_mse_gt_1": split_stats.get("rows_mse_gt_1"),
                "rows_mse_gt_5": split_stats.get("rows_mse_gt_5"),
                "rows_mse_gt_10": split_stats.get("rows_mse_gt_10"),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No split reconstruction metrics found in {metrics_path}")

    train_mean = float(table.loc[table["split"] == "train", "mean_mse"].iloc[0])
    valid_mean = float(table.loc[table["split"] == "validation", "mean_mse"].iloc[0])
    test_mean = float(table.loc[table["split"] == "test", "mean_mse"].iloc[0])
    summary = {
        "latent_dim": latent_dim,
        "autoencoder_dir": str(autoencoder_dir),
        "loss": metrics.get("loss"),
        "best_epoch": metrics.get("best_epoch"),
        "best_validation_loss": metrics.get("best_validation_loss"),
        "observed_v_cell_rate": metrics.get("observed_v_cell_rate"),
        "train_mean_mse": train_mean,
        "validation_mean_mse": valid_mean,
        "test_mean_mse": test_mean,
        "validation_to_train_mean_ratio": valid_mean / train_mean if train_mean else None,
        "test_to_train_mean_ratio": test_mean / train_mean if train_mean else None,
        "test_to_validation_mean_ratio": test_mean / valid_mean if valid_mean else None,
    }
    return table, summary


def reconstruction_error_by_fraud_class(
    autoencoder_dir: Path,
    latent_dim: int,
) -> pd.DataFrame:
    manifest_path = autoencoder_dir / "latent_split_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing latent split manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []
    for split_name in RECONSTRUCTION_SPLITS:
        error_path = autoencoder_dir / RECONSTRUCTION_ERROR_FILENAMES[split_name]
        if not error_path.exists():
            raise FileNotFoundError(f"Missing reconstruction errors: {error_path}")
        errors = pd.read_csv(error_path)["reconstruction_mse"].to_numpy(dtype=float)
        split_manifest = manifest.loc[manifest["split"] == split_name].sort_values(
            "row_position"
        )
        if len(split_manifest) != len(errors):
            raise ValueError(
                f"{split_name} manifest rows {len(split_manifest)} do not match "
                f"reconstruction errors {len(errors)} for {autoencoder_dir}."
            )
        labels = split_manifest[TARGET_COL].astype(int).to_numpy()
        for fraud_label, label_name in ((0, "non_fraud"), (1, "fraud")):
            mask = labels == fraud_label
            split_errors = errors[mask]
            rows.append(
                {
                    "latent_dim": latent_dim,
                    "split": split_name,
                    "class": label_name,
                    "rows": int(mask.sum()),
                    "mean_mse": float(np.mean(split_errors)),
                    "median_mse": float(np.median(split_errors)),
                    "p95_mse": float(np.percentile(split_errors, 95)),
                    "p99_mse": float(np.percentile(split_errors, 99)),
                    "max_mse": float(np.max(split_errors)),
                }
            )
        fraud_mean = float(np.mean(errors[labels == 1]))
        non_fraud_mean = float(np.mean(errors[labels == 0]))
        rows.append(
            {
                "latent_dim": latent_dim,
                "split": split_name,
                "class": "fraud_to_non_fraud_mean_ratio",
                "rows": int(len(errors)),
                "mean_mse": fraud_mean / non_fraud_mean if non_fraud_mean else np.nan,
                "median_mse": np.nan,
                "p95_mse": np.nan,
                "p99_mse": np.nan,
                "max_mse": np.nan,
            }
        )
    return pd.DataFrame(rows)


def baseline_v_to_ae_missing_bridge(
    baseline_importance: pd.DataFrame,
    ae_importance: pd.DataFrame,
    model_name: str,
    top_n: int = 25,
) -> pd.DataFrame:
    top_v = (
        baseline_importance.loc[
            baseline_importance["feature"].map(lambda name: bool(V_FEATURE_REGEX.match(name)))
        ]
        .sort_values("importance_gain", ascending=False)
        .head(top_n)
        .copy()
    )
    ae_map = ae_importance.set_index("feature")
    total_latent_gain = float(
        ae_importance.loc[
            ae_importance["feature"].map(lambda name: bool(LATENT_FEATURE_REGEX.match(name))),
            "importance_gain",
        ].sum()
    )
    rows: list[dict[str, Any]] = []
    for _, row in top_v.iterrows():
        v_feature = row["feature"]
        indicator = f"v_missing_{v_feature}"
        indicator_gain = float(ae_map.loc[indicator, "importance_gain"]) if indicator in ae_map.index else 0.0
        indicator_split = int(ae_map.loc[indicator, "importance_split"]) if indicator in ae_map.index else 0
        rows.append(
            {
                "model_name": model_name,
                "baseline_v_feature": v_feature,
                "baseline_gain": float(row["importance_gain"]),
                "baseline_split": int(row["importance_split"]),
                "ae_missing_indicator": indicator,
                "ae_missing_indicator_gain": indicator_gain,
                "ae_missing_indicator_split": indicator_split,
                "ae_total_latent_gain": total_latent_gain,
            }
        )
    return pd.DataFrame(rows)


def load_proposal_metrics(comparison_path: Path) -> pd.DataFrame:
    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing comparison table: {comparison_path}. "
            "Run build_initial_proposal_comparison.py first."
        )
    return pd.read_csv(comparison_path)


def build_diagnostic_notes(payload: dict[str, Any]) -> str:
    lines = [
        "# Initial Proposal Diagnostics",
        "",
        "Generated from saved P01-P04 artifacts. No model training is performed.",
        "",
        "## Proposal Metrics Context",
        "",
    ]
    for row in payload["proposal_metrics"]:
        lines.append(
            f"- {row['legacy_id']} ({row['model_name']}): "
            f"test PR-AUC={row['test_average_precision']:.6f}, "
            f"features={row['total_features']}"
        )

    lines.extend(
        [
            "",
            "## V Missingness Signal",
            "",
            f"- Train cell missing rate: {payload['missingness']['train_missing_cell_rate']:.4%}",
            f"- Train row fraud rate: {payload['missingness']['train_fraud_rate']:.4%}",
            f"- Lowest missing-count bin fraud rate (train): "
            f"{payload['missingness']['lowest_missing_bin_fraud_rate']:.4%}",
            f"- Highest missing-count bin fraud rate (train): "
            f"{payload['missingness']['highest_missing_bin_fraud_rate']:.4%}",
            "",
            "## Baseline V Gain Share",
            "",
            f"- V-value gain share: {payload['gain_share']['baseline_v_value_gain_share_pct']:.2f}%",
            f"- Non-V gain share: {payload['gain_share']['baseline_non_v_gain_share_pct']:.2f}%",
            "",
            "## AE Feature Groups (P03)",
            "",
            f"- Latent gain share: {payload['gain_share']['ae_ld32_latent_gain_share_pct']:.2f}%",
            f"- Missing-indicator gain share: "
            f"{payload['gain_share']['ae_ld32_missing_indicator_gain_share_pct']:.2f}%",
            f"- Non-V gain share: {payload['gain_share']['ae_ld32_non_v_gain_share_pct']:.2f}%",
            f"- Missing indicators with non-zero gain: "
            f"{payload['gain_share']['ae_ld32_missing_indicators_with_gain']}",
            "",
            "## AE Reconstruction Drift",
            "",
        ]
    )
    for item in payload["reconstruction_drift"]:
        lines.append(
            f"- LD{item['latent_dim']}: train={item['train_mean_mse']:.6f}, "
            f"valid={item['validation_mean_mse']:.6f}, "
            f"test={item['test_mean_mse']:.6f}, "
            f"test/train={item['test_to_train_mean_ratio']:.2f}x"
        )

    lines.extend(
        [
            "",
            "## Interpretation Anchors",
            "",
            "- If V-value gain share is high in baseline but AE latents plus missing masks still lose to baseline, the bottleneck is likely representation replacement rather than downstream LightGBM tuning.",
            "- Large test/train reconstruction drift suggests the AE encodes patterns that do not generalize temporally, even when validation loss looks stable.",
            "- Missingness is a fraud signal, but preserving `v_missing_*` alone does not recover the supervised value of original `V*` entries.",
            "",
        ]
    )
    return "\n".join(lines)


def run_diagnostics(
    initial_proposal_dir: Path,
    output_dir: Path,
    top_n: int,
) -> dict[str, Any]:
    paths = {
        "baseline_default": initial_proposal_dir / "baseline_lgbm_default",
        "ae_ld32_default": initial_proposal_dir / "ae_lgbm_ld32_default",
        "ae_ld128_tuned": initial_proposal_dir / "optuna" / "ae_lgbm_ld128_tuned",
        "autoencoder_ld32": initial_proposal_dir / "autoencoder_robust_ld32",
        "autoencoder_ld128": initial_proposal_dir / "autoencoder_robust_ld128",
        "comparison": initial_proposal_dir / "final_comparison" / "initial_proposal_comparison.csv",
    }
    ensure_dir(output_dir)

    log("Loading labeled data and chronological split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)
    split_frames = {"train": train_df, "validation": valid_df, "test": test_df}

    log("Analyzing V missingness signal.")
    cell_missingness = cell_level_missingness_summary(split_frames, v_columns)
    row_missing_fraud = row_level_missingness_fraud_table(split_frames, v_columns)
    row_deciles_train = row_missing_count_deciles_train(train_df, v_columns)
    baseline_fraud_rate = float(train_df[TARGET_COL].mean())
    per_column_lift = per_column_missing_fraud_lift(train_df, v_columns, baseline_fraud_rate)

    log("Analyzing feature importance and gain share.")
    baseline_importance = load_feature_importance(
        paths["baseline_default"] / "feature_importance.csv"
    )
    ae_ld32_importance = load_feature_importance(
        paths["ae_ld32_default"] / "feature_importance.csv"
    )
    ae_ld128_importance = load_feature_importance(
        paths["ae_ld128_tuned"] / "feature_importance.csv"
    )

    baseline_groups, baseline_group_summary = feature_group_gain_summary(
        baseline_importance,
        "baseline_lgbm_default",
    )
    ae_ld32_groups, ae_ld32_group_summary = feature_group_gain_summary(
        ae_ld32_importance,
        "ae_lgbm_ld32_default",
    )
    ae_ld128_groups, ae_ld128_group_summary = feature_group_gain_summary(
        ae_ld128_importance,
        "ae_lgbm_ld128_tuned",
    )

    top_baseline_v = top_features_by_gain(
        baseline_importance,
        "baseline_lgbm_default",
        feature_group="v_value",
        top_n=top_n,
    )
    top_ae_latent_ld32 = top_features_by_gain(
        ae_ld32_importance,
        "ae_lgbm_ld32_default",
        feature_group="ae_latent",
        top_n=top_n,
    )
    top_missing_indicators_ld32 = top_features_by_gain(
        ae_ld32_importance,
        "ae_lgbm_ld32_default",
        feature_group="v_missing_indicator",
        top_n=top_n,
    )
    bridge_ld32 = baseline_v_to_ae_missing_bridge(
        baseline_importance,
        ae_ld32_importance,
        "ae_lgbm_ld32_default",
        top_n=top_n,
    )

    log("Analyzing Autoencoder reconstruction drift and fraud-class separation.")
    recon_tables: list[pd.DataFrame] = []
    recon_summaries: list[dict[str, Any]] = []
    fraud_class_tables: list[pd.DataFrame] = []
    for latent_dim, autoencoder_dir in ((32, paths["autoencoder_ld32"]), (128, paths["autoencoder_ld128"])):
        recon_table, recon_summary = reconstruction_metrics_table(
            autoencoder_dir,
            latent_dim,
        )
        recon_tables.append(recon_table)
        recon_summaries.append(recon_summary)
        fraud_class_tables.append(
            reconstruction_error_by_fraud_class(autoencoder_dir, latent_dim)
        )
    reconstruction_drift = pd.concat(recon_tables, ignore_index=True)
    reconstruction_by_fraud = pd.concat(fraud_class_tables, ignore_index=True)

    proposal_metrics = load_proposal_metrics(paths["comparison"])

    train_bins = row_missing_fraud.loc[row_missing_fraud["split"] == "train"]
    lowest_bin = train_bins.sort_values("missing_v_count_min").iloc[0]
    highest_bin = train_bins.sort_values("missing_v_count_max").iloc[-1]

    missing_indicators_with_gain = int(
        (
            ae_ld32_importance.loc[
                ae_ld32_importance["feature"].map(
                    lambda name: bool(MISSING_INDICATOR_REGEX.match(name))
                ),
                "importance_gain",
            ]
            > 0
        ).sum()
    )

    payload: dict[str, Any] = {
        "initial_proposal_dir": str(initial_proposal_dir),
        "output_dir": str(output_dir),
        "v_feature_count": len(v_columns),
        "proposal_metrics": proposal_metrics.to_dict(orient="records"),
        "missingness": {
            "train_missing_cell_rate": float(
                cell_missingness.loc[cell_missingness["split"] == "train", "missing_cell_rate"].iloc[0]
            ),
            "train_fraud_rate": baseline_fraud_rate,
            "lowest_missing_bin": str(lowest_bin["missing_count_bin"]),
            "lowest_missing_bin_fraud_rate": float(lowest_bin["fraud_rate"]),
            "highest_missing_bin": str(highest_bin["missing_count_bin"]),
            "highest_missing_bin_fraud_rate": float(highest_bin["fraud_rate"]),
        },
        "gain_share": {
            "baseline_v_value_gain_share_pct": baseline_group_summary["groups"]
            .get("v_value", {})
            .get("gain_share_pct", 0.0),
            "baseline_non_v_gain_share_pct": baseline_group_summary["groups"]
            .get("non_v", {})
            .get("gain_share_pct", 0.0),
            "ae_ld32_latent_gain_share_pct": ae_ld32_group_summary["groups"]
            .get("ae_latent", {})
            .get("gain_share_pct", 0.0),
            "ae_ld32_missing_indicator_gain_share_pct": ae_ld32_group_summary["groups"]
            .get("v_missing_indicator", {})
            .get("gain_share_pct", 0.0),
            "ae_ld32_non_v_gain_share_pct": ae_ld32_group_summary["groups"]
            .get("non_v", {})
            .get("gain_share_pct", 0.0),
            "ae_ld32_missing_indicators_with_gain": missing_indicators_with_gain,
            "ae_ld128_latent_gain_share_pct": ae_ld128_group_summary["groups"]
            .get("ae_latent", {})
            .get("gain_share_pct", 0.0),
            "ae_ld128_missing_indicator_gain_share_pct": ae_ld128_group_summary["groups"]
            .get("v_missing_indicator", {})
            .get("gain_share_pct", 0.0),
        },
        "reconstruction_drift": recon_summaries,
        "feature_group_summaries": {
            "baseline_lgbm_default": baseline_group_summary,
            "ae_lgbm_ld32_default": ae_ld32_group_summary,
            "ae_lgbm_ld128_tuned": ae_ld128_group_summary,
        },
        "artifacts": {
            "cell_missingness_csv": "v_cell_missingness_by_split.csv",
            "row_missing_fraud_csv": "v_row_missing_count_fraud_rate_by_split.csv",
            "row_missing_deciles_train_csv": "v_row_missing_count_deciles_train.csv",
            "per_column_lift_csv": "v_column_missing_fraud_lift_train.csv",
            "baseline_group_gain_csv": "baseline_feature_group_gain_share.csv",
            "ae_ld32_group_gain_csv": "ae_ld32_feature_group_gain_share.csv",
            "ae_ld128_group_gain_csv": "ae_ld128_feature_group_gain_share.csv",
            "top_baseline_v_csv": "top_baseline_v_features_by_gain.csv",
            "top_ae_latent_ld32_csv": "top_ae_latent_features_ld32_by_gain.csv",
            "top_missing_indicators_ld32_csv": "top_v_missing_indicators_ld32_by_gain.csv",
            "baseline_to_ae_missing_bridge_csv": "baseline_top_v_to_ae_missing_indicator_bridge.csv",
            "reconstruction_drift_csv": "ae_reconstruction_drift_by_split.csv",
            "reconstruction_by_fraud_csv": "ae_reconstruction_error_by_fraud_class.csv",
            "proposal_metrics_csv": "proposal_metrics_context.csv",
            "notes_md": "diagnostic_notes.md",
        },
    }

    log("Writing diagnostic tables and summary files.")
    cell_missingness.to_csv(output_dir / "v_cell_missingness_by_split.csv", index=False)
    row_missing_fraud.to_csv(
        output_dir / "v_row_missing_count_fraud_rate_by_split.csv",
        index=False,
    )
    row_deciles_train.to_csv(
        output_dir / "v_row_missing_count_deciles_train.csv",
        index=False,
    )
    per_column_lift.to_csv(output_dir / "v_column_missing_fraud_lift_train.csv", index=False)
    baseline_groups.to_csv(output_dir / "baseline_feature_group_gain_share.csv", index=False)
    ae_ld32_groups.to_csv(output_dir / "ae_ld32_feature_group_gain_share.csv", index=False)
    ae_ld128_groups.to_csv(output_dir / "ae_ld128_feature_group_gain_share.csv", index=False)
    top_baseline_v.to_csv(output_dir / "top_baseline_v_features_by_gain.csv", index=False)
    top_ae_latent_ld32.to_csv(output_dir / "top_ae_latent_features_ld32_by_gain.csv", index=False)
    top_missing_indicators_ld32.to_csv(
        output_dir / "top_v_missing_indicators_ld32_by_gain.csv",
        index=False,
    )
    bridge_ld32.to_csv(
        output_dir / "baseline_top_v_to_ae_missing_indicator_bridge.csv",
        index=False,
    )
    reconstruction_drift.to_csv(
        output_dir / "ae_reconstruction_drift_by_split.csv",
        index=False,
    )
    reconstruction_by_fraud.to_csv(
        output_dir / "ae_reconstruction_error_by_fraud_class.csv",
        index=False,
    )
    proposal_metrics.to_csv(output_dir / "proposal_metrics_context.csv", index=False)
    save_json(payload, output_dir / "diagnostic_summary.json")
    (output_dir / "diagnostic_notes.md").write_text(
        build_diagnostic_notes(payload),
        encoding="utf-8",
    )

    return payload


def print_summary(payload: dict[str, Any], output_dir: Path) -> None:
    print()
    print("Initial Proposal Diagnostics Summary")
    print("====================================")
    print(f"V features analyzed     : {payload['v_feature_count']}")
    print(
        "Train V-cell missing rate: "
        f"{payload['missingness']['train_missing_cell_rate']:.4%}"
    )
    print(
        "Baseline V gain share   : "
        f"{payload['gain_share']['baseline_v_value_gain_share_pct']:.2f}%"
    )
    print(
        "AE LD32 latent gain share: "
        f"{payload['gain_share']['ae_ld32_latent_gain_share_pct']:.2f}%"
    )
    print(
        "AE LD32 missing-mask share: "
        f"{payload['gain_share']['ae_ld32_missing_indicator_gain_share_pct']:.2f}%"
    )
    print(
        "AE LD32 masks with gain : "
        f"{payload['gain_share']['ae_ld32_missing_indicators_with_gain']} / 339"
    )
    for item in payload["reconstruction_drift"]:
        print(
            f"AE LD{item['latent_dim']} test/train MSE: "
            f"{item['test_to_train_mean_ratio']:.2f}x"
        )
    print(f"Outputs saved to        : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate in-depth diagnostics for the initial proposal P01-P04 path."
    )
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=DEFAULT_INITIAL_PROPOSAL_DIR,
        help="Root directory containing the isolated initial proposal rerun artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for diagnostic CSV/JSON/Markdown outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top features to export for ranked importance tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_diagnostics(
        initial_proposal_dir=args.initial_proposal_dir,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    print_summary(payload, args.output_dir)


if __name__ == "__main__":
    main()