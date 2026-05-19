"""Utility helpers for the thesis experiment report notebook.

The notebook is intended for supervisor-facing reporting, so implementation
details live here to keep the notebook readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "outputs").exists() and (candidate / "notebooks").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MISSING_NOTES: list[str] = []

ARTIFACTS = {
    "train_transaction": "data/raw/train_transaction.csv",
    "train_identity": "data/raw/train_identity.csv",
    "split_summary": "outputs/split_summary.json",
    "final_summary": "outputs/final_report/final_summary.json",
    "final_model_comparison": "outputs/final_report/final_model_comparison.csv",
    "diagnostic_notes": "outputs/final_report/diagnostic_notes.md",
    "final_defense_notes": "outputs/final_diagnostics/final_defense_notes.md",
    "feature_importance": "outputs/optuna/baseline_lgbm_entity_time_amount_features/feature_importance.csv",
    "feature_set_summary": "outputs/baseline_lgbm_entity_time_amount_features/feature_set_summary.json",
    "next_controlled": "outputs/final_comparison/next_controlled_experiments.csv",
    "fe_ae_fine_ensemble": "outputs/final_comparison/fe_ae_fine_ensemble_comparison.csv",
    "latent_ablation": "outputs/final_comparison/latent_dim_ablation.csv",
    "ae_augmented": "outputs/final_comparison/ae_augmented_comparison.csv",
    "three_model_ensemble": "outputs/final_comparison/three_model_ensemble_comparison.csv",
    "scores_test": "outputs/fe_ae_controlled_experiments/A_score_ensemble_fe_tuned_ae_tuned/scores_test.csv",
    "bootstrap_ci": "outputs/final_diagnostics/bootstrap_pr_auc_ci.csv",
    "bootstrap_delta_summary": "outputs/final_diagnostics/bootstrap_delta_summary.json",
    "temporal_degradation": "outputs/final_diagnostics/temporal_degradation_by_bin.csv",
    "score_correlation": "outputs/final_diagnostics/score_correlation_summary.json",
    "reconstruction_summary": "outputs/final_diagnostics/reconstruction_error_class_summary.csv",
    "inference_complexity": "outputs/final_diagnostics/inference_complexity_summary.csv",
    "operational_fpr": "outputs/final_diagnostics/business_impact/operational_fpr_simulation.csv",
    "operational_delta": "outputs/final_diagnostics/business_impact/operational_fpr_delta_summary.csv",
    "business_notes": "outputs/final_diagnostics/business_impact/business_impact_notes.md",
}

RAW_TRANSACTION_COLS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "dist1",
]
RAW_IDENTITY_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]

HIGHLIGHT_FEATURES = [
    "amt_mean_by_card1_addr1",
    "amt_std_by_card1_addr1",
    "count_card1_addr1_P_emaildomain",
    "count_card1_P_emaildomain",
    "amt_mean_by_card1_P_emaildomain",
]


def configure_notebook_display() -> None:
    pd.set_option("display.max_columns", 90)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda value: f"{value:.6f}")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def resolve_path(path: str | Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path
    return PROJECT_ROOT / artifact_path


def register_note(message: str) -> None:
    MISSING_NOTES.append(message)
    print(message)


def safe_read_csv(path: str | Path, context: str | None = None, **kwargs) -> pd.DataFrame | None:
    artifact_path = resolve_path(path)
    label = context or str(path)
    if not artifact_path.exists():
        register_note(f"Catatan: artifact untuk {label} tidak ditemukan: {artifact_path}")
        return None
    try:
        return pd.read_csv(artifact_path, **kwargs)
    except Exception as exc:  # pragma: no cover - report notebook guardrail
        register_note(f"Catatan: artifact untuk {label} gagal dibaca: {exc}")
        return None


def safe_read_json(path: str | Path, context: str | None = None) -> dict | None:
    artifact_path = resolve_path(path)
    label = context or str(path)
    if not artifact_path.exists():
        register_note(f"Catatan: artifact untuk {label} tidak ditemukan: {artifact_path}")
        return None
    try:
        with artifact_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:  # pragma: no cover - report notebook guardrail
        register_note(f"Catatan: artifact untuk {label} gagal dibaca: {exc}")
        return None


def safe_read_text(path: str | Path, context: str | None = None) -> str | None:
    artifact_path = resolve_path(path)
    label = context or str(path)
    if not artifact_path.exists():
        register_note(f"Catatan: artifact untuk {label} tidak ditemukan: {artifact_path}")
        return None
    try:
        return artifact_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - report notebook guardrail
        register_note(f"Catatan: artifact untuk {label} gagal dibaca: {exc}")
        return None


def note_table(message: str) -> pd.DataFrame:
    return pd.DataFrame({"Catatan": [message]})


def numeric_round_table(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    table = df.copy()
    numeric_columns = table.select_dtypes(include=["number"]).columns
    table[numeric_columns] = table[numeric_columns].round(digits)
    return table


def prepare_comparison(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    data = df.copy()
    if "model_name" not in data.columns and "model_id" in data.columns:
        data["model_name"] = data["model_id"]
    if "model_name" not in data.columns:
        data["model_name"] = data.index.astype(str)

    metric_cols = [c for c in data.columns if c.startswith(("validation_", "test_"))]
    metric_cols += ["selected_threshold", "total_features", "best_iteration"]
    for column in sorted(set(metric_cols)):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "rank" not in data.columns and "test_pr_auc" in data.columns:
        data = data.sort_values("test_pr_auc", ascending=False).reset_index(drop=True)
        data.insert(0, "rank", np.arange(1, len(data) + 1))
    return data


def format_metric_table(df: pd.DataFrame | None) -> pd.DataFrame:
    data = prepare_comparison(df)
    if data.empty:
        return note_table("Data perbandingan model tidak tersedia.")
    desired = [
        "rank",
        "model_name",
        "validation_pr_auc",
        "test_pr_auc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_mcc",
        "selected_threshold",
        "total_features",
        "is_standalone",
    ]
    available = [column for column in desired if column in data.columns]
    table = data[available].copy()
    table = table.rename(
        columns={
            "rank": "Rank",
            "model_name": "Model",
            "validation_pr_auc": "Validation PR-AUC",
            "test_pr_auc": "Test PR-AUC",
            "test_roc_auc": "Test ROC-AUC",
            "test_precision": "Precision",
            "test_recall": "Recall",
            "test_f1": "F1",
            "test_mcc": "MCC",
            "selected_threshold": "Threshold",
            "total_features": "Jumlah Fitur",
            "is_standalone": "Standalone",
        }
    )
    return numeric_round_table(table)


def plot_bar(
    df: pd.DataFrame | None,
    metric_col: str,
    label_col: str = "model_name",
    title: str = "",
    xlabel: str | None = None,
    top_n: int = 12,
    color: str = "#4c78a8",
) -> None:
    data = prepare_comparison(df)
    if data.empty or metric_col not in data.columns or label_col not in data.columns:
        print(f"Catatan: data untuk plot {metric_col} tidak tersedia.")
        return
    plot_data = data[[label_col, metric_col]].copy()
    plot_data[metric_col] = pd.to_numeric(plot_data[metric_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[metric_col]).sort_values(metric_col, ascending=False).head(top_n).iloc[::-1]
    if plot_data.empty:
        print(f"Catatan: data numerik untuk plot {metric_col} tidak tersedia.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_data))))
    ax.barh(plot_data[label_col], plot_data[metric_col], color=color)
    ax.set_title(title or metric_col)
    ax.set_xlabel(xlabel or metric_col)
    ax.set_ylabel("")
    upper = min(1.0, max(float(plot_data[metric_col].max()) * 1.16, 0.05))
    ax.set_xlim(left=0, right=upper)
    for bar in ax.patches:
        width = bar.get_width()
        ax.text(width + upper * 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_grouped_metric_bars(
    df: pd.DataFrame | None,
    metric_cols: list[str],
    label_col: str = "model_name",
    top_n: int = 9,
    title: str = "Perbandingan Metrik Test",
) -> None:
    data = prepare_comparison(df)
    available = [column for column in metric_cols if column in data.columns]
    if data.empty or not available or label_col not in data.columns:
        print("Catatan: data untuk grouped bar chart tidak tersedia.")
        return
    sort_col = "test_pr_auc" if "test_pr_auc" in data.columns else available[0]
    plot_data = data[[label_col, sort_col] + available].copy()
    plot_data = plot_data.dropna(subset=available, how="all").sort_values(sort_col, ascending=False).head(top_n).iloc[::-1]

    labels = plot_data[label_col].tolist()
    y = np.arange(len(labels))
    bar_height = 0.8 / len(available)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
    pretty = {"test_roc_auc": "ROC-AUC", "test_f1": "F1", "test_mcc": "MCC"}
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(labels))))
    for index, metric in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * bar_height
        ax.barh(
            y + offset,
            plot_data[metric].astype(float),
            height=bar_height,
            label=pretty.get(metric, metric),
            color=colors[index % len(colors)],
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Skor")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def category_fraud_summary(df: pd.DataFrame | None, column: str, top_n: int = 10) -> pd.DataFrame:
    if df is None or df.empty or column not in df.columns:
        return pd.DataFrame()
    temp = df[[column, "isFraud"]].copy()
    temp[column] = temp[column].astype("string").fillna("MISSING")
    grouped = temp.groupby(column, dropna=False)["isFraud"].agg(["count", "sum"]).reset_index()
    grouped = grouped.rename(columns={"sum": "fraud_count"})
    grouped["fraud_rate"] = grouped["fraud_count"] / grouped["count"]
    grouped = grouped.sort_values("count", ascending=False).head(top_n)
    return grouped.sort_values("fraud_rate", ascending=True)


def plot_category_fraud_rate(df: pd.DataFrame | None, column: str, top_n: int = 10) -> None:
    summary = category_fraud_summary(df, column, top_n=top_n)
    if summary.empty:
        print(f"Catatan: kolom {column} tidak tersedia untuk fraud-rate plot.")
        return
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(summary))))
    ax.barh(summary[column].astype(str), summary["fraud_rate"] * 100, color="#4c78a8")
    ax.set_title(f"Fraud Rate per {column} (top {top_n} by volume)")
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def markdown_bullets_to_frame(text: str | None) -> pd.DataFrame:
    if not text:
        return note_table("Catatan markdown tidak tersedia.")
    rows = []
    section = "Umum"
    for line in text.splitlines():
        if line.startswith("## "):
            section = line.replace("## ", "", 1).strip()
        elif line.startswith("- "):
            rows.append({"Area": section, "Temuan": line[2:].strip()})
    if not rows:
        return note_table("File markdown tersedia, tetapi tidak ada bullet ringkasan.")
    return pd.DataFrame(rows)
