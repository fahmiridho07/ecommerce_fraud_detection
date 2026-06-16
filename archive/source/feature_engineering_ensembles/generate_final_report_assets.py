"""Generate final thesis report assets from completed experiment outputs.

This script is intentionally read-only for previous experiment folders. It
collects available metrics from known output directories, uses fixed final
values as fallbacks, and writes report-ready tables under outputs/final_report.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR, PROJECT_ROOT
from utils import ensure_dir, save_json


FINAL_REPORT_DIR = OUTPUT_DIR / "final_report"
FE_TUNED_OUTPUT_DIR = OUTPUT_DIR / "optuna" / "baseline_lgbm_entity_time_amount_features"

COMPARISON_COLUMNS = [
    "descriptive_test_rank",
    "model_id",
    "model_name",
    "model_family",
    "variant",
    "is_tuned",
    "is_standalone",
    "is_baseline",
    "validation_pr_auc",
    "test_pr_auc",
    "test_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "selected_threshold",
    "best_iteration",
    "total_features",
    "metrics_source",
    "output_dir",
]

METRIC_MAP = {
    "test_pr_auc": "average_precision",
    "test_roc_auc": "roc_auc",
    "test_precision": "precision",
    "test_recall": "recall",
    "test_f1": "f1",
    "test_mcc": "mcc",
    "selected_threshold": "threshold",
}

FINAL_MODELS = [
    {
        "model_id": "fe_ae_tuned_score_ensemble",
        "model_name": "FE-LGBM tuned + AE-LGBM tuned score ensemble",
        "model_family": "score_ensemble",
        "variant": "feature_engineered_plus_autoencoder",
        "is_tuned": True,
        "is_standalone": False,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR
        / "fe_ae_controlled_experiments"
        / "A_score_ensemble_fe_tuned_ae_tuned",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {
            "test_pr_auc": 0.533935,
            "test_roc_auc": 0.904215,
            "test_f1": 0.520402,
            "test_mcc": 0.523285,
        },
    },
    {
        "model_id": "fe_lgbm_tuned",
        "model_name": "Feature-engineered LightGBM tuned",
        "model_family": "feature_engineered_lgbm",
        "variant": "entity_time_amount_features",
        "is_tuned": True,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": FE_TUNED_OUTPUT_DIR,
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {
            "test_pr_auc": 0.529857,
            "test_roc_auc": 0.894602,
            "test_f1": 0.522828,
            "test_mcc": 0.520060,
        },
    },
    {
        "model_id": "fe_lgbm_default",
        "model_name": "Feature-engineered LightGBM default",
        "model_family": "feature_engineered_lgbm",
        "variant": "entity_time_amount_features",
        "is_tuned": False,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR / "baseline_lgbm_entity_time_amount_features",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.509117},
    },
    {
        "model_id": "baseline_lgbm_tuned",
        "model_name": "Baseline LightGBM tuned",
        "model_family": "baseline_lgbm",
        "variant": "original_features",
        "is_tuned": True,
        "is_standalone": True,
        "is_baseline": True,
        "output_dir": OUTPUT_DIR / "optuna" / "baseline_lgbm",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.501438},
    },
    {
        "model_id": "ae_reconstruction_error",
        "model_name": "AE reconstruction error only",
        "model_family": "ae_reconstruction_error",
        "variant": "robust_ld128_reconstruction_mse",
        "is_tuned": False,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR / "baseline_lgbm_plus_log1p_ae_reconstruction_mse",
        "alternate_output_dirs": [
            OUTPUT_DIR / "baseline_lgbm_plus_ae_reconstruction_mse",
        ],
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.496067},
    },
    {
        "model_id": "ae_lgbm_ld32_default",
        "model_name": "AE-LightGBM LD32 replacement default",
        "model_family": "ae_lgbm",
        "variant": "latent_dim_32_replacement",
        "is_tuned": False,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR / "ae_lgbm",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {
            "validation_pr_auc": 0.591398,
            "test_pr_auc": 0.481593,
        },
    },
    {
        "model_id": "ae_lgbm_ld128_tuned",
        "model_name": "AE-LightGBM LD128 tuned",
        "model_family": "ae_lgbm",
        "variant": "latent_dim_128",
        "is_tuned": True,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR / "optuna" / "ae_lgbm_ld128",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.490686},
    },
    {
        "model_id": "ae_lgbm_ld128_default",
        "model_name": "AE-LightGBM LD128 default",
        "model_family": "ae_lgbm",
        "variant": "latent_dim_128",
        "is_tuned": False,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR / "ae_lgbm_ld128",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.489417},
    },
    {
        "model_id": "normal_only_ae_reconstruction_error",
        "model_name": "Normal-only AE reconstruction error",
        "model_family": "ae_reconstruction_error",
        "variant": "normal_only_ld128_reconstruction_mse",
        "is_tuned": False,
        "is_standalone": True,
        "is_baseline": False,
        "output_dir": OUTPUT_DIR / "baseline_lgbm_plus_normal_only_ae_reconstruction_mse",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.487441},
    },
    {
        "model_id": "baseline_lgbm_default",
        "model_name": "Baseline LightGBM default",
        "model_family": "baseline_lgbm",
        "variant": "original_features",
        "is_tuned": False,
        "is_standalone": True,
        "is_baseline": True,
        "output_dir": OUTPUT_DIR / "baseline_lgbm",
        "test_metrics_files": ["metrics_test_selected_threshold.json"],
        "validation_metrics_files": ["metrics_validation_selected_threshold.json"],
        "fallback": {"test_pr_auc": 0.485756},
    },
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def candidate_dirs(spec: dict[str, Any]) -> list[Path]:
    dirs = [Path(spec["output_dir"])]
    dirs.extend(Path(path) for path in spec.get("alternate_output_dirs", []))
    return dirs


def find_existing_file(dirs: list[Path], file_names: list[str]) -> Path | None:
    for directory in dirs:
        for file_name in file_names:
            path = directory / file_name
            if path.exists():
                return path
    return None


def value_from_metrics(
    metrics: dict[str, Any] | None,
    output_key: str,
    fallback: dict[str, Any],
) -> Any:
    if metrics is not None:
        metric_key = METRIC_MAP.get(output_key)
        if metric_key in metrics:
            return metrics[metric_key]
    return fallback.get(output_key)


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def run_config_value(run_config: dict[str, Any] | None, key: str) -> Any:
    if not run_config:
        return None

    if key == "best_iteration":
        return (
            nested_get(run_config, ["early_stopping", "best_iteration"])
            or nested_get(run_config, ["optuna", "best_trial_validation_best_iteration"])
        )

    if key == "total_features":
        return (
            run_config.get("model_features_count")
            or nested_get(run_config, ["feature_construction", "model_features_count"])
            or nested_get(run_config, ["feature_construction", "total_feature_count"])
            or nested_get(run_config, ["feature_construction", "total_final_features"])
        )

    return None


def build_model_row(spec: dict[str, Any]) -> dict[str, Any]:
    dirs = candidate_dirs(spec)
    test_metrics_path = find_existing_file(dirs, spec["test_metrics_files"])
    validation_metrics_path = find_existing_file(dirs, spec["validation_metrics_files"])
    run_config_path = find_existing_file(dirs, ["run_config.json"])

    test_metrics = load_json(test_metrics_path) if test_metrics_path else None
    validation_metrics = (
        load_json(validation_metrics_path) if validation_metrics_path else None
    )
    run_config = load_json(run_config_path) if run_config_path else None
    fallback = spec.get("fallback", {})

    row = {
        "descriptive_test_rank": None,
        "model_id": spec["model_id"],
        "model_name": spec["model_name"],
        "model_family": spec["model_family"],
        "variant": spec["variant"],
        "is_tuned": spec["is_tuned"],
        "is_standalone": spec["is_standalone"],
        "is_baseline": spec["is_baseline"],
        "validation_pr_auc": (
            validation_metrics.get("average_precision")
            if validation_metrics is not None
            else fallback.get("validation_pr_auc")
        ),
        "test_pr_auc": value_from_metrics(test_metrics, "test_pr_auc", fallback),
        "test_roc_auc": value_from_metrics(test_metrics, "test_roc_auc", fallback),
        "test_precision": value_from_metrics(test_metrics, "test_precision", fallback),
        "test_recall": value_from_metrics(test_metrics, "test_recall", fallback),
        "test_f1": value_from_metrics(test_metrics, "test_f1", fallback),
        "test_mcc": value_from_metrics(test_metrics, "test_mcc", fallback),
        "selected_threshold": value_from_metrics(
            test_metrics,
            "selected_threshold",
            fallback,
        ),
        "best_iteration": run_config_value(run_config, "best_iteration"),
        "total_features": run_config_value(run_config, "total_features"),
        "metrics_source": (
            str(test_metrics_path.relative_to(PROJECT_ROOT))
            if test_metrics_path
            else "fixed_final_value_fallback"
        ),
        "output_dir": str(Path(spec["output_dir"]).relative_to(PROJECT_ROOT)),
        "selected_fe_weight": nested_get(
            run_config or {},
            ["ensemble", "selected_fe_lgbm_tuned_weight"],
        ),
        "selected_ae_weight": nested_get(
            run_config or {},
            ["ensemble", "selected_ae_lgbm_ld128_tuned_weight"],
        ),
    }
    return row


def sort_by_descriptive_test_rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order rows by observed test AP for descriptive reporting only."""
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["test_pr_auc"] is not None,
            float(row["test_pr_auc"] or 0.0),
        ),
        reverse=True,
    )
    for rank, row in enumerate(sorted_rows, start=1):
        row["descriptive_test_rank"] = rank
    return sorted_rows


THESIS_PRIMARY_MODEL_IDS = (
    "baseline_lgbm_default",
    "baseline_lgbm_tuned",
    "ae_lgbm_ld32_default",
    "ae_lgbm_ld128_tuned",
)


def thesis_primary_validation_leader(
    rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the primary-model row with the highest validation AP."""
    primary_rows = [
        row
        for row in rows
        if row["model_id"] in THESIS_PRIMARY_MODEL_IDS
        and row.get("validation_pr_auc") is not None
    ]
    if not primary_rows:
        return None
    return max(primary_rows, key=lambda row: float(row["validation_pr_auc"]))


def write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_model_comparison_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        ("descriptive_test_rank", "Descriptive test rank"),
        ("model_name", "Model"),
        ("validation_pr_auc", "Validation PR-AUC"),
        ("test_pr_auc", "Test PR-AUC"),
        ("test_roc_auc", "Test ROC-AUC"),
        ("test_f1", "F1"),
        ("test_mcc", "MCC"),
        ("is_standalone", "Standalone"),
    ]
    lines = [
        "# Final Model Comparison",
        "",
        "Descriptive test ranks are ordered by observed test PR-AUC on the "
        "chronological test split. This table is **not** a model-selection rule.",
        "Thesis-primary model choice must use validation AP on the frozen primary "
        "comparison defined in `docs/EXPERIMENT_SCOPE_FREEZE.md`.",
        "",
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, bool):
                value = "yes" if value else "no"
            values.append(markdown_value(value))
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_summary_model(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "descriptive_test_rank": row.get("descriptive_test_rank"),
        "model_id": row.get("model_id"),
        "model_name": row.get("model_name"),
        "validation_pr_auc": row.get("validation_pr_auc"),
        "test_pr_auc": row.get("test_pr_auc"),
        "test_roc_auc": row.get("test_roc_auc"),
        "test_f1": row.get("test_f1"),
        "test_mcc": row.get("test_mcc"),
        "metrics_source": row.get("metrics_source"),
    }


def find_row(rows: list[dict[str, Any]], model_id: str) -> dict[str, Any] | None:
    return next((row for row in rows if row["model_id"] == model_id), None)


def metric_delta(
    rows: list[dict[str, Any]],
    left_id: str,
    right_id: str,
    metric: str = "test_pr_auc",
) -> float | None:
    left = find_row(rows, left_id)
    right = find_row(rows, right_id)
    if left is None or right is None:
        return None
    if left.get(metric) is None or right.get(metric) is None:
        return None
    return float(left[metric]) - float(right[metric])


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    highest_observed_test_ap = rows[0]
    standalone_rows = [row for row in rows if row["is_standalone"]]
    baseline_rows = [row for row in rows if row["is_baseline"]]
    highest_observed_test_ap_standalone = standalone_rows[0]
    highest_observed_test_ap_baseline = baseline_rows[0]
    validation_leader = thesis_primary_validation_leader(rows)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ranking_purpose": (
            "descriptive_test_ranking_not_model_selection"
        ),
        "ranking_metric": "test_pr_auc",
        "evaluation_split": "chronological test split",
        "governance_note": (
            "Highest observed test AP is descriptive only. Thesis-primary model "
            "selection must use validation AP on the frozen primary comparison."
        ),
        "highest_observed_test_ap_model": as_summary_model(
            highest_observed_test_ap
        ),
        "highest_observed_test_ap_standalone_model": as_summary_model(
            highest_observed_test_ap_standalone
        ),
        "highest_observed_test_ap_baseline_model": as_summary_model(
            highest_observed_test_ap_baseline
        ),
        "thesis_primary_validation_leader": (
            as_summary_model(validation_leader)
            if validation_leader is not None
            else None
        ),
        # Deprecated aliases retained for backward compatibility when reading
        # old summaries. Do not use these fields for model selection.
        "best_overall_model": as_summary_model(highest_observed_test_ap),
        "best_standalone_model": as_summary_model(
            highest_observed_test_ap_standalone
        ),
        "best_baseline_model": as_summary_model(
            highest_observed_test_ap_baseline
        ),
        "contributions": {
            "ae_contribution": {
                "ensemble_vs_fe_tuned_test_pr_auc_delta": metric_delta(
                    rows,
                    "fe_ae_tuned_score_ensemble",
                    "fe_lgbm_tuned",
                ),
                "selected_fe_weight": find_row(
                    rows,
                    "fe_ae_tuned_score_ensemble",
                ).get("selected_fe_weight"),
                "selected_ae_weight": find_row(
                    rows,
                    "fe_ae_tuned_score_ensemble",
                ).get("selected_ae_weight"),
                "ae_lgbm_tuned_vs_baseline_tuned_test_pr_auc_delta": metric_delta(
                    rows,
                    "ae_lgbm_ld128_tuned",
                    "baseline_lgbm_tuned",
                ),
                "interpretation": (
                    "AE standalone did not outperform tuned LightGBM, but the "
                    "AE-LGBM score contributed complementary signal in the "
                    "FE-LGBM + AE-LGBM ensemble."
                ),
            },
            "feature_engineering_contribution": {
                "fe_default_vs_baseline_default_test_pr_auc_delta": metric_delta(
                    rows,
                    "fe_lgbm_default",
                    "baseline_lgbm_default",
                ),
                "fe_tuned_vs_baseline_tuned_test_pr_auc_delta": metric_delta(
                    rows,
                    "fe_lgbm_tuned",
                    "baseline_lgbm_tuned",
                ),
                "interpretation": (
                    "Entity, time, and amount feature engineering produced the "
                    "largest standalone improvement."
                ),
            },
        },
        "assets": {
            "model_comparison_csv": "outputs/final_report/final_model_comparison.csv",
            "model_comparison_markdown": (
                "outputs/final_report/final_model_comparison.md"
            ),
            "interpretation_notes": "outputs/final_report/interpretation_notes.md",
            "top_30_feature_importance_fe_tuned": (
                "outputs/final_report/top_30_feature_importance_fe_tuned.csv"
            ),
            "engineered_feature_importance_fe_tuned": (
                "outputs/final_report/engineered_feature_importance_fe_tuned.csv"
            ),
        },
    }


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "not available"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.6f}"


def write_interpretation_notes(summary: dict[str, Any], path: Path) -> None:
    contributions = summary["contributions"]
    ae_delta = contributions["ae_contribution"][
        "ensemble_vs_fe_tuned_test_pr_auc_delta"
    ]
    ae_standalone_delta = contributions["ae_contribution"][
        "ae_lgbm_tuned_vs_baseline_tuned_test_pr_auc_delta"
    ]
    fe_default_delta = contributions["feature_engineering_contribution"][
        "fe_default_vs_baseline_default_test_pr_auc_delta"
    ]
    fe_tuned_delta = contributions["feature_engineering_contribution"][
        "fe_tuned_vs_baseline_tuned_test_pr_auc_delta"
    ]

    validation_leader = summary.get("thesis_primary_validation_leader")
    highest_test = summary["highest_observed_test_ap_model"]

    lines = [
        "# Interpretation Notes",
        "",
        "## Governance",
        "",
        "- Descriptive test ranks are not a model-selection rule.",
        "- Thesis-primary model choice must use validation AP on the frozen "
        "primary comparison in `docs/EXPERIMENT_SCOPE_FREEZE.md`.",
        "",
        "## Thesis-ready findings",
        "",
        "- Highest observed test AP (descriptive only): "
        f"{highest_test['model_name']} "
        f"(test PR-AUC {highest_test['test_pr_auc']:.6f}).",
    ]
    if validation_leader is not None:
        lines.extend(
            [
                "- Thesis-primary validation leader among frozen primary models: "
                f"{validation_leader['model_name']} "
                f"(validation PR-AUC {validation_leader['validation_pr_auc']:.6f}).",
            ]
        )
    lines.extend(
        [
        "- Highest observed test AP standalone model: "
        f"{summary['highest_observed_test_ap_standalone_model']['model_name']} "
        f"(test PR-AUC "
        f"{summary['highest_observed_test_ap_standalone_model']['test_pr_auc']:.6f}).",
        "- Highest observed test AP baseline model: "
        f"{summary['highest_observed_test_ap_baseline_model']['model_name']} "
        f"(test PR-AUC "
        f"{summary['highest_observed_test_ap_baseline_model']['test_pr_auc']:.6f}).",
        ]
    )
    lines.extend(
        [
            "- AE standalone did not outperform tuned LightGBM "
            f"(delta vs tuned baseline: {fmt_delta(ae_standalone_delta)} test PR-AUC).",
            "- AE latent replacement or augmentation appears less effective because "
            "IEEE-CIS V-features are already highly engineered.",
            "- AE still provides complementary probabilistic signal when ensembled "
            f"with FE-LGBM (delta vs FE-LGBM tuned: {fmt_delta(ae_delta)} test PR-AUC).",
            "- Entity, time, and amount feature engineering produced the largest "
            "standalone improvement "
            f"(default delta: {fmt_delta(fe_default_delta)}, "
            f"tuned delta: {fmt_delta(fe_tuned_delta)} test PR-AUC).",
            "- The chronological split makes the evaluation harder and more realistic "
            "than a random split because later transactions are held out for testing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric_sort_value(row: dict[str, str], column: str) -> float:
    try:
        return float(row.get(column, "") or 0.0)
    except ValueError:
        return 0.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_dict_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fe_tuned_engineered_features() -> set[str]:
    run_config = load_json(FE_TUNED_OUTPUT_DIR / "run_config.json")
    if not run_config:
        return set()
    engineered_features = nested_get(
        run_config,
        ["feature_construction", "engineered_features"],
    )
    if not isinstance(engineered_features, list):
        return set()
    return {str(feature) for feature in engineered_features}


def write_feature_importance_assets() -> dict[str, Any]:
    source_path = FE_TUNED_OUTPUT_DIR / "feature_importance.csv"
    result = {
        "feature_importance_source": (
            str(source_path.relative_to(PROJECT_ROOT))
            if source_path.exists()
            else None
        ),
        "top_30_written": False,
        "engineered_written": False,
    }
    if not source_path.exists():
        return result

    rows = read_csv_rows(source_path)
    if not rows:
        return result

    fieldnames = list(rows[0].keys())
    rows = sorted(
        rows,
        key=lambda row: numeric_sort_value(row, "importance_gain"),
        reverse=True,
    )

    top_30_path = FINAL_REPORT_DIR / "top_30_feature_importance_fe_tuned.csv"
    write_dict_rows(top_30_path, rows[:30], fieldnames)
    result["top_30_written"] = True

    engineered_features = fe_tuned_engineered_features()
    if engineered_features:
        engineered_rows = [
            row for row in rows if row.get("feature") in engineered_features
        ]
        if engineered_rows:
            engineered_path = (
                FINAL_REPORT_DIR / "engineered_feature_importance_fe_tuned.csv"
            )
            write_dict_rows(engineered_path, engineered_rows, fieldnames)
            result["engineered_written"] = True
            result["engineered_feature_count"] = len(engineered_rows)

    return result


def main() -> None:
    ensure_dir(FINAL_REPORT_DIR)

    rows = sort_by_descriptive_test_rank(
        [build_model_row(spec) for spec in FINAL_MODELS]
    )
    write_csv(
        rows,
        FINAL_REPORT_DIR / "final_model_comparison.csv",
        COMPARISON_COLUMNS,
    )
    write_model_comparison_markdown(
        rows,
        FINAL_REPORT_DIR / "final_model_comparison.md",
    )

    summary = build_summary(rows)
    feature_importance_summary = write_feature_importance_assets()
    summary["feature_importance_assets"] = feature_importance_summary
    save_json(summary, FINAL_REPORT_DIR / "final_summary.json")
    write_interpretation_notes(
        summary,
        FINAL_REPORT_DIR / "interpretation_notes.md",
    )

    print("Final report assets generated:")
    print(f"- {FINAL_REPORT_DIR / 'final_model_comparison.csv'}")
    print(f"- {FINAL_REPORT_DIR / 'final_model_comparison.md'}")
    print(f"- {FINAL_REPORT_DIR / 'final_summary.json'}")
    print(f"- {FINAL_REPORT_DIR / 'interpretation_notes.md'}")
    print(f"- {FINAL_REPORT_DIR / 'top_30_feature_importance_fe_tuned.csv'}")
    if feature_importance_summary.get("engineered_written"):
        print(f"- {FINAL_REPORT_DIR / 'engineered_feature_importance_fe_tuned.csv'}")


if __name__ == "__main__":
    main()
