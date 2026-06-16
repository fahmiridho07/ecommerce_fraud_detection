"""Sweep top-K V retention for hybrid AE-LightGBM and pick the best validation AP."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from utils import ensure_dir, log, save_json

DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_SWEEP_K = (10, 50, 75, 100, 150)
SWEEP_COLUMNS = [
    "top_k",
    "model_name",
    "validation_average_precision",
    "test_average_precision",
    "test_roc_auc",
    "total_features",
    "output_dir",
    "status",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def output_dir_for_k(root: Path, top_k: int) -> Path:
    return root / f"ae_lgbm_ld32_top{top_k}v_default"


def run_complete(output_dir: Path) -> bool:
    required = (
        output_dir / "metrics_test_selected_threshold.json",
        output_dir / "metrics_validation_selected_threshold.json",
        output_dir / "run_config.json",
    )
    return all(path.exists() for path in required)


def train_top_k(
    top_k: int,
    initial_proposal_dir: Path,
    autoencoder_dir: Path,
    baseline_importance: Path,
    baseline_metrics: Path,
    force: bool,
) -> str:
    output_dir = output_dir_for_k(initial_proposal_dir, top_k)
    if run_complete(output_dir) and not force:
        log(f"Skipping top-{top_k}; complete outputs already exist.")
        return "skipped_existing"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "train_ae_lgbm.py"),
        "--autoencoder-output-dir",
        str(autoencoder_dir),
        "--output-dir",
        str(output_dir),
        "--phase-name",
        f"4_ae_lgbm_ld32_top{top_k}v_representation_sweep",
        "--retain-top-v-features",
        str(top_k),
        "--baseline-importance-path",
        str(baseline_importance),
        "--baseline-metrics-path",
        str(baseline_metrics),
    ]
    log(f"Training hybrid AE-LightGBM with top-{top_k} retained V-features.")
    subprocess.run(command, check=True)
    return "trained"


def sweep_row(top_k: int, initial_proposal_dir: Path, status: str) -> dict[str, object]:
    output_dir = output_dir_for_k(initial_proposal_dir, top_k)
    row = {
        "top_k": top_k,
        "model_name": f"ae_lgbm_ld32_top{top_k}v_default",
        "validation_average_precision": None,
        "test_average_precision": None,
        "test_roc_auc": None,
        "total_features": None,
        "output_dir": str(output_dir),
        "status": status,
    }
    if not run_complete(output_dir):
        row["status"] = "missing_metrics"
        return row

    valid_metrics = load_json(output_dir / "metrics_validation_selected_threshold.json")
    test_metrics = load_json(output_dir / "metrics_test_selected_threshold.json")
    run_config = load_json(output_dir / "run_config.json")
    feature_construction = run_config.get("feature_construction", {})
    row.update(
        {
            "validation_average_precision": float(valid_metrics["average_precision"]),
            "test_average_precision": float(test_metrics["average_precision"]),
            "test_roc_auc": float(test_metrics["roc_auc"]),
            "total_features": feature_construction.get("total_feature_count"),
            "status": status,
        }
    )
    return row


def add_reference_rows(
    table: pd.DataFrame,
    initial_proposal_dir: Path,
) -> pd.DataFrame:
    references = [
        ("P01", initial_proposal_dir / "baseline_lgbm_default"),
        ("P02", initial_proposal_dir / "optuna" / "baseline_lgbm_tuned"),
        ("P03", initial_proposal_dir / "ae_lgbm_ld32_default"),
    ]
    rows: list[dict[str, object]] = []
    for label, output_dir in references:
        if not run_complete(output_dir):
            continue
        valid_metrics = load_json(output_dir / "metrics_validation_selected_threshold.json")
        test_metrics = load_json(output_dir / "metrics_test_selected_threshold.json")
        run_config = load_json(output_dir / "run_config.json")
        feature_construction = run_config.get("feature_construction", {})
        total_features = feature_construction.get("total_feature_count")
        if total_features is None:
            total_features = run_config.get("model_features_count")
        rows.append(
            {
                "top_k": label,
                "model_name": output_dir.name,
                "validation_average_precision": float(valid_metrics["average_precision"]),
                "test_average_precision": float(test_metrics["average_precision"]),
                "test_roc_auc": float(test_metrics["roc_auc"]),
                "total_features": total_features,
                "output_dir": str(output_dir),
                "status": "reference",
            }
        )
    if not rows:
        return table
    return pd.concat([pd.DataFrame(rows), table], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep top-K retained V-features for hybrid AE-LightGBM."
    )
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=DEFAULT_INITIAL_PROPOSAL_DIR,
    )
    parser.add_argument(
        "--autoencoder-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--baseline-importance-path", type=Path, default=None)
    parser.add_argument("--baseline-metrics-path", type=Path, default=None)
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_K),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even when complete outputs already exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    root = args.initial_proposal_dir
    args.autoencoder_dir = args.autoencoder_dir or root / "autoencoder_robust_ld32"
    args.baseline_importance_path = (
        args.baseline_importance_path or root / "baseline_lgbm_default" / "feature_importance.csv"
    )
    args.baseline_metrics_path = (
        args.baseline_metrics_path
        or root / "baseline_lgbm_default" / "metrics_test_selected_threshold.json"
    )
    args.output_dir = args.output_dir or root / "representation_ablation"
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    statuses: list[dict[str, object]] = []
    for top_k in args.top_k_values:
        status = train_top_k(
            top_k=top_k,
            initial_proposal_dir=args.initial_proposal_dir,
            autoencoder_dir=args.autoencoder_dir,
            baseline_importance=args.baseline_importance_path,
            baseline_metrics=args.baseline_metrics_path,
            force=args.force,
        )
        statuses.append(sweep_row(top_k, args.initial_proposal_dir, status))

    table = pd.DataFrame(statuses, columns=SWEEP_COLUMNS)
    table = add_reference_rows(table, args.initial_proposal_dir)
    table.to_csv(args.output_dir / "top_v_retention_sweep.csv", index=False)

    sweep_only = table.loc[table["status"].isin(("trained", "skipped_existing"))]
    if sweep_only.empty:
        raise RuntimeError("No sweep rows with metrics are available.")

    best_row = sweep_only.sort_values(
        "validation_average_precision",
        ascending=False,
    ).iloc[0]
    selection = {
        "selected_top_k": int(best_row["top_k"]),
        "selected_model_name": str(best_row["model_name"]),
        "selected_validation_average_precision": float(
            best_row["validation_average_precision"]
        ),
        "selected_test_average_precision": float(best_row["test_average_precision"]),
        "selected_output_dir": str(best_row["output_dir"]),
        "selection_metric": "validation_average_precision",
    }
    save_json(selection, args.output_dir / "top_v_retention_sweep_selection.json")

    print()
    print("Top-V Retention Sweep")
    print("=====================")
    print(table.to_string(index=False))
    print()
    print(
        "Selected for tuning: "
        f"top_k={selection['selected_top_k']} "
        f"(val AP={selection['selected_validation_average_precision']:.6f}, "
        f"test AP={selection['selected_test_average_precision']:.6f})"
    )
    print(f"Saved sweep table to: {args.output_dir / 'top_v_retention_sweep.csv'}")


if __name__ == "__main__":
    main()