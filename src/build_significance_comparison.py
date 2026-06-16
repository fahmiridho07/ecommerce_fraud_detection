"""Build final significance comparison across P01-P04 and hybrid AE runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from utils import ensure_dir, save_json

DEFAULT_ROOT = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_OUTPUT = DEFAULT_ROOT / "representation_ablation"

COLUMNS = [
    "tier",
    "model_name",
    "tuned",
    "retained_top_v",
    "validation_average_precision",
    "test_average_precision",
    "delta_test_pr_auc_vs_p01",
    "delta_test_pr_auc_vs_p02",
    "delta_test_pr_auc_vs_p03",
    "total_features",
    "n_trials",
    "output_dir",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def row_from_dir(
    tier: str,
    model_name: str,
    output_dir: Path,
    tuned: bool,
    retained_top_v: int | str,
) -> dict[str, object] | None:
    metrics_path = output_dir / "metrics_test_selected_threshold.json"
    valid_path = output_dir / "metrics_validation_selected_threshold.json"
    run_config_path = output_dir / "run_config.json"
    if not metrics_path.exists() or not valid_path.exists() or not run_config_path.exists():
        return None

    test_metrics = load_json(metrics_path)
    valid_metrics = load_json(valid_path)
    run_config = load_json(run_config_path)
    feature_construction = run_config.get("feature_construction", {})
    total_features = feature_construction.get("total_feature_count")
    if total_features is None:
        total_features = run_config.get("model_features_count")
    optuna = run_config.get("optuna", {})
    if not isinstance(optuna, dict):
        optuna = {}
    n_trials = optuna.get("n_trials_completed", 0 if not tuned else None)
    if n_trials is None and run_config.get("training_mode") == "fixed_params":
        n_trials = 0
    return {
        "tier": tier,
        "model_name": model_name,
        "tuned": tuned,
        "retained_top_v": retained_top_v,
        "validation_average_precision": float(valid_metrics["average_precision"]),
        "test_average_precision": float(test_metrics["average_precision"]),
        "delta_test_pr_auc_vs_p01": None,
        "delta_test_pr_auc_vs_p02": None,
        "delta_test_pr_auc_vs_p03": None,
        "total_features": total_features,
        "n_trials": n_trials,
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build significance comparison table.")
    parser.add_argument("--initial-proposal-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.initial_proposal_dir
    output_dir = ensure_dir(args.output_dir)

    candidates = [
        ("canonical", "P01_baseline_default", root / "baseline_lgbm_default", False, 0),
        ("canonical", "P02_baseline_tuned", root / "optuna" / "baseline_lgbm_tuned", True, 0),
        ("canonical", "P03_ae_full_replacement", root / "ae_lgbm_ld32_default", False, 0),
        ("canonical", "P04_ae_ld128_tuned", root / "optuna" / "ae_lgbm_ld128_tuned", True, 0),
        ("hybrid", "AE_top10v_default", root / "ae_lgbm_ld32_top10v_default", False, 10),
        ("hybrid", "AE_top25v_default", root / "ae_lgbm_ld32_top25v_default", False, 25),
        ("hybrid", "AE_top50v_default", root / "ae_lgbm_ld32_top50v_default", False, 50),
        ("hybrid", "AE_top25v_tuned", root / "optuna" / "ae_lgbm_ld32_top25v_tuned", True, 25),
        (
            "hybrid_reconstruction",
            "AE_top25v_recon_fixed_from_hybrid_tuned",
            root / "ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned",
            True,
            25,
        ),
        (
            "hybrid_reconstruction",
            "AE_top25v_recon_tuned",
            root / "optuna" / "ae_lgbm_ld32_top25v_recon_tuned",
            True,
            25,
        ),
    ]

    rows: list[dict[str, object]] = []
    missing: dict[str, str] = {}
    for tier, model_name, path, tuned, retained in candidates:
        row = row_from_dir(tier, model_name, path, tuned, retained)
        if row is None:
            missing[model_name] = f"missing metrics in {path}"
            continue
        rows.append(row)

    table = pd.DataFrame(rows, columns=COLUMNS)
    if not table.empty:
        p01 = table.loc[table["model_name"] == "P01_baseline_default", "test_average_precision"]
        p02 = table.loc[table["model_name"] == "P02_baseline_tuned", "test_average_precision"]
        p03 = table.loc[table["model_name"] == "P03_ae_full_replacement", "test_average_precision"]
        p01_ap = float(p01.iloc[0]) if not p01.empty else None
        p02_ap = float(p02.iloc[0]) if not p02.empty else None
        p03_ap = float(p03.iloc[0]) if not p03.empty else None
        if p01_ap is not None:
            table["delta_test_pr_auc_vs_p01"] = table["test_average_precision"] - p01_ap
        if p02_ap is not None:
            table["delta_test_pr_auc_vs_p02"] = table["test_average_precision"] - p02_ap
        if p03_ap is not None:
            table["delta_test_pr_auc_vs_p03"] = table["test_average_precision"] - p03_ap

    table = table.sort_values("test_average_precision", ascending=False)
    table.to_csv(output_dir / "significance_comparison.csv", index=False)
    save_json(missing, output_dir / "significance_comparison_missing.json")

    print()
    print("Significance Comparison")
    print("=========================")
    if table.empty:
        print("No rows available.")
    else:
        print(table.to_string(index=False))
    print(f"\nSaved to: {output_dir / 'significance_comparison.csv'}")


if __name__ == "__main__":
    main()
