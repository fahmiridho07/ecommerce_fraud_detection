"""Build comparison table for representation ablation runs (P03 vs hybrid top-V)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT
from utils import ensure_dir, save_json

DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_OUTPUT_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "representation_ablation"

COMPARISON_COLUMNS = [
    "model_name",
    "representation_mode",
    "test_average_precision",
    "validation_average_precision",
    "test_roc_auc",
    "test_f1",
    "test_mcc",
    "total_features",
    "retained_v_features",
    "delta_test_pr_auc_vs_p03",
    "delta_test_pr_auc_vs_p01",
    "output_dir",
    "metrics_path",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def metric_row(
    model_name: str,
    output_dir: Path,
    representation_mode: str,
    retained_v_features: int,
    baseline_p01_ap: float | None,
    baseline_p03_ap: float | None,
) -> dict[str, object] | None:
    metrics_path = output_dir / "metrics_test_selected_threshold.json"
    run_config_path = output_dir / "run_config.json"
    if not metrics_path.exists() or not run_config_path.exists():
        return None

    metrics = load_json(metrics_path)
    run_config = load_json(run_config_path)
    feature_construction = run_config.get("feature_construction", {})
    total_features = feature_construction.get("total_feature_count")
    test_ap = float(metrics["average_precision"])
    return {
        "model_name": model_name,
        "representation_mode": representation_mode,
        "test_average_precision": test_ap,
        "validation_average_precision": load_json(
            output_dir / "metrics_validation_selected_threshold.json"
        )["average_precision"],
        "test_roc_auc": metrics.get("roc_auc"),
        "test_f1": metrics.get("f1"),
        "test_mcc": metrics.get("mcc"),
        "total_features": total_features,
        "retained_v_features": retained_v_features,
        "delta_test_pr_auc_vs_p03": (
            test_ap - baseline_p03_ap if baseline_p03_ap is not None else None
        ),
        "delta_test_pr_auc_vs_p01": (
            test_ap - baseline_p01_ap if baseline_p01_ap is not None else None
        ),
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
    }


def build_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = [
        (
            "baseline_lgbm_default",
            args.baseline_default_dir,
            "full_original_features",
            0,
        ),
        (
            "ae_lgbm_ld32_default",
            args.ae_lgbm_default_dir,
            "full_latent_replacement",
            0,
        ),
        (
            "ae_lgbm_ld32_top25v_default",
            args.ae_lgbm_top25v_dir,
            "hybrid_latent_plus_top_v_retention",
            25,
        ),
    ]
    rows: list[dict[str, object]] = []
    missing: dict[str, str] = {}
    for model_name, output_dir, mode, retained in candidates:
        if not output_dir.exists():
            missing[model_name] = f"missing directory: {output_dir}"
            continue
        row = metric_row(model_name, output_dir, mode, retained, None, None)
        if row is None:
            missing[model_name] = f"missing metrics in {output_dir}"
            continue
        rows.append(row)

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if not table.empty:
        p01_ap = table.loc[
            table["model_name"] == "baseline_lgbm_default",
            "test_average_precision",
        ]
        p03_ap = table.loc[
            table["model_name"] == "ae_lgbm_ld32_default",
            "test_average_precision",
        ]
        p01_value = float(p01_ap.iloc[0]) if not p01_ap.empty else None
        p03_value = float(p03_ap.iloc[0]) if not p03_ap.empty else None
        if p01_value is not None:
            table["delta_test_pr_auc_vs_p01"] = (
                table["test_average_precision"] - p01_value
            )
        if p03_value is not None:
            table["delta_test_pr_auc_vs_p03"] = (
                table["test_average_precision"] - p03_value
            )
    return table, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full latent replacement against hybrid top-V retention."
    )
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=DEFAULT_INITIAL_PROPOSAL_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--baseline-default-dir", type=Path, default=None)
    parser.add_argument("--ae-lgbm-default-dir", type=Path, default=None)
    parser.add_argument("--ae-lgbm-top25v-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.initial_proposal_dir
    args.baseline_default_dir = args.baseline_default_dir or root / "baseline_lgbm_default"
    args.ae_lgbm_default_dir = args.ae_lgbm_default_dir or root / "ae_lgbm_ld32_default"
    args.ae_lgbm_top25v_dir = (
        args.ae_lgbm_top25v_dir or root / "ae_lgbm_ld32_top25v_default"
    )
    return args


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    table, missing = build_table(args)
    table.to_csv(output_dir / "representation_ablation_comparison.csv", index=False)
    save_json(missing, output_dir / "representation_ablation_missing_artifacts.json")

    print()
    print("Representation Ablation Comparison")
    print("==================================")
    if table.empty:
        print("No comparison rows are available yet.")
    else:
        print(table.to_string(index=False))
    if missing:
        print()
        print("Missing artifacts:")
        for model_name, reason in missing.items():
            print(f"- {model_name}: {reason}")
    print(f"\nSaved comparison to: {output_dir / 'representation_ablation_comparison.csv'}")


if __name__ == "__main__":
    main()