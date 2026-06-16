"""Build extended thesis comparison: P01-P04 plus post-diagnostic AE-05."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_initial_proposal_comparison import (
    COMPARISON_COLUMNS,
    build_initial_proposal_comparison_table,
    comparison_row,
)
from config import PROJECT_ROOT
from utils import ensure_dir, save_json

EXTENDED_COMPARISON_FILENAME = "extended_proposal_comparison.csv"
EXTENDED_MISSING_ARTIFACTS_FILENAME = "extended_proposal_missing_artifacts.json"

AE05_CANDIDATE = {
    "canonical_id": "AE-05",
    "legacy_id": "P05",
    "model_name": "ae_lgbm_ld32_top25v_recon_hybrid",
    "tuned": True,
    "output_dir_key": "ae05_hybrid_recon",
}

INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_AE05_DIR = (
    INITIAL_PROPOSAL_DIR / "ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned"
)
DEFAULT_COMPARISON_OUTPUT_DIR = INITIAL_PROPOSAL_DIR / "final_comparison"


def build_extended_proposal_comparison_table(
    output_dirs: dict[str, Path],
    ae05_dir: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    base_table, base_missing = build_initial_proposal_comparison_table(output_dirs)
    rows = base_table.to_dict(orient="records")
    missing = dict(base_missing)

    row, missing_paths = comparison_row(AE05_CANDIDATE, ae05_dir)
    if row is not None:
        rows.append(row)
    elif missing_paths:
        missing[AE05_CANDIDATE["model_name"]] = missing_paths

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    if not table.empty:
        table = table.sort_values("test_average_precision", ascending=False)
    return table, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build extended comparison table (P01-P04 + AE-05)."
    )
    parser.add_argument(
        "--ae05-dir",
        type=Path,
        default=DEFAULT_AE05_DIR,
        help="AE-05 hybrid + reconstruction-error artifact directory.",
    )
    parser.add_argument(
        "--initial-proposal-dir",
        type=Path,
        default=INITIAL_PROPOSAL_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT_DIR,
    )
    parser.add_argument("--baseline-default-dir", type=Path, default=None)
    parser.add_argument("--baseline-tuned-dir", type=Path, default=None)
    parser.add_argument("--ae-lgbm-default-dir", type=Path, default=None)
    parser.add_argument("--ae-lgbm-ld128-tuned-dir", type=Path, default=None)
    parser.set_defaults(
        baseline_default_dir=None,
        baseline_tuned_dir=None,
        ae_lgbm_default_dir=None,
        ae_lgbm_ld128_tuned_dir=None,
    )
    return parser.parse_args()


def resolve_extended_output_dirs(args: argparse.Namespace) -> dict[str, Path]:
    root = args.initial_proposal_dir
    return {
        "baseline_default": args.baseline_default_dir or (root / "baseline_lgbm_default"),
        "baseline_tuned": args.baseline_tuned_dir or (root / "optuna" / "baseline_lgbm_tuned"),
        "ae_default": args.ae_lgbm_default_dir or (root / "ae_lgbm_ld32_default"),
        "ae_ld128_tuned": args.ae_lgbm_ld128_tuned_dir or (root / "optuna" / "ae_lgbm_ld128_tuned"),
    }


def main() -> pd.DataFrame:
    args = parse_args()
    output_dirs = resolve_extended_output_dirs(args)
    comparison_output_dir = ensure_dir(args.output_dir)
    comparison_file = comparison_output_dir / EXTENDED_COMPARISON_FILENAME
    missing_file = comparison_output_dir / EXTENDED_MISSING_ARTIFACTS_FILENAME

    table, missing = build_extended_proposal_comparison_table(output_dirs, args.ae05_dir)
    table.to_csv(comparison_file, index=False)
    save_json(missing, missing_file)

    print()
    print("Extended Proposal Comparison (P01-P04 + AE-05)")
    print("==============================================")
    if table.empty:
        print("No completed rows available.")
    else:
        print(table.to_string(index=False))
    if missing:
        print()
        print("Missing artifacts:")
        for model_name, paths in missing.items():
            print(f"- {model_name}: {len(paths)} path(s)")
    print(f"\nSaved comparison to: {comparison_file}")
    print(f"Saved missing-artifact log to: {missing_file}")
    return table


if __name__ == "__main__":
    main()