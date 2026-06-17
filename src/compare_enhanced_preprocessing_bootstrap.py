"""Paired bootstrap comparison for enhanced-preprocessing LightGBM outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import (
    DEFAULT_SPLIT_STRATEGY,
    PROJECT_ROOT,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
)
from data_loader import load_labeled_train_data
from enhanced_preprocessing import apply_enhanced_preprocessing
from preprocessing import split_features_target
from splitting import create_holdout_split
from utils import ensure_dir, save_json, set_seed


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "initial_proposal"
    / "preprocessing_ablation"
    / "bootstrap_preprocessing_comparison"
)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def score_enhanced_baseline(
    output_dir: Path,
    test_df: pd.DataFrame,
    split_strategy: str,
) -> np.ndarray:
    model_path = output_dir / "final_model.pkl"
    preprocessing_path = output_dir / "enhanced_preprocessing.pkl"
    run_config_path = output_dir / "run_config.json"
    for path in (model_path, preprocessing_path, run_config_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    run_config = load_json(run_config_path)
    if run_config.get("model_type") != "baseline":
        raise ValueError(
            "This bootstrap helper currently supports enhanced baseline outputs only. "
            f"Got model_type={run_config.get('model_type')} for {output_dir}."
        )
    artifact_split_strategy = run_config.get("split_strategy", "chronological")
    if artifact_split_strategy != split_strategy:
        raise ValueError(
            f"{output_dir} was produced with split_strategy={artifact_split_strategy!r}, "
            f"but this comparison requested {split_strategy!r}."
        )

    model = joblib.load(model_path)
    preprocessing = joblib.load(preprocessing_path)
    X_raw, _y = split_features_target(test_df)
    X = apply_enhanced_preprocessing(X_raw, preprocessing)
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    return model.predict_proba(X, num_iteration=best_iteration)[:, 1]


def paired_bootstrap_delta(
    y_true: np.ndarray,
    candidate_score: np.ndarray,
    reference_score: np.ndarray,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    n_rows = len(y_true)
    rows: list[dict[str, float | int]] = []
    for bootstrap_id in range(n_bootstrap):
        indices = rng.integers(0, n_rows, size=n_rows)
        sampled_y = y_true[indices]
        if np.unique(sampled_y).size < 2:
            continue
        candidate_ap = average_precision_score(sampled_y, candidate_score[indices])
        reference_ap = average_precision_score(sampled_y, reference_score[indices])
        rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "candidate_average_precision": float(candidate_ap),
                "reference_average_precision": float(reference_ap),
                "delta_average_precision": float(candidate_ap - reference_ap),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    set_seed(args.random_seed)
    output_dir = ensure_dir(args.output_dir)

    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    _train_df, _valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=args.split_strategy,
    )
    _X_test_raw, y_test = split_features_target(test_df)
    y_true = y_test.to_numpy()

    candidate_score = score_enhanced_baseline(
        args.candidate_dir,
        test_df,
        args.split_strategy,
    )
    reference_score = score_enhanced_baseline(
        args.reference_dir,
        test_df,
        args.split_strategy,
    )
    candidate_ap = average_precision_score(y_true, candidate_score)
    reference_ap = average_precision_score(y_true, reference_score)
    delta = candidate_ap - reference_ap

    bootstrap = paired_bootstrap_delta(
        y_true,
        candidate_score,
        reference_score,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
    )
    bootstrap_path = output_dir / "paired_bootstrap_pr_auc_delta.csv"
    bootstrap.to_csv(bootstrap_path, index=False)

    deltas = bootstrap["delta_average_precision"].to_numpy()
    summary = {
        "candidate_dir": str(args.candidate_dir),
        "reference_dir": str(args.reference_dir),
        "split_strategy": args.split_strategy,
        "test_rows": int(len(y_true)),
        "test_fraud_rows": int(y_true.sum()),
        "candidate_average_precision": float(candidate_ap),
        "reference_average_precision": float(reference_ap),
        "delta_average_precision": float(delta),
        "n_bootstrap_requested": int(args.n_bootstrap),
        "n_bootstrap_completed": int(len(bootstrap)),
        "ci_95_lower": float(np.quantile(deltas, 0.025)),
        "ci_95_upper": float(np.quantile(deltas, 0.975)),
        "one_sided_p_delta_le_0": float(np.mean(deltas <= 0.0)),
        "bootstrap_csv": str(bootstrap_path),
    }
    save_json(summary, output_dir / "bootstrap_summary.json")

    print("Paired Bootstrap PR-AUC Delta")
    print("=============================")
    print(f"Candidate AP : {candidate_ap:.6f}")
    print(f"Reference AP : {reference_ap:.6f}")
    print(f"Delta AP     : {delta:+.6f}")
    print(f"95% CI       : [{summary['ci_95_lower']:+.6f}, {summary['ci_95_upper']:+.6f}]")
    print(f"p(delta<=0)  : {summary['one_sided_p_delta_le_0']:.4f}")
    print(f"Saved to     : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two enhanced baseline outputs with paired bootstrap."
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--split-strategy",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help="Holdout split strategy. Default is the active thesis stratified reset.",
    )
    args = parser.parse_args()
    if args.n_bootstrap <= 0:
        raise SystemExit("--n-bootstrap must be positive.")
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
