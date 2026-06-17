"""Paired bootstrap comparison for tuned baseline vs AE hybrid reconstruction.

The script rebuilds tuned-baseline scores from the saved model, loads AE hybrid
scores saved by `tune_ae_hybrid_reconstruction_lgbm.py`, and estimates a paired
bootstrap confidence interval for the test average-precision delta.
"""

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
    ID_COL,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
    TARGET_COL,
)
from data_loader import load_labeled_train_data
from preprocessing import apply_baseline_preprocessing, split_features_target
from splitting import create_holdout_split
from utils import ensure_dir, save_json


DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_BASELINE_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "optuna" / "baseline_lgbm_tuned"
DEFAULT_AE_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned"
DEFAULT_OUTPUT_DIR = DEFAULT_AE_DIR / "bootstrap_comparison"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_split_data(split_strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    _train_df, valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=split_strategy,
    )
    return valid_df, test_df


def baseline_scores_for_split(
    split_df: pd.DataFrame,
    baseline_dir: Path,
    split_strategy: str,
) -> pd.DataFrame:
    preprocessing_path = baseline_dir / "preprocessing.pkl"
    model_path = baseline_dir / "final_model.pkl"
    run_config_path = baseline_dir / "run_config.json"
    for path in (preprocessing_path, model_path, run_config_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline artifact: {path}")

    preprocessing = joblib.load(preprocessing_path)
    model = joblib.load(model_path)
    run_config = load_json(run_config_path)
    artifact_split_strategy = run_config.get("split_strategy", "chronological")
    if artifact_split_strategy != split_strategy:
        raise ValueError(
            f"{baseline_dir} was produced with split_strategy={artifact_split_strategy!r}, "
            f"but this comparison requested {split_strategy!r}."
        )
    best_iteration = int(run_config["early_stopping"]["best_iteration"])

    X_raw, y = split_features_target(split_df)
    X = apply_baseline_preprocessing(X_raw, preprocessing)
    score = model.predict_proba(X, num_iteration=best_iteration)[:, 1]
    return pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
            "baseline_tuned_score": score,
        }
    )


def load_ae_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing AE score file: {path}")
    frame = pd.read_csv(path)
    required = {ID_COL, TARGET_COL, "score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame.loc[:, [ID_COL, TARGET_COL, "score"]].rename(
        columns={"score": "ae_hybrid_recon_score"}
    )


def aligned_scores(baseline: pd.DataFrame, ae: pd.DataFrame) -> pd.DataFrame:
    if baseline[ID_COL].tolist() != ae[ID_COL].tolist():
        raise ValueError("Baseline and AE TransactionID order does not match.")
    if baseline[TARGET_COL].tolist() != ae[TARGET_COL].tolist():
        raise ValueError("Baseline and AE labels do not match.")
    return baseline.merge(
        ae[[ID_COL, "ae_hybrid_recon_score"]],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )


def paired_bootstrap_average_precision_delta(
    y_true: np.ndarray,
    baseline_score: np.ndarray,
    ae_score: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n_rows = int(y_true.shape[0])
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_rows, size=n_rows)
        sample_y = y_true[indices]
        if sample_y.min() == sample_y.max():
            continue
        baseline_ap = average_precision_score(sample_y, baseline_score[indices])
        ae_ap = average_precision_score(sample_y, ae_score[indices])
        deltas.append(float(ae_ap - baseline_ap))

    delta_array = np.asarray(deltas, dtype="float64")
    observed_baseline_ap = float(average_precision_score(y_true, baseline_score))
    observed_ae_ap = float(average_precision_score(y_true, ae_score))
    observed_delta = observed_ae_ap - observed_baseline_ap
    return {
        "metric": "average_precision",
        "n_bootstrap_requested": int(n_bootstrap),
        "n_bootstrap_used": int(delta_array.shape[0]),
        "observed_baseline_average_precision": observed_baseline_ap,
        "observed_ae_hybrid_recon_average_precision": observed_ae_ap,
        "observed_delta": float(observed_delta),
        "delta_ci_95_percentile": [
            float(np.percentile(delta_array, 2.5)),
            float(np.percentile(delta_array, 97.5)),
        ],
        "one_sided_p_delta_le_0": float(np.mean(delta_array <= 0.0)),
        "bootstrap_delta_mean": float(delta_array.mean()),
        "bootstrap_delta_std": float(delta_array.std(ddof=1)),
    }


def run(args: argparse.Namespace) -> None:
    output_dir = ensure_dir(args.output_dir)
    valid_df, test_df = load_split_data(args.split_strategy)

    baseline_valid = baseline_scores_for_split(
        valid_df,
        args.baseline_dir,
        args.split_strategy,
    )
    baseline_test = baseline_scores_for_split(
        test_df,
        args.baseline_dir,
        args.split_strategy,
    )
    baseline_valid.to_csv(output_dir / "baseline_tuned_scores_validation.csv", index=False)
    baseline_test.to_csv(output_dir / "baseline_tuned_scores_test.csv", index=False)

    ae_valid = load_ae_scores(args.ae_dir / "scores_validation.csv")
    ae_test = load_ae_scores(args.ae_dir / "scores_test.csv")
    ae_run_config = load_json(args.ae_dir / "run_config.json")
    ae_split_strategy = ae_run_config.get("split_strategy", "chronological")
    if ae_split_strategy != args.split_strategy:
        raise ValueError(
            f"{args.ae_dir} was produced with split_strategy={ae_split_strategy!r}, "
            f"but this comparison requested {args.split_strategy!r}."
        )
    aligned_valid = aligned_scores(baseline_valid, ae_valid)
    aligned_test = aligned_scores(baseline_test, ae_test)
    aligned_valid.to_csv(output_dir / "aligned_scores_validation.csv", index=False)
    aligned_test.to_csv(output_dir / "aligned_scores_test.csv", index=False)

    bootstrap = paired_bootstrap_average_precision_delta(
        y_true=aligned_test[TARGET_COL].to_numpy(dtype=int),
        baseline_score=aligned_test["baseline_tuned_score"].to_numpy(dtype="float64"),
        ae_score=aligned_test["ae_hybrid_recon_score"].to_numpy(dtype="float64"),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    save_json(bootstrap, output_dir / "paired_bootstrap_pr_auc_delta.json")

    summary = pd.DataFrame(
        [
            {
                "comparison": "ae_hybrid_recon_minus_baseline_tuned",
                "metric": bootstrap["metric"],
                "baseline_average_precision": bootstrap[
                    "observed_baseline_average_precision"
                ],
                "ae_average_precision": bootstrap[
                    "observed_ae_hybrid_recon_average_precision"
                ],
                "delta": bootstrap["observed_delta"],
                "ci95_low": bootstrap["delta_ci_95_percentile"][0],
                "ci95_high": bootstrap["delta_ci_95_percentile"][1],
                "one_sided_p_delta_le_0": bootstrap["one_sided_p_delta_le_0"],
            }
        ]
    )
    summary.to_csv(output_dir / "paired_bootstrap_pr_auc_delta.csv", index=False)

    print()
    print("Paired Bootstrap PR-AUC Delta")
    print("==============================")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap test PR-AUC delta for baseline vs AE hybrid recon."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--ae-dir", type=Path, default=DEFAULT_AE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
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
