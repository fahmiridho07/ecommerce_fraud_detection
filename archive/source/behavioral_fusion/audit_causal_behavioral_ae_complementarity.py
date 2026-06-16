"""Validation-only complementarity audit for CBA01R and P04 expert scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR,
    ID_COL,
    LATE_FUSION_COMPLEMENTARITY_SUMMARY_FILE,
    TARGET_COL,
)
from utils import ensure_dir, save_json


TOPK_FRACTIONS = (0.01, 0.03, 0.05)
DISAGREEMENT_RANK_GAP = 0.10


def classify_complementarity(spearman: float) -> str:
    abs_corr = abs(spearman)
    if abs_corr >= 0.95:
        return "negligible"
    if abs_corr >= 0.85:
        return "weak"
    if abs_corr >= 0.70:
        return "moderate"
    return "strong"


def score_distribution_summary(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    series = pd.Series(scores)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def topk_overlap_row(
    df: pd.DataFrame,
    fraction: float,
    y_true: np.ndarray,
) -> dict[str, object]:
    n = len(df)
    k = max(1, int(np.ceil(n * fraction)))
    cba_top = set(df.nlargest(k, "cba01r_score")[ID_COL].tolist())
    p04_top = set(df.nlargest(k, "p04_ae_score")[ID_COL].tolist())
    both_top = cba_top & p04_top
    union_top = cba_top | p04_top

    fraud_mask = y_true.astype(bool)
    fraud_ids = set(df.loc[fraud_mask, ID_COL].tolist())
    cba_fraud = len(cba_top & fraud_ids)
    p04_fraud = len(p04_top & fraud_ids)
    both_fraud = len(both_top & fraud_ids)
    union_fraud = len(union_top & fraud_ids)
    only_cba_fraud = len((cba_top - p04_top) & fraud_ids)
    only_p04_fraud = len((p04_top - cba_top) & fraud_ids)

    overlap_coefficient = (
        len(both_top) / min(len(cba_top), len(p04_top)) if k else 0.0
    )
    jaccard = len(both_top) / len(union_top) if union_top else 0.0
    total_fraud = int(y_true.sum())

    return {
        "top_fraction": fraction,
        "reviewed_transactions": k,
        "fraud_captured_cba01r": cba_fraud,
        "fraud_captured_p04": p04_fraud,
        "fraud_captured_both": both_fraud,
        "fraud_captured_only_cba01r": only_cba_fraud,
        "fraud_captured_only_p04": only_p04_fraud,
        "fraud_captured_union": union_fraud,
        "overlap_coefficient": float(overlap_coefficient),
        "jaccard_overlap": float(jaccard),
        "precision_union": float(union_fraud / len(union_top)) if union_top else 0.0,
        "recall_union": float(union_fraud / total_fraud) if total_fraud else 0.0,
    }


def run_complementarity_audit(
    validation_table: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if TARGET_COL not in validation_table.columns:
        raise ValueError("Validation table must include isFraud labels.")

    output_dir = ensure_dir(output_dir or CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR)
    y_true = validation_table[TARGET_COL].astype(int).to_numpy()
    cba_score = validation_table["cba01r_score"].to_numpy(dtype=float)
    p04_score = validation_table["p04_ae_score"].to_numpy(dtype=float)

    pearson = float(pd.Series(cba_score).corr(pd.Series(p04_score), method="pearson"))
    spearman = float(pd.Series(cba_score).corr(pd.Series(p04_score), method="spearman"))

    topk_rows = [
        topk_overlap_row(validation_table, fraction, y_true)
        for fraction in TOPK_FRACTIONS
    ]
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(output_dir / "topk_complementarity.csv", index=False)

    cba_rank = percentile_ranks(cba_score)
    p04_rank = percentile_ranks(p04_score)
    rank_gap = cba_rank - p04_rank

    cba_higher = rank_gap >= DISAGREEMENT_RANK_GAP
    p04_higher = (-rank_gap) >= DISAGREEMENT_RANK_GAP

    top5_k = max(1, int(np.ceil(len(validation_table) * 0.05)))
    cba_top5_ids = set(
        validation_table.nlargest(top5_k, "cba01r_score")[ID_COL].tolist()
    )
    p04_top5_ids = set(
        validation_table.nlargest(top5_k, "p04_ae_score")[ID_COL].tolist()
    )
    fraud_ids = set(validation_table.loc[y_true.astype(bool), ID_COL].tolist())
    p04_top5_only_fraud = len((p04_top5_ids - cba_top5_ids) & fraud_ids)
    cba_top5_only_fraud = len((cba_top5_ids - p04_top5_ids) & fraud_ids)

    complementarity_summary = {
        "split": "validation",
        "row_count": int(len(validation_table)),
        "fraud_prevalence": float(np.mean(y_true)),
        "score_association": {
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "cba01r_distribution": score_distribution_summary(cba_score),
            "p04_distribution": score_distribution_summary(p04_score),
        },
        "complementarity_classification": classify_complementarity(spearman),
        "topk_unique_fraud_capture": {
            str(row["top_fraction"]): {
                "only_cba01r": int(row["fraud_captured_only_cba01r"]),
                "only_p04": int(row["fraud_captured_only_p04"]),
            }
            for row in topk_rows
        },
    }
    save_json(complementarity_summary, output_dir / "complementarity_summary.json")

    disagreement_summary = {
        "disagreement_rank_gap_threshold": DISAGREEMENT_RANK_GAP,
        "cba01r_rank_substantially_higher_count": int(cba_higher.sum()),
        "p04_rank_substantially_higher_count": int(p04_higher.sum()),
        "fraud_in_cba01r_higher_rank_direction": int(
            y_true[cba_higher].sum()
        ),
        "fraud_in_p04_higher_rank_direction": int(
            y_true[p04_higher].sum()
        ),
        "fraud_in_p04_top5_outside_cba01r_top5": p04_top5_only_fraud,
        "fraud_in_cba01r_top5_outside_p04_top5": cba_top5_only_fraud,
        "p04_finds_fraud_missed_by_cba01r_top5": p04_top5_only_fraud > 0,
    }
    save_json(disagreement_summary, output_dir / "disagreement_summary.json")

    thesis_safe = {
        "experiment_id": "LF01",
        "split": "validation",
        "spearman_correlation": spearman,
        "pearson_correlation": pearson,
        "complementarity_classification": complementarity_summary[
            "complementarity_classification"
        ],
        "topk_unique_fraud_capture": complementarity_summary["topk_unique_fraud_capture"],
        "p04_finds_fraud_missed_by_cba01r_top5": disagreement_summary[
            "p04_finds_fraud_missed_by_cba01r_top5"
        ],
        "fraud_in_p04_top5_outside_cba01r_top5": p04_top5_only_fraud,
        "fraud_in_cba01r_top5_outside_p04_top5": cba_top5_only_fraud,
    }
    save_json(thesis_safe, LATE_FUSION_COMPLEMENTARITY_SUMMARY_FILE)

    return {
        "complementarity_summary": complementarity_summary,
        "disagreement_summary": disagreement_summary,
        "thesis_safe_summary": thesis_safe,
        "topk_df": topk_df,
    }


def main() -> dict[str, object]:
    output_dir = CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR
    validation_path = output_dir / "validation_expert_scores.csv"
    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation expert scores not found: {validation_path}. "
            "Run src/run_causal_behavioral_ae_late_fusion.py first."
        )
    validation_table = pd.read_csv(validation_path)
    result = run_complementarity_audit(validation_table, output_dir=output_dir)
    print(json.dumps(result["thesis_safe_summary"], indent=2))
    return result


if __name__ == "__main__":
    main()