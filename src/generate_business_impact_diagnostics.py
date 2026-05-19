"""Generate operational business-impact diagnostics for thesis defense.

This script uses existing artifacts only:
- labeled train files are used only to recreate the chronological test split;
- saved score files are used when available;
- saved model artifacts are loaded only by the existing helper fallback when
  score files are missing;
- no model training and no Kaggle competition test files are used.

TransactionAmt is treated as a nominal transaction amount for operational
simulation. It is not assumed to be actual financial loss or monetary saving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import (  # noqa: E402
    FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR,
    ID_COL,
    OUTPUT_DIR,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data  # noqa: E402
from splitting import chronological_split  # noqa: E402


OUTPUT_SUBDIR = OUTPUT_DIR / "final_diagnostics" / "business_impact"
FPR_TARGETS = [0.005, 0.01, 0.02, 0.05]
FOCUS_FPR_TARGET = 0.01
FE_MODEL_NAME = "FE-LGBM tuned"
ENSEMBLE_MODEL_NAME = "FE+AE score ensemble"
AE_MODEL_NAME = "AE-LGBM tuned"

SCORE_COLUMNS = {
    FE_MODEL_NAME: "fe_score",
    ENSEMBLE_MODEL_NAME: "ensemble_score",
}

PROFILE_COLUMNS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
]
MISSING_RATE_COLUMNS = ["DeviceInfo", "addr1", "dist1"]


def log(message: str) -> None:
    print(f"[business-impact] {message}", flush=True)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, indent=2, sort_keys=True)
        file.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value):.{digits}f}"


def format_amount(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except TypeError:
        pass
    return f"{float(value):,.2f}"


def fpr_label(fpr_target: float) -> str:
    return f"{int(round(fpr_target * 1000)):03d}"


def final_weights(warnings: list[str]) -> tuple[float, float, str]:
    run_config_path = FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR / "run_config.json"
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        ensemble = run_config.get("ensemble", {})
        if isinstance(ensemble, dict):
            fe_weight = ensemble.get("selected_fe_lgbm_tuned_weight")
            ae_weight = ensemble.get("selected_ae_lgbm_ld128_tuned_weight")
            if fe_weight is not None and ae_weight is not None:
                return float(fe_weight), float(ae_weight), str(run_config_path)

    warnings.append(
        "Final ensemble weights were not found in run_config.json; using "
        "FE=0.78 and AE=0.22 from the stated final model."
    )
    return 0.78, 0.22, "stated_final_model_defaults"


def standardize_score_frame(
    score_df: pd.DataFrame,
    split_name: str,
    fe_weight: float,
    ae_weight: float,
    source_path: Path,
) -> pd.DataFrame:
    required = [ID_COL, TARGET_COL, "fe_lgbm_tuned_score", "ae_lgbm_ld128_tuned_score"]
    missing = [column for column in required if column not in score_df.columns]
    if missing:
        raise KeyError(f"{source_path} is missing required column(s): {missing}")

    standardized = pd.DataFrame(
        {
            ID_COL: score_df[ID_COL].to_numpy(),
            TARGET_COL: score_df[TARGET_COL].astype(int).to_numpy(),
            "fe_score": pd.to_numeric(
                score_df["fe_lgbm_tuned_score"],
                errors="coerce",
            ).to_numpy(dtype="float64"),
            "ae_score": pd.to_numeric(
                score_df["ae_lgbm_ld128_tuned_score"],
                errors="coerce",
            ).to_numpy(dtype="float64"),
        }
    )
    if "ensemble_score" in score_df.columns:
        standardized["ensemble_score"] = pd.to_numeric(
            score_df["ensemble_score"],
            errors="coerce",
        ).to_numpy(dtype="float64")
    else:
        standardized["ensemble_score"] = (
            fe_weight * standardized["fe_score"] + ae_weight * standardized["ae_score"]
        )

    if standardized[["fe_score", "ae_score", "ensemble_score"]].isna().any().any():
        raise ValueError(f"{source_path} contains missing or non-numeric scores.")

    standardized["split"] = split_name
    standardized["score_source"] = str(source_path)
    return standardized


def load_or_regenerate_existing_scores(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    warnings: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    fe_weight, ae_weight, weight_source = final_weights(warnings)
    score_paths = {
        "validation": FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR / "scores_validation.csv",
        "test": FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR / "scores_test.csv",
    }

    if all(path.exists() for path in score_paths.values()):
        log("Loading saved final validation/test score files.")
        scores = {
            split_name: standardize_score_frame(
                pd.read_csv(path),
                split_name,
                fe_weight,
                ae_weight,
                path,
            )
            for split_name, path in score_paths.items()
        }
    else:
        missing = [str(path) for path in score_paths.values() if not path.exists()]
        warnings.append(
            "Final score file(s) missing; regenerating scores in memory from "
            f"saved artifacts only: {missing}"
        )
        from generate_final_defense_diagnostics import (  # noqa: PLC0415
            load_or_regenerate_scores as helper_load_or_regenerate_scores,
        )

        scores, helper_metadata = helper_load_or_regenerate_scores(
            train_df,
            valid_df,
            test_df,
            warnings,
        )
        helper_metadata["warnings"] = warnings
        return scores, helper_metadata

    metadata = {
        "fe_weight": fe_weight,
        "ae_weight": ae_weight,
        "weight_source": weight_source,
        "validation_score_source": str(scores["validation"]["score_source"].iloc[0]),
        "test_score_source": str(scores["test"]["score_source"].iloc[0]),
        "warnings": warnings,
    }
    return scores, metadata


def load_test_data_and_scores() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    log("Loading labeled data and recreating chronological train/valid/test split.")
    full_df = load_labeled_train_data(sample_size=None)
    train_df, valid_df, test_df = chronological_split(full_df)

    warnings: list[str] = []
    scores, score_metadata = load_or_regenerate_existing_scores(
        train_df,
        valid_df,
        test_df,
        warnings,
    )

    test_scores = scores["test"].copy()
    aligned = align_test_scores(test_df, test_scores)
    return test_df, aligned, score_metadata


def align_test_scores(test_df: pd.DataFrame, test_scores: pd.DataFrame) -> pd.DataFrame:
    required_score_columns = [
        ID_COL,
        TARGET_COL,
        "fe_score",
        "ae_score",
        "ensemble_score",
    ]
    missing_scores = [
        column for column in required_score_columns if column not in test_scores.columns
    ]
    if missing_scores:
        raise KeyError(f"Test score file is missing column(s): {missing_scores}")

    profile_columns = [
        column
        for column in [
            TIME_COL,
            "TransactionAmt",
            *PROFILE_COLUMNS,
            *MISSING_RATE_COLUMNS,
        ]
        if column in test_df.columns
    ]
    required_test_columns = [ID_COL, TARGET_COL, "TransactionAmt"]
    missing_test = [column for column in required_test_columns if column not in test_df]
    if missing_test:
        raise KeyError(f"Chronological test split is missing column(s): {missing_test}")

    if test_df[ID_COL].duplicated().any():
        raise ValueError(f"Test split contains duplicate {ID_COL} values.")
    if test_scores[ID_COL].duplicated().any():
        raise ValueError(f"Test scores contain duplicate {ID_COL} values.")

    left = test_df[[ID_COL, TARGET_COL, *profile_columns]].copy()
    right = test_scores[required_score_columns].copy()
    aligned = left.merge(
        right,
        on=ID_COL,
        how="inner",
        suffixes=("_split", "_score"),
        validate="one_to_one",
    )
    if len(aligned) != len(test_df):
        raise ValueError(
            f"Score alignment failed: matched {len(aligned)} rows, expected "
            f"{len(test_df)} test rows."
        )
    if not np.array_equal(
        aligned[f"{TARGET_COL}_split"].astype(int).to_numpy(),
        aligned[f"{TARGET_COL}_score"].astype(int).to_numpy(),
    ):
        raise ValueError("Test labels differ between split data and score file.")

    aligned = aligned.rename(columns={f"{TARGET_COL}_split": TARGET_COL})
    aligned = aligned.drop(columns=[f"{TARGET_COL}_score"])
    aligned[TARGET_COL] = aligned[TARGET_COL].astype(int)
    aligned["TransactionAmt"] = pd.to_numeric(
        aligned["TransactionAmt"],
        errors="coerce",
    ).fillna(0.0)
    return aligned


def threshold_at_or_below_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    candidates = np.flatnonzero(fpr <= target_fpr)
    if candidates.size == 0:
        return float(np.nextafter(np.nanmax(y_score), np.inf))

    candidate_frame = pd.DataFrame(
        {
            "index": candidates,
            "fpr": fpr[candidates],
            "tpr": tpr[candidates],
            "threshold": thresholds[candidates],
        }
    )
    finite_candidates = candidate_frame[np.isfinite(candidate_frame["threshold"])]
    if not finite_candidates.empty:
        candidate_frame = finite_candidates

    selected = candidate_frame.sort_values(
        ["fpr", "tpr", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    threshold = float(selected["threshold"])
    if np.isinf(threshold):
        return float(np.nextafter(np.nanmax(y_score), np.inf))
    return threshold


def confusion_counts(y_true: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    positive = y_true == 1
    negative = ~positive
    return {
        "tp": int(np.sum(flagged & positive)),
        "fp": int(np.sum(flagged & negative)),
        "fn": int(np.sum(~flagged & positive)),
        "tn": int(np.sum(~flagged & negative)),
    }


def operational_row(
    df: pd.DataFrame,
    model_name: str,
    score_column: str,
    target_fpr: float,
) -> dict[str, Any]:
    y_true = df[TARGET_COL].astype(int).to_numpy()
    y_score = df[score_column].to_numpy(dtype="float64")
    threshold = threshold_at_or_below_fpr(y_true, y_score, target_fpr)
    flagged = y_score >= threshold
    counts = confusion_counts(y_true, flagged)

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    positive_count = tp + fn
    negative_count = fp + tn
    flagged_count = tp + fp
    fraud_mask = df[TARGET_COL].to_numpy() == 1
    amount = df["TransactionAmt"].to_numpy(dtype="float64")

    return {
        "model": model_name,
        "score_column": score_column,
        "fpr_target": target_fpr,
        "selected_threshold": threshold,
        "actual_fpr": float(fp / negative_count) if negative_count else np.nan,
        "tpr_recall": float(tp / positive_count) if positive_count else np.nan,
        "precision": float(tp / flagged_count) if flagged_count else np.nan,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "false_negative_count": fn,
        "true_negative_count": tn,
        "detected_fraud_transaction_amt_sum": float(np.sum(amount[flagged & fraud_mask])),
        "missed_fraud_transaction_amt_sum": float(np.sum(amount[(~flagged) & fraud_mask])),
        "blocked_legitimate_transaction_amt_sum": float(
            np.sum(amount[flagged & (~fraud_mask)])
        ),
        "total_flagged_transaction_amt_sum": float(np.sum(amount[flagged])),
        "flagged_count": int(flagged_count),
        "test_rows": int(len(df)),
        "test_fraud_count": int(positive_count),
        "test_nonfraud_count": int(negative_count),
        "transaction_amt_interpretation": (
            "TransactionAmt is nominal transaction amount, not guaranteed real "
            "financial loss."
        ),
        "analysis_type": "operational simulation, not actual monetary saving",
    }


def run_operational_simulation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Running operational threshold simulation at fixed FPR targets.")
    rows = []
    for fpr_target in FPR_TARGETS:
        for model_name, score_column in SCORE_COLUMNS.items():
            rows.append(operational_row(df, model_name, score_column, fpr_target))

    simulation = pd.DataFrame(rows)
    delta_rows = []
    for fpr_target in FPR_TARGETS:
        fe = simulation[
            (simulation["fpr_target"] == fpr_target)
            & (simulation["model"] == FE_MODEL_NAME)
        ].iloc[0]
        ensemble = simulation[
            (simulation["fpr_target"] == fpr_target)
            & (simulation["model"] == ENSEMBLE_MODEL_NAME)
        ].iloc[0]
        delta_rows.append(
            {
                "fpr_target": fpr_target,
                "ensemble_minus_fe_detected_fraud_transaction_amt_sum": float(
                    ensemble["detected_fraud_transaction_amt_sum"]
                    - fe["detected_fraud_transaction_amt_sum"]
                ),
                "ensemble_minus_fe_true_positive_count": int(
                    ensemble["true_positive_count"] - fe["true_positive_count"]
                ),
                "ensemble_minus_fe_false_positive_count": int(
                    ensemble["false_positive_count"] - fe["false_positive_count"]
                ),
                "ensemble_minus_fe_precision": float(
                    ensemble["precision"] - fe["precision"]
                ),
                "ensemble_minus_fe_recall": float(
                    ensemble["tpr_recall"] - fe["tpr_recall"]
                ),
                "fe_detected_fraud_transaction_amt_sum": float(
                    fe["detected_fraud_transaction_amt_sum"]
                ),
                "ensemble_detected_fraud_transaction_amt_sum": float(
                    ensemble["detected_fraud_transaction_amt_sum"]
                ),
                "fe_true_positive_count": int(fe["true_positive_count"]),
                "ensemble_true_positive_count": int(
                    ensemble["true_positive_count"]
                ),
                "fe_false_positive_count": int(fe["false_positive_count"]),
                "ensemble_false_positive_count": int(
                    ensemble["false_positive_count"]
                ),
                "fe_precision": float(fe["precision"]),
                "ensemble_precision": float(ensemble["precision"]),
                "fe_recall": float(fe["tpr_recall"]),
                "ensemble_recall": float(ensemble["tpr_recall"]),
                "analysis_type": "operational simulation, not actual monetary saving",
            }
        )

    return simulation, pd.DataFrame(delta_rows)


def flags_for_model(
    df: pd.DataFrame,
    simulation: pd.DataFrame,
    model_name: str,
    fpr_target: float,
) -> np.ndarray:
    row = simulation[
        (simulation["model"] == model_name)
        & (simulation["fpr_target"] == fpr_target)
    ].iloc[0]
    return (
        df[str(row["score_column"])].to_numpy(dtype="float64")
        >= float(row["selected_threshold"])
    )


def value_profile(series: pd.Series, top_n: int = 8) -> str | None:
    if series.empty:
        return ""
    values = series.astype("object").where(series.notna(), "MISSING")
    counts = values.value_counts(dropna=False).head(top_n)
    total = float(len(series))
    parts = [
        f"{value}: {int(count)} ({count / total:.1%})"
        for value, count in counts.items()
    ]
    return "; ".join(parts)


def missing_rate(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float(series.isna().mean())


def segment_profile_rows(
    df: pd.DataFrame,
    simulation: pd.DataFrame,
    fpr_target: float,
) -> list[dict[str, Any]]:
    fe_flag = flags_for_model(df, simulation, FE_MODEL_NAME, fpr_target)
    ensemble_flag = flags_for_model(df, simulation, ENSEMBLE_MODEL_NAME, fpr_target)
    fraud = df[TARGET_COL].to_numpy() == 1

    segment_masks = {
        "caught_by_both": fraud & fe_flag & ensemble_flag,
        "caught_by_fe_only": fraud & fe_flag & (~ensemble_flag),
        "caught_by_ensemble_only": fraud & (~fe_flag) & ensemble_flag,
        "missed_by_both": fraud & (~fe_flag) & (~ensemble_flag),
    }

    rows: list[dict[str, Any]] = []
    for segment_name, mask in segment_masks.items():
        segment = df.loc[mask].copy()
        amount = segment["TransactionAmt"]
        row: dict[str, Any] = {
            "fpr_target": fpr_target,
            "segment": segment_name,
            "row_count": int(len(segment)),
            "total_transaction_amt": float(amount.sum()) if len(segment) else 0.0,
            "mean_transaction_amt": float(amount.mean()) if len(segment) else np.nan,
            "median_transaction_amt": float(amount.median()) if len(segment) else np.nan,
            "p90_transaction_amt": (
                float(amount.quantile(0.90)) if len(segment) else np.nan
            ),
        }

        for column in PROFILE_COLUMNS:
            output_column = (
                f"{column}_top_values"
                if "emaildomain" in column
                else f"{column}_distribution"
            )
            row[output_column] = (
                value_profile(segment[column]) if column in segment.columns else None
            )

        for column in MISSING_RATE_COLUMNS:
            row[f"{column}_missing_rate"] = (
                missing_rate(segment[column]) if column in segment.columns else None
            )

        rows.append(row)
    return rows


def run_fraud_segmentation(
    df: pd.DataFrame,
    simulation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log("Profiling fraud segments caught or missed by FE versus ensemble.")
    rows = []
    for fpr_target in FPR_TARGETS:
        rows.extend(segment_profile_rows(df, simulation, fpr_target))

    profiles = pd.DataFrame(rows)
    focus = profiles[profiles["fpr_target"] == FOCUS_FPR_TARGET].copy()
    return profiles, focus


def average_ae_scores_by_segment(
    df: pd.DataFrame,
    simulation: pd.DataFrame,
    fpr_target: float,
) -> dict[str, Any]:
    fe_flag = flags_for_model(df, simulation, FE_MODEL_NAME, fpr_target)
    ensemble_flag = flags_for_model(df, simulation, ENSEMBLE_MODEL_NAME, fpr_target)
    fraud = df[TARGET_COL].to_numpy() == 1
    ae_score = df["ae_score"].to_numpy(dtype="float64")

    masks = {
        "caught_by_both": fraud & fe_flag & ensemble_flag,
        "caught_by_fe_only": fraud & fe_flag & (~ensemble_flag),
        "caught_by_ensemble_only": fraud & (~fe_flag) & ensemble_flag,
        "missed_by_both": fraud & (~fe_flag) & (~ensemble_flag),
    }

    summary: dict[str, Any] = {}
    for segment, mask in masks.items():
        summary[segment] = {
            "row_count": int(np.sum(mask)),
            "average_ae_score": float(np.mean(ae_score[mask]))
            if np.sum(mask)
            else None,
        }

    ensemble_only = summary["caught_by_ensemble_only"]["average_ae_score"]
    missed = summary["missed_by_both"]["average_ae_score"]
    summary["caught_by_ensemble_only_minus_missed_by_both"] = (
        float(ensemble_only - missed)
        if ensemble_only is not None and missed is not None
        else None
    )
    return summary


def run_score_complementarity(
    df: pd.DataFrame,
    simulation: pd.DataFrame,
    score_metadata: dict[str, Any],
) -> dict[str, Any]:
    log("Computing FE/AE score complementarity summary.")
    y_true = df[TARGET_COL].astype(int).to_numpy()
    fe_score = df["fe_score"].to_numpy(dtype="float64")
    ae_score = df["ae_score"].to_numpy(dtype="float64")
    ensemble_score = df["ensemble_score"].to_numpy(dtype="float64")

    segment_ae_scores = {
        fpr_label(fpr_target): average_ae_scores_by_segment(
            df,
            simulation,
            fpr_target,
        )
        for fpr_target in FPR_TARGETS
    }

    return {
        "scope": {
            "split": "chronological labeled test split",
            "kaggle_competition_test_files_used": False,
            "training_performed": False,
        },
        "score_sources": {
            "test_score_source": score_metadata.get("test_score_source"),
            "validation_score_source": score_metadata.get("validation_score_source"),
        },
        "ensemble_weights": {
            "fe_weight": score_metadata.get("fe_weight"),
            "ae_weight": score_metadata.get("ae_weight"),
            "weight_source": score_metadata.get("weight_source"),
        },
        "correlations": {
            "pearson_fe_vs_ae": float(pd.Series(fe_score).corr(pd.Series(ae_score))),
            "spearman_fe_vs_ae": float(
                pd.Series(fe_score).corr(pd.Series(ae_score), method="spearman")
            ),
        },
        "pr_auc": {
            "fe_lgbm_tuned": float(average_precision_score(y_true, fe_score)),
            "ae_lgbm_ld128_tuned": float(average_precision_score(y_true, ae_score)),
            "fe_ae_ensemble": float(
                average_precision_score(y_true, ensemble_score)
            ),
        },
        "average_ae_score_by_fraud_segment": {
            "definition": (
                "Fraud-only caught/missed segments are defined using each model's "
                "test-split threshold at the stated FPR target."
            ),
            "focus_fpr_target": FOCUS_FPR_TARGET,
            "focus_fpr_label": fpr_label(FOCUS_FPR_TARGET),
            "by_fpr_target": segment_ae_scores,
        },
        "warnings": score_metadata.get("warnings", []),
    }


def plot_detected_fraud_amount(simulation: pd.DataFrame, output_dir: Path) -> None:
    pivot = simulation.pivot(
        index="fpr_target",
        columns="model",
        values="detected_fraud_transaction_amt_sum",
    ).loc[FPR_TARGETS]
    x = np.arange(len(pivot.index))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, pivot[FE_MODEL_NAME], width, label=FE_MODEL_NAME)
    ax.bar(
        x + width / 2,
        pivot[ENSEMBLE_MODEL_NAME],
        width,
        label=ENSEMBLE_MODEL_NAME,
    )
    ax.set_title("Detected Fraud TransactionAmt by FPR Target")
    ax.set_xlabel("FPR Target")
    ax.set_ylabel("Detected Fraud TransactionAmt Sum")
    ax.set_xticks(x)
    ax.set_xticklabels([format_float(value, 3) for value in pivot.index])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "detected_fraud_amount_by_fpr.png", dpi=170)
    plt.close(fig)


def plot_tp_fp_counts(simulation: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
    metrics = [
        ("true_positive_count", "True Positives"),
        ("false_positive_count", "False Positives"),
    ]
    x = np.arange(len(FPR_TARGETS))
    width = 0.36

    for ax, (column, title) in zip(axes, metrics):
        pivot = simulation.pivot(
            index="fpr_target",
            columns="model",
            values=column,
        ).loc[FPR_TARGETS]
        ax.bar(x - width / 2, pivot[FE_MODEL_NAME], width, label=FE_MODEL_NAME)
        ax.bar(
            x + width / 2,
            pivot[ENSEMBLE_MODEL_NAME],
            width,
            label=ENSEMBLE_MODEL_NAME,
        )
        ax.set_title(title)
        ax.set_xlabel("FPR Target")
        ax.set_xticks(x)
        ax.set_xticklabels([format_float(value, 3) for value in FPR_TARGETS])
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Count")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tp_fp_count_by_fpr.png", dpi=170)
    plt.close(fig)


def plot_segment_amount_distribution(
    df: pd.DataFrame,
    simulation: pd.DataFrame,
    output_dir: Path,
) -> None:
    fe_flag = flags_for_model(df, simulation, FE_MODEL_NAME, FOCUS_FPR_TARGET)
    ensemble_flag = flags_for_model(
        df,
        simulation,
        ENSEMBLE_MODEL_NAME,
        FOCUS_FPR_TARGET,
    )
    fraud = df[TARGET_COL].to_numpy() == 1
    amount = df["TransactionAmt"].to_numpy(dtype="float64")

    segment_masks = {
        "caught_by_both": fraud & fe_flag & ensemble_flag,
        "caught_by_fe_only": fraud & fe_flag & (~ensemble_flag),
        "caught_by_ensemble_only": fraud & (~fe_flag) & ensemble_flag,
        "missed_by_both": fraud & (~fe_flag) & (~ensemble_flag),
    }
    positive_amounts = amount[fraud & (amount > 0)]
    if positive_amounts.size == 0:
        return

    lower = max(float(np.percentile(positive_amounts, 1)), 0.01)
    upper = max(float(np.percentile(positive_amounts, 99)), lower * 2.0)
    bins = np.logspace(np.log10(lower), np.log10(upper), 35)

    fig, ax = plt.subplots(figsize=(8, 5))
    for segment, mask in segment_masks.items():
        values = amount[mask & (amount > 0)]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=f"{segment} (n={values.size})",
        )
    ax.set_xscale("log")
    ax.set_title("Fraud Segment TransactionAmt Distribution at FPR 0.01")
    ax.set_xlabel("TransactionAmt")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        output_dir / "fraud_segment_transaction_amount_distribution_fpr_001.png",
        dpi=170,
    )
    plt.close(fig)


def make_plots(df: pd.DataFrame, simulation: pd.DataFrame, output_dir: Path) -> None:
    log("Saving optional plots.")
    plot_detected_fraud_amount(simulation, output_dir)
    plot_tp_fp_counts(simulation, output_dir)
    plot_segment_amount_distribution(df, simulation, output_dir)


def row_for(
    table: pd.DataFrame,
    model_name: str,
    fpr_target: float,
) -> pd.Series:
    return table[
        (table["model"] == model_name) & (table["fpr_target"] == fpr_target)
    ].iloc[0]


def profile_row_for(profiles: pd.DataFrame, segment: str) -> pd.Series:
    return profiles[profiles["segment"] == segment].iloc[0]


def build_notes(
    simulation: pd.DataFrame,
    delta: pd.DataFrame,
    focus_profiles: pd.DataFrame,
    complementarity: dict[str, Any],
    output_dir: Path,
) -> str:
    fe_focus = row_for(simulation, FE_MODEL_NAME, FOCUS_FPR_TARGET)
    ensemble_focus = row_for(simulation, ENSEMBLE_MODEL_NAME, FOCUS_FPR_TARGET)
    delta_focus = delta[delta["fpr_target"] == FOCUS_FPR_TARGET].iloc[0]
    ensemble_only = profile_row_for(focus_profiles, "caught_by_ensemble_only")
    fe_only = profile_row_for(focus_profiles, "caught_by_fe_only")
    missed_both = profile_row_for(focus_profiles, "missed_by_both")
    score_summary = complementarity["pr_auc"]
    correlations = complementarity["correlations"]
    ae_segment_scores = complementarity["average_ae_score_by_fraud_segment"][
        "by_fpr_target"
    ][fpr_label(FOCUS_FPR_TARGET)]

    lines = [
        "# Business-Impact Diagnostics",
        "",
        "## Scope",
        "",
        "- Existing artifacts only; no model training is performed.",
        "- The analysis uses only the labeled chronological test split.",
        "- Kaggle competition test files are not used.",
        "- TransactionAmt is treated as nominal transaction amount, not guaranteed real financial loss.",
        "- This is an operational threshold simulation, not an actual monetary-saving estimate.",
        "",
        "## Model Context",
        "",
        (
            f"- FE-LGBM tuned test PR-AUC: "
            f"{format_float(score_summary['fe_lgbm_tuned'])}."
        ),
        (
            f"- AE-LGBM tuned test PR-AUC: "
            f"{format_float(score_summary['ae_lgbm_ld128_tuned'])}."
        ),
        (
            f"- FE+AE score ensemble test PR-AUC: "
            f"{format_float(score_summary['fe_ae_ensemble'])} "
            f"(FE weight={format_float(complementarity['ensemble_weights']['fe_weight'], 2)}, "
            f"AE weight={format_float(complementarity['ensemble_weights']['ae_weight'], 2)})."
        ),
        "",
        "## Operational Simulation at FPR 0.01",
        "",
        (
            f"- FE threshold={format_float(fe_focus['selected_threshold'])}, "
            f"actual FPR={format_float(fe_focus['actual_fpr'])}, "
            f"precision={format_float(fe_focus['precision'])}, "
            f"recall={format_float(fe_focus['tpr_recall'])}."
        ),
        (
            f"- Ensemble threshold={format_float(ensemble_focus['selected_threshold'])}, "
            f"actual FPR={format_float(ensemble_focus['actual_fpr'])}, "
            f"precision={format_float(ensemble_focus['precision'])}, "
            f"recall={format_float(ensemble_focus['tpr_recall'])}."
        ),
        (
            "- Ensemble minus FE at FPR 0.01: "
            f"detected fraud TransactionAmt {format_amount(delta_focus['ensemble_minus_fe_detected_fraud_transaction_amt_sum'])}, "
            f"TP {int(delta_focus['ensemble_minus_fe_true_positive_count']):+d}, "
            f"FP {int(delta_focus['ensemble_minus_fe_false_positive_count']):+d}, "
            f"precision {float(delta_focus['ensemble_minus_fe_precision']):+.6f}, "
            f"recall {float(delta_focus['ensemble_minus_fe_recall']):+.6f}."
        ),
        "",
        "## Saved-by-Ensemble Segmentation at FPR 0.01",
        "",
        (
            f"- Fraud caught by ensemble only: n={int(ensemble_only['row_count'])}, "
            f"TransactionAmt total={format_amount(ensemble_only['total_transaction_amt'])}, "
            f"median={format_amount(ensemble_only['median_transaction_amt'])}, "
            f"p90={format_amount(ensemble_only['p90_transaction_amt'])}."
        ),
        (
            f"- Fraud caught by FE only: n={int(fe_only['row_count'])}, "
            f"TransactionAmt total={format_amount(fe_only['total_transaction_amt'])}, "
            f"median={format_amount(fe_only['median_transaction_amt'])}, "
            f"p90={format_amount(fe_only['p90_transaction_amt'])}."
        ),
        (
            f"- Fraud missed by both: n={int(missed_both['row_count'])}, "
            f"TransactionAmt total={format_amount(missed_both['total_transaction_amt'])}."
        ),
        (
            "- Average AE score among fraud caught by ensemble only: "
            f"{format_float(ae_segment_scores['caught_by_ensemble_only']['average_ae_score'])}; "
            "among fraud missed by both: "
            f"{format_float(ae_segment_scores['missed_by_both']['average_ae_score'])}."
        ),
        "",
        "## Complementarity",
        "",
        (
            f"- FE and AE score correlation on test: Pearson="
            f"{format_float(correlations['pearson_fe_vs_ae'])}, "
            f"Spearman={format_float(correlations['spearman_fe_vs_ae'])}."
        ),
        (
            "- Interpretation: the ensemble gain is small in PR-AUC, so it should be "
            "defended as a targeted ranking improvement with operational trade-offs, "
            "not as evidence that the AE model is better standalone."
        ),
        "",
        "## Files",
        "",
        f"- operational_fpr_simulation.csv: {output_dir / 'operational_fpr_simulation.csv'}",
        f"- operational_fpr_delta_summary.csv: {output_dir / 'operational_fpr_delta_summary.csv'}",
        f"- fraud_segment_profiles_fpr_001.csv: {output_dir / 'fraud_segment_profiles_fpr_001.csv'}",
        f"- score_complementarity_summary.json: {output_dir / 'score_complementarity_summary.json'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    test_df, aligned, score_metadata = load_test_data_and_scores()
    _ = test_df

    simulation, delta = run_operational_simulation(aligned)
    all_profiles, focus_profiles = run_fraud_segmentation(aligned, simulation)
    complementarity = run_score_complementarity(aligned, simulation, score_metadata)
    make_plots(aligned, simulation, OUTPUT_SUBDIR)

    simulation.to_csv(OUTPUT_SUBDIR / "operational_fpr_simulation.csv", index=False)
    delta.to_csv(OUTPUT_SUBDIR / "operational_fpr_delta_summary.csv", index=False)
    focus_profiles.to_csv(
        OUTPUT_SUBDIR / "fraud_segment_profiles_fpr_001.csv",
        index=False,
    )
    all_profiles.to_csv(
        OUTPUT_SUBDIR / "fraud_segment_profiles_all_fpr.csv",
        index=False,
    )
    save_json(
        complementarity,
        OUTPUT_SUBDIR / "score_complementarity_summary.json",
    )
    notes = build_notes(
        simulation,
        delta,
        focus_profiles,
        complementarity,
        OUTPUT_SUBDIR,
    )
    (OUTPUT_SUBDIR / "business_impact_notes.md").write_text(
        notes,
        encoding="utf-8",
    )

    log(f"Business-impact diagnostics saved to {OUTPUT_SUBDIR}")


if __name__ == "__main__":
    main()
