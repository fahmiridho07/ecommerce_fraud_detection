"""Generate preprocessing diagnostics for the IEEE-CIS thesis pipeline.

This script performs no model training. It inspects the data transformations
that happen before baseline/AE modeling:
- configured split composition
- feature type counts
- missingness by feature family and split
- train-fitted categorical mapping unknown rates
- numeric train-to-test distribution shift
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import (
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
    PROJECT_ROOT,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from preprocessing import (
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_VALUE,
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from splitting import create_holdout_split
from utils import ensure_dir, log, save_json


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "initial_proposal" / "preprocessing_diagnostics"


def split_summary_frame(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        rows.append(
            {
                "split": split_name,
                "rows": int(len(frame)),
                "fraud_count": int(frame[TARGET_COL].sum()),
                "fraud_rate": float(frame[TARGET_COL].mean()),
                "TransactionDT_min": int(frame[TIME_COL].min()),
                "TransactionDT_max": int(frame[TIME_COL].max()),
                "TransactionID_min": int(frame[ID_COL].min()),
                "TransactionID_max": int(frame[ID_COL].max()),
            }
        )
    return pd.DataFrame(rows)


def feature_type_counts_frame(
    X_train_raw: pd.DataFrame,
    categorical_columns: list[str],
    v_columns: list[str],
) -> pd.DataFrame:
    feature_count = int(X_train_raw.shape[1])
    rows = [
        {"group": "all_model_features", "count": feature_count},
        {"group": "categorical_object_features", "count": len(categorical_columns)},
        {
            "group": "numeric_features",
            "count": feature_count - len(categorical_columns),
        },
        {"group": "V_features", "count": len(v_columns)},
        {"group": "non_V_features", "count": feature_count - len(v_columns)},
    ]
    return pd.DataFrame(rows)


def broad_feature_groups(v_columns: list[str]) -> list[tuple[str, Callable[[str], bool]]]:
    return [
        ("V", lambda column: column in v_columns),
        ("C", lambda column: column.startswith("C")),
        ("D", lambda column: column.startswith("D")),
        ("M", lambda column: column.startswith("M")),
        ("card", lambda column: column.startswith("card")),
        ("addr", lambda column: column.startswith("addr")),
        ("dist", lambda column: column.startswith("dist")),
        ("email", lambda column: column.endswith("emaildomain")),
        ("id", lambda column: column.startswith("id_")),
    ]


def missingness_by_group_frame(
    splits: dict[str, pd.DataFrame],
    v_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = broad_feature_groups(v_columns)
    for split_name, frame in splits.items():
        X = frame.drop(columns=[TARGET_COL, ID_COL])
        for group_name, matcher in groups:
            columns = [column for column in X.columns if matcher(column)]
            if not columns:
                continue
            missing = X[columns].isna()
            rows.append(
                {
                    "split": split_name,
                    "group": group_name,
                    "feature_count": len(columns),
                    "cell_missing_rate": float(missing.to_numpy().mean()),
                    "row_any_missing_rate": float(missing.any(axis=1).mean()),
                    "row_all_missing_rate": float(missing.all(axis=1).mean()),
                }
            )
    return pd.DataFrame(rows)


def feature_missingness_drift_frame(
    raw_features: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    v_columns: list[str],
) -> pd.DataFrame:
    X_train = raw_features["train"]
    rows: list[dict[str, object]] = []
    for column in X_train.columns:
        train_missing = float(raw_features["train"][column].isna().mean())
        valid_missing = float(raw_features["validation"][column].isna().mean())
        test_missing = float(raw_features["test"][column].isna().mean())
        rows.append(
            {
                "feature": column,
                "is_v_feature": column in v_columns,
                "is_categorical": column in categorical_columns,
                "train_missing_rate": train_missing,
                "validation_missing_rate": valid_missing,
                "test_missing_rate": test_missing,
                "test_minus_train_missing_rate": test_missing - train_missing,
                "abs_test_minus_train_missing_rate": abs(test_missing - train_missing),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_test_minus_train_missing_rate",
        ascending=False,
    )


def categorical_unknown_rates_frame(
    raw_features: dict[str, pd.DataFrame],
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    categorical_columns = preprocessing["categorical_columns"]
    mappings = preprocessing["categorical_mappings"]
    rows: list[dict[str, object]] = []
    for split_name, X_raw in raw_features.items():
        X_transformed = apply_baseline_preprocessing(X_raw, preprocessing)
        for column in categorical_columns:
            mapping = mappings[column]
            raw = X_raw[column]
            transformed = X_transformed[column]
            train_categories = set(mapping.keys()) - {MISSING_CATEGORY}
            raw_as_string = raw.astype("string")
            observed = raw_as_string[raw_as_string.notna()]
            unseen_rate_among_observed = (
                float((~observed.isin(train_categories)).mean())
                if len(observed)
                else 0.0
            )
            rows.append(
                {
                    "split": split_name,
                    "feature": column,
                    "train_category_count_in_mapping": len(mapping),
                    "raw_missing_rate": float(raw.isna().mean()),
                    "encoded_missing_category_rate": float(
                        (transformed == mapping[MISSING_CATEGORY]).mean()
                    ),
                    "encoded_unknown_rate": float(
                        (transformed == UNKNOWN_CATEGORY_VALUE).mean()
                    ),
                    "unseen_rate_among_observed": unseen_rate_among_observed,
                    "unique_raw_nonmissing": int(raw.nunique(dropna=True)),
                }
            )
    return pd.DataFrame(rows)


def numeric_distribution_shift_frame(
    raw_features: dict[str, pd.DataFrame],
    categorical_columns: list[str],
    v_columns: list[str],
) -> pd.DataFrame:
    numeric_columns = [
        column for column in raw_features["train"].columns if column not in categorical_columns
    ]
    rows: list[dict[str, object]] = []
    for column in numeric_columns:
        train_values = raw_features["train"][column]
        test_values = raw_features["test"][column]
        if train_values.notna().sum() < 100 or test_values.notna().sum() < 100:
            continue
        train_iqr = float(train_values.quantile(0.75) - train_values.quantile(0.25))
        test_iqr = float(test_values.quantile(0.75) - test_values.quantile(0.25))
        fallback_scale = float(train_values.std(skipna=True) or 1.0)
        denominator = train_iqr if abs(train_iqr) > 1e-12 else fallback_scale
        median_shift = float(
            (float(test_values.median()) - float(train_values.median())) / denominator
        )
        rows.append(
            {
                "feature": column,
                "is_v_feature": column in v_columns,
                "train_median": float(train_values.median()),
                "test_median": float(test_values.median()),
                "median_shift_over_train_iqr": median_shift,
                "abs_median_shift_over_train_iqr": abs(median_shift),
                "train_iqr": train_iqr,
                "test_iqr": test_iqr,
                "test_iqr_over_train_iqr": (
                    float(test_iqr / train_iqr) if abs(train_iqr) > 1e-12 else None
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_median_shift_over_train_iqr",
        ascending=False,
    )


def records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, object]]:
    working = frame.head(limit) if limit is not None else frame
    return working.replace({np.nan: None}).to_dict(orient="records")


def run(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> None:
    output_dir = ensure_dir(output_dir)
    log("Loading merged labeled train data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log(f"Creating {split_strategy} split.")
    train_df, valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=split_strategy,
    )
    splits = {"train": train_df, "validation": valid_df, "test": test_df}

    raw_features: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.Series] = {}
    for split_name, frame in splits.items():
        X, y = split_features_target(frame)
        raw_features[split_name] = X
        targets[split_name] = y

    preprocessing = fit_baseline_preprocessing(raw_features["train"])
    categorical_columns = preprocessing["categorical_columns"]
    v_columns = get_v_feature_columns(train_df)

    log("Computing preprocessing diagnostics.")
    split_summary = split_summary_frame(splits)
    feature_counts = feature_type_counts_frame(
        raw_features["train"],
        categorical_columns,
        v_columns,
    )
    missingness_by_group = missingness_by_group_frame(splits, v_columns)
    missingness_drift = feature_missingness_drift_frame(
        raw_features,
        categorical_columns,
        v_columns,
    )
    categorical_unknown = categorical_unknown_rates_frame(raw_features, preprocessing)
    numeric_shift = numeric_distribution_shift_frame(
        raw_features,
        categorical_columns,
        v_columns,
    )

    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    feature_counts.to_csv(output_dir / "feature_type_counts.csv", index=False)
    missingness_by_group.to_csv(output_dir / "missingness_by_group_split.csv", index=False)
    missingness_drift.to_csv(output_dir / "feature_missingness_drift.csv", index=False)
    categorical_unknown.to_csv(output_dir / "categorical_unknown_rates.csv", index=False)
    numeric_shift.to_csv(output_dir / "numeric_distribution_shift.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "split_strategy": split_strategy,
        "split_summary": records(split_summary),
        "feature_counts": {
            row["group"]: int(row["count"]) for row in records(feature_counts)
        },
        "top_categorical_unknown_validation": records(
            categorical_unknown.loc[categorical_unknown["split"].eq("validation")]
            .sort_values("encoded_unknown_rate", ascending=False),
            10,
        ),
        "top_categorical_unknown_test": records(
            categorical_unknown.loc[categorical_unknown["split"].eq("test")]
            .sort_values("encoded_unknown_rate", ascending=False),
            10,
        ),
        "top_missingness_drift": records(missingness_drift, 15),
        "top_numeric_distribution_shift": records(numeric_shift, 15),
        "artifacts": {
            "split_summary_csv": "split_summary.csv",
            "feature_type_counts_csv": "feature_type_counts.csv",
            "missingness_by_group_csv": "missingness_by_group_split.csv",
            "feature_missingness_drift_csv": "feature_missingness_drift.csv",
            "categorical_unknown_rates_csv": "categorical_unknown_rates.csv",
            "numeric_distribution_shift_csv": "numeric_distribution_shift.csv",
        },
    }
    save_json(summary, output_dir / "preprocessing_diagnostic_summary.json")

    print()
    print("Preprocessing Diagnostics")
    print("=========================")
    print(split_summary.to_string(index=False))
    print()
    print("Feature Counts")
    print(feature_counts.to_string(index=False))
    print(f"\nSaved diagnostics to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate preprocessing diagnostics without model training."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-strategy",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help="Holdout split strategy. Default is the active thesis stratified reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(output_dir=args.output_dir, split_strategy=args.split_strategy)


if __name__ == "__main__":
    main()
