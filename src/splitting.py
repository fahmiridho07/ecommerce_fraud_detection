"""Chronological splitting utilities."""

from __future__ import annotations

import pandas as pd

from config import (
    ID_COL,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import validate_labeled_data, validate_required_columns


def sort_by_transaction_time(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
) -> pd.DataFrame:
    """Sort labeled data by TransactionDT ascending."""
    validate_required_columns(df, [time_col], "labeled data")
    return df.sort_values(time_col, ascending=True).reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
    time_col: str = TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by time order into train, validation, and test sets."""
    validate_labeled_data(df)
    if round(train_ratio + valid_ratio + test_ratio, 10) != 1.0:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    df_sorted = sort_by_transaction_time(df, time_col=time_col)
    n_rows = len(df_sorted)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    valid_df = df_sorted.iloc[train_end:valid_end].copy()
    test_df = df_sorted.iloc[valid_end:].copy()
    validate_split_integrity(df_sorted, train_df, valid_df, test_df, time_col=time_col)
    return train_df, valid_df, test_df


def validate_split_integrity(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = TIME_COL,
) -> None:
    """Validate chronological order and TransactionID separation across splits."""
    validate_labeled_data(full_df)
    for name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        validate_required_columns(split_df, [ID_COL, TARGET_COL, time_col], name)
        if split_df.empty:
            raise ValueError(f"{name} split is empty. Increase sample_size or use full data.")
        if split_df[ID_COL].duplicated().any():
            duplicate_count = int(split_df[ID_COL].duplicated().sum())
            raise ValueError(f"Duplicate {ID_COL} values found in {name}: {duplicate_count}")

    if len(train_df) + len(valid_df) + len(test_df) != len(full_df):
        raise ValueError("Split row counts do not add up to the full dataset row count.")

    train_ids = set(train_df[ID_COL])
    valid_ids = set(valid_df[ID_COL])
    test_ids = set(test_df[ID_COL])

    if train_ids & valid_ids:
        raise ValueError(f"{ID_COL} overlap found between train and validation splits.")
    if train_ids & test_ids:
        raise ValueError(f"{ID_COL} overlap found between train and test splits.")
    if valid_ids & test_ids:
        raise ValueError(f"{ID_COL} overlap found between validation and test splits.")

    if train_df[time_col].max() > valid_df[time_col].min():
        raise ValueError(f"Temporal order violated: train max {time_col} > validation min {time_col}.")
    if valid_df[time_col].max() > test_df[time_col].min():
        raise ValueError(f"Temporal order violated: validation max {time_col} > test min {time_col}.")


def _split_stats(df: pd.DataFrame, time_col: str = TIME_COL) -> dict[str, int | float]:
    """Build compact row, fraud, and time statistics for one split."""
    fraud_count = int(df[TARGET_COL].sum())
    row_count = int(len(df))
    return {
        "row_count": row_count,
        "fraud_count": fraud_count,
        "fraud_rate": float(fraud_count / row_count) if row_count else 0.0,
        f"{time_col}_min": int(df[time_col].min()) if row_count else None,
        f"{time_col}_max": int(df[time_col].max()) if row_count else None,
    }


def build_split_summary(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_size: int | None = SAMPLE_SIZE,
    time_col: str = TIME_COL,
) -> dict[str, object]:
    """Create the Phase 1 split summary payload."""
    validate_split_integrity(full_df, train_df, valid_df, test_df, time_col=time_col)

    total_rows = int(len(full_df))
    total_fraud_count = int(full_df[TARGET_COL].sum())
    train_stats = _split_stats(train_df, time_col=time_col)
    valid_stats = _split_stats(valid_df, time_col=time_col)
    test_stats = _split_stats(test_df, time_col=time_col)

    return {
        "full_dataset_shape": [int(full_df.shape[0]), int(full_df.shape[1])],
        "number_of_features": int(full_df.shape[1] - 1),
        "total_fraud_count": total_fraud_count,
        "total_fraud_rate": float(total_fraud_count / total_rows) if total_rows else 0.0,
        "train_row_count": train_stats["row_count"],
        "validation_row_count": valid_stats["row_count"],
        "test_row_count": test_stats["row_count"],
        "train_fraud_count": train_stats["fraud_count"],
        "validation_fraud_count": valid_stats["fraud_count"],
        "test_fraud_count": test_stats["fraud_count"],
        "train_fraud_rate": train_stats["fraud_rate"],
        "validation_fraud_rate": valid_stats["fraud_rate"],
        "test_fraud_rate": test_stats["fraud_rate"],
        "train_transactiondt_min": train_stats[f"{time_col}_min"],
        "train_transactiondt_max": train_stats[f"{time_col}_max"],
        "validation_transactiondt_min": valid_stats[f"{time_col}_min"],
        "validation_transactiondt_max": valid_stats[f"{time_col}_max"],
        "test_transactiondt_min": test_stats[f"{time_col}_min"],
        "test_transactiondt_max": test_stats[f"{time_col}_max"],
        "sample_size": sample_size,
        "is_local_debugging_sample": sample_size is not None,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "splits": {
            "train": train_stats,
            "validation": valid_stats,
            "test": test_stats,
        },
    }


if __name__ == "__main__":
    raise SystemExit(
        "Use `python src/check_data_split.py` to validate Phase 1 data loading and splitting."
    )
