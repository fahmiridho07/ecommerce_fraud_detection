"""Chronological and stratified splitting utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import validate_labeled_data, validate_required_columns

# Duplicate TransactionDT values are expected in IEEE-CIS data.
# TransactionID is used only as a deterministic tie-breaker for split ordering,
# not as a model feature.
CHRONOLOGICAL_SORT_ORDER_NOTE = (
    "Events are ordered by TransactionDT ascending; duplicate TransactionDT "
    "values are expected and TransactionID ascending is used only as a "
    "deterministic tie-breaker, not as a model feature."
)


def sort_by_transaction_time(
    df: pd.DataFrame,
    time_col: str = TIME_COL,
    id_col: str = ID_COL,
) -> pd.DataFrame:
    """Sort labeled data by TransactionDT ascending, then TransactionID ascending.

    Uses stable sorting so row order is deterministic when TransactionDT ties.
    """
    validate_required_columns(df, [time_col, id_col], "labeled data")
    return (
        df.sort_values([time_col, id_col], ascending=[True, True], kind="mergesort")
        .reset_index(drop=True)
    )


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
    time_col: str = TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by deterministic event order into train, validation, and test sets.

    Event ordering is TransactionDT ascending with TransactionID ascending as the
    tie-breaker. See CHRONOLOGICAL_SORT_ORDER_NOTE.
    """
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


def validate_holdout_split_integrity(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = TIME_COL,
) -> None:
    """Validate row counts and TransactionID separation without temporal ordering."""
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


def temporal_order_preserved(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_col: str = TIME_COL,
) -> bool:
    """Return True when train, validation, and test periods are strictly ordered."""
    if train_df.empty or valid_df.empty or test_df.empty:
        return False
    return (
        train_df[time_col].max() <= valid_df[time_col].min()
        and valid_df[time_col].max() <= test_df[time_col].min()
    )


def stratified_holdout_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
    target_col: str = TARGET_COL,
    random_seed: int = RANDOM_SEED,
    time_col: str = TIME_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data with stratified sampling while preserving global fraud rate."""
    validate_labeled_data(df)
    if round(train_ratio + valid_ratio + test_ratio, 10) != 1.0:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    train_valid_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[target_col],
        random_state=random_seed,
    )
    valid_fraction = valid_ratio / (train_ratio + valid_ratio)
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_fraction,
        stratify=train_valid_df[target_col],
        random_state=random_seed,
    )

    train_df = train_df.reset_index(drop=True).copy()
    valid_df = valid_df.reset_index(drop=True).copy()
    test_df = test_df.reset_index(drop=True).copy()
    validate_holdout_split_integrity(df, train_df, valid_df, test_df, time_col=time_col)
    return train_df, valid_df, test_df


def stratified_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = TARGET_COL,
    random_seed: int = RANDOM_SEED,
) -> StratifiedKFold:
    """Return a configured StratifiedKFold splitter for labeled experiment data."""
    validate_labeled_data(df)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if n_splits > len(df):
        raise ValueError("n_splits cannot exceed the number of rows.")

    labels = df[target_col].to_numpy()
    class_counts = pd.Series(labels).value_counts()
    if class_counts.min() < n_splits:
        raise ValueError(
            "Stratified K-fold requires at least n_splits rows per class. "
            f"Smallest class count={int(class_counts.min())}, n_splits={n_splits}."
        )

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed,
    )


def build_holdout_split_summary(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_strategy: str,
    sample_size: int | None = SAMPLE_SIZE,
    time_col: str = TIME_COL,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Create a split summary for chronological or stratified holdout experiments."""
    validate_holdout_split_integrity(full_df, train_df, valid_df, test_df, time_col=time_col)

    total_rows = int(len(full_df))
    total_fraud_count = int(full_df[TARGET_COL].sum())
    train_stats = _split_stats(train_df, time_col=time_col)
    valid_stats = _split_stats(valid_df, time_col=time_col)
    test_stats = _split_stats(test_df, time_col=time_col)

    return {
        "split_strategy": split_strategy,
        "temporal_order_preserved": bool(
            temporal_order_preserved(
                train_df,
                valid_df,
                test_df,
                time_col=time_col,
            )
        ),
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
        "random_seed": random_seed,
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


def build_split_summary(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_size: int | None = SAMPLE_SIZE,
    time_col: str = TIME_COL,
) -> dict[str, object]:
    """Create the Phase 1 chronological split summary payload."""
    validate_split_integrity(full_df, train_df, valid_df, test_df, time_col=time_col)
    summary = build_holdout_split_summary(
        full_df,
        train_df,
        valid_df,
        test_df,
        split_strategy="chronological",
        sample_size=sample_size,
        time_col=time_col,
    )
    summary["temporal_order_preserved"] = True
    return summary


if __name__ == "__main__":
    raise SystemExit(
        "Use `python src/check_data_split.py` to validate Phase 1 data loading and splitting."
    )
