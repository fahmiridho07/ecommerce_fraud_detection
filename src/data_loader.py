"""Data loading helpers for the IEEE-CIS Fraud Detection dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    ID_COL,
    SAMPLE_SIZE,
    TARGET_COL,
    TIME_COL,
    TRAIN_IDENTITY_FILE,
    TRAIN_TRANSACTION_FILE,
)


def validate_train_files() -> None:
    """Fail early if required labeled training files are missing."""
    missing_files = [
        path
        for path in (TRAIN_TRANSACTION_FILE, TRAIN_IDENTITY_FILE)
        if not Path(path).exists()
    ]
    if missing_files:
        missing = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(
            "Required training data files were not found:\n"
            f"{missing}\n"
            "Place Kaggle files in data/raw locally, or attach the dataset in Kaggle."
        )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str],
    dataset_name: str,
) -> None:
    """Check that a dataframe contains required columns."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise KeyError(f"{dataset_name} is missing required column(s): {missing}")


def load_train_transaction() -> pd.DataFrame:
    """Load labeled transaction data from train_transaction.csv."""
    validate_train_files()
    return pd.read_csv(TRAIN_TRANSACTION_FILE)


def load_train_identity() -> pd.DataFrame:
    """Load identity data from train_identity.csv."""
    validate_train_files()
    return pd.read_csv(TRAIN_IDENTITY_FILE)


def apply_temporal_debug_sample(
    transaction_df: pd.DataFrame,
    sample_size: int | None = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Use the earliest N rows after sorting by TransactionDT for local debugging.

    This is only for quick local checks. Final experiments should keep
    SAMPLE_SIZE = None.
    """
    if sample_size is None:
        return transaction_df
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer or None.")

    validate_required_columns(transaction_df, [TIME_COL], "train_transaction")
    return (
        transaction_df.sort_values(TIME_COL, ascending=True)
        .head(sample_size)
        .reset_index(drop=True)
        .copy()
    )


def validate_labeled_data(df: pd.DataFrame) -> None:
    """Run basic checks on the merged labeled training dataframe."""
    validate_required_columns(df, [ID_COL, TARGET_COL, TIME_COL], "merged training data")
    if df[ID_COL].duplicated().any():
        duplicate_count = int(df[ID_COL].duplicated().sum())
        raise ValueError(f"Duplicate {ID_COL} values found: {duplicate_count}")
    if df[TARGET_COL].isna().any():
        raise ValueError(f"{TARGET_COL} contains missing values.")


def merge_train_data(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left join train_transaction and train_identity on TransactionID."""
    validate_required_columns(
        transaction_df,
        [ID_COL, TARGET_COL, TIME_COL],
        "train_transaction",
    )
    validate_required_columns(identity_df, [ID_COL], "train_identity")

    if transaction_df[ID_COL].duplicated().any():
        duplicate_count = int(transaction_df[ID_COL].duplicated().sum())
        raise ValueError(f"Duplicate {ID_COL} values found in train_transaction: {duplicate_count}")
    if identity_df[ID_COL].duplicated().any():
        duplicate_count = int(identity_df[ID_COL].duplicated().sum())
        raise ValueError(f"Duplicate {ID_COL} values found in train_identity: {duplicate_count}")

    merged_df = transaction_df.merge(identity_df, on=ID_COL, how="left")
    validate_labeled_data(merged_df)
    return merged_df


def load_labeled_train_data(sample_size: int | None = SAMPLE_SIZE) -> pd.DataFrame:
    """Load, optionally temporally sample, and merge the labeled training data."""
    transaction_df = load_train_transaction()
    transaction_df = apply_temporal_debug_sample(transaction_df, sample_size)
    identity_df = load_train_identity()
    return merge_train_data(transaction_df, identity_df)


def load_train_data(sample_size: int | None = SAMPLE_SIZE) -> pd.DataFrame:
    """Backward-compatible alias for loading merged labeled training data."""
    return load_labeled_train_data(sample_size=sample_size)


if __name__ == "__main__":
    raise SystemExit(
        "Use `python src/check_data_split.py` to validate Phase 1 data loading and splitting."
    )
