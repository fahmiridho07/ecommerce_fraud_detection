"""Validate Phase 1 data loading, merging, and chronological splitting."""

from __future__ import annotations

from config import SAMPLE_SIZE, SPLIT_SUMMARY_FILE
from data_loader import load_labeled_train_data
from splitting import build_split_summary, chronological_split
from utils import log, save_json


def _format_rate(value: float) -> str:
    return f"{value:.4%}"


def main() -> None:
    sample_mode = "debug sample" if SAMPLE_SIZE is not None else "full experiment"
    log(f"Loading labeled training data ({sample_mode}).")
    merged_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(merged_df)
    summary = build_split_summary(
        merged_df,
        train_df,
        valid_df,
        test_df,
        sample_size=SAMPLE_SIZE,
    )
    save_json(summary, SPLIT_SUMMARY_FILE)

    print()
    print("Phase 1 Data Split Summary")
    print("==========================")
    print(f"Merged dataset shape : {tuple(summary['full_dataset_shape'])}")
    print(f"Sample mode          : {sample_mode}")
    print(f"Split summary saved  : {SPLIT_SUMMARY_FILE}")
    print()
    print("Split sizes")
    print(f"  Train      : {summary['train_row_count']:,}")
    print(f"  Validation : {summary['validation_row_count']:,}")
    print(f"  Test       : {summary['test_row_count']:,}")
    print()
    print("Fraud rates")
    print(f"  Train      : {_format_rate(summary['train_fraud_rate'])}")
    print(f"  Validation : {_format_rate(summary['validation_fraud_rate'])}")
    print(f"  Test       : {_format_rate(summary['test_fraud_rate'])}")
    print()
    print("TransactionDT ranges")
    print(
        "  Train      : "
        f"{summary['train_transactiondt_min']} - {summary['train_transactiondt_max']}"
    )
    print(
        "  Validation : "
        f"{summary['validation_transactiondt_min']} - {summary['validation_transactiondt_max']}"
    )
    print(
        "  Test       : "
        f"{summary['test_transactiondt_min']} - {summary['test_transactiondt_max']}"
    )


if __name__ == "__main__":
    main()
