"""Validate thesis data loading, merging, and the default holdout split."""

from __future__ import annotations

import argparse

from config import (
    DEFAULT_SPLIT_STRATEGY,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SPLIT_SUMMARY_FILE,
    SUPPORTED_SPLIT_STRATEGIES,
)
from data_loader import load_labeled_train_data
from splitting import build_holdout_split_summary, create_holdout_split
from utils import log, save_json


def _format_rate(value: float) -> str:
    return f"{value:.4%}"


def run(split_strategy: str = DEFAULT_SPLIT_STRATEGY) -> None:
    sample_mode = "debug sample" if SAMPLE_SIZE is not None else "full experiment"
    log(f"Loading labeled training data ({sample_mode}).")
    merged_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log(f"Creating {split_strategy} train/validation/test split.")
    train_df, valid_df, test_df = create_holdout_split(
        merged_df,
        split_strategy=split_strategy,
    )
    summary = build_holdout_split_summary(
        merged_df,
        train_df,
        valid_df,
        test_df,
        split_strategy=split_strategy,
        sample_size=SAMPLE_SIZE,
        random_seed=RANDOM_SEED,
    )
    save_json(summary, SPLIT_SUMMARY_FILE)

    print()
    print("Phase 1 Data Split Summary")
    print("==========================")
    print(f"Merged dataset shape : {tuple(summary['full_dataset_shape'])}")
    print(f"Sample mode          : {sample_mode}")
    print(f"Split strategy       : {summary['split_strategy']}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate data loading and the configured holdout split."
    )
    parser.add_argument(
        "--split-strategy",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help="Holdout split strategy. Default is the active thesis stratified reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(split_strategy=args.split_strategy)


if __name__ == "__main__":
    main()
