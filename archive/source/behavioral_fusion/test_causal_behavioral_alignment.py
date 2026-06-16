"""Identity and leakage tests for corrected causal behavioral feature generation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ID_COL, TARGET_COL, TIME_COL  # noqa: E402
from causal_behavioral_features import (  # noqa: E402
    CAUSAL_ENTITY_GROUPS,
    causal_behavioral_feature_names,
    generate_causal_behavioral_features,
    reproduce_legacy_positional_feature_slices,
    restore_behavioral_features_to_input_order,
)
from feature_engineering import AMOUNT_COL  # noqa: E402


def _base_columns() -> list[str]:
    return [
        ID_COL,
        TIME_COL,
        AMOUNT_COL,
        "card1",
        "addr1",
        "P_emaildomain",
    ]


def _misaligned_legacy_generator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deliberately broken positional slice implementation for negative control."""
    from causal_behavioral_features import _compute_features_for_sorted_split

    feature_names = causal_behavioral_feature_names()
    states = {entity: {} for entity in CAUSAL_ENTITY_GROUPS}
    split_lengths = {
        "train": len(train_df),
        "validation": len(valid_df),
        "test": len(test_df),
    }
    all_df = pd.concat(
        [train_df, valid_df, test_df],
        ignore_index=True,
    ).sort_values([TIME_COL, ID_COL], kind="mergesort").reset_index(drop=True)
    generated = _compute_features_for_sorted_split(all_df, states, feature_names)
    train_end = split_lengths["train"]
    valid_end = train_end + split_lengths["validation"]
    return (
        generated.iloc[:train_end][feature_names].reset_index(drop=True),
        generated.iloc[train_end:valid_end][feature_names].reset_index(drop=True),
        generated.iloc[valid_end:][feature_names].reset_index(drop=True),
    )


def test_shuffled_input_order_restores_original_ids() -> None:
    columns = _base_columns()
    train = pd.DataFrame(
        [
            [300, 20, 150.0, "A", "X", "email.test"],
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 10, 200.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[400, 30, 120.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[500, 40, 180.0, "A", "X", "email.test"]], columns=columns)
    ordered_train = train.sort_values([TIME_COL, ID_COL], kind="mergesort").reset_index(
        drop=True
    )
    ordered_train_cb, _, _, _ = generate_causal_behavioral_features(
        ordered_train,
        valid,
        test,
    )
    shuffled_train_cb, _, _, _ = generate_causal_behavioral_features(train, valid, test)
    expected_by_id = ordered_train_cb.copy()
    expected_by_id[ID_COL] = ordered_train[ID_COL].tolist()
    expected_by_id = expected_by_id.set_index(ID_COL).loc[train[ID_COL].tolist()].reset_index(
        drop=True
    )
    assert shuffled_train_cb.equals(expected_by_id)
    assert shuffled_train_cb.loc[1, "cb_transaction_count_before_card1"] == 0
    assert shuffled_train_cb.loc[0, "cb_transaction_count_before_card1"] == 2


def test_duplicate_timestamps_within_split_use_lower_id_as_prior() -> None:
    columns = _base_columns()
    train = pd.DataFrame(
        [
            [200, 10, 200.0, "A", "X", "email.test"],
            [100, 10, 100.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[300, 20, 120.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[400, 30, 180.0, "A", "X", "email.test"]], columns=columns)
    train_cb, _, _, _ = generate_causal_behavioral_features(train, valid, test)
    assert train_cb.loc[0, "cb_transaction_count_before_card1"] == 1
    assert train_cb.loc[1, "cb_transaction_count_before_card1"] == 0


def test_duplicate_timestamps_across_boundaries_preserve_split_membership() -> None:
    columns = _base_columns()
    train = pd.DataFrame(
        [
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 20, 150.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[300, 20, 120.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[400, 30, 180.0, "A", "X", "email.test"]], columns=columns)
    legacy_train_ids, legacy_valid_ids, _ = reproduce_legacy_positional_feature_slices(
        train,
        valid,
        test,
    )
    train_cb, valid_cb, _, _ = generate_causal_behavioral_features(train, valid, test)
    assert train[ID_COL].tolist() == [100, 200]
    assert valid[ID_COL].tolist() == [300]
    assert valid_cb.loc[0, "cb_transaction_count_before_card1"] == 2
    assert legacy_valid_ids != valid[ID_COL].tolist() or train_cb.shape[0] == 2


def test_transaction_id_feature_identity_for_historical_mean() -> None:
    columns = _base_columns()
    train = pd.DataFrame(
        [
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 20, 300.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[300, 30, 500.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[400, 40, 180.0, "A", "X", "email.test"]], columns=columns)
    train_cb, valid_cb, _, _ = generate_causal_behavioral_features(train, valid, test)
    assert train_cb.loc[1, "cb_historical_mean_amount_before_card1"] == 100.0
    assert valid_cb.loc[0, "cb_historical_mean_amount_before_card1"] == 200.0


def test_future_rows_do_not_change_earlier_features() -> None:
    columns = _base_columns()
    train = pd.DataFrame(
        [
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 20, 150.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[300, 30, 120.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[400, 40, 180.0, "A", "X", "email.test"]], columns=columns)
    baseline_train, _, _, _ = generate_causal_behavioral_features(train, valid, test)
    extended_train = pd.concat(
        [
            train,
            pd.DataFrame([[250, 25, 999.0, "B", "Y", "other.test"]], columns=columns),
        ],
        ignore_index=True,
    )
    rerun_train, _, _, _ = generate_causal_behavioral_features(
        extended_train,
        valid,
        test,
    )
    assert baseline_train.equals(rerun_train.loc[:1].reset_index(drop=True))


def test_label_invariance() -> None:
    columns = _base_columns() + [TARGET_COL]
    train = pd.DataFrame(
        [
            [100, 10, 100.0, "A", "X", "email.test", 0],
            [200, 20, 150.0, "A", "X", "email.test", 1],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[300, 30, 120.0, "A", "X", "email.test", 0]], columns=columns)
    test = pd.DataFrame([[400, 40, 180.0, "A", "X", "email.test", 1]], columns=columns)
    with_label, _, _, _ = generate_causal_behavioral_features(
        train,
        valid.drop(columns=[TARGET_COL]),
        test.drop(columns=[TARGET_COL]),
    )
    no_label, _, _, _ = generate_causal_behavioral_features(
        train.drop(columns=[TARGET_COL]),
        valid.drop(columns=[TARGET_COL]),
        test.drop(columns=[TARGET_COL]),
    )
    assert with_label.equals(no_label)


def test_one_to_one_identity_manifest() -> None:
    columns = _base_columns()
    train = pd.DataFrame([[100, 10, 100.0, "A", "X", "email.test"]], columns=columns)
    valid = pd.DataFrame([[200, 20, 120.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[300, 30, 180.0, "A", "X", "email.test"]], columns=columns)
    _, _, _, summary = generate_causal_behavioral_features(train, valid, test)
    manifests = summary["transaction_id_manifests"]
    assert manifests["train"]["restored_order_matches_input"] is True
    assert manifests["validation"]["duplicate_ids"] is False
    assert manifests["test"]["missing_ids"] is False


def test_original_order_restoration_helper() -> None:
    columns = _base_columns()
    split_df = pd.DataFrame(
        [
            [300, 20, 150.0, "A", "X", "email.test"],
            [100, 10, 100.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    feature_names = causal_behavioral_feature_names()
    generated = pd.DataFrame(
        {
            ID_COL: [100, 300],
            feature_names[0]: [0.0, 1.0],
        }
    )
    for feature in feature_names[1:]:
        generated[feature] = 0.0
    restored = restore_behavioral_features_to_input_order(
        split_df,
        generated,
        feature_names,
    )
    assert restored.loc[0, feature_names[0]] == 1.0
    assert restored.loc[1, feature_names[0]] == 0.0


def test_misaligned_legacy_generator_fails_shuffled_restore_case() -> None:
    columns = _base_columns()
    train = pd.DataFrame(
        [
            [300, 20, 150.0, "A", "X", "email.test"],
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 10, 200.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame([[400, 30, 120.0, "A", "X", "email.test"]], columns=columns)
    test = pd.DataFrame([[500, 40, 180.0, "A", "X", "email.test"]], columns=columns)
    repaired_train, _, _, _ = generate_causal_behavioral_features(train, valid, test)
    legacy_train, _, _ = _misaligned_legacy_generator(train, valid, test)
    assert not legacy_train.equals(repaired_train)


def run_all_tests() -> None:
    tests = [
        test_shuffled_input_order_restores_original_ids,
        test_duplicate_timestamps_within_split_use_lower_id_as_prior,
        test_duplicate_timestamps_across_boundaries_preserve_split_membership,
        test_transaction_id_feature_identity_for_historical_mean,
        test_future_rows_do_not_change_earlier_features,
        test_label_invariance,
        test_one_to_one_identity_manifest,
        test_original_order_restoration_helper,
        test_misaligned_legacy_generator_fails_shuffled_restore_case,
    ]
    for test_func in tests:
        test_func()
    print(f"All {len(tests)} causal behavioral alignment tests passed.")


if __name__ == "__main__":
    run_all_tests()