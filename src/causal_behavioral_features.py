"""Leakage-safe causal behavioral feature helpers for B2/B3 experiments."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import pandas as pd

from config import ID_COL, TARGET_COL, TIME_COL
from feature_engineering import AMOUNT_COL, build_combo_key


CAUSAL_ENTITY_GROUPS = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_P_emaildomain": ["card1", "P_emaildomain"],
}

WINDOW_ENTITY_GROUPS = {
    "card1",
    "card1_P_emaildomain",
}

WINDOWS_SECONDS = {
    "1h": 3600,
    "24h": 24 * 3600,
}

BASE_FEATURE_SUFFIXES = [
    "transaction_count_before",
    "time_since_previous_transaction",
    "historical_mean_amount_before",
    "historical_std_amount_before",
    "amount_deviation_from_historical_mean",
]

WINDOW_FEATURE_SUFFIXES = [
    "count_in_previous_1_hour",
    "count_in_previous_24_hours",
]


@dataclass
class EntityState:
    count: int = 0
    last_time: float | None = None
    amount_count: int = 0
    amount_sum: float = 0.0
    amount_sum_sq: float = 0.0
    times: list[float] = field(default_factory=list)


def causal_behavioral_feature_names() -> list[str]:
    """Return deterministic causal behavioral feature column names."""
    feature_names: list[str] = []
    for entity in CAUSAL_ENTITY_GROUPS:
        for suffix in BASE_FEATURE_SUFFIXES:
            feature_names.append(f"cb_{suffix}_{entity}")
        if entity in WINDOW_ENTITY_GROUPS:
            for suffix in WINDOW_FEATURE_SUFFIXES:
                feature_names.append(f"cb_{suffix}_{entity}")
    return feature_names


def build_feature_definition_metadata() -> dict[str, Any]:
    """Build JSON-safe feature metadata for audit artifacts."""
    formulas = {
        "cb_transaction_count_before_{entity}": (
            "Number of prior transactions for entity key strictly before current row."
        ),
        "cb_time_since_previous_transaction_{entity}": (
            "TransactionDT delta from most recent prior entity transaction; "
            "NaN if no prior transaction."
        ),
        "cb_historical_mean_amount_before_{entity}": (
            "Expanding mean of TransactionAmt over prior entity transactions only."
        ),
        "cb_historical_std_amount_before_{entity}": (
            "Expanding sample std of TransactionAmt over prior entity transactions; "
            "requires at least two prior amounts."
        ),
        "cb_amount_deviation_from_historical_mean_{entity}": (
            "TransactionAmt minus cb_historical_mean_amount_before_{entity}."
        ),
        "cb_count_in_previous_1_hour_{entity}": (
            "Count of prior entity transactions with TransactionDT in (t-3600, t]."
        ),
        "cb_count_in_previous_24_hours_{entity}": (
            "Count of prior entity transactions with TransactionDT in (t-86400, t]."
        ),
    }
    return {
        "feature_names": causal_behavioral_feature_names(),
        "feature_count": len(causal_behavioral_feature_names()),
        "entity_definitions": CAUSAL_ENTITY_GROUPS,
        "entity_count": len(CAUSAL_ENTITY_GROUPS),
        "window_entity_groups": sorted(WINDOW_ENTITY_GROUPS),
        "windows_seconds": WINDOWS_SECONDS,
        "calculation_formulas": formulas,
        "causal_policy": (
            "Each row uses only transactions strictly before the current row ordered by "
            "TransactionDT then TransactionID."
        ),
        "state_transition_policy": (
            "Online state updates after each row during a single chronological pass over "
            "train, then validation, then test."
        ),
        "tie_breaking_policy": (
            "Rows with equal TransactionDT are ordered by ascending TransactionID; "
            "lower TransactionID rows are visible to higher TransactionID rows at the "
            "same timestamp."
        ),
        "target_not_used": True,
        "future_rows_not_used": True,
        "labels_not_used_in_feature_state": True,
    }


def generate_causal_behavioral_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create causal behavioral features with state continuation across splits."""
    _validate_split_inputs(train_df, valid_df, test_df)
    split_lengths = {
        "train": len(train_df),
        "validation": len(valid_df),
        "test": len(test_df),
    }

    all_df = pd.concat(
        [
            train_df.reset_index(drop=True),
            valid_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        ],
        axis=0,
        ignore_index=True,
    )
    all_df = all_df.sort_values(
        [TIME_COL, ID_COL],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    time_values = pd.to_numeric(all_df[TIME_COL], errors="coerce").to_numpy(
        dtype="float64",
        copy=False,
    )
    amount_values = pd.to_numeric(all_df[AMOUNT_COL], errors="coerce").to_numpy(
        dtype="float64",
        copy=False,
    )
    if np.isnan(time_values).any():
        raise ValueError(f"{TIME_COL} contains missing or non-numeric values.")

    keys_by_entity = {
        entity: build_combo_key(all_df, columns).to_numpy()
        for entity, columns in CAUSAL_ENTITY_GROUPS.items()
    }
    feature_names = causal_behavioral_feature_names()
    feature_arrays = _initialize_feature_arrays(len(all_df), feature_names)
    states = {entity: {} for entity in CAUSAL_ENTITY_GROUPS}

    for row_index in range(len(all_df)):
        current_time = time_values[row_index]
        current_amount = amount_values[row_index]
        for entity in CAUSAL_ENTITY_GROUPS:
            key = keys_by_entity[entity][row_index]
            state = states[entity].get(key)
            _write_row_features(
                entity=entity,
                row_index=row_index,
                current_time=current_time,
                current_amount=current_amount,
                state=state,
                feature_arrays=feature_arrays,
            )
            states[entity][key] = _update_entity_state(
                state=state,
                current_time=current_time,
                current_amount=current_amount,
            )

    features_all = pd.DataFrame(feature_arrays, columns=feature_names)
    train_end = split_lengths["train"]
    valid_end = train_end + split_lengths["validation"]
    X_train_cb = features_all.iloc[:train_end].reset_index(drop=True)
    X_valid_cb = features_all.iloc[train_end:valid_end].reset_index(drop=True)
    X_test_cb = features_all.iloc[valid_end:].reset_index(drop=True)

    duplicate_timestamp_rows = int(
        pd.Series(time_values).duplicated(keep=False).sum()
    )
    summary = {
        **build_feature_definition_metadata(),
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "split_row_counts": split_lengths,
        "sort_columns": [TIME_COL, ID_COL],
    }
    return X_train_cb, X_valid_cb, X_test_cb, summary


def validate_causal_behavioral_features(
    X_train_cb: pd.DataFrame,
    X_valid_cb: pd.DataFrame,
    X_test_cb: pd.DataFrame,
) -> dict[str, object]:
    """Validate causal behavioral invariants."""
    expected_features = causal_behavioral_feature_names()
    checks: dict[str, object] = {
        "expected_feature_count": len(expected_features),
        "expected_features_present": True,
        "no_infinite_values": True,
        "non_negative_time_since_prev": True,
        "window_counts_monotonic": True,
        "synthetic_fixture_passed": True,
        "future_rows_do_not_change_past_features": True,
        "labels_do_not_affect_features": True,
    }

    for split_name, X in (
        ("train", X_train_cb),
        ("validation", X_valid_cb),
        ("test", X_test_cb),
    ):
        if X.columns.tolist() != expected_features:
            checks["expected_features_present"] = False
            raise ValueError(f"{split_name} causal behavioral columns do not match.")

        values = X.to_numpy(dtype="float64", copy=False)
        if np.isinf(values).any():
            checks["no_infinite_values"] = False
            raise ValueError(f"{split_name} causal behavioral features contain inf.")

        for entity in CAUSAL_ENTITY_GROUPS:
            time_col = f"cb_time_since_previous_transaction_{entity}"
            valid_values = X[time_col].dropna()
            if (valid_values < 0).any():
                checks["non_negative_time_since_prev"] = False
                raise ValueError(
                    f"{split_name} {time_col} contains negative prior gaps."
                )

        for entity in WINDOW_ENTITY_GROUPS:
            one_hour = X[f"cb_count_in_previous_1_hour_{entity}"]
            one_day = X[f"cb_count_in_previous_24_hours_{entity}"]
            count_before = X[f"cb_transaction_count_before_{entity}"]
            if not ((one_hour <= one_day).all() and (one_day <= count_before).all()):
                checks["window_counts_monotonic"] = False
                raise ValueError(
                    f"{split_name} window count ordering failed for {entity}."
                )

    run_causal_behavioral_fixture_check()
    run_future_immutability_check()
    run_label_invariance_check()
    return checks


def run_causal_behavioral_fixture_check() -> dict[str, object]:
    """Deterministic fixture for tie-breaking and split-boundary continuation."""
    columns = [
        ID_COL,
        TIME_COL,
        AMOUNT_COL,
        "card1",
        "addr1",
        "P_emaildomain",
    ]
    train = pd.DataFrame(
        [
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 10, 200.0, "A", "X", "email.test"],
            [300, 20, 150.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame(
        [[400, 30, 120.0, "A", "X", "email.test"]],
        columns=columns,
    )
    test = pd.DataFrame(
        [[500, 40, 180.0, "A", "X", "email.test"]],
        columns=columns,
    )
    train_cb, valid_cb, test_cb, _ = generate_causal_behavioral_features(
        train,
        valid,
        test,
    )

    if train_cb.loc[0, "cb_transaction_count_before_card1"] != 0:
        raise ValueError("Fixture failed: first row should have count 0.")
    if train_cb.loc[1, "cb_transaction_count_before_card1"] != 1:
        raise ValueError(
            "Fixture failed: second row at same timestamp should see first row."
        )
    if train_cb.loc[2, "cb_transaction_count_before_card1"] != 2:
        raise ValueError("Fixture failed: third train row should see two prior rows.")
    if train_cb.loc[2, "cb_time_since_previous_transaction_card1"] != 10:
        raise ValueError("Fixture failed: time gap should be 10 seconds.")
    if valid_cb.loc[0, "cb_transaction_count_before_card1"] != 3:
        raise ValueError("Fixture failed: validation should see train history.")
    if test_cb.loc[0, "cb_transaction_count_before_card1"] != 4:
        raise ValueError("Fixture failed: test should see train and validation history.")

    return {
        "transactionid_tie_breaking_within_timestamp": True,
        "validation_uses_prior_train_history": True,
        "test_uses_prior_train_validation_history": True,
    }


def run_future_immutability_check() -> dict[str, object]:
    """Verify later rows do not change earlier computed features."""
    columns = [
        ID_COL,
        TIME_COL,
        AMOUNT_COL,
        "card1",
        "addr1",
        "P_emaildomain",
    ]
    train = pd.DataFrame(
        [
            [100, 10, 100.0, "A", "X", "email.test"],
            [200, 20, 150.0, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame(
        [[300, 30, 120.0, "A", "X", "email.test"]],
        columns=columns,
    )
    test = pd.DataFrame(
        [[400, 40, 180.0, "A", "X", "email.test"]],
        columns=columns,
    )
    baseline_train, baseline_valid, baseline_test, _ = generate_causal_behavioral_features(
        train,
        valid,
        test,
    )

    extended_train = pd.concat(
        [
            train,
            pd.DataFrame(
                [[250, 25, 999.0, "B", "Y", "other.test"]],
                columns=columns,
            ),
        ],
        ignore_index=True,
    )
    extended_train = extended_train.sort_values(
        [TIME_COL, ID_COL],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    rerun_train, _, _, _ = generate_causal_behavioral_features(
        extended_train,
        valid,
        test,
    )
    common_ids = [100, 200]
    baseline_subset = baseline_train.loc[
        baseline_train.index.isin([0, 1])
    ].reset_index(drop=True)
    rerun_subset = rerun_train.loc[
        rerun_train.index < 2
    ].reset_index(drop=True)
    if not baseline_subset.equals(rerun_subset):
        raise ValueError(
            "Fixture failed: inserting a future train row changed earlier features."
        )
    return {"future_rows_do_not_change_past_features": True}


def run_label_invariance_check() -> dict[str, object]:
    """Verify isFraud is not required and does not change features."""
    columns = [
        ID_COL,
        TIME_COL,
        AMOUNT_COL,
        TARGET_COL,
        "card1",
        "addr1",
        "P_emaildomain",
    ]
    train = pd.DataFrame(
        [
            [100, 10, 100.0, 0, "A", "X", "email.test"],
            [200, 20, 150.0, 1, "A", "X", "email.test"],
        ],
        columns=columns,
    )
    train_no_label = train.drop(columns=[TARGET_COL])
    valid = pd.DataFrame(
        [[300, 30, 120.0, 0, "A", "X", "email.test"]],
        columns=columns,
    )
    valid_no_label = valid.drop(columns=[TARGET_COL])
    test = pd.DataFrame(
        [[400, 40, 180.0, 1, "A", "X", "email.test"]],
        columns=columns,
    )
    test_no_label = test.drop(columns=[TARGET_COL])

    with_label_train, _, _, _ = generate_causal_behavioral_features(
        train,
        valid.drop(columns=[TARGET_COL]),
        test.drop(columns=[TARGET_COL]),
    )
    no_label_train, _, _, _ = generate_causal_behavioral_features(
        train_no_label,
        valid_no_label,
        test_no_label,
    )
    if not with_label_train.equals(no_label_train):
        raise ValueError("Fixture failed: label presence changed causal features.")
    return {"labels_do_not_affect_features": True}


def _validate_split_inputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    required = [ID_COL, TIME_COL, AMOUNT_COL]
    for entity_columns in CAUSAL_ENTITY_GROUPS.values():
        required.extend(entity_columns)
    required = sorted(set(required))

    for split_name, df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise KeyError(
                f"{split_name} is missing required column(s): " + ", ".join(missing)
            )


def _initialize_feature_arrays(
    n_rows: int,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    feature_arrays = {
        feature_name: np.full(n_rows, np.nan, dtype="float32")
        for feature_name in feature_names
    }
    for entity in CAUSAL_ENTITY_GROUPS:
        feature_arrays[f"cb_transaction_count_before_{entity}"] = np.zeros(
            n_rows,
            dtype="float32",
        )
        if entity in WINDOW_ENTITY_GROUPS:
            for suffix in WINDOW_FEATURE_SUFFIXES:
                feature_arrays[f"cb_{suffix}_{entity}"] = np.zeros(
                    n_rows,
                    dtype="float32",
                )
    return feature_arrays


def _write_row_features(
    entity: str,
    row_index: int,
    current_time: float,
    current_amount: float,
    state: EntityState | None,
    feature_arrays: dict[str, np.ndarray],
) -> None:
    if state is None:
        return

    feature_arrays[f"cb_transaction_count_before_{entity}"][row_index] = float(
        state.count
    )
    if state.last_time is not None:
        feature_arrays[f"cb_time_since_previous_transaction_{entity}"][row_index] = (
            current_time - state.last_time
        )
    if state.amount_count > 0:
        mean_before = state.amount_sum / state.amount_count
        feature_arrays[f"cb_historical_mean_amount_before_{entity}"][row_index] = (
            mean_before
        )
        if np.isfinite(current_amount):
            feature_arrays[
                f"cb_amount_deviation_from_historical_mean_{entity}"
            ][row_index] = current_amount - mean_before
        if state.amount_count > 1:
            variance = (
                state.amount_sum_sq
                - (state.amount_sum * state.amount_sum / state.amount_count)
            ) / (state.amount_count - 1)
            feature_arrays[f"cb_historical_std_amount_before_{entity}"][row_index] = (
                math.sqrt(max(0.0, variance))
            )

    if entity in WINDOW_ENTITY_GROUPS:
        for window_name, window_seconds in WINDOWS_SECONDS.items():
            suffix = (
                "count_in_previous_1_hour"
                if window_name == "1h"
                else "count_in_previous_24_hours"
            )
            lower_bound = current_time - window_seconds
            start_index = bisect_left(state.times, lower_bound)
            feature_arrays[f"cb_{suffix}_{entity}"][row_index] = float(
                len(state.times) - start_index
            )


def _update_entity_state(
    state: EntityState | None,
    current_time: float,
    current_amount: float,
) -> EntityState:
    if state is None:
        state = EntityState()
    state.count += 1
    state.last_time = current_time
    if np.isfinite(current_amount):
        state.amount_count += 1
        state.amount_sum += float(current_amount)
        state.amount_sum_sq += float(current_amount * current_amount)
    state.times.append(current_time)
    return state


if __name__ == "__main__":
    run_causal_behavioral_fixture_check()
    run_future_immutability_check()
    run_label_invariance_check()
    print("Causal behavioral fixture checks passed.")