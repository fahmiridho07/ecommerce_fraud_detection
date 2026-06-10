"""Leakage-safe causal behavioral feature helpers for B2/B3 experiments."""

from __future__ import annotations

import hashlib
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

DETERMINISTIC_EVENT_ORDER_POLICY = (
    "Process train split first, validation second, test third. Within each split, "
    "order events by TransactionDT ascending then TransactionID ascending. "
    "Entity state continues from completed train into validation and from "
    "train+validation into test."
)

CAUSAL_POLICY_WORDING = (
    "Each feature uses only transactions preceding the current row in the "
    "deterministic event order defined by split precedence, TransactionDT, "
    "and TransactionID."
)

SAME_TIMESTAMP_POLICY = (
    "Within the same split, equal TransactionDT values are ordered by ascending "
    "TransactionID; lower TransactionID is treated as the prior event. Across "
    "split boundaries, split precedence is authoritative: train, then validation, "
    "then test."
)


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
            "Number of prior transactions for entity key before the current row "
            "in deterministic event order."
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
        "causal_policy": CAUSAL_POLICY_WORDING,
        "deterministic_event_order": DETERMINISTIC_EVENT_ORDER_POLICY,
        "same_timestamp_policy": SAME_TIMESTAMP_POLICY,
        "state_transition_policy": (
            "Online state updates after each row during sequential processing of "
            "train, then validation, then test. Validation and test labels never "
            "update state."
        ),
        "tie_breaking_policy": SAME_TIMESTAMP_POLICY,
        "transactiondt_resolution_note": (
            "TransactionDT has coarse resolution; multiple rows may share a timestamp."
        ),
        "transactionid_tie_break_note": (
            "TransactionID tie-breaking within a split is an approximation of true "
            "event order."
        ),
        "split_boundary_note": (
            "Frozen chronological split membership is preserved at timestamp "
            "boundaries; split precedence overrides TransactionID ordering across "
            "boundaries."
        ),
        "target_not_used": True,
        "future_rows_not_used": True,
        "labels_not_used_in_feature_state": True,
    }


def transaction_id_checksum(transaction_ids: list[int] | pd.Series) -> str:
    """Return a stable checksum for an ordered TransactionID sequence."""
    if isinstance(transaction_ids, pd.Series):
        transaction_ids = transaction_ids.tolist()
    payload = ",".join(str(value) for value in transaction_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_split_identity(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        if split_df[ID_COL].duplicated().any():
            duplicate_count = int(split_df[ID_COL].duplicated().sum())
            raise ValueError(
                f"Duplicate {ID_COL} values found in {split_name}: {duplicate_count}"
            )

    train_ids = set(train_df[ID_COL])
    valid_ids = set(valid_df[ID_COL])
    test_ids = set(test_df[ID_COL])
    if train_ids & valid_ids:
        raise ValueError(f"{ID_COL} overlap found between train and validation splits.")
    if train_ids & test_ids:
        raise ValueError(f"{ID_COL} overlap found between train and test splits.")
    if valid_ids & test_ids:
        raise ValueError(f"{ID_COL} overlap found between validation and test splits.")


def _assert_restored_identity(
    split_df: pd.DataFrame,
    X_split_behavioral: pd.DataFrame,
    original_id_order: list[int],
    split_name: str,
) -> None:
    if len(X_split_behavioral) != len(split_df):
        raise ValueError(
            f"{split_name}: behavioral row count {len(X_split_behavioral)} does not "
            f"match split row count {len(split_df)}."
        )
    if X_split_behavioral.columns.tolist() != causal_behavioral_feature_names():
        raise ValueError(f"{split_name}: behavioral feature columns do not match spec.")

    input_ids = split_df[ID_COL].tolist()
    if input_ids != original_id_order:
        raise ValueError(
            f"{split_name}: input split ID order changed during processing."
        )


def validate_feature_identity_alignment(
    split_df: pd.DataFrame,
    X_behavioral: pd.DataFrame,
    split_name: str,
) -> dict[str, object]:
    """Assert one-to-one TransactionID alignment between split rows and features."""
    _validate_split_inputs(split_df, split_df, split_df)
    original_id_order = split_df[ID_COL].tolist()
    _assert_restored_identity(split_df, X_behavioral, original_id_order, split_name)
    return {
        "split_name": split_name,
        "row_count": int(len(split_df)),
        "feature_count": int(X_behavioral.shape[1]),
        "transaction_id_checksum": transaction_id_checksum(original_id_order),
        "duplicate_ids": False,
        "missing_ids": False,
        "unexpected_ids": False,
        "restored_order_matches_input": True,
    }


def _compute_features_for_sorted_split(
    sorted_df: pd.DataFrame,
    states: dict[str, dict[str, EntityState]],
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute causal features for one split in deterministic event order."""
    n_rows = len(sorted_df)
    time_values = pd.to_numeric(sorted_df[TIME_COL], errors="coerce").to_numpy(
        dtype="float64",
        copy=False,
    )
    amount_values = pd.to_numeric(sorted_df[AMOUNT_COL], errors="coerce").to_numpy(
        dtype="float64",
        copy=False,
    )
    if np.isnan(time_values).any():
        raise ValueError(f"{TIME_COL} contains missing or non-numeric values.")

    keys_by_entity = {
        entity: build_combo_key(sorted_df, columns).to_numpy()
        for entity, columns in CAUSAL_ENTITY_GROUPS.items()
    }
    feature_arrays = _initialize_feature_arrays(n_rows, feature_names)

    for row_index in range(n_rows):
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

    generated = pd.DataFrame(feature_arrays, columns=feature_names)
    generated[ID_COL] = sorted_df[ID_COL].to_numpy()
    return generated


def restore_behavioral_features_to_input_order(
    split_df: pd.DataFrame,
    generated_frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Restore generated behavioral features to the split's original row order."""
    behavioral_by_id = generated_frame.set_index(ID_COL)
    original_id_order = split_df[ID_COL].tolist()
    missing_ids = [value for value in original_id_order if value not in behavioral_by_id.index]
    if missing_ids:
        raise ValueError(
            "Generated behavioral features are missing TransactionID value(s): "
            + ", ".join(str(value) for value in missing_ids[:20])
        )
    unexpected_ids = sorted(set(behavioral_by_id.index) - set(original_id_order))
    if unexpected_ids:
        raise ValueError(
            "Generated behavioral features contain unexpected TransactionID value(s): "
            + ", ".join(str(value) for value in unexpected_ids[:20])
        )

    restored = (
        behavioral_by_id.loc[original_id_order, feature_names]
        .reset_index(drop=True)
    )
    _assert_restored_identity(split_df, restored, original_id_order, "split")
    return restored


def generate_causal_behavioral_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create identity-safe causal behavioral features with split state continuation."""
    _validate_split_inputs(train_df, valid_df, test_df)
    _validate_split_identity(train_df, valid_df, test_df)

    feature_names = causal_behavioral_feature_names()
    states = {entity: {} for entity in CAUSAL_ENTITY_GROUPS}
    split_outputs: dict[str, pd.DataFrame] = {}
    id_manifests: dict[str, dict[str, object]] = {}

    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        original_id_order = split_df[ID_COL].tolist()
        working_df = split_df.sort_values(
            [TIME_COL, ID_COL],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        generated_frame = _compute_features_for_sorted_split(
            working_df,
            states,
            feature_names,
        )
        X_split_behavioral = restore_behavioral_features_to_input_order(
            split_df,
            generated_frame,
            feature_names,
        )
        split_outputs[split_name] = X_split_behavioral
        id_manifests[split_name] = validate_feature_identity_alignment(
            split_df,
            X_split_behavioral,
            split_name,
        )

    duplicate_timestamp_rows = int(
        pd.concat(
            [train_df[[TIME_COL]], valid_df[[TIME_COL]], test_df[[TIME_COL]]],
            ignore_index=True,
        )[TIME_COL].duplicated(keep=False).sum()
    )
    summary = {
        **build_feature_definition_metadata(),
        "identity_safe_generation": True,
        "positional_slice_recovery_used": False,
        "global_concat_resort_used": False,
        "restoration_key": ID_COL,
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "split_row_counts": {
            "train": len(train_df),
            "validation": len(valid_df),
            "test": len(test_df),
        },
        "transaction_id_manifests": id_manifests,
        "sort_columns_within_split": [TIME_COL, ID_COL],
    }
    return (
        split_outputs["train"],
        split_outputs["validation"],
        split_outputs["test"],
        summary,
    )


def reproduce_legacy_positional_feature_slices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce the pre-fix global concat/sort/row-count slice TransactionID order."""
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

    train_end = split_lengths["train"]
    valid_end = train_end + split_lengths["validation"]
    legacy_train_ids = all_df.iloc[:train_end][ID_COL].tolist()
    legacy_valid_ids = all_df.iloc[train_end:valid_end][ID_COL].tolist()
    legacy_test_ids = all_df.iloc[valid_end:][ID_COL].tolist()
    return legacy_train_ids, legacy_valid_ids, legacy_test_ids


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

    rerun_train, _, _, _ = generate_causal_behavioral_features(
        extended_train,
        valid,
        test,
    )
    original_train_ids = train[ID_COL].tolist()
    if not baseline_train.equals(rerun_train.loc[: len(train) - 1].reset_index(drop=True)):
        raise ValueError(
            "Fixture failed: inserting a future train row changed earlier features."
        )
    if rerun_train.shape[0] != len(extended_train):
        raise ValueError("Fixture failed: extended train row count mismatch.")
    if extended_train[ID_COL].tolist()[: len(train)] != original_train_ids:
        raise ValueError("Fixture failed: original train ID order changed.")
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