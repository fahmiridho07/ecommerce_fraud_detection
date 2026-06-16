"""Leakage-safe historical velocity feature helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_left
import math
from typing import Any

import numpy as np
import pandas as pd

from config import TARGET_COL, TIME_COL
from feature_engineering import AMOUNT_COL, build_combo_key


HISTORICAL_ENTITY_GROUPS = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_P_emaildomain": ["card1", "P_emaildomain"],
    "P_emaildomain": ["P_emaildomain"],
    "DeviceInfo": ["DeviceInfo"],
}

WINDOW_ENTITY_GROUPS = {
    "card1",
    "card1_P_emaildomain",
}

WINDOWS_SECONDS = {
    "1h": 3600,
    "24h": 24 * 3600,
    "7d": 7 * 24 * 3600,
}

BASE_FEATURE_TEMPLATES = [
    "hist_count_before_{entity}",
    "hist_time_since_prev_{entity}",
    "hist_amt_prev_{entity}",
    "hist_amt_diff_prev_{entity}",
    "hist_amt_ratio_prev_{entity}",
    "hist_amt_mean_before_{entity}",
    "hist_amt_std_before_{entity}",
]


@dataclass
class HistoricalState:
    count: int = 0
    last_time: float | None = None
    last_amount: float = math.nan
    amount_count: int = 0
    amount_sum: float = 0.0
    amount_sum_sq: float = 0.0
    times: list[float] = field(default_factory=list)


def historical_feature_names() -> list[str]:
    """Return the ordered historical feature columns."""
    feature_names: list[str] = []
    for entity in HISTORICAL_ENTITY_GROUPS:
        feature_names.extend(
            template.format(entity=entity) for template in BASE_FEATURE_TEMPLATES
        )
        if entity in WINDOW_ENTITY_GROUPS:
            feature_names.extend(
                f"hist_count_last_{window_name}_{entity}"
                for window_name in WINDOWS_SECONDS
            )
    return feature_names


def generate_historical_velocity_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create historical features over train, validation, and test in time order."""
    _validate_inputs(X_train, X_valid, X_test)
    split_lengths = {
        "train": len(X_train),
        "validation": len(X_valid),
        "test": len(X_test),
    }
    all_df = pd.concat(
        [
            X_train.reset_index(drop=True),
            X_valid.reset_index(drop=True),
            X_test.reset_index(drop=True),
        ],
        axis=0,
        ignore_index=True,
    )
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
    if np.any(np.diff(time_values) < 0):
        raise ValueError("Historical feature input must be chronological.")

    keys_by_entity = {
        entity: build_combo_key(all_df, columns).to_numpy()
        for entity, columns in HISTORICAL_ENTITY_GROUPS.items()
    }
    feature_names = historical_feature_names()
    feature_arrays = _initialize_feature_arrays(len(all_df), feature_names)
    states = {entity: {} for entity in HISTORICAL_ENTITY_GROUPS}

    start = 0
    n_rows = len(all_df)
    duplicate_timestamp_rows = int(pd.Series(time_values).duplicated(keep=False).sum())
    while start < n_rows:
        current_time = time_values[start]
        end = start + 1
        while end < n_rows and time_values[end] == current_time:
            end += 1

        for entity in HISTORICAL_ENTITY_GROUPS:
            _write_group_features(
                entity=entity,
                row_start=start,
                row_end=end,
                current_time=current_time,
                amount_values=amount_values,
                keys=keys_by_entity[entity],
                states=states[entity],
                feature_arrays=feature_arrays,
            )

        for entity in HISTORICAL_ENTITY_GROUPS:
            _update_group_state(
                row_start=start,
                row_end=end,
                current_time=current_time,
                amount_values=amount_values,
                keys=keys_by_entity[entity],
                states=states[entity],
            )

        start = end

    features_all = pd.DataFrame(feature_arrays, columns=feature_names)
    train_end = split_lengths["train"]
    valid_end = train_end + split_lengths["validation"]
    X_train_hist = features_all.iloc[:train_end].reset_index(drop=True)
    X_valid_hist = features_all.iloc[train_end:valid_end].reset_index(drop=True)
    X_test_hist = features_all.iloc[valid_end:].reset_index(drop=True)

    summary = historical_feature_summary(
        X_train_hist,
        X_valid_hist,
        X_test_hist,
        duplicate_timestamp_rows=duplicate_timestamp_rows,
    )
    return X_train_hist, X_valid_hist, X_test_hist, summary


def validate_historical_velocity_features(
    X_train_hist: pd.DataFrame,
    X_valid_hist: pd.DataFrame,
    X_test_hist: pd.DataFrame,
) -> dict[str, object]:
    """Validate historical feature invariants that protect against leakage."""
    expected_features = historical_feature_names()
    checks: dict[str, object] = {
        "expected_feature_count": len(expected_features),
        "expected_features_present": True,
        "feature_columns_aligned": True,
        "no_infinite_values": True,
        "positive_time_since_prev": True,
        "window_counts_monotonic": True,
        "same_timestamp_fixture_passed": True,
    }

    for split_name, X in (
        ("train", X_train_hist),
        ("validation", X_valid_hist),
        ("test", X_test_hist),
    ):
        if X.columns.tolist() != expected_features:
            checks["expected_features_present"] = False
            raise ValueError(f"{split_name} historical feature columns do not match.")

        values = X.to_numpy(dtype="float64", copy=False)
        if np.isinf(values).any():
            checks["no_infinite_values"] = False
            raise ValueError(f"{split_name} historical features contain infinite values.")

        for column in _time_since_columns():
            valid_values = X[column].dropna()
            if (valid_values <= 0).any():
                checks["positive_time_since_prev"] = False
                raise ValueError(
                    f"{split_name} {column} contains non-positive prior time gaps."
                )

        for entity in WINDOW_ENTITY_GROUPS:
            one_hour = X[f"hist_count_last_1h_{entity}"]
            one_day = X[f"hist_count_last_24h_{entity}"]
            seven_days = X[f"hist_count_last_7d_{entity}"]
            count_before = X[f"hist_count_before_{entity}"]
            if not (
                (one_hour <= one_day).all()
                and (one_day <= seven_days).all()
                and (seven_days <= count_before).all()
            ):
                checks["window_counts_monotonic"] = False
                raise ValueError(
                    f"{split_name} window count ordering failed for {entity}."
                )

    run_historical_velocity_fixture_check()
    return checks


def run_historical_velocity_fixture_check() -> dict[str, object]:
    """Run a deterministic duplicate-timestamp and split-boundary fixture check."""
    columns = [
        TIME_COL,
        AMOUNT_COL,
        "card1",
        "addr1",
        "P_emaildomain",
        "DeviceInfo",
    ]
    train = pd.DataFrame(
        [
            [10, 100.0, "A", "X", "email.test", "device-a"],
            [10, 200.0, "A", "X", "email.test", "device-a"],
            [20, 150.0, "A", "X", "email.test", "device-a"],
        ],
        columns=columns,
    )
    valid = pd.DataFrame(
        [[30, 120.0, "A", "X", "email.test", "device-a"]],
        columns=columns,
    )
    test = pd.DataFrame(
        [[40, 180.0, "A", "X", "email.test", "device-a"]],
        columns=columns,
    )
    train_hist, valid_hist, test_hist, _ = generate_historical_velocity_features(
        train,
        valid,
        test,
    )

    if train_hist.loc[0, "hist_count_before_card1"] != 0:
        raise ValueError("Fixture failed: first row should have no history.")
    if train_hist.loc[1, "hist_count_before_card1"] != 0:
        raise ValueError("Fixture failed: same-timestamp row saw peer history.")
    if train_hist.loc[2, "hist_count_before_card1"] != 2:
        raise ValueError("Fixture failed: later train row should see both first rows.")
    if train_hist.loc[2, "hist_time_since_prev_card1"] != 10:
        raise ValueError("Fixture failed: prior time gap should exclude same timestamp.")
    if valid_hist.loc[0, "hist_count_before_card1"] != 3:
        raise ValueError("Fixture failed: validation should see earlier train history.")
    if test_hist.loc[0, "hist_count_before_card1"] != 4:
        raise ValueError("Fixture failed: test should see earlier train/validation history.")

    return {
        "same_timestamp_rows_excluded": True,
        "validation_uses_prior_train_history": True,
        "test_uses_prior_train_validation_history": True,
    }


def historical_feature_summary(
    X_train_hist: pd.DataFrame,
    X_valid_hist: pd.DataFrame,
    X_test_hist: pd.DataFrame,
    duplicate_timestamp_rows: int,
) -> dict[str, object]:
    """Build a compact JSON-safe summary of historical feature generation."""
    combined = pd.concat(
        [X_train_hist, X_valid_hist, X_test_hist],
        axis=0,
        ignore_index=True,
    )
    first_history_rates = {}
    for entity in HISTORICAL_ENTITY_GROUPS:
        column = f"hist_count_before_{entity}"
        first_history_rates[entity] = {
            "train": float((X_train_hist[column] == 0).mean()),
            "validation": float((X_valid_hist[column] == 0).mean()),
            "test": float((X_test_hist[column] == 0).mean()),
        }

    null_rates = {
        column: float(combined[column].isna().mean()) for column in combined.columns
    }
    return {
        "entity_groups": HISTORICAL_ENTITY_GROUPS,
        "window_entity_groups": sorted(WINDOW_ENTITY_GROUPS),
        "windows_seconds": WINDOWS_SECONDS,
        "feature_names": combined.columns.tolist(),
        "feature_count": int(combined.shape[1]),
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "same_timestamp_policy": (
            "Rows sharing TransactionDT are computed from the prior state first; "
            "the state is updated only after the full timestamp group is complete."
        ),
        "first_history_rates": first_history_rates,
        "null_rates": null_rates,
    }


def _validate_inputs(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    for split_name, X in (
        ("train", X_train),
        ("validation", X_valid),
        ("test", X_test),
    ):
        if TARGET_COL in X.columns:
            raise ValueError(
                f"{split_name} historical feature input must not contain {TARGET_COL}."
            )
        missing_columns = [
            column
            for column in _required_columns()
            if column not in X.columns
        ]
        if missing_columns:
            raise KeyError(
                f"{split_name} is missing historical input column(s): "
                + ", ".join(missing_columns)
            )


def _required_columns() -> list[str]:
    columns = [TIME_COL, AMOUNT_COL]
    for group_columns in HISTORICAL_ENTITY_GROUPS.values():
        columns.extend(group_columns)
    return sorted(set(columns))


def _initialize_feature_arrays(
    n_rows: int,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    feature_arrays = {
        feature_name: np.full(n_rows, np.nan, dtype="float32")
        for feature_name in feature_names
    }
    for entity in HISTORICAL_ENTITY_GROUPS:
        feature_arrays[f"hist_count_before_{entity}"] = np.zeros(
            n_rows,
            dtype="float32",
        )
        if entity in WINDOW_ENTITY_GROUPS:
            for window_name in WINDOWS_SECONDS:
                feature_arrays[f"hist_count_last_{window_name}_{entity}"] = np.zeros(
                    n_rows,
                    dtype="float32",
                )
    return feature_arrays


def _write_group_features(
    entity: str,
    row_start: int,
    row_end: int,
    current_time: float,
    amount_values: np.ndarray,
    keys: np.ndarray,
    states: dict[str, HistoricalState],
    feature_arrays: dict[str, np.ndarray],
) -> None:
    for row_index in range(row_start, row_end):
        key = keys[row_index]
        state = states.get(key)
        current_amount = amount_values[row_index]
        if state is None:
            continue

        feature_arrays[f"hist_count_before_{entity}"][row_index] = state.count
        if state.last_time is not None:
            feature_arrays[f"hist_time_since_prev_{entity}"][row_index] = (
                current_time - state.last_time
            )
        if np.isfinite(state.last_amount):
            feature_arrays[f"hist_amt_prev_{entity}"][row_index] = state.last_amount
            if np.isfinite(current_amount):
                feature_arrays[f"hist_amt_diff_prev_{entity}"][row_index] = (
                    current_amount - state.last_amount
                )
                if state.last_amount > 0:
                    feature_arrays[f"hist_amt_ratio_prev_{entity}"][row_index] = (
                        current_amount / state.last_amount
                    )
        if state.amount_count > 0:
            mean_before = state.amount_sum / state.amount_count
            feature_arrays[f"hist_amt_mean_before_{entity}"][row_index] = mean_before
            if state.amount_count > 1:
                variance = (
                    state.amount_sum_sq
                    - (state.amount_sum * state.amount_sum / state.amount_count)
                ) / (state.amount_count - 1)
                feature_arrays[f"hist_amt_std_before_{entity}"][row_index] = math.sqrt(
                    max(0.0, variance)
                )

        if entity in WINDOW_ENTITY_GROUPS:
            for window_name, window_seconds in WINDOWS_SECONDS.items():
                lower_bound = current_time - window_seconds
                start_index = bisect_left(state.times, lower_bound)
                feature_arrays[f"hist_count_last_{window_name}_{entity}"][
                    row_index
                ] = len(state.times) - start_index


def _update_group_state(
    row_start: int,
    row_end: int,
    current_time: float,
    amount_values: np.ndarray,
    keys: np.ndarray,
    states: dict[str, HistoricalState],
) -> None:
    for row_index in range(row_start, row_end):
        key = keys[row_index]
        state = states.get(key)
        if state is None:
            state = HistoricalState()
            states[key] = state

        current_amount = amount_values[row_index]
        state.count += 1
        state.last_time = current_time
        state.last_amount = current_amount if np.isfinite(current_amount) else math.nan
        if np.isfinite(current_amount):
            state.amount_count += 1
            state.amount_sum += float(current_amount)
            state.amount_sum_sq += float(current_amount * current_amount)
        state.times.append(current_time)


def _time_since_columns() -> list[str]:
    return [
        f"hist_time_since_prev_{entity}"
        for entity in HISTORICAL_ENTITY_GROUPS
    ]


if __name__ == "__main__":
    run_historical_velocity_fixture_check()
