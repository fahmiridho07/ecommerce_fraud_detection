"""Leakage-safe entity, time, and amount feature engineering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import ID_COL, TARGET_COL, TIME_COL
from preprocessing import MISSING_CATEGORY


AMOUNT_COL = "TransactionAmt"
KEY_SEPARATOR = "||"
MIN_AMOUNT_STAT_COUNT = 2
INTERNAL_KEY_PREFIX = "__fe_key_"

AMOUNT_FEATURES = [
    "TransactionAmt_log",
    "TransactionAmt_decimal",
    "TransactionAmt_cents",
    "TransactionAmt_is_round",
    "TransactionAmt_num_decimals",
]

TIME_FEATURES = [
    "transaction_day",
    "transaction_hour",
    "transaction_week",
    "transaction_dayofweek_proxy",
    "sin_hour",
    "cos_hour",
]

COUNT_FREQUENCY_GROUPS = {
    "card1": ["card1"],
    "card2": ["card2"],
    "card3": ["card3"],
    "card5": ["card5"],
    "addr1": ["addr1"],
    "addr2": ["addr2"],
    "P_emaildomain": ["P_emaildomain"],
    "R_emaildomain": ["R_emaildomain"],
    "DeviceInfo": ["DeviceInfo"],
    "ProductCD": ["ProductCD"],
    "card1_card2": ["card1", "card2"],
    "card1_addr1": ["card1", "addr1"],
    "card1_P_emaildomain": ["card1", "P_emaildomain"],
    "card1_addr1_P_emaildomain": ["card1", "addr1", "P_emaildomain"],
    "card1_DeviceInfo": ["card1", "DeviceInfo"],
    "ProductCD_card1": ["ProductCD", "card1"],
    "uid_card_addr": ["card1", "card2", "addr1"],
}

AMOUNT_STAT_GROUPS = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_P_emaildomain": ["card1", "P_emaildomain"],
    "P_emaildomain": ["P_emaildomain"],
    "ProductCD": ["ProductCD"],
    "addr1": ["addr1"],
}

UID_AMOUNT_STAT_GROUPS = {
    "uid_card_addr": ["card1", "card2", "addr1"],
    "uid_card_addr_email": ["card1", "addr1", "P_emaildomain"],
    "uid_card_product": ["card1", "ProductCD"],
    "uid_card_device": ["card1", "DeviceInfo"],
}

NUNIQUE_RELATIONSHIP_GROUPS = {
    "nunique_P_emaildomain_by_card1": {
        "group_column": "card1",
        "value_column": "P_emaildomain",
    },
    "nunique_addr1_by_card1": {
        "group_column": "card1",
        "value_column": "addr1",
    },
    "nunique_DeviceInfo_by_card1": {
        "group_column": "card1",
        "value_column": "DeviceInfo",
    },
    "nunique_card1_by_DeviceInfo": {
        "group_column": "DeviceInfo",
        "value_column": "card1",
    },
    "nunique_card1_by_P_emaildomain": {
        "group_column": "P_emaildomain",
        "value_column": "card1",
    },
    "nunique_addr1_by_P_emaildomain": {
        "group_column": "P_emaildomain",
        "value_column": "addr1",
    },
    "nunique_ProductCD_by_card1": {
        "group_column": "card1",
        "value_column": "ProductCD",
    },
}

UID_SKIPPED_BY_DESIGN_GROUPS = [
    {
        "group": "uid_card_email",
        "columns": ["card1", "P_emaildomain"],
        "reason": (
            "Represented by existing card1_P_emaildomain count/frequency and "
            "amount-stat features to avoid duplicate aliases."
        ),
    },
    {
        "group": "uid_card_addr_email_product",
        "columns": ["card1", "addr1", "P_emaildomain", "ProductCD"],
        "reason": (
            "Skipped in first controlled UID run because read-only profiling "
            "showed high cardinality and sparse train support."
        ),
    },
    {
        "group": "uid_card_addr_device",
        "columns": ["card1", "addr1", "DeviceInfo"],
        "reason": (
            "Skipped in first controlled UID run because DeviceInfo/address "
            "combinations are sparse and risk overfitting."
        ),
    },
]

AMOUNT_STAT_FEATURE_TEMPLATES = [
    "amt_mean_by_{group}",
    "amt_median_by_{group}",
    "amt_std_by_{group}",
    "amt_to_mean_by_{group}",
    "amt_diff_mean_by_{group}",
    "amt_to_median_by_{group}",
    "amt_zscore_by_{group}",
]


def validate_raw_feature_input(X: pd.DataFrame) -> None:
    """Validate that feature engineering sees feature-only raw matrices."""
    forbidden_columns = [column for column in (ID_COL, TARGET_COL) if column in X.columns]
    if forbidden_columns:
        raise ValueError(
            "Feature engineering input must not include target or ID columns: "
            + ", ".join(forbidden_columns)
        )

    missing_required = [
        column for column in (TIME_COL, AMOUNT_COL) if column not in X.columns
    ]
    if missing_required:
        raise KeyError(
            "Feature engineering input is missing required column(s): "
            + ", ".join(missing_required)
        )


def build_combo_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Build a stable string key for one or more entity columns."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError("Missing key column(s): " + ", ".join(missing_columns))
    if not columns:
        raise ValueError("At least one column is required to build a combo key.")

    key = _normalize_key_part(df[columns[0]])
    for column in columns[1:]:
        key = key.str.cat(_normalize_key_part(df[column]), sep=KEY_SEPARATOR)
    return key


def fit_count_frequency_mappings(
    X_train: pd.DataFrame,
    group_specs: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    """Fit count and frequency mappings using training rows only."""
    group_specs = group_specs or COUNT_FREQUENCY_GROUPS
    mappings: dict[str, dict[str, Any]] = {}
    skipped_groups: list[dict[str, object]] = []
    train_row_count = int(len(X_train))

    for group_name, columns in group_specs.items():
        missing_columns = [column for column in columns if column not in X_train.columns]
        if missing_columns:
            skipped_groups.append(
                {
                    "group": group_name,
                    "columns": columns,
                    "missing_columns": missing_columns,
                }
            )
            continue

        key = build_combo_key(X_train, columns)
        counts = key.value_counts(dropna=False).astype("int32")
        frequencies = (counts / train_row_count).astype("float32")
        mappings[group_name] = {
            "columns": columns,
            "counts": counts,
            "frequencies": frequencies,
            "mapping_size": int(len(counts)),
            "train_row_count": train_row_count,
        }

    return mappings, skipped_groups


def fit_amount_stat_mappings(
    X_train: pd.DataFrame,
    group_specs: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    """Fit train-only TransactionAmt statistics for entity groups."""
    group_specs = group_specs or AMOUNT_STAT_GROUPS
    mappings: dict[str, dict[str, Any]] = {}
    skipped_groups: list[dict[str, object]] = []
    train_row_count = int(len(X_train))
    amount = _amount_series(X_train)

    for group_name, columns in group_specs.items():
        missing_columns = [column for column in columns if column not in X_train.columns]
        if missing_columns:
            skipped_groups.append(
                {
                    "group": group_name,
                    "columns": columns,
                    "missing_columns": missing_columns,
                }
            )
            continue

        key = build_combo_key(X_train, columns)
        stats_source = pd.DataFrame({"key": key, AMOUNT_COL: amount})
        stats = stats_source.groupby("key", dropna=False)[AMOUNT_COL].agg(
            ["count", "mean", "median", "std"]
        )
        stats["count"] = stats["count"].astype("int32")
        for column in ("mean", "median", "std"):
            stats[column] = stats[column].astype("float32")
        mappings[group_name] = {
            "columns": columns,
            "stats": stats,
            "mapping_size": int(len(stats)),
            "train_row_count": train_row_count,
            "min_stat_count": MIN_AMOUNT_STAT_COUNT,
        }

    return mappings, skipped_groups


def fit_nunique_mappings(
    X_train: pd.DataFrame,
    relationship_specs: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    """Fit train-only nunique relationship mappings for entity columns."""
    relationship_specs = relationship_specs or NUNIQUE_RELATIONSHIP_GROUPS
    mappings: dict[str, dict[str, Any]] = {}
    skipped_groups: list[dict[str, object]] = []
    train_row_count = int(len(X_train))

    for feature_name, spec in relationship_specs.items():
        group_column = spec["group_column"]
        value_column = spec["value_column"]
        missing_columns = [
            column
            for column in (group_column, value_column)
            if column not in X_train.columns
        ]
        if missing_columns:
            skipped_groups.append(
                {
                    "feature": feature_name,
                    "group_column": group_column,
                    "value_column": value_column,
                    "missing_columns": missing_columns,
                }
            )
            continue

        keys = _normalize_key_part(X_train[group_column])
        values = _normalize_key_part(X_train[value_column])
        nunique = (
            pd.DataFrame({"key": keys, "value": values})
            .groupby("key", dropna=False)["value"]
            .nunique(dropna=False)
            .astype("int32")
        )
        mappings[feature_name] = {
            "group_column": group_column,
            "value_column": value_column,
            "nunique": nunique,
            "mapping_size": int(len(nunique)),
            "train_row_count": train_row_count,
        }

    return mappings, skipped_groups


def fit_entity_time_amount_features(X_train: pd.DataFrame) -> dict[str, Any]:
    """Fit all train-only artifacts needed by the engineered feature set."""
    validate_raw_feature_input(X_train)
    amount = _amount_series(X_train)
    global_stats = _global_amount_stats(amount)
    count_mappings, skipped_count_groups = fit_count_frequency_mappings(X_train)
    amount_mappings, skipped_amount_groups = fit_amount_stat_mappings(X_train)

    count_feature_names = [
        feature_name
        for group_name in count_mappings
        for feature_name in (f"count_{group_name}", f"freq_{group_name}")
    ]
    amount_stat_feature_names = [
        template.format(group=group_name)
        for group_name in amount_mappings
        for template in AMOUNT_STAT_FEATURE_TEMPLATES
    ]
    engineered_feature_names = (
        AMOUNT_FEATURES
        + TIME_FEATURES
        + count_feature_names
        + amount_stat_feature_names
    )

    artifacts = {
        "source_feature_columns": X_train.columns.tolist(),
        "fit_row_count": int(len(X_train)),
        "train_time_min": _safe_int_min(X_train[TIME_COL]),
        "train_time_max": _safe_int_max(X_train[TIME_COL]),
        "required_columns": [TIME_COL, AMOUNT_COL],
        "amount_features": AMOUNT_FEATURES,
        "time_features": TIME_FEATURES,
        "count_frequency_mappings": count_mappings,
        "amount_stat_mappings": amount_mappings,
        "skipped_count_frequency_groups": skipped_count_groups,
        "skipped_amount_stat_groups": skipped_amount_groups,
        "global_amount_stats": global_stats,
        "engineered_feature_names": engineered_feature_names,
        "internal_key_prefix": INTERNAL_KEY_PREFIX,
        "internal_combo_key_columns_retained": False,
    }
    _validate_fitted_artifacts(artifacts)
    return artifacts


def fit_uid_entity_time_amount_features(X_train: pd.DataFrame) -> dict[str, Any]:
    """Fit base features plus controlled UID-inspired relationship features."""
    artifacts = fit_entity_time_amount_features(X_train)

    uid_amount_mappings, skipped_uid_amount_groups = fit_amount_stat_mappings(
        X_train,
        UID_AMOUNT_STAT_GROUPS,
    )
    artifacts["amount_stat_mappings"].update(uid_amount_mappings)
    artifacts["skipped_uid_amount_stat_groups"] = skipped_uid_amount_groups

    nunique_mappings, skipped_nunique_groups = fit_nunique_mappings(X_train)
    artifacts["nunique_mappings"] = nunique_mappings
    artifacts["skipped_nunique_groups"] = skipped_nunique_groups
    artifacts["skipped_uid_groups_by_design"] = UID_SKIPPED_BY_DESIGN_GROUPS
    artifacts["uid_alias_policy"] = {
        "uid_card_email": "Existing card1_P_emaildomain features are reused.",
        "uid_card_product": "Existing ProductCD_card1 count/frequency features are reused.",
        "uid_card_device": "Existing card1_DeviceInfo count/frequency features are reused.",
        "uid_card_addr_email": (
            "Existing card1_addr1_P_emaildomain count/frequency features are reused."
        ),
    }

    amount_stat_feature_names = [
        template.format(group=group_name)
        for group_name in uid_amount_mappings
        for template in AMOUNT_STAT_FEATURE_TEMPLATES
    ]
    nunique_feature_names = list(nunique_mappings)
    artifacts["base_engineered_feature_names"] = list(
        artifacts["engineered_feature_names"]
    )
    artifacts["uid_engineered_feature_names"] = (
        amount_stat_feature_names + nunique_feature_names
    )
    artifacts["engineered_feature_names"] = (
        artifacts["engineered_feature_names"]
        + artifacts["uid_engineered_feature_names"]
    )
    artifacts["experiment_variant"] = "uid_entity_time_amount"
    _validate_fitted_artifacts(artifacts)
    return artifacts


def apply_entity_time_amount_features(
    X: pd.DataFrame,
    artifacts: dict[str, Any],
) -> pd.DataFrame:
    """Apply train-fitted entity/time/amount feature artifacts to one split."""
    validate_raw_feature_input(X)
    source_columns = artifacts["source_feature_columns"]
    missing_source = [column for column in source_columns if column not in X.columns]
    if missing_source:
        raise KeyError(
            "Input split is missing source feature column(s): "
            + ", ".join(missing_source)
        )

    transformed = X.loc[:, source_columns].copy()
    _add_amount_features(transformed)
    _add_time_features(transformed)
    _add_count_frequency_features(transformed, artifacts)
    _add_amount_stat_features(transformed, artifacts)
    _add_nunique_features(transformed, artifacts)
    return transformed


def validate_engineered_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    artifacts: dict[str, Any],
) -> None:
    """Validate split alignment and leakage-sensitive engineered columns."""
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation engineered columns do not align with train.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test engineered columns do not align with train.")

    engineered_feature_names = artifacts["engineered_feature_names"]
    for split_name, X in (
        ("train", X_train),
        ("validation", X_valid),
        ("test", X_test),
    ):
        forbidden_columns = [
            column for column in (ID_COL, TARGET_COL) if column in X.columns
        ]
        if forbidden_columns:
            raise ValueError(
                f"{split_name} engineered matrix contains forbidden column(s): "
                + ", ".join(forbidden_columns)
            )

        missing_engineered = [
            column for column in engineered_feature_names if column not in X.columns
        ]
        if missing_engineered:
            raise ValueError(
                f"{split_name} is missing engineered feature(s): "
                + ", ".join(missing_engineered[:20])
            )

        internal_columns = [
            column
            for column in X.columns
            if str(column).startswith(artifacts["internal_key_prefix"])
            or str(column).startswith("uid_")
        ]
        if internal_columns:
            raise ValueError(
                f"{split_name} retained internal combo key column(s): "
                + ", ".join(internal_columns)
            )

        for column in engineered_feature_names:
            values = pd.to_numeric(X[column], errors="coerce").to_numpy(
                dtype="float64",
                copy=False,
            )
            if np.isinf(values).any():
                raise ValueError(
                    f"{split_name} engineered feature contains infinite values: "
                    f"{column}"
                )

    _validate_fitted_artifacts(artifacts)


def unknown_rate_summary(
    X: pd.DataFrame,
    artifacts: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Summarize unseen validation/test entity keys against train-fitted maps."""
    validate_raw_feature_input(X)
    summary = {
        "count_frequency": {},
        "amount_stats": {},
        "nunique": {},
    }

    for group_name, mapping in artifacts["count_frequency_mappings"].items():
        key = build_combo_key(X, mapping["columns"])
        summary["count_frequency"][group_name] = float(
            (~key.isin(mapping["counts"].index)).mean()
        )

    for group_name, mapping in artifacts["amount_stat_mappings"].items():
        key = build_combo_key(X, mapping["columns"])
        summary["amount_stats"][group_name] = float(
            (~key.isin(mapping["stats"].index)).mean()
        )

    for feature_name, mapping in artifacts.get("nunique_mappings", {}).items():
        key = _normalize_key_part(X[mapping["group_column"]])
        summary["nunique"][feature_name] = float(
            (~key.isin(mapping["nunique"].index)).mean()
        )

    return summary


def feature_engineering_summary(artifacts: dict[str, Any]) -> dict[str, object]:
    """Build a JSON-safe summary of feature-engineering artifacts."""
    count_groups = []
    for group_name, mapping in artifacts["count_frequency_mappings"].items():
        count_groups.append(
            {
                "group": group_name,
                "columns": mapping["columns"],
                "count_feature": f"count_{group_name}",
                "frequency_feature": f"freq_{group_name}",
                "mapping_size": int(mapping["mapping_size"]),
                "train_row_count": int(mapping["train_row_count"]),
            }
        )

    amount_stat_groups = []
    for group_name, mapping in artifacts["amount_stat_mappings"].items():
        amount_stat_groups.append(
            {
                "group": group_name,
                "columns": mapping["columns"],
                "features": [
                    template.format(group=group_name)
                    for template in AMOUNT_STAT_FEATURE_TEMPLATES
                ],
                "mapping_size": int(mapping["mapping_size"]),
                "train_row_count": int(mapping["train_row_count"]),
                "min_stat_count": int(mapping["min_stat_count"]),
            }
        )

    nunique_relationship_groups = []
    for feature_name, mapping in artifacts.get("nunique_mappings", {}).items():
        nunique_relationship_groups.append(
            {
                "feature": feature_name,
                "group_column": mapping["group_column"],
                "value_column": mapping["value_column"],
                "mapping_size": int(mapping["mapping_size"]),
                "train_row_count": int(mapping["train_row_count"]),
            }
        )

    return {
        "fit_row_count": int(artifacts["fit_row_count"]),
        "train_time_min": artifacts["train_time_min"],
        "train_time_max": artifacts["train_time_max"],
        "required_columns": artifacts["required_columns"],
        "amount_features": artifacts["amount_features"],
        "time_features": artifacts["time_features"],
        "count_frequency_groups": count_groups,
        "amount_stat_groups": amount_stat_groups,
        "nunique_relationship_groups": nunique_relationship_groups,
        "skipped_count_frequency_groups": artifacts["skipped_count_frequency_groups"],
        "skipped_amount_stat_groups": artifacts["skipped_amount_stat_groups"],
        "skipped_uid_amount_stat_groups": artifacts.get(
            "skipped_uid_amount_stat_groups",
            [],
        ),
        "skipped_nunique_groups": artifacts.get("skipped_nunique_groups", []),
        "skipped_uid_groups_by_design": artifacts.get(
            "skipped_uid_groups_by_design",
            [],
        ),
        "global_amount_stats": artifacts["global_amount_stats"],
        "engineered_feature_names": artifacts["engineered_feature_names"],
        "engineered_feature_count": int(len(artifacts["engineered_feature_names"])),
        "base_engineered_feature_count": len(
            artifacts.get(
                "base_engineered_feature_names",
                artifacts["engineered_feature_names"],
            )
        ),
        "uid_engineered_feature_names": artifacts.get(
            "uid_engineered_feature_names",
            [],
        ),
        "uid_engineered_feature_count": len(
            artifacts.get("uid_engineered_feature_names", [])
        ),
        "internal_combo_key_columns_retained": bool(
            artifacts["internal_combo_key_columns_retained"]
        ),
        "uid_alias_policy": artifacts.get("uid_alias_policy", {}),
        "uid_duplicate_policy": (
            "Duplicate count/frequency aliases are not emitted; existing "
            "card1_P_emaildomain, card1_addr1_P_emaildomain, ProductCD_card1, "
            "and card1_DeviceInfo features are reused where applicable."
        ),
    }


def _normalize_key_part(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CATEGORY)


def _amount_series(X: pd.DataFrame) -> pd.Series:
    amount = pd.to_numeric(X[AMOUNT_COL], errors="coerce")
    if amount.notna().sum() == 0:
        raise ValueError(f"{AMOUNT_COL} has no non-missing numeric values.")
    return amount


def _global_amount_stats(amount: pd.Series) -> dict[str, float]:
    std = float(amount.std())
    if not np.isfinite(std) or std <= 0:
        std = 1.0
    return {
        "mean": float(amount.mean()),
        "median": float(amount.median()),
        "std": std,
    }


def _safe_int_min(series: pd.Series) -> int | None:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return None
    return int(values.min())


def _safe_int_max(series: pd.Series) -> int | None:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return None
    return int(values.max())


def _decimal_place_counts(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[0], np.nan, dtype="float32")
    valid = np.isfinite(values)
    unresolved = valid.copy()
    for decimals in range(0, 7):
        scale = 10**decimals
        scaled = values * scale
        hits = unresolved & np.isclose(
            scaled,
            np.rint(scaled),
            rtol=0.0,
            atol=1e-6,
        )
        result[hits] = float(decimals)
        unresolved[hits] = False
    result[unresolved] = 6.0
    return result


def _add_amount_features(X: pd.DataFrame) -> None:
    amount = _amount_series(X)
    values = amount.to_numpy(dtype="float64", copy=False)
    valid = np.isfinite(values)

    decimal = values - np.floor(values)
    decimal[~valid] = np.nan

    cents = np.full(values.shape[0], np.nan, dtype="float64")
    cents[valid] = np.mod(np.rint(values[valid] * 100.0), 100.0)

    is_round = np.full(values.shape[0], np.nan, dtype="float64")
    is_round[valid] = np.isclose(
        values[valid],
        np.rint(values[valid]),
        rtol=0.0,
        atol=1e-6,
    ).astype("float64")

    X["TransactionAmt_log"] = np.log1p(amount).astype("float32")
    X["TransactionAmt_decimal"] = decimal.astype("float32")
    X["TransactionAmt_cents"] = cents.astype("float32")
    X["TransactionAmt_is_round"] = is_round.astype("float32")
    X["TransactionAmt_num_decimals"] = _decimal_place_counts(values)


def _add_time_features(X: pd.DataFrame) -> None:
    transaction_dt = pd.to_numeric(X[TIME_COL], errors="coerce")
    transaction_day = np.floor(transaction_dt / 86400.0)
    transaction_hour = np.mod(np.floor(transaction_dt / 3600.0), 24.0)
    transaction_week = np.floor(transaction_day / 7.0)
    transaction_dayofweek_proxy = np.mod(transaction_day, 7.0)

    X["transaction_day"] = transaction_day.astype("float32")
    X["transaction_hour"] = transaction_hour.astype("float32")
    X["transaction_week"] = transaction_week.astype("float32")
    X["transaction_dayofweek_proxy"] = transaction_dayofweek_proxy.astype("float32")
    X["sin_hour"] = np.sin(2.0 * np.pi * transaction_hour / 24.0).astype("float32")
    X["cos_hour"] = np.cos(2.0 * np.pi * transaction_hour / 24.0).astype("float32")


def _add_count_frequency_features(
    X: pd.DataFrame,
    artifacts: dict[str, Any],
) -> None:
    for group_name, mapping in artifacts["count_frequency_mappings"].items():
        key = build_combo_key(X, mapping["columns"])
        X[f"count_{group_name}"] = (
            key.map(mapping["counts"]).fillna(0).astype("int32")
        )
        X[f"freq_{group_name}"] = (
            key.map(mapping["frequencies"]).fillna(0.0).astype("float32")
        )


def _add_amount_stat_features(
    X: pd.DataFrame,
    artifacts: dict[str, Any],
) -> None:
    amount = _amount_series(X)
    global_stats = artifacts["global_amount_stats"]
    global_mean = float(global_stats["mean"])
    global_median = float(global_stats["median"])
    global_std = float(global_stats["std"])
    if not np.isfinite(global_std) or global_std <= 0:
        global_std = 1.0

    for group_name, mapping in artifacts["amount_stat_mappings"].items():
        key = build_combo_key(X, mapping["columns"])
        stats = mapping["stats"]
        group_count = key.map(stats["count"]).fillna(0).astype("int32")
        stable_group = group_count >= int(mapping["min_stat_count"])

        mean = key.map(stats["mean"]).astype("float64")
        median = key.map(stats["median"]).astype("float64")
        std = key.map(stats["std"]).astype("float64")

        mean = mean.where(stable_group & mean.notna(), global_mean)
        median = median.where(stable_group & median.notna(), global_median)
        std = std.where(stable_group & std.notna() & (std > 0), global_std)

        mean_denominator = mean.where(mean != 0, np.nan)
        median_denominator = median.where(median != 0, np.nan)
        std_denominator = std.where(std > 0, global_std)

        X[f"amt_mean_by_{group_name}"] = mean.astype("float32")
        X[f"amt_median_by_{group_name}"] = median.astype("float32")
        X[f"amt_std_by_{group_name}"] = std.astype("float32")
        X[f"amt_to_mean_by_{group_name}"] = (amount / mean_denominator).astype(
            "float32"
        )
        X[f"amt_diff_mean_by_{group_name}"] = (amount - mean).astype("float32")
        X[f"amt_to_median_by_{group_name}"] = (
            amount / median_denominator
        ).astype("float32")
        X[f"amt_zscore_by_{group_name}"] = (
            (amount - mean) / std_denominator
        ).astype("float32")


def _add_nunique_features(
    X: pd.DataFrame,
    artifacts: dict[str, Any],
) -> None:
    for feature_name, mapping in artifacts.get("nunique_mappings", {}).items():
        key = _normalize_key_part(X[mapping["group_column"]])
        X[feature_name] = (
            key.map(mapping["nunique"]).fillna(0).astype("int32")
        )


def _validate_fitted_artifacts(artifacts: dict[str, Any]) -> None:
    fit_row_count = int(artifacts["fit_row_count"])
    for group_name, mapping in artifacts["count_frequency_mappings"].items():
        if int(mapping["train_row_count"]) != fit_row_count:
            raise ValueError(
                f"Count/frequency mapping row count mismatch for {group_name}."
            )
        if int(mapping["mapping_size"]) != len(mapping["counts"]):
            raise ValueError(
                f"Count/frequency mapping size mismatch for {group_name}."
            )

    for group_name, mapping in artifacts["amount_stat_mappings"].items():
        if int(mapping["train_row_count"]) != fit_row_count:
            raise ValueError(f"Amount-stat mapping row count mismatch for {group_name}.")
        if int(mapping["mapping_size"]) != len(mapping["stats"]):
            raise ValueError(f"Amount-stat mapping size mismatch for {group_name}.")

    for feature_name, mapping in artifacts.get("nunique_mappings", {}).items():
        if int(mapping["train_row_count"]) != fit_row_count:
            raise ValueError(f"Nunique mapping row count mismatch for {feature_name}.")
        if int(mapping["mapping_size"]) != len(mapping["nunique"]):
            raise ValueError(f"Nunique mapping size mismatch for {feature_name}.")
