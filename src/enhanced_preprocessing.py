"""Enhanced preprocessing ablations for IEEE-CIS categorical drift.

These helpers are intentionally separate from `preprocessing.py` so the
canonical P01-P04 pipeline remains unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from preprocessing import (
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_VALUE,
    get_categorical_columns,
)


RARE_CATEGORY = "__RARE__"
DEVICEINFO_COLUMN = "DeviceInfo"
ID30_COLUMN = "id_30"
ID31_COLUMN = "id_31"
ID33_COLUMN = "id_33"
DEFAULT_RARE_MIN_COUNT = 50
DEFAULT_FREQUENCY_COLUMNS = (
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "id_31_browser_family",
    "id_30_os_family",
    "DeviceInfo_family",
    "id_33_size_bucket",
)
MISSINGNESS_GROUP_PREFIXES = {
    "v": "V",
    "d": "D",
    "m": "M",
    "id": "id_",
}


def _as_string(series: pd.Series) -> pd.Series:
    return series.astype("string")


def normalize_browser_family(value: object) -> str:
    text = str(value).strip().lower()
    if text in ("", "<na>", "nan", "none"):
        return MISSING_CATEGORY
    if "edge" in text:
        return "edge"
    if "chrome" in text or "chromium" in text:
        return "chrome"
    if "firefox" in text:
        return "firefox"
    if "safari" in text:
        return "safari"
    if "samsung" in text:
        return "samsung"
    if "opera" in text:
        return "opera"
    if "ie" in text or "internet explorer" in text:
        return "ie"
    return "other"


def extract_major_version(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)", str(value))
    return float(match.group(1)) if match else np.nan


def normalize_os_family(value: object) -> str:
    text = str(value).strip().lower()
    if text in ("", "<na>", "nan", "none"):
        return MISSING_CATEGORY
    if "windows" in text:
        return "windows"
    if "ios" in text:
        return "ios"
    if "mac" in text:
        return "mac"
    if "android" in text:
        return "android"
    if "linux" in text:
        return "linux"
    return "other"


def normalize_device_family(value: object) -> str:
    text = str(value).strip().lower()
    if text in ("", "<na>", "nan", "none"):
        return MISSING_CATEGORY
    if "iphone" in text:
        return "iphone"
    if "ipad" in text:
        return "ipad"
    if "mac" in text:
        return "mac"
    if "windows" in text:
        return "windows"
    if "sm-" in text or "samsung" in text:
        return "samsung"
    if "moto" in text or "xt" in text:
        return "motorola"
    if "huawei" in text:
        return "huawei"
    if "lg-" in text or text.startswith("lg"):
        return "lg"
    if "rv:" in text or "linux" in text:
        return "linux_device"
    return "other"


def parse_screen_dimension(value: object) -> tuple[float, float, str]:
    if pd.isna(value):
        return np.nan, np.nan, MISSING_CATEGORY
    match = re.search(r"(\d+)\s*x\s*(\d+)", str(value).lower())
    if not match:
        return np.nan, np.nan, "other"
    width = float(match.group(1))
    height = float(match.group(2))
    short_side = min(width, height)
    long_side = max(width, height)
    if long_side >= 2500:
        bucket = "very_large"
    elif long_side >= 1900:
        bucket = "large"
    elif long_side >= 1300:
        bucket = "medium"
    elif long_side > 0:
        bucket = "small"
    else:
        bucket = "other"
    if short_side > 0 and long_side / short_side > 1.8:
        bucket = f"{bucket}_wide"
    return width, height, bucket


def add_normalized_identity_device_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add lower-cardinality identity/device fields and drop raw drift-prone strings."""
    transformed = X.copy()
    dropped_columns: list[str] = []

    if ID31_COLUMN in transformed.columns:
        transformed["id_31_browser_family"] = transformed[ID31_COLUMN].map(
            normalize_browser_family
        )
        transformed["id_31_major_version"] = transformed[ID31_COLUMN].map(
            extract_major_version
        )
        dropped_columns.append(ID31_COLUMN)

    if ID30_COLUMN in transformed.columns:
        transformed["id_30_os_family"] = transformed[ID30_COLUMN].map(
            normalize_os_family
        )
        transformed["id_30_major_version"] = transformed[ID30_COLUMN].map(
            extract_major_version
        )
        dropped_columns.append(ID30_COLUMN)

    if DEVICEINFO_COLUMN in transformed.columns:
        transformed["DeviceInfo_family"] = transformed[DEVICEINFO_COLUMN].map(
            normalize_device_family
        )
        dropped_columns.append(DEVICEINFO_COLUMN)

    if ID33_COLUMN in transformed.columns:
        parsed = transformed[ID33_COLUMN].map(parse_screen_dimension)
        transformed["id_33_width"] = parsed.map(lambda item: item[0])
        transformed["id_33_height"] = parsed.map(lambda item: item[1])
        transformed["id_33_area"] = transformed["id_33_width"] * transformed["id_33_height"]
        transformed["id_33_aspect_ratio"] = transformed["id_33_width"] / transformed[
            "id_33_height"
        ].replace(0, np.nan)
        transformed["id_33_size_bucket"] = parsed.map(lambda item: item[2])
        dropped_columns.append(ID33_COLUMN)

    return transformed.drop(columns=dropped_columns, errors="ignore")


def add_time_amount_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add simple temporal and amount transforms without fitting on holdout data."""
    transformed = X.copy()
    if "TransactionDT" in transformed.columns:
        transaction_day = np.floor(transformed["TransactionDT"] / 86_400.0)
        transformed["TransactionDT_day"] = transaction_day
        transformed["TransactionDT_week"] = np.floor(transaction_day / 7.0)
        transformed["TransactionDT_day_of_week"] = np.mod(transaction_day, 7.0)
        transformed["TransactionDT_hour_of_day"] = np.floor(
            np.mod(transformed["TransactionDT"], 86_400.0) / 3_600.0
        )
    if "TransactionAmt" in transformed.columns:
        amount = transformed["TransactionAmt"]
        transformed["TransactionAmt_log1p"] = np.log1p(amount)
        cents = np.mod(np.round(amount * 100), 100)
        transformed["TransactionAmt_cents"] = cents
    return transformed


def add_missingness_summary_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add compact missingness summaries by IEEE-CIS feature family."""
    transformed = X.copy()
    missing_counts: list[pd.Series] = []
    for group_name, prefix in MISSINGNESS_GROUP_PREFIXES.items():
        columns = [column for column in transformed.columns if column.startswith(prefix)]
        if not columns:
            continue
        missing = transformed[columns].isna()
        count = missing.sum(axis=1).astype("float32")
        transformed[f"missing_count_{group_name}"] = count
        transformed[f"missing_rate_{group_name}"] = (
            count / float(len(columns))
        ).astype("float32")
        missing_counts.append(count)

    email_columns = [
        column for column in ("P_emaildomain", "R_emaildomain") if column in transformed.columns
    ]
    if email_columns:
        missing = transformed[email_columns].isna()
        count = missing.sum(axis=1).astype("float32")
        transformed["missing_count_email"] = count
        transformed["missing_rate_email"] = (
            count / float(len(email_columns))
        ).astype("float32")
        missing_counts.append(count)

    distance_columns = [
        column for column in ("dist1", "dist2") if column in transformed.columns
    ]
    if distance_columns:
        missing = transformed[distance_columns].isna()
        count = missing.sum(axis=1).astype("float32")
        transformed["missing_count_dist"] = count
        transformed["missing_rate_dist"] = (
            count / float(len(distance_columns))
        ).astype("float32")
        missing_counts.append(count)

    if missing_counts:
        total_missing = sum(missing_counts)
        transformed["missing_count_selected_total"] = total_missing.astype("float32")

    id_columns = [column for column in transformed.columns if column.startswith("id_")]
    if id_columns:
        transformed["has_identity_observed"] = (
            transformed[id_columns].notna().any(axis=1).astype("int8")
        )
    return transformed


def _normalize_category_series(series: pd.Series) -> pd.Series:
    return _as_string(series).fillna(MISSING_CATEGORY)


def fit_rare_categories(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    min_count: int,
) -> dict[str, set[str]]:
    keepers: dict[str, set[str]] = {}
    for column in categorical_columns:
        values = _normalize_category_series(df[column])
        counts = values.value_counts(dropna=False)
        keepers[column] = set(counts[counts >= min_count].index.tolist())
        keepers[column].add(MISSING_CATEGORY)
    return keepers


def apply_rare_categories(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
    keepers: dict[str, set[str]],
) -> pd.DataFrame:
    transformed = df.copy()
    for column in categorical_columns:
        values = _normalize_category_series(transformed[column])
        allowed = keepers[column]
        transformed[column] = values.where(values.isin(allowed), RARE_CATEGORY)
    return transformed


def fit_categorical_mappings(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = _normalize_category_series(df[column])
        categories = sorted(set(values.tolist()) - {MISSING_CATEGORY})
        ordered = [MISSING_CATEGORY, *categories]
        mappings[column] = {category: index for index, category in enumerate(ordered)}
    return mappings


def transform_categorical_columns(
    df: pd.DataFrame,
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    transformed = df.copy()
    for column, mapping in mappings.items():
        values = _normalize_category_series(transformed[column])
        transformed[column] = values.map(mapping).fillna(UNKNOWN_CATEGORY_VALUE).astype(
            "int32"
        )
    return transformed


def fit_frequency_encoding_maps(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = _normalize_category_series(df[column])
        counts = values.value_counts(dropna=False)
        maps[column] = {
            str(category): float(count)
            for category, count in counts.items()
        }
    return maps


def add_frequency_encoded_features(
    df: pd.DataFrame,
    frequency_maps: dict[str, dict[str, float]],
    train_rows: int,
) -> pd.DataFrame:
    transformed = df.copy()
    denominator = float(train_rows) if train_rows else 1.0
    for column, mapping in frequency_maps.items():
        if column not in transformed.columns:
            continue
        values = _normalize_category_series(transformed[column]).astype(str)
        count_feature = f"{column}_train_count"
        frequency_feature = f"{column}_train_frequency"
        counts = values.map(mapping).fillna(0.0).astype("float32")
        transformed[count_feature] = counts
        transformed[frequency_feature] = (counts / denominator).astype("float32")
    return transformed


def fit_enhanced_preprocessing(
    X_train: pd.DataFrame,
    rare_min_count: int = DEFAULT_RARE_MIN_COUNT,
    enable_frequency_encoding: bool = False,
    enable_missingness_summary: bool = False,
    enable_time_amount_features: bool = False,
) -> dict[str, object]:
    transformed = X_train.copy()
    if enable_missingness_summary:
        transformed = add_missingness_summary_features(transformed)
    if enable_time_amount_features:
        transformed = add_time_amount_features(transformed)
    transformed = add_normalized_identity_device_features(transformed)
    frequency_maps = (
        fit_frequency_encoding_maps(transformed, DEFAULT_FREQUENCY_COLUMNS)
        if enable_frequency_encoding
        else {}
    )
    if enable_frequency_encoding:
        transformed = add_frequency_encoded_features(
            transformed,
            frequency_maps,
            train_rows=len(X_train),
        )
    categorical_columns = get_categorical_columns(transformed)
    rare_category_keepers = fit_rare_categories(
        transformed,
        categorical_columns,
        rare_min_count,
    )
    bucketed = apply_rare_categories(
        transformed,
        categorical_columns,
        rare_category_keepers,
    )
    categorical_mappings = fit_categorical_mappings(bucketed, categorical_columns)
    return {
        "feature_columns_raw": X_train.columns.tolist(),
        "feature_columns_transformed": bucketed.columns.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_mappings": categorical_mappings,
        "rare_category_keepers": {
            column: sorted(values) for column, values in rare_category_keepers.items()
        },
        "rare_min_count": rare_min_count,
        "enable_frequency_encoding": enable_frequency_encoding,
        "enable_missingness_summary": enable_missingness_summary,
        "enable_time_amount_features": enable_time_amount_features,
        "frequency_encoding_maps": frequency_maps,
        "frequency_encoding_columns": sorted(frequency_maps),
        "train_rows": int(len(X_train)),
        "missing_category": MISSING_CATEGORY,
        "rare_category": RARE_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "normalization": {
            "dropped_raw_columns": [
                column
                for column in (ID31_COLUMN, ID30_COLUMN, DEVICEINFO_COLUMN, ID33_COLUMN)
                if column in X_train.columns
            ],
            "added_columns": [
                column
                for column in (
                    "id_31_browser_family",
                    "id_31_major_version",
                    "id_30_os_family",
                    "id_30_major_version",
                    "DeviceInfo_family",
                    "id_33_width",
                    "id_33_height",
                    "id_33_area",
                    "id_33_aspect_ratio",
                    "id_33_size_bucket",
                )
                if column in bucketed.columns
            ],
        },
        "engineered_feature_groups": {
            "frequency_encoded_features": [
                feature
                for column in sorted(frequency_maps)
                for feature in (
                    f"{column}_train_count",
                    f"{column}_train_frequency",
                )
            ],
            "missingness_summary_features": [
                column
                for column in bucketed.columns
                if column.startswith("missing_") or column == "has_identity_observed"
            ],
            "time_amount_features": [
                column
                for column in (
                    "TransactionDT_day",
                    "TransactionDT_week",
                    "TransactionDT_day_of_week",
                    "TransactionDT_hour_of_day",
                    "TransactionAmt_log1p",
                    "TransactionAmt_cents",
                )
                if column in bucketed.columns
            ],
        },
    }


def apply_enhanced_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    X = X.loc[:, preprocessing["feature_columns_raw"]].copy()
    transformed = X.copy()
    if preprocessing.get("enable_missingness_summary", False):
        transformed = add_missingness_summary_features(transformed)
    if preprocessing.get("enable_time_amount_features", False):
        transformed = add_time_amount_features(transformed)
    transformed = add_normalized_identity_device_features(transformed)
    if preprocessing.get("enable_frequency_encoding", False):
        transformed = add_frequency_encoded_features(
            transformed,
            preprocessing["frequency_encoding_maps"],
            train_rows=int(preprocessing["train_rows"]),
        )
    transformed = transformed.loc[:, preprocessing["feature_columns_transformed"]].copy()
    keepers = {
        column: set(values)
        for column, values in preprocessing["rare_category_keepers"].items()
    }
    transformed = apply_rare_categories(
        transformed,
        preprocessing["categorical_columns"],
        keepers,
    )
    return transform_categorical_columns(
        transformed,
        preprocessing["categorical_mappings"],
    )
