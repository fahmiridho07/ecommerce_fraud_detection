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


def _as_string(series: pd.Series) -> pd.Series:
    return series.astype("string")


def normalize_browser_family(value: object) -> str:
    text = str(value).strip().lower()
    if text in ("", "<na>", "nan", "none"):
        return MISSING_CATEGORY
    if "chrome" in text or "chromium" in text:
        return "chrome"
    if "firefox" in text:
        return "firefox"
    if "safari" in text:
        return "safari"
    if "edge" in text:
        return "edge"
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


def fit_enhanced_preprocessing(
    X_train: pd.DataFrame,
    rare_min_count: int = DEFAULT_RARE_MIN_COUNT,
) -> dict[str, object]:
    transformed = add_normalized_identity_device_features(X_train)
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
    }


def apply_enhanced_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    X = X.loc[:, preprocessing["feature_columns_raw"]].copy()
    transformed = add_normalized_identity_device_features(X)
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
