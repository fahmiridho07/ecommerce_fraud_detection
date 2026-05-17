"""Preprocessing helpers shared by experiment scripts."""

from __future__ import annotations

import re

import pandas as pd

from config import ID_COL, TARGET_COL, V_FEATURE_PATTERN


MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY_VALUE = -1


def get_v_feature_columns(df: pd.DataFrame, pattern: str = V_FEATURE_PATTERN) -> list[str]:
    """Return Vesta engineered numerical V-feature columns."""
    regex = re.compile(pattern)
    return [column for column in df.columns if regex.match(column)]


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target and drop non-feature ID column."""
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if id_col not in df.columns:
        raise KeyError(f"Missing ID column: {id_col}")

    y = df[target_col].astype(int).copy()
    X = df.drop(columns=[target_col, id_col]).copy()
    return X, y


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Detect categorical columns from object and category dtypes."""
    categorical_dtypes = ["object", "category"]
    return df.select_dtypes(include=categorical_dtypes).columns.tolist()


def _normalize_category_series(series: pd.Series) -> pd.Series:
    """Convert categorical values to strings and mark missing values explicitly."""
    return series.astype("string").fillna(MISSING_CATEGORY)


def fit_categorical_mappings(
    df: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, dict[str, int]]:
    """Fit category-to-integer mappings using training data only."""
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = _normalize_category_series(df[column])
        categories = sorted(set(values.dropna().tolist()) - {MISSING_CATEGORY})
        ordered_categories = [MISSING_CATEGORY, *categories]
        mappings[column] = {
            category: index for index, category in enumerate(ordered_categories)
        }
    return mappings


def transform_categorical_columns(
    df: pd.DataFrame,
    mappings: dict[str, dict[str, int]],
    unknown_value: int = UNKNOWN_CATEGORY_VALUE,
) -> pd.DataFrame:
    """Apply train-fitted categorical mappings.

    Validation/test categories that were not seen in train are mapped to -1.
    Numeric columns are left untouched so LightGBM can handle missing values.
    """
    transformed = df.copy()
    for column, mapping in mappings.items():
        values = _normalize_category_series(transformed[column])
        transformed[column] = (
            values.map(mapping).fillna(unknown_value).astype("int32")
        )
    return transformed


def fit_baseline_preprocessing(X_train: pd.DataFrame) -> dict[str, object]:
    """Fit baseline preprocessing artifacts on train features only."""
    categorical_columns = get_categorical_columns(X_train)
    categorical_mappings = fit_categorical_mappings(X_train, categorical_columns)
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_mappings": categorical_mappings,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "dropped_columns": [ID_COL],
    }


def apply_baseline_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    """Apply baseline preprocessing fitted on train data."""
    feature_columns = preprocessing["feature_columns"]
    categorical_mappings = preprocessing["categorical_mappings"]

    X = X.loc[:, feature_columns].copy()
    return transform_categorical_columns(X, categorical_mappings)


if __name__ == "__main__":
    raise SystemExit(
        "Preprocessing helpers are imported by training scripts."
    )
