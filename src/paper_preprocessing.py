"""Paper-anchored preprocessing branches for active thesis reruns.

The first active branch is an Alharbi-style IEEE-CIS preprocessing adapter:
categorical frequency encoding, numeric median imputation, and numeric z-score
scaling. All fitted statistics are learned from the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import MISSING_CATEGORY, UNKNOWN_CATEGORY_VALUE, get_categorical_columns


ALHARBI_STYLE_ANCHOR = "Alharbi et al. (2026) IEEE-CIS preprocessing"


def _normalize_category_series(series: pd.Series) -> pd.Series:
    """Return string categories with an explicit missing token."""
    return series.astype("string").fillna(MISSING_CATEGORY).astype(str)


def _numeric_frame(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values for imputation/scaling."""
    if not numeric_columns:
        return pd.DataFrame(index=df.index)
    return df.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")


def _safe_population_std(frame: pd.DataFrame) -> pd.Series:
    """Return train standard deviations, replacing empty/constant columns with 1."""
    std = frame.std(axis=0, ddof=0).replace(0.0, np.nan)
    return std.fillna(1.0)


def fit_alharbi_style_preprocessing(X_train: pd.DataFrame) -> dict[str, object]:
    """Fit the A1 Alharbi-style preprocessing contract on train features only."""
    categorical_columns = get_categorical_columns(X_train)
    numeric_columns = [
        column for column in X_train.columns if column not in categorical_columns
    ]

    numeric_train = _numeric_frame(X_train, numeric_columns)
    numeric_medians = numeric_train.median(axis=0, skipna=True).fillna(0.0)
    numeric_imputed = numeric_train.fillna(numeric_medians)
    numeric_means = numeric_imputed.mean(axis=0).fillna(0.0)
    numeric_stds = _safe_population_std(numeric_imputed)

    frequency_maps: dict[str, dict[str, float]] = {}
    train_rows = float(len(X_train)) if len(X_train) else 1.0
    for column in categorical_columns:
        values = _normalize_category_series(X_train[column])
        counts = values.value_counts(dropna=False)
        frequency_maps[column] = {
            str(category): float(count / train_rows)
            for category, count in counts.items()
        }

    categorical_frequency_columns = [
        f"{column}_frequency" for column in categorical_columns
    ]
    return {
        "kind": "alharbi_style_frequency_median_zscore",
        "anchor": ALHARBI_STYLE_ANCHOR,
        "feature_columns_raw": X_train.columns.tolist(),
        "feature_columns_transformed": numeric_columns + categorical_frequency_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "categorical_frequency_columns": categorical_frequency_columns,
        "numeric_medians": numeric_medians.to_dict(),
        "numeric_means": numeric_means.to_dict(),
        "numeric_stds": numeric_stds.to_dict(),
        "frequency_maps": frequency_maps,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "train_rows": int(len(X_train)),
        "fit_scope": "train split only",
        "notes": {
            "categorical": "Missing values become a dedicated category before train-frequency encoding; unseen validation/test categories map to 0 frequency.",
            "numeric": "Missing values are filled with train medians, then z-score scaled with train means and standard deviations.",
        },
    }


def apply_alharbi_style_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    """Apply train-fitted A1 preprocessing to train/validation/test features."""
    feature_columns_raw = list(preprocessing["feature_columns_raw"])
    numeric_columns = list(preprocessing["numeric_columns"])
    categorical_columns = list(preprocessing["categorical_columns"])

    X = X.loc[:, feature_columns_raw].copy()

    numeric = _numeric_frame(X, numeric_columns)
    if numeric_columns:
        medians = pd.Series(preprocessing["numeric_medians"], dtype="float64")
        means = pd.Series(preprocessing["numeric_means"], dtype="float64")
        stds = pd.Series(preprocessing["numeric_stds"], dtype="float64")
        numeric = numeric.fillna(medians)
        numeric = ((numeric - means) / stds).astype("float32")

    categorical_features: list[pd.Series] = []
    frequency_maps = preprocessing["frequency_maps"]
    for column in categorical_columns:
        values = _normalize_category_series(X[column])
        mapping = frequency_maps[column]
        encoded = values.map(mapping).fillna(0.0).astype("float32")
        encoded.name = f"{column}_frequency"
        categorical_features.append(encoded)

    if categorical_features:
        categorical = pd.concat(categorical_features, axis=1)
        transformed = pd.concat([numeric, categorical], axis=1)
    else:
        transformed = numeric

    transformed = transformed.loc[:, preprocessing["feature_columns_transformed"]]
    if not np.isfinite(transformed.to_numpy(dtype="float32")).all():
        raise ValueError("A1 preprocessing produced non-finite values.")
    return transformed


if __name__ == "__main__":
    raise SystemExit("Import this module from training scripts.")
