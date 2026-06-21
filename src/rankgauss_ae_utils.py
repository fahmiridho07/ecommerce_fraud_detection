"""RankGauss helpers for proposal-consistent Autoencoder experiments.

The helpers in this module deliberately fit every statistic on the training
split only. Missing values are represented by a zero placeholder in transformed
space and by a separate observed-value mask, so masked AE losses can ignore
synthetic placeholders instead of treating them as real observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


@dataclass
class RankGaussColumn:
    """Train-fitted one-column RankGauss transform metadata."""

    column: str
    transformer: QuantileTransformer | None
    fill_value: float
    n_observed: int
    n_unique: int


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def select_v_columns_by_missingness_correlation(
    X_train: pd.DataFrame,
    v_columns: list[str],
    *,
    missing_rate_round: int = 4,
    corr_threshold: float = 0.75,
    max_columns: int | None = 150,
) -> tuple[list[str], pd.DataFrame]:
    """Reduce redundant V columns using missingness groups and correlation.

    Features with the same rounded missingness rate often come from the same
    Vesta feature family. Within each such family, this function keeps a greedy
    representative set sorted by observed variance and cardinality, dropping a
    later feature when its absolute Pearson correlation with an already-kept
    feature is at least ``corr_threshold``.
    """
    if not v_columns:
        return [], pd.DataFrame()
    if not 0.0 <= corr_threshold <= 1.0:
        raise ValueError("corr_threshold must be in [0, 1].")

    numeric = X_train.loc[:, v_columns].apply(pd.to_numeric, errors="coerce")
    missing_rate = numeric.isna().mean(axis=0)
    observed_count = numeric.notna().sum(axis=0)
    nunique = numeric.nunique(dropna=True)
    variance = numeric.var(axis=0, skipna=True).fillna(0.0)
    missing_group = missing_rate.round(missing_rate_round)

    selected: list[str] = []
    rows: list[dict[str, Any]] = []
    selected_set: set[str] = set()

    for group_value, group_columns_index in missing_group.groupby(missing_group).groups.items():
        group_columns = list(group_columns_index)
        ordered = sorted(
            group_columns,
            key=lambda column: (
                float(variance[column]),
                int(nunique[column]),
                int(observed_count[column]),
                column,
            ),
            reverse=True,
        )
        corr = numeric.loc[:, ordered].corr().abs()
        kept_in_group: list[str] = []
        for column in ordered:
            max_corr = 0.0
            if kept_in_group:
                max_corr = float(corr.loc[column, kept_in_group].max(skipna=True))
                if np.isnan(max_corr):
                    max_corr = 0.0
            keep = not kept_in_group or max_corr < corr_threshold
            if keep:
                kept_in_group.append(column)
                selected.append(column)
                selected_set.add(column)
            rows.append(
                {
                    "feature": column,
                    "missing_group": float(group_value),
                    "missing_rate": float(missing_rate[column]),
                    "observed_count": int(observed_count[column]),
                    "n_unique": int(nunique[column]),
                    "variance": float(variance[column]),
                    "max_corr_with_kept": max_corr,
                    "selected_before_cap": bool(keep),
                }
            )

    if max_columns is not None and len(selected) > max_columns:
        rank_table = pd.DataFrame(rows)
        rank_table = rank_table.loc[rank_table["feature"].isin(selected)].copy()
        rank_table["selection_score"] = (
            rank_table["variance"].rank(method="dense", ascending=True)
            + rank_table["n_unique"].rank(method="dense", ascending=True)
            + rank_table["observed_count"].rank(method="dense", ascending=True)
        )
        selected = (
            rank_table.sort_values(["selection_score", "feature"], ascending=[False, True])
            .head(max_columns)["feature"]
            .tolist()
        )
        selected_set = set(selected)

    report = pd.DataFrame(rows)
    if not report.empty:
        report["selected"] = report["feature"].isin(selected_set)
        report = report.sort_values(
            ["selected", "missing_group", "variance", "n_unique", "feature"],
            ascending=[False, True, False, False, True],
        ).reset_index(drop=True)
    return selected, report


def fit_observed_rankgauss(
    X_train: pd.DataFrame,
    columns: list[str],
    *,
    max_quantiles: int = 1000,
    random_state: int = 42,
) -> list[RankGaussColumn]:
    """Fit one QuantileTransformer per column using observed train values only."""
    fitted: list[RankGaussColumn] = []
    for column in columns:
        values = _numeric_series(X_train, column).dropna().astype("float64")
        n_observed = int(values.shape[0])
        n_unique = int(values.nunique(dropna=True))
        fill_value = float(values.median()) if n_observed else 0.0
        transformer: QuantileTransformer | None = None
        if n_observed >= 2 and n_unique >= 2:
            n_quantiles = max(2, min(int(max_quantiles), n_observed))
            transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=random_state,
                copy=True,
            )
            transformer.fit(values.to_numpy(dtype="float64").reshape(-1, 1))
        fitted.append(
            RankGaussColumn(
                column=column,
                transformer=transformer,
                fill_value=fill_value,
                n_observed=n_observed,
                n_unique=n_unique,
            )
        )
    return fitted


def transform_observed_rankgauss(
    X: pd.DataFrame,
    fitted: list[RankGaussColumn],
    *,
    clip_value: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform observed values and return ``(values, observed_mask)`` arrays."""
    n_rows = len(X)
    values_out = np.zeros((n_rows, len(fitted)), dtype="float32")
    observed_out = np.zeros((n_rows, len(fitted)), dtype="float32")
    for index, column_fit in enumerate(fitted):
        raw = _numeric_series(X, column_fit.column)
        observed = raw.notna().to_numpy()
        observed_out[:, index] = observed.astype("float32")
        if not observed.any():
            continue
        if column_fit.transformer is None:
            transformed = np.zeros(int(observed.sum()), dtype="float32")
        else:
            transformed = column_fit.transformer.transform(
                raw.loc[observed].to_numpy(dtype="float64").reshape(-1, 1)
            ).ravel()
            transformed = np.clip(transformed, -clip_value, clip_value).astype("float32")
        values_out[observed, index] = transformed
    return values_out, observed_out


def inverse_observed_rankgauss(
    transformed: np.ndarray,
    fitted: list[RankGaussColumn],
    *,
    clip_value: float = 5.0,
) -> np.ndarray:
    """Map RankGauss values back to raw feature scale column by column."""
    if transformed.shape[1] != len(fitted):
        raise ValueError("transformed column count does not match fitted metadata.")
    raw = np.empty_like(transformed, dtype="float32")
    for index, column_fit in enumerate(fitted):
        values = np.clip(transformed[:, index], -clip_value, clip_value)
        if column_fit.transformer is None:
            raw[:, index] = column_fit.fill_value
        else:
            raw[:, index] = column_fit.transformer.inverse_transform(
                values.reshape(-1, 1)
            ).ravel().astype("float32")
    return raw


def observed_reconstruction_error_features(
    values: np.ndarray,
    reconstructed: np.ndarray,
    observed: np.ndarray,
    *,
    prefix: str = "rg_ae",
) -> pd.DataFrame:
    """Build compact reconstruction-error features from observed cells only."""
    if values.shape != reconstructed.shape or values.shape != observed.shape:
        raise ValueError("values, reconstructed, and observed must have identical shapes.")
    abs_error = np.abs(values - reconstructed).astype("float32")
    squared_error = np.square(values - reconstructed).astype("float32")
    observed_count = observed.sum(axis=1)
    denom = np.maximum(observed_count, 1.0)
    masked_abs = abs_error * observed
    masked_sq = squared_error * observed
    max_abs = np.divide(
        masked_abs.max(axis=1),
        np.where(observed_count > 0, 1.0, np.nan),
    )
    max_abs = np.nan_to_num(max_abs, nan=0.0).astype("float32")
    return pd.DataFrame(
        {
            f"{prefix}_mse_observed": (masked_sq.sum(axis=1) / denom).astype("float32"),
            f"{prefix}_mae_observed": (masked_abs.sum(axis=1) / denom).astype("float32"),
            f"{prefix}_max_abs_observed": max_abs,
            f"{prefix}_observed_rate": observed.mean(axis=1).astype("float32"),
        }
    )
