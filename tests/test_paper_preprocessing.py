"""Tests for paper-anchored preprocessing contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paper_preprocessing import (  # noqa: E402
    apply_alharbi_style_preprocessing,
    fit_alharbi_style_preprocessing,
)
from preprocessing import MISSING_CATEGORY  # noqa: E402


def test_alharbi_style_preprocessing_uses_train_frequency_maps_only() -> None:
    train = pd.DataFrame(
        {
            "num": [1.0, 3.0, np.nan, 5.0],
            "cat": ["a", "a", "b", None],
        }
    )
    valid = pd.DataFrame(
        {
            "num": [np.nan, 7.0, 1.0],
            "cat": ["a", "new_category", None],
        }
    )

    preprocessing = fit_alharbi_style_preprocessing(train)
    transformed = apply_alharbi_style_preprocessing(valid, preprocessing)

    assert preprocessing["fit_scope"] == "train split only"
    assert preprocessing["frequency_maps"]["cat"] == {
        "a": 0.5,
        "b": 0.25,
        MISSING_CATEGORY: 0.25,
    }
    assert transformed["cat_frequency"].tolist() == [0.5, 0.0, 0.25]


def test_alharbi_style_numeric_median_and_zscore_are_train_fitted() -> None:
    train = pd.DataFrame(
        {
            "num": [1.0, 3.0, np.nan, 5.0],
            "cat": ["a", "a", "b", None],
        }
    )
    valid = pd.DataFrame(
        {
            "num": [np.nan, 7.0],
            "cat": ["a", "b"],
        }
    )

    preprocessing = fit_alharbi_style_preprocessing(train)
    transformed = apply_alharbi_style_preprocessing(valid, preprocessing)

    assert preprocessing["numeric_medians"]["num"] == 3.0
    assert preprocessing["numeric_means"]["num"] == 3.0
    assert np.isclose(preprocessing["numeric_stds"]["num"], np.sqrt(2.0))
    assert np.isclose(transformed.loc[0, "num"], 0.0)
    assert np.isclose(transformed.loc[1, "num"], (7.0 - 3.0) / np.sqrt(2.0))


def test_alharbi_style_preprocessing_returns_stable_float_matrix() -> None:
    train = pd.DataFrame(
        {
            "num": [np.nan, np.nan, np.nan],
            "constant": [2.0, 2.0, 2.0],
            "cat": ["x", None, "y"],
        }
    )
    preprocessing = fit_alharbi_style_preprocessing(train)
    transformed = apply_alharbi_style_preprocessing(train, preprocessing)

    assert transformed.columns.tolist() == ["num", "constant", "cat_frequency"]
    assert transformed.dtypes.astype(str).tolist() == ["float32", "float32", "float32"]
    assert np.isfinite(transformed.to_numpy()).all()
    assert transformed["num"].eq(0.0).all()
    assert transformed["constant"].eq(0.0).all()
