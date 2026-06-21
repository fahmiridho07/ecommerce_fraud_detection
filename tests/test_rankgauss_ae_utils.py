"""Tests for RankGauss AE helper contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rankgauss_ae_utils import (  # noqa: E402
    fit_observed_rankgauss,
    inverse_observed_rankgauss,
    observed_reconstruction_error_features,
    select_v_columns_by_missingness_correlation,
    transform_observed_rankgauss,
)


def test_rankgauss_fit_uses_observed_train_values_and_masks_missing() -> None:
    train = pd.DataFrame({"V1": [1.0, 2.0, 100.0, np.nan], "V2": [5.0, 5.0, np.nan, 5.0]})
    valid = pd.DataFrame({"V1": [np.nan, 2.0, 500.0], "V2": [5.0, np.nan, 5.0]})

    fitted = fit_observed_rankgauss(train, ["V1", "V2"], max_quantiles=8, random_state=42)
    values, observed = transform_observed_rankgauss(valid, fitted, clip_value=5.0)

    assert values.shape == (3, 2)
    assert observed.tolist() == [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    assert values[0, 0] == 0.0
    assert values[1, 1] == 0.0
    assert np.isfinite(values).all()

    raw = inverse_observed_rankgauss(values, fitted)
    assert raw.shape == values.shape
    assert np.isfinite(raw).all()


def test_select_v_columns_prunes_correlated_columns_inside_missingness_group() -> None:
    frame = pd.DataFrame(
            {
                "V1": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
                "V2": [2.0, 4.0, 6.0, np.nan, 10.0, 12.0],
                "V3": [1.0, np.nan, 8.0, 4.0, np.nan, 16.0],
            }
        )

    selected, report = select_v_columns_by_missingness_correlation(
        frame,
        ["V1", "V2", "V3"],
        corr_threshold=0.75,
        max_columns=None,
    )

    assert len({"V1", "V2"} & set(selected)) == 1
    assert "V3" in selected
    assert set(report["feature"]) == {"V1", "V2", "V3"}


def test_observed_reconstruction_error_ignores_missing_cells() -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0]], dtype="float32")
    recon = np.array([[2.0, 99.0], [4.0, 25.0]], dtype="float32")
    observed = np.array([[1.0, 0.0], [1.0, 1.0]], dtype="float32")

    features = observed_reconstruction_error_features(values, recon, observed, prefix="x")

    assert np.isclose(features.loc[0, "x_mse_observed"], 1.0)
    assert np.isclose(features.loc[0, "x_mae_observed"], 1.0)
    assert np.isclose(features.loc[1, "x_mse_observed"], (4.0 + 25.0) / 2.0)
    assert np.isclose(features.loc[1, "x_observed_rate"], 1.0)
