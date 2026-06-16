"""Tests for AE reconstruction-error augmentation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_ae_reconstruction_error_lgbm import (  # noqa: E402
    ADDED_FEATURES,
    LOG_RECONSTRUCTION_ERROR_FEATURE,
    RECONSTRUCTION_ERROR_FEATURE,
    add_reconstruction_error_features,
    reconstruction_error_features,
    validate_feature_matrix,
)


def test_reconstruction_error_features_add_raw_and_log1p() -> None:
    errors = np.array([0.0, 1.0, 3.0], dtype="float32")
    features = reconstruction_error_features(errors)

    assert features.columns.tolist() == [
        RECONSTRUCTION_ERROR_FEATURE,
        LOG_RECONSTRUCTION_ERROR_FEATURE,
    ]
    assert features[RECONSTRUCTION_ERROR_FEATURE].tolist() == [0.0, 1.0, 3.0]
    assert np.allclose(
        features[LOG_RECONSTRUCTION_ERROR_FEATURE].to_numpy(),
        np.log1p(errors),
    )


def test_validate_feature_matrix_requires_all_original_v_features() -> None:
    original = pd.DataFrame(
        {
            "TransactionDT": [1, 2],
            "V1": [0.1, 0.2],
            "V2": [0.3, 0.4],
        }
    )
    augmented = add_reconstruction_error_features(
        original,
        np.array([0.5, 0.7], dtype="float32"),
    )

    counts = validate_feature_matrix(
        augmented,
        augmented.copy(),
        augmented.copy(),
        v_columns=["V1", "V2"],
        original_feature_count=original.shape[1],
    )

    assert counts["original_v_feature_count"] == 2
    assert counts["reconstruction_error_feature_count"] == len(ADDED_FEATURES)
    assert counts["total_feature_count"] == original.shape[1] + len(ADDED_FEATURES)


def test_validate_feature_matrix_rejects_latent_columns() -> None:
    original = pd.DataFrame(
        {
            "TransactionDT": [1, 2],
            "V1": [0.1, 0.2],
            "ae_latent_001": [0.3, 0.4],
        }
    )
    augmented = add_reconstruction_error_features(
        original,
        np.array([0.5, 0.7], dtype="float32"),
    )

    with pytest.raises(ValueError, match="Latent AE features"):
        validate_feature_matrix(
            augmented,
            augmented.copy(),
            augmented.copy(),
            v_columns=["V1"],
            original_feature_count=original.shape[1],
        )
