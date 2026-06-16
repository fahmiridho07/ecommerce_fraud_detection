"""Tests for hybrid AE-LightGBM top-V retention helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_ae_lgbm import (  # noqa: E402
    load_top_v_features_from_importance,
    resolve_replaced_v_columns,
    validate_feature_alignment,
)


def test_resolve_replaced_v_columns_splits_retained_features() -> None:
    v_columns = [f"V{index}" for index in range(1, 6)]
    retained = ["V2", "V5"]
    replaced, retained_out = resolve_replaced_v_columns(v_columns, retained)
    assert retained_out == retained
    assert replaced == ["V1", "V3", "V4"]


def test_load_top_v_features_from_importance(tmp_path: Path) -> None:
    importance = pd.DataFrame(
        {
            "feature": ["V2", "V1", "TransactionAmt", "V3"],
            "importance_gain": [30.0, 50.0, 10.0, 20.0],
            "importance_split": [3, 5, 1, 2],
        }
    )
    path = tmp_path / "feature_importance.csv"
    importance.to_csv(path, index=False)
    selected = load_top_v_features_from_importance(path, top_k=2, v_columns=["V1", "V2", "V3"])
    assert selected == ["V1", "V2"]


def test_validate_feature_alignment_allows_retained_v_columns() -> None:
    train = pd.DataFrame(
        {
            "TransactionAmt": [1.0, 2.0],
            "ae_latent_001": [0.1, 0.2],
            "V1": [1.0, None],
            "v_missing_V2": [0, 1],
        }
    )
    validate_feature_alignment(
        train,
        train.copy(),
        train.copy(),
        replaced_v_columns=["V2"],
        retained_v_columns=["V1"],
    )


def test_validate_feature_alignment_rejects_replaced_v_leakage() -> None:
    train = pd.DataFrame(
        {
            "TransactionAmt": [1.0, 2.0],
            "ae_latent_001": [0.1, 0.2],
            "V2": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Replaced V-features were found"):
        validate_feature_alignment(
            train,
            train.copy(),
            train.copy(),
            replaced_v_columns=["V2"],
            retained_v_columns=[],
        )