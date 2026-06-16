"""Tests for initial proposal diagnostic helpers."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_initial_proposal_diagnostics import (  # noqa: E402
    assign_missing_count_bin,
    classify_feature_group,
    feature_group_gain_summary,
)
import pandas as pd


def test_classify_feature_group() -> None:
    assert classify_feature_group("V12") == "v_value"
    assert classify_feature_group("v_missing_V12") == "v_missing_indicator"
    assert classify_feature_group("ae_latent_001") == "ae_latent"
    assert classify_feature_group("TransactionAmt") == "non_v"


def test_assign_missing_count_bin_edges() -> None:
    assert assign_missing_count_bin(0) == "000_000"
    assert assign_missing_count_bin(1) == "001_009"
    assert assign_missing_count_bin(9) == "001_009"
    assert assign_missing_count_bin(10) == "010_049"
    assert assign_missing_count_bin(339) == "339_339"


def test_feature_group_gain_summary_percentages() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["V1", "TransactionAmt", "ae_latent_001", "v_missing_V1"],
            "importance_gain": [30.0, 50.0, 15.0, 5.0],
            "importance_split": [3, 5, 2, 1],
        }
    )
    grouped, summary = feature_group_gain_summary(importance, "demo")
    assert summary["total_gain"] == 100.0
    assert summary["groups"]["v_value"]["gain_share_pct"] == 30.0
    assert summary["groups"]["non_v"]["gain_share_pct"] == 50.0
    assert grouped["gain_share_pct"].sum() == 100.0