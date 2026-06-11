"""Static tests for AE integration strategy ablation pipeline helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_ae_strategy_ablation_comparison import (  # noqa: E402
    EXPECTED_MODEL_NAMES,
    EXPECTED_STRATEGY_IDS,
    OUT_OF_SCOPE_MODEL_NAMES,
    STRATEGY_CANDIDATES,
    build_ae_strategy_ablation_comparison_table,
)
from train_ae_integration_strategy_ablation import (  # noqa: E402
    RECONSTRUCTION_ERROR_FEATURES,
    SUPPORTED_VARIANTS,
    reconstructed_v_feature_names,
    reconstruction_error_feature_names,
    requires_autoencoder_output_dir,
)


def test_strategy_registry_contains_exactly_four_strategies() -> None:
    assert len(STRATEGY_CANDIDATES) == 4
    assert EXPECTED_STRATEGY_IDS == ("STR-B0", "STR-AE1", "STR-AE2", "STR-AE3")
    assert set(EXPECTED_MODEL_NAMES) == set(SUPPORTED_VARIANTS)


def test_comparison_expected_rows_exclude_out_of_scope_names() -> None:
    assert OUT_OF_SCOPE_MODEL_NAMES.isdisjoint(set(EXPECTED_MODEL_NAMES))
    table, _ = build_ae_strategy_ablation_comparison_table(
        {
            "baseline_fixed": Path("missing/b0"),
            "du_latent_replacement": Path("missing/ae1"),
            "ding_reconstructed_replacement": Path("missing/ae2"),
            "reconstruction_error_augmentation": Path("missing/ae3"),
        }
    )
    assert table.empty


def test_reconstructed_feature_naming_for_v_columns() -> None:
    names = reconstructed_v_feature_names(["V1", "V2", "V339"])
    assert names == [
        "ae_reconstructed_V1",
        "ae_reconstructed_V2",
        "ae_reconstructed_V339",
    ]


def test_reconstruction_error_feature_naming_is_stable() -> None:
    assert reconstruction_error_feature_names() == [
        "v_ae_reconstruction_mse",
        "v_ae_reconstruction_log1p_mse",
    ]
    assert RECONSTRUCTION_ERROR_FEATURES == tuple(
        reconstruction_error_feature_names()
    )


def test_baseline_strategy_does_not_require_ae_output_dir() -> None:
    assert requires_autoencoder_output_dir("baseline_fixed") is False


@pytest.mark.parametrize(
    "variant",
    [
        "du_latent_replacement",
        "ding_reconstructed_replacement",
        "reconstruction_error_augmentation",
    ],
)
def test_ae_strategies_require_ae_output_dir(variant: str) -> None:
    assert requires_autoencoder_output_dir(variant) is True