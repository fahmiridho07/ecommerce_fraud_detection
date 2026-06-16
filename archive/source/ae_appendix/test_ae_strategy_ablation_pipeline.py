"""Static tests for AE integration strategy ablation pipeline helpers."""

from __future__ import annotations

import json
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
    comparison_row,
)
from train_ae_integration_strategy_ablation import (  # noqa: E402
    RECONSTRUCTION_ERROR_FEATURES,
    SUPPORTED_VARIANTS,
    reconstructed_v_feature_names,
    reconstruction_error_feature_names,
    requires_autoencoder_output_dir,
    validate_v_columns_match_ae_run_config,
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


def test_ding_v_column_mismatch_is_fail_fast_not_silent_fallback() -> None:
    source = (SRC_DIR / "train_ae_integration_strategy_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "v_columns = saved_v_columns" not in source

    with pytest.raises(ValueError, match="V column scope mismatch"):
        validate_v_columns_match_ae_run_config(
            ["V1", "V2", "V3"],
            ["V1", "V2", "V4"],
            Path("outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32"),
        )


def test_comparison_row_rejects_wrong_variant_family(tmp_path: Path) -> None:
    output_dir = tmp_path / "wrong_variant_output"
    output_dir.mkdir()

    metrics_payload = {
        "average_precision": 0.5,
        "roc_auc": 0.5,
        "threshold": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "mcc": 0.0,
    }
    (output_dir / "metrics_validation_selected_threshold.json").write_text(
        '{"average_precision": 0.5, "roc_auc": 0.5}',
        encoding="utf-8",
    )
    (output_dir / "metrics_test_selected_threshold.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "experiment_family": "ae_integration_strategy_ablation",
                "variant": "du_latent_replacement",
            }
        ),
        encoding="utf-8",
    )

    candidate = STRATEGY_CANDIDATES[1]
    row, missing = comparison_row(candidate, output_dir)
    assert row is None
    assert any("run_config variant/family mismatch" in message for message in missing)