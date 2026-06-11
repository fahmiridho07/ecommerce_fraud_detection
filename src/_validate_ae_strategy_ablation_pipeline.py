"""Lightweight static checks for AE integration strategy ablation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
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
    ABLATION_BASE_OUTPUT_DIR,
    RECONSTRUCTION_ERROR_FEATURES,
    RECONSTRUCTED_V_NAME_PREFIX,
    SUPPORTED_VARIANTS,
    reconstructed_v_feature_names,
    requires_autoencoder_output_dir,
)


def _check_required_files_exist() -> None:
    repo_root = SRC_DIR.parent
    required = [
        SRC_DIR / "train_ae_integration_strategy_ablation.py",
        SRC_DIR / "build_ae_strategy_ablation_comparison.py",
        repo_root / "docs" / "AE_INTEGRATION_STRATEGY_ABLATION.md",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"


def _check_supported_variants() -> None:
    assert SUPPORTED_VARIANTS == (
        "baseline_fixed",
        "du_latent_replacement",
        "ding_reconstructed_replacement",
        "reconstruction_error_augmentation",
    )


def _check_no_optuna_dependency() -> None:
    source = (SRC_DIR / "train_ae_integration_strategy_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "optuna" not in source.lower()


def _check_output_path_isolation() -> None:
    assert str(ABLATION_BASE_OUTPUT_DIR).endswith(
        "outputs/ae_integration_strategy_ablation"
    ) or ABLATION_BASE_OUTPUT_DIR.name == "ae_integration_strategy_ablation"


def _check_manifest_alignment_reference() -> None:
    source = (SRC_DIR / "train_ae_integration_strategy_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "validate_latent_split_manifest_alignment" in source


def _check_ding_strategy_references_decoder_reconstruction() -> None:
    source = (SRC_DIR / "train_ae_integration_strategy_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "generate_decoder_reconstructed_v" in source
    assert RECONSTRUCTED_V_NAME_PREFIX in source
    assert "ae_reconstructed_V" in reconstructed_v_feature_names(["V1", "V2"])[0]


def _check_reconstruction_error_features() -> None:
    assert RECONSTRUCTION_ERROR_FEATURES == (
        "v_ae_reconstruction_mse",
        "v_ae_reconstruction_log1p_mse",
    )
    source = (SRC_DIR / "train_ae_integration_strategy_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "v_ae_reconstruction_mse" in source
    assert "v_ae_reconstruction_log1p_mse" in source
    assert "grouped reconstruction error" not in source.lower()


def _check_comparison_scope() -> None:
    assert EXPECTED_STRATEGY_IDS == ("STR-B0", "STR-AE1", "STR-AE2", "STR-AE3")
    assert len(STRATEGY_CANDIDATES) == 4
    assert set(EXPECTED_MODEL_NAMES) == set(SUPPORTED_VARIANTS)
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


def _check_autoencoder_output_dir_requirements() -> None:
    assert requires_autoencoder_output_dir("baseline_fixed") is False
    for variant in (
        "du_latent_replacement",
        "ding_reconstructed_replacement",
        "reconstruction_error_augmentation",
    ):
        assert requires_autoencoder_output_dir(variant) is True

    source = (SRC_DIR / "train_ae_integration_strategy_ablation.py").read_text(
        encoding="utf-8"
    )
    assert "--autoencoder-output-dir" in source
    assert "is required for variant" in source


def _check_guide_command_order() -> None:
    guide = (SRC_DIR.parent / "docs" / "AE_INTEGRATION_STRATEGY_ABLATION.md").read_text(
        encoding="utf-8"
    )
    assert "_validate_ae_strategy_ablation_pipeline.py" in guide
    assert "outputs/ae_integration_strategy_ablation" in guide
    assert "build_ae_strategy_ablation_comparison.py" in guide
    assert "initial_proposal" in guide
    assert "STR-B0" in guide
    assert "STR-AE3" in guide


def main() -> None:
    checks = [
        _check_required_files_exist,
        _check_supported_variants,
        _check_no_optuna_dependency,
        _check_output_path_isolation,
        _check_manifest_alignment_reference,
        _check_ding_strategy_references_decoder_reconstruction,
        _check_reconstruction_error_features,
        _check_comparison_scope,
        _check_autoencoder_output_dir_requirements,
        _check_guide_command_order,
    ]
    for check in checks:
        check()
    print("AE integration strategy ablation pipeline checks passed.")


if __name__ == "__main__":
    main()