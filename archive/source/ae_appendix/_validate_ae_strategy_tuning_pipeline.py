"""Lightweight static checks for AE strategy tuning pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_ae_strategy_tuned_comparison import (  # noqa: E402
    EXPECTED_MODEL_NAMES,
    EXPECTED_STRATEGY_IDS,
    OUT_OF_SCOPE_MODEL_NAMES,
    TUNED_CANDIDATES,
    build_ae_strategy_tuned_comparison_table,
)
from tune_ae_strategy_ablation import (  # noqa: E402
    RECONSTRUCTION_ERROR_FEATURES,
    SUPPORTED_VARIANTS,
    TUNING_BASE_OUTPUT_DIR,
    requires_autoencoder_output_dir,
)


def _check_required_files_exist() -> None:
    repo_root = SRC_DIR.parent
    required = [
        SRC_DIR / "tune_ae_strategy_ablation.py",
        SRC_DIR / "build_ae_strategy_tuned_comparison.py",
        repo_root / "docs" / "AE_STRATEGY_TUNING_PLAN.md",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"


def _check_supported_variants() -> None:
    assert SUPPORTED_VARIANTS == (
        "baseline_lgbm_tuned",
        "ae3_reconstruction_error_lgbm_ld128_tuned",
    )


def _check_output_root() -> None:
    assert TUNING_BASE_OUTPUT_DIR.name == "optuna"
    assert TUNING_BASE_OUTPUT_DIR.parent.name == "ae_integration_strategy_ablation_ld128"


def _check_ae3_feature_construction() -> None:
    source = (SRC_DIR / "tune_ae_strategy_ablation.py").read_text(encoding="utf-8")
    assert "build_reconstruction_error_features" in source
    assert "reconstruction_error_augmentation" in source
    assert "RECONSTRUCTION_ERROR_FEATURES" in source
    assert "validate_ae3_tuned_feature_matrix" in source
    assert "ae_latent_" in source
    assert RECONSTRUCTION_ERROR_FEATURES == (
        "v_ae_reconstruction_mse",
        "v_ae_reconstruction_log1p_mse",
    )
    ablation_source = (
        SRC_DIR / "train_ae_integration_strategy_ablation.py"
    ).read_text(encoding="utf-8")
    assert "v_ae_reconstruction_mse" in ablation_source
    assert "v_ae_reconstruction_log1p_mse" in ablation_source


def _check_ae3_excludes_latent_and_reconstructed() -> None:
    source = (SRC_DIR / "tune_ae_strategy_ablation.py").read_text(encoding="utf-8")
    assert "combine_non_v_and_latent" not in source
    assert "combine_non_v_and_reconstructed_v" not in source
    assert "RECONSTRUCTED_V_NAME_PREFIX" in source
    assert "must not include latent features" in source
    assert "must not include reconstructed V" in source


def _check_comparison_scope() -> None:
    assert EXPECTED_STRATEGY_IDS == ("TUNE-B0", "TUNE-AE3")
    assert len(TUNED_CANDIDATES) == 2
    assert set(EXPECTED_MODEL_NAMES) == set(SUPPORTED_VARIANTS)
    assert OUT_OF_SCOPE_MODEL_NAMES.isdisjoint(set(EXPECTED_MODEL_NAMES))

    table, _ = build_ae_strategy_tuned_comparison_table(
        {
            "baseline_lgbm_tuned": Path("missing/tune_b0"),
            "ae3_reconstruction_error_lgbm_ld128_tuned": Path("missing/tune_ae3"),
        }
    )
    assert table.empty


def _check_comparison_run_config_guards() -> None:
    source = (SRC_DIR / "build_ae_strategy_tuned_comparison.py").read_text(
        encoding="utf-8"
    )
    assert "experiment_family" in source
    assert "validate_tuned_run_config" in source
    assert "final_training_completed" in source
    assert 'run_config.get("variant")' in source


def _check_autoencoder_output_dir_requirements() -> None:
    assert requires_autoencoder_output_dir("baseline_lgbm_tuned") is False
    assert requires_autoencoder_output_dir(
        "ae3_reconstruction_error_lgbm_ld128_tuned"
    ) is True


def _check_tuning_plan_commands() -> None:
    guide = (SRC_DIR.parent / "docs" / "AE_STRATEGY_TUNING_PLAN.md").read_text(
        encoding="utf-8"
    )
    assert "_validate_ae_strategy_tuning_pipeline.py" in guide
    assert "baseline_lgbm_tuned" in guide
    assert "ae3_reconstruction_error_lgbm_ld128_tuned" in guide
    assert "outputs/ae_integration_strategy_ablation_ld128/optuna" in guide


def main() -> None:
    checks = [
        _check_required_files_exist,
        _check_supported_variants,
        _check_output_root,
        _check_ae3_feature_construction,
        _check_ae3_excludes_latent_and_reconstructed,
        _check_comparison_scope,
        _check_comparison_run_config_guards,
        _check_autoencoder_output_dir_requirements,
        _check_tuning_plan_commands,
    ]
    for check in checks:
        check()
    print("AE strategy tuning pipeline checks passed.")


if __name__ == "__main__":
    main()