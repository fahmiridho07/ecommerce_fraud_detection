"""Lightweight static checks for the GBDT baseline comparison pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_gbdt_baseline_comparison import (  # noqa: E402
    DECISION_GATE_MIN_VAL_AP_DELTA,
    EXPECTED_PHASE1_IDS,
    OUT_OF_SCOPE_MODEL_NAMES,
    PHASE1_CANDIDATES,
    TUNE_B0_REFERENCE,
    build_gbdt_baseline_comparison_table,
    build_decision_gate_summary,
)
from gbdt_backends import (  # noqa: E402
    SUPPORTED_BACKENDS,
    SUPPORTED_PREPROCESSING_MODES,
)
from train_gbdt_ae3_integration import (  # noqa: E402
    DEFAULT_AE3_AUTOENCODER_DIR,
    EXPERIMENT_FAMILY as AE3_EXPERIMENT_FAMILY,
)
from train_gbdt_baseline import (  # noqa: E402
    BACKEND_DEFAULT_SUBDIRS,
    EXPERIMENT_FAMILY,
    GBDT_BASE_OUTPUT_DIR,
)
from tune_gbdt_baseline import (  # noqa: E402
    BACKEND_TUNED_SUBDIRS,
    SUPPORTED_FEATURE_SETS,
)


def _check_required_files_exist() -> None:
    repo_root = SRC_DIR.parent
    required = [
        SRC_DIR / "gbdt_backends.py",
        SRC_DIR / "train_gbdt_baseline.py",
        SRC_DIR / "tune_gbdt_baseline.py",
        SRC_DIR / "train_gbdt_ae3_integration.py",
        SRC_DIR / "build_gbdt_baseline_comparison.py",
        repo_root / "docs" / "GBDT_BASELINE_COMPARISON_PLAN.md",
        repo_root / "requirements.txt",
    ]
    for path in required:
        assert path.exists(), f"Missing required file: {path}"


def _check_backends_and_modes() -> None:
    assert SUPPORTED_BACKENDS == ("lightgbm", "xgboost", "catboost")
    assert SUPPORTED_PREPROCESSING_MODES == ("native", "shared_lgbm")
    assert SUPPORTED_FEATURE_SETS == ("raw", "ae3")


def _check_output_path_isolation() -> None:
    assert GBDT_BASE_OUTPUT_DIR.name == "gbdt_baseline_comparison"
    assert EXPERIMENT_FAMILY == "gbdt_baseline_comparison"
    assert AE3_EXPERIMENT_FAMILY == EXPERIMENT_FAMILY


def _check_phase1_subdirs() -> None:
    assert BACKEND_DEFAULT_SUBDIRS["lightgbm"] == "LGBM_fixed"
    assert BACKEND_DEFAULT_SUBDIRS["xgboost"] == "XGB_fixed"
    assert BACKEND_DEFAULT_SUBDIRS["catboost"] == "CAT_fixed"
    assert BACKEND_TUNED_SUBDIRS["lightgbm"] == "optuna/LGBM_tuned"
    assert BACKEND_TUNED_SUBDIRS["xgboost"] == "optuna/XGB_tuned"
    assert BACKEND_TUNED_SUBDIRS["catboost"] == "optuna/CAT_tuned"


def _check_comparison_scope() -> None:
    assert len(PHASE1_CANDIDATES) == 6
    assert EXPECTED_PHASE1_IDS == (
        "GBDT-LGBM-FIX",
        "GBDT-XGB-FIX",
        "GBDT-CAT-FIX",
        "GBDT-LGBM-TUNE",
        "GBDT-XGB-TUNE",
        "GBDT-CAT-TUNE",
    )
    assert OUT_OF_SCOPE_MODEL_NAMES.isdisjoint(
        {candidate["experiment_id"] for candidate in PHASE1_CANDIDATES}
    )

    table, _ = build_gbdt_baseline_comparison_table({})
    assert table.empty

    gate = build_decision_gate_summary(table)
    assert gate["gate_passed"] is False
    assert gate["min_validation_delta"] == DECISION_GATE_MIN_VAL_AP_DELTA
    assert TUNE_B0_REFERENCE["validation_average_precision"] == 0.6378


def _check_requirements_include_gbdt_deps() -> None:
    requirements = (SRC_DIR.parent / "requirements.txt").read_text(encoding="utf-8")
    assert "xgboost" in requirements
    assert "catboost" in requirements


def _check_cli_surface() -> None:
    train_source = (SRC_DIR / "train_gbdt_baseline.py").read_text(encoding="utf-8")
    tune_source = (SRC_DIR / "tune_gbdt_baseline.py").read_text(encoding="utf-8")
    ae3_source = (SRC_DIR / "train_gbdt_ae3_integration.py").read_text(encoding="utf-8")

    assert "--backend" in train_source
    assert "--preprocessing-mode" in train_source
    assert "--feature-set" in tune_source
    assert "--autoencoder-output-dir" in tune_source
    assert "--autoencoder-output-dir" in ae3_source
    assert str(DEFAULT_AE3_AUTOENCODER_DIR).endswith("autoencoder_robust_ld128")


def _check_native_preprocessing_hooks() -> None:
    backends_source = (SRC_DIR / "gbdt_backends.py").read_text(encoding="utf-8")
    assert "fit_catboost_native_preprocessing" in backends_source
    assert "fit_xgboost_native_preprocessing" in backends_source
    assert "map_trial_params_to_backend" in backends_source


def main() -> None:
    _check_required_files_exist()
    _check_backends_and_modes()
    _check_output_path_isolation()
    _check_phase1_subdirs()
    _check_comparison_scope()
    _check_requirements_include_gbdt_deps()
    _check_cli_surface()
    _check_native_preprocessing_hooks()
    print("GBDT baseline pipeline validation passed.")


if __name__ == "__main__":
    main()