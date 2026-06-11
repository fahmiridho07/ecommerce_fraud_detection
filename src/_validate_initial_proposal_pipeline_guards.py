"""Lightweight static checks for initial proposal pipeline guards."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_initial_proposal_comparison import (  # noqa: E402
    INITIAL_PROPOSAL_MODEL_NAMES,
    build_initial_proposal_comparison_table,
)
from config import ID_COL, TARGET_COL, TIME_COL  # noqa: E402
from splitting import chronological_split, sort_by_transaction_time  # noqa: E402
from train_ae_lgbm import (  # noqa: E402
    LATENT_SPLIT_MANIFEST_CSV,
    LATENT_SPLIT_MANIFEST_JSON,
    save_latent_split_manifest,
    validate_latent_split_manifest_alignment,
)


def _check_splitting_source_uses_stable_tie_break() -> None:
    source = (SRC_DIR / "splitting.py").read_text(encoding="utf-8")
    assert 'kind="mergesort"' in source
    assert "TransactionID ascending is used only as a" in source


def _check_autoencoder_robust_saves_manifest() -> None:
    source = (SRC_DIR / "train_autoencoder_robust.py").read_text(encoding="utf-8")
    assert "save_latent_split_manifest" in source
    assert LATENT_SPLIT_MANIFEST_CSV.replace(".csv", "") in source or LATENT_SPLIT_MANIFEST_CSV in source


def _check_ae_lgbm_and_optuna_validate_manifest() -> None:
    ae_source = (SRC_DIR / "train_ae_lgbm.py").read_text(encoding="utf-8")
    optuna_source = (SRC_DIR / "tune_lgbm_optuna.py").read_text(encoding="utf-8")
    assert "validate_latent_split_manifest_alignment" in ae_source
    assert "validate_latent_split_manifest_alignment" in optuna_source


def _check_tune_lgbm_optuna_supports_autoencoder_output_dir() -> None:
    source = (SRC_DIR / "tune_lgbm_optuna.py").read_text(encoding="utf-8")
    assert "--autoencoder-output-dir" in source
    assert "autoencoder_output_dir: Path" in source
    assert "prepare_ae_lgbm_ld128_data(" in source
    assert "model_type=ae_lgbm_ld128" in source


def _check_build_script_scope() -> None:
    excluded = {
        "baseline_lgbm_entity_time_amount_features_tuned",
        "ae_augmented_lgbm_ld128_tuned",
        "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned",
    }
    assert excluded.isdisjoint(set(INITIAL_PROPOSAL_MODEL_NAMES))
    table, _ = build_initial_proposal_comparison_table(
        {
            "baseline_default": Path("missing/baseline"),
            "baseline_tuned": Path("missing/baseline_tuned"),
            "ae_default": Path("missing/ae"),
            "ae_ld128_tuned": Path("missing/ae_tuned"),
        }
    )
    assert table.empty


def _check_runtime_guards() -> None:
    df = pd.DataFrame(
        {
            ID_COL: [300, 100, 200, 400],
            TIME_COL: [10, 10, 10, 20],
            TARGET_COL: [0, 0, 1, 0],
            "feature_a": [1.0, 2.0, 3.0, 4.0],
        }
    )
    sorted_ids = sort_by_transaction_time(df)[ID_COL].tolist()
    assert sorted_ids == [100, 200, 300, 400]

    train_df, valid_df, test_df = chronological_split(
        df.sample(frac=1.0, random_state=3).reset_index(drop=True),
        train_ratio=0.5,
        valid_ratio=0.25,
        test_ratio=0.25,
    )
    combined_ids = pd.concat([train_df, valid_df, test_df], ignore_index=True)[ID_COL]
    assert combined_ids.tolist() == sorted_ids

    manifest_dir = Path("_tmp_manifest_validation")
    manifest_dir.mkdir(exist_ok=True)
    try:
        save_latent_split_manifest(train_df, valid_df, test_df, manifest_dir)
        assert (manifest_dir / LATENT_SPLIT_MANIFEST_CSV).exists()
        assert (manifest_dir / LATENT_SPLIT_MANIFEST_JSON).exists()
        validate_latent_split_manifest_alignment(
            manifest_dir,
            train_df,
            valid_df,
            test_df,
        )
        shuffled_train = train_df.iloc[::-1].reset_index(drop=True)
        try:
            validate_latent_split_manifest_alignment(
                manifest_dir,
                shuffled_train,
                valid_df,
                test_df,
            )
        except ValueError as exc:
            assert "TransactionID order does not match" in str(exc)
        else:
            raise AssertionError("Expected shuffled TransactionID validation to fail.")
    finally:
        for path in manifest_dir.glob("*"):
            path.unlink()
        manifest_dir.rmdir()


def _check_train_baseline_lgbm_supports_output_dir() -> None:
    source = (SRC_DIR / "train_baseline_lgbm.py").read_text(encoding="utf-8")
    assert '--output-dir' in source or '"--output-dir"' in source
    assert "--phase-name" in source
    assert "phase_name: str" in source
    assert '"phase": phase_name' in source


def _check_build_script_supports_artifact_directory_args() -> None:
    source = (SRC_DIR / "build_initial_proposal_comparison.py").read_text(
        encoding="utf-8"
    )
    for arg in (
        "--baseline-default-dir",
        "--baseline-tuned-dir",
        "--ae-lgbm-default-dir",
        "--ae-lgbm-ld128-tuned-dir",
        "--output-dir",
    ):
        assert arg in source
    assert "comparison_output_paths" in source


def _check_guide_and_readme_links() -> None:
    guide = (SRC_DIR.parent / "docs" / "INITIAL_PROPOSAL_RERUN_GUIDE.md").read_text(
        encoding="utf-8"
    )
    assert "initial_proposal_comparison.csv" in guide
    assert "optuna_comparison.csv" in guide
    assert "outputs/initial_proposal/baseline_lgbm_default" in guide
    assert "outputs/initial_proposal/final_comparison" in guide
    assert (
        "--autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld128"
        in guide
    )
    assert "--model_type ae_lgbm_ld128" in guide
    readme = (SRC_DIR.parent / "README.md").read_text(encoding="utf-8")
    assert "INITIAL_PROPOSAL_RERUN_GUIDE.md" in readme
    assert "initial_proposal_comparison.csv" in readme


def main() -> None:
    checks = [
        _check_splitting_source_uses_stable_tie_break,
        _check_autoencoder_robust_saves_manifest,
        _check_ae_lgbm_and_optuna_validate_manifest,
        _check_tune_lgbm_optuna_supports_autoencoder_output_dir,
        _check_train_baseline_lgbm_supports_output_dir,
        _check_build_script_supports_artifact_directory_args,
        _check_build_script_scope,
        _check_runtime_guards,
        _check_guide_and_readme_links,
    ]
    for check in checks:
        check()
    print("Initial proposal pipeline guard checks passed.")


if __name__ == "__main__":
    main()