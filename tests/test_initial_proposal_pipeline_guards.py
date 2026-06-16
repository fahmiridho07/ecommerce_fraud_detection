"""Guards for the initial thesis proposal experiment path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_initial_proposal_comparison import (  # noqa: E402
    INITIAL_PROPOSAL_CANDIDATES,
    INITIAL_PROPOSAL_MODEL_NAMES,
    build_initial_proposal_comparison_table,
)
from config import ID_COL, TARGET_COL, TIME_COL  # noqa: E402
from splitting import chronological_split, sort_by_transaction_time  # noqa: E402
from train_ae_lgbm import (  # noqa: E402
    save_latent_split_manifest,
    validate_latent_split_manifest_alignment,
)


def _labeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: [300, 100, 200, 400, 500, 600],
            TIME_COL: [10, 10, 10, 20, 20, 30],
            TARGET_COL: [0, 0, 1, 0, 1, 0],
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_sort_by_transaction_time_is_deterministic_on_ties() -> None:
    df = _labeled_frame()
    sorted_df = sort_by_transaction_time(df)
    assert sorted_df[ID_COL].tolist() == [100, 200, 300, 400, 500, 600]
    assert sorted_df[TIME_COL].tolist() == [10, 10, 10, 20, 20, 30]


def test_chronological_split_preserves_deterministic_order() -> None:
    shuffled = _labeled_frame().sample(frac=1.0, random_state=7).reset_index(drop=True)
    train_df, valid_df, test_df = chronological_split(
        shuffled,
        train_ratio=0.5,
        valid_ratio=0.1666666667,
        test_ratio=0.3333333333,
    )
    combined_ids = pd.concat([train_df, valid_df, test_df], ignore_index=True)[ID_COL]
    assert combined_ids.tolist() == [100, 200, 300, 400, 500, 600]


def test_latent_manifest_alignment_passes_when_transaction_ids_match(
    tmp_path: Path,
) -> None:
    train_df = pd.DataFrame(
        {ID_COL: [100, 200], TIME_COL: [1, 2], TARGET_COL: [0, 1], "x": [1.0, 2.0]}
    )
    valid_df = pd.DataFrame(
        {ID_COL: [300], TIME_COL: [3], TARGET_COL: [0], "x": [3.0]}
    )
    test_df = pd.DataFrame(
        {ID_COL: [400, 500], TIME_COL: [4, 5], TARGET_COL: [1, 0], "x": [4.0, 5.0]}
    )
    save_latent_split_manifest(train_df, valid_df, test_df, tmp_path)
    validate_latent_split_manifest_alignment(tmp_path, train_df, valid_df, test_df)


def test_latent_manifest_alignment_fails_when_transaction_ids_are_shuffled(
    tmp_path: Path,
) -> None:
    train_df = pd.DataFrame(
        {ID_COL: [100, 200], TIME_COL: [1, 2], TARGET_COL: [0, 1], "x": [1.0, 2.0]}
    )
    valid_df = pd.DataFrame(
        {ID_COL: [300], TIME_COL: [3], TARGET_COL: [0], "x": [3.0]}
    )
    test_df = pd.DataFrame(
        {ID_COL: [400, 500], TIME_COL: [4, 5], TARGET_COL: [1, 0], "x": [4.0, 5.0]}
    )
    save_latent_split_manifest(train_df, valid_df, test_df, tmp_path)
    shuffled_train = train_df.iloc[[1, 0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="TransactionID order does not match"):
        validate_latent_split_manifest_alignment(
            tmp_path,
            shuffled_train,
            valid_df,
            test_df,
        )


def test_autoencoder_preprocessing_preserves_missingness_signal_by_contract() -> None:
    source = (SRC_DIR / "train_autoencoder_robust.py").read_text(encoding="utf-8")
    assert "SimpleImputer" in source
    assert "v_imputer.pkl" in source
    assert "masked_mse_loss" in source
    assert 'activation="linear", name="latent"' in source


def test_ae_lgbm_appends_v_missing_indicators() -> None:
    ae_source = (SRC_DIR / "train_ae_lgbm.py").read_text(encoding="utf-8")
    optuna_source = (SRC_DIR / "tune_lgbm_optuna.py").read_text(encoding="utf-8")
    assert "build_v_missing_indicators" in ae_source
    assert "v_missing_indicators_included" in ae_source
    assert "validate_autoencoder_preprocessing_contract" in ae_source
    assert "build_v_missing_indicators" in optuna_source
    assert "v_missing_indicators_included" in optuna_source


def test_initial_proposal_comparison_excludes_out_of_scope_model_names() -> None:
    assert set(INITIAL_PROPOSAL_MODEL_NAMES) == {
        candidate["model_name"] for candidate in INITIAL_PROPOSAL_CANDIDATES
    }
    excluded_models = {
        "baseline_lgbm_entity_time_amount_features_tuned",
        "ae_augmented_lgbm_ld128_tuned",
        "score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned",
        "causal_behavioral_lgbm_id_aligned",
    }
    assert excluded_models.isdisjoint(set(INITIAL_PROPOSAL_MODEL_NAMES))

    table, _ = build_initial_proposal_comparison_table(
        {
            "baseline_default": Path("missing/baseline"),
            "baseline_tuned": Path("missing/baseline_tuned"),
            "ae_default": Path("missing/ae"),
            "ae_ld128_tuned": Path("missing/ae_tuned"),
        }
    )
    assert table.empty
    assert list(table.columns) == [
        "canonical_id",
        "legacy_id",
        "model_name",
        "tuned",
        "feature_setup",
        "validation_average_precision",
        "test_average_precision",
        "validation_roc_auc",
        "test_roc_auc",
        "selected_threshold",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_mcc",
        "best_iteration",
        "n_trials",
        "total_features",
        "output_dir",
        "run_config_path",
        "metrics_path",
    ]
