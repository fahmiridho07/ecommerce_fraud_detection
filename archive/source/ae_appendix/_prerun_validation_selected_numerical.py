"""Phase 9 pre-run validation for selected-numerical AE experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoencoder_helpers import build_dense_autoencoder, prepare_median_scaled_feature_block
from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_USE_SCALED_CLIPPING,
    AE_LGBM_LD128_OUTPUT_DIR,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    ID_COL,
    RANDOM_SEED,
    SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import set_seed

SELECTED_NUMERICAL_LATENT_DIM = 128
SMOKE_ROWS = 2000


def main() -> None:
    set_seed(RANDOM_SEED)

    with SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE.open("r", encoding="utf-8") as file:
        audit = json.load(file)

    feature_names = audit["feature_names"]
    input_dim = len(feature_names)

    print("=" * 60)
    print("PHASE 9 PRE-RUN VALIDATION")
    print("=" * 60)
    print(f"Selected numerical feature count: {input_dim}")
    print(f"V-feature count: {audit['v_feature_count']}")
    print(f"Additional numerical count: {audit['additional_numerical_feature_count']}")
    print()
    print("Included groups:")
    for group in audit["group_summaries"]:
        print(f"  - {group['feature_group']}: {group['column_count']} columns")
    print()
    print("Excluded ambiguous / coded columns:")
    for column, reason in audit["review_required_features"].items():
        print(f"  - {column}: {reason}")
    print()
    print("Leakage checks:")
    print(f"  Target absent from AE input: {TARGET_COL not in feature_names}")
    print(f"  TransactionID absent: {ID_COL not in feature_names}")
    print(f"  TransactionDT absent: {TIME_COL not in feature_names}")
    print()

    required_refs = [
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json",
        AE_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json",
    ]
    for path in required_refs:
        if not path.exists():
            raise FileNotFoundError(f"Missing reference artifact: {path}")
        print(f"Reference artifact OK: {path}")

    if input_dim <= SELECTED_NUMERICAL_LATENT_DIM:
        raise ValueError("Autoencoder would not be undercomplete.")

    print()
    print("Preprocessing smoke test (train-only fit on temporal sample)...")
    full_df = load_labeled_train_data(sample_size=SMOKE_ROWS)
    train_df, valid_df, test_df = chronological_split(full_df)
    X_train, X_valid, X_test, imputer, scaler = prepare_median_scaled_feature_block(
        train_df,
        valid_df,
        test_df,
        feature_names,
        use_scaled_clipping=AE_USE_SCALED_CLIPPING,
        clip_min=AE_CLIP_MIN,
        clip_max=AE_CLIP_MAX,
    )
    assert imputer.statistics_.shape[0] == input_dim
    assert scaler.mean_.shape[0] == input_dim
    assert np.isfinite(X_train).all()
    print(f"  Smoke arrays: train {X_train.shape}, valid {X_valid.shape}, test {X_test.shape}")

    print("Building undercomplete Autoencoder architecture...")
    autoencoder, encoder = build_dense_autoencoder(
        input_dim=input_dim,
        latent_dim=SELECTED_NUMERICAL_LATENT_DIM,
        learning_rate=0.001,
        input_name="selected_numerical_features",
        model_name="selected_numerical_autoencoder",
        encoder_name="selected_numerical_encoder",
    )
    assert autoencoder.count_params() > 0
    print(f"  input_dim={input_dim}, latent_dim={SELECTED_NUMERICAL_LATENT_DIM}")

    retained_raw = 432 - input_dim
    final_lgbm = retained_raw + SELECTED_NUMERICAL_LATENT_DIM
    print()
    print("Expected downstream dimensions:")
    print(f"  Removed numerical features: {input_dim}")
    print(f"  Retained raw features: {retained_raw}")
    print(f"  Latent features: {SELECTED_NUMERICAL_LATENT_DIM}")
    print(f"  Final LightGBM feature count: {final_lgbm}")

    mem_mb = (X_train.nbytes + X_valid.nbytes + X_test.nbytes) / (1024 ** 2)
    print()
    print(f"Smoke-sample memory (~float32 AE matrices): {mem_mb:.1f} MB")
    print(f"Configured AE batch size: {AE_BATCH_SIZE}")
    print(f"Output dir (must be empty/new): {AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR}")
    print()
    print("PRE-RUN VALIDATION PASSED")


if __name__ == "__main__":
    main()