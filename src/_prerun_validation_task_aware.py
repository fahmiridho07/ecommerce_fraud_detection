"""Phase 10 pre-run validation for task-aware AE experiment (TAE01)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autoencoder_helpers import (
    apply_frozen_median_scaled_feature_block,
    build_task_aware_autoencoder,
)
from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_USE_SCALED_CLIPPING,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    ID_COL,
    RANDOM_SEED,
    SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    TARGET_COL,
    TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import set_seed

SELECTED_NUMERICAL_LATENT_DIM = 128
SMOKE_ROWS = 256
LAMBDA_CANDIDATES = [0.1, 0.5, 1.0]


def main() -> None:
    set_seed(RANDOM_SEED)

    with SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE.open("r", encoding="utf-8") as file:
        audit = json.load(file)

    feature_names = audit["feature_names"]
    input_dim = len(feature_names)

    print("=" * 60)
    print("TAE01 PRE-RUN VALIDATION")
    print("=" * 60)
    print(f"Selected numerical feature count: {input_dim}")
    if input_dim != 387:
        raise ValueError(f"Expected 387 selected numerical features, got {input_dim}.")
    print()

    forbidden = {TARGET_COL, ID_COL, TIME_COL}
    leaked = sorted(set(feature_names) & forbidden)
    if leaked:
        raise ValueError(f"Forbidden columns in AE input: {leaked}")
    print("Leakage checks:")
    print(f"  Target absent from AE input: {TARGET_COL not in feature_names}")
    print(f"  TransactionID absent: {ID_COL not in feature_names}")
    print(f"  TransactionDT absent: {TIME_COL not in feature_names}")
    print()

    frozen_paths = [
        AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR / "numerical_imputer.pkl",
        AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR / "numerical_scaler.pkl",
        AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR / "selected_numerical_feature_names.json",
    ]
    for path in frozen_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen AAE01 artifact: {path}")
        print(f"Frozen preprocessing artifact OK: {path}")

    required_refs = [
        BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json",
        SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR / "metrics_validation_selected_threshold.json",
    ]
    for path in required_refs:
        if not path.exists():
            raise FileNotFoundError(f"Missing reference artifact: {path}")
        print(f"Reference metric artifact OK: {path}")

    full_df = load_labeled_train_data(sample_size=SMOKE_ROWS)
    train_df, valid_df, test_df = chronological_split(full_df)
    print()
    print("Chronological split row counts (smoke sample):")
    print(f"  train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    import joblib

    imputer = joblib.load(frozen_paths[0])
    scaler = joblib.load(frozen_paths[1])
    with frozen_paths[2].open("r", encoding="utf-8") as file:
        frozen_feature_names = json.load(file)
    if frozen_feature_names != feature_names:
        raise ValueError("Frozen AAE01 feature list does not match audit feature list.")

    X_train, X_valid, X_test = apply_frozen_median_scaled_feature_block(
        train_df,
        valid_df,
        test_df,
        feature_names,
        imputer,
        scaler,
        use_scaled_clipping=AE_USE_SCALED_CLIPPING,
        clip_min=AE_CLIP_MIN,
        clip_max=AE_CLIP_MAX,
    )
    assert np.isfinite(X_train).all()
    print(f"Frozen preprocessing smoke arrays: train {X_train.shape}")

    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    positive_class_weight = negative_count / positive_count if positive_count else 1.0
    print(f"Train-only positive class weight: {positive_class_weight:.6f}")

    for lambda_classification in LAMBDA_CANDIDATES:
        autoencoder, encoder, classification_head = build_task_aware_autoencoder(
            input_dim=input_dim,
            latent_dim=SELECTED_NUMERICAL_LATENT_DIM,
            learning_rate=0.001,
            lambda_classification=lambda_classification,
            positive_class_weight=positive_class_weight,
        )
        recon, fraud_prob = autoencoder.predict(X_train[:32], verbose=0)
        latent = encoder.predict(X_train[:32], verbose=0)
        head_prob = classification_head.predict(X_train[:32], verbose=0)
        assert recon.shape == (32, input_dim)
        assert fraud_prob.shape == (32, 1)
        assert latent.shape == (32, SELECTED_NUMERICAL_LATENT_DIM)
        assert head_prob.shape == (32, 1)
        print(
            f"Architecture shape check OK for lambda={lambda_classification}: "
            f"decoder={recon.shape}, latent={latent.shape}, head={head_prob.shape}"
        )

    retained_raw = 432 - input_dim
    final_lgbm = retained_raw + SELECTED_NUMERICAL_LATENT_DIM
    print()
    print("Expected downstream dimensions:")
    print(f"  Removed numerical features: {input_dim}")
    print(f"  Retained raw features: {retained_raw}")
    print(f"  Latent features: {SELECTED_NUMERICAL_LATENT_DIM}")
    print(f"  Final LightGBM feature count: {final_lgbm}")
    if final_lgbm != 173:
        raise ValueError(f"Expected final feature count 173, got {final_lgbm}.")

    mem_mb = (X_train.nbytes + X_valid.nbytes + X_test.nbytes) / (1024 ** 2)
    print()
    print(f"Smoke-sample memory (~float32 AE matrices): {mem_mb:.1f} MB")
    print(f"Configured AE batch size: {AE_BATCH_SIZE}")
    print(f"Lambda candidates: {LAMBDA_CANDIDATES}")
    print(f"Output dir (must be empty/new): {TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR}")
    print()
    print("PRE-RUN VALIDATION PASSED")


if __name__ == "__main__":
    main()