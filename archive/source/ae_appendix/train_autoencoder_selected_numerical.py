"""Train selected-numerical Autoencoder representations (anchor-alignment experiment)."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from autoencoder_helpers import (
    add_flattened_reconstruction_stats,
    build_dense_autoencoder,
    prepare_median_scaled_feature_block,
    reconstruction_errors,
    reconstruction_stats,
    save_reconstruction_errors,
)
from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_LEARNING_RATE,
    AE_MAX_EPOCHS,
    AE_PATIENCE,
    AE_USE_SCALED_CLIPPING,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    DATA_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed

SELECTED_NUMERICAL_LATENT_DIM = 128
EXPERIMENT_NAME = "selected_numerical_ae_ld128"


def load_selected_numerical_features(audit_file: Path) -> dict[str, object]:
    if not audit_file.exists():
        raise FileNotFoundError(
            f"Selected numerical feature audit not found: {audit_file}\n"
            "Run Phase 1 feature audit before training."
        )
    with audit_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    required_keys = {
        "feature_names",
        "feature_count",
        "v_feature_names",
        "additional_numerical_feature_names",
        "target_not_used",
        "validation_not_used_for_selection",
        "test_not_used_for_selection",
    }
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise KeyError(f"Audit JSON missing required keys: {sorted(missing_keys)}")
    return payload


def validate_selected_features(
    feature_names: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    forbidden = {TARGET_COL, ID_COL, TIME_COL}
    leaked = sorted(set(feature_names) & forbidden)
    if leaked:
        raise ValueError(f"Forbidden columns found in AE input: {leaked}")

    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        missing = [column for column in feature_names if column not in split_df.columns]
        if missing:
            raise KeyError(
                f"{split_name} split is missing selected AE feature(s): "
                + ", ".join(missing[:10])
            )


def validate_latent_arrays(
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray,
    latent_feature_names: list[str],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    latent_dim: int,
) -> None:
    expected = {
        "train": (latent_train, train_rows),
        "validation": (latent_valid, valid_rows),
        "test": (latent_test, test_rows),
    }
    for split_name, (latent, row_count) in expected.items():
        if latent.shape[0] != row_count:
            raise ValueError(
                f"{split_name} latent row count {latent.shape[0]} does not match "
                f"split row count {row_count}."
            )
    if latent_train.shape[1] != latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: expected {latent_dim}, "
            f"got {latent_train.shape[1]}."
        )
    for array_name, latent in (
        ("latent_train", latent_train),
        ("latent_valid", latent_valid),
        ("latent_test", latent_test),
    ):
        if not np.isfinite(latent).all():
            raise ValueError(f"{array_name} contains non-finite values.")
    if len(set(latent_feature_names)) != len(latent_feature_names):
        raise ValueError("Duplicate latent feature names found.")


def save_latent_features(
    encoder: keras.Model,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("Encoding train, validation, and test selected numerical features.")
    latent_train = encoder.predict(X_train, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    latent_valid = encoder.predict(X_valid, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    latent_test = encoder.predict(X_test, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    np.save(output_dir / "latent_train.npy", latent_train)
    np.save(output_dir / "latent_valid.npy", latent_valid)
    np.save(output_dir / "latent_test.npy", latent_test)
    return latent_train, latent_valid, latent_test


def main(
    output_dir: Path = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    audit_file: Path = SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    latent_dim: int = SELECTED_NUMERICAL_LATENT_DIM,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    audit = load_selected_numerical_features(audit_file)
    feature_names = list(audit["feature_names"])
    input_dim = len(feature_names)

    if input_dim <= latent_dim:
        raise ValueError(
            f"Input dimension {input_dim} must exceed latent dimension {latent_dim} "
            "for an undercomplete Autoencoder."
        )

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_selected_features(feature_names, train_df, valid_df, test_df)

    latent_feature_names = [
        f"ae_latent_{index:03d}" for index in range(1, latent_dim + 1)
    ]

    log(
        f"Preparing {input_dim} selected numerical features with train-median "
        "imputation and train-fitted scaling."
    )
    X_train, X_valid, X_test, imputer, scaler = prepare_median_scaled_feature_block(
        train_df,
        valid_df,
        test_df,
        feature_names,
        use_scaled_clipping=AE_USE_SCALED_CLIPPING,
        clip_min=AE_CLIP_MIN,
        clip_max=AE_CLIP_MAX,
    )

    log("Building selected-numerical Autoencoder.")
    autoencoder, encoder = build_dense_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        learning_rate=AE_LEARNING_RATE,
        input_name="selected_numerical_features",
        model_name="selected_numerical_autoencoder",
        encoder_name="selected_numerical_encoder",
    )

    log("Training selected-numerical Autoencoder on chronological train rows only.")
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_valid, X_valid),
        epochs=AE_MAX_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=AE_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "ae_training_history.csv", index=False)

    best_epoch_index = int(history_df["val_loss"].idxmin())
    best_epoch = int(history_df.loc[best_epoch_index, "epoch"])
    best_validation_loss = float(history_df.loc[best_epoch_index, "val_loss"])

    latent_train, latent_valid, latent_test = save_latent_features(
        encoder,
        X_train,
        X_valid,
        X_test,
        output_dir,
    )
    validate_latent_arrays(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
        latent_dim,
    )

    log("Computing reconstruction metrics.")
    train_errors = reconstruction_errors(autoencoder, X_train, AE_BATCH_SIZE)
    valid_errors = reconstruction_errors(autoencoder, X_valid, AE_BATCH_SIZE)
    test_errors = reconstruction_errors(autoencoder, X_test, AE_BATCH_SIZE)

    save_reconstruction_errors(train_errors, output_dir / "reconstruction_error_train.csv")
    save_reconstruction_errors(valid_errors, output_dir / "reconstruction_error_valid.csv")
    save_reconstruction_errors(test_errors, output_dir / "reconstruction_error_test.csv")

    train_stats = reconstruction_stats(train_errors)
    valid_stats = reconstruction_stats(valid_errors)
    test_stats = reconstruction_stats(test_errors)
    reconstruction_metrics: dict[str, object] = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "selected_numerical_feature_count": input_dim,
        "v_feature_count": audit["v_feature_count"],
        "additional_numerical_feature_count": audit["additional_numerical_feature_count"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "clipping_enabled": AE_USE_SCALED_CLIPPING,
        "clipping_min": AE_CLIP_MIN,
        "clipping_max": AE_CLIP_MAX,
        "splits": {
            "train": train_stats,
            "validation": valid_stats,
            "test": test_stats,
        },
    }
    add_flattened_reconstruction_stats(reconstruction_metrics, "train", train_stats)
    add_flattened_reconstruction_stats(reconstruction_metrics, "validation", valid_stats)
    add_flattened_reconstruction_stats(reconstruction_metrics, "test", test_stats)
    save_json(reconstruction_metrics, output_dir / "reconstruction_metrics.json")

    log("Saving selected-numerical Autoencoder artifacts.")
    autoencoder.save(output_dir / "autoencoder_model.keras")
    encoder.save(output_dir / "encoder_model.keras")
    joblib.dump(imputer, output_dir / "numerical_imputer.pkl")
    joblib.dump(scaler, output_dir / "numerical_scaler.pkl")
    save_json(feature_names, output_dir / "selected_numerical_feature_names.json")
    save_json(latent_feature_names, output_dir / "latent_feature_names.json")

    run_config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_purpose": (
            "Anchor-alignment diagnostic: broaden Autoencoder input from V-only "
            "to suitable numerical predictors and replace those columns with "
            "latent features in downstream LightGBM."
        ),
        "anchor_alignment_rationale": (
            "Anchor studies use broader standardized numerical predictor inputs; "
            "this experiment tests whether input scope explains prior negative "
            "V-only AE-LightGBM results."
        ),
        "split_strategy": "chronological TransactionDT holdout",
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "selected_feature_count": input_dim,
        "v_feature_count": audit["v_feature_count"],
        "additional_numerical_feature_count": audit["additional_numerical_feature_count"],
        "selected_feature_names": feature_names,
        "excluded_transactiondt_from_ae": True,
        "excluded_identifiers": [ID_COL],
        "imputation_strategy": "train-median per feature",
        "preprocessing_fit_split": "train",
        "scaling_strategy": "StandardScaler fitted on imputed train data only",
        "clipping_strategy": {
            "enabled": AE_USE_SCALED_CLIPPING,
            "clip_min": AE_CLIP_MIN,
            "clip_max": AE_CLIP_MAX,
            "note": "Fixed config bounds; not estimated from validation or test.",
        },
        "input_dimension": input_dim,
        "latent_dimension": latent_dim,
        "architecture": {
            "encoder": [256, 128, latent_dim],
            "decoder": [128, 256, input_dim],
            "hidden_activation": "relu",
            "output_activation": "linear",
        },
        "loss": "mse",
        "optimizer": "Adam",
        "random_seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "validation_loss": best_validation_loss,
        "target_not_used_for_autoencoder": True,
        "test_not_used_for_model_selection": True,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "audit_file": str(audit_file),
        "sample_size": SAMPLE_SIZE,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Selected-Numerical Autoencoder Summary")
    print("======================================")
    print(f"Selected features   : {input_dim}")
    print(f"V features          : {audit['v_feature_count']}")
    print(f"Additional numerical: {audit['additional_numerical_feature_count']}")
    print(f"Latent dimension    : {latent_dim}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Latent train shape  : {latent_train.shape}")
    print(f"Outputs saved to    : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


if __name__ == "__main__":
    main()