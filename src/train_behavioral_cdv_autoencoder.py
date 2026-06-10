"""Train LD128 Autoencoder reconstruction errors for C + D + V features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from autoencoder_helpers import (
    EXPECTED_CDV_FEATURE_COUNT,
    add_flattened_reconstruction_stats,
    build_dense_autoencoder,
    get_ordered_cdv_feature_columns,
    prepare_output_dir,
    prepare_scaled_feature_block,
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
    BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
    DATA_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import log, save_json, set_seed


DEFAULT_CDV_LATENT_DIM = 128


def train_cdv_autoencoder(
    latent_dim: int,
    output_dir: Path,
    overwrite: bool,
    phase_name: str = "behavioral_cdv_autoencoder_ld128",
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    cdv_columns = get_ordered_cdv_feature_columns(train_df)
    input_dim = len(cdv_columns)
    if input_dim != EXPECTED_CDV_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CDV_FEATURE_COUNT} CDV features, found {input_dim}."
        )

    log(
        f"Preparing {input_dim} CDV features with train-fitted scaling "
        "and fixed clipping."
    )
    X_train, X_valid, X_test, scaler = prepare_scaled_feature_block(
        train_df,
        valid_df,
        test_df,
        cdv_columns,
        use_scaled_clipping=AE_USE_SCALED_CLIPPING,
        clip_min=AE_CLIP_MIN,
        clip_max=AE_CLIP_MAX,
    )

    log("Building behavioral CDV Autoencoder.")
    autoencoder, encoder = build_dense_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        learning_rate=AE_LEARNING_RATE,
        input_name="cdv_features",
        model_name="behavioral_cdv_autoencoder",
        encoder_name="behavioral_cdv_encoder",
    )

    log("Training behavioral CDV Autoencoder on train rows only.")
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_valid, X_valid),
        epochs=AE_MAX_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
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

    log("Computing reconstruction errors for train, validation, and test.")
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
        "cdv_feature_count": input_dim,
        "c_feature_count": 14,
        "d_feature_count": 15,
        "v_feature_count": 339,
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
    add_flattened_reconstruction_stats(
        reconstruction_metrics,
        "train",
        train_stats,
    )
    add_flattened_reconstruction_stats(
        reconstruction_metrics,
        "validation",
        valid_stats,
    )
    add_flattened_reconstruction_stats(reconstruction_metrics, "test", test_stats)
    save_json(reconstruction_metrics, output_dir / "reconstruction_metrics.json")

    log("Saving behavioral CDV Autoencoder artifacts.")
    autoencoder.save(output_dir / "autoencoder_model.keras")
    encoder.save(output_dir / "encoder_model.keras")
    joblib.dump(scaler, output_dir / "cdv_scaler.pkl")
    save_json(cdv_columns, output_dir / "cdv_feature_columns.json")

    run_config = {
        "phase": phase_name,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "split_row_counts": {
            "train": int(len(train_df)),
            "validation": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "feature_block": {
            "name": "C1-C14 + D1-D15 + V1-V339",
            "cdv_feature_count": input_dim,
            "expected_cdv_feature_count": EXPECTED_CDV_FEATURE_COUNT,
            "c_columns": cdv_columns[:14],
            "d_columns": cdv_columns[14:29],
            "v_columns": cdv_columns[29:],
            "ordered_columns": cdv_columns,
        },
        "target_usage": "isFraud is not used for Autoencoder training.",
        "leakage_prevention": {
            "split": "Existing chronological 60/20/20 labeled-train split.",
            "missing_value_strategy": "Fill CDV feature missing values with 0.",
            "scaler_fit": "StandardScaler fitted on train CDV features only.",
            "autoencoder_fit": "Autoencoder fitted on train CDV features only.",
            "validation_usage": "Validation used only for reconstruction-loss early stopping.",
            "test_usage": "Test transformed and evaluated once after AE training.",
            "kaggle_competition_test_files_used": False,
            "fraud_labels_used_for_autoencoder": False,
        },
        "preprocessing": {
            "scaled_clipping_enabled": AE_USE_SCALED_CLIPPING,
            "clip_min": AE_CLIP_MIN,
            "clip_max": AE_CLIP_MAX,
            "clipping_leakage_note": (
                "Clipping bounds are fixed config constants and are not fitted "
                "from validation or test distributions."
            ),
        },
        "architecture": {
            "input_dim": input_dim,
            "encoder": [256, 128, latent_dim],
            "decoder": [128, 256, input_dim],
            "hidden_activation": "relu",
            "output_activation": "linear",
        },
        "training": {
            "loss": "mse",
            "optimizer": "Adam",
            "learning_rate": AE_LEARNING_RATE,
            "batch_size": AE_BATCH_SIZE,
            "max_epochs": AE_MAX_EPOCHS,
            "patience": AE_PATIENCE,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "random_seed": RANDOM_SEED,
        },
        "outputs": {
            "reconstruction_error_train": "reconstruction_error_train.csv",
            "reconstruction_error_valid": "reconstruction_error_valid.csv",
            "reconstruction_error_test": "reconstruction_error_test.csv",
            "latent_arrays_saved": False,
            "latent_features_used_downstream": False,
            "autoencoder_model": "autoencoder_model.keras",
            "encoder_model": "encoder_model.keras",
            "scaler": "cdv_scaler.pkl",
        },
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Behavioral CDV Autoencoder Summary")
    print("==================================")
    print(f"CDV features         : {input_dim}")
    print(f"Latent dimension     : {latent_dim}")
    print(f"Scaled clipping      : {AE_USE_SCALED_CLIPPING} [{AE_CLIP_MIN}, {AE_CLIP_MAX}]")
    print(f"Best epoch           : {best_epoch}")
    print(f"Best validation loss : {best_validation_loss:.6f}")
    print(f"Train MSE mean       : {train_stats['mean']:.6f}")
    print(f"Validation MSE mean  : {valid_stats['mean']:.6f}")
    print(f"Test MSE mean        : {test_stats['mean']:.6f}")
    print(f"Test MSE p99         : {test_stats['p99']:.6f}")
    print(f"Test MSE max         : {test_stats['max']:.6f}")
    print(f"Latent arrays saved  : False")
    print(f"Outputs saved to     : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "latent_dim": latent_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "train_stats": train_stats,
        "validation_stats": valid_stats,
        "test_stats": test_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LD128 Autoencoder reconstruction errors for CDV features."
    )
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_CDV_LATENT_DIM)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BEHAVIORAL_CDV_AUTOENCODER_LD128_OUTPUT_DIR,
        help="Output directory for behavioral CDV AE artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing a non-empty behavioral CDV AE output directory.",
    )
    parser.add_argument(
        "--phase-name",
        default="behavioral_cdv_autoencoder_ld128",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    return train_cdv_autoencoder(
        latent_dim=args.latent_dim,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        phase_name=args.phase_name,
    )


if __name__ == "__main__":
    main()
