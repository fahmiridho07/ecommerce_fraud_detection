"""Train a robust Autoencoder on normal train transactions only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_LEARNING_RATE,
    AE_MAX_EPOCHS,
    AE_PATIENCE,
    AE_USE_SCALED_CLIPPING,
    AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR,
    DATA_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns
from splitting import chronological_split
from train_autoencoder_robust import (
    add_flattened_stats,
    build_autoencoder,
    reconstruction_errors,
    reconstruction_stats,
    save_reconstruction_errors,
)
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_NORMAL_ONLY_LATENT_DIM = 128


def raw_v_matrix(df: pd.DataFrame, v_columns: list[str]) -> np.ndarray:
    return df.loc[:, v_columns].fillna(0).astype("float32").to_numpy()


def maybe_clip_scaled_values(X: np.ndarray) -> np.ndarray:
    if not AE_USE_SCALED_CLIPPING:
        return X.astype("float32")
    return np.clip(X, AE_CLIP_MIN, AE_CLIP_MAX).astype("float32")


def prepare_normal_only_v_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    v_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on normal train rows only and transform all split rows."""
    if not v_columns:
        raise ValueError("No V-features were detected. Check V_FEATURE_PATTERN.")

    normal_train_df = train_df.loc[train_df[TARGET_COL] == 0]
    normal_valid_df = valid_df.loc[valid_df[TARGET_COL] == 0]
    if normal_train_df.empty:
        raise ValueError("Normal-only train subset is empty.")
    if normal_valid_df.empty:
        raise ValueError("Normal-only validation subset is empty.")

    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(raw_v_matrix(normal_train_df, v_columns))
    X_valid_normal = scaler.transform(raw_v_matrix(normal_valid_df, v_columns))
    X_train_all = scaler.transform(raw_v_matrix(train_df, v_columns))
    X_valid_all = scaler.transform(raw_v_matrix(valid_df, v_columns))
    X_test_all = scaler.transform(raw_v_matrix(test_df, v_columns))

    return (
        maybe_clip_scaled_values(X_train_all),
        maybe_clip_scaled_values(X_valid_all),
        maybe_clip_scaled_values(X_test_all),
        maybe_clip_scaled_values(X_train_normal),
        maybe_clip_scaled_values(X_valid_normal),
        scaler,
    )


def normal_subset_counts(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    return {
        "train_normal_rows": int((train_df[TARGET_COL] == 0).sum()),
        "train_fraud_rows": int((train_df[TARGET_COL] == 1).sum()),
        "validation_normal_rows": int((valid_df[TARGET_COL] == 0).sum()),
        "validation_fraud_rows": int((valid_df[TARGET_COL] == 1).sum()),
        "test_normal_rows": int((test_df[TARGET_COL] == 0).sum()),
        "test_fraud_rows": int((test_df[TARGET_COL] == 1).sum()),
    }


def main(
    latent_dim: int = DEFAULT_NORMAL_ONLY_LATENT_DIM,
    output_dir: Path = AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR,
    phase_name: str = "next_D_normal_only_autoencoder_ld128",
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    v_columns = get_v_feature_columns(train_df)
    input_dim = len(v_columns)
    subset_counts = normal_subset_counts(train_df, valid_df, test_df)

    log(
        "Preparing V-features with normal-train-fitted scaling and fixed clipping."
    )
    (
        X_train,
        X_valid,
        X_test,
        X_train_normal,
        X_valid_normal,
        scaler,
    ) = prepare_normal_only_v_features(train_df, valid_df, test_df, v_columns)

    log("Building normal-only robust Autoencoder.")
    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        learning_rate=AE_LEARNING_RATE,
    )

    log("Training Autoencoder on normal train V-features only.")
    history = autoencoder.fit(
        X_train_normal,
        X_train_normal,
        validation_data=(X_valid_normal, X_valid_normal),
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

    log("Computing reconstruction errors for all split rows.")
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
        "number_of_v_features": input_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "clipping_enabled": AE_USE_SCALED_CLIPPING,
        "clipping_min": AE_CLIP_MIN,
        "clipping_max": AE_CLIP_MAX,
        "training_subset": subset_counts,
        "splits": {
            "train": train_stats,
            "validation": valid_stats,
            "test": test_stats,
        },
    }
    add_flattened_stats(reconstruction_metrics, "train", train_stats)
    add_flattened_stats(reconstruction_metrics, "validation", valid_stats)
    add_flattened_stats(reconstruction_metrics, "test", test_stats)
    save_json(reconstruction_metrics, output_dir / "reconstruction_metrics.json")

    log("Saving normal-only Autoencoder artifacts.")
    autoencoder.save(output_dir / "autoencoder_model.keras")
    encoder.save(output_dir / "encoder_model.keras")
    joblib.dump(scaler, output_dir / "v_scaler.pkl")

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
        "v_feature_pattern": r"^V\d+$",
        "v_feature_count": input_dim,
        "v_columns": v_columns,
        "target_usage": (
            "isFraud is used only to select normal train rows for Autoencoder "
            "training and normal validation rows for Autoencoder early stopping."
        ),
        "training_subset": subset_counts,
        "preprocessing": {
            "missing_value_strategy": "Fill V-feature missing values with 0.",
            "scaler": (
                "StandardScaler fitted on normal train V-feature rows only, "
                "then applied to all train/validation/test rows."
            ),
            "scaled_clipping_enabled": AE_USE_SCALED_CLIPPING,
            "clip_min": AE_CLIP_MIN,
            "clip_max": AE_CLIP_MAX,
            "clipping_leakage_note": (
                "Clipping bounds are fixed config constants and are not fitted "
                "from validation or test data."
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
            "train_rows_used": subset_counts["train_normal_rows"],
            "validation_rows_used_for_early_stopping": (
                subset_counts["validation_normal_rows"]
            ),
        },
        "outputs": {
            "reconstruction_error_train": "reconstruction_error_train.csv",
            "reconstruction_error_valid": "reconstruction_error_valid.csv",
            "reconstruction_error_test": "reconstruction_error_test.csv",
            "autoencoder_model": "autoencoder_model.keras",
            "encoder_model": "encoder_model.keras",
            "scaler": "v_scaler.pkl",
        },
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Normal-Only Autoencoder Summary")
    print("===============================")
    print(f"V features              : {input_dim}")
    print(f"Latent dimension        : {latent_dim}")
    print(f"Normal train rows       : {subset_counts['train_normal_rows']:,}")
    print(f"Normal validation rows  : {subset_counts['validation_normal_rows']:,}")
    print(f"Scaled clipping         : {AE_USE_SCALED_CLIPPING} [{AE_CLIP_MIN}, {AE_CLIP_MAX}]")
    print(f"Best epoch              : {best_epoch}")
    print(f"Best normal val loss    : {best_validation_loss:.6f}")
    print(f"Train MSE mean          : {train_stats['mean']:.6f}")
    print(f"Validation MSE mean     : {valid_stats['mean']:.6f}")
    print(f"Test MSE mean           : {test_stats['mean']:.6f}")
    print(f"Test MSE p99            : {test_stats['p99']:.6f}")
    print(f"Test MSE max            : {test_stats['max']:.6f}")
    print(f"Outputs saved to        : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "latent_dim": latent_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "train_stats": train_stats,
        "validation_stats": valid_stats,
        "test_stats": test_stats,
        "training_subset": subset_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train normal-only robust Autoencoder for V-features."
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=DEFAULT_NORMAL_ONLY_LATENT_DIM,
        help="Latent dimension for the normal-only Autoencoder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR,
        help="Output directory for normal-only Autoencoder artifacts.",
    )
    parser.add_argument(
        "--phase-name",
        default="next_D_normal_only_autoencoder_ld128",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        latent_dim=args.latent_dim,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
    )
