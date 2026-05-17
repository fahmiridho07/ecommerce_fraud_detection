"""Train Phase 3 Autoencoder representations for V-features only."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AE_BATCH_SIZE,
    AE_LATENT_DIM,
    AE_LEARNING_RATE,
    AE_MAX_EPOCHS,
    AE_PATIENCE,
    AUTOENCODER_OUTPUT_DIR,
    DATA_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed


def prepare_v_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    v_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fill missing V values with 0 and fit StandardScaler on train only."""
    if not v_columns:
        raise ValueError("No V-features were detected. Check V_FEATURE_PATTERN.")

    # Leakage prevention: both imputation choice and scaler fitting are based on
    # train V-features only. Validation/test are transformed with the fitted scaler.
    X_train_raw = train_df.loc[:, v_columns].fillna(0).astype("float32")
    X_valid_raw = valid_df.loc[:, v_columns].fillna(0).astype("float32")
    X_test_raw = test_df.loc[:, v_columns].fillna(0).astype("float32")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype("float32")
    X_valid = scaler.transform(X_valid_raw).astype("float32")
    X_test = scaler.transform(X_test_raw).astype("float32")
    return X_train, X_valid, X_test, scaler


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
) -> tuple[keras.Model, keras.Model]:
    """Build an undercomplete dense autoencoder and separate encoder model."""
    inputs = keras.Input(shape=(input_dim,), name="v_features")
    x = keras.layers.Dense(256, activation="relu", name="encoder_dense_256")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="encoder_dense_128")(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = keras.layers.Dense(128, activation="relu", name="decoder_dense_128")(latent)
    x = keras.layers.Dense(256, activation="relu", name="decoder_dense_256")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="v_feature_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="v_feature_encoder")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss="mse")
    return autoencoder, encoder


def reconstruction_errors(model: keras.Model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Compute per-row reconstruction MSE."""
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def reconstruction_stats(errors: np.ndarray) -> dict[str, float]:
    """Summarize reconstruction errors."""
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
    }


def save_reconstruction_errors(
    errors: np.ndarray,
    output_path,
) -> None:
    """Save one-column reconstruction error CSV."""
    pd.DataFrame({"reconstruction_mse": errors}).to_csv(output_path, index=False)


def main() -> None:
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = ensure_dir(AUTOENCODER_OUTPUT_DIR)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    v_columns = get_v_feature_columns(train_df)
    input_dim = len(v_columns)
    latent_feature_names = [
        f"ae_latent_{index:03d}" for index in range(1, AE_LATENT_DIM + 1)
    ]

    log(f"Preparing {input_dim} V-features with train-fitted scaling.")
    X_train, X_valid, X_test, scaler = prepare_v_features(
        train_df,
        valid_df,
        test_df,
        v_columns,
    )

    log("Building Autoencoder.")
    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        latent_dim=AE_LATENT_DIM,
        learning_rate=AE_LEARNING_RATE,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=AE_PATIENCE,
            restore_best_weights=True,
        )
    ]

    log("Training Autoencoder on train V-features only.")
    history = autoencoder.fit(
        X_train,
        X_train,
        validation_data=(X_valid, X_valid),
        epochs=AE_MAX_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "ae_training_history.csv", index=False)

    best_epoch_index = int(history_df["val_loss"].idxmin())
    best_epoch = int(history_df.loc[best_epoch_index, "epoch"])
    best_validation_loss = float(history_df.loc[best_epoch_index, "val_loss"])

    log("Encoding train, validation, and test V-features.")
    latent_train = encoder.predict(X_train, batch_size=AE_BATCH_SIZE, verbose=0).astype("float32")
    latent_valid = encoder.predict(X_valid, batch_size=AE_BATCH_SIZE, verbose=0).astype("float32")
    latent_test = encoder.predict(X_test, batch_size=AE_BATCH_SIZE, verbose=0).astype("float32")
    np.save(output_dir / "latent_train.npy", latent_train)
    np.save(output_dir / "latent_valid.npy", latent_valid)
    np.save(output_dir / "latent_test.npy", latent_test)

    log("Computing reconstruction metrics.")
    train_errors = reconstruction_errors(autoencoder, X_train, AE_BATCH_SIZE)
    valid_errors = reconstruction_errors(autoencoder, X_valid, AE_BATCH_SIZE)
    test_errors = reconstruction_errors(autoencoder, X_test, AE_BATCH_SIZE)

    save_reconstruction_errors(
        train_errors,
        output_dir / "reconstruction_error_train.csv",
    )
    save_reconstruction_errors(
        valid_errors,
        output_dir / "reconstruction_error_valid.csv",
    )
    save_reconstruction_errors(
        test_errors,
        output_dir / "reconstruction_error_test.csv",
    )

    train_stats = reconstruction_stats(train_errors)
    valid_stats = reconstruction_stats(valid_errors)
    test_stats = reconstruction_stats(test_errors)
    reconstruction_metrics = {
        "train_reconstruction_mse_mean": train_stats["mean"],
        "validation_reconstruction_mse_mean": valid_stats["mean"],
        "test_reconstruction_mse_mean": test_stats["mean"],
        "train_reconstruction_mse_std": train_stats["std"],
        "validation_reconstruction_mse_std": valid_stats["std"],
        "test_reconstruction_mse_std": test_stats["std"],
        "input_dim": input_dim,
        "latent_dim": AE_LATENT_DIM,
        "number_of_v_features": input_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }
    save_json(reconstruction_metrics, output_dir / "reconstruction_metrics.json")

    log("Saving Autoencoder artifacts.")
    autoencoder.save(output_dir / "autoencoder_model.keras")
    encoder.save(output_dir / "encoder_model.keras")
    joblib.dump(scaler, output_dir / "v_scaler.pkl")
    save_json(latent_feature_names, output_dir / "latent_feature_names.json")

    run_config = {
        "phase": "3_autoencoder_representation_learning",
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
        "target_usage": "isFraud is not used for Autoencoder training.",
        "leakage_prevention": {
            "imputation": "Missing V-feature values are filled with 0 for all splits.",
            "scaler_fit": "StandardScaler is fit on train V-features only.",
            "autoencoder_fit": "Autoencoder is fit on train V-features only.",
            "validation_usage": "Validation is used only for reconstruction-loss early stopping.",
            "test_usage": "Test is transformed and evaluated only after training.",
        },
        "architecture": {
            "input_dim": input_dim,
            "encoder": [256, 128, AE_LATENT_DIM],
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
        "latent_features": latent_feature_names,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Autoencoder Representation Summary")
    print("==================================")
    print(f"V features          : {input_dim}")
    print(f"Latent dimension    : {AE_LATENT_DIM}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Train MSE mean      : {train_stats['mean']:.6f}")
    print(f"Validation MSE mean : {valid_stats['mean']:.6f}")
    print(f"Test MSE mean       : {test_stats['mean']:.6f}")
    print(f"Latent train shape  : {latent_train.shape}")
    print(f"Latent valid shape  : {latent_valid.shape}")
    print(f"Latent test shape   : {latent_test.shape}")
    print(f"Outputs saved to    : {output_dir}")


if __name__ == "__main__":
    main()
