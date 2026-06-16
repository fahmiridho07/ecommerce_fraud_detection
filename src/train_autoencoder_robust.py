"""Train robust Phase 3B Autoencoder representations for V-features only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_LATENT_DIM,
    AE_LEARNING_RATE,
    AE_MAX_EPOCHS,
    AE_PATIENCE,
    AE_USE_SCALED_CLIPPING,
    AUTOENCODER_OUTPUT_DIR,
    AUTOENCODER_ROBUST_OUTPUT_DIR,
    DATA_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TEST_RATIO,
    TRAIN_RATIO,
    VALID_RATIO,
)
from train_ae_lgbm import save_latent_split_manifest
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed


MASKED_TARGET_VALUE_SUFFIX = "_target_value"
MASKED_TARGET_OBSERVED_SUFFIX = "_observed_mask"


def build_median_imputer() -> SimpleImputer:
    """Create a median imputer that keeps all V-feature columns when supported."""
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - older sklearn compatibility
        return SimpleImputer(strategy="median")


def fit_transform_v_imputer(
    imputer: SimpleImputer,
    X_train_raw: pd.DataFrame,
) -> np.ndarray:
    X_train_imputed = imputer.fit_transform(X_train_raw)
    if hasattr(imputer, "statistics_"):
        imputer.statistics_ = np.where(np.isnan(imputer.statistics_), 0.0, imputer.statistics_)
    if X_train_imputed.shape[1] != X_train_raw.shape[1]:
        raise ValueError(
            "Median imputation changed the number of V-features. Upgrade "
            "scikit-learn or use an imputer that preserves empty features."
        )
    return X_train_imputed


def build_masked_targets(X: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Pack reconstruction targets and observed-cell mask into y_true."""
    return np.concatenate(
        [X.astype("float32"), observed_mask.astype("float32")],
        axis=1,
    )


@keras.utils.register_keras_serializable(package="thesis")
def masked_mse_loss(y_true, y_pred):
    """Mean squared reconstruction loss over originally observed V-feature cells."""
    input_dim = tf.shape(y_pred)[1]
    target = y_true[:, :input_dim]
    observed_mask = y_true[:, input_dim:]
    squared_error = tf.square(target - y_pred) * observed_mask
    denominator = tf.maximum(tf.reduce_sum(observed_mask, axis=1), 1.0)
    return tf.reduce_sum(squared_error, axis=1) / denominator


def prepare_clipped_v_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    v_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    SimpleImputer,
]:
    """Median-impute V-features, scale train-only, then apply fixed clipping."""
    if not v_columns:
        raise ValueError("No V-features were detected. Check V_FEATURE_PATTERN.")

    X_train_raw = train_df.loc[:, v_columns].astype("float32")
    X_valid_raw = valid_df.loc[:, v_columns].astype("float32")
    X_test_raw = test_df.loc[:, v_columns].astype("float32")

    observed_train = (~X_train_raw.isna()).to_numpy(dtype="float32")
    observed_valid = (~X_valid_raw.isna()).to_numpy(dtype="float32")
    observed_test = (~X_test_raw.isna()).to_numpy(dtype="float32")

    imputer = build_median_imputer()
    X_train_imputed = fit_transform_v_imputer(imputer, X_train_raw)
    X_valid_imputed = imputer.transform(X_valid_raw)
    X_test_imputed = imputer.transform(X_test_raw)
    if (
        X_valid_imputed.shape[1] != len(v_columns)
        or X_test_imputed.shape[1] != len(v_columns)
    ):
        raise ValueError("Median imputation changed validation/test V-feature width.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed).astype("float32")
    X_valid = scaler.transform(X_valid_imputed).astype("float32")
    X_test = scaler.transform(X_test_imputed).astype("float32")

    if AE_USE_SCALED_CLIPPING:
        # Leakage-safe robust preprocessing: these bounds are fixed config
        # constants, not estimated from validation or test distributions.
        X_train = np.clip(X_train, AE_CLIP_MIN, AE_CLIP_MAX)
        X_valid = np.clip(X_valid, AE_CLIP_MIN, AE_CLIP_MAX)
        X_test = np.clip(X_test, AE_CLIP_MIN, AE_CLIP_MAX)

    return (
        X_train,
        X_valid,
        X_test,
        observed_train,
        observed_valid,
        observed_test,
        scaler,
        imputer,
    )


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
) -> tuple[keras.Model, keras.Model]:
    """Build the undercomplete dense Autoencoder used in the thesis design."""
    inputs = keras.Input(shape=(input_dim,), name="v_features")
    x = keras.layers.Dense(256, activation="relu", name="encoder_dense_256")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="encoder_dense_128")(x)
    latent = keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)
    x = keras.layers.Dense(128, activation="relu", name="decoder_dense_128")(latent)
    x = keras.layers.Dense(256, activation="relu", name="decoder_dense_256")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="robust_v_feature_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="robust_v_feature_encoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_mse_loss,
    )
    return autoencoder, encoder


def reconstruction_errors(
    model: keras.Model,
    X: np.ndarray,
    observed_mask: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Compute per-row reconstruction MSE over originally observed V-feature cells."""
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    squared_error = np.square(X - reconstructed) * observed_mask
    denominator = np.maximum(observed_mask.sum(axis=1), 1.0)
    return squared_error.sum(axis=1) / denominator


def reconstruction_stats(errors: np.ndarray) -> dict[str, float | int]:
    """Summarize reconstruction error distribution and high-error counts."""
    return {
        "mean": float(np.mean(errors)),
        "std": float(np.std(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(np.max(errors)),
        "rows_mse_gt_1": int(np.sum(errors > 1)),
        "rows_mse_gt_5": int(np.sum(errors > 5)),
        "rows_mse_gt_10": int(np.sum(errors > 10)),
        "rows_mse_gt_50": int(np.sum(errors > 50)),
        "rows_mse_gt_100": int(np.sum(errors > 100)),
    }


def add_flattened_stats(
    payload: dict[str, object],
    split_name: str,
    stats: dict[str, float | int],
) -> None:
    """Add requested flat reconstruction metric keys."""
    for metric_name, value in stats.items():
        payload[f"{split_name}_reconstruction_mse_{metric_name}"] = value


def save_reconstruction_errors(errors: np.ndarray, output_path: Path) -> None:
    pd.DataFrame({"reconstruction_mse": errors}).to_csv(output_path, index=False)


def save_latent_features(
    encoder: keras.Model,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log("Encoding train, validation, and test V-features.")
    latent_train = encoder.predict(X_train, batch_size=AE_BATCH_SIZE, verbose=0).astype("float32")
    latent_valid = encoder.predict(X_valid, batch_size=AE_BATCH_SIZE, verbose=0).astype("float32")
    latent_test = encoder.predict(X_test, batch_size=AE_BATCH_SIZE, verbose=0).astype("float32")
    np.save(output_dir / "latent_train.npy", latent_train)
    np.save(output_dir / "latent_valid.npy", latent_valid)
    np.save(output_dir / "latent_test.npy", latent_test)
    return latent_train, latent_valid, latent_test


def load_previous_autoencoder_test_metrics() -> dict[str, float | None]:
    """Load old Phase 3 test metrics for a compact comparison if available."""
    metrics_path = AUTOENCODER_OUTPUT_DIR / "reconstruction_metrics.json"
    if not metrics_path.exists():
        return {}

    with metrics_path.open("r", encoding="utf-8") as file:
        old_metrics = json.load(file)

    comparison = {
        "old_test_mean_mse": old_metrics.get("test_reconstruction_mse_mean"),
        "old_test_p99_mse": old_metrics.get("test_reconstruction_mse_p99"),
        "old_test_max_mse": old_metrics.get("test_reconstruction_mse_max"),
    }

    old_errors_path = AUTOENCODER_OUTPUT_DIR / "reconstruction_error_test.csv"
    if old_errors_path.exists():
        old_errors = pd.read_csv(old_errors_path)["reconstruction_mse"].to_numpy()
        comparison["old_test_p99_mse"] = float(np.percentile(old_errors, 99))
        comparison["old_test_max_mse"] = float(np.max(old_errors))

    return comparison


def print_previous_comparison(robust_stats: dict[str, float | int]) -> None:
    previous = load_previous_autoencoder_test_metrics()
    if not previous:
        return

    print()
    print("Previous vs Robust Autoencoder")
    print("==============================")
    print(
        "Test mean MSE: "
        f"{previous.get('old_test_mean_mse'):.6f} -> {robust_stats['mean']:.6f}"
    )
    old_p99 = previous.get("old_test_p99_mse")
    old_max = previous.get("old_test_max_mse")
    if old_p99 is not None:
        print(f"Test p99 MSE : {old_p99:.6f} -> {robust_stats['p99']:.6f}")
    if old_max is not None:
        print(f"Test max MSE : {old_max:.6f} -> {robust_stats['max']:.6f}")


def main(
    latent_dim: int = AE_LATENT_DIM,
    output_dir: Path = AUTOENCODER_ROBUST_OUTPUT_DIR,
    phase_name: str = "3B_robust_autoencoder_representation_learning",
    print_old_comparison: bool = True,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    if latent_dim != AE_LATENT_DIM and output_dir == AUTOENCODER_ROBUST_OUTPUT_DIR:
        raise SystemExit(
            "Non-default latent_dim runs must pass an explicit --output-dir so "
            "LD32 and LD128 artifacts cannot overwrite each other."
        )
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    v_columns = get_v_feature_columns(train_df)
    input_dim = len(v_columns)
    latent_feature_names = [
        f"ae_latent_{index:03d}" for index in range(1, latent_dim + 1)
    ]

    log(
        f"Preparing {input_dim} V-features with train-fitted median imputation, "
        "scaling, and fixed clipping."
    )
    (
        X_train,
        X_valid,
        X_test,
        observed_train,
        observed_valid,
        observed_test,
        scaler,
        imputer,
    ) = prepare_clipped_v_features(
        train_df,
        valid_df,
        test_df,
        v_columns,
    )

    log("Building robust Autoencoder.")
    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        learning_rate=AE_LEARNING_RATE,
    )

    y_train_masked = build_masked_targets(X_train, observed_train)
    y_valid_masked = build_masked_targets(X_valid, observed_valid)

    log("Training robust Autoencoder with masked reconstruction loss on observed V-feature cells.")
    history = autoencoder.fit(
        X_train,
        y_train_masked,
        validation_data=(X_valid, y_valid_masked),
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
    save_latent_split_manifest(train_df, valid_df, test_df, output_dir)

    log("Computing robust reconstruction metrics.")
    train_errors = reconstruction_errors(autoencoder, X_train, observed_train, AE_BATCH_SIZE)
    valid_errors = reconstruction_errors(autoencoder, X_valid, observed_valid, AE_BATCH_SIZE)
    test_errors = reconstruction_errors(autoencoder, X_test, observed_test, AE_BATCH_SIZE)

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
        "loss": "masked_mse_loss",
        "observed_v_cell_rate": {
            "train": float(observed_train.mean()),
            "validation": float(observed_valid.mean()),
            "test": float(observed_test.mean()),
        },
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

    log("Saving robust Autoencoder artifacts.")
    autoencoder.save(output_dir / "autoencoder_model.keras")
    encoder.save(output_dir / "encoder_model.keras")
    joblib.dump(scaler, output_dir / "v_scaler.pkl")
    joblib.dump(imputer, output_dir / "v_imputer.pkl")
    save_json(latent_feature_names, output_dir / "latent_feature_names.json")

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
        "target_usage": "isFraud is not used for Autoencoder training.",
        "preprocessing": {
            "missing_value_strategy": (
                "SimpleImputer(strategy='median') fitted on train V-features only."
            ),
            "imputer_artifact": "v_imputer.pkl",
            "scaler": "StandardScaler fitted on train median-imputed V-features only.",
            "scaled_clipping_enabled": AE_USE_SCALED_CLIPPING,
            "clip_min": AE_CLIP_MIN,
            "clip_max": AE_CLIP_MAX,
            "clipping_leakage_note": (
                "Clipping is leakage-safe here because bounds are fixed config "
                "constants and are not fitted from validation or test data."
            ),
        },
        "architecture": {
            "input_dim": input_dim,
            "encoder": [256, 128, latent_dim],
            "decoder": [128, 256, input_dim],
            "hidden_activation": "relu",
            "latent_activation": "linear",
            "output_activation": "linear",
        },
        "training": {
            "loss": "masked_mse_loss",
            "loss_observed_cells_only": True,
            "optimizer": "Adam",
            "learning_rate": AE_LEARNING_RATE,
            "batch_size": AE_BATCH_SIZE,
            "max_epochs": AE_MAX_EPOCHS,
            "patience": AE_PATIENCE,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "random_seed": RANDOM_SEED,
        },
        "observed_v_cell_rate": {
            "train": float(observed_train.mean()),
            "validation": float(observed_valid.mean()),
            "test": float(observed_test.mean()),
        },
        "latent_features": latent_feature_names,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Robust Autoencoder Representation Summary")
    print("=========================================")
    print(f"V features          : {input_dim}")
    print(f"Latent dimension    : {latent_dim}")
    print(f"Scaled clipping     : {AE_USE_SCALED_CLIPPING} [{AE_CLIP_MIN}, {AE_CLIP_MAX}]")
    print("Imputation          : train-fitted median")
    print("Reconstruction loss : masked MSE on observed V cells")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Train MSE mean      : {train_stats['mean']:.6f}")
    print(f"Validation MSE mean : {valid_stats['mean']:.6f}")
    print(f"Test MSE mean       : {test_stats['mean']:.6f}")
    print(f"Test MSE p99        : {test_stats['p99']:.6f}")
    print(f"Test MSE max        : {test_stats['max']:.6f}")
    print(f"Test rows MSE > 100 : {test_stats['rows_mse_gt_100']:,}")
    print(f"Latent train shape  : {latent_train.shape}")
    print(f"Latent valid shape  : {latent_valid.shape}")
    print(f"Latent test shape   : {latent_test.shape}")
    print(f"Outputs saved to    : {output_dir}")

    if print_old_comparison:
        print_previous_comparison(test_stats)

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
        description="Train robust Autoencoder representations for V-features."
    )
    parser.add_argument("--latent-dim", type=int, default=AE_LATENT_DIM)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUTOENCODER_ROBUST_OUTPUT_DIR,
    )
    parser.add_argument(
        "--phase-name",
        default="3B_robust_autoencoder_representation_learning",
    )
    parser.add_argument(
        "--no-old-comparison",
        action="store_true",
        help="Do not print comparison against the original unstable Autoencoder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        latent_dim=args.latent_dim,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        print_old_comparison=not args.no_old_comparison,
    )
