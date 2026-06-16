"""Reusable helpers for leakage-safe Autoencoder experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils import ensure_dir


EXPECTED_CDV_FEATURE_COUNT = 368
RECONSTRUCTION_ERROR_COLUMN = "reconstruction_mse"


def ordered_range_columns(prefix: str, start: int, end: int) -> list[str]:
    """Return feature names such as C1..C14 in numeric order."""
    if start > end:
        raise ValueError("start must be less than or equal to end.")
    return [f"{prefix}{index}" for index in range(start, end + 1)]


def expected_cdv_feature_columns() -> list[str]:
    """Return the exact behavioral CDV feature block in experiment order."""
    columns = (
        ordered_range_columns("C", 1, 14)
        + ordered_range_columns("D", 1, 15)
        + ordered_range_columns("V", 1, 339)
    )
    if len(columns) != EXPECTED_CDV_FEATURE_COUNT:
        raise AssertionError("CDV feature count constant is inconsistent.")
    return columns


def get_ordered_cdv_feature_columns(df: pd.DataFrame) -> list[str]:
    """Validate and return C1-C14, D1-D15, V1-V339 in numeric order."""
    columns = expected_cdv_feature_columns()
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Input data is missing expected CDV feature column(s): "
            + ", ".join(missing_columns[:30])
        )
    return columns


def output_dir_is_non_empty(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    """Create an output directory, refusing accidental non-empty overwrites."""
    if output_dir_is_non_empty(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Pass --overwrite only when you intentionally want to replace this "
            "experiment output."
        )
    return ensure_dir(output_dir)


def raw_float_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Select a numeric AE block, fill missing values with zero, and cast."""
    if not feature_columns:
        raise ValueError("At least one Autoencoder feature column is required.")
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Autoencoder input is missing feature column(s): "
            + ", ".join(missing_columns[:30])
        )
    return df.loc[:, feature_columns].fillna(0).astype("float32")


def raw_float_feature_matrix_unfilled(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Select a numeric AE block and cast without imputing missing values."""
    if not feature_columns:
        raise ValueError("At least one Autoencoder feature column is required.")
    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "Autoencoder input is missing feature column(s): "
            + ", ".join(missing_columns[:30])
        )
    return df.loc[:, feature_columns].astype("float32")


def apply_frozen_median_scaled_feature_block(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    use_scaled_clipping: bool,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform splits with train-fitted median imputer and StandardScaler."""
    X_train_raw = raw_float_feature_matrix_unfilled(train_df, feature_columns)
    X_valid_raw = raw_float_feature_matrix_unfilled(valid_df, feature_columns)
    X_test_raw = raw_float_feature_matrix_unfilled(test_df, feature_columns)

    X_train_imputed = imputer.transform(X_train_raw).astype("float32")
    X_valid_imputed = imputer.transform(X_valid_raw).astype("float32")
    X_test_imputed = imputer.transform(X_test_raw).astype("float32")

    X_train = scaler.transform(X_train_imputed).astype("float32")
    X_valid = scaler.transform(X_valid_imputed).astype("float32")
    X_test = scaler.transform(X_test_imputed).astype("float32")

    if use_scaled_clipping:
        X_train = np.clip(X_train, clip_min, clip_max).astype("float32")
        X_valid = np.clip(X_valid, clip_min, clip_max).astype("float32")
        X_test = np.clip(X_test, clip_min, clip_max).astype("float32")

    return X_train, X_valid, X_test


def prepare_median_scaled_feature_block(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    use_scaled_clipping: bool,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    """Fit train-median imputer and StandardScaler on train only."""
    X_train_raw = raw_float_feature_matrix_unfilled(train_df, feature_columns)
    X_valid_raw = raw_float_feature_matrix_unfilled(valid_df, feature_columns)
    X_test_raw = raw_float_feature_matrix_unfilled(test_df, feature_columns)

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_raw).astype("float32")
    X_valid_imputed = imputer.transform(X_valid_raw).astype("float32")
    X_test_imputed = imputer.transform(X_test_raw).astype("float32")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed).astype("float32")
    X_valid = scaler.transform(X_valid_imputed).astype("float32")
    X_test = scaler.transform(X_test_imputed).astype("float32")

    if use_scaled_clipping:
        X_train = np.clip(X_train, clip_min, clip_max).astype("float32")
        X_valid = np.clip(X_valid, clip_min, clip_max).astype("float32")
        X_test = np.clip(X_test, clip_min, clip_max).astype("float32")

    return X_train, X_valid, X_test, imputer, scaler


def prepare_scaled_feature_block(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    use_scaled_clipping: bool,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train only, transform all splits, and clip."""
    X_train_raw = raw_float_feature_matrix(train_df, feature_columns)
    X_valid_raw = raw_float_feature_matrix(valid_df, feature_columns)
    X_test_raw = raw_float_feature_matrix(test_df, feature_columns)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype("float32")
    X_valid = scaler.transform(X_valid_raw).astype("float32")
    X_test = scaler.transform(X_test_raw).astype("float32")

    if use_scaled_clipping:
        X_train = np.clip(X_train, clip_min, clip_max).astype("float32")
        X_valid = np.clip(X_valid, clip_min, clip_max).astype("float32")
        X_test = np.clip(X_test, clip_min, clip_max).astype("float32")

    return X_train, X_valid, X_test, scaler


def build_dense_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
    input_name: str,
    model_name: str,
    encoder_name: str,
):
    """Build the dense undercomplete AE architecture used by controlled runs."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")

    from tensorflow import keras

    inputs = keras.Input(shape=(input_dim,), name=input_name)
    x = keras.layers.Dense(256, activation="relu", name="encoder_dense_256")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="encoder_dense_128")(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = keras.layers.Dense(128, activation="relu", name="decoder_dense_128")(latent)
    x = keras.layers.Dense(256, activation="relu", name="decoder_dense_256")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    encoder = keras.Model(inputs=inputs, outputs=latent, name=encoder_name)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder, encoder


def build_task_aware_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
    lambda_classification: float,
    positive_class_weight: float,
    input_name: str = "selected_numerical_features",
    model_name: str = "task_aware_autoencoder",
    encoder_name: str = "task_aware_encoder",
    classification_head_name: str = "task_aware_classification_head",
):
    """Build joint reconstruction-classification Autoencoder (TAE01)."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if lambda_classification <= 0:
        raise ValueError("lambda_classification must be positive.")
    if positive_class_weight <= 0:
        raise ValueError("positive_class_weight must be positive.")

    import tensorflow as tf
    from tensorflow import keras

    def weighted_binary_crossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        positive_loss = positive_class_weight * y_true * tf.math.log(y_pred)
        negative_loss = (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return -tf.reduce_mean(positive_loss + negative_loss)

    inputs = keras.Input(shape=(input_dim,), name=input_name)
    x = keras.layers.Dense(256, activation="relu", name="encoder_dense_256")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="encoder_dense_128")(x)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="latent")(x)

    decoder_x = keras.layers.Dense(128, activation="relu", name="decoder_dense_128")(latent)
    decoder_x = keras.layers.Dense(256, activation="relu", name="decoder_dense_256")(decoder_x)
    reconstruction = keras.layers.Dense(
        input_dim,
        activation="linear",
        name="reconstruction",
    )(decoder_x)

    classifier_x = keras.layers.Dense(64, activation="relu", name="classifier_dense_64")(latent)
    classifier_x = keras.layers.Dropout(0.2, name="classifier_dropout_02")(classifier_x)
    fraud_probability = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="fraud_probability",
    )(classifier_x)

    autoencoder = keras.Model(
        inputs=inputs,
        outputs=[reconstruction, fraud_probability],
        name=model_name,
    )
    encoder = keras.Model(inputs=inputs, outputs=latent, name=encoder_name)
    classification_head = keras.Model(
        inputs=inputs,
        outputs=fraud_probability,
        name=classification_head_name,
    )
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "reconstruction": "mse",
            "fraud_probability": weighted_binary_crossentropy,
        },
        loss_weights={
            "reconstruction": 1.0,
            "fraud_probability": float(lambda_classification),
        },
    )
    return autoencoder, encoder, classification_head


def reconstruction_errors(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Compute per-row reconstruction MSE."""
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(X - reconstructed), axis=1).astype("float32")
    if not np.isfinite(errors).all():
        raise ValueError("Computed reconstruction errors contain non-finite values.")
    if np.any(errors < 0):
        raise ValueError("Computed reconstruction errors contain negative values.")
    return errors


def reconstruction_stats(errors: np.ndarray) -> dict[str, float | int]:
    """Summarize reconstruction error distribution."""
    if errors.size == 0:
        raise ValueError("Cannot summarize an empty reconstruction-error array.")
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


def add_flattened_reconstruction_stats(
    payload: dict[str, object],
    split_name: str,
    stats: dict[str, float | int],
) -> None:
    for metric_name, value in stats.items():
        payload[f"{split_name}_reconstruction_mse_{metric_name}"] = value


def save_reconstruction_errors(errors: np.ndarray, output_path: Path) -> None:
    pd.DataFrame({RECONSTRUCTION_ERROR_COLUMN: errors}).to_csv(
        output_path,
        index=False,
    )


def reconstruction_error_file_paths(source_dir: Path) -> dict[str, Path]:
    return {
        "train": source_dir / "reconstruction_error_train.csv",
        "validation": source_dir / "reconstruction_error_valid.csv",
        "test": source_dir / "reconstruction_error_test.csv",
    }


def load_reconstruction_error_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    if RECONSTRUCTION_ERROR_COLUMN not in df.columns:
        raise KeyError(f"{path} is missing {RECONSTRUCTION_ERROR_COLUMN} column.")

    values = df[RECONSTRUCTION_ERROR_COLUMN].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite reconstruction errors.")
    if np.any(values < 0):
        raise ValueError(f"{path} contains negative reconstruction errors.")
    return values


def load_reconstruction_errors(source_dir: Path) -> dict[str, np.ndarray]:
    paths = reconstruction_error_file_paths(source_dir)
    return {
        split_name: load_reconstruction_error_csv(path)
        for split_name, path in paths.items()
    }


def validate_reconstruction_error_lengths(
    errors: dict[str, np.ndarray],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    expected = {
        "train": train_rows,
        "validation": valid_rows,
        "test": test_rows,
    }
    for split_name, row_count in expected.items():
        if split_name not in errors:
            raise KeyError(f"Missing reconstruction errors for split: {split_name}")
        if errors[split_name].shape[0] != row_count:
            raise ValueError(
                f"{split_name} reconstruction-error length "
                f"{errors[split_name].shape[0]} does not match split rows "
                f"{row_count}."
            )
