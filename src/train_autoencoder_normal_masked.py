"""Train a normal-only mask-aware Autoencoder for V-feature anomaly signals."""

from __future__ import annotations

import argparse
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
    DATA_DIR,
    PROJECT_ROOT,
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
from train_ae_lgbm import save_latent_split_manifest
from train_autoencoder_robust import (
    build_median_imputer,
    build_masked_targets,
    fit_transform_v_imputer,
    masked_mse_loss,
    reconstruction_stats,
    save_reconstruction_errors,
)
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_OUTPUT_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "normal_masked_autoencoder_ld128"
DEFAULT_LATENT_DIM = 128
DEFAULT_INPUT_NOISE_STD = 0.02

V_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("v001_v095", 1, 95),
    ("v096_v137", 96, 137),
    ("v138_v166", 138, 166),
    ("v167_v216", 167, 216),
    ("v217_v278", 217, 278),
    ("v279_v339", 279, 339),
)


def group_column_indices(v_columns: list[str]) -> dict[str, list[int]]:
    index_by_name = {column: index for index, column in enumerate(v_columns)}
    groups: dict[str, list[int]] = {}
    for group_name, start, end in V_GROUPS:
        names = [f"V{index}" for index in range(start, end + 1)]
        groups[group_name] = [index_by_name[name] for name in names if name in index_by_name]
    return groups


def prepare_mask_aware_v_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    v_columns: list[str],
    training_subset: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    SimpleImputer,
]:
    """Fit train-only median/scaler and append observed masks as AE inputs."""
    if not v_columns:
        raise ValueError("No V-features were detected. Check V_FEATURE_PATTERN.")

    normal_train_df = train_df.loc[train_df[TARGET_COL] == 0]
    if normal_train_df.empty:
        raise ValueError("Normal-only train subset is empty.")
    if training_subset not in {"normal", "all"}:
        raise ValueError(f"Unsupported training_subset: {training_subset}")

    X_train_raw = train_df.loc[:, v_columns].astype("float32")
    X_valid_raw = valid_df.loc[:, v_columns].astype("float32")
    X_test_raw = test_df.loc[:, v_columns].astype("float32")
    if training_subset == "normal":
        X_fit_raw = normal_train_df.loc[:, v_columns].astype("float32")
    else:
        X_fit_raw = X_train_raw

    observed_train = (~X_train_raw.isna()).to_numpy(dtype="float32")
    observed_valid = (~X_valid_raw.isna()).to_numpy(dtype="float32")
    observed_test = (~X_test_raw.isna()).to_numpy(dtype="float32")
    observed_fit = (~X_fit_raw.isna()).to_numpy(dtype="float32")

    imputer = build_median_imputer()
    X_fit_imputed = fit_transform_v_imputer(imputer, X_fit_raw)
    X_train_imputed = imputer.transform(X_train_raw)
    X_valid_imputed = imputer.transform(X_valid_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_imputed).astype("float32")
    X_train_scaled = scaler.transform(X_train_imputed).astype("float32")
    X_valid_scaled = scaler.transform(X_valid_imputed).astype("float32")
    X_test_scaled = scaler.transform(X_test_imputed).astype("float32")

    if AE_USE_SCALED_CLIPPING:
        X_fit_scaled = np.clip(X_fit_scaled, AE_CLIP_MIN, AE_CLIP_MAX)
        X_train_scaled = np.clip(X_train_scaled, AE_CLIP_MIN, AE_CLIP_MAX)
        X_valid_scaled = np.clip(X_valid_scaled, AE_CLIP_MIN, AE_CLIP_MAX)
        X_test_scaled = np.clip(X_test_scaled, AE_CLIP_MIN, AE_CLIP_MAX)

    X_fit_input = np.concatenate(
        [X_fit_scaled, observed_fit],
        axis=1,
    ).astype("float32")
    X_train_input = np.concatenate([X_train_scaled, observed_train], axis=1).astype("float32")
    X_valid_input = np.concatenate([X_valid_scaled, observed_valid], axis=1).astype("float32")
    X_test_input = np.concatenate([X_test_scaled, observed_test], axis=1).astype("float32")

    return (
        X_fit_input,
        X_train_input,
        X_valid_input,
        X_test_input,
        X_fit_scaled,
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled,
        observed_fit,
        observed_train,
        observed_valid,
        observed_test,
        scaler,
        imputer,
    )


def build_mask_aware_autoencoder(
    value_dim: int,
    latent_dim: int,
    learning_rate: float,
    input_noise_std: float,
) -> tuple[keras.Model, keras.Model]:
    """Build AE that receives scaled values plus observed masks and reconstructs values."""
    input_dim = value_dim * 2
    inputs = keras.Input(shape=(input_dim,), name="v_values_plus_observed_mask")
    if input_noise_std > 0:
        value_part = keras.layers.Lambda(
            lambda tensor: tensor[:, :value_dim],
            name="scaled_value_slice",
        )(inputs)
        mask_part = keras.layers.Lambda(
            lambda tensor: tensor[:, value_dim:],
            name="observed_mask_slice",
        )(inputs)
        value_part = keras.layers.GaussianNoise(
            input_noise_std,
            name="value_denoising_noise",
        )(value_part)
        x = keras.layers.Concatenate(name="denoised_values_plus_mask")(
            [value_part, mask_part]
        )
    else:
        x = inputs
    x = keras.layers.Dense(384, activation="relu", name="encoder_dense_384")(x)
    x = keras.layers.Dropout(0.05, name="encoder_dropout_005")(x)
    x = keras.layers.Dense(192, activation="relu", name="encoder_dense_192")(x)
    latent = keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)
    x = keras.layers.Dense(192, activation="relu", name="decoder_dense_192")(latent)
    x = keras.layers.Dense(384, activation="relu", name="decoder_dense_384")(x)
    outputs = keras.layers.Dense(value_dim, activation="linear", name="reconstruction")(x)

    autoencoder = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="normal_only_mask_aware_v_autoencoder",
    )
    encoder = keras.Model(inputs=inputs, outputs=latent, name="normal_only_mask_encoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_mse_loss,
    )
    return autoencoder, encoder


def reconstruction_errors(
    model: keras.Model,
    X_input: np.ndarray,
    X_target: np.ndarray,
    observed_mask: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    reconstructed = model.predict(X_input, batch_size=batch_size, verbose=0)
    squared_error = np.square(X_target - reconstructed) * observed_mask
    denominator = np.maximum(observed_mask.sum(axis=1), 1.0)
    return (squared_error.sum(axis=1) / denominator).astype("float32")


def grouped_reconstruction_features(
    model: keras.Model,
    X_input: np.ndarray,
    X_target: np.ndarray,
    observed_mask: np.ndarray,
    group_indices: dict[str, list[int]],
    batch_size: int,
    feature_prefix: str = "normal_masked_ae",
) -> pd.DataFrame:
    reconstructed = model.predict(X_input, batch_size=batch_size, verbose=0)
    squared_error = np.square(X_target - reconstructed) * observed_mask
    features: dict[str, np.ndarray] = {}
    global_denominator = np.maximum(observed_mask.sum(axis=1), 1.0)
    global_mse = squared_error.sum(axis=1) / global_denominator
    features[f"{feature_prefix}_mse"] = global_mse.astype("float32")
    features[f"{feature_prefix}_log1p_mse"] = np.log1p(global_mse).astype("float32")
    features[f"{feature_prefix}_observed_v_rate"] = observed_mask.mean(axis=1).astype("float32")

    for group_name, indices in group_indices.items():
        if not indices:
            continue
        group_error = squared_error[:, indices]
        group_mask = observed_mask[:, indices]
        denominator = np.maximum(group_mask.sum(axis=1), 1.0)
        group_mse = group_error.sum(axis=1) / denominator
        observed_rate = group_mask.mean(axis=1)
        features[f"{feature_prefix}_{group_name}_mse"] = group_mse.astype("float32")
        features[f"{feature_prefix}_{group_name}_log1p_mse"] = np.log1p(
            group_mse
        ).astype("float32")
        features[f"{feature_prefix}_{group_name}_observed_rate"] = observed_rate.astype(
            "float32"
        )
    return pd.DataFrame(features)


def save_feature_frame(frame: pd.DataFrame, output_path: Path) -> None:
    frame.to_csv(output_path, index=False)


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
    latent_dim: int = DEFAULT_LATENT_DIM,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    phase_name: str = "normal_only_mask_aware_autoencoder_ld128",
    input_noise_std: float = DEFAULT_INPUT_NOISE_STD,
    training_subset: str = "normal",
) -> dict[str, object]:
    if training_subset not in {"normal", "all"}:
        raise ValueError(f"Unsupported training_subset: {training_subset}")
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)
    value_dim = len(v_columns)
    subset_counts = normal_subset_counts(train_df, valid_df, test_df)
    group_indices = group_column_indices(v_columns)

    feature_prefix = (
        "normal_masked_ae" if training_subset == "normal" else "all_masked_ae"
    )
    log(f"Preparing {training_subset}-train-fitted values plus observed-mask inputs.")
    (
        X_fit_input,
        X_train_input,
        X_valid_input,
        X_test_input,
        X_fit_target,
        X_train_target,
        X_valid_target,
        X_test_target,
        observed_fit,
        observed_train,
        observed_valid,
        observed_test,
        scaler,
        imputer,
    ) = prepare_mask_aware_v_features(
        train_df,
        valid_df,
        test_df,
        v_columns,
        training_subset=training_subset,
    )

    log(f"Building {training_subset}-train mask-aware Autoencoder.")
    autoencoder, encoder = build_mask_aware_autoencoder(
        value_dim=value_dim,
        latent_dim=latent_dim,
        learning_rate=AE_LEARNING_RATE,
        input_noise_std=input_noise_std,
    )

    y_fit_masked = build_masked_targets(X_fit_target, observed_fit)
    if training_subset == "normal":
        valid_fit_mask = valid_df[TARGET_COL].to_numpy(dtype=int) == 0
        if not valid_fit_mask.any():
            raise ValueError("Normal-only validation subset is empty.")
        X_valid_fit_input = X_valid_input[valid_fit_mask]
        y_valid_fit_masked = build_masked_targets(
            X_valid_target[valid_fit_mask],
            observed_valid[valid_fit_mask],
        )
    else:
        X_valid_fit_input = X_valid_input
        y_valid_fit_masked = build_masked_targets(X_valid_target, observed_valid)

    log(f"Training on {training_subset} train rows; early stopping on matching validation rows.")
    history = autoencoder.fit(
        X_fit_input,
        y_fit_masked,
        validation_data=(X_valid_fit_input, y_valid_fit_masked),
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

    log("Encoding all split rows.")
    latent_train = encoder.predict(X_train_input, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    latent_valid = encoder.predict(X_valid_input, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    latent_test = encoder.predict(X_test_input, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    np.save(output_dir / "latent_train.npy", latent_train)
    np.save(output_dir / "latent_valid.npy", latent_valid)
    np.save(output_dir / "latent_test.npy", latent_test)
    latent_feature_names = [
        f"{feature_prefix}_latent_{index:03d}" for index in range(1, latent_dim + 1)
    ]
    save_json(latent_feature_names, output_dir / "latent_feature_names.json")
    save_latent_split_manifest(train_df, valid_df, test_df, output_dir)

    log("Computing global and grouped reconstruction-error features.")
    train_errors = reconstruction_errors(
        autoencoder,
        X_train_input,
        X_train_target,
        observed_train,
        AE_BATCH_SIZE,
    )
    valid_errors = reconstruction_errors(
        autoencoder,
        X_valid_input,
        X_valid_target,
        observed_valid,
        AE_BATCH_SIZE,
    )
    test_errors = reconstruction_errors(
        autoencoder,
        X_test_input,
        X_test_target,
        observed_test,
        AE_BATCH_SIZE,
    )
    save_reconstruction_errors(train_errors, output_dir / "reconstruction_error_train.csv")
    save_reconstruction_errors(valid_errors, output_dir / "reconstruction_error_valid.csv")
    save_reconstruction_errors(test_errors, output_dir / "reconstruction_error_test.csv")

    save_feature_frame(
        grouped_reconstruction_features(
            autoencoder,
            X_train_input,
            X_train_target,
            observed_train,
            group_indices,
            AE_BATCH_SIZE,
            feature_prefix=feature_prefix,
        ),
        output_dir / "reconstruction_features_train.csv",
    )
    save_feature_frame(
        grouped_reconstruction_features(
            autoencoder,
            X_valid_input,
            X_valid_target,
            observed_valid,
            group_indices,
            AE_BATCH_SIZE,
            feature_prefix=feature_prefix,
        ),
        output_dir / "reconstruction_features_valid.csv",
    )
    save_feature_frame(
        grouped_reconstruction_features(
            autoencoder,
            X_test_input,
            X_test_target,
            observed_test,
            group_indices,
            AE_BATCH_SIZE,
            feature_prefix=feature_prefix,
        ),
        output_dir / "reconstruction_features_test.csv",
    )

    reconstruction_metrics: dict[str, object] = {
        "input_value_dim": value_dim,
        "input_model_dim": value_dim * 2,
        "latent_dim": latent_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "loss": "masked_mse_loss",
        "training_subset": subset_counts,
        "observed_v_cell_rate": {
            "train": float(observed_train.mean()),
            "validation": float(observed_valid.mean()),
            "test": float(observed_test.mean()),
        },
        "splits": {
            "train": reconstruction_stats(train_errors),
            "validation": reconstruction_stats(valid_errors),
            "test": reconstruction_stats(test_errors),
        },
    }
    save_json(reconstruction_metrics, output_dir / "reconstruction_metrics.json")

    log("Saving Autoencoder artifacts.")
    autoencoder.save(output_dir / "autoencoder_model.keras")
    encoder.save(output_dir / "encoder_model.keras")
    joblib.dump(scaler, output_dir / "v_scaler.pkl")
    joblib.dump(imputer, output_dir / "v_imputer.pkl")

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
        "v_feature_count": value_dim,
        "v_columns": v_columns,
        "target_usage": (
            "isFraud is used only to select normal train/validation rows."
            if training_subset == "normal"
            else "isFraud is not used for Autoencoder fitting or early stopping."
        ),
        "literature_alignment": {
            "anomaly_detection": (
                "Normal-only training with reconstruction error as anomaly signal."
                if training_subset == "normal"
                else "Mask-aware denoising representation learning; reconstruction error is tested as an anomaly signal."
            ),
            "mask_aware_tabular": (
                "Observed-cell mask is appended to the AE input and used in the loss."
            ),
            "denoising": (
                "Gaussian noise is applied only to scaled value inputs, not masks."
            ),
        },
        "preprocessing": {
            "missing_value_strategy": (
                "SimpleImputer(strategy='median') fitted on normal train V-features only."
                if training_subset == "normal"
                else "SimpleImputer(strategy='median') fitted on train V-features only."
            ),
            "imputer_artifact": "v_imputer.pkl",
            "scaler": (
                "StandardScaler fitted on normal train median-imputed V-features only."
                if training_subset == "normal"
                else "StandardScaler fitted on train median-imputed V-features only."
            ),
            "scaled_clipping_enabled": AE_USE_SCALED_CLIPPING,
            "clip_min": AE_CLIP_MIN,
            "clip_max": AE_CLIP_MAX,
        },
        "architecture": {
            "input_value_dim": value_dim,
            "input_model_dim": value_dim * 2,
            "encoder": [value_dim * 2, 384, 192, latent_dim],
            "decoder": [192, 384, value_dim],
            "hidden_activation": "relu",
            "latent_activation": "linear",
            "output_activation": "linear",
            "input_noise_std": input_noise_std,
            "dropout": 0.05,
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
            "training_subset": training_subset,
        },
        "reconstruction_features": {
            "global": [f"{feature_prefix}_mse", f"{feature_prefix}_log1p_mse"],
            "observed_rate": True,
            "feature_prefix": feature_prefix,
            "groups": [
                {
                    "name": name,
                    "start": start,
                    "end": end,
                    "feature_count": len(group_indices[name]),
                }
                for name, start, end in V_GROUPS
            ],
        },
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("Mask-Aware Autoencoder Summary")
    print("==============================")
    print(f"V features          : {value_dim}")
    print(f"Model input dim     : {value_dim * 2}")
    print(f"Latent dimension    : {latent_dim}")
    print(f"Training subset     : {training_subset}")
    print(f"Normal train rows   : {subset_counts['train_normal_rows']:,}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Train MSE mean      : {reconstruction_metrics['splits']['train']['mean']:.6f}")
    print(f"Validation MSE mean : {reconstruction_metrics['splits']['validation']['mean']:.6f}")
    print(f"Test MSE mean       : {reconstruction_metrics['splits']['test']['mean']:.6f}")
    print(f"Outputs saved to    : {output_dir}")

    return {
        "output_dir": str(output_dir),
        "latent_dim": latent_dim,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "reconstruction_metrics": reconstruction_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train normal-only mask-aware AE and save reconstruction features."
    )
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase-name",
        default="normal_only_mask_aware_autoencoder_ld128",
    )
    parser.add_argument("--input-noise-std", type=float, default=DEFAULT_INPUT_NOISE_STD)
    parser.add_argument(
        "--training-subset",
        choices=("normal", "all"),
        default="normal",
        help="Rows used to fit the AE: normal train rows or the full train split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        latent_dim=args.latent_dim,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        input_noise_std=args.input_noise_std,
        training_subset=args.training_subset,
    )
