"""Run advisor-requested AE feature-improvement ladder.

The ladder stays inside the original Autoencoder + LightGBM idea but tests
whether AE should improve, complement, denoise, or partially reconstruct V
features instead of replacing all V features with a small latent vector.

Stop rule:
- Use the original proposal tuned LightGBM test AP as the reference.
- Run stages sequentially.
- For each candidate, train LightGBM with fixed tuned parameter profiles and
  select the profile by validation AP.
- Stop once the selected candidate test AP exceeds the tuned LightGBM reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("TensorFlow is not installed.") from exc

from config import RANDOM_SEED, SAMPLE_SIZE, TEST_RATIO, TRAIN_RATIO, VALID_RATIO
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns, split_features_target
from run_original_proposal_stratified import (
    compact_result,
    encode_latents,
    encode_splits,
    evaluate_model,
    fit_lgbm,
    paired_bootstrap_ap_delta,
)
from splitting import stratified_holdout_split
from train_baseline_lgbm import save_feature_importance
from utils import ensure_dir, log, save_json, set_seed


BASELINE_REFERENCE_AP = 0.873133233975772
BASELINE_REFERENCE_VALID_AP = 0.8745885416185524
DEFAULT_ORIGINAL_OUTPUT = Path("outputs/stratified_reset/original_proposal_v_latent_replacement")


@dataclass(frozen=True)
class CandidateSpec:
    stage: int
    candidate_id: str
    description: str
    mode: str
    latent_dim: int | None = None
    denoising: bool = False
    partial_strategy: str | None = None
    high_missing_threshold: float | None = None


@dataclass
class LadderData:
    X_base_train: pd.DataFrame
    X_base_valid: pd.DataFrame
    X_base_test: pd.DataFrame
    X_non_v_train: pd.DataFrame
    X_non_v_valid: pd.DataFrame
    X_non_v_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    baseline_categorical_columns: list[str]
    non_v_categorical_columns: list[str]
    v_columns: list[str]
    V_train: np.ndarray
    V_valid: np.ndarray
    V_test: np.ndarray
    X_train_raw: pd.DataFrame
    X_valid_raw: pd.DataFrame
    X_test_raw: pd.DataFrame


def zero_fill_matrix(X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = X.loc[:, columns].to_numpy(dtype="float32", copy=True)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def fit_zero_zscore(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    V_train = zero_fill_matrix(X_train, columns)
    V_valid = zero_fill_matrix(X_valid, columns)
    V_test = zero_fill_matrix(X_test, columns)
    mean = V_train.mean(axis=0, dtype="float64").astype("float32")
    scale = V_train.std(axis=0, dtype="float64").astype("float32")
    scale[scale == 0.0] = 1.0

    def transform(values: np.ndarray) -> np.ndarray:
        values -= mean
        values /= scale
        return values.astype("float32", copy=False)

    return (
        transform(V_train),
        transform(V_valid),
        transform(V_test),
        {"columns": columns, "mean": mean.tolist(), "scale": scale.tolist()},
    )


def inverse_zero_zscore(values: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype="float32")
    scale = np.asarray(scaler["scale"], dtype="float32")
    return (values * scale + mean).astype("float32", copy=False)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_params(path: Path, n_jobs: int, seed: int) -> dict[str, object]:
    params = dict(load_json(path)["best_params_lightgbm"])  # type: ignore[index]
    params["n_jobs"] = n_jobs
    params["random_state"] = seed
    params["metric"] = "None"
    params["verbosity"] = -1
    return params


def load_reference_scores(path: Path, y_test: pd.Series) -> np.ndarray:
    df = pd.read_csv(path)
    if not np.array_equal(df["isFraud"].to_numpy(dtype=int), y_test.to_numpy(dtype=int)):
        raise ValueError("Reference score file does not match the current test split order.")
    score_columns = [column for column in df.columns if column.endswith("_score")]
    if len(score_columns) != 1:
        raise ValueError(f"Expected exactly one score column in {path}, found {score_columns}.")
    return df[score_columns[0]].to_numpy(dtype="float64")


def load_test_scores(path: Path, y_test: pd.Series) -> np.ndarray:
    df = pd.read_csv(path)
    if not np.array_equal(df["isFraud"].to_numpy(dtype=int), y_test.to_numpy(dtype=int)):
        raise ValueError(f"Score file does not match the current test split order: {path}")
    score_columns = [column for column in df.columns if column.endswith("_score")]
    if len(score_columns) != 1:
        raise ValueError(f"Expected exactly one score column in {path}, found {score_columns}.")
    return df[score_columns[0]].to_numpy(dtype="float64")


def prepare_ladder_data(output_dir: Path, seed: int) -> LadderData:
    log("Loading IEEE-CIS data for AE feature-improvement ladder.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = stratified_holdout_split(full_df, random_seed=seed)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    v_columns = get_v_feature_columns(X_train_raw)
    log(f"Found {len(v_columns)} V columns.")

    baseline_encoded = encode_splits(X_train_raw, X_valid_raw, X_test_raw)
    non_v_encoded = encode_splits(
        X_train_raw.drop(columns=v_columns),
        X_valid_raw.drop(columns=v_columns),
        X_test_raw.drop(columns=v_columns),
    )
    V_train, V_valid, V_test, scaler = fit_zero_zscore(
        X_train_raw,
        X_valid_raw,
        X_test_raw,
        v_columns,
    )
    save_json(scaler, output_dir / "v_full_zero_zscore_scaler.json")

    save_json(
        {
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "train_fraud_rate": float(y_train.mean()),
            "valid_fraud_rate": float(y_valid.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "baseline_feature_count": int(baseline_encoded.X_train.shape[1]),
            "non_v_feature_count": int(non_v_encoded.X_train.shape[1]),
            "v_feature_count": len(v_columns),
            "split_strategy": "stratified_holdout",
            "split_ratios": {"train": TRAIN_RATIO, "validation": VALID_RATIO, "test": TEST_RATIO},
            "seed": seed,
        },
        output_dir / "data_contract.json",
    )

    return LadderData(
        X_base_train=baseline_encoded.X_train,
        X_base_valid=baseline_encoded.X_valid,
        X_base_test=baseline_encoded.X_test,
        X_non_v_train=non_v_encoded.X_train,
        X_non_v_valid=non_v_encoded.X_valid,
        X_non_v_test=non_v_encoded.X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        baseline_categorical_columns=baseline_encoded.categorical_columns,
        non_v_categorical_columns=non_v_encoded.categorical_columns,
        v_columns=v_columns,
        V_train=V_train,
        V_valid=V_valid,
        V_test=V_test,
        X_train_raw=X_train_raw,
        X_valid_raw=X_valid_raw,
        X_test_raw=X_test_raw,
    )


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
    latent_activation: str,
    denoising_noise_std: float,
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(input_dim,), name="v_features")
    x = inputs
    if denoising_noise_std > 0.0:
        x = keras.layers.GaussianNoise(denoising_noise_std, name="denoising_noise")(x)
    x = keras.layers.Dense(256, activation="relu", name="encoder_dense_256")(x)
    x = keras.layers.Dense(128, activation="relu", name="encoder_dense_128")(x)
    latent = keras.layers.Dense(latent_dim, activation=latent_activation, name="latent")(x)
    x = keras.layers.Dense(128, activation="relu", name="decoder_dense_128")(latent)
    x = keras.layers.Dense(256, activation="relu", name="decoder_dense_256")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="ladder_v_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="ladder_v_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return autoencoder, encoder


def train_autoencoder(
    V_train: np.ndarray,
    V_valid: np.ndarray,
    output_dir: Path,
    latent_dim: int,
    latent_activation: str,
    denoising_noise_std: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[keras.Model, keras.Model, pd.DataFrame]:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    autoencoder, encoder = build_autoencoder(
        input_dim=V_train.shape[1],
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        latent_activation=latent_activation,
        denoising_noise_std=denoising_noise_std,
    )
    history = autoencoder.fit(
        V_train,
        V_train,
        validation_data=(V_valid, V_valid),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "ae_training_history.csv", index=False)
    autoencoder.save(output_dir / "autoencoder.keras")
    encoder.save(output_dir / "encoder.keras")
    return autoencoder, encoder, history_df


def latent_cache_key(latent_dim: int, denoising: bool, latent_activation: str) -> str:
    mode = "dae" if denoising else "ae"
    return f"{mode}_ld{latent_dim}_{latent_activation}"


def original_ld32_cache_available(original_output: Path) -> bool:
    return all(
        (original_output / filename).exists()
        for filename in ["latent_train.npy", "latent_valid.npy", "latent_test.npy"]
    )


def get_latents(
    data: LadderData,
    cache_dir: Path,
    original_output: Path,
    latent_dim: int,
    latent_activation: str,
    denoising: bool,
    denoising_noise_std: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    key = latent_cache_key(latent_dim, denoising, latent_activation)
    output_dir = ensure_dir(cache_dir / key)
    train_path = output_dir / "latent_train.npy"
    valid_path = output_dir / "latent_valid.npy"
    test_path = output_dir / "latent_test.npy"
    if train_path.exists() and valid_path.exists() and test_path.exists():
        log(f"Loading cached latents: {key}.")
        return np.load(train_path), np.load(valid_path), np.load(test_path), output_dir

    if (
        latent_dim == 32
        and not denoising
        and latent_activation == "relu"
        and original_ld32_cache_available(original_output)
    ):
        log("Reusing original proposal LD32 latents.")
        latent_train = np.load(original_output / "latent_train.npy")
        latent_valid = np.load(original_output / "latent_valid.npy")
        latent_test = np.load(original_output / "latent_test.npy")
        np.save(train_path, latent_train)
        np.save(valid_path, latent_valid)
        np.save(test_path, latent_test)
        save_json({"source": str(original_output), "reused_original_ld32": True}, output_dir / "latent_source.json")
        return latent_train, latent_valid, latent_test, output_dir

    log(f"Training {'denoising ' if denoising else ''}AE latent cache: {key}.")
    _, encoder, _ = train_autoencoder(
        V_train=data.V_train,
        V_valid=data.V_valid,
        output_dir=output_dir,
        latent_dim=latent_dim,
        latent_activation=latent_activation,
        denoising_noise_std=denoising_noise_std if denoising else 0.0,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    latent_train, latent_valid, latent_test = encode_latents(
        encoder,
        data.V_train,
        data.V_valid,
        data.V_test,
        batch_size=batch_size,
    )
    np.save(train_path, latent_train)
    np.save(valid_path, latent_valid)
    np.save(test_path, latent_test)
    return latent_train, latent_valid, latent_test, output_dir


def add_latent_features(
    base_train: pd.DataFrame,
    base_valid: pd.DataFrame,
    base_test: pd.DataFrame,
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = [f"{prefix}_{index:03d}" for index in range(1, latent_train.shape[1] + 1)]

    def combine(base: pd.DataFrame, latent: np.ndarray) -> pd.DataFrame:
        latent_df = pd.DataFrame(latent, columns=names, index=base.index)
        return pd.concat([base.reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)

    return combine(base_train, latent_train), combine(base_valid, latent_valid), combine(base_test, latent_test)


def build_latent_feature_matrices(
    data: LadderData,
    spec: CandidateSpec,
    cache_dir: Path,
    original_output: Path,
    latent_activation: str,
    denoising_noise_std: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    if spec.latent_dim is None:
        raise ValueError("Latent candidate requires latent_dim.")
    latent_train, latent_valid, latent_test, source_dir = get_latents(
        data=data,
        cache_dir=cache_dir,
        original_output=original_output,
        latent_dim=spec.latent_dim,
        latent_activation=latent_activation,
        denoising=spec.denoising,
        denoising_noise_std=denoising_noise_std,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + spec.stage + spec.latent_dim,
    )
    prefix = "dae_latent" if spec.denoising else "ae_latent"
    if spec.mode == "replace":
        X_train, X_valid, X_test = add_latent_features(
            data.X_non_v_train,
            data.X_non_v_valid,
            data.X_non_v_test,
            latent_train,
            latent_valid,
            latent_test,
            prefix,
        )
        categorical_columns = data.non_v_categorical_columns
    elif spec.mode == "concat":
        X_train, X_valid, X_test = add_latent_features(
            data.X_base_train,
            data.X_base_valid,
            data.X_base_test,
            latent_train,
            latent_valid,
            latent_test,
            prefix,
        )
        categorical_columns = data.baseline_categorical_columns
    else:
        raise ValueError(f"Unsupported latent feature mode: {spec.mode}")
    return X_train, X_valid, X_test, categorical_columns, source_dir


def prepare_subset_zero_scaled(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    return fit_zero_zscore(X_train, X_valid, X_test, columns)


def high_missing_v_columns(data: LadderData, threshold: float, fallback_top_k: int) -> list[str]:
    missing_rate = data.X_train_raw.loc[:, data.v_columns].isna().mean()
    selected = missing_rate[missing_rate >= threshold].sort_values(ascending=False).index.tolist()
    if not selected:
        selected = missing_rate.sort_values(ascending=False).head(fallback_top_k).index.tolist()
    return selected


def build_partial_reconstruction_matrices(
    data: LadderData,
    spec: CandidateSpec,
    output_dir: Path,
    latent_activation: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    threshold = spec.high_missing_threshold if spec.high_missing_threshold is not None else 0.75
    selected_columns = high_missing_v_columns(data, threshold=threshold, fallback_top_k=80)
    candidate_dir = ensure_dir(output_dir / "partial_reconstruction_cache" / spec.candidate_id)
    V_train, V_valid, V_test, scaler = prepare_subset_zero_scaled(
        data.X_train_raw,
        data.X_valid_raw,
        data.X_test_raw,
        selected_columns,
    )
    save_json(scaler, candidate_dir / "subset_zero_zscore_scaler.json")
    save_json(
        {
            "selected_v_columns": selected_columns,
            "selected_v_count": len(selected_columns),
            "missing_threshold": threshold,
            "partial_strategy": spec.partial_strategy,
        },
        candidate_dir / "partial_reconstruction_contract.json",
    )
    latent_dim = spec.latent_dim if spec.latent_dim is not None else min(128, max(16, len(selected_columns) // 2))
    autoencoder, _, _ = train_autoencoder(
        V_train=V_train,
        V_valid=V_valid,
        output_dir=candidate_dir,
        latent_dim=latent_dim,
        latent_activation=latent_activation,
        denoising_noise_std=0.0,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + spec.stage + latent_dim,
    )
    recon_train_scaled = autoencoder.predict(V_train, batch_size=batch_size, verbose=0).astype("float32")
    recon_valid_scaled = autoencoder.predict(V_valid, batch_size=batch_size, verbose=0).astype("float32")
    recon_test_scaled = autoencoder.predict(V_test, batch_size=batch_size, verbose=0).astype("float32")
    np.save(candidate_dir / "recon_train_scaled.npy", recon_train_scaled)
    np.save(candidate_dir / "recon_valid_scaled.npy", recon_valid_scaled)
    np.save(candidate_dir / "recon_test_scaled.npy", recon_test_scaled)

    if spec.partial_strategy == "replace":
        recon_train_raw = inverse_zero_zscore(recon_train_scaled, scaler)
        recon_valid_raw = inverse_zero_zscore(recon_valid_scaled, scaler)
        recon_test_raw = inverse_zero_zscore(recon_test_scaled, scaler)

        def replace(base: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
            transformed = base.copy()
            for index, column in enumerate(selected_columns):
                transformed[column] = values[:, index]
            return transformed

        return (
            replace(data.X_base_train, recon_train_raw),
            replace(data.X_base_valid, recon_valid_raw),
            replace(data.X_base_test, recon_test_raw),
            data.baseline_categorical_columns,
            candidate_dir,
        )

    if spec.partial_strategy == "append":
        feature_names = [f"ae_recon_{column}" for column in selected_columns]

        def append(base: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
            recon_df = pd.DataFrame(values, columns=feature_names, index=base.index)
            return pd.concat([base.reset_index(drop=True), recon_df.reset_index(drop=True)], axis=1)

        return (
            append(data.X_base_train, recon_train_scaled),
            append(data.X_base_valid, recon_valid_scaled),
            append(data.X_base_test, recon_test_scaled),
            data.baseline_categorical_columns,
            candidate_dir,
        )

    raise ValueError(f"Unsupported partial reconstruction strategy: {spec.partial_strategy}")


def train_with_profile(
    data: LadderData,
    output_dir: Path,
    candidate_id: str,
    profile_name: str,
    params: dict[str, object],
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, object]:
    profile_dir = ensure_dir(output_dir / profile_name)
    metrics_path = profile_dir / "metrics_and_config.json"
    scores_path = profile_dir / f"{candidate_id}_{profile_name}_test_scores.csv"
    if metrics_path.exists() and scores_path.exists():
        log(f"Loading cached {candidate_id} profile {profile_name}.")
        metrics = load_json(metrics_path)
        return {
            "profile_name": profile_name,
            "compact": metrics["compact"],
            "validation_score": None,
            "test_score": load_test_scores(scores_path, data.y_test),
            "output_dir": str(profile_dir),
        }

    log(f"Training {candidate_id} with profile {profile_name}.")
    model, best_iteration = fit_lgbm(
        X_train,
        data.y_train,
        X_valid,
        data.y_valid,
        categorical_columns,
        params,
    )
    joblib.dump(model, profile_dir / "model.pkl")
    model.booster_.save_model(str(profile_dir / "model.txt"))
    save_feature_importance(model, profile_dir / "feature_importance.csv")
    result = evaluate_model(
        model,
        best_iteration,
        X_valid,
        data.y_valid,
        X_test,
        data.y_test,
        profile_dir,
        f"{candidate_id}_{profile_name}",
    )
    compact = compact_result(result)
    save_json(
        {
            "best_iteration": int(result["best_iteration"]),
            "selected_threshold": float(result["selected_threshold"]),
            "validation_selected": result["validation_selected"],
            "test_selected": result["test_selected"],
            "model_params": params,
            "compact": compact,
        },
        profile_dir / "metrics_and_config.json",
    )
    return {
        "profile_name": profile_name,
        "compact": compact,
        "validation_score": result["validation_score"],
        "test_score": result["test_score"],
        "output_dir": str(profile_dir),
    }


def evaluate_candidate(
    data: LadderData,
    spec: CandidateSpec,
    output_dir: Path,
    param_profiles: dict[str, dict[str, object]],
    reference_scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    feature_source_dir: Path,
) -> dict[str, object]:
    candidate_dir = ensure_dir(output_dir / spec.candidate_id)
    save_json(asdict(spec), candidate_dir / "candidate_spec.json")
    profile_results = []
    for profile_name, params in param_profiles.items():
        profile_results.append(
            train_with_profile(
                data=data,
                output_dir=candidate_dir,
                candidate_id=spec.candidate_id,
                profile_name=profile_name,
                params=params,
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                categorical_columns=categorical_columns,
            )
        )
        gc.collect()

    selected = max(
        profile_results,
        key=lambda item: item["compact"]["validation_average_precision"],  # type: ignore[index]
    )
    selected_compact = selected["compact"]  # type: ignore[assignment]
    test_score = selected["test_score"]  # type: ignore[assignment]
    comparison = paired_bootstrap_ap_delta(
        data.y_test.to_numpy(dtype=int),
        reference_scores,
        test_score,  # type: ignore[arg-type]
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = {
        "candidate": asdict(spec),
        "feature_source_dir": str(feature_source_dir),
        "feature_count": int(X_train.shape[1]),
        "categorical_feature_count": len(categorical_columns),
        "selected_profile": selected["profile_name"],
        "selected_result": selected_compact,
        "delta_vs_baseline_tuned_test_ap": float(
            selected_compact["test_average_precision"] - BASELINE_REFERENCE_AP  # type: ignore[index]
        ),
        "comparison_vs_baseline_tuned": comparison,
        "profile_results": [
            {
                "profile_name": item["profile_name"],
                "output_dir": item["output_dir"],
                "compact": item["compact"],
            }
            for item in profile_results
        ],
        "beats_baseline_tuned": bool(selected_compact["test_average_precision"] > BASELINE_REFERENCE_AP),  # type: ignore[index]
    }
    save_json(summary, candidate_dir / "candidate_summary.json")
    log(
        f"{spec.candidate_id}: selected={summary['selected_profile']} "
        f"val_AP={selected_compact['validation_average_precision']:.6f} "
        f"test_AP={selected_compact['test_average_precision']:.6f} "
        f"delta={summary['delta_vs_baseline_tuned_test_ap']:+.6f}"
    )
    return summary


def stage_specs(
    stage: int,
    best_concat_dim: int | None,
    high_missing_threshold: float,
) -> list[CandidateSpec]:
    if stage == 1:
        return [
            CandidateSpec(1, "s1_replace_ld64", "Replace V with larger latent_dim=64.", "replace", latent_dim=64),
            CandidateSpec(1, "s1_replace_ld128", "Replace V with larger latent_dim=128.", "replace", latent_dim=128),
            CandidateSpec(1, "s1_replace_ld256", "Replace V with larger latent_dim=256.", "replace", latent_dim=256),
        ]
    if stage == 2:
        return [
            CandidateSpec(2, "s2_concat_ld32", "Keep original V and append AE latent_dim=32.", "concat", latent_dim=32),
            CandidateSpec(2, "s2_concat_ld64", "Keep original V and append AE latent_dim=64.", "concat", latent_dim=64),
            CandidateSpec(2, "s2_concat_ld128", "Keep original V and append AE latent_dim=128.", "concat", latent_dim=128),
        ]
    if stage == 3:
        latent_dim = best_concat_dim or 128
        return [
            CandidateSpec(
                3,
                f"s3_denoising_concat_ld{latent_dim}",
                f"Denoising AE latent_dim={latent_dim} appended to original V.",
                "concat",
                latent_dim=latent_dim,
                denoising=True,
            )
        ]
    if stage == 4:
        return [
            CandidateSpec(
                4,
                "s4_partial_recon_replace_high_missing",
                "Replace only high-missing V columns with partial AE reconstruction.",
                "partial_reconstruction",
                latent_dim=128,
                partial_strategy="replace",
                high_missing_threshold=high_missing_threshold,
            ),
            CandidateSpec(
                4,
                "s4_partial_recon_append_high_missing",
                "Append partial AE reconstructions only for high-missing V columns.",
                "partial_reconstruction",
                latent_dim=128,
                partial_strategy="append",
                high_missing_threshold=high_missing_threshold,
            ),
        ]
    raise ValueError(f"Unsupported stage: {stage}")


def build_candidate_matrices(
    data: LadderData,
    spec: CandidateSpec,
    output_dir: Path,
    cache_dir: Path,
    original_output: Path,
    latent_activation: str,
    denoising_noise_std: float,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    if spec.mode in {"replace", "concat"}:
        return build_latent_feature_matrices(
            data=data,
            spec=spec,
            cache_dir=cache_dir,
            original_output=original_output,
            latent_activation=latent_activation,
            denoising_noise_std=denoising_noise_std,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
    if spec.mode == "partial_reconstruction":
        return build_partial_reconstruction_matrices(
            data=data,
            spec=spec,
            output_dir=output_dir,
            latent_activation=latent_activation,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
    raise ValueError(f"Unsupported candidate mode: {spec.mode}")


def run_ladder(
    output_dir: Path,
    original_output: Path,
    baseline_params_path: Path,
    ae_params_path: Path,
    reference_scores_path: Path,
    latent_activation: str,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    denoising_noise_std: float,
    high_missing_threshold: float,
    n_bootstrap: int,
    n_jobs: int,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)
    cache_dir = ensure_dir(output_dir / "latent_cache")
    data = prepare_ladder_data(output_dir, seed=seed)
    reference_scores = load_reference_scores(reference_scores_path, data.y_test)
    reference_ap = float(average_precision_score(data.y_test, reference_scores))
    if abs(reference_ap - BASELINE_REFERENCE_AP) > 1e-12:
        log(f"Reference AP from scores is {reference_ap:.12f}; constant is {BASELINE_REFERENCE_AP:.12f}.")

    param_profiles = {
        "baseline_tuned": load_params(baseline_params_path, n_jobs=n_jobs, seed=seed),
        "ae_tuned_ld32": load_params(ae_params_path, n_jobs=n_jobs, seed=seed),
    }
    all_results: list[dict[str, object]] = []
    stopped_after_stage: int | None = None
    winning_candidate: dict[str, object] | None = None
    best_concat_dim: int | None = None

    for stage in [1, 2, 3, 4]:
        log(f"Starting stage {stage}.")
        stage_results: list[dict[str, object]] = []
        for spec in stage_specs(stage, best_concat_dim, high_missing_threshold):
            cached_summary_path = output_dir / spec.candidate_id / "candidate_summary.json"
            if cached_summary_path.exists():
                log(f"Loading cached candidate summary: {spec.candidate_id}.")
                result = load_json(cached_summary_path)
                all_results.append(result)
                stage_results.append(result)
                if result["beats_baseline_tuned"]:
                    winning_candidate = result
                    stopped_after_stage = stage
                    log(f"Stop rule met by cached {spec.candidate_id}.")
                    break
                continue

            X_train, X_valid, X_test, categorical_columns, feature_source_dir = build_candidate_matrices(
                data=data,
                spec=spec,
                output_dir=output_dir,
                cache_dir=cache_dir,
                original_output=original_output,
                latent_activation=latent_activation,
                denoising_noise_std=denoising_noise_std,
                max_epochs=ae_max_epochs,
                patience=ae_patience,
                batch_size=ae_batch_size,
                learning_rate=ae_learning_rate,
                seed=seed,
            )
            result = evaluate_candidate(
                data=data,
                spec=spec,
                output_dir=output_dir,
                param_profiles=param_profiles,
                reference_scores=reference_scores,
                n_bootstrap=n_bootstrap,
                seed=seed + stage,
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                categorical_columns=categorical_columns,
                feature_source_dir=feature_source_dir,
            )
            all_results.append(result)
            stage_results.append(result)
            del X_train, X_valid, X_test
            gc.collect()
            if result["beats_baseline_tuned"]:
                winning_candidate = result
                stopped_after_stage = stage
                log(f"Stop rule met by {spec.candidate_id}.")
                break

        if stage == 2 and stage_results:
            best_stage2 = max(
                stage_results,
                key=lambda item: item["selected_result"]["validation_average_precision"],  # type: ignore[index]
            )
            best_concat_dim = int(best_stage2["candidate"]["latent_dim"])  # type: ignore[index]
            log(f"Best concat dim by validation AP: {best_concat_dim}.")
        if winning_candidate is not None:
            break

    summary = {
        "experiment": "ae_feature_improvement_ladder",
        "output_dir": str(output_dir),
        "source_original_output": str(original_output),
        "split_strategy": "stratified_holdout",
        "split_ratios": {"train": TRAIN_RATIO, "validation": VALID_RATIO, "test": TEST_RATIO},
        "seed": seed,
        "sample_size": SAMPLE_SIZE,
        "reference": {
            "name": "original_proposal_baseline_tuned",
            "validation_average_precision": BASELINE_REFERENCE_VALID_AP,
            "test_average_precision": reference_ap,
            "reference_scores": str(reference_scores_path),
        },
        "stop_rule": "stop after first selected candidate with test AP > tuned LightGBM reference",
        "stopped_after_stage": stopped_after_stage,
        "winning_candidate": winning_candidate,
        "results": all_results,
    }
    save_json(summary, output_dir / "ladder_summary.json")
    print_ladder_summary(summary)
    return summary


def print_ladder_summary(summary: dict[str, object]) -> None:
    print()
    print("AE Feature Improvement Ladder")
    print("=============================")
    ref = summary["reference"]  # type: ignore[index]
    print(f"Reference tuned LightGBM test AP: {ref['test_average_precision']:.6f}")
    print()
    for result in summary["results"]:  # type: ignore[union-attr]
        selected = result["selected_result"]
        print(
            f"{result['candidate']['candidate_id']:40s} "
            f"profile={result['selected_profile']:15s} "
            f"val_AP={selected['validation_average_precision']:.6f} "
            f"test_AP={selected['test_average_precision']:.6f} "
            f"delta={result['delta_vs_baseline_tuned_test_ap']:+.6f}"
        )
    if summary["winning_candidate"] is None:
        print("\nNo candidate beat the tuned LightGBM reference.")
    else:
        winner = summary["winning_candidate"]
        print(f"\nWinner: {winner['candidate']['candidate_id']}")
    print(f"\nSaved: {summary['output_dir']}/ladder_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE feature-improvement ladder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stratified_reset/ae_feature_improvement_ladder"),
    )
    parser.add_argument("--original-output", type=Path, default=DEFAULT_ORIGINAL_OUTPUT)
    parser.add_argument(
        "--baseline-params-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "baseline_tpe_best_params.json",
    )
    parser.add_argument(
        "--ae-params-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "ae_latent_replacement_tpe_best_params.json",
    )
    parser.add_argument(
        "--reference-scores-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "baseline_tuned" / "baseline_tuned_test_scores.csv",
    )
    parser.add_argument("--latent-activation", choices=["relu", "linear"], default="relu")
    parser.add_argument("--ae-max-epochs", type=int, default=60)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--denoising-noise-std", type=float, default=0.10)
    parser.add_argument("--high-missing-threshold", type=float, default=0.75)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ladder(
        output_dir=args.output_dir,
        original_output=args.original_output,
        baseline_params_path=args.baseline_params_path,
        ae_params_path=args.ae_params_path,
        reference_scores_path=args.reference_scores_path,
        latent_activation=args.latent_activation,
        ae_max_epochs=args.ae_max_epochs,
        ae_patience=args.ae_patience,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        denoising_noise_std=args.denoising_noise_std,
        high_missing_threshold=args.high_missing_threshold,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
