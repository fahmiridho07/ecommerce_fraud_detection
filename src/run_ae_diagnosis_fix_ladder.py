"""Run diagnosis-driven AE fixes while staying inside Autoencoder + LightGBM.

This runner continues after ``run_ae_feature_improvement_ladder.py``. It uses
the same stratified split, same tuned LightGBM reference, and the same stop rule:
stop after the first selected candidate with test AP greater than the tuned
LightGBM reference.

The candidates are derived from the deep diagnosis:
- preserve or explicitly expose V missingness;
- avoid full V replacement;
- use masked reconstruction where missing targets are not treated as real zero;
- use reconstruction errors as auxiliary AE features;
- test a low-priority latent activation ablation.
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
from run_ae_feature_improvement_ladder import (
    BASELINE_REFERENCE_AP,
    BASELINE_REFERENCE_VALID_AP,
    DEFAULT_ORIGINAL_OUTPUT,
    CandidateSpec,
    LadderData,
    add_latent_features,
    fit_zero_zscore,
    get_latents,
    high_missing_v_columns,
    inverse_zero_zscore,
    load_json,
    load_params,
    load_reference_scores,
    load_test_scores,
    paired_bootstrap_ap_delta,
    prepare_ladder_data,
    train_with_profile,
)
from run_original_proposal_stratified import fit_lgbm
from train_baseline_lgbm import save_feature_importance
from utils import ensure_dir, log, save_json, set_seed


@dataclass(frozen=True)
class FixCandidate:
    stage: int
    candidate_id: str
    description: str
    kind: str
    high_missing_threshold: float | None = None
    top_k: int | None = None
    latent_dim: int | None = None
    latent_activation: str = "relu"
    input_missing_mask: bool = False
    masked_loss: bool = False
    replace_observed_only: bool = False
    append_missing_mask: bool = False
    append_error_features: bool = False


@dataclass
class ReconstructionBundle:
    selected_columns: list[str]
    scaler: dict[str, object]
    V_train: np.ndarray
    V_valid: np.ndarray
    V_test: np.ndarray
    missing_train: np.ndarray
    missing_valid: np.ndarray
    missing_test: np.ndarray
    recon_train_scaled: np.ndarray
    recon_valid_scaled: np.ndarray
    recon_test_scaled: np.ndarray
    source_dir: Path


def missing_masks(data: LadderData, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        data.X_train_raw.loc[:, columns].isna().to_numpy(dtype="float32"),
        data.X_valid_raw.loc[:, columns].isna().to_numpy(dtype="float32"),
        data.X_test_raw.loc[:, columns].isna().to_numpy(dtype="float32"),
    )


def select_v_by_importance_and_missingness(
    data: LadderData,
    baseline_importance_path: Path,
    high_missing_threshold: float,
    top_k: int,
    output_path: Path,
) -> list[str]:
    """Select high-missing V columns with importance and class-missingness gap."""
    importance = pd.read_csv(baseline_importance_path)
    importance = importance.loc[importance["feature"].isin(data.v_columns), ["feature", "importance_gain"]]
    gain_by_feature = importance.set_index("feature")["importance_gain"].astype(float)

    Xv = data.X_train_raw.loc[:, data.v_columns]
    y = data.y_train.to_numpy(dtype=int)
    missing_rate = Xv.isna().mean()
    nonfraud_missing = Xv.loc[y == 0].isna().mean()
    fraud_missing = Xv.loc[y == 1].isna().mean()
    class_gap = (fraud_missing - nonfraud_missing).abs()

    table = pd.DataFrame(
        {
            "feature": data.v_columns,
            "missing_rate": missing_rate.reindex(data.v_columns).to_numpy(dtype=float),
            "nonfraud_missing_rate": nonfraud_missing.reindex(data.v_columns).to_numpy(dtype=float),
            "fraud_missing_rate": fraud_missing.reindex(data.v_columns).to_numpy(dtype=float),
            "class_missingness_gap": class_gap.reindex(data.v_columns).to_numpy(dtype=float),
            "baseline_gain": gain_by_feature.reindex(data.v_columns).fillna(0.0).to_numpy(dtype=float),
        }
    )
    candidate_mask = table["missing_rate"] >= high_missing_threshold
    if not bool(candidate_mask.any()):
        candidate_mask = table["missing_rate"].rank(method="first", ascending=False) <= top_k
    candidates = table.loc[candidate_mask].copy()

    log_gain = np.log1p(candidates["baseline_gain"].to_numpy(dtype=float))
    gap = candidates["class_missingness_gap"].to_numpy(dtype=float)
    miss = candidates["missing_rate"].to_numpy(dtype=float)
    candidates["importance_score"] = log_gain / max(float(log_gain.max()), 1e-12)
    candidates["gap_score"] = gap / max(float(gap.max()), 1e-12)
    candidates["missing_score"] = miss / max(float(miss.max()), 1e-12)
    candidates["selection_score"] = (
        0.50 * candidates["importance_score"]
        + 0.40 * candidates["gap_score"]
        + 0.10 * candidates["missing_score"]
    )
    candidates = candidates.sort_values(
        ["selection_score", "baseline_gain", "class_missingness_gap"],
        ascending=False,
    )
    ensure_dir(output_path.parent)
    candidates.to_csv(output_path, index=False)
    selected = candidates.head(top_k)["feature"].tolist()
    if not selected:
        raise ValueError("No V columns selected for selective partial AE.")
    return selected


def build_mask_aware_autoencoder(
    input_dim: int,
    output_dim: int,
    latent_dim: int,
    learning_rate: float,
    latent_activation: str,
    masked_loss: bool,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="v_values_and_optional_mask")
    width = max(64, min(256, input_dim))
    x = keras.layers.Dense(width, activation="relu", name="encoder_dense_wide")(inputs)
    x = keras.layers.Dense(max(32, width // 2), activation="relu", name="encoder_dense_mid")(x)
    latent = keras.layers.Dense(latent_dim, activation=latent_activation, name="latent")(x)
    x = keras.layers.Dense(max(32, width // 2), activation="relu", name="decoder_dense_mid")(latent)
    x = keras.layers.Dense(width, activation="relu", name="decoder_dense_wide")(x)
    outputs = keras.layers.Dense(output_dim, activation="linear", name="reconstruction")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="diagnosis_partial_v_autoencoder")

    if masked_loss:

        def observed_only_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            target = y_true[:, :output_dim]
            observed = y_true[:, output_dim:]
            squared_error = tf.square(target - y_pred) * observed
            denominator = tf.reduce_sum(observed, axis=-1) + tf.keras.backend.epsilon()
            return tf.reduce_sum(squared_error, axis=-1) / denominator

        loss = observed_only_mse
    else:
        loss = "mse"

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)
    return model


def train_or_load_reconstruction(
    data: LadderData,
    spec: FixCandidate,
    output_dir: Path,
    baseline_importance_path: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> ReconstructionBundle:
    candidate_dir = ensure_dir(output_dir / "feature_cache" / spec.candidate_id)
    contract_path = candidate_dir / "feature_contract.json"
    train_path = candidate_dir / "recon_train_scaled.npy"
    valid_path = candidate_dir / "recon_valid_scaled.npy"
    test_path = candidate_dir / "recon_test_scaled.npy"

    if spec.kind == "selective_partial_reconstruction":
        if contract_path.exists():
            selected_columns = list(load_json(contract_path)["selected_columns"])
        else:
            selected_columns = select_v_by_importance_and_missingness(
                data=data,
                baseline_importance_path=baseline_importance_path,
                high_missing_threshold=spec.high_missing_threshold or 0.75,
                top_k=spec.top_k or 64,
                output_path=candidate_dir / "selection_metrics.csv",
            )
    else:
        selected_columns = high_missing_v_columns(
            data,
            threshold=spec.high_missing_threshold or 0.75,
            fallback_top_k=spec.top_k or 80,
        )

    V_train, V_valid, V_test, scaler = fit_zero_zscore(
        data.X_train_raw,
        data.X_valid_raw,
        data.X_test_raw,
        selected_columns,
    )
    missing_train, missing_valid, missing_test = missing_masks(data, selected_columns)

    if train_path.exists() and valid_path.exists() and test_path.exists() and contract_path.exists():
        log(f"Loading cached reconstruction features: {spec.candidate_id}.")
        return ReconstructionBundle(
            selected_columns=selected_columns,
            scaler=scaler,
            V_train=V_train,
            V_valid=V_valid,
            V_test=V_test,
            missing_train=missing_train,
            missing_valid=missing_valid,
            missing_test=missing_test,
            recon_train_scaled=np.load(train_path),
            recon_valid_scaled=np.load(valid_path),
            recon_test_scaled=np.load(test_path),
            source_dir=candidate_dir,
        )

    latent_dim = spec.latent_dim or min(128, max(16, len(selected_columns) // 2))
    input_train = V_train
    input_valid = V_valid
    input_test = V_test
    if spec.input_missing_mask:
        input_train = np.concatenate([V_train, missing_train], axis=1).astype("float32")
        input_valid = np.concatenate([V_valid, missing_valid], axis=1).astype("float32")
        input_test = np.concatenate([V_test, missing_test], axis=1).astype("float32")

    if spec.masked_loss:
        target_train = np.concatenate([V_train, 1.0 - missing_train], axis=1).astype("float32")
        target_valid = np.concatenate([V_valid, 1.0 - missing_valid], axis=1).astype("float32")
    else:
        target_train = V_train
        target_valid = V_valid

    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    model = build_mask_aware_autoencoder(
        input_dim=input_train.shape[1],
        output_dim=V_train.shape[1],
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        latent_activation=spec.latent_activation,
        masked_loss=spec.masked_loss,
    )
    log(
        f"Training {spec.candidate_id}: columns={len(selected_columns)} "
        f"input_dim={input_train.shape[1]} latent_dim={latent_dim} masked_loss={spec.masked_loss}."
    )
    history = model.fit(
        input_train,
        target_train,
        validation_data=(input_valid, target_valid),
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
    history_df.to_csv(candidate_dir / "ae_training_history.csv", index=False)
    model.save(candidate_dir / "autoencoder.keras", include_optimizer=False)

    recon_train = model.predict(input_train, batch_size=batch_size, verbose=0).astype("float32")
    recon_valid = model.predict(input_valid, batch_size=batch_size, verbose=0).astype("float32")
    recon_test = model.predict(input_test, batch_size=batch_size, verbose=0).astype("float32")
    np.save(train_path, recon_train)
    np.save(valid_path, recon_valid)
    np.save(test_path, recon_test)
    save_json(
        {
            "candidate": asdict(spec),
            "selected_columns": selected_columns,
            "selected_count": len(selected_columns),
            "scaler": scaler,
            "latent_dim": latent_dim,
            "input_dim": int(input_train.shape[1]),
            "target": "observed-only masked MSE" if spec.masked_loss else "zero-imputed MSE",
        },
        contract_path,
    )
    tf.keras.backend.clear_session()
    return ReconstructionBundle(
        selected_columns=selected_columns,
        scaler=scaler,
        V_train=V_train,
        V_valid=V_valid,
        V_test=V_test,
        missing_train=missing_train,
        missing_valid=missing_valid,
        missing_test=missing_test,
        recon_train_scaled=recon_train,
        recon_valid_scaled=recon_valid,
        recon_test_scaled=recon_test,
        source_dir=candidate_dir,
    )


def append_missing_mask_features(
    base: pd.DataFrame,
    columns: list[str],
    missing: np.ndarray,
) -> pd.DataFrame:
    names = [f"ae_missing_{column}" for column in columns]
    mask_df = pd.DataFrame(missing, columns=names, index=base.index)
    return pd.concat([base.reset_index(drop=True), mask_df.reset_index(drop=True)], axis=1)


def replace_selected_v(
    base: pd.DataFrame,
    columns: list[str],
    recon_raw: np.ndarray,
    missing: np.ndarray,
    observed_only: bool,
) -> pd.DataFrame:
    transformed = base.copy()
    for index, column in enumerate(columns):
        values = recon_raw[:, index]
        if observed_only:
            original = transformed[column].to_numpy(copy=True)
            observed = missing[:, index] < 0.5
            original[observed] = values[observed]
            transformed[column] = original
        else:
            transformed[column] = values
    return transformed


def build_reconstruction_error_features(
    bundle: ReconstructionBundle,
    split: str,
) -> pd.DataFrame:
    if split == "train":
        values = bundle.V_train
        recon = bundle.recon_train_scaled
        missing = bundle.missing_train
    elif split == "valid":
        values = bundle.V_valid
        recon = bundle.recon_valid_scaled
        missing = bundle.missing_valid
    elif split == "test":
        values = bundle.V_test
        recon = bundle.recon_test_scaled
        missing = bundle.missing_test
    else:
        raise ValueError(f"Unsupported split: {split}")

    observed = 1.0 - missing
    abs_error = np.abs(values - recon).astype("float32")
    squared_error = np.square(values - recon).astype("float32")
    observed_count = observed.sum(axis=1)
    denominator = np.maximum(observed_count, 1.0)

    per_feature_error = np.where(observed > 0.5, abs_error, np.nan).astype("float32")
    features = pd.DataFrame(
        per_feature_error,
        columns=[f"ae_abs_err_{column}" for column in bundle.selected_columns],
    )
    features["ae_recon_mse_observed"] = (squared_error * observed).sum(axis=1) / denominator
    features["ae_recon_mae_observed"] = (abs_error * observed).sum(axis=1) / denominator
    features["ae_recon_max_abs_observed"] = np.nanmax(per_feature_error, axis=1)
    features["ae_recon_missing_rate"] = missing.mean(axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def build_partial_replacement_matrices(
    data: LadderData,
    bundle: ReconstructionBundle,
    spec: FixCandidate,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    train_raw = inverse_zero_zscore(bundle.recon_train_scaled, bundle.scaler)
    valid_raw = inverse_zero_zscore(bundle.recon_valid_scaled, bundle.scaler)
    test_raw = inverse_zero_zscore(bundle.recon_test_scaled, bundle.scaler)

    X_train = replace_selected_v(
        data.X_base_train,
        bundle.selected_columns,
        train_raw,
        bundle.missing_train,
        observed_only=spec.replace_observed_only,
    )
    X_valid = replace_selected_v(
        data.X_base_valid,
        bundle.selected_columns,
        valid_raw,
        bundle.missing_valid,
        observed_only=spec.replace_observed_only,
    )
    X_test = replace_selected_v(
        data.X_base_test,
        bundle.selected_columns,
        test_raw,
        bundle.missing_test,
        observed_only=spec.replace_observed_only,
    )
    if spec.append_missing_mask:
        X_train = append_missing_mask_features(X_train, bundle.selected_columns, bundle.missing_train)
        X_valid = append_missing_mask_features(X_valid, bundle.selected_columns, bundle.missing_valid)
        X_test = append_missing_mask_features(X_test, bundle.selected_columns, bundle.missing_test)
    return X_train, X_valid, X_test, data.baseline_categorical_columns, bundle.source_dir


def build_error_feature_matrices(
    data: LadderData,
    bundle: ReconstructionBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    def append(base: pd.DataFrame, error_features: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([base.reset_index(drop=True), error_features.reset_index(drop=True)], axis=1)

    return (
        append(data.X_base_train, build_reconstruction_error_features(bundle, "train")),
        append(data.X_base_valid, build_reconstruction_error_features(bundle, "valid")),
        append(data.X_base_test, build_reconstruction_error_features(bundle, "test")),
        data.baseline_categorical_columns,
        bundle.source_dir,
    )


def build_linear_latent_concat_matrices(
    data: LadderData,
    spec: FixCandidate,
    output_dir: Path,
    original_output: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    latent_train, latent_valid, latent_test, source_dir = get_latents(
        data=data,
        cache_dir=ensure_dir(output_dir / "latent_cache"),
        original_output=original_output,
        latent_dim=spec.latent_dim or 32,
        latent_activation=spec.latent_activation,
        denoising=False,
        denoising_noise_std=0.0,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + spec.stage + (spec.latent_dim or 32),
    )
    X_train, X_valid, X_test = add_latent_features(
        data.X_base_train,
        data.X_base_valid,
        data.X_base_test,
        latent_train,
        latent_valid,
        latent_test,
        "ae_linear_latent",
    )
    return X_train, X_valid, X_test, data.baseline_categorical_columns, source_dir


def build_candidate_matrices(
    data: LadderData,
    spec: FixCandidate,
    output_dir: Path,
    original_output: Path,
    baseline_importance_path: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    if spec.kind in {"partial_reconstruction", "selective_partial_reconstruction"}:
        bundle = train_or_load_reconstruction(
            data=data,
            spec=spec,
            output_dir=output_dir,
            baseline_importance_path=baseline_importance_path,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + spec.stage,
        )
        return build_partial_replacement_matrices(data, bundle, spec)
    if spec.kind == "reconstruction_error":
        source_spec = FixCandidate(
            stage=spec.stage,
            candidate_id="shared_high_missing_maskedloss_reconstruction",
            description="Shared masked-loss high-missing V reconstruction for error features.",
            kind="partial_reconstruction",
            high_missing_threshold=spec.high_missing_threshold,
            latent_dim=spec.latent_dim,
            input_missing_mask=True,
            masked_loss=True,
            append_missing_mask=True,
        )
        bundle = train_or_load_reconstruction(
            data=data,
            spec=source_spec,
            output_dir=output_dir,
            baseline_importance_path=baseline_importance_path,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + spec.stage,
        )
        return build_error_feature_matrices(data, bundle)
    if spec.kind == "linear_latent_concat":
        return build_linear_latent_concat_matrices(
            data=data,
            spec=spec,
            output_dir=output_dir,
            original_output=original_output,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
    raise ValueError(f"Unsupported candidate kind: {spec.kind}")


def evaluate_fix_candidate(
    data: LadderData,
    spec: FixCandidate,
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


def candidate_sequence(high_missing_threshold: float, selective_top_k: int) -> list[FixCandidate]:
    return [
        FixCandidate(
            stage=1,
            candidate_id="fix1_partial_mask_recon_replace",
            description="Mask-aware partial AE: replace high-missing V reconstruction and append missing masks.",
            kind="partial_reconstruction",
            high_missing_threshold=high_missing_threshold,
            latent_dim=128,
            input_missing_mask=True,
            masked_loss=False,
            replace_observed_only=False,
            append_missing_mask=True,
        ),
        FixCandidate(
            stage=2,
            candidate_id="fix2_partial_maskedloss_observed_replace",
            description="Masked-loss partial AE: replace only observed high-missing V values and append missing masks.",
            kind="partial_reconstruction",
            high_missing_threshold=high_missing_threshold,
            latent_dim=128,
            input_missing_mask=True,
            masked_loss=True,
            replace_observed_only=True,
            append_missing_mask=True,
        ),
        FixCandidate(
            stage=3,
            candidate_id=f"fix3_select{selective_top_k}_maskedloss_observed_replace",
            description="Selective masked-loss partial AE using high-missing, importance, and class-gap V columns.",
            kind="selective_partial_reconstruction",
            high_missing_threshold=high_missing_threshold,
            top_k=selective_top_k,
            latent_dim=64,
            input_missing_mask=True,
            masked_loss=True,
            replace_observed_only=True,
            append_missing_mask=True,
        ),
        FixCandidate(
            stage=4,
            candidate_id="fix4_recon_error_append_high_missing",
            description="Keep original V and append high-missing AE reconstruction-error features.",
            kind="reconstruction_error",
            high_missing_threshold=high_missing_threshold,
            latent_dim=128,
            append_error_features=True,
        ),
        FixCandidate(
            stage=5,
            candidate_id="fix5_linear_latent_concat_ld32",
            description="Keep original V and append full-V AE latent_dim=32 with linear latent activation.",
            kind="linear_latent_concat",
            latent_dim=32,
            latent_activation="linear",
        ),
    ]


def run_fix_ladder(
    output_dir: Path,
    original_output: Path,
    baseline_params_path: Path,
    ae_params_path: Path,
    reference_scores_path: Path,
    baseline_importance_path: Path,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    high_missing_threshold: float,
    selective_top_k: int,
    n_bootstrap: int,
    n_jobs: int,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)
    data = prepare_ladder_data(output_dir, seed=seed)
    reference_scores = load_reference_scores(reference_scores_path, data.y_test)
    reference_ap = float(average_precision_score(data.y_test, reference_scores))
    if abs(reference_ap - BASELINE_REFERENCE_AP) > 1e-12:
        log(f"Reference AP from scores is {reference_ap:.12f}; constant is {BASELINE_REFERENCE_AP:.12f}.")

    param_profiles = {
        "baseline_tuned": load_params(baseline_params_path, n_jobs=n_jobs, seed=seed),
        "ae_tuned_ld32": load_params(ae_params_path, n_jobs=n_jobs, seed=seed),
    }

    results: list[dict[str, object]] = []
    winning_candidate: dict[str, object] | None = None
    stopped_after_stage: int | None = None
    for spec in candidate_sequence(high_missing_threshold, selective_top_k):
        cached_summary_path = output_dir / spec.candidate_id / "candidate_summary.json"
        if cached_summary_path.exists():
            log(f"Loading cached candidate summary: {spec.candidate_id}.")
            result = load_json(cached_summary_path)
        else:
            log(f"Starting candidate {spec.candidate_id}.")
            X_train, X_valid, X_test, categorical_columns, feature_source_dir = build_candidate_matrices(
                data=data,
                spec=spec,
                output_dir=output_dir,
                original_output=original_output,
                baseline_importance_path=baseline_importance_path,
                max_epochs=ae_max_epochs,
                patience=ae_patience,
                batch_size=ae_batch_size,
                learning_rate=ae_learning_rate,
                seed=seed,
            )
            result = evaluate_fix_candidate(
                data=data,
                spec=spec,
                output_dir=output_dir,
                param_profiles=param_profiles,
                reference_scores=reference_scores,
                n_bootstrap=n_bootstrap,
                seed=seed + spec.stage,
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                categorical_columns=categorical_columns,
                feature_source_dir=feature_source_dir,
            )
            del X_train, X_valid, X_test
            gc.collect()

        results.append(result)
        if result["beats_baseline_tuned"]:
            winning_candidate = result
            stopped_after_stage = int(result["candidate"]["stage"])  # type: ignore[index]
            log(f"Stop rule met by {result['candidate']['candidate_id']}.")  # type: ignore[index]
            break

    summary = {
        "experiment": "ae_diagnosis_fix_ladder",
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
        "results": results,
    }
    save_json(summary, output_dir / "fix_ladder_summary.json")
    print_fix_summary(summary)
    return summary


def print_fix_summary(summary: dict[str, object]) -> None:
    print()
    print("AE Diagnosis-Driven Fix Ladder")
    print("==============================")
    ref = summary["reference"]  # type: ignore[index]
    print(f"Reference tuned LightGBM test AP: {ref['test_average_precision']:.6f}")
    print()
    for result in summary["results"]:  # type: ignore[union-attr]
        selected = result["selected_result"]
        print(
            f"{result['candidate']['candidate_id']:46s} "
            f"profile={result['selected_profile']:15s} "
            f"val_AP={selected['validation_average_precision']:.6f} "
            f"test_AP={selected['test_average_precision']:.6f} "
            f"delta={result['delta_vs_baseline_tuned_test_ap']:+.6f}"
        )
    if summary["winning_candidate"] is None:
        print("\nNo candidate beat the tuned LightGBM reference.")
    else:
        winner = summary["winning_candidate"]
        print(f"\nWinner: {winner['candidate']['candidate_id']}")  # type: ignore[index]
    print(f"\nSaved: {summary['output_dir']}/fix_ladder_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnosis-driven AE fix ladder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stratified_reset/ae_diagnosis_fix_ladder"),
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
    parser.add_argument(
        "--baseline-importance-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "baseline_tuned" / "feature_importance.csv",
    )
    parser.add_argument("--ae-max-epochs", type=int, default=60)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--high-missing-threshold", type=float, default=0.75)
    parser.add_argument("--selective-top-k", type=int, default=64)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_fix_ladder(
        output_dir=args.output_dir,
        original_output=args.original_output,
        baseline_params_path=args.baseline_params_path,
        ae_params_path=args.ae_params_path,
        reference_scores_path=args.reference_scores_path,
        baseline_importance_path=args.baseline_importance_path,
        ae_max_epochs=args.ae_max_epochs,
        ae_patience=args.ae_patience,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        high_missing_threshold=args.high_missing_threshold,
        selective_top_k=args.selective_top_k,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
