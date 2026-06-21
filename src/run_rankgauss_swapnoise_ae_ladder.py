"""Run RankGauss + swap-noise Autoencoder fixes for the proposal path.

This runner operationalizes the external deep-research suggestions without
changing the thesis method family: Autoencoder remains the representation
learner and LightGBM remains the classifier. It adds:

- V-feature reduction by missingness group and within-group correlation;
- train-only observed-value RankGauss transforms for selected V columns;
- swap-noise denoising Autoencoder;
- observed-only masked MSE;
- LightGBM evaluation using the existing PR-AUC-centered tuned profiles.

The runner is intentionally separate from completed canonical results so these
diagnostic variants can be tested without rewriting the current Bab 4 record.
"""

from __future__ import annotations

import argparse
import gc
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
from rankgauss_ae_utils import (
    RankGaussColumn,
    fit_observed_rankgauss,
    inverse_observed_rankgauss,
    observed_reconstruction_error_features,
    select_v_columns_by_missingness_correlation,
    transform_observed_rankgauss,
)
from run_ae_feature_improvement_ladder import (
    BASELINE_REFERENCE_AP,
    BASELINE_REFERENCE_VALID_AP,
    DEFAULT_ORIGINAL_OUTPUT,
    LadderData,
    add_latent_features,
    load_json,
    load_params,
    load_reference_scores,
    paired_bootstrap_ap_delta,
    prepare_ladder_data,
    train_with_profile,
)
from utils import ensure_dir, log, save_json, set_seed


@dataclass(frozen=True)
class RankGaussCandidate:
    stage: int
    candidate_id: str
    description: str
    kind: str
    latent_dim: int = 64
    append_latent: bool = True
    append_error: bool = True
    replace_observed_only: bool = False
    append_missing_mask: bool = False


@dataclass
class RankGaussBundle:
    selected_columns: list[str]
    fitted: list[RankGaussColumn]
    V_train: np.ndarray
    V_valid: np.ndarray
    V_test: np.ndarray
    observed_train: np.ndarray
    observed_valid: np.ndarray
    observed_test: np.ndarray
    latent_train: np.ndarray
    latent_valid: np.ndarray
    latent_test: np.ndarray
    recon_train: np.ndarray
    recon_valid: np.ndarray
    recon_test: np.ndarray
    source_dir: Path


class SwapNoise(keras.layers.Layer):
    """Column-wise swap noise for tabular denoising autoencoders."""

    def __init__(self, swap_rate: float, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= swap_rate <= 1.0:
            raise ValueError("swap_rate must be in [0, 1].")
        self.swap_rate = float(swap_rate)

    def call(self, inputs, training=None):  # type: ignore[override]
        if not training or self.swap_rate <= 0.0:
            return inputs
        batch_size = tf.shape(inputs)[0]
        shuffled_rows = tf.random.shuffle(tf.range(batch_size))
        shuffled = tf.gather(inputs, shuffled_rows)
        swap_mask = tf.random.uniform(tf.shape(inputs)) < self.swap_rate
        return tf.where(swap_mask, shuffled, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"swap_rate": self.swap_rate})
        return config


def build_swapnoise_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
    swap_rate: float,
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(input_dim,), name="rankgauss_v_values")
    x = SwapNoise(swap_rate, name="swap_noise")(inputs)
    width = max(64, min(256, input_dim * 2))
    x = keras.layers.Dense(width, activation="relu", name="encoder_dense_wide")(x)
    x = keras.layers.Dense(max(32, width // 2), activation="relu", name="encoder_dense_mid")(x)
    latent = keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)
    x = keras.layers.Dense(max(32, width // 2), activation="relu", name="decoder_dense_mid")(latent)
    x = keras.layers.Dense(width, activation="relu", name="decoder_dense_wide")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="rankgauss_swapnoise_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="rankgauss_swapnoise_encoder")

    def observed_only_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target = y_true[:, :input_dim]
        observed = y_true[:, input_dim:]
        squared_error = tf.square(target - y_pred) * observed
        denom = tf.reduce_sum(observed, axis=-1) + tf.keras.backend.epsilon()
        return tf.reduce_sum(squared_error, axis=-1) / denom

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=observed_only_mse,
    )
    return autoencoder, encoder


def candidate_sequence() -> list[RankGaussCandidate]:
    return [
        RankGaussCandidate(
            stage=1,
            candidate_id="rg_s1_swapdae_latent_error_append",
            description="Keep original features; append RankGauss swap-noise AE latent and observed reconstruction errors.",
            kind="append_latent_error",
            latent_dim=64,
            append_latent=True,
            append_error=True,
        ),
        RankGaussCandidate(
            stage=2,
            candidate_id="rg_s2_swapdae_observed_replace_mask",
            description="Replace observed selected V values with inverse RankGauss reconstruction and append V missing masks.",
            kind="observed_replace",
            latent_dim=64,
            replace_observed_only=True,
            append_missing_mask=True,
        ),
    ]


def train_or_load_rankgauss_bundle(
    data: LadderData,
    spec: RankGaussCandidate,
    output_dir: Path,
    *,
    corr_threshold: float,
    max_v_columns: int,
    max_quantiles: int,
    clip_value: float,
    swap_rate: float,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    seed: int,
) -> RankGaussBundle:
    cache_dir = ensure_dir(output_dir / "feature_cache" / spec.candidate_id)
    contract_path = cache_dir / "rankgauss_contract.json"
    fitted_path = cache_dir / "rankgauss_transformers.pkl"
    paths = {
        "latent_train": cache_dir / "latent_train.npy",
        "latent_valid": cache_dir / "latent_valid.npy",
        "latent_test": cache_dir / "latent_test.npy",
        "recon_train": cache_dir / "recon_train.npy",
        "recon_valid": cache_dir / "recon_valid.npy",
        "recon_test": cache_dir / "recon_test.npy",
    }
    cache_ready = (
        contract_path.exists()
        and fitted_path.exists()
        and all(path.exists() for path in paths.values())
    )

    if contract_path.exists():
        selected_columns = list(load_json(contract_path)["selected_columns"])
    else:
        selected_columns, selection_report = select_v_columns_by_missingness_correlation(
            data.X_train_raw,
            data.v_columns,
            corr_threshold=corr_threshold,
            max_columns=max_v_columns,
        )
        selection_report.to_csv(cache_dir / "v_selection_report.csv", index=False)

    if fitted_path.exists():
        fitted: list[RankGaussColumn] = joblib.load(fitted_path)
    else:
        fitted = fit_observed_rankgauss(
            data.X_train_raw,
            selected_columns,
            max_quantiles=max_quantiles,
            random_state=seed,
        )
        joblib.dump(fitted, fitted_path)

    V_train, observed_train = transform_observed_rankgauss(data.X_train_raw, fitted, clip_value=clip_value)
    V_valid, observed_valid = transform_observed_rankgauss(data.X_valid_raw, fitted, clip_value=clip_value)
    V_test, observed_test = transform_observed_rankgauss(data.X_test_raw, fitted, clip_value=clip_value)

    if cache_ready:
        log(f"Loading cached RankGauss swap-noise features: {spec.candidate_id}.")
        return RankGaussBundle(
            selected_columns=selected_columns,
            fitted=fitted,
            V_train=V_train,
            V_valid=V_valid,
            V_test=V_test,
            observed_train=observed_train,
            observed_valid=observed_valid,
            observed_test=observed_test,
            latent_train=np.load(paths["latent_train"]),
            latent_valid=np.load(paths["latent_valid"]),
            latent_test=np.load(paths["latent_test"]),
            recon_train=np.load(paths["recon_train"]),
            recon_valid=np.load(paths["recon_valid"]),
            recon_test=np.load(paths["recon_test"]),
            source_dir=cache_dir,
        )

    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    log(
        f"Training {spec.candidate_id}: selected_v={len(selected_columns)} "
        f"latent_dim={spec.latent_dim} swap_rate={swap_rate} masked_loss=True."
    )
    autoencoder, encoder = build_swapnoise_autoencoder(
        input_dim=V_train.shape[1],
        latent_dim=spec.latent_dim,
        learning_rate=ae_learning_rate,
        swap_rate=swap_rate,
    )
    target_train = np.concatenate([V_train, observed_train], axis=1).astype("float32")
    target_valid = np.concatenate([V_valid, observed_valid], axis=1).astype("float32")
    history = autoencoder.fit(
        V_train,
        target_train,
        validation_data=(V_valid, target_valid),
        epochs=ae_max_epochs,
        batch_size=ae_batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=ae_patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(cache_dir / "ae_training_history.csv", index=False)
    autoencoder.save(cache_dir / "autoencoder.keras", include_optimizer=False)
    encoder.save(cache_dir / "encoder.keras", include_optimizer=False)

    latent_train = encoder.predict(V_train, batch_size=ae_batch_size, verbose=0).astype("float32")
    latent_valid = encoder.predict(V_valid, batch_size=ae_batch_size, verbose=0).astype("float32")
    latent_test = encoder.predict(V_test, batch_size=ae_batch_size, verbose=0).astype("float32")
    recon_train = autoencoder.predict(V_train, batch_size=ae_batch_size, verbose=0).astype("float32")
    recon_valid = autoencoder.predict(V_valid, batch_size=ae_batch_size, verbose=0).astype("float32")
    recon_test = autoencoder.predict(V_test, batch_size=ae_batch_size, verbose=0).astype("float32")
    np.save(paths["latent_train"], latent_train)
    np.save(paths["latent_valid"], latent_valid)
    np.save(paths["latent_test"], latent_test)
    np.save(paths["recon_train"], recon_train)
    np.save(paths["recon_valid"], recon_valid)
    np.save(paths["recon_test"], recon_test)

    save_json(
        {
            "candidate": asdict(spec),
            "selected_columns": selected_columns,
            "selected_count": len(selected_columns),
            "rankgauss": {
                "fit_scope": "train observed values only",
                "missing_placeholder_after_transform": 0.0,
                "max_quantiles": max_quantiles,
                "clip_value": clip_value,
            },
            "autoencoder": {
                "denoising": "swap_noise",
                "swap_rate": swap_rate,
                "loss": "observed-only masked MSE",
                "latent_dim": spec.latent_dim,
            },
        },
        contract_path,
    )
    tf.keras.backend.clear_session()
    return RankGaussBundle(
        selected_columns=selected_columns,
        fitted=fitted,
        V_train=V_train,
        V_valid=V_valid,
        V_test=V_test,
        observed_train=observed_train,
        observed_valid=observed_valid,
        observed_test=observed_test,
        latent_train=latent_train,
        latent_valid=latent_valid,
        latent_test=latent_test,
        recon_train=recon_train,
        recon_valid=recon_valid,
        recon_test=recon_test,
        source_dir=cache_dir,
    )


def append_error_features(
    base_train: pd.DataFrame,
    base_valid: pd.DataFrame,
    base_test: pd.DataFrame,
    bundle: RankGaussBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def append(base: pd.DataFrame, values: np.ndarray, recon: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
        error_df = observed_reconstruction_error_features(
            values,
            recon,
            observed,
            prefix="rg_swapdae",
        )
        return pd.concat([base.reset_index(drop=True), error_df.reset_index(drop=True)], axis=1)

    return (
        append(base_train, bundle.V_train, bundle.recon_train, bundle.observed_train),
        append(base_valid, bundle.V_valid, bundle.recon_valid, bundle.observed_valid),
        append(base_test, bundle.V_test, bundle.recon_test, bundle.observed_test),
    )


def append_missing_masks(
    base: pd.DataFrame,
    selected_columns: list[str],
    observed: np.ndarray,
) -> pd.DataFrame:
    missing = 1.0 - observed
    names = [f"rg_missing_{column}" for column in selected_columns]
    mask_df = pd.DataFrame(missing, columns=names)
    return pd.concat([base.reset_index(drop=True), mask_df.reset_index(drop=True)], axis=1)


def replace_observed_v_values(
    base: pd.DataFrame,
    selected_columns: list[str],
    recon_raw: np.ndarray,
    observed: np.ndarray,
) -> pd.DataFrame:
    transformed = base.copy()
    for index, column in enumerate(selected_columns):
        if column not in transformed.columns:
            continue
        values = transformed[column].to_numpy(copy=True)
        observed_mask = observed[:, index] > 0.5
        values[observed_mask] = recon_raw[observed_mask, index]
        transformed[column] = values
    return transformed


def build_candidate_matrices(
    data: LadderData,
    spec: RankGaussCandidate,
    bundle: RankGaussBundle,
    *,
    clip_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    if spec.kind == "append_latent_error":
        X_train, X_valid, X_test = data.X_base_train, data.X_base_valid, data.X_base_test
        if spec.append_latent:
            X_train, X_valid, X_test = add_latent_features(
                X_train,
                X_valid,
                X_test,
                bundle.latent_train,
                bundle.latent_valid,
                bundle.latent_test,
                "rg_swapdae_latent",
            )
        if spec.append_error:
            X_train, X_valid, X_test = append_error_features(X_train, X_valid, X_test, bundle)
        return X_train, X_valid, X_test, data.baseline_categorical_columns, bundle.source_dir

    if spec.kind == "observed_replace":
        recon_train_raw = inverse_observed_rankgauss(bundle.recon_train, bundle.fitted, clip_value=clip_value)
        recon_valid_raw = inverse_observed_rankgauss(bundle.recon_valid, bundle.fitted, clip_value=clip_value)
        recon_test_raw = inverse_observed_rankgauss(bundle.recon_test, bundle.fitted, clip_value=clip_value)
        X_train = replace_observed_v_values(
            data.X_base_train,
            bundle.selected_columns,
            recon_train_raw,
            bundle.observed_train,
        )
        X_valid = replace_observed_v_values(
            data.X_base_valid,
            bundle.selected_columns,
            recon_valid_raw,
            bundle.observed_valid,
        )
        X_test = replace_observed_v_values(
            data.X_base_test,
            bundle.selected_columns,
            recon_test_raw,
            bundle.observed_test,
        )
        if spec.append_missing_mask:
            X_train = append_missing_masks(X_train, bundle.selected_columns, bundle.observed_train)
            X_valid = append_missing_masks(X_valid, bundle.selected_columns, bundle.observed_valid)
            X_test = append_missing_masks(X_test, bundle.selected_columns, bundle.observed_test)
        return X_train, X_valid, X_test, data.baseline_categorical_columns, bundle.source_dir

    raise ValueError(f"Unsupported candidate kind: {spec.kind}")


def evaluate_candidate(
    data: LadderData,
    spec: RankGaussCandidate,
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
        seed=seed + spec.stage,
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


def run_ladder(
    output_dir: Path,
    original_output: Path,
    baseline_params_path: Path,
    ae_params_path: Path,
    reference_scores_path: Path,
    corr_threshold: float,
    max_v_columns: int,
    max_quantiles: int,
    rankgauss_clip: float,
    swap_rate: float,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
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
    for spec in candidate_sequence():
        cached_summary_path = output_dir / spec.candidate_id / "candidate_summary.json"
        if cached_summary_path.exists():
            log(f"Loading cached candidate summary: {spec.candidate_id}.")
            result = load_json(cached_summary_path)
        else:
            bundle = train_or_load_rankgauss_bundle(
                data=data,
                spec=spec,
                output_dir=output_dir,
                corr_threshold=corr_threshold,
                max_v_columns=max_v_columns,
                max_quantiles=max_quantiles,
                clip_value=rankgauss_clip,
                swap_rate=swap_rate,
                ae_max_epochs=ae_max_epochs,
                ae_patience=ae_patience,
                ae_batch_size=ae_batch_size,
                ae_learning_rate=ae_learning_rate,
                seed=seed + spec.stage,
            )
            X_train, X_valid, X_test, categorical_columns, feature_source_dir = build_candidate_matrices(
                data=data,
                spec=spec,
                bundle=bundle,
                clip_value=rankgauss_clip,
            )
            result = evaluate_candidate(
                data=data,
                spec=spec,
                output_dir=output_dir,
                param_profiles=param_profiles,
                reference_scores=reference_scores,
                n_bootstrap=n_bootstrap,
                seed=seed,
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                categorical_columns=categorical_columns,
                feature_source_dir=feature_source_dir,
            )
            del X_train, X_valid, X_test, bundle
            gc.collect()

        results.append(result)
        if result["beats_baseline_tuned"]:
            winning_candidate = result
            stopped_after_stage = int(result["candidate"]["stage"])  # type: ignore[index]
            log(f"Stop rule met by {result['candidate']['candidate_id']}.")  # type: ignore[index]
            break

    summary = {
        "experiment": "rankgauss_swapnoise_ae_ladder",
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
        "deep_research_adaptations": {
            "v_reduction": "missingness-group correlation pruning",
            "rankgauss": "QuantileTransformer(output_distribution='normal') fit on train observed values only",
            "denoising": "swap noise",
            "loss": "observed-only masked MSE",
            "lightgbm_selection": "validation Average Precision / PR-AUC",
        },
        "stop_rule": "stop after first selected candidate with test AP > tuned LightGBM reference",
        "stopped_after_stage": stopped_after_stage,
        "winning_candidate": winning_candidate,
        "results": results,
    }
    save_json(summary, output_dir / "rankgauss_swapnoise_summary.json")
    print_summary(summary)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print()
    print("RankGauss + Swap-Noise AE Ladder")
    print("================================")
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
        print(f"\nWinner: {winner['candidate']['candidate_id']}")  # type: ignore[index]
    print(f"\nSaved: {summary['output_dir']}/rankgauss_swapnoise_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RankGauss + swap-noise AE ladder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stratified_reset/rankgauss_swapnoise_ae_ladder"),
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
    parser.add_argument("--corr-threshold", type=float, default=0.75)
    parser.add_argument("--max-v-columns", type=int, default=150)
    parser.add_argument("--max-quantiles", type=int, default=1000)
    parser.add_argument("--rankgauss-clip", type=float, default=5.0)
    parser.add_argument("--swap-rate", type=float, default=0.15)
    parser.add_argument("--ae-max-epochs", type=int, default=60)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-bootstrap", type=int, default=300)
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
        corr_threshold=args.corr_threshold,
        max_v_columns=args.max_v_columns,
        max_quantiles=args.max_quantiles,
        rankgauss_clip=args.rankgauss_clip,
        swap_rate=args.swap_rate,
        ae_max_epochs=args.ae_max_epochs,
        ae_patience=args.ae_patience,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

