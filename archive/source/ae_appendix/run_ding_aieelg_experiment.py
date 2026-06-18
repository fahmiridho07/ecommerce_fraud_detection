"""Ding-style AEELG reproduction adapted to IEEE-CIS.

Anchor: Ding et al. (2024), "An AutoEncoder enhanced LightGBM method for
credit card fraud detection".

The paper pipeline is:

    standardize -> split -> SMOTE on train -> train AutoEncoder
    -> reconstruct train/test features -> train LightGBM

This script adapts the recipe to IEEE-CIS while keeping thesis guardrails:

* stratified or chronological split is created before fitted preprocessing;
* A1 dense IEEE-CIS preprocessing is fit on train only;
* validation/test are never oversampled;
* SMOTE-style interpolation is applied only on the train split;
* the AutoEncoder reconstructs dense A1 features, not raw mixed IEEE-CIS values.

The paper pseudocode is slightly ambiguous about whether LightGBM is trained on
the reconstructed original training set or the reconstructed SMOTE-balanced
training set. We report both variants, plus A1 and SMOTE-only controls.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc

from config import (
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
    TARGET_COL,
)
from data_loader import load_labeled_train_data
from paper_preprocessing import (
    apply_alharbi_style_preprocessing,
    fit_alharbi_style_preprocessing,
)
from preprocessing import split_features_target
from run_ae_augmentation_experiment import (
    evaluate,
    latent_smote_synthesis,
    train_lgbm,
)
from run_vae_augmentation_experiment import paired_bootstrap_ap_delta
from splitting import create_holdout_split, stratified_holdout_split
from utils import ensure_dir, log, save_json, set_seed


def build_ding_autoencoder(
    input_dim: int,
    latent_dim: int,
    hidden_dim: int,
    learning_rate: float,
    l1_penalty: float,
    output_activation: str,
) -> keras.Model:
    """Build a dense reconstruction AE in the spirit of Ding et al.

    Ding's public code uses a 30 -> 16 -> 8 -> 8 -> 30 ReLU autoencoder. IEEE-CIS
    A1 features are z-score scaled and can be negative, so the default output
    activation is linear; `--output-activation relu` is available for strict
    code-level replication.
    """
    input_layer = keras.Input(shape=(input_dim,), name="a1_features")
    x = keras.layers.Dense(
        hidden_dim,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(l1_penalty),
        name="encoder_hidden",
    )(input_layer)
    bottleneck = keras.layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)
    x = keras.layers.Dense(latent_dim, activation="relu", name="decoder_hidden")(bottleneck)
    output_layer = keras.layers.Dense(
        input_dim,
        activation=output_activation,
        name="reconstructed_features",
    )(x)
    autoencoder = keras.Model(input_layer, output_layer, name="ding_style_aieelg_autoencoder")
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return autoencoder


def _synthetic_count(n_normal: int, n_fraud: int, target_fraud_rate: float) -> int:
    if not (0.0 < target_fraud_rate < 1.0):
        raise ValueError("target_fraud_rate must be between 0 and 1.")
    target_total = n_normal / (1.0 - target_fraud_rate)
    return max(0, int(round(target_total - (n_normal + n_fraud))))


def _train_and_score(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    scale_pos_weight: float | None,
    n_estimators: int | None,
) -> tuple[dict[str, float], float, np.ndarray, int]:
    model, best_iteration = train_lgbm(
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_columns=[],
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
    )
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]
    metrics, threshold = evaluate(y_valid, valid_score, y_test, test_score)
    return metrics, threshold, test_score, best_iteration


def _reconstruct_frame(
    autoencoder: keras.Model,
    frame: pd.DataFrame,
    columns: pd.Index,
    batch_size: int,
) -> pd.DataFrame:
    reconstructed = autoencoder.predict(frame.to_numpy(dtype="float32"), batch_size=batch_size, verbose=0)
    return pd.DataFrame(reconstructed.astype("float32"), columns=columns)


def main(
    output_dir: Path,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    split_seed: int | None = None,
    target_fraud_rate: float = 0.50,
    ae_epochs: int = 40,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 2048,
    k_neighbors: int = 5,
    n_estimators: int | None = 800,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
    output_activation: str = "linear",
) -> dict[str, object]:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)

    log("Loading IEEE-CIS data and creating holdout split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    if split_seed is not None and split_strategy == "stratified_holdout":
        train_df, valid_df, test_df = stratified_holdout_split(full_df, random_seed=split_seed)
    else:
        train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    y_train_np = y_train.to_numpy()
    y_valid_np = y_valid.to_numpy()
    y_test_np = y_test.to_numpy()

    log("Fitting A1 dense IEEE-CIS preprocessing on train only.")
    preprocessing = fit_alharbi_style_preprocessing(X_train_raw)
    X_train = apply_alharbi_style_preprocessing(X_train_raw, preprocessing).astype("float32")
    X_valid = apply_alharbi_style_preprocessing(X_valid_raw, preprocessing).astype("float32")
    X_test = apply_alharbi_style_preprocessing(X_test_raw, preprocessing).astype("float32")
    feature_columns = X_train.columns

    log("Training A1 baseline control.")
    baseline_metrics, baseline_threshold, baseline_score, baseline_iter = _train_and_score(
        X_train,
        y_train_np,
        X_valid,
        y_valid_np,
        X_test,
        y_test_np,
        scale_pos_weight=None,
        n_estimators=n_estimators,
    )
    log(f"A1 baseline AP={baseline_metrics['average_precision']:.6f}")

    fraud_mask = y_train_np == 1
    n_fraud = int(fraud_mask.sum())
    n_normal = int((~fraud_mask).sum())
    n_synthetic = _synthetic_count(n_normal, n_fraud, target_fraud_rate)
    log(
        f"Creating SMOTE-style train-only fraud rows: n_synthetic={n_synthetic}, "
        f"target_fraud_rate={target_fraud_rate:.2f}."
    )
    X_fraud = X_train.loc[fraud_mask].to_numpy(dtype="float32")
    synthetic_fraud, _anchors = latent_smote_synthesis(X_fraud, n_synthetic, k_neighbors, rng)
    X_synthetic = pd.DataFrame(synthetic_fraud.astype("float32"), columns=feature_columns)
    X_balanced = pd.concat([X_train, X_synthetic], axis=0, ignore_index=True)
    y_balanced = np.concatenate([y_train_np, np.ones(n_synthetic, dtype=int)])

    log("Training SMOTE-only control on A1 dense features.")
    smote_metrics, smote_threshold, smote_score, smote_iter = _train_and_score(
        X_balanced,
        y_balanced,
        X_valid,
        y_valid_np,
        X_test,
        y_test_np,
        scale_pos_weight=1.0,
        n_estimators=n_estimators,
    )
    smote_bootstrap = paired_bootstrap_ap_delta(
        y_test_np,
        baseline_score,
        smote_score,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    log(
        f"SMOTE-only AP={smote_metrics['average_precision']:.6f}, "
        f"delta={smote_bootstrap['observed_delta_ap']:+.6f}."
    )

    log("Training Ding-style reconstruction AutoEncoder on SMOTE-balanced train.")
    autoencoder = build_ding_autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=1e-3,
        l1_penalty=1e-4,
        output_activation=output_activation,
    )
    history = autoencoder.fit(
        X_balanced.to_numpy(dtype="float32"),
        X_balanced.to_numpy(dtype="float32"),
        validation_data=(X_valid.to_numpy(dtype="float32"), X_valid.to_numpy(dtype="float32")),
        epochs=ae_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(output_dir / "ae_training_history.csv", index=False)

    log("Reconstructing original and SMOTE-balanced feature matrices.")
    X_rec_train = _reconstruct_frame(autoencoder, X_train, feature_columns, batch_size)
    X_rec_valid = _reconstruct_frame(autoencoder, X_valid, feature_columns, batch_size)
    X_rec_test = _reconstruct_frame(autoencoder, X_test, feature_columns, batch_size)
    X_rec_balanced = _reconstruct_frame(autoencoder, X_balanced, feature_columns, batch_size)

    log("Training Ding literal variant: LGBM on reconstructed original train.")
    rec_orig_metrics, rec_orig_threshold, rec_orig_score, rec_orig_iter = _train_and_score(
        X_rec_train,
        y_train_np,
        X_rec_valid,
        y_valid_np,
        X_rec_test,
        y_test_np,
        scale_pos_weight=None,
        n_estimators=n_estimators,
    )
    rec_orig_bootstrap = paired_bootstrap_ap_delta(
        y_test_np,
        baseline_score,
        rec_orig_score,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    log(
        f"Ding literal reconstructed-original AP={rec_orig_metrics['average_precision']:.6f}, "
        f"delta={rec_orig_bootstrap['observed_delta_ap']:+.6f}."
    )

    log("Training Ding carried-SMOTE variant: LGBM on reconstructed balanced train.")
    rec_bal_metrics, rec_bal_threshold, rec_bal_score, rec_bal_iter = _train_and_score(
        X_rec_balanced,
        y_balanced,
        X_rec_valid,
        y_valid_np,
        X_rec_test,
        y_test_np,
        scale_pos_weight=1.0,
        n_estimators=n_estimators,
    )
    rec_bal_bootstrap = paired_bootstrap_ap_delta(
        y_test_np,
        baseline_score,
        rec_bal_score,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    rec_bal_vs_smote = paired_bootstrap_ap_delta(
        y_test_np,
        smote_score,
        rec_bal_score,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    log(
        f"Ding reconstructed-balanced AP={rec_bal_metrics['average_precision']:.6f}, "
        f"delta={rec_bal_bootstrap['observed_delta_ap']:+.6f}."
    )

    results = {
        "a1_baseline": {
            "test_average_precision": baseline_metrics["average_precision"],
            "test_roc_auc": baseline_metrics["roc_auc"],
            "test_f1": baseline_metrics["f1"],
            "test_mcc": baseline_metrics["mcc"],
            "selected_threshold": baseline_threshold,
            "best_iteration": baseline_iter,
        },
        "a1_smote_only": {
            "test_average_precision": smote_metrics["average_precision"],
            "test_roc_auc": smote_metrics["roc_auc"],
            "test_f1": smote_metrics["f1"],
            "test_mcc": smote_metrics["mcc"],
            "selected_threshold": smote_threshold,
            "best_iteration": smote_iter,
            "bootstrap_vs_a1_baseline": smote_bootstrap,
        },
        "ding_reconstructed_original_train": {
            "interpretation": "Paper pseudocode literal: AE trained on SMOTE-balanced train; LightGBM trained on reconstructed original train.",
            "test_average_precision": rec_orig_metrics["average_precision"],
            "test_roc_auc": rec_orig_metrics["roc_auc"],
            "test_f1": rec_orig_metrics["f1"],
            "test_mcc": rec_orig_metrics["mcc"],
            "selected_threshold": rec_orig_threshold,
            "best_iteration": rec_orig_iter,
            "bootstrap_vs_a1_baseline": rec_orig_bootstrap,
        },
        "ding_reconstructed_balanced_train": {
            "interpretation": "SMOTE carried through to LightGBM: LightGBM trained on reconstructed SMOTE-balanced train.",
            "test_average_precision": rec_bal_metrics["average_precision"],
            "test_roc_auc": rec_bal_metrics["roc_auc"],
            "test_f1": rec_bal_metrics["f1"],
            "test_mcc": rec_bal_metrics["mcc"],
            "selected_threshold": rec_bal_threshold,
            "best_iteration": rec_bal_iter,
            "bootstrap_vs_a1_baseline": rec_bal_bootstrap,
            "bootstrap_vs_a1_smote_only": rec_bal_vs_smote,
        },
    }

    summary = {
        "experiment_id": "DING-AEELG-IEEE-CIS",
        "anchor": "Ding et al. (2024) AEELG: SMOTE + AutoEncoder feature reconstruction + LightGBM",
        "adaptation": {
            "dataset": "IEEE-CIS Fraud Detection labeled train set",
            "preprocessing": "A1 dense Alharbi-style frequency encoding + median imputation + z-score, fit on train only",
            "smote_scope": "train split only",
            "validation_test_scope": "never oversampled; used only for validation/test evaluation",
            "reason_for_a1": "Ding's datasets are already numeric; IEEE-CIS requires dense numeric conversion before AE reconstruction.",
            "output_activation": output_activation,
        },
        "split_strategy": split_strategy,
        "split_seed": split_seed,
        "seed": seed,
        "n_estimators": n_estimators,
        "n_bootstrap": n_bootstrap,
        "test_prevalence": float(y_test_np.mean()),
        "smote": {
            "target_fraud_rate": target_fraud_rate,
            "k_neighbors": k_neighbors,
            "train_fraud_before": n_fraud,
            "train_normal": n_normal,
            "n_synthetic_fraud": n_synthetic,
            "train_fraud_after": int(n_fraud + n_synthetic),
            "train_rows_after": int(len(y_balanced)),
        },
        "autoencoder": {
            "input_dim": int(X_train.shape[1]),
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "epochs_requested": ae_epochs,
            "epochs_ran": int(len(history.history["loss"])),
            "batch_size": batch_size,
            "loss": "mean_squared_error",
            "optimizer": "Adam",
            "l1_penalty": 1e-4,
            "best_val_loss": float(np.min(history.history["val_loss"])),
        },
        "results": results,
    }
    save_json(summary, output_dir / "experiment_summary.json")

    scores = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].to_numpy(),
            TARGET_COL: y_test_np,
            "score_a1_baseline": baseline_score,
            "score_a1_smote_only": smote_score,
            "score_ding_reconstructed_original_train": rec_orig_score,
            "score_ding_reconstructed_balanced_train": rec_bal_score,
        }
    )
    scores.to_csv(output_dir / "test_scores.csv", index=False)

    print("\nDing-style AEELG on IEEE-CIS")
    print("============================")
    for name, row in results.items():
        print(
            f"{name:38s} AP={row['test_average_precision']:.6f} "
            f"ROC={row['test_roc_auc']:.6f} F1={row['test_f1']:.6f} MCC={row['test_mcc']:.6f}"
        )
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ding-style AEELG on IEEE-CIS.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--target-fraud-rate", type=float, default=0.50)
    parser.add_argument("--ae-epochs", type=int, default=40)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-activation", choices=("linear", "relu"), default="linear")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
        target_fraud_rate=args.target_fraud_rate,
        ae_epochs=args.ae_epochs,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        k_neighbors=args.k_neighbors,
        n_estimators=args.n_estimators,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        output_activation=args.output_activation,
    )
