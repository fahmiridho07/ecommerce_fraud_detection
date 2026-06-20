"""Run a stricter Ding et al. AEELG adaptation on IEEE-CIS.

This runner answers a narrow audit question:

    Was the previous Ding-style IEEE-CIS experiment unfair because the LightGBM
    and reported metrics were too thesis-specific?

It keeps the unavoidable IEEE-CIS adaptation (dense numeric A1 preprocessing),
but uses Ding-like LightGBM defaults and reports Ding's metrics:
precision, recall, F1/F-measure, ROC-AUC, MCC, and BCR. Average Precision is
also saved because it remains the thesis primary metric.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
    TARGET_COL,
)
from data_loader import load_labeled_train_data
from paper_preprocessing import fit_alharbi_style_preprocessing
from preprocessing import split_features_target
from run_ding_anchor_replication import (
    build_autoencoder,
    compact_result_row,
    lgbm_params,
    paired_bootstrap_ap_delta,
    reconstruct_frame,
    score_model,
    smote_interpolate,
    synthetic_count,
)
from splitting import create_holdout_split, stratified_holdout_split
from utils import ensure_dir, log, save_json, set_seed


def _apply_a1_float32(X: pd.DataFrame, preprocessing: dict[str, object]) -> pd.DataFrame:
    """Apply A1 preprocessing with lower peak memory for full IEEE-CIS runs."""
    raw_columns = list(preprocessing["feature_columns_raw"])
    numeric_columns = list(preprocessing["numeric_columns"])
    categorical_columns = list(preprocessing["categorical_columns"])
    transformed_columns = list(preprocessing["feature_columns_transformed"])

    X = X.loc[:, raw_columns]
    parts: list[pd.DataFrame] = []

    if numeric_columns:
        numeric = X.loc[:, numeric_columns].astype("float32", copy=True)
        medians = pd.Series(preprocessing["numeric_medians"], dtype="float32")
        means = pd.Series(preprocessing["numeric_means"], dtype="float32")
        stds = pd.Series(preprocessing["numeric_stds"], dtype="float32")
        numeric = numeric.fillna(medians)
        numeric = ((numeric - means) / stds).astype("float32")
        parts.append(numeric)

    missing_token = str(preprocessing["missing_category"])
    frequency_maps = preprocessing["frequency_maps"]
    if categorical_columns:
        categorical_data: dict[str, pd.Series] = {}
        for column in categorical_columns:
            values = X[column].astype("string").fillna(missing_token).astype(str)
            encoded = values.map(frequency_maps[column]).fillna(0.0).astype("float32")
            categorical_data[f"{column}_frequency"] = encoded
        parts.append(pd.DataFrame(categorical_data, index=X.index))

    transformed = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
    transformed = transformed.loc[:, transformed_columns].astype("float32")
    if not np.isfinite(transformed.to_numpy(dtype="float32", copy=False)).all():
        raise ValueError("A1 preprocessing produced non-finite values.")
    return transformed


def _serializable_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"valid_score", "test_score"}
    }


def _save_metrics_tables(results: list[dict[str, object]], output_dir: Path) -> None:
    default_rows = [compact_result_row(result, "test_default") for result in results]
    selected_rows = [compact_result_row(result, "test_selected") for result in results]
    pd.DataFrame(default_rows).to_csv(output_dir / "metrics_test_default_threshold.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "metrics_test_selected_threshold.csv", index=False)


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = ensure_dir(args.output_dir)

    log("Loading IEEE-CIS data and creating holdout split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    if args.split_seed is not None and args.split_strategy == "stratified_holdout":
        train_df, valid_df, test_df = stratified_holdout_split(full_df, random_seed=args.split_seed)
    else:
        train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=args.split_strategy)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    test_ids = test_df[ID_COL].to_numpy()
    del train_df, valid_df, test_df
    gc.collect()

    y_train_np = y_train.to_numpy(dtype=int)
    y_valid_np = y_valid.to_numpy(dtype=int)
    y_test_np = y_test.to_numpy(dtype=int)
    del y_train, y_valid, y_test

    log("Fitting dense A1 preprocessing on train only.")
    preprocessing = fit_alharbi_style_preprocessing(X_train_raw)
    log("Applying dense A1 preprocessing to train.")
    X_train = _apply_a1_float32(X_train_raw, preprocessing)
    del X_train_raw
    gc.collect()
    log("Applying dense A1 preprocessing to validation.")
    X_valid = _apply_a1_float32(X_valid_raw, preprocessing)
    del X_valid_raw
    gc.collect()
    log("Applying dense A1 preprocessing to test.")
    X_test = _apply_a1_float32(X_test_raw, preprocessing)
    del X_test_raw
    gc.collect()

    feature_columns = X_train.columns
    params = lgbm_params("ding", args.n_estimators, args.lgbm_seed, args.n_jobs)

    log("Training Ding-like GOSS LightGBM baseline.")
    baseline = score_model(
        "ding_goss_baseline",
        X_train,
        y_train_np,
        X_valid,
        y_valid_np,
        X_test,
        y_test_np,
        params=params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    fraud_mask = y_train_np == 1
    n_fraud = int(fraud_mask.sum())
    n_normal = int((~fraud_mask).sum())
    n_synthetic = synthetic_count(n_normal, n_fraud, args.target_fraud_rate)
    log(
        f"Creating Ding-like SMOTE train rows: fraud={n_fraud}, "
        f"normal={n_normal}, synthetic={n_synthetic}."
    )
    X_fraud = X_train.loc[fraud_mask].to_numpy(dtype="float32")
    synthetic_values = smote_interpolate(X_fraud, n_synthetic, args.k_neighbors, rng)
    X_synthetic = pd.DataFrame(synthetic_values, columns=feature_columns)
    X_balanced = pd.concat([X_train, X_synthetic], axis=0, ignore_index=True)
    y_balanced = np.concatenate([y_train_np, np.ones(n_synthetic, dtype=int)])

    log("Training Ding-like SMOTE + GOSS LightGBM control.")
    smote_control = score_model(
        "ding_smote_goss",
        X_balanced,
        y_balanced,
        X_valid,
        y_valid_np,
        X_test,
        y_test_np,
        params=params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    log("Training Ding-style AutoEncoder on SMOTE-balanced A1 train features.")
    autoencoder = build_autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        decoder_dim=args.decoder_dim,
        output_activation=args.output_activation,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
    )
    history = autoencoder.fit(
        X_balanced.to_numpy(dtype="float32"),
        X_balanced.to_numpy(dtype="float32"),
        validation_data=(X_valid.to_numpy(dtype="float32"), X_valid.to_numpy(dtype="float32")),
        epochs=args.ae_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.ae_patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(output_dir / "ae_training_history.csv", index=False)

    log("Reconstructing original and SMOTE-balanced matrices.")
    X_rec_train = reconstruct_frame(autoencoder, X_train, args.batch_size)
    X_rec_valid = reconstruct_frame(autoencoder, X_valid, args.batch_size)
    X_rec_test = reconstruct_frame(autoencoder, X_test, args.batch_size)
    X_rec_balanced = reconstruct_frame(autoencoder, X_balanced, args.batch_size)

    log("Training Ding pseudocode variant: reconstructed original train.")
    rec_original = score_model(
        "ding_reconstructed_original_goss",
        X_rec_train,
        y_train_np,
        X_rec_valid,
        y_valid_np,
        X_rec_test,
        y_test_np,
        params=params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    log("Training carried-SMOTE variant: reconstructed balanced train.")
    rec_balanced = score_model(
        "ding_reconstructed_balanced_goss",
        X_rec_balanced,
        y_balanced,
        X_rec_valid,
        y_valid_np,
        X_rec_test,
        y_test_np,
        params=params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    results = [baseline, smote_control, rec_original, rec_balanced]
    _save_metrics_tables(results, output_dir)

    baseline_score = baseline["test_score"]
    smote_score = smote_control["test_score"]
    for result in results[1:]:
        result["bootstrap_ap_vs_ding_goss_baseline"] = paired_bootstrap_ap_delta(
            y_test_np,
            baseline_score,
            result["test_score"],
            args.n_bootstrap,
            args.seed,
        )
    rec_balanced["bootstrap_ap_vs_ding_smote_goss"] = paired_bootstrap_ap_delta(
        y_test_np,
        smote_score,
        rec_balanced["test_score"],
        args.n_bootstrap,
        args.seed,
    )

    scores = pd.DataFrame(
        {
            ID_COL: test_ids,
            TARGET_COL: y_test_np,
            "score_ding_goss_baseline": baseline["test_score"],
            "score_ding_smote_goss": smote_control["test_score"],
            "score_ding_reconstructed_original_goss": rec_original["test_score"],
            "score_ding_reconstructed_balanced_goss": rec_balanced["test_score"],
        }
    )
    scores.to_csv(output_dir / "test_scores.csv", index=False)

    summary = {
        "experiment_id": "DING-STRICT-IEEE-CIS",
        "anchor": "Ding et al. (2024) AEELG strict-ish transfer to IEEE-CIS",
        "adaptation_boundary": {
            "unavoidable": "IEEE-CIS is converted to dense A1 numeric features before AE reconstruction.",
            "strict_parts": [
                "SMOTE target fraud rate 0.50 by default",
                "GOSS LightGBM with Ding-like hyperparameters",
                "Ding metrics: precision, recall, F1, ROC-AUC, MCC, BCR",
                "AE reconstructs full feature matrix before LightGBM",
            ],
            "guardrails": [
                "split before preprocessing",
                "preprocessing fit on train only",
                "SMOTE fit/applied to train only",
                "validation/test never oversampled",
            ],
        },
        "split_strategy": args.split_strategy,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "lgbm_seed": args.lgbm_seed,
        "test_prevalence": float(y_test_np.mean()),
        "preprocessing": {
            "kind": preprocessing["kind"],
            "anchor": preprocessing["anchor"],
            "feature_count": int(X_train.shape[1]),
        },
        "smote": {
            "target_fraud_rate": args.target_fraud_rate,
            "k_neighbors": args.k_neighbors,
            "train_fraud_before": n_fraud,
            "train_normal": n_normal,
            "synthetic_fraud": int(n_synthetic),
            "train_fraud_after": int(n_fraud + n_synthetic),
            "train_rows_after": int(len(y_balanced)),
            "fraud_rate_after": float(y_balanced.mean()),
        },
        "autoencoder": {
            "input_dim": int(X_train.shape[1]),
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "decoder_dim": args.decoder_dim,
            "output_activation": args.output_activation,
            "learning_rate": args.learning_rate,
            "l1_penalty": args.l1_penalty,
            "batch_size": args.batch_size,
            "epochs_requested": args.ae_epochs,
            "epochs_ran": int(len(history.history["loss"])),
            "best_val_loss": float(np.min(history.history["val_loss"])),
        },
        "lightgbm_params": params,
        "results": {
            str(result["arm"]): _serializable_result(result)
            for result in results
        },
    }
    save_json(summary, output_dir / "experiment_summary.json")

    print("\nDing-strict IEEE-CIS")
    print("====================")
    table = pd.read_csv(output_dir / "metrics_test_selected_threshold.csv")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict-ish Ding AEELG transfer on IEEE-CIS.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--target-fraud-rate", type=float, default=0.50)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--lgbm-seed", type=int, default=2018)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--output-activation", choices=("linear", "relu"), default="linear")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1-penalty", type=float, default=1e-4)
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
