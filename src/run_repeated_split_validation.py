"""Repeated stratified holdout validation of the AE augmentation win.

A single holdout split estimates one number; it does not capture variance from
the split itself. This harness re-runs the fair comparison across several
stratified split seeds and reports the mean +/- std of the test-AP deltas, plus
how many splits show a positive delta. This is the cross-validation-style
robustness check recommended in the methodology audit (single holdout ->
repeated holdout).

Per split seed it trains, on identical features and LightGBM budget:
- baseline           : no oversampling, scale_pos_weight from data
- smote_nc           : raw-space SMOTE-NC oversampling, spw=1.0
- ae_latent_smote    : AE latent-space oversampling, spw=1.0 (proposed)

Reported deltas: ae - baseline (proposed vs baseline) and ae - smote_nc
(AE-specific contribution beyond generic oversampling).

Run from repo root:
    python src/run_repeated_split_validation.py \
        --output-dir outputs/stratified_reset/repeated_split_validation \
        --split-seeds 42 1 2 3 4 --n-estimators 800
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, SAMPLE_SIZE, TARGET_COL
from data_loader import load_labeled_train_data
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    split_features_target,
)
from splitting import stratified_holdout_split
from run_ae_augmentation_experiment import (
    build_fraud_autoencoder,
    evaluate,
    latent_smote_synthesis,
    numeric_columns,
    train_lgbm,
)
from run_fair_augmentation_comparison import assemble_synthetic_df
from utils import ensure_dir, log, save_json, set_seed

try:
    from tensorflow import keras
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc


def build_synthetic(method, Xb_train, y_train_np, categorical_columns, num_cols,
                    target_fraud_rate, latent_dim, ae_epochs, k_neighbors, rng, seed):
    """Return (X_aug, y_aug) for the requested oversampling method."""
    imputer = SimpleImputer(strategy="median").fit(Xb_train[num_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[num_cols]))
    fraud_mask = y_train_np == 1
    Xn_fraud = np.clip(
        scaler.transform(imputer.transform(Xb_train.loc[fraud_mask, num_cols])).astype("float32"),
        -10.0, 10.0,
    )
    fraud_cat = Xb_train.loc[fraud_mask, categorical_columns].reset_index(drop=True)
    n_fraud = int(fraud_mask.sum())
    n_normal = int((~fraud_mask).sum())
    n_synth = max(0, int(round(n_normal / (1.0 - target_fraud_rate) - (n_normal + n_fraud))))

    if method == "smote_nc":
        synth_scaled, anchors = latent_smote_synthesis(Xn_fraud, n_synth, k_neighbors, rng)
        raw = scaler.inverse_transform(synth_scaled)
    elif method == "ae_latent_smote":
        ae, enc, dec = build_fraud_autoencoder(Xn_fraud.shape[1], latent_dim)
        ae.fit(Xn_fraud, Xn_fraud, validation_split=0.1, epochs=ae_epochs, batch_size=256,
               shuffle=True,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                        restore_best_weights=True)],
               verbose=0)
        latent_fraud = enc.predict(Xn_fraud, batch_size=1024, verbose=0)
        synth_latent, anchors = latent_smote_synthesis(latent_fraud, n_synth, k_neighbors, rng)
        raw = scaler.inverse_transform(dec.predict(synth_latent, batch_size=1024, verbose=0))
    else:
        raise ValueError(method)

    synth_df = assemble_synthetic_df(raw, anchors, num_cols, categorical_columns, fraud_cat, Xb_train)
    X_aug = pd.concat([Xb_train, synth_df], axis=0, ignore_index=True)
    y_aug = np.concatenate([y_train_np, np.ones(n_synth, dtype=int)])
    return X_aug, y_aug


def main(output_dir: Path, split_seeds: list[int], target_fraud_rate: float = 0.15,
         latent_dim: int = 16, ae_epochs: int = 60, k_neighbors: int = 5,
         n_estimators: int | None = 800) -> dict:
    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading data once.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    per_seed = []
    for seed in split_seeds:
        log(f"===== split seed {seed} =====")
        train_df, valid_df, test_df = stratified_holdout_split(full_df, random_seed=seed)
        X_train_raw, y_train = split_features_target(train_df)
        X_valid_raw, y_valid = split_features_target(valid_df)
        X_test_raw, y_test = split_features_target(test_df)
        pre = fit_baseline_preprocessing(X_train_raw)
        Xb_train = apply_baseline_preprocessing(X_train_raw, pre)
        Xb_valid = apply_baseline_preprocessing(X_valid_raw, pre)
        Xb_test = apply_baseline_preprocessing(X_test_raw, pre)
        cat_cols = pre["categorical_columns"]
        num_cols = numeric_columns(Xb_train, cat_cols)
        y_tr, y_va, y_te = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()

        # baseline
        bm, bi = train_lgbm(Xb_train, y_tr, Xb_valid, y_va, cat_cols, n_estimators=n_estimators)
        base_test = bm.predict_proba(Xb_test, num_iteration=bi)[:, 1]
        base_ap = float(average_precision_score(y_te, base_test))

        method_ap = {}
        for method in ("smote_nc", "ae_latent_smote"):
            X_aug, y_aug = build_synthetic(
                method, Xb_train, y_tr, cat_cols, num_cols, target_fraud_rate,
                latent_dim, ae_epochs, k_neighbors, np.random.default_rng(seed), seed)
            m, mi = train_lgbm(X_aug, y_aug, Xb_valid, y_va, cat_cols,
                               scale_pos_weight=1.0, n_estimators=n_estimators)
            tsc = m.predict_proba(Xb_test, num_iteration=mi)[:, 1]
            method_ap[method] = float(average_precision_score(y_te, tsc))
            del X_aug, y_aug
            gc.collect()

        row = {
            "split_seed": seed,
            "baseline_ap": base_ap,
            "smote_nc_ap": method_ap["smote_nc"],
            "ae_latent_smote_ap": method_ap["ae_latent_smote"],
            "delta_ae_vs_baseline": method_ap["ae_latent_smote"] - base_ap,
            "delta_ae_vs_smote_nc": method_ap["ae_latent_smote"] - method_ap["smote_nc"],
            "delta_smote_vs_baseline": method_ap["smote_nc"] - base_ap,
        }
        per_seed.append(row)
        log(f"seed {seed}: base={base_ap:.5f} smote={method_ap['smote_nc']:.5f} "
            f"ae={method_ap['ae_latent_smote']:.5f} "
            f"d(ae-base)={row['delta_ae_vs_baseline']:+.5f} d(ae-smote)={row['delta_ae_vs_smote_nc']:+.5f}")
        del Xb_train, Xb_valid, Xb_test, train_df, valid_df, test_df
        gc.collect()

    df = pd.DataFrame(per_seed)
    def agg(col):
        v = df[col].to_numpy()
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
                "min": float(v.min()), "max": float(v.max()), "n_positive": int((v > 0).sum()),
                "n": int(len(v))}
    summary = {
        "split_seeds": split_seeds,
        "n_estimators": n_estimators,
        "target_fraud_rate": target_fraud_rate,
        "per_seed": per_seed,
        "aggregate": {
            "delta_ae_vs_baseline": agg("delta_ae_vs_baseline"),
            "delta_ae_vs_smote_nc": agg("delta_ae_vs_smote_nc"),
            "delta_smote_vs_baseline": agg("delta_smote_vs_baseline"),
            "baseline_ap": agg("baseline_ap"),
            "ae_latent_smote_ap": agg("ae_latent_smote_ap"),
        },
    }
    save_json(summary, output_dir / "experiment_summary.json")
    df.to_csv(output_dir / "per_seed_results.csv", index=False)

    print()
    print("Repeated Stratified Holdout Validation")
    print("======================================")
    print(df.to_string(index=False))
    a = summary["aggregate"]
    print()
    print("Aggregate across %d splits:" % len(split_seeds))
    print("  delta ae_vs_baseline : mean %+.5f +/- %.5f  (positive in %d/%d)" % (
        a["delta_ae_vs_baseline"]["mean"], a["delta_ae_vs_baseline"]["std"],
        a["delta_ae_vs_baseline"]["n_positive"], a["delta_ae_vs_baseline"]["n"]))
    print("  delta ae_vs_smote_nc : mean %+.5f +/- %.5f  (positive in %d/%d)" % (
        a["delta_ae_vs_smote_nc"]["mean"], a["delta_ae_vs_smote_nc"]["std"],
        a["delta_ae_vs_smote_nc"]["n_positive"], a["delta_ae_vs_smote_nc"]["n"]))
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated stratified holdout validation.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    p.add_argument("--target-fraud-rate", type=float, default=0.15)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--ae-epochs", type=int, default=60)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=800)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        split_seeds=args.split_seeds,
        target_fraud_rate=args.target_fraud_rate,
        latent_dim=args.latent_dim,
        ae_epochs=args.ae_epochs,
        k_neighbors=args.k_neighbors,
        n_estimators=args.n_estimators,
    )
