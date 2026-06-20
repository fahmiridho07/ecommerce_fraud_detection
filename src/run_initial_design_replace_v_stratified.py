"""Desain awal usulan pada protokol STRATIFIED (agar sebanding dgn tabel lain).

Desain awal: blok fitur V direkonstruksi oleh autoencoder lalu MENGGANTI V asli,
digabung dengan fitur non-V, dan dilatih LightGBM. Dibandingkan dengan baseline
(tanpa AE) pada split, preprocessing, budget, dan threshold-rule yang sama persis.

Hanya satu variabel yang berbeda: blok V asli vs blok V hasil rekonstruksi AE.

Run:
    python src/run_initial_design_replace_v_stratified.py \
        --output-dir outputs/stratified_reset/initial_design_replace_v
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_SPLIT_STRATEGY, RANDOM_SEED, SAMPLE_SIZE, SUPPORTED_SPLIT_STRATEGIES
from data_loader import load_labeled_train_data
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from splitting import create_holdout_split
from run_ae_augmentation_experiment import (
    build_fraud_autoencoder,
    evaluate,
    paired_bootstrap_ap_delta,
    train_lgbm,
)
from utils import ensure_dir, log, save_json, set_seed

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc


def reconstruct_v_block(Xb_train, Xb_valid, Xb_test, v_cols, latent_dim, ae_epochs, seed):
    """Train an AE on the (scaled) V block of train rows; return reconstructed V
    for each split in the original feature space (median-imputed, AE-reconstructed)."""
    imputer = SimpleImputer(strategy="median").fit(Xb_train[v_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[v_cols]))

    def scaled(df):
        return np.clip(scaler.transform(imputer.transform(df[v_cols])).astype("float32"), -10.0, 10.0)

    Vtr, Vva, Vte = scaled(Xb_train), scaled(Xb_valid), scaled(Xb_test)

    ae, enc, dec = build_fraud_autoencoder(Vtr.shape[1], latent_dim)
    ae.fit(
        Vtr, Vtr, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
        verbose=2,
    )

    def recon(V):
        scaled_recon = ae.predict(V, batch_size=4096, verbose=0)
        return scaler.inverse_transform(scaled_recon).astype("float32")

    return recon(Vtr), recon(Vva), recon(Vte)


def main(output_dir: Path, split_strategy: str = DEFAULT_SPLIT_STRATEGY, latent_dim: int = 32,
         ae_epochs: int = 30, n_bootstrap: int = 2000, seed: int = RANDOM_SEED,
         n_estimators: int | None = None) -> dict:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)

    log("Loading data and splitting (stratified).")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    pre = fit_baseline_preprocessing(X_train_raw)
    Xb_train = apply_baseline_preprocessing(X_train_raw, pre)
    Xb_valid = apply_baseline_preprocessing(X_valid_raw, pre)
    Xb_test = apply_baseline_preprocessing(X_test_raw, pre)
    categorical_columns = pre["categorical_columns"]
    v_cols = get_v_feature_columns(Xb_train)
    y_tr, y_va, y_te = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()
    log(f"V columns: {len(v_cols)}; non-V columns: {Xb_train.shape[1] - len(v_cols)}")

    # ----- Baseline: original V retained -----
    log("Training baseline LightGBM (original features).")
    base_model, base_it = train_lgbm(Xb_train, y_tr, Xb_valid, y_va, categorical_columns, n_estimators=n_estimators)
    base_valid = base_model.predict_proba(Xb_valid, num_iteration=base_it)[:, 1]
    base_test = base_model.predict_proba(Xb_test, num_iteration=base_it)[:, 1]
    base_m, base_thr = evaluate(y_va, base_valid, y_te, base_test)
    log(f"BASELINE test AP={base_m['average_precision']:.6f}")

    # ----- Desain awal: replace V with AE reconstruction -----
    log("Training autoencoder on V block and reconstructing.")
    Rtr, Rva, Rte = reconstruct_v_block(Xb_train, Xb_valid, Xb_test, v_cols, latent_dim, ae_epochs, seed)
    Xr_train, Xr_valid, Xr_test = Xb_train.copy(), Xb_valid.copy(), Xb_test.copy()
    Xr_train[v_cols] = Rtr
    Xr_valid[v_cols] = Rva
    Xr_test[v_cols] = Rte

    log("Training LightGBM on reconstructed-V representation.")
    rep_model, rep_it = train_lgbm(Xr_train, y_tr, Xr_valid, y_va, categorical_columns, n_estimators=n_estimators)
    rep_valid = rep_model.predict_proba(Xr_valid, num_iteration=rep_it)[:, 1]
    rep_test = rep_model.predict_proba(Xr_test, num_iteration=rep_it)[:, 1]
    rep_m, rep_thr = evaluate(y_va, rep_valid, y_te, rep_test)
    boot = paired_bootstrap_ap_delta(y_te, base_test, rep_test, n_bootstrap=n_bootstrap)
    log(f"REPLACE-V test AP={rep_m['average_precision']:.6f} delta={boot['observed_delta_ap']:+.6f} p={boot['p_delta_le_0']:.3f}")

    summary = {
        "design": "initial_proposal_replace_v_with_ae_reconstruction",
        "split_strategy": split_strategy,
        "seed": seed,
        "latent_dim": latent_dim,
        "n_v_features": len(v_cols),
        "test_prevalence": float(y_te.mean()),
        "results": {
            "baseline": {
                "test_average_precision": base_m["average_precision"], "test_roc_auc": base_m["roc_auc"],
                "test_f1": base_m["f1"], "test_mcc": base_m["mcc"], "selected_threshold": base_thr,
            },
            "replace_v_ae_reconstruction": {
                "test_average_precision": rep_m["average_precision"], "test_roc_auc": rep_m["roc_auc"],
                "test_f1": rep_m["f1"], "test_mcc": rep_m["mcc"], "selected_threshold": rep_thr,
                "bootstrap_vs_baseline": boot,
            },
        },
    }
    save_json(summary, output_dir / "experiment_summary.json")

    print("\nInitial design (replace V) — stratified")
    print("=======================================")
    print(f"{'variant':32s} {'test_AP':>10s} {'ROC':>9s} {'F1':>8s} {'MCC':>8s} {'d_vs_base':>11s} {'p':>7s}")
    b = summary["results"]["baseline"]
    print(f"{'baseline (V asli)':32s} {b['test_average_precision']:10.6f} {b['test_roc_auc']:9.6f} {b['test_f1']:8.6f} {b['test_mcc']:8.6f}")
    r = summary["results"]["replace_v_ae_reconstruction"]
    print(f"{'replace V -> AE recon':32s} {r['test_average_precision']:10.6f} {r['test_roc_auc']:9.6f} {r['test_f1']:8.6f} {r['test_mcc']:8.6f} {boot['observed_delta_ap']:+11.6f} {boot['p_delta_le_0']:7.3f}")
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initial design (replace V with AE reconstruction) on stratified split.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--ae-epochs", type=int, default=30)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-estimators", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(output_dir=a.output_dir, split_strategy=a.split_strategy, latent_dim=a.latent_dim,
         ae_epochs=a.ae_epochs, n_bootstrap=a.n_bootstrap, seed=a.seed, n_estimators=a.n_estimators)
