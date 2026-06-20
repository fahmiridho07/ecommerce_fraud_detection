"""Diagnosis: MENGAPA desain awal (ganti V dgn rekonstruksi AE) menurunkan PR-AUC?

Mengikuti arahan: tetap pada usulan semula, kaji penyebab kekurangannya secara
empiris sebelum mencari perbaikan yang sejalan tujuan awal.

Hipotesis: autoencoder bersifat lossy. Memampatkan 339 fitur V ke latent kecil
lalu merekonstruksi menghilangkan informasi diskriminatif yang dipakai LightGBM.

Bukti yang dikumpulkan (protokol stratified, sama dgn tabel lain):
1. Pentingnya fitur V di baseline (gain importance share V vs non-V).
2. Sweep latent_dim {16, 32, 64, 128, 256}: untuk tiap dim ukur
   - kualitas rekonstruksi V (R^2 di ruang ter-skala), dan
   - test PR-AUC LightGBM saat V diganti rekonstruksi.
   Jika AP naik seiring latent_dim membesar (rekonstruksi membaik), maka penyebab
   penurunan TERBUKTI: information loss akibat bottleneck yang terlalu sempit.

Run (budget hemat utk diagnosis):
    python src/diagnose_initial_design_replace_v.py \
        --output-dir outputs/stratified_reset/diagnose_replace_v --n-estimators 800
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


def r2_score_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main(output_dir: Path, split_strategy: str = DEFAULT_SPLIT_STRATEGY,
         latent_dims=(16, 32, 64, 128, 256), ae_epochs: int = 30,
         n_estimators: int = 800, n_bootstrap: int = 2000, seed: int = RANDOM_SEED) -> dict:
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

    # ----- 1) Baseline + V importance share -----
    log("Training baseline LightGBM (original features).")
    base_model, base_it = train_lgbm(Xb_train, y_tr, Xb_valid, y_va, categorical_columns, n_estimators=n_estimators)
    base_valid = base_model.predict_proba(Xb_valid, num_iteration=base_it)[:, 1]
    base_test = base_model.predict_proba(Xb_test, num_iteration=base_it)[:, 1]
    base_m, _ = evaluate(y_va, base_valid, y_te, base_test)

    importances = base_model.feature_importances_
    feat_names = list(Xb_train.columns)
    v_set = set(v_cols)
    total_imp = float(importances.sum())
    v_imp = float(sum(imp for name, imp in zip(feat_names, importances) if name in v_set))
    v_share = v_imp / total_imp if total_imp > 0 else 0.0
    log(f"BASELINE AP={base_m['average_precision']:.6f} | V importance share={v_share:.3f} ({len(v_cols)} V feats)")

    # ----- Shared imputer/scaler for the V block -----
    imputer = SimpleImputer(strategy="median").fit(Xb_train[v_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[v_cols]))

    def scaled(df):
        return np.clip(scaler.transform(imputer.transform(df[v_cols])).astype("float32"), -10.0, 10.0)

    Vtr, Vva, Vte = scaled(Xb_train), scaled(Xb_valid), scaled(Xb_test)

    # ----- 2) Sweep latent_dim -----
    sweep = []

    def write_checkpoint():
        save_json({
            "diagnosis": "why_replace_v_with_ae_reconstruction_underperforms",
            "split_strategy": split_strategy, "seed": seed, "n_estimators": n_estimators,
            "n_v_features": len(v_cols),
            "baseline_test_average_precision": base_m["average_precision"],
            "baseline_v_importance_share": v_share,
            "latent_dim_sweep": sweep,
        }, output_dir / "diagnosis_summary.json")

    for ld in latent_dims:
        keras.backend.clear_session()  # prevent TF memory creep across iterations
        log(f"--- latent_dim={ld}: train AE, reconstruct, retrain LightGBM ---")
        ae, enc, dec = build_fraud_autoencoder(Vtr.shape[1], ld)
        ae.fit(
            Vtr, Vtr, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
            verbose=2,
        )
        Rtr = ae.predict(Vtr, batch_size=4096, verbose=0)
        Rva = ae.predict(Vva, batch_size=4096, verbose=0)
        Rte = ae.predict(Vte, batch_size=4096, verbose=0)
        recon_r2_test = r2_score_overall(Vte, Rte)
        recon_mse_test = float(np.mean((Vte - Rte) ** 2))

        Xr_train, Xr_valid, Xr_test = Xb_train.copy(), Xb_valid.copy(), Xb_test.copy()
        Xr_train[v_cols] = scaler.inverse_transform(Rtr).astype("float32")
        Xr_valid[v_cols] = scaler.inverse_transform(Rva).astype("float32")
        Xr_test[v_cols] = scaler.inverse_transform(Rte).astype("float32")

        model, it = train_lgbm(Xr_train, y_tr, Xr_valid, y_va, categorical_columns, n_estimators=n_estimators)
        vsc = model.predict_proba(Xr_valid, num_iteration=it)[:, 1]
        tsc = model.predict_proba(Xr_test, num_iteration=it)[:, 1]
        m, _ = evaluate(y_va, vsc, y_te, tsc)
        boot = paired_bootstrap_ap_delta(y_te, base_test, tsc, n_bootstrap=n_bootstrap)
        row = {
            "latent_dim": ld,
            "compression_ratio": round(len(v_cols) / ld, 2),
            "recon_r2_test": recon_r2_test,
            "recon_mse_test": recon_mse_test,
            "test_average_precision": m["average_precision"],
            "delta_vs_baseline": boot["observed_delta_ap"],
            "p_delta_le_0": boot["p_delta_le_0"],
        }
        sweep.append(row)
        write_checkpoint()  # persist after each latent_dim so a crash doesn't lose progress
        log(f"latent_dim={ld}: recon_R2={recon_r2_test:.4f} AP={m['average_precision']:.6f} delta={boot['observed_delta_ap']:+.6f}")
        del ae, enc, dec, Xr_train, Xr_valid, Xr_test, model
        gc.collect()

    summary = {
        "diagnosis": "why_replace_v_with_ae_reconstruction_underperforms",
        "split_strategy": split_strategy,
        "seed": seed,
        "n_estimators": n_estimators,
        "n_v_features": len(v_cols),
        "baseline_test_average_precision": base_m["average_precision"],
        "baseline_v_importance_share": v_share,
        "latent_dim_sweep": sweep,
    }
    save_json(summary, output_dir / "diagnosis_summary.json")

    print("\nDiagnosis: replace-V dengan rekonstruksi AE (stratified)")
    print("========================================================")
    print(f"Baseline AP={base_m['average_precision']:.6f} | V importance share={v_share:.1%} ({len(v_cols)} fitur V)")
    print(f"{'latent_dim':>10s} {'kompresi':>9s} {'recon_R2':>9s} {'test_AP':>10s} {'d_vs_base':>11s} {'p':>7s}")
    for r in sweep:
        print(f"{r['latent_dim']:>10d} {r['compression_ratio']:>8.1f}x {r['recon_r2_test']:>9.4f} "
              f"{r['test_average_precision']:>10.6f} {r['delta_vs_baseline']:>+11.6f} {r['p_delta_le_0']:>7.3f}")
    print(f"\nSaved: {output_dir / 'diagnosis_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose why replace-V AE reconstruction underperforms.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dims", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    p.add_argument("--ae-epochs", type=int, default=30)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(output_dir=a.output_dir, split_strategy=a.split_strategy, latent_dims=tuple(a.latent_dims),
         ae_epochs=a.ae_epochs, n_estimators=a.n_estimators, n_bootstrap=a.n_bootstrap, seed=a.seed)
