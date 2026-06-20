"""Perbaikan desain awal (AE memperbaiki fitur V) yang sejalan tujuan proposal.

Mengikuti arahan Pak Arif: dari penyebab (rekonstruksi AE menghaluskan sinyal
halus penanda fraud), uji perbaikan yang TETAP dalam koridor "AE memperbaiki
fitur untuk LightGBM" — bukan ganti metode.

Empat varian (pilih via --variant), semua dibandingkan ke baseline yang sama
(protokol stratified, fitur non-V dipertahankan apa adanya):

  concat     : V asli DIPERTAHANKAN + tambahkan latent AE sebagai fitur pelengkap
  denoise    : ganti V dengan rekonstruksi DENOISING-AE (input diberi noise besar,
               target bersih) -> AE membersihkan noise, bukan meniru identik
  recon_error: V asli dipertahankan + tambahkan fitur ERROR rekonstruksi (sinyal anomali)
  selective  : rekonstruksi & ganti HANYA kolom V dengan missing-rate tinggi

Memori: frame di-downcast IN-PLACE ke float32 + dtype category dan TIDAK pernah
disalin penuh (hindari OOM). Array V dibebaskan sebelum training LightGBM.
Jalankan SATU varian per proses.

Run:
    python archive/source/ae_appendix/run_ae_feature_improvements.py --variant concat \
        --output-dir outputs/stratified_reset/improve_concat --n-estimators 800
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
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_SPLIT_STRATEGY, RANDOM_SEED, SAMPLE_SIZE, SUPPORTED_SPLIT_STRATEGIES
from data_loader import load_labeled_train_data, merge_train_data, load_train_identity, TRAIN_TRANSACTION_FILE
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from splitting import create_holdout_split
from run_ae_augmentation_experiment import build_fraud_autoencoder, evaluate, paired_bootstrap_ap_delta, train_lgbm
from utils import ensure_dir, log, save_json, set_seed

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc

VARIANTS = ("concat", "denoise", "recon_error", "selective", "normal_recon_error")


def downcast_inplace(df: pd.DataFrame, categorical_columns: list[str]) -> None:
    """float32 for NUMERIC columns only (halves LightGBM memory vs float64). Categorical
    columns are left as their integer codes — converting high-cardinality columns to the
    pandas 'category' dtype makes LightGBM allocate huge bins and can exhaust RAM."""
    cat_set = set(categorical_columns)
    for col in df.columns:
        if col not in cat_set:
            df[col] = df[col].astype("float32")


def build_denoising_ae(input_dim: int, latent_dim: int, noise: float):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.GaussianNoise(noise)(inp)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent_dim, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae


def scaled_v(df, v_cols, imputer, scaler):
    return np.clip(scaler.transform(imputer.transform(df[v_cols])).astype("float32"), -10.0, 10.0)


def main(variant: str, output_dir: Path, split_strategy: str = DEFAULT_SPLIT_STRATEGY,
         latent_dim: int = 32, ae_epochs: int = 30, n_estimators: int = 800,
         n_bootstrap: int = 2000, seed: int = RANDOM_SEED, sample_size: int | None = None) -> dict:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)

    eff_sample = sample_size if sample_size is not None else SAMPLE_SIZE
    log(f"Loading data and splitting (stratified). sample_size={eff_sample}")
    if eff_sample is not None:
        # Low-memory path: read only the first N transaction rows (never materialize
        # the full ~1.65 GB CSV), then merge the small identity table.
        tx = pd.read_csv(TRAIN_TRANSACTION_FILE, nrows=int(eff_sample))
        full_df = merge_train_data(tx, load_train_identity())
        del tx
        gc.collect()
    else:
        full_df = load_labeled_train_data(sample_size=None)
    train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df
    gc.collect()

    X_train, y_train = split_features_target(train_df)
    X_valid, y_valid = split_features_target(valid_df)
    X_test, y_test = split_features_target(test_df)
    pre = fit_baseline_preprocessing(X_train)
    Xb_train = apply_baseline_preprocessing(X_train, pre)
    Xb_valid = apply_baseline_preprocessing(X_valid, pre)
    Xb_test = apply_baseline_preprocessing(X_test, pre)
    del X_train, X_valid, X_test, train_df, valid_df, test_df
    gc.collect()
    cat_cols = pre["categorical_columns"]
    v_cols = get_v_feature_columns(Xb_train)
    y_tr, y_va, y_te = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()

    # V-block scaler for AE (compute BEFORE downcast so NaN handling is identical).
    imputer = SimpleImputer(strategy="median").fit(Xb_train[v_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[v_cols]))

    # Downcast in place once (no copies) -> halves LightGBM memory, avoids float64 OOM.
    for df in (Xb_train, Xb_valid, Xb_test):
        downcast_inplace(df, cat_cols)
    gc.collect()

    # ----- Baseline (trained on the same frames, before they are modified) -----
    log("Training baseline LightGBM.")
    bm, bit = train_lgbm(Xb_train, y_tr, Xb_valid, y_va, cat_cols, n_estimators=n_estimators)
    base_valid = bm.predict_proba(Xb_valid, num_iteration=bit)[:, 1]
    base_test = bm.predict_proba(Xb_test, num_iteration=bit)[:, 1]
    base_m, _ = evaluate(y_va, base_valid, y_te, base_test)
    log(f"BASELINE AP={base_m['average_precision']:.6f}")
    del bm
    gc.collect()

    # ----- Build the variant by MODIFYING the frames in place -----
    Vtr = scaled_v(Xb_train, v_cols, imputer, scaler)
    Vva = scaled_v(Xb_valid, v_cols, imputer, scaler)
    Vte = scaled_v(Xb_test, v_cols, imputer, scaler)
    detail: dict = {"variant": variant}

    if variant == "concat":
        ae, enc, dec = build_fraud_autoencoder(Vtr.shape[1], latent_dim)
        ae.fit(Vtr, Vtr, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
        cols = [f"ae_latent_{j}" for j in range(latent_dim)]
        lat_tr = pd.DataFrame(enc.predict(Vtr, batch_size=4096, verbose=0).astype("float32"), columns=cols, index=Xb_train.index)
        lat_va = pd.DataFrame(enc.predict(Vva, batch_size=4096, verbose=0).astype("float32"), columns=cols, index=Xb_valid.index)
        lat_te = pd.DataFrame(enc.predict(Vte, batch_size=4096, verbose=0).astype("float32"), columns=cols, index=Xb_test.index)
        Xb_train = pd.concat([Xb_train, lat_tr], axis=1)  # single concat avoids frame fragmentation
        Xb_valid = pd.concat([Xb_valid, lat_va], axis=1)
        Xb_test = pd.concat([Xb_test, lat_te], axis=1)
        del lat_tr, lat_va, lat_te
        detail["added_features"] = latent_dim

    elif variant == "denoise":
        ae = build_denoising_ae(Vtr.shape[1], latent_dim, noise=0.25)
        ae.fit(Vtr, Vtr, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
        for X, V in ((Xb_train, Vtr), (Xb_valid, Vva), (Xb_test, Vte)):
            X[v_cols] = scaler.inverse_transform(ae.predict(V, batch_size=4096, verbose=0)).astype("float32")
        detail["noise"] = 0.25

    elif variant == "recon_error":
        ae, enc, dec = build_fraud_autoencoder(Vtr.shape[1], latent_dim)
        ae.fit(Vtr, Vtr, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
        for X, V in ((Xb_train, Vtr), (Xb_valid, Vva), (Xb_test, Vte)):
            err = np.abs(V - ae.predict(V, batch_size=4096, verbose=0))
            X["ae_recon_err_mean"] = err.mean(axis=1).astype("float32")
            X["ae_recon_err_max"] = err.max(axis=1).astype("float32")
            X["ae_recon_err_std"] = err.std(axis=1).astype("float32")
        detail["added_features"] = 3

    elif variant == "normal_recon_error":
        # One-class AE: train ONLY on normal rows so fraud reconstructs poorly ->
        # reconstruction error becomes an anomaly score (a NEW signal, not redundant).
        normal_mask = (y_tr == 0)
        Vtr_normal = Vtr[normal_mask]
        log(f"Training one-class AE on {Vtr_normal.shape[0]} normal rows only.")
        ae, enc, dec = build_fraud_autoencoder(Vtr.shape[1], latent_dim)
        ae.fit(Vtr_normal, Vtr_normal, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
        del Vtr_normal
        for X, V in ((Xb_train, Vtr), (Xb_valid, Vva), (Xb_test, Vte)):
            err = np.abs(V - ae.predict(V, batch_size=4096, verbose=0))
            X["ae_anom_err_mean"] = err.mean(axis=1).astype("float32")
            X["ae_anom_err_max"] = err.max(axis=1).astype("float32")
            X["ae_anom_err_std"] = err.std(axis=1).astype("float32")
        detail["added_features"] = 3
        detail["ae_trained_on"] = "normal_only"

    elif variant == "selective":
        miss_rate = Xb_train[v_cols].isna().mean()
        hi_missing = [c for c in v_cols if float(miss_rate[c]) >= 0.30]
        log(f"high-missing V columns (>=30% NaN): {len(hi_missing)} of {len(v_cols)}")
        if hi_missing:
            sub_imputer = SimpleImputer(strategy="median").fit(Xb_train[hi_missing])
            sub_scaler = StandardScaler().fit(sub_imputer.transform(Xb_train[hi_missing]))
            S = lambda df: np.clip(sub_scaler.transform(sub_imputer.transform(df[hi_missing])).astype("float32"), -10, 10)
            ld = min(latent_dim, max(2, len(hi_missing) // 2))
            ae, enc, dec = build_fraud_autoencoder(len(hi_missing), ld)
            Str = S(Xb_train)
            ae.fit(Str, Str, validation_split=0.1, epochs=ae_epochs, batch_size=2048, shuffle=True,
                   callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
            del Str
            for X in (Xb_train, Xb_valid, Xb_test):
                X[hi_missing] = sub_scaler.inverse_transform(ae.predict(S(X), batch_size=4096, verbose=0)).astype("float32")
        detail["reconstructed_columns"] = len(hi_missing)
    else:
        raise ValueError(variant)

    del Vtr, Vva, Vte
    keras.backend.clear_session()
    gc.collect()

    # ----- Train + evaluate the variant on the (modified) frames -----
    log(f"Training LightGBM for variant={variant} (features={Xb_train.shape[1]}).")
    vm, vit = train_lgbm(Xb_train, y_tr, Xb_valid, y_va, cat_cols, n_estimators=n_estimators)
    vsc = vm.predict_proba(Xb_valid, num_iteration=vit)[:, 1]
    tsc = vm.predict_proba(Xb_test, num_iteration=vit)[:, 1]
    vmetrics, vthr = evaluate(y_va, vsc, y_te, tsc)
    boot = paired_bootstrap_ap_delta(y_te, base_test, tsc, n_bootstrap=n_bootstrap)

    summary = {
        "variant": variant, "detail": detail, "split_strategy": split_strategy, "seed": seed,
        "n_estimators": n_estimators, "n_features_variant": int(Xb_train.shape[1]),
        "results": {
            "baseline": {
                "test_average_precision": base_m["average_precision"], "test_roc_auc": base_m["roc_auc"],
                "test_f1": base_m["f1"], "test_mcc": base_m["mcc"],
            },
            variant: {
                "test_average_precision": vmetrics["average_precision"], "test_roc_auc": vmetrics["roc_auc"],
                "test_f1": vmetrics["f1"], "test_mcc": vmetrics["mcc"], "selected_threshold": vthr,
                "bootstrap_vs_baseline": boot,
            },
        },
    }
    save_json(summary, output_dir / "experiment_summary.json")

    print(f"\nPerbaikan AE-fitur: variant={variant} (stratified)")
    print("=" * 52)
    print(f"baseline      AP={base_m['average_precision']:.6f}")
    print(f"{variant:13s} AP={vmetrics['average_precision']:.6f}  delta={boot['observed_delta_ap']:+.6f} "
          f"ci=[{boot['ci_2_5']:+.5f},{boot['ci_97_5']:+.5f}] p(d<=0)={boot['p_delta_le_0']:.3f}")
    verdict = "MENANG signifikan" if (boot["observed_delta_ap"] > 0 and boot["p_delta_le_0"] < 0.05) else (
        "tie/tidak signifikan" if boot["observed_delta_ap"] > 0 else "lebih buruk")
    print(f"VERDICT: {verdict}")
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE feature-improvement variants (stick-to-proposal).")
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--ae-epochs", type=int, default=30)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--sample-size", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(variant=a.variant, output_dir=a.output_dir, split_strategy=a.split_strategy,
         latent_dim=a.latent_dim, ae_epochs=a.ae_epochs, n_estimators=a.n_estimators,
         n_bootstrap=a.n_bootstrap, seed=a.seed, sample_size=a.sample_size)
