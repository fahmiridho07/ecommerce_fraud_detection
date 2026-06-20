"""IEEE-CIS AE feature experiments — versi MANDIRI untuk Kaggle (RAM 16-30 GB).

Tujuan: menjalankan, pada FULL DATA tanpa kendala RAM 8 GB, perbandingan adil:
  - baseline LightGBM (fitur asli, NaN-native, scale_pos_weight dari data)
  - one_class_anomaly : AE dilatih HANYA pada transaksi normal -> error rekonstruksi
                        blok V jadi SKOR ANOMALI (fitur baru utk LightGBM)  [ide utama]
  - recon_error       : AE dilatih pada SEMUA V -> error rekonstruksi sbg fitur
  - concat_latent     : V asli + latent AE sebagai fitur pelengkap

Protokol identik dgn kerja lokal: stratified holdout 60/20/20, seed 42,
metrik utama PR-AUC (Average Precision), threshold dipilih di validation (MCC),
signifikansi paired bootstrap (2000) terhadap baseline.

CARA PAKAI DI KAGGLE
--------------------
1. New Notebook -> Add Input -> cari "IEEE-CIS Fraud Detection" (competition data).
   Pastikan path: /kaggle/input/ieee-fraud-detection/train_transaction.csv & train_identity.csv
2. Settings -> Accelerator: None (CPU cukup) atau GPU; -> pastikan Internet off tak masalah.
3. Tempel SELURUH isi file ini ke satu cell, Run All. Hasil tabel tercetak di akhir
   dan tersimpan ke /kaggle/working/ae_experiment_results.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras

# ----------------------------- konfigurasi -----------------------------
DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000          # full budget; turunkan ke 800 utk iterasi cepat
EARLY_STOPPING = 100
LATENT_DIM = 32
AE_EPOCHS = 30
N_BOOTSTRAP = 2000
TARGET = "isFraud"
ID = "TransactionID"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ----------------------------- load & split -----------------------------
def load_data():
    tx = pd.read_csv(f"{DATA_DIR}/train_transaction.csv")
    idf = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    df = tx.merge(idf, on=ID, how="left")
    del tx, idf
    return df


def split_60_20_20(df):
    y = df[TARGET].to_numpy()
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=y[tmp])
    return df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def preprocess(train, valid, test):
    """Integer-encode kolom object (categorical) berbasis train; numerik NaN dibiarkan."""
    drop = [c for c in (ID, TARGET) if c in train.columns]
    feat = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in feat if train[c].dtype == "object"]

    def enc(df_):
        X = df_[feat].copy()
        return X

    Xtr, Xva, Xte = enc(train), enc(valid), enc(test)
    for c in cat_cols:
        codes, uniques = pd.factorize(Xtr[c], sort=True)
        mapping = {v: i for i, v in enumerate(uniques)}
        Xtr[c] = codes.astype("int32")
        Xva[c] = Xva[c].map(mapping).fillna(-1).astype("int32")
        Xte[c] = Xte[c].map(mapping).fillna(-1).astype("int32")
    # numerik -> float32 (hemat memori), NaN dipertahankan
    num_cols = [c for c in feat if c not in cat_cols]
    for X in (Xtr, Xva, Xte):
        X[num_cols] = X[num_cols].astype("float32")
    return Xtr, Xva, Xte, cat_cols, [c for c in feat if c.startswith("V") and c[1:].isdigit()]


# ----------------------------- model & util -----------------------------
def ap_eval(y_true, y_pred):
    return "ap", average_precision_score(y_true, y_pred), True


def train_lgbm(Xtr, ytr, Xva, yva, cat_cols, spw):
    params = dict(objective="binary", boosting_type="gbdt", n_estimators=N_ESTIMATORS,
                  learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0,
                  scale_pos_weight=spw, n_jobs=-1, random_state=SEED, metric="None", verbosity=-1)
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=ap_eval,
          categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True)])
    return m, int(m.best_iteration_ or N_ESTIMATORS)


def pick_threshold(yva, pva):
    best_t, best_mcc = 0.5, -1
    for t in np.arange(0.01, 1.0, 0.01):
        mcc = matthews_corrcoef(yva, (pva >= t).astype(int))
        if mcc > best_mcc:
            best_mcc, best_t = mcc, t
    return float(best_t)


def metrics(yte, pte, thr):
    pred = (pte >= thr).astype(int)
    return dict(average_precision=float(average_precision_score(yte, pte)),
                roc_auc=float(roc_auc_score(yte, pte)),
                f1=float(f1_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)))


def bootstrap_delta(yte, ref, cand, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED)
    nrow = len(yte)
    base = average_precision_score(yte, cand) - average_precision_score(yte, ref)
    deltas = np.empty(n)
    for i in range(n):
        s = rng.integers(0, nrow, nrow)
        deltas[i] = average_precision_score(yte[s], cand[s]) - average_precision_score(yte[s], ref[s])
    return dict(observed_delta_ap=float(base), ci_2_5=float(np.percentile(deltas, 2.5)),
                ci_97_5=float(np.percentile(deltas, 97.5)), p_delta_le_0=float((deltas <= 0).mean()))


def build_ae(input_dim, latent_dim, noise=0.0):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.GaussianNoise(noise)(inp) if noise > 0 else inp
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent_dim, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    enc = keras.Model(inp, z)
    return ae, enc


# ----------------------------- main -----------------------------
def main():
    print("Loading data...")
    df = load_data()
    train, valid, test = split_60_20_20(df)
    del df
    ytr, yva, yte = train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy()
    Xtr, Xva, Xte, cat_cols, v_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} | cat={len(cat_cols)} V={len(v_cols)} spw={spw:.1f}")

    # baseline
    print("\n[baseline] training...")
    bm, bit = train_lgbm(Xtr, ytr, Xva, yva, cat_cols, spw)
    base_va = bm.predict_proba(Xva, num_iteration=bit)[:, 1]
    base_te = bm.predict_proba(Xte, num_iteration=bit)[:, 1]
    thr = pick_threshold(yva, base_va)
    results = {"baseline": metrics(yte, base_te, thr)}
    print("baseline AP =", round(results["baseline"]["average_precision"], 6))

    # AE pada blok V (skala: impute median + standardize + clip), fit di train
    imp = SimpleImputer(strategy="median").fit(Xtr[v_cols])
    sc = StandardScaler().fit(imp.transform(Xtr[v_cols]))
    def Vs(X):
        return np.clip(sc.transform(imp.transform(X[v_cols])).astype("float32"), -10, 10)
    Vtr, Vva, Vte = Vs(Xtr), Vs(Xva), Vs(Xte)

    def eval_variant(name, Xtr2, Xva2, Xte2):
        m, it = train_lgbm(Xtr2, ytr, Xva2, yva, cat_cols, spw)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        t = pick_threshold(yva, vva)
        mm = metrics(yte, vte, t)
        mm["bootstrap_vs_baseline"] = bootstrap_delta(yte, base_te, vte)
        results[name] = mm
        d = mm["bootstrap_vs_baseline"]
        print(f"[{name}] AP={mm['average_precision']:.6f} delta={d['observed_delta_ap']:+.6f} "
              f"ci=[{d['ci_2_5']:+.5f},{d['ci_97_5']:+.5f}] p={d['p_delta_le_0']:.3f}")

    # --- ide utama: one-class anomaly (AE dilatih hanya pada normal) ---
    print("\n[one_class_anomaly] training AE on NORMAL rows only...")
    ae, _ = build_ae(Vtr.shape[1], LATENT_DIM)
    nmask = ytr == 0
    ae.fit(Vtr[nmask], Vtr[nmask], validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048,
           shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
           restore_best_weights=True)], verbose=2)
    def add_err(X, V, tag):
        e = np.abs(V - ae.predict(V, batch_size=8192, verbose=0))
        out = X.copy()
        out[f"{tag}_mean"] = e.mean(1).astype("float32")
        out[f"{tag}_max"] = e.max(1).astype("float32")
        out[f"{tag}_std"] = e.std(1).astype("float32")
        return out
    eval_variant("one_class_anomaly",
                 add_err(Xtr, Vtr, "anom"), add_err(Xva, Vva, "anom"), add_err(Xte, Vte, "anom"))

    # --- recon_error: AE dilatih semua V ---
    print("\n[recon_error] training AE on ALL V...")
    keras.backend.clear_session()
    ae2, _ = build_ae(Vtr.shape[1], LATENT_DIM)
    ae2.fit(Vtr, Vtr, validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
            restore_best_weights=True)], verbose=2)
    def add_err2(X, V):
        e = np.abs(V - ae2.predict(V, batch_size=8192, verbose=0))
        out = X.copy()
        out["recon_mean"] = e.mean(1).astype("float32")
        out["recon_max"] = e.max(1).astype("float32")
        out["recon_std"] = e.std(1).astype("float32")
        return out
    eval_variant("recon_error", add_err2(Xtr, Vtr), add_err2(Xva, Vva), add_err2(Xte, Vte))

    # --- concat_latent: V asli + latent AE ---
    print("\n[concat_latent] training AE on ALL V, append latent...")
    keras.backend.clear_session()
    ae3, enc3 = build_ae(Vtr.shape[1], LATENT_DIM)
    ae3.fit(Vtr, Vtr, validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
            restore_best_weights=True)], verbose=2)
    cols = [f"lat_{j}" for j in range(LATENT_DIM)]
    def add_lat(X, V):
        L = pd.DataFrame(enc3.predict(V, batch_size=8192, verbose=0).astype("float32"), columns=cols, index=X.index)
        return pd.concat([X, L], axis=1)
    eval_variant("concat_latent", add_lat(Xtr, Vtr), add_lat(Xva, Vva), add_lat(Xte, Vte))

    # ----------------------------- ringkasan -----------------------------
    print("\n================ RINGKASAN (full data, stratified) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    with open("/kaggle/working/ae_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results.json")


if __name__ == "__main__":
    main()
