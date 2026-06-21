"""IEEE-CIS — EKSPLORASI (BUKAN bagian tesis; ruang lingkup tesis tetap terkunci).

Pertanyaan yang diuji
---------------------
Diagnosis sebelumnya: mengganti blok V dengan rekonstruksi AE MENURUNKAN PR-AUC
(baseline 0.8218 -> 0.7690). Penyebab: AE meminimalkan MSE rata-rata seluruh fitur
secara SETARA, sehingga "menghaluskan" deviasi halus pada V yang justru menandai
fraud langka (efek low-pass). Pertanyaan eksplorasi:

  Jika loss rekonstruksi DITIMBANG per-fitur dengan importance LightGBM (gain),
  sehingga AE DIPAKSA mempertahankan fitur diskriminatif alih-alih fitur ber-varians
  besar — apakah informasi penting yang biasanya hilang bisa dipulihkan?

Ini menyerang langsung akar masalah dan BELUM pernah diuji (14 varian sebelumnya
melakukan SELEKSI fitur top-k, bukan PENIMBANGAN loss). Tetap dalam kerangka
"AE sebagai feature extractor" (selaras proposal asli), murni untuk diagnosis.

Skenario (semua: non-V + [representasi V] -> LightGBM, TANPA SMOTE, agar kontribusi
fitur-AE terisolasi vs baseline — sama seperti uji feature-level negatif terdahulu):
  baseline          : non-V + V mentah (informasi penuh, acuan)
  replace_uniform   : non-V + rekonstruksi V (AE MSE seragam)           [KONTROL]
  replace_iw        : non-V + rekonstruksi V (AE MSE ditimbang gain)    [IDE BARU]
  concat_iw         : non-V + V mentah + laten AE-iw (augmentasi)       [tak buang info]

Hipotesis jujur: replace_iw memulihkan sebagian AP yang hilang di replace_uniform
tetapi kemungkinan tetap < baseline; concat_iw kemungkinan seri (tie) dengan baseline
seperti varian concat sebelumnya. Hasil negatif tetap informatif: menegaskan batas
fundamental AE-feature pada IEEE-CIS (V Vesta sudah padat/engineered).

Protokol: data berstrata 60/20/20, seed 42, metrik utama PR-AUC, threshold di validasi
(MCC), paired bootstrap. AE & scaler hanya di-fit pada train.

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_importance_weighted_results.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 64
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 2048
N_BOOTSTRAP = 2000
TARGET, ID = "isFraud", "TransactionID"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def load_data():
    tx = pd.read_csv(f"{DATA_DIR}/train_transaction.csv")
    idf = pd.read_csv(f"{DATA_DIR}/train_identity.csv")
    df = tx.merge(idf, on=ID, how="left"); del tx, idf
    return df


def split_60_20_20(df):
    y = df[TARGET].to_numpy(); idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=y[tmp])
    return (df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True),
            df.iloc[te].reset_index(drop=True))


def preprocess_numeric(train, valid, test):
    """Frequency-encode kategorikal, imputasi median, standarisasi. Fit hanya pada train."""
    drop = [c for c in (ID, TARGET) if c in train.columns]
    feat = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in feat if train[c].dtype == "object"]
    num_cols = [c for c in feat if c not in cat_cols]
    out = {}
    for nm, df in (("tr", train), ("va", valid), ("te", test)):
        cols = {}
        for c in cat_cols:
            freq = train[c].value_counts(normalize=True)
            cols[c] = df[c].map(freq).fillna(0.0).astype("float32")
        for c in num_cols:
            cols[c] = df[c].fillna(float(train[c].median())).astype("float32")
        out[nm] = pd.DataFrame(cols)
    order = list(out["tr"].columns)
    mu = out["tr"].mean(); sd = out["tr"].std().replace(0, 1.0)
    for nm in out:
        out[nm] = ((out[nm][order] - mu) / sd).clip(-10, 10).astype("float32")
    v_cols = [c for c in order if c.startswith("V") and c[1:].isdigit()]
    return out["tr"].to_numpy(), out["va"].to_numpy(), out["te"].to_numpy(), order, v_cols


def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def default_params(spw):
    return dict(objective="binary", boosting_type="gbdt", n_estimators=N_ESTIMATORS,
                learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                random_state=SEED, metric="None", verbosity=-1)


def fit_lgbm(params, Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=ap_eval,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True)])
    return m, int(m.best_iteration_ or N_ESTIMATORS)


def pick_threshold(yva, pva):
    best_t, best = 0.5, -1
    for t in np.arange(0.01, 1.0, 0.01):
        v = matthews_corrcoef(yva, (pva >= t).astype(int))
        if v > best: best, best_t = v, t
    return float(best_t)


def metrics(yte, pte, thr):
    pred = (pte >= thr).astype(int)
    return dict(average_precision=float(average_precision_score(yte, pte)),
                roc_auc=float(roc_auc_score(yte, pte)),
                f1=float(f1_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)))


def bootstrap_delta(yte, ref, cand, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED); nrow = len(yte)
    obs = average_precision_score(yte, cand) - average_precision_score(yte, ref)
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, nrow, nrow)
        d[i] = average_precision_score(yte[s], cand[s]) - average_precision_score(yte[s], ref[s])
    return dict(observed_delta_ap=float(obs), ci_2_5=float(np.percentile(d, 2.5)),
                ci_97_5=float(np.percentile(d, 97.5)), p_delta_le_0=float((d <= 0).mean()))


def v_importance_weights(Xtr_v, ytr, Xva_v, yva, v_cols, spw):
    """Latih LightGBM ringan HANYA pada V utk dapat gain importance per fitur V.
    Bobot rekonstruksi = importance dinormalisasi (mean=1) agar skala loss tetap wajar.
    Fit hanya pada train+valid (early stopping); tidak menyentuh test."""
    p = default_params(spw); p["n_estimators"] = 400; p["learning_rate"] = 0.05
    m, it = fit_lgbm(p, Xtr_v, ytr, Xva_v, yva)
    gain = np.asarray(m.booster_.feature_importance(importance_type="gain"), dtype="float64")
    if gain.sum() <= 0:
        return np.ones(len(v_cols), dtype="float32")
    w = gain / gain.mean()                      # mean(w) = 1
    w = np.clip(w, 0.05, 20.0)                  # cegah fitur dominan menelan loss
    return w.astype("float32")


def build_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    return keras.Model(inp, out), keras.Model(inp, z)


def train_ae(Xtr_v, weights):
    """AE dengan loss MSE ditimbang per-fitur. weights=None -> MSE seragam (kontrol)."""
    ae, enc = build_ae(Xtr_v.shape[1], LATENT_DIM)
    if weights is None:
        loss = "mse"
    else:
        w = tf.constant(weights.reshape(1, -1), dtype=tf.float32)
        def weighted_mse(y_true, y_pred):
            return tf.reduce_mean(tf.square(y_true - y_pred) * w, axis=-1)
        loss = weighted_mse
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    ae.fit(Xtr_v, Xtr_v, validation_split=0.1, epochs=AE_EPOCHS, batch_size=AE_BATCH,
           shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])
    return ae, enc


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, cols, v_cols = preprocess_numeric(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    vmask = np.array([c in set(v_cols) for c in cols])
    nonv = ~vmask
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} feats={len(cols)} V={int(vmask.sum())}")

    results, scores = {}, {}

    def evaluate(name, Xtr2, Xva2, Xte2):
        m, it = fit_lgbm(default_params(spw), Xtr2, ytr, Xva2, yva)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    # ---- baseline: non-V + V mentah (informasi penuh) ----
    print("\n--- baseline (non-V + V mentah) ---")
    evaluate("baseline", Xtr, Xva, Xte)

    # ---- importance per fitur V (untuk penimbangan loss) ----
    print("\nMenghitung importance gain per fitur V (LightGBM ringan, train/valid saja)...")
    w = v_importance_weights(Xtr[:, vmask], ytr, Xva[:, vmask], yva, v_cols, spw)
    print(f"  bobot: min={w.min():.3f} median={np.median(w):.3f} max={w.max():.3f} "
          f"(>1: {int((w > 1).sum())}/{len(w)} fitur diutamakan)")

    def recon(ae):
        return (ae.predict(Xtr[:, vmask], batch_size=8192, verbose=0),
                ae.predict(Xva[:, vmask], batch_size=8192, verbose=0),
                ae.predict(Xte[:, vmask], batch_size=8192, verbose=0))

    def latent(enc):
        return (enc.predict(Xtr[:, vmask], batch_size=8192, verbose=0),
                enc.predict(Xva[:, vmask], batch_size=8192, verbose=0),
                enc.predict(Xte[:, vmask], batch_size=8192, verbose=0))

    # ---- KONTROL: replace_uniform (AE MSE seragam) ----
    print("\nTraining AE (MSE seragam) [KONTROL]...")
    ae_u, _ = train_ae(Xtr[:, vmask], None)
    rtr, rva, rte = recon(ae_u)
    print("\n--- replace_uniform (non-V + rekonstruksi V seragam) ---")
    evaluate("replace_uniform",
             np.hstack([Xtr[:, nonv], rtr]).astype("float32"),
             np.hstack([Xva[:, nonv], rva]).astype("float32"),
             np.hstack([Xte[:, nonv], rte]).astype("float32"))
    del ae_u, rtr, rva, rte

    # ---- IDE BARU: replace_iw (AE MSE ditimbang importance) ----
    print("\nTraining AE (MSE ditimbang importance) [IDE BARU]...")
    ae_w, enc_w = train_ae(Xtr[:, vmask], w)
    rtr, rva, rte = recon(ae_w)
    print("\n--- replace_iw (non-V + rekonstruksi V ditimbang) ---")
    evaluate("replace_iw",
             np.hstack([Xtr[:, nonv], rtr]).astype("float32"),
             np.hstack([Xva[:, nonv], rva]).astype("float32"),
             np.hstack([Xte[:, nonv], rte]).astype("float32"))
    del rtr, rva, rte

    # ---- concat_iw: non-V + V mentah + laten AE-iw (tak membuang info) ----
    ltr, lva, lte = latent(enc_w)
    print("\n--- concat_iw (non-V + V mentah + laten AE-iw) ---")
    evaluate("concat_iw",
             np.hstack([Xtr, ltr]).astype("float32"),
             np.hstack([Xva, lva]).astype("float32"),
             np.hstack([Xte, lte]).astype("float32"))

    comp = {
        "replace_uniform_vs_baseline": bootstrap_delta(yte, scores["baseline"], scores["replace_uniform"]),
        "replace_iw_vs_baseline":      bootstrap_delta(yte, scores["baseline"], scores["replace_iw"]),
        "replace_iw_vs_uniform":       bootstrap_delta(yte, scores["replace_uniform"], scores["replace_iw"]),
        "concat_iw_vs_baseline":       bootstrap_delta(yte, scores["baseline"], scores["concat_iw"]),
    }

    print("\n========== EKSPLORASI — AE importance-weighted (data berstrata 60/20/20) ==========")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k in ("baseline", "replace_uniform", "replace_iw", "concat_iw"):
        v = results[k]; print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k, b in comp.items():
        print(f"  {k:30s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "ae_importance_weighted_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp,
                   "v_weight_stats": {"min": float(w.min()), "median": float(np.median(w)),
                                      "max": float(w.max()), "n_emphasized": int((w > 1).sum())},
                   "config": {"latent_dim": LATENT_DIM, "ae_epochs": AE_EPOCHS}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
