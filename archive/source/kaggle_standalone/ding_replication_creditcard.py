"""POSITIVE CONTROL / REPLIKASI — uji kebenaran implementasi AE+LightGBM kita
pada dataset REFERENSI Ding et al. (2024): ULB credit-card fraud (31 fitur).

Tujuan (sesuai arahan Pak Arif): membuktikan kode AE+LightGBM kita BENAR dengan
mereproduksi kasus di mana AE seharusnya membantu. Jika di dataset Ding pun AE-fitur
gagal namun SMOTE yang menaikkan, itu memvalidasi kode kita SEKALIGUS mengonfirmasi
temuan kita (peningkatan = oversampling, bukan AE-fitur).

Dekomposisi (semua dibanding baseline, paired bootstrap pada AP):
  baseline        : LightGBM fitur asli
  ae_recon        : AE merekonstruksi seluruh fitur -> ganti -> LightGBM  (efek AE-fitur, gaya Ding)
  ae_concat       : fitur asli + latent AE                                 (AE menambah)
  smote           : SMOTE oversampling -> LightGBM                          (efek oversampling)
  ae_recon_smote  : AE rekonstruksi + SMOTE -> LightGBM                     (klaim penuh Ding)

Metrik: ROC-AUC (utama di Ding), PR-AUC, F1, MCC, Recall. Bandingkan AUC dgn ~0.968 (Ding).

CARA PAKAI DI KAGGLE:
1. Add Input -> "Credit Card Fraud Detection" (mlg-ulb/creditcardfraud).
   Path: /kaggle/input/creditcardfraud/creditcard.csv
2. Tempel file ini ke satu cell, Run All.
Hasil -> /kaggle/working/ding_replication_results.json
(Dataset kecil ~284k x 31 -> bisa juga jalan di laptop biasa.)
"""

from __future__ import annotations

import argparse
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras

CSV = os.environ.get("DING_CSV", "/kaggle/input/creditcardfraud/creditcard.csv")
OUT_JSON = os.environ.get("DING_OUT", "/kaggle/working/ding_replication_results.json")
SEED = 42
N_ESTIMATORS = 1000
EARLY_STOPPING = 100
LATENT_DIM = 16
AE_EPOCHS = 40
N_BOOTSTRAP = 2000
TARGET = "Class"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def train_lgbm(Xtr, ytr, Xva, yva, spw):
    params = dict(objective="binary", boosting_type="gbdt", n_estimators=N_ESTIMATORS,
                  learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                  random_state=SEED, metric="None", verbosity=-1)
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
    return dict(roc_auc=float(roc_auc_score(yte, pte)),
                average_precision=float(average_precision_score(yte, pte)),
                f1=float(f1_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)),
                recall=float(recall_score(yte, pred)))


def bootstrap_delta(yte, ref, cand, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED); nrow = len(yte)
    obs = average_precision_score(yte, cand) - average_precision_score(yte, ref)
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, nrow, nrow)
        d[i] = average_precision_score(yte[s], cand[s]) - average_precision_score(yte[s], ref[s])
    return dict(observed_delta_ap=float(obs), ci_2_5=float(np.percentile(d, 2.5)),
                ci_97_5=float(np.percentile(d, 97.5)), p_delta_le_0=float((d <= 0).mean()))


def build_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(64, activation="relu")(inp)
    x = keras.layers.Dense(32, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(z)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def smote(Xn, y, k=5, seed=SEED):
    """SMOTE sederhana di ruang fitur (scaled). Menyeimbangkan minoritas ke jumlah mayoritas."""
    rng = np.random.default_rng(seed)
    Xmin = Xn[y == 1]; n_maj = int((y == 0).sum()); n_need = n_maj - len(Xmin)
    if n_need <= 0:
        return Xn, y
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    _, idx = nn.kneighbors(Xmin)
    anchors = rng.integers(0, len(Xmin), n_need)
    synth = np.empty((n_need, Xn.shape[1]), dtype="float32")
    for i, a in enumerate(anchors):
        nbrs = idx[a][1:]
        b = nbrs[rng.integers(0, len(nbrs))]
        lam = rng.random()
        synth[i] = Xmin[a] + lam * (Xmin[b] - Xmin[a])
    Xb = np.vstack([Xn, synth]).astype("float32")
    yb = np.concatenate([y, np.ones(n_need, int)])
    return Xb, yb


def main():
    print("Loading creditcard.csv ...")
    df = pd.read_csv(CSV)
    y = df[TARGET].to_numpy()
    feats = [c for c in df.columns if c != TARGET]
    X = df[feats].astype("float32")
    print(f"rows={len(df)} feats={len(feats)} fraud_rate={y.mean():.5f}")

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    Xtr = Xtr.reset_index(drop=True); Xva = Xva.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} spw={spw:.1f}")

    # scaler untuk AE & SMOTE (fit train)
    sc = StandardScaler().fit(Xtr.values)
    def S(d): return np.clip(sc.transform(d.values).astype("float32"), -10, 10)
    Str, Sva, Ste = S(Xtr), S(Xva), S(Xte)

    results, scores = {}, {}

    # baseline
    print("\n[baseline] ...")
    bm, bit = train_lgbm(Xtr, ytr, Xva, yva, spw)
    base_va = bm.predict_proba(Xva, num_iteration=bit)[:, 1]
    base_te = bm.predict_proba(Xte, num_iteration=bit)[:, 1]
    results["baseline"] = metrics(yte, base_te, pick_threshold(yva, base_va))
    print(f"baseline ROC-AUC={results['baseline']['roc_auc']:.5f} AP={results['baseline']['average_precision']:.5f}")

    def evaluate(name, Xtr2, ytr2, Xva2, Xte2, spw2):
        m, it = train_lgbm(Xtr2, ytr2, Xva2, yva, spw2)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        mm["bootstrap_vs_baseline"] = bootstrap_delta(yte, base_te, vte)
        results[name] = mm; scores[name] = vte
        d = mm["bootstrap_vs_baseline"]
        print(f"[{name}] ROC={mm['roc_auc']:.5f} AP={mm['average_precision']:.5f} "
              f"dAP={d['observed_delta_ap']:+.5f} p={d['p_delta_le_0']:.3f}")

    # AE rekonstruksi (gaya Ding): ganti fitur dgn rekonstruksi
    print("\n[ae_recon] training AE...")
    ae, enc = build_ae(Str.shape[1], LATENT_DIM)
    ae.fit(Str, Str, validation_split=0.1, epochs=AE_EPOCHS, batch_size=512, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
    def recon(s): return sc.inverse_transform(ae.predict(s, batch_size=8192, verbose=0)).astype("float32")
    Rtr = pd.DataFrame(recon(Str), columns=feats); Rva = pd.DataFrame(recon(Sva), columns=feats); Rte = pd.DataFrame(recon(Ste), columns=feats)
    evaluate("ae_recon", Rtr, ytr, Rva, Rte, spw)

    # AE concat (asli + latent)
    print("\n[ae_concat] ...")
    def addlat(Xdf, s):
        L = pd.DataFrame(enc.predict(s, batch_size=8192, verbose=0).astype("float32"),
                         columns=[f"ae_{j}" for j in range(LATENT_DIM)], index=Xdf.index)
        return pd.concat([Xdf, L], axis=1)
    evaluate("ae_concat", addlat(Xtr, Str), ytr, addlat(Xva, Sva), addlat(Xte, Ste), spw)

    # SMOTE saja (train di-balance; spw=1)
    print("\n[smote] ...")
    Xb, yb = smote(Str, ytr)
    Xb_raw = pd.DataFrame(sc.inverse_transform(Xb), columns=feats)
    evaluate("smote", Xb_raw, yb, Xva, Xte, 1.0)

    # AE rekonstruksi + SMOTE (klaim penuh Ding)
    print("\n[ae_recon_smote] ...")
    Rtr_s = np.clip(sc.transform(Rtr.values).astype("float32"), -10, 10)
    Xb2, yb2 = smote(Rtr_s, ytr)
    Xb2_raw = pd.DataFrame(sc.inverse_transform(Xb2), columns=feats)
    evaluate("ae_recon_smote", Xb2_raw, yb2, Rva, Rte, 1.0)

    print("\n================ REPLIKASI DING — creditcard ULB (stratified 60/20/20) ================")
    print(f"(Ding melaporkan AUC ~0.968 dgn SMOTE+AE+LightGBM)")
    print(f"{'skenario':16s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Recall':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>9s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:16s} {v['roc_auc']:9.5f} {v['average_precision']:9.5f} {v['recall']:8.4f} "
              f"{v['f1']:8.4f} {v['mcc']:8.4f} {d.get('observed_delta_ap',0):+9.5f} {d.get('p_delta_le_0',float('nan')):7.3f}")
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nInterpretasi:")
    print(" - ae_recon > baseline  -> AE-fitur membantu di data Ding -> kode kita BENAR (IEEE-CIS negatif itu nyata).")
    print(" - ae_recon ~/< baseline TAPI smote > baseline -> kode BENAR; peningkatan Ding dari OVERSAMPLING, bukan AE-fitur.")
    print(" - semua jauh di bawah ~0.968 -> periksa implementasi.")
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
