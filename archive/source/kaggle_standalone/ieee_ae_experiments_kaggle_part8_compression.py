"""IEEE-CIS PART 8 (Kaggle) — Studi KOMPRESI: AE vs PCA sebagai feature extractor.

Tetap 100% pada tujuan awal (AE sebagai feature extractor / dimensionality
reduction ala Ding), tanpa peran baru. Pertanyaan diubah dari "apakah AE menaikkan
AP?" menjadi "sebagai kompresor, apakah encoding NON-LINEAR autoencoder lebih baik
daripada kompresi LINEAR (PCA) pada dimensi yang sama?".

Untuk tiap k in {16,32,64,128}: blok fitur V (339 kolom) DIGANTI oleh k komponen,
fitur non-V dipertahankan, lalu LightGBM. Tiga representasi dibandingkan adil:
  - vfull   : V utuh (baseline, tanpa kompresi)
  - pca_k   : V -> PCA k komponen
  - ae_k    : V -> AE latent k dimensi

Perbandingan kunci (paired bootstrap): ae_k vs pca_k pada k yang sama
-> mengisolasi kontribusi NON-LINEARITAS autoencoder.

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000. Budget 800 trees (cukup utk kontras relatif; ubah bila perlu).

PAKAI: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_experiment_results_part8.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 800
EARLY_STOPPING = 100
AE_EPOCHS = 30
N_BOOTSTRAP = 2000
K_LIST = [16, 32, 64, 128]
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


def preprocess(train, valid, test):
    drop = [c for c in (ID, TARGET) if c in train.columns]
    feat = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in feat if train[c].dtype == "object"]
    Xtr, Xva, Xte = train[feat].copy(), valid[feat].copy(), test[feat].copy()
    for c in cat_cols:
        codes, uniques = pd.factorize(Xtr[c], sort=True)
        mp = {v: i for i, v in enumerate(uniques)}
        Xtr[c] = codes.astype("int32")
        Xva[c] = Xva[c].map(mp).fillna(-1).astype("int32")
        Xte[c] = Xte[c].map(mp).fillna(-1).astype("int32")
    num_cols = [c for c in feat if c not in cat_cols]
    for X in (Xtr, Xva, Xte):
        X[num_cols] = X[num_cols].astype("float32")
    v_cols = [c for c in feat if c.startswith("V") and c[1:].isdigit()]
    return Xtr, Xva, Xte, cat_cols, v_cols


def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def train_lgbm(Xtr, ytr, Xva, yva, cat_cols, spw):
    params = dict(objective="binary", boosting_type="gbdt", n_estimators=N_ESTIMATORS,
                  learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                  random_state=SEED, metric="None", verbosity=-1)
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=ap_eval, categorical_feature=cat_cols,
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


def build_encoder(dim, k):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(k, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def main():
    print("Loading...")
    train, valid, test = split_60_20_20(load_data())
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    Xtr, Xva, Xte, cat_cols, v_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} cat={len(cat_cols)} V={len(v_cols)}")

    # frame non-V (dipertahankan di semua skenario)
    nonV = [c for c in Xtr.columns if c not in v_cols]
    Xtr_nonV, Xva_nonV, Xte_nonV = Xtr[nonV].copy(), Xva[nonV].copy(), Xte[nonV].copy()

    # V ter-skala untuk PCA & AE
    imp = SimpleImputer(strategy="median").fit(Xtr[v_cols])
    sc = StandardScaler().fit(imp.transform(Xtr[v_cols]))
    def Vs(X): return np.clip(sc.transform(imp.transform(X[v_cols])).astype("float32"), -10, 10)
    Vtr, Vva, Vte = Vs(Xtr), Vs(Xva), Vs(Xte)
    del Xtr, Xva, Xte

    # baseline: V utuh
    print("\n[vfull] baseline (V utuh)...")
    def with_comp(nonV_df, comp, tag):
        cols = [f"{tag}_{j}" for j in range(comp.shape[1])]
        C = pd.DataFrame(comp, columns=cols, index=nonV_df.index)
        return pd.concat([nonV_df, C], axis=1)
    # untuk vfull pakai V mentah (bukan skala) supaya benar2 baseline asli
    # (gabung kembali V asli)
    # NOTE: kita rekonstruksi vfull dgn menambgrafik V scaled tidak setara; jadi latih vfull
    # langsung dgn V scaled sebagai referensi kompresi yang adil.
    Xtr_full = with_comp(Xtr_nonV, Vtr, "v"); Xva_full = with_comp(Xva_nonV, Vva, "v"); Xte_full = with_comp(Xte_nonV, Vte, "v")
    bm, bit = train_lgbm(Xtr_full, ytr, Xva_full, yva, cat_cols, spw)
    base_va = bm.predict_proba(Xva_full, num_iteration=bit)[:, 1]
    base_te = bm.predict_proba(Xte_full, num_iteration=bit)[:, 1]
    results = {"vfull": metrics(yte, base_te, pick_threshold(yva, base_va))}
    print("vfull (V utuh) AP =", round(results["vfull"]["average_precision"], 6))
    del Xtr_full, Xva_full, Xte_full

    # PCA(128) sekali, ambil k pertama
    print("\nFitting PCA(128) on scaled V...")
    pca = PCA(n_components=max(K_LIST), random_state=SEED).fit(Vtr)
    Ptr, Pva, Pte = pca.transform(Vtr).astype("float32"), pca.transform(Vva).astype("float32"), pca.transform(Vte).astype("float32")
    evr = float(pca.explained_variance_ratio_[:max(K_LIST)].sum())
    print(f"PCA-{max(K_LIST)} explained variance = {evr:.3f}")

    store = {}  # simpan skor test utk perbandingan ae vs pca
    def eval_named(name, Xtr2, Xva2, Xte2):
        m, it = train_lgbm(Xtr2, ytr, Xva2, yva, cat_cols, spw)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        mm["bootstrap_vs_vfull"] = bootstrap_delta(yte, base_te, vte)
        results[name] = mm; store[name] = vte
        d = mm["bootstrap_vs_vfull"]
        print(f"[{name}] AP={mm['average_precision']:.6f} vs_vfull={d['observed_delta_ap']:+.6f} p={d['p_delta_le_0']:.3f}")

    for k in K_LIST:
        # PCA-k
        eval_named(f"pca_{k}", with_comp(Xtr_nonV, Ptr[:, :k], "pca"),
                   with_comp(Xva_nonV, Pva[:, :k], "pca"), with_comp(Xte_nonV, Pte[:, :k], "pca"))
        # AE-k
        keras.backend.clear_session()
        ae, enc = build_encoder(Vtr.shape[1], k)
        ae.fit(Vtr, Vtr, validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
        Atr = enc.predict(Vtr, batch_size=8192, verbose=0)
        Ava = enc.predict(Vva, batch_size=8192, verbose=0)
        Ate = enc.predict(Vte, batch_size=8192, verbose=0)
        eval_named(f"ae_{k}", with_comp(Xtr_nonV, Atr, "ae"),
                   with_comp(Xva_nonV, Ava, "ae"), with_comp(Xte_nonV, Ate, "ae"))
        # KUNCI: ae_k vs pca_k
        b = bootstrap_delta(yte, store[f"pca_{k}"], store[f"ae_{k}"])
        results[f"ae_vs_pca_{k}"] = b
        print(f"  >> ae_{k} vs pca_{k}: delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(ae<=pca)={b['p_delta_le_0']:.3f}")

    print("\n================ RINGKASAN PART 8 — KOMPRESI (full data, stratified, 800 trees) ================")
    print(f"vfull (V utuh) AP = {results['vfull']['average_precision']:.6f}  | PCA-128 explained var = {evr:.3f}")
    print(f"\n{'k':>4s} {'pca_AP':>9s} {'ae_AP':>9s} {'ae−pca':>9s} {'p(ae<=pca)':>11s}")
    for k in K_LIST:
        pa = results[f"pca_{k}"]["average_precision"]; aa = results[f"ae_{k}"]["average_precision"]
        b = results[f"ae_vs_pca_{k}"]
        print(f"{k:>4d} {pa:9.6f} {aa:9.6f} {b['observed_delta_ap']:+9.6f} {b['p_delta_le_0']:>11.3f}")
    with open("/kaggle/working/ae_experiment_results_part8.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part8.json")
    print("Baca: ae−pca > 0 dan p kecil  -> non-linearitas AE berkontribusi (klaim simpel & kuat).")


if __name__ == "__main__":
    main()
