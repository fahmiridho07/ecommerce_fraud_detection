"""IEEE-CIS AE experiments PART 2 (Kaggle) — alternatif lanjutan, full data.

Tiga eksperimen, semua tetap dalam koridor "AE menghasilkan fitur untuk LightGBM"
(sesuai arahan: tetap di tujuan proposal, perbaiki dari penyebab):

  sae               : SUPERVISED autoencoder — laten dilatih utk rekonstruksi V
                      DAN memprediksi fraud, sehingga laten menyimpan sinyal
                      diskriminatif fraud (memperbaiki sebab: AE unsupervised
                      menghaluskan sinyal langka). Laten -> fitur LightGBM.
  perfeat_anomaly   : AE one-class (dilatih normal) -> error rekonstruksi PER-FITUR
                      (339 kolom), bukan ringkasan -> LightGBM tahu fitur mana anomali.
  allnum_anomaly    : AE one-class pada SELURUH fitur numerik (V+C+D+dist+addr+...),
                      skor anomali global (mean/max/std) -> fitur LightGBM.

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000 vs baseline.

PAKAI: sama seperti part 1 — Add Input "IEEE-CIS Fraud Detection", tempel file ini
ke satu cell, Run All. Hasil -> /kaggle/working/ae_experiment_results_part2.json
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

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 32
AE_EPOCHS = 30
N_BOOTSTRAP = 2000
TARGET, ID = "isFraud", "TransactionID"

np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ---------------- data ----------------
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
    return Xtr, Xva, Xte, cat_cols, num_cols, v_cols


# ---------------- model & util ----------------
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


def scaler_for(Xtr, cols):
    imp = SimpleImputer(strategy="median").fit(Xtr[cols])
    sc = StandardScaler().fit(imp.transform(Xtr[cols]))
    def f(X): return np.clip(sc.transform(imp.transform(X[cols])).astype("float32"), -10, 10)
    return f


def plain_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def supervised_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu", name="latent")(x)
    d = keras.layers.Dense(128, activation="relu")(z)
    d = keras.layers.Dense(256, activation="relu")(d)
    recon = keras.layers.Dense(dim, activation="linear", name="recon")(d)
    clf = keras.layers.Dense(1, activation="sigmoid", name="clf")(z)
    model = keras.Model(inp, [recon, clf])
    # List-based losses (Keras 3 safe): order = [recon, clf]. clf ditekan kuat lewat
    # loss_weight tinggi sebagai ganti sample_weight (yang bermasalah di Keras 3).
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=["mse", "binary_crossentropy"], loss_weights=[1.0, 8.0])
    return model, keras.Model(inp, z)


# ---------------- main ----------------
def main():
    print("Loading...")
    train, valid, test = split_60_20_20(load_data())
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    Xtr, Xva, Xte, cat_cols, num_cols, v_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} cat={len(cat_cols)} num={len(num_cols)} V={len(v_cols)}")

    print("\n[baseline] training...")
    bm, bit = train_lgbm(Xtr, ytr, Xva, yva, cat_cols, spw)
    base_va = bm.predict_proba(Xva, num_iteration=bit)[:, 1]
    base_te = bm.predict_proba(Xte, num_iteration=bit)[:, 1]
    base_thr = pick_threshold(yva, base_va)
    results = {"baseline": metrics(yte, base_te, base_thr)}
    print("baseline AP =", round(results["baseline"]["average_precision"], 6))

    def eval_variant(name, Xtr2, Xva2, Xte2):
        m, it = train_lgbm(Xtr2, ytr, Xva2, yva, cat_cols, spw)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        mm["bootstrap_vs_baseline"] = bootstrap_delta(yte, base_te, vte)
        results[name] = mm
        d = mm["bootstrap_vs_baseline"]
        print(f"[{name}] AP={mm['average_precision']:.6f} delta={d['observed_delta_ap']:+.6f} "
              f"ci=[{d['ci_2_5']:+.5f},{d['ci_97_5']:+.5f}] p={d['p_delta_le_0']:.3f}")

    Vf = scaler_for(Xtr, v_cols)
    Vtr, Vva, Vte = Vf(Xtr), Vf(Xva), Vf(Xte)

    # ---- 1) Supervised AE: latent (fraud-aware) sebagai fitur ----
    print("\n[sae] training supervised autoencoder...")
    model, enc = supervised_ae(Vtr.shape[1], LATENT_DIM)
    model.fit(Vtr, [Vtr, ytr.astype("float32")],
              validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
              verbose=2)
    cols = [f"sae_{j}" for j in range(LATENT_DIM)]
    def add_lat(X, V):
        L = pd.DataFrame(enc.predict(V, batch_size=8192, verbose=0).astype("float32"), columns=cols, index=X.index)
        return pd.concat([X, L], axis=1)
    eval_variant("sae_latent", add_lat(Xtr, Vtr), add_lat(Xva, Vva), add_lat(Xte, Vte))

    # ---- 2) Per-feature reconstruction error (one-class) ----
    print("\n[perfeat_anomaly] one-class AE, per-feature error (339 dims)...")
    keras.backend.clear_session()
    ae, _ = plain_ae(Vtr.shape[1], LATENT_DIM)
    nmask = ytr == 0
    ae.fit(Vtr[nmask], Vtr[nmask], validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
    ecols = [f"err_{c}" for c in v_cols]
    def add_perr(X, V):
        E = pd.DataFrame(np.abs(V - ae.predict(V, batch_size=8192, verbose=0)).astype("float32"), columns=ecols, index=X.index)
        return pd.concat([X, E], axis=1)
    eval_variant("perfeat_anomaly", add_perr(Xtr, Vtr), add_perr(Xva, Vva), add_perr(Xte, Vte))

    # ---- 3) One-class AE pada SEMUA fitur numerik ----
    print("\n[allnum_anomaly] one-class AE on ALL numeric features...")
    keras.backend.clear_session()
    Nf = scaler_for(Xtr, num_cols)
    Ntr, Nva, Nte = Nf(Xtr), Nf(Xva), Nf(Xte)
    ae2, _ = plain_ae(Ntr.shape[1], min(64, LATENT_DIM * 2))
    ae2.fit(Ntr[nmask], Ntr[nmask], validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
    def add_anom(X, N):
        e = np.abs(N - ae2.predict(N, batch_size=8192, verbose=0))
        out = X.copy()
        out["allnum_mean"] = e.mean(1).astype("float32")
        out["allnum_max"] = e.max(1).astype("float32")
        out["allnum_std"] = e.std(1).astype("float32")
        return out
    eval_variant("allnum_anomaly", add_anom(Xtr, Ntr), add_anom(Xva, Nva), add_anom(Xte, Nte))

    # ---- ringkasan ----
    print("\n================ RINGKASAN PART 2 (full data, stratified) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    with open("/kaggle/working/ae_experiment_results_part2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part2.json")


if __name__ == "__main__":
    main()
