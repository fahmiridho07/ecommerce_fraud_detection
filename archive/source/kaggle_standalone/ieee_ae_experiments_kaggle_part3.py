"""IEEE-CIS AE experiments PART 3 (Kaggle) — tiga ide BARU, full data.

Distinct dari part 1/2. Tetap koridor "AE menghasilkan fitur untuk LightGBM":

  contrast_anomaly : latih DUA AE one-class — satu pada transaksi NORMAL, satu pada
                     transaksi FRAUD. Fitur = error_normal, error_fraud, dan
                     selisihnya. Ide: "baris ini lebih dekat manifold normal atau
                     manifold fraud?" (sinyal kontras yang GBDT tak punya).
  latent_distance  : AE one-class (normal) -> jarak MAHALANOBIS laten tiap baris ke
                     pusat laten-normal sebagai skor keanehan (anomaly berbasis jarak).
  sae_allnum       : SUPERVISED AE pada SEMUA fitur numerik (bukan hanya V) ->
                     laten fraud-aware -> fitur LightGBM (gabungan ide SAE + full-numeric).

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000 vs baseline.

PAKAI: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_experiment_results_part3.json
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


def fit_ae(ae, X, val_split=0.1):
    ae.fit(X, X, validation_split=val_split, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
           verbose=2)


def row_err(ae, V):
    return np.abs(V - ae.predict(V, batch_size=8192, verbose=0))


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
    nmask, fmask = (ytr == 0), (ytr == 1)

    # ---- 1) contrast_anomaly: normal-AE vs fraud-AE error + diff ----
    print("\n[contrast_anomaly] training normal-AE and fraud-AE...")
    ae_n, _ = plain_ae(Vtr.shape[1], LATENT_DIM); fit_ae(ae_n, Vtr[nmask])
    keras.backend.clear_session()
    ae_f, _ = plain_ae(Vtr.shape[1], LATENT_DIM); fit_ae(ae_f, Vtr[fmask])
    def add_contrast(X, V):
        en = row_err(ae_n, V).mean(1).astype("float32")
        ef = row_err(ae_f, V).mean(1).astype("float32")
        out = X.copy()
        out["err_normal"] = en
        out["err_fraud"] = ef
        out["err_diff"] = (en - ef).astype("float32")  # >0 => lebih mirip fraud
        return out
    eval_variant("contrast_anomaly", add_contrast(Xtr, Vtr), add_contrast(Xva, Vva), add_contrast(Xte, Vte))

    # ---- 2) latent_distance: Mahalanobis di laten normal-AE ----
    print("\n[latent_distance] one-class AE + Mahalanobis latent distance...")
    keras.backend.clear_session()
    ae_l, enc_l = plain_ae(Vtr.shape[1], LATENT_DIM); fit_ae(ae_l, Vtr[nmask])
    Ztr_n = enc_l.predict(Vtr[nmask], batch_size=8192, verbose=0)
    mu = Ztr_n.mean(0)
    cov = np.cov(Ztr_n, rowvar=False) + 1e-3 * np.eye(Ztr_n.shape[1])
    inv = np.linalg.inv(cov).astype("float32")
    mu = mu.astype("float32")
    def maha(V):
        z = enc_l.predict(V, batch_size=8192, verbose=0).astype("float32")
        diff = z - mu
        m = np.einsum("ij,jk,ik->i", diff, inv, diff)
        return np.sqrt(np.clip(m, 0, None)).astype("float32"), np.linalg.norm(diff, axis=1).astype("float32")
    def add_dist(X, V):
        md, ed = maha(V)
        out = X.copy()
        out["latent_maha"] = md
        out["latent_eucl"] = ed
        return out
    eval_variant("latent_distance", add_dist(Xtr, Vtr), add_dist(Xva, Vva), add_dist(Xte, Vte))

    # ---- 3) sae_allnum: supervised AE pada SEMUA fitur numerik ----
    print("\n[sae_allnum] supervised AE on ALL numeric features...")
    keras.backend.clear_session()
    Nf = scaler_for(Xtr, num_cols)
    Ntr, Nva, Nte = Nf(Xtr), Nf(Xva), Nf(Xte)
    ld = min(64, LATENT_DIM * 2)
    model, enc = supervised_ae(Ntr.shape[1], ld)
    model.fit(Ntr, [Ntr, ytr.astype("float32")],
              validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
              verbose=2)
    cols = [f"saen_{j}" for j in range(ld)]
    def add_lat(X, N):
        L = pd.DataFrame(enc.predict(N, batch_size=8192, verbose=0).astype("float32"), columns=cols, index=X.index)
        return pd.concat([X, L], axis=1)
    eval_variant("sae_allnum", add_lat(Xtr, Ntr), add_lat(Xva, Nva), add_lat(Xte, Nte))

    print("\n================ RINGKASAN PART 3 (full data, stratified) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    with open("/kaggle/working/ae_experiment_results_part3.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part3.json")


if __name__ == "__main__":
    main()
