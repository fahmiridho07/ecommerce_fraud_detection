"""IEEE-CIS AE experiments PART 4 (Kaggle) — tiga ide BARU, full data.

Distinct dari part 1/2/3. Tetap koridor "AE menghasilkan fitur untuk LightGBM":

  vae_anomaly      : Variational Autoencoder (probabilistik) dilatih pada transaksi
                     NORMAL -> error rekonstruksi sebagai skor anomali. Keluarga AE
                     berbeda; banyak dipakai di literatur fraud.
  missingness_ae   : AE pada MATRIKS BINER pola-missing seluruh fitur -> embedding
                     "record macam apa ini" sbg fitur (ide missingness, langsung uji).
  iforest_latent   : AE one-class (normal) -> laten -> IsolationForest skor anomali
                     pada laten sbg fitur (representasi AE + detektor anomali khusus).

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000 vs baseline.

PAKAI: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_experiment_results_part4.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
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
    """Integer-encode object cols; bangun juga MASK missing (sebelum imputasi)."""
    drop = [c for c in (ID, TARGET) if c in train.columns]
    feat = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in feat if train[c].dtype == "object"]
    # mask missing dari nilai ASLI (NaN), sebelum encoding
    Mtr = train[feat].isna().astype("float32")
    Mva = valid[feat].isna().astype("float32")
    Mte = test[feat].isna().astype("float32")
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
    return Xtr, Xva, Xte, cat_cols, num_cols, v_cols, Mtr, Mva, Mte


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


def plain_ae(dim, latent, out_act="linear", loss="mse"):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation=out_act)(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    return ae, keras.Model(inp, z)


def fit_ae(ae, X):
    ae.fit(X, X, validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)


class VAE(keras.Model):
    """VAE subclass — Keras 3 safe (train_step kustom, tanpa add_loss)."""
    def __init__(self, dim, latent, beta=1.0):
        super().__init__()
        self.beta = beta
        self.enc = keras.Sequential([keras.layers.Input((dim,)),
                                     keras.layers.Dense(256, activation="relu"),
                                     keras.layers.Dense(128, activation="relu")])
        self.zm = keras.layers.Dense(latent)
        self.zlv = keras.layers.Dense(latent)
        self.dec = keras.Sequential([keras.layers.Input((latent,)),
                                     keras.layers.Dense(128, activation="relu"),
                                     keras.layers.Dense(256, activation="relu"),
                                     keras.layers.Dense(dim)])
        self.tracker = keras.metrics.Mean(name="loss")

    def encode(self, x):
        h = self.enc(x); return self.zm(h), self.zlv(h)

    def reparam(self, zm, zlv):
        return zm + tf.exp(0.5 * zlv) * tf.random.normal(tf.shape(zm))

    def call(self, x):
        zm, zlv = self.encode(x); return self.dec(self.reparam(zm, zlv))

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            zm, zlv = self.encode(x)
            recon = self.dec(self.reparam(zm, zlv))
            rl = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + zlv - tf.square(zm) - tf.exp(zlv), axis=1))
            loss = rl + self.beta * kl
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        self.tracker.update_state(loss)
        return {"loss": self.tracker.result()}

    @property
    def metrics(self):
        return [self.tracker]

    def recon_error(self, X):
        zm, _ = self.encode(X)
        r = self.dec(zm).numpy()
        return np.abs(X - r)


def main():
    print("Loading...")
    train, valid, test = split_60_20_20(load_data())
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    Xtr, Xva, Xte, cat_cols, num_cols, v_cols, Mtr, Mva, Mte = preprocess(train, valid, test)
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
    nmask = ytr == 0

    # ---- 1) VAE anomaly (one-class, normal) ----
    print("\n[vae_anomaly] training VAE on NORMAL rows...")
    vae = VAE(Vtr.shape[1], LATENT_DIM, beta=1.0)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    vae.fit(Vtr[nmask], epochs=AE_EPOCHS, batch_size=2048, shuffle=True, verbose=2)
    def add_vae(X, V):
        e = vae.recon_error(V)
        out = X.copy()
        out["vae_mean"] = e.mean(1).astype("float32")
        out["vae_max"] = e.max(1).astype("float32")
        out["vae_std"] = e.std(1).astype("float32")
        return out
    eval_variant("vae_anomaly", add_vae(Xtr, Vtr), add_vae(Xva, Vva), add_vae(Xte, Vte))

    # ---- 2) missingness AE: embedding pola-missing ----
    print("\n[missingness_ae] AE on binary missingness mask...")
    keras.backend.clear_session()
    Mtr_a, Mva_a, Mte_a = Mtr.to_numpy(), Mva.to_numpy(), Mte.to_numpy()
    # buang kolom yang tak pernah / selalu missing (varian nol) agar AE stabil
    keep = (Mtr_a.mean(0) > 0.001) & (Mtr_a.mean(0) < 0.999)
    Mtr_a, Mva_a, Mte_a = Mtr_a[:, keep], Mva_a[:, keep], Mte_a[:, keep]
    print(f"  mask columns used: {int(keep.sum())}")
    ae_m, enc_m = plain_ae(Mtr_a.shape[1], LATENT_DIM, out_act="sigmoid", loss="binary_crossentropy")
    fit_ae(ae_m, Mtr_a)
    cols = [f"miss_{j}" for j in range(LATENT_DIM)]
    def add_miss(X, M):
        L = pd.DataFrame(enc_m.predict(M, batch_size=8192, verbose=0).astype("float32"), columns=cols, index=X.index)
        return pd.concat([X, L], axis=1)
    eval_variant("missingness_ae", add_miss(Xtr, Mtr_a), add_miss(Xva, Mva_a), add_miss(Xte, Mte_a))

    # ---- 3) IsolationForest on AE latent ----
    print("\n[iforest_latent] one-class AE latent -> IsolationForest score...")
    keras.backend.clear_session()
    ae_l, enc_l = plain_ae(Vtr.shape[1], LATENT_DIM)
    fit_ae(ae_l, Vtr[nmask])
    Ztr = enc_l.predict(Vtr, batch_size=8192, verbose=0)
    Zva = enc_l.predict(Vva, batch_size=8192, verbose=0)
    Zte = enc_l.predict(Vte, batch_size=8192, verbose=0)
    iforest = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    iforest.fit(Ztr[nmask])
    def add_if(X, Z):
        out = X.copy()
        out["iforest_score"] = (-iforest.score_samples(Z)).astype("float32")  # tinggi = makin anomali
        return out
    eval_variant("iforest_latent", add_if(Xtr, Ztr), add_if(Xva, Zva), add_if(Xte, Zte))

    print("\n================ RINGKASAN PART 4 (full data, stratified) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    with open("/kaggle/working/ae_experiment_results_part4.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part4.json")


if __name__ == "__main__":
    main()
