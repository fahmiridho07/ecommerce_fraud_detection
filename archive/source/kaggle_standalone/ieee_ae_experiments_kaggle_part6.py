"""IEEE-CIS AE experiments PART 6 (Kaggle) — menutup CELAH: relasional & kategorikal.

Dua titik buta LightGBM yang BELUM kita garap (semua part 1-5 per-baris & numerik):

  entity_ae      : RELASIONAL. Bangun profil agregat per-entitas (count & statistik
                   TransactionAmt per card1/addr1/email/DeviceInfo) dari TRAIN, lalu
                   AE memampatkan profil itu -> embedding entitas ditempel ke tiap
                   transaksi. Memberi info "perilaku entitas" yang TAK BISA diturunkan
                   LightGBM dari satu baris. (proxy graph/entity-aggregation, ala juara IEEE-CIS)
  cat_embedding  : KATEGORIKAL. Entity-embedding terpelajar (Guo & Berkhahn 2016) untuk
                   kategori berkardinalitas tinggi (card1/card2/card5/addr1/email) via
                   jaringan supervised -> vektor embedding padat ditempel sbg fitur.
                   Memberi METRIK KEMIRIPAN antar kategori yang split LightGBM tak punya.

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000 vs baseline.

PAKAI: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_experiment_results_part6.json
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
AE_EPOCHS = 30
N_BOOTSTRAP = 2000
TARGET, ID = "isFraud", "TransactionID"
ENT_KEYS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
EMB_COLS = ["card1", "card2", "card5", "addr1", "P_emaildomain", "R_emaildomain"]
EMB_DIM = 4

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
    return Xtr, Xva, Xte, cat_cols


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


def plain_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(64, activation="relu")(inp)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(z)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


# ---------- RELASIONAL: profil agregat entitas (tanpa label) ----------
def entity_profile_matrix(train_raw, frames):
    """Untuk tiap key: count + statistik TransactionAmt (mean/std/max) dari TRAIN,
    dipetakan ke setiap baris di tiap frame. Tanpa memakai label (anti-bocor)."""
    aggs = {}
    glob = {"count": 0.0, "mean": float(train_raw["TransactionAmt"].mean()),
            "std": float(train_raw["TransactionAmt"].std()), "max": float(train_raw["TransactionAmt"].max())}
    for k in ENT_KEYS:
        g = train_raw.groupby(k)["TransactionAmt"].agg(["count", "mean", "std", "max"])
        aggs[k] = g
    cols = []
    for k in ENT_KEYS:
        for s in ("count", "mean", "std", "max"):
            cols.append(f"{k}__{s}")
    out = []
    for df in frames:
        mat = np.zeros((len(df), len(cols)), dtype="float32")
        ci = 0
        for k in ENT_KEYS:
            g = aggs[k]
            for s in ("count", "mean", "std", "max"):
                vals = df[k].map(g[s]).fillna(glob[s] if s != "count" else 0.0).astype("float32")
                mat[:, ci] = vals.to_numpy(); ci += 1
        out.append(np.nan_to_num(mat))
    return out, cols


# ---------- KATEGORIKAL: entity embeddings terpelajar ----------
def build_codes(train_raw, frames, col):
    uniques = pd.Index(train_raw[col].dropna().unique())
    mp = {v: i + 1 for i, v in enumerate(uniques)}  # 0 = unknown/NaN
    card = len(uniques) + 1
    arrs = [df[col].map(mp).fillna(0).astype("int32").to_numpy() for df in frames]
    return arrs, card


def embedding_net(cards):
    inputs, embs = [], []
    for card in cards:
        inp = keras.Input(shape=(1,), dtype="int32")
        e = keras.layers.Embedding(card, EMB_DIM)(inp)
        e = keras.layers.Flatten()(e)
        inputs.append(inp); embs.append(e)
    cat = keras.layers.Concatenate()(embs)
    x = keras.layers.Dense(64, activation="relu")(cat)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model, keras.Model(inputs, cat)


def main():
    print("Loading...")
    df = load_data()
    train, valid, test = split_60_20_20(df)
    del df
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    # simpan kolom mentah yang dibutuhkan SEBELUM preprocess
    raw_keep = list(set(ENT_KEYS + EMB_COLS + ["TransactionAmt"]))
    train_raw = train[raw_keep].copy(); valid_raw = valid[raw_keep].copy(); test_raw = test[raw_keep].copy()
    Xtr, Xva, Xte, cat_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} cat={len(cat_cols)}")

    print("\n[baseline] training...")
    bm, bit = train_lgbm(Xtr, ytr, Xva, yva, cat_cols, spw)
    base_va = bm.predict_proba(Xva, num_iteration=bit)[:, 1]
    base_te = bm.predict_proba(Xte, num_iteration=bit)[:, 1]
    results = {"baseline": metrics(yte, base_te, pick_threshold(yva, base_va))}
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

    # ---- 1) entity_ae (relasional) ----
    print("\n[entity_ae] building entity profiles + AE embedding...")
    (Ptr, Pva, Pte), pcols = entity_profile_matrix(train_raw, [train_raw, valid_raw, test_raw])
    sc = StandardScaler().fit(Ptr)
    Ptr_s = np.clip(sc.transform(Ptr), -10, 10).astype("float32")
    Pva_s = np.clip(sc.transform(Pva), -10, 10).astype("float32")
    Pte_s = np.clip(sc.transform(Pte), -10, 10).astype("float32")
    ld = 8
    ae, enc = plain_ae(Ptr_s.shape[1], ld)
    ae.fit(Ptr_s, Ptr_s, validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
    cols = [f"ent_{j}" for j in range(ld)]
    def add_ent(X, Ps):
        L = pd.DataFrame(enc.predict(Ps, batch_size=8192, verbose=0).astype("float32"), columns=cols, index=X.index)
        return pd.concat([X, L], axis=1)
    eval_variant("entity_ae", add_ent(Xtr, Ptr_s), add_ent(Xva, Pva_s), add_ent(Xte, Pte_s))

    # ---- 2) cat_embedding (kategorikal) ----
    print("\n[cat_embedding] learning entity embeddings for high-card categoricals...")
    keras.backend.clear_session()
    code_tr, code_va, code_te, cards = [], [], [], []
    for c in EMB_COLS:
        arrs, card = build_codes(train_raw, [train_raw, valid_raw, test_raw], c)
        code_tr.append(arrs[0]); code_va.append(arrs[1]); code_te.append(arrs[2]); cards.append(card)
        print(f"  {c}: cardinality {card}")
    model, emb_model = embedding_net(cards)
    cw = {0: 1.0, 1: spw}
    model.fit(code_tr, ytr.astype("float32"), validation_data=(code_va, yva.astype("float32")),
              epochs=AE_EPOCHS, batch_size=2048, shuffle=True, class_weight=cw,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)], verbose=2)
    ecols = [f"emb_{j}" for j in range(EMB_DIM * len(EMB_COLS))]
    def add_emb(X, codes):
        E = emb_model.predict(codes, batch_size=8192, verbose=0).astype("float32")
        return pd.concat([X, pd.DataFrame(E, columns=ecols, index=X.index)], axis=1)
    eval_variant("cat_embedding", add_emb(Xtr, code_tr), add_emb(Xva, code_va), add_emb(Xte, code_te))

    print("\n================ RINGKASAN PART 6 (full data, stratified) ================")
    print(f"{'skenario':16s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:16s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    with open("/kaggle/working/ae_experiment_results_part6.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part6.json")


if __name__ == "__main__":
    main()
