"""IEEE-CIS — STRATEGI MUTAKHIR (EKSPLORASI lanjutan; diagnosis-driven).

DIAGNOSIS (kenapa Ding/Du gagal di IEEE-CIS):
  - Ding pakai one-class AE -> error rekonstruksi sbg skor anomali. Itu hanya bekerja bila
    fraud = POINT-ANOMALY (OOD per-baris), benar di ULB creditcard (fitur PCA bersih).
    Di IEEE-CIS fraud bersifat KONTEKSTUAL/RELASIONAL (menyimpang relatif thd riwayat
    entitas), jadi error per-baris buta konteks -> tak ada sinyal. <-- celah utama Ding.
  - Du memampatkan SEMUA fitur heterogen (NaN + cat high-card) -> laten lossy parah.
  - Bukti internal: AE hanya menolong saat diberi profil ENTITAS; entity_raw (+0.0207) =
    kemenangan bersih terbesar. Gap-nya RELASIONAL/TEMPORAL, tak terlihat per-baris.

STRATEGI (tiga lapis, AE tetap sentral -> judul/proposal aman):
  L1. uid (card1+addr1+(hari-D1)) + fitur entitas/temporal (count, amt mean/std/ratio,
      waktu sejak txn terakhir per uid). Sinyal relasional yang hilang.
  L2. CONTEXTUAL AE (perbaikan Ding): AE merekonstruksi fitur transaksi dari KONTEKS
      entitas -> residu = deviasi txn dari kebiasaan kartunya = anomali kontekstual.
  L3. AE-latent SMOTE pada representasi dense diperkaya (peran AE yang terbukti menang).

Skenario (stratified 60/20/20, seed 42, PR-AUC utama, threshold MCC, paired bootstrap;
semua statistik entitas & AE & SMOTE di-FIT pada TRAIN saja -> bebas leakage):
  baseline         : fitur numerik mentah -> LightGBM
  entity           : baseline + fitur uid/entitas/temporal (L1)            [diharapkan menang]
  entity_ctxae     : entity + contextual-AE residual (L2)                  [perbaikan Ding]
  entity_smote     : entity + SMOTE biasa                                   [kontrol]
  entity_aesmote   : entity + AE-latent SMOTE (L3)                          [peran AE menang]

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/entity_contextual_ae_results.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 32
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 2048
SMOTE_RATE = 0.15        # rasio rendah (PR-AUC optimum biasanya rendah, bukan 0.5)
K_NEIGHBORS = 5
N_BOOTSTRAP = 2000
TARGET, ID = "isFraud", "TransactionID"
DAY = 86400

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
    return tr, va, te


# ----------------------- L1: fitur entitas / uid / temporal -----------------------
def build_entity_features(df, tr_idx):
    """uid = card1_addr1_(hari-D1). Agregasi & statistik di-FIT pada TRAIN saja (anti-leak).
    Waktu-sejak-txn-terakhir bersifat kausal (urut waktu per uid) -> aman lintas split."""
    d = df[["card1", "addr1", "D1", "TransactionAmt", "TransactionDT"]].copy()
    day = (d["TransactionDT"] / DAY).astype("float32")
    reg = np.floor(day - d["D1"].fillna(-999)).astype("float32")   # hari registrasi kartu
    uid = (d["card1"].astype("float32").astype("Int64").astype(str) + "_"
           + d["addr1"].astype("float32").astype("Int64").astype(str) + "_"
           + reg.astype("Int64").astype(str))
    d["uid"] = uid.values

    tr_mask = np.zeros(len(d), dtype=bool); tr_mask[tr_idx] = True

    # statistik per-uid di-fit pada TRAIN saja
    g = d.loc[tr_mask].groupby("uid")["TransactionAmt"]
    amt_mean = g.mean(); amt_std = g.std().fillna(0.0); cnt = d.loc[tr_mask].groupby("uid").size()
    glob_mean = float(d.loc[tr_mask, "TransactionAmt"].mean())

    feat = pd.DataFrame(index=d.index)
    feat["uid_count"] = d["uid"].map(cnt).fillna(0.0).astype("float32")
    m = d["uid"].map(amt_mean).fillna(glob_mean).astype("float32")
    s = d["uid"].map(amt_std).fillna(0.0).astype("float32")
    feat["uid_amt_mean"] = m
    feat["uid_amt_std"] = s
    # deviasi terstandar txn ini dari rata-rata kartunya (anomali sederhana)
    feat["uid_amt_z"] = ((d["TransactionAmt"].astype("float32") - m) / (s + 1e-3)).astype("float32")
    feat["uid_amt_ratio"] = (d["TransactionAmt"].astype("float32") / (m + 1e-3)).astype("float32")
    feat["card_reg_day"] = reg

    # waktu sejak txn terakhir untuk uid yang sama (kausal: urut waktu)
    order = d[["uid", "TransactionDT"]].copy()
    order["row"] = np.arange(len(order))
    order = order.sort_values(["uid", "TransactionDT"])
    dt_prev = order.groupby("uid")["TransactionDT"].diff()
    tsl = pd.Series(np.nan, index=order["row"].values)
    tsl.loc[order["row"].values] = dt_prev.values
    feat["uid_time_since_last"] = tsl.reindex(np.arange(len(d))).fillna(-1.0).astype("float32")
    return feat.reset_index(drop=True), d["uid"].values


# ----------------------- preprocessing numerik (utk SMOTE & AE) -----------------------
def preprocess_numeric(df, tr_idx):
    drop = [c for c in (ID, TARGET) if c in df.columns]
    feat = [c for c in df.columns if c not in drop]
    cat_cols = [c for c in feat if df[c].dtype == "object"]
    num_cols = [c for c in feat if c not in cat_cols]
    tr = df.iloc[tr_idx]
    cols = {}
    for c in cat_cols:
        freq = tr[c].value_counts(normalize=True)
        cols[c] = df[c].map(freq).fillna(0.0).astype("float32")
    for c in num_cols:
        cols[c] = df[c].fillna(float(tr[c].median())).astype("float32")
    X = pd.DataFrame(cols)
    order = list(X.columns)
    mu = X.iloc[tr_idx].mean(); sd = X.iloc[tr_idx].std().replace(0, 1.0)
    X = ((X[order] - mu) / sd).clip(-10, 10).astype("float32")
    return X, order


def zscore_fit(A, tr_idx):
    mu = A[tr_idx].mean(0); sd = A[tr_idx].std(0); sd[sd == 0] = 1.0
    return ((A - mu) / sd).clip(-10, 10).astype("float32")


# ----------------------- model utils -----------------------
def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def params(spw):
    return dict(objective="binary", boosting_type="gbdt", n_estimators=N_ESTIMATORS,
                learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                random_state=SEED, metric="None", verbosity=-1)


def fit_lgbm(p, Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(**p)
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


def smote(X, y, rate=SMOTE_RATE, k=K_NEIGHBORS, seed=SEED):
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]; n_maj = int((y == 0).sum())
    n_need = int(round(n_maj * rate / (1 - rate))) - len(Xmin)
    if n_need <= 0 or len(Xmin) < 2:
        return X, y
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    _, idx = nn.kneighbors(Xmin)
    anchors = rng.integers(0, len(Xmin), n_need)
    syn = np.empty((n_need, X.shape[1]), dtype="float32")
    for i, a in enumerate(anchors):
        nbrs = idx[a][1:]; b = nbrs[rng.integers(0, len(nbrs))]
        lam = rng.random(); syn[i] = Xmin[a] + lam * (Xmin[b] - Xmin[a])
    return np.vstack([X, syn]).astype("float32"), np.concatenate([y, np.ones(n_need, int)])


def build_ae(in_dim, out_dim, latent):
    inp = keras.Input(shape=(in_dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="linear")(x)
    dec1 = keras.layers.Dense(128, activation="relu")
    dec2 = keras.layers.Dense(256, activation="relu")
    dec3 = keras.layers.Dense(out_dim, activation="linear")
    out = dec3(dec2(dec1(z)))
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    enc = keras.Model(inp, z)
    # decoder berdiri sendiri: laten -> fitur (untuk decode sampel SMOTE sintetis)
    lat_in = keras.Input(shape=(latent,))
    dec = keras.Model(lat_in, dec3(dec2(dec1(lat_in))))
    return ae, enc, dec


def main():
    print("Loading...")
    df = load_data()
    y = df[TARGET].to_numpy()
    tr, va, te = split_60_20_20(df)
    ytr, yva, yte = y[tr], y[va], y[te]
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    print("L1: building entity/uid/temporal features...")
    ent_df, uid = build_entity_features(df, tr)
    ent = ent_df.to_numpy().astype("float32")
    ent = zscore_fit(ent, tr)
    n_uid = len(pd.unique(uid)); print(f"  uid unik={n_uid} fitur entitas={ent.shape[1]}")

    print("Preprocessing numeric base...")
    Xdf, cols = preprocess_numeric(df, tr)
    X = Xdf.to_numpy(); del Xdf
    print(f"  base feats={X.shape[1]}")

    Xe = np.hstack([X, ent]).astype("float32")   # representasi diperkaya entitas

    results, scores = {}, {}

    def run(name, M):
        Mtr, Mva, Mte = M[tr], M[va], M[te]
        m, it = fit_lgbm(params(spw), Mtr, ytr, Mva, yva)
        vva = m.predict_proba(Mva, num_iteration=it)[:, 1]
        vte = m.predict_proba(Mte, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")
        return m, it

    def run_smote(name, Mtr_full, idx_tr, Mva, Mte):
        Xs, ys = smote(Mtr_full, ytr)
        m, it = fit_lgbm(params(1.0), Xs, ys, Mva, yva)
        vva = m.predict_proba(Mva, num_iteration=it)[:, 1]
        vte = m.predict_proba(Mte, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    print("\n--- baseline ---"); run("baseline", X)
    print("\n--- entity (L1) ---"); run("entity", Xe)

    # ----- L2: contextual AE -> residual deviasi txn dari konteks entitas -----
    print("\nL2: training contextual AE (konteks entitas -> rekonstruksi fitur txn)...")
    # input = fitur entitas (konteks kartu); target = ringkas fitur transaksi penting.
    txn_idx = [i for i, c in enumerate(cols) if c in ("TransactionAmt", "ProductCD", "card1",
               "card2", "addr1", "dist1", "P_emaildomain", "C1", "C13", "D15")]
    Tgt = X[:, txn_idx]
    ctx_ae, _, _ = build_ae(ent.shape[1], Tgt.shape[1], LATENT_DIM)
    ctx_ae.fit(ent[tr], Tgt[tr], validation_data=(ent[va], Tgt[va]), epochs=AE_EPOCHS,
               batch_size=AE_BATCH, shuffle=True, verbose=2,
               callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                          patience=AE_PATIENCE, restore_best_weights=True)])
    pred = ctx_ae.predict(ent, batch_size=8192, verbose=0)
    ctx_res = np.hstack([
        np.mean((Tgt - pred) ** 2, axis=1, keepdims=True),
        np.mean(np.abs(Tgt - pred), axis=1, keepdims=True),
        (Tgt - pred).astype("float32"),     # residu per-fitur (deviasi kontekstual)
    ]).astype("float32")
    ctx_res = zscore_fit(ctx_res, tr)
    Xctx = np.hstack([Xe, ctx_res]).astype("float32")
    print("\n--- entity_ctxae (L2, perbaikan Ding) ---"); run("entity_ctxae", Xctx)
    tf.keras.backend.clear_session()

    # ----- L3: AE-latent SMOTE pada representasi entitas dense -----
    print("\nL3: training AE for latent-space SMOTE on enriched dense rep...")
    ae, enc, dec = build_ae(Xe.shape[1], Xe.shape[1], LATENT_DIM)
    ae.fit(Xe[tr], Xe[tr], validation_data=(Xe[va], Xe[va]), epochs=AE_EPOCHS,
           batch_size=AE_BATCH, shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])
    Ztr = enc.predict(Xe[tr], batch_size=8192, verbose=0).astype("float32")

    print("\n--- entity_smote (kontrol) ---")
    run_smote("entity_smote", Xe[tr], tr, Xe[va], Xe[te])

    # AE-latent SMOTE: oversample di ruang laten, decode kembali, gabung ke fitur asli
    print("\n--- entity_aesmote (L3, AE-latent SMOTE) ---")
    Zs, ys = smote(Ztr, ytr)
    syn_lat = Zs[len(Ztr):]
    syn_dec = dec.predict(syn_lat, batch_size=8192, verbose=0)  # decode sintetis ke ruang fitur
    Xtr_aug = np.vstack([Xe[tr], syn_dec]).astype("float32")
    m, it = fit_lgbm(params(1.0), Xtr_aug, ys, Xe[va], yva)
    vva = m.predict_proba(Xe[va], num_iteration=it)[:, 1]
    vte = m.predict_proba(Xe[te], num_iteration=it)[:, 1]
    mm = metrics(yte, vte, pick_threshold(yva, vva))
    results["entity_aesmote"] = mm; scores["entity_aesmote"] = vte
    print(f"[entity_aesmote] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
          f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    comp = {
        "entity_vs_baseline":       bootstrap_delta(yte, scores["baseline"], scores["entity"]),
        "entity_ctxae_vs_entity":   bootstrap_delta(yte, scores["entity"], scores["entity_ctxae"]),
        "entity_smote_vs_entity":   bootstrap_delta(yte, scores["entity"], scores["entity_smote"]),
        "entity_aesmote_vs_smote":  bootstrap_delta(yte, scores["entity_smote"], scores["entity_aesmote"]),
        "entity_aesmote_vs_baseline": bootstrap_delta(yte, scores["baseline"], scores["entity_aesmote"]),
    }

    print("\n========== STRATEGI MUTAKHIR — entity + contextual AE (stratified 60/20/20) ==========")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k in ("baseline", "entity", "entity_ctxae", "entity_smote", "entity_aesmote"):
        v = results[k]; print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k, b in comp.items():
        print(f"  {k:28s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "entity_contextual_ae_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp,
                   "config": {"latent_dim": LATENT_DIM, "smote_rate": SMOTE_RATE,
                              "n_uid": int(n_uid), "entity_feats": int(ent.shape[1])}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
