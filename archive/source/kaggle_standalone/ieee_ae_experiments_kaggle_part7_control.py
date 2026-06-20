"""IEEE-CIS PART 7 (Kaggle) — KONTROL ADIL untuk pemenang Part 6.

Part 6 menunjukkan entity_ae (+0.0171) dan cat_embedding (+0.0188) MENGALAHKAN
baseline. Pertanyaan wajib: kemenangan dari AUTOENCODER-nya, atau dari FITUR
relasional/entitas yang mendasarinya? Part 7 menjawab dengan kontrol tanpa AE:

  entity_raw    : 16 fitur agregat entitas MENTAH (count + amt mean/std/max per
                  card1/addr1/P_emaildomain/DeviceInfo) langsung ke LightGBM,
                  TANPA autoencoder. Kontrol untuk entity_ae.
  target_encode : OOF smoothed target-encoding (KFold anti-bocor) untuk kategori
                  yang sama (card1/card2/card5/addr1/email). Kontrol untuk cat_embedding.

Interpretasi:
  - AE >= kontrol  -> autoencoder memberi nilai tambah (klaim terkuat)
  - AE  ≈ kontrol  -> yang membantu adalah FITUR relasional; AE setara cara sederhana

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000 vs baseline.

PAKAI: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_experiment_results_part7.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
N_BOOTSTRAP = 2000
TARGET, ID = "isFraud", "TransactionID"
ENT_KEYS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
TE_COLS = ["card1", "card2", "card5", "addr1", "P_emaildomain", "R_emaildomain"]
TE_FOLDS = 5
TE_ALPHA = 20.0  # smoothing

np.random.seed(SEED)


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


def entity_aggregates(train_raw, frames):
    """16 fitur agregat entitas (tanpa label), dipetakan ke tiap frame."""
    aggs, glob = {}, {"mean": float(train_raw["TransactionAmt"].mean()),
                      "std": float(train_raw["TransactionAmt"].std()),
                      "max": float(train_raw["TransactionAmt"].max())}
    for k in ENT_KEYS:
        aggs[k] = train_raw.groupby(k)["TransactionAmt"].agg(["count", "mean", "std", "max"])
    cols = [f"{k}__{s}" for k in ENT_KEYS for s in ("count", "mean", "std", "max")]
    outs = []
    for df in frames:
        mat = np.zeros((len(df), len(cols)), dtype="float32"); ci = 0
        for k in ENT_KEYS:
            g = aggs[k]
            for s in ("count", "mean", "std", "max"):
                fill = 0.0 if s == "count" else glob[s]
                mat[:, ci] = df[k].map(g[s]).fillna(fill).astype("float32").to_numpy(); ci += 1
        outs.append(np.nan_to_num(mat))
    return outs, cols


def target_encode_oof(train_raw, ytr, valid_raw, test_raw):
    """OOF smoothed target encoding (anti-bocor) untuk TE_COLS."""
    gmean = float(ytr.mean())
    tr = np.zeros((len(train_raw), len(TE_COLS)), dtype="float32")
    va = np.zeros((len(valid_raw), len(TE_COLS)), dtype="float32")
    te = np.zeros((len(test_raw), len(TE_COLS)), dtype="float32")
    skf = StratifiedKFold(n_splits=TE_FOLDS, shuffle=True, random_state=SEED)
    for ci, c in enumerate(TE_COLS):
        s = pd.Series(ytr, index=train_raw.index)
        # OOF untuk train
        oof = np.full(len(train_raw), gmean, dtype="float32")
        for tri, vli in skf.split(train_raw, ytr):
            grp = pd.DataFrame({"k": train_raw[c].iloc[tri].values, "y": ytr[tri]})
            agg = grp.groupby("k")["y"].agg(["sum", "count"])
            enc = (agg["sum"] + gmean * TE_ALPHA) / (agg["count"] + TE_ALPHA)
            oof[vli] = train_raw[c].iloc[vli].map(enc).fillna(gmean).astype("float32").values
        tr[:, ci] = oof
        # valid/test pakai full-train encoding
        grp = pd.DataFrame({"k": train_raw[c].values, "y": ytr})
        agg = grp.groupby("k")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + gmean * TE_ALPHA) / (agg["count"] + TE_ALPHA)
        va[:, ci] = valid_raw[c].map(enc).fillna(gmean).astype("float32").values
        te[:, ci] = test_raw[c].map(enc).fillna(gmean).astype("float32").values
    return tr, va, te, [f"te_{c}" for c in TE_COLS]


def main():
    print("Loading...")
    df = load_data()
    train, valid, test = split_60_20_20(df); del df
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    raw_keep = list(set(ENT_KEYS + TE_COLS + ["TransactionAmt"]))
    train_raw, valid_raw, test_raw = train[raw_keep].copy(), valid[raw_keep].copy(), test[raw_keep].copy()
    Xtr, Xva, Xte, cat_cols = preprocess(train, valid, test); del train, valid, test
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

    # ---- kontrol 1: agregat entitas mentah (tanpa AE) ----
    print("\n[entity_raw] 16 raw entity aggregates (no AE)...")
    (Atr, Ava, Ate), acols = entity_aggregates(train_raw, [train_raw, valid_raw, test_raw])
    def add_raw(X, A):
        return pd.concat([X, pd.DataFrame(A, columns=acols, index=X.index)], axis=1)
    eval_variant("entity_raw", add_raw(Xtr, Atr), add_raw(Xva, Ava), add_raw(Xte, Ate))

    # ---- kontrol 2: OOF target encoding (tanpa embedding) ----
    print("\n[target_encode] OOF smoothed target encoding (no embedding)...")
    Ttr, Tva, Tte, tcols = target_encode_oof(train_raw, ytr, valid_raw, test_raw)
    def add_te(X, T):
        return pd.concat([X, pd.DataFrame(T, columns=tcols, index=X.index)], axis=1)
    eval_variant("target_encode", add_te(Xtr, Ttr), add_te(Xva, Tva), add_te(Xte, Tte))

    print("\n================ RINGKASAN PART 7 — KONTROL (full data, stratified) ================")
    print(f"{'skenario':16s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:16s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    print("\nBandingkan: entity_raw vs entity_ae(0.84002) | target_encode vs cat_embedding(0.84168)")
    with open("/kaggle/working/ae_experiment_results_part7.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part7.json")


if __name__ == "__main__":
    main()
