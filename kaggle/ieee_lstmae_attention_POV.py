"""IEEE-CIS — LSTM-AE + attention sebagai feature extractor (POV dari precedent Prabha 2024).

Latar: Prabha & Priscilla (2024) memakai PERSIS pipeline usulan ini (AE feature extractor ->
booster) di IEEE-CIS dan melaporkan berhasil, dengan dua beda dari percobaan feedforward-AE
yang gagal: (1) arsitektur LSTM-AE + ATTENTION (bukan feedforward), (2) evaluasi pada
RECALL/F1 di threshold tuned (regime operasional), bukan AP global.

POV (faithful: hanya GANTI ARSITEKTUR AE; pipeline AE-feature->LightGBM & judul utuh):
  AE = LSTM-autoencoder + self-attention atas blok V (V di-reshape jadi sekuens chunk).
  Laten -> digabung dengan non-V -> LightGBM. (replace & concat diuji.)

Evaluasi dua-sisi (kunci dari Prabha): selain PR-AUC, laporkan METRIK OPERASIONAL tempat
AE diharapkan berkontribusi: Recall@FPR=0.1%/1%, Precision@top-k, F1 di threshold terbaik.

Skenario (stratified 60/20/20, seed 42, paired bootstrap AP; AE/scaler fit TRAIN saja):
  baseline           : non-V + semua V mentah -> LightGBM
  lstmae_replace     : non-V + laten LSTM-AE(V)            [usulan asli, arsitektur LSTM]
  lstmae_concat      : non-V + V mentah + laten LSTM-AE(V) [additive]

PAKAI DI KAGGLE (disarankan GPU): Add Input "IEEE-CIS Fraud Detection", Run All.
Hasil -> /kaggle/working/lstmae_attention_results.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score, precision_score, recall_score, roc_curve)
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 32
FEAT_PER_STEP = 16        # V di-reshape jadi sekuens (T, FEAT_PER_STEP)
AE_EPOCHS = 40
AE_PATIENCE = 6
AE_BATCH = 1024
N_BOOTSTRAP = 2000
TOPK_LIST = [100, 500, 1000]
FPR_LIST = [0.001, 0.01]
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


def to_sequence(Xv, feat_per_step):
    """Reshape (n, d_V) -> (n, T, feat_per_step) dgn padding (gaya LSTM-AE tabular)."""
    n, d = Xv.shape
    T = int(np.ceil(d / feat_per_step))
    pad = T * feat_per_step - d
    if pad:
        Xv = np.hstack([Xv, np.zeros((n, pad), dtype="float32")])
    return Xv.reshape(n, T, feat_per_step).astype("float32"), T


def build_lstmae_attention(T, F, latent):
    """LSTM-AE + self-attention. Encoder LSTM(return_seq) -> MHA -> pool -> laten;
    decoder RepeatVector -> LSTM -> TimeDistributed(Dense(F))."""
    inp = keras.Input(shape=(T, F))
    e = keras.layers.LSTM(64, return_sequences=True)(inp)
    att = keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(e, e)
    e = keras.layers.Add()([e, att])                      # residual attention
    pooled = keras.layers.GlobalAveragePooling1D()(e)
    z = keras.layers.Dense(latent, activation="linear", name="latent")(pooled)
    d = keras.layers.RepeatVector(T)(z)
    d = keras.layers.LSTM(64, return_sequences=True)(d)
    out = keras.layers.TimeDistributed(keras.layers.Dense(F))(d)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


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


def best_f1_threshold(yva, pva):
    best_t, best = 0.5, -1
    for t in np.unique(np.quantile(pva, np.linspace(0.5, 0.999, 80))):
        f = f1_score(yva, (pva >= t).astype(int))
        if f > best: best, best_t = f, float(t)
    return best_t


def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def operational_metrics(yte, pte, thr):
    pred = (pte >= thr).astype(int)
    out = {"f1_best": float(f1_score(yte, pred)),
           "recall_best": float(recall_score(yte, pred)),
           "precision_best": float(precision_score(yte, pred, zero_division=0))}
    for fpr in FPR_LIST:
        out[f"recall@fpr{fpr}"] = recall_at_fpr(yte, pte, fpr)
    order = np.argsort(-pte)
    for k in TOPK_LIST:
        topk = order[:k]
        out[f"precision@top{k}"] = float(yte[topk].mean())
    return out


def metrics(yte, pte, thr):
    pred = (pte >= thr).astype(int)
    base = dict(average_precision=float(average_precision_score(yte, pte)),
                roc_auc=float(roc_auc_score(yte, pte)),
                f1=float(f1_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)))
    base.update(operational_metrics(yte, pte, thr))
    return base


def bootstrap_delta(yte, ref, cand, n=N_BOOTSTRAP):
    rng = np.random.default_rng(SEED); nrow = len(yte)
    obs = average_precision_score(yte, cand) - average_precision_score(yte, ref)
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, nrow, nrow)
        d[i] = average_precision_score(yte[s], cand[s]) - average_precision_score(yte[s], ref[s])
    return dict(observed_delta_ap=float(obs), ci_2_5=float(np.percentile(d, 2.5)),
                ci_97_5=float(np.percentile(d, 97.5)), p_delta_le_0=float((d <= 0).mean()))


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, cols, v_cols = preprocess_numeric(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    vmask = np.array([c in set(v_cols) for c in cols]); nonv = ~vmask
    print(f"train={len(Xtr)} feats={len(cols)} V={int(vmask.sum())}")

    results, scores = {}, {}

    def evaluate(name, Mtr, Mva, Mte):
        m, it = fit_lgbm(params(spw), Mtr, ytr, Mva, yva)
        pva = m.predict_proba(Mva, num_iteration=it)[:, 1]
        pte = m.predict_proba(Mte, num_iteration=it)[:, 1]
        thr = best_f1_threshold(yva, pva)
        mm = metrics(yte, pte, thr)
        results[name] = mm; scores[name] = pte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1*={mm['f1_best']:.4f} R@1%={mm['recall@fpr0.01']:.4f} "
              f"R@0.1%={mm['recall@fpr0.001']:.4f} P@500={mm['precision@top500']:.4f}")

    print("\n--- baseline ---")
    evaluate("baseline", Xtr, Xva, Xte)

    print("\nTraining LSTM-AE + attention on V (sequence)...")
    Str, T = to_sequence(Xtr[:, vmask], FEAT_PER_STEP)
    Sva, _ = to_sequence(Xva[:, vmask], FEAT_PER_STEP)
    Ste, _ = to_sequence(Xte[:, vmask], FEAT_PER_STEP)
    ae, enc = build_lstmae_attention(T, FEAT_PER_STEP, LATENT_DIM)
    ae.fit(Str, Str, validation_data=(Sva, Sva), epochs=AE_EPOCHS, batch_size=AE_BATCH,
           shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])
    Lt = enc.predict(Str, batch_size=4096, verbose=0).astype("float32")
    Lv = enc.predict(Sva, batch_size=4096, verbose=0).astype("float32")
    Le = enc.predict(Ste, batch_size=4096, verbose=0).astype("float32")

    print("\n--- lstmae_replace (non-V + laten LSTM-AE) ---")
    evaluate("lstmae_replace",
             np.hstack([Xtr[:, nonv], Lt]).astype("float32"),
             np.hstack([Xva[:, nonv], Lv]).astype("float32"),
             np.hstack([Xte[:, nonv], Le]).astype("float32"))

    print("\n--- lstmae_concat (non-V + V mentah + laten LSTM-AE) ---")
    evaluate("lstmae_concat",
             np.hstack([Xtr, Lt]).astype("float32"),
             np.hstack([Xva, Lv]).astype("float32"),
             np.hstack([Xte, Le]).astype("float32"))

    comp = {}
    for name in results:
        if name != "baseline":
            comp[f"{name}_vs_baseline"] = bootstrap_delta(yte, scores["baseline"], scores[name])

    print("\n========== LSTM-AE + ATTENTION (Prabha POV) — stratified 60/20/20 ==========")
    hdr = f"{'skenario':16s} {'PR-AUC':>9s} {'ROC':>7s} {'F1*':>7s} {'R@1%':>7s} {'R@.1%':>7s} {'P@100':>7s} {'P@1000':>7s}"
    print(hdr)
    for k in results:
        v = results[k]
        print(f"{k:16s} {v['average_precision']:9.6f} {v['roc_auc']:7.4f} {v['f1_best']:7.4f} "
              f"{v['recall@fpr0.01']:7.4f} {v['recall@fpr0.001']:7.4f} "
              f"{v['precision@top100']:7.4f} {v['precision@top1000']:7.4f}")
    print("\nPerbandingan AP (paired bootstrap):")
    for k, b in comp.items():
        print(f"  {k:26s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "lstmae_attention_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp,
                   "config": {"latent_dim": LATENT_DIM, "feat_per_step": FEAT_PER_STEP,
                              "seq_len_T": int(T)}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
