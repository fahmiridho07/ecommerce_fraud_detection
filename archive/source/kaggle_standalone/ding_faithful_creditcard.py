"""REPLIKASI FAITHFUL metode Ding et al. (2024) sesuai KODE ASLI mereka.

Berbeda dari replikasi pertama: di sini AE diperlakukan PERSIS seperti kode Ding —
  * AE dilatih HANYA pada transaksi NORMAL (one-class),
  * arsitektur Ding: input -> Dense(16, relu, L1=1e-5) -> Dense(8) -> Dense(8) -> Dense(input, relu),
  * OUTPUT yang dipakai = ERROR rekonstruksi (MSE & MAE) sebagai FITUR untuk LightGBM,
    BUKAN fitur rekonstruksi / laten yang menggantikan input.

Dekomposisi (vs baseline, paired bootstrap pada AP):
  baseline        : fitur asli -> LightGBM
  ae_error        : fitur asli + MSE + MAE (skor anomali AE) -> LightGBM   [integrasi Ding]
  ae_error_only   : HANYA MSE + MAE -> LightGBM                            [AE murni sbg detektor]
  smote           : fitur asli + SMOTE
  ae_error_smote  : fitur asli + MSE + MAE + SMOTE                          [klaim penuh Ding]

Preprocessing meniru Ding: Hour = Time//3600; StandardScaler pd Amount & Hour; V1-28 apa adanya.
Evaluasi: stratified 60/20/20 seed 42 utk LightGBM; AE dilatih pd baris normal di train split.

Jalankan (laptop):
  DING_CSV=".../creditcard.csv" DING_OUT=".../results.json" python ding_faithful_creditcard.py
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

CSV = os.environ.get("DING_CSV", "/kaggle/input/creditcardfraud/creditcard.csv")
OUT_JSON = os.environ.get("DING_OUT", "/kaggle/working/ding_faithful_results.json")
SEED = 42
N_ESTIMATORS = 1000
EARLY_STOPPING = 100
AE_EPOCHS = 60
N_BOOTSTRAP = 2000

np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)


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


def ding_ae(input_dim):
    """Arsitektur PERSIS Ding: 16(L1) -> 8 -> 8 -> input, relu, MSE."""
    inp = keras.Input(shape=(input_dim,))
    e = keras.layers.Dense(16, activation="relu", activity_regularizer=regularizers.l1(1e-5))(inp)
    e = keras.layers.Dense(8, activation="relu")(e)
    d = keras.layers.Dense(8, activation="relu")(e)
    d = keras.layers.Dense(input_dim, activation="relu")(d)
    ae = keras.Model(inp, d); ae.compile(optimizer="adam", loss="mse")
    return ae


def smote(Xn, y, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    Xmin = Xn[y == 1]; n_need = int((y == 0).sum()) - len(Xmin)
    if n_need <= 0 or len(Xmin) < 2:
        return Xn, y
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    _, idx = nn.kneighbors(Xmin)
    anchors = rng.integers(0, len(Xmin), n_need)
    synth = np.empty((n_need, Xn.shape[1]), dtype="float32")
    for i, a in enumerate(anchors):
        nbrs = idx[a][1:]; b = nbrs[rng.integers(0, len(nbrs))]
        lam = rng.random(); synth[i] = Xmin[a] + lam * (Xmin[b] - Xmin[a])
    return np.vstack([Xn, synth]).astype("float32"), np.concatenate([y, np.ones(n_need, int)])


def main():
    print("Loading + preprocessing (Ding-style)...")
    d = pd.read_csv(CSV)
    d["Hour"] = (d["Time"] // 3600).astype("float32")
    d[["Amount", "Hour"]] = StandardScaler().fit_transform(d[["Amount", "Hour"]])
    d.drop("Time", axis=1, inplace=True)
    feats = [c for c in d.columns if c != "Class"]
    X = d[feats].astype("float32"); y = d["Class"].to_numpy()
    print(f"rows={len(d)} feats={len(feats)} fraud_rate={y.mean():.5f}")

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    Xtr = Xtr.reset_index(drop=True); Xva = Xva.reset_index(drop=True); Xte = Xte.reset_index(drop=True)
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} spw={spw:.1f}")

    # AE one-class: latih HANYA pada normal di train split (persis Ding)
    print("\nTraining one-class AE on NORMAL train rows (Ding arch)...")
    Vtr = Xtr.values.astype("float32")
    normal = Vtr[ytr == 0]
    ae = ding_ae(Vtr.shape[1])
    ae.fit(normal, normal, epochs=AE_EPOCHS, batch_size=256, shuffle=True, validation_split=0.1,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)

    def recon_err(Xdf):
        v = Xdf.values.astype("float32")
        pred = ae.predict(v, batch_size=8192, verbose=0)
        mse = np.mean((v - pred) ** 2, axis=1).astype("float32")
        mae = np.mean(np.abs(v - pred), axis=1).astype("float32")
        return mse, mae

    mse_tr, mae_tr = recon_err(Xtr); mse_va, mae_va = recon_err(Xva); mse_te, mae_te = recon_err(Xte)

    def add_err(Xdf, mse, mae):
        out = Xdf.copy(); out["ae_mse"] = mse; out["ae_mae"] = mae; return out

    results, scores = {}, {}
    print("\n[baseline] ...")
    bm, bit = train_lgbm(Xtr, ytr, Xva, yva, spw)
    base_te = bm.predict_proba(Xte, num_iteration=bit)[:, 1]
    base_va = bm.predict_proba(Xva, num_iteration=bit)[:, 1]
    results["baseline"] = metrics(yte, base_te, pick_threshold(yva, base_va))
    print(f"baseline ROC={results['baseline']['roc_auc']:.5f} AP={results['baseline']['average_precision']:.5f}")

    def evaluate(name, Xtr2, ytr2, Xva2, Xte2, spw2):
        m, it = train_lgbm(Xtr2, ytr2, Xva2, yva, spw2)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        mm["bootstrap_vs_baseline"] = bootstrap_delta(yte, base_te, vte)
        results[name] = mm; scores[name] = vte
        b = mm["bootstrap_vs_baseline"]
        print(f"[{name}] ROC={mm['roc_auc']:.5f} AP={mm['average_precision']:.5f} dAP={b['observed_delta_ap']:+.5f} p={b['p_delta_le_0']:.3f}")

    # ae_error: raw + MSE + MAE (integrasi Ding)
    evaluate("ae_error", add_err(Xtr, mse_tr, mae_tr), ytr, add_err(Xva, mse_va, mae_va), add_err(Xte, mse_te, mae_te), spw)

    # ae_error_only: hanya MSE + MAE
    evaluate("ae_error_only",
             pd.DataFrame({"ae_mse": mse_tr, "ae_mae": mae_tr}), ytr,
             pd.DataFrame({"ae_mse": mse_va, "ae_mae": mae_va}),
             pd.DataFrame({"ae_mse": mse_te, "ae_mae": mae_te}), spw)

    # smote
    Xb, yb = smote(Xtr.values.astype("float32"), ytr)
    evaluate("smote", pd.DataFrame(Xb, columns=feats), yb, Xva, Xte, 1.0)

    # ae_error_smote (klaim penuh Ding): raw+MSE+MAE lalu SMOTE
    Atr = add_err(Xtr, mse_tr, mae_tr)
    Xb2, yb2 = smote(Atr.values.astype("float32"), ytr)
    evaluate("ae_error_smote", pd.DataFrame(Xb2, columns=list(Atr.columns)), yb2,
             add_err(Xva, mse_va, mae_va), add_err(Xte, mse_te, mae_te), 1.0)

    print("\n========= REPLIKASI FAITHFUL DING — creditcard ULB =========")
    print("(Ding melaporkan AUC ~0.968 dgn SMOTE+AE+LightGBM)")
    print(f"{'skenario':16s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'Recall':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>9s} {'p':>7s}")
    for k, v in results.items():
        b = v.get("bootstrap_vs_baseline", {})
        print(f"{k:16s} {v['roc_auc']:9.5f} {v['average_precision']:9.5f} {v['recall']:8.4f} "
              f"{v['f1']:8.4f} {v['mcc']:8.4f} {b.get('observed_delta_ap',0):+9.5f} {b.get('p_delta_le_0',float('nan')):7.3f}")
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
