"""IEEE-CIS AE experiments PART 5 (Kaggle) — dua mekanisme BARU, full data.

Distinct dari part 1-4. Tetap koridor "AE menghasilkan fitur untuk LightGBM":

  blockwise_ae : fitur V dikelompokkan ke beberapa blok berkorelasi (clustering),
                 tiap blok punya AE kecil sendiri -> gabungan laten + error per blok
                 jadi fitur. Mengeksploitasi struktur grup V khas Vesta.
  ae_mlp_stack : encoder AE -> kepala klasifikasi fraud (neural), prediksinya diambil
                 OUT-OF-FOLD (KFold, anti-bocor) sebagai SATU fitur skor untuk LightGBM
                 (stacking: AE menyumbang sinyal terpelajar).

Protokol identik: stratified 60/20/20 seed 42, PR-AUC, threshold MCC di validation,
paired bootstrap 2000 vs baseline.

PAKAI: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_experiment_results_part5.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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
N_V_BLOCKS = 11          # jumlah grup korelasi untuk blockwise_ae
STACK_FOLDS = 4          # KFold untuk OOF stacking
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


def small_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(max(32, dim), activation="relu")(inp)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(max(32, dim), activation="relu")(z)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def ae_mlp_classifier(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(z)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m


def main():
    print("Loading...")
    train, valid, test = split_60_20_20(load_data())
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    Xtr, Xva, Xte, cat_cols, num_cols, v_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} cat={len(cat_cols)} V={len(v_cols)}")

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

    Vf = scaler_for(Xtr, v_cols)
    Vtr, Vva, Vte = Vf(Xtr), Vf(Xva), Vf(Xte)
    nmask = ytr == 0

    # ---- 1) blockwise_ae: kelompokkan V via korelasi, AE per blok ----
    print(f"\n[blockwise_ae] clustering V into {N_V_BLOCKS} blocks by correlation...")
    samp = Vtr[np.random.default_rng(SEED).choice(len(Vtr), size=min(20000, len(Vtr)), replace=False)]
    corr = np.corrcoef(samp.T)
    corr = np.nan_to_num(corr)
    dist = 1.0 - np.abs(corr)
    labels = AgglomerativeClustering(n_clusters=N_V_BLOCKS, metric="precomputed",
                                     linkage="average").fit_predict(dist)
    Xtr2, Xva2, Xte2 = Xtr.copy(), Xva.copy(), Xte.copy()
    for g in range(N_V_BLOCKS):
        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            continue
        ld = max(2, min(6, len(idx) // 3))
        keras.backend.clear_session()
        ae, enc = small_ae(len(idx), ld)
        ae.fit(Vtr[:, idx], Vtr[:, idx], validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048,
               shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
               restore_best_weights=True)], verbose=0)
        for X, V in ((Xtr2, Vtr), (Xva2, Vva), (Xte2, Vte)):
            sub = V[:, idx]
            lat = enc.predict(sub, batch_size=8192, verbose=0)
            for j in range(lat.shape[1]):
                X[f"blk{g}_lat{j}"] = lat[:, j].astype("float32")
            X[f"blk{g}_err"] = np.abs(sub - ae.predict(sub, batch_size=8192, verbose=0)).mean(1).astype("float32")
        print(f"  block {g}: {len(idx)} V-cols -> latent {ld}")
    eval_variant("blockwise_ae", Xtr2, Xva2, Xte2)
    del Xtr2, Xva2, Xte2

    # ---- 2) ae_mlp_stack: skor neural OOF sebagai fitur ----
    print(f"\n[ae_mlp_stack] AE->MLP fraud score, {STACK_FOLDS}-fold OOF...")
    keras.backend.clear_session()
    oof = np.zeros(len(Vtr), dtype="float32")
    va_acc = np.zeros(len(Vva), dtype="float32")
    te_acc = np.zeros(len(Vte), dtype="float32")
    cw = {0: 1.0, 1: spw}
    skf = StratifiedKFold(n_splits=STACK_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tri, vli) in enumerate(skf.split(Vtr, ytr)):
        keras.backend.clear_session()
        clf = ae_mlp_classifier(Vtr.shape[1], LATENT_DIM)
        clf.fit(Vtr[tri], ytr[tri].astype("float32"), validation_data=(Vtr[vli], ytr[vli].astype("float32")),
                epochs=AE_EPOCHS, batch_size=2048, shuffle=True, class_weight=cw,
                callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
                verbose=0)
        oof[vli] = clf.predict(Vtr[vli], batch_size=8192, verbose=0).ravel()
        va_acc += clf.predict(Vva, batch_size=8192, verbose=0).ravel() / STACK_FOLDS
        te_acc += clf.predict(Vte, batch_size=8192, verbose=0).ravel() / STACK_FOLDS
        print(f"  fold {fold} done")
    Xtr3, Xva3, Xte3 = Xtr.copy(), Xva.copy(), Xte.copy()
    Xtr3["ae_mlp_score"] = oof
    Xva3["ae_mlp_score"] = va_acc.astype("float32")
    Xte3["ae_mlp_score"] = te_acc.astype("float32")
    eval_variant("ae_mlp_stack", Xtr3, Xva3, Xte3)

    print("\n================ RINGKASAN PART 5 (full data, stratified) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC':>8s} {'F1':>8s} {'MCC':>8s} {'dAP':>10s} {'p':>7s}")
    for k, v in results.items():
        d = v.get("bootstrap_vs_baseline", {})
        print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:8.5f} {v['f1']:8.5f} {v['mcc']:8.5f} "
              f"{d.get('observed_delta_ap', 0):+10.6f} {d.get('p_delta_le_0', float('nan')):7.3f}")
    with open("/kaggle/working/ae_experiment_results_part5.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /kaggle/working/ae_experiment_results_part5.json")


if __name__ == "__main__":
    main()
