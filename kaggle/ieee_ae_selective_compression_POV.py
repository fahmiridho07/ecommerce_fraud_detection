"""IEEE-CIS — USULAN ASLI, POV "kompresi SELEKTIF" (modifikasi minimal & defensible).

Usulan asli: Autoencoder (feature extractor) merekonstruksi/mengompres blok V -> gabung
dengan non-V -> LightGBM. Diagnosis: mengompres SELURUH V menghaluskan V diskriminatif
(baseline 0.8218 -> 0.7690). AE vs PCA tak pernah menang -> V sudah PCA-like.

POV BARU (tetap usulan asli, hanya GRANULARITAS yang berubah):
  Jangan kompres semua V. PARTISI V berdasar importance LightGBM:
    - V importance-TINGGI  -> DIPERTAHANKAN MENTAH (sinyal diskriminatif tak dihaluskan)
    - V importance-RENDAH  -> dikompres AE jadi laten kecil (reduksi dimensi yg redundan)
  Gabung: non-V + V-top mentah + laten(V-redundan) -> LightGBM.
  Ini varian "selective" yang dulu OOM & belum pernah tuntas. Minimal, faithful,
  rasional langsung dari diagnosis. AE tetap feature extractor -> judul/tujuan utuh.

Skenario (stratified 60/20/20, seed 42, PR-AUC utama, threshold MCC, paired bootstrap;
AE/scaler fit di TRAIN saja). TANPA SMOTE -> mengisolasi kontribusi AE-feature (uji bersih
usulan asli sebelum penyeimbangan):
  baseline      : non-V + SEMUA V mentah (info penuh, acuan)
  ae_full       : non-V + laten(SEMUA V)            [usulan asli klasik -> diharapkan turun]
  ae_selective  : non-V + V-top mentah + laten(V-redundan)   [POV BARU]

Sweep TOP_K (berapa V dipertahankan mentah) untuk kurva: makin banyak dipertahankan,
makin dekat baseline; titik manis = reduksi dimensi tanpa kehilangan sinyal.

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_selective_compression_results.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from sklearn.neighbors import NearestNeighbors

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 16
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 2048
N_BOOTSTRAP = 2000
TOP_K_GRID = [64, 128, 192]    # berapa V importance-tinggi dipertahankan mentah
SMOTE_RATE = 0.15              # untuk palang ke-2: ae_selective_smote vs smote
K_NEIGHBORS = 5
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


def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def default_params(spw):
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


def build_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="linear")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def train_ae_latent(Xtr_sub, Xva_sub, Xte_sub):
    ae, enc = build_ae(Xtr_sub.shape[1], LATENT_DIM)
    ae.fit(Xtr_sub, Xtr_sub, validation_data=(Xva_sub, Xva_sub), epochs=AE_EPOCHS,
           batch_size=AE_BATCH, shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])
    return (enc.predict(Xtr_sub, batch_size=8192, verbose=0).astype("float32"),
            enc.predict(Xva_sub, batch_size=8192, verbose=0).astype("float32"),
            enc.predict(Xte_sub, batch_size=8192, verbose=0).astype("float32"))


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, cols, v_cols = preprocess_numeric(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    vmask = np.array([c in set(v_cols) for c in cols]); nonv = ~vmask
    v_idx = np.where(vmask)[0]
    print(f"train={len(Xtr)} feats={len(cols)} V={len(v_idx)}")

    results, scores = {}, {}

    def evaluate(name, Mtr, Mva, Mte, use_smote=False):
        if use_smote:
            Xs, ys = smote(Mtr, ytr); p = default_params(1.0)
        else:
            Xs, ys = Mtr, ytr; p = default_params(spw)
        m, it = fit_lgbm(p, Xs, ys, Mva, yva)
        vva = m.predict_proba(Mva, num_iteration=it)[:, 1]
        vte = m.predict_proba(Mte, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")
        return Mtr, Mva, Mte

    reps = {}   # simpan representasi tiap skenario untuk versi SMOTE

    # ---- baseline: non-V + semua V mentah ----
    print("\n--- baseline (non-V + semua V mentah) ---")
    reps["baseline"] = evaluate("baseline", Xtr, Xva, Xte)

    # ---- importance V dari baseline (untuk partisi); fit train/valid saja ----
    print("\nMenghitung importance V (LightGBM ringan)...")
    p = default_params(spw); p["n_estimators"] = 400; p["learning_rate"] = 0.05
    m0, _ = fit_lgbm(p, Xtr, ytr, Xva, yva)
    gain = np.asarray(m0.booster_.feature_importance(importance_type="gain"), dtype="float64")
    v_gain = gain[v_idx]
    v_rank = v_idx[np.argsort(-v_gain)]    # indeks kolom V urut importance menurun

    # ---- ae_full: non-V + laten(SEMUA V) (usulan asli klasik) ----
    print("\nTraining AE pada SEMUA V (usulan asli)...")
    lt, lv, le = train_ae_latent(Xtr[:, vmask], Xva[:, vmask], Xte[:, vmask])
    print("\n--- ae_full (non-V + laten semua V) ---")
    reps["ae_full"] = evaluate("ae_full",
             np.hstack([Xtr[:, nonv], lt]).astype("float32"),
             np.hstack([Xva[:, nonv], lv]).astype("float32"),
             np.hstack([Xte[:, nonv], le]).astype("float32"))

    # ---- ae_selective: non-V + V-top mentah + laten(V-redundan), sweep TOP_K ----
    for K in TOP_K_GRID:
        top = v_rank[:K]; rest = v_rank[K:]
        if len(rest) < LATENT_DIM + 2:
            print(f"  K={K}: sisa V terlalu sedikit, lewati."); continue
        print(f"\nTraining AE pada {len(rest)} V-redundan (pertahankan {K} V-top mentah)...")
        rt, rv, re = train_ae_latent(Xtr[:, rest], Xva[:, rest], Xte[:, rest])
        name = f"ae_selective_top{K}"
        print(f"\n--- {name} (non-V + {K} V-top mentah + laten {len(rest)} V-redundan) ---")
        reps[name] = evaluate(name,
                 np.hstack([Xtr[:, nonv], Xtr[:, top], rt]).astype("float32"),
                 np.hstack([Xva[:, nonv], Xva[:, top], rv]).astype("float32"),
                 np.hstack([Xte[:, nonv], Xte[:, top], re]).astype("float32"))
        tf.keras.backend.clear_session()

    # ===== PALANG KE-2: dengan SMOTE -> ae_selective_smote harus > smote =====
    print("\n##### Versi SMOTE (uji palang ke-2: AE harus mengalahkan SMOTE) #####")
    for name, (Mtr, Mva, Mte) in list(reps.items()):
        sname = f"{name}_smote"
        print(f"\n--- {sname} ---")
        evaluate(sname, Mtr, Mva, Mte, use_smote=True)

    comp = {}
    for name in results:
        if name == "baseline":
            continue
        comp[f"{name}_vs_baseline"] = bootstrap_delta(yte, scores["baseline"], scores[name])
    # PALANG KE-2: tiap representasi+SMOTE vs SMOTE-baseline (baseline_smote = kontrol "smote")
    if "baseline_smote" in scores:
        for name in results:
            if name.endswith("_smote") and name != "baseline_smote":
                comp[f"{name}_vs_smote"] = bootstrap_delta(
                    yte, scores["baseline_smote"], scores[name])

    print("\n========== USULAN ASLI — KOMPRESI SELEKTIF (stratified 60/20/20) ==========")
    print(f"{'skenario':22s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k_ in results:
        v = results[k_]; print(f"{k_:22s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k_, b in comp.items():
        print(f"  {k_:28s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")
    best = max(results, key=lambda n: results[n]["average_precision"])
    print(f"\n>>> TERBAIK: {best} (AP={results[best]['average_precision']:.6f}, "
          f"baseline={results['baseline']['average_precision']:.6f})")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "ae_selective_compression_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp, "best": best,
                   "config": {"latent_dim": LATENT_DIM, "top_k_grid": TOP_K_GRID,
                              "n_v": int(len(v_idx))}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
