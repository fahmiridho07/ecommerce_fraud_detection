"""IEEE-CIS — AE+SMOTE+LightGBM+BO, MODIFIKASI MINIMAL (tweak, metode tetap).

Tujuan: menutup "celah" antara hasil paper (bagus) dan implementasi awal (jelek) TANPA
mengubah metode secara signifikan. Dari hasil FINAL: smote_tuned (0.8610) SUDAH mengalahkan
baseline_tuned (0.8430); hanya langkah AE-mengganti-V (substitutif) yang menariknya turun
(ae_smote_tuned 0.8086). Tweak di sini mengembalikan kemenangan itu sambil AE tetap dipakai.

TWEAK (tetap = Autoencoder feature extractor + SMOTE + LightGBM + Bayesian Optimization):
  1. AE ADDITIVE, bukan substitutif: SEMUA fitur asli DIPERTAHANKAN, lalu DITAMBAH laten(V)
     + error rekonstruksi V (gaya Ding/Du yang menambah sinyal, tidak membuang info).
  2. Rasio SMOTE (target_fraud_rate) DI-TUNE (bukan dipaku 0.5) — optimum PR-AUC biasanya rendah.
  3. Laten AE: aktivasi linear (bukan relu) + dimensi kecil, agar sinyal tak terpotong.

Skenario (stratified 60/20/20, seed 42, PR-AUC utama, threshold MCC, paired bootstrap):
  baseline_tuned       : seluruh fitur, LightGBM+BO (acuan)
  smote_tuned          : + SMOTE rasio di-tune, LightGBM+BO            [sudah > baseline]
  ae_add_smote_tuned   : seluruh fitur + laten(V) + error V, SMOTE rasio di-tune  [USULAN-TWEAK]

Target: ae_add_smote_tuned >= baseline_tuned (idealnya ~ smote_tuned), membuktikan model
usulan AE+SMOTE mengungguli baseline setelah integrasi diperbaiki.

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_smote_additive_tweak_results.json
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
import optuna
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 8             # TWEAK: SANGAT kecil + linear. Bukti concat_iw: laten 64-dim
                           # mengencerkan -0.027; pengenceran ~ jumlah dim, jadi minimalkan.
                           # Sinyal additive utama = error rekonstruksi (MSE/MAE, gaya Ding).
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 2048
N_TRIALS = 20
N_BOOTSTRAP = 2000
SMOTE_RATIO_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]   # TWEAK: di-tune, bukan 0.5
K_GRID = [3, 5, 7, 10]
TARGET, ID = "isFraud", "TransactionID"

optuna.logging.set_verbosity(optuna.logging.WARNING)
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


def fit_lgbm(params, Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=ap_eval,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True)])
    return m, int(m.best_iteration_ or N_ESTIMATORS)


def smote(X, y, target_rate, k, seed=SEED):
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]; n_maj = int((y == 0).sum())
    n_target = int(round(n_maj * target_rate / (1 - target_rate)))
    n_need = n_target - len(Xmin)
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


def lgbm_search_space(trial, spw):
    p = default_params(spw)
    p.update(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
             num_leaves=trial.suggest_int("num_leaves", 16, 256),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    return p


def tune_plain(Xtr, ytr, Xva, yva, spw):
    """BO LightGBM tanpa SMOTE (baseline_tuned)."""
    def objective(trial):
        p = lgbm_search_space(trial, spw)
        m, it = fit_lgbm(p, Xtr, ytr, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(objective, n_trials=N_TRIALS)
    p = default_params(spw); p.update(st.best_params)
    return fit_lgbm(p, Xtr, ytr, Xva, yva)


def tune_with_smote(Xtr, ytr, Xva, yva):
    """BO bersama: rasio SMOTE + k + hiperparameter LightGBM (TWEAK rasio di-tune)."""
    def objective(trial):
        rate = trial.suggest_categorical("smote_rate", SMOTE_RATIO_GRID)
        k = trial.suggest_categorical("smote_k", K_GRID)
        Xs, ys = smote(Xtr, ytr, rate, k)
        p = lgbm_search_space(trial, 1.0)
        m, it = fit_lgbm(p, Xs, ys, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(objective, n_trials=N_TRIALS)
    bp = st.best_params
    Xs, ys = smote(Xtr, ytr, bp["smote_rate"], bp["smote_k"])
    p = default_params(1.0)
    p.update({k: v for k, v in bp.items() if k not in ("smote_rate", "smote_k")})
    m, it = fit_lgbm(p, Xs, ys, Xva, yva)
    return m, it, bp["smote_rate"], bp["smote_k"]


def build_ae(dim, latent):
    """AE undercomplete; TWEAK: laten LINEAR (bukan relu) agar sinyal tak terpotong."""
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="linear")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def metrics(yte, pte, thr):
    pred = (pte >= thr).astype(int)
    return dict(average_precision=float(average_precision_score(yte, pte)),
                roc_auc=float(roc_auc_score(yte, pte)),
                f1=float(f1_score(yte, pred)), mcc=float(matthews_corrcoef(yte, pred)))


def pick_threshold(yva, pva):
    best_t, best = 0.5, -1
    for t in np.arange(0.01, 1.0, 0.01):
        v = matthews_corrcoef(yva, (pva >= t).astype(int))
        if v > best: best, best_t = v, t
    return float(best_t)


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
    vmask = np.array([c in set(v_cols) for c in cols])
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} feats={len(cols)} V={int(vmask.sum())}")

    results, scores, meta = {}, {}, {}

    def record(name, m, it, Xva2, Xte2):
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    # ---- baseline_tuned ----
    print("\n--- baseline_tuned (BO) ---")
    m, it = tune_plain(Xtr, ytr, Xva, yva, spw)
    record("baseline_tuned", m, it, Xva, Xte)

    # ---- smote_tuned (rasio di-tune) ----
    print("\n--- smote_tuned (rasio SMOTE + BO) ---")
    m, it, rate, k = tune_with_smote(Xtr, ytr, Xva, yva)
    meta["smote_tuned"] = {"smote_rate": rate, "smote_k": k}
    print(f"  rasio SMOTE terpilih={rate} k={k}")
    record("smote_tuned", m, it, Xva, Xte)

    # ---- ae_add_smote_tuned: ADDITIVE (semua fitur + laten V + error V) + SMOTE tuned ----
    print("\nTraining AE pada blok V (additive feature extractor)...")
    ae, enc = build_ae(int(vmask.sum()), LATENT_DIM)
    ae.fit(Xtr[:, vmask], Xtr[:, vmask], validation_split=0.1, epochs=AE_EPOCHS,
           batch_size=AE_BATCH, shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])

    def add_repr(X):
        Xv = X[:, vmask]
        lat = enc.predict(Xv, batch_size=8192, verbose=0)
        rec = ae.predict(Xv, batch_size=8192, verbose=0)
        mse = np.mean((Xv - rec) ** 2, axis=1, keepdims=True)
        mae = np.mean(np.abs(Xv - rec), axis=1, keepdims=True)
        return np.hstack([X, lat, mse, mae]).astype("float32")   # SEMUA fitur + laten + error

    Atr, Ava, Ate = add_repr(Xtr), add_repr(Xva), add_repr(Xte)
    print(f"  fitur additive: {Xtr.shape[1]} -> {Atr.shape[1]} (+{Atr.shape[1]-Xtr.shape[1]})")
    print("\n--- ae_add_smote_tuned [USULAN-TWEAK] ---")
    m, it, rate, k = tune_with_smote(Atr, ytr, Ava, yva)
    meta["ae_add_smote_tuned"] = {"smote_rate": rate, "smote_k": k}
    print(f"  rasio SMOTE terpilih={rate} k={k}")
    record("ae_add_smote_tuned", m, it, Ava, Ate)

    comp = {
        "smote_vs_baseline":        bootstrap_delta(yte, scores["baseline_tuned"], scores["smote_tuned"]),
        "ae_add_vs_baseline":       bootstrap_delta(yte, scores["baseline_tuned"], scores["ae_add_smote_tuned"]),
        "ae_add_vs_smote":          bootstrap_delta(yte, scores["smote_tuned"], scores["ae_add_smote_tuned"]),
    }

    print("\n========== AE+SMOTE ADDITIVE TWEAK (data berstrata 60/20/20) ==========")
    print(f"{'skenario':22s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k_ in ("baseline_tuned", "smote_tuned", "ae_add_smote_tuned"):
        v = results[k_]; print(f"{k_:22s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k_, b in comp.items():
        print(f"  {k_:24s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "ae_smote_additive_tweak_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp, "selected_smote": meta,
                   "config": {"latent_dim": LATENT_DIM, "latent_activation": "linear",
                              "integration": "additive (all features + V latent + V recon error)",
                              "smote_ratio_grid": SMOTE_RATIO_GRID, "n_trials": N_TRIALS}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
