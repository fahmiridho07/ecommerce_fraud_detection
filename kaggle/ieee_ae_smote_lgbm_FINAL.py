"""IEEE-CIS — LEGACY FEATURE-EXTRACTOR SCRIPT, NOT THE ACTIVE THESIS FINAL.

Active thesis Kaggle script: `ieee_final_oversampling_kaggle.py`.

This file is kept for traceability of the older feature-extractor + SMOTE
framing. Do not use it as the Bab 4 headline unless the thesis scope is
reopened deliberately.

Original note:
IEEE-CIS — EKSPERIMEN FINAL (ruang lingkup terkunci, mengikuti Ding & Du).

Judul tetap: "Deteksi Penipuan ... Integrasi Autoencoder dan LightGBM dengan Bayesian Optimization".
Pipeline = Autoencoder (feature extractor) + SMOTE + LightGBM + Bayesian Optimization (Optuna/TPE).
Peran AE TIDAK berubah (tetap representasi fitur) -> tujuan & judul utuh; SMOTE mengikuti Ding/Du.

Skenario (default & tuned) — tabel rapi 6 baris:
  baseline      : LightGBM tanpa penyeimbangan
  smote         : LightGBM + SMOTE (penyeimbangan kelas, mengikuti Ding/Du)
  ae_smote      : Autoencoder (fitur V -> laten) + non-V, lalu SMOTE -> LightGBM  [MODEL USULAN]

Perbandingan:
  ae_smote vs baseline  -> efektivitas model usulan
  ae_smote vs smote     -> kontribusi autoencoder di atas SMOTE (transparansi atribusi)
  smote vs baseline     -> efek penyeimbangan SMOTE

Protokol: data berstrata 60/20/20, seed 42, metrik utama PR-AUC, threshold dipilih di
validasi (MCC), uji signifikansi paired bootstrap. SMOTE & AE hanya di-fit pada data latih.

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ae_smote_lgbm_results.json
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
LATENT_DIM = 64           # autoencoder undercomplete (V dikompres ke ruang laten)
AE_EPOCHS = 40
TARGET_FRAUD_RATE = 0.5   # SMOTE menyeimbangkan kelas (mengikuti Ding/Du)
K_NEIGHBORS = 5
N_TRIALS = 12
N_BOOTSTRAP = 2000
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
    """Representasi numerik penuh agar SMOTE dapat diterapkan: frequency-encoding utk
    kategorikal, imputasi median utk numerik, lalu standarisasi. Fit hanya pada train."""
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
        out[nm] = pd.DataFrame(cols)  # bangun sekaligus (tanpa fragmentasi)
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


def tune_lgbm(Xtr, ytr, Xva, yva, spw):
    def objective(trial):
        p = default_params(spw)
        p.update(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 16, 256),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        m, it = fit_lgbm(p, Xtr, ytr, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    p = default_params(spw); p.update(study.best_params)
    return fit_lgbm(p, Xtr, ytr, Xva, yva)


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


def smote(X, y, target_rate=TARGET_FRAUD_RATE, k=K_NEIGHBORS, seed=SEED):
    """SMOTE (Chawla et al., 2002) pada ruang fitur numerik. Hanya untuk data latih."""
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


def build_ae(dim, latent):
    """Autoencoder undercomplete (fitur V -> ruang laten -> rekonstruksi)."""
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, cols, v_cols = preprocess_numeric(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    vmask = np.array([c in set(v_cols) for c in cols])
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} feats={len(cols)} V={int(vmask.sum())}")

    results, scores = {}, {}

    def evaluate(name, Xtr2, ytr2, Xte2, spw2, tuned):
        m, it = (tune_lgbm(Xtr2, ytr2, Xva_use[name], yva, spw2) if tuned
                 else fit_lgbm(default_params(spw2), Xtr2, ytr2, Xva_use[name], yva))
        vva = m.predict_proba(Xva_use[name], num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    # validasi memakai representasi yang sesuai tiap skenario (baseline/smote pakai fitur asli;
    # ae_smote pakai fitur ber-AE). Disiapkan di bawah.
    Xva_use = {}

    # ----- baseline & smote: fitur numerik apa adanya -----
    Xva_use["baseline"] = Xva; Xva_use["baseline_tuned"] = Xva
    Xva_use["smote"] = Xva; Xva_use["smote_tuned"] = Xva
    print("\n--- baseline ---")
    evaluate("baseline", Xtr, ytr, Xte, spw, False)
    evaluate("baseline_tuned", Xtr, ytr, Xte, spw, True)
    print("\n--- smote ---")
    Xs, ys = smote(Xtr, ytr)
    evaluate("smote", Xs, ys, Xte, 1.0, False)
    evaluate("smote_tuned", Xs, ys, Xte, 1.0, True)

    # ----- ae_smote (MODEL USULAN): AE(fitur V -> laten) + non-V, lalu SMOTE -----
    print("\nTraining autoencoder on V block (feature extractor)...")
    ae, enc = build_ae(int(vmask.sum()), LATENT_DIM)
    ae.fit(Xtr[:, vmask], Xtr[:, vmask], validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048,
           shuffle=True, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
           restore_best_weights=True)], verbose=2)
    def to_ae_repr(X):
        lat = enc.predict(X[:, vmask], batch_size=8192, verbose=0)
        return np.hstack([X[:, ~vmask], lat]).astype("float32")  # non-V + laten
    Atr, Ava, Ate = to_ae_repr(Xtr), to_ae_repr(Xva), to_ae_repr(Xte)
    Xva_use["ae_smote"] = Ava; Xva_use["ae_smote_tuned"] = Ava
    print("\n--- ae_smote (MODEL USULAN) ---")
    As, ays = smote(Atr, ytr)
    evaluate("ae_smote", As, ays, Ate, 1.0, False)
    evaluate("ae_smote_tuned", As, ays, Ate, 1.0, True)

    comp = {
        "ae_smote_vs_baseline": bootstrap_delta(yte, scores["baseline"], scores["ae_smote"]),
        "ae_smote_vs_smote": bootstrap_delta(yte, scores["smote"], scores["ae_smote"]),
        "smote_vs_baseline": bootstrap_delta(yte, scores["baseline"], scores["smote"]),
        "ae_smote_vs_baseline_tuned": bootstrap_delta(yte, scores["baseline_tuned"], scores["ae_smote_tuned"]),
        "ae_smote_vs_smote_tuned": bootstrap_delta(yte, scores["smote_tuned"], scores["ae_smote_tuned"]),
    }

    print("\n================ HASIL FINAL — AE + SMOTE + LightGBM (data berstrata 60/20/20) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k in ("baseline","baseline_tuned","smote","smote_tuned","ae_smote","ae_smote_tuned"):
        v = results[k]; print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k, b in comp.items():
        print(f"  {k:28s} delta={b['observed_delta_ap']:+.6f} ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    with open("/kaggle/working/ae_smote_lgbm_results.json", "w") as f:
        json.dump({"results": results, "comparisons": comp,
                   "config": {"latent_dim": LATENT_DIM, "target_fraud_rate": TARGET_FRAUD_RATE, "n_trials": N_TRIALS}}, f, indent=2)
    print("\nSaved: /kaggle/working/ae_smote_lgbm_results.json")


if __name__ == "__main__":
    main()
