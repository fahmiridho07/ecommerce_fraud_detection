"""IEEE-CIS — PERBANDINGAN 4-CARA SESUAI PROPOSAL (tanpa SMOTE).

Setia desain proposal & referensi (Du, Ding): hanya autoencoder + LightGBM +
Bayesian Optimization (Optuna/TPE). Empat skenario:

  baseline         : LightGBM (parameter default)
  baseline_tuned   : LightGBM + Optuna (TPE)
  integration      : AE menggantikan fitur V dengan representasi laten + gabung non-V -> LightGBM (default)
  integration_tuned: integrasi di atas + Optuna (TPE)

Definisi integrasi mengikuti Bab 3: fitur V (V1..V339) diganti latent undercomplete AE,
digabung dengan fitur non-V, lalu diklasifikasi LightGBM.

Protokol: stratified 60/20/20 seed 42, metrik utama PR-AUC, threshold MCC di validation,
paired bootstrap 2000. Perbandingan kunci: integration_tuned vs baseline_tuned (apple-to-apple).

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/proposal_4way_results.json
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
import optuna
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 64        # undercomplete (339 -> 64)
AE_EPOCHS = 40
N_TRIALS = 15          # Optuna trials per pipeline
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
    return Xtr, Xva, Xte, cat_cols, v_cols


def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def default_params(spw):
    return dict(objective="binary", boosting_type="gbdt", n_estimators=N_ESTIMATORS,
                learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                random_state=SEED, metric="None", verbosity=-1)


def fit_lgbm(params, Xtr, ytr, Xva, yva, cat_cols):
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=ap_eval, categorical_feature=cat_cols,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True)])
    return m, int(m.best_iteration_ or N_ESTIMATORS)


def tune_lgbm(Xtr, ytr, Xva, yva, cat_cols, spw):
    def objective(trial):
        p = default_params(spw)
        p.update(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 16, 256),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
        m, it = fit_lgbm(p, Xtr, ytr, Xva, yva, cat_cols)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    p = default_params(spw); p.update(study.best_params)
    return fit_lgbm(p, Xtr, ytr, Xva, yva, cat_cols)


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


def build_ae(dim, latent):
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
    print("Loading...")
    train, valid, test = split_60_20_20(load_data())
    ytr, yva, yte = (train[TARGET].to_numpy(), valid[TARGET].to_numpy(), test[TARGET].to_numpy())
    Xtr, Xva, Xte, cat_cols, v_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} cat={len(cat_cols)} V={len(v_cols)}")

    results, scores = {}, {}

    def run(name, Xtr2, Xva2, Xte2, tuned):
        if tuned:
            m, it = tune_lgbm(Xtr2, ytr, Xva2, yva, cat_cols, spw)
        else:
            m, it = fit_lgbm(default_params(spw), Xtr2, ytr, Xva2, yva, cat_cols)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")
        return vte

    # ---- Baseline (V utuh) ----
    print("\n[baseline] default ..."); run("baseline", Xtr, Xva, Xte, tuned=False)
    print("[baseline_tuned] Optuna ..."); run("baseline_tuned", Xtr, Xva, Xte, tuned=True)

    # ---- Integrasi: ganti V dgn latent AE (undercomplete) + gabung non-V ----
    print("\nTraining undercomplete AE on V block...")
    imp = SimpleImputer(strategy="median").fit(Xtr[v_cols])
    sc = StandardScaler().fit(imp.transform(Xtr[v_cols]))
    def Vs(X): return np.clip(sc.transform(imp.transform(X[v_cols])).astype("float32"), -10, 10)
    Vtr, Vva, Vte = Vs(Xtr), Vs(Xva), Vs(Xte)
    ae, enc = build_ae(Vtr.shape[1], LATENT_DIM)
    ae.fit(Vtr, Vtr, validation_split=0.1, epochs=AE_EPOCHS, batch_size=2048, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=2)
    nonV = [c for c in Xtr.columns if c not in v_cols]
    lat_cols = [f"ae_lat_{j}" for j in range(LATENT_DIM)]
    def integ(Xdf, V):
        L = pd.DataFrame(enc.predict(V, batch_size=8192, verbose=0).astype("float32"), columns=lat_cols, index=Xdf.index)
        return pd.concat([Xdf[nonV].reset_index(drop=True), L.reset_index(drop=True)], axis=1)
    Itr, Iva, Ite = integ(Xtr, Vtr), integ(Xva, Vva), integ(Xte, Vte)
    print(f"integration features: {Itr.shape[1]} (non-V {len(nonV)} + latent {LATENT_DIM})")

    print("\n[integration] default ..."); run("integration", Itr, Iva, Ite, tuned=False)
    print("[integration_tuned] Optuna ..."); run("integration_tuned", Itr, Iva, Ite, tuned=True)

    # ---- Perbandingan kunci (paired bootstrap) ----
    comp = {
        "tuning_effect_baseline": bootstrap_delta(yte, scores["baseline"], scores["baseline_tuned"]),
        "ae_effect_default": bootstrap_delta(yte, scores["baseline"], scores["integration"]),
        "ae_effect_tuned": bootstrap_delta(yte, scores["baseline_tuned"], scores["integration_tuned"]),
        "tuning_effect_integration": bootstrap_delta(yte, scores["integration"], scores["integration_tuned"]),
    }

    print("\n================ PERBANDINGAN 4-CARA (full data, stratified) ================")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k in ("baseline", "baseline_tuned", "integration", "integration_tuned"):
        v = results[k]
        print(f"{k:18s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan kunci (paired bootstrap on AP):")
    for k, b in comp.items():
        print(f"  {k:26s} delta={b['observed_delta_ap']:+.6f} ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out = {"results": results, "comparisons": comp, "latent_dim": LATENT_DIM, "n_trials": N_TRIALS}
    with open("/kaggle/working/proposal_4way_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: /kaggle/working/proposal_4way_results.json")
    print("Baca: ae_effect_tuned = kontribusi integrasi AE pada kondisi sama-sama tuned (apple-to-apple).")


if __name__ == "__main__":
    main()
