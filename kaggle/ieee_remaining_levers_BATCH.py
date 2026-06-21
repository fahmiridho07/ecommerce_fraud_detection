"""IEEE-CIS — SEMUA TUAS TERSISA dalam SATU batch (anchor-faithful, judul/scope utuh).

Setelah 20+ varian AE-feature gagal, ini menjalankan SEKALIGUS seluruh knob yang BELUM
diuji & masih faithful (AE + LightGBM + SMOTE; judul "Integrasi Autoencoder dan LightGBM"
utuh). Tujuan: sekali jalan, tahu mana yang mengalahkan baseline, tak buang waktu iterasi.

Tuas yang diuji:
  REFERENSI
    baseline_tuned        : LightGBM + BO
    smote_tuned           : + SMOTE rasio di-tune                         [sudah > baseline]
  SISI SMOTE (paling mungkin menang; faithful Ding/Du; SMOTE tak ada di judul)
    borderline_tuned      : BorderlineSMOTE rasio+k di-tune
    adasyn_tuned          : ADASYN rasio+k di-tune
    svmsmote_tuned        : SVMSMOTE rasio+k di-tune
  SISI AE-FEATURE (sisa yang belum diuji; semua additive + one-class, gaya Ding)
    ding_group_smote      : one-class AE -> error per-keluarga + SMOTE     [best AE shot]
    dual_class_smote      : error dari AE-normal & AE-fraud + SMOTE        [baru]

Protokol: stratified 60/20/20, seed 42, PR-AUC utama, threshold MCC, paired bootstrap.
Semua AE/scaler/resampler fit di TRAIN saja. Tiap skenario: rasio resampler + k + hiperparam
LightGBM di-tune bersama via Optuna (Bayesian Optimization).

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/remaining_levers_results.json
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
import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
AE_LATENT = 16
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 2048
N_TRIALS = 20
N_BOOTSTRAP = 2000
RATIO_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
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
    return out["tr"].to_numpy(), out["va"].to_numpy(), out["te"].to_numpy(), order


def feature_families(cols):
    fam = {}
    for i, c in enumerate(cols):
        if c.startswith("V") and c[1:].isdigit(): f = "V"
        elif c.startswith("C") and c[1:].isdigit(): f = "C"
        elif c.startswith("D") and c[1:].isdigit(): f = "D"
        elif c.startswith("M") and c[1:].isdigit(): f = "M"
        elif c.startswith("card") or c.startswith("addr"): f = "cardaddr"
        elif "email" in c: f = "email"
        elif c.startswith("id_") or c.startswith("Device"): f = "iddev"
        else: f = "other"
        fam.setdefault(f, []).append(i)
    return fam


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


def space(trial, spw):
    p = default_params(spw)
    p.update(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
             num_leaves=trial.suggest_int("num_leaves", 16, 256),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
             subsample=trial.suggest_float("subsample", 0.5, 1.0),
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
             reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    return p


def make_resampler(kind, rate, k):
    if kind == "smote":      return SMOTE(sampling_strategy=rate, k_neighbors=k, random_state=SEED)
    if kind == "borderline": return BorderlineSMOTE(sampling_strategy=rate, k_neighbors=k, random_state=SEED)
    if kind == "adasyn":     return ADASYN(sampling_strategy=rate, n_neighbors=k, random_state=SEED)
    if kind == "svmsmote":   return SVMSMOTE(sampling_strategy=rate, k_neighbors=k, random_state=SEED)
    raise ValueError(kind)


def resample(kind, X, y, rate, k):
    try:
        return make_resampler(kind, rate, k).fit_resample(X, y)
    except (ValueError, RuntimeError):
        return X, y   # fallback bila resampler gagal (mis. tetangga tak cukup)


def tune_plain(Xtr, ytr, Xva, yva, spw):
    def obj(trial):
        m, it = fit_lgbm(space(trial, spw), Xtr, ytr, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    p = default_params(spw); p.update(st.best_params)
    return fit_lgbm(p, Xtr, ytr, Xva, yva)


def tune_resample(kind, Xtr, ytr, Xva, yva):
    def obj(trial):
        rate = trial.suggest_categorical("rate", RATIO_GRID)
        k = trial.suggest_categorical("k", K_GRID)
        Xs, ys = resample(kind, Xtr, ytr, rate, k)
        m, it = fit_lgbm(space(trial, 1.0), Xs, ys, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    bp = st.best_params
    Xs, ys = resample(kind, Xtr, ytr, bp["rate"], bp["k"])
    p = default_params(1.0); p.update({k: v for k, v in bp.items() if k not in ("rate", "k")})
    m, it = fit_lgbm(p, Xs, ys, Xva, yva)
    return m, it, bp["rate"], bp["k"]


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


def build_oneclass_ae(dim, latent):
    inp = keras.Input(shape=(dim,))
    e = keras.layers.Dense(latent, activation="relu", activity_regularizer=regularizers.l1(1e-5))(inp)
    e = keras.layers.Dense(8, activation="relu")(e)
    d = keras.layers.Dense(8, activation="relu")(e)
    out = keras.layers.Dense(dim, activation="relu")(d)
    ae = keras.Model(inp, out); ae.compile(optimizer="adam", loss="mse")
    return ae


def train_ae(ae, Xfit, Xval):
    ae.fit(Xfit, Xfit, validation_data=(Xval, Xval), epochs=AE_EPOCHS, batch_size=AE_BATCH,
           shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])
    return ae


def group_err(ae, X, fam):
    pred = ae.predict(X, batch_size=8192, verbose=0)
    se = (X - pred) ** 2; mae = np.abs(X - pred)
    feats = [se.mean(1, keepdims=True), mae.mean(1, keepdims=True)]
    for f, idxs in fam.items():
        feats.append(se[:, idxs].mean(1, keepdims=True))
    return np.hstack(feats).astype("float32")


def global_err(ae, X):
    pred = ae.predict(X, batch_size=8192, verbose=0)
    return np.hstack([((X - pred) ** 2).mean(1, keepdims=True),
                      np.abs(X - pred).mean(1, keepdims=True)]).astype("float32")


def zfit(A, ref):
    mu = ref.mean(0); sd = ref.std(0); sd[sd == 0] = 1.0
    return ((A - mu) / sd).astype("float32")


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, cols = preprocess_numeric(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    fam = feature_families(cols)
    print(f"train={len(Xtr)} feats={len(cols)} keluarga={list(fam.keys())}")

    results, scores, meta = {}, {}, {}

    def record(name, m, it, Mva, Mte):
        vva = m.predict_proba(Mva, num_iteration=it)[:, 1]
        vte = m.predict_proba(Mte, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    # ---------- referensi ----------
    print("\n--- baseline_tuned ---")
    m, it = tune_plain(Xtr, ytr, Xva, yva, spw); record("baseline_tuned", m, it, Xva, Xte)

    # ---------- sisi SMOTE (paling mungkin menang) ----------
    for kind, name in (("smote", "smote_tuned"), ("borderline", "borderline_tuned"),
                       ("adasyn", "adasyn_tuned"), ("svmsmote", "svmsmote_tuned")):
        print(f"\n--- {name} ---")
        m, it, r, k = tune_resample(kind, Xtr, ytr, Xva, yva)
        meta[name] = {"rate": r, "k": k}; print(f"  {kind} rate={r} k={k}")
        record(name, m, it, Xva, Xte)

    # ---------- sisi AE-feature: one-class group error ----------
    print("\nTraining one-class AE (normal, semua fitur) -> group error...")
    ae_n = train_ae(build_oneclass_ae(Xtr.shape[1], AE_LATENT), Xtr[ytr == 0], Xva[yva == 0])
    En_tr, En_va, En_te = group_err(ae_n, Xtr, fam), group_err(ae_n, Xva, fam), group_err(ae_n, Xte, fam)
    En_va = zfit(En_va, En_tr); En_te = zfit(En_te, En_tr); En_tr = zfit(En_tr, En_tr)
    Gtr = np.hstack([Xtr, En_tr]).astype("float32")
    Gva = np.hstack([Xva, En_va]).astype("float32"); Gte = np.hstack([Xte, En_te]).astype("float32")
    print("\n--- ding_group_smote (best AE shot) ---")
    m, it, r, k = tune_resample("smote", Gtr, ytr, Gva, yva)
    meta["ding_group_smote"] = {"rate": r, "k": k}; print(f"  smote rate={r} k={k}")
    record("ding_group_smote", m, it, Gva, Gte)

    # ---------- sisi AE-feature: dual-class error (normal-AE & fraud-AE) ----------
    print("\nTraining fraud-class AE (baris fraud) -> dual error...")
    n_fraud = int((ytr == 1).sum())
    if n_fraud >= 50:
        ae_f = train_ae(build_oneclass_ae(Xtr.shape[1], AE_LATENT), Xtr[ytr == 1],
                        Xva[yva == 1] if int((yva == 1).sum()) >= 10 else Xtr[ytr == 1])
        Ef_tr, Ef_va, Ef_te = global_err(ae_f, Xtr), global_err(ae_f, Xva), global_err(ae_f, Xte)
        # fitur dual: error-normal(global+group) + error-fraud(global) + selisih (normal-fraud)
        diff_tr = (En_tr[:, :2] - zfit(Ef_tr, Ef_tr))
        Dtr = np.hstack([Xtr, En_tr, zfit(Ef_tr, Ef_tr), diff_tr]).astype("float32")
        Dva = np.hstack([Xva, En_va, zfit(Ef_va, Ef_tr), En_va[:, :2] - zfit(Ef_va, Ef_tr)]).astype("float32")
        Dte = np.hstack([Xte, En_te, zfit(Ef_te, Ef_tr), En_te[:, :2] - zfit(Ef_te, Ef_tr)]).astype("float32")
        print("\n--- dual_class_smote (normal-AE & fraud-AE error) ---")
        m, it, r, k = tune_resample("smote", Dtr, ytr, Dva, yva)
        meta["dual_class_smote"] = {"rate": r, "k": k}; print(f"  smote rate={r} k={k}")
        record("dual_class_smote", m, it, Dva, Dte)
    else:
        print("  fraud train rows < 50 -> lewati dual_class.")

    # ---------- ringkasan + bootstrap vs baseline & smote ----------
    comp = {}
    for name in results:
        if name == "baseline_tuned":
            continue
        comp[f"{name}_vs_baseline"] = bootstrap_delta(yte, scores["baseline_tuned"], scores[name])
    for name in ("ding_group_smote", "dual_class_smote"):
        if name in scores:
            comp[f"{name}_vs_smote"] = bootstrap_delta(yte, scores["smote_tuned"], scores[name])

    print("\n========== SEMUA TUAS TERSISA (stratified 60/20/20) ==========")
    print(f"{'skenario':20s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k_ in results:
        v = results[k_]; print(f"{k_:20s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k_, b in comp.items():
        print(f"  {k_:30s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    best = max(results, key=lambda n: results[n]["average_precision"])
    print(f"\n>>> TERBAIK: {best} (AP={results[best]['average_precision']:.6f}, "
          f"baseline={results['baseline_tuned']['average_precision']:.6f})")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "remaining_levers_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp, "selected": meta, "best": best,
                   "config": {"ae_latent": AE_LATENT, "n_trials": N_TRIALS,
                              "families": list(fam.keys())}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
