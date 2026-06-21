"""IEEE-CIS — DING faithful, modifikasi terdalam (BEST SHOT, judul/scope utuh).

Telaah terdalam
---------------
Semua kegagalan AE-feature berakar satu hal: AE bekerja pada ruang input yang SAMA dengan
LightGBM -> redundan; dan menambah banyak dim laten malah mengencerkan. SATU-satunya yang
AE beri & GBDT mustahil hitung sendiri = SKOR NORMALITAS NONLINIER (jarak dari manifold
normal). Itu persis ide Ding (one-class AE -> error rekonstruksi). Kegagalan one_class
sebelumnya: error hanya di blok V & sbg 1 skalar global -> terlalu encer.

Modifikasi berarti (tetap = Autoencoder feature extractor + SMOTE + LightGBM + BO):
  - One-class AE (latih BARIS NORMAL saja) atas SEMUA fitur (inti Ding).
  - Error rekonstruksi PER-KELUARGA-FITUR (V/C/D/M/card-addr/email/id-device) + global
    MSE/MAE -> ~8-10 skor normalitas nonlinier; sinyal AE maksimal, dilusi minimal.
  - ADDITIVE: semua fitur asli dipertahankan (tak buang info).
  - SMOTE rasio DI-TUNE (bukan 0.5) via Bayesian Optimization.

Skenario (stratified 60/20/20, seed 42, PR-AUC utama, threshold MCC, paired bootstrap;
AE/scaler/SMOTE fit di TRAIN saja):
  baseline_tuned        : seluruh fitur, LightGBM+BO (acuan)
  smote_tuned           : + SMOTE rasio di-tune                         [sudah > baseline]
  ding_global_smote     : + global MSE/MAE (Ding klasik) + SMOTE tuned  [faithful]
  ding_group_smote      : + error per-keluarga + global + SMOTE tuned   [MODIFIKASI TERDALAM]

Target: ding_group_smote >= baseline (idealnya menambah di atas smote_tuned -> kontribusi
AE nyata). Jika hanya ~ smote_tuned, model usulan tetap mengalahkan baseline & AE tetap ada.

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ding_oneclass_grouperr_results.json
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
from tensorflow.keras import regularizers

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
SMOTE_RATIO_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
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
    """Petakan tiap kolom ke keluarga fitur untuk error rekonstruksi per-grup."""
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


def tune_plain(Xtr, ytr, Xva, yva, spw):
    def obj(trial):
        m, it = fit_lgbm(space(trial, spw), Xtr, ytr, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    p = default_params(spw); p.update(st.best_params)
    return fit_lgbm(p, Xtr, ytr, Xva, yva)


def tune_smote(Xtr, ytr, Xva, yva):
    def obj(trial):
        rate = trial.suggest_categorical("smote_rate", SMOTE_RATIO_GRID)
        k = trial.suggest_categorical("smote_k", K_GRID)
        Xs, ys = smote(Xtr, ytr, rate, k)
        m, it = fit_lgbm(space(trial, 1.0), Xs, ys, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    bp = st.best_params
    Xs, ys = smote(Xtr, ytr, bp["smote_rate"], bp["smote_k"])
    p = default_params(1.0); p.update({k: v for k, v in bp.items() if k not in ("smote_rate", "smote_k")})
    m, it = fit_lgbm(p, Xs, ys, Xva, yva)
    return m, it, bp["smote_rate"], bp["smote_k"]


def smote(X, y, rate, k, seed=SEED):
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
    """Arsitektur Ding (diperluas): input->latent(relu,L1)->8->8->input(relu). One-class."""
    inp = keras.Input(shape=(dim,))
    e = keras.layers.Dense(latent, activation="relu", activity_regularizer=regularizers.l1(1e-5))(inp)
    e = keras.layers.Dense(8, activation="relu")(e)
    d = keras.layers.Dense(8, activation="relu")(e)
    out = keras.layers.Dense(dim, activation="relu")(d)
    ae = keras.Model(inp, out); ae.compile(optimizer="adam", loss="mse")
    return ae


def family_error_feats(ae, X, fam):
    """Global MSE/MAE + MSE per-keluarga-fitur (skor normalitas nonlinier per grup)."""
    pred = ae.predict(X, batch_size=8192, verbose=0)
    se = (X - pred) ** 2; ae_abs = np.abs(X - pred)
    feats = [se.mean(1, keepdims=True), ae_abs.mean(1, keepdims=True)]   # global MSE, MAE
    for f, idxs in fam.items():
        feats.append(se[:, idxs].mean(1, keepdims=True))                 # MSE keluarga f
    return np.hstack(feats).astype("float32")


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

    print("\n--- baseline_tuned ---")
    m, it = tune_plain(Xtr, ytr, Xva, yva, spw); record("baseline_tuned", m, it, Xva, Xte)
    print("\n--- smote_tuned ---")
    m, it, r, k = tune_smote(Xtr, ytr, Xva, yva); meta["smote_tuned"] = {"rate": r, "k": k}
    print(f"  SMOTE rate={r} k={k}"); record("smote_tuned", m, it, Xva, Xte)

    # ---- one-class AE (baris normal saja, semua fitur) ----
    print("\nTraining one-class AE (baris normal, semua fitur)...")
    ae = build_oneclass_ae(Xtr.shape[1], AE_LATENT)
    ae.fit(Xtr[ytr == 0], Xtr[ytr == 0], validation_data=(Xva[yva == 0], Xva[yva == 0]),
           epochs=AE_EPOCHS, batch_size=AE_BATCH, shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])

    # error features (global + per-keluarga), z-score fit di train
    Etr, Eva, Ete = (family_error_feats(ae, Xtr, fam), family_error_feats(ae, Xva, fam),
                     family_error_feats(ae, Xte, fam))
    mu = Etr.mean(0); sd = Etr.std(0); sd[sd == 0] = 1.0
    Etr = ((Etr - mu) / sd).astype("float32"); Eva = ((Eva - mu) / sd).astype("float32")
    Ete = ((Ete - mu) / sd).astype("float32")

    # ---- ding_global_smote: hanya 2 error global (Ding klasik) ----
    Gtr = np.hstack([Xtr, Etr[:, :2]]); Gva = np.hstack([Xva, Eva[:, :2]]); Gte = np.hstack([Xte, Ete[:, :2]])
    print("\n--- ding_global_smote (Ding klasik: +MSE/MAE) ---")
    m, it, r, k = tune_smote(Gtr.astype("float32"), ytr, Gva.astype("float32"), yva)
    meta["ding_global_smote"] = {"rate": r, "k": k}; print(f"  SMOTE rate={r} k={k}")
    record("ding_global_smote", m, it, Gva.astype("float32"), Gte.astype("float32"))

    # ---- ding_group_smote: + error per-keluarga (modifikasi terdalam) ----
    Htr = np.hstack([Xtr, Etr]); Hva = np.hstack([Xva, Eva]); Hte = np.hstack([Xte, Ete])
    print(f"\n--- ding_group_smote (+error per-keluarga, +{Etr.shape[1]} fitur) ---")
    m, it, r, k = tune_smote(Htr.astype("float32"), ytr, Hva.astype("float32"), yva)
    meta["ding_group_smote"] = {"rate": r, "k": k}; print(f"  SMOTE rate={r} k={k}")
    record("ding_group_smote", m, it, Hva.astype("float32"), Hte.astype("float32"))

    comp = {
        "smote_vs_baseline":        bootstrap_delta(yte, scores["baseline_tuned"], scores["smote_tuned"]),
        "ding_global_vs_baseline":  bootstrap_delta(yte, scores["baseline_tuned"], scores["ding_global_smote"]),
        "ding_global_vs_smote":     bootstrap_delta(yte, scores["smote_tuned"], scores["ding_global_smote"]),
        "ding_group_vs_baseline":   bootstrap_delta(yte, scores["baseline_tuned"], scores["ding_group_smote"]),
        "ding_group_vs_smote":      bootstrap_delta(yte, scores["smote_tuned"], scores["ding_group_smote"]),
        "ding_group_vs_global":     bootstrap_delta(yte, scores["ding_global_smote"], scores["ding_group_smote"]),
    }

    print("\n========== DING ONE-CLASS GROUP-ERROR BEST SHOT (stratified 60/20/20) ==========")
    print(f"{'skenario':20s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k_ in ("baseline_tuned", "smote_tuned", "ding_global_smote", "ding_group_smote"):
        v = results[k_]; print(f"{k_:20s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k_, b in comp.items():
        print(f"  {k_:26s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "ding_oneclass_grouperr_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp, "selected_smote": meta,
                   "config": {"ae_latent": AE_LATENT, "families": list(fam.keys()),
                              "n_error_feats": int(Etr.shape[1]), "n_trials": N_TRIALS}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
