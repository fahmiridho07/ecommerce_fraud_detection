"""IEEE-CIS — Ding & Du FAITHFUL (EKSPLORASI/diagnosis, BUKAN scope tesis terkunci).

Latar
-----
Pipeline tesis saat ini (AE: V -> laten 64-d, MENGGANTI V) BUKAN Ding maupun Du, dan
merupakan desain terburuk: substitutif (membuang V) + hanya pada blok V. Itulah sebab
"AE menghilangkan info penting" — info-nya memang dibuang.

Dari source code Ding & paper Du (dibaca langsung):
  - DING 2024: one-class AE (baris normal saja) atas SELURUH fitur -> pakai RECONSTRUCTION
    ERROR (MSE & MAE per baris) sebagai fitur ANOMALI tambahan (additive) + SMOTE.
    Arsitektur Ding: input -> Dense(16, relu, L1=1e-5) -> Dense(8) -> Dense(8) -> Dense(input, relu).
  - DU 2023 (AED-LGB): AE simetris atas SELURUH fitur -> pakai LATEN (low-dim) sebagai
    representasi fitur LightGBM. (Du melaporkan SMOTE tidak menambah; tetap diuji sebagai kontrol.)

Skenario (data berstrata 60/20/20, seed 42, PR-AUC utama, threshold MCC di validasi, paired bootstrap):
  baseline        : SELURUH fitur numerik mentah -> LightGBM (acuan, info penuh)
  smote           : baseline + SMOTE (kontrol penyeimbangan)
  ding_error      : SELURUH fitur + MSE/MAE (one-class AE atas semua fitur)   [DING, ADDITIVE]
  ding_error_smote: ding_error + SMOTE                                        [DING FAITHFUL]
  du_latent       : laten(semua fitur) -> LightGBM                            [DU, substitutif]
  du_latent_smote : du_latent + SMOTE                                         [DU + kontrol]

Hipotesis jujur (dari bukti terdahulu): ding_error ~ seri dgn baseline (additive, tak buang
info; SMOTE yang menggerakkan); du_latent < baseline (kompresi seluruh fitur = info loss).
Hasil ini mengkonfirmasi: jalan keluar dari info-loss = pendekatan ADDITIVE (Ding), bukan
substitutif (Du / desain lama).

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", tempel ke satu cell, Run All.
Hasil -> /kaggle/working/ding_du_faithful_results.json
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
DING_LATENT = 16          # encoding_dim Ding (input->16->8->8->input)
DU_LATENT = 32            # laten low-dim Du (representasi fitur)
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 2048
TARGET_FRAUD_RATE = 0.5
K_NEIGHBORS = 5
N_BOOTSTRAP = 2000
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
    """Frequency-encode kategorikal, imputasi median, standarisasi. Fit hanya pada train."""
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


def build_ding_ae(dim):
    """Arsitektur Ding: input -> 16(relu,L1) -> 8 -> 8 -> input(relu). One-class."""
    inp = keras.Input(shape=(dim,))
    e = keras.layers.Dense(DING_LATENT, activation="relu",
                           activity_regularizer=regularizers.l1(1e-5))(inp)
    e = keras.layers.Dense(8, activation="relu")(e)
    d = keras.layers.Dense(8, activation="relu")(e)
    out = keras.layers.Dense(dim, activation="relu")(d)
    ae = keras.Model(inp, out); ae.compile(optimizer="adam", loss="mse")
    return ae


def build_du_ae(dim, latent):
    """AE simetris Du -> laten low-dim sebagai representasi fitur."""
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(256, activation="relu")(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(z)
    x = keras.layers.Dense(256, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, keras.Model(inp, z)


def fit_ae(ae, Xfit, Xval):
    ae.fit(Xfit, Xfit, validation_data=(Xval, Xval), epochs=AE_EPOCHS, batch_size=AE_BATCH,
           shuffle=True, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])
    return ae


def recon_error_feats(ae, X):
    """MSE & MAE per baris (gaya Ding) sebagai 2 kolom anomali."""
    pred = ae.predict(X, batch_size=8192, verbose=0)
    mse = np.mean((X - pred) ** 2, axis=1, keepdims=True)
    mae = np.mean(np.abs(X - pred), axis=1, keepdims=True)
    return np.hstack([mse, mae]).astype("float32")


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, cols = preprocess_numeric(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} feats={len(cols)}")

    results, scores = {}, {}

    def evaluate(name, Xtr2, ytr2, Xva2, Xte2, spw2):
        m, it = fit_lgbm(default_params(spw2), Xtr2, ytr2, Xva2, yva)
        vva = m.predict_proba(Xva2, num_iteration=it)[:, 1]
        vte = m.predict_proba(Xte2, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    # ---- baseline + smote (info penuh) ----
    print("\n--- baseline ---")
    evaluate("baseline", Xtr, ytr, Xva, Xte, spw)
    print("\n--- smote ---")
    Xs, ys = smote(Xtr, ytr)
    evaluate("smote", Xs, ys, Xva, Xte, 1.0)

    # ---- DING faithful: one-class AE atas SEMUA fitur -> MSE/MAE additive ----
    print("\nTraining DING one-class AE (baris normal, semua fitur)...")
    normal = ytr == 0
    ding = build_ding_ae(Xtr.shape[1])
    # validasi one-class juga pakai baris normal (seperti Ding)
    ding = fit_ae(ding, Xtr[normal], Xva[yva == 0])
    etr, eva, ete = recon_error_feats(ding, Xtr), recon_error_feats(ding, Xva), recon_error_feats(ding, Xte)
    Dtr = np.hstack([Xtr, etr]).astype("float32")
    Dva = np.hstack([Xva, eva]).astype("float32")
    Dte = np.hstack([Xte, ete]).astype("float32")
    print("\n--- ding_error (semua fitur + MSE/MAE) ---")
    evaluate("ding_error", Dtr, ytr, Dva, Dte, spw)
    print("\n--- ding_error_smote [DING FAITHFUL] ---")
    Ds, dys = smote(Dtr, ytr)
    evaluate("ding_error_smote", Ds, dys, Dva, Dte, 1.0)
    del ding, etr, eva, ete, Dtr, Dva, Dte, Ds
    tf.keras.backend.clear_session()

    # ---- DU faithful: AE simetris atas SEMUA fitur -> laten sebagai representasi ----
    print("\nTraining DU symmetric AE (semua fitur -> laten)...")
    du, enc = build_du_ae(Xtr.shape[1], DU_LATENT)
    du = fit_ae(du, Xtr, Xva)
    Ltr = enc.predict(Xtr, batch_size=8192, verbose=0).astype("float32")
    Lva = enc.predict(Xva, batch_size=8192, verbose=0).astype("float32")
    Lte = enc.predict(Xte, batch_size=8192, verbose=0).astype("float32")
    print("\n--- du_latent (laten semua fitur) ---")
    evaluate("du_latent", Ltr, ytr, Lva, Lte, spw)
    print("\n--- du_latent_smote ---")
    Ls, lys = smote(Ltr, ytr)
    evaluate("du_latent_smote", Ls, lys, Lva, Lte, 1.0)

    comp = {
        "smote_vs_baseline":            bootstrap_delta(yte, scores["baseline"], scores["smote"]),
        "ding_error_vs_baseline":       bootstrap_delta(yte, scores["baseline"], scores["ding_error"]),
        "ding_error_smote_vs_smote":    bootstrap_delta(yte, scores["smote"], scores["ding_error_smote"]),
        "ding_error_smote_vs_baseline": bootstrap_delta(yte, scores["baseline"], scores["ding_error_smote"]),
        "du_latent_vs_baseline":        bootstrap_delta(yte, scores["baseline"], scores["du_latent"]),
        "du_latent_smote_vs_smote":     bootstrap_delta(yte, scores["smote"], scores["du_latent_smote"]),
    }

    print("\n========== DING & DU FAITHFUL (data berstrata 60/20/20) ==========")
    print(f"{'skenario':20s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k in ("baseline", "smote", "ding_error", "ding_error_smote", "du_latent", "du_latent_smote"):
        v = results[k]; print(f"{k:20s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k, b in comp.items():
        print(f"  {k:30s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "ding_du_faithful_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp,
                   "config": {"ding_latent": DING_LATENT, "du_latent": DU_LATENT,
                              "ae_epochs": AE_EPOCHS, "target_fraud_rate": TARGET_FRAUD_RATE}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
