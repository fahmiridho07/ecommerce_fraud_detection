"""IEEE-CIS — Swap-Noise Denoising AE + RankGauss + Masked MSE (POV laporan deep-research).

Kartu terakhir yang BELUM diuji & faithful (AE feature extractor + LightGBM, judul utuh).
Bukan teori kosong: swap-noise DAE = teknik juara-1 Kaggle Porto Seguro (M. Jahrer) untuk
representation learning tabular. Beda dari denoising-Gaussian (sudah gagal): swap-noise
menukar nilai antar-baris per-kolom -> memaksa AE belajar korelasi gabungan antar-fitur,
bukan peta identitas.

Tiga perbaikan kualitas-AE (sesuai laporan), semua faithful:
  1. RankGauss (QuantileTransformer output='normal') pada input AE -> distribusi mulus.
  2. Swap-noise (p=0.15) saat latih DAE -> patahkan identity mapping.
  3. Masked MSE -> rugi dihitung HANYA pada sel teramati (bukan sel imputasi).
Lalu ADDITIVE: semua fitur asli + laten DAE + recon-error -> LightGBM (+SMOTE tuned, BO).

CAVEAT JUJUR: kemenangan Porto Seguro pakai NN di hilir, bukan GBDT. Laten DAE tetap f(X);
redundansi terhadap LightGBM mungkin tetap ada. Ini peluang terbaik tersisa, BUKAN jaminan.

Skenario (stratified 60/20/20, seed 42, PR-AUC + bootstrap; AE/scaler/SMOTE fit TRAIN saja):
  baseline_tuned   : seluruh fitur -> LightGBM+BO
  smote_tuned      : + SMOTE rasio di-tune                  [palang ke-2]
  dae_concat_smote : semua fitur + laten DAE + recon-error + SMOTE tuned   [USULAN]

Bandingkan dae_concat_smote vs baseline (palang 1) DAN vs smote (palang 2).

PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", Run All.
Hasil -> /kaggle/working/swapnoise_dae_results.json
"""

from __future__ import annotations

import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors

import lightgbm as lgb
import optuna
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000
EARLY_STOPPING = 100
LATENT_DIM = 32
SWAP_P = 0.15
AE_EPOCHS = 50
AE_PATIENCE = 6
AE_BATCH = 1024
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


def preprocess(train, valid, test):
    """Untuk LightGBM: freq-enc + median impute + z-score (representasi standar).
    Juga kembalikan MASK observasi (pra-imputasi) untuk masked-MSE & matriks V."""
    drop = [c for c in (ID, TARGET) if c in train.columns]
    feat = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in feat if train[c].dtype == "object"]
    num_cols = [c for c in feat if c not in cat_cols]
    out, mask = {}, {}
    for nm, df in (("tr", train), ("va", valid), ("te", test)):
        cols, mcols = {}, {}
        for c in cat_cols:
            freq = train[c].value_counts(normalize=True)
            mcols[c] = df[c].notna().astype("float32")
            cols[c] = df[c].map(freq).fillna(0.0).astype("float32")
        for c in num_cols:
            mcols[c] = df[c].notna().astype("float32")
            cols[c] = df[c].fillna(float(train[c].median())).astype("float32")
        out[nm] = pd.DataFrame(cols); mask[nm] = pd.DataFrame(mcols)
    order = list(out["tr"].columns)
    mu = out["tr"].mean(); sd = out["tr"].std().replace(0, 1.0)
    for nm in out:
        out[nm] = ((out[nm][order] - mu) / sd).clip(-10, 10).astype("float32")
    v_cols = [c for c in order if c.startswith("V") and c[1:].isdigit()]
    return (out["tr"].to_numpy(), out["va"].to_numpy(), out["te"].to_numpy(),
            mask["tr"][order].to_numpy(), mask["va"][order].to_numpy(), mask["te"][order].to_numpy(),
            order, v_cols)


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
        rate = trial.suggest_categorical("rate", RATIO_GRID)
        k = trial.suggest_categorical("k", K_GRID)
        Xs, ys = smote(Xtr, ytr, rate, k)
        m, it = fit_lgbm(space(trial, 1.0), Xs, ys, Xva, yva)
        return average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=N_TRIALS)
    bp = st.best_params
    Xs, ys = smote(Xtr, ytr, bp["rate"], bp["k"])
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


def masked_mse(value_dim):
    def loss(y_true, y_pred):
        target = y_true[:, :value_dim]; mask = y_true[:, value_dim:]
        se = tf.square(target - y_pred) * mask
        denom = tf.reduce_sum(mask, axis=-1) + tf.keras.backend.epsilon()
        return tf.reduce_sum(se, axis=-1) / denom
    return loss


def build_dae(dim, latent):
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(512, activation="relu")(inp)
    x = keras.layers.Dense(256, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="linear", name="latent")(x)
    x = keras.layers.Dense(256, activation="relu")(z)
    x = keras.layers.Dense(512, activation="relu")(x)
    out = keras.layers.Dense(dim, activation="linear")(x)
    ae = keras.Model(inp, out)
    return ae, keras.Model(inp, z)


class SwapNoiseSeq(keras.utils.Sequence):
    """Yield (input ber-swap-noise, target=[nilai bersih | mask]) dgn korupsi baru tiap epoch."""
    def __init__(self, X, M, batch, p, seed):
        self.X = X.astype("float32"); self.M = M.astype("float32")
        self.batch = batch; self.p = p; self.rng = np.random.default_rng(seed)
        self.n = len(X); self.on_epoch_end()

    def __len__(self): return int(np.ceil(self.n / self.batch))

    def on_epoch_end(self):
        self.perm = self.rng.permutation(self.n)

    def __getitem__(self, i):
        idx = self.perm[i * self.batch:(i + 1) * self.batch]
        xb = self.X[idx].copy(); mb = self.M[idx]
        # swap noise: tiap sel dgn prob p diganti nilai kolom yg sama dari baris acak
        swap = self.rng.random(xb.shape) < self.p
        if swap.any():
            donor = self.rng.integers(0, self.n, xb.shape[0])
            src = self.X[donor]
            xb[swap] = src[swap]
        target = np.hstack([self.X[idx], mb]).astype("float32")
        return xb, target


def main():
    print("Loading + preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    Xtr, Xva, Xte, Mtr, Mva, Mte, cols, v_cols = preprocess(train, valid, test)
    del train, valid, test
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    vmask = np.array([c in set(v_cols) for c in cols])
    print(f"train={len(Xtr)} feats={len(cols)} V={int(vmask.sum())}")

    results, scores, meta = {}, {}, {}

    def record(name, m, it, Mva_, Mte_):
        vva = m.predict_proba(Mva_, num_iteration=it)[:, 1]
        vte = m.predict_proba(Mte_, num_iteration=it)[:, 1]
        mm = metrics(yte, vte, pick_threshold(yva, vva))
        results[name] = mm; scores[name] = vte
        print(f"[{name}] AP={mm['average_precision']:.6f} ROC={mm['roc_auc']:.5f} "
              f"F1={mm['f1']:.4f} MCC={mm['mcc']:.4f}")

    print("\n--- baseline_tuned ---")
    m, it = tune_plain(Xtr, ytr, Xva, yva, spw); record("baseline_tuned", m, it, Xva, Xte)
    print("\n--- smote_tuned ---")
    m, it, r, k = tune_smote(Xtr, ytr, Xva, yva); meta["smote_tuned"] = {"rate": r, "k": k}
    print(f"  SMOTE rate={r} k={k}"); record("smote_tuned", m, it, Xva, Xte)

    # ---- RankGauss pada blok V (untuk input DAE) ----
    print("\nRankGauss (QuantileTransformer) pada V untuk input DAE...")
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(Xtr)),
                             subsample=10**9, random_state=SEED)
    Vtr = qt.fit_transform(Xtr[:, vmask]).astype("float32")
    Vva = qt.transform(Xva[:, vmask]).astype("float32")
    Vte = qt.transform(Xte[:, vmask]).astype("float32")
    MVtr, MVva, MVte = Mtr[:, vmask], Mva[:, vmask], Mte[:, vmask]

    # ---- latih Swap-Noise DAE + Masked MSE ----
    print("Training Swap-Noise DAE (masked MSE)...")
    dim = Vtr.shape[1]
    ae, enc = build_dae(dim, LATENT_DIM)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss=masked_mse(dim))
    seq = SwapNoiseSeq(Vtr, MVtr, AE_BATCH, SWAP_P, SEED)
    val_target = np.hstack([Vva, MVva]).astype("float32")
    ae.fit(seq, validation_data=(Vva, val_target), epochs=AE_EPOCHS, verbose=2,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss",
                      patience=AE_PATIENCE, restore_best_weights=True)])

    def feats(V, MV):
        z = enc.predict(V, batch_size=8192, verbose=0).astype("float32")
        rec = ae.predict(V, batch_size=8192, verbose=0)
        se = ((V - rec) ** 2) * MV
        denom = np.maximum(MV.sum(1, keepdims=True), 1.0)
        err = np.hstack([se.sum(1, keepdims=True) / denom,
                         (np.abs(V - rec) * MV).sum(1, keepdims=True) / denom]).astype("float32")
        return np.hstack([z, err]).astype("float32")

    Ftr, Fva, Fte = feats(Vtr, MVtr), feats(Vva, MVva), feats(Vte, MVte)
    mu = Ftr.mean(0); sd = Ftr.std(0); sd[sd == 0] = 1.0
    Ftr = (Ftr - mu) / sd; Fva = (Fva - mu) / sd; Fte = (Fte - mu) / sd

    Ctr = np.hstack([Xtr, Ftr]).astype("float32")     # additive: semua fitur + DAE feats
    Cva = np.hstack([Xva, Fva]).astype("float32")
    Cte = np.hstack([Xte, Fte]).astype("float32")
    print(f"  fitur: {Xtr.shape[1]} -> {Ctr.shape[1]} (+{Ftr.shape[1]} DAE)")

    print("\n--- dae_concat_smote (USULAN) ---")
    m, it, r, k = tune_smote(Ctr, ytr, Cva, yva); meta["dae_concat_smote"] = {"rate": r, "k": k}
    print(f"  SMOTE rate={r} k={k}"); record("dae_concat_smote", m, it, Cva, Cte)

    comp = {
        "smote_vs_baseline":      bootstrap_delta(yte, scores["baseline_tuned"], scores["smote_tuned"]),
        "dae_vs_baseline":        bootstrap_delta(yte, scores["baseline_tuned"], scores["dae_concat_smote"]),
        "dae_vs_smote":           bootstrap_delta(yte, scores["smote_tuned"], scores["dae_concat_smote"]),
    }

    print("\n========== SWAP-NOISE DAE + RankGauss + Masked MSE (stratified 60/20/20) ==========")
    print(f"{'skenario':18s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'F1':>8s} {'MCC':>8s}")
    for k_ in ("baseline_tuned", "smote_tuned", "dae_concat_smote"):
        v = results[k_]; print(f"{k_:18s} {v['average_precision']:9.6f} {v['roc_auc']:9.5f} {v['f1']:8.4f} {v['mcc']:8.4f}")
    print("\nPerbandingan (paired bootstrap on AP):")
    for k_, b in comp.items():
        print(f"  {k_:22s} delta={b['observed_delta_ap']:+.6f} "
              f"ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    out_dir = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    out_path = os.path.join(out_dir, "swapnoise_dae_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "comparisons": comp, "selected_smote": meta,
                   "config": {"latent_dim": LATENT_DIM, "swap_p": SWAP_P, "n_trials": N_TRIALS}}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
