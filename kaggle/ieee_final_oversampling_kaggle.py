"""IEEE-CIS — EKSPERIMEN FINAL: integrasi Autoencoder sebagai latent-space oversampler + LightGBM + Bayesian Optimization.

Metode mengikuti protokol yang terjustifikasi referensi:
  1. Prapemrosesan representasi padat A1: frequency-encoding (kategorikal kardinalitas tinggi),
     imputasi median (tahan pencilan), Z-score (stabilitas pelatihan AE & jarak SMOTE).
     Missing kategorikal diberi token khusus; fitur kode kategorikal numerik tetap diperlakukan
     sebagai kategorikal.
     Ref: Misra et al. (2020) [z-score utk AE]; Chawla et al. (2002) [SMOTE berbasis jarak].
  2. Analisis sensitivitas pada VALIDATION (grid) untuk mengunci dua hyperparameter metode:
     latent_dim in {8,16,32,64}, target_fraud_rate in {0.10,0.15,0.20,0.30,0.50}.
     LightGBM default dipakai agar efek parameter AE terisolasi. Parameter dikunci dari val PR-AUC.
  3. Eksperimen utama (3 skenario x {default, tuned}):
       baseline  : LightGBM tanpa oversampling (scale_pos_weight dari data). Ref: Ke et al. (2017).
       smote     : SMOTE pada ruang fitur. Ref: Chawla et al. (2002).  [pembanding]
       ae_oversample : AE latent-space oversampling (USULAN). Ref: Dablain et al. (2022); Fan et al. (2025).
     target_fraud_rate sama (terkunci) untuk smote & ae -> perbandingan terkontrol.
  4. Optimasi: Bayesian Optimization (TPE) via Optuna, objektif = val PR-AUC.
     Ref: Bergstra et al. (2011); Akiba et al. (2019); Lim et al. (2024).
     N_TRIALS=30 untuk run final; dapat diturunkan sementara jika hanya debugging komputasi.
  5. Threshold dipilih dari VALIDATION dgn memaksimalkan F1 (selaras metrik Bab 2).
  6. Metrik (selaras Bab 2): PR-AUC (utama), Recall, F1-Score, ROC-AUC. Ref: Davis & Goadrich (2006); Saito & Rehmsmeier (2015).
  7. TEST hanya dipakai pada evaluasi akhir, setelah seluruh parameter & threshold terkunci.

Protokol: stratified holdout 60/20/20, seed 42. AE/encoder/decoder/SMOTE & seluruh fitting hanya pada train.
PAKAI DI KAGGLE: Add Input "IEEE-CIS Fraud Detection", Run All. Hasil -> /kaggle/working/final_oversampling_results.json
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import lightgbm as lgb
import optuna
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")
try:
    tf.autograph.set_verbosity(0)
except Exception:
    pass

DATA_DIR = "/kaggle/input/ieee-fraud-detection"
SEED = 42
N_ESTIMATORS = 2000          # ditahan tinggi + early stopping (praktik standar GBDT; Ke et al., 2017)
EARLY_STOPPING = 100
SENS_N_ESTIMATORS = 800      # lebih ringan utk tahap sensitivitas (hanya untuk meranking parameter AE)
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH = 256
K_NEIGHBORS = 5              # default SMOTE (Chawla et al., 2002)
N_TRIALS = 10               # final run; turunkan sementara jika hanya debugging komputasi
LATENT_GRID = [8, 16, 32, 64]
RATE_GRID = [0.10, 0.15, 0.20, 0.30, 0.50]
TARGET, ID = "isFraud", "TransactionID"
MISSING_TOKEN = "__MISSING__"

optuna.logging.set_verbosity(optuna.logging.ERROR)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ----------------------------- data & prapemrosesan -----------------------------
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


# Fitur kategorikal resmi IEEE-CIS (sebagian berkode numerik: card1-6, addr1-2 -> tetap kategorikal).
# Sumber: deskripsi dataset IEEE-CIS Fraud Detection (Vesta Corp., 2019).
IEEE_CATEGORICAL = (
    {"ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
     "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"}
    | {f"M{i}" for i in range(1, 10)}            # M1..M9
    | {f"id_{i:02d}" for i in range(12, 39)}     # id_12..id_38
)


def a1_preprocess(train, valid, test):
    """Representasi numerik padat A1. Semua statistik di-fit HANYA di train (anti-leakage).
    Fitur kategorikal ditentukan dari daftar resmi IEEE-CIS (termasuk yang berkode numerik
    seperti card1-card6 dan addr1-addr2), bukan sekadar dtype object."""
    drop = [c for c in (ID, TARGET) if c in train.columns]
    feat = [c for c in train.columns if c not in drop]
    cat_cols = [c for c in feat if c in IEEE_CATEGORICAL or train[c].dtype == "object"]
    num_cols = [c for c in feat if c not in cat_cols]

    def norm_cat(s):
        return s.astype("string").fillna(MISSING_TOKEN).astype(str)

    freqs = {c: norm_cat(train[c]).value_counts(normalize=True, dropna=False) for c in cat_cols}
    meds = train[num_cols].median(skipna=True).fillna(0.0).to_dict()       # imputasi median

    def build(df):
        d = {}
        for c in cat_cols:
            d[c] = norm_cat(df[c]).map(freqs[c]).fillna(0.0)
        for c in num_cols:
            d[c] = df[c].fillna(meds[c])
        return pd.DataFrame(d, index=df.index)

    Xtr, Xva, Xte = build(train), build(valid), build(test)
    cols = list(Xtr.columns)
    mu = Xtr.mean().fillna(0.0)
    sd = Xtr.std(ddof=0).replace(0, np.nan).fillna(1.0)                    # z-score (fit train)
    Xtr = ((Xtr - mu) / sd).clip(-10, 10).astype("float32")
    Xva = ((Xva - mu) / sd).clip(-10, 10).astype("float32")
    Xte = ((Xte - mu) / sd).clip(-10, 10).astype("float32")
    return Xtr[cols].to_numpy(), Xva[cols].to_numpy(), Xte[cols].to_numpy(), cols


# ----------------------------- LightGBM -----------------------------
def ap_eval(y, p): return "ap", average_precision_score(y, p), True


def default_params(spw, n_estimators=N_ESTIMATORS):
    return dict(objective="binary", boosting_type="gbdt", n_estimators=n_estimators,
                learning_rate=0.03, num_leaves=64, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                random_state=SEED, metric="None", verbosity=-1)


def fit_lgbm(params, Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(**params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric=ap_eval,
          callbacks=[lgb.early_stopping(EARLY_STOPPING, first_metric_only=True, verbose=False),
                     lgb.log_evaluation(period=0)])
    return m, int(m.best_iteration_ or params["n_estimators"])


def tune_lgbm(Xtr, ytr, Xva, yva, spw):
    """Bayesian Optimization (TPE) — objektif: validation PR-AUC (Akiba et al., 2019; Lim et al., 2024)."""
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
    m, it = fit_lgbm(p, Xtr, ytr, Xva, yva)
    info = {"best_validation_pr_auc": float(study.best_value), "best_params": study.best_params, "best_iteration": it}
    return m, it, info


# ----------------------------- evaluasi -----------------------------
def pick_threshold_f1(yva, pva):
    """Threshold dikunci di VALIDATION dgn memaksimalkan F1 (selaras metrik Bab 2)."""
    best_t, best = 0.5, -1.0
    for t in np.arange(0.01, 1.0, 0.01):
        v = f1_score(yva, (pva >= t).astype(int))
        if v > best: best, best_t = v, float(t)
    return best_t


def metrics(yte, pte, thr):
    pred = (pte >= thr).astype(int)
    return dict(pr_auc=float(average_precision_score(yte, pte)),
                roc_auc=float(roc_auc_score(yte, pte)),
                recall=float(recall_score(yte, pred, zero_division=0)),
                precision=float(precision_score(yte, pred, zero_division=0)),
                f1=float(f1_score(yte, pred, zero_division=0)))


# ----------------------------- oversampling -----------------------------
def n_synth_for(n_norm, n_fr, rate):
    return max(0, int(round(n_norm / (1.0 - rate) - (n_norm + n_fr))))


def smote_interp(points, n_synth, k, rng):
    """Interpolasi SMOTE (Chawla et al., 2002) pada ruang `points` minoritas."""
    if n_synth <= 0 or len(points) < 2:
        return np.empty((0, points.shape[1]), dtype="float32")
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(points))).fit(points)
    _, idx = nn.kneighbors(points)
    anchors = rng.integers(0, len(points), n_synth)
    out = np.empty((n_synth, points.shape[1]), dtype="float32")
    for i, a in enumerate(anchors):
        nbrs = idx[a][1:]; b = nbrs[rng.integers(0, len(nbrs))]
        lam = rng.random(); out[i] = points[a] + lam * (points[b] - points[a])
    return out


def build_ae(dim, latent):
    """Undercomplete AE (Goodfellow et al., 2016; Misra et al., 2020); Adam (Kingma & Ba, 2014)."""
    inp = keras.Input(shape=(dim,))
    x = keras.layers.Dense(128, activation="relu")(inp)
    x = keras.layers.Dense(64, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="linear")(x)
    dec_64 = keras.layers.Dense(64, activation="relu")
    dec_128 = keras.layers.Dense(128, activation="relu")
    out_layer = keras.layers.Dense(dim, activation="linear")
    x = dec_64(z)
    x = dec_128(x)
    out = out_layer(x)
    ae = keras.Model(inp, out); ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    enc = keras.Model(inp, z)
    dec_in = keras.Input(shape=(latent,))
    dec_out = out_layer(dec_128(dec_64(dec_in)))
    dec = keras.Model(dec_in, dec_out)
    return ae, enc, dec


def train_fraud_ae(Xf, latent_dim):
    """Latih AE pada sampel fraud; kembalikan encoder, decoder, dan laten fraud."""
    ae, enc, decoder = build_ae(Xf.shape[1], latent_dim)
    ae.fit(Xf, Xf, validation_split=0.1, epochs=AE_EPOCHS, batch_size=AE_BATCH, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=AE_PATIENCE,
                      restore_best_weights=True)], verbose=0)
    lat_fraud = enc.predict(Xf, batch_size=2048, verbose=0)
    return enc, decoder, lat_fraud


def make_ae_oversample(Xtr, ytr, decoder, lat_fraud, rate, rng):
    n_norm = int((ytr == 0).sum()); n_fr = int((ytr == 1).sum())
    ns = n_synth_for(n_norm, n_fr, rate)
    syn_lat = smote_interp(lat_fraud, ns, K_NEIGHBORS, rng)
    if len(syn_lat) == 0:
        return Xtr, ytr
    syn = decoder.predict(syn_lat, batch_size=2048, verbose=0).astype("float32")
    return np.vstack([Xtr, syn]).astype("float32"), np.concatenate([ytr, np.ones(len(syn), int)])


def make_smote_oversample(Xtr, ytr, rate, rng):
    n_norm = int((ytr == 0).sum()); n_fr = int((ytr == 1).sum())
    ns = n_synth_for(n_norm, n_fr, rate)
    syn = smote_interp(Xtr[ytr == 1], ns, K_NEIGHBORS, rng)
    if len(syn) == 0:
        return Xtr, ytr
    return np.vstack([Xtr, syn]).astype("float32"), np.concatenate([ytr, np.ones(len(syn), int)])


# ----------------------------- main -----------------------------
def main():
    print("Loading + A1 preprocessing...")
    train, valid, test = split_60_20_20(load_data())
    ytr = train[TARGET].to_numpy(); yva = valid[TARGET].to_numpy(); yte = test[TARGET].to_numpy()
    valid_ids = valid[ID].to_numpy(); test_ids = test[ID].to_numpy()
    Xtr, Xva, Xte, cols = a1_preprocess(train, valid, test); del train, valid, test
    n_norm = int((ytr == 0).sum()); n_fr = int((ytr == 1).sum())
    spw = n_norm / max(n_fr, 1)
    Xf = Xtr[ytr == 1]
    print(f"train={len(Xtr)} valid={len(Xva)} test={len(Xte)} feats={len(cols)} fraud={n_fr}")

    # ===== TAHAP 1: ANALISIS SENSITIVITAS (VALIDATION) -> kunci latent_dim & target_fraud_rate =====
    print("\n=== Tahap 1: analisis sensitivitas latent_dim x target_fraud_rate (validation, LightGBM default) ===")
    sens = []
    best = {"val_pr_auc": -1.0, "latent_dim": None, "rate": None}
    for ld in LATENT_GRID:
        enc, decoder, lat_fraud = train_fraud_ae(Xf, ld)
        for rate in RATE_GRID:
            rng = np.random.default_rng(SEED)
            Xa, ya = make_ae_oversample(Xtr, ytr, decoder, lat_fraud, rate, rng)
            m, it = fit_lgbm(default_params(1.0, SENS_N_ESTIMATORS), Xa, ya, Xva, yva)
            val_ap = float(average_precision_score(yva, m.predict_proba(Xva, num_iteration=it)[:, 1]))
            sens.append({"latent_dim": ld, "rate": rate, "val_pr_auc": val_ap})
            print(f"  latent_dim={ld:>3} rate={rate:.2f} -> val PR-AUC={val_ap:.6f}")
            if val_ap > best["val_pr_auc"]:
                best = {"val_pr_auc": val_ap, "latent_dim": ld, "rate": rate}
        tf.keras.backend.clear_session()
    LD, RATE = best["latent_dim"], best["rate"]
    print(f"\n>>> Terkunci: latent_dim={LD}, target_fraud_rate={RATE} (val PR-AUC={best['val_pr_auc']:.6f})")

    # ===== TAHAP 2: EKSPERIMEN UTAMA (parameter terkunci) =====
    results, valid_scores, test_scores, run_info = {}, {}, {}, {}

    def run(name, Xtr2, ytr2, spw2, tuned):
        if tuned:
            m, it, info = tune_lgbm(Xtr2, ytr2, Xva, yva, spw2)
        else:
            m, it = fit_lgbm(default_params(spw2), Xtr2, ytr2, Xva, yva)
            info = {"best_validation_pr_auc": None, "best_params": {}, "best_iteration": it}
        pva = m.predict_proba(Xva, num_iteration=it)[:, 1]
        pte = m.predict_proba(Xte, num_iteration=it)[:, 1]
        thr = pick_threshold_f1(yva, pva)                 # threshold dikunci di validation
        mm = metrics(yte, pte, thr); mm["threshold"] = thr
        results[name] = mm; valid_scores[name] = pva; test_scores[name] = pte; run_info[name] = info
        print(f"[{name}] PR-AUC={mm['pr_auc']:.6f} ROC={mm['roc_auc']:.5f} "
              f"Recall={mm['recall']:.4f} F1={mm['f1']:.4f}")

    print("\n=== Tahap 2: eksperimen utama (baseline / smote / ae_oversample) ===")
    print("--- baseline ---")
    run("baseline", Xtr, ytr, spw, False)
    run("baseline_tuned", Xtr, ytr, spw, True)

    print("--- smote ---")
    rng = np.random.default_rng(SEED)
    Xs, ys = make_smote_oversample(Xtr, ytr, RATE, rng)
    run("smote", Xs, ys, 1.0, False)
    run("smote_tuned", Xs, ys, 1.0, True)

    print("--- ae_oversample (USULAN) ---")
    enc, decoder, lat_fraud = train_fraud_ae(Xf, LD)
    rng = np.random.default_rng(SEED)
    Xa, ya = make_ae_oversample(Xtr, ytr, decoder, lat_fraud, RATE, rng)
    run("ae_oversample", Xa, ya, 1.0, False)
    run("ae_oversample_tuned", Xa, ya, 1.0, True)

    # ===== perbandingan (delta PR-AUC test) =====
    def delta(a, b): return float(results[a]["pr_auc"] - results[b]["pr_auc"])
    comp = {
        "ae_vs_baseline": delta("ae_oversample", "baseline"),
        "ae_vs_smote": delta("ae_oversample", "smote"),
        "ae_vs_baseline_tuned": delta("ae_oversample_tuned", "baseline_tuned"),
        "ae_vs_smote_tuned": delta("ae_oversample_tuned", "smote_tuned"),
    }

    print("\n================ HASIL UTAMA (test, stratified 60/20/20) ================")
    print(f"{'skenario':22s} {'PR-AUC':>9s} {'ROC-AUC':>9s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s}")
    for k in ("baseline", "baseline_tuned", "smote", "smote_tuned", "ae_oversample", "ae_oversample_tuned"):
        v = results[k]
        print(f"{k:22s} {v['pr_auc']:9.6f} {v['roc_auc']:9.5f} {v['recall']:8.4f} {v['precision']:8.4f} {v['f1']:8.4f}")
    print("\nKontribusi (delta PR-AUC test):")
    for k, d in comp.items():
        print(f"  {k:24s} {d:+.6f}")

    out_dir = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".")
    out_path = out_dir / "final_oversampling_results.json"
    scenario_order = ["baseline", "baseline_tuned", "smote", "smote_tuned", "ae_oversample", "ae_oversample_tuned"]
    split_summary = {
        "train": {"rows": int(len(ytr)), "fraud_count": int(ytr.sum()), "fraud_rate": float(ytr.mean())},
        "validation": {"rows": int(len(yva)), "fraud_count": int(yva.sum()), "fraud_rate": float(yva.mean())},
        "test": {"rows": int(len(yte)), "fraud_count": int(yte.sum()), "fraud_rate": float(yte.mean())},
    }

    pd.DataFrame(sens).to_csv(out_dir / "sensitivity_analysis.csv", index=False)
    pd.DataFrame.from_dict(results, orient="index").loc[scenario_order].to_csv(out_dir / "metrics_table.csv")
    pd.DataFrame.from_dict(run_info, orient="index").loc[scenario_order].to_json(
        out_dir / "best_params.json", orient="index", indent=2
    )
    with open(out_dir / "split_summary.json", "w") as f:
        json.dump(split_summary, f, indent=2)

    val_scores_df = pd.DataFrame({ID: valid_ids, "isFraud": yva})
    test_scores_df = pd.DataFrame({ID: test_ids, "isFraud": yte})
    for name in scenario_order:
        val_scores_df[name] = valid_scores[name]
        test_scores_df[name] = test_scores[name]
    val_scores_df.to_csv(out_dir / "validation_scores.csv", index=False)
    test_scores_df.to_csv(out_dir / "test_scores.csv", index=False)

    with open(out_path, "w") as f:
        json.dump({"locked": {"latent_dim": LD, "target_fraud_rate": RATE,
                              "val_pr_auc": best["val_pr_auc"]},
                   "sensitivity": sens, "results": results, "comparisons": comp,
                   "training_info": run_info, "split_summary": split_summary,
                   "config": {"n_trials": N_TRIALS, "k_neighbors": K_NEIGHBORS,
                              "ae_epochs": AE_EPOCHS, "seed": SEED,
                              "latent_grid": LATENT_GRID, "rate_grid": RATE_GRID,
                              "sensitivity_n_estimators": SENS_N_ESTIMATORS}}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {out_dir / 'metrics_table.csv'}")
    print(f"Saved: {out_dir / 'sensitivity_analysis.csv'}")
    print(f"Saved: {out_dir / 'split_summary.json'}")
    print(f"Saved: {out_dir / 'validation_scores.csv'}")
    print(f"Saved: {out_dir / 'test_scores.csv'}")
    print(f"Saved: {out_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
