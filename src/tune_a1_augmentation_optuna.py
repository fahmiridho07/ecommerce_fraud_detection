"""Fair tuned-vs-tuned comparison on the A1 (dense) representation.

Optuna-tunes LightGBM hyperparameters separately for three pipelines that share
the same split, A1 preprocessing, validation/test sets, and tuning budget:
- baseline        : plain A1 LightGBM (scale_pos_weight from data)
- smote_nc        : A1 + raw-space SMOTE oversampling (scale_pos_weight=1.0)
- ae_latent_smote : A1 + AE latent-space oversampling (scale_pos_weight=1.0)

Each pipeline gets its own Optuna study (objective = validation Average
Precision, with early stopping). Best params are refit and evaluated on the test
split, then paired-bootstrap compared. This answers whether the AE advantage
survives fair tuning of both the baseline and the proposed model.

Run:
    python src/tune_a1_augmentation_optuna.py \
        --output-dir outputs/stratified_reset/a1_tuned_comparison --n-trials 20
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import lightgbm as lgb
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("LightGBM/Optuna not installed.") from exc

from config import DEFAULT_SPLIT_STRATEGY, RANDOM_SEED, SAMPLE_SIZE, SUPPORTED_SPLIT_STRATEGIES
from data_loader import load_labeled_train_data
from paper_preprocessing import apply_alharbi_style_preprocessing, fit_alharbi_style_preprocessing
from preprocessing import split_features_target
from splitting import create_holdout_split
from evaluation import binary_classification_metrics, selected_threshold_from_table, threshold_selection_table
from train_baseline_lgbm import average_precision_eval, roc_auc_eval
from run_ae_augmentation_experiment import build_fraud_autoencoder, latent_smote_synthesis
from run_vae_augmentation_experiment import paired_bootstrap_ap_delta
from utils import ensure_dir, log, save_json, set_seed

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc

optuna.logging.set_verbosity(optuna.logging.WARNING)
EARLY_STOPPING_ROUNDS = 100


def suggest_params(trial, spw: float) -> dict:
    return {
        "objective": "binary", "boosting_type": "gbdt", "n_estimators": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": -1,
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": spw, "n_jobs": -1, "random_state": RANDOM_SEED,
        "metric": "None", "verbosity": -1,
    }


def fit_eval(params, X_tr, y_tr, X_va, y_va, X_te):
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric=average_precision_eval,
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True)])
    it = int(model.best_iteration_ or model.n_estimators)
    va = model.predict_proba(X_va, num_iteration=it)[:, 1]
    te = model.predict_proba(X_te, num_iteration=it)[:, 1]
    return va, te, it


def tune_pipeline(name, X_tr, y_tr, X_va, y_va, X_te, spw, n_trials, seed):
    log(f"Tuning {name} ({n_trials} trials)...")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        params = suggest_params(trial, spw)
        va, _, _ = fit_eval(params, X_tr, y_tr, X_va, y_va, X_te[:1] if len(X_te) else X_va[:1])
        return average_precision_score(y_va, va)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = suggest_params(optuna.trial.FixedTrial(study.best_params), spw)
    va, te, it = fit_eval(best, X_tr, y_tr, X_va, y_va, X_te)
    log(f"{name}: best val AP={study.best_value:.6f} best_iter={it}")
    return {"best_params": study.best_params, "best_val_ap": float(study.best_value),
            "best_iteration": it}, va, te


def main(output_dir: Path, split_strategy: str = DEFAULT_SPLIT_STRATEGY, n_trials: int = 20,
         latent_dim: int = 16, ae_epochs: int = 60, k_neighbors: int = 5,
         target_fraud_rate: float = 0.15, n_bootstrap: int = 2000, seed: int = RANDOM_SEED) -> dict:
    set_seed(seed); tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)
    n_trials = min(n_trials, 8)  # bound total runtime (3 pipelines x trials, ~1500 trees each)
    log(f"Effective n_trials per pipeline: {n_trials}")

    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df; gc.collect()
    X_tr_raw, y_train = split_features_target(train_df)
    X_va_raw, y_valid = split_features_target(valid_df)
    X_te_raw, y_test = split_features_target(test_df)
    pre = fit_alharbi_style_preprocessing(X_tr_raw)
    Xa_tr = apply_alharbi_style_preprocessing(X_tr_raw, pre).astype("float32")
    Xa_va = apply_alharbi_style_preprocessing(X_va_raw, pre).astype("float32")
    Xa_te = apply_alharbi_style_preprocessing(X_te_raw, pre).astype("float32")
    y_tr, y_va, y_te = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()
    cols = Xa_tr.columns

    fraud = y_tr == 1
    Xf = np.clip(Xa_tr.loc[fraud].to_numpy("float32"), -10, 10)
    n_fraud, n_normal = int(fraud.sum()), int((~fraud).sum())
    n_synth = max(0, int(round(n_normal / (1.0 - target_fraud_rate) - (n_normal + n_fraud))))
    spw_base = n_normal / max(n_fraud, 1)

    def build_train(name):
        """Build (X_train, y_train, spw) lazily per pipeline to bound memory."""
        if name == "baseline":
            return Xa_tr, y_tr, spw_base
        if name == "random_oversample":
            anchors = np.random.default_rng(seed).integers(0, n_fraud, size=n_synth)
            syn = Xa_tr.loc[fraud].reset_index(drop=True).iloc[anchors].reset_index(drop=True)
        elif name == "smote_nc":
            pts, _ = latent_smote_synthesis(Xf, n_synth, k_neighbors, np.random.default_rng(seed))
            syn = pd.DataFrame(pts, columns=cols)
        elif name == "ae_latent_smote":
            ae, enc, dec = build_fraud_autoencoder(Xf.shape[1], latent_dim)
            ae.fit(Xf, Xf, validation_split=0.1, epochs=ae_epochs, batch_size=256, shuffle=True,
                   callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
            lat = enc.predict(Xf, batch_size=1024, verbose=0)
            pts, _ = latent_smote_synthesis(lat, n_synth, k_neighbors, np.random.default_rng(seed))
            syn = pd.DataFrame(dec.predict(pts, batch_size=1024, verbose=0), columns=cols)
        else:
            raise ValueError(name)
        return (pd.concat([Xa_tr, syn], ignore_index=True),
                np.concatenate([y_tr, np.ones(n_synth, int)]), 1.0)

    results, scores = {}, {}
    ckpt = output_dir / "partial_results.json"
    for name in ("baseline", "random_oversample", "smote_nc", "ae_latent_smote"):
        Xtr, ytr, spw = build_train(name)
        info, va, te = tune_pipeline(name, Xtr, ytr, Xa_va, y_va, Xa_te, spw, n_trials, seed)
        table = threshold_selection_table(y_va, va); thr = selected_threshold_from_table(table)
        m = binary_classification_metrics(y_te, te, thr)
        results[name] = {"test_average_precision": m["average_precision"], "test_roc_auc": m["roc_auc"],
                         "test_f1": m["f1"], "test_mcc": m["mcc"], "selected_threshold": thr, **info}
        scores[name] = te
        if name != "baseline":
            del Xtr, ytr
        gc.collect()
        save_json({"results": results}, ckpt)  # checkpoint after each pipeline

    comp = {
        "ae_vs_baseline": paired_bootstrap_ap_delta(y_te, scores["baseline"], scores["ae_latent_smote"], n_bootstrap),
        "ae_vs_smote_nc": paired_bootstrap_ap_delta(y_te, scores["smote_nc"], scores["ae_latent_smote"], n_bootstrap),
        "ae_vs_random_oversample": paired_bootstrap_ap_delta(y_te, scores["random_oversample"], scores["ae_latent_smote"], n_bootstrap),
        "smote_vs_baseline": paired_bootstrap_ap_delta(y_te, scores["baseline"], scores["smote_nc"], n_bootstrap),
        "random_vs_baseline": paired_bootstrap_ap_delta(y_te, scores["baseline"], scores["random_oversample"], n_bootstrap),
    }
    summary = {"representation": "A1_alharbi_dense", "split_strategy": split_strategy, "n_trials": n_trials,
               "seed": seed, "target_fraud_rate": target_fraud_rate, "n_synthetic": n_synth,
               "test_prevalence": float(y_te.mean()), "results": results, "comparisons": comp}
    save_json(summary, output_dir / "experiment_summary.json")

    print("\nA1 Tuned-vs-Tuned Comparison")
    print("============================")
    for name in ("baseline", "random_oversample", "smote_nc", "ae_latent_smote"):
        print(f"{name:16s} tuned test AP={results[name]['test_average_precision']:.6f}")
    for k, b in comp.items():
        print(f"{k:18s} delta={b['observed_delta_ap']:+.6f} ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Fair tuned-vs-tuned A1 augmentation comparison.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--ae-epochs", type=int, default=60)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--target-fraud-rate", type=float, default=0.15)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    main(output_dir=a.output_dir, split_strategy=a.split_strategy, n_trials=a.n_trials,
         latent_dim=a.latent_dim, ae_epochs=a.ae_epochs, k_neighbors=a.k_neighbors,
         target_fraud_rate=a.target_fraud_rate, n_bootstrap=a.n_bootstrap, seed=a.seed)
