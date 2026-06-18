"""VAE generative augmentation - the genuinely AE-specific generation test.

The fair comparison showed AE latent-SMOTE ties standard SMOTE-NC, because both
just interpolate existing fraud. A variational autoencoder instead learns a
regularised latent distribution and GENERATES new fraud by sampling the prior
N(0, I), which can fill the fraud manifold more richly than linear interpolation.
If VAE-generated fraud beats SMOTE-NC, the autoencoder component has specific
value beyond classical oversampling. Anchor: Alharbi et al. (2026) VAE-GAN
generative augmentation on IEEE-CIS.

Same fairness discipline as run_fair_augmentation_comparison.py: identical split,
test/validation never augmented, same LightGBM budget, scale_pos_weight=1.0 for
all oversamplers, train-fraud-only generation (no leakage). Compared variants:
baseline, smote_nc (control), vae_prior (proposed).

Run:
    python src/run_vae_augmentation_experiment.py \
        --output-dir outputs/stratified_reset/vae_augmentation_experiment \
        --target-fraud-rate 0.15 --n-estimators 800
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_SPLIT_STRATEGY, ID_COL, RANDOM_SEED, SAMPLE_SIZE, SUPPORTED_SPLIT_STRATEGIES, TARGET_COL
from data_loader import load_labeled_train_data
from preprocessing import apply_baseline_preprocessing, fit_baseline_preprocessing, split_features_target
from splitting import create_holdout_split
from run_ae_augmentation_experiment import evaluate, latent_smote_synthesis, numeric_columns, train_lgbm
from run_fair_augmentation_comparison import assemble_synthetic_df
from utils import ensure_dir, log, save_json, set_seed

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc


class VAE(keras.Model):
    """Minimal VAE with Gaussian encoder/decoder and KL regularisation."""

    def __init__(self, input_dim: int, latent_dim: int, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.encoder = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
        ])
        self.z_mean = keras.layers.Dense(latent_dim)
        self.z_log_var = keras.layers.Dense(latent_dim)
        self.decoder = keras.Sequential([
            keras.layers.Input(shape=(latent_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ])
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def encode(self, x):
        h = self.encoder(x)
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(x)
            z = self.reparameterize(z_mean, z_log_var)
            recon = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - recon), axis=1))
            kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            loss = recon_loss + self.beta * kl
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss)
        return {"loss": self.total_loss_tracker.result()}


def paired_bootstrap_ap_delta(y_true, ref_score, cand_score, n_bootstrap=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = int(y_true.shape[0]); deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n); sy = y_true[idx]
        if sy.min() == sy.max():
            continue
        deltas.append(average_precision_score(sy, cand_score[idx]) - average_precision_score(sy, ref_score[idx]))
    d = np.asarray(deltas, "float64")
    return {
        "reference_ap": float(average_precision_score(y_true, ref_score)),
        "candidate_ap": float(average_precision_score(y_true, cand_score)),
        "observed_delta_ap": float(average_precision_score(y_true, cand_score) - average_precision_score(y_true, ref_score)),
        "ci_2_5": float(np.percentile(d, 2.5)), "ci_50": float(np.percentile(d, 50)),
        "ci_97_5": float(np.percentile(d, 97.5)), "p_delta_le_0": float(np.mean(d <= 0.0)), "n_bootstrap": int(d.shape[0]),
    }


def main(output_dir: Path, split_strategy: str = DEFAULT_SPLIT_STRATEGY, latent_dim: int = 16,
         vae_epochs: int = 100, beta: float = 1.0, k_neighbors: int = 5,
         target_fraud_rate: float = 0.15, n_bootstrap: int = 2000, seed: int = RANDOM_SEED,
         n_estimators: int | None = 800) -> dict:
    set_seed(seed); tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    output_dir = ensure_dir(output_dir)

    log("Loading data and splitting.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df; gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    pre = fit_baseline_preprocessing(X_train_raw)
    Xb_train = apply_baseline_preprocessing(X_train_raw, pre)
    Xb_valid = apply_baseline_preprocessing(X_valid_raw, pre)
    Xb_test = apply_baseline_preprocessing(X_test_raw, pre)
    cat_cols = pre["categorical_columns"]; num_cols = numeric_columns(Xb_train, cat_cols)
    y_tr, y_va, y_te = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()

    bm, bi = train_lgbm(Xb_train, y_tr, Xb_valid, y_va, cat_cols, n_estimators=n_estimators)
    base_test = bm.predict_proba(Xb_test, num_iteration=bi)[:, 1]
    base_m, base_thr = evaluate(y_va, bm.predict_proba(Xb_valid, num_iteration=bi)[:, 1], y_te, base_test)
    log(f"baseline AP={base_m['average_precision']:.6f}")

    imputer = SimpleImputer(strategy="median").fit(Xb_train[num_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[num_cols]))
    fraud_mask = y_tr == 1
    Xn_fraud = np.clip(scaler.transform(imputer.transform(Xb_train.loc[fraud_mask, num_cols])).astype("float32"), -10, 10)
    fraud_cat = Xb_train.loc[fraud_mask, cat_cols].reset_index(drop=True)
    n_fraud, n_normal = int(fraud_mask.sum()), int((~fraud_mask).sum())
    n_synth = max(0, int(round(n_normal / (1.0 - target_fraud_rate) - (n_normal + n_fraud))))

    # smote_nc control
    log("Building smote_nc control.")
    synth_scaled, anchors = latent_smote_synthesis(Xn_fraud, n_synth, k_neighbors, np.random.default_rng(seed))
    smote_df = assemble_synthetic_df(scaler.inverse_transform(synth_scaled), anchors, num_cols, cat_cols, fraud_cat, Xb_train)

    # vae_prior proposed
    log("Training VAE on train fraud.")
    vae = VAE(Xn_fraud.shape[1], latent_dim, beta=beta)
    vae.compile(optimizer=keras.optimizers.Adam(1e-3))
    vae.fit(Xn_fraud, epochs=vae_epochs, batch_size=256, shuffle=True, verbose=0)
    z = rng.normal(size=(n_synth, latent_dim)).astype("float32")
    vae_scaled = vae.decoder.predict(z, batch_size=1024, verbose=0)
    vae_anchors = rng.integers(0, n_fraud, size=n_synth)  # categoricals from random real fraud
    vae_df = assemble_synthetic_df(scaler.inverse_transform(vae_scaled), vae_anchors, num_cols, cat_cols, fraud_cat, Xb_train)

    results = {"baseline": {"test_average_precision": base_m["average_precision"], "test_roc_auc": base_m["roc_auc"],
                            "test_f1": base_m["f1"], "test_mcc": base_m["mcc"], "selected_threshold": base_thr}}
    scores = {"baseline": base_test}
    for name, sdf in (("smote_nc", smote_df), ("vae_prior", vae_df)):
        log(f"Training variant: {name}")
        X_aug = pd.concat([Xb_train, sdf], axis=0, ignore_index=True)
        y_aug = np.concatenate([y_tr, np.ones(n_synth, dtype=int)])
        m, it = train_lgbm(X_aug, y_aug, Xb_valid, y_va, cat_cols, scale_pos_weight=1.0, n_estimators=n_estimators)
        tsc = m.predict_proba(Xb_test, num_iteration=it)[:, 1]
        mm, thr = evaluate(y_va, m.predict_proba(Xb_valid, num_iteration=it)[:, 1], y_te, tsc)
        boot = paired_bootstrap_ap_delta(y_te, base_test, tsc, n_bootstrap=n_bootstrap)
        results[name] = {"test_average_precision": mm["average_precision"], "test_roc_auc": mm["roc_auc"],
                         "test_f1": mm["f1"], "test_mcc": mm["mcc"], "selected_threshold": thr,
                         "bootstrap_vs_baseline": boot}
        scores[name] = tsc
        del X_aug, y_aug; gc.collect()
        log(f"{name}: AP={mm['average_precision']:.6f} d_vs_base={boot['observed_delta_ap']:+.6f} p={boot['p_delta_le_0']:.3f}")

    vae_vs_smote = paired_bootstrap_ap_delta(y_te, scores["smote_nc"], scores["vae_prior"], n_bootstrap=n_bootstrap)
    log(f"vae_prior vs smote_nc: delta={vae_vs_smote['observed_delta_ap']:+.6f} ci=[{vae_vs_smote['ci_2_5']:+.5f},{vae_vs_smote['ci_97_5']:+.5f}] p(d<=0)={vae_vs_smote['p_delta_le_0']:.3f}")

    summary = {"split_strategy": split_strategy, "seed": seed, "n_estimators": n_estimators, "beta": beta,
               "target_fraud_rate": target_fraud_rate, "n_synthetic_per_method": n_synth, "latent_dim": latent_dim,
               "test_prevalence": float(y_te.mean()), "results": results,
               "ae_specific_contribution": {"vae_vs_smote_nc": vae_vs_smote}}
    save_json(summary, output_dir / "experiment_summary.json")

    print("\nVAE Augmentation Experiment")
    print("===========================")
    print(f"baseline AP={results['baseline']['test_average_precision']:.6f}")
    for name in ("smote_nc", "vae_prior"):
        r = results[name]; b = r["bootstrap_vs_baseline"]
        print(f"{name:12s} AP={r['test_average_precision']:.6f} d_vs_base={b['observed_delta_ap']:+.6f} p={b['p_delta_le_0']:.3f}")
    print(f"vae_prior vs smote_nc: delta={vae_vs_smote['observed_delta_ap']:+.6f} ci=[{vae_vs_smote['ci_2_5']:+.5f},{vae_vs_smote['ci_97_5']:+.5f}] p(d<=0)={vae_vs_smote['p_delta_le_0']:.3f}")
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAE generative augmentation experiment.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--vae-epochs", type=int, default=100)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--target-fraud-rate", type=float, default=0.15)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-estimators", type=int, default=800)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir, split_strategy=args.split_strategy, latent_dim=args.latent_dim,
         vae_epochs=args.vae_epochs, beta=args.beta, k_neighbors=args.k_neighbors,
         target_fraud_rate=args.target_fraud_rate, n_bootstrap=args.n_bootstrap, seed=args.seed,
         n_estimators=args.n_estimators)
