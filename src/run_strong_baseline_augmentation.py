"""Does augmentation help on top of the STRONGEST baseline, or only rescue A0?

The augmentation win so far is measured against the A0 raw-feature control. The
decisive defensibility question is whether augmentation still helps when the
baseline already uses strong frequency-encoding preprocessing (Alharbi-style A1),
which is much stronger than A0. If a plain frequency-encoded LightGBM already
captures what augmentation provides, the thesis "win" against the best plain
baseline disappears.

A1 preprocessing transforms every feature to numeric (categorical frequency
encoding + median imputation + z-score), so the augmentation works directly in
the dense A1 feature space (interpolation for SMOTE-NC, AE latent for the AE
variant). Same fairness discipline: train-only fitting, test/validation never
augmented, identical LightGBM budget, scale_pos_weight=1.0 for augmenters.

Run:
    python src/run_strong_baseline_augmentation.py \
        --output-dir outputs/stratified_reset/strong_baseline_augmentation \
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
from sklearn.metrics import average_precision_score

from config import DEFAULT_SPLIT_STRATEGY, ID_COL, RANDOM_SEED, SAMPLE_SIZE, SUPPORTED_SPLIT_STRATEGIES, TARGET_COL
from data_loader import load_labeled_train_data
from paper_preprocessing import apply_alharbi_style_preprocessing, fit_alharbi_style_preprocessing
from preprocessing import split_features_target
from splitting import create_holdout_split, stratified_holdout_split
from run_ae_augmentation_experiment import build_fraud_autoencoder, evaluate, latent_smote_synthesis, train_lgbm
from run_vae_augmentation_experiment import paired_bootstrap_ap_delta
from utils import ensure_dir, log, save_json, set_seed

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc


def main(output_dir: Path, split_strategy: str = DEFAULT_SPLIT_STRATEGY, latent_dim: int = 16,
         ae_epochs: int = 60, k_neighbors: int = 5, target_fraud_rate: float = 0.15,
         n_bootstrap: int = 2000, seed: int = RANDOM_SEED, n_estimators: int | None = 800,
         split_seed: int | None = None) -> dict:
    set_seed(seed); tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)

    log("Loading and splitting.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    if split_seed is not None and split_strategy == "stratified_holdout":
        train_df, valid_df, test_df = stratified_holdout_split(full_df, random_seed=split_seed)
    else:
        train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df; gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)

    log("Fitting Alharbi-style (frequency + median + z-score) preprocessing on train only.")
    pre = fit_alharbi_style_preprocessing(X_train_raw)
    Xa_train = apply_alharbi_style_preprocessing(X_train_raw, pre).astype("float32")
    Xa_valid = apply_alharbi_style_preprocessing(X_valid_raw, pre).astype("float32")
    Xa_test = apply_alharbi_style_preprocessing(X_test_raw, pre).astype("float32")
    cat_cols: list[str] = []  # all features numeric after A1 transform
    y_tr, y_va, y_te = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()

    # strong baseline
    log("Training strong (A1) baseline.")
    bm, bi = train_lgbm(Xa_train, y_tr, Xa_valid, y_va, cat_cols, n_estimators=n_estimators)
    base_test = bm.predict_proba(Xa_test, num_iteration=bi)[:, 1]
    base_m, base_thr = evaluate(y_va, bm.predict_proba(Xa_valid, num_iteration=bi)[:, 1], y_te, base_test)
    log(f"strong baseline AP={base_m['average_precision']:.6f}")

    fraud_mask = y_tr == 1
    Xf = np.clip(Xa_train.loc[fraud_mask].to_numpy(dtype="float32"), -10, 10)
    n_fraud, n_normal = int(fraud_mask.sum()), int((~fraud_mask).sum())
    n_synth = max(0, int(round(n_normal / (1.0 - target_fraud_rate) - (n_normal + n_fraud))))
    cols = Xa_train.columns

    # smote_nc (in dense A1 space = plain SMOTE interpolation)
    log("Building SMOTE augmentation in A1 space.")
    smote_pts, _ = latent_smote_synthesis(Xf, n_synth, k_neighbors, np.random.default_rng(seed))
    smote_df = pd.DataFrame(smote_pts, columns=cols)

    # AE latent-SMOTE in A1 space
    log("Training AE on A1 fraud and building AE augmentation.")
    ae, enc, dec = build_fraud_autoencoder(Xf.shape[1], latent_dim)
    ae.fit(Xf, Xf, validation_split=0.1, epochs=ae_epochs, batch_size=256, shuffle=True,
           callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)], verbose=0)
    latent = enc.predict(Xf, batch_size=1024, verbose=0)
    ae_latent_pts, _ = latent_smote_synthesis(latent, n_synth, k_neighbors, np.random.default_rng(seed))
    ae_df = pd.DataFrame(dec.predict(ae_latent_pts, batch_size=1024, verbose=0), columns=cols)

    results = {"baseline": {"test_average_precision": base_m["average_precision"], "test_roc_auc": base_m["roc_auc"],
                            "test_f1": base_m["f1"], "test_mcc": base_m["mcc"], "selected_threshold": base_thr}}
    scores = {"baseline": base_test}
    for name, sdf in (("smote_nc", smote_df), ("ae_latent_smote", ae_df)):
        log(f"Training A1 + {name}.")
        X_aug = pd.concat([Xa_train, sdf], axis=0, ignore_index=True)
        y_aug = np.concatenate([y_tr, np.ones(n_synth, dtype=int)])
        m, it = train_lgbm(X_aug, y_aug, Xa_valid, y_va, cat_cols, scale_pos_weight=1.0, n_estimators=n_estimators)
        tsc = m.predict_proba(Xa_test, num_iteration=it)[:, 1]
        mm, thr = evaluate(y_va, m.predict_proba(Xa_valid, num_iteration=it)[:, 1], y_te, tsc)
        boot = paired_bootstrap_ap_delta(y_te, base_test, tsc, n_bootstrap=n_bootstrap)
        results[name] = {"test_average_precision": mm["average_precision"], "test_roc_auc": mm["roc_auc"],
                         "test_f1": mm["f1"], "test_mcc": mm["mcc"], "selected_threshold": thr,
                         "bootstrap_vs_strong_baseline": boot}
        scores[name] = tsc
        del X_aug, y_aug; gc.collect()
        log(f"{name}: AP={mm['average_precision']:.6f} d_vs_strong_base={boot['observed_delta_ap']:+.6f} p={boot['p_delta_le_0']:.3f}")

    ae_vs_smote = paired_bootstrap_ap_delta(y_te, scores["smote_nc"], scores["ae_latent_smote"], n_bootstrap=n_bootstrap)
    summary = {"split_strategy": split_strategy, "baseline": "alharbi_style_frequency_encoded", "seed": seed,
               "n_estimators": n_estimators, "target_fraud_rate": target_fraud_rate, "n_synthetic": n_synth,
               "test_prevalence": float(y_te.mean()), "results": results,
               "ae_vs_smote_nc": ae_vs_smote}
    save_json(summary, output_dir / "experiment_summary.json")

    print("\nStrong-Baseline Augmentation (A1 frequency-encoded)")
    print("===================================================")
    print(f"strong baseline AP={results['baseline']['test_average_precision']:.6f}")
    for name in ("smote_nc", "ae_latent_smote"):
        r = results[name]; b = r["bootstrap_vs_strong_baseline"]
        verdict = "helps" if b["ci_2_5"] > 0 else ("tie" if b["ci_97_5"] > 0 else "hurts")
        print(f"{name:16s} AP={r['test_average_precision']:.6f} d_vs_strong=%+.6f ci=[%+.5f,%+.5f] p=%.3f -> %s" % (
            b["observed_delta_ap"], b["ci_2_5"], b["ci_97_5"], b["p_delta_le_0"], verdict))
    print(f"ae vs smote_nc: delta={ae_vs_smote['observed_delta_ap']:+.6f} ci=[{ae_vs_smote['ci_2_5']:+.5f},{ae_vs_smote['ci_97_5']:+.5f}] p(d<=0)={ae_vs_smote['p_delta_le_0']:.3f}")
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmentation on the strongest (A1) baseline.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--ae-epochs", type=int, default=60)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--target-fraud-rate", type=float, default=0.15)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--split-seed", type=int, default=None, help="If set (stratified), re-split with this seed for split-variance robustness.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(output_dir=args.output_dir, split_strategy=args.split_strategy, latent_dim=args.latent_dim,
         ae_epochs=args.ae_epochs, k_neighbors=args.k_neighbors, target_fraud_rate=args.target_fraud_rate,
         n_bootstrap=args.n_bootstrap, seed=args.seed, n_estimators=args.n_estimators, split_seed=args.split_seed)
