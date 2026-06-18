"""Fair augmentation comparison: isolate the Autoencoder's specific contribution.

The Option B win (AE latent-SMOTE oversampling) changed TWO things at once versus
the baseline: it added oversampling AND it used the AE. This script controls for
that by holding everything fixed except the oversampling mechanism, so the only
variable that distinguishes the proposed method from the controls is the AE.

All oversampling variants use the SAME target fraud rate, the SAME LightGBM
budget, and scale_pos_weight=1.0 (training already rebalanced). The baseline uses
its proper imbalance handling (scale_pos_weight from the data). Validation and
test are never augmented; synthesis uses train fraud only (no leakage).

Variants:
- baseline        : LightGBM, no oversampling, scale_pos_weight from data
- random_oversample : duplicate real train-fraud rows (keep their NaN/categoricals)
- smote_nc        : SMOTE-NC style interpolation in raw imputed-scaled numeric space
- ae_latent_smote : interpolation in the AE latent space then decode (proposed)

Key fairness comparisons (paired bootstrap on test AP):
- ae_latent_smote vs baseline       -> proposed vs baseline
- ae_latent_smote vs smote_nc       -> AE-specific contribution (same oversampling, only the space differs)
- ae_latent_smote vs random_oversample

Run from repo root:
    python src/run_fair_augmentation_comparison.py \
        --output-dir outputs/stratified_reset/fair_augmentation_comparison \
        --target-fraud-rate 0.15
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

from config import (
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
    TARGET_COL,
)
from data_loader import load_labeled_train_data
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    split_features_target,
)
from splitting import create_holdout_split
from run_ae_augmentation_experiment import (
    build_fraud_autoencoder,
    evaluate,
    latent_smote_synthesis,
    numeric_columns,
    paired_bootstrap_ap_delta,
    train_lgbm,
)
from utils import ensure_dir, log, save_json, set_seed

try:
    from tensorflow import keras
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc


def assemble_synthetic_df(
    synth_numeric_raw: np.ndarray,
    anchors: np.ndarray,
    num_cols: list[str],
    categorical_columns: list[str],
    fraud_cat: pd.DataFrame,
    template: pd.DataFrame,
) -> pd.DataFrame:
    """Build synthetic rows in template column space (numeric generated, categoricals from anchor)."""
    synth_numeric_df = pd.DataFrame(synth_numeric_raw, columns=num_cols)
    synth_cat_df = fraud_cat.iloc[anchors].reset_index(drop=True)
    synth_df = pd.concat([synth_numeric_df, synth_cat_df], axis=1)[template.columns]
    return synth_df.astype(template.dtypes.to_dict())


def main(
    output_dir: Path,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    latent_dim: int = 16,
    ae_epochs: int = 60,
    k_neighbors: int = 5,
    target_fraud_rate: float = 0.15,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_SEED,
    n_estimators: int | None = None,
) -> dict:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)

    log("Loading data and splitting.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    pre = fit_baseline_preprocessing(X_train_raw)
    Xb_train = apply_baseline_preprocessing(X_train_raw, pre)
    Xb_valid = apply_baseline_preprocessing(X_valid_raw, pre)
    Xb_test = apply_baseline_preprocessing(X_test_raw, pre)
    categorical_columns = pre["categorical_columns"]
    num_cols = numeric_columns(Xb_train, categorical_columns)
    y_train_np = y_train.to_numpy()
    y_valid_np = y_valid.to_numpy()
    y_test_np = y_test.to_numpy()

    # ----- Baseline (no oversampling, proper imbalance handling) -----
    log("Training baseline LightGBM (scale_pos_weight from data).")
    base_model, base_iter = train_lgbm(
        Xb_train, y_train_np, Xb_valid, y_valid_np, categorical_columns, n_estimators=n_estimators
    )
    base_valid = base_model.predict_proba(Xb_valid, num_iteration=base_iter)[:, 1]
    base_test = base_model.predict_proba(Xb_test, num_iteration=base_iter)[:, 1]
    base_metrics, base_thr = evaluate(y_valid_np, base_valid, y_test_np, base_test)
    log(f"BASELINE test AP={base_metrics['average_precision']:.6f}")

    # ----- Shared imputation/scaling for synthetic numeric space -----
    imputer = SimpleImputer(strategy="median").fit(Xb_train[num_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[num_cols]))
    fraud_mask = y_train_np == 1
    Xn_fraud = np.clip(
        scaler.transform(imputer.transform(Xb_train.loc[fraud_mask, num_cols])).astype("float32"),
        -10.0, 10.0,
    )
    fraud_cat = Xb_train.loc[fraud_mask, categorical_columns].reset_index(drop=True)
    fraud_full = Xb_train.loc[fraud_mask].reset_index(drop=True)

    n_fraud = int(fraud_mask.sum())
    n_normal = int((~fraud_mask).sum())
    n_synth = max(0, int(round(n_normal / (1.0 - target_fraud_rate) - (n_normal + n_fraud))))
    log(f"n_synth={n_synth} per method (fraud={n_fraud}, normal={n_normal}).")

    # ----- Train fraud autoencoder once (for ae_latent_smote) -----
    log("Training fraud autoencoder.")
    ae, enc, dec = build_fraud_autoencoder(Xn_fraud.shape[1], latent_dim)
    ae.fit(
        Xn_fraud, Xn_fraud, validation_split=0.1, epochs=ae_epochs, batch_size=256, shuffle=True,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
        verbose=2,
    )
    latent_fraud = enc.predict(Xn_fraud, batch_size=1024, verbose=0)

    # ----- Build synthetic sets for each method -----
    def synth_random(rng):
        anchors = rng.integers(0, n_fraud, size=n_synth)
        return fraud_full.iloc[anchors].reset_index(drop=True).astype(Xb_train.dtypes.to_dict())

    def synth_smote_nc(rng):
        synth_scaled, anchors = latent_smote_synthesis(Xn_fraud, n_synth, k_neighbors, rng)
        raw = scaler.inverse_transform(synth_scaled)
        return assemble_synthetic_df(raw, anchors, num_cols, categorical_columns, fraud_cat, Xb_train)

    def synth_ae(rng):
        synth_latent, anchors = latent_smote_synthesis(latent_fraud, n_synth, k_neighbors, rng)
        scaled = dec.predict(synth_latent, batch_size=1024, verbose=0)
        raw = scaler.inverse_transform(scaled)
        return assemble_synthetic_df(raw, anchors, num_cols, categorical_columns, fraud_cat, Xb_train)

    methods = {
        "random_oversample": synth_random,
        "smote_nc": synth_smote_nc,
        "ae_latent_smote": synth_ae,
    }

    results = {
        "baseline": {
            "test_average_precision": base_metrics["average_precision"],
            "test_roc_auc": base_metrics["roc_auc"],
            "test_f1": base_metrics["f1"],
            "test_mcc": base_metrics["mcc"],
            "selected_threshold": base_thr,
        }
    }
    scores_store = {"baseline": base_test}

    for name, synth_fn in methods.items():
        log(f"Building + training variant: {name}")
        synth_df = synth_fn(np.random.default_rng(seed))
        X_aug = pd.concat([Xb_train, synth_df], axis=0, ignore_index=True)
        y_aug = np.concatenate([y_train_np, np.ones(n_synth, dtype=int)])
        model, it = train_lgbm(
            X_aug, y_aug, Xb_valid, y_valid_np, categorical_columns,
            scale_pos_weight=1.0, n_estimators=n_estimators,
        )
        vsc = model.predict_proba(Xb_valid, num_iteration=it)[:, 1]
        tsc = model.predict_proba(Xb_test, num_iteration=it)[:, 1]
        m, thr = evaluate(y_valid_np, vsc, y_test_np, tsc)
        boot = paired_bootstrap_ap_delta(y_test_np, base_test, tsc, n_bootstrap=n_bootstrap)
        results[name] = {
            "test_average_precision": m["average_precision"],
            "test_roc_auc": m["roc_auc"],
            "test_f1": m["f1"],
            "test_mcc": m["mcc"],
            "selected_threshold": thr,
            "best_iteration": it,
            "bootstrap_vs_baseline": boot,
        }
        scores_store[name] = tsc
        log(f"{name}: AP={m['average_precision']:.6f} delta_vs_base={boot['observed_delta_ap']:+.6f} p={boot['p_delta_le_0']:.3f}")

    # ----- AE-specific contribution: ae vs smote_nc and ae vs random -----
    ae_isolation = {}
    for ref in ("smote_nc", "random_oversample"):
        b = paired_bootstrap_ap_delta(y_test_np, scores_store[ref], scores_store["ae_latent_smote"], n_bootstrap=n_bootstrap)
        ae_isolation[f"ae_vs_{ref}"] = b
        log(f"ae_latent_smote vs {ref}: delta={b['observed_delta_ap']:+.6f} ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")

    summary = {
        "split_strategy": split_strategy,
        "seed": seed,
        "n_estimators_override": n_estimators,
        "target_fraud_rate": target_fraud_rate,
        "n_synthetic_per_method": n_synth,
        "latent_dim": latent_dim,
        "k_neighbors": k_neighbors,
        "test_prevalence": float(y_test_np.mean()),
        "results": results,
        "ae_specific_contribution": ae_isolation,
    }
    save_json(summary, output_dir / "experiment_summary.json")
    scores_df = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: y_test_np})
    for name, sc in scores_store.items():
        scores_df[f"score_{name}"] = sc
    scores_df.to_csv(output_dir / "test_scores.csv", index=False)

    print()
    print("Fair Augmentation Comparison")
    print("============================")
    print(f"{'variant':20s} {'test_AP':>10s} {'d_vs_base':>11s} {'p(d<=0)':>9s}")
    print(f"{'baseline':20s} {results['baseline']['test_average_precision']:10.6f}")
    for name in ("random_oversample", "smote_nc", "ae_latent_smote"):
        r = results[name]; b = r["bootstrap_vs_baseline"]
        print(f"{name:20s} {r['test_average_precision']:10.6f} {b['observed_delta_ap']:+11.6f} {b['p_delta_le_0']:9.3f}")
    print("\nAE-specific contribution (proposed minus control):")
    for k, b in ae_isolation.items():
        print(f"  {k:22s} delta={b['observed_delta_ap']:+.6f} ci=[{b['ci_2_5']:+.5f},{b['ci_97_5']:+.5f}] p(d<=0)={b['p_delta_le_0']:.3f}")
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fair augmentation comparison isolating the AE contribution.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--ae-epochs", type=int, default=60)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--target-fraud-rate", type=float, default=0.15)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-estimators", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        latent_dim=args.latent_dim,
        ae_epochs=args.ae_epochs,
        k_neighbors=args.k_neighbors,
        target_fraud_rate=args.target_fraud_rate,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
