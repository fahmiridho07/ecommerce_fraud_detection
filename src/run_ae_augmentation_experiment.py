"""AE-based minority oversampling experiment (Option B).

Autoencoder latent-space oversampling of the fraud class, then LightGBM.
A small denoising autoencoder is trained on TRAIN fraud numeric features only.
Fraud rows are encoded to the latent space; new synthetic fraud samples are
created by SMOTE-style interpolation between a fraud anchor and one of its
latent-space fraud neighbours, then decoded back to numeric feature space.
Categorical values for each synthetic row are copied from the real fraud anchor
(SMOTE-NC style) so categories stay valid. Synthetic fraud rows are appended to
the training split only; validation and test stay untouched.

Anchors: Ding et al. (2024) AE+SMOTE+LightGBM; Alharbi et al. (2026) AE-based
generative augmentation on IEEE-CIS; Kabane & Ouali (2024) train-only resampling
to avoid leakage.

Run from repo root:
    python src/run_ae_augmentation_experiment.py \
        --output-dir outputs/stratified_reset/ae_augmentation_experiment \
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit("LightGBM not installed.") from exc

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover
    raise SystemExit("TensorFlow not installed.") from exc

from config import (
    DEFAULT_SPLIT_STRATEGY,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
    TARGET_COL,
)
from data_loader import load_labeled_train_data
from evaluation import (
    binary_classification_metrics,
    selected_threshold_from_table,
    threshold_selection_table,
)
from preprocessing import (
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    split_features_target,
)
from splitting import create_holdout_split
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
)
from utils import ensure_dir, log, save_json, set_seed


def numeric_columns(X: pd.DataFrame, categorical_columns: list[str]) -> list[str]:
    return [c for c in X.columns if c not in set(categorical_columns)]


def build_fraud_autoencoder(input_dim: int, latent_dim: int, lr: float = 1e-3):
    inp = keras.Input(shape=(input_dim,))
    x = keras.layers.GaussianNoise(0.05)(inp)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    latent = keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)
    x = keras.layers.Dense(64, activation="relu")(latent)
    x = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(input_dim, activation="linear")(x)
    ae = keras.Model(inp, out)
    enc = keras.Model(inp, latent)
    # decoder as standalone for sampling
    latent_in = keras.Input(shape=(latent_dim,))
    d = ae.layers[-3](latent_in)
    d = ae.layers[-2](d)
    d = ae.layers[-1](d)
    dec = keras.Model(latent_in, d)
    ae.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return ae, enc, dec


def latent_smote_synthesis(
    latent_fraud: np.ndarray,
    n_synth: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic latent points and the anchor index used for each."""
    n = latent_fraud.shape[0]
    k_eff = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(latent_fraud)
    _, neigh = nn.kneighbors(latent_fraud)
    neigh = neigh[:, 1:]  # drop self
    anchors = rng.integers(0, n, size=n_synth)
    synth = np.empty((n_synth, latent_fraud.shape[1]), dtype="float32")
    for i, a in enumerate(anchors):
        b = neigh[a, rng.integers(0, k_eff)]
        u = rng.random()
        synth[i] = latent_fraud[a] + u * (latent_fraud[b] - latent_fraud[a])
    return synth, anchors


def train_lgbm(X_train, y_train, X_valid, y_valid, categorical_columns, scale_pos_weight=None, n_estimators=None):
    params = build_model_params(pd.Series(y_train))
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=categorical_columns,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True)],
    )
    return model, int(model.best_iteration_ or model.n_estimators)


def evaluate(y_valid, valid_score, y_test, test_score):
    table = threshold_selection_table(y_valid, valid_score)
    thr = selected_threshold_from_table(table)
    return binary_classification_metrics(y_test, test_score, thr), thr


def paired_bootstrap_ap_delta(y_true, ref_score, cand_score, n_bootstrap=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = int(y_true.shape[0])
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sy = y_true[idx]
        if sy.min() == sy.max():
            continue
        deltas.append(
            average_precision_score(sy, cand_score[idx]) - average_precision_score(sy, ref_score[idx])
        )
    d = np.asarray(deltas, "float64")
    obs_ref = float(average_precision_score(y_true, ref_score))
    obs_cand = float(average_precision_score(y_true, cand_score))
    return {
        "reference_ap": obs_ref,
        "candidate_ap": obs_cand,
        "observed_delta_ap": obs_cand - obs_ref,
        "ci_2_5": float(np.percentile(d, 2.5)),
        "ci_50": float(np.percentile(d, 50)),
        "ci_97_5": float(np.percentile(d, 97.5)),
        "p_delta_le_0": float(np.mean(d <= 0.0)),
        "n_bootstrap": int(d.shape[0]),
    }


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
    rng = np.random.default_rng(seed)
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
    y_train_np = y_train.to_numpy()
    y_valid_np = y_valid.to_numpy()
    y_test_np = y_test.to_numpy()

    num_cols = numeric_columns(Xb_train, categorical_columns)

    # ----- Baseline reference (no augmentation) -----
    log("Training baseline reference LightGBM (no augmentation).")
    base_model, base_iter = train_lgbm(Xb_train, y_train_np, Xb_valid, y_valid_np, categorical_columns, n_estimators=n_estimators)
    base_valid = base_model.predict_proba(Xb_valid, num_iteration=base_iter)[:, 1]
    base_test = base_model.predict_proba(Xb_test, num_iteration=base_iter)[:, 1]
    base_metrics, base_thr = evaluate(y_valid_np, base_valid, y_test_np, base_test)
    log(f"BASELINE test AP={base_metrics['average_precision']:.6f}")

    # ----- Fit AE on TRAIN fraud numeric features -----
    log("Fitting train-only imputer/scaler on numeric features.")
    imputer = SimpleImputer(strategy="median").fit(Xb_train[num_cols])
    scaler = StandardScaler().fit(imputer.transform(Xb_train[num_cols]))

    fraud_mask = y_train_np == 1
    Xn_fraud = scaler.transform(imputer.transform(Xb_train.loc[fraud_mask, num_cols])).astype("float32")
    Xn_fraud = np.clip(Xn_fraud, -10.0, 10.0)
    log(f"Training fraud autoencoder on {Xn_fraud.shape[0]} fraud rows, dim={Xn_fraud.shape[1]}.")
    ae, enc, dec = build_fraud_autoencoder(Xn_fraud.shape[1], latent_dim)
    ae.fit(
        Xn_fraud,
        Xn_fraud,
        validation_split=0.1,
        epochs=ae_epochs,
        batch_size=256,
        shuffle=True,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)],
        verbose=2,
    )

    latent_fraud = enc.predict(Xn_fraud, batch_size=1024, verbose=0)

    n_normal = int((~fraud_mask).sum())
    n_fraud = int(fraud_mask.sum())
    # number of synthetic fraud to reach target_fraud_rate
    n_total_target = n_normal / (1.0 - target_fraud_rate)
    n_synth = max(0, int(round(n_total_target - (n_normal + n_fraud))))
    log(f"Generating {n_synth} synthetic fraud (current fraud={n_fraud}, normal={n_normal}).")

    synth_latent, anchors = latent_smote_synthesis(latent_fraud, n_synth, k_neighbors, rng)
    synth_num_scaled = dec.predict(synth_latent, batch_size=1024, verbose=0)
    synth_num_raw = scaler.inverse_transform(synth_num_scaled)
    synth_numeric_df = pd.DataFrame(synth_num_raw, columns=num_cols)

    # categorical from anchors (SMOTE-NC style)
    fraud_cat = Xb_train.loc[fraud_mask, categorical_columns].reset_index(drop=True)
    synth_cat_df = fraud_cat.iloc[anchors].reset_index(drop=True)

    synth_df = pd.concat([synth_numeric_df, synth_cat_df], axis=1)[Xb_train.columns]
    synth_df = synth_df.astype(Xb_train.dtypes.to_dict())

    Xb_train_aug = pd.concat([Xb_train, synth_df], axis=0, ignore_index=True)
    y_train_aug = np.concatenate([y_train_np, np.ones(n_synth, dtype=int)])

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

    # ----- Augmented variants -----
    for variant, spw in (("augment_scale_pos_weight", None), ("augment_no_spw", 1.0)):
        log(f"Training augmented LightGBM: {variant}")
        model, it = train_lgbm(
            Xb_train_aug, y_train_aug, Xb_valid, y_valid_np, categorical_columns,
            scale_pos_weight=spw, n_estimators=n_estimators,
        )
        vsc = model.predict_proba(Xb_valid, num_iteration=it)[:, 1]
        tsc = model.predict_proba(Xb_test, num_iteration=it)[:, 1]
        m, thr = evaluate(y_valid_np, vsc, y_test_np, tsc)
        boot = paired_bootstrap_ap_delta(y_test_np, base_test, tsc, n_bootstrap=n_bootstrap)
        results[variant] = {
            "test_average_precision": m["average_precision"],
            "test_roc_auc": m["roc_auc"],
            "test_f1": m["f1"],
            "test_mcc": m["mcc"],
            "selected_threshold": thr,
            "best_iteration": it,
            "bootstrap_vs_baseline": boot,
        }
        scores_store[variant] = tsc
        log(
            f"{variant}: test AP={m['average_precision']:.6f} "
            f"delta={boot['observed_delta_ap']:+.6f} p(delta<=0)={boot['p_delta_le_0']:.3f}"
        )

    summary = {
        "split_strategy": split_strategy,
        "seed": seed,
        "n_estimators_override": n_estimators,
        "augmentation": {
            "method": "ae_latent_smote_oversampling",
            "latent_dim": latent_dim,
            "k_neighbors": k_neighbors,
            "target_fraud_rate": target_fraud_rate,
            "n_synthetic_fraud": n_synth,
            "train_fraud_before": n_fraud,
            "train_normal": n_normal,
        },
        "n_bootstrap": n_bootstrap,
        "test_prevalence": float(y_test_np.mean()),
        "results": results,
    }
    save_json(summary, output_dir / "experiment_summary.json")
    scores_df = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: y_test_np})
    for name, sc in scores_store.items():
        scores_df[f"score_{name}"] = sc
    scores_df.to_csv(output_dir / "test_scores.csv", index=False)

    print()
    print("AE Augmentation Experiment Summary")
    print("==================================")
    print(f"Baseline AP: {results['baseline']['test_average_precision']:.6f}")
    for name, r in results.items():
        if "bootstrap_vs_baseline" in r:
            b = r["bootstrap_vs_baseline"]
            print(
                f"{name:28s} AP={r['test_average_precision']:.6f} "
                f"delta={b['observed_delta_ap']:+.6f} p(d<=0)={b['p_delta_le_0']:.3f}"
            )
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE latent-space oversampling experiment.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--ae-epochs", type=int, default=60)
    p.add_argument("--k-neighbors", type=int, default=5)
    p.add_argument("--target-fraud-rate", type=float, default=0.15)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--n-estimators", type=int, default=None, help="Override LightGBM n_estimators (applied to baseline and augmented equally).")
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
