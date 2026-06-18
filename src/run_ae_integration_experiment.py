"""Controlled AE-integration experiment harness.

Trains one A0-style LightGBM baseline on the active stratified split, then
evaluates several Autoencoder-integration variants against that SAME baseline on
the SAME split with paired-bootstrap AP deltas. All variants share the identical
base feature set, split, and LightGBM budget, so the only thing that changes is
the AE augmentation. This isolates the AE contribution.

Variants:
- baseline                : A0 original features (reference)
- recon_global            : baseline + global reconstruction error (mse, log1p)
- recon_grouped           : baseline + global + per-group reconstruction features
- latent                  : baseline + AE latent features
- recon_grouped_plus_latent : baseline + grouped recon + latent
- score_ensemble          : blend of baseline prob and AE anomaly score (alpha tuned on validation)

Run from repo root, e.g.:
    python src/run_ae_integration_experiment.py \
        --ae-dir outputs/stratified_reset/normal_masked_ae_ld32 \
        --output-dir outputs/stratified_reset/ae_integration_experiment_normal_ld32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install requirements then rerun."
    ) from exc

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
    get_v_feature_columns,
    split_features_target,
)
from splitting import create_holdout_split
from train_ae_lgbm import validate_latent_split_manifest_alignment
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
)
from utils import ensure_dir, log, save_json, set_seed


def load_recon_features(ae_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for split, fname in (
        ("train", "reconstruction_features_train.csv"),
        ("validation", "reconstruction_features_valid.csv"),
        ("test", "reconstruction_features_test.csv"),
    ):
        path = ae_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing recon features: {path}")
        out[split] = pd.read_csv(path)
    return out


def load_recon_error(ae_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for split, fname in (
        ("train", "reconstruction_error_train.csv"),
        ("validation", "reconstruction_error_valid.csv"),
        ("test", "reconstruction_error_test.csv"),
    ):
        path = ae_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing recon error: {path}")
        out[split] = pd.read_csv(path)["reconstruction_mse"].to_numpy(dtype="float64")
    return out


def load_latent(ae_dir: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    names_path = ae_dir / "latent_feature_names.json"
    with names_path.open("r", encoding="utf-8") as f:
        names = json.load(f)
    out = {
        "train": np.load(ae_dir / "latent_train.npy"),
        "validation": np.load(ae_dir / "latent_valid.npy"),
        "test": np.load(ae_dir / "latent_test.npy"),
    }
    return out, names


def concat_features(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1
    )


def train_lgbm(X_train, y_train, X_valid, y_valid, categorical_columns):
    params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=categorical_columns,
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True),
        ],
    )
    best_iter = int(model.best_iteration_ or model.n_estimators)
    return model, best_iter


def score_variant(model, best_iter, X_valid, X_test):
    valid_score = model.predict_proba(X_valid, num_iteration=best_iter)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
    return valid_score, test_score


def evaluate(y_valid, valid_score, y_test, test_score):
    table = threshold_selection_table(y_valid, valid_score)
    thr = selected_threshold_from_table(table)
    m = binary_classification_metrics(y_test, test_score, thr)
    return m, thr


def paired_bootstrap_ap_delta(y_true, ref_score, cand_score, n_bootstrap=2000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = int(y_true.shape[0])
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sy = y_true[idx]
        if sy.min() == sy.max():
            continue
        ref_ap = average_precision_score(sy, ref_score[idx])
        cand_ap = average_precision_score(sy, cand_score[idx])
        deltas.append(cand_ap - ref_ap)
    d = np.asarray(deltas, dtype="float64")
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


def zscore_fit(x: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x)) or 1.0
    return mu, sd


def main(
    ae_dir: Path,
    output_dir: Path,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    n_bootstrap: int = 2000,
) -> dict:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    log(f"Creating {split_strategy} split.")
    train_df, valid_df, test_df = create_holdout_split(full_df, split_strategy=split_strategy)
    del full_df  # free ~3.5GB; stratified split returns independent copies
    import gc

    gc.collect()

    log("Validating AE row-order alignment to current split.")
    validate_latent_split_manifest_alignment(ae_dir, train_df, valid_df, test_df)

    v_columns = get_v_feature_columns(train_df)

    log("Building A0 baseline feature matrices (train-only fit).")
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

    log("Loading AE artifacts.")
    recon_feats = load_recon_features(ae_dir)
    recon_err = load_recon_error(ae_dir)
    latent, latent_names = load_latent(ae_dir)

    # global recon columns = global mse + log1p (the columns without a vXXX_vYYY group token)
    group_token = lambda c: any(tok in c for tok in ["v001", "v096", "v138", "v167", "v217", "v279"])
    global_recon_cols = [c for c in recon_feats["train"].columns if c.endswith(("_mse", "_log1p_mse")) and not group_token(c)]

    def recon_global_df(split):
        return recon_feats[split][global_recon_cols].reset_index(drop=True)

    def recon_grouped_df(split):
        return recon_feats[split].reset_index(drop=True)

    def latent_df(split):
        return pd.DataFrame(latent[split], columns=latent_names)

    variants_specs = {
        "recon_global": lambda s: recon_global_df(s),
        "recon_grouped": lambda s: recon_grouped_df(s),
        "latent": lambda s: latent_df(s),
        "recon_grouped_plus_latent": lambda s: pd.concat(
            [recon_grouped_df(s), latent_df(s)], axis=1
        ),
    }

    results = {}
    scores_store = {}

    # ---- Baseline ----
    log("Training BASELINE LightGBM.")
    base_model, base_iter = train_lgbm(Xb_train, y_train_np, Xb_valid, y_valid_np, categorical_columns)
    base_valid_score, base_test_score = score_variant(base_model, base_iter, Xb_valid, Xb_test)
    base_metrics, base_thr = evaluate(y_valid_np, base_valid_score, y_test_np, base_test_score)
    results["baseline"] = {
        "test_average_precision": base_metrics["average_precision"],
        "test_roc_auc": base_metrics["roc_auc"],
        "test_f1": base_metrics["f1"],
        "test_mcc": base_metrics["mcc"],
        "selected_threshold": base_thr,
        "best_iteration": base_iter,
        "n_features": int(Xb_train.shape[1]),
    }
    scores_store["baseline"] = base_test_score
    log(f"BASELINE test AP={base_metrics['average_precision']:.6f} ROC={base_metrics['roc_auc']:.6f}")

    # ---- Feature-augmentation variants ----
    for name, builder in variants_specs.items():
        log(f"Training variant: {name}")
        Xv_train = concat_features(Xb_train, builder("train"))
        Xv_valid = concat_features(Xb_valid, builder("validation"))
        Xv_test = concat_features(Xb_test, builder("test"))
        model, it = train_lgbm(Xv_train, y_train_np, Xv_valid, y_valid_np, categorical_columns)
        vsc, tsc = score_variant(model, it, Xv_valid, Xv_test)
        m, thr = evaluate(y_valid_np, vsc, y_test_np, tsc)
        boot = paired_bootstrap_ap_delta(y_test_np, base_test_score, tsc, n_bootstrap=n_bootstrap)
        results[name] = {
            "test_average_precision": m["average_precision"],
            "test_roc_auc": m["roc_auc"],
            "test_f1": m["f1"],
            "test_mcc": m["mcc"],
            "selected_threshold": thr,
            "best_iteration": it,
            "n_features": int(Xv_train.shape[1]),
            "bootstrap_vs_baseline": boot,
        }
        scores_store[name] = tsc
        log(
            f"{name}: test AP={m['average_precision']:.6f} "
            f"delta={boot['observed_delta_ap']:+.6f} p(delta<=0)={boot['p_delta_le_0']:.3f}"
        )

    # ---- Score ensemble (baseline prob + AE anomaly score) ----
    log("Building score ensemble (validation-tuned alpha).")
    bmu, bsd = zscore_fit(base_valid_score)
    amu, asd = zscore_fit(recon_err["validation"])
    base_valid_z = (base_valid_score - bmu) / bsd
    base_test_z = (base_test_score - bmu) / bsd
    ae_valid_z = (recon_err["validation"] - amu) / asd
    ae_test_z = (recon_err["test"] - amu) / asd
    best_alpha, best_valid_ap = 1.0, -1.0
    for alpha in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        blend_valid = alpha * base_valid_z + (1.0 - alpha) * ae_valid_z
        ap = average_precision_score(y_valid_np, blend_valid)
        if ap > best_valid_ap:
            best_valid_ap, best_alpha = ap, float(alpha)
    blend_test = best_alpha * base_test_z + (1.0 - best_alpha) * ae_test_z
    ens_boot = paired_bootstrap_ap_delta(y_test_np, base_test_score, blend_test, n_bootstrap=n_bootstrap)
    results["score_ensemble"] = {
        "test_average_precision": float(average_precision_score(y_test_np, blend_test)),
        "test_roc_auc": float(roc_auc_score(y_test_np, blend_test)),
        "best_alpha_on_validation": best_alpha,
        "validation_ap_at_best_alpha": float(best_valid_ap),
        "bootstrap_vs_baseline": ens_boot,
    }
    scores_store["score_ensemble"] = blend_test
    log(
        f"score_ensemble: alpha={best_alpha} test AP="
        f"{results['score_ensemble']['test_average_precision']:.6f} "
        f"delta={ens_boot['observed_delta_ap']:+.6f} p(delta<=0)={ens_boot['p_delta_le_0']:.3f}"
    )

    # ---- Standalone AE anomaly score reference ----
    results["ae_anomaly_score_standalone"] = {
        "test_average_precision": float(average_precision_score(y_test_np, recon_err["test"])),
        "test_roc_auc": float(roc_auc_score(y_test_np, recon_err["test"])),
    }

    summary = {
        "ae_dir": str(ae_dir),
        "split_strategy": split_strategy,
        "n_bootstrap": n_bootstrap,
        "test_prevalence": float(y_test_np.mean()),
        "results": results,
    }
    save_json(summary, output_dir / "experiment_summary.json")

    # per-row test scores for later reuse
    scores_df = pd.DataFrame({ID_COL: test_df[ID_COL].to_numpy(), TARGET_COL: y_test_np})
    for name, sc in scores_store.items():
        scores_df[f"score_{name}"] = sc
    scores_df.to_csv(output_dir / "test_scores.csv", index=False)
    joblib.dump(pre, output_dir / "baseline_preprocessing.pkl")

    # ---- Console summary ----
    print()
    print("AE Integration Experiment Summary")
    print("=================================")
    print(f"AE dir       : {ae_dir}")
    print(f"Baseline AP  : {results['baseline']['test_average_precision']:.6f}")
    print(f"AE anomaly standalone AP: {results['ae_anomaly_score_standalone']['test_average_precision']:.6f}")
    print()
    print(f"{'variant':32s} {'test_AP':>10s} {'deltaAP':>10s} {'p(d<=0)':>9s} {'roc_auc':>9s}")
    for name, r in results.items():
        if name in ("baseline", "ae_anomaly_score_standalone"):
            ap = r["test_average_precision"]
            print(f"{name:32s} {ap:10.6f} {'':>10s} {'':>9s} {r.get('test_roc_auc', float('nan')):9.5f}")
    for name, r in results.items():
        if "bootstrap_vs_baseline" in r:
            b = r["bootstrap_vs_baseline"]
            print(
                f"{name:32s} {r['test_average_precision']:10.6f} "
                f"{b['observed_delta_ap']:+10.6f} {b['p_delta_le_0']:9.3f} "
                f"{r.get('test_roc_auc', float('nan')):9.5f}"
            )
    print(f"\nSaved: {output_dir / 'experiment_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled AE-integration experiment harness.")
    p.add_argument("--ae-dir", type=Path, required=True, help="Autoencoder output dir.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-strategy", choices=SUPPORTED_SPLIT_STRATEGIES, default=DEFAULT_SPLIT_STRATEGY)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        ae_dir=args.ae_dir,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        n_bootstrap=args.n_bootstrap,
    )
