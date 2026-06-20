"""Replicate the Ding et al. AEELG anchor on Ding's original datasets.

This script is a sanity-check gate for the thesis method:

    standardize numeric data
    -> split into train/validation/test
    -> apply SMOTE-style interpolation to the train split only
    -> train an AutoEncoder on the balanced train split
    -> reconstruct train/validation/test features
    -> train LightGBM on reconstructed features

Ding et al. (2024) evaluate the method on the ULB European credit-card dataset
and Santander Customer Transaction Prediction. This runner reports the paper's
metrics (ROC-AUC, recall, F1, MCC, BCR) plus Average Precision for thesis
comparability.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from config import RANDOM_SEED
from utils import ensure_dir, log, save_json, set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DING_ROOT = WORKSPACE_ROOT / "Eksperimen Ding et al"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    target_col: str
    id_col: str | None
    default_scaling: str
    paper_reference: str


def dataset_spec(name: str, path: Path | None = None) -> DatasetSpec:
    if name == "ulb":
        return DatasetSpec(
            name="ulb",
            path=path or DEFAULT_DING_ROOT / "ULB Dataset" / "creditcard.csv",
            target_col="Class",
            id_col=None,
            default_scaling="amount_hour",
            paper_reference=(
                "Ding et al. report SMOTE+AEELG AUC 96.83% and F-measure 80.27% "
                "on the 284,807-row ULB credit-card dataset."
            ),
        )
    if name == "santander":
        return DatasetSpec(
            name="santander",
            path=path or DEFAULT_DING_ROOT / "Santander Dataset" / "train.csv",
            target_col="target",
            id_col="ID_code",
            default_scaling="all",
            paper_reference=(
                "Ding et al. report strong Santander AEELG variants, including "
                "AEELG-G/AEELG-S around 89% ROC-AUC and about 53% F-measure."
            ),
        )
    raise ValueError(f"Unsupported dataset: {name}")


def load_anchor_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, np.ndarray, pd.Series | None, dict[str, object]]:
    if not spec.path.exists():
        raise FileNotFoundError(f"Dataset not found: {spec.path}")

    log(f"Loading {spec.name} data from {spec.path}.")
    df = pd.read_csv(spec.path)
    metadata: dict[str, object] = {
        "dataset": spec.name,
        "path": str(spec.path),
        "raw_shape": [int(df.shape[0]), int(df.shape[1])],
    }

    if spec.name == "ulb":
        if "Time" not in df.columns or "Amount" not in df.columns:
            raise ValueError("ULB data must contain Time and Amount columns.")
        df["Hour"] = df["Time"].apply(lambda value: divmod(value, 3600)[0])
        df = df.drop(columns=["Time"])

    if spec.target_col not in df.columns:
        raise ValueError(f"Target column missing: {spec.target_col}")

    ids = df[spec.id_col].copy() if spec.id_col else None
    drop_cols = [spec.target_col]
    if spec.id_col:
        drop_cols.append(spec.id_col)
    y = df[spec.target_col].astype(int).to_numpy()
    X = df.drop(columns=drop_cols)

    non_numeric = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns are not supported here: {non_numeric[:10]}")

    metadata.update(
        {
            "feature_count": int(X.shape[1]),
            "positive_count": int(y.sum()),
            "negative_count": int(len(y) - y.sum()),
            "positive_rate": float(y.mean()),
        }
    )
    return X.astype("float32"), y, ids, metadata


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    ids: pd.Series | None,
    test_size: float,
    valid_size: float,
    seed: int,
) -> dict[str, object]:
    row_ids = pd.Series(np.arange(len(y), dtype=np.int64), name="row_id") if ids is None else ids.reset_index(drop=True)
    idx = np.arange(len(y))
    train_valid_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    y_train_valid = y[train_valid_idx]
    relative_valid_size = valid_size / (1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=relative_valid_size,
        stratify=y_train_valid,
        random_state=seed,
    )

    def take(index: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
        return (
            X.iloc[index].reset_index(drop=True),
            y[index].astype(int),
            row_ids.iloc[index].reset_index(drop=True),
        )

    X_train, y_train, id_train = take(train_idx)
    X_valid, y_valid, id_valid = take(valid_idx)
    X_test, y_test, id_test = take(test_idx)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "id_train": id_train,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "id_valid": id_valid,
        "X_test": X_test,
        "y_test": y_test,
        "id_test": id_test,
        "sizes": {
            "train": int(len(y_train)),
            "validation": int(len(y_valid)),
            "test": int(len(y_test)),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_valid.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
    }


def fit_scaler(X_train: pd.DataFrame, mode: str) -> tuple[StandardScaler | None, list[str]]:
    if mode == "none":
        return None, []
    if mode == "amount_hour":
        columns = [col for col in ("Amount", "Hour") if col in X_train.columns]
    elif mode == "all":
        columns = list(X_train.columns)
    else:
        raise ValueError(f"Unsupported scaling mode: {mode}")
    if not columns:
        return None, []
    scaler = StandardScaler()
    scaler.fit(X_train[columns])
    return scaler, columns


def apply_scaler(X: pd.DataFrame, scaler: StandardScaler | None, columns: list[str]) -> pd.DataFrame:
    out = X.copy()
    if scaler is not None and columns:
        out.loc[:, columns] = scaler.transform(out[columns]).astype("float32")
    return out.astype("float32")


def synthetic_count(n_negative: int, n_positive: int, target_positive_rate: float) -> int:
    if not (0.0 < target_positive_rate < 1.0):
        raise ValueError("--target-fraud-rate must be between 0 and 1.")
    target_total = n_negative / (1.0 - target_positive_rate)
    return max(0, int(round(target_total - n_negative - n_positive)))


def smote_interpolate(
    X_positive: np.ndarray,
    n_synthetic: int,
    k_neighbors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_synthetic <= 0:
        return np.empty((0, X_positive.shape[1]), dtype="float32")
    if X_positive.shape[0] < 2:
        raise ValueError("At least two positive rows are required for SMOTE interpolation.")

    k_eff = min(k_neighbors, X_positive.shape[0] - 1)
    neighbours = NearestNeighbors(n_neighbors=k_eff + 1).fit(X_positive)
    _, neighbour_idx = neighbours.kneighbors(X_positive)
    neighbour_idx = neighbour_idx[:, 1:]

    anchors = rng.integers(0, X_positive.shape[0], size=n_synthetic)
    chosen_neighbour = neighbour_idx[anchors, rng.integers(0, k_eff, size=n_synthetic)]
    gap = rng.random((n_synthetic, 1), dtype="float32")
    synthetic = X_positive[anchors] + gap * (X_positive[chosen_neighbour] - X_positive[anchors])
    return synthetic.astype("float32")


def build_autoencoder(
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    decoder_dim: int,
    output_activation: str,
    learning_rate: float,
    l1_penalty: float,
) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(
        hidden_dim,
        activation="relu",
        activity_regularizer=keras.regularizers.l1(l1_penalty),
        name="encoder_hidden",
    )(inputs)
    latent = keras.layers.Dense(latent_dim, activation="relu", name="bottleneck")(x)
    x = keras.layers.Dense(decoder_dim, activation="relu", name="decoder_hidden")(latent)
    outputs = keras.layers.Dense(input_dim, activation=output_activation, name="reconstruction")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="ding_anchor_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def lgbm_params(preset: str, n_estimators: int, seed: int, n_jobs: int) -> dict[str, object]:
    if preset == "ding":
        return {
            "boosting_type": "goss",
            "objective": "binary",
            "metric": "auc",
            "n_estimators": n_estimators,
            "learning_rate": 0.1,
            "num_leaves": 32,
            "reg_lambda": 10.0,
            "min_child_weight": 1.5,
            "is_unbalance": False,
            "n_jobs": n_jobs,
            "random_state": seed,
            "verbosity": -1,
        }
    if preset == "thesis":
        return {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": n_jobs,
            "random_state": seed,
            "verbosity": -1,
        }
    raise ValueError(f"Unsupported LightGBM preset: {preset}")


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    params: dict[str, object],
    early_stopping_rounds: int,
) -> tuple[lgb.LGBMClassifier, int]:
    model = lgb.LGBMClassifier(**params)
    callbacks: list[object] = [lgb.log_evaluation(period=50)]
    if early_stopping_rounds > 0:
        callbacks.insert(
            0,
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
            ),
        )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        callbacks=callbacks,
    )
    return model, int(model.best_iteration_ or model.n_estimators)


def threshold_table(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = classification_metrics(y_true, y_score, float(threshold))
        rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                "bcr": metrics["bcr"],
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(["mcc", "f1", "bcr", "threshold"], ascending=[False, False, False, True]).index[0]
    table["selected"] = False
    table.loc[best, "selected"] = True
    return table


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "bcr": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix_labels_0_1": cm.astype(int).tolist(),
    }


def score_model(
    arm: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    params: dict[str, object],
    early_stopping_rounds: int,
    output_dir: Path,
) -> dict[str, object]:
    log(f"Training LightGBM arm: {arm}.")
    model, best_iteration = fit_lgbm(
        X_train,
        y_train,
        X_valid,
        y_valid,
        params=params,
        early_stopping_rounds=early_stopping_rounds,
    )
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]
    table = threshold_table(y_valid, valid_score)
    table.to_csv(output_dir / f"thresholds_{arm}.csv", index=False)
    selected_threshold = float(table.loc[table["selected"], "threshold"].iloc[0])
    return {
        "arm": arm,
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "validation_default": classification_metrics(y_valid, valid_score, 0.5),
        "validation_selected": classification_metrics(y_valid, valid_score, selected_threshold),
        "test_default": classification_metrics(y_test, test_score, 0.5),
        "test_selected": classification_metrics(y_test, test_score, selected_threshold),
        "valid_score": valid_score,
        "test_score": test_score,
    }


def paired_bootstrap_ap_delta(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sampled_y = y_true[idx]
        if sampled_y.min() == sampled_y.max():
            continue
        deltas.append(
            average_precision_score(sampled_y, candidate_score[idx])
            - average_precision_score(sampled_y, reference_score[idx])
        )
    values = np.asarray(deltas, dtype="float64")
    observed = float(average_precision_score(y_true, candidate_score) - average_precision_score(y_true, reference_score))
    return {
        "observed_delta_ap": observed,
        "ci_2_5": float(np.percentile(values, 2.5)),
        "ci_50": float(np.percentile(values, 50)),
        "ci_97_5": float(np.percentile(values, 97.5)),
        "p_delta_le_0": float(np.mean(values <= 0.0)),
        "n_bootstrap": int(values.shape[0]),
    }


def reconstruct_frame(model: keras.Model, X: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    values = model.predict(X.to_numpy(dtype="float32"), batch_size=batch_size, verbose=0)
    return pd.DataFrame(values.astype("float32"), columns=X.columns)


def compact_result_row(result: dict[str, object], threshold_key: str = "test_default") -> dict[str, object]:
    metrics = result[threshold_key]
    if not isinstance(metrics, dict):
        raise TypeError("Unexpected metrics payload.")
    return {
        "arm": result["arm"],
        "best_iteration": result["best_iteration"],
        "selected_threshold": result["selected_threshold"],
        "test_average_precision": metrics["average_precision"],
        "test_roc_auc": metrics["roc_auc"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_mcc": metrics["mcc"],
        "test_bcr": metrics["bcr"],
        "test_threshold": metrics["threshold"],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output_dir = ensure_dir(args.output_dir)

    spec = dataset_spec(args.dataset, args.data_path)
    scaling = spec.default_scaling if args.scaling == "dataset_default" else args.scaling
    X, y, ids, metadata = load_anchor_dataset(spec)
    split = split_data(X, y, ids, args.test_size, args.valid_size, args.seed)
    del X
    gc.collect()

    scaler, scaled_columns = fit_scaler(split["X_train"], scaling)
    X_train = apply_scaler(split["X_train"], scaler, scaled_columns)
    X_valid = apply_scaler(split["X_valid"], scaler, scaled_columns)
    X_test = apply_scaler(split["X_test"], scaler, scaled_columns)
    y_train = split["y_train"]
    y_valid = split["y_valid"]
    y_test = split["y_test"]

    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    n_synthetic = synthetic_count(n_negative, n_positive, args.target_fraud_rate)
    log(
        f"Creating train-only SMOTE rows: positives={n_positive}, "
        f"negatives={n_negative}, synthetic={n_synthetic}."
    )
    X_positive = X_train.loc[y_train == 1].to_numpy(dtype="float32")
    X_synthetic_values = smote_interpolate(X_positive, n_synthetic, args.k_neighbors, rng)
    X_synthetic = pd.DataFrame(X_synthetic_values, columns=X_train.columns)
    X_balanced = pd.concat([X_train, X_synthetic], axis=0, ignore_index=True)
    y_balanced = np.concatenate([y_train, np.ones(n_synthetic, dtype=int)])

    params = lgbm_params(args.lgbm_preset, args.n_estimators, args.seed, args.n_jobs)
    if args.lgbm_preset == "thesis":
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        params["scale_pos_weight"] = neg / pos if pos else 1.0

    baseline = score_model(
        "baseline_lgbm",
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        params=params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    smote_params = dict(params)
    smote_params["is_unbalance"] = False
    smote_params["scale_pos_weight"] = 1.0
    smote_only = score_model(
        "smote_lgbm",
        X_balanced,
        y_balanced,
        X_valid,
        y_valid,
        X_test,
        y_test,
        params=smote_params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    log("Training Ding-style AutoEncoder on SMOTE-balanced train features.")
    autoencoder = build_autoencoder(
        input_dim=X_train.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        decoder_dim=args.decoder_dim,
        output_activation=args.output_activation,
        learning_rate=args.learning_rate,
        l1_penalty=args.l1_penalty,
    )
    history = autoencoder.fit(
        X_balanced.to_numpy(dtype="float32"),
        X_balanced.to_numpy(dtype="float32"),
        validation_data=(X_valid.to_numpy(dtype="float32"), X_valid.to_numpy(dtype="float32")),
        epochs=args.ae_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.ae_patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(output_dir / "ae_training_history.csv", index=False)

    log("Reconstructing original and SMOTE-balanced matrices.")
    X_rec_train = reconstruct_frame(autoencoder, X_train, args.batch_size)
    X_rec_valid = reconstruct_frame(autoencoder, X_valid, args.batch_size)
    X_rec_test = reconstruct_frame(autoencoder, X_test, args.batch_size)
    X_rec_balanced = reconstruct_frame(autoencoder, X_balanced, args.batch_size)

    reconstructed_original = score_model(
        "ding_reconstructed_original_train",
        X_rec_train,
        y_train,
        X_rec_valid,
        y_valid,
        X_rec_test,
        y_test,
        params=params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )
    reconstructed_balanced = score_model(
        "ding_reconstructed_balanced_train",
        X_rec_balanced,
        y_balanced,
        X_rec_valid,
        y_valid,
        X_rec_test,
        y_test,
        params=smote_params,
        early_stopping_rounds=args.early_stopping_rounds,
        output_dir=output_dir,
    )

    results = [
        baseline,
        smote_only,
        reconstructed_original,
        reconstructed_balanced,
    ]
    scores = pd.DataFrame(
        {
            "row_id": split["id_test"],
            "target": y_test,
            "score_baseline_lgbm": baseline["test_score"],
            "score_smote_lgbm": smote_only["test_score"],
            "score_ding_reconstructed_original_train": reconstructed_original["test_score"],
            "score_ding_reconstructed_balanced_train": reconstructed_balanced["test_score"],
        }
    )
    scores.to_csv(output_dir / "test_scores.csv", index=False)

    rows = [compact_result_row(result, "test_default") for result in results]
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(output_dir / "metrics_test_default_threshold.csv", index=False)
    selected_rows = [compact_result_row(result, "test_selected") for result in results]
    pd.DataFrame(selected_rows).to_csv(output_dir / "metrics_test_selected_threshold.csv", index=False)

    baseline_score = baseline["test_score"]
    for result in results[1:]:
        result["bootstrap_ap_vs_baseline"] = paired_bootstrap_ap_delta(
            y_test,
            baseline_score,
            result["test_score"],
            args.n_bootstrap,
            args.seed,
        )
    reconstructed_balanced["bootstrap_ap_vs_smote_lgbm"] = paired_bootstrap_ap_delta(
        y_test,
        smote_only["test_score"],
        reconstructed_balanced["test_score"],
        args.n_bootstrap,
        args.seed,
    )

    serializable_results = {}
    for result in results:
        serializable_results[str(result["arm"])] = {
            key: value
            for key, value in result.items()
            if key not in {"valid_score", "test_score"}
        }

    summary = {
        "experiment_id": f"DING-ANCHOR-{spec.name.upper()}",
        "anchor": "Ding et al. (2024) AEELG: SMOTE + AutoEncoder reconstruction + LightGBM",
        "paper_reference": spec.paper_reference,
        "dataset": metadata,
        "split": split["sizes"],
        "config": {
            "seed": args.seed,
            "test_size": args.test_size,
            "valid_size": args.valid_size,
            "scaling": scaling,
            "scaled_columns_count": len(scaled_columns),
            "target_fraud_rate": args.target_fraud_rate,
            "k_neighbors": args.k_neighbors,
            "lgbm_preset": args.lgbm_preset,
            "n_estimators": args.n_estimators,
            "early_stopping_rounds": args.early_stopping_rounds,
            "ae_epochs_requested": args.ae_epochs,
            "ae_epochs_ran": int(len(history.history["loss"])),
            "ae_batch_size": args.batch_size,
            "ae_hidden_dim": args.hidden_dim,
            "ae_latent_dim": args.latent_dim,
            "ae_decoder_dim": args.decoder_dim,
            "ae_output_activation": args.output_activation,
            "ae_learning_rate": args.learning_rate,
            "ae_l1_penalty": args.l1_penalty,
            "best_ae_val_loss": float(np.min(history.history["val_loss"])),
        },
        "smote": {
            "train_positive_before": n_positive,
            "train_negative": n_negative,
            "synthetic_positive": int(n_synthetic),
            "train_positive_after": int(n_positive + n_synthetic),
            "train_rows_after": int(len(y_balanced)),
            "positive_rate_after": float(y_balanced.mean()),
        },
        "results": serializable_results,
    }
    save_json(summary, output_dir / "experiment_summary.json")

    print("\nDing-anchor replication")
    print("=======================")
    print(f"Dataset: {spec.name}")
    print(f"Output:  {output_dir}")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ding et al. AEELG anchor replication.")
    parser.add_argument("--dataset", choices=("ulb", "santander"), required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scaling",
        choices=("dataset_default", "none", "amount_hour", "all"),
        default="dataset_default",
        help="dataset_default uses amount/hour scaling for ULB and all-feature scaling for Santander.",
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.14,
        help="Validation fraction of the full dataset; carved from the non-test split.",
    )
    parser.add_argument("--target-fraud-rate", type=float, default=0.50)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--lgbm-preset", choices=("ding", "thesis"), default="ding")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--ae-epochs", type=int, default=60)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--decoder-dim", type=int, default=16)
    parser.add_argument("--output-activation", choices=("linear", "relu"), default="linear")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1-penalty", type=float, default=1e-4)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
