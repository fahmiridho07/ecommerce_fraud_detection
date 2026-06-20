"""Run the original seminar-proposal design under the stated stratified split.

Original proposal contract:
- IEEE-CIS train transaction + identity data.
- Stratified train/validation/test split = 60/20/20, seed 42.
- Baseline LightGBM on preprocessed original features.
- Autoencoder trained only on V1-V339, unsupervised, zero-imputed, z-score scaled.
- Encoder latent representation replaces the original V1-V339 block.
- Non-V features are joined back with the V latent features for LightGBM.
- No resampling of train/validation/test.
- PR-AUC / Average Precision is the primary metric.
- Optional Optuna/TPE tuning is applied fairly to baseline and proposed model.

This runner is intentionally separate from later thesis branches such as
AE latent-SMOTE augmentation, Ding-style reconstruction, and missingness-
preserving robust AE variants. It exists to answer whether the original proposal
design itself has been tested on IEEE-CIS.
"""

from __future__ import annotations

import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("LightGBM is not installed.") from exc

try:
    import optuna
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Optuna is not installed.") from exc

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("TensorFlow is not installed.") from exc

from config import (
    DATA_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from evaluation import (
    binary_classification_metrics,
    confusion_matrix_table,
    selected_threshold_from_table,
    threshold_selection_table,
)
from preprocessing import (
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_VALUE,
    get_v_feature_columns,
    split_features_target,
)
from splitting import stratified_holdout_split
from train_baseline_lgbm import average_precision_eval, roc_auc_eval, save_feature_importance
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_THRESHOLD = 0.5
EARLY_STOPPING_ROUNDS = 100
PROPOSAL_CATEGORICAL_EXACT = {
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
}


@dataclass
class EncodedSplits:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    categorical_columns: list[str]
    preprocessing: dict[str, object]


@dataclass
class ProposalData:
    X_base_train: pd.DataFrame
    X_base_valid: pd.DataFrame
    X_base_test: pd.DataFrame
    X_ae_train: pd.DataFrame
    X_ae_valid: pd.DataFrame
    X_ae_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    baseline_categorical_columns: list[str]
    ae_categorical_columns: list[str]
    v_columns: list[str]
    latent_feature_names: list[str]
    preprocessing_baseline: dict[str, object]
    preprocessing_non_v: dict[str, object]
    ae_preprocessing: dict[str, object]


def proposal_categorical_columns(X: pd.DataFrame) -> list[str]:
    """Return categorical columns according to the original proposal text."""
    categorical = set(X.select_dtypes(include=["object", "category"]).columns)
    categorical |= {column for column in PROPOSAL_CATEGORICAL_EXACT if column in X.columns}
    categorical |= {
        column
        for column in X.columns
        if column.startswith("M") and column[1:].isdigit() and 1 <= int(column[1:]) <= 9
    }
    categorical |= {
        column
        for column in X.columns
        if column.startswith("id_")
        and column[3:].isdigit()
        and 12 <= int(column[3:]) <= 38
    }
    return [column for column in X.columns if column in categorical]


def fit_proposal_encoding(X_train: pd.DataFrame) -> dict[str, object]:
    """Fit train-only ordinal encoding for proposal-defined categorical columns."""
    categorical_columns = proposal_categorical_columns(X_train)
    mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = X_train[column].astype("string").fillna(MISSING_CATEGORY)
        categories = sorted(set(values.tolist()) - {MISSING_CATEGORY})
        ordered = [MISSING_CATEGORY, *categories]
        mappings[column] = {category: index for index, category in enumerate(ordered)}
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_mappings": mappings,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "encoding": "train-fitted ordinal encoding for proposal categorical columns",
    }


def apply_proposal_encoding(X: pd.DataFrame, preprocessing: dict[str, object]) -> pd.DataFrame:
    """Apply train-fitted categorical encoding; leave numeric columns unchanged."""
    transformed = X.loc[:, preprocessing["feature_columns"]].copy()
    mappings: dict[str, dict[str, int]] = preprocessing["categorical_mappings"]  # type: ignore[assignment]
    for column, mapping in mappings.items():
        values = transformed[column].astype("string").fillna(MISSING_CATEGORY)
        transformed[column] = values.map(mapping).fillna(UNKNOWN_CATEGORY_VALUE).astype("int32")
    return transformed


def encode_splits(X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame) -> EncodedSplits:
    preprocessing = fit_proposal_encoding(X_train)
    return EncodedSplits(
        X_train=apply_proposal_encoding(X_train, preprocessing),
        X_valid=apply_proposal_encoding(X_valid, preprocessing),
        X_test=apply_proposal_encoding(X_test, preprocessing),
        categorical_columns=list(preprocessing["categorical_columns"]),
        preprocessing=preprocessing,
    )


def prepare_v_zero_scaled(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Zero-impute V features and fit z-score scaler on train only."""
    Xv_train = X_train.loc[:, v_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xv_valid = X_valid.loc[:, v_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xv_test = X_test.loc[:, v_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    scaler = StandardScaler()
    V_train = scaler.fit_transform(Xv_train).astype("float32")
    V_valid = scaler.transform(Xv_valid).astype("float32")
    V_test = scaler.transform(Xv_test).astype("float32")
    return V_train, V_valid, V_test, scaler


def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
    latent_activation: str,
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(input_dim,), name="v_features")
    x = keras.layers.Dense(256, activation="relu", name="encoder_dense_256")(inputs)
    x = keras.layers.Dense(128, activation="relu", name="encoder_dense_128")(x)
    latent = keras.layers.Dense(latent_dim, activation=latent_activation, name="latent")(x)
    x = keras.layers.Dense(128, activation="relu", name="decoder_dense_128")(latent)
    x = keras.layers.Dense(256, activation="relu", name="decoder_dense_256")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="proposal_v_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="proposal_v_encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return autoencoder, encoder


def train_v_autoencoder(
    V_train: np.ndarray,
    V_valid: np.ndarray,
    latent_dim: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    latent_activation: str,
) -> tuple[keras.Model, keras.Model, pd.DataFrame]:
    autoencoder, encoder = build_autoencoder(
        input_dim=V_train.shape[1],
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        latent_activation=latent_activation,
    )
    history = autoencoder.fit(
        V_train,
        V_train,
        validation_data=(V_valid, V_valid),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    return autoencoder, encoder, history_df


def encode_latents(
    encoder: keras.Model,
    V_train: np.ndarray,
    V_valid: np.ndarray,
    V_test: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        encoder.predict(V_train, batch_size=batch_size, verbose=0).astype("float32"),
        encoder.predict(V_valid, batch_size=batch_size, verbose=0).astype("float32"),
        encoder.predict(V_test, batch_size=batch_size, verbose=0).astype("float32"),
    )


def combine_non_v_and_latent(
    encoded_train: pd.DataFrame,
    encoded_valid: pd.DataFrame,
    encoded_test: pd.DataFrame,
    v_columns: list[str],
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    non_v_columns = [column for column in encoded_train.columns if column not in set(v_columns)]
    latent_names = [f"ae_latent_{index:03d}" for index in range(1, latent_train.shape[1] + 1)]

    def combine(encoded: pd.DataFrame, latent: np.ndarray) -> pd.DataFrame:
        latent_df = pd.DataFrame(latent, columns=latent_names, index=encoded.index)
        return pd.concat([encoded.loc[:, non_v_columns].reset_index(drop=True), latent_df.reset_index(drop=True)], axis=1)

    return (
        combine(encoded_train, latent_train),
        combine(encoded_valid, latent_valid),
        combine(encoded_test, latent_test),
        latent_names,
    )


def default_lgbm_params(y_train: pd.Series, n_jobs: int) -> dict[str, object]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "scale_pos_weight": negatives / max(positives, 1),
        "n_jobs": n_jobs,
        "random_state": RANDOM_SEED,
        "metric": "None",
        "verbosity": -1,
    }


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
    params: dict[str, object],
) -> tuple[lgb.LGBMClassifier, int]:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=[column for column in categorical_columns if column in X_train.columns],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True),
            lgb.log_evaluation(period=100),
        ],
    )
    return model, int(model.best_iteration_ or model.n_estimators)


def evaluate_model(
    model: lgb.LGBMClassifier,
    best_iteration: int,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    name: str,
) -> dict[str, object]:
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]
    threshold_table = threshold_selection_table(y_valid.to_numpy(), valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / f"{name}_threshold_selection.csv", index=False)
    confusion_matrix_table(
        y_test.to_numpy(),
        test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / f"{name}_confusion_matrix_test.csv", index=False)
    pd.DataFrame(
        {
            "isFraud": y_test.to_numpy(dtype=int),
            f"{name}_score": test_score,
        }
    ).to_csv(output_dir / f"{name}_test_scores.csv", index=False)
    return {
        "best_iteration": best_iteration,
        "selected_threshold": selected_threshold,
        "validation_default": binary_classification_metrics(y_valid.to_numpy(), valid_score, DEFAULT_THRESHOLD),
        "validation_selected": binary_classification_metrics(y_valid.to_numpy(), valid_score, selected_threshold),
        "test_default": binary_classification_metrics(y_test.to_numpy(), test_score, DEFAULT_THRESHOLD),
        "test_selected": binary_classification_metrics(y_test.to_numpy(), test_score, selected_threshold),
        "validation_score": valid_score,
        "test_score": test_score,
    }


def tune_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
    n_trials: int,
    n_jobs: int,
    seed: int,
    study_name: str,
    output_dir: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    categorical_feature = [column for column in categorical_columns if column in X_train.columns]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "colsample_bytree": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 30.0),
            "subsample": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "min_child_samples": trial.suggest_int("min_data_in_leaf", 20, 200),
            "n_jobs": n_jobs,
            "random_state": seed,
            "metric": "None",
            "verbosity": -1,
        }
        model, best_iteration = fit_lgbm(
            X_train,
            y_train,
            X_valid,
            y_valid,
            categorical_feature,
            params,
        )
        score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
        return float(average_precision_score(y_valid, score))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    trials = study.trials_dataframe()
    trials.to_csv(output_dir / f"{study_name}_trials.csv", index=False)
    best_params = dict(study.best_params)
    mapped = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": int(best_params["num_leaves"]),
        "max_depth": int(best_params["max_depth"]),
        "learning_rate": float(best_params["learning_rate"]),
        "n_estimators": int(best_params["n_estimators"]),
        "colsample_bytree": float(best_params["feature_fraction"]),
        "scale_pos_weight": float(best_params["scale_pos_weight"]),
        "subsample": float(best_params["bagging_fraction"]),
        "subsample_freq": int(best_params["bagging_freq"]),
        "reg_alpha": float(best_params["lambda_l1"]),
        "reg_lambda": float(best_params["lambda_l2"]),
        "min_child_samples": int(best_params["min_data_in_leaf"]),
        "n_jobs": n_jobs,
        "random_state": seed,
        "metric": "None",
        "verbosity": -1,
    }
    save_json(
        {
            "study_name": study_name,
            "n_trials": n_trials,
            "best_validation_average_precision": float(study.best_value),
            "best_params_raw": best_params,
            "best_params_lightgbm": mapped,
        },
        output_dir / f"{study_name}_best_params.json",
    )
    return mapped, trials


def paired_bootstrap_ap_delta(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    observed = float(
        average_precision_score(y_true, candidate_score)
        - average_precision_score(y_true, reference_score)
    )
    deltas = np.empty(n_bootstrap, dtype="float64")
    for index in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        y_sample = y_true[sample_idx]
        if len(np.unique(y_sample)) < 2:
            deltas[index] = 0.0
            continue
        deltas[index] = (
            average_precision_score(y_sample, candidate_score[sample_idx])
            - average_precision_score(y_sample, reference_score[sample_idx])
        )
    return {
        "observed_delta_ap": observed,
        "ci_2_5": float(np.percentile(deltas, 2.5)),
        "ci_50": float(np.percentile(deltas, 50.0)),
        "ci_97_5": float(np.percentile(deltas, 97.5)),
        "p_delta_le_0": float(np.mean(deltas <= 0.0)),
        "n_bootstrap": int(n_bootstrap),
    }


def prepare_proposal_data(
    output_dir: Path,
    latent_dim: int,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    latent_activation: str,
    seed: int,
) -> ProposalData:
    log("Loading IEEE-CIS labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    log("Creating stratified 60/20/20 split.")
    train_df, valid_df, test_df = stratified_holdout_split(full_df, random_seed=seed)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    v_columns = get_v_feature_columns(X_train_raw)
    if len(v_columns) != 339:
        log(f"Warning: expected 339 V columns, found {len(v_columns)}.")

    log("Fitting proposal categorical encoders on train only.")
    baseline_encoded = encode_splits(X_train_raw, X_valid_raw, X_test_raw)

    log("Preparing proposal V block for Autoencoder: zero-impute then z-score.")
    V_train, V_valid, V_test, scaler = prepare_v_zero_scaled(
        X_train_raw,
        X_valid_raw,
        X_test_raw,
        v_columns,
    )

    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    log("Training V-only proposal Autoencoder.")
    autoencoder, encoder, history_df = train_v_autoencoder(
        V_train=V_train,
        V_valid=V_valid,
        latent_dim=latent_dim,
        max_epochs=ae_max_epochs,
        patience=ae_patience,
        batch_size=ae_batch_size,
        learning_rate=ae_learning_rate,
        latent_activation=latent_activation,
    )
    history_df.to_csv(output_dir / "ae_training_history.csv", index=False)
    autoencoder.save(output_dir / "proposal_v_autoencoder.keras")
    encoder.save(output_dir / "proposal_v_encoder.keras")
    joblib.dump(scaler, output_dir / "v_zero_impute_zscore_scaler.pkl")

    latent_train, latent_valid, latent_test = encode_latents(
        encoder,
        V_train,
        V_valid,
        V_test,
        batch_size=ae_batch_size,
    )
    np.save(output_dir / "latent_train.npy", latent_train)
    np.save(output_dir / "latent_valid.npy", latent_valid)
    np.save(output_dir / "latent_test.npy", latent_test)

    log("Building AE-LightGBM matrices: non-V + V latent replacement.")
    non_v_train_raw = X_train_raw.drop(columns=v_columns)
    non_v_valid_raw = X_valid_raw.drop(columns=v_columns)
    non_v_test_raw = X_test_raw.drop(columns=v_columns)
    non_v_encoded = encode_splits(non_v_train_raw, non_v_valid_raw, non_v_test_raw)
    X_ae_train, X_ae_valid, X_ae_test, latent_names = combine_non_v_and_latent(
        non_v_encoded.X_train,
        non_v_encoded.X_valid,
        non_v_encoded.X_test,
        [],
        latent_train,
        latent_valid,
        latent_test,
    )

    ae_preprocessing = {
        "input_scope": "V1-V339 only",
        "missing_value_strategy": "fill missing V values with 0 before scaling",
        "scaling": "StandardScaler fit on train V block only",
        "loss": "mean_squared_error",
        "optimizer": "Adam",
        "hidden_activation": "relu",
        "latent_activation": latent_activation,
        "latent_dim": latent_dim,
        "validation_source": "validation split",
        "original_v_replaced": True,
        "v_missing_indicators_included": False,
    }
    save_json(
        {
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "train_fraud_rate": float(y_train.mean()),
            "valid_fraud_rate": float(y_valid.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "v_feature_count": len(v_columns),
            "baseline_feature_count": int(baseline_encoded.X_train.shape[1]),
            "ae_feature_count": int(X_ae_train.shape[1]),
            "non_v_feature_count": int(non_v_encoded.X_train.shape[1]),
            "latent_feature_count": len(latent_names),
            "categorical_columns_baseline": baseline_encoded.categorical_columns,
            "categorical_columns_ae": non_v_encoded.categorical_columns,
            "ae_preprocessing": ae_preprocessing,
        },
        output_dir / "data_contract.json",
    )

    return ProposalData(
        X_base_train=baseline_encoded.X_train,
        X_base_valid=baseline_encoded.X_valid,
        X_base_test=baseline_encoded.X_test,
        X_ae_train=X_ae_train,
        X_ae_valid=X_ae_valid,
        X_ae_test=X_ae_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        baseline_categorical_columns=baseline_encoded.categorical_columns,
        ae_categorical_columns=non_v_encoded.categorical_columns,
        v_columns=v_columns,
        latent_feature_names=latent_names,
        preprocessing_baseline=baseline_encoded.preprocessing,
        preprocessing_non_v=non_v_encoded.preprocessing,
        ae_preprocessing=ae_preprocessing,
    )


def train_and_record(
    data: ProposalData,
    output_dir: Path,
    name: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    params: dict[str, object],
) -> dict[str, object]:
    log(f"Training {name}.")
    model, best_iteration = fit_lgbm(
        X_train,
        data.y_train,
        X_valid,
        data.y_valid,
        categorical_columns,
        params,
    )
    model_dir = ensure_dir(output_dir / name)
    joblib.dump(model, model_dir / "model.pkl")
    model.booster_.save_model(str(model_dir / "model.txt"))
    save_feature_importance(model, model_dir / "feature_importance.csv")
    result = evaluate_model(
        model,
        best_iteration,
        X_valid,
        data.y_valid,
        X_test,
        data.y_test,
        model_dir,
        name,
    )
    save_json(
        {
            "best_iteration": result["best_iteration"],
            "selected_threshold": result["selected_threshold"],
            "validation_default": result["validation_default"],
            "validation_selected": result["validation_selected"],
            "test_default": result["test_default"],
            "test_selected": result["test_selected"],
            "model_params": params,
        },
        model_dir / "metrics_and_config.json",
    )
    return result


def main(
    output_dir: Path,
    latent_dim: int,
    latent_activation: str,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    n_trials: int,
    n_bootstrap: int,
    n_jobs: int,
    seed: int,
) -> dict[str, object]:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)

    data = prepare_proposal_data(
        output_dir=output_dir,
        latent_dim=latent_dim,
        ae_max_epochs=ae_max_epochs,
        ae_patience=ae_patience,
        ae_batch_size=ae_batch_size,
        ae_learning_rate=ae_learning_rate,
        latent_activation=latent_activation,
        seed=seed,
    )

    save_json(data.preprocessing_baseline, output_dir / "baseline_preprocessing.json")
    save_json(data.preprocessing_non_v, output_dir / "ae_non_v_preprocessing.json")

    base_default_params = default_lgbm_params(data.y_train, n_jobs=n_jobs)
    ae_default_params = default_lgbm_params(data.y_train, n_jobs=n_jobs)

    baseline_default = train_and_record(
        data,
        output_dir,
        "baseline_default",
        data.X_base_train,
        data.X_base_valid,
        data.X_base_test,
        data.baseline_categorical_columns,
        base_default_params,
    )
    ae_default = train_and_record(
        data,
        output_dir,
        "ae_latent_replacement_default",
        data.X_ae_train,
        data.X_ae_valid,
        data.X_ae_test,
        data.ae_categorical_columns,
        ae_default_params,
    )

    results: dict[str, object] = {
        "baseline_default": compact_result(baseline_default),
        "ae_latent_replacement_default": compact_result(ae_default),
    }
    comparisons: dict[str, object] = {
        "ae_default_vs_baseline_default": paired_bootstrap_ap_delta(
            data.y_test.to_numpy(),
            baseline_default["test_score"],  # type: ignore[arg-type]
            ae_default["test_score"],  # type: ignore[arg-type]
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    }

    if n_trials > 0:
        log(f"Running Optuna/TPE tuning for baseline ({n_trials} trials).")
        tuned_base_params, _ = tune_params(
            data.X_base_train,
            data.y_train,
            data.X_base_valid,
            data.y_valid,
            data.baseline_categorical_columns,
            n_trials=n_trials,
            n_jobs=n_jobs,
            seed=seed,
            study_name="baseline_tpe",
            output_dir=output_dir,
        )
        baseline_tuned = train_and_record(
            data,
            output_dir,
            "baseline_tuned",
            data.X_base_train,
            data.X_base_valid,
            data.X_base_test,
            data.baseline_categorical_columns,
            tuned_base_params,
        )
        log(f"Running Optuna/TPE tuning for AE latent replacement ({n_trials} trials).")
        tuned_ae_params, _ = tune_params(
            data.X_ae_train,
            data.y_train,
            data.X_ae_valid,
            data.y_valid,
            data.ae_categorical_columns,
            n_trials=n_trials,
            n_jobs=n_jobs,
            seed=seed,
            study_name="ae_latent_replacement_tpe",
            output_dir=output_dir,
        )
        ae_tuned = train_and_record(
            data,
            output_dir,
            "ae_latent_replacement_tuned",
            data.X_ae_train,
            data.X_ae_valid,
            data.X_ae_test,
            data.ae_categorical_columns,
            tuned_ae_params,
        )
        results["baseline_tuned"] = compact_result(baseline_tuned)
        results["ae_latent_replacement_tuned"] = compact_result(ae_tuned)
        comparisons["ae_tuned_vs_baseline_tuned"] = paired_bootstrap_ap_delta(
            data.y_test.to_numpy(),
            baseline_tuned["test_score"],  # type: ignore[arg-type]
            ae_tuned["test_score"],  # type: ignore[arg-type]
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

    summary = {
        "experiment": "original_proposal_stratified_v_latent_replacement",
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "split_strategy": "stratified_holdout",
        "split_ratios": {"train": TRAIN_RATIO, "validation": VALID_RATIO, "test": TEST_RATIO},
        "seed": seed,
        "sample_size": SAMPLE_SIZE,
        "primary_metric": "Average Precision / PR-AUC",
        "proposal_contract": {
            "baseline": "LightGBM on original preprocessed features, no resampling",
            "autoencoder": "V1-V339 only, zero imputation, z-score scaling, MSE, Adam, ReLU hidden layers",
            "integration": "replace original V1-V339 with encoder latent features, then LightGBM",
            "validation_usage": "early stopping, Optuna objective, threshold selection",
            "test_usage": "final evaluation only",
            "resampling": "none",
        },
        "feature_counts": {
            "baseline": int(data.X_base_train.shape[1]),
            "ae_latent_replacement": int(data.X_ae_train.shape[1]),
            "v_features_replaced": len(data.v_columns),
            "latent_features": len(data.latent_feature_names),
            "v_missing_indicators": 0,
        },
        "n_trials_per_tuned_model": n_trials,
        "n_bootstrap": n_bootstrap,
        "results": results,
        "comparisons": comparisons,
    }
    save_json(summary, output_dir / "experiment_summary.json")
    print_summary(summary)
    return summary


def compact_result(result: dict[str, object]) -> dict[str, object]:
    validation = result["validation_selected"]  # type: ignore[index]
    test = result["test_selected"]  # type: ignore[index]
    return {
        "best_iteration": int(result["best_iteration"]),
        "selected_threshold": float(result["selected_threshold"]),
        "validation_average_precision": float(validation["average_precision"]),  # type: ignore[index]
        "validation_roc_auc": float(validation["roc_auc"]),  # type: ignore[index]
        "validation_f1": float(validation["f1"]),  # type: ignore[index]
        "validation_mcc": float(validation["mcc"]),  # type: ignore[index]
        "test_average_precision": float(test["average_precision"]),  # type: ignore[index]
        "test_roc_auc": float(test["roc_auc"]),  # type: ignore[index]
        "test_precision": float(test["precision"]),  # type: ignore[index]
        "test_recall": float(test["recall"]),  # type: ignore[index]
        "test_f1": float(test["f1"]),  # type: ignore[index]
        "test_mcc": float(test["mcc"]),  # type: ignore[index]
    }


def print_summary(summary: dict[str, object]) -> None:
    print()
    print("Original Proposal Stratified V-Latent Replacement")
    print("=================================================")
    results = summary["results"]  # type: ignore[index]
    for name, result in results.items():  # type: ignore[union-attr]
        print(
            f"{name:34s} "
            f"val_AP={result['validation_average_precision']:.6f} "
            f"test_AP={result['test_average_precision']:.6f} "
            f"ROC={result['test_roc_auc']:.6f} "
            f"F1={result['test_f1']:.6f} "
            f"MCC={result['test_mcc']:.6f}"
        )
    print()
    comparisons = summary["comparisons"]  # type: ignore[index]
    for name, comp in comparisons.items():  # type: ignore[union-attr]
        print(
            f"{name:34s} "
            f"delta_AP={comp['observed_delta_ap']:+.6f} "
            f"CI=[{comp['ci_2_5']:+.6f}, {comp['ci_97_5']:+.6f}] "
            f"p(delta<=0)={comp['p_delta_le_0']:.3f}"
        )
    print(f"\nSaved: {summary['output_dir']}/experiment_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run original proposal stratified V-latent replacement experiment.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--latent-activation", choices=["relu", "linear"], default="relu")
    parser.add_argument("--ae-max-epochs", type=int, default=60)
    parser.add_argument("--ae-patience", type=int, default=8)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        output_dir=args.output_dir,
        latent_dim=args.latent_dim,
        latent_activation=args.latent_activation,
        ae_max_epochs=args.ae_max_epochs,
        ae_patience=args.ae_patience,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        n_trials=args.n_trials,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
