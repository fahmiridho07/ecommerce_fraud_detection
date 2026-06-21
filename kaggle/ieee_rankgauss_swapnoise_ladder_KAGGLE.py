"""IEEE-CIS - RankGauss + Swap-Noise DAE + Masked MSE for Kaggle.

Standalone Kaggle version of the proposal-consistent diagnostic branch:

  baseline_tuned:
      proposal-style LightGBM on original features, Optuna objective = PR-AUC

  rg_append_latent_error:
      original features + RankGauss swap-noise DAE latent + observed recon errors

  rg_observed_replace_mask:
      replace only observed selected V values with inverse-RankGauss
      reconstruction, keep missing cells as missing, append V missing masks

All split, preprocessing, RankGauss fitting, Autoencoder training, and LightGBM
tuning use TRAIN only. Validation is used for early stopping, threshold
selection, and Optuna objective. Test is used only for final metrics and paired
bootstrap.

Kaggle use:
  1. Add Input: "IEEE-CIS Fraud Detection".
  2. Upload this file or paste it into a notebook cell.
  3. Run.

Outputs:
  /kaggle/working/rankgauss_swapnoise_ladder_results.json
  /kaggle/working/rankgauss_swapnoise_ladder_summary.csv
  /kaggle/working/rankgauss_v_selection_report.csv
  /kaggle/working/rankgauss_swapnoise_test_scores.csv
"""

from __future__ import annotations

import gc
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from tensorflow import keras


# =========================
# Config - edit in Kaggle
# =========================

SEED = 42
TRAIN_RATIO = 0.60
VALID_RATIO = 0.20
TEST_RATIO = 0.20

N_ESTIMATORS = 2000
EARLY_STOPPING = 100
N_TRIALS_BASELINE = 12
N_TRIALS_CANDIDATE = 10
N_BOOTSTRAP = 2000

MAX_V_COLUMNS = 150
V_CORR_THRESHOLD = 0.75
RANKGAUSS_MAX_QUANTILES = 1000
RANKGAUSS_SUBSAMPLE = 200_000
RANKGAUSS_CLIP = 5.0

LATENT_DIM = 64
SWAP_RATE = 0.15
AE_EPOCHS = 60
AE_PATIENCE = 8
AE_BATCH_SIZE = 2048
AE_LEARNING_RATE = 1e-3

RUN_APPEND_LATENT_ERROR = True
RUN_OBSERVED_REPLACE_MASK = True
SAVE_MODELS = False

# Optional quick smoke. Keep None for full dataset.
SAMPLE_ROWS = None

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
MISSING_CATEGORY = "__MISSING__"
UNKNOWN_CATEGORY_VALUE = -1

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.get_logger().setLevel("ERROR")
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)


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
class RankGaussColumn:
    column: str
    transformer: QuantileTransformer | None
    fill_value: float
    n_observed: int
    n_unique: int


def find_data_dir() -> Path:
    candidates = [
        Path("/kaggle/input/ieee-fraud-detection"),
        Path("/kaggle/input/competitions/ieee-fraud-detection"),
    ]
    for candidate in candidates:
        if (candidate / "train_transaction.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find IEEE-CIS files. Add the Kaggle input 'IEEE-CIS Fraud Detection'."
    )


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def load_data() -> pd.DataFrame:
    data_dir = find_data_dir()
    print(f"Data dir: {data_dir}")
    tx = pd.read_csv(data_dir / "train_transaction.csv")
    identity = pd.read_csv(data_dir / "train_identity.csv")
    df = tx.merge(identity, on=ID_COL, how="left")
    del tx, identity
    gc.collect()
    if SAMPLE_ROWS is not None:
        df = df.sort_values(TIME_COL).head(int(SAMPLE_ROWS)).reset_index(drop=True)
    return df


def split_60_20_20(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].to_numpy(dtype=int)
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=VALID_RATIO + TEST_RATIO,
        stratify=y,
        random_state=SEED,
    )
    valid_fraction = TEST_RATIO / (VALID_RATIO + TEST_RATIO)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=valid_fraction,
        stratify=y[temp_idx],
        random_state=SEED,
    )
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[valid_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def v_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("V") and column[1:].isdigit()]


def proposal_categorical_columns(X: pd.DataFrame) -> list[str]:
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


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[TARGET].astype(int).copy()
    X = df.drop(columns=[TARGET, ID_COL]).copy()
    return X, y


def fit_proposal_encoding(X_train: pd.DataFrame) -> dict[str, object]:
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
    }


def apply_proposal_encoding(X: pd.DataFrame, preprocessing: dict[str, object]) -> pd.DataFrame:
    transformed = X.loc[:, preprocessing["feature_columns"]].copy()
    mappings: dict[str, dict[str, int]] = preprocessing["categorical_mappings"]  # type: ignore[assignment]
    for column, mapping in mappings.items():
        values = transformed[column].astype("string").fillna(MISSING_CATEGORY)
        transformed[column] = values.map(mapping).fillna(UNKNOWN_CATEGORY_VALUE).astype("int32")
    numeric_columns = [column for column in transformed.columns if column not in mappings]
    for column in numeric_columns:
        transformed[column] = pd.to_numeric(transformed[column], errors="coerce").astype("float32")
    return transformed


def fit_apply_baseline(
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, object]]:
    pre = fit_proposal_encoding(X_train_raw)
    X_train = apply_proposal_encoding(X_train_raw, pre)
    X_valid = apply_proposal_encoding(X_valid_raw, pre)
    X_test = apply_proposal_encoding(X_test_raw, pre)
    return X_train, X_valid, X_test, list(pre["categorical_columns"]), pre


def ap_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    return "average_precision", float(average_precision_score(y_true, y_pred)), True


def base_lgbm_params(y_train: pd.Series) -> dict[str, object]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    return {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": N_ESTIMATORS,
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
        "n_jobs": -1,
        "random_state": SEED,
        "metric": "None",
        "verbosity": -1,
    }


def suggest_lgbm_params(trial: optuna.Trial, y_train: pd.Series) -> dict[str, object]:
    params = base_lgbm_params(y_train)
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0),
        }
    )
    return params


def lgbm_params_from_best_params(best_params: dict[str, object], y_train: pd.Series) -> dict[str, object]:
    params = base_lgbm_params(y_train)
    params.update(best_params)
    return params


def fit_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
    params: dict[str, object],
) -> tuple[lgb.LGBMClassifier, int]:
    categorical_feature = [column for column in categorical_columns if column in X_train.columns]
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=ap_eval,
        categorical_feature=categorical_feature,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model, int(model.best_iteration_ or model.n_estimators)


def tune_lgbm(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_columns: list[str],
    n_trials: int,
) -> tuple[lgb.LGBMClassifier, int, dict[str, object], pd.DataFrame]:
    print(f"\nTuning {name}: {n_trials} Optuna/TPE trials, objective=validation PR-AUC")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgbm_params(trial, y_train)
        model, best_iteration = fit_lgbm(
            X_train, y_train, X_valid, y_valid, categorical_columns, params
        )
        score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
        return float(average_precision_score(y_valid, score))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)
    best_params = lgbm_params_from_best_params(dict(study.best_params), y_train)
    model, best_iteration = fit_lgbm(
        X_train, y_train, X_valid, y_valid, categorical_columns, best_params
    )
    trials = study.trials_dataframe()
    trials.to_csv(OUT_DIR / f"{name}_optuna_trials.csv", index=False)
    save_json(
        {
            "study_name": name,
            "n_trials": n_trials,
            "best_validation_average_precision": float(study.best_value),
            "best_params": best_params,
        },
        OUT_DIR / f"{name}_best_params.json",
    )
    return model, best_iteration, best_params, trials


def pick_threshold(y_valid: np.ndarray, score_valid: np.ndarray) -> float:
    best_threshold = 0.5
    best_mcc = -2.0
    best_f1 = -1.0
    for threshold in np.arange(0.01, 1.0, 0.01):
        pred = (score_valid >= threshold).astype(int)
        mcc = matthews_corrcoef(y_valid, pred)
        f1 = f1_score(y_valid, pred, zero_division=0)
        if mcc > best_mcc or (mcc == best_mcc and f1 > best_f1):
            best_mcc = float(mcc)
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
    }


def evaluate_model(
    name: str,
    model: lgb.LGBMClassifier,
    best_iteration: int,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]
    threshold = pick_threshold(y_valid.to_numpy(dtype=int), valid_score)
    result = metrics(y_test.to_numpy(dtype=int), test_score, threshold)
    print(
        f"{name:28s} AP={result['average_precision']:.6f} "
        f"ROC={result['roc_auc']:.5f} F1={result['f1']:.4f} MCC={result['mcc']:.4f}"
    )
    return result, valid_score, test_score


def paired_bootstrap_ap_delta(
    y_true: np.ndarray,
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float | int]:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if sample_y.min() == sample_y.max():
            continue
        deltas.append(
            average_precision_score(sample_y, candidate_score[idx])
            - average_precision_score(sample_y, reference_score[idx])
        )
    d = np.asarray(deltas, dtype="float64")
    ref_ap = float(average_precision_score(y_true, reference_score))
    cand_ap = float(average_precision_score(y_true, candidate_score))
    return {
        "reference_ap": ref_ap,
        "candidate_ap": cand_ap,
        "observed_delta_ap": cand_ap - ref_ap,
        "ci_2_5": float(np.percentile(d, 2.5)),
        "ci_50": float(np.percentile(d, 50.0)),
        "ci_97_5": float(np.percentile(d, 97.5)),
        "p_delta_le_0": float(np.mean(d <= 0.0)),
        "n_bootstrap": int(d.shape[0]),
    }


def select_v_by_missingness_correlation(
    X_train_raw: pd.DataFrame,
    raw_v_columns: list[str],
) -> tuple[list[str], pd.DataFrame]:
    print("\nSelecting V columns by missingness group + correlation pruning...")
    numeric = X_train_raw.loc[:, raw_v_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    missing_rate = numeric.isna().mean(axis=0)
    observed_count = numeric.notna().sum(axis=0)
    nunique = numeric.nunique(dropna=True)
    variance = numeric.var(axis=0, skipna=True).fillna(0.0)
    missing_group = missing_rate.round(4)

    selected: list[str] = []
    selected_set: set[str] = set()
    rows: list[dict[str, object]] = []

    for group_value, group_columns_index in missing_group.groupby(missing_group).groups.items():
        group_columns = list(group_columns_index)
        ordered = sorted(
            group_columns,
            key=lambda column: (
                float(variance[column]),
                int(nunique[column]),
                int(observed_count[column]),
                column,
            ),
            reverse=True,
        )
        corr = numeric.loc[:, ordered].corr().abs()
        kept: list[str] = []
        for column in ordered:
            max_corr = 0.0
            if kept:
                max_corr = float(corr.loc[column, kept].max(skipna=True))
                if np.isnan(max_corr):
                    max_corr = 0.0
            keep = not kept or max_corr < V_CORR_THRESHOLD
            if keep:
                kept.append(column)
                selected.append(column)
                selected_set.add(column)
            rows.append(
                {
                    "feature": column,
                    "missing_group": float(group_value),
                    "missing_rate": float(missing_rate[column]),
                    "observed_count": int(observed_count[column]),
                    "n_unique": int(nunique[column]),
                    "variance": float(variance[column]),
                    "max_corr_with_kept": max_corr,
                    "selected_before_cap": bool(keep),
                }
            )

    report = pd.DataFrame(rows)
    if len(selected) > MAX_V_COLUMNS:
        kept_report = report.loc[report["feature"].isin(selected)].copy()
        kept_report["selection_score"] = (
            kept_report["variance"].rank(method="dense", ascending=True)
            + kept_report["n_unique"].rank(method="dense", ascending=True)
            + kept_report["observed_count"].rank(method="dense", ascending=True)
        )
        selected = (
            kept_report.sort_values(["selection_score", "feature"], ascending=[False, True])
            .head(MAX_V_COLUMNS)["feature"]
            .tolist()
        )
        selected_set = set(selected)

    report["selected"] = report["feature"].isin(selected_set)
    report = report.sort_values(
        ["selected", "missing_group", "variance", "n_unique", "feature"],
        ascending=[False, True, False, False, True],
    ).reset_index(drop=True)
    report.to_csv(OUT_DIR / "rankgauss_v_selection_report.csv", index=False)
    print(f"Selected {len(selected)} of {len(raw_v_columns)} V columns.")
    del numeric
    gc.collect()
    return selected, report


def fit_rankgauss(X_train_raw: pd.DataFrame, selected_columns: list[str]) -> list[RankGaussColumn]:
    print("Fitting train-only observed-value RankGauss transformers...")
    fitted: list[RankGaussColumn] = []
    for index, column in enumerate(selected_columns, start=1):
        values = pd.to_numeric(X_train_raw[column], errors="coerce").dropna().astype("float64")
        n_observed = int(values.shape[0])
        n_unique = int(values.nunique(dropna=True))
        fill_value = float(values.median()) if n_observed else 0.0
        transformer: QuantileTransformer | None = None
        if n_observed >= 2 and n_unique >= 2:
            n_quantiles = max(2, min(RANKGAUSS_MAX_QUANTILES, n_observed))
            transformer = QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                subsample=min(RANKGAUSS_SUBSAMPLE, n_observed),
                random_state=SEED,
                copy=True,
            )
            transformer.fit(values.to_numpy(dtype="float64").reshape(-1, 1))
        fitted.append(RankGaussColumn(column, transformer, fill_value, n_observed, n_unique))
        if index % 25 == 0 or index == len(selected_columns):
            print(f"  fitted {index}/{len(selected_columns)}")
    joblib.dump(fitted, OUT_DIR / "rankgauss_transformers.pkl")
    return fitted


def transform_rankgauss(
    X_raw: pd.DataFrame,
    fitted: list[RankGaussColumn],
) -> tuple[np.ndarray, np.ndarray]:
    values_out = np.zeros((len(X_raw), len(fitted)), dtype="float32")
    observed_out = np.zeros((len(X_raw), len(fitted)), dtype="float32")
    for index, column_fit in enumerate(fitted):
        raw = pd.to_numeric(X_raw[column_fit.column], errors="coerce")
        observed = raw.notna().to_numpy()
        observed_out[:, index] = observed.astype("float32")
        if not observed.any():
            continue
        if column_fit.transformer is None:
            transformed = np.zeros(int(observed.sum()), dtype="float32")
        else:
            transformed = column_fit.transformer.transform(
                raw.loc[observed].to_numpy(dtype="float64").reshape(-1, 1)
            ).ravel()
            transformed = np.clip(transformed, -RANKGAUSS_CLIP, RANKGAUSS_CLIP).astype("float32")
        values_out[observed, index] = transformed
    return values_out, observed_out


def inverse_rankgauss(transformed: np.ndarray, fitted: list[RankGaussColumn]) -> np.ndarray:
    raw = np.empty_like(transformed, dtype="float32")
    for index, column_fit in enumerate(fitted):
        values = np.clip(transformed[:, index], -RANKGAUSS_CLIP, RANKGAUSS_CLIP)
        if column_fit.transformer is None:
            raw[:, index] = column_fit.fill_value
        else:
            raw[:, index] = column_fit.transformer.inverse_transform(
                values.reshape(-1, 1)
            ).ravel().astype("float32")
    return raw


class SwapNoise(keras.layers.Layer):
    def __init__(self, swap_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.swap_rate = float(swap_rate)

    def call(self, inputs, training=None):  # type: ignore[override]
        if not training or self.swap_rate <= 0.0:
            return inputs
        batch_size = tf.shape(inputs)[0]
        shuffled_rows = tf.random.shuffle(tf.range(batch_size))
        shuffled = tf.gather(inputs, shuffled_rows)
        swap_mask = tf.random.uniform(tf.shape(inputs)) < self.swap_rate
        return tf.where(swap_mask, shuffled, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"swap_rate": self.swap_rate})
        return config


def build_swapnoise_autoencoder(input_dim: int) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(input_dim,), name="rankgauss_v_values")
    x = SwapNoise(SWAP_RATE, name="swap_noise")(inputs)
    width = max(64, min(256, input_dim * 2))
    x = keras.layers.Dense(width, activation="relu", name="encoder_dense_wide")(x)
    x = keras.layers.Dense(max(32, width // 2), activation="relu", name="encoder_dense_mid")(x)
    latent = keras.layers.Dense(LATENT_DIM, activation="linear", name="latent")(x)
    x = keras.layers.Dense(max(32, width // 2), activation="relu", name="decoder_dense_mid")(latent)
    x = keras.layers.Dense(width, activation="relu", name="decoder_dense_wide")(x)
    outputs = keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    autoencoder = keras.Model(inputs=inputs, outputs=outputs, name="rankgauss_swapnoise_autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="rankgauss_swapnoise_encoder")

    def observed_only_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target = y_true[:, :input_dim]
        observed = y_true[:, input_dim:]
        squared_error = tf.square(target - y_pred) * observed
        denom = tf.reduce_sum(observed, axis=-1) + tf.keras.backend.epsilon()
        return tf.reduce_sum(squared_error, axis=-1) / denom

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(AE_LEARNING_RATE),
        loss=observed_only_mse,
    )
    return autoencoder, encoder


def train_swapnoise_ae(
    V_train: np.ndarray,
    V_valid: np.ndarray,
    observed_train: np.ndarray,
    observed_valid: np.ndarray,
) -> tuple[keras.Model, keras.Model]:
    print("\nTraining swap-noise DAE with observed-only masked MSE...")
    autoencoder, encoder = build_swapnoise_autoencoder(V_train.shape[1])
    target_train = np.concatenate([V_train, observed_train], axis=1).astype("float32")
    target_valid = np.concatenate([V_valid, observed_valid], axis=1).astype("float32")
    history = autoencoder.fit(
        V_train,
        target_train,
        validation_data=(V_valid, target_valid),
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=AE_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    pd.DataFrame(history.history).to_csv(OUT_DIR / "rankgauss_swapnoise_ae_history.csv", index=False)
    if SAVE_MODELS:
        autoencoder.save(OUT_DIR / "rankgauss_swapnoise_autoencoder.keras", include_optimizer=False)
        encoder.save(OUT_DIR / "rankgauss_swapnoise_encoder.keras", include_optimizer=False)
    return autoencoder, encoder


def reconstruction_error_features(
    values: np.ndarray,
    recon: np.ndarray,
    observed: np.ndarray,
) -> pd.DataFrame:
    abs_error = np.abs(values - recon).astype("float32")
    squared_error = np.square(values - recon).astype("float32")
    observed_count = observed.sum(axis=1)
    denom = np.maximum(observed_count, 1.0)
    masked_abs = abs_error * observed
    masked_sq = squared_error * observed
    return pd.DataFrame(
        {
            "rg_swapdae_mse_observed": (masked_sq.sum(axis=1) / denom).astype("float32"),
            "rg_swapdae_mae_observed": (masked_abs.sum(axis=1) / denom).astype("float32"),
            "rg_swapdae_max_abs_observed": masked_abs.max(axis=1).astype("float32"),
            "rg_swapdae_observed_rate": observed.mean(axis=1).astype("float32"),
        }
    )


def append_latent_error(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    V_train: np.ndarray,
    V_valid: np.ndarray,
    V_test: np.ndarray,
    observed_train: np.ndarray,
    observed_valid: np.ndarray,
    observed_test: np.ndarray,
    encoder: keras.Model,
    autoencoder: keras.Model,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Building append-latent-error candidate matrices...")
    latent_train = encoder.predict(V_train, batch_size=8192, verbose=0).astype("float32")
    latent_valid = encoder.predict(V_valid, batch_size=8192, verbose=0).astype("float32")
    latent_test = encoder.predict(V_test, batch_size=8192, verbose=0).astype("float32")
    recon_train = autoencoder.predict(V_train, batch_size=8192, verbose=0).astype("float32")
    recon_valid = autoencoder.predict(V_valid, batch_size=8192, verbose=0).astype("float32")
    recon_test = autoencoder.predict(V_test, batch_size=8192, verbose=0).astype("float32")

    latent_names = [f"rg_swapdae_latent_{i:03d}" for i in range(1, latent_train.shape[1] + 1)]

    def combine(
        base: pd.DataFrame,
        latent: np.ndarray,
        values: np.ndarray,
        recon: np.ndarray,
        observed: np.ndarray,
    ) -> pd.DataFrame:
        latent_df = pd.DataFrame(latent, columns=latent_names)
        err_df = reconstruction_error_features(values, recon, observed)
        return pd.concat(
            [base.reset_index(drop=True), latent_df.reset_index(drop=True), err_df.reset_index(drop=True)],
            axis=1,
        )

    return (
        combine(X_train, latent_train, V_train, recon_train, observed_train),
        combine(X_valid, latent_valid, V_valid, recon_valid, observed_valid),
        combine(X_test, latent_test, V_test, recon_test, observed_test),
    )


def observed_replace_mask(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    selected_columns: list[str],
    fitted: list[RankGaussColumn],
    V_train: np.ndarray,
    V_valid: np.ndarray,
    V_test: np.ndarray,
    observed_train: np.ndarray,
    observed_valid: np.ndarray,
    observed_test: np.ndarray,
    autoencoder: keras.Model,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Building observed-replace-mask candidate matrices...")
    recon_train = autoencoder.predict(V_train, batch_size=8192, verbose=0).astype("float32")
    recon_valid = autoencoder.predict(V_valid, batch_size=8192, verbose=0).astype("float32")
    recon_test = autoencoder.predict(V_test, batch_size=8192, verbose=0).astype("float32")
    raw_train = inverse_rankgauss(recon_train, fitted)
    raw_valid = inverse_rankgauss(recon_valid, fitted)
    raw_test = inverse_rankgauss(recon_test, fitted)

    def replace(base: pd.DataFrame, raw_recon: np.ndarray, observed: np.ndarray) -> pd.DataFrame:
        out = base.copy()
        for index, column in enumerate(selected_columns):
            values = out[column].to_numpy(copy=True)
            observed_mask = observed[:, index] > 0.5
            values[observed_mask] = raw_recon[observed_mask, index]
            out[column] = values
        missing = 1.0 - observed
        mask_df = pd.DataFrame(
            missing,
            columns=[f"rg_missing_{column}" for column in selected_columns],
        ).astype("float32")
        return pd.concat([out.reset_index(drop=True), mask_df.reset_index(drop=True)], axis=1)

    return (
        replace(X_train, raw_train, observed_train),
        replace(X_valid, raw_valid, observed_valid),
        replace(X_test, raw_test, observed_test),
    )


def run_candidate(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_columns: list[str],
    n_trials: int,
) -> tuple[dict[str, float | int], np.ndarray, dict[str, object]]:
    model, best_iteration, params, _ = tune_lgbm(
        name,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_columns,
        n_trials=n_trials,
    )
    result, _, test_score = evaluate_model(
        name,
        model,
        best_iteration,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )
    return result, test_score, params


def main() -> None:
    print("Loading IEEE-CIS and creating stratified 60/20/20 split...")
    full_df = load_data()
    train_df, valid_df, test_df = split_60_20_20(full_df)
    del full_df
    gc.collect()

    X_train_raw, y_train = split_features_target(train_df)
    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    test_ids = test_df[ID_COL].to_numpy()
    del train_df, valid_df, test_df
    gc.collect()

    raw_v_columns = v_columns(X_train_raw)
    print(
        f"Rows train/valid/test: {len(y_train)}/{len(y_valid)}/{len(y_test)} | "
        f"fraud rates: {y_train.mean():.5f}/{y_valid.mean():.5f}/{y_test.mean():.5f} | "
        f"V columns: {len(raw_v_columns)}"
    )

    print("\nFitting proposal-style baseline encoding on train only...")
    X_train, X_valid, X_test, categorical_columns, baseline_pre = fit_apply_baseline(
        X_train_raw, X_valid_raw, X_test_raw
    )
    save_json(
        {
            "categorical_columns": categorical_columns,
            "feature_count": int(X_train.shape[1]),
            "fit_scope": "train split only",
        },
        OUT_DIR / "proposal_baseline_preprocessing_contract.json",
    )

    selected_v, selection_report = select_v_by_missingness_correlation(X_train_raw, raw_v_columns)
    fitted_rankgauss = fit_rankgauss(X_train_raw, selected_v)
    print("Transforming RankGauss matrices...")
    V_train, observed_train = transform_rankgauss(X_train_raw, fitted_rankgauss)
    V_valid, observed_valid = transform_rankgauss(X_valid_raw, fitted_rankgauss)
    V_test, observed_test = transform_rankgauss(X_test_raw, fitted_rankgauss)

    autoencoder, encoder = train_swapnoise_ae(V_train, V_valid, observed_train, observed_valid)

    results: dict[str, dict[str, float | int]] = {}
    scores: dict[str, np.ndarray] = {}
    params_store: dict[str, dict[str, object]] = {}

    print("\n================ Baseline =================")
    baseline_result, baseline_score, baseline_params = run_candidate(
        "baseline_tuned",
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        categorical_columns,
        n_trials=N_TRIALS_BASELINE,
    )
    results["baseline_tuned"] = baseline_result
    scores["baseline_tuned"] = baseline_score
    params_store["baseline_tuned"] = baseline_params

    candidate_order: list[str] = []

    if RUN_APPEND_LATENT_ERROR:
        print("\n================ Candidate: append latent + error =================")
        Xa_train, Xa_valid, Xa_test = append_latent_error(
            X_train,
            X_valid,
            X_test,
            V_train,
            V_valid,
            V_test,
            observed_train,
            observed_valid,
            observed_test,
            encoder,
            autoencoder,
        )
        result, score, params = run_candidate(
            "rg_append_latent_error",
            Xa_train,
            y_train,
            Xa_valid,
            y_valid,
            Xa_test,
            y_test,
            categorical_columns,
            n_trials=N_TRIALS_CANDIDATE,
        )
        results["rg_append_latent_error"] = result
        scores["rg_append_latent_error"] = score
        params_store["rg_append_latent_error"] = params
        candidate_order.append("rg_append_latent_error")
        del Xa_train, Xa_valid, Xa_test
        gc.collect()

    if RUN_OBSERVED_REPLACE_MASK:
        print("\n================ Candidate: observed replace + mask =================")
        Xr_train, Xr_valid, Xr_test = observed_replace_mask(
            X_train,
            X_valid,
            X_test,
            selected_v,
            fitted_rankgauss,
            V_train,
            V_valid,
            V_test,
            observed_train,
            observed_valid,
            observed_test,
            autoencoder,
        )
        result, score, params = run_candidate(
            "rg_observed_replace_mask",
            Xr_train,
            y_train,
            Xr_valid,
            y_valid,
            Xr_test,
            y_test,
            categorical_columns,
            n_trials=N_TRIALS_CANDIDATE,
        )
        results["rg_observed_replace_mask"] = result
        scores["rg_observed_replace_mask"] = score
        params_store["rg_observed_replace_mask"] = params
        candidate_order.append("rg_observed_replace_mask")
        del Xr_train, Xr_valid, Xr_test
        gc.collect()

    print("\n================ Paired Bootstrap vs Baseline =================")
    y_test_np = y_test.to_numpy(dtype=int)
    comparisons = {}
    for candidate in candidate_order:
        comparisons[f"{candidate}_vs_baseline_tuned"] = paired_bootstrap_ap_delta(
            y_test_np,
            scores["baseline_tuned"],
            scores[candidate],
            n_bootstrap=N_BOOTSTRAP,
        )
        comp = comparisons[f"{candidate}_vs_baseline_tuned"]
        print(
            f"{candidate:28s} delta={comp['observed_delta_ap']:+.6f} "
            f"CI=[{comp['ci_2_5']:+.5f}, {comp['ci_97_5']:+.5f}] "
            f"p(d<=0)={comp['p_delta_le_0']:.3f}"
        )

    summary_rows = []
    for name, result in results.items():
        summary_rows.append({"model": name, **result})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "rankgauss_swapnoise_ladder_summary.csv", index=False)

    score_df = pd.DataFrame({ID_COL: test_ids, TARGET: y_test_np})
    for name, score in scores.items():
        score_df[f"score_{name}"] = score
    score_df.to_csv(OUT_DIR / "rankgauss_swapnoise_test_scores.csv", index=False)

    output = {
        "experiment": "rankgauss_swapnoise_ae_ladder_kaggle",
        "protocol": {
            "split": "stratified_holdout_60_20_20",
            "seed": SEED,
            "primary_metric": "Average Precision / PR-AUC",
            "train_only_fitting": True,
        },
        "config": {
            "n_trials_baseline": N_TRIALS_BASELINE,
            "n_trials_candidate": N_TRIALS_CANDIDATE,
            "n_bootstrap": N_BOOTSTRAP,
            "max_v_columns": MAX_V_COLUMNS,
            "v_corr_threshold": V_CORR_THRESHOLD,
            "latent_dim": LATENT_DIM,
            "swap_rate": SWAP_RATE,
            "ae_epochs": AE_EPOCHS,
            "rankgauss_clip": RANKGAUSS_CLIP,
            "rankgauss_subsample": RANKGAUSS_SUBSAMPLE,
        },
        "data": {
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "train_fraud_rate": float(y_train.mean()),
            "valid_fraud_rate": float(y_valid.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "raw_v_count": len(raw_v_columns),
            "selected_v_count": len(selected_v),
        },
        "results": results,
        "comparisons": comparisons,
        "best_params": params_store,
    }
    save_json(output, OUT_DIR / "rankgauss_swapnoise_ladder_results.json")

    print("\n================ Final Table =================")
    print(summary_df[["model", "average_precision", "roc_auc", "f1", "mcc"]].to_string(index=False))
    print("\nSaved:")
    print(f"  {OUT_DIR / 'rankgauss_swapnoise_ladder_results.json'}")
    print(f"  {OUT_DIR / 'rankgauss_swapnoise_ladder_summary.csv'}")
    print(f"  {OUT_DIR / 'rankgauss_v_selection_report.csv'}")
    print(f"  {OUT_DIR / 'rankgauss_swapnoise_test_scores.csv'}")
    print("\nDecision rule: promote only if candidate AP > baseline_tuned AP and bootstrap CI is positive.")


if __name__ == "__main__":
    main()
