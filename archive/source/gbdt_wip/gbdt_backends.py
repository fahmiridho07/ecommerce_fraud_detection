"""Shared GBDT backend helpers for the baseline comparison shootout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import RANDOM_SEED
from preprocessing import (
    MISSING_CATEGORY,
    UNKNOWN_CATEGORY_VALUE,
    _normalize_category_series,
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_categorical_columns,
)
from tune_lgbm_optuna import TUNING_PROFILES, _suggest_from_spec

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - environment dependent
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - environment dependent
    xgb = None

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:  # pragma: no cover - environment dependent
    CatBoostClassifier = None
    Pool = None


SUPPORTED_BACKENDS = ("lightgbm", "xgboost", "catboost")
SUPPORTED_PREPROCESSING_MODES = ("native", "shared_lgbm")

DEFAULT_THRESHOLD = 0.5
EARLY_STOPPING_ROUNDS = 100


@dataclass
class PreparedMatrices:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    preprocessing: dict[str, object]
    categorical_columns: list[str]
    cat_feature_indices: list[int] | None
    preprocessing_mode: str
    backend: str

    @property
    def total_features(self) -> int:
        return int(self.X_train.shape[1])


def average_precision_eval(y_true, y_pred):
    """LightGBM custom validation metric for PR-AUC / Average Precision."""
    return "average_precision", average_precision_score(y_true, y_pred), True


def roc_auc_eval(y_true, y_pred):
    """LightGBM custom validation metric for ROC-AUC."""
    if len(set(y_true)) < 2:
        return "roc_auc", 0.0, True
    return "roc_auc", roc_auc_score(y_true, y_pred), True


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    return negative_count / positive_count if positive_count else 1.0


def fit_catboost_native_preprocessing(X_train: pd.DataFrame) -> dict[str, object]:
    categorical_columns = get_categorical_columns(X_train)
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "preprocessing_mode": "catboost_native",
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
    }


def apply_catboost_native_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    feature_columns = preprocessing["feature_columns"]
    categorical_columns = preprocessing["categorical_columns"]
    transformed = X.loc[:, feature_columns].copy()
    for column in categorical_columns:
        transformed[column] = _normalize_category_series(transformed[column])
    return transformed


def fit_xgboost_native_preprocessing(X_train: pd.DataFrame) -> dict[str, object]:
    categorical_columns = get_categorical_columns(X_train)
    category_levels: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = _normalize_category_series(X_train[column])
        levels = sorted(set(values.dropna().tolist()))
        category_levels[column] = levels
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "category_levels": category_levels,
        "preprocessing_mode": "xgboost_native",
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
    }


def apply_xgboost_native_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    feature_columns = preprocessing["feature_columns"]
    categorical_columns = preprocessing["categorical_columns"]
    category_levels = preprocessing["category_levels"]
    transformed = X.loc[:, feature_columns].copy()
    for column in categorical_columns:
        values = _normalize_category_series(transformed[column])
        transformed[column] = pd.Categorical(
            values,
            categories=category_levels[column],
        )
    return transformed


def fit_preprocessing_for_backend(
    X_train: pd.DataFrame,
    backend: str,
    preprocessing_mode: str,
) -> dict[str, object]:
    if preprocessing_mode == "shared_lgbm":
        preprocessing = fit_baseline_preprocessing(X_train)
        preprocessing["preprocessing_mode"] = "shared_lgbm"
        return preprocessing

    if preprocessing_mode != "native":
        raise ValueError(f"Unsupported preprocessing_mode: {preprocessing_mode}")

    if backend == "lightgbm":
        preprocessing = fit_baseline_preprocessing(X_train)
        preprocessing["preprocessing_mode"] = "lightgbm_native"
        return preprocessing
    if backend == "catboost":
        return fit_catboost_native_preprocessing(X_train)
    if backend == "xgboost":
        return fit_xgboost_native_preprocessing(X_train)
    raise ValueError(f"Unsupported backend: {backend}")


def apply_preprocessing_for_backend(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
    backend: str,
) -> pd.DataFrame:
    mode = preprocessing.get("preprocessing_mode", "shared_lgbm")
    if mode == "shared_lgbm" or (mode == "lightgbm_native" and backend == "lightgbm"):
        return apply_baseline_preprocessing(X, preprocessing)
    if mode == "catboost_native":
        return apply_catboost_native_preprocessing(X, preprocessing)
    if mode == "xgboost_native":
        return apply_xgboost_native_preprocessing(X, preprocessing)
    raise ValueError(
        f"Unsupported preprocessing mode {mode!r} for backend {backend!r}."
    )


def cat_feature_indices_for_matrix(
    X: pd.DataFrame,
    categorical_columns: list[str],
) -> list[int]:
    column_positions = {column: index for index, column in enumerate(X.columns)}
    return [column_positions[column] for column in categorical_columns]


def prepare_matrices_from_raw_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    backend: str,
    preprocessing_mode: str = "native",
    y_train: pd.Series | None = None,
    y_valid: pd.Series | None = None,
    y_test: pd.Series | None = None,
) -> PreparedMatrices:
    from preprocessing import split_features_target

    if y_train is None or y_valid is None or y_test is None:
        _, y_train = split_features_target(train_df)
        _, y_valid = split_features_target(valid_df)
        _, y_test = split_features_target(test_df)

    X_train_raw, _ = split_features_target(train_df)
    X_valid_raw, _ = split_features_target(valid_df)
    X_test_raw, _ = split_features_target(test_df)

    preprocessing = fit_preprocessing_for_backend(
        X_train_raw,
        backend,
        preprocessing_mode,
    )
    X_train = apply_preprocessing_for_backend(X_train_raw, preprocessing, backend)
    X_valid = apply_preprocessing_for_backend(X_valid_raw, preprocessing, backend)
    X_test = apply_preprocessing_for_backend(X_test_raw, preprocessing, backend)

    categorical_columns = list(preprocessing.get("categorical_columns", []))
    use_native_categoricals = (
        preprocessing_mode == "native"
        and backend in {"catboost", "xgboost"}
        and preprocessing.get("preprocessing_mode") != "shared_lgbm"
    )
    cat_feature_indices = (
        cat_feature_indices_for_matrix(X_train, categorical_columns)
        if use_native_categoricals
        else None
    )

    return PreparedMatrices(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        preprocessing=preprocessing,
        categorical_columns=categorical_columns,
        cat_feature_indices=cat_feature_indices,
        preprocessing_mode=preprocessing_mode,
        backend=backend,
    )


def build_default_params(
    backend: str,
    y_train: pd.Series,
    preprocessing_mode: str = "native",
    n_jobs: int = -1,
) -> dict[str, object]:
    scale_pos_weight = compute_scale_pos_weight(y_train)

    if backend == "lightgbm":
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": n_jobs,
            "random_state": RANDOM_SEED,
            "metric": "None",
            "verbosity": -1,
        }

    if backend == "xgboost":
        params: dict[str, object] = {
            "objective": "binary:logistic",
            "n_estimators": 2000,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "scale_pos_weight": scale_pos_weight,
            "n_jobs": n_jobs,
            "random_state": RANDOM_SEED,
            "eval_metric": "aucpr",
            "verbosity": 0,
            "tree_method": "hist",
        }
        if preprocessing_mode == "native":
            params["enable_categorical"] = True
        return params

    if backend == "catboost":
        return {
            "loss_function": "Logloss",
            "iterations": 2000,
            "learning_rate": 0.03,
            "depth": 6,
            "min_data_in_leaf": 50,
            "subsample": 0.8,
            "rsm": 0.8,
            "l2_leaf_reg": 0.0,
            "class_weights": [1.0, scale_pos_weight],
            "random_seed": RANDOM_SEED,
            "verbose": False,
            "eval_metric": "PRAUC",
            "allow_writing_files": False,
            "thread_count": n_jobs if n_jobs > 0 else -1,
        }

    raise ValueError(f"Unsupported backend: {backend}")


def _require_backend(backend: str) -> None:
    if backend == "lightgbm" and lgb is None:
        raise SystemExit(
            "LightGBM is not installed. Install project requirements with "
            "`pip install -r requirements.txt`."
        )
    if backend == "xgboost" and xgb is None:
        raise SystemExit(
            "XGBoost is not installed. Install project requirements with "
            "`pip install -r requirements.txt`."
        )
    if backend == "catboost" and CatBoostClassifier is None:
        raise SystemExit(
            "CatBoost is not installed. Install project requirements with "
            "`pip install -r requirements.txt`."
        )


def fit_model(
    prepared: PreparedMatrices,
    params: dict[str, object],
    log_period: int = 50,
) -> Any:
    backend = prepared.backend
    _require_backend(backend)

    if backend == "lightgbm":
        model = lgb.LGBMClassifier(**params)
        model.fit(
            prepared.X_train,
            prepared.y_train,
            eval_set=[(prepared.X_valid, prepared.y_valid)],
            eval_metric=[average_precision_eval, roc_auc_eval],
            categorical_feature=prepared.categorical_columns,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=False,
                ),
                lgb.log_evaluation(period=log_period),
            ],
        )
        return model

    if backend == "xgboost":
        fit_params = dict(params)
        fit_params.setdefault("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        model = xgb.XGBClassifier(**fit_params)
        model.fit(
            prepared.X_train,
            prepared.y_train,
            eval_set=[(prepared.X_valid, prepared.y_valid)],
            verbose=log_period > 0,
        )
        return model

    if backend == "catboost":
        train_pool = Pool(
            prepared.X_train,
            prepared.y_train,
            cat_features=prepared.cat_feature_indices,
        )
        valid_pool = Pool(
            prepared.X_valid,
            prepared.y_valid,
            cat_features=prepared.cat_feature_indices,
        )
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
            verbose=log_period > 0,
        )
        return model

    raise ValueError(f"Unsupported backend: {backend}")


def best_iteration_for(model: Any, backend: str, params: dict[str, object]) -> int:
    if backend == "lightgbm":
        return int(model.best_iteration_ or params.get("n_estimators", 2000))
    if backend == "xgboost":
        if getattr(model, "best_iteration", None) is not None:
            return int(model.best_iteration)
        return int(params.get("n_estimators", 2000))
    if backend == "catboost":
        return int(model.get_best_iteration() or params.get("iterations", 2000))
    raise ValueError(f"Unsupported backend: {backend}")


def predict_positive_proba(
    model: Any,
    X: pd.DataFrame,
    backend: str,
    best_iteration: int,
    cat_feature_indices: list[int] | None,
) -> Any:
    if backend == "lightgbm":
        return model.predict_proba(X, num_iteration=best_iteration)[:, 1]
    if backend == "xgboost":
        return model.predict_proba(
            X,
            iteration_range=(0, best_iteration + 1),
        )[:, 1]
    if backend == "catboost":
        pool = Pool(X, cat_features=cat_feature_indices)
        return model.predict_proba(pool)[:, 1]
    raise ValueError(f"Unsupported backend: {backend}")


def validation_average_precision(
    model: Any,
    prepared: PreparedMatrices,
    params: dict[str, object],
) -> tuple[float, int]:
    best_iteration = best_iteration_for(model, prepared.backend, params)
    valid_score = predict_positive_proba(
        model,
        prepared.X_valid,
        prepared.backend,
        best_iteration,
        prepared.cat_feature_indices,
    )
    score = average_precision_score(prepared.y_valid.to_numpy(), valid_score)
    return float(score), best_iteration


def fixed_trial_params(
    backend: str,
    n_jobs: int,
    preprocessing_mode: str = "native",
) -> dict[str, object]:
    if backend == "lightgbm":
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_jobs": n_jobs,
            "random_state": RANDOM_SEED,
            "metric": "None",
            "verbosity": -1,
        }
    if backend == "xgboost":
        params = {
            "objective": "binary:logistic",
            "n_jobs": n_jobs,
            "random_state": RANDOM_SEED,
            "eval_metric": "aucpr",
            "verbosity": 0,
            "tree_method": "hist",
        }
        if preprocessing_mode == "native":
            params["enable_categorical"] = True
        return params
    if backend == "catboost":
        return {
            "loss_function": "Logloss",
            "random_seed": RANDOM_SEED,
            "verbose": False,
            "eval_metric": "PRAUC",
            "allow_writing_files": False,
            "thread_count": n_jobs if n_jobs > 0 else -1,
        }
    raise ValueError(f"Unsupported backend: {backend}")


def suggest_unified_trial_params(trial, tuning_profile: str) -> dict[str, object]:
    space = TUNING_PROFILES[tuning_profile]
    return {
        name: _suggest_from_spec(trial, name, spec)
        for name, spec in space.items()
    }


def map_trial_params_to_backend(
    backend: str,
    trial_params: dict[str, object],
) -> dict[str, object]:
    scale_pos_weight = float(trial_params["scale_pos_weight"])

    if backend == "lightgbm":
        return {
            "num_leaves": int(trial_params["num_leaves"]),
            "max_depth": int(trial_params["max_depth"]),
            "learning_rate": float(trial_params["learning_rate"]),
            "n_estimators": int(trial_params["n_estimators"]),
            "min_child_samples": int(trial_params["min_child_samples"]),
            "subsample": float(trial_params["subsample"]),
            "subsample_freq": int(trial_params["subsample_freq"]),
            "colsample_bytree": float(trial_params["colsample_bytree"]),
            "reg_alpha": float(trial_params["reg_alpha"]),
            "reg_lambda": float(trial_params["reg_lambda"]),
            "scale_pos_weight": scale_pos_weight,
        }

    if backend == "xgboost":
        max_depth = int(trial_params["max_depth"])
        if max_depth < 0:
            max_depth = int(trial_params["num_leaves"]).bit_length() + 2
        return {
            "max_depth": max_depth,
            "learning_rate": float(trial_params["learning_rate"]),
            "n_estimators": int(trial_params["n_estimators"]),
            "min_child_weight": float(trial_params["min_child_samples"]),
            "subsample": float(trial_params["subsample"]),
            "colsample_bytree": float(trial_params["colsample_bytree"]),
            "reg_alpha": float(trial_params["reg_alpha"]),
            "reg_lambda": float(trial_params["reg_lambda"]),
            "scale_pos_weight": scale_pos_weight,
        }

    if backend == "catboost":
        max_depth = int(trial_params["max_depth"])
        if max_depth < 0:
            depth = min(16, max(4, int(round(trial_params["num_leaves"] ** 0.25 * 2))))
        else:
            depth = max_depth
        return {
            "depth": depth,
            "learning_rate": float(trial_params["learning_rate"]),
            "iterations": int(trial_params["n_estimators"]),
            "min_data_in_leaf": int(trial_params["min_child_samples"]),
            "subsample": float(trial_params["subsample"]),
            "rsm": float(trial_params["colsample_bytree"]),
            "l2_leaf_reg": float(trial_params["reg_lambda"]),
            "class_weights": [1.0, scale_pos_weight],
        }

    raise ValueError(f"Unsupported backend: {backend}")


def save_feature_importance(
    model: Any,
    backend: str,
    output_path: Path,
    feature_names: list[str],
) -> None:
    if backend == "lightgbm":
        booster = model.booster_
        importance = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance_split": booster.feature_importance(
                    importance_type="split"
                ),
                "importance_gain": booster.feature_importance(
                    importance_type="gain"
                ),
            }
        )
    elif backend == "xgboost":
        booster = model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        score_weight = booster.get_score(importance_type="weight")
        importance = pd.DataFrame({"feature": feature_names})
        importance["importance_gain"] = (
            importance["feature"].map(score_gain).fillna(0.0)
        )
        importance["importance_split"] = (
            importance["feature"].map(score_weight).fillna(0.0)
        )
    elif backend == "catboost":
        gain = model.get_feature_importance()
        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_gain": gain,
                "importance_split": gain,
            }
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    importance = importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)
    importance.to_csv(output_path, index=False)


def save_model_artifacts(
    model: Any,
    backend: str,
    output_dir: Path,
    model_stem: str = "model",
) -> None:
    joblib.dump(model, output_dir / f"{model_stem}.pkl")
    if backend == "lightgbm":
        model.booster_.save_model(str(output_dir / f"{model_stem}.txt"))
    elif backend == "xgboost":
        model.save_model(str(output_dir / f"{model_stem}.json"))
    elif backend == "catboost":
        model.save_model(str(output_dir / f"{model_stem}.cbm"))


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload