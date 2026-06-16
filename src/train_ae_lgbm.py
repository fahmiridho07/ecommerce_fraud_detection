"""Train Phase 4 AE-LightGBM using robust Autoencoder latent V-features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AE_LGBM_OUTPUT_DIR,
    AUTOENCODER_ROBUST_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
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
    fit_categorical_mappings,
    get_categorical_columns,
    get_v_feature_columns,
    transform_categorical_columns,
)
from splitting import chronological_split
from utils import ensure_dir, log, save_json, set_seed


DEFAULT_THRESHOLD = 0.5
EARLY_STOPPING_ROUNDS = 100
LATENT_SPLIT_MANIFEST_CSV = "latent_split_manifest.csv"
LATENT_SPLIT_MANIFEST_JSON = "latent_split_manifest_summary.json"
LATENT_SPLIT_MANIFEST_SORT_ORDER = "TransactionDT asc, TransactionID asc"


def average_precision_eval(y_true, y_pred):
    """LightGBM custom validation metric for PR-AUC / Average Precision."""
    return "average_precision", average_precision_score(y_true, y_pred), True


def roc_auc_eval(y_true, y_pred):
    """LightGBM custom validation metric for ROC-AUC."""
    if len(set(y_true)) < 2:
        return "roc_auc", 0.0, True
    return "roc_auc", roc_auc_score(y_true, y_pred), True


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_robust_latent_outputs(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_OUTPUT_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    """Load robust Autoencoder latent arrays and metadata."""
    required_files = {
        "latent_train": autoencoder_output_dir / "latent_train.npy",
        "latent_valid": autoencoder_output_dir / "latent_valid.npy",
        "latent_test": autoencoder_output_dir / "latent_test.npy",
        "latent_feature_names": autoencoder_output_dir / "latent_feature_names.json",
        "v_imputer": autoencoder_output_dir / "v_imputer.pkl",
        "run_config": autoencoder_output_dir / "run_config.json",
    }
    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing robust Autoencoder output(s):\n"
            + "\n".join(missing)
            + "\nRun `python src/train_autoencoder_robust.py` first."
        )

    latent_train = np.load(required_files["latent_train"])
    latent_valid = np.load(required_files["latent_valid"])
    latent_test = np.load(required_files["latent_test"])
    latent_feature_names = load_json(required_files["latent_feature_names"])
    run_config = load_json(required_files["run_config"])

    if not isinstance(latent_feature_names, list):
        raise TypeError("latent_feature_names.json must contain a list of feature names.")
    validate_autoencoder_preprocessing_contract(run_config, autoencoder_output_dir)

    return latent_train, latent_valid, latent_test, latent_feature_names, run_config


def validate_autoencoder_preprocessing_contract(
    run_config: dict[str, object],
    autoencoder_output_dir: Path,
) -> None:
    """Reject stale zero-fill Autoencoder artifacts after the missingness fix."""
    preprocessing = run_config.get("preprocessing", {})
    training = run_config.get("training", {})
    if not isinstance(preprocessing, dict) or not isinstance(training, dict):
        raise ValueError(
            f"{autoencoder_output_dir} has an invalid run_config.json schema."
        )
    missing_strategy = str(preprocessing.get("missing_value_strategy", ""))
    loss_name = str(training.get("loss", ""))
    if "SimpleImputer" not in missing_strategy or loss_name != "masked_mse_loss":
        raise ValueError(
            "Stale Autoencoder artifacts detected. Re-run "
            "`python src/train_autoencoder_robust.py` with the current "
            "median-imputation and masked-loss pipeline before training AE-LightGBM."
        )


def build_latent_split_manifest_frame(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the row manifest used to guard AE latent alignment."""
    manifest_rows: list[dict[str, object]] = []
    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        for row_position, (_, row) in enumerate(split_df.iterrows()):
            manifest_rows.append(
                {
                    "split": split_name,
                    "row_position": row_position,
                    ID_COL: row[ID_COL],
                    TIME_COL: row[TIME_COL],
                    TARGET_COL: int(row[TARGET_COL]),
                }
            )
    return pd.DataFrame(manifest_rows)


def save_latent_split_manifest(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist split row order beside latent arrays for downstream alignment checks."""
    manifest_df = build_latent_split_manifest_frame(train_df, valid_df, test_df)
    manifest_df.to_csv(output_dir / LATENT_SPLIT_MANIFEST_CSV, index=False)

    split_summaries: dict[str, dict[str, object]] = {}
    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        split_summaries[split_name] = {
            "rows": int(len(split_df)),
            "first_transaction_id": int(split_df[ID_COL].iloc[0]),
            "last_transaction_id": int(split_df[ID_COL].iloc[-1]),
        }

    summary = {
        "train_rows": split_summaries["train"]["rows"],
        "valid_rows": split_summaries["validation"]["rows"],
        "test_rows": split_summaries["test"]["rows"],
        "train_first_transaction_id": split_summaries["train"]["first_transaction_id"],
        "valid_first_transaction_id": split_summaries["validation"][
            "first_transaction_id"
        ],
        "test_first_transaction_id": split_summaries["test"]["first_transaction_id"],
        "train_last_transaction_id": split_summaries["train"]["last_transaction_id"],
        "valid_last_transaction_id": split_summaries["validation"]["last_transaction_id"],
        "test_last_transaction_id": split_summaries["test"]["last_transaction_id"],
        "sort_order": LATENT_SPLIT_MANIFEST_SORT_ORDER,
    }
    save_json(summary, output_dir / LATENT_SPLIT_MANIFEST_JSON)


def load_latent_split_manifest(autoencoder_output_dir: Path) -> pd.DataFrame:
    """Load the AE latent split manifest saved during robust Autoencoder training."""
    manifest_path = autoencoder_output_dir / LATENT_SPLIT_MANIFEST_CSV
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Missing latent split manifest:\n"
            f"{manifest_path}\n"
            "Re-run `python src/train_autoencoder_robust.py` for the matching "
            "autoencoder output directory before training AE-LightGBM."
        )
    manifest_df = pd.read_csv(manifest_path)
    required_columns = {"split", "row_position", ID_COL, TIME_COL, TARGET_COL}
    missing_columns = sorted(required_columns - set(manifest_df.columns))
    if missing_columns:
        raise ValueError(
            "Latent split manifest is missing required column(s): "
            + ", ".join(missing_columns)
        )
    return manifest_df


def validate_latent_split_manifest_alignment(
    autoencoder_output_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Fail fast when current split TransactionID order differs from AE latent manifest."""
    manifest_df = load_latent_split_manifest(autoencoder_output_dir)
    split_frames = {
        "train": train_df,
        "validation": valid_df,
        "test": test_df,
    }
    for split_name, split_df in split_frames.items():
        split_manifest = manifest_df.loc[
            manifest_df["split"] == split_name
        ].sort_values("row_position")
        if len(split_manifest) != len(split_df):
            raise ValueError(
                f"{split_name} manifest row count {len(split_manifest)} does not match "
                f"current split row count {len(split_df)} for "
                f"{autoencoder_output_dir}."
            )

        expected_ids = split_manifest[ID_COL].to_numpy()
        actual_ids = split_df[ID_COL].to_numpy()
        if not np.array_equal(expected_ids, actual_ids):
            mismatch_index = int(np.argmax(expected_ids != actual_ids))
            raise ValueError(
                f"{split_name} TransactionID order does not match latent manifest for "
                f"{autoencoder_output_dir}. First mismatch at row {mismatch_index}: "
                f"manifest={expected_ids[mismatch_index]!r}, "
                f"current={actual_ids[mismatch_index]!r}. "
                "Latent row i must align to split row i; rerun the matching "
                "Autoencoder training or restore the frozen chronological split."
            )


def validate_latent_outputs(
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray,
    latent_feature_names: list[str],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    """Validate robust latent arrays align to the Phase 1 temporal split."""
    expected = {
        "train": (latent_train, train_rows),
        "validation": (latent_valid, valid_rows),
        "test": (latent_test, test_rows),
    }
    for split_name, (latent, row_count) in expected.items():
        if latent.shape[0] != row_count:
            raise ValueError(
                f"{split_name} latent row count {latent.shape[0]} does not match "
                f"split row count {row_count}."
            )

    latent_dim = latent_train.shape[1]
    if latent_valid.shape[1] != latent_dim or latent_test.shape[1] != latent_dim:
        raise ValueError("Latent arrays do not have the same number of columns.")
    if len(latent_feature_names) != latent_dim:
        raise ValueError(
            "Latent feature name count does not match latent array dimension: "
            f"{len(latent_feature_names)} vs {latent_dim}."
        )
    if len(set(latent_feature_names)) != len(latent_feature_names):
        raise ValueError("Duplicate latent feature names found.")


def load_top_v_features_from_importance(
    importance_path: Path,
    top_k: int,
    v_columns: list[str],
) -> list[str]:
    """Select the top-K V-features by baseline LightGBM gain."""
    if top_k <= 0:
        raise ValueError("retain_top_v_features must be a positive integer.")
    if not importance_path.exists():
        raise FileNotFoundError(f"Missing baseline feature importance file: {importance_path}")

    importance = pd.read_csv(importance_path)
    if "feature" not in importance.columns or "importance_gain" not in importance.columns:
        raise ValueError(
            f"{importance_path} must contain feature and importance_gain columns."
        )
    allowed_v = set(v_columns)
    ranked_v = (
        importance.loc[importance["feature"].isin(allowed_v)]
        .sort_values(["importance_gain", "importance_split"], ascending=False)
        .drop_duplicates(subset=["feature"])
    )
    if ranked_v.empty:
        raise ValueError(
            f"No V-features were found in baseline importance file: {importance_path}"
        )
    if len(ranked_v) < top_k:
        raise ValueError(
            f"Requested top {top_k} V-features, but only {len(ranked_v)} are available."
        )
    return ranked_v.head(top_k)["feature"].tolist()


def resolve_replaced_v_columns(
    v_columns: list[str],
    retained_v_columns: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return replaced and retained V-feature lists for hybrid AE-LightGBM runs."""
    if not retained_v_columns:
        return list(v_columns), []
    retained = list(retained_v_columns)
    unknown = sorted(set(retained) - set(v_columns))
    if unknown:
        raise ValueError(
            "retained_v_columns contains unknown V-features: " + ", ".join(unknown[:10])
        )
    replaced = [column for column in v_columns if column not in set(retained)]
    return replaced, retained


def split_non_v_features_target(
    df: pd.DataFrame,
    excluded_v_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features excluding selected V columns plus target labels."""
    y = df[TARGET_COL].astype(int).copy()
    excluded = set(excluded_v_columns + [TARGET_COL, ID_COL])
    feature_columns = [column for column in df.columns if column not in excluded]
    return df.loc[:, feature_columns].copy(), y


def build_retained_v_features(
    df: pd.DataFrame,
    retained_v_columns: list[str],
) -> pd.DataFrame:
    """Keep selected original V-features with NaN preserved for LightGBM."""
    return df.loc[:, retained_v_columns].copy()


def fit_non_v_preprocessing(
    X_train: pd.DataFrame,
    replaced_v_columns: list[str],
    retained_v_columns: list[str] | None = None,
) -> dict[str, object]:
    """Fit categorical mappings on train non-V features only."""
    categorical_columns = get_categorical_columns(X_train)
    categorical_mappings = fit_categorical_mappings(X_train, categorical_columns)
    return {
        "feature_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_mappings": categorical_mappings,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "dropped_columns": [ID_COL],
        "excluded_original_v_features": replaced_v_columns,
        "retained_original_v_features": list(retained_v_columns or []),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
    }


def apply_non_v_preprocessing(
    X: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    """Apply train-fitted non-V categorical mappings."""
    feature_columns = preprocessing["feature_columns"]
    categorical_mappings = preprocessing["categorical_mappings"]
    X = X.loc[:, feature_columns].copy()
    return transform_categorical_columns(X, categorical_mappings)


def combine_non_v_and_latent(
    X_non_v: pd.DataFrame,
    latent: np.ndarray,
    latent_feature_names: list[str],
    missing_indicators: pd.DataFrame | None = None,
    retained_v_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Concatenate non-V, latent, retained V-values, and missing indicators."""
    latent_df = pd.DataFrame(latent, columns=latent_feature_names)
    parts = [X_non_v.reset_index(drop=True), latent_df.reset_index(drop=True)]
    if retained_v_features is not None:
        parts.append(retained_v_features.reset_index(drop=True))
    if missing_indicators is not None:
        parts.append(missing_indicators.reset_index(drop=True))
    return pd.concat(parts, axis=1)


def v_missing_indicator_names(v_columns: list[str]) -> list[str]:
    return [f"v_missing_{column}" for column in v_columns]


def build_v_missing_indicators(
    df: pd.DataFrame,
    v_columns: list[str],
) -> pd.DataFrame:
    """Preserve original V-feature missingness as supervised LightGBM inputs."""
    return df.loc[:, v_columns].isna().astype("int8").set_axis(
        v_missing_indicator_names(v_columns),
        axis=1,
    )


def validate_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    replaced_v_columns: list[str],
    retained_v_columns: list[str] | None = None,
) -> None:
    """Ensure matrices align and only allowed original V-features are present."""
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    allowed_v = set(retained_v_columns or [])
    leaked_v_columns = sorted(
        (set(X_train.columns) & set(replaced_v_columns)) - allowed_v
    )
    if leaked_v_columns:
        raise ValueError(
            "Replaced V-features were found in final AE-LightGBM features: "
            + ", ".join(leaked_v_columns[:10])
        )
    if retained_v_columns:
        missing_retained = sorted(allowed_v - set(X_train.columns))
        if missing_retained:
            raise ValueError(
                "Retained V-features are missing from final AE-LightGBM features: "
                + ", ".join(missing_retained[:10])
            )


def build_model_params(y_train: pd.Series) -> dict[str, object]:
    """Build fixed LightGBM parameters consistent with the Phase 2 baseline."""
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

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
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "metric": "None",
        "verbosity": -1,
    }


def save_feature_importance(model: lgb.LGBMClassifier, output_path: Path) -> None:
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    )
    importance = importance.sort_values(
        ["importance_gain", "importance_split"],
        ascending=False,
    ).reset_index(drop=True)
    importance.to_csv(output_path, index=False)


def load_baseline_selected_metrics() -> dict[str, float] | None:
    baseline_path = BASELINE_OUTPUT_DIR / "metrics_test_selected_threshold.json"
    if not baseline_path.exists():
        return None
    return load_json(baseline_path)


def load_baseline_metrics_from_path(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    return load_json(path)


def build_baseline_comparison(
    ae_metrics: dict[str, object],
    baseline_metrics_path: Path | None = None,
) -> dict[str, float] | None:
    baseline_metrics = (
        load_baseline_metrics_from_path(baseline_metrics_path)
        if baseline_metrics_path is not None
        else load_baseline_selected_metrics()
    )
    if baseline_metrics is None:
        return None

    return {
        "baseline_test_pr_auc": baseline_metrics["average_precision"],
        "ae_lgbm_test_pr_auc": ae_metrics["average_precision"],
        "delta_pr_auc": ae_metrics["average_precision"] - baseline_metrics["average_precision"],
        "baseline_test_roc_auc": baseline_metrics["roc_auc"],
        "ae_lgbm_test_roc_auc": ae_metrics["roc_auc"],
        "delta_roc_auc": ae_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        "baseline_test_f1": baseline_metrics["f1"],
        "ae_lgbm_test_f1": ae_metrics["f1"],
        "delta_f1": ae_metrics["f1"] - baseline_metrics["f1"],
        "baseline_test_mcc": baseline_metrics["mcc"],
        "ae_lgbm_test_mcc": ae_metrics["mcc"],
        "delta_mcc": ae_metrics["mcc"] - baseline_metrics["mcc"],
    }


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    save_json(metrics, path)


@dataclass
class AELGBMTrainingData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    categorical_columns: list[str]
    preprocessing_non_v: dict[str, object]
    v_columns: list[str]
    replaced_v_columns: list[str]
    retained_v_columns: list[str]
    latent_feature_names: list[str]
    robust_ae_run_config: dict[str, object]
    missing_indicator_names: list[str]
    retain_top_v_features: int | None
    baseline_importance_path: str | None

    @property
    def total_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def representation_mode(self) -> str:
        if self.retained_v_columns:
            return "hybrid_latent_plus_top_v_retention"
        return "full_latent_replacement"


def prepare_ae_lgbm_training_data(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_OUTPUT_DIR,
    retain_top_v_features: int | None = None,
    baseline_importance_path: Path | None = None,
) -> AELGBMTrainingData:
    """Build AE-LightGBM feature matrices without fitting the classifier."""
    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    v_columns = get_v_feature_columns(train_df)
    retained_v_columns: list[str] = []
    if retain_top_v_features is not None:
        if baseline_importance_path is None:
            raise ValueError(
                "baseline_importance_path is required when retain_top_v_features is set."
            )
        retained_v_columns = load_top_v_features_from_importance(
            baseline_importance_path,
            retain_top_v_features,
            v_columns,
        )
        log(
            "Retaining top baseline V-features alongside AE latents: "
            f"{len(retained_v_columns)} columns."
        )
    replaced_v_columns, retained_v_columns = resolve_replaced_v_columns(
        v_columns,
        retained_v_columns,
    )

    log("Loading robust Autoencoder latent features.")
    (
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        robust_ae_run_config,
    ) = load_robust_latent_outputs(autoencoder_output_dir)
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    validate_latent_split_manifest_alignment(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
    )

    log("Building non-V feature matrices and fitting train-only preprocessing.")
    X_train_non_v_raw, y_train = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, y_valid = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test = split_non_v_features_target(test_df, v_columns)

    preprocessing_non_v = fit_non_v_preprocessing(
        X_train_non_v_raw,
        replaced_v_columns,
        retained_v_columns,
    )
    X_train_non_v = apply_non_v_preprocessing(X_train_non_v_raw, preprocessing_non_v)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing_non_v)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing_non_v)

    retained_train = (
        build_retained_v_features(train_df, retained_v_columns)
        if retained_v_columns
        else None
    )
    retained_valid = (
        build_retained_v_features(valid_df, retained_v_columns)
        if retained_v_columns
        else None
    )
    retained_test = (
        build_retained_v_features(test_df, retained_v_columns)
        if retained_v_columns
        else None
    )

    log("Combining processed non-V features with robust latent V features.")
    missing_train = build_v_missing_indicators(train_df, replaced_v_columns)
    missing_valid = build_v_missing_indicators(valid_df, replaced_v_columns)
    missing_test = build_v_missing_indicators(test_df, replaced_v_columns)
    missing_indicator_names = missing_train.columns.tolist()
    X_train = combine_non_v_and_latent(
        X_train_non_v,
        latent_train,
        latent_feature_names,
        missing_train,
        retained_train,
    )
    X_valid = combine_non_v_and_latent(
        X_valid_non_v,
        latent_valid,
        latent_feature_names,
        missing_valid,
        retained_valid,
    )
    X_test = combine_non_v_and_latent(
        X_test_non_v,
        latent_test,
        latent_feature_names,
        missing_test,
        retained_test,
    )
    validate_feature_alignment(
        X_train,
        X_valid,
        X_test,
        replaced_v_columns,
        retained_v_columns,
    )

    return AELGBMTrainingData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=preprocessing_non_v["categorical_columns"],
        preprocessing_non_v=preprocessing_non_v,
        v_columns=v_columns,
        replaced_v_columns=replaced_v_columns,
        retained_v_columns=retained_v_columns,
        latent_feature_names=latent_feature_names,
        robust_ae_run_config=robust_ae_run_config,
        missing_indicator_names=missing_indicator_names,
        retain_top_v_features=retain_top_v_features,
        baseline_importance_path=(
            str(baseline_importance_path) if baseline_importance_path else None
        ),
    )


def main(
    autoencoder_output_dir: Path = AUTOENCODER_ROBUST_OUTPUT_DIR,
    output_dir: Path = AE_LGBM_OUTPUT_DIR,
    phase_name: str = "4_ae_lgbm",
    retain_top_v_features: int | None = None,
    baseline_importance_path: Path | None = None,
    baseline_metrics_path: Path | None = None,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    prepared = prepare_ae_lgbm_training_data(
        autoencoder_output_dir=autoencoder_output_dir,
        retain_top_v_features=retain_top_v_features,
        baseline_importance_path=baseline_importance_path,
    )
    X_train = prepared.X_train
    X_valid = prepared.X_valid
    X_test = prepared.X_test
    y_train = prepared.y_train
    y_valid = prepared.y_valid
    y_test = prepared.y_test
    preprocessing_non_v = prepared.preprocessing_non_v
    v_columns = prepared.v_columns
    replaced_v_columns = prepared.replaced_v_columns
    retained_v_columns = prepared.retained_v_columns
    latent_feature_names = prepared.latent_feature_names
    robust_ae_run_config = prepared.robust_ae_run_config
    missing_indicator_names = prepared.missing_indicator_names
    categorical_columns = prepared.categorical_columns
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training AE-LightGBM with validation early stopping.")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[average_precision_eval, roc_auc_eval],
        categorical_feature=categorical_columns,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iteration = int(model.best_iteration_ or model.n_estimators)

    log("Generating validation and test probabilities.")
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]

    log("Selecting classification threshold on validation only.")
    threshold_table = threshold_selection_table(y_valid.to_numpy(), valid_score)
    selected_threshold = selected_threshold_from_table(threshold_table)
    threshold_table.to_csv(output_dir / "threshold_selection.csv", index=False)

    metrics_valid_default = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        DEFAULT_THRESHOLD,
    )
    metrics_valid_selected = binary_classification_metrics(
        y_valid.to_numpy(),
        valid_score,
        selected_threshold,
    )
    metrics_test_default = binary_classification_metrics(
        y_test.to_numpy(),
        test_score,
        DEFAULT_THRESHOLD,
    )
    metrics_test_selected = binary_classification_metrics(
        y_test.to_numpy(),
        test_score,
        selected_threshold,
    )

    log("Saving AE-LightGBM outputs.")
    save_metrics(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_metrics(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_metrics(
        metrics_test_default,
        output_dir / "metrics_test_default_threshold.json",
    )
    save_metrics(
        metrics_test_selected,
        output_dir / "metrics_test_selected_threshold.json",
    )

    confusion_matrix_table(
        y_valid.to_numpy(),
        valid_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "validation",
    ).to_csv(output_dir / "confusion_matrix_validation.csv", index=False)
    confusion_matrix_table(
        y_test.to_numpy(),
        test_score,
        {"default": DEFAULT_THRESHOLD, "selected": selected_threshold},
        "test",
    ).to_csv(output_dir / "confusion_matrix_test.csv", index=False)

    save_feature_importance(model, output_dir / "feature_importance.csv")
    joblib.dump(model, output_dir / "model.pkl")
    model.booster_.save_model(str(output_dir / "model.txt"))
    joblib.dump(preprocessing_non_v, output_dir / "preprocessing_non_v.pkl")

    robust_preprocessing = robust_ae_run_config.get("preprocessing", {})
    feature_set_summary = {
        "number_of_non_v_features": int(len(preprocessing_non_v["feature_columns"])),
        "number_of_latent_v_features": int(len(latent_feature_names)),
        "number_of_retained_original_v_features": int(len(retained_v_columns)),
        "number_of_v_missing_indicator_features": int(len(missing_indicator_names)),
        "total_final_features": int(X_train.shape[1]),
        "original_v_features_fully_replaced": not bool(retained_v_columns),
        "retained_original_v_features_included": bool(retained_v_columns),
        "v_missing_indicators_included": True,
        "number_of_original_v_features_excluded": int(len(replaced_v_columns)),
        "robust_autoencoder_output_path_used": str(autoencoder_output_dir),
        "robust_autoencoder_clipping": {
            "enabled": robust_preprocessing.get("scaled_clipping_enabled"),
            "clip_min": robust_preprocessing.get("clip_min"),
            "clip_max": robust_preprocessing.get("clip_max"),
        },
    }
    save_json(feature_set_summary, output_dir / "feature_set_summary.json")
    if retained_v_columns:
        save_json(retained_v_columns, output_dir / "retained_v_features.json")

    comparison = build_baseline_comparison(
        metrics_test_selected,
        baseline_metrics_path=baseline_metrics_path,
    )
    if comparison is not None:
        save_json(comparison, output_dir / "comparison_against_baseline.json")

    run_config = {
        "phase": phase_name,
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "transactiondt_note": (
            "TransactionDT is kept as a non-V model feature and was also used "
            "to create the chronological split."
        ),
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "feature_construction": {
            "representation_mode": (
                "hybrid_latent_plus_top_v_retention"
                if retained_v_columns
                else "full_latent_replacement"
            ),
            "retain_top_v_features": retain_top_v_features,
            "baseline_importance_path": (
                str(baseline_importance_path) if baseline_importance_path else None
            ),
            "retained_original_v_features": retained_v_columns,
            "replaced_original_v_feature_count": len(replaced_v_columns),
            "original_v_feature_count": len(v_columns),
            "non_v_feature_count": int(len(preprocessing_non_v["feature_columns"])),
            "latent_feature_count": len(latent_feature_names),
            "retained_original_v_feature_count": len(retained_v_columns),
            "v_missing_indicator_count": len(missing_indicator_names),
            "v_missing_indicators_included": True,
            "total_feature_count": int(X_train.shape[1]),
            "robust_autoencoder_output_dir": str(autoencoder_output_dir),
        },
        "preprocessing": {
            "non_v_categorical_fit": "Categorical mappings fit on train non-V features only.",
            "categorical_columns": categorical_columns,
            "categorical_columns_count": len(categorical_columns),
            "categorical_missing_value": MISSING_CATEGORY,
            "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
            "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
        },
        "threshold_selection": {
            "source_split": "validation",
            "search_range": "0.01 to 0.99 step 0.01",
            "selection_metric": "MCC, with F1 as tie-breaker",
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": selected_threshold,
        },
        "early_stopping": {
            "validation_split": "validation",
            "metric": "average_precision",
            "stopping_rounds": EARLY_STOPPING_ROUNDS,
            "best_iteration": best_iteration,
        },
        "class_imbalance": {
            "method": "scale_pos_weight",
            "computed_from": "training labels only",
            "value": model_params["scale_pos_weight"],
        },
        "model_params": model_params,
    }
    save_json(run_config, output_dir / "run_config.json")

    print()
    print("AE-LightGBM Summary")
    print("===================")
    print(f"Validation PR-AUC : {metrics_valid_selected['average_precision']:.6f}")
    print(f"Test PR-AUC       : {metrics_test_selected['average_precision']:.6f}")
    print(f"Validation ROC-AUC: {metrics_valid_selected['roc_auc']:.6f}")
    print(f"Test ROC-AUC      : {metrics_test_selected['roc_auc']:.6f}")
    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Test precision    : {metrics_test_selected['precision']:.6f}")
    print(f"Test recall       : {metrics_test_selected['recall']:.6f}")
    print(f"Test F1           : {metrics_test_selected['f1']:.6f}")
    print(f"Test MCC          : {metrics_test_selected['mcc']:.6f}")
    print(f"Best iteration    : {best_iteration}")
    print(f"Outputs saved to  : {output_dir}")

    if comparison is not None:
        print()
        print("Comparison Against Baseline")
        print("===========================")
        print(f"Delta test PR-AUC : {comparison['delta_pr_auc']:+.6f}")
        print(f"Delta test ROC-AUC: {comparison['delta_roc_auc']:+.6f}")
        print(f"Delta test F1     : {comparison['delta_f1']:+.6f}")
        print(f"Delta test MCC    : {comparison['delta_mcc']:+.6f}")

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
        "comparison_against_baseline": comparison,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AE-LightGBM using robust Autoencoder latent V-features."
    )
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=AUTOENCODER_ROBUST_OUTPUT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=AE_LGBM_OUTPUT_DIR)
    parser.add_argument("--phase-name", default="4_ae_lgbm")
    parser.add_argument(
        "--retain-top-v-features",
        type=int,
        default=None,
        help=(
            "Retain the top-K original V-features by baseline gain alongside AE "
            "latents and missing indicators for the remaining V columns."
        ),
    )
    parser.add_argument(
        "--baseline-importance-path",
        type=Path,
        default=None,
        help="Baseline feature_importance.csv used to rank retained V-features.",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        type=Path,
        default=None,
        help="Optional baseline metrics JSON for comparison_against_baseline.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        autoencoder_output_dir=args.autoencoder_output_dir,
        output_dir=args.output_dir,
        phase_name=args.phase_name,
        retain_top_v_features=args.retain_top_v_features,
        baseline_importance_path=args.baseline_importance_path,
        baseline_metrics_path=args.baseline_metrics_path,
    )
