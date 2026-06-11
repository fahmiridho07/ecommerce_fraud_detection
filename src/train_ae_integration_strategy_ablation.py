"""Train fixed/default LightGBM variants for AE integration strategy ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "LightGBM is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from config import (
    AE_BATCH_SIZE,
    DATA_DIR,
    ID_COL,
    OUTPUT_DIR,
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
    apply_baseline_preprocessing,
    fit_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from splitting import chronological_split
from train_ae_lgbm import (
    apply_non_v_preprocessing,
    combine_non_v_and_latent,
    fit_non_v_preprocessing,
    load_robust_latent_outputs,
    split_non_v_features_target,
    validate_feature_alignment as validate_latent_feature_alignment,
    validate_latent_outputs,
    validate_latent_split_manifest_alignment,
)
from train_baseline_lgbm import (
    DEFAULT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    average_precision_eval,
    build_model_params,
    roc_auc_eval,
    save_feature_importance,
)
from utils import ensure_dir, log, save_json, set_seed


ABLATION_BASE_OUTPUT_DIR = OUTPUT_DIR / "ae_integration_strategy_ablation"

SUPPORTED_VARIANTS = (
    "baseline_fixed",
    "du_latent_replacement",
    "ding_reconstructed_replacement",
    "reconstruction_error_augmentation",
)

VARIANT_DEFAULT_SUBDIRS = {
    "baseline_fixed": "baseline_fixed",
    "du_latent_replacement": "du_latent_replacement",
    "ding_reconstructed_replacement": "ding_reconstructed_replacement",
    "reconstruction_error_augmentation": "reconstruction_error_augmentation",
}

VARIANT_DEFAULT_PHASE_NAMES = {
    "baseline_fixed": "STR_B0_baseline_fixed",
    "du_latent_replacement": "STR_AE1_du_latent_replacement",
    "ding_reconstructed_replacement": "STR_AE2_ding_reconstructed_replacement",
    "reconstruction_error_augmentation": "STR_AE3_reconstruction_error_augmentation",
}

RECONSTRUCTION_ERROR_FEATURES = (
    "v_ae_reconstruction_mse",
    "v_ae_reconstruction_log1p_mse",
)
RECONSTRUCTED_V_NAME_PREFIX = "ae_reconstructed_"
SUPPORTED_RECONSTRUCTION_SPACES = ("scaled",)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def default_output_dir(variant: str) -> Path:
    return ABLATION_BASE_OUTPUT_DIR / VARIANT_DEFAULT_SUBDIRS[variant]


def requires_autoencoder_output_dir(variant: str) -> bool:
    return variant != "baseline_fixed"


def reconstructed_v_feature_names(v_columns: list[str]) -> list[str]:
    return [f"{RECONSTRUCTED_V_NAME_PREFIX}{column}" for column in v_columns]


def reconstruction_error_feature_names() -> list[str]:
    return list(RECONSTRUCTION_ERROR_FEATURES)


def validate_v_columns_match_ae_run_config(
    current_v_columns: list[str],
    saved_v_columns: list[str],
    autoencoder_output_dir: Path,
) -> None:
    """Fail fast when AE run_config V scope does not match the current split."""
    if saved_v_columns == current_v_columns:
        return

    current_set = set(current_v_columns)
    saved_set = set(saved_v_columns)
    missing_in_saved = sorted(current_set - saved_set)
    extra_in_saved = sorted(saved_set - current_set)

    positional_differences: list[str] = []
    for index in range(min(len(current_v_columns), len(saved_v_columns))):
        if current_v_columns[index] != saved_v_columns[index]:
            positional_differences.append(
                f"index {index}: current={current_v_columns[index]!r}, "
                f"saved={saved_v_columns[index]!r}"
            )

    detail_parts: list[str] = []
    if missing_in_saved:
        detail_parts.append(
            "missing in saved AE run_config: " + ", ".join(missing_in_saved[:10])
        )
    if extra_in_saved:
        detail_parts.append(
            "extra in saved AE run_config: " + ", ".join(extra_in_saved[:10])
        )
    if positional_differences:
        detail_parts.append(
            "positional differences: " + "; ".join(positional_differences[:5])
        )

    raise ValueError(
        "V column scope mismatch between current chronological split and AE run_config: "
        f"current V columns={len(current_v_columns)}, "
        f"saved V columns={len(saved_v_columns)}, "
        f"autoencoder_output_dir={autoencoder_output_dir}. "
        + "; ".join(detail_parts)
    )


def load_ae_run_config_v_columns(autoencoder_output_dir: Path) -> list[str]:
    """Load and validate the V column list stored in AE run_config.json."""
    run_config_path = autoencoder_output_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing AE run_config: {run_config_path}")

    run_config = load_json(run_config_path)
    if not isinstance(run_config, dict):
        raise TypeError(f"Expected JSON object in {run_config_path}")

    saved_v_columns = run_config.get("v_columns")
    if not isinstance(saved_v_columns, list) or not saved_v_columns:
        raise ValueError(
            f"{run_config_path} is missing a non-empty v_columns list."
        )
    return saved_v_columns


def apply_saved_v_preprocessing(
    df: pd.DataFrame,
    v_columns: list[str],
    scaler,
    ae_run_config: dict[str, object],
) -> np.ndarray:
    """Transform V columns with train-fitted AE scaler and saved clipping settings."""
    preprocessing = ae_run_config.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        preprocessing = {}

    use_clipping = bool(preprocessing.get("scaled_clipping_enabled", False))
    clip_min = float(preprocessing.get("clip_min", -5.0))
    clip_max = float(preprocessing.get("clip_max", 5.0))

    X_raw = df.loc[:, v_columns].fillna(0).astype("float32")
    X_scaled = scaler.transform(X_raw).astype("float32")
    if use_clipping:
        X_scaled = np.clip(X_scaled, clip_min, clip_max).astype("float32")
    return X_scaled


def load_saved_autoencoder_artifacts(
    autoencoder_output_dir: Path,
) -> tuple[object, object, dict[str, object], list[str]]:
    """Load frozen V-only AE model, scaler, run config, and V column list."""
    model_path = autoencoder_output_dir / "autoencoder_model.keras"
    scaler_path = autoencoder_output_dir / "v_scaler.pkl"
    run_config_path = autoencoder_output_dir / "run_config.json"
    missing = [
        str(path)
        for path in (model_path, scaler_path, run_config_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Autoencoder artifact(s) required for this strategy:\n"
            + "\n".join(missing)
        )

    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "TensorFlow is not installed. Install project requirements with "
            "`pip install -r requirements.txt`, then rerun this script."
        ) from exc

    run_config = load_json(run_config_path)
    if not isinstance(run_config, dict):
        raise TypeError(f"Expected JSON object in {run_config_path}")

    v_columns = run_config.get("v_columns")
    if not isinstance(v_columns, list) or not v_columns:
        raise ValueError(
            f"{run_config_path} is missing a non-empty v_columns list."
        )

    autoencoder = keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return autoencoder, scaler, run_config, v_columns


def generate_decoder_reconstructed_v(
    autoencoder,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """Return decoder reconstruction in scaled representation space."""
    reconstructed = autoencoder.predict(X_scaled, batch_size=AE_BATCH_SIZE, verbose=0)
    if not np.isfinite(reconstructed).all():
        raise ValueError("Decoder reconstruction contains non-finite values.")
    return reconstructed.astype("float32")


def reconstruction_error_file_paths(source_dir: Path) -> dict[str, Path]:
    return {
        "train": source_dir / "reconstruction_error_train.csv",
        "validation": source_dir / "reconstruction_error_valid.csv",
        "test": source_dir / "reconstruction_error_test.csv",
    }


def load_reconstruction_error_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    if "reconstruction_mse" not in df.columns:
        raise KeyError(f"{path} is missing reconstruction_mse column.")

    values = df["reconstruction_mse"].to_numpy(dtype="float32")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite reconstruction errors.")
    if np.any(values < 0):
        raise ValueError(f"{path} contains negative reconstruction errors.")
    return values


def load_or_compute_reconstruction_errors(
    autoencoder_output_dir: Path,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    v_columns: list[str],
) -> dict[str, np.ndarray]:
    """Load saved reconstruction-error CSVs or compute from frozen AE artifacts."""
    saved_v_columns = load_ae_run_config_v_columns(autoencoder_output_dir)
    validate_v_columns_match_ae_run_config(
        v_columns,
        saved_v_columns,
        autoencoder_output_dir,
    )

    paths = reconstruction_error_file_paths(autoencoder_output_dir)
    if all(path.exists() for path in paths.values()):
        return {
            split_name: load_reconstruction_error_csv(path)
            for split_name, path in paths.items()
        }

    log(
        "Reconstruction-error CSVs not found; computing from saved AE artifacts "
        "without fitting on validation/test."
    )
    autoencoder, scaler, ae_run_config, _ = load_saved_autoencoder_artifacts(
        autoencoder_output_dir
    )

    split_frames = {
        "train": train_df,
        "validation": valid_df,
        "test": test_df,
    }
    errors: dict[str, np.ndarray] = {}
    for split_name, split_df in split_frames.items():
        X_scaled = apply_saved_v_preprocessing(
            split_df,
            v_columns,
            scaler,
            ae_run_config,
        )
        reconstructed = generate_decoder_reconstructed_v(autoencoder, X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1).astype("float32")
        if not np.isfinite(mse).all():
            raise ValueError(
                f"{split_name} computed reconstruction errors contain non-finite values."
            )
        errors[split_name] = mse
    return errors


def validate_reconstruction_error_lengths(
    errors: dict[str, np.ndarray],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
) -> None:
    expected = {
        "train": train_rows,
        "validation": valid_rows,
        "test": test_rows,
    }
    for split_name, row_count in expected.items():
        if errors[split_name].shape[0] != row_count:
            raise ValueError(
                f"{split_name} reconstruction-error length "
                f"{errors[split_name].shape[0]} does not match split rows "
                f"{row_count}."
            )


def reconstruction_error_dataframe(errors: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            RECONSTRUCTION_ERROR_FEATURES[0]: errors.astype("float32"),
            RECONSTRUCTION_ERROR_FEATURES[1]: np.log1p(errors).astype("float32"),
        }
    )


def combine_with_reconstruction_errors(
    X: pd.DataFrame,
    errors: np.ndarray,
) -> pd.DataFrame:
    error_df = reconstruction_error_dataframe(errors)
    return pd.concat(
        [X.reset_index(drop=True), error_df.reset_index(drop=True)],
        axis=1,
    )


def reconstructed_v_dataframe(
    reconstructed: np.ndarray,
    v_columns: list[str],
) -> pd.DataFrame:
    column_names = reconstructed_v_feature_names(v_columns)
    if reconstructed.shape[1] != len(column_names):
        raise ValueError(
            f"Reconstructed column count {reconstructed.shape[1]} does not match "
            f"V column count {len(column_names)}."
        )
    return pd.DataFrame(reconstructed, columns=column_names)


def combine_non_v_and_reconstructed_v(
    X_non_v: pd.DataFrame,
    reconstructed: np.ndarray,
    v_columns: list[str],
) -> pd.DataFrame:
    reconstructed_df = reconstructed_v_dataframe(reconstructed, v_columns)
    return pd.concat(
        [X_non_v.reset_index(drop=True), reconstructed_df.reset_index(drop=True)],
        axis=1,
    )


def validate_reconstructed_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
    reconstructed_columns: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    leaked_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if leaked_v_columns:
        raise ValueError(
            "Original V-features were found in final matrix: "
            + ", ".join(leaked_v_columns[:10])
        )

    missing_reconstructed = [
        column for column in reconstructed_columns if column not in X_train.columns
    ]
    if missing_reconstructed:
        raise ValueError(
            "Missing reconstructed V feature(s): " + ", ".join(missing_reconstructed[:10])
        )


def validate_reconstruction_error_feature_alignment(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    v_columns: list[str],
) -> None:
    if X_valid.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Validation feature columns do not align with train columns.")
    if X_test.columns.tolist() != X_train.columns.tolist():
        raise ValueError("Test feature columns do not align with train columns.")

    retained_v_columns = sorted(set(X_train.columns) & set(v_columns))
    if len(retained_v_columns) != len(v_columns):
        raise ValueError(
            "Original V-features must be retained; "
            f"retained {len(retained_v_columns)} of {len(v_columns)}."
        )

    for feature in RECONSTRUCTION_ERROR_FEATURES:
        if feature not in X_train.columns:
            raise ValueError(f"Missing reconstruction error feature: {feature}")


def build_baseline_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], list[str]]:
    X_train_raw, _ = split_features_target(train_df)
    X_valid_raw, _ = split_features_target(valid_df)
    X_test_raw, _ = split_features_target(test_df)

    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test = apply_baseline_preprocessing(X_test_raw, preprocessing)
    return X_train, X_valid, X_test, preprocessing, preprocessing["categorical_columns"]


def build_du_latent_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autoencoder_output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    list[str],
    dict[str, object],
]:
    v_columns = get_v_feature_columns(train_df)
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

    X_train_non_v_raw, _ = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, _ = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, _ = split_non_v_features_target(test_df, v_columns)

    preprocessing = fit_non_v_preprocessing(X_train_non_v_raw, v_columns)
    X_train_non_v = apply_non_v_preprocessing(X_train_non_v_raw, preprocessing)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing)

    X_train = combine_non_v_and_latent(X_train_non_v, latent_train, latent_feature_names)
    X_valid = combine_non_v_and_latent(X_valid_non_v, latent_valid, latent_feature_names)
    X_test = combine_non_v_and_latent(X_test_non_v, latent_test, latent_feature_names)
    validate_latent_feature_alignment(X_train, X_valid, X_test, v_columns)

    feature_set_summary = {
        "experiment_family": "ae_integration_strategy_ablation",
        "strategy": "du_latent_replacement",
        "paper_anchor": "Du et al. latent representation",
        "original_v_features_retained": False,
        "latent_features_used": True,
        "reconstructed_features_used": False,
        "reconstruction_error_used": False,
        "number_of_non_v_features": int(X_train_non_v.shape[1]),
        "number_of_latent_v_features": int(len(latent_feature_names)),
        "total_final_features": int(X_train.shape[1]),
        "robust_autoencoder_output_path_used": str(autoencoder_output_dir),
        "robust_autoencoder_clipping": robust_ae_run_config.get("preprocessing", {}),
    }
    return (
        X_train,
        X_valid,
        X_test,
        preprocessing,
        preprocessing["categorical_columns"],
        feature_set_summary,
    )


def build_ding_reconstructed_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autoencoder_output_dir: Path,
    reconstruction_space: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    list[str],
    dict[str, object],
]:
    if reconstruction_space not in SUPPORTED_RECONSTRUCTION_SPACES:
        raise ValueError(
            f"Unsupported reconstruction_space: {reconstruction_space}. "
            f"Supported: {SUPPORTED_RECONSTRUCTION_SPACES}"
        )

    current_v_columns = get_v_feature_columns(train_df)
    autoencoder, scaler, ae_run_config, saved_v_columns = load_saved_autoencoder_artifacts(
        autoencoder_output_dir
    )
    validate_v_columns_match_ae_run_config(
        current_v_columns,
        saved_v_columns,
        autoencoder_output_dir,
    )
    v_columns = current_v_columns

    validate_latent_split_manifest_alignment(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
    )

    X_train_non_v_raw, _ = split_non_v_features_target(train_df, v_columns)
    X_valid_non_v_raw, _ = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, _ = split_non_v_features_target(test_df, v_columns)

    preprocessing = fit_non_v_preprocessing(X_train_non_v_raw, v_columns)
    X_train_non_v = apply_non_v_preprocessing(X_train_non_v_raw, preprocessing)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing)

    reconstructed_columns = reconstructed_v_feature_names(v_columns)
    split_inputs = {
        "train": (train_df, len(train_df)),
        "validation": (valid_df, len(valid_df)),
        "test": (test_df, len(test_df)),
    }
    reconstructed_by_split: dict[str, np.ndarray] = {}
    for split_name, (split_df, row_count) in split_inputs.items():
        X_scaled = apply_saved_v_preprocessing(
            split_df,
            v_columns,
            scaler,
            ae_run_config,
        )
        reconstructed = generate_decoder_reconstructed_v(autoencoder, X_scaled)
        if reconstructed.shape != (row_count, len(v_columns)):
            raise ValueError(
                f"{split_name} reconstructed shape {reconstructed.shape} does not match "
                f"expected {(row_count, len(v_columns))}."
            )
        reconstructed_by_split[split_name] = reconstructed

    X_train = combine_non_v_and_reconstructed_v(
        X_train_non_v,
        reconstructed_by_split["train"],
        v_columns,
    )
    X_valid = combine_non_v_and_reconstructed_v(
        X_valid_non_v,
        reconstructed_by_split["validation"],
        v_columns,
    )
    X_test = combine_non_v_and_reconstructed_v(
        X_test_non_v,
        reconstructed_by_split["test"],
        v_columns,
    )
    validate_reconstructed_feature_alignment(
        X_train,
        X_valid,
        X_test,
        v_columns,
        reconstructed_columns,
    )

    feature_set_summary = {
        "experiment_family": "ae_integration_strategy_ablation",
        "strategy": "ding_reconstructed_replacement",
        "paper_anchor": "Ding et al. reconstructed features",
        "reconstruction_space": reconstruction_space,
        "original_v_features_retained": False,
        "latent_features_used": False,
        "reconstructed_features_used": True,
        "reconstruction_error_used": False,
        "number_of_non_v_features": int(X_train_non_v.shape[1]),
        "number_of_reconstructed_v_features": int(len(v_columns)),
        "total_final_features": int(X_train.shape[1]),
        "reconstructed_feature_names_sample": reconstructed_columns[:5],
        "robust_autoencoder_output_path_used": str(autoencoder_output_dir),
    }
    return (
        X_train,
        X_valid,
        X_test,
        preprocessing,
        preprocessing["categorical_columns"],
        feature_set_summary,
    )


def build_reconstruction_error_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autoencoder_output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    list[str],
    dict[str, object],
]:
    v_columns = get_v_feature_columns(train_df)
    validate_latent_split_manifest_alignment(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
    )

    errors = load_or_compute_reconstruction_errors(
        autoencoder_output_dir,
        train_df,
        valid_df,
        test_df,
        v_columns,
    )
    validate_reconstruction_error_lengths(
        errors,
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    X_train_raw, _ = split_features_target(train_df)
    X_valid_raw, _ = split_features_target(valid_df)
    X_test_raw, _ = split_features_target(test_df)

    preprocessing = fit_baseline_preprocessing(X_train_raw)
    X_train_base = apply_baseline_preprocessing(X_train_raw, preprocessing)
    X_valid_base = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test_base = apply_baseline_preprocessing(X_test_raw, preprocessing)

    X_train = combine_with_reconstruction_errors(X_train_base, errors["train"])
    X_valid = combine_with_reconstruction_errors(X_valid_base, errors["validation"])
    X_test = combine_with_reconstruction_errors(X_test_base, errors["test"])
    validate_reconstruction_error_feature_alignment(
        X_train,
        X_valid,
        X_test,
        v_columns,
    )

    feature_set_summary = {
        "experiment_family": "ae_integration_strategy_ablation",
        "strategy": "reconstruction_error_augmentation",
        "paper_anchor": "Autoencoder anomaly detection reconstruction error",
        "original_v_features_retained": True,
        "latent_features_used": False,
        "reconstructed_features_used": False,
        "reconstruction_error_used": True,
        "reconstruction_error_features": list(RECONSTRUCTION_ERROR_FEATURES),
        "total_final_features": int(X_train.shape[1]),
        "robust_autoencoder_output_path_used": str(autoencoder_output_dir),
    }
    return (
        X_train,
        X_valid,
        X_test,
        preprocessing,
        preprocessing["categorical_columns"],
        feature_set_summary,
    )


def build_feature_matrices(
    variant: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    autoencoder_output_dir: Path | None,
    reconstruction_space: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
    list[str],
    dict[str, object] | None,
]:
    if variant == "baseline_fixed":
        X_train, X_valid, X_test, preprocessing, categorical_columns = build_baseline_features(
            train_df,
            valid_df,
            test_df,
        )
        feature_set_summary = {
            "experiment_family": "ae_integration_strategy_ablation",
            "strategy": "baseline_fixed",
            "paper_anchor": "Raw LightGBM fixed/default baseline",
            "original_v_features_retained": True,
            "latent_features_used": False,
            "reconstructed_features_used": False,
            "reconstruction_error_used": False,
            "total_final_features": int(X_train.shape[1]),
        }
        return (
            X_train,
            X_valid,
            X_test,
            preprocessing,
            categorical_columns,
            feature_set_summary,
        )

    if autoencoder_output_dir is None:
        raise ValueError(
            f"--autoencoder-output-dir is required for variant '{variant}'."
        )

    if variant == "du_latent_replacement":
        return build_du_latent_features(
            train_df,
            valid_df,
            test_df,
            autoencoder_output_dir,
        )
    if variant == "ding_reconstructed_replacement":
        return build_ding_reconstructed_features(
            train_df,
            valid_df,
            test_df,
            autoencoder_output_dir,
            reconstruction_space,
        )
    if variant == "reconstruction_error_augmentation":
        return build_reconstruction_error_features(
            train_df,
            valid_df,
            test_df,
            autoencoder_output_dir,
        )
    raise ValueError(f"Unsupported variant: {variant}")


def train_and_save(
    variant: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    categorical_columns: list[str],
    preprocessing: dict[str, object],
    feature_set_summary: dict[str, object] | None,
    output_dir: Path,
    phase_name: str,
    autoencoder_output_dir: Path | None,
    reconstruction_space: str,
) -> dict[str, object]:
    model_params = build_model_params(y_train)
    model = lgb.LGBMClassifier(**model_params)

    log("Training LightGBM with validation early stopping.")
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
    valid_score = model.predict_proba(X_valid, num_iteration=best_iteration)[:, 1]
    test_score = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]

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

    save_json(
        metrics_valid_default,
        output_dir / "metrics_validation_default_threshold.json",
    )
    save_json(
        metrics_valid_selected,
        output_dir / "metrics_validation_selected_threshold.json",
    )
    save_json(metrics_test_default, output_dir / "metrics_test_default_threshold.json")
    save_json(
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
    joblib.dump(preprocessing, output_dir / "preprocessing.pkl")

    if feature_set_summary is not None:
        save_json(feature_set_summary, output_dir / "feature_set_summary.json")

    run_config = {
        "phase": phase_name,
        "variant": variant,
        "experiment_family": "ae_integration_strategy_ablation",
        "data_dir": str(DATA_DIR),
        "output_dir": str(output_dir),
        "sample_size": SAMPLE_SIZE,
        "is_local_debugging_sample": SAMPLE_SIZE is not None,
        "target_column": TARGET_COL,
        "id_column_dropped_from_features": ID_COL,
        "time_column": TIME_COL,
        "transactiondt_note": (
            "TransactionDT is kept as a model feature and was also used "
            "to create the chronological split."
        ),
        "split_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "model_features_count": int(X_train.shape[1]),
        "categorical_columns": categorical_columns,
        "categorical_columns_count": len(categorical_columns),
        "numeric_missing_values": "Preserved as NaN for LightGBM native handling.",
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
    if autoencoder_output_dir is not None:
        run_config["autoencoder_output_dir"] = str(autoencoder_output_dir)
    if variant == "ding_reconstructed_replacement":
        run_config["reconstruction_space"] = reconstruction_space
    if feature_set_summary is not None:
        run_config["feature_set_summary"] = feature_set_summary
    save_json(run_config, output_dir / "run_config.json")

    print()
    print(f"AE Integration Strategy Ablation — {variant}")
    print("=" * (38 + len(variant)))
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

    return {
        "output_dir": str(output_dir),
        "metrics_validation_selected": metrics_valid_selected,
        "metrics_test_selected": metrics_test_selected,
        "selected_threshold": selected_threshold,
        "best_iteration": best_iteration,
        "feature_set_summary": feature_set_summary,
    }


def main(
    variant: str,
    output_dir: Path,
    phase_name: str,
    autoencoder_output_dir: Path | None = None,
    reconstruction_space: str = "scaled",
) -> dict[str, object]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant '{variant}'. Supported: {SUPPORTED_VARIANTS}"
        )
    if requires_autoencoder_output_dir(variant) and autoencoder_output_dir is None:
        raise ValueError(
            f"--autoencoder-output-dir is required for variant '{variant}'."
        )

    set_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)

    _, y_train = split_features_target(train_df)
    _, y_valid = split_features_target(valid_df)
    _, y_test = split_features_target(test_df)

    log(f"Building feature matrices for variant '{variant}'.")
    (
        X_train,
        X_valid,
        X_test,
        preprocessing,
        categorical_columns,
        feature_set_summary,
    ) = build_feature_matrices(
        variant,
        train_df,
        valid_df,
        test_df,
        autoencoder_output_dir,
        reconstruction_space,
    )

    return train_and_save(
        variant=variant,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        categorical_columns=categorical_columns,
        preprocessing=preprocessing,
        feature_set_summary=feature_set_summary,
        output_dir=output_dir,
        phase_name=phase_name,
        autoencoder_output_dir=autoencoder_output_dir,
        reconstruction_space=reconstruction_space,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed/default LightGBM variants for AE integration strategy ablation."
        )
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=SUPPORTED_VARIANTS,
        help="AE integration strategy variant to train.",
    )
    parser.add_argument(
        "--autoencoder-output-dir",
        type=Path,
        default=None,
        help="Frozen V-only AE output directory (required for AE variants).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Variant output directory. Defaults to "
            "outputs/ae_integration_strategy_ablation/<variant_name>."
        ),
    )
    parser.add_argument(
        "--phase-name",
        default=None,
        help="Optional run phase label saved in run_config.json.",
    )
    parser.add_argument(
        "--reconstruction-space",
        choices=SUPPORTED_RECONSTRUCTION_SPACES,
        default="scaled",
        help="Decoder reconstruction representation for ding_reconstructed_replacement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    variant = args.variant
    output_dir = args.output_dir or default_output_dir(variant)
    phase_name = args.phase_name or VARIANT_DEFAULT_PHASE_NAMES[variant]
    main(
        variant=variant,
        output_dir=output_dir,
        phase_name=phase_name,
        autoencoder_output_dir=args.autoencoder_output_dir,
        reconstruction_space=args.reconstruction_space,
    )