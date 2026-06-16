"""Generate decoder-reconstructed selected-numerical features from a frozen Autoencoder."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from autoencoder_helpers import raw_float_feature_matrix_unfilled
from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_USE_SCALED_CLIPPING,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR,
    TARGET_COL,
    TIME_COL,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from utils import log, save_json, set_seed

RECONSTRUCTED_REPRESENTATION_SPACE = "scaled"
RECONSTRUCTED_NAME_PREFIX = "ae_reconstructed_"


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def reconstructed_feature_names(selected_feature_names: list[str]) -> list[str]:
    return [f"{RECONSTRUCTED_NAME_PREFIX}{name}" for name in selected_feature_names]


def verify_existing_artifacts(autoencoder_output_dir: Path) -> dict[str, object]:
    required = {
        "audit": SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
        "selected_numerical_feature_names": (
            autoencoder_output_dir / "selected_numerical_feature_names.json"
        ),
        "numerical_imputer": autoencoder_output_dir / "numerical_imputer.pkl",
        "numerical_scaler": autoencoder_output_dir / "numerical_scaler.pkl",
        "autoencoder_model": autoencoder_output_dir / "autoencoder_model.keras",
        "latent_train": autoencoder_output_dir / "latent_train.npy",
        "p01_validation_metrics": (
            BASELINE_OUTPUT_DIR / "metrics_validation_selected_threshold.json"
        ),
        "latent_lgbm_validation_metrics": (
            SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        ),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s) for reconstruction generation: "
            + ", ".join(missing)
        )
    return required


def apply_saved_preprocessing(
    df: pd.DataFrame,
    feature_names: list[str],
    imputer,
    scaler,
) -> np.ndarray:
    raw = raw_float_feature_matrix_unfilled(df, feature_names)
    imputed = imputer.transform(raw).astype("float32")
    scaled = scaler.transform(imputed).astype("float32")
    if AE_USE_SCALED_CLIPPING:
        scaled = np.clip(scaled, AE_CLIP_MIN, AE_CLIP_MAX).astype("float32")
    return scaled


def generate_reconstructed_arrays(
    autoencoder_output_dir: Path = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    smoke_rows: int | None = None,
) -> dict[str, object]:
    set_seed(RANDOM_SEED)
    artifacts = verify_existing_artifacts(autoencoder_output_dir)

    selected_feature_names = load_json(artifacts["selected_numerical_feature_names"])
    if not isinstance(selected_feature_names, list):
        raise TypeError("selected_numerical_feature_names.json must contain a list.")

    input_dim = len(selected_feature_names)
    imputer = joblib.load(artifacts["numerical_imputer"])
    scaler = joblib.load(artifacts["numerical_scaler"])
    autoencoder = keras.models.load_model(artifacts["autoencoder_model"])

    output_dim = int(autoencoder.output_shape[-1])
    if output_dim != input_dim:
        raise ValueError(
            f"Decoder output dimension {output_dim} does not match selected "
            f"input dimension {input_dim}."
        )

    sample_size = smoke_rows if smoke_rows is not None else SAMPLE_SIZE
    full_df = load_labeled_train_data(sample_size=sample_size)
    train_df, valid_df, test_df = chronological_split(full_df)

    expected_rows = (len(train_df), len(valid_df), len(test_df))
    if smoke_rows is None:
        latent_train = np.load(artifacts["latent_train"])
        if latent_train.shape[0] != expected_rows[0]:
            raise ValueError(
                "Chronological train row count does not match saved latent artifact rows."
            )

    X_train = apply_saved_preprocessing(train_df, selected_feature_names, imputer, scaler)
    X_valid = apply_saved_preprocessing(valid_df, selected_feature_names, imputer, scaler)
    X_test = apply_saved_preprocessing(test_df, selected_feature_names, imputer, scaler)

    log("Generating decoder reconstructions in standardized/scaled space.")
    reconstructed_train = autoencoder.predict(X_train, batch_size=AE_BATCH_SIZE, verbose=0)
    reconstructed_valid = autoencoder.predict(X_valid, batch_size=AE_BATCH_SIZE, verbose=0)
    reconstructed_test = autoencoder.predict(X_test, batch_size=AE_BATCH_SIZE, verbose=0)

    for split_name, array, row_count in (
        ("train", reconstructed_train, expected_rows[0]),
        ("validation", reconstructed_valid, expected_rows[1]),
        ("test", reconstructed_test, expected_rows[2]),
    ):
        if array.shape != (row_count, input_dim):
            raise ValueError(
                f"{split_name} reconstructed shape {array.shape} != "
                f"expected {(row_count, input_dim)}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{split_name} reconstructed array contains non-finite values.")

    recon_names = reconstructed_feature_names(selected_feature_names)
    if len(recon_names) != len(set(recon_names)):
        raise ValueError("Duplicate reconstructed feature names detected.")

    result = {
        "reconstructed_train": reconstructed_train.astype("float32"),
        "reconstructed_valid": reconstructed_valid.astype("float32"),
        "reconstructed_test": reconstructed_test.astype("float32"),
        "reconstructed_feature_names": recon_names,
        "selected_feature_names": selected_feature_names,
        "input_dim": input_dim,
        "representation_space": RECONSTRUCTED_REPRESENTATION_SPACE,
        "train_rows": expected_rows[0],
        "valid_rows": expected_rows[1],
        "test_rows": expected_rows[2],
    }
    return result


def save_reconstructed_artifacts(
    result: dict[str, object],
    autoencoder_output_dir: Path = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
) -> None:
    output_files = {
        "reconstructed_train.npy": result["reconstructed_train"],
        "reconstructed_valid.npy": result["reconstructed_valid"],
        "reconstructed_test.npy": result["reconstructed_test"],
    }
    for filename, array in output_files.items():
        path = autoencoder_output_dir / filename
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing reconstructed artifact: {path}"
            )
        np.save(path, array)

    names_path = autoencoder_output_dir / "reconstructed_feature_names.json"
    if names_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {names_path}")
    save_json(result["reconstructed_feature_names"], names_path)


def print_prerun_summary(
    autoencoder_output_dir: Path,
    result: dict[str, object],
) -> None:
    print("=" * 60)
    print("PHASE 9 PRE-RUN VALIDATION")
    print("=" * 60)
    print(f"Autoencoder source path     : {autoencoder_output_dir}")
    print(f"Input dimension             : {result['input_dim']}")
    print(f"Decoder output dimension    : {result['input_dim']}")
    print(f"Selected numerical count    : {result['input_dim']}")
    print(f"Reconstructed feature count   : {len(result['reconstructed_feature_names'])}")
    print(f"Retained raw count (expected): 45")
    print(f"Final expected count        : 432")
    print(f"TransactionDT policy        : retained downstream among 45 raw features")
    print("Latent absent               : yes")
    print("Reconstruction error absent : yes")
    print()
    print("Reconstructed shapes:")
    print(f"  train: {result['reconstructed_train'].shape}")
    print(f"  valid: {result['reconstructed_valid'].shape}")
    print(f"  test : {result['reconstructed_test'].shape}")
    print(f"Representation space        : {result['representation_space']}")
    print(f"Finite values               : {np.isfinite(result['reconstructed_train']).all()}")
    print("PRE-RUN VALIDATION PASSED")


def main(smoke_rows: int | None = None) -> dict[str, object]:
    autoencoder_output_dir = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR
    verify_existing_artifacts(autoencoder_output_dir)

    if smoke_rows is None:
        smoke_result = generate_reconstructed_arrays(
            autoencoder_output_dir=autoencoder_output_dir,
            smoke_rows=256,
        )
        print_prerun_summary(autoencoder_output_dir, smoke_result)

    result = generate_reconstructed_arrays(
        autoencoder_output_dir=autoencoder_output_dir,
        smoke_rows=smoke_rows,
    )

    if smoke_rows is None:
        save_reconstructed_artifacts(result, autoencoder_output_dir)
        print()
        print("Reconstruction Generation Summary")
        print("=================================")
        print(f"Train shape : {result['reconstructed_train'].shape}")
        print(f"Valid shape : {result['reconstructed_valid'].shape}")
        print(f"Test shape  : {result['reconstructed_test'].shape}")
        print(f"Representation: {result['representation_space']}")
        print(f"Saved to    : {autoencoder_output_dir}")

    return result


if __name__ == "__main__":
    main()