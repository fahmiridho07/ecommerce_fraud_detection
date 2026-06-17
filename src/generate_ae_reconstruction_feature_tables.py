"""Generate global and grouped reconstruction feature tables from a saved AE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

from config import (
    DEFAULT_SPLIT_STRATEGY,
    PROJECT_ROOT,
    SAMPLE_SIZE,
    SUPPORTED_SPLIT_STRATEGIES,
)
from data_loader import load_labeled_train_data
from preprocessing import get_v_feature_columns
from splitting import create_holdout_split
from train_ae_lgbm import validate_latent_split_manifest_alignment
from train_autoencoder_normal_masked import V_GROUPS, group_column_indices
from utils import ensure_dir, log, save_json


DEFAULT_INITIAL_PROPOSAL_DIR = PROJECT_ROOT / "outputs" / "initial_proposal"
DEFAULT_AUTOENCODER_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "autoencoder_robust_ld128"
DEFAULT_OUTPUT_DIR = DEFAULT_INITIAL_PROPOSAL_DIR / "autoencoder_robust_ld128_grouped_features"


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def load_saved_autoencoder(autoencoder_dir: Path):
    model_path = autoencoder_dir / "autoencoder_model.keras"
    imputer_path = autoencoder_dir / "v_imputer.pkl"
    scaler_path = autoencoder_dir / "v_scaler.pkl"
    run_config_path = autoencoder_dir / "run_config.json"
    missing = [
        str(path)
        for path in (model_path, imputer_path, scaler_path, run_config_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing AE artifact(s):\n" + "\n".join(missing))
    return (
        keras.models.load_model(model_path, compile=False),
        joblib.load(imputer_path),
        joblib.load(scaler_path),
        load_json(run_config_path),
    )


def transform_v_block(
    df: pd.DataFrame,
    v_columns: list[str],
    imputer,
    scaler,
    run_config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    raw = df.loc[:, v_columns].astype("float32")
    observed_mask = (~raw.isna()).to_numpy(dtype="float32")
    imputed = imputer.transform(raw)
    scaled = scaler.transform(imputed).astype("float32")
    preprocessing = run_config.get("preprocessing", {})
    if isinstance(preprocessing, dict) and preprocessing.get("scaled_clipping_enabled"):
        clip_min = float(preprocessing.get("clip_min", -10.0))
        clip_max = float(preprocessing.get("clip_max", 10.0))
        scaled = np.clip(scaled, clip_min, clip_max).astype("float32")
    return scaled, observed_mask


def reconstruction_feature_frame(
    model,
    X: np.ndarray,
    observed_mask: np.ndarray,
    group_indices: dict[str, list[int]],
    batch_size: int,
    prefix: str,
) -> pd.DataFrame:
    reconstructed = model.predict(X, batch_size=batch_size, verbose=0)
    squared_error = np.square(X - reconstructed) * observed_mask
    global_denominator = np.maximum(observed_mask.sum(axis=1), 1.0)
    global_mse = squared_error.sum(axis=1) / global_denominator

    features: dict[str, np.ndarray] = {
        f"{prefix}_mse": global_mse.astype("float32"),
        f"{prefix}_log1p_mse": np.log1p(global_mse).astype("float32"),
        f"{prefix}_observed_v_rate": observed_mask.mean(axis=1).astype("float32"),
    }
    for group_name, indices in group_indices.items():
        if not indices:
            continue
        group_error = squared_error[:, indices]
        group_mask = observed_mask[:, indices]
        denominator = np.maximum(group_mask.sum(axis=1), 1.0)
        group_mse = group_error.sum(axis=1) / denominator
        features[f"{prefix}_{group_name}_mse"] = group_mse.astype("float32")
        features[f"{prefix}_{group_name}_log1p_mse"] = np.log1p(group_mse).astype(
            "float32"
        )
        features[f"{prefix}_{group_name}_observed_rate"] = group_mask.mean(axis=1).astype(
            "float32"
        )
    return pd.DataFrame(features)


def main(
    autoencoder_dir: Path = DEFAULT_AUTOENCODER_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    feature_prefix: str = "robust_ae_ld128",
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
) -> dict[str, object]:
    output_dir = ensure_dir(output_dir)

    log("Loading saved Autoencoder artifacts.")
    model, imputer, scaler, run_config = load_saved_autoencoder(autoencoder_dir)
    batch_size = int(run_config.get("training", {}).get("batch_size", 1024))

    log(f"Loading labeled training data and {split_strategy} split.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)
    train_df, valid_df, test_df = create_holdout_split(
        full_df,
        split_strategy=split_strategy,
    )
    validate_latent_split_manifest_alignment(
        autoencoder_dir,
        train_df,
        valid_df,
        test_df,
    )
    v_columns = [str(column) for column in run_config.get("v_columns", [])]
    if not v_columns:
        v_columns = get_v_feature_columns(train_df)
    group_indices = group_column_indices(v_columns)

    outputs = {}
    for split_name, split_df, filename in (
        ("train", train_df, "reconstruction_features_train.csv"),
        ("validation", valid_df, "reconstruction_features_valid.csv"),
        ("test", test_df, "reconstruction_features_test.csv"),
    ):
        log(f"Generating grouped reconstruction features for {split_name}.")
        X, observed_mask = transform_v_block(split_df, v_columns, imputer, scaler, run_config)
        frame = reconstruction_feature_frame(
            model,
            X,
            observed_mask,
            group_indices,
            batch_size,
            feature_prefix,
        )
        frame.to_csv(output_dir / filename, index=False)
        outputs[split_name] = {
            "rows": int(len(frame)),
            "features": int(frame.shape[1]),
            "path": str(output_dir / filename),
        }

    config = {
        "source_autoencoder_dir": str(autoencoder_dir),
        "source_autoencoder_phase": run_config.get("phase"),
        "split_strategy": split_strategy,
        "feature_prefix": feature_prefix,
        "feature_count": int(len(pd.read_csv(output_dir / "reconstruction_features_train.csv", nrows=1).columns)),
        "groups": [
            {"name": name, "start": start, "end": end}
            for name, start, end in V_GROUPS
        ],
        "outputs": outputs,
        "training": run_config.get("training", {}),
        "preprocessing": run_config.get("preprocessing", {}),
    }
    save_json(config, output_dir / "run_config.json")
    print()
    print("AE Reconstruction Feature Tables")
    print("================================")
    print(f"Source AE     : {autoencoder_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Feature prefix: {feature_prefix}")
    print(f"Feature count : {config['feature_count']}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grouped reconstruction feature CSVs from a saved AE."
    )
    parser.add_argument("--autoencoder-dir", type=Path, default=DEFAULT_AUTOENCODER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-prefix", default="robust_ae_ld128")
    parser.add_argument(
        "--split-strategy",
        choices=SUPPORTED_SPLIT_STRATEGIES,
        default=DEFAULT_SPLIT_STRATEGY,
        help="Holdout split strategy. Default is the active thesis stratified reset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        autoencoder_dir=args.autoencoder_dir,
        output_dir=args.output_dir,
        feature_prefix=args.feature_prefix,
        split_strategy=args.split_strategy,
    )
