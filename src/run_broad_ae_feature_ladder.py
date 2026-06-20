"""Run broader AE feature-learning fixes while staying in AE + LightGBM.

This runner tests the next proposal-near alternatives suggested by the
diagnosis: the AE should not be restricted to the V block only. It keeps the
strong original proposal tuned LightGBM as reference and appends AE-derived
features to the original LightGBM feature matrix instead of replacing it.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("TensorFlow is not installed.") from exc

from config import (
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TARGET_COL,
    TEST_RATIO,
    TIME_COL,
    TRAIN_IDENTITY_FILE,
    TRAIN_RATIO,
    TRAIN_TRANSACTION_FILE,
    VALID_RATIO,
)
from data_loader import validate_train_files
from preprocessing import get_v_feature_columns
from preprocessing import MISSING_CATEGORY, UNKNOWN_CATEGORY_VALUE
from run_ae_feature_improvement_ladder import (
    BASELINE_REFERENCE_AP,
    BASELINE_REFERENCE_VALID_AP,
    DEFAULT_ORIGINAL_OUTPUT,
    load_json,
    load_params,
    load_reference_scores,
    paired_bootstrap_ap_delta,
    train_with_profile,
)
from run_original_proposal_stratified import PROPOSAL_CATEGORICAL_EXACT
from utils import ensure_dir, log, save_json, set_seed


IEEE_CIS_OBJECT_COLUMNS = {
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "DeviceType",
    "DeviceInfo",
}


@dataclass(frozen=True)
class BroadCandidate:
    stage: int
    candidate_id: str
    description: str
    kind: str
    latent_dim: int = 64
    group_latent_dim: int = 16
    training_subset: str = "all"
    include_input_mask: bool = True
    reconstruct_mask: bool = False
    supervised_auxiliary: bool = False
    append_latent: bool = True
    append_error: bool = True
    append_aux_score: bool = False
    aux_loss_weight: float = 0.05
    mask_loss_weight: float = 0.10
    top_feature_count: int | None = None
    group_top_k: int | None = None


@dataclass
class BroadData:
    X_base_train: pd.DataFrame
    X_base_valid: pd.DataFrame
    X_base_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series
    baseline_categorical_columns: list[str]
    v_columns: list[str]
    observed_train: pd.DataFrame
    observed_valid: pd.DataFrame
    observed_test: pd.DataFrame


@dataclass
class DenseFeatureBundle:
    source_dir: Path
    columns: list[str]
    latent_train: np.ndarray | None
    latent_valid: np.ndarray | None
    latent_test: np.ndarray | None
    recon_train: np.ndarray
    recon_valid: np.ndarray
    recon_test: np.ndarray
    values_train: np.ndarray
    values_valid: np.ndarray
    values_test: np.ndarray
    observed_train: np.ndarray
    observed_valid: np.ndarray
    observed_test: np.ndarray
    mask_pred_train: np.ndarray | None = None
    mask_pred_valid: np.ndarray | None = None
    mask_pred_test: np.ndarray | None = None
    aux_train: np.ndarray | None = None
    aux_valid: np.ndarray | None = None
    aux_test: np.ndarray | None = None


def release_feature_storage(data: BroadData) -> None:
    data.X_base_train = pd.DataFrame()
    data.X_base_valid = pd.DataFrame()
    data.X_base_test = pd.DataFrame()
    data.observed_train = pd.DataFrame()
    data.observed_valid = pd.DataFrame()
    data.observed_test = pd.DataFrame()
    gc.collect()


def arrow_column_types(path: Path) -> dict[str, pa.DataType]:
    columns = pd.read_csv(path, nrows=0).columns
    dtypes: dict[str, pa.DataType] = {}
    for column in columns:
        if column == TARGET_COL:
            dtypes[column] = pa.int8()
        elif column in {ID_COL, TIME_COL}:
            dtypes[column] = pa.int32()
        elif column in IEEE_CIS_OBJECT_COLUMNS:
            dtypes[column] = pa.string()
        else:
            dtypes[column] = pa.float32()
    return dtypes


def read_csv_arrow(path: Path, dataset_name: str) -> pa.Table:
    log(f"Reading {dataset_name} with PyArrow memory-light dtypes.")
    return pacsv.read_csv(
        path,
        read_options=pacsv.ReadOptions(use_threads=False, block_size=1 << 20),
        convert_options=pacsv.ConvertOptions(
            column_types=arrow_column_types(path),
            strings_can_be_null=True,
        ),
    )


def table_to_feature_target(table: pa.Table) -> tuple[pd.DataFrame, pd.Series]:
    y_values = (
        table.column(TARGET_COL)
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype("int8", copy=False)
    )
    feature_table = table.drop([TARGET_COL, ID_COL])
    frame = feature_table.to_pandas(strings_to_categorical=True, split_blocks=True)
    del feature_table
    y = pd.Series(y_values.astype(int, copy=False), name=TARGET_COL)
    return frame.reset_index(drop=True), y


def load_stratified_raw_splits_memory_light(
    seed: int,
    sample_size: int | None = SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    validate_train_files()
    transaction_table = read_csv_arrow(TRAIN_TRANSACTION_FILE, "train_transaction.csv")
    if sample_size is not None:
        transaction_table = transaction_table.sort_by([(TIME_COL, "ascending")]).slice(0, sample_size)

    row_id = "__source_row_id__"
    transaction_table = transaction_table.append_column(
        row_id,
        pa.array(np.arange(transaction_table.num_rows, dtype=np.int32)),
    )

    identity_table = read_csv_arrow(TRAIN_IDENTITY_FILE, "train_identity.csv")
    log("Joining transaction and identity tables with Arrow left join.")
    merged_table = transaction_table.join(identity_table, keys=ID_COL, join_type="left outer")
    del transaction_table, identity_table
    gc.collect()
    merged_table = merged_table.sort_by([(row_id, "ascending")]).drop([row_id])

    labels = (
        merged_table.column(TARGET_COL)
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype("int8", copy=False)
    )
    indices = np.arange(merged_table.num_rows, dtype=np.int32)
    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=seed,
    )
    train_valid_labels = labels[train_valid_idx]
    valid_fraction = VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_fraction,
        stratify=train_valid_labels,
        random_state=seed,
    )
    del indices, train_valid_idx, train_valid_labels, labels
    gc.collect()

    log("Converting stratified train split from Arrow to pandas.")
    train_table = merged_table.take(pa.array(train_idx, type=pa.int32()))
    X_train_raw, y_train = table_to_feature_target(train_table)
    del train_table, train_idx
    gc.collect()

    log("Converting stratified validation split from Arrow to pandas.")
    valid_table = merged_table.take(pa.array(valid_idx, type=pa.int32()))
    X_valid_raw, y_valid = table_to_feature_target(valid_table)
    del valid_table, valid_idx
    gc.collect()

    log("Converting stratified test split from Arrow to pandas.")
    test_table = merged_table.take(pa.array(test_idx, type=pa.int32()))
    X_test_raw, y_test = table_to_feature_target(test_table)
    del test_table, test_idx, merged_table
    gc.collect()

    return X_train_raw, y_train, X_valid_raw, y_valid, X_test_raw, y_test


def pandas_dtype_map(path: Path, usecols: list[str] | None = None) -> dict[str, str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if usecols is not None:
        wanted = set(usecols)
        columns = [column for column in columns if column in wanted]
    dtypes: dict[str, str] = {}
    for column in columns:
        if column == TARGET_COL:
            dtypes[column] = "int8"
        elif column in {ID_COL, TIME_COL}:
            dtypes[column] = "int32"
        elif column in IEEE_CIS_OBJECT_COLUMNS:
            dtypes[column] = "category"
        else:
            dtypes[column] = "float32"
    return dtypes


def proposal_categorical_columns_from_names(columns: list[str]) -> list[str]:
    categorical = set(IEEE_CIS_OBJECT_COLUMNS)
    categorical |= {column for column in PROPOSAL_CATEGORICAL_EXACT if column in columns}
    categorical |= {
        column
        for column in columns
        if column.startswith("M") and column[1:].isdigit() and 1 <= int(column[1:]) <= 9
    }
    categorical |= {
        column
        for column in columns
        if column.startswith("id_")
        and column[3:].isdigit()
        and 12 <= int(column[3:]) <= 38
    }
    return [column for column in columns if column in categorical]


def build_feature_columns() -> list[str]:
    transaction_columns = pd.read_csv(TRAIN_TRANSACTION_FILE, nrows=0).columns.tolist()
    identity_columns = pd.read_csv(TRAIN_IDENTITY_FILE, nrows=0).columns.tolist()
    merged_columns = transaction_columns + [column for column in identity_columns if column != ID_COL]
    return [column for column in merged_columns if column not in {ID_COL, TARGET_COL}]


def read_identity_frame(usecols: list[str] | None = None) -> pd.DataFrame:
    dtype_map = pandas_dtype_map(TRAIN_IDENTITY_FILE, usecols=usecols)
    return pd.read_csv(
        TRAIN_IDENTITY_FILE,
        usecols=usecols,
        dtype=dtype_map,
        low_memory=False,
    )


def iter_transaction_chunks(
    chunksize: int,
    usecols: list[str] | None = None,
):
    dtype_map = pandas_dtype_map(TRAIN_TRANSACTION_FILE, usecols=usecols)
    return pd.read_csv(
        TRAIN_TRANSACTION_FILE,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=chunksize,
        low_memory=False,
    )


def fit_chunked_proposal_encoding(
    split_code: np.ndarray,
    feature_columns: list[str],
    categorical_columns: list[str],
    chunksize: int,
) -> dict[str, object]:
    log("Fitting proposal categorical mappings from train chunks.")
    category_values = {column: {MISSING_CATEGORY} for column in categorical_columns}
    transaction_columns = pd.read_csv(TRAIN_TRANSACTION_FILE, nrows=0).columns.tolist()
    transaction_cat_columns = [column for column in categorical_columns if column in transaction_columns]
    identity_cat_columns = [column for column in categorical_columns if column not in transaction_columns]
    usecols = [ID_COL, *transaction_cat_columns]
    identity_usecols = [ID_COL, *identity_cat_columns] if identity_cat_columns else [ID_COL]
    identity_df = read_identity_frame(usecols=identity_usecols)

    start = 0
    for chunk in iter_transaction_chunks(chunksize=chunksize, usecols=usecols):
        end = start + len(chunk)
        train_mask = split_code[start:end] == 0
        if train_mask.any():
            merged = chunk.merge(identity_df, on=ID_COL, how="left", copy=False)
            train_features = merged.loc[train_mask, categorical_columns]
            for column in categorical_columns:
                values = train_features[column].astype("string").fillna(MISSING_CATEGORY)
                category_values[column].update(values.unique().tolist())
            del merged, train_features
        del chunk
        start = end
        gc.collect()

    mappings = {}
    for column in categorical_columns:
        ordered = [MISSING_CATEGORY, *sorted(category_values[column] - {MISSING_CATEGORY})]
        mappings[column] = {category: index for index, category in enumerate(ordered)}
    return {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "categorical_mappings": mappings,
        "missing_category": MISSING_CATEGORY,
        "unknown_category_value": UNKNOWN_CATEGORY_VALUE,
        "encoding": "chunked train-fitted ordinal encoding for proposal categorical columns",
    }


def load_encoded_broad_data_chunked(
    output_dir: Path,
    seed: int,
    sample_size: int | None = SAMPLE_SIZE,
    chunksize: int = 20_000,
) -> BroadData:
    validate_train_files()
    if sample_size is not None:
        raise ValueError("Chunked broad loader currently supports full SAMPLE_SIZE=None runs only.")

    log("Reading lightweight target metadata for stratified split indices.")
    meta = pd.read_csv(
        TRAIN_TRANSACTION_FILE,
        usecols=[ID_COL, TARGET_COL, TIME_COL],
        dtype={ID_COL: "int32", TARGET_COL: "int8", TIME_COL: "int32"},
    )
    labels = meta[TARGET_COL].to_numpy(dtype="int8", copy=False)
    indices = np.arange(len(meta), dtype=np.int32)
    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=seed,
    )
    valid_fraction = VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_fraction,
        stratify=labels[train_valid_idx],
        random_state=seed,
    )

    split_code = np.full(len(meta), -1, dtype=np.int8)
    split_code[train_idx] = 0
    split_code[valid_idx] = 1
    split_code[test_idx] = 2
    order_rank = {
        "train": np.full(len(meta), -1, dtype=np.int32),
        "valid": np.full(len(meta), -1, dtype=np.int32),
        "test": np.full(len(meta), -1, dtype=np.int32),
    }
    order_rank["train"][train_idx] = np.arange(len(train_idx), dtype=np.int32)
    order_rank["valid"][valid_idx] = np.arange(len(valid_idx), dtype=np.int32)
    order_rank["test"][test_idx] = np.arange(len(test_idx), dtype=np.int32)

    y_train = pd.Series(labels[train_idx].astype(int, copy=False), name=TARGET_COL)
    y_valid = pd.Series(labels[valid_idx].astype(int, copy=False), name=TARGET_COL)
    y_test = pd.Series(labels[test_idx].astype(int, copy=False), name=TARGET_COL)
    del train_valid_idx, train_idx, valid_idx, test_idx, labels, indices, meta
    gc.collect()

    feature_columns = build_feature_columns()
    v_columns = get_v_feature_columns(pd.DataFrame(columns=feature_columns))
    categorical_columns = proposal_categorical_columns_from_names(feature_columns)
    baseline_preprocessing = fit_chunked_proposal_encoding(
        split_code=split_code,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        chunksize=chunksize,
    )
    save_json(baseline_preprocessing, output_dir / "baseline_preprocessing.json")

    log("Encoding proposal matrices from CSV chunks.")
    identity_df = read_identity_frame()
    buckets = {
        "train": {"X": [], "observed": []},
        "valid": {"X": [], "observed": []},
        "test": {"X": [], "observed": []},
    }

    start = 0
    for chunk in iter_transaction_chunks(chunksize=chunksize):
        end = start + len(chunk)
        merged = chunk.merge(identity_df, on=ID_COL, how="left", copy=False)
        features = merged.loc[:, feature_columns]
        observed = observed_mask_frame(features)
        encoded = apply_proposal_encoding_memory_light(features, baseline_preprocessing)

        for name, code in (("train", 0), ("valid", 1), ("test", 2)):
            mask = split_code[start:end] == code
            if not mask.any():
                continue
            ranks = order_rank[name][start:end][mask]
            X_part = encoded.loc[mask].copy()
            X_part.insert(0, "__split_order__", ranks)
            observed_part = observed.loc[mask].copy()
            observed_part.insert(0, "__split_order__", ranks)
            buckets[name]["X"].append(X_part)
            buckets[name]["observed"].append(observed_part)

        del chunk, merged, features, observed, encoded
        start = end
        gc.collect()

    def finalize(parts: list[pd.DataFrame]) -> pd.DataFrame:
        frame = pd.concat(parts, axis=0, ignore_index=True)
        frame = frame.sort_values("__split_order__", kind="mergesort")
        frame = frame.drop(columns=["__split_order__"]).reset_index(drop=True)
        return frame

    log("Finalizing encoded train/validation/test matrices.")
    X_base_train = finalize(buckets["train"]["X"])
    buckets["train"]["X"].clear()
    gc.collect()
    observed_train = finalize(buckets["train"]["observed"])
    buckets["train"]["observed"].clear()
    gc.collect()
    X_base_valid = finalize(buckets["valid"]["X"])
    buckets["valid"]["X"].clear()
    gc.collect()
    observed_valid = finalize(buckets["valid"]["observed"])
    buckets["valid"]["observed"].clear()
    gc.collect()
    X_base_test = finalize(buckets["test"]["X"])
    buckets["test"]["X"].clear()
    gc.collect()
    observed_test = finalize(buckets["test"]["observed"])
    buckets["test"]["observed"].clear()
    del buckets, identity_df, split_code, order_rank
    gc.collect()

    save_json(
        {
            "train_rows": int(len(y_train)),
            "valid_rows": int(len(y_valid)),
            "test_rows": int(len(y_test)),
            "train_fraud_rate": float(y_train.mean()),
            "valid_fraud_rate": float(y_valid.mean()),
            "test_fraud_rate": float(y_test.mean()),
            "baseline_feature_count": int(X_base_train.shape[1]),
            "v_feature_count": len(v_columns),
            "categorical_feature_count": len(categorical_columns),
            "split_strategy": "stratified_holdout",
            "split_ratios": {"train": TRAIN_RATIO, "validation": VALID_RATIO, "test": TEST_RATIO},
            "seed": seed,
            "prepare_mode": "chunked_encoded_plus_missing_masks",
            "chunksize": chunksize,
            "numeric_dtype_note": "Float columns are downcast to float32 for memory; categorical proposal encodings remain int32.",
        },
        output_dir / "data_contract.json",
    )
    return BroadData(
        X_base_train=X_base_train,
        X_base_valid=X_base_valid,
        X_base_test=X_base_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        baseline_categorical_columns=categorical_columns,
        v_columns=v_columns,
        observed_train=observed_train,
        observed_valid=observed_valid,
        observed_test=observed_test,
    )


def numeric_values(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    try:
        return frame.loc[:, columns].to_numpy(dtype="float32", copy=True)
    except ValueError:
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        return values.to_numpy(dtype="float32", copy=True)


def observed_mask_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.notna().astype("bool")


def observed_values(mask_frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return mask_frame.loc[:, columns].to_numpy(dtype="float32", copy=True)


def apply_proposal_encoding_memory_light(
    frame: pd.DataFrame,
    preprocessing: dict[str, object],
) -> pd.DataFrame:
    """Apply the proposal ordinal encoding while keeping the working matrix small."""
    feature_columns: list[str] = preprocessing["feature_columns"]  # type: ignore[assignment]
    if frame.columns.tolist() != feature_columns:
        frame = frame.loc[:, feature_columns].copy()

    mappings: dict[str, dict[str, int]] = preprocessing["categorical_mappings"]  # type: ignore[assignment]
    missing_category = str(preprocessing["missing_category"])
    unknown_value = int(preprocessing["unknown_category_value"])
    for column, mapping in mappings.items():
        values = frame[column].astype("string").fillna(missing_category)
        frame[column] = values.map(mapping).fillna(unknown_value).astype("int32")

    categorical = set(mappings)
    for column in frame.columns:
        if column in categorical:
            continue
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].astype("float32", copy=False)
    return frame


def prepare_broad_data(output_dir: Path, seed: int) -> BroadData:
    log("Loading IEEE-CIS data for broad AE feature ladder.")
    data = load_encoded_broad_data_chunked(output_dir=output_dir, seed=seed)
    log(f"Prepared chunked broad data with {len(data.v_columns)} V columns.")
    return data


def load_importance_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "feature" not in frame.columns:
        return {}
    if "importance_gain" in frame.columns:
        importance_column = "importance_gain"
    else:
        numeric_columns = [
            column
            for column in frame.columns
            if column != "feature" and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric_columns:
            return {}
        importance_column = numeric_columns[0]
    return frame.set_index("feature")[importance_column].astype(float).to_dict()


def select_top_columns(
    columns: list[str],
    importance_scores: dict[str, float],
    top_k: int | None,
) -> list[str]:
    if top_k is None or len(columns) <= top_k or not importance_scores:
        return columns
    ranked = sorted(columns, key=lambda column: (importance_scores.get(column, 0.0), column), reverse=True)
    selected = set(ranked[:top_k])
    return [column for column in columns if column in selected]


def fit_dense_preprocessor(
    data: BroadData,
    columns: list[str],
    fit_subset: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    train_raw = numeric_values(data.X_base_train, columns)
    valid_raw = numeric_values(data.X_base_valid, columns)
    test_raw = numeric_values(data.X_base_test, columns)
    observed_train = observed_values(data.observed_train, columns)
    observed_valid = observed_values(data.observed_valid, columns)
    observed_test = observed_values(data.observed_test, columns)

    if fit_subset == "normal":
        fit_mask = data.y_train.to_numpy(dtype=int) == 0
    elif fit_subset == "all":
        fit_mask = np.ones(train_raw.shape[0], dtype=bool)
    else:
        raise ValueError(f"Unsupported fit_subset: {fit_subset}")

    fit_values = train_raw[fit_mask]
    median = np.nanmedian(fit_values, axis=0).astype("float32")
    median = np.where(np.isfinite(median), median, 0.0).astype("float32")

    def impute(values: np.ndarray) -> np.ndarray:
        out = values.copy()
        row_idx, col_idx = np.where(~np.isfinite(out))
        if row_idx.size:
            out[row_idx, col_idx] = median[col_idx]
        return out

    fit_imputed = impute(fit_values)
    mean = fit_imputed.mean(axis=0, dtype="float64").astype("float32")
    scale = fit_imputed.std(axis=0, dtype="float64").astype("float32")
    scale[~np.isfinite(scale) | (scale == 0.0)] = 1.0

    def transform(values: np.ndarray) -> np.ndarray:
        out = impute(values)
        out -= mean
        out /= scale
        np.clip(out, -10.0, 10.0, out=out)
        return out.astype("float32", copy=False)

    scaler = {
        "columns": columns,
        "fit_subset": fit_subset,
        "imputation": "train-fitted median",
        "scaler": "train-fitted zscore on imputed values",
        "clip": [-10.0, 10.0],
        "median": median.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    return (
        transform(train_raw),
        transform(valid_raw),
        transform(test_raw),
        observed_train,
        observed_valid,
        observed_test,
        scaler,
    )


def make_inputs(values: np.ndarray, observed: np.ndarray, include_mask: bool) -> np.ndarray:
    if not include_mask:
        return values
    return np.concatenate([values, observed], axis=1).astype("float32", copy=False)


def masked_target(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.concatenate([values, observed], axis=1).astype("float32", copy=False)


def build_dense_autoencoder(
    value_dim: int,
    input_dim: int,
    latent_dim: int,
    learning_rate: float,
    reconstruct_mask: bool,
    supervised_auxiliary: bool,
    mask_loss_weight: float,
    aux_loss_weight: float,
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(input_dim,), name="dense_values_plus_optional_mask")
    width = min(512, max(128, input_dim))
    x = keras.layers.Dense(width, activation="relu", name="encoder_dense_wide")(inputs)
    x = keras.layers.Dropout(0.05, name="encoder_dropout_005")(x)
    x = keras.layers.Dense(max(64, width // 2), activation="relu", name="encoder_dense_mid")(x)
    latent = keras.layers.Dense(latent_dim, activation="linear", name="latent")(x)
    x = keras.layers.Dense(max(64, width // 2), activation="relu", name="decoder_dense_mid")(latent)
    x = keras.layers.Dense(width, activation="relu", name="decoder_dense_wide")(x)
    recon_values = keras.layers.Dense(value_dim, activation="linear", name="reconstruction_values")(x)

    def observed_only_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target = y_true[:, :value_dim]
        observed = y_true[:, value_dim:]
        squared_error = tf.square(target - y_pred) * observed
        denominator = tf.reduce_sum(observed, axis=-1) + tf.keras.backend.epsilon()
        return tf.reduce_sum(squared_error, axis=-1) / denominator

    if supervised_auxiliary:
        fraud_score = keras.layers.Dense(1, activation="sigmoid", name="fraud_aux")(latent)
        model = keras.Model(inputs=inputs, outputs=[recon_values, fraud_score], name="dense_aux_autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={"reconstruction_values": observed_only_mse, "fraud_aux": "binary_crossentropy"},
            loss_weights={"reconstruction_values": 1.0, "fraud_aux": aux_loss_weight},
        )
    elif reconstruct_mask:
        mask_pred = keras.layers.Dense(value_dim, activation="sigmoid", name="observed_mask_reconstruction")(x)
        combined_output = keras.layers.Concatenate(name="value_and_mask_reconstruction")([recon_values, mask_pred])

        def value_and_mask_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            target = y_true[:, :value_dim]
            observed = y_true[:, value_dim:]
            pred_values = y_pred[:, :value_dim]
            pred_mask = y_pred[:, value_dim:]
            squared_error = tf.square(target - pred_values) * observed
            denominator = tf.reduce_sum(observed, axis=-1) + tf.keras.backend.epsilon()
            value_loss = tf.reduce_sum(squared_error, axis=-1) / denominator
            mask_loss = tf.keras.backend.binary_crossentropy(observed, pred_mask)
            return value_loss + mask_loss_weight * tf.reduce_mean(mask_loss, axis=-1)

        model = keras.Model(inputs=inputs, outputs=combined_output, name="dense_value_mask_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=value_and_mask_loss)
    else:
        model = keras.Model(inputs=inputs, outputs=recon_values, name="dense_autoencoder")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=observed_only_mse)

    encoder = keras.Model(inputs=inputs, outputs=latent, name="dense_encoder")
    return model, encoder


def predict_bundle_outputs(
    model: keras.Model,
    encoder: keras.Model,
    X_train_input: np.ndarray,
    X_valid_input: np.ndarray,
    X_test_input: np.ndarray,
    value_dim: int,
    spec: BroadCandidate,
    batch_size: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    latent_train = latent_valid = latent_test = None
    if spec.append_latent:
        latent_train = encoder.predict(X_train_input, batch_size=batch_size, verbose=0).astype("float32")
        latent_valid = encoder.predict(X_valid_input, batch_size=batch_size, verbose=0).astype("float32")
        latent_test = encoder.predict(X_test_input, batch_size=batch_size, verbose=0).astype("float32")

    pred_train = model.predict(X_train_input, batch_size=batch_size, verbose=0)
    pred_valid = model.predict(X_valid_input, batch_size=batch_size, verbose=0)
    pred_test = model.predict(X_test_input, batch_size=batch_size, verbose=0)

    aux_train = aux_valid = aux_test = None
    mask_train = mask_valid = mask_test = None
    if spec.supervised_auxiliary:
        recon_train, aux_train = pred_train
        recon_valid, aux_valid = pred_valid
        recon_test, aux_test = pred_test
        aux_train = aux_train.astype("float32").reshape(-1)
        aux_valid = aux_valid.astype("float32").reshape(-1)
        aux_test = aux_test.astype("float32").reshape(-1)
    elif spec.reconstruct_mask:
        recon_train = pred_train[:, :value_dim]
        recon_valid = pred_valid[:, :value_dim]
        recon_test = pred_test[:, :value_dim]
        mask_train = pred_train[:, value_dim:]
        mask_valid = pred_valid[:, value_dim:]
        mask_test = pred_test[:, value_dim:]
    else:
        recon_train = pred_train
        recon_valid = pred_valid
        recon_test = pred_test

    return (
        latent_train,
        latent_valid,
        latent_test,
        recon_train.astype("float32"),
        recon_valid.astype("float32"),
        recon_test.astype("float32"),
        None if mask_train is None else mask_train.astype("float32"),
        None if mask_valid is None else mask_valid.astype("float32"),
        None if mask_test is None else mask_test.astype("float32"),
        aux_train,
        aux_valid,
        aux_test,
    )


def train_or_load_dense_bundle(
    data: BroadData,
    spec: BroadCandidate,
    output_dir: Path,
    columns: list[str],
    cache_name: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> DenseFeatureBundle:
    cache_dir = ensure_dir(output_dir / "feature_cache" / cache_name)
    log(
        f"Preparing dense AE features for {cache_name}: "
        f"columns={len(columns)} subset={spec.training_subset}."
    )
    contract_path = cache_dir / "feature_contract.json"
    latent_train_path = cache_dir / "latent_train.npy"
    latent_valid_path = cache_dir / "latent_valid.npy"
    latent_test_path = cache_dir / "latent_test.npy"
    recon_train_path = cache_dir / "recon_train.npy"
    recon_valid_path = cache_dir / "recon_valid.npy"
    recon_test_path = cache_dir / "recon_test.npy"
    mask_train_path = cache_dir / "mask_pred_train.npy"
    mask_valid_path = cache_dir / "mask_pred_valid.npy"
    mask_test_path = cache_dir / "mask_pred_test.npy"
    aux_train_path = cache_dir / "aux_train.npy"
    aux_valid_path = cache_dir / "aux_valid.npy"
    aux_test_path = cache_dir / "aux_test.npy"

    (
        values_train,
        values_valid,
        values_test,
        observed_train,
        observed_valid,
        observed_test,
        scaler,
    ) = fit_dense_preprocessor(data, columns, spec.training_subset)

    required = [recon_train_path, recon_valid_path, recon_test_path, contract_path]
    if spec.append_latent:
        required += [latent_train_path, latent_valid_path, latent_test_path]
    if spec.reconstruct_mask:
        required += [mask_train_path, mask_valid_path, mask_test_path]
    if spec.supervised_auxiliary:
        required += [aux_train_path, aux_valid_path, aux_test_path]
    if all(path.exists() for path in required):
        log(f"Loading cached broad AE features: {cache_name}.")
        return DenseFeatureBundle(
            source_dir=cache_dir,
            columns=columns,
            latent_train=np.load(latent_train_path) if spec.append_latent else None,
            latent_valid=np.load(latent_valid_path) if spec.append_latent else None,
            latent_test=np.load(latent_test_path) if spec.append_latent else None,
            recon_train=np.load(recon_train_path),
            recon_valid=np.load(recon_valid_path),
            recon_test=np.load(recon_test_path),
            values_train=values_train,
            values_valid=values_valid,
            values_test=values_test,
            observed_train=observed_train,
            observed_valid=observed_valid,
            observed_test=observed_test,
            mask_pred_train=np.load(mask_train_path) if spec.reconstruct_mask else None,
            mask_pred_valid=np.load(mask_valid_path) if spec.reconstruct_mask else None,
            mask_pred_test=np.load(mask_test_path) if spec.reconstruct_mask else None,
            aux_train=np.load(aux_train_path) if spec.supervised_auxiliary else None,
            aux_valid=np.load(aux_valid_path) if spec.supervised_auxiliary else None,
            aux_test=np.load(aux_test_path) if spec.supervised_auxiliary else None,
        )

    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    X_train_input = make_inputs(values_train, observed_train, spec.include_input_mask)
    X_valid_input = make_inputs(values_valid, observed_valid, spec.include_input_mask)
    X_test_input = make_inputs(values_test, observed_test, spec.include_input_mask)
    y_train_target = masked_target(values_train, observed_train)
    y_valid_target = masked_target(values_valid, observed_valid)

    if spec.training_subset == "normal":
        fit_mask = data.y_train.to_numpy(dtype=int) == 0
        valid_fit_mask = data.y_valid.to_numpy(dtype=int) == 0
        X_fit = X_train_input[fit_mask]
        y_fit = y_train_target[fit_mask]
        X_valid_fit = X_valid_input[valid_fit_mask]
        y_valid_fit = y_valid_target[valid_fit_mask]
    else:
        X_fit = X_train_input
        y_fit = y_train_target
        X_valid_fit = X_valid_input
        y_valid_fit = y_valid_target

    value_dim = values_train.shape[1]
    model, encoder = build_dense_autoencoder(
        value_dim=value_dim,
        input_dim=X_train_input.shape[1],
        latent_dim=spec.latent_dim,
        learning_rate=learning_rate,
        reconstruct_mask=spec.reconstruct_mask,
        supervised_auxiliary=spec.supervised_auxiliary,
        mask_loss_weight=spec.mask_loss_weight,
        aux_loss_weight=spec.aux_loss_weight,
    )
    log(
        f"Training {cache_name}: columns={len(columns)} input_dim={X_train_input.shape[1]} "
        f"latent_dim={spec.latent_dim} subset={spec.training_subset}."
    )
    if spec.supervised_auxiliary:
        fit_y = {
            "reconstruction_values": y_fit,
            "fraud_aux": data.y_train.to_numpy(dtype="float32")[fit_mask] if spec.training_subset == "normal" else data.y_train.to_numpy(dtype="float32"),
        }
        valid_y = {
            "reconstruction_values": y_valid_fit,
            "fraud_aux": data.y_valid.to_numpy(dtype="float32")[valid_fit_mask] if spec.training_subset == "normal" else data.y_valid.to_numpy(dtype="float32"),
        }
    else:
        fit_y = y_fit
        valid_y = y_valid_fit

    history = model.fit(
        X_fit,
        fit_y,
        validation_data=(X_valid_fit, valid_y),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            ),
            keras.callbacks.CSVLogger(str(cache_dir / "ae_training_live_log.csv"), append=False),
        ],
        verbose=2,
    )
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(cache_dir / "ae_training_history.csv", index=False)
    model.save(cache_dir / "autoencoder.keras", include_optimizer=False)
    encoder.save(cache_dir / "encoder.keras", include_optimizer=False)

    (
        latent_train,
        latent_valid,
        latent_test,
        recon_train,
        recon_valid,
        recon_test,
        mask_train,
        mask_valid,
        mask_test,
        aux_train,
        aux_valid,
        aux_test,
    ) = predict_bundle_outputs(
        model,
        encoder,
        X_train_input,
        X_valid_input,
        X_test_input,
        value_dim,
        spec,
        batch_size,
    )
    if latent_train is not None:
        np.save(latent_train_path, latent_train)
        np.save(latent_valid_path, latent_valid)
        np.save(latent_test_path, latent_test)
    np.save(recon_train_path, recon_train)
    np.save(recon_valid_path, recon_valid)
    np.save(recon_test_path, recon_test)
    if mask_train is not None:
        np.save(mask_train_path, mask_train)
        np.save(mask_valid_path, mask_valid)
        np.save(mask_test_path, mask_test)
    if aux_train is not None:
        np.save(aux_train_path, aux_train)
        np.save(aux_valid_path, aux_valid)
        np.save(aux_test_path, aux_test)

    save_json(
        {
            "candidate": asdict(spec),
            "columns": columns,
            "column_count": len(columns),
            "value_dim": value_dim,
            "input_dim": int(X_train_input.shape[1]),
            "scaler": scaler,
            "loss": (
                "observed-only reconstruction + auxiliary fraud BCE"
                if spec.supervised_auxiliary
                else "observed-only reconstruction + observed-mask BCE"
                if spec.reconstruct_mask
                else "observed-only reconstruction"
            ),
            "training": {
                "max_epochs": max_epochs,
                "patience": patience,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "seed": seed,
            },
        },
        contract_path,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    return DenseFeatureBundle(
        source_dir=cache_dir,
        columns=columns,
        latent_train=latent_train,
        latent_valid=latent_valid,
        latent_test=latent_test,
        recon_train=recon_train,
        recon_valid=recon_valid,
        recon_test=recon_test,
        values_train=values_train,
        values_valid=values_valid,
        values_test=values_test,
        observed_train=observed_train,
        observed_valid=observed_valid,
        observed_test=observed_test,
        mask_pred_train=mask_train,
        mask_pred_valid=mask_valid,
        mask_pred_test=mask_test,
        aux_train=aux_train,
        aux_valid=aux_valid,
        aux_test=aux_test,
    )


def reconstruction_error_frame(bundle: DenseFeatureBundle, split: str, prefix: str) -> pd.DataFrame:
    if split == "train":
        values, recon, observed = bundle.values_train, bundle.recon_train, bundle.observed_train
        mask_pred, aux = bundle.mask_pred_train, bundle.aux_train
    elif split == "valid":
        values, recon, observed = bundle.values_valid, bundle.recon_valid, bundle.observed_valid
        mask_pred, aux = bundle.mask_pred_valid, bundle.aux_valid
    elif split == "test":
        values, recon, observed = bundle.values_test, bundle.recon_test, bundle.observed_test
        mask_pred, aux = bundle.mask_pred_test, bundle.aux_test
    else:
        raise ValueError(f"Unsupported split: {split}")

    abs_error = np.abs(values - recon).astype("float32")
    squared_error = np.square(values - recon).astype("float32")
    denominator = np.maximum(observed.sum(axis=1), 1.0)
    observed_abs = abs_error * observed
    observed_sq = squared_error * observed
    max_abs = np.max(observed_abs, axis=1)
    frame = pd.DataFrame(
        {
            f"{prefix}_mse": (observed_sq.sum(axis=1) / denominator).astype("float32"),
            f"{prefix}_mae": (observed_abs.sum(axis=1) / denominator).astype("float32"),
            f"{prefix}_max_abs": max_abs.astype("float32"),
            f"{prefix}_observed_rate": observed.mean(axis=1).astype("float32"),
        }
    )
    if mask_pred is not None:
        eps = 1e-7
        clipped = np.clip(mask_pred, eps, 1.0 - eps)
        mask_bce = -(observed * np.log(clipped) + (1.0 - observed) * np.log(1.0 - clipped))
        frame[f"{prefix}_mask_bce"] = mask_bce.mean(axis=1).astype("float32")
        frame[f"{prefix}_mask_mae"] = np.abs(observed - mask_pred).mean(axis=1).astype("float32")
        frame[f"{prefix}_pred_observed_rate"] = mask_pred.mean(axis=1).astype("float32")
    if aux is not None:
        frame[f"{prefix}_aux_fraud_score"] = aux.astype("float32")
    return frame.replace([np.inf, -np.inf], np.nan)


def latent_frame(latent: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = [f"{prefix}_latent_{index:03d}" for index in range(1, latent.shape[1] + 1)]
    return pd.DataFrame(latent, columns=columns)


def append_bundle_features(
    base_train: pd.DataFrame,
    base_valid: pd.DataFrame,
    base_test: pd.DataFrame,
    bundle: DenseFeatureBundle,
    prefix: str,
    append_latent: bool,
    append_error: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    extras_train: list[pd.DataFrame] = []
    extras_valid: list[pd.DataFrame] = []
    extras_test: list[pd.DataFrame] = []
    if append_latent:
        if bundle.latent_train is None or bundle.latent_valid is None or bundle.latent_test is None:
            raise ValueError("Latent features requested but bundle has no latents.")
        extras_train.append(latent_frame(bundle.latent_train, prefix))
        extras_valid.append(latent_frame(bundle.latent_valid, prefix))
        extras_test.append(latent_frame(bundle.latent_test, prefix))
    if append_error:
        extras_train.append(reconstruction_error_frame(bundle, "train", prefix))
        extras_valid.append(reconstruction_error_frame(bundle, "valid", prefix))
        extras_test.append(reconstruction_error_frame(bundle, "test", prefix))

    def combine(base: pd.DataFrame, extras: list[pd.DataFrame]) -> pd.DataFrame:
        if not extras:
            return base
        return pd.concat([base.reset_index(drop=True), *[item.reset_index(drop=True) for item in extras]], axis=1)

    return combine(base_train, extras_train), combine(base_valid, extras_valid), combine(base_test, extras_test)


def feature_groups(data: BroadData) -> dict[str, list[str]]:
    columns = data.X_base_train.columns.tolist()
    used: set[str] = set()

    def select(name: str, predicate) -> list[str]:
        selected = [column for column in columns if column not in used and predicate(column)]
        used.update(selected)
        return selected

    groups = {
        "v": select("v", lambda c: c in data.v_columns),
        "identity_device": select(
            "identity_device",
            lambda c: c.startswith("id_") or c in {"DeviceType", "DeviceInfo"},
        ),
        "payment_identity": select(
            "payment_identity",
            lambda c: c.startswith("card")
            or c.startswith("addr")
            or c in {"ProductCD", "P_emaildomain", "R_emaildomain"},
        ),
        "amount_time_behavior": select(
            "amount_time_behavior",
            lambda c: c in {"TransactionAmt", "TransactionDT"}
            or c.startswith("C")
            or c.startswith("D")
            or c.startswith("dist"),
        ),
        "match_flags": select("match_flags", lambda c: c.startswith("M") and c[1:].isdigit()),
    }
    remaining = [column for column in columns if column not in used]
    if remaining:
        groups["other"] = remaining
    return {name: selected for name, selected in groups.items() if selected}


def build_groupwise_matrices(
    data: BroadData,
    spec: BroadCandidate,
    output_dir: Path,
    importance_scores: dict[str, float],
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    group_dir = ensure_dir(output_dir / "feature_cache" / spec.candidate_id)
    X_train = data.X_base_train.copy()
    X_valid = data.X_base_valid.copy()
    X_test = data.X_base_test.copy()
    group_manifest: dict[str, object] = {}
    for idx, (group_name, columns) in enumerate(feature_groups(data).items(), start=1):
        selected_columns = select_top_columns(columns, importance_scores, spec.group_top_k)
        group_spec = BroadCandidate(
            stage=spec.stage,
            candidate_id=f"{spec.candidate_id}_{group_name}",
            description=f"Group-wise AE for {group_name}",
            kind="group_component",
            latent_dim=spec.group_latent_dim,
            training_subset="all",
            include_input_mask=True,
            append_latent=True,
            append_error=True,
        )
        bundle = train_or_load_dense_bundle(
            data=data,
            spec=group_spec,
            output_dir=output_dir,
            columns=selected_columns,
            cache_name=f"{spec.candidate_id}_{group_name}",
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed + spec.stage * 100 + idx,
        )
        X_train, X_valid, X_test = append_bundle_features(
            X_train,
            X_valid,
            X_test,
            bundle,
            prefix=f"ae_group_{group_name}",
            append_latent=True,
            append_error=True,
        )
        group_manifest[group_name] = {
            "columns": selected_columns,
            "column_count": len(selected_columns),
            "original_column_count": len(columns),
            "source_dir": str(bundle.source_dir),
            "latent_dim": spec.group_latent_dim,
        }
        del bundle
        gc.collect()
    save_json(group_manifest, group_dir / "group_manifest.json")
    return X_train, X_valid, X_test, data.baseline_categorical_columns, group_dir


def build_candidate_matrices(
    data: BroadData,
    spec: BroadCandidate,
    output_dir: Path,
    importance_scores: dict[str, float],
    max_epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    if spec.kind == "groupwise":
        return build_groupwise_matrices(
            data=data,
            spec=spec,
            output_dir=output_dir,
            importance_scores=importance_scores,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )

    columns = select_top_columns(
        data.X_base_train.columns.tolist(),
        importance_scores,
        spec.top_feature_count,
    )
    bundle = train_or_load_dense_bundle(
        data=data,
        spec=spec,
        output_dir=output_dir,
        columns=columns,
        cache_name=spec.candidate_id,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed + spec.stage,
    )
    X_train, X_valid, X_test = append_bundle_features(
        data.X_base_train,
        data.X_base_valid,
        data.X_base_test,
        bundle,
        prefix=spec.candidate_id,
        append_latent=spec.append_latent,
        append_error=spec.append_error,
    )
    return X_train, X_valid, X_test, data.baseline_categorical_columns, bundle.source_dir


def evaluate_broad_candidate(
    data: BroadData,
    spec: BroadCandidate,
    output_dir: Path,
    param_profiles: dict[str, dict[str, object]],
    reference_scores: np.ndarray,
    n_bootstrap: int,
    seed: int,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
    feature_source_dir: Path,
) -> dict[str, object]:
    candidate_dir = ensure_dir(output_dir / spec.candidate_id)
    save_json(asdict(spec), candidate_dir / "candidate_spec.json")
    profile_results = []
    for profile_name, params in param_profiles.items():
        profile_results.append(
            train_with_profile(
                data=data,
                output_dir=candidate_dir,
                candidate_id=spec.candidate_id,
                profile_name=profile_name,
                params=params,
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                categorical_columns=categorical_columns,
            )
        )
        gc.collect()

    selected = max(
        profile_results,
        key=lambda item: item["compact"]["validation_average_precision"],  # type: ignore[index]
    )
    selected_compact = selected["compact"]  # type: ignore[assignment]
    comparison = paired_bootstrap_ap_delta(
        data.y_test.to_numpy(dtype=int),
        reference_scores,
        selected["test_score"],  # type: ignore[arg-type]
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = {
        "candidate": asdict(spec),
        "feature_source_dir": str(feature_source_dir),
        "feature_count": int(X_train.shape[1]),
        "categorical_feature_count": len(categorical_columns),
        "selected_profile": selected["profile_name"],
        "selected_result": selected_compact,
        "delta_vs_baseline_tuned_test_ap": float(
            selected_compact["test_average_precision"] - BASELINE_REFERENCE_AP  # type: ignore[index]
        ),
        "comparison_vs_baseline_tuned": comparison,
        "profile_results": [
            {
                "profile_name": item["profile_name"],
                "output_dir": item["output_dir"],
                "compact": item["compact"],
            }
            for item in profile_results
        ],
        "beats_baseline_tuned": bool(selected_compact["test_average_precision"] > BASELINE_REFERENCE_AP),  # type: ignore[index]
    }
    save_json(summary, candidate_dir / "candidate_summary.json")
    log(
        f"{spec.candidate_id}: selected={summary['selected_profile']} "
        f"val_AP={selected_compact['validation_average_precision']:.6f} "
        f"test_AP={selected_compact['test_average_precision']:.6f} "
        f"delta={summary['delta_vs_baseline_tuned_test_ap']:+.6f}"
    )
    return summary


def candidate_sequence() -> list[BroadCandidate]:
    return [
        BroadCandidate(
            stage=1,
            candidate_id="broad1_all_feature_masked_latent_error_ld64",
            description="AE on top cross-family proposal features; append latent and reconstruction-error features.",
            kind="all_feature",
            latent_dim=64,
            training_subset="all",
            include_input_mask=True,
            append_latent=True,
            append_error=True,
            top_feature_count=192,
        ),
        BroadCandidate(
            stage=2,
            candidate_id="broad2_groupwise_ae_latent_error",
            description="Group-wise AEs for V, identity, payment, behavior, and match features.",
            kind="groupwise",
            group_latent_dim=16,
            group_top_k=96,
        ),
        BroadCandidate(
            stage=3,
            candidate_id="broad3_all_feature_value_mask_recon_ld64",
            description="AE reconstructs both values and observed/missing masks; append latent, value error, and mask error.",
            kind="all_feature_value_mask",
            latent_dim=64,
            training_subset="all",
            include_input_mask=True,
            reconstruct_mask=True,
            append_latent=True,
            append_error=True,
            top_feature_count=192,
        ),
        BroadCandidate(
            stage=4,
            candidate_id="broad4_normal_only_all_feature_anomaly",
            description="Normal-only all-feature AE; append anomaly reconstruction features only.",
            kind="normal_only_anomaly",
            latent_dim=64,
            training_subset="normal",
            include_input_mask=True,
            append_latent=False,
            append_error=True,
            top_feature_count=192,
        ),
        BroadCandidate(
            stage=5,
            candidate_id="broad5_supervised_aux_all_feature_ld64",
            description="All-feature AE with auxiliary fraud head; append latent, reconstruction error, and aux score.",
            kind="supervised_auxiliary",
            latent_dim=64,
            training_subset="all",
            include_input_mask=True,
            supervised_auxiliary=True,
            append_latent=True,
            append_error=True,
            append_aux_score=True,
            aux_loss_weight=0.05,
            top_feature_count=192,
        ),
    ]


def run_broad_ladder(
    output_dir: Path,
    original_output: Path,
    baseline_params_path: Path,
    ae_params_path: Path,
    reference_scores_path: Path,
    baseline_importance_path: Path,
    ae_max_epochs: int,
    ae_patience: int,
    ae_batch_size: int,
    ae_learning_rate: float,
    max_lgbm_estimators: int | None,
    n_bootstrap: int,
    n_jobs: int,
    seed: int,
    candidates: list[str] | None,
) -> dict[str, object]:
    set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    output_dir = ensure_dir(output_dir)
    data = prepare_broad_data(output_dir, seed=seed)
    reference_scores = load_reference_scores(reference_scores_path, data.y_test)
    reference_ap = float(average_precision_score(data.y_test, reference_scores))
    if abs(reference_ap - BASELINE_REFERENCE_AP) > 1e-12:
        log(f"Reference AP from scores is {reference_ap:.12f}; constant is {BASELINE_REFERENCE_AP:.12f}.")

    param_profiles = {
        "baseline_tuned": load_params(baseline_params_path, n_jobs=n_jobs, seed=seed),
        "ae_tuned_ld32": load_params(ae_params_path, n_jobs=n_jobs, seed=seed),
    }
    if max_lgbm_estimators is not None:
        for params in param_profiles.values():
            params["n_estimators"] = min(int(params["n_estimators"]), max_lgbm_estimators)
    importance_scores = load_importance_scores(baseline_importance_path)

    specs = candidate_sequence()
    if candidates:
        wanted = set(candidates)
        specs = [spec for spec in specs if spec.candidate_id in wanted]
        missing = wanted - {spec.candidate_id for spec in specs}
        if missing:
            raise ValueError(f"Unknown candidate id(s): {sorted(missing)}")
    single_candidate_mode = len(specs) == 1

    results: list[dict[str, object]] = []
    winning_candidate: dict[str, object] | None = None
    stopped_after_stage: int | None = None
    for spec in specs:
        cached_summary_path = output_dir / spec.candidate_id / "candidate_summary.json"
        if cached_summary_path.exists():
            log(f"Loading cached candidate summary: {spec.candidate_id}.")
            result = load_json(cached_summary_path)
        else:
            log(f"Starting candidate {spec.candidate_id}.")
            X_train, X_valid, X_test, categorical_columns, feature_source_dir = build_candidate_matrices(
                data=data,
                spec=spec,
                output_dir=output_dir,
                importance_scores=importance_scores,
                max_epochs=ae_max_epochs,
                patience=ae_patience,
                batch_size=ae_batch_size,
                learning_rate=ae_learning_rate,
                seed=seed,
            )
            if single_candidate_mode:
                log("Single-candidate mode: releasing base/missingness matrices before LightGBM evaluation.")
                release_feature_storage(data)
            result = evaluate_broad_candidate(
                data=data,
                spec=spec,
                output_dir=output_dir,
                param_profiles=param_profiles,
                reference_scores=reference_scores,
                n_bootstrap=n_bootstrap,
                seed=seed + spec.stage,
                X_train=X_train,
                X_valid=X_valid,
                X_test=X_test,
                categorical_columns=categorical_columns,
                feature_source_dir=feature_source_dir,
            )
            del X_train, X_valid, X_test
            gc.collect()

        results.append(result)
        if result["beats_baseline_tuned"]:
            winning_candidate = result
            stopped_after_stage = int(result["candidate"]["stage"])  # type: ignore[index]
            log(f"Stop rule met by {result['candidate']['candidate_id']}.")  # type: ignore[index]
            break

    summary = {
        "experiment": "broad_ae_feature_ladder",
        "output_dir": str(output_dir),
        "source_original_output": str(original_output),
        "split_strategy": "stratified_holdout",
        "split_ratios": {"train": TRAIN_RATIO, "validation": VALID_RATIO, "test": TEST_RATIO},
        "seed": seed,
        "sample_size": SAMPLE_SIZE,
        "reference": {
            "name": "original_proposal_baseline_tuned",
            "validation_average_precision": BASELINE_REFERENCE_VALID_AP,
            "test_average_precision": reference_ap,
            "reference_scores": str(reference_scores_path),
        },
        "ae_training": {
            "max_epochs": ae_max_epochs,
            "patience": ae_patience,
            "batch_size": ae_batch_size,
            "learning_rate": ae_learning_rate,
        },
        "lightgbm_runtime": {
            "max_lgbm_estimators": max_lgbm_estimators,
            "note": "If set, tuned LightGBM profile n_estimators are capped only to avoid local native LightGBM crashes on augmented matrices.",
        },
        "feature_selection": {
            "baseline_importance_path": str(baseline_importance_path),
            "score": "importance_gain",
            "note": "Broad all-feature candidates use top cross-family features to avoid unstable full-matrix AE training.",
        },
        "stop_rule": "stop after first selected candidate with test AP > tuned LightGBM reference",
        "stopped_after_stage": stopped_after_stage,
        "winning_candidate": winning_candidate,
        "results": results,
    }
    save_json(summary, output_dir / "broad_ladder_summary.json")
    print_summary(summary)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print()
    print("Broad AE Feature Ladder")
    print("=======================")
    ref = summary["reference"]  # type: ignore[index]
    print(f"Reference tuned LightGBM test AP: {ref['test_average_precision']:.6f}")
    print()
    for result in summary["results"]:  # type: ignore[union-attr]
        selected = result["selected_result"]
        print(
            f"{result['candidate']['candidate_id']:48s} "
            f"profile={result['selected_profile']:15s} "
            f"val_AP={selected['validation_average_precision']:.6f} "
            f"test_AP={selected['test_average_precision']:.6f} "
            f"delta={result['delta_vs_baseline_tuned_test_ap']:+.6f}"
        )
    if summary["winning_candidate"] is None:
        print("\nNo candidate beat the tuned LightGBM reference.")
    else:
        winner = summary["winning_candidate"]
        print(f"\nWinner: {winner['candidate']['candidate_id']}")  # type: ignore[index]
    print(f"\nSaved: {summary['output_dir']}/broad_ladder_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run broader AE feature-learning ladder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stratified_reset/broad_ae_feature_ladder"),
    )
    parser.add_argument("--original-output", type=Path, default=DEFAULT_ORIGINAL_OUTPUT)
    parser.add_argument(
        "--baseline-params-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "baseline_tpe_best_params.json",
    )
    parser.add_argument(
        "--ae-params-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "ae_latent_replacement_tpe_best_params.json",
    )
    parser.add_argument(
        "--reference-scores-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "baseline_tuned" / "baseline_tuned_test_scores.csv",
    )
    parser.add_argument(
        "--baseline-importance-path",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT / "baseline_tuned" / "feature_importance.csv",
    )
    parser.add_argument("--ae-max-epochs", type=int, default=50)
    parser.add_argument("--ae-patience", type=int, default=7)
    parser.add_argument("--ae-batch-size", type=int, default=2048)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-lgbm-estimators",
        type=int,
        default=None,
        help="Optional runtime cap for tuned LightGBM n_estimators on augmented matrices.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--candidates",
        nargs="*",
        help="Optional candidate ids to run. Default runs the full sequence.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_broad_ladder(
        output_dir=args.output_dir,
        original_output=args.original_output,
        baseline_params_path=args.baseline_params_path,
        ae_params_path=args.ae_params_path,
        reference_scores_path=args.reference_scores_path,
        baseline_importance_path=args.baseline_importance_path,
        ae_max_epochs=args.ae_max_epochs,
        ae_patience=args.ae_patience,
        ae_batch_size=args.ae_batch_size,
        ae_learning_rate=args.ae_learning_rate,
        max_lgbm_estimators=args.max_lgbm_estimators,
        n_bootstrap=args.n_bootstrap,
        n_jobs=args.n_jobs,
        seed=args.seed,
        candidates=args.candidates,
    )
