"""Train task-aware selected-numerical Autoencoder (TAE01) with lambda ablation."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "TensorFlow is not installed. Install project requirements with "
        "`pip install -r requirements.txt`, then rerun this script."
    ) from exc

from autoencoder_helpers import (
    apply_frozen_median_scaled_feature_block,
    build_task_aware_autoencoder,
    output_dir_is_non_empty,
)
from config import (
    AE_BATCH_SIZE,
    AE_CLIP_MAX,
    AE_CLIP_MIN,
    AE_LEARNING_RATE,
    AE_MAX_EPOCHS,
    AE_PATIENCE,
    AE_USE_SCALED_CLIPPING,
    AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    DATA_DIR,
    FINAL_COMPARISON_OUTPUT_DIR,
    ID_COL,
    RANDOM_SEED,
    SAMPLE_SIZE,
    SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    TARGET_COL,
    TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR,
    TASK_AWARE_LAMBDA_SELECTION_FILE,
    TEST_RATIO,
    TIME_COL,
    TRAIN_RATIO,
    VALID_RATIO,
)
from data_loader import load_labeled_train_data
from splitting import chronological_split
from train_task_aware_ae_lgbm import train_task_aware_downstream_lgbm
from utils import ensure_dir, log, save_json, set_seed

SELECTED_NUMERICAL_LATENT_DIM = 128
EXPERIMENT_NAME = "task_aware_autoencoder_selected_numerical_ld128"
LAMBDA_CANDIDATES = [0.1, 0.5, 1.0]
SELECTED_AE_SUBDIR = "selected"


def lambda_dir_name(lambda_classification: float) -> str:
    token = str(lambda_classification).replace(".", "_")
    return f"lambda_{token}"


def load_selected_numerical_features(audit_file: Path) -> dict[str, object]:
    if not audit_file.exists():
        raise FileNotFoundError(
            f"Selected numerical feature audit not found: {audit_file}\n"
            "Run Phase 1 feature audit before training."
        )
    with audit_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    required_keys = {
        "feature_names",
        "feature_count",
        "v_feature_names",
        "additional_numerical_feature_names",
        "target_not_used",
    }
    missing_keys = required_keys - set(payload)
    if missing_keys:
        raise KeyError(f"Audit JSON missing required keys: {sorted(missing_keys)}")
    return payload


def validate_selected_features(
    feature_names: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    forbidden = {TARGET_COL, ID_COL, TIME_COL}
    leaked = sorted(set(feature_names) & forbidden)
    if leaked:
        raise ValueError(f"Forbidden columns found in AE input: {leaked}")

    for split_name, split_df in (
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ):
        missing = [column for column in feature_names if column not in split_df.columns]
        if missing:
            raise KeyError(
                f"{split_name} split is missing selected AE feature(s): "
                + ", ".join(missing[:10])
            )


def load_frozen_preprocessing(
    source_dir: Path,
    feature_names: list[str],
) -> tuple[object, object, list[str]]:
    imputer_path = source_dir / "numerical_imputer.pkl"
    scaler_path = source_dir / "numerical_scaler.pkl"
    feature_names_path = source_dir / "selected_numerical_feature_names.json"

    for path in (imputer_path, scaler_path, feature_names_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen preprocessing artifact from AAE01: {path}"
            )

    with feature_names_path.open("r", encoding="utf-8") as file:
        frozen_feature_names = json.load(file)

    if frozen_feature_names != feature_names:
        raise ValueError(
            "Frozen AAE01 feature list does not match audit feature list."
        )

    return joblib.load(imputer_path), joblib.load(scaler_path), frozen_feature_names


def classification_head_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    ap = float(average_precision_score(y_true, y_prob))
    if len(np.unique(y_true)) < 2:
        roc_auc = 0.0
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    return {
        "classification_head_validation_ap": ap,
        "classification_head_validation_roc_auc": roc_auc,
    }


class ValidationDiagnosticsCallback(keras.callbacks.Callback):
    """Log classification-head validation AP/ROC-AUC each epoch."""

    def __init__(self, X_valid: np.ndarray, y_valid: np.ndarray) -> None:
        super().__init__()
        self.X_valid = X_valid
        self.y_valid = y_valid.astype("float32").reshape(-1, 1)
        self.records: list[dict[str, float | int]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        _, y_prob = self.model.predict(self.X_valid, batch_size=AE_BATCH_SIZE, verbose=0)
        head_metrics = classification_head_metrics(
            self.y_valid.reshape(-1),
            y_prob.reshape(-1),
        )
        record = {
            "epoch": epoch + 1,
            "validation_total_loss": float(logs.get("val_loss", np.nan)),
            "validation_reconstruction_loss": float(
                logs.get("val_reconstruction_loss", np.nan)
            ),
            "validation_classification_loss": float(
                logs.get("val_fraud_probability_loss", np.nan)
            ),
            **head_metrics,
        }
        self.records.append(record)


def save_latent_features(
    encoder: keras.Model,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray | None,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    log("Encoding latent features.")
    latent_train = encoder.predict(X_train, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    latent_valid = encoder.predict(X_valid, batch_size=AE_BATCH_SIZE, verbose=0).astype(
        "float32"
    )
    latent_test = None
    np.save(output_dir / "latent_train.npy", latent_train)
    np.save(output_dir / "latent_valid.npy", latent_valid)
    if X_test is not None:
        latent_test = encoder.predict(X_test, batch_size=AE_BATCH_SIZE, verbose=0).astype(
            "float32"
        )
        np.save(output_dir / "latent_test.npy", latent_test)
    return latent_train, latent_valid, latent_test


def validate_latent_arrays(
    latent_train: np.ndarray,
    latent_valid: np.ndarray,
    latent_test: np.ndarray | None,
    latent_feature_names: list[str],
    train_rows: int,
    valid_rows: int,
    test_rows: int,
    latent_dim: int,
) -> None:
    expected = {
        "train": (latent_train, train_rows),
        "validation": (latent_valid, valid_rows),
    }
    if latent_test is not None:
        expected["test"] = (latent_test, test_rows)

    for split_name, (latent, row_count) in expected.items():
        if latent.shape[0] != row_count:
            raise ValueError(
                f"{split_name} latent row count {latent.shape[0]} does not match "
                f"split row count {row_count}."
            )
    if latent_train.shape[1] != latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: expected {latent_dim}, "
            f"got {latent_train.shape[1]}."
        )
    for array_name, latent in (
        ("latent_train", latent_train),
        ("latent_valid", latent_valid),
    ):
        if not np.isfinite(latent).all():
            raise ValueError(f"{array_name} contains non-finite values.")
    if latent_test is not None and not np.isfinite(latent_test).all():
        raise ValueError("latent_test contains non-finite values.")
    if len(set(latent_feature_names)) != len(latent_feature_names):
        raise ValueError("Duplicate latent feature names found.")


def train_lambda_candidate(
    lambda_classification: float,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    audit: dict[str, object],
    positive_class_weight: float,
    output_dir: Path,
    imputer: object,
    scaler: object,
) -> dict[str, object]:
    candidate_dir = ensure_dir(output_dir / lambda_dir_name(lambda_classification))
    latent_feature_names = [
        f"tae_latent_{index:03d}" for index in range(1, SELECTED_NUMERICAL_LATENT_DIM + 1)
    ]

    log(f"Building task-aware Autoencoder for lambda={lambda_classification}.")
    autoencoder, encoder, classification_head = build_task_aware_autoencoder(
        input_dim=X_train.shape[1],
        latent_dim=SELECTED_NUMERICAL_LATENT_DIM,
        learning_rate=AE_LEARNING_RATE,
        lambda_classification=lambda_classification,
        positive_class_weight=positive_class_weight,
    )

    diagnostics_callback = ValidationDiagnosticsCallback(X_valid, y_valid)
    history = autoencoder.fit(
        X_train,
        {
            "reconstruction": X_train,
            "fraud_probability": y_train.reshape(-1, 1).astype("float32"),
        },
        validation_data=(
            X_valid,
            {
                "reconstruction": X_valid,
                "fraud_probability": y_valid.reshape(-1, 1).astype("float32"),
            },
        ),
        epochs=AE_MAX_EPOCHS,
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=AE_PATIENCE,
                restore_best_weights=True,
            ),
            diagnostics_callback,
        ],
        verbose=2,
    )

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(candidate_dir / "training_history.csv", index=False)

    best_epoch_index = int(history_df["val_loss"].idxmin())
    best_epoch = int(history_df.loc[best_epoch_index, "epoch"])
    validation_total_loss = float(history_df.loc[best_epoch_index, "val_loss"])
    validation_reconstruction_loss = float(
        history_df.loc[best_epoch_index, "val_reconstruction_loss"]
    )
    validation_classification_loss = float(
        history_df.loc[best_epoch_index, "val_fraud_probability_loss"]
    )
    convergence_limitation = best_epoch >= AE_MAX_EPOCHS

    diagnostics_df = pd.DataFrame(diagnostics_callback.records)
    diagnostics_df.to_csv(candidate_dir / "validation_diagnostics.csv", index=False)
    best_diag = diagnostics_df.loc[diagnostics_df["epoch"] == best_epoch].iloc[0]
    validation_diagnostics = {
        "best_epoch": best_epoch,
        "validation_total_loss": validation_total_loss,
        "validation_reconstruction_loss": validation_reconstruction_loss,
        "validation_classification_loss": validation_classification_loss,
        "classification_head_validation_ap": float(
            best_diag["classification_head_validation_ap"]
        ),
        "classification_head_validation_roc_auc": float(
            best_diag["classification_head_validation_roc_auc"]
        ),
        "convergence_limitation": convergence_limitation,
        "classification_head_metrics_diagnostic_only": True,
        "lambda_selection_uses_downstream_lgbm_validation_ap": True,
    }
    save_json(validation_diagnostics, candidate_dir / "validation_diagnostics.json")

    latent_train, latent_valid, _ = save_latent_features(
        encoder,
        X_train,
        X_valid,
        None,
        candidate_dir,
    )
    validate_latent_arrays(
        latent_train,
        latent_valid,
        None,
        latent_feature_names,
        X_train.shape[0],
        X_valid.shape[0],
        0,
        SELECTED_NUMERICAL_LATENT_DIM,
    )

    autoencoder.save(candidate_dir / "autoencoder_model.keras")
    encoder.save(candidate_dir / "encoder_model.keras")
    classification_head.save(candidate_dir / "classification_head_model.keras")
    save_json(feature_names, candidate_dir / "selected_numerical_feature_names.json")
    save_json(latent_feature_names, candidate_dir / "latent_feature_names.json")

    run_config = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_purpose": (
            "Task-aware integration diagnostic: jointly optimize reconstruction and "
            "fraud classification during Autoencoder training, then replace selected "
            "numerical features with latent features in downstream LightGBM."
        ),
        "selected_numerical_feature_count": len(feature_names),
        "exact_feature_list_artifact": str(SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE),
        "latent_dimension": SELECTED_NUMERICAL_LATENT_DIM,
        "shared_encoder_architecture": [256, 128, SELECTED_NUMERICAL_LATENT_DIM],
        "decoder_architecture": [128, 256, len(feature_names)],
        "classification_head_architecture": [64, "dropout_0.2", 1],
        "reconstruction_loss": "mse",
        "classification_loss": "weighted_binary_crossentropy",
        "lambda_classification": lambda_classification,
        "class_weight": positive_class_weight,
        "preprocessing_source": str(AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR),
        "preprocessing_reused_from_aae01": True,
        "split_strategy": "chronological TransactionDT holdout",
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "maximum_epoch": AE_MAX_EPOCHS,
        "convergence_limitation": convergence_limitation,
        "target_used_only_in_classification_head": True,
        "target_not_used_in_preprocessing": True,
        "validation_used_for_model_selection": True,
        "test_not_used_for_lambda_selection": True,
        "test_not_used_for_architecture_selection": True,
        "v_feature_count": audit["v_feature_count"],
        "additional_numerical_feature_count": audit["additional_numerical_feature_count"],
        "selected_feature_names": feature_names,
        "excluded_transactiondt_from_ae": True,
        "excluded_identifiers": [ID_COL],
        "imputation_strategy": "train-median per feature (frozen AAE01 imputer)",
        "scaling_strategy": "StandardScaler (frozen AAE01 scaler)",
        "clipping_strategy": {
            "enabled": AE_USE_SCALED_CLIPPING,
            "clip_min": AE_CLIP_MIN,
            "clip_max": AE_CLIP_MAX,
        },
        "optimizer": "Adam",
        "learning_rate": AE_LEARNING_RATE,
        "batch_size": AE_BATCH_SIZE,
        "early_stopping_patience": AE_PATIENCE,
        "early_stopping_monitor": "val_loss",
        "data_dir": str(DATA_DIR),
        "output_dir": str(candidate_dir),
        "sample_size": SAMPLE_SIZE,
    }
    save_json(run_config, candidate_dir / "run_config.json")

    return {
        "lambda_classification": lambda_classification,
        "candidate_dir": candidate_dir,
        "latent_train": latent_train,
        "latent_valid": latent_valid,
        "latent_feature_names": latent_feature_names,
        "best_epoch": best_epoch,
        "validation_total_loss": validation_total_loss,
        "validation_reconstruction_loss": validation_reconstruction_loss,
        "validation_classification_loss": validation_classification_loss,
        "classification_head_validation_ap": float(
            best_diag["classification_head_validation_ap"]
        ),
        "validation_diagnostics": validation_diagnostics,
        "run_config": run_config,
        "convergence_limitation": convergence_limitation,
    }


def copy_selected_candidate(
    source_dir: Path,
    selected_dir: Path,
    selected_lambda: float,
) -> None:
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(source_dir, selected_dir)
    save_json(
        {
            "selected_lambda": selected_lambda,
            "selection_criterion": "downstream_lgbm_validation_ap",
            "source_candidate_directory": str(source_dir),
            "test_latent_extracted_only_for_selected_lambda": True,
        },
        selected_dir / "selected_lambda.json",
    )


def main(
    output_dir: Path = TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    audit_file: Path = SELECTED_NUMERICAL_AE_FEATURE_AUDIT_FILE,
    preprocessing_source_dir: Path = AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_dir_is_non_empty(output_dir) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is non-empty: {output_dir}\n"
            "Pass overwrite=True only when intentionally replacing TAE01 outputs."
        )

    set_seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    output_dir = ensure_dir(output_dir)

    audit = load_selected_numerical_features(audit_file)
    feature_names = list(audit["feature_names"])
    input_dim = len(feature_names)
    if input_dim != 387:
        raise ValueError(f"Expected 387 selected numerical features, got {input_dim}.")

    log("Loading labeled training data.")
    full_df = load_labeled_train_data(sample_size=SAMPLE_SIZE)

    log("Creating chronological train/validation/test split.")
    train_df, valid_df, test_df = chronological_split(full_df)
    validate_selected_features(feature_names, train_df, valid_df, test_df)

    log("Loading frozen median imputer and scaler from AAE01.")
    imputer, scaler, frozen_feature_names = load_frozen_preprocessing(
        preprocessing_source_dir,
        feature_names,
    )

    log("Applying frozen preprocessing to chronological splits.")
    X_train, X_valid, X_test = apply_frozen_median_scaled_feature_block(
        train_df,
        valid_df,
        test_df,
        frozen_feature_names,
        imputer,
        scaler,
        use_scaled_clipping=AE_USE_SCALED_CLIPPING,
        clip_min=AE_CLIP_MIN,
        clip_max=AE_CLIP_MAX,
    )

    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    y_valid = valid_df[TARGET_COL].astype(int).to_numpy()
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    positive_class_weight = negative_count / positive_count if positive_count else 1.0
    log(f"Classification positive class weight (train only): {positive_class_weight:.6f}")

    lambda_results: list[dict[str, object]] = []
    for lambda_classification in LAMBDA_CANDIDATES:
        log(f"=== Lambda candidate {lambda_classification} ===")
        candidate_result = train_lambda_candidate(
            lambda_classification=lambda_classification,
            X_train=X_train,
            X_valid=X_valid,
            y_train=y_train,
            y_valid=y_valid,
            feature_names=feature_names,
            audit=audit,
            positive_class_weight=positive_class_weight,
            output_dir=output_dir,
            imputer=imputer,
            scaler=scaler,
        )

        log(
            "Training validation-only downstream LightGBM for lambda selection "
            f"(lambda={lambda_classification})."
        )
        downstream_result = train_task_aware_downstream_lgbm(
            train_df=train_df,
            valid_df=valid_df,
            test_df=None,
            latent_train=candidate_result["latent_train"],
            latent_valid=candidate_result["latent_valid"],
            latent_test=None,
            latent_feature_names=candidate_result["latent_feature_names"],
            selected_numerical_features=feature_names,
            output_dir=(
                TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR
                / lambda_dir_name(lambda_classification)
            ),
            save_artifacts=False,
        )

        candidate_result["downstream_lgbm_validation_ap"] = downstream_result[
            "validation_average_precision"
        ]
        candidate_result["downstream_lgbm_best_iteration"] = downstream_result[
            "best_iteration"
        ]
        lambda_results.append(candidate_result)

    selection_rows = []
    best_result = max(
        lambda_results,
        key=lambda row: float(row["downstream_lgbm_validation_ap"]),
    )
    selected_lambda = float(best_result["lambda_classification"])

    for result in lambda_results:
        is_selected = float(result["lambda_classification"]) == selected_lambda
        selection_rows.append(
            {
                "lambda_classification": result["lambda_classification"],
                "ae_best_epoch": result["best_epoch"],
                "validation_total_loss": result["validation_total_loss"],
                "validation_reconstruction_loss": result["validation_reconstruction_loss"],
                "validation_classification_loss": result["validation_classification_loss"],
                "classification_head_validation_ap": result["classification_head_validation_ap"],
                "downstream_lgbm_validation_ap": result["downstream_lgbm_validation_ap"],
                "downstream_lgbm_best_iteration": result["downstream_lgbm_best_iteration"],
                "selected": is_selected,
                "selection_reason": (
                    "Highest downstream_lgbm_validation_ap among bounded lambda candidates."
                    if is_selected
                    else "Not selected."
                ),
                "metric_source": "downstream_lgbm_validation_average_precision",
                "run_config_source": str(result["candidate_dir"] / "run_config.json"),
            }
        )

    selection_df = pd.DataFrame(selection_rows)
    ensure_dir(FINAL_COMPARISON_OUTPUT_DIR)
    selection_df.to_csv(TASK_AWARE_LAMBDA_SELECTION_FILE, index=False)
    selection_df.to_csv(output_dir / "model_selection_summary.csv", index=False)

    log(f"Selected lambda={selected_lambda} based on downstream validation AP.")
    selected_source_dir = Path(best_result["candidate_dir"])
    selected_dir = ensure_dir(output_dir / SELECTED_AE_SUBDIR)
    copy_selected_candidate(selected_source_dir, selected_dir, selected_lambda)

    log("Extracting test latent features for selected lambda only.")
    encoder = keras.models.load_model(selected_source_dir / "encoder_model.keras")
    _, _, latent_test = save_latent_features(
        encoder,
        X_train,
        X_valid,
        X_test,
        selected_dir,
    )
    latent_train = np.load(selected_dir / "latent_train.npy")
    latent_valid = np.load(selected_dir / "latent_valid.npy")
    latent_feature_names = list(best_result["latent_feature_names"])
    validate_latent_arrays(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
        SELECTED_NUMERICAL_LATENT_DIM,
    )

    print()
    print("Task-Aware Autoencoder Lambda Selection")
    print("=======================================")
    for row in selection_rows:
        marker = " <-- SELECTED" if row["selected"] else ""
        print(
            f"lambda={row['lambda_classification']}: "
            f"downstream val AP={row['downstream_lgbm_validation_ap']:.6f}{marker}"
        )
    print(f"Outputs saved to: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "selected_lambda": selected_lambda,
        "selection_rows": selection_rows,
        "positive_class_weight": positive_class_weight,
    }


if __name__ == "__main__":
    main()