"""Shared identity-safe expert score generation for LF01 late fusion."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from causal_behavioral_features import transaction_id_checksum
from config import (
    AUTOENCODER_ROBUST_LD128_OUTPUT_DIR,
    BASELINE_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    ID_COL,
    OPTUNA_OUTPUT_DIR,
    TARGET_COL,
)
from preprocessing import (
    apply_baseline_preprocessing,
    get_v_feature_columns,
    split_features_target,
)
from train_ae_lgbm import (
    apply_non_v_preprocessing,
    combine_non_v_and_latent,
    load_robust_latent_outputs,
    split_non_v_features_target,
    validate_feature_alignment,
    validate_latent_outputs,
)
from train_causal_behavioral_lgbm import prepare_causal_behavioral_splits


CBA01R_OUTPUT_DIR = CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR
P04_OUTPUT_DIR = OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128"
P02_OUTPUT_DIR = OPTUNA_OUTPUT_DIR / "baseline_lgbm"
P01_OUTPUT_DIR = BASELINE_OUTPUT_DIR

REFERENCE_AP = {
    "CBA01R": {"validation": 0.6151223584166937, "test": 0.4938379650483288},
    "P04": {"validation": 0.610631101672744, "test": 0.490686},
    "P02": {"validation": 0.624072, "test": 0.501438},
    "P01": {"validation": 0.602433, "test": 0.485756},
}

BEHAVIORAL_WEIGHT_GRID = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
]

METRIC_TOLERANCE_STRICT = 1e-6
METRIC_TOLERANCE_RELAXED = 1e-4
AP_TIE_TOLERANCE = 1e-8
PRACTICAL_IMPROVEMENT_THRESHOLD = 0.002
BOOTSTRAP_RESAMPLES = 1000


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def require_files(paths: list[Path], context: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s) for {context}:\n" + "\n".join(missing)
        )


def best_iteration_from_config(model, run_config: dict[str, object]) -> int:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict) and early_stopping.get("best_iteration"):
        return int(early_stopping["best_iteration"])

    best_iteration = getattr(model, "best_iteration_", None)
    if best_iteration:
        return int(best_iteration)

    model_params = run_config.get("model_params", {})
    if isinstance(model_params, dict) and model_params.get("n_estimators"):
        return int(model_params["n_estimators"])

    n_estimators = getattr(model, "n_estimators", None)
    if n_estimators:
        return int(n_estimators)

    raise ValueError("Could not determine model best_iteration.")


def model_feature_names(model) -> list[str]:
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise ValueError("Loaded model does not expose LightGBM feature names.")


def validate_model_features(model, X: pd.DataFrame, model_name: str) -> None:
    expected = model_feature_names(model)
    observed = X.columns.tolist()
    if expected != observed:
        expected_only = [column for column in expected if column not in observed][:10]
        observed_only = [column for column in observed if column not in expected][:10]
        raise ValueError(
            f"{model_name} feature-name mismatch. "
            f"expected_count={len(expected)}, observed_count={len(observed)}, "
            f"expected_only={expected_only}, observed_only={observed_only}"
        )


def predict_scores(model, X: pd.DataFrame, best_iteration: int) -> np.ndarray:
    return model.predict_proba(X, num_iteration=best_iteration)[:, 1]


def validate_probabilities(scores: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{label}: non-finite probabilities detected.")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError(f"{label}: probabilities outside [0, 1].")


def validate_metric_reproduction(
    model_id: str,
    split_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    tolerance: float = METRIC_TOLERANCE_STRICT,
) -> dict[str, float]:
    reference = REFERENCE_AP[model_id][split_name]
    regenerated = float(average_precision_score(y_true, y_score))
    diff = abs(regenerated - reference)
    if diff > tolerance:
        raise ValueError(
            f"{model_id} {split_name} regenerated AP {regenerated:.12f} "
            f"differs from reference {reference:.12f} by {diff:.12f} "
            f"(tolerance {tolerance})."
        )
    return {
        "reference_ap": reference,
        "regenerated_ap": regenerated,
        "absolute_difference": diff,
    }


def build_cba01r_matrices(
    prepared: dict[str, object],
    preprocessing: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = apply_baseline_preprocessing(
        prepared["X_train_combined"],
        preprocessing,
    )
    X_valid = apply_baseline_preprocessing(
        prepared["X_valid_combined"],
        preprocessing,
    )
    X_test = apply_baseline_preprocessing(
        prepared["X_test_combined"],
        preprocessing,
    )
    return X_train, X_valid, X_test


def regenerate_cba01r_scores() -> dict[str, object]:
    require_files(
        [
            CBA01R_OUTPUT_DIR / "model.pkl",
            CBA01R_OUTPUT_DIR / "preprocessing.pkl",
            CBA01R_OUTPUT_DIR / "run_config.json",
        ],
        "CBA01R",
    )
    prepared = prepare_causal_behavioral_splits()
    model = joblib.load(CBA01R_OUTPUT_DIR / "model.pkl")
    preprocessing = joblib.load(CBA01R_OUTPUT_DIR / "preprocessing.pkl")
    run_config = load_json(CBA01R_OUTPUT_DIR / "run_config.json")

    _, X_valid, X_test = build_cba01r_matrices(prepared, preprocessing)
    validate_model_features(model, X_valid, "CBA01R")
    validate_model_features(model, X_test, "CBA01R")

    best_iteration = best_iteration_from_config(model, run_config)
    valid_score = predict_scores(model, X_valid, best_iteration)
    test_score = predict_scores(model, X_test, best_iteration)
    validate_probabilities(valid_score, "CBA01R validation")
    validate_probabilities(test_score, "CBA01R test")

    y_valid = prepared["y_valid"].to_numpy()
    y_test = prepared["y_test"].to_numpy()
    metric_checks = {
        "validation": validate_metric_reproduction("CBA01R", "validation", y_valid, valid_score),
        "test": validate_metric_reproduction("CBA01R", "test", y_test, test_score),
    }

    return {
        "prepared": prepared,
        "valid_df": prepared["valid_df"],
        "test_df": prepared["test_df"],
        "y_valid": prepared["y_valid"],
        "y_test": prepared["y_test"],
        "valid_score": valid_score,
        "test_score": test_score,
        "best_iteration": best_iteration,
        "metric_checks": metric_checks,
        "run_config": run_config,
    }


def regenerate_p04_scores(prepared: dict[str, object] | None = None) -> dict[str, object]:
    require_files(
        [
            P04_OUTPUT_DIR / "final_model.pkl",
            P04_OUTPUT_DIR / "preprocessing_non_v.pkl",
            P04_OUTPUT_DIR / "run_config.json",
        ],
        "P04",
    )
    if prepared is None:
        prepared = prepare_causal_behavioral_splits()

    train_df = prepared["train_df"]
    valid_df = prepared["valid_df"]
    test_df = prepared["test_df"]
    v_columns = get_v_feature_columns(train_df)

    model = joblib.load(P04_OUTPUT_DIR / "final_model.pkl")
    preprocessing = joblib.load(P04_OUTPUT_DIR / "preprocessing_non_v.pkl")
    run_config = load_json(P04_OUTPUT_DIR / "run_config.json")

    latent_train, latent_valid, latent_test, latent_feature_names, _ = (
        load_robust_latent_outputs(AUTOENCODER_ROBUST_LD128_OUTPUT_DIR)
    )
    validate_latent_outputs(
        latent_train,
        latent_valid,
        latent_test,
        latent_feature_names,
        len(train_df),
        len(valid_df),
        len(test_df),
    )
    if latent_valid.shape[1] != 128:
        raise ValueError(f"Expected LD128 latent features, found {latent_valid.shape[1]}.")

    X_valid_non_v_raw, y_valid = split_non_v_features_target(valid_df, v_columns)
    X_test_non_v_raw, y_test = split_non_v_features_target(test_df, v_columns)
    X_valid_non_v = apply_non_v_preprocessing(X_valid_non_v_raw, preprocessing)
    X_test_non_v = apply_non_v_preprocessing(X_test_non_v_raw, preprocessing)
    X_valid = combine_non_v_and_latent(X_valid_non_v, latent_valid, latent_feature_names)
    X_test = combine_non_v_and_latent(X_test_non_v, latent_test, latent_feature_names)
    validate_feature_alignment(X_valid, X_valid, X_test, v_columns)
    validate_model_features(model, X_valid, "P04")
    validate_model_features(model, X_test, "P04")

    best_iteration = best_iteration_from_config(model, run_config)
    valid_score = predict_scores(model, X_valid, best_iteration)
    test_score = predict_scores(model, X_test, best_iteration)
    validate_probabilities(valid_score, "P04 validation")
    validate_probabilities(test_score, "P04 test")

    y_valid_np = y_valid.to_numpy()
    y_test_np = y_test.to_numpy()
    metric_checks = {
        "validation": validate_metric_reproduction("P04", "validation", y_valid_np, valid_score),
        "test": validate_metric_reproduction("P04", "test", y_test_np, test_score),
    }

    return {
        "valid_df": valid_df,
        "test_df": test_df,
        "y_valid": y_valid,
        "y_test": y_test,
        "valid_score": valid_score,
        "test_score": test_score,
        "best_iteration": best_iteration,
        "metric_checks": metric_checks,
        "run_config": run_config,
        "latent_dim": latent_valid.shape[1],
    }


def regenerate_p02_scores(prepared: dict[str, object] | None = None) -> dict[str, object] | None:
    """Regenerate P02 scores when artifacts exist; return None if unavailable."""
    required = [
        P02_OUTPUT_DIR / "final_model.pkl",
        P02_OUTPUT_DIR / "preprocessing.pkl",
        P02_OUTPUT_DIR / "run_config.json",
    ]
    if not all(path.exists() for path in required):
        return None

    if prepared is None:
        prepared = prepare_causal_behavioral_splits()

    valid_df = prepared["valid_df"]
    test_df = prepared["test_df"]
    model = joblib.load(P02_OUTPUT_DIR / "final_model.pkl")
    preprocessing = joblib.load(P02_OUTPUT_DIR / "preprocessing.pkl")
    run_config = load_json(P02_OUTPUT_DIR / "run_config.json")

    X_valid_raw, y_valid = split_features_target(valid_df)
    X_test_raw, y_test = split_features_target(test_df)
    X_valid = apply_baseline_preprocessing(X_valid_raw, preprocessing)
    X_test = apply_baseline_preprocessing(X_test_raw, preprocessing)
    validate_model_features(model, X_valid, "P02")
    validate_model_features(model, X_test, "P02")

    best_iteration = best_iteration_from_config(model, run_config)
    valid_score = predict_scores(model, X_valid, best_iteration)
    test_score = predict_scores(model, X_test, best_iteration)
    validate_probabilities(valid_score, "P02 validation")
    validate_probabilities(test_score, "P02 test")

    return {
        "valid_score": valid_score,
        "test_score": test_score,
        "y_valid": y_valid,
        "y_test": y_test,
        "best_iteration": best_iteration,
    }


def build_single_expert_score_frame(
    split_df: pd.DataFrame,
    y: pd.Series,
    score: np.ndarray,
    score_column: str,
    split_name: str,
) -> pd.DataFrame:
    if len(split_df) != len(y):
        raise ValueError(f"{split_name}: split row count does not match labels.")
    if len(score) != len(split_df):
        raise ValueError(f"{split_name}: score length does not match split rows.")

    table = pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            score_column: score,
        }
    )
    if table[ID_COL].duplicated().any():
        raise ValueError(f"{split_name}: duplicate TransactionID in score frame.")
    validate_probabilities(table[score_column].to_numpy(), f"{split_name} {score_column}")
    return table


def align_expert_scores_by_transaction_id(
    split_df: pd.DataFrame,
    y: pd.Series,
    cba_score: np.ndarray,
    p04_score: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    cba_frame = build_single_expert_score_frame(
        split_df,
        y,
        cba_score,
        "cba01r_score",
        split_name,
    )
    p04_frame = build_single_expert_score_frame(
        split_df,
        y,
        p04_score,
        "p04_ae_score",
        split_name,
    )
    merged = pd.DataFrame(
        {
            ID_COL: split_df[ID_COL].to_numpy(),
            TARGET_COL: y.to_numpy(),
        }
    ).merge(cba_frame, on=ID_COL, how="inner", validate="one_to_one")
    merged = merged.merge(
        p04_frame,
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(split_df):
        raise ValueError(f"{split_name}: TransactionID join row count mismatch.")
    if merged[ID_COL].tolist() != split_df[ID_COL].tolist():
        raise ValueError(f"{split_name}: merged TransactionID order differs from split.")
    if not np.allclose(merged["cba01r_score"], cba_score):
        raise ValueError(f"{split_name}: CBA01R scores changed after join.")
    if not np.allclose(merged["p04_ae_score"], p04_score):
        raise ValueError(f"{split_name}: P04 scores changed after join.")
    if merged.isna().any().any():
        raise ValueError(f"{split_name}: missing values in aligned score table.")
    return merged


def fusion_score(
    behavioral_weight: float,
    cba_score: np.ndarray,
    ae_score: np.ndarray,
) -> np.ndarray:
    ae_weight = 1.0 - behavioral_weight
    return behavioral_weight * cba_score + ae_weight * ae_score


def build_weight_search_table(
    y_valid: np.ndarray,
    cba_valid_score: np.ndarray,
    p04_valid_score: np.ndarray,
    cba01r_val_ap: float,
    p04_val_ap: float,
) -> pd.DataFrame:
    rows = []
    for behavioral_weight in BEHAVIORAL_WEIGHT_GRID:
        ae_weight = 1.0 - behavioral_weight
        fused = fusion_score(behavioral_weight, cba_valid_score, p04_valid_score)
        val_ap = float(average_precision_score(y_valid, fused))
        val_roc = float(roc_auc_score(y_valid, fused))
        rows.append(
            {
                "behavioral_weight": behavioral_weight,
                "ae_weight": ae_weight,
                "validation_average_precision": val_ap,
                "validation_delta_vs_cba01r": val_ap - cba01r_val_ap,
                "validation_delta_vs_p04": val_ap - p04_val_ap,
                "validation_roc_auc": val_roc,
            }
        )

    table = pd.DataFrame(rows)
    max_ap = table["validation_average_precision"].max()
    tied = table.loc[
        np.isclose(
            table["validation_average_precision"],
            max_ap,
            atol=AP_TIE_TOLERANCE,
            rtol=0.0,
        )
    ]
    best_index = tied.sort_values(
        ["validation_average_precision", "behavioral_weight"],
        ascending=[False, False],
    ).index[0]
    table["selected"] = False
    table.loc[best_index, "selected"] = True
    tie_break_applied = len(tied) > 1
    table["tie_break_applied"] = tie_break_applied
    table["selection_reason"] = ""
    table.loc[best_index, "selection_reason"] = (
        "highest validation_average_precision"
        + ("; tie-break prefers larger behavioral_weight" if tie_break_applied else "")
    )
    return table


def selected_weights_from_table(weight_table: pd.DataFrame) -> tuple[float, float]:
    selected = weight_table.loc[weight_table["selected"]]
    if selected.empty:
        raise ValueError("No selected fusion weight found.")
    behavioral_weight = float(selected.iloc[0]["behavioral_weight"])
    ae_weight = float(selected.iloc[0]["ae_weight"])
    return behavioral_weight, ae_weight


def classify_practical_result(
    ae_weight: float,
    fusion_val_ap: float,
    cba01r_val_ap: float,
    p02_val_ap: float,
) -> str:
    if ae_weight == 0.0 or fusion_val_ap <= cba01r_val_ap:
        return "no contribution"
    delta = fusion_val_ap - cba01r_val_ap
    if delta >= PRACTICAL_IMPROVEMENT_THRESHOLD and fusion_val_ap > p02_val_ap:
        return "strong success"
    if delta >= PRACTICAL_IMPROVEMENT_THRESHOLD and fusion_val_ap <= p02_val_ap:
        return "partial success"
    if delta > 0.0:
        return "marginal signal"
    return "no contribution"


def topk_metrics_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.03, 0.05),
) -> pd.DataFrame:
    n = len(y_true)
    total_fraud = int(y_true.sum())
    order = np.argsort(-y_score)
    rows = []
    for fraction in fractions:
        reviewed = max(1, int(np.ceil(n * fraction)))
        top_idx = order[:reviewed]
        fraud_captured = int(y_true[top_idx].sum())
        rows.append(
            {
                "top_fraction": fraction,
                "reviewed_transactions": reviewed,
                "fraud_captured": fraud_captured,
                "precision_at_top": float(fraud_captured / reviewed),
                "recall_at_top": float(fraud_captured / total_fraud)
                if total_fraud
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_ap_delta(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    attempts = 0
    max_attempts = n_resamples * 20
    while len(deltas) < n_resamples and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        y_sample = y_true[idx]
        if np.unique(y_sample).size < 2:
            continue
        ap_a = average_precision_score(y_sample, score_a[idx])
        ap_b = average_precision_score(y_sample, score_b[idx])
        deltas.append(float(ap_b - ap_a))

    if len(deltas) < n_resamples:
        raise RuntimeError(
            f"Paired bootstrap produced only {len(deltas)} valid resamples "
            f"(requested {n_resamples})."
        )

    delta_array = np.asarray(deltas, dtype=float)
    return {
        "mean_delta": float(delta_array.mean()),
        "median_delta": float(np.median(delta_array)),
        "ci_lower_2_5": float(np.percentile(delta_array, 2.5)),
        "ci_upper_97_5": float(np.percentile(delta_array, 97.5)),
        "proportion_delta_gt_zero": float(np.mean(delta_array > 0.0)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def score_alignment_summary(
    table: pd.DataFrame,
    expected_checksum: str | None,
    split_name: str,
) -> dict[str, object]:
    checksum = transaction_id_checksum(table[ID_COL])
    return {
        "split": split_name,
        "row_count": int(len(table)),
        "transaction_id_checksum": checksum,
        "expected_checksum": expected_checksum,
        "checksum_matches_expected": (
            checksum == expected_checksum if expected_checksum else None
        ),
        "duplicate_transaction_ids": int(table[ID_COL].duplicated().sum()),
        "missing_transaction_ids": False,
        "finite_probabilities": bool(
            np.isfinite(table["cba01r_score"]).all()
            and np.isfinite(table["p04_ae_score"]).all()
        ),
        "probability_range_valid": bool(
            table["cba01r_score"].between(0.0, 1.0).all()
            and table["p04_ae_score"].between(0.0, 1.0).all()
        ),
    }