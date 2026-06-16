"""Build comparison table for the GBDT baseline shootout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from utils import ensure_dir, save_json


GBDT_BASE_OUTPUT_DIR = OUTPUT_DIR / "gbdt_baseline_comparison"
COMPARISON_FILENAME = "comparison.csv"
DECISION_GATE_FILENAME = "decision_gate.json"
MISSING_ARTIFACTS_FILENAME = "missing_artifacts.json"
EXPECTED_EXPERIMENT_FAMILY = "gbdt_baseline_comparison"

TUNE_B0_REFERENCE = {
    "strategy_id": "TUNE-B0",
    "validation_average_precision": 0.6378,
    "test_average_precision": 0.5060,
    "source": "docs/AE_STRATEGY_TUNING_RESULTS.md",
}
TUNE_AE3_REFERENCE = {
    "strategy_id": "TUNE-AE3",
    "validation_average_precision": 0.6290,
    "test_average_precision": 0.4994,
    "source": "docs/AE_STRATEGY_TUNING_RESULTS.md",
}
DECISION_GATE_MIN_VAL_AP_DELTA = 0.003

COMPARISON_COLUMNS = [
    "experiment_id",
    "backend",
    "feature_set",
    "tuned",
    "preprocessing_mode",
    "validation_average_precision",
    "test_average_precision",
    "validation_roc_auc",
    "test_roc_auc",
    "selected_threshold",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mcc",
    "best_iteration",
    "n_trials",
    "total_features",
    "reconstruction_error_used",
    "output_dir",
    "run_config_path",
    "metrics_path",
]

PHASE1_CANDIDATES = [
    {
        "experiment_id": "GBDT-LGBM-FIX",
        "backend": "lightgbm",
        "feature_set": "raw",
        "tuned": False,
        "output_subdir": "LGBM_fixed",
    },
    {
        "experiment_id": "GBDT-XGB-FIX",
        "backend": "xgboost",
        "feature_set": "raw",
        "tuned": False,
        "output_subdir": "XGB_fixed",
    },
    {
        "experiment_id": "GBDT-CAT-FIX",
        "backend": "catboost",
        "feature_set": "raw",
        "tuned": False,
        "output_subdir": "CAT_fixed",
    },
    {
        "experiment_id": "GBDT-LGBM-TUNE",
        "backend": "lightgbm",
        "feature_set": "raw",
        "tuned": True,
        "output_subdir": "optuna/LGBM_tuned",
    },
    {
        "experiment_id": "GBDT-XGB-TUNE",
        "backend": "xgboost",
        "feature_set": "raw",
        "tuned": True,
        "output_subdir": "optuna/XGB_tuned",
    },
    {
        "experiment_id": "GBDT-CAT-TUNE",
        "backend": "catboost",
        "feature_set": "raw",
        "tuned": True,
        "output_subdir": "optuna/CAT_tuned",
    },
]

PHASE2_CANDIDATES = [
    {
        "experiment_id": "GBDT-WIN-AE3-FIX",
        "backend": "winner",
        "feature_set": "ae3",
        "tuned": False,
        "output_subdir_template": "AE3_fixed/{backend}",
    },
    {
        "experiment_id": "GBDT-WIN-AE3-TUNE",
        "backend": "winner",
        "feature_set": "ae3",
        "tuned": True,
        "output_subdir_template": "optuna/AE3_tuned/{backend}",
    },
]

EXPECTED_PHASE1_IDS = tuple(candidate["experiment_id"] for candidate in PHASE1_CANDIDATES)
OUT_OF_SCOPE_MODEL_NAMES = {
    "baseline_lgbm_default",
    "baseline_lgbm_tuned",
    "ae3_reconstruction_error_lgbm_ld128_tuned",
    "du_latent_replacement",
    "ding_reconstructed_replacement",
    "reconstruction_error_augmentation",
    "causal_behavioral_lgbm_id_aligned",
}


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def metric_value(metrics: dict[str, object], key: str) -> object:
    return metrics.get(key)


def total_features_from_run_config(run_config: dict[str, object]) -> object:
    if "model_features_count" in run_config:
        return run_config["model_features_count"]
    feature_set_summary = run_config.get("feature_set_summary", {})
    if isinstance(feature_set_summary, dict):
        for key in ("total_final_features", "total_feature_count"):
            if key in feature_set_summary:
                return feature_set_summary[key]
    return None


def best_iteration_from_run_config(run_config: dict[str, object]) -> object:
    early_stopping = run_config.get("early_stopping", {})
    if isinstance(early_stopping, dict):
        return early_stopping.get("best_iteration")
    return None


def n_trials_from_run_config(run_config: dict[str, object], tuned: bool) -> object:
    if not tuned:
        return 0
    optuna_config = run_config.get("optuna", {})
    if isinstance(optuna_config, dict):
        return optuna_config.get("n_trials_completed")
    return None


def validate_run_config(
    run_config: dict[str, object],
    candidate: dict[str, object],
    run_config_path: Path,
) -> str | None:
    if run_config.get("final_training_completed") is False:
        return f"{run_config_path} (final_training_completed=false)"

    if run_config.get("experiment_family") != EXPECTED_EXPERIMENT_FAMILY:
        return (
            "run_config family mismatch: "
            f"expected {EXPECTED_EXPERIMENT_FAMILY!r}, "
            f"found {run_config.get('experiment_family')!r} ({run_config_path})"
        )

    expected_id = candidate["experiment_id"]
    actual_id = run_config.get("experiment_id")
    if actual_id != expected_id:
        return (
            "run_config experiment_id mismatch: "
            f"expected {expected_id!r}, found {actual_id!r} ({run_config_path})"
        )
    return None


def required_artifact_paths(output_dir: Path, tuned: bool) -> list[Path]:
    metrics_name = (
        "metrics_test_selected_threshold.json"
        if tuned or True
        else "metrics_test_selected_threshold.json"
    )
    return [
        output_dir / "metrics_validation_selected_threshold.json",
        output_dir / metrics_name,
        output_dir / "run_config.json",
    ]


def comparison_row(
    candidate: dict[str, object],
    output_dir: Path,
) -> tuple[dict[str, object] | None, list[str]]:
    required_paths = required_artifact_paths(output_dir, candidate["tuned"])
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        return None, missing_paths

    run_config_path = output_dir / "run_config.json"
    valid_metrics_path = output_dir / "metrics_validation_selected_threshold.json"
    test_metrics_path = output_dir / "metrics_test_selected_threshold.json"

    run_config = load_json(run_config_path)
    mismatch_message = validate_run_config(run_config, candidate, run_config_path)
    if mismatch_message:
        return None, [mismatch_message]

    valid_metrics = load_json(valid_metrics_path)
    test_metrics = load_json(test_metrics_path)
    feature_set_summary = run_config.get("feature_set_summary", {})
    if not isinstance(feature_set_summary, dict):
        feature_set_summary = {}

    return (
        {
            "experiment_id": candidate["experiment_id"],
            "backend": run_config.get("backend", candidate.get("backend")),
            "feature_set": run_config.get("feature_set", candidate["feature_set"]),
            "tuned": candidate["tuned"],
            "preprocessing_mode": (
                feature_set_summary.get("preprocessing_mode")
                or run_config.get("preprocessing", {}).get("mode")
            ),
            "validation_average_precision": metric_value(
                valid_metrics,
                "average_precision",
            ),
            "test_average_precision": metric_value(test_metrics, "average_precision"),
            "validation_roc_auc": metric_value(valid_metrics, "roc_auc"),
            "test_roc_auc": metric_value(test_metrics, "roc_auc"),
            "selected_threshold": metric_value(test_metrics, "threshold"),
            "test_precision": metric_value(test_metrics, "precision"),
            "test_recall": metric_value(test_metrics, "recall"),
            "test_f1": metric_value(test_metrics, "f1"),
            "test_mcc": metric_value(test_metrics, "mcc"),
            "best_iteration": best_iteration_from_run_config(run_config),
            "n_trials": n_trials_from_run_config(run_config, candidate["tuned"]),
            "total_features": total_features_from_run_config(run_config),
            "reconstruction_error_used": feature_set_summary.get(
                "reconstruction_error_used",
                candidate["feature_set"] == "ae3",
            ),
            "output_dir": str(output_dir),
            "run_config_path": str(run_config_path),
            "metrics_path": str(test_metrics_path),
        },
        [],
    )


def resolve_output_dirs(
    base_output_dir: Path,
    winner_backend: str | None = None,
) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for candidate in PHASE1_CANDIDATES:
        mapping[candidate["experiment_id"]] = (
            base_output_dir / candidate["output_subdir"]
        )

    if winner_backend:
        for candidate in PHASE2_CANDIDATES:
            subdir = candidate["output_subdir_template"].format(backend=winner_backend)
            mapping[candidate["experiment_id"]] = base_output_dir / subdir
    return mapping


def build_gbdt_baseline_comparison_table(
    output_dirs: dict[str, Path],
    include_phase2: bool = False,
    winner_backend: str | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    candidates = list(PHASE1_CANDIDATES)
    if include_phase2 and winner_backend:
        candidates.extend(PHASE2_CANDIDATES)

    rows: list[dict[str, object]] = []
    missing_artifacts: dict[str, list[str]] = {}

    for candidate in candidates:
        output_dir = output_dirs.get(candidate["experiment_id"])
        if output_dir is None:
            missing_artifacts[candidate["experiment_id"]] = ["output_dir_not_configured"]
            continue

        row, missing = comparison_row(candidate, output_dir)
        if row is not None:
            rows.append(row)
        else:
            missing_artifacts[candidate["experiment_id"]] = missing

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    return table, missing_artifacts


def select_tuned_winner(table: pd.DataFrame) -> pd.Series | None:
    tuned_raw = table.loc[
        (table["tuned"] == True) & (table["feature_set"] == "raw")  # noqa: E712
    ].copy()
    if tuned_raw.empty:
        return None
    tuned_raw = tuned_raw.dropna(subset=["validation_average_precision"])
    if tuned_raw.empty:
        return None
    return tuned_raw.sort_values(
        "validation_average_precision",
        ascending=False,
    ).iloc[0]


def build_decision_gate_summary(table: pd.DataFrame) -> dict[str, object]:
    winner_row = select_tuned_winner(table)
    if winner_row is None:
        return {
            "gate_passed": False,
            "reason": "No completed tuned raw GBDT runs found.",
            "reference": TUNE_B0_REFERENCE,
            "min_validation_delta": DECISION_GATE_MIN_VAL_AP_DELTA,
        }

    winner_val_ap = float(winner_row["validation_average_precision"])
    reference_val_ap = float(TUNE_B0_REFERENCE["validation_average_precision"])
    delta = winner_val_ap - reference_val_ap
    gate_passed = delta >= DECISION_GATE_MIN_VAL_AP_DELTA

    return {
        "gate_passed": gate_passed,
        "winner_experiment_id": winner_row["experiment_id"],
        "winner_backend": winner_row["backend"],
        "winner_validation_average_precision": winner_val_ap,
        "winner_test_average_precision": winner_row["test_average_precision"],
        "reference": TUNE_B0_REFERENCE,
        "validation_delta_vs_tune_b0": delta,
        "min_validation_delta": DECISION_GATE_MIN_VAL_AP_DELTA,
        "phase2_recommended": gate_passed,
        "tune_ae3_reference": TUNE_AE3_REFERENCE,
    }


def main(
    base_output_dir: Path = GBDT_BASE_OUTPUT_DIR,
    winner_backend: str | None = None,
    include_phase2: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dir(base_output_dir)
    output_dirs = resolve_output_dirs(base_output_dir, winner_backend=winner_backend)

    if include_phase2 and not winner_backend:
        gate_preview = build_decision_gate_summary(
            build_gbdt_baseline_comparison_table(output_dirs)[0]
        )
        winner_backend = gate_preview.get("winner_backend")
        if winner_backend:
            output_dirs = resolve_output_dirs(
                base_output_dir,
                winner_backend=str(winner_backend),
            )

    table, missing_artifacts = build_gbdt_baseline_comparison_table(
        output_dirs,
        include_phase2=include_phase2 or winner_backend is not None,
        winner_backend=winner_backend,
    )
    decision_gate = build_decision_gate_summary(table)

    table.to_csv(base_output_dir / COMPARISON_FILENAME, index=False)
    save_json(missing_artifacts, base_output_dir / MISSING_ARTIFACTS_FILENAME)
    save_json(decision_gate, base_output_dir / DECISION_GATE_FILENAME)

    print()
    print("GBDT Baseline Comparison")
    print("========================")
    if table.empty:
        print("No completed runs found yet.")
    else:
        ranked = table.sort_values(
            "validation_average_precision",
            ascending=False,
            na_position="last",
        )
        for _, row in ranked.iterrows():
            print(
                f"{row['experiment_id']}: "
                f"val AP={row['validation_average_precision']}, "
                f"test AP={row['test_average_precision']}"
            )

    print()
    print("Decision Gate vs TUNE-B0")
    print("========================")
    print(f"Gate passed          : {decision_gate['gate_passed']}")
    if decision_gate.get("winner_experiment_id"):
        print(f"Winner               : {decision_gate['winner_experiment_id']}")
        print(
            "Validation delta     : "
            f"{decision_gate['validation_delta_vs_tune_b0']:+.6f}"
        )
    print(f"Phase 2 recommended  : {decision_gate.get('phase2_recommended', False)}")
    print(f"Saved comparison to  : {base_output_dir / COMPARISON_FILENAME}")
    return table, decision_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GBDT baseline comparison table and decision gate summary."
    )
    parser.add_argument("--base-output-dir", type=Path, default=GBDT_BASE_OUTPUT_DIR)
    parser.add_argument(
        "--winner-backend",
        choices=("lightgbm", "xgboost", "catboost"),
        default=None,
        help="Backend winner for Phase 2 AE3 rows.",
    )
    parser.add_argument(
        "--include-phase2",
        action="store_true",
        help="Include AE3 fixed/tuned rows when winner backend is known.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        base_output_dir=args.base_output_dir,
        winner_backend=args.winner_backend,
        include_phase2=args.include_phase2,
    )