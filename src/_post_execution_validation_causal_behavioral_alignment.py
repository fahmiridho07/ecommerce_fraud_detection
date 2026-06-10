"""Phase 12 post-run validation for causal behavioral alignment correction."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from build_causal_behavioral_alignment_correction_comparison import main as build_comparison
from config import (
    CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
)


REQUIRED_ARTIFACTS = [
    "run_config.json",
    "metrics_validation_selected_threshold.json",
    "metrics_test_selected_threshold.json",
    "alignment_validation.json",
    "model.pkl",
    "preprocessing.pkl",
]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def assert_output_complete(output_dir: Path, label: str) -> None:
    for artifact in REQUIRED_ARTIFACTS:
        path = output_dir / artifact
        if not path.exists():
            raise FileNotFoundError(f"{label} missing artifact: {path}")


def main() -> None:
    print("=" * 72)
    print("PHASE 12 — CAUSAL BEHAVIORAL ALIGNMENT POST-RUN VALIDATION")
    print("=" * 72)

    assert_output_complete(CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR, "CBA01R")
    assert_output_complete(
        CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR,
        "CBA02R",
    )
    for legacy_dir in (
        CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR,
        CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR,
    ):
        if not legacy_dir.exists():
            raise FileNotFoundError(f"Legacy output preserved check failed: {legacy_dir}")

    b2_run = load_json(CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR / "run_config.json")
    b3_run = load_json(
        CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR
        / "run_config.json"
    )
    if int(b2_run["final_feature_count"]) != 451:
        raise ValueError(f"CBA01R feature count expected 451, found {b2_run['final_feature_count']}")
    if int(b3_run["final_feature_count"]) != 452:
        raise ValueError(f"CBA02R feature count expected 452, found {b3_run['final_feature_count']}")
    if b2_run.get("positional_join_used"):
        raise ValueError("CBA01R run_config still reports positional_join_used=true")
    if b3_run.get("autoencoder_retrained"):
        raise ValueError("CBA02R reports autoencoder_retrained=true")
    if not b3_run.get("test_not_used_for_model_selection", True):
        raise ValueError("CBA02R test_not_used_for_model_selection is false")

    comparison = build_comparison()
    if not CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION_FILE.exists():
        raise FileNotFoundError("Comparison CSV was not created.")

    cba01r_val = float(
        load_json(
            CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )
    cba02r_val = float(
        load_json(
            CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_ID_ALIGNED_OUTPUT_DIR
            / "metrics_validation_selected_threshold.json"
        )["average_precision"]
    )
    row_cba01r = comparison.loc[comparison["model_id"] == "CBA01R"].iloc[0]
    if abs(float(row_cba01r["validation_average_precision"]) - cba01r_val) > 1e-9:
        raise ValueError("Comparison CSV CBA01R validation AP mismatch.")

    print("Running alignment tests...")
    tests_path = Path(__file__).resolve().parents[1] / "tests" / "test_causal_behavioral_alignment.py"
    subprocess.run([sys.executable, str(tests_path)], check=True)

    print("Running causal behavioral fixture checks...")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "causal_behavioral_features.py")],
        check=True,
    )

    print("Running syntax/import checks...")
    for module in [
        "audit_causal_behavioral_row_alignment",
        "train_causal_behavioral_lgbm",
        "train_causal_behavioral_cdv_reconstruction_lgbm",
        "regenerate_cdv_reconstruction_errors_id_aligned",
        "build_causal_behavioral_alignment_correction_comparison",
    ]:
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(Path(__file__).parent / f"{module}.py")],
            check=True,
        )

    print("Checking git status/diff...")
    subprocess.run(["git", "status", "--short"], cwd=Path(__file__).resolve().parents[1])
    subprocess.run(["git", "diff", "--stat"], cwd=Path(__file__).resolve().parents[1])

    print("\n" + "=" * 72)
    print("POST-RUN VALIDATION: ALL CHECKS PASSED")
    print(f"CBA01R validation AP: {cba01r_val:.6f}")
    print(f"CBA02R validation AP: {cba02r_val:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPOST-RUN VALIDATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)