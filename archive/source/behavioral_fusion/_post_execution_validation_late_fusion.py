"""Phase 13 post-run validation for LF01 late fusion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import (
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE,
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR,
    CAUSAL_BEHAVIORAL_AE_LATE_FUSION_WEIGHT_SEARCH_FILE,
    CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR,
    OPTUNA_OUTPUT_DIR,
    PROJECT_ROOT,
)
from late_fusion_experts import (
    BEHAVIORAL_WEIGHT_GRID,
    fusion_score,
    selected_weights_from_table,
)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def main() -> None:
    output_dir = CAUSAL_BEHAVIORAL_AE_LATE_FUSION_OUTPUT_DIR
    frozen_path = output_dir / "frozen_fusion_config.json"
    if not frozen_path.exists():
        fail("frozen_fusion_config.json missing")

    frozen = load_json(frozen_path)
    weight_table = pd.read_csv(CAUSAL_BEHAVIORAL_AE_LATE_FUSION_WEIGHT_SEARCH_FILE)
    behavioral_weight, ae_weight = selected_weights_from_table(weight_table)

    max_ap = weight_table["validation_average_precision"].max()
    tied = weight_table.loc[
        np.isclose(
            weight_table["validation_average_precision"],
            max_ap,
            atol=1e-8,
            rtol=0.0,
        )
    ]
    best = tied.sort_values(
        ["validation_average_precision", "behavioral_weight"],
        ascending=[False, False],
    ).iloc[0]
    if not bool(best["selected"]):
        fail("selected weight is not maximum validation AP under tie-break")

    if float(frozen["behavioral_weight"]) != behavioral_weight:
        fail("frozen behavioral_weight does not match weight search selection")

    if (output_dir / "metrics_test_selected_threshold.json").stat().st_mtime < frozen_path.stat().st_mtime:
        fail("test metrics written before frozen config")

    forbidden = output_dir / "test_weight_search.csv"
    if forbidden.exists():
        fail("test-weight-search artifact exists")

    valid_table = pd.read_csv(output_dir / "validation_expert_scores.csv")
    recomputed = fusion_score(
        behavioral_weight,
        valid_table["cba01r_score"].to_numpy(),
        valid_table["p04_ae_score"].to_numpy(),
    )
    frozen_ap = float(frozen["validation_average_precision"])
    if abs(average_precision_score(valid_table["isFraud"], recomputed) - frozen_ap) > 1e-8:
        fail("fusion scores do not match documented convex formula")

    comparison = pd.read_csv(CAUSAL_BEHAVIORAL_AE_LATE_FUSION_COMPARISON_FILE)
    lf01 = comparison.loc[comparison["model_id"] == "LF01"].iloc[0]
    test_metrics = load_json(output_dir / "metrics_test_selected_threshold.json")
    if abs(float(lf01["test_average_precision"]) - float(test_metrics["average_precision"])) > 1e-8:
        fail("comparison CSV test AP does not match LF01 metric artifact")

    cba_mtime = (CAUSAL_BEHAVIORAL_LGBM_ID_ALIGNED_OUTPUT_DIR / "model.pkl").stat().st_mtime
    p04_mtime = (OPTUNA_OUTPUT_DIR / "ae_lgbm_ld128" / "final_model.pkl").stat().st_mtime
    fusion_mtime = frozen_path.stat().st_mtime
    if cba_mtime > fusion_mtime or p04_mtime > fusion_mtime:
        print("WARNING: expert model artifacts newer than frozen config; verify no retraining.")

    checks = [
        "src/late_fusion_experts.py",
        "src/audit_causal_behavioral_ae_complementarity.py",
        "src/run_causal_behavioral_ae_late_fusion.py",
        "src/build_causal_behavioral_ae_late_fusion_comparison.py",
    ]
    for script in checks:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(PROJECT_ROOT / script)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            fail(f"syntax check failed for {script}: {result.stderr}")

    if list(weight_table["behavioral_weight"]) != BEHAVIORAL_WEIGHT_GRID:
        fail("weight grid in artifact differs from predefined grid")

    print("POST-RUN VALIDATION: PASSED")
    print(f"Selected behavioral_weight={behavioral_weight} ae_weight={ae_weight}")
    print(f"Practical category={frozen['practical_result_category']}")


if __name__ == "__main__":
    main()