from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INITIAL_PROPOSAL = REPO_ROOT / "outputs" / "initial_proposal"
AE05_DIR = INITIAL_PROPOSAL / "ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned"
P02_DIR = INITIAL_PROPOSAL / "optuna" / "baseline_lgbm_tuned"
EXTENDED_CSV = INITIAL_PROPOSAL / "final_comparison" / "extended_proposal_comparison.csv"
BOOTSTRAP_CSV = (
    INITIAL_PROPOSAL
    / "representation_ablation"
    / "bootstrap_ae05_vs_p02"
    / "paired_bootstrap_pr_auc_delta.csv"
)


def _load_test_ap(metrics_path: Path) -> float:
    with metrics_path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return float(payload["average_precision"])


def test_ae05_beats_p02_on_test_pr_auc() -> None:
    ae05_ap = _load_test_ap(AE05_DIR / "metrics_test_selected_threshold.json")
    p02_ap = _load_test_ap(P02_DIR / "metrics_test_selected_threshold.json")
    assert ae05_ap > p02_ap


def test_ae05_required_artifacts_exist() -> None:
    required = [
        AE05_DIR / "metrics_test_selected_threshold.json",
        AE05_DIR / "metrics_validation_selected_threshold.json",
        AE05_DIR / "run_config.json",
        AE05_DIR / "feature_importance.csv",
        AE05_DIR / "scores_test.csv",
        AE05_DIR / "scores_validation.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Missing AE-05 artifacts: {missing}"


def test_extended_comparison_includes_ae05_ranked_first() -> None:
    assert EXTENDED_CSV.is_file(), "Run build_extended_proposal_comparison.py first."
    table = pd.read_csv(EXTENDED_CSV)
    assert "AE-05" in table["canonical_id"].values
    top = table.sort_values("test_average_precision", ascending=False).iloc[0]
    assert top["canonical_id"] == "AE-05"


def test_bootstrap_delta_csv_positive() -> None:
    if not BOOTSTRAP_CSV.is_file():
        return
    summary = pd.read_csv(BOOTSTRAP_CSV)
    delta = float(summary.iloc[0]["delta"])
    assert delta > 0.0