"""Phase 12 pre-run gates for LF01 late fusion."""

from __future__ import annotations

import sys

from audit_causal_behavioral_ae_complementarity import run_complementarity_audit
from late_fusion_experts import (
    BEHAVIORAL_WEIGHT_GRID,
    align_expert_scores_by_transaction_id,
    regenerate_cba01r_scores,
    regenerate_p04_scores,
)
from run_causal_behavioral_ae_late_fusion import print_prerun_gates


def main() -> None:
    cba = regenerate_cba01r_scores()
    p04 = regenerate_p04_scores(cba["prepared"])

    valid_table = align_expert_scores_by_transaction_id(
        cba["valid_df"],
        cba["y_valid"],
        cba["valid_score"],
        p04["valid_score"],
        "validation",
    )
    test_table = align_expert_scores_by_transaction_id(
        cba["test_df"],
        cba["y_test"],
        cba["test_score"],
        p04["test_score"],
        "test",
    )
    complementarity = run_complementarity_audit(valid_table)

    if BEHAVIORAL_WEIGHT_GRID != [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
    ]:
        print("FAIL: weight grid differs from predefined grid.")
        sys.exit(1)

    print_prerun_gates(cba, p04, valid_table, test_table, complementarity)
    print("PRE-RUN GATES: PASSED")


if __name__ == "__main__":
    main()