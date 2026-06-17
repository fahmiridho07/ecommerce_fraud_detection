# Experiment Registry

Status: active stratified registry after repository cleanup.

This file tracks only thesis-facing experiments under the active protocol:

```text
split_strategy=stratified_holdout
train/validation/test = 60/20/20
random_state = 42
primary metric = Average Precision / PR-AUC
```

Historical chronological evidence was moved to:

```text
archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md
```

Do not mix historical chronological numbers with new stratified result tables.

## Active Stratified Results

Only the split validation has been completed in the canonical active path.

| ID | Experiment | Script | Output | Status | Test AP |
|----|------------|--------|--------|--------|--------:|
| S0 | Split validation | `src/check_data_split.py` | `outputs/split_summary.json` | Complete | N/A |
| A0 | Original-feature LightGBM default | `src/train_baseline_lgbm.py` | `outputs/stratified_reset/baseline_lgbm_default/` | Pending | N/A |
| A0-T | Original-feature LightGBM tuned | `src/tune_lgbm_optuna.py --model_type baseline_lgbm` | `outputs/stratified_reset/optuna/baseline_lgbm_tuned/` | Pending | N/A |
| A1 | Alharbi-style preprocessing LightGBM default | `src/train_paper_preprocessing_lgbm.py` | `outputs/stratified_reset/alharbi_style_lgbm_default/` | Pending | N/A |
| A1-T | Alharbi-style preprocessing LightGBM tuned | `src/tune_lgbm_optuna.py --model_type alharbi_lgbm` | `outputs/stratified_reset/optuna/alharbi_lgbm_tuned/` | Pending | N/A |
| A1-AE | AE-LightGBM matched to strongest A1 baseline | TBD after A1-T | `outputs/stratified_reset/` | Blocked until A1-T | N/A |
| A1-E | AE feature/score integration | TBD after A1-AE | `outputs/stratified_reset/` | Blocked until A1-AE | N/A |

## S0 Split Validation

Observed on the full local IEEE-CIS training data:

| Split | Rows | Fraud count | Fraud rate |
|-------|-----:|------------:|-----------:|
| Train | 354,324 | 12,398 | 3.4991% |
| Validation | 118,108 | 4,132 | 3.4985% |
| Test | 118,108 | 4,133 | 3.4993% |

The stratified split preserves class ratio across all three sets. Temporal order
is not preserved by design; chronological evaluation is now a limitation and
future-work discussion, not the active S1 protocol.

## Staging Results Not Yet Canonical

There are older stratified staging artifacts under:

```text
outputs/initial_proposal/split_strategy_current/stratified_holdout/
```

They are useful diagnostics but are not the canonical post-cleanup registry
because they were produced before the current `outputs/stratified_reset/`
boundary and A1 paper-anchored preprocessing cleanup.

Observed staging snapshot:

| Model | Test AP | Test ROC-AUC | Test F1 | Test MCC |
|-------|--------:|-------------:|--------:|---------:|
| Baseline frequency/missingness/time/amount | 0.855734 | 0.968481 | 0.815113 | 0.814175 |
| AE latent LD32 add-on | 0.835256 | 0.964357 | 0.788943 | 0.790996 |
| Fixed 0.50 score ensemble | 0.850940 | 0.968245 | 0.804722 | 0.804903 |
| Alpha-tuned score ensemble | 0.855618 | 0.968595 | 0.813573 | 0.812696 |

Interpretation: under this staging stratified split, the tabular preprocessing
baseline is very strong and AE does not yet beat it on PR-AUC. Rerun the active
A0/A1 ladder before using this in thesis prose.

## Decision Rules

- Promote A1 only if it improves or clarifies the strongest A0 baseline under
  the same stratified split.
- Promote any AE branch only if it beats the strongest matched A1 baseline on
  test AP and the paired-bootstrap AP delta supports a positive improvement.
- If AE does not beat the strongest baseline, document that result honestly.

## Historical Evidence

Historical chronological results are still traceable:

| Evidence | Location |
|----------|----------|
| P01-P04 proposal block | `archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md` |
| AE-05 hybrid candidate | `archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md` |
| Best old score ensemble | `archive/docs/chronological_evidence/FINAL_CANDIDATE_VALIDATION.md` |
| Old preprocessing diagnostics | `archive/docs/chronological_evidence/PREPROCESSING_DIAGNOSTIC.md` |
