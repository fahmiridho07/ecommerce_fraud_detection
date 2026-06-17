# Thesis Scope

Status: active cleanup reset, 2026-06-17.

This document is the source of truth for the cleaned repository after the split
protocol reset. The active thesis protocol is now **stratified holdout**. Older
chronological experiments remain documented as historical evidence only.

For a compact agent-facing orientation, read `docs/AI_AGENT_BRIEF.md` after this
file.

## Active Protocol

The thesis studies fraud detection on the IEEE-CIS Fraud Detection dataset using:

- LightGBM as the supervised tabular baseline.
- Paper-anchored preprocessing branches, starting from Alharbi-style IEEE-CIS
  preprocessing.
- Autoencoder representation learning on anonymized numerical `V*` features.
- AE-LightGBM variants only after the stratified baseline is rerun.
- Average Precision / PR-AUC as the primary metric, with ROC-AUC, F1, MCC, and
  confusion matrices as supporting metrics.

The active data split is:

```text
stratified_holdout, 60% train / 20% validation / 20% test, random_state=42
```

Guardrails:

- Fit all imputers, encoders, scalers, frequency maps, AE preprocessors, and
  sampling objects on the training split only.
- Use validation only for early stopping, Optuna objective, and threshold
  selection.
- Use test only for final evaluation after model/threshold selection.
- Do not balance validation or test.
- Treat chronological/time-aware evaluation, concept drift, rolling windows, and
  deployment realism as limitation/future-work discussion for the S1 thesis.

## Current Active Result

There is **no active thesis-facing winner yet** after this reset.

All metrics produced before the stratified reset are historical and must not be
compared directly with future stratified reruns. The next valid thesis result
must come from a clean stratified rerun.

## Historical Archive

The chronological proposal and post-diagnostic results stay traceable because
they explain how the project arrived here:

| Block | Status | Notes |
|-------|--------|-------|
| P01-P04 proposal block | Archived historical chronological results | P02 was best inside this block. |
| AE-05 hybrid branch | Archived post-diagnostic chronological candidate | First AE branch that beat P02 under the old split. |
| Fixed 0.50 score ensemble | Archived post-diagnostic chronological candidate | Strongest old score-level evidence, not active after reset. |
| `frequency_missingness_time_amount` preprocessing | Archived empirical diagnostic branch | Useful evidence, but not the final paper-anchored protocol. |

Historical metrics can be cited only with the phrase "under the previous
chronological protocol" and should not be mixed into new stratified tables.

Detailed historical evidence lives in:

```text
archive/docs/chronological_evidence/
```

## Active Rerun Ladder

Run from the narrowest defensible branch outward:

1. `S0`: validate stratified split summary.
2. `A0`: baseline LightGBM on original features under stratified holdout.
3. `A0-T`: tuned baseline LightGBM under stratified holdout.
4. `A1`: Alharbi-style preprocessing baseline under stratified holdout.
5. `A1-T`: tuned A1 baseline under stratified holdout.
6. `A1-AE`: AE-LightGBM branch using the same split and train-only fitted
   preprocessing.
7. `A1-E`: score-level or feature-level AE integration only if it beats the
   strongest A1 baseline under the same stratified test split.

Decision rule:

- Promote a preprocessing branch only if it improves or clarifies the strongest
  baseline under the same stratified test split.
- Promote an AE branch only if it beats the strongest stratified baseline on
  test AP and the paired-bootstrap confidence interval supports a positive
  delta.
- Otherwise, conclude that the paper-anchored LightGBM baseline is stronger
  than the tested AE integration.

## Out Of Scope

These remain outside active thesis claims unless a new written decision gate is
created:

- Chronological or time-aware deployment evaluation as the main experiment.
- SMOTE/ADASYN as a mainline method instead of appendix/robustness branch.
- Target encoding without strict out-of-fold leakage controls.
- Broad UID, velocity, rolling-window, behavioral, or causal feature families.
- GBDT backend shootouts with XGBoost/CatBoost.
- Stacking many model families or leaderboard-style broad ensembles.
- Rewriting the thesis into a general feature-engineering benchmark.

## Source-Of-Truth Order

When documents disagree, use this order:

1. `docs/THESIS_SCOPE.md`
2. `docs/STRATIFIED_SPLIT_RESET.md`
3. `docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`
4. `docs/EXPERIMENT_REGISTRY.md`
5. `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`
6. `src/README.md`
