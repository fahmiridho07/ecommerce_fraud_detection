# Thesis Scope

Status: active Bab 4 writing scope, 2026-06-18.

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
- Autoencoder latent-space oversampling as the final thesis-facing AE mechanism.
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

The active thesis-facing result is now:

- AE latent and reconstruction-error features **do not help** LightGBM; all
  feature-level AE variants lose or tie against the matched baseline.
- Minority augmentation improves the baseline under both stratified and temporal
  checks.
- The AE-specific contribution is representation-dependent: AE latent-space
  oversampling ties SMOTE-NC on raw/NaN-native A0 features, but significantly
  beats SMOTE-NC on dense Alharbi-style A1 features.
- The final tuned A1 comparison is: baseline AP 0.838988, SMOTE-NC AP 0.843476,
  AE latent-SMOTE AP 0.850031. AE beats the tuned baseline by +0.011043 AP and
  tuned SMOTE-NC by +0.006555 AP, both with paired-bootstrap support.

Use `docs/THESIS_RESULTS_BAB4.md` for write-ready prose and
`docs/AE_INTEGRATION_EXPERIMENT_RESULTS.md` for the detailed empirical record.

All metrics produced before the stratified reset are historical and must not be
compared directly with active stratified result tables.

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

## Completed Rerun Ladder

The active ladder has been executed from simple to complex:

1. `S0`: validate stratified split summary.
2. `A0`: baseline LightGBM on original features under stratified holdout.
3. `AE-F`: feature-level and score-level AE integration on A0.
4. `AE-G`: AE latent-space minority augmentation on A0.
5. `AE-G-fair`: matched random oversampling and SMOTE-NC controls.
6. `AE-G-rep`: repeated split validation.
7. `AE-G-temp`: chronological robustness check.
8. `AE-A1`: dense Alharbi-style representation, AE vs SMOTE-NC.
9. `AE-A1-FB`: full-budget confirmation.
10. `AE-A1-TUNED`: fair Optuna tuned-vs-tuned comparison.

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
- Classical SMOTE/ADASYN as the standalone thesis contribution. SMOTE-NC is used
  as a required control for isolating the AE-specific contribution.
- Target encoding without strict out-of-fold leakage controls.
- Broad UID, velocity, rolling-window, behavioral, or causal feature families.
- GBDT backend shootouts with XGBoost/CatBoost.
- Stacking many model families or leaderboard-style broad ensembles.
- Rewriting the thesis into a general feature-engineering benchmark.
- RankGauss/swap-noise AE variants as a new mainline method; they remain
  diagnostic/future-work unless promoted by a new written scope decision.

## Source-Of-Truth Order

When documents disagree, use this order:

1. `docs/THESIS_SCOPE.md`
2. `docs/THESIS_RESULTS_BAB4.md`
3. `docs/AE_INTEGRATION_EXPERIMENT_RESULTS.md`
4. `docs/EXPERIMENT_REGISTRY.md`
5. `docs/STRATIFIED_SPLIT_RESET.md`
6. `docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`
7. `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`
8. `src/README.md`
