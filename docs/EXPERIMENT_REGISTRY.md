# Experiment Registry

Status: active stratified registry for Bab 4 writing.

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

The active post-reset experiment loop is complete enough for Bab 4 drafting.
Detailed bootstrap CIs and supporting runs are in
`AE_INTEGRATION_EXPERIMENT_RESULTS.md`; write-ready Indonesian prose is in
`THESIS_RESULTS_BAB4.md`.

| ID | Experiment | Output | Status | Test AP |
|----|------------|--------|--------|--------:|
| S0 | Split validation | `outputs/split_summary.json` | Complete | N/A |
| A0 | Raw/NaN-native LightGBM control | `outputs/stratified_reset/ae_integration_experiment_normal_ld32/` | Complete | 0.821840 |
| AE-F | A0 + AE feature/score integration | `outputs/stratified_reset/ae_integration_experiment_normal_ld32/` | Complete, loses/ties | 0.798149-0.821840 |
| AE-G | A0 + AE latent-SMOTE augmentation | `outputs/stratified_reset/ae_augmentation_experiment/` | Complete, beats A0 | 0.837371 |
| AE-G-fair | A0 augmentation controls: random, SMOTE-NC, AE | `outputs/stratified_reset/fair_augmentation_comparison/` | Complete, AE ties SMOTE-NC | 0.837371 |
| AE-G-temp | Chronological robustness for A0 augmentation | `outputs/stratified_reset/fair_augmentation_chronological/` | Complete, augmentation helps but AE loses to SMOTE-NC | 0.501923 |
| AE-A1 | Dense A1 AE vs SMOTE-NC | `outputs/stratified_reset/strong_baseline_augmentation/` | Complete, AE wins | 0.784061 |
| AE-A1-FB | Dense A1 full-budget confirmation | `outputs/stratified_reset/strong_baseline_full_budget/` | Complete, AE wins | 0.784061 |
| AE-A1-TUNED | Dense A1 tuned-vs-tuned comparison | `outputs/stratified_reset/a1_tuned_comparison/` | Complete, final headline | 0.850031 |

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
baseline is very strong and AE does not beat it on PR-AUC. Use it only as
diagnostic context, not as an active Bab 4 result table.

## Active Controlled AE Experiments (2026-06-18)

Matched comparison on the A0 baseline (original features, train-only fitted
preprocessing) under the active stratified split. Full detail and bootstrap CIs
are in `AE_INTEGRATION_EXPERIMENT_RESULTS.md`. Harnesses:
`src/run_ae_integration_experiment.py`, `src/run_ae_augmentation_experiment.py`.

| ID | Experiment | Test AP | Delta vs A0 | p(delta<=0) | Status |
|----|------------|--------:|------------:|------------:|--------|
| A0 | Baseline LightGBM (original features) | 0.821840 | reference | - | Complete |
| AE-F1 | A0 + AE recon-error (global) | 0.816413 | -0.005426 | 1.000 | Complete, loses |
| AE-F2 | A0 + AE recon-error (grouped) | 0.813677 | -0.008163 | 1.000 | Complete, loses |
| AE-F3 | A0 + AE latent (32) | 0.800850 | -0.020990 | 1.000 | Complete, loses |
| AE-F4 | A0 + grouped recon + latent | 0.798149 | -0.023691 | 1.000 | Complete, loses |
| AE-E1 | Score ensemble (A0 prob + AE anomaly) | 0.821840 | +0.000000 | 1.000 | Complete, tie (alpha=1.0) |
| AE-G1 | A0 + AE latent-SMOTE augmentation (no spw) | 0.837371 | **+0.015531** | **0.000** | Complete, **significant win** |
| AE-G2 | A0 + AE latent-SMOTE augmentation (spw) | 0.830381 | +0.008542 | 0.000 | Complete, win |

### Fairness controls and representation-dependence (final)

| ID | Experiment | Key result | Status |
|----|------------|-----------|--------|
| AE-G-fair | AE-latent-SMOTE vs SMOTE-NC vs random on A0 (raw) | AE beats random (+0.0067); AE ties SMOTE-NC (-0.0006, p=0.73) | Complete |
| AE-G-rep  | repeated split (4 seeds) A0: ae vs smote | tie, mean -0.0005, 0/4 AE wins | Complete |
| AE-G-temp | chronological A0: ae vs smote | AE worse than SMOTE (-0.0048, p=1.0); aug still beats baseline +0.0176 | Complete |
| AE-VAE    | VAE prior vs SMOTE-NC on A0 | tie (-0.0005, p=0.71) | Complete |
| AE-A1     | AE-latent-SMOTE vs SMOTE-NC on A1 (dense freq-encoded), 4 splits | **AE beats SMOTE +0.0204 mean, 4/4 splits, p<0.001** | Complete |
| AE-A1-FB  | A1 AE-vs-SMOTE at full 2000-tree budget | **AE beats SMOTE +0.02589, p<0.001** (matches 800 trees) | Complete |
| AE-A1-TUNED | A1 tuned-vs-tuned (Optuna 8 trials each): base 0.8390 / smote 0.8435 / ae 0.8500 | **AE beats baseline +0.0110 and SMOTE +0.0066, both p<0.001** | Complete |

Final headline: (1) feature/score AE integration hurts or ties; (2) minority
augmentation robustly beats the baseline (stratified +0.0155, temporal +0.0176);
(3) the AE earns a specific, robust, fair advantage over classical SMOTE ONLY as
a latent-space oversampler on DENSE frequency-encoded representations
(+0.020 AP, 4/4 splits) - representation-dependent, consistent with DeepSMOTE.
On raw NaN-native features AE ties SMOTE. Defensible thesis claim: the AE helps
as a generator on dense representations, not as a feature extractor. Full detail:
`AE_INTEGRATION_EXPERIMENT_RESULTS.md`.

### Ding-style AEELG anchor replication on IEEE-CIS (closed, 2026-06-18)

Isolated direct adaptation of Ding et al. (2024): A1 dense preprocessing,
train-only full-balance SMOTE, AutoEncoder feature reconstruction, then
LightGBM. Closed as negative-transfer evidence. Full detail:
`archive/docs/ae_appendix/DING_AEELG_IEEE_CIS_EXPERIMENT.md`.

| ID | Experiment | Test AP | Delta | Status |
|----|------------|--------:|------:|--------|
| DING-AEELG-B0 | A1 dense LightGBM control | 0.746013 | reference | Closed |
| DING-AEELG-SMOTE | A1 dense SMOTE-only control | 0.734853 | -0.011160 vs B0 | Closed, loses |
| DING-AEELG-R1 | AE reconstructed original train | 0.399351 | -0.346662 vs B0 | Closed, loses badly |
| DING-AEELG-R2 | AE reconstructed SMOTE-balanced train | 0.408320 | -0.337693 vs B0 | Closed, loses badly |

Conclusion: direct Ding-style reconstructed-feature AEELG is not competitive on
IEEE-CIS under the active stratified protocol. This supports using Ding et al.
as an initial methodological anchor and negative transfer test, but not as the
final winning AE mechanism.

## Final Thesis Claim Boundary

- Claim that AE helps as a latent-space oversampler on dense A1
  frequency-encoded features.
- Do not claim that AE helps as a feature extractor; those variants lose or tie.
- Do not claim that AE universally beats SMOTE-NC; it ties SMOTE-NC on raw A0
  features and loses to SMOTE-NC under the chronological A0 fair comparison.
- Do not claim absolute SOTA; compare literature numbers only with protocol
  caveats.

## Historical Evidence

Historical chronological results are still traceable:

| Evidence | Location |
|----------|----------|
| P01-P04 proposal block | `archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md` |
| AE-05 hybrid candidate | `archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md` |
| Best old score ensemble | `archive/docs/chronological_evidence/FINAL_CANDIDATE_VALIDATION.md` |
| Old preprocessing diagnostics | `archive/docs/chronological_evidence/PREPROCESSING_DIAGNOSTIC.md` |
