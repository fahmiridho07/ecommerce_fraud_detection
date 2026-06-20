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
| OP-FIX | Original proposal AE feature-improvement ladder | `outputs/stratified_reset/ae_feature_improvement_ladder/` | Complete, no AE variant beats tuned LightGBM | 0.871309 |
| OP-FIX2 | Diagnosis-driven AE fix ladder | `outputs/stratified_reset/ae_diagnosis_fix_ladder/` | Complete, no AE variant beats tuned LightGBM | 0.871646 |

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

## Advisor Diagnostic: Original Proposal Literal Rerun (2026-06-19)

This rerun answers the advisor-facing question: whether the seminar proposal
original design had already been tested under its own stated stratified protocol.
Full diagnosis is in `docs/PROPOSAL_EXPERIMENT_ALIGNMENT_AUDIT.md`. Harness:
`src/run_original_proposal_stratified.py`.

Contract reproduced: IEEE-CIS stratified 60/20/20, baseline LightGBM, V-only
Autoencoder on `V1-V339` with zero-imputation and z-score scaling, encoder
latent replacement of original V features, no resampling, and Optuna/TPE for
both baseline and AE-LightGBM.

Output:

```text
outputs/stratified_reset/original_proposal_v_latent_replacement/
```

| ID | Experiment | Test AP | Delta vs matched baseline | p(delta<=0) | Status |
|----|------------|--------:|--------------------------:|------------:|--------|
| OP-B0 | Original proposal baseline LightGBM default | 0.859857 | reference | - | Complete |
| OP-AE0 | Original proposal AE V-latent replacement default | 0.849081 | -0.010776 | 1.000 | Complete, loses |
| OP-BT | Original proposal baseline LightGBM tuned | 0.873133 | reference | - | Complete |
| OP-AET | Original proposal AE V-latent replacement tuned | 0.860110 | -0.013023 | 1.000 | Complete, loses |

Conclusion: the exact original proposal mechanism has now been tested and is
not competitive against its matched LightGBM baseline. The likely cause is
information loss from replacing 339 raw V features, including useful missingness
patterns, with a 32-dimensional unsupervised reconstruction latent. This supports
keeping original proposal results as diagnostic evidence and applying any
improvement as an Autoencoder + LightGBM integration fix, not as an unrelated
method jump.

## Advisor Diagnostic Follow-Up: AE Feature Improvement Ladder (2026-06-19)

Full detail is in `docs/AE_FEATURE_IMPROVEMENT_LADDER_RESULTS.md`. Harness:
`src/run_ae_feature_improvement_ladder.py`.

Question: after the literal original-proposal AE replacement loses, do
proposal-consistent fixes recover enough AP to beat tuned LightGBM?

Reference:

```text
original proposal tuned LightGBM test AP = 0.873133
```

Output:

```text
outputs/stratified_reset/ae_feature_improvement_ladder/
```

| ID | Experiment | Test AP | Delta vs tuned LightGBM | p(delta<=0) | Status |
|----|------------|--------:|------------------------:|------------:|--------|
| OP-FIX-S1 | Replace V with larger latent dim, best LD64 | 0.859875 | -0.013258 | 1.00 | Complete, loses |
| OP-FIX-S2 | Concatenate original V + AE latent, best LD32 | 0.864217 | -0.008917 | 1.00 | Complete, loses |
| OP-FIX-S3 | Denoising AE concat LD32 | 0.864170 | -0.008963 | 1.00 | Complete, loses |
| OP-FIX-S4R | Partial reconstruction replace high-missing V | 0.871309 | -0.001824 | 0.89 | Complete, closest but does not beat |
| OP-FIX-S4A | Partial reconstruction append high-missing V | 0.868841 | -0.004292 | 1.00 | Complete, loses |

Conclusion: no AE feature-improvement variant beat tuned LightGBM. Partial
replacement of 159 high-missing V features is the closest and substantially
narrows the gap, but still remains below the tuned LightGBM reference. This
strengthens the diagnosis that full V replacement causes information loss, while
also showing that preserving original V features and limiting AE reconstruction
to noisy/missing V columns is the most defensible proposal-consistent direction.
The deeper cause analysis and next proposal-consistent fix plan are documented
in `docs/AE_FAILURE_DEEP_DIAGNOSIS_AND_FIX_PLAN.md`.

## Advisor Diagnostic Follow-Up 2: Diagnosis-Driven AE Fix Ladder (2026-06-20)

Full detail is in `docs/AE_DIAGNOSIS_FIX_LADDER_RESULTS.md`. Harness:
`src/run_ae_diagnosis_fix_ladder.py`.

Question: after partial reconstruction became the closest proposal-consistent
direction, do missingness-aware, masked-loss, selective subset, reconstruction
error, or latent-activation fixes beat tuned LightGBM?

Reference:

```text
original proposal tuned LightGBM test AP = 0.873133
```

Output:

```text
outputs/stratified_reset/ae_diagnosis_fix_ladder/
```

| ID | Experiment | Test AP | Delta vs tuned LightGBM | p(delta<=0) | Status |
|----|------------|--------:|------------------------:|------------:|--------|
| OP-FIX2-S1 | Missingness-aware partial AE replace + mask | 0.870329 | -0.002804 | 0.973 | Complete, loses |
| OP-FIX2-S2 | Masked-loss partial AE, observed-only replace | 0.871646 | -0.001487 | 0.900 | Complete, closest but does not beat |
| OP-FIX2-S3 | Selective top-64 masked-loss partial AE | 0.871624 | -0.001509 | 0.897 | Complete, loses |
| OP-FIX2-S4 | Reconstruction-error features, keep original V | 0.869771 | -0.003363 | 0.997 | Complete, loses |
| OP-FIX2-S5 | Linear latent concat LD32, keep original V | 0.864650 | -0.008483 | 1.000 | Complete, loses |

Conclusion: no diagnosis-driven AE fix beat tuned LightGBM. The best result is
masked-loss partial reconstruction with observed-only replacement, which narrows
the AP gap to `-0.001487` but remains below the tuned LightGBM reference. This
supports the thesis-facing conclusion that the current AE feature-engineering
family is not stronger than a tuned LightGBM on original IEEE-CIS features, even
after proposal-consistent fixes.

## Advisor Diagnostic Follow-Up 3: Broad AE Feature Ladder (2026-06-20)

Full detail is in `docs/BROAD_AE_FEATURE_LADDER_RESULTS.md`. Harness:
`src/run_broad_ae_feature_ladder.py`.

Question: if the AE is not restricted to only `V` features, and instead learns
cross-family or group-wise representations that are appended to the original
LightGBM feature matrix, can it beat the tuned LightGBM reference while staying
inside the original AE + LightGBM corridor?

Reference:

```text
original proposal tuned LightGBM test AP = 0.873133
```

Output:

```text
outputs/stratified_reset/broad_ae_feature_ladder/
```

Runtime note: augmented LightGBM matrices repeatedly triggered native LightGBM
crashes locally at the full tuned 999-estimator setting, so these broad
diagnostic variants were completed with `max_lgbm_estimators=600`.

| ID | Experiment | Test AP | Delta vs tuned LightGBM | p(delta<=0) | Status |
|----|------------|--------:|------------------------:|------------:|--------|
| OP-BROAD-S1 | All-feature AE top-192, append latent + error | 0.857028 | -0.016105 | 1.000 | Complete, loses |
| OP-BROAD-S2 | Group-wise AE, append latent + error | 0.868076 | -0.005057 | 1.000 | Complete, best broad variant but loses |
| OP-BROAD-S3 | All-feature AE reconstructs value + missingness mask | 0.857716 | -0.015418 | 1.000 | Complete, loses |
| OP-BROAD-S4 | Normal-only all-feature AE anomaly error | 0.867750 | -0.005383 | 1.000 | Complete, loses |
| OP-BROAD-S5 | All-feature AE with auxiliary fraud head | 0.848394 | -0.024739 | 1.000 | Complete, loses |

Conclusion: broadening the AE beyond the `V` block does not recover enough
signal to beat tuned LightGBM. The best broad result is group-wise AE
(`-0.005057` AP), which suggests feature-family separation is better than one
global AE, but still remains below the original-feature tuned LightGBM. This
supports the cause analysis that original IEEE-CIS features already contain
strong granular and missingness signal that AE compression/reconstruction does
not improve for this protocol.

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
