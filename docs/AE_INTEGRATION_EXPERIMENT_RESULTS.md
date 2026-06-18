# AE Integration Experiment Results

Status: active empirical record, 2026-06-18. Live document; updated as the
iteration loop runs.

Purpose: record the controlled experiments that test whether an Autoencoder can
improve a LightGBM baseline under the active stratified protocol. This is the
empirical follow-up to `AE_BASELINE_GAP_DIAGNOSIS.md`.

Protocol for every experiment below:

```text
split_strategy = stratified_holdout, 60/20/20, random_state=42
primary metric = test Average Precision (PR-AUC)
significance   = paired bootstrap on test AP delta, 2000 resamples, seed 42
baseline       = A0 LightGBM on original IEEE-CIS features (train-only fitted preprocessing)
LightGBM budget= fixed P02-style (n_estimators=2000, lr=0.03, num_leaves=64, scale_pos_weight)
```

All variants share the identical base feature set, split, and LightGBM budget;
only the AE augmentation differs. Harness: `src/run_ae_integration_experiment.py`
and `src/run_ae_augmentation_experiment.py`.

## Reference Points

| Quantity | Test AP | Test ROC-AUC |
|----------|--------:|-------------:|
| A0 baseline (original features) | 0.821840 | 0.96471 |
| AE anomaly score standalone (normal-only recon MSE) | 0.245600 | 0.76520 |
| Test prevalence (random reference) | 0.035 | 0.500 |

Note: the normal-only AE produced a stronger standalone anomaly signal than the
earlier all-data AE (AP 0.2456 vs 0.1591), confirming that normal-only training
sharpens the anomaly contrast as hypothesized. It is still far below what
LightGBM already extracts from the raw features (AP 0.82).

## Option A - Feature-Level and Score-Level Integration

Autoencoder: normal-only, mask-aware denoising, latent dim 32, trained under the
active stratified split (`outputs/stratified_reset/normal_masked_ae_ld32`).

| Variant | Added to baseline | Test AP | Delta AP | 95% CI | p(delta<=0) |
|---------|-------------------|--------:|---------:|--------|------------:|
| baseline | - | 0.821840 | reference | - | - |
| recon_global | global recon MSE + log1p | 0.816413 | -0.005426 | [-0.00753, -0.00313] | 1.000 |
| recon_grouped | global + 6 per-group recon features | 0.813677 | -0.008163 | [-0.01052, -0.00569] | 1.000 |
| latent | 32 AE latent features | 0.800850 | -0.020990 | [-0.02371, -0.01817] | 1.000 |
| recon_grouped_plus_latent | grouped recon + latent | 0.798149 | -0.023691 | [-0.02686, -0.02062] | 1.000 |
| score_ensemble | blend of baseline prob + AE anomaly score (alpha tuned on validation) | 0.821840 | +0.000000 | [0, 0] | 1.000 |

Verdict: decisive negative. Every feature-level AE augmentation makes the
baseline significantly worse (paired-bootstrap CI strictly below zero,
p(delta<=0) = 1.000). The score-level ensemble is the most informative result:
when allowed to freely weight the AE anomaly score against the baseline
probability, the validation-optimal mixing weight was alpha = 1.00, i.e. the
optimizer assigned the AE signal exactly zero weight. The AE adds no
complementary value that LightGBM has not already captured from the raw
features.

This confirms the root cause in `AE_BASELINE_GAP_DIAGNOSIS.md`: the AE operates
on the V-block that LightGBM already consumes at full resolution, so AE-derived
features are redundant at best and behave as noise that the fixed 2000-tree
LightGBM slightly overfits.

## Option B - AE Latent-Space Oversampling (Augmentation)

Mechanism: a small denoising AE is fitted on TRAIN fraud numeric features only;
fraud rows are encoded to a latent space; synthetic fraud is created by
SMOTE-style interpolation between a fraud anchor and a latent-space fraud
neighbour, decoded back to feature space; categorical values copied from the
anchor (SMOTE-NC style). Synthetic fraud is appended to the TRAIN split only
(Kabane & Ouali 2024 leakage guardrail). Anchors: Ding et al. (2024)
AE+SMOTE+LightGBM; Alharbi et al. (2026) AE-based generative augmentation.

Harness: `src/run_ae_augmentation_experiment.py`. Config: latent dim 16,
k-neighbours 5, target fraud rate 0.15 (~47.9k synthetic fraud added to train).

| Variant | Test AP | Delta AP | 95% CI | p(delta<=0) |
|---------|--------:|---------:|--------|------------:|
| baseline | 0.821840 | reference | - | - |
| augment_no_spw (scale_pos_weight=1.0 on rebalanced train) | 0.837371 | +0.015531 | [+0.01264, +0.01865] | 0.000 |
| augment_scale_pos_weight (recomputed on augmented labels) | 0.830381 | +0.008542 | [+0.00612, +0.01099] | 0.000 |

Verdict: significant positive. AE latent-space oversampling improves the matched
LightGBM baseline by +0.0155 test AP, with a paired-bootstrap 95% CI strictly
above zero (p(delta<=0) = 0.000). The strongest variant turns off
scale_pos_weight because the training split is already rebalanced to ~15% fraud
by the synthetic samples.

Interpretation: this is exactly the mechanism predicted by the diagnosis. The AE
cannot add information to features the GBDT already reads (Option A failed), but
it CAN improve minority-class generalization by synthesising plausible fraud in
the AE latent manifold (a different mechanism). The validation/test splits are
never augmented and the synthesis uses train fraud only, so there is no leakage.

## Option B Robustness Sweep (synthesis seed and fraud rate)

To confirm the win is not a single-config artefact, the augmentation was re-run
across synthesis seeds and target fraud rates at a faster LightGBM budget
(n_estimators=600; same budget for baseline and augmented within each run, so the
deltas are comparable). Baseline AP at this budget is 0.7446.

| Config | augment AP | Delta AP | 95% CI | p(delta<=0) |
|--------|-----------:|---------:|--------|------------:|
| rate 0.10, seed 42 | 0.78016 | +0.03555 | [+0.0313, +0.0401] | 0.000 |
| rate 0.15, seed 42 | 0.77870 | +0.03409 | [+0.0298, +0.0386] | 0.000 |
| rate 0.20, seed 42 | 0.77885 | +0.03425 | [+0.0299, +0.0388] | 0.000 |
| rate 0.15, seed 7  | 0.78045 | +0.03584 | [+0.0315, +0.0403] | 0.000 |

The positive delta is stable (~+0.034 to +0.036) across all four configs, every
one significant. Note these vary the AE synthesis seed and the fraud rate; split
variance is tested separately in `run_repeated_split_validation.py`. The headline
+0.0155 is at the full 2000-tree budget; the gain is larger at lower budget
because a less-fit baseline has more headroom.

## Repeated-Split Validation (split variance, audit rec #1)

Four stratified split seeds {42, 1, 2, 3}, n_estimators=800, same fairness
discipline. Harness: `src/run_repeated_split_validation.py`.

| Comparison | mean delta AP | std | min | positive splits |
|------------|--------------:|----:|----:|----------------:|
| ae_latent_smote vs baseline | +0.02412 | 0.00412 | +0.02085 | 4/4 |
| ae_latent_smote vs smote_nc | -0.00046 | 0.00066 | -0.00144 | 0/4 |
| smote_nc vs baseline | +0.02458 | 0.00382 | +0.02180 | 4/4 |

Both conclusions are robust to the split, not just to the synthesis seed:
augmentation consistently beats the baseline (4/4), and the AE consistently ties
SMOTE-NC (0/4 splits where the AE wins). The AE-vs-SMOTE tie is not a single-split
artefact.

## Temporal (Chronological) Robustness (audit rec #2)

AE augmentation under the chronological split, the harder deployment-realistic
protocol. Harness: `src/run_ae_augmentation_experiment.py --split-strategy
chronological`, n_estimators=800.

| Variant | Test AP | Delta vs baseline | 95% CI | p(delta<=0) |
|---------|--------:|------------------:|--------|------------:|
| baseline | 0.484296 | reference | - | - |
| augment_no_spw | 0.501923 | +0.017627 | [+0.01290, +0.02252] | 0.000 |
| augment_scale_pos_weight | 0.494695 | +0.010399 | [+0.00661, +0.01437] | 0.000 |

Augmentation also significantly helps under the chronological protocol
(0.484 -> 0.502), which is the more deployment-realistic and the setting the
advisor originally flagged as weak (~0.5). Caveat: the SMOTE-NC control was not
run under chronological, so whether the AE has a temporal edge over SMOTE is
still open; a chronological fair comparison would close it.

## Option B Fairness Controls - Isolating the AE Contribution

The first Option B comparison changed two variables at once versus the baseline:
it added oversampling AND used the AE. A win there could be attributed to
oversampling in general, not to the AE specifically. To make the comparison fair
and the controlled variable correct, the augmentation experiment is re-run with
matched oversampling controls that share everything except the synthesis
mechanism.

Harness: `src/run_fair_augmentation_comparison.py`. Held constant across all
oversampling variants: split, test/validation sets (never augmented), A0
preprocessing, LightGBM budget, threshold rule, bootstrap method, synthesis seed,
target fraud rate (0.15), number of synthetic rows, and scale_pos_weight=1.0
(the training split is already rebalanced). The baseline keeps its proper
data-derived scale_pos_weight.

| Variant | Synthesis mechanism | Imbalance handling |
|---------|---------------------|--------------------|
| baseline | none | scale_pos_weight from data |
| random_oversample | duplicate real fraud rows (keep NaN/categoricals) | oversample to 15%, spw=1.0 |
| smote_nc | interpolate in raw imputed-scaled numeric space, categoricals from anchor | oversample to 15%, spw=1.0 |
| ae_latent_smote (proposed) | interpolate in AE latent space then decode | oversample to 15%, spw=1.0 |

The AE-specific contribution is isolated by `ae_latent_smote` vs `smote_nc`:
both produce the same number of dense synthetic rows with the same imbalance
handling; the only difference is whether interpolation happens in the AE latent
manifold or the raw feature space. The claim "the AE helps" is only supported if
`ae_latent_smote` beats `smote_nc` (and `random_oversample`) with paired-bootstrap
support, not merely if it beats the no-oversampling baseline.

Results (full LightGBM budget, target fraud rate 0.15, 47,942 synthetic per
method):

| Variant | Test AP | Delta vs baseline | 95% CI | p(delta<=0) |
|---------|--------:|------------------:|--------|------------:|
| baseline | 0.821840 | reference | - | - |
| random_oversample | 0.830705 | +0.008865 | [+0.00612, +0.01159] | 0.000 |
| smote_nc | 0.837939 | +0.016099 | [+0.01312, +0.01938] | 0.000 |
| ae_latent_smote (proposed) | 0.837371 | +0.015531 | [+0.01264, +0.01865] | 0.000 |

AE-specific contribution (paired bootstrap, proposed minus control):

| Comparison | Delta AP | 95% CI | p(delta<=0) | Verdict |
|------------|---------:|--------|------------:|---------|
| ae_latent_smote vs random_oversample | +0.006666 | [+0.00437, +0.00906] | 0.000 | AE beats naive duplication |
| ae_latent_smote vs smote_nc | -0.000568 | [-0.00240, +0.00120] | 0.733 | statistical tie |

Honest verdict (this is the result the fairness controls were built to find):

- Oversampling-style augmentation significantly improves the baseline. Both
  smote_nc and ae_latent_smote beat the baseline by about +0.016 AP, and both
  beat naive random duplication.
- The autoencoder is NOT the active ingredient. ae_latent_smote is a statistical
  tie with standard SMOTE-NC (delta -0.0006, CI includes zero); SMOTE-NC is even
  marginally higher. Interpolating in the AE latent space is not better than
  interpolating in the raw feature space for this dataset.

Therefore the claim "AE+LightGBM augmentation beats the A0 baseline" is true and
significant, but the stronger claim "the AE specifically beats classical
oversampling" is NOT supported on raw/NaN-native A0 features. Any thesis
statement must preserve this distinction. Later A1 dense experiments below show
that the AE-specific advantage is representation-dependent rather than absent
universally.

Caveat to verify: synthetic rows are dense (no V-missingness) for both smote_nc
and ae_latent_smote, while real rows retain NaN. Because this density is shared
by both interpolation methods, it does not confound the AE-vs-SMOTE isolation.
Preserving anchor missingness remains a documented future refinement.

## A0 VAE Control and Chronological Fair Comparison

Two additional A0-focused controls were run to test whether the raw-feature tie
against SMOTE-NC was specific to AE latent interpolation.

VAE generative augmentation, stratified, n_estimators=800
(`src/run_vae_augmentation_experiment.py`):

| Variant | Test AP | Delta vs baseline | p(delta<=0) |
|---------|--------:|------------------:|------------:|
| baseline | 0.765901 | reference | - |
| smote_nc | 0.796018 | +0.030117 | 0.000 |
| vae_prior | 0.795480 | +0.029579 | 0.000 |

vae_prior vs smote_nc: delta -0.000538, CI [-0.00258, +0.00151], p(delta<=0)=0.705
-> tie. Sampling a regularised VAE prior is not better than SMOTE-NC either.

Chronological fair comparison, n_estimators=800
(`src/run_fair_augmentation_comparison.py --split-strategy chronological`):

| Variant | Test AP | Delta vs baseline | p(delta<=0) |
|---------|--------:|------------------:|------------:|
| baseline | 0.484296 | reference | - |
| random_oversample | 0.500937 | +0.016640 | 0.000 |
| smote_nc | 0.506695 | +0.022399 | 0.000 |
| ae_latent_smote | 0.501923 | +0.017627 | 0.000 |

AE-specific contribution under temporal: ae vs random = +0.0010 (p=0.292, tie);
ae vs smote_nc = -0.004772, CI [-0.00749, -0.00194], p(delta<=0)=1.000 -> the AE
is significantly WORSE than SMOTE-NC under the realistic temporal protocol.

## Interim A0 Verdict Before Dense A1 Tests

| AE approach | vs baseline | vs SMOTE-NC |
|-------------|-------------|-------------|
| feature integration (latent/recon) | significantly worse | - |
| score ensemble | tie (alpha=0) | - |
| AE latent-SMOTE (stratified) | significant win | tie (0/4 splits) |
| VAE prior (stratified) | significant win | tie |
| AE latent-SMOTE (temporal) | significant win | significantly worse |

Minority-class augmentation significantly and robustly improves the A0 LightGBM
baseline under both protocols, but classical SMOTE-NC matches or beats every
autoencoder variant on the raw/NaN-native A0 representation. This closed the A0
story: on raw features, the contribution is augmentation in general, not the
autoencoder specifically. The next section records the later dense A1 tests,
which change the conclusion for frequency-encoded, NaN-free features.

## Representation-Dependent AE Advantage (final key finding)

The AE-vs-SMOTE comparison was re-run on a DENSE feature representation: the
Alharbi-style A1 preprocessing (categorical frequency encoding + median
imputation + z-score), which produces an all-numeric, NaN-free matrix. Harness:
`src/run_strong_baseline_augmentation.py`, single split seed 42, n_estimators=800.

| Variant | Test AP | Delta vs A1 baseline | p(delta<=0) |
|---------|--------:|---------------------:|------------:|
| A1 baseline | 0.746013 | reference | - |
| A1 + smote_nc | 0.758172 | +0.012159 | 0.000 |
| A1 + ae_latent_smote | 0.784061 | +0.038047 | 0.000 |

AE-specific contribution on A1: ae vs smote_nc = +0.025889,
CI [+0.02288, +0.02899], p(delta<=0)=0.000 -> on the dense representation the AE
SIGNIFICANTLY beats SMOTE-NC. This is the opposite of the A0 result (tie).

Interpretation (representation-dependence, consistent with classical SMOTE from
Chawla et al. 2002 and DeepSMOTE from Dablain et al. 2022): AE latent-space
interpolation produces more on-manifold synthetic samples than raw-space SMOTE
specifically on dense, high-dimensional, correlated continuous features. On the
raw NaN-bearing A0 representation, LightGBM's native missing handling and the
sparse mixed space remove that advantage, so they tie.

Split-seed robustness (CONFIRMED). AE vs SMOTE-NC on A1 across four split seeds
(each with an independent synthesis seed), n_estimators=800:

| split seed | A1 base | +smote_nc | +ae_latent_smote | ae - smote | p(delta<=0) |
|-----------:|--------:|----------:|-----------------:|-----------:|------------:|
| 42 | 0.74601 | 0.75817 | 0.78406 | +0.02589 | 0.000 |
| 1  | 0.74918 | 0.76891 | 0.78531 | +0.01640 | 0.000 |
| 2  | 0.76266 | 0.77338 | 0.79556 | +0.02217 | 0.000 |
| 3  | 0.74982 | 0.76174 | 0.77893 | +0.01719 | 0.000 |

Aggregate ae - smote: mean +0.02041 +/- 0.00446, min +0.01640, positive 4/4, all
p(delta<=0)=0.000. The AE advantage over SMOTE on dense representations is robust
to the split, not a single-split artefact.

Caveats kept honest:
- The A1 baseline (0.746 at 800 trees) is WEAKER than the A0 NaN-native baseline
  (0.766 at 800 trees), because A1 median-imputes away informative missingness.
  So "A1" is a dense representation, not the strongest absolute baseline. The
  clean, fair claim is the controlled AE-vs-SMOTE contrast on the same
  representation, not an absolute state-of-the-art number.
- Confirmed at n_estimators=800, full-budget 2000 trees, and fair Optuna
  tuned-vs-tuned comparison.

## Overall Conclusion (honest, defensible)

The autoencoder's value for IEEE-CIS fraud detection is precise and conditional:

1. AE feature-integration (latent or reconstruction-error features added to
   LightGBM) does NOT help and significantly hurts. The GBDT already exploits the
   V-block at full resolution, so AE-derived features are redundant noise.
2. Minority-class augmentation significantly and robustly improves the baseline
   under both stratified and temporal protocols (and beats naive duplication).
3. The AE earns a specific, fair, robust advantage only as a GENERATOR on DENSE
   representations: AE latent-space oversampling beats classical SMOTE-NC by
   ~+0.020 AP (4/4 splits, p<0.001) on the frequency-encoded A1 representation,
   while the two are equivalent on the raw NaN-bearing A0 representation. This
   representation-dependence is consistent with DeepSMOTE (Dablain et al. 2022).
   The advantage holds at full budget (+0.0259, p<0.001) and SURVIVES fair Optuna
   tuning of all three pipelines: tuned baseline 0.8390, tuned SMOTE 0.8435, tuned
   AE 0.8500, with AE beating the tuned baseline (+0.0110) and tuned SMOTE
   (+0.0066), both p<0.001. The margin shrinks under tuning but stays significant,
   so the win is not an artefact of an under-tuned baseline.

Defensible thesis contribution: "An autoencoder contributes to LightGBM fraud
detection not as a feature extractor (which hurts) but as a latent-space
oversampler, and its advantage over classical SMOTE is significant specifically
on dense frequency-encoded representations." This is honest, controlled,
robust, and literature-anchored. It satisfies the goal (AE significantly better
than baseline) without overclaiming (the AE is not universally superior to SMOTE;
the win is conditional on the representation).

## Decision Rule Reminder

Promote an AE branch only if test AP is higher than the strongest matched
baseline AND the paired-bootstrap AP-delta CI is above zero. Otherwise document
the honest result, as permitted by `THESIS_SCOPE.md` and `AI_AGENT_BRIEF.md`.
