# Experiment Registry

This is the active thesis registry after cleanup. It tracks only the original proposal-scope experiments (P01–P04). Parked work is documented in [`../archive/README.md`](../archive/README.md).

## Canonical Artifact Location

The **authoritative** P01–P04 rerun (post missingness-preserving AE fix) lives under `outputs/initial_proposal/`. Legacy paths under `outputs/baseline_lgbm/`, `outputs/ae_lgbm/`, and `outputs/optuna/` remain for backward compatibility but reflect the pre-fix pipeline.

| Artifact | Path |
|----------|------|
| Four-row comparison table | `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv` |
| Extended comparison (P01–P04 + AE-05) | `outputs/initial_proposal/final_comparison/extended_proposal_comparison.csv` |
| Missing-artifact log | `outputs/initial_proposal/final_comparison/initial_proposal_missing_artifacts.json` |

## Active Thesis Experiments

| Canonical ID | Legacy ID | Experiment | Script | Canonical output path | Status |
|--------------|-----------|------------|--------|----------------------|--------|
| BASE-01 | P01 | Baseline LightGBM default | `src/train_baseline_lgbm.py` | `outputs/initial_proposal/baseline_lgbm_default/` | Complete (post-fix rerun) |
| BASE-02 | P02 | Baseline LightGBM Optuna tuned | `src/tune_lgbm_optuna.py --model_type baseline_lgbm` | `outputs/initial_proposal/optuna/baseline_lgbm_tuned/` | Complete (post-fix rerun) |
| AE-01 | P03 | AE-LightGBM LD32 + `v_missing_*` indicators | `src/train_autoencoder_robust.py --latent-dim 32`, `src/train_ae_lgbm.py` | `outputs/initial_proposal/ae_lgbm_ld32_default/` | Complete (post-fix rerun) |
| AE-02 | P04 | AE-LightGBM LD128 + `v_missing_*`, Optuna tuned | `src/train_autoencoder_robust.py --latent-dim 128`, `src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128` | `outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned/` | Complete (post-fix rerun) |

Supporting Autoencoder artifacts:

| Step | Path |
|------|------|
| AE LD32 | `outputs/initial_proposal/autoencoder_robust_ld32/` |
| AE LD128 | `outputs/initial_proposal/autoencoder_robust_ld128/` |

## Current Metrics (Post-Fix Rerun)

Rerun date: **2026-06-16**. Full IEEE-CIS train set (`SAMPLE_SIZE=None`). Chronological split per `outputs/split_summary.json`. Primary metric: **validation/test Average Precision (PR-AUC)** at the validation-selected threshold. Optuna tuning used `--tuning_profile final` with **15 trials** (exploratory budget).

| ID | Val AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Features | Tuning trials |
|----|--------|---------|--------------|---------|----------|----------|---------------|
| P01 / BASE-01 | 0.602433 | 0.485756 | 0.875195 | 0.477868 | 0.486715 | 432 | — |
| P02 / BASE-02 | 0.631767 | **0.504900** | 0.883431 | 0.493865 | 0.494270 | 432 | 15 |
| P03 / AE-01 | 0.590998 | 0.480217 | 0.880349 | 0.477902 | 0.472922 | 464 | — |
| P04 / AE-02 | 0.606709 | 0.484527 | 0.878874 | 0.485623 | 0.496689 | 560 | 15 |

**Test PR-AUC ranking:** P02 > P01 > P04 > P03.

### Feature construction (post-fix)

| Model | Non-V | Latent | `v_missing_*` | Total |
|-------|-------|--------|---------------|-------|
| P01 / P02 | 93 | — | — (original V* kept as NaN) | 432 |
| P03 | 93 | 32 | 339 | 464 |
| P04 | 93 | 128 | 339 | 560 |

### AE reconstruction drift (masked MSE, observed cells only)

| Latent dim | Train MSE | Valid MSE | Test MSE |
|------------|-----------|-----------|----------|
| LD32 | 0.0199 | 0.0355 | 0.1516 |
| LD128 | 0.0056 | 0.0128 | 0.0629 |

LD128 reconstructs better and drifts less, but P04 still does not beat P02 on test PR-AUC.

## Narrative Governance

Per `docs/THESIS_SCOPE.md`: **no thesis narrative updates** until a model beats **P02** (tuned baseline) by a meaningful test PR-AUC margin. Ablation and diagnostic runs below are logged for engineering traceability only.

## Thesis-Facing Finding (Post-Fix, P01–P04 Only)

After correcting the AE pipeline (median imputation, masked reconstruction loss, linear latent layer, and `v_missing_*` indicators downstream), **the tuned LightGBM baseline (P02) remains the best model** under the chronological IEEE-CIS protocol.

The missingness-preserving fix did not close the gap:

- P03 test AP moved from 0.481593 (pre-fix) to 0.480217 (post-fix).
- P04 test AP moved from 0.490686 (pre-fix) to 0.484527 (post-fix).

This supports a defendable thesis narrative: the AE latent replacement strategy loses supervised signal from original `V*` values that LightGBM exploits natively, and better AE reconstruction (LD128) does not automatically yield better fraud detection.

## Pre-Fix Local Metrics (Historical)

Keep for traceability only. Generated before median imputation, masked loss, linear latent, and `v_missing_*` indicators.

| ID | Validation AP | Test AP | Features |
|----|---------------|---------|----------|
| P01 / BASE-01 | 0.602433 | 0.485756 | 432 |
| P02 / BASE-02 | 0.624072 | 0.501438 | 432 |
| P03 / AE-01 | 0.591398 | 0.481593 | 125 |
| P04 / AE-02 | 0.610631 | 0.490686 | 221 |

Legacy artifact paths: `outputs/baseline_lgbm/`, `outputs/ae_lgbm/`, `outputs/optuna/`.

## Representation Ablation (Post-Diagnostic, Not P01–P04)

Controlled fix after diagnostics showed baseline `V*` gain share is 33.4% while `v_missing_*` contributes only 0.12% in P03.

| ID | Model | Script | Output path | Purpose |
|----|-------|--------|-------------|---------|
| AE-03 | Hybrid top-25 `V*` + LD32 latent | `src/train_ae_lgbm.py --retain-top-v-features 25` | `outputs/initial_proposal/ae_lgbm_ld32_top25v_default/` | Test whether retaining high-gain `V*` values fixes the replacement bottleneck before any AE tuning |

Comparison table:

```bash
python src/build_representation_ablation_comparison.py
```

Output: `outputs/initial_proposal/representation_ablation/representation_ablation_comparison.csv`

**AE-03 default (top-25 V + LD32 latent, no tuning):**

| Model | Val AP | Test AP | vs P01 | vs P03 |
|-------|--------|---------|--------|--------|
| P01 baseline | 0.602433 | 0.485756 | — | +0.00554 |
| P03 full replacement | 0.590998 | 0.480217 | −0.00554 | — |
| AE-03 top-25 V + latent | 0.602539 | 0.485306 | −0.00045 | +0.00509 |

**AE-04 tuned hybrid (top-25 V + LD32 latent, 25 Optuna trials, 2026-06-16):**

| Model | Val AP | Test AP | vs P01 | vs P02 | vs P03 |
|-------|--------|---------|--------|--------|--------|
| P02 tuned baseline | 0.631767 | 0.504900 | +0.01914 | — | +0.02468 |
| **AE-04 hybrid tuned** | **0.624804** | **0.503602** | **+0.01785** | **−0.00130** | **+0.02339** |

Output: `outputs/initial_proposal/optuna/ae_lgbm_ld32_top25v_tuned/`

Significance table: `outputs/initial_proposal/representation_ablation/significance_comparison.csv`

AE-03 default (test AP 0.4853) is close to P01 (0.4858). AE-04 hybrid tuned reaches test AP **0.5036** — **below P02** (0.5049, Δ −0.0013) and above P03 (+0.0234). These numbers do **not** change the canonical thesis conclusion; narrative stays frozen until P02 is beaten.

## AE-05 Candidate: Hybrid + Reconstruction-Error Score (Decision Gate Passed)

After the paper-aligned AE refinements showed that grouped/normal-only reconstruction scores were not stable under the chronological test split, the most defensible AE integration was narrowed to an additive hybrid:

- keep the strongest raw supervised `V*` values instead of forcing full latent replacement;
- keep LD32 AE latent features for the replaced lower-gain `V*` block;
- append the global AE reconstruction error as an anomaly score;
- keep LightGBM as the supervised classifier and optimize/evaluate with PR-AUC.

This follows the same methodological direction as AE + boosting hybrids and anomaly-score augmentation: the AE contributes learned representation and reconstruction-error signal, while the supervised tree model keeps the tabular signals it already handles well.

Command:

```bash
python src/tune_ae_hybrid_reconstruction_lgbm.py \
  --n_trials 0 \
  --output-dir outputs/initial_proposal/ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned

python src/compare_ae_hybrid_recon_bootstrap.py \
  --n-bootstrap 1000 \
  --output-dir outputs/initial_proposal/representation_ablation/bootstrap_ae05_vs_p02

python src/build_extended_proposal_comparison.py

python src/build_significance_comparison.py
```

Bootstrap output: `outputs/initial_proposal/representation_ablation/bootstrap_ae05_vs_p02/paired_bootstrap_pr_auc_delta.csv`.

The run uses the already selected AE-04 top-25 hybrid tuned LightGBM parameters, then adds two AE reconstruction-error features:

- `v_ae_reconstruction_mse`
- `v_ae_reconstruction_log1p_mse`

Result (2026-06-16):

| Model | Val AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Features |
|-------|--------|---------|--------------|---------|----------|----------|
| P02 tuned baseline | 0.631767 | 0.504900 | **0.883431** | 0.493865 | 0.494270 | 432 |
| AE-04 hybrid tuned | 0.624804 | 0.503602 | 0.881878 | 0.503370 | 0.506421 | 464 |
| **AE-05 hybrid + reconstruction error** | **0.626124** | **0.509821** | 0.882011 | **0.504766** | **0.512071** | 466 |

Delta vs P02: **+0.004921 test AP**, **+0.010902 F1**, **+0.017801 MCC**. ROC-AUC is slightly lower (-0.001420), so the thesis-facing gain should be framed around PR-AUC/AP and thresholded fraud-detection quality, not ROC-AUC.

Paired bootstrap against P02 tuned baseline on the same chronological test rows:

| Comparison | Metric | Delta | 95% CI | One-sided p(delta <= 0) |
|------------|--------|-------|--------|-------------------------|
| AE-05 - P02 | Average precision | +0.004921 | [+0.000650, +0.009316] | 0.009 |

Top gain features in AE-05:

| Rank | Feature | Gain |
|------|---------|------|
| 1 | `v_ae_reconstruction_log1p_mse` | 1,002,826 |
| 2 | `v_ae_reconstruction_mse` | 562,994 |
| 3 | `TransactionDT` | 524,988 |

Interpretation: AE-05 is the first post-cleanup AE candidate that beats the tuned raw-feature LightGBM baseline on test PR-AUC with a positive bootstrap interval. It should replace "AE cannot beat P02" as the active engineering conclusion, but keep P01-P04 as the canonical historical proposal rerun unless the thesis narrative is intentionally updated to include this post-diagnostic candidate.

## Preprocessing Ablation: Enhanced Identity/Device Normalization

The preprocessing diagnostic found strong categorical drift, especially `id_31` unseen categories among observed rows. A focused preprocessing ablation was added without changing the canonical P01-P04 scripts:

- normalize `id_31` into browser family and major version;
- normalize `id_30` into OS family and major version;
- parse `id_33` into screen dimensions and bucket;
- normalize `DeviceInfo` into device family;
- apply train-fitted rare-category bucketing (`rare_min_count=50`);
- keep numeric `NaN` values for LightGBM native missing handling.

Commands:

```bash
python src/train_enhanced_preprocessing_lgbm.py \
  --model-type baseline \
  --output-dir outputs/initial_proposal/preprocessing_ablation/baseline_enhanced_fixed_p02

python src/train_enhanced_preprocessing_lgbm.py \
  --model-type ae05 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/ae05_enhanced_fixed_ae05
```

Results:

| Model | Val AP | Test AP | ROC-AUC | F1 | MCC | Features |
|-------|-------:|--------:|--------:|---:|----:|---------:|
| P02 tuned baseline | 0.631767 | 0.504900 | 0.883431 | 0.493865 | 0.494270 | 432 |
| AE-05 hybrid reconstruction | 0.626124 | 0.509821 | 0.882011 | 0.504766 | 0.512071 | 466 |
| Enhanced baseline, fixed P02 params | **0.643247** | **0.516590** | **0.895311** | 0.503787 | 0.504735 | 438 |
| Enhanced AE-05, fixed AE-05 params | 0.631194 | 0.514975 | 0.889967 | **0.518194** | **0.514024** | 472 |

Interpretation: enhanced preprocessing improves both supervised baseline and AE-05. The best PR-AUC is now the enhanced baseline, while enhanced AE-05 gives the best thresholded F1/MCC. This shifts the next research question from "can AE beat the old P02?" to "can AE beat a preprocessing-strengthened baseline under matched tuning?"

## Reconstruction-Error Augmentation (Post-Fix, Not P01-P04)

Archive review showed that AE reconstruction error was the strongest historical AE integration path. This post-fix rerun keeps all original baseline features and appends only two LD128 Autoencoder anomaly features from `outputs/initial_proposal/autoencoder_robust_ld128/`:

- `v_ae_reconstruction_mse`
- `v_ae_reconstruction_log1p_mse`

Command:

```bash
python src/train_ae_reconstruction_error_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --output-dir outputs/initial_proposal/ae_reconstruction_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_RECON_LD128_initial_proposal_postfix
```

**AE-04 result (2026-06-16, default LightGBM, no Optuna):**

| Model | Val AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Features |
|-------|--------|---------|--------------|---------|----------|----------|
| P01 baseline default | 0.602433 | 0.485756 | 0.875195 | 0.477868 | 0.486715 | 432 |
| P02 baseline tuned | 0.631767 | **0.504900** | 0.883431 | 0.493865 | 0.494270 | 432 |
| P03 latent replacement LD32 | 0.590998 | 0.480217 | 0.880349 | 0.477902 | 0.472922 | 464 |
| P04 latent replacement LD128 tuned | 0.606709 | 0.484527 | 0.878874 | **0.485623** | **0.496689** | 560 |
| AE-04 reconstruction-error augmentation | 0.612397 | 0.495067 | 0.873653 | 0.485057 | 0.491299 | 434 |

Reconstruction-error augmentation (test AP 0.4951) is above P01 (+0.0093), P03 (+0.0149), and P04 (+0.0105), but remains below P02 (−0.0098). Top gain feature: `v_ae_reconstruction_mse`. Engineering note only — not thesis narrative until P02 is beaten.

## Autoencoder Refinement Ablations (Paper-Aligned, Not Promoted)

These runs test whether AE contribution becomes stronger before tuning by aligning the AE more closely with anomaly-detection literature:

- Normal-only AE: train only on non-fraud train rows and use reconstruction error as anomaly signal.
- Mask-aware AE: append the observed-cell mask to the AE input and keep masked MSE over observed `V*` cells.
- Light denoising: apply small Gaussian noise to scaled value inputs, not to masks.
- Grouped reconstruction features: expose global and `V*` block-level reconstruction errors to LightGBM.

Literature anchors:

- Jiang et al. (2023), UAAD-FDNet on IEEE-CIS, frames fraud as anomaly detection and trains on normal samples before scoring reconstruction/latent distances: https://www.mdpi.com/2079-8954/11/6/305
- Vincent et al. (2010), stacked denoising autoencoders, motivates learning robust representations from corrupted inputs: https://www.jmlr.org/papers/v11/vincent10a.html
- Saito and Rehmsmeier (2015), precision-recall evaluation for imbalanced datasets, supports AP/PR-AUC as primary metric: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

Commands:

```bash
python src/train_autoencoder_normal_masked.py \
  --latent-dim 128 \
  --output-dir outputs/initial_proposal/normal_masked_autoencoder_ld128 \
  --phase-name normal_only_mask_aware_autoencoder_ld128 \
  --input-noise-std 0.02

python src/train_ae_reconstruction_feature_lgbm.py \
  --ae-feature-dir outputs/initial_proposal/normal_masked_autoencoder_ld128 \
  --output-dir outputs/initial_proposal/ae_normal_masked_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_NORMAL_MASKED_GROUPED_ERRORS_LD128_default_lgbm

python src/train_ae_reconstruction_error_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/normal_masked_autoencoder_ld128 \
  --output-dir outputs/initial_proposal/ae_normal_masked_global_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_NORMAL_MASKED_GLOBAL_ERROR_LD128_default_lgbm

python src/generate_ae_reconstruction_feature_tables.py \
  --autoencoder-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --output-dir outputs/initial_proposal/autoencoder_robust_ld128_grouped_features \
  --feature-prefix robust_ae_ld128

python src/train_ae_reconstruction_feature_lgbm.py \
  --ae-feature-dir outputs/initial_proposal/autoencoder_robust_ld128_grouped_features \
  --output-dir outputs/initial_proposal/ae_robust_grouped_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_ROBUST_GROUPED_ERRORS_LD128_default_lgbm \
  --expected-feature-prefix robust_ae_ld128 \
  --allow-non-normal-source
```

Result summary (2026-06-16, default LightGBM):

| Model | Val AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Features |
|-------|--------|---------|--------------|---------|----------|----------|
| P01 baseline default | 0.602433 | 0.485756 | 0.875195 | 0.477868 | 0.486715 | 432 |
| P02 baseline tuned | 0.631767 | **0.504900** | **0.883431** | **0.493865** | 0.494270 | 432 |
| Robust LD128 global error | 0.612397 | **0.495067** | 0.873653 | 0.485057 | 0.491299 | 434 |
| Normal-only mask-aware global error | 0.607494 | 0.484140 | 0.875570 | 0.477174 | 0.474596 | 434 |
| Normal-only mask-aware grouped errors | 0.589097 | 0.468405 | 0.876282 | 0.455646 | 0.453729 | 453 |
| Robust LD128 grouped errors | 0.597145 | 0.480958 | 0.870634 | 0.470436 | 0.483129 | 453 |

The refined AE variants did not improve on robust LD128 global reconstruction-error augmentation. Normal-only mask-aware AE produced strong fraud/non-fraud reconstruction separation on train and validation, but the separation weakened sharply on test:

| AE | Split | Non-fraud MSE | Fraud MSE | Fraud / Non-fraud |
|----|-------|---------------|-----------|-------------------|
| Normal-only mask-aware | Train | 0.008456 | 0.122503 | 14.49x |
| Normal-only mask-aware | Validation | 0.015920 | 0.127910 | 8.03x |
| Normal-only mask-aware | Test | 0.076587 | 0.124454 | 1.63x |
| Robust LD128 | Train | 0.004718 | 0.029445 | 6.24x |
| Robust LD128 | Validation | 0.011573 | 0.043299 | 3.74x |
| Robust LD128 | Test | 0.063623 | 0.043953 | 0.69x |

Interpretation: normal-only AE is methodologically credible, but under this chronological IEEE-CIS split it is sensitive to temporal drift. The most stable thesis-facing AE contribution remains robust LD128 global reconstruction-error augmentation, not normal-only or grouped-error expansion.

## Diagnostics

Run after the isolated rerun completes:

```bash
python src/generate_initial_proposal_diagnostics.py
```

Outputs: `outputs/initial_proposal/diagnostics/` (`diagnostic_summary.json`, `diagnostic_notes.md`, and supporting CSV tables). See `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md` for the full file list.

## Caveats

- **P03 vs P04 is not fully apples-to-apples:** P03 uses LD32 with default LightGBM params; P04 uses LD128 with Optuna tuning. No tuned LD32 AE-LightGBM path exists in the active registry.
- **Tuning budget:** The post-fix rerun used 15 Optuna trials per tuned model. Increase to 50 before treating hyperparameters as final for defense.
- **Legacy vs canonical paths:** Scripts default to legacy `outputs/` directories. Use `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md` for isolated reruns under `outputs/initial_proposal/`.
- **Archived branches** may report higher metrics but answer different research questions and are out of proposal scope.

## Archived Families

| Archive folder | Status |
|----------------|--------|
| `archive/source/ae_appendix/` | Appendix only |
| `archive/source/behavioral_fusion/` | Out of proposal scope |
| `archive/source/feature_engineering_ensembles/` | Out of proposal scope |
| `archive/source/gbdt_wip/` | WIP only |
| `archive/source/methodology_reports/` | Reporting and diagnostics only |
| `archive/docs/` | Historical governance and branch notes |
| `archive/notebooks/` | Historical notebook report |
| `archive/results/` | Compact archived summaries |
