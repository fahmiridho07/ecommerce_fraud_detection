# Experiment Scope Freeze

## Purpose

This document freezes which experiments belong in the thesis narrative, which are supporting evidence only, and which remain archived. It does not move files or rerun experiments.

**Effective policy:** validation AP on chronological splits determines model-design conclusions; test AP is reported once for the frozen comparison set.

## Primary thesis comparison

Include **only** these four roles in the main results comparison table:

| Role | ID | Output path | Latent dim | Tuning | Validation AP | Test AP |
|------|-----|-------------|------------|--------|---------------|---------|
| P01 original-feature LightGBM default | P01 | `outputs/baseline_lgbm/` | — | Default | 0.602433 | 0.485756 |
| P02 original-feature LightGBM tuned | P02 | `outputs/optuna/baseline_lgbm/` | — | Optuna 15 trials | 0.624072 | 0.501438 |
| P03 AE replacement default | P03 | `outputs/ae_lgbm/` | **32** | Default | 0.591398 | 0.481593 |
| P04 AE replacement tuned | P04 | `outputs/optuna/ae_lgbm_ld128/` | **128** | Optuna 15 trials | 0.610631 | 0.490686 |

Metric sources: `metrics_validation_selected_threshold.json` and `metrics_test_selected_threshold.json` in each output directory.

### LD32 / LD128 comparability caveat

- **P03** is the thesis-original replacement pipeline (LD32 robust AE → latent replacement).
- **P04** tunes **LD128** only because `src/tune_lgbm_optuna.py` implements `ae_lgbm_ld128` and does not expose LD32 tuned training.
- P03 and P04 are **not** a pure default-vs-tuned pair at the same latent dimension.
- P04 may still be reported as the best available tuned AE replacement candidate, with the caveat documented.
- Do **not** claim LD128 was formally selected from a written validation-based rule in artifacts; see Phase 1 evidence in `docs/FINAL_EXPERIMENT_PLAN.md` and registry notes.

### Current primary conclusion (validation-based)

**P02 outperforms P04** on validation AP (0.624072 vs 0.610631) and test AP (0.501438 vs 0.490686). AE latent replacement did not improve over the original-feature baseline in the executed pipeline.

## Supporting baseline

| ID | Experiment | Role | Validation AP | Test AP | Evidence |
|----|------------|------|---------------|---------|----------|
| EX01 | FE LightGBM default | Supporting benchmark | 0.627793 | 0.509117 | `outputs/baseline_lgbm_entity_time_amount_features/` |
| EX02 | FE LightGBM tuned | Supporting benchmark | 0.654316 | 0.529857 | `outputs/optuna/baseline_lgbm_entity_time_amount_features/` |

FE models are **supporting benchmarks only**. Higher test AP (EX02 > P02) does **not** move FE into the central thesis comparison, because the research question is Autoencoder integration on the original-feature pipeline under chronological evaluation.

## Autoencoder diagnostics retained

Smallest distinct set for thesis ablation chapter / appendix:

| ID | Diagnostic | Why retained | Redundancy note |
|----|------------|--------------|-----------------|
| AE04–AE05 | Latent dimension replacement (LD64, LD128 defaults) | Isolates capacity under replacement design | Aggregated in `outputs/final_comparison/latent_dim_ablation.csv` |
| AE06 | Replacement vs augmentation (confounded) | Historical integration design | **Confounded:** retains V + latent + recon error |
| AE17 | Clean latent-only augmentation LD128 | Tests complement vs replace fairly | **Executed** — see `latent_integration_strategy_comparison.csv` |
| AAE01 | Selected-numerical AE replacement LD128 | Anchor-alignment input-scope diagnostic | **Executed** — see `autoencoder_input_scope_comparison.csv` |
| AAE02 | Selected-numerical reconstructed replacement | Ding-alignment output-strategy diagnostic | **Executed** — see `autoencoder_output_strategy_comparison.csv` |
| AE08 | Reconstruction error only (raw robust) | Tests latent vs scalar AE signal | Treat AE09 (log1p) as duplicate unless distinction verified |

**Do not retain as separate thesis evidence:**

- AE09 log1p recon — identical test AP to AE08 (0.496067) in saved artifacts
- AE10, AE13, AE14 — config-only paths, not executed
- AE15 behavioral/CDV — FE-space branch, not primary AE question
- AE07 tuned augmentation — secondary to AE06 design question

## Methodological appendix

| ID | Content | Thesis role |
|----|---------|-------------|
| MD01–MD04 | Chronological vs stratified holdout (baseline + FE) | Supports chronological protocol choice |
| MD05–MD06 | Stratified 5-fold CV OOF benchmarks | Non-temporal sensitivity only |

**Policy:** Appendix demonstrates that stratified evaluation yields much higher AP than chronological evaluation. It does **not** select the final model and must not replace primary results in the main chapter.

Evidence: `outputs/split_strategy_appendix/split_strategy_comparison.csv`, `stratified_cv_summary.csv`, `appendix_summary.json`.

## Archived experiments

These registry IDs remain in the repository for reproducibility but **must not drive** the final thesis narrative:

| ID | Experiment | Archive reason |
|----|------------|----------------|
| EX03 | UID FE branch | Parallel FE exploration |
| EX04 | Historical velocity FE | Parallel FE exploration |
| EX05 | Baseline+AE score ensemble | Post-hoc ensemble, not AE integration |
| EX06 | Three-model ensemble | Post-hoc ensemble |
| EX07 | FE+AE fine ensemble | Highest test AP branch; test-peeking risk |
| EX08–EX10 | Controlled FE+AE arms | FE-space, not primary question |
| EX11 | Optuna FE extended smoke | Incomplete smoke test |
| EX12 | `final_report/` assets | Contains descriptive test ranking; see `docs/FINAL_REPORT_GOVERNANCE_NOTE.md` |
| MD07–MD08 | Bootstrap / business diagnostics | Supplementary defense material |

Do not delete or move archived outputs.

## Resolved integration question (AE17)

**Question:** Does adding LD128 latent features while retaining original V-features improve validation AP versus P01?

**Experiment:** `outputs/ae_latent_only_augmented_lgbm_ld128/` (`src/train_ae_latent_only_augmented_lgbm.py`)

**Verified result (validation AP, selected threshold):**

| Model | Validation AP | Delta vs P01 |
|-------|---------------|--------------|
| P01 baseline default | 0.602433 | — |
| AE17 clean latent-only augmentation | 0.591898 | −0.010535 |
| AE05 replacement LD128 default | 0.594149 | −0.008284 |
| AE06 confounded augmentation | 0.598198 | −0.004235 |

**Conclusion:** Clean latent-only augmentation **does not improve** validation AP versus the original-feature baseline under the executed protocol. Evidence: `outputs/final_comparison/latent_integration_strategy_comparison.csv`.

AE17 remains an **Autoencoder diagnostic**, not a primary thesis model.

## Resolved anchor-alignment question (AAE01)

**Question:** Does broadening Autoencoder input from V-only to suitable numerical predictors improve chronological validation AP for AE-LightGBM latent replacement?

**Experiment:** `outputs/selected_numerical_ae_lgbm_ld128/` (`src/train_autoencoder_selected_numerical.py`, `src/train_selected_numerical_ae_lgbm.py`)

**Verified result (validation AP, selected threshold):**

| Model | Validation AP | Delta vs P01 |
|-------|---------------|--------------|
| P01 baseline default | 0.602433 | — |
| AE05 V-only replacement LD128 | 0.594149 | −0.008284 |
| **AAE01 selected-numerical replacement LD128** | **0.525103** | **−0.077330** |

**Conclusion (Rule C):** Broadening Autoencoder input does **not** improve the latent-replacement approach under the executed chronological protocol. Evidence: `outputs/final_comparison/autoencoder_input_scope_comparison.csv`, `docs/ANCHOR_ALIGNMENT_EXPERIMENT.md`.

AAE01 is an **anchor-alignment Autoencoder diagnostic**, not a primary thesis model and not permission to create further AE variants.

## Resolved Ding-alignment question (AAE02)

**Question:** Does using decoder-reconstructed numerical features instead of bottleneck latent features improve chronological validation AP for selected-numerical AE–LightGBM?

**Experiment:** `outputs/selected_numerical_reconstructed_lgbm/` (`src/generate_selected_numerical_reconstructed_features.py`, `src/train_selected_numerical_reconstructed_lgbm.py`)

**Verified result (validation AP, selected threshold):**

| Model | Validation AP | Delta vs P01 |
|-------|---------------|--------------|
| P01 baseline default | 0.602433 | — |
| AAE01 latent replacement | 0.525103 | −0.077330 |
| **AAE02 reconstructed replacement** | **0.549737** | **−0.052696** |

**Conclusion (Rule B):** Decoder reconstruction preserves more useful information than latent replacement (+0.024634 vs AAE01) but does **not** outperform original-feature LightGBM. Evidence: `outputs/final_comparison/autoencoder_output_strategy_comparison.csv`, `docs/DING_RECONSTRUCTION_ALIGNMENT_EXPERIMENT.md`.

AAE02 is the **final permitted anchor-alignment diagnostic**. No further AE branches are authorized.

## Resolved causal behavioral question (CBA01R — corrected authoritative 2026-06-10)

**Question:** Do identity-aligned causal behavioral features improve chronological validation AP versus P01?

**Experiment:** `outputs/causal_behavioral_lgbm_id_aligned/` (`src/train_causal_behavioral_lgbm.py --id-aligned`, `src/causal_behavioral_features.py`)

| Model | Validation AP | Delta vs P01 | Status |
|-------|---------------|--------------|--------|
| P01 baseline default | 0.602433 | — | original reference |
| CBA01 B2 (legacy) | 0.613738 | +0.011305 | provisional / superseded |
| **CBA01R B2 corrected** | **0.615122** | **+0.012689** | **corrected authoritative** |

**Conclusion (Rule A):** Identity-aligned causal behavioral features improve LightGBM under the executed chronological protocol. Evidence: `outputs/final_comparison/causal_behavioral_alignment_correction.csv`, `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md`.

CBA01 remains archived as **provisional** due to 16,309 within-split TransactionID mismatches under the legacy generator. CBA01R is the authoritative B2 result.

## Resolved CDV-after-behavioral question (CBA02R — corrected authoritative 2026-06-10)

**Question:** Does ID-aligned CDV reconstruction error add complementary validation AP after corrected causal behavioral features?

**Experiment:** `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/` (`src/train_causal_behavioral_cdv_reconstruction_lgbm.py --id-aligned`)

| Model | Validation AP | Delta vs CBA01R | Status |
|-------|---------------|-----------------|--------|
| CBA01R corrected B2 | 0.615122 | — | corrected authoritative |
| CBA02 B3 (legacy) | 0.600659 | −0.013079 vs provisional B2 | provisional / superseded |
| **CBA02R B3 corrected** | **0.600607** | **−0.014515** | **corrected authoritative** |

**Conclusion (Rule D):** ID-aligned CDV reconstruction error does **not** provide additional validation benefit beyond corrected causal behavioral features. Frozen CDV AE reused; no retraining.

CBA02 is archived as provisional. CBA02R is the authoritative B3 result. **Late fusion remains blocked** pending supervisor approval.

## Resolved task-aware AE question (TAE01 — executed 2026-06-10)

**Question:** Does adding a supervised fraud-classification objective during Autoencoder training produce a latent representation more useful for downstream LightGBM than unsupervised selected-numerical latent replacement (AAE01)?

**Experiment:** `outputs/task_aware_ae_lgbm_ld128/selected/` (`src/train_task_aware_autoencoder_selected_numerical.py`, `src/train_task_aware_ae_lgbm.py`)

| Model | Validation AP | Delta vs AAE01 |
|-------|---------------|----------------|
| AAE01 unsupervised latent replacement | 0.525103 | — |
| **TAE01 task-aware latent replacement (λ=0.1)** | **0.524481** | **−0.000621** |

**Conclusion (Rule C):** Task-aware supervision does **not** improve selected-numerical latent replacement for downstream LightGBM under the executed protocol. Evidence: `outputs/final_comparison/task_aware_ae_comparison.csv`, `docs/TASK_AWARE_AUTOENCODER_EXPERIMENT.md`.

TAE01 is the **final permitted AE integration diagnostic**. It is not automatically a replacement for primary thesis models P01–P04.

## Test freeze rule (post-TAE01)

1. **Model-design conclusions use validation AP** on chronological splits.
2. **No further experiments** unless supervisor approves an exception. Anchor-alignment (AAE01, AAE02), causal behavioral (CBA01, CBA02), and task-aware AE (TAE01) phases are closed.
3. **Do not run** additional entity keys, time windows, multiple AE error signals, behavioral tuning, stacking, GNNs, VAE, GAN, attention, LSTM, SMOTE, lambda expansion, or Autoencoder architecture changes after TAE01.
3. **Stratified appendix results must not determine model selection.**
4. **Historical test inspection** across EX07/EX08/`final_summary.json` remains a documented limitation.

## Related documents

- [`docs/EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) — full inventory
- [`docs/FINAL_EXPERIMENT_PLAN.md`](FINAL_EXPERIMENT_PLAN.md) — reporting template and fair-comparison rules
- [`docs/FINAL_REPORT_GOVERNANCE_NOTE.md`](FINAL_REPORT_GOVERNANCE_NOTE.md) — historical test-ranking correction
- [`docs/RESULT_ARTIFACT_MANIFEST.md`](RESULT_ARTIFACT_MANIFEST.md) — Git-tracking recommendations