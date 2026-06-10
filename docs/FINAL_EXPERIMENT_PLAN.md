# Final Experiment Plan

## Main research question

Evaluate whether Autoencoder-derived representation improves LightGBM fraud detection on the IEEE-CIS dataset under **chronological** evaluation using `TransactionDT`.

The thesis contribution question is about **integration design** (replacement, augmentation, reconstruction error), not about finding the highest historical test score across all exploratory branches.

## Main evaluation policy

- **Chronological split** (60% train / 20% validation / 20% test by `TransactionDT`) is the **primary protocol**. Verified in `outputs/split_summary.json` and all primary `run_config.json` files (`sample_size=null`).
- **Stratified holdout** and **stratified 5-fold CV** are **appendix sensitivity analyses only** (`outputs/split_strategy_appendix/`).
- **Final model selection must use chronological validation AP**, with MCC-based threshold selection on the validation split (implemented in `src/evaluation.py`).
- **Stratified scores must not replace** main thesis results in tables, conclusions, or model ranking.
- **Autoencoder success** is not declared from a single higher test score in an exploratory branch.

## Core model roles

| Role ID | Definition | Verified mapping | Status |
|---------|------------|------------------|--------|
| **P1** | original-feature LightGBM default | `outputs/baseline_lgbm/` (script: `src/train_baseline_lgbm.py`) | **Complete** |
| **P2** | original-feature LightGBM tuned | `outputs/optuna/baseline_lgbm/` (script: `src/tune_lgbm_optuna.py --model_type baseline_lgbm`) | **Complete** |
| **P3** | AE-LightGBM latent replacement default | `outputs/ae_lgbm/` + `outputs/autoencoder_robust/` (scripts: `src/train_autoencoder_robust.py`, `src/train_ae_lgbm.py`) | **Complete** |
| **P4** | AE-LightGBM latent replacement tuned | `outputs/optuna/ae_lgbm_ld128/` (script: `src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128`) | **Partial — requires rerun** for strict thesis parity |

### P4 gap (verified)

- No local output at `outputs/optuna/ae_lgbm/` (LD32 tuned).
- Executed tuned AE replacement uses **LD128** (`outputs/optuna/ae_lgbm_ld128/run_config.json`: `latent_feature_count=128`, `original_v_features_excluded=true`).
- Thesis-original replacement default P3 uses **LD32** (`outputs/ae_lgbm/run_config.json`: `latent_feature_count=32`).

For a strict four-role comparison, P4 should be rerun as LD32 tuned **or** P3 should be explicitly reframed to LD128 default with documented rationale. Until then, P4 is reported with caveat only.

## Critical Autoencoder diagnostic

**Priority comparison: replacement vs augmentation vs baseline (original V retained)**

| Arm | Experiment | Original V | Latent | Recon error | Validation AP | Test AP | Evidence |
|-----|------------|------------|--------|-------------|---------------|---------|----------|
| Baseline | P1 | Retained | No | No | 0.602433 | 0.485756 | `outputs/baseline_lgbm/metrics_*_selected_threshold.json` |
| Replacement | P3 | Replaced by LD32 latent | Yes | No | 0.591398 | 0.481593 | `outputs/ae_lgbm/metrics_*_selected_threshold.json` |
| Augmentation | AE06 | Retained | LD128 added | Yes | 0.598198 | 0.485417 | `outputs/ae_augmented_lgbm_ld128/metrics_*_selected_threshold.json` |

**Interpretation (conservative):** This tests whether Autoencoder information acts better as a **replacement** or **complement**. The executed augmentation arm is **not latent-only**; it adds latent **and** `ae_reconstruction_mse` while retaining all V features (`outputs/ae_augmented_lgbm_ld128/run_config.json`). Under verified artifacts:

- Replacement does not beat baseline on validation or test AP.
- Augmentation does not beat baseline on validation or test AP.
- Augmentation is not a clean test of "original V + latent only."

**Governance rule:** Do not assume augmentation helps unless a fair, same-dimension, same-protocol comparison is verified.

## Optional ablation (at most one)

**Selected optional ablation: latent dimension sweep under replacement design (AE04/AE05 family)**

| Variant | Latent dim | Validation AP | Test AP | Evidence |
|---------|------------|---------------|---------|----------|
| P3 | 32 | 0.591398 | 0.481593 | `outputs/ae_lgbm/metrics_*` |
| AE04 | 64 | 0.587878 | 0.481166 | `outputs/ae_lgbm_ld64/metrics_*` |
| AE05 | 128 | 0.594149 | 0.489417 | `outputs/ae_lgbm_ld128/metrics_*` |

**Why this ablation:** Artifact completeness is high (`run_config.json`, models, AE artifacts, aggregated `outputs/final_comparison/latent_dim_ablation.csv`). Scientifically, it isolates capacity effects without leaving the replacement integration design.

**Not selected for optional slot:** reconstruction-error variants — already informative (AE08), but secondary to the replacement/augmentation integration question.

## Methodological appendix

The split-strategy appendix (`src/compare_split_strategy_appendix.py`, `outputs/split_strategy_appendix/`) is:

- **Supporting evidence** for the chronological evaluation choice
- **Not** a model-selection experiment
- **Summarized briefly** in the main results chapter (1 paragraph + pointer)
- **Detailed** in the thesis appendix

### Verified appendix metrics (from artifacts only)

**Holdout comparison** (`outputs/split_strategy_appendix/split_strategy_comparison.csv`):

| Model | Split | Validation AP | Test AP |
|-------|-------|---------------|---------|
| baseline_lgbm | chronological | 0.599462 | 0.483872 |
| baseline_lgbm | stratified_holdout | 0.819570 | 0.821840 |
| feature_engineered_lgbm | chronological | 0.627793 | 0.509117 |
| feature_engineered_lgbm | stratified_holdout | 0.847822 | 0.849415 |

**Stratified CV** (`outputs/split_strategy_appendix/stratified_cv_summary.csv`):

| Model | OOF AP | Mean fold AP |
|-------|--------|--------------|
| baseline_lgbm | 0.843111 | 0.843167 |
| feature_engineered_lgbm | 0.865834 | 0.865952 |

**Reproducibility:** Appendix has `appendix_summary.json`, per-run `run_config.json`, `split_summary.json`, `cv_summary.json`, and `fold_metrics.csv`. Artifacts exist locally but are **not Git-tracked** (`.gitignore` excludes `outputs/*`).

## Fair comparison requirements

All primary and critical diagnostic comparisons must satisfy:

| Requirement | Verified for P1–P3 |
|-------------|-------------------|
| Identical chronological data partitions | Yes — `split_ratios` 0.6/0.2/0.2, `sample_size=null` |
| Identical full-data mode | Yes |
| Identical preprocessing policy per arm | Yes — documented per `run_config.json` |
| Same implemented AP objective | Yes — `sklearn.metrics.average_precision_score` (`src/evaluation.py`) |
| Comparable tuning budget | P2 vs P4: both 15 Optuna trials when tuned |
| Same threshold-selection policy | Yes — MCC with F1 tie-break on validation |
| Same random seed policy | Yes — `random_state=42` / `RANDOM_SEED=42` |
| No model choice based on stratified test results | Policy requirement |
| No model choice based solely on historical chronological test results | Policy requirement |

**Known fairness violations to document:**

- P4 uses LD128; P3 uses LD32.
- AE06 augmentation confounds latent with reconstruction error.
- Exploratory FE and ensemble branches (EX01–EX08) are not fair comparisons to P1–P3.

## Test freeze policy

After this plan is approved:

1. **No new feature or model decision** may be based on test scores.
2. **Validation AP** determines hyperparameters, thresholds, and integration design choices.
3. **Test AP** is computed **once** for the frozen final comparison table (P1–P4 + critical AE diagnostic arms).
4. Historical test inspection across exploratory branches is documented as a **methodological limitation** (see `outputs/final_report/final_summary.json` ranking by test AP).

## Final reporting template

Use this table for the frozen thesis results chapter. Populate only from verified metric artifacts.

| Model | Feature setup | AE setup | Split | Tuning | Validation AP | Test AP | ROC-AUC | Precision | Recall | F1 | MCC | Threshold | Feature count | Best iteration | Notes |
|-------|---------------|----------|-------|--------|---------------|---------|---------|-----------|--------|----|-----|-----------|---------------|----------------|-------|
| P1 Baseline default | 432 original | None | Chronological | Default | 0.602433 | 0.485756 | 0.875195 | 0.672622 | 0.370571 | 0.477868 | 0.486715 | 0.70 | 432 | 1062 | `outputs/baseline_lgbm/metrics_*_selected_threshold.json` |
| P2 Baseline tuned | 432 original | None | Chronological | Optuna 15 | 0.624072 | 0.501438 | 0.880276 | 0.684424 | 0.373031 | 0.482879 | 0.493001 | 0.53 | 432 | 1727 | `outputs/optuna/baseline_lgbm/metrics_*` |
| P3 AE replacement default | 93 non-V + 32 latent | Robust all-class LD32 | Chronological | Default | 0.591398 | 0.481593 | 0.881512 | 0.626176 | 0.376722 | 0.470426 | 0.472018 | 0.65 | 125 | 1459 | `outputs/ae_lgbm/metrics_*` |
| P4 AE replacement tuned | 93 non-V + 128 latent | Robust all-class LD128 | Chronological | Optuna 15 | 0.610631 | 0.490686 | 0.892513 | 0.601197 | 0.395423 | 0.477067 | 0.473170 | 0.28 | 221 | 1262 | LD128 caveat — `outputs/optuna/ae_lgbm_ld128/metrics_*` |
| AE06 Augmentation default | 432 + 128 latent + recon | Robust LD128 | Chronological | Default | 0.598198 | 0.485417 | 0.876654 | 0.648370 | 0.372047 | 0.472795 | 0.478038 | 0.70 | 561 | 1156 | Confounded augmentation — `outputs/ae_augmented_lgbm_ld128/metrics_*` |

Metric source paths use the `metrics_validation_selected_threshold.json` and `metrics_test_selected_threshold.json` pattern cited in `docs/EXPERIMENT_REGISTRY.md`.

## Stopping rule

The allowed unresolved Autoencoder questions have been **executed and closed**:

| Experiment | Output | Validation AP | Delta vs P01 | Status |
|------------|--------|---------------|--------------|--------|
| AE17 clean latent-only augmentation LD128 | `outputs/ae_latent_only_augmented_lgbm_ld128/` | 0.591898 | −0.010535 | Complete |
| AAE01 selected-numerical AE replacement LD128 | `outputs/selected_numerical_ae_lgbm_ld128/` | 0.525103 | −0.077330 | Complete |
| AAE02 selected-numerical reconstructed replacement | `outputs/selected_numerical_reconstructed_lgbm/` | 0.549737 | −0.052696 | Complete |

**AE17 result:** Latent-only augmentation does **not** improve validation AP versus P01. Comparison: `outputs/final_comparison/latent_integration_strategy_comparison.csv`.

**AAE01 result (anchor-alignment diagnostic):** Broadening Autoencoder input from V-only to 387 selected numerical predictors does **not** improve validation AP versus P01 or V-only LD128 replacement. Comparison: `outputs/final_comparison/autoencoder_input_scope_comparison.csv`. Details: `docs/ANCHOR_ALIGNMENT_EXPERIMENT.md`.

**AAE02 result (Ding-alignment diagnostic):** Decoder-reconstructed replacement improves validation AP versus AAE01 latent replacement (+0.024634) but remains below P01 (−0.052696). Comparison: `outputs/final_comparison/autoencoder_output_strategy_comparison.csv`. Details: `docs/DING_RECONSTRUCTION_ALIGNMENT_EXPERIMENT.md`.

**No further experiments** are permitted without explicit supervisor approval. The anchor-alignment experimental phase is closed.

## Final permitted experiment family — causal behavioral + AE signal (executed 2026-06-10)

| Experiment | Output | Validation AP | Delta vs P01 | Status |
|------------|--------|---------------|--------------|--------|
| B2 causal behavioral LightGBM | `outputs/causal_behavioral_lgbm_default/` | 0.613738 | +0.011305 | Complete |
| B3 causal behavioral + CDV recon | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` | 0.600659 | −0.001774 | Complete |

**B2 result (Rule A):** Causal behavioral features improve validation AP versus P01 under chronological evaluation.

**B3 result (Rule D):** CDV reconstruction error does not improve validation AP beyond causal behavioral features (−0.013079 vs B2).

Comparison: `outputs/final_comparison/causal_behavioral_ae_comparison.csv`. Details: `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md`.

**The experimental phase is closed.** No further entity expansions, AE signals, tuning, or ensemble branches.

## Final permitted AE integration experiment — task-aware latent learning (executed 2026-06-10)

| Experiment | Output | Validation AP | Delta vs AAE01 | Status |
|------------|--------|---------------|------------------|--------|
| TAE01 task-aware selected-numerical latent replacement LD128 | `outputs/task_aware_ae_lgbm_ld128/selected/` | 0.524481 | −0.000621 | Complete |

**TAE01 result (Rule C):** Joint reconstruction-classification supervision does **not** improve selected-numerical latent replacement versus unsupervised AAE01 under the executed chronological protocol.

Comparison: `outputs/final_comparison/task_aware_ae_comparison.csv`. Details: `docs/TASK_AWARE_AUTOENCODER_EXPERIMENT.md`.

**TAE01 is the final permitted AE integration experiment.** No attention, VAE, GAN, SMOTE, graph, ensemble, lambda expansion, or architecture changes after TAE01.

## Proposed physical organization

Do **not** execute yet. Future layout proposal:

```text
outputs/
├── primary/                      # P1–P4 chronological core
├── autoencoder_ablations/        # AE04–AE12, augmentation, recon error
├── methodological_diagnostics/   # split_strategy_appendix, bootstrap, audit
└── exploratory_archive/          # FE branches, ensembles, final_report
```

Current paths remain authoritative until a governed migration is approved and documented.

## Smallest defensible final experiment scope

**Minimum scope for thesis defense:**

1. P1 — baseline default
2. P2 — baseline tuned
3. P3 — AE replacement default (thesis-original pipeline)
4. Critical diagnostic table — P1 vs P3 vs AE06 (with augmentation confound noted)
5. MD01–MD06 — split appendix (brief main-text mention, full appendix)

**Optional single ablation:** latent dimension sweep (LD32/64/128 replacement defaults).

**Explicitly excluded from main narrative:** FE branches, score ensembles, UID/velocity, behavioral CDV under FE, `final_report` test-based rankings.

## Immediate next step

**Write final thesis diagnostic text for AE17, AAE01, AAE02, CBA01/CBA02, and TAE01 using validation AP only. The experimental phase is closed. TAE01 is the final permitted AE integration experiment.**