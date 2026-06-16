# Ablation Experiment Map

## Purpose

This document lists experiments that **support scientific reasoning** but are **not part of the final active thesis path**. Use canonical IDs in thesis writing; keep legacy IDs in reproduction commands and output paths.

**Legacy IDs are preserved for reproducibility** because output directories, scripts, and historical records use them. **Canonical IDs are used for thesis writing and final reporting.**

Active path: [`docs/ACTIVE_EXPERIMENT_MAP.md`](ACTIVE_EXPERIMENT_MAP.md)

---

## AE-01 / P03 — V-only AE-LightGBM Default LD32

| Field | Value |
|-------|-------|
| **Why it is not final** | Thesis-original replacement default; superseded as fusion expert by AE-02 / P04 (LD128 tuned) |
| **Insight contributed** | Latent replacement under default params trails BASE-01 on both splits (validation −0.011035 AP) |
| **Bab 4 or appendix** | **Bab 4** — primary AE integration comparison table (P01 vs P03 vs confounded augmentation reference) |
| **Validation AP / Test AP** | 0.591398 / 0.481593 |
| **Output** | `outputs/ae_lgbm/` |

---

## AE-03 / AE17 — Latent-only Augmentation

| Field | Value |
|-------|-------|
| **Why it is not final** | Clean augmentation does not beat BASE-01; feature-level AE integration abandoned |
| **Insight contributed** | Fair test shows retaining V + adding latent does not improve validation AP (−0.010535 vs BASE-01) |
| **Bab 4 or appendix** | **Bab 4** — integration-strategy ablation subsection |
| **Validation AP / Test AP** | 0.591898 / 0.483013 |
| **Output** | `outputs/ae_latent_only_augmented_lgbm_ld128/` |

---

## AE-04 / AAE01 — Selected-Numerical Latent Replacement

| Field | Value |
|-------|-------|
| **Why it is not final** | Anchor-alignment broadening of AE input severely hurt validation AP |
| **Insight contributed** | Rule C — wider numerical scope (387 features) did not help latent replacement |
| **Bab 4 or appendix** | **Bab 4** — anchor-alignment diagnostic paragraph; full detail in **appendix** |
| **Validation AP / Test AP** | 0.525103 / 0.398658 |
| **Output** | `outputs/selected_numerical_ae_lgbm_ld128/` |
| **Documentation** | `docs/ANCHOR_ALIGNMENT_EXPERIMENT.md` |

---

## AE-05 / AAE02 — Selected-Numerical Reconstruction

| Field | Value |
|-------|-------|
| **Why it is not final** | Decoder reconstruction beats AE-04 but still below BASE-01 |
| **Insight contributed** | Rule B — reconstruction preserves more information than latent bottleneck (+0.024634 vs AE-04) |
| **Bab 4 or appendix** | **Bab 4** — output-strategy comparison; detail in **appendix** |
| **Validation AP / Test AP** | 0.549737 / 0.430796 |
| **Output** | `outputs/selected_numerical_reconstructed_lgbm/` |
| **Documentation** | `docs/DING_RECONSTRUCTION_ALIGNMENT_EXPERIMENT.md` |

---

## AE-07 / TAE01 — Task-Aware AE-LightGBM

| Field | Value |
|-------|-------|
| **Why it is not final** | Final permitted AE integration experiment; supervised latent did not beat AE-04 |
| **Insight contributed** | Rule C — joint fraud supervision (λ=0.1) does not improve downstream LightGBM (−0.000621 vs AE-04) |
| **Bab 4 or appendix** | **Bab 4** — closes AE integration search space; lambda ablation in **appendix** |
| **Validation AP / Test AP** | 0.524481 / 0.407953 |
| **Output** | `outputs/task_aware_ae_lgbm_ld128/selected/` |
| **Documentation** | `docs/TASK_AWARE_AUTOENCODER_EXPERIMENT.md` |

---

## BEH-02 / CBA02R — Behavioral + CDV Reconstruction Error

| Field | Value |
|-------|-------|
| **Why it is not final** | Adding CDV recon error after BEH-01 **degrades** validation AP |
| **Insight contributed** | Rule D — feature-level CDV signal is not complementary after causal behavioral context (−0.014515 vs BEH-01); motivates decision-level fusion instead |
| **Bab 4 or appendix** | **Bab 4** — behavioral family results table (B1/B2/B3 corrected) |
| **Validation AP / Test AP** | 0.600607 / 0.483831 |
| **Output** | `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/` |
| **Documentation** | `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md` |

---

## APP-01 / MD01–MD06 — Split Strategy Appendix

| Field | Value |
|-------|-------|
| **Why it is not final** | Diagnostic only; stratified splits yield much higher AP and must not select models |
| **Insight contributed** | Supports chronological protocol choice; pattern consistent with temporal shift / prevalence differences |
| **Bab 4 or appendix** | **Bab 4** — one paragraph summary; **appendix** — full tables |
| **Key finding** | Chronological baseline test AP ≈ 0.484 vs stratified holdout ≈ 0.822 |
| **Output** | `outputs/split_strategy_appendix/` |
| **Documentation** | `docs/FINAL_EXPERIMENT_PLAN.md` (methodological appendix section) |

---

## LEGACY-01 / CBA01 — Provisional Causal Behavioral LightGBM

| Field | Value |
|-------|-------|
| **Why it is not final** | Superseded by BEH-01 / CBA01R after alignment audit (16,309 TransactionID mismatches) |
| **Insight contributed** | Confirms behavioral direction was positive even before correction (+0.011305 vs BASE-01) |
| **Bab 4 or appendix** | **Appendix** — alignment correction audit only; do not cite as authoritative |
| **Validation AP / Test AP** | 0.613738 / 0.495350 |
| **Output** | `outputs/causal_behavioral_lgbm_default/` |
| **Documentation** | `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md` |

---

## LEGACY-02 / CBA02 — Provisional Behavioral + CDV Recon

| Field | Value |
|-------|-------|
| **Why it is not final** | Superseded by BEH-02 / CBA02R; same alignment risk as LEGACY-01 |
| **Insight contributed** | CDV degradation signal persisted after correction (−0.014515 vs BEH-01) |
| **Bab 4 or appendix** | **Appendix** — alignment correction audit only |
| **Validation AP / Test AP** | 0.600659 / 0.484615 |
| **Output** | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` |
| **Documentation** | `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md` |

---

## AE-06 / AE15 — FE LightGBM + CDV Reconstruction Error (moved to legacy)

> **Reclassified:** `legacy_archived` — not normal AE ablation evidence. See [`docs/EXPERIMENT_NAMING_GUIDE.md`](EXPERIMENT_NAMING_GUIDE.md) AE-06 audit summary.

| Field | Value |
|-------|-------|
| **What it actually is** | Experiment Arm A: EX01-style static FE LightGBM (87 engineered features + 432 originals, V retained) plus one `cdv_ae_reconstruction_mse` scalar from CDV AE (368 C+D+V inputs). Single default LightGBM — **not** score ensemble, **not** latent replacement. |
| **Why it is not final** | Exploratory FE-space branch with static train-fitted aggregations; excluded from freeze scope; `run_config.json` `stopping_criteria` references FE+AE ensemble test scores |
| **Why high AP does not promote it** | Validation AP 0.635954 exceeds FUS-01 (0.629600) only because static FE baseline (EX01) is already 0.627793 (+0.008160 delta). Still below tuned FE (0.654316) and FE+AE ensemble ref (0.659935). Governed BEH-02 shows CDV recon degrades causal behavioral model. |
| **Insight contributed** | CDV AE training artifacts are reusable; weak additive signal under FE; negative result on governed path motivates decision-level fusion instead of feature-level CDV injection |
| **Comparable to active path?** | **No** — not comparable to BASE-02 / P02, BEH-01 / CBA01R, or FUS-01 / LF01 |
| **Bab 4 or appendix** | **Appendix only** — CDV AE artifact lineage; cite BEH-02 for governed CDV conclusion |
| **Validation AP / Test AP** | 0.635954 / 0.511667 |
| **Scripts** | `src/train_behavioral_cdv_autoencoder.py`, `src/train_fe_cdv_reconstruction_error_lgbm.py`, `src/compare_behavioral_cdv_ae_experiment.py` |
| **Output** | `outputs/behavioral_cdv_ae_experiment/A_fe_lgbm_cdv_reconstruction_mse_default/`, `comparison.csv` |
| **Documentation** | `docs/CAUSAL_BEHAVIORAL_FEATURE_AUDIT.md`, `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md` |

**High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

---

## LEGACY-03 / EX05–EX08 — Static FE Score Ensemble Branches

| Field | Value |
|-------|-------|
| **Why it is not final** | Exploratory FE-space ensembles; test-peeking risk (EX07); not AE integration on original features |
| **Insight contributed** | Historical hint that score-level combination can help — later formalized in controlled FUS-01 / LF01 |
| **Bab 4 or appendix** | **Appendix** or omit; cite only as methodological limitation if mentioned |
| **Representative test AP** | EX05: 0.508124; EX07: 0.5340; EX08: 0.533935 |
| **Outputs** | `outputs/score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned/`, `outputs/fe_ae_fine_ensemble/`, `outputs/fe_ae_controlled_experiments/A_score_ensemble_fe_tuned_ae_tuned/` |
| **Documentation** | `docs/FINAL_REPORT_GOVERNANCE_NOTE.md`, `docs/EXPERIMENT_REGISTRY.md` |

---

## Related documents

- [`docs/EXPERIMENT_NAMING_GUIDE.md`](EXPERIMENT_NAMING_GUIDE.md)
- [`docs/EXPERIMENT_SCOPE_FREEZE.md`](EXPERIMENT_SCOPE_FREEZE.md)