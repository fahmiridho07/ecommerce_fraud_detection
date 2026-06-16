# Experiment Naming Guide

## Purpose

This document defines the **canonical experiment naming system** for thesis writing and final reporting. It maps stable canonical IDs to legacy repository IDs without renaming files, moving outputs, or altering metrics.

**Legacy IDs are preserved for reproducibility** because output directories, scripts, and historical records use them. **Canonical IDs are used for thesis writing and final reporting.**

## Prefix families

| Prefix | Meaning | Examples |
|--------|---------|----------|
| **BASE** | Baseline LightGBM experiments on original features | BASE-01, BASE-02 |
| **AE** | Autoencoder-related representation experiments | AE-01 … AE-07 |
| **BEH** | Behavioral / historical causal feature experiments | BEH-01, BEH-02 |
| **FUS** | Fusion / ensemble experiments | FUS-01 |
| **WIP** | Active work-in-progress branches not yet thesis evidence | WIP-GBDT |
| **APP** | Appendix / diagnostic comparison experiments | APP-01 |
| **LEGACY** | Superseded or provisional historical branches | LEGACY-01 … LEGACY-03 |

## Status definitions

| Status | Role in thesis |
|--------|----------------|
| `active_reference` | Used as baseline/reference in final comparison |
| `active_expert` | Used as a component in the final thesis-candidate method |
| `thesis_candidate` | Candidate final method pending supervisor approval |
| `ablation_evidence` | Not final, but supports scientific reasoning |
| `diagnostic_appendix` | Used to explain protocol sensitivity or validity |
| `active_wip` | In-progress branch isolated from thesis conclusions until reviewed |
| `provisional_superseded` | Old result replaced by corrected rerun |
| `legacy_archived` | Historical branch not part of final argument |

> **High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

## Current thesis-candidate path

```text
BASE-01 / P01  →  BASE-02 / P02
                        ↓
BEH-01 / CBA01R  +  AE-02 / P04  →  FUS-01 / LF01
```

- **BASE-01 / P01** — raw LightGBM default baseline
- **BASE-02 / P02** — tuned raw LightGBM baseline
- **AE-02 / P04** — tuned V-only AE-LightGBM LD128 representation expert
- **BEH-01 / CBA01R** — identity-aligned behavioral LightGBM
- **FUS-01 / LF01** — score-level fusion between CBA01R and P04

Active WIP branches such as **WIP-GBDT / GBDT-\*** are not part of this thesis-candidate path.

## Canonical experiment registry

| Canonical ID | Legacy ID(s) | Experiment name | Short description | Role in thesis | Status | Primary metric | Validation AP | Test AP | Main script | Output path | Documentation path | Notes / caveats |
|--------------|--------------|-----------------|-------------------|----------------|--------|----------------|---------------|---------|-------------|-------------|---------------------|-----------------|
| **BASE-01** | P01 | Raw LightGBM Default | 432 original IEEE-CIS features; default LightGBM; chronological 60/20/20 | Primary chronological reference for all integration comparisons | `active_reference` | `average_precision` (validation-selected threshold) | 0.602433 | 0.485756 | `src/train_baseline_lgbm.py` | `outputs/baseline_lgbm/` | `docs/EXPERIMENT_REGISTRY.md`, `docs/FINAL_EXPERIMENT_PLAN.md` | B1 reference in causal behavioral family |
| **BASE-02** | P02 | Tuned Raw LightGBM | Same 432 original features; Optuna TPE 15 trials | Strongest original-feature tuned benchmark in primary comparison | `active_reference` | `average_precision` | 0.624072 | 0.501438 | `src/tune_lgbm_optuna.py --model_type baseline_lgbm` | `outputs/optuna/baseline_lgbm/` | `docs/EXPERIMENT_REGISTRY.md`, `docs/FINAL_EXPERIMENT_PLAN.md` | Outperforms AE-02 / P04 on both splits |
| **AE-01** | P03 | V-only AE-LightGBM Default LD32 | V-features replaced by LD32 robust AE latent; default LightGBM | Thesis-original AE replacement default; ablation vs BASE-01 | `ablation_evidence` | `average_precision` | 0.591398 | 0.481593 | `src/train_autoencoder_robust.py`, `src/train_ae_lgbm.py` | `outputs/ae_lgbm/`, `outputs/autoencoder_robust/` | `docs/EXPERIMENT_REGISTRY.md`, `docs/FINAL_EXPERIMENT_PLAN.md` | LD32; not the LD128 expert used in FUS-01 |
| **AE-02** | P04 | Tuned V-only AE-LightGBM LD128 | 93 non-V + 128 LD128 latent (V replaced); Optuna tuned | Frozen AE representation expert in FUS-01 / LF01 | `active_expert` | `average_precision` | 0.610631 | 0.490686 | `src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128` | `outputs/optuna/ae_lgbm_ld128/`, `outputs/autoencoder_robust_ld128/` | `docs/EXPERIMENT_REGISTRY.md`, `docs/CAUSAL_BEHAVIORAL_AE_LATE_FUSION_EXPERIMENT.md` | LD128 only; not LD32 tuned parity; Partial completion label in registry |
| **AE-03** | AE17 | Latent-only Augmentation | Original V retained + LD128 latent added; recon error excluded | Fair augmentation test vs replacement | `ablation_evidence` | `average_precision` | 0.591898 | 0.483013 | `src/train_ae_latent_only_augmented_lgbm.py` | `outputs/ae_latent_only_augmented_lgbm_ld128/` | `docs/EXPERIMENT_REGISTRY.md`, `docs/EXPERIMENT_SCOPE_FREEZE.md` | Validation AP −0.010535 vs BASE-01; supersedes confounded legacy AE06 augmentation |
| **AE-04** | AAE01 | Selected-Numerical Latent Replacement | 387 selected numerical predictors → LD128 latent replacement | Anchor-alignment input-scope diagnostic | `ablation_evidence` | `average_precision` | 0.525103 | 0.398658 | `src/train_autoencoder_selected_numerical.py`, `src/train_selected_numerical_ae_lgbm.py` | `outputs/autoencoder_selected_numerical_ld128/`, `outputs/selected_numerical_ae_lgbm_ld128/` | `docs/ANCHOR_ALIGNMENT_EXPERIMENT.md`, `docs/SELECTED_NUMERICAL_AE_FEATURE_AUDIT.md` | Rule C — broadening AE input did not help |
| **AE-05** | AAE02 | Selected-Numerical Reconstruction | Decoder reconstruction replaces latent bottleneck | Ding-alignment output-strategy diagnostic | `ablation_evidence` | `average_precision` | 0.549737 | 0.430796 | `src/generate_selected_numerical_reconstructed_features.py`, `src/train_selected_numerical_reconstructed_lgbm.py` | `outputs/selected_numerical_reconstructed_lgbm/` | `docs/DING_RECONSTRUCTION_ALIGNMENT_EXPERIMENT.md` | Rule B — beats AE-04 on validation but below BASE-01 |
| **AE-06** | AE15, Arm A | FE LightGBM + CDV Reconstruction Error (Exploratory) | EX01-style static FE (87 features) + 432 originals (V retained) + one `cdv_ae_reconstruction_mse` scalar; default LightGBM; **not** score ensemble | Historical FE-space CDV diagnostic; CDV AE artifact source for BEH-02 | `legacy_archived` | `average_precision` | 0.635954 | 0.511667 | `src/train_behavioral_cdv_autoencoder.py`, `src/train_fe_cdv_reconstruction_error_lgbm.py`, `src/compare_behavioral_cdv_ae_experiment.py` | `outputs/behavioral_cdv_ae_experiment/A_fe_lgbm_cdv_reconstruction_mse_default/`, `outputs/behavioral_cdv_ae_experiment/autoencoder_cdv_ld128/`, `outputs/behavioral_cdv_ae_experiment/comparison.csv` | `docs/CAUSAL_BEHAVIORAL_FEATURE_AUDIT.md`, `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md`, `docs/EXPERIMENT_REGISTRY.md` | **Not comparable** to BASE-02, BEH-01, or FUS-01. High validation AP rides on static FE baseline (EX01 val AP 0.627793); still below tuned FE (0.654316) and FE+AE ensemble ref (0.659935). `run_config.json` `stopping_criteria` embeds exploratory FE-ensemble references. Governed BEH-02 test shows CDV recon **hurts** (−0.014515 vs BEH-01). Single model, not latent/recon replacement/score fusion. Chronological split yes. |
| **AE-07** | TAE01 | Task-Aware AE-LightGBM | Joint reconstruction + fraud-classification AE; λ=0.1 selected | Final permitted AE integration diagnostic | `ablation_evidence` | `average_precision` | 0.524481 | 0.407953 | `src/train_task_aware_autoencoder_selected_numerical.py`, `src/train_task_aware_ae_lgbm.py` | `outputs/task_aware_autoencoder_selected_numerical_ld128/`, `outputs/task_aware_ae_lgbm_ld128/selected/` | `docs/TASK_AWARE_AUTOENCODER_EXPERIMENT.md` | Rule C — supervised latent did not beat AE-04 |
| **BEH-01** | CBA01R | Identity-Aligned Behavioral LightGBM | 432 original + 19 past-only causal behavioral features; TransactionID-safe alignment | Primary behavioral expert in FUS-01 / LF01 | `active_expert` | `average_precision` | 0.615122 | 0.493838 | `src/causal_behavioral_features.py`, `src/train_causal_behavioral_lgbm.py --id-aligned` | `outputs/causal_behavioral_lgbm_id_aligned/` | `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md`, `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md` | Rule A — +0.012689 validation AP vs BASE-01; supersedes LEGACY-01 |
| **BEH-02** | CBA02R | Behavioral + CDV Reconstruction Error | BEH-01 features + one ID-aligned `cdv_ae_reconstruction_mse` | Tests whether CDV recon adds value after behavioral context | `ablation_evidence` | `average_precision` | 0.600607 | 0.483831 | `src/train_causal_behavioral_cdv_reconstruction_lgbm.py --id-aligned` | `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/` | `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md`, `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md` | Rule D — −0.014515 vs BEH-01; supersedes LEGACY-02 |
| **FUS-01** | LF01 | Behavioral + AE-LightGBM Late Fusion | Validation-selected 50/50 convex fusion of BEH-01 and AE-02 scores | Conditional thesis-candidate final method | `thesis_candidate` | `average_precision` | 0.629600 | 0.505543 | `src/run_causal_behavioral_ae_late_fusion.py`, `src/audit_causal_behavioral_ae_complementarity.py` | `outputs/causal_behavioral_ae_late_fusion/` | `docs/CAUSAL_BEHAVIORAL_AE_LATE_FUSION_EXPERIMENT.md` | Strong success vs BEH-01 (+0.014478) and BASE-02 (+0.005528); supervisor approval required |
| **WIP-GBDT** | GBDT-* | GBDT Backend Comparison | Raw-feature LightGBM vs XGBoost vs CatBoost shootout; conditional AE3 on winner | Active WIP, isolated from thesis conclusions | `active_wip` | `average_precision` | Incomplete | Incomplete | `src/train_gbdt_baseline.py`, `src/tune_gbdt_baseline.py`, `src/train_gbdt_ae3_integration.py`, `src/build_gbdt_baseline_comparison.py` | `outputs/gbdt_baseline_comparison/` | `docs/GBDT_BASELINE_COMPARISON_PLAN.md`, `docs/EXPERIMENT_REGISTRY.md` | Do not promote until `comparison.csv` and `decision_gate.json` are complete and supervisor-reviewed |
| **APP-01** | MD01–MD06 | Split Strategy Appendix | Chronological vs stratified holdout and stratified 5-fold CV | Protocol sensitivity and evaluation validity | `diagnostic_appendix` | `average_precision` (holdout); OOF AP (CV) | See appendix table below | See appendix table below | `src/compare_split_strategy_appendix.py` | `outputs/split_strategy_appendix/` | `docs/FINAL_EXPERIMENT_PLAN.md`, `docs/EXPERIMENT_SCOPE_FREEZE.md` | Must not select final model; stratified AP much higher than chronological |
| **LEGACY-01** | CBA01, B2 | Provisional Causal Behavioral LightGBM | Legacy B2 before TransactionID alignment correction | Historical record only | `provisional_superseded` | `average_precision` | 0.613738 | 0.495350 | `src/train_causal_behavioral_lgbm.py` | `outputs/causal_behavioral_lgbm_default/` | `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md` | 16,309 within-split TransactionID mismatches; replaced by BEH-01 / CBA01R |
| **LEGACY-02** | CBA02, B3 | Provisional Behavioral + CDV Recon | Legacy B3 before alignment correction | Historical record only | `provisional_superseded` | `average_precision` | 0.600659 | 0.484615 | `src/train_causal_behavioral_cdv_reconstruction_lgbm.py` | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` | `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md` | Replaced by BEH-02 / CBA02R |
| **LEGACY-03** | EX05, EX06, EX07, EX08 | Static FE Score Ensemble Branches | Post-hoc score-level ensembles in FE/exploratory space | Archived exploratory branches | `legacy_archived` | `average_precision` | Varies (see notes) | Varies (see notes) | `src/run_score_ensemble.py`, `src/run_three_model_score_ensemble.py`, `src/run_fe_ae_fine_ensemble.py`, `src/run_fe_ae_score_ensemble.py` | `outputs/score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned/`, `outputs/three_model_score_ensemble/`, `outputs/fe_ae_fine_ensemble/`, `outputs/fe_ae_controlled_experiments/A_score_ensemble_fe_tuned_ae_tuned/` | `docs/EXPERIMENT_REGISTRY.md`, `docs/FINAL_REPORT_GOVERNANCE_NOTE.md` | EX07 test AP 0.5340 (test-peeking risk); not valid for model selection |

### APP-01 / MD01–MD06 appendix metrics (descriptive)

| Legacy ID | Split / scope | Validation AP | Test AP |
|-----------|---------------|---------------|---------|
| MD01 | Baseline chronological holdout | 0.599462 | 0.483872 |
| MD02 | Baseline stratified holdout | 0.819570 | 0.821840 |
| MD03 | FE chronological holdout | 0.627793 | 0.509117 |
| MD04 | FE stratified holdout | 0.847822 | 0.849415 |
| MD05 | Baseline stratified 5-fold CV | OOF 0.843111 | — |
| MD06 | FE stratified 5-fold CV | OOF 0.865834 | — |

Sources: `outputs/split_strategy_appendix/split_strategy_comparison.csv`, `outputs/split_strategy_appendix/stratified_cv_summary.csv`.

### AE-06 / AE15 audit summary (corrected 2026-06-11)

| Property | Verified value |
|----------|----------------|
| What it is | Experiment **Arm A** in `behavioral_cdv_ae_experiment`: FE-LightGBM + one CDV AE reconstruction-error feature |
| Model | Single default LightGBM (520 features after preprocessing) |
| AE signal used | Reconstruction error scalar only (`cdv_ae_reconstruction_mse`); no latent, no decoder reconstruction, no model-score fusion |
| Original V retained | Yes — all 339 V-features kept alongside 432 originals |
| Static FE | Yes — 87 train-fitted count/frequency and amount-stat features (EX01 family) |
| Score ensemble | No |
| Split | Chronological 60/20/20 (`sample_size=null`) |
| Comparable to BASE-02 / P02 | **No** — different feature space (432 original vs 432+87 FE+1 CDV error) |
| Comparable to BEH-01 / CBA01R | **No** — static FE aggregations vs 19 causal behavioral features |
| Comparable to FUS-01 / LF01 | **No** — single FE model vs validation-selected score fusion of governed experts |
| Why high AP does not promote it | Validation AP exceeds FUS-01 because static FE alone reaches 0.627793; governed path shows CDV recon fails after causal behavioral features (BEH-02). Branch excluded from freeze scope. |
| CDV AE reuse | Frozen `autoencoder_cdv_ld128/` artifacts reused by BEH-02 / CBA02R without retraining |

## Dual-notation usage rules

1. **In thesis tables and conclusions**, prefer canonical IDs (e.g., FUS-01) with legacy ID in parentheses on first mention: **FUS-01 / LF01**.
2. **In scripts, paths, and reproduction commands**, keep legacy IDs unchanged.
3. **Do not rename** output directories, metric files, or experiment JSON manifests.
4. **Do not replace** legacy IDs in historical CSV column names or `run_config.json` fields.

## Related documents

- [`docs/ACTIVE_EXPERIMENT_MAP.md`](ACTIVE_EXPERIMENT_MAP.md) — final active thesis path only
- [`docs/ABLATION_EXPERIMENT_MAP.md`](ABLATION_EXPERIMENT_MAP.md) — non-final supporting experiments
- [`docs/EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) — full legacy inventory
- [`docs/EXPERIMENT_SCOPE_FREEZE.md`](EXPERIMENT_SCOPE_FREEZE.md) — frozen narrative scope
