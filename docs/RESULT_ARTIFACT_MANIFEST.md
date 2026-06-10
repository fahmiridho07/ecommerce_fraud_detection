# Result Artifact Manifest

## Purpose

Recommend a **minimal** set of small, thesis-relevant artifacts that may later be tracked in Git for reproducibility and supervisor review. This document does **not** modify `.gitignore` yet.

## Canonical naming

Artifacts below are keyed by legacy output paths. Canonical experiment IDs for thesis writing: **BASE-01 / P01**, **BASE-02 / P02**, **AE-02 / P04**, **BEH-01 / CBA01R**, **FUS-01 / LF01**.

**Legacy IDs are preserved for reproducibility** because output directories, scripts, and historical records use them. **Canonical IDs are used for thesis writing and final reporting.**

See [`docs/EXPERIMENT_NAMING_GUIDE.md`](EXPERIMENT_NAMING_GUIDE.md).

**Current state:** `outputs/*` is entirely gitignored (`.gitignore` line 27). All artifacts below exist locally but are untracked.

**Exclusions (do not track):** model binaries, `.pkl`, `.keras`, latent `.npy`, `oof_scores.csv`, prediction/score CSVs, Optuna `study.db`, raw data, full `outputs/` trees.

## Primary model artifacts

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/baseline_lgbm/run_config.json` | BASE-01 / P01 protocol record | 2.4 KB | **Yes** | Defines chronological split, features, threshold policy |
| `outputs/baseline_lgbm/metrics_validation_selected_threshold.json` | BASE-01 / P01 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/baseline_lgbm/metrics_test_selected_threshold.json` | BASE-01 / P01 test AP | 0.4 KB | **Yes** | Frozen final reporting |
| `outputs/optuna/baseline_lgbm/run_config.json` | BASE-02 / P02 tuning record | 4.6 KB | **Yes** | Optuna budget, best params context |
| `outputs/optuna/baseline_lgbm/best_params.json` | BASE-02 / P02 selected hyperparameters | small | **Yes** | Reproducibility without DB |
| `outputs/optuna/baseline_lgbm/metrics_validation_selected_threshold.json` | BASE-02 / P02 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/optuna/baseline_lgbm/metrics_test_selected_threshold.json` | BASE-02 / P02 test AP | 0.4 KB | **Yes** | Frozen final reporting |
| `outputs/ae_lgbm/run_config.json` | AE-01 / P03 replacement LD32 record | 2.9 KB | **Yes** | Feature construction proof |
| `outputs/ae_lgbm/metrics_validation_selected_threshold.json` | AE-01 / P03 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/ae_lgbm/metrics_test_selected_threshold.json` | AE-01 / P03 test AP | 0.4 KB | **Yes** | Frozen final reporting |
| `outputs/optuna/ae_lgbm_ld128/run_config.json` | AE-02 / P04 tuned LD128 record | 4.9 KB | **Yes** | Documents LD128 tuned candidate |
| `outputs/optuna/ae_lgbm_ld128/best_params.json` | AE-02 / P04 selected hyperparameters | small | **Yes** | Reproducibility without DB |
| `outputs/optuna/ae_lgbm_ld128/metrics_validation_selected_threshold.json` | AE-02 / P04 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/optuna/ae_lgbm_ld128/metrics_test_selected_threshold.json` | AE-02 / P04 test AP | 0.4 KB | **Yes** | Frozen final reporting |

## Comparison and ablation summaries

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/final_comparison/latent_integration_strategy_comparison.csv` | Clean integration-strategy comparison (P01, replacement, AE17, confounded AE06) | ~1 KB | **Yes** | Primary AE17 evidence; validation-based interpretation |
| `outputs/final_comparison/latent_dim_ablation.csv` | LD32/64/128 replacement comparison | 0.7 KB | **Yes** | Small ablation summary; documents dimension sweep |
| `outputs/final_comparison/ae_augmented_comparison.csv` | Replacement vs augmentation table | 0.9 KB | **Yes** | Key integration-design evidence |
| `outputs/final_comparison/optuna_comparison.csv` | Default vs tuned summary | 1.5 KB | **Yes** | Primary tuning comparison |
| `outputs/final_comparison/next_controlled_experiments.csv` | Exploratory ranking CSV | 3.8 KB | **Optional** | Useful archive; includes non-primary models |
| `outputs/final_comparison/fe_ae_fine_ensemble_comparison.csv` | Ensemble exploration | 1.7 KB | **No** | Exploratory archive only |
| `outputs/final_comparison/three_model_ensemble_comparison.csv` | Ensemble exploration | 1.1 KB | **No** | Exploratory archive only |

## Methodological appendix artifacts

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/split_strategy_appendix/appendix_summary.json` | Appendix orchestration metadata | 0.9 KB | **Yes** | Documents appendix scope and seeds |
| `outputs/split_strategy_appendix/split_strategy_comparison.csv` | Holdout strategy comparison | 1.9 KB | **Yes** | Core appendix evidence |
| `outputs/split_strategy_appendix/stratified_cv_summary.csv` | CV OOF summary | 0.8 KB | **Yes** | Core appendix evidence |
| `outputs/split_strategy_appendix/holdout/*/run_config.json` | Per-run appendix configs | ~2–3 KB each | **Optional** | Detailed reproducibility; CSV may suffice |
| `outputs/split_strategy_appendix/stratified_cv/*/cv_summary.json` | Per-model CV stats | small | **Optional** | Redundant with `stratified_cv_summary.csv` |
| `outputs/split_strategy_appendix/stratified_cv/*/fold_metrics.csv` | Per-fold metrics | larger | **No** | Too granular for Git summary layer |
| `outputs/split_strategy_appendix/stratified_cv/*/oof_scores.csv` | OOF predictions | large | **No** | Prediction file; excluded by policy |

## Global split reference

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/split_summary.json` | Chronological split statistics | small | **Yes** | Shared split evidence for all primary runs |

## Reporting metadata (governance-sensitive)

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/final_report/final_summary.json` | Historical report summary | 2.9 KB | **No** (or track with warning) | Contains test-based `best_overall_model`; see `docs/FINAL_REPORT_GOVERNANCE_NOTE.md` |
| `outputs/final_report/final_model_comparison.csv` | Descriptive test ranking table | small | **Optional** | Only if regenerated with governance labels |

## Autoencoder diagnostic metrics (optional track)

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/ae_latent_only_augmented_lgbm_ld128/run_config.json` | AE17 clean augmentation protocol | ~3 KB | **Yes** | Documents latent-only design (`reconstruction_error_included: false`) |
| `outputs/ae_latent_only_augmented_lgbm_ld128/metrics_validation_selected_threshold.json` | AE17 validation AP | 0.4 KB | **Yes** | Primary integration diagnostic metric |
| `outputs/ae_latent_only_augmented_lgbm_ld128/metrics_test_selected_threshold.json` | AE17 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/selected_numerical_ae_feature_audit/selected_numerical_features.json` | AAE01 feature eligibility audit | ~21 KB | **Yes** | Documents AE input scope before training |
| `outputs/autoencoder_selected_numerical_ld128/run_config.json` | AAE01 AE protocol record | ~3 KB | **Yes** | Anchor-alignment AE preprocessing and architecture |
| `outputs/autoencoder_selected_numerical_ld128/reconstruction_metrics.json` | AAE01 AE quality diagnostic | small | **Optional** | AE training evidence |
| `outputs/selected_numerical_ae_lgbm_ld128/run_config.json` | AAE01 replacement protocol | ~3 KB | **Yes** | Feature construction proof |
| `outputs/selected_numerical_ae_lgbm_ld128/metrics_validation_selected_threshold.json` | AAE01 validation AP | 0.4 KB | **Yes** | Primary anchor-alignment metric |
| `outputs/selected_numerical_ae_lgbm_ld128/metrics_test_selected_threshold.json` | AAE01 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/final_comparison/autoencoder_input_scope_comparison.csv` | P01 vs V-only vs selected-numerical comparison | ~1 KB | **Yes** | Anchor-alignment evidence |
| `outputs/autoencoder_selected_numerical_ld128/reconstructed_feature_names.json` | AAE02 reconstructed feature names | small | **Yes** | Documents decoder output naming |
| `outputs/selected_numerical_reconstructed_lgbm/run_config.json` | AAE02 reconstructed replacement protocol | ~3 KB | **Yes** | `autoencoder_retrained: false` proof |
| `outputs/selected_numerical_reconstructed_lgbm/metrics_validation_selected_threshold.json` | AAE02 validation AP | 0.4 KB | **Yes** | Primary Ding-alignment metric |
| `outputs/selected_numerical_reconstructed_lgbm/metrics_test_selected_threshold.json` | AAE02 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/final_comparison/autoencoder_output_strategy_comparison.csv` | P01 vs latent vs reconstructed comparison | ~1 KB | **Yes** | Ding-alignment evidence |
| `outputs/causal_behavioral_feature_audit/feature_definition.json` | CBA causal behavioral feature policy | ~3 KB | **Yes** | Documents entity keys, windows, causal policy |
| `outputs/causal_behavioral_lgbm_default/run_config.json` | B2 legacy protocol | ~4 KB | **Optional** | Provisional archive only |
| `outputs/causal_behavioral_lgbm_default/metrics_validation_selected_threshold.json` | B2 legacy validation AP | 0.4 KB | **Optional** | Superseded by CBA01R |
| `outputs/causal_behavioral_lgbm_id_aligned/run_config.json` | CBA01R corrected protocol | ~5 KB | **Yes** | Identity-safe B2 proof |
| `outputs/causal_behavioral_lgbm_id_aligned/alignment_validation.json` | CBA01R ID join validation | small | **Yes** | TransactionID alignment evidence |
| `outputs/causal_behavioral_lgbm_id_aligned/metrics_validation_selected_threshold.json` | BEH-01 / CBA01R validation AP | 0.4 KB | **Yes** | **Authoritative BEH-01 / CBA01R metric** |
| `outputs/causal_behavioral_lgbm_id_aligned/metrics_test_selected_threshold.json` | CBA01R test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/run_config.json` | CBA02R corrected protocol | ~5 KB | **Yes** | Identity-safe B3 proof |
| `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/metrics_validation_selected_threshold.json` | CBA02R validation AP | 0.4 KB | **Yes** | **Authoritative CBA02R metric** |
| `outputs/causal_behavioral_alignment_audit/pre_fix_alignment_report.json` | Alignment risk audit | ~15 KB | **Yes** | Documents 16,309 legacy mismatches |
| `outputs/final_comparison/causal_behavioral_alignment_correction.csv` | Corrected CBA comparison | ~1 KB | **Yes** | Authoritative CBA evidence |
| `results/causal_behavioral_alignment_correction.csv` | Trackable summary CSV | ~1 KB | **Yes** | Git-tracked mirror of comparison |
| `results/causal_behavioral_alignment_manifest.json` | Trackable result manifest | small | **Yes** | Git-tracked summary metadata |
| `outputs/final_comparison/causal_behavioral_ae_comparison.csv` | Legacy B1/B2/B3 comparison | ~1 KB | **Optional** | Historical archive only |
| `outputs/task_aware_autoencoder_selected_numerical_ld128/selected/run_config.json` | TAE01 AE protocol record | ~4 KB | **Yes** | Joint-loss architecture and lambda-selection policy |
| `outputs/task_aware_autoencoder_selected_numerical_ld128/model_selection_summary.csv` | TAE01 lambda ablation summary | ~1 KB | **Yes** | Validation-only downstream AP selection |
| `outputs/task_aware_ae_lgbm_ld128/selected/run_config.json` | TAE01 replacement protocol | ~4 KB | **Yes** | Feature construction proof |
| `outputs/task_aware_ae_lgbm_ld128/selected/metrics_validation_selected_threshold.json` | TAE01 validation AP | 0.4 KB | **Yes** | Primary task-aware metric |
| `outputs/task_aware_ae_lgbm_ld128/selected/metrics_test_selected_threshold.json` | TAE01 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/final_comparison/task_aware_lambda_selection.csv` | Lambda ablation comparison | ~1 KB | **Yes** | TAE01 lambda-selection evidence |
| `outputs/final_comparison/task_aware_ae_comparison.csv` | P01 vs AAE01 vs TAE01 comparison | ~1 KB | **Yes** | Final task-aware integration evidence |
| `outputs/causal_behavioral_ae_late_fusion/frozen_fusion_config.json` | LF01 frozen fusion weights and threshold | ~1 KB | **Yes** | Decision-level integration proof |
| `outputs/causal_behavioral_ae_late_fusion/run_config.json` | LF01 protocol record | ~4 KB | **Yes** | Identity-safe score fusion lineage |
| `outputs/causal_behavioral_ae_late_fusion/metrics_validation_selected_threshold.json` | FUS-01 / LF01 validation AP | 0.4 KB | **Yes** | Primary FUS-01 / LF01 metric |
| `outputs/causal_behavioral_ae_late_fusion/metrics_test_selected_threshold.json` | FUS-01 / LF01 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/causal_behavioral_ae_late_fusion/complementarity_summary.json` | Validation complementarity audit | ~2 KB | **Yes** | Expert complementarity evidence |
| `outputs/causal_behavioral_ae_late_fusion/paired_bootstrap_summary.json` | Paired AP-delta bootstrap | ~2 KB | **Yes** | Uncertainty quantification |
| `outputs/final_comparison/causal_behavioral_ae_late_fusion_weight_search.csv` | Predefined weight grid results | ~1 KB | **Yes** | Validation-only weight selection |
| `outputs/final_comparison/causal_behavioral_ae_late_fusion_comparison.csv` | P01–P04 + CBA01R + LF01 table | ~2 KB | **Yes** | Controlled comparison evidence |
| `results/causal_behavioral_ae_late_fusion.csv` | Trackable LF01 summary | ~1 KB | **Yes** | Git-tracked thesis-facing metrics |
| `results/causal_behavioral_ae_late_fusion_manifest.json` | LF01 result manifest | ~2 KB | **Yes** | Git-tracked artifact lineage |
| `results/late_fusion_complementarity_summary.json` | Thesis-safe complementarity summary | ~1 KB | **Yes** | No transaction-level records |
| `outputs/final_comparison/causal_behavioral_feature_importance.csv` | B2/B3 importance by feature group | ~15 KB | **Optional** | Supports importance narrative |
| `outputs/ae_augmented_lgbm_ld128/run_config.json` | Confounded augmentation design record | ~3 KB | **Optional** | Documents V+latent+recon confound |
| `outputs/ae_augmented_lgbm_ld128/metrics_validation_selected_threshold.json` | AE06 validation AP | 0.4 KB | **Optional** | Supports integration diagnostic |
| `outputs/baseline_lgbm_plus_ae_reconstruction_mse/metrics_*_selected_threshold.json` | AE08 recon-only metrics | 0.4 KB each | **Optional** | One variant sufficient (skip log1p duplicate) |
| `outputs/autoencoder_robust/reconstruction_metrics.json` | AE training diagnostic | small | **Optional** | AE quality evidence; not LGBM selection |

## Proposed future `.gitignore` allow-list pattern (not applied yet)

When ready, a narrow allow-list could look like:

```gitignore
outputs/*
!outputs/split_summary.json
!outputs/final_comparison/*.csv
!outputs/selected_numerical_ae_feature_audit/selected_numerical_features.json
!outputs/split_strategy_appendix/appendix_summary.json
!outputs/split_strategy_appendix/split_strategy_comparison.csv
!outputs/split_strategy_appendix/stratified_cv_summary.csv
!outputs/baseline_lgbm/run_config.json
!outputs/baseline_lgbm/metrics_*_selected_threshold.json
# ... explicit per-file rules for P02–P04 only
```

**Safety rule:** allow-list by explicit file path, never `!outputs/` recursively. Model binaries and pickles must remain ignored.

## Estimated tracked footprint

If **Yes** and primary **Optional** configs/metrics only: approximately **30–45 KB** of JSON/CSV — suitable for Git without bloating the repository.

## Related documents

- [`docs/EXPERIMENT_SCOPE_FREEZE.md`](EXPERIMENT_SCOPE_FREEZE.md) — which artifacts matter for thesis narrative
- [`docs/FINAL_REPORT_GOVERNANCE_NOTE.md`](FINAL_REPORT_GOVERNANCE_NOTE.md) — reporting metadata caution