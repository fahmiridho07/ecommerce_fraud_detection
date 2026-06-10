# Result Artifact Manifest

## Purpose

Recommend a **minimal** set of small, thesis-relevant artifacts that may later be tracked in Git for reproducibility and supervisor review. This document does **not** modify `.gitignore` yet.

**Current state:** `outputs/*` is entirely gitignored (`.gitignore` line 27). All artifacts below exist locally but are untracked.

**Exclusions (do not track):** model binaries, `.pkl`, `.keras`, latent `.npy`, `oof_scores.csv`, prediction/score CSVs, Optuna `study.db`, raw data, full `outputs/` trees.

## Primary model artifacts

| Artifact | Purpose | Size (approx.) | Track recommendation | Reason |
|----------|---------|----------------|----------------------|--------|
| `outputs/baseline_lgbm/run_config.json` | P01 protocol record | 2.4 KB | **Yes** | Defines chronological split, features, threshold policy |
| `outputs/baseline_lgbm/metrics_validation_selected_threshold.json` | P01 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/baseline_lgbm/metrics_test_selected_threshold.json` | P01 test AP | 0.4 KB | **Yes** | Frozen final reporting |
| `outputs/optuna/baseline_lgbm/run_config.json` | P02 tuning record | 4.6 KB | **Yes** | Optuna budget, best params context |
| `outputs/optuna/baseline_lgbm/best_params.json` | P02 selected hyperparameters | small | **Yes** | Reproducibility without DB |
| `outputs/optuna/baseline_lgbm/metrics_validation_selected_threshold.json` | P02 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/optuna/baseline_lgbm/metrics_test_selected_threshold.json` | P02 test AP | 0.4 KB | **Yes** | Frozen final reporting |
| `outputs/ae_lgbm/run_config.json` | P03 replacement LD32 record | 2.9 KB | **Yes** | Feature construction proof |
| `outputs/ae_lgbm/metrics_validation_selected_threshold.json` | P03 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/ae_lgbm/metrics_test_selected_threshold.json` | P03 test AP | 0.4 KB | **Yes** | Frozen final reporting |
| `outputs/optuna/ae_lgbm_ld128/run_config.json` | P04 tuned LD128 record | 4.9 KB | **Yes** | Documents LD128 tuned candidate |
| `outputs/optuna/ae_lgbm_ld128/best_params.json` | P04 selected hyperparameters | small | **Yes** | Reproducibility without DB |
| `outputs/optuna/ae_lgbm_ld128/metrics_validation_selected_threshold.json` | P04 validation AP | 0.4 KB | **Yes** | Primary selection metric |
| `outputs/optuna/ae_lgbm_ld128/metrics_test_selected_threshold.json` | P04 test AP | 0.4 KB | **Yes** | Frozen final reporting |

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
| `outputs/causal_behavioral_lgbm_default/run_config.json` | B2 causal behavioral protocol | ~4 KB | **Yes** | B2 feature construction proof |
| `outputs/causal_behavioral_lgbm_default/metrics_validation_selected_threshold.json` | B2 validation AP | 0.4 KB | **Yes** | Primary CBA01 metric |
| `outputs/causal_behavioral_lgbm_default/metrics_test_selected_threshold.json` | B2 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/run_config.json` | B3 protocol (`only_added_feature`) | ~4 KB | **Yes** | B3 isolation proof |
| `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/metrics_validation_selected_threshold.json` | B3 validation AP | 0.4 KB | **Yes** | Primary CBA02 metric |
| `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/metrics_test_selected_threshold.json` | B3 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/source_ae_validation.json` | CDV AE reuse validation | small | **Yes** | `autoencoder_retrained: false` proof |
| `outputs/final_comparison/causal_behavioral_ae_comparison.csv` | B1/B2/B3 controlled comparison | ~1 KB | **Yes** | CBA evidence |
| `outputs/task_aware_autoencoder_selected_numerical_ld128/selected/run_config.json` | TAE01 AE protocol record | ~4 KB | **Yes** | Joint-loss architecture and lambda-selection policy |
| `outputs/task_aware_autoencoder_selected_numerical_ld128/model_selection_summary.csv` | TAE01 lambda ablation summary | ~1 KB | **Yes** | Validation-only downstream AP selection |
| `outputs/task_aware_ae_lgbm_ld128/selected/run_config.json` | TAE01 replacement protocol | ~4 KB | **Yes** | Feature construction proof |
| `outputs/task_aware_ae_lgbm_ld128/selected/metrics_validation_selected_threshold.json` | TAE01 validation AP | 0.4 KB | **Yes** | Primary task-aware metric |
| `outputs/task_aware_ae_lgbm_ld128/selected/metrics_test_selected_threshold.json` | TAE01 test AP | 0.4 KB | **Yes** | Descriptive final evaluation |
| `outputs/final_comparison/task_aware_lambda_selection.csv` | Lambda ablation comparison | ~1 KB | **Yes** | TAE01 lambda-selection evidence |
| `outputs/final_comparison/task_aware_ae_comparison.csv` | P01 vs AAE01 vs TAE01 comparison | ~1 KB | **Yes** | Final task-aware integration evidence |
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