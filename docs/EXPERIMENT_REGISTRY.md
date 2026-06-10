# Experiment Registry

## Purpose

This registry is the authoritative map of experiment status and thesis relevance for the `ecommerce_fraud_detection` repository. Every status claim is tied to locally inspected source scripts, `src/config.py` entries, and `outputs/` artifacts. Config presence alone does not imply execution. Output directory presence alone does not imply completeness.

The registry supports documentation-first experiment governance. It does not modify code, move outputs, or select a final thesis model from historical test scores.

## Evidence and status definitions

| Term | Definition |
|------|------------|
| **Configured** | A path, constant, or `OUTPUT_PATHS` registry entry exists in `src/config.py`. |
| **Implemented** | A runnable source script or function exists under `src/`. |
| **Executed** | Output artifacts or metric JSON/CSV files exist under `outputs/`. |
| **Complete** | Source-output mapping is verified; `run_config.json` (or equivalent), validation metrics, test metrics, and model/preprocessing artifacts are present and consistent. |

### Completion labels

| Label | Meaning |
|-------|---------|
| **Complete** | Executed with verified source-output lineage and sufficient artifacts for reproduction audit. |
| **Results-only** | Metrics and summary files exist, but one or more expected model/preprocessing artifacts are missing or use nonstandard names. |
| **Partial** | Executed, but missing artifacts, protocol mismatch, or incomplete comparison fairness. |
| **Code-only** | Script exists; no verified output directory or metrics. |
| **Config-only** | Configured output path exists in code; local output directory not found. |
| **Unverified** | Evidence is ambiguous or producer mapping is unclear. |

### Metric terminology

- **Validation AP** and **Test AP** refer to `average_precision` from `sklearn.metrics.average_precision_score`, stored in `metrics_validation_selected_threshold.json` and `metrics_test_selected_threshold.json` (see `src/evaluation.py`).
- **OOF AP** refers to out-of-fold average precision from stratified CV appendix artifacts (`cv_summary.json`, `stratified_cv_summary.csv`).
- README and some CSV columns use **PR-AUC** as shorthand for the same implemented metric; this registry uses **AP** unless citing a file that literally says `pr_auc`.

## Master experiment inventory

| ID | Experiment | Category | Configured | Implemented | Executed | Completion status | Source script | Config path | Output path | Key artifacts |
|----|------------|----------|------------|-------------|----------|-------------------|---------------|-------------|-------------|---------------|
| P01 | Baseline LightGBM default | Primary | Yes | Yes | Yes | Complete | `src/train_baseline_lgbm.py` | `BASELINE_OUTPUT_DIR` | `outputs/baseline_lgbm/` | `run_config.json`, `metrics_validation_selected_threshold.json`, `metrics_test_selected_threshold.json`, `model.pkl`, `preprocessing.pkl` |
| P02 | Baseline LightGBM Optuna tuned | Primary | Yes | Yes | Yes | Complete | `src/tune_lgbm_optuna.py` | `OPTUNA_OUTPUT_DIR/baseline_lgbm` | `outputs/optuna/baseline_lgbm/` | `run_config.json`, `best_params.json`, `final_model.pkl`, `metrics_*_selected_threshold.json`, `trials.csv` |
| P03 | AE-LightGBM latent replacement default (LD32) | Primary | Yes | Yes | Yes | Complete | `src/train_ae_lgbm.py` | `AE_LGBM_OUTPUT_DIR` | `outputs/ae_lgbm/` | `run_config.json`, `metrics_*_selected_threshold.json`, `model.pkl`, `preprocessing_non_v.pkl`, `comparison_against_baseline.json` |
| P04 | AE-LightGBM latent replacement Optuna tuned (LD128) | Primary | Yes | Yes | Yes | Partial | `src/tune_lgbm_optuna.py` | `OPTUNA_OUTPUT_DIR/ae_lgbm_ld128` | `outputs/optuna/ae_lgbm_ld128/` | `run_config.json`, `best_params.json`, `final_model.pkl`, `metrics_*_selected_threshold.json` |
| AE01 | Robust Autoencoder LD32 (all-class) | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_autoencoder_robust.py` | `AUTOENCODER_ROBUST_OUTPUT_DIR` | `outputs/autoencoder_robust/` | `run_config.json`, `reconstruction_metrics.json`, `encoder_model.keras`, `v_scaler.pkl`, latent `.npy` files |
| AE02 | Robust Autoencoder LD64 | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_autoencoder_robust.py --latent-dim 64` | `AUTOENCODER_ROBUST_LD64_OUTPUT_DIR` | `outputs/autoencoder_robust_ld64/` | `run_config.json`, `reconstruction_metrics.json`, encoder/scaler artifacts |
| AE03 | Robust Autoencoder LD128 | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_autoencoder_robust.py --latent-dim 128` | `AUTOENCODER_ROBUST_LD128_OUTPUT_DIR` | `outputs/autoencoder_robust_ld128/` | `run_config.json`, `reconstruction_metrics.json`, encoder/scaler artifacts |
| AE04 | AE-LightGBM replacement LD64 | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_ae_lgbm.py` | `AE_LGBM_LD64_OUTPUT_DIR` | `outputs/ae_lgbm_ld64/` | `run_config.json`, `metrics_*_selected_threshold.json`, `model.pkl` |
| AE05 | AE-LightGBM replacement LD128 default | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_ae_lgbm.py` | `AE_LGBM_LD128_OUTPUT_DIR` | `outputs/ae_lgbm_ld128/` | `run_config.json`, `metrics_*_selected_threshold.json`, `model.pkl` |
| AE06 | AE augmentation LD128 (V retained + latent + recon error) | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_ae_augmented_lgbm.py` | `AE_AUGMENTED_LGBM_LD128_OUTPUT_DIR` | `outputs/ae_augmented_lgbm_ld128/` | `run_config.json`, `feature_set_summary.json`, `metrics_*_selected_threshold.json` |
| AE17 | Clean latent-only augmentation LD128 | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_ae_latent_only_augmented_lgbm.py` | `AE_LATENT_ONLY_AUGMENTED_LGBM_LD128_OUTPUT_DIR` | `outputs/ae_latent_only_augmented_lgbm_ld128/` | `run_config.json`, `metrics_*_selected_threshold.json`, `model.pkl`, `preprocessing.pkl` |
| AAE01 | Selected-numerical AE replacement LD128 | AE diagnostic (anchor alignment) | Yes | Yes | Yes | Complete | `src/train_autoencoder_selected_numerical.py`, `src/train_selected_numerical_ae_lgbm.py` | `AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR`, `SELECTED_NUMERICAL_AE_LGBM_LD128_OUTPUT_DIR` | `outputs/autoencoder_selected_numerical_ld128/`, `outputs/selected_numerical_ae_lgbm_ld128/` | `run_config.json`, `reconstruction_metrics.json`, `metrics_*_selected_threshold.json`, `model.pkl`, `preprocessing_non_ae_features.pkl` |
| AAE02 | Selected-numerical reconstructed replacement | AE diagnostic (Ding alignment) | Yes | Yes | Yes | Complete | `src/generate_selected_numerical_reconstructed_features.py`, `src/train_selected_numerical_reconstructed_lgbm.py` | `SELECTED_NUMERICAL_RECONSTRUCTED_LGBM_OUTPUT_DIR` | `outputs/selected_numerical_reconstructed_lgbm/` | `reconstructed_*.npy`, `run_config.json`, `metrics_*_selected_threshold.json`, `model.pkl`, `preprocessing_retained_features.pkl` |
| TAE01 | Task-aware selected-numerical latent replacement LD128 | AE diagnostic (final integration) | Yes | Yes | Yes | Complete | `src/train_task_aware_autoencoder_selected_numerical.py`, `src/train_task_aware_ae_lgbm.py` | `TASK_AWARE_AUTOENCODER_SELECTED_NUMERICAL_LD128_OUTPUT_DIR`, `TASK_AWARE_AE_LGBM_LD128_OUTPUT_DIR` | `outputs/task_aware_autoencoder_selected_numerical_ld128/`, `outputs/task_aware_ae_lgbm_ld128/selected/` | `run_config.json`, `model_selection_summary.csv`, `metrics_*_selected_threshold.json`, `model.pkl`, `preprocessing_retained_features.pkl` |
| AE07 | AE augmentation LD128 Optuna tuned | AE diagnostic | Yes | Yes | Yes | Complete | `src/tune_lgbm_optuna.py` | `OPTUNA_OUTPUT_DIR/ae_augmented_lgbm_ld128` | `outputs/optuna/ae_augmented_lgbm_ld128/` | `run_config.json`, `best_params.json`, `final_model.pkl`, `metrics_*_selected_threshold.json` |
| AE08 | Reconstruction error only — robust raw MSE | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_reconstruction_error_lgbm.py` | `RECON_ERROR_LGBM_ROBUST_RAW_OUTPUT_DIR` | `outputs/baseline_lgbm_plus_ae_reconstruction_mse/` | `run_config.json`, `metrics_*_selected_threshold.json`, `feature_set_summary.json` |
| AE09 | Reconstruction error only — robust log1p MSE | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_reconstruction_error_lgbm.py` | `RECON_ERROR_LGBM_ROBUST_LOG1P_OUTPUT_DIR` | `outputs/baseline_lgbm_plus_log1p_ae_reconstruction_mse/` | `run_config.json`, `metrics_*_selected_threshold.json` |
| AE10 | Reconstruction error only — robust raw+log1p | AE diagnostic | Yes | Yes | No | Config-only | `src/train_reconstruction_error_lgbm.py` | `RECON_ERROR_LGBM_ROBUST_RAW_LOG1P_OUTPUT_DIR` | `outputs/baseline_lgbm_plus_raw_log1p_ae_reconstruction_mse/` | Not found locally |
| AE11 | Normal-only Autoencoder LD128 | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_autoencoder_normal_only.py` | `AUTOENCODER_NORMAL_ONLY_LD128_OUTPUT_DIR` | `outputs/autoencoder_normal_only_ld128/` | `run_config.json`, `reconstruction_metrics.json`, encoder/scaler artifacts |
| AE12 | Reconstruction error only — normal-only raw MSE | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_reconstruction_error_lgbm.py` | `RECON_ERROR_LGBM_NORMAL_ONLY_RAW_OUTPUT_DIR` | `outputs/baseline_lgbm_plus_normal_only_ae_reconstruction_mse/` | `run_config.json`, `metrics_*_selected_threshold.json` |
| AE13 | Reconstruction error only — normal-only log1p | AE diagnostic | Yes | Yes | No | Config-only | `src/train_reconstruction_error_lgbm.py` | `RECON_ERROR_LGBM_NORMAL_ONLY_LOG1P_OUTPUT_DIR` | `outputs/baseline_lgbm_plus_log1p_normal_only_ae_reconstruction_mse/` | Not found locally |
| AE14 | Reconstruction error only — normal-only raw+log1p | AE diagnostic | Yes | Yes | No | Config-only | `src/train_reconstruction_error_lgbm.py` | `RECON_ERROR_LGBM_NORMAL_ONLY_RAW_LOG1P_OUTPUT_DIR` | `outputs/baseline_lgbm_plus_raw_log1p_normal_only_ae_reconstruction_mse/` | Not found locally |
| AE15 | Behavioral/CDV Autoencoder + FE recon LGBM | AE diagnostic | Yes | Yes | Yes | Complete | `src/train_behavioral_cdv_autoencoder.py`, `src/train_fe_cdv_reconstruction_error_lgbm.py`, `src/compare_behavioral_cdv_ae_experiment.py` | `BEHAVIORAL_CDV_AE_EXPERIMENT_OUTPUT_DIR` | `outputs/behavioral_cdv_ae_experiment/` | `comparison.csv`, `autoencoder_cdv_ld128/run_config.json`, `A_fe_lgbm_cdv_reconstruction_mse_default/metrics_*` |
| CBA01 | Causal behavioral LightGBM default (B2) | Literature-motivated diagnostic | Yes | Yes | Yes | Complete | `src/causal_behavioral_features.py`, `src/train_causal_behavioral_lgbm.py` | `CAUSAL_BEHAVIORAL_LGBM_OUTPUT_DIR` | `outputs/causal_behavioral_lgbm_default/` | `run_config.json`, `metrics_*_selected_threshold.json`, `behavioral_feature_importance.csv`, `feature_definition.json` |
| CBA02 | Causal behavioral + CDV recon error (B3) | Literature-motivated diagnostic | Yes | Yes | Yes | Complete | `src/train_causal_behavioral_cdv_reconstruction_lgbm.py` | `CAUSAL_BEHAVIORAL_CDV_RECONSTRUCTION_LGBM_OUTPUT_DIR` | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` | `run_config.json`, `metrics_*_selected_threshold.json`, `cdv_feature_importance.csv`, `source_ae_validation.json` |
| AE16 | Legacy non-robust Autoencoder LD32 | AE diagnostic | Yes | Yes | Yes | Complete (superseded) | `src/train_autoencoder.py` | `AUTOENCODER_OUTPUT_DIR` | `outputs/autoencoder/` | `run_config.json`, `reconstruction_metrics.json` |
| MD01 | Split appendix — baseline chronological holdout | Methodological | Yes | Yes | Yes | Complete | `src/compare_split_strategy_appendix.py` | `SPLIT_STRATEGY_APPENDIX_OUTPUT_DIR` | `outputs/split_strategy_appendix/holdout/baseline_lgbm/chronological/` | `run_config.json`, `metrics_test_selected_threshold.json`, `split_summary.json` |
| MD02 | Split appendix — baseline stratified holdout | Methodological | Yes | Yes | Yes | Complete | `src/compare_split_strategy_appendix.py` | same | `outputs/split_strategy_appendix/holdout/baseline_lgbm/stratified_holdout/` | `run_config.json`, `metrics_test_selected_threshold.json` |
| MD03 | Split appendix — FE chronological holdout | Methodological | Yes | Yes | Yes | Complete | `src/compare_split_strategy_appendix.py` | same | `outputs/split_strategy_appendix/holdout/feature_engineered_lgbm/chronological/` | `run_config.json`, `metrics_test_selected_threshold.json` |
| MD04 | Split appendix — FE stratified holdout | Methodological | Yes | Yes | Yes | Complete | `src/compare_split_strategy_appendix.py` | same | `outputs/split_strategy_appendix/holdout/feature_engineered_lgbm/stratified_holdout/` | `run_config.json`, `metrics_test_selected_threshold.json` |
| MD05 | Split appendix — baseline stratified 5-fold CV | Methodological | Yes | Yes | Yes | Complete | `src/compare_split_strategy_appendix.py` | same | `outputs/split_strategy_appendix/stratified_cv/baseline_lgbm/` | `run_config.json`, `cv_summary.json`, `fold_metrics.csv`, `oof_scores.csv` |
| MD06 | Split appendix — FE stratified 5-fold CV | Methodological | Yes | Yes | Yes | Complete | `src/compare_split_strategy_appendix.py` | same | `outputs/split_strategy_appendix/stratified_cv/feature_engineered_lgbm/` | `run_config.json`, `cv_summary.json`, `fold_metrics.csv` |
| MD07 | Autoencoder audit diagnostics | Methodological | No | Yes | Yes | Results-only | `src/generate_diagnostic_analysis.py` | — | `outputs/autoencoder_audit/` | `diagnosis_summary.json`, distribution CSVs |
| MD08 | Bootstrap / business-impact diagnostics | Methodological | No | Yes | Yes | Results-only | `src/generate_final_defense_diagnostics.py`, `src/generate_business_impact_diagnostics.py` | — | `outputs/final_diagnostics/` | `bootstrap_delta_summary.json`, `bootstrap_pr_auc_ci.csv` |
| EX01 | Entity/time/amount FE LightGBM default | Exploratory | Yes | Yes | Yes | Complete | `src/train_feature_engineered_lgbm.py` | `FEATURE_ENGINEERED_LGBM_OUTPUT_DIR` | `outputs/baseline_lgbm_entity_time_amount_features/` | `run_config.json`, `metrics_*_selected_threshold.json` |
| EX02 | Entity/time/amount FE LightGBM Optuna tuned | Exploratory | Yes | Yes | Yes | Complete | `src/tune_lgbm_optuna.py` | `OPTUNA_OUTPUT_DIR/baseline_lgbm_entity_time_amount_features` | `outputs/optuna/baseline_lgbm_entity_time_amount_features/` | `run_config.json`, `best_params.json`, `metrics_*` |
| EX03 | FE + UID features LightGBM default | Exploratory | Yes | Yes | Yes | Complete | `src/train_uid_feature_engineered_lgbm.py` | `UID_FEATURE_ENGINEERED_LGBM_OUTPUT_DIR` | `outputs/baseline_lgbm_entity_time_amount_uid_features/` | `run_config.json`, `metrics_*` |
| EX04 | FE + historical velocity LightGBM default | Exploratory | Yes | Yes | Yes | Complete | `src/train_historical_velocity_lgbm.py` | `HISTORICAL_VELOCITY_LGBM_OUTPUT_DIR` | `outputs/baseline_lgbm_entity_time_amount_historical_velocity_features/` | `run_config.json`, `metrics_*` |
| EX05 | Score ensemble baseline tuned + AE tuned | Exploratory | Yes | Yes | Yes | Complete | `src/run_score_ensemble.py` | `SCORE_ENSEMBLE_TUNED_OUTPUT_DIR` | `outputs/score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned/` | `run_config.json`, `weight_selection.csv`, `metrics_*` |
| EX06 | Three-model score ensemble | Exploratory | No | Yes | Yes | Complete | `src/run_three_model_score_ensemble.py` | — | `outputs/three_model_score_ensemble/` | `run_config.json`, `metrics_test_selected_threshold.json` |
| EX07 | FE+AE fine score ensemble | Exploratory | No | Yes | Yes | Complete | `src/run_fe_ae_fine_ensemble.py` | — | `outputs/fe_ae_fine_ensemble/` | `run_config.json`, `metrics_test_probability.json`, `metrics_test_rank.json` |
| EX08 | Controlled FE+AE score ensemble | Exploratory | Yes | Yes | Yes | Complete | `src/run_fe_ae_score_ensemble.py` | `FE_AE_SCORE_ENSEMBLE_OUTPUT_DIR` | `outputs/fe_ae_controlled_experiments/A_score_ensemble_fe_tuned_ae_tuned/` | `run_config.json`, `comparison.csv` (parent), `metrics_*` |
| EX09 | Controlled FE recon error LGBM | Exploratory | Yes | Yes | Yes | Complete | `src/train_fe_reconstruction_error_lgbm.py` | `FE_RECON_ERROR_LGBM_OUTPUT_DIR` | `outputs/fe_ae_controlled_experiments/B_fe_lgbm_reconstruction_mse_default/` | `run_config.json`, `metrics_*` |
| EX10 | Controlled FE latent128 + recon LGBM | Exploratory | Yes | Yes | Yes | Complete | `src/train_fe_ae_augmented_lgbm.py` | `FE_AE_AUGMENTED_LGBM_OUTPUT_DIR` | `outputs/fe_ae_controlled_experiments/C_fe_lgbm_latent128_reconstruction_mse_default/` | `run_config.json`, `metrics_*` |
| EX11 | Optuna FE extended smoke test | Exploratory | No | Yes | Yes | Partial | `src/compare_extended_optuna.py` | — | `outputs/_smoke_fe_extended/` | `run_config.json`, `best_params.json`, `trials.csv`; no final holdout metrics |
| EX12 | Final report asset generator outputs | Exploratory | No | Yes | Yes | Results-only | `src/generate_final_report_assets.py` | — | `outputs/final_report/` | `final_summary.json`, `final_model_comparison.csv` |
| U01 | AE-LightGBM LD32 Optuna tuned | Unverified | No | Yes (`--model_type ae_lgbm`) | No | Config-only / missing | `src/tune_lgbm_optuna.py` | — | `outputs/optuna/ae_lgbm/` (not found) | Only `ae_lgbm_ld128` tuned output exists |
| U02 | Latent-dim ablation aggregator | Unverified | Yes | Yes | Yes | Results-only | `src/run_latent_dim_ablation.py` | `LATENT_DIM_ABLATION_FILE` | `outputs/final_comparison/latent_dim_ablation.csv` | Aggregated CSV only |
| U03 | Appendix vs main baseline metric gap | Unverified | — | — | Yes | Ambiguous | `src/compare_split_strategy_appendix.py` vs `src/train_baseline_lgbm.py` | — | see MD01 vs P01 | Chronological test AP differs slightly between pipelines |

## Primary thesis experiments

| ID | Model role | Feature setup | Original V retained | Latent used | AE regime | Latent dimension | LightGBM tuning | Validation AP | Test AP | Metric sources | Comparability notes | Status |
|----|------------|---------------|---------------------|-------------|-----------|------------------|-----------------|---------------|---------|----------------|---------------------|--------|
| P01 | original-feature LightGBM default | 432 original tabular features incl. raw V and TransactionDT | Yes | No | — | — | Default params | 0.602433 | 0.485756 | `outputs/baseline_lgbm/metrics_validation_selected_threshold.json`, `outputs/baseline_lgbm/metrics_test_selected_threshold.json` | Chronological 60/20/20; full data (`sample_size=null`); MCC threshold on validation | Complete |
| P02 | original-feature LightGBM tuned | Same 432 original features | Yes | No | — | — | Optuna TPE, 15 trials, `tuning_profile=final` | 0.624072 | 0.501438 | `outputs/optuna/baseline_lgbm/metrics_validation_selected_threshold.json`, `outputs/optuna/baseline_lgbm/metrics_test_selected_threshold.json` | Same split protocol as P01; comparable tuning budget to P04 | Complete |
| P03 | AE-LightGBM replacement default | 93 non-V + 32 latent (V replaced) | No | Yes | All-class robust AE | 32 | Default params | 0.591398 | 0.481593 | `outputs/ae_lgbm/metrics_validation_selected_threshold.json`, `outputs/ae_lgbm/metrics_test_selected_threshold.json` | Uses `outputs/autoencoder_robust/`; directly comparable to P01 on split and sample mode | Complete |
| P04 | AE-LightGBM replacement tuned | 93 non-V + 128 latent (V replaced) | No | Yes | All-class robust AE | 128 | Optuna TPE, 15 trials | 0.610631 | 0.490686 | `outputs/optuna/ae_lgbm_ld128/metrics_validation_selected_threshold.json`, `outputs/optuna/ae_lgbm_ld128/metrics_test_selected_threshold.json` | **Not directly comparable to P03**: different latent dimension (128 vs 32). No executed `ae_lgbm` LD32 tuned run found. | Partial — requires rerun for strict LD32 tuned role |

### Primary role gaps

| Role | Status |
|------|--------|
| original-feature LightGBM default | **Mapped (P01), Complete** |
| original-feature LightGBM tuned | **Mapped (P02), Complete** |
| AE-LightGBM replacement default | **Mapped (P03), Complete** |
| AE-LightGBM replacement tuned | **Partial (P04)**: executed only for LD128, not thesis-original LD32 |

## Autoencoder diagnostics

| ID | Research question | Experiment | Comparison target | AE input | AE regime | Latent dimension | Original V retained | Reconstruction error | Validation AP | Test AP | Evidence paths | Conservative interpretation | Confidence |
|----|-------------------|------------|-------------------|----------|-----------|------------------|----------------------|------------------------|---------------|---------|----------------|----------------------------|------------|
| AE04 | Does latent dimension affect replacement performance? | AE-LightGBM LD64 replacement | P03 (LD32), AE05 (LD128) | V-features | All-class robust | 64 | No | No | 0.587878 | 0.481166 | `outputs/ae_lgbm_ld64/metrics_*_selected_threshold.json` | LD64 did not improve test AP vs LD32/LD128 under default params | High |
| AE05 | Does larger latent help replacement? | AE-LightGBM LD128 default | P03 | V-features | All-class robust | 128 | No | No | 0.594149 | 0.489417 | `outputs/ae_lgbm_ld128/metrics_*_selected_threshold.json` | LD128 default slightly above LD32 on test AP, still below P01/P02 | High |
| AE06 | Replacement vs augmentation? | AE augmentation LD128 | P01, AE05 | V-features | All-class robust | 128 | Yes | Yes (`ae_reconstruction_mse`) | 0.598198 | 0.485417 | `outputs/ae_augmented_lgbm_ld128/metrics_*_selected_threshold.json` | Confounded: V + latent + recon error; superseded by AE17 for fair augmentation test | Medium |
| AE17 | Does clean latent-only augmentation help? | Latent-only augmentation LD128 | P01, AE05 | V-features | All-class robust LD128 | 128 | Yes | No | 0.591898 | 0.483013 | `outputs/ae_latent_only_augmented_lgbm_ld128/metrics_*_selected_threshold.json` | Validation AP below P01 (−0.010535); does not support complementary latent utility under fair comparison | High |
| AAE01 | Does broadening AE input to selected numerical predictors help replacement? | Selected-numerical AE replacement LD128 | P01, AE05 | 387 selected numerical (V + amt/C/D/dist/id) | Selected-numerical all-class LD128 | 128 | No | No | 0.525103 | 0.398658 | `outputs/selected_numerical_ae_lgbm_ld128/metrics_*_selected_threshold.json` | Validation AP below P01 and V-only LD128; input-scope broadening did not improve replacement under executed protocol | High |
| TAE01 | Does joint fraud supervision improve selected-numerical latent replacement? | Task-aware selected-numerical AE replacement LD128 | P01, AAE01 | 387 selected numerical | Joint reconstruction + classification LD128 (λ=0.1 selected) | 128 | No | No | 0.524481 | 0.407953 | `outputs/task_aware_ae_lgbm_ld128/selected/metrics_*_selected_threshold.json` | Validation AP below AAE01 (−0.000621) and P01; Rule C — supervised latent learning did not improve downstream LightGBM | High |
| AAE02 | Does decoder reconstruction outperform latent replacement? | Selected-numerical reconstructed replacement | P01, AAE01 | 387 selected numerical | Frozen selected-numerical AE (no retrain) | 387 recon | No | No | 0.549737 | 0.430796 | `outputs/selected_numerical_reconstructed_lgbm/metrics_*_selected_threshold.json` | Validation AP above AAE01 (+0.024634) but below P01; Rule B | High |
| AE07 | Does tuning help augmentation? | AE augmentation LD128 tuned | AE06 | V-features | All-class robust | 128 | Yes | Yes | 0.612999 | 0.483597 | `outputs/optuna/ae_augmented_lgbm_ld128/metrics_*_selected_threshold.json` | Tuned augmentation validation AP rises, but test AP remains below P02 | High |
| AE08 | Is reconstruction error alone useful? | Baseline + raw recon MSE | P01 | V-features | All-class robust LD128 | 128 (AE only) | Yes | Yes (raw) | 0.612429 | 0.496067 | `outputs/baseline_lgbm_plus_ae_reconstruction_mse/metrics_*_selected_threshold.json` | Validation AP > P01; test AP still < P02. Does not establish AE superiority for thesis main claim | High |
| AE09 | Does log1p transform change recon-error utility? | Baseline + log1p recon MSE | AE08 | V-features | All-class robust LD128 | 128 (AE only) | Yes | Yes (log1p) | 0.612429 | 0.496067 | `outputs/baseline_lgbm_plus_log1p_ae_reconstruction_mse/metrics_*_selected_threshold.json` | Identical validation/test AP to AE08 in saved artifacts | High |
| AE12 | Does normal-only AE recon error differ from all-class? | Baseline + normal-only raw recon | AE08 | V-features | Normal-only | 128 (AE only) | Yes | Yes (raw) | 0.609163 | 0.487441 | `outputs/baseline_lgbm_plus_normal_only_ae_reconstruction_mse/metrics_*_selected_threshold.json` | Normal-only AE recon did not outperform all-class recon on test AP | High |
| AE15 | Do behavioral/CDV AE features help under FE? | FE-LGBM + CDV recon error | EX01 | CDV behavioral subset | CDV AE | 128 | Yes (FE + originals) | Yes | 0.635954 | 0.511667 | `outputs/behavioral_cdv_ae_experiment/A_fe_lgbm_cdv_reconstruction_mse_default/metrics_*`, `comparison.csv` | Weak additive signal under FE; not a replacement for main AE thesis pipeline | Medium |
| CBA01 | Do causal behavioral features improve P01? | Causal behavioral B2 | P01 | Original + 19 causal behavioral | — | — | Yes | No | 0.613738 | 0.495350 | `outputs/causal_behavioral_lgbm_default/metrics_*_selected_threshold.json` | Validation AP +0.011305 vs P01; Rule A | High |
| CBA02 | Does CDV recon error help after causal behavioral context? | B2 + CDV recon B3 | CBA01 | B2 + `cdv_ae_reconstruction_mse` | CDV AE (frozen) | 128 (AE only) | Yes | Yes (one feature) | 0.600659 | 0.484615 | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/metrics_*` | Validation AP −0.013079 vs B2; Rule D; high recon gain but no AP gain | High |
| AE16 | Is robust AE necessary? | Legacy non-robust AE | AE01 | V-features | Legacy | 32 | — | — | Not available | Not available | `outputs/autoencoder/run_config.json`, `reconstruction_metrics.json` | Superseded by robust AE; no downstream LGBM replacement artifact tied to legacy AE in main path | Low |

Aggregated ablation evidence: `outputs/final_comparison/latent_dim_ablation.csv`, `outputs/final_comparison/ae_augmented_comparison.csv`, `outputs/final_comparison/optuna_comparison.csv`.

## Methodological diagnostics

| ID | Diagnostic | Models evaluated | Evaluation strategies | Metric scope | Verified result | Evidence paths | Thesis role | Limitations |
|----|------------|------------------|----------------------|--------------|-----------------|----------------|-------------|-------------|
| MD01 | Chronological vs stratified holdout — baseline | Baseline LightGBM default recipe | Chronological holdout | Validation + test AP | Chronological: val AP 0.599462, test AP 0.483872; Stratified holdout: val AP 0.819570, test AP 0.821840 | `outputs/split_strategy_appendix/split_strategy_comparison.csv`, holdout `run_config.json` files | Supports keeping chronological evaluation as main protocol | Appendix chronological baseline test AP differs slightly from P01 (0.485756); not causal proof of shift |
| MD02 | Chronological vs stratified holdout — FE | FE LightGBM default recipe | Chronological vs stratified holdout | Validation + test AP | Chronological: val AP 0.627793, test AP 0.509117; Stratified: val AP 0.847822, test AP 0.849415 | same CSV | Appendix sensitivity only; must not select final model | FE features fitted within train partition per `compare_split_strategy_appendix.py` |
| MD03 | Stratified 5-fold CV benchmark | Baseline and FE LightGBM | Stratified K-fold OOF | OOF AP + fold means | Baseline OOF AP 0.843111; FE OOF AP 0.865834 | `outputs/split_strategy_appendix/stratified_cv_summary.csv`, `cv_summary.json`, `fold_metrics.csv` | Non-temporal benchmark; each fold validation used for early stopping | CV shuffles time; optimistic vs chronological test |
| MD04 | Appendix orchestration metadata | Baseline + FE | All appendix modes | Configuration | `sample_size=null`, `n_folds=5`, `random_seed=42` | `outputs/split_strategy_appendix/appendix_summary.json` | Documents appendix scope | `skip_existing=true` may affect reruns |

### Split-strategy appendix policy (verified)

- **Chronological holdout** is the main thesis evaluation protocol (`outputs/split_summary.json`, all primary `run_config.json` files).
- **Stratified holdout** is a non-temporal sensitivity analysis (`temporal_order_preserved=False` in `split_strategy_comparison.csv`).
- **Stratified K-fold CV** is an OOF non-temporal benchmark (`evaluation_scope=out_of_fold`).
- Preprocessing and feature engineering are fitted within training partitions per appendix `run_config.json` leakage notes.
- Each CV validation fold is also used for early stopping (`src/compare_split_strategy_appendix.py` docstring).
- Findings are **consistent with** temporal distribution shift and higher distribution similarity under stratified splitting, but do **not** isolate causality.
- AP gaps may also be influenced by fraud prevalence differences across chronological splits (see `train_fraud_rate`, `test_fraud_rate` in `split_strategy_comparison.csv`).
- Raw `TransactionDT` remains a model feature in baseline runs and may influence chronological vs stratified gaps.
- **Appendix results must not select the final thesis model.**

## Exploratory experiments

| ID | Experiment | Original purpose | Source | Output | Completion | Verified result | Why excluded from main narrative | Reusable insight |
|----|------------|------------------|--------|--------|------------|-----------------|-------------------------------|------------------|
| EX01 | FE LightGBM default | Improve tabular baseline with entity/time/amount features | `src/train_feature_engineered_lgbm.py` | `outputs/baseline_lgbm_entity_time_amount_features/` | Complete | Test AP 0.509117 (`metrics_test_selected_threshold.json`) | Shifts thesis from AE-on-original-features question | FE improves chronological test AP over P01 |
| EX02 | FE LightGBM tuned | Tune FE baseline | `src/tune_lgbm_optuna.py` | `outputs/optuna/baseline_lgbm_entity_time_amount_features/` | Complete | Test AP 0.529857 | Best standalone test AP in `next_controlled_experiments.csv`; not the AE integration question | Strong tuned FE benchmark |
| EX03 | UID FE branch | Add UID aggregations | `src/train_uid_feature_engineered_lgbm.py` | `outputs/baseline_lgbm_entity_time_amount_uid_features/` | Complete | Test AP 0.507254 | Parallel FE branch | UID gains modest vs EX01 |
| EX04 | Historical velocity FE | Add velocity history | `src/train_historical_velocity_lgbm.py` | `outputs/baseline_lgbm_entity_time_amount_historical_velocity_features/` | Complete | Test AP 0.502726 | Parallel FE branch | Velocity did not beat EX01 on test AP |
| EX05 | Baseline+AE score ensemble | Combine tuned models at score level | `src/run_score_ensemble.py` | `outputs/score_ensemble_baseline_tuned_ae_lgbm_ld128_tuned/` | Complete | Test AP 0.508124 | Ensemble is not AE feature integration | Score combination ≠ representation integration |
| EX06 | Three-model ensemble | Combine multiple tuned models | `src/run_three_model_score_ensemble.py` | `outputs/three_model_score_ensemble/` | Complete | Test AP in `final_comparison/three_model_ensemble_comparison.csv` | Post-hoc ensemble exploration | Useful only as archive |
| EX07 | FE+AE fine ensemble | Fine-grained weight search | `src/run_fe_ae_fine_ensemble.py` | `outputs/fe_ae_fine_ensemble/` | Complete | Test AP probability 0.5340 (`metrics_test_probability.json`) | Highest test AP in some summaries; invalid for frozen thesis selection | Documents test-peeking risk |
| EX08 | Controlled FE+AE ensemble | Controlled FE+AE comparison arm A | `src/run_fe_ae_score_ensemble.py` | `outputs/fe_ae_controlled_experiments/A_score_ensemble_fe_tuned_ae_tuned/` | Complete | Test AP 0.533935 (`comparison.csv`) | FE-space experiment, not original-feature AE question | Best test AP in controlled CSV |
| EX09 | Controlled FE recon | Controlled arm B | `src/train_fe_reconstruction_error_lgbm.py` | `outputs/fe_ae_controlled_experiments/B_fe_lgbm_reconstruction_mse_default/` | Complete | Test AP 0.509774 | FE branch diagnostic | Recon under FE ≈ EX01 |
| EX10 | Controlled FE latent+recon | Controlled arm C | `src/train_fe_ae_augmented_lgbm.py` | `outputs/fe_ae_controlled_experiments/C_fe_lgbm_latent128_reconstruction_mse_default/` | Complete | Test AP 0.502211 | FE branch diagnostic | Latent+recon under FE did not help |
| EX11 | Optuna FE extended smoke | Extended search smoke test | `src/compare_extended_optuna.py` | `outputs/_smoke_fe_extended/` | Partial | No holdout metrics | Debugging only | Not thesis evidence |
| EX12 | Final report assets | Notebook/report aggregation | `src/generate_final_report_assets.py` | `outputs/final_report/final_summary.json` | Results-only | Ranks FE ensemble highest on **test** AP | Report generator reflects exploratory ranking, not governance policy | Shows historical test-inspection bias |

## Unverified and inconsistent items

| Issue | Exact paths | Notes |
|-------|-------------|-------|
| Config-only recon raw+log1p (robust) | `src/config.py` → `outputs/baseline_lgbm_plus_raw_log1p_ae_reconstruction_mse/` | Directory missing locally |
| Config-only recon log1p (normal-only) | `src/config.py` → `outputs/baseline_lgbm_plus_log1p_normal_only_ae_reconstruction_mse/` | Directory missing locally |
| Config-only recon raw+log1p (normal-only) | `src/config.py` → `outputs/baseline_lgbm_plus_raw_log1p_normal_only_ae_reconstruction_mse/` | Directory missing locally |
| Missing tuned AE LD32 run | `src/tune_lgbm_optuna.py` supports `ae_lgbm`; no `outputs/optuna/ae_lgbm/` | Only `outputs/optuna/ae_lgbm_ld128/` executed |
| Model filename inconsistency | `outputs/optuna/baseline_lgbm/final_model.pkl` (not `model.pkl`) | Still reproducible; naming differs from default trainers |
| Appendix vs main baseline gap | P01 test AP 0.485756 vs MD01 test AP 0.483872 | Different `best_iteration` / threshold; same chronological protocol claimed |
| Identical raw vs log1p recon metrics | `outputs/baseline_lgbm_plus_ae_reconstruction_mse/metrics_test_selected_threshold.json` and `outputs/baseline_lgbm_plus_log1p_ae_reconstruction_mse/metrics_test_selected_threshold.json` | Both report test AP 0.496067 |
| Stale README statements | `README.md` lines 23–24, 105–109 | Says "Optuna final tuning: running" and "Final comparison summary: pending"; contradicted by completed `outputs/optuna/` and `outputs/final_comparison/` |
| README structure outdated | `README.md` Project Structure | Omits most `src/train_*.py`, compare scripts, and `docs/` |
| Duplicate experiment naming | `ae_lgbm` vs `ae_lgbm_ld32` in `latent_dim_ablation.csv` | Same output dir `outputs/ae_lgbm/` |
| Local outputs not tracked by Git | `.gitignore` lines 27–28 (`outputs/*`) | 0 tracked files under `outputs/` (`git ls-files outputs/` empty) |
| Partial appendix artifact naming | Appendix holdout dirs lack `metrics_validation_selected_threshold.json` in some trees | Test metrics present; validation metrics available via `split_strategy_comparison.csv` |
| `final_summary.json` test-based ranking | `outputs/final_report/final_summary.json` | Declares FE+AE ensemble "best overall" using test AP — governance violation if used for model selection |
| Smoke output in production tree | `outputs/_smoke_fe_extended/` | Incomplete experiment artifacts |

## Comparability matrix

| Pair | Classification | Reason |
|------|----------------|--------|
| P01 baseline default vs P03 AE replacement default | **Directly comparable** | Same chronological split, full data, default LightGBM params, same AP implementation; only feature integration differs (verified in `run_config.json`) |
| P02 baseline tuned vs P04 AE replacement tuned | **Comparable with caveat** | Same Optuna budget (15 trials) and split, but P04 uses LD128 AE while thesis-original replacement uses LD32 (P03) |
| P03 AE replacement default vs AE06 augmentation default | **Comparable with caveat** | Same chronological protocol and default LGBM params, but AE06 retains all original V **and** adds latent **and** recon error (561 features) |
| P03 vs AE05 (LD128 replacement default) | **Comparable with caveat** | Same replacement design, different latent dimension |
| Latent dimension variants (LD32/64/128) | **Directly comparable** | All replacement design, default params, verified in `latent_dim_ablation.csv` |
| P01 vs MD01 appendix chronological baseline | **Comparable with caveat** | Same stated protocol; small metric differences suggest separate training runs or recipe drift |
| EX01 FE chronological vs MD03 appendix FE chronological | **Directly comparable** | Test AP 0.509117 matches across `metrics_test_selected_threshold.json` and `split_strategy_comparison.csv` |
| P01 vs EX02 FE tuned | **Not directly comparable** | Different feature engineering and tuning; FE tuned exceeds baseline on test AP but answers a different research question |
| Chronological primary results vs stratified appendix | **Not directly comparable** | Different split semantics (`temporal_order_preserved` false for stratified) |
| Any primary model vs EX07/EX08 ensembles | **Not directly comparable** | Score-level ensembles with different model sets and weight selection |

## Current defensible findings

### Model performance (chronological, verified AP)

- **P01 baseline default**: validation AP 0.602433, test AP 0.485756.
- **P02 baseline tuned**: validation AP 0.624072, test AP 0.501438 — improves over P01 on both splits.
- **P03 AE replacement LD32 default**: validation AP 0.591398, test AP 0.481593 — below P01 on both splits.
- **P04 AE replacement LD128 tuned**: validation AP 0.610631, test AP 0.490686 — below P02 on both splits.
- Under default replacement design, **no executed AE-LightGBM variant beats the baseline default on chronological test AP** (`latent_dim_ablation.csv`, primary metrics JSON files).

### Autoencoder integration

- **Latent replacement underperformed the original-feature baseline** in the executed thesis pipeline (P01 vs P03).
- **LD128 default replacement** slightly improves test AP over LD32 (0.489417 vs 0.481593) but still trails baseline default.
- **Augmentation (AE06)** did not demonstrate clear benefit over baseline on test AP; the experiment confounds latent addition with reconstruction-error addition.
- **Reconstruction-error-only features (AE08)** improved validation AP over P01 but did not outperform tuned baseline P02 on test AP.
- **Normal-only AE recon (AE12)** did not outperform all-class recon error on test AP.

### Split strategy

- Stratified holdout and CV yield **much higher AP** than chronological holdout for the same recipe (`split_strategy_comparison.csv`, `stratified_cv_summary.csv`).
- This pattern is **consistent with** temporal distribution shift and/or prevalence/feature-distribution differences, but causality is **not isolated**.

### P04 LD128 selection evidence (Phase 1)

| Evidence source | Finding | Confidence |
|-----------------|---------|------------|
| `src/tune_lgbm_optuna.py` header | Optuna implements `ae_lgbm_ld128` only; no `ae_lgbm` LD32 tuned path | High |
| `src/run_latent_dim_ablation.py` | Phase 4B ablation extended to LD64/LD128; ranking printed by **test** AP | High |
| `outputs/final_comparison/latent_dim_ablation.csv` | LD128 replacement has highest **test** AP among replacement variants (0.489417) | High |
| `outputs/ae_lgbm_ld128/metrics_validation_selected_threshold.json` | LD128 default validation AP 0.594149 > LD32 0.591398 > LD64 0.587878 | Medium |
| Any artifact with explicit validation-selection rule for LD128 | **Not found** | — |

**Conclusion:** P04 uses LD128 because tuning was implemented for the Phase 4B LD128 branch, with supporting but informal evidence that LD128 default had the highest validation AP among replacement defaults. Do **not** claim a formal validation-based selection protocol was documented at tuning time.

### Anchor-alignment selected-numerical AE (AAE01 — executed 2026-06-10)

| Criterion | Status |
|-----------|--------|
| Broader numerical AE input with latent replacement | **Executed** |
| Scripts | `src/train_autoencoder_selected_numerical.py`, `src/train_selected_numerical_ae_lgbm.py` |
| Feature audit | `docs/SELECTED_NUMERICAL_AE_FEATURE_AUDIT.md` |
| AE output | `outputs/autoencoder_selected_numerical_ld128/` |
| LGBM output | `outputs/selected_numerical_ae_lgbm_ld128/` |
| Validation AP | 0.525103 |
| Delta vs P01 validation AP | −0.077330 |
| Delta vs AE05 validation AP | −0.069046 |
| Interpretation | Rule C — broadening AE input did not improve replacement |

Comparison: `outputs/final_comparison/autoencoder_input_scope_comparison.csv`. See `docs/ANCHOR_ALIGNMENT_EXPERIMENT.md`.

### Ding-alignment reconstructed replacement (AAE02 — executed 2026-06-10)

| Criterion | Status |
|-----------|--------|
| Frozen AE weights; decoder reconstruction used | **Executed** |
| Scripts | `src/generate_selected_numerical_reconstructed_features.py`, `src/train_selected_numerical_reconstructed_lgbm.py` |
| Output | `outputs/selected_numerical_reconstructed_lgbm/` |
| `autoencoder_retrained` in run_config | `false` (verified) |
| Validation AP | 0.549737 |
| Delta vs P01 validation AP | −0.052696 |
| Delta vs AAE01 validation AP | +0.024634 |
| Interpretation | Rule B — reconstruction beats latent replacement, not P01 |

Comparison: `outputs/final_comparison/autoencoder_output_strategy_comparison.csv`. See `docs/DING_RECONSTRUCTION_ALIGNMENT_EXPERIMENT.md`.

### Clean latent-only augmentation (AE17 — executed 2026-06-10)

| Criterion | Status |
|-----------|--------|
| Original V retained + latent added + recon error excluded | **Executed** |
| Script | `src/train_ae_latent_only_augmented_lgbm.py` |
| Output | `outputs/ae_latent_only_augmented_lgbm_ld128/` |
| `reconstruction_error_included` in run_config | `false` (verified) |
| Validation AP | 0.591898 (`metrics_validation_selected_threshold.json`) |
| Delta vs P01 validation AP | −0.010535 |

AE06 remains **confounded** (V + latent + recon). AE17 is the fair latent-only augmentation diagnostic.

### Unresolved questions

- Whether **fair LD32 tuned AE replacement** would close the gap to P02 (no executed run).
- Why appendix chronological baseline test AP differs slightly from P01 despite the same stated split policy.

## Thesis-facing recommendation

### Retain as primary

- P01, P02, P03 (minimum defensible core)
- P04 only with explicit LD128 caveat, or mark **requires rerun** for LD32 tuned parity

### Retain as Autoencoder ablations

- AE04, AE05 (latent dimension)
- AE06, AE07 (augmentation vs replacement family)
- AE08, AE09, AE12 (reconstruction error path)
- AE01–AE03 (AE training artifacts)

### Split appendix treatment

- Retain MD01–MD06 as **methodological appendix only**
- Summarize briefly in main results; detail in appendix
- Do not use appendix scores for model selection

### Archive (keep, do not delete)

- EX01–EX10, EX12, ensemble branches, FE branches, `final_report/`, `final_diagnostics/`

### Requires rerun (if strict governance adopted)

- AE-LightGBM **LD32 tuned** (`outputs/optuna/ae_lgbm/`) for P04 parity
- Optional: latent-only augmentation without recon error (not executed)

### Integration strategy comparison (AE17)

Controlled comparison: `outputs/final_comparison/latent_integration_strategy_comparison.csv`

| Model | Validation AP | Test AP | Notes |
|-------|---------------|---------|-------|
| P01 baseline | 0.602433 | 0.485756 | Primary control |
| AE05 replacement LD128 | 0.594149 | 0.489417 | Primary control |
| **AE17 latent-only augmentation** | **0.591898** | **0.483013** | Clean augmentation; validation AP below P01 |
| AE06 confounded augmentation | 0.598198 | 0.485417 | Reference only |

## Test-use risk note

Multiple exploratory branches (`outputs/final_comparison/next_controlled_experiments.csv`, `outputs/final_report/final_summary.json`, ensemble scripts) were developed after inspecting many test scores. This does **not** automatically invalidate chronological primary experiments (P01–P03), which have clear `run_config.json` lineage and pre-registered-style split protocol.

### Historical final-report ranking warning

`outputs/final_report/final_summary.json` ranks `fe_ae_tuned_score_ensemble` as `best_overall_model` by **test AP only**. This is **descriptive test ranking**, not a valid model-selection rule. Do not cite it for thesis conclusions. See `docs/FINAL_REPORT_GOVERNANCE_NOTE.md`. Generator logic in `src/generate_final_report_assets.py` was corrected to distinguish descriptive test ranks from validation-selected primary models.

**Freeze policy:** see `docs/EXPERIMENT_SCOPE_FREEZE.md`. Validation AP determines configuration; test AP is reported once for the frozen comparison set.

### Causal behavioral + AE-signal experiment (CBA01/CBA02 — executed 2026-06-10)

| Criterion | Status |
|-----------|--------|
| B1 P01 baseline reused | **Yes** — `outputs/baseline_lgbm/` |
| B2 causal behavioral implemented | **Yes** — `outputs/causal_behavioral_lgbm_default/` |
| B3 B2 + one CDV recon error | **Yes** — `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` |
| CDV AE retrained | **No** — reused `outputs/behavioral_cdv_ae_experiment/autoencoder_cdv_ld128/` |
| B2 validation AP | 0.613738 (+0.011305 vs P01) |
| B3 validation AP | 0.600659 (−0.013079 vs B2) |
| Interpretation | Rule A (B1 vs B2); Rule D (B2 vs B3) |

Comparison: `outputs/final_comparison/causal_behavioral_ae_comparison.csv`. Details: `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md`, `docs/CAUSAL_BEHAVIORAL_FEATURE_AUDIT.md`.

### Task-aware Autoencoder experiment (TAE01 — executed 2026-06-10)

| Criterion | Status |
|-----------|--------|
| Joint reconstruction + classification AE | **Executed** |
| Lambda ablation {0.1, 0.5, 1.0} with validation-only downstream AP selection | **Executed** |
| Frozen AAE01 preprocessing reused | **Yes** |
| Scripts | `src/train_task_aware_autoencoder_selected_numerical.py`, `src/train_task_aware_ae_lgbm.py` |
| AE output | `outputs/task_aware_autoencoder_selected_numerical_ld128/` |
| LGBM output | `outputs/task_aware_ae_lgbm_ld128/selected/` |
| Selected lambda | 0.1 |
| TAE01 validation AP | 0.524481 |
| Delta vs AAE01 validation AP | −0.000621 |
| Delta vs P01 validation AP | −0.077952 |
| Interpretation | Rule C |

Comparison: `outputs/final_comparison/task_aware_ae_comparison.csv`, `outputs/final_comparison/task_aware_lambda_selection.csv`. Details: `docs/TASK_AWARE_AUTOENCODER_EXPERIMENT.md`.

## Immediate next step

**Incorporate TAE01 results into the thesis diagnostic chapter using validation AP only. The experimental phase is closed; TAE01 is the final permitted AE integration experiment. No further branches without supervisor approval.**