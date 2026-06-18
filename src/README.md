# Source Script Index

Status: active source tree for Bab 4 writing after the stratified split reset.

All active training scripts default to:

```text
--split-strategy stratified_holdout
```

Use `--split-strategy chronological` only when intentionally reproducing
historical archived results.

## Core Support

- `config.py` - dataset paths, output paths, random seed, split ratios, and default split strategy.
- `data_loader.py` - IEEE-CIS train data loading.
- `splitting.py` - stratified and chronological split helpers.
- `preprocessing.py` - feature/target preparation and baseline categorical preprocessing.
- `paper_preprocessing.py` - A1 Alharbi-style frequency/median/z-score preprocessing.
- `enhanced_preprocessing.py` - exploratory preprocessing ablation helpers.
- `evaluation.py` - AP, ROC-AUC, threshold, and confusion-matrix metrics.
- `utils.py` - JSON, logging, seed, and filesystem helpers.

## Active Thesis Scripts

- `check_data_split.py` - validates the default stratified split.
- `train_baseline_lgbm.py` - baseline LightGBM default.
- `tune_lgbm_optuna.py --model_type baseline_lgbm` - tuned baseline LightGBM.
- `train_paper_preprocessing_lgbm.py` - A1 Alharbi-style preprocessing baseline.
- `tune_lgbm_optuna.py --model_type alharbi_lgbm` - tuned A1 baseline.
- `train_autoencoder_normal_masked.py` - mask-aware AE diagnostic branch.
- `run_ae_integration_experiment.py` - A0 feature/score AE integration; final verdict is negative.
- `run_ae_augmentation_experiment.py` - AE latent-space augmentation on A0.
- `run_fair_augmentation_comparison.py` - random/SMOTE-NC/AE matched controls.
- `run_repeated_split_validation.py` - split-seed robustness for augmentation.
- `run_strong_baseline_augmentation.py` - A1 dense AE-vs-SMOTE comparison.
- `run_vae_augmentation_experiment.py` - VAE prior control against SMOTE-NC.
- `tune_a1_augmentation_optuna.py` - final A1 tuned-vs-tuned comparison.
- `generate_thesis_figures.py` - generates active Bab 4 figures from `outputs/stratified_reset/`.

## Legacy Active-Path Entrypoints

These remain available for reproducibility but are not the shortest path for
Bab 4 writing because the consolidated harnesses above already produced the
active results.

- `train_autoencoder_robust.py` - robust AE latent feature generation.
- `train_ae_lgbm.py` - AE-LightGBM latent integration.
- `train_enhanced_preprocessing_lgbm.py` - preprocessing ablations and AE add-on branches.
- `train_score_ensemble.py` - score-level ensemble helper.

## Historical / Diagnostic Scripts

These scripts remain for traceability, diagnostics, or archived result
reproduction. They should not define active thesis claims unless rerun under the
stratified reset and documented in `docs/EXPERIMENT_REGISTRY.md`.

- `build_initial_proposal_comparison.py`
- `build_extended_proposal_comparison.py`
- `generate_initial_proposal_diagnostics.py`
- `generate_preprocessing_diagnostics.py`
- `build_representation_ablation_comparison.py`
- `run_top_v_retention_sweep.py`
- `build_significance_comparison.py`
- `train_ae_reconstruction_error_lgbm.py`
- `train_ae_reconstruction_feature_lgbm.py`
- `generate_ae_reconstruction_feature_tables.py`
- `tune_ae_hybrid_reconstruction_lgbm.py`
- `compare_enhanced_preprocessing_bootstrap.py`
- `compare_ae_hybrid_recon_bootstrap.py`

## Validation

- `_validate_initial_proposal_pipeline_guards.py` - historical proposal guard script.
- `../tests/test_initial_proposal_pipeline_guards.py` - split and proposal-scope safety tests.

## Archive Map

Parked exploratory experiment families live under `../archive/`.
