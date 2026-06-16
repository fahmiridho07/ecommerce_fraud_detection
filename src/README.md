# Source Script Index

This directory is now intentionally narrow. It contains only the active proposal-scope pipeline and lightweight support code. Archived experiments are under `../archive/source/`.

## Core Support

- `config.py` - dataset paths, output paths, random seed, and split constants.
- `data_loader.py` - IEEE-CIS train data loading.
- `splitting.py` - chronological split helpers.
- `preprocessing.py` - feature/target preparation and categorical preprocessing.
- `evaluation.py` - AP, ROC-AUC, threshold, and confusion-matrix metrics.
- `utils.py` - JSON, logging, seed, and filesystem helpers.

## Active Thesis Pipeline

- `check_data_split.py` - verifies the chronological split.
- `train_baseline_lgbm.py` - P01 / BASE-01 baseline LightGBM default.
- `train_autoencoder_robust.py` - robust autoencoder latent feature generation with train-fitted median imputation, masked reconstruction loss, and linear latent features.
- `train_ae_lgbm.py` - P03 / AE-01 AE-LightGBM default latent replacement plus `V*` missing indicators.
- `tune_lgbm_optuna.py` - P02 and P04 Optuna tuning only:
  - `--model_type baseline_lgbm`
  - `--model_type ae_lgbm_ld128`
- `build_initial_proposal_comparison.py` - builds the four-row P01-P04 comparison.
- `generate_initial_proposal_diagnostics.py` - in-depth diagnostics from saved P01-P04 artifacts (missingness signal, V gain share, AE drift).
- `build_representation_ablation_comparison.py` - compares P01, P03, and hybrid top-V AE runs.
- `train_ae_lgbm.py --retain-top-v-features K` - hybrid representation ablation (latent + top baseline `V*`).
- `train_ae_reconstruction_error_lgbm.py` - post-fix ablation that keeps all original baseline features and appends LD128 AE reconstruction-error features.

Canonical thesis artifacts (post-fix rerun): `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv`. Script defaults still write to legacy `outputs/baseline_lgbm/` and `outputs/ae_lgbm/`; use directory overrides from `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md` for isolated reruns.

## Validation

- `_validate_initial_proposal_pipeline_guards.py` - static guard for the proposal pipeline.
- `../tests/test_initial_proposal_pipeline_guards.py` - unit tests for proposal-scope safety rules.

## Archived Work

The following experiment families were parked outside `src/`:

- AE appendices and additional AE strategy ablations.
- Behavioral and causal behavioral branches.
- Late fusion branches.
- Feature engineering, velocity, UID, and score-ensemble branches.
- GBDT backend comparison WIP.
- Methodology/report generators and the old notebook report.

See `../archive/README.md` for the archive map.
