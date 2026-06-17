# Initial Proposal Rerun Guide

Status: historical chronological reproduction guide.

The active code default is now `--split-strategy stratified_holdout`. To
reproduce the 2026-06-16 P01-P04 numbers in this guide, pass
`--split-strategy chronological` to split-aware scripts.

This guide covers only the original thesis proposal experiment path. It does not cover behavioral features, causal behavioral alignment, CDV AE, feature-engineered static branches, late fusion, score ensembles, GBDT backend comparisons, or task-aware AE experiments.

## Latest Rerun Status

| Field | Value |
|-------|-------|
| Date | 2026-06-16 |
| Output tree | `outputs/initial_proposal/` |
| Comparison table | `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv` |
| Tuning budget | 15 trials per tuned model (`--tuning_profile final`) |
| Dataset | Full IEEE-CIS train (`SAMPLE_SIZE=None`) |

### Results summary

| ID | Test PR-AUC | Features | Notes |
|----|-------------|----------|-------|
| P01 | 0.485756 | 432 | Baseline default |
| P02 | **0.504900** | 432 | Best model; 15 Optuna trials |
| P03 | 0.480217 | 464 | LD32 + 339 `v_missing_*` |
| P04 | 0.484527 | 560 | LD128 tuned + 339 `v_missing_*`; 15 Optuna trials |

To reproduce these numbers, run the command sequence below. Legacy `outputs/baseline_lgbm/` and `outputs/ae_lgbm/` are not updated by this isolated rerun.

## Active Models

| Canonical ID | Legacy ID | Role |
|--------------|-----------|------|
| BASE-01 | P01 | Baseline LightGBM default |
| BASE-02 | P02 | Tuned baseline LightGBM |
| AE-01 | P03 | AE-LightGBM default, LD32 latent replacement |
| AE-02 | P04 | Tuned AE-LightGBM, LD128 latent replacement |

The current AE path preserves `V*` missingness. `train_autoencoder_robust.py` uses train-fitted median imputation, masked reconstruction loss over originally observed `V*` cells, and a linear latent layer. `train_ae_lgbm.py` and the `ae_lgbm_ld128` Optuna path append `v_missing_*` indicators beside the latent features.

## Out Of Scope

Do not use these branches when reproducing the initial proposal comparison:

- Feature-engineered static FE, UID, and velocity branches.
- AE augmented, CDV AE, selected-numerical AE, task-aware AE, and reconstruction appendix branches.
- Behavioral and causal behavioral LightGBM branches.
- Late fusion and score ensembles.
- GBDT backend comparisons with XGBoost or CatBoost.

## Isolated Output Directories

Use a dedicated tree under `outputs/initial_proposal/` so reruns do not overwrite legacy artifacts.

| Step | Isolated directory |
|------|-------------------|
| BASE-01 | `outputs/initial_proposal/baseline_lgbm_default` |
| AE LD32 | `outputs/initial_proposal/autoencoder_robust_ld32` |
| AE-01 | `outputs/initial_proposal/ae_lgbm_ld32_default` |
| AE LD128 | `outputs/initial_proposal/autoencoder_robust_ld128` |
| BASE-02 | `outputs/initial_proposal/optuna/baseline_lgbm_tuned` |
| AE-02 | `outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned` |
| Comparison | `outputs/initial_proposal/final_comparison/` |

Legacy locations such as `outputs/baseline_lgbm/` and `outputs/ae_lgbm/` remain the script defaults for backward compatibility.

## Required Command Order

Run from the repository root after installing `requirements.txt` and placing IEEE-CIS files under `data/raw/`.

```bash
python src/check_data_split.py \
  --split-strategy chronological

python src/train_baseline_lgbm.py \
  --output-dir outputs/initial_proposal/baseline_lgbm_default \
  --phase-name 2_baseline_lgbm_initial_proposal \
  --split-strategy chronological

python src/train_autoencoder_robust.py \
  --latent-dim 32 \
  --output-dir outputs/initial_proposal/autoencoder_robust_ld32 \
  --phase-name 3B_robust_autoencoder_representation_learning_ld32 \
  --split-strategy chronological

python src/train_ae_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld32 \
  --output-dir outputs/initial_proposal/ae_lgbm_ld32_default \
  --phase-name 4_ae_lgbm_ld32_default_initial_proposal \
  --split-strategy chronological

python src/train_autoencoder_robust.py \
  --latent-dim 128 \
  --output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --phase-name 3B_robust_autoencoder_representation_learning_ld128 \
  --split-strategy chronological

python src/tune_lgbm_optuna.py \
  --model_type baseline_lgbm \
  --tuning_profile final \
  --n_trials 15 \
  --storage sqlite:///outputs/initial_proposal/optuna/baseline_lgbm_tuned/study.db \
  --study_name initial_proposal_baseline_lgbm \
  --output-dir outputs/initial_proposal/optuna/baseline_lgbm_tuned \
  --split-strategy chronological \
  --skip-global-comparison-update

python src/tune_lgbm_optuna.py \
  --model_type ae_lgbm_ld128 \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --tuning_profile final \
  --n_trials 15 \
  --storage sqlite:///outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned/study.db \
  --study_name initial_proposal_ae_lgbm_ld128 \
  --output-dir outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned \
  --split-strategy chronological \
  --skip-global-comparison-update

python src/build_initial_proposal_comparison.py \
  --baseline-default-dir outputs/initial_proposal/baseline_lgbm_default \
  --baseline-tuned-dir outputs/initial_proposal/optuna/baseline_lgbm_tuned \
  --ae-lgbm-default-dir outputs/initial_proposal/ae_lgbm_ld32_default \
  --ae-lgbm-ld128-tuned-dir outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned \
  --output-dir outputs/initial_proposal/final_comparison
```

Use `--skip-global-comparison-update` when you want the isolated rerun to avoid updating the default Optuna comparison table.

### Tuning budget

| Profile | `--n_trials` | Use case |
|---------|--------------|----------|
| Exploratory (current rerun) | 15 | Quick validation after pipeline changes |
| Defense-grade | 50 | Final thesis rerun if reviewers question tuning stability |

Both use `--tuning_profile final` for the wider search space. The script default is `--tuning_profile quick` with 20 trials; always pass flags explicitly for thesis reruns.

To read legacy artifacts instead, omit the directory overrides and rely on script defaults:

```bash
python src/build_initial_proposal_comparison.py
```

## Expected Output Folders

| Step | Output directory | Key artifacts |
|------|------------------|---------------|
| Split check | `outputs/split_summary.json` | Chronological split summary |
| BASE-01 | `outputs/initial_proposal/baseline_lgbm_default/` | `metrics_*_selected_threshold.json`, `run_config.json` |
| AE LD32 | `outputs/initial_proposal/autoencoder_robust_ld32/` | `latent_*.npy`, `latent_split_manifest.csv`, `latent_split_manifest_summary.json`, `v_imputer.pkl`, `v_scaler.pkl` |
| AE-01 | `outputs/initial_proposal/ae_lgbm_ld32_default/` | `metrics_*_selected_threshold.json`, `feature_set_summary.json` (expect 464 features), `model.pkl` |
| AE LD128 | `outputs/initial_proposal/autoencoder_robust_ld128/` | Same manifest, imputer, scaler, and latent arrays for LD128 |
| BASE-02 | `outputs/initial_proposal/optuna/baseline_lgbm_tuned/` | `best_params.json`, `final_model.pkl`, metrics JSON |
| AE-02 | `outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned/` | Same tuned artifacts |
| Comparison | `outputs/initial_proposal/final_comparison/` | `initial_proposal_comparison.csv`, `initial_proposal_missing_artifacts.json` |

## Latent TransactionID Alignment

Autoencoder latent vectors are saved as NumPy arrays without embedded row keys.
For this historical guide, row `i` in `latent_train.npy` must correspond to row
`i` in the chronological train split created with `--split-strategy
chronological`.

`train_autoencoder_robust.py` writes `latent_split_manifest.csv` beside the latent arrays. `train_ae_lgbm.py` and the `ae_lgbm_ld128` path in `tune_lgbm_optuna.py` load that manifest and fail fast when `TransactionID` order does not match the current split.

`train_ae_lgbm.py` also rejects stale zero-fill Autoencoder artifacts. If `v_imputer.pkl` is missing or `run_config.json` does not report `masked_mse_loss`, rerun the matching Autoencoder command before training AE-LightGBM.

## Comparison Tables

- `initial_proposal_comparison.csv` - four rows only: BASE-01, BASE-02, AE-01, AE-02.
- `optuna_comparison.csv` - Optuna-focused table for the active proposal-scope tuning paths: `baseline_lgbm` and `ae_lgbm_ld128`.

Older broader Optuna branches were parked under `archive/`. Keeping the proposal comparison separate prevents reruns from mixing FE, ensemble, or fusion experiments into the thesis narrative.

## Representation Ablation (After Diagnostics)

If full latent replacement (P03) underperforms because high-gain `V*` values are discarded, run the controlled hybrid fix:

```bash
python src/train_ae_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld32 \
  --output-dir outputs/initial_proposal/ae_lgbm_ld32_top25v_default \
  --phase-name 4_ae_lgbm_ld32_top25v_representation_fix \
  --retain-top-v-features 25 \
  --baseline-importance-path outputs/initial_proposal/baseline_lgbm_default/feature_importance.csv \
  --baseline-metrics-path outputs/initial_proposal/baseline_lgbm_default/metrics_test_selected_threshold.json

python src/build_representation_ablation_comparison.py
```

This keeps the top-25 baseline `V*` columns by gain, LD32 latents for the remaining `V*` columns, and `v_missing_*` indicators only for the replaced subset. No Optuna tuning.

## Reconstruction-Error Augmentation (After Archive Review)

If the goal is to test the most promising AE integration strategy from archived ablations, keep all original baseline features and append LD128 reconstruction-error features from the post-fix Autoencoder:

```bash
python src/train_ae_reconstruction_error_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --output-dir outputs/initial_proposal/ae_reconstruction_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_RECON_LD128_initial_proposal_postfix
```

This writes `comparison_against_initial_proposal.json` beside the model outputs. The 2026-06-16 run reached validation AP 0.612397 and test AP 0.495067 with 434 features. Treat this as post-diagnostic supporting evidence, not as a replacement for the four-row P01-P04 canonical comparison.

### AE-05 Hybrid + Reconstruction-Error Candidate

The first AE candidate that beats P02 keeps the top-25 supervised `V*` values, uses LD32 latent features for the replaced lower-gain `V*` block, and appends the global AE reconstruction-error score.

```bash
python src/tune_ae_hybrid_reconstruction_lgbm.py \
  --n_trials 0 \
  --output-dir outputs/initial_proposal/ae_lgbm_ld32_top25v_recon_fixed_from_hybrid_tuned

python src/compare_ae_hybrid_recon_bootstrap.py \
  --n-bootstrap 1000 \
  --output-dir outputs/initial_proposal/representation_ablation/bootstrap_ae05_vs_p02

python src/build_extended_proposal_comparison.py

python src/build_significance_comparison.py

# Independent Optuna for AE-05 (15 trials, optional verification):
python src/tune_ae_hybrid_reconstruction_lgbm.py \
  --n_trials 15 \
  --tuning_profile final
```

The 2026-06-16 run reached validation AP 0.626124 and test AP 0.509821 with 466 features. It beats P02 tuned baseline by +0.004921 test AP; paired bootstrap CI for the AP delta is [+0.000650, +0.009316] with one-sided p(delta <= 0) = 0.009.

### AE Refinement Ablations

To test paper-aligned AE refinements before tuning, run:

```bash
python src/train_autoencoder_normal_masked.py \
  --latent-dim 128 \
  --output-dir outputs/initial_proposal/normal_masked_autoencoder_ld128 \
  --phase-name normal_only_mask_aware_autoencoder_ld128 \
  --input-noise-std 0.02

python src/train_ae_reconstruction_error_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/normal_masked_autoencoder_ld128 \
  --output-dir outputs/initial_proposal/ae_normal_masked_global_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_NORMAL_MASKED_GLOBAL_ERROR_LD128_default_lgbm

python src/train_ae_reconstruction_feature_lgbm.py \
  --ae-feature-dir outputs/initial_proposal/normal_masked_autoencoder_ld128 \
  --output-dir outputs/initial_proposal/ae_normal_masked_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_NORMAL_MASKED_GROUPED_ERRORS_LD128_default_lgbm
```

Optional grouped-error diagnostic from the existing robust LD128 AE:

```bash
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

The 2026-06-16 refinement runs did not beat the robust LD128 global reconstruction-error augmentation. Keep them as diagnostic evidence for temporal drift and avoid promoting them without a new validation result.

## Diagnostics (No Retraining)

After P01-P04 artifacts exist under `outputs/initial_proposal/`, generate technical diagnostics (not thesis narrative until P02 is beaten):

```bash
python src/generate_initial_proposal_diagnostics.py
```

Outputs go to `outputs/initial_proposal/diagnostics/`:

| File | Purpose |
|------|---------|
| `diagnostic_summary.json` | Machine-readable summary |
| `diagnostic_notes.md` | Short technical interpretation (engineering only) |
| `v_cell_missingness_by_split.csv` | Cell-level `V*` missing rate per split |
| `v_row_missing_count_fraud_rate_by_split.csv` | Fraud rate by row missing-count bins |
| `v_column_missing_fraud_lift_train.csv` | Per-`V*` missing vs observed fraud lift (train only) |
| `baseline_feature_group_gain_share.csv` | Baseline gain split: `V*` vs non-V |
| `ae_ld32_feature_group_gain_share.csv` | P03 gain split: latent / `v_missing_*` / non-V |
| `ae_reconstruction_drift_by_split.csv` | Train/valid/test reconstruction MSE drift |
| `ae_reconstruction_error_by_fraud_class.csv` | Reconstruction MSE by fraud class |
| `baseline_top_v_to_ae_missing_indicator_bridge.csv` | Top baseline `V*` vs matching `v_missing_*` gain |

## Lightweight Validation

```bash
python src/_validate_initial_proposal_pipeline_guards.py
python -m pytest tests/test_initial_proposal_pipeline_guards.py -q
python -m pytest tests/test_initial_proposal_diagnostics.py -q
```
