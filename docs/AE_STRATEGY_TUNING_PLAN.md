# AE Strategy Tuning Plan

## Background

The fixed/default AE integration strategy ablation has been completed for LD32 and LD128.

**LD128 ablation result (fixed/default LightGBM):**

| Strategy | Validation AP | Test AP |
|----------|---------------|---------|
| STR-B0 / baseline_fixed | 0.6024 | 0.4858 |
| STR-AE3 / reconstruction_error_augmentation | **0.6114** | **0.4892** |

STR-AE3 is the **only** AE integration strategy that beat the fixed/default baseline on test AP (+0.0034).

**STR-AE3 LD128 feature importance:**

- `v_ae_reconstruction_mse` ranked **#1** by `importance_gain` (~830,020)
- `v_ae_reconstruction_log1p_mse` ranked lower (~36,799)

Du-style latent replacement (STR-AE1) and Ding-style reconstructed V replacement (STR-AE2) underperformed baseline and are **not** tuned in this phase.

## Tuning scope

This pipeline tunes only:

1. **TUNE-B0** — raw baseline LightGBM (`baseline_lgbm_tuned`)
2. **TUNE-AE3** — LightGBM + AE LD128 reconstruction-error features (`ae3_reconstruction_error_lgbm_ld128_tuned`)

**TUNE-AE3 feature matrix:**

- Original raw LightGBM features (including V1–V339)
- `v_ae_reconstruction_mse`
- `v_ae_reconstruction_log1p_mse`

**Explicitly excluded from TUNE-AE3:**

- AE latent features (`ae_latent_*`)
- Reconstructed V columns (`ae_reconstructed_V*`)
- Legacy `ae_augmented_lgbm_ld128` path (latent + reconstruction error)

## Goal

Test whether the STR-AE3 advantage over fixed/default baseline remains under a **fair tuned comparison** between:

- tuned raw baseline LightGBM
- tuned LightGBM + AE LD128 reconstruction-error anomaly signal

Do not claim a final thesis improvement until both tuned runs complete and `comparison.csv` is reviewed.

## Isolation

Output root:

```text
outputs/ae_integration_strategy_ablation_ld128/optuna/
```

This tuning path is separate from:

- `outputs/initial_proposal/` — initial proposal literal rerun
- `outputs/ae_integration_strategy_ablation/` — LD32 fixed/default ablation
- LF01 / fusion experiments
- behavioral, FE, CDV, task-aware AE, and selected-numerical branches

## Leakage prevention

- Chronological 60/20/20 split by `TransactionDT`
- Preprocessing fit on train only
- Validation: Optuna objective, early stopping, threshold selection
- Test: final evaluation only after best hyperparameters are selected

## Recommended command order

```bash
python src/_validate_ae_strategy_tuning_pipeline.py

python src/tune_ae_strategy_ablation.py \
--variant baseline_lgbm_tuned \
--tuning-profile final \
--n-trials 50 \
--storage sqlite:///outputs/ae_integration_strategy_ablation_ld128/optuna/baseline_lgbm_tuned/study.db \
--study-name ae_strategy_baseline_lgbm_tuned \
--output-dir outputs/ae_integration_strategy_ablation_ld128/optuna/baseline_lgbm_tuned

python src/tune_ae_strategy_ablation.py \
--variant ae3_reconstruction_error_lgbm_ld128_tuned \
--autoencoder-output-dir outputs/ae_integration_strategy_ablation_ld128/autoencoder_robust_ld128 \
--tuning-profile final \
--n-trials 50 \
--storage sqlite:///outputs/ae_integration_strategy_ablation_ld128/optuna/AE3_reconstruction_error_tuned/study.db \
--study-name ae_strategy_ae3_reconstruction_error_tuned \
--output-dir outputs/ae_integration_strategy_ablation_ld128/optuna/AE3_reconstruction_error_tuned

python src/build_ae_strategy_tuned_comparison.py \
--base-output-dir outputs/ae_integration_strategy_ablation_ld128/optuna
```

## Related documents

- [`AE_INTEGRATION_STRATEGY_ABLATION.md`](AE_INTEGRATION_STRATEGY_ABLATION.md) — fixed/default ablation protocol
- [`AE_INTEGRATION_STRATEGY_ABLATION_RESULTS.md`](AE_INTEGRATION_STRATEGY_ABLATION_RESULTS.md) — LD32 fixed/default results summary