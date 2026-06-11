# AE Strategy Tuning — Results

Completed runs under gitignored `outputs/`. This document records the compact summary for thesis traceability.

**Run dates:** 2026-06-11 (ablation LD32/LD128, Optuna tuning)  
**Protocol:** chronological 60/20/20 split; validation AP for model design; test AP descriptive only  
**Tuning:** Optuna TPE, `final` profile, 50 trials per tuned variant

Related documents:

- [`AE_INTEGRATION_STRATEGY_ABLATION.md`](AE_INTEGRATION_STRATEGY_ABLATION.md) — fixed/default ablation protocol
- [`AE_INTEGRATION_STRATEGY_ABLATION_RESULTS.md`](AE_INTEGRATION_STRATEGY_ABLATION_RESULTS.md) — LD32 fixed/default summary
- [`AE_STRATEGY_TUNING_PLAN.md`](AE_STRATEGY_TUNING_PLAN.md) — tuning protocol

## 1. LD32 ablation result (fixed/default LightGBM)

Shared AE artifact: `outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32` (latent dim 32)

| Strategy ID | Variant | Val AP | Test AP | Δ Test AP vs STR-B0 | Test MCC | Features |
|-------------|---------|--------|---------|---------------------|----------|----------|
| **STR-B0** | `baseline_fixed` | **0.6024** | **0.4858** | — | **0.4867** | 432 |
| STR-AE3 | `reconstruction_error_augmentation` | 0.6001 | 0.4832 | −0.0025 | 0.4806 | 434 |
| STR-AE1 | `du_latent_replacement` | 0.5915 | 0.4821 | −0.0037 | 0.4769 | 125 |
| STR-AE2 | `ding_reconstructed_replacement` | 0.5827 | 0.4754 | −0.0104 | 0.4684 | 432 |

**Test AP ranking:** STR-B0 > STR-AE3 > STR-AE1 > STR-AE2

**Conclusion (LD32):** All AE integration strategies underperform raw fixed/default baseline. Replacement paths (AE1, AE2) are weakest.

Artifact: `outputs/ae_integration_strategy_ablation/comparison.csv`

## 2. LD128 ablation result (fixed/default LightGBM)

Shared AE artifact: `outputs/ae_integration_strategy_ablation_ld128/autoencoder_robust_ld128` (latent dim 128)

| Strategy ID | Variant | Val AP | Test AP | Δ Test AP vs STR-B0 | Test MCC | Features |
|-------------|---------|--------|---------|---------------------|----------|----------|
| **STR-AE3** | `reconstruction_error_augmentation` | **0.6114** | **0.4892** | **+0.0034** | **0.4892** | 434 |
| **STR-B0** | `baseline_fixed` | 0.6024 | 0.4858 | — | 0.4867 | 432 |
| STR-AE1 | `du_latent_replacement` | 0.5869 | 0.4811 | −0.0047 | 0.4773 | 221 |
| STR-AE2 | `ding_reconstructed_replacement` | 0.5899 | 0.4792 | −0.0066 | 0.4844 | 432 |

**Test AP ranking:** STR-AE3 > STR-B0 > STR-AE1 > STR-AE2

**Conclusion (LD128):** STR-AE3 is the **only** fixed/default strategy that beats baseline on test AP. AE replacement strategies remain below baseline. LD128 improves AE3 more than LD32 (+0.0060 test AP vs LD32 STR-AE3).

Artifact: `outputs/ae_integration_strategy_ablation_ld128/comparison.csv`

## 3. Tuned comparison result

Output root: `outputs/ae_integration_strategy_ablation_ld128/optuna/`  
Tuning profile: `final`, 50 Optuna trials each

| Strategy ID | Variant | Val AP | Test AP | Δ Test AP vs TUNE-B0 | Test MCC | n_trials | Features |
|-------------|---------|--------|---------|----------------------|----------|----------|----------|
| **TUNE-B0** | `baseline_lgbm_tuned` | **0.6378** | **0.5060** | — | **0.4984** | 50 | 432 |
| TUNE-AE3 | `ae3_reconstruction_error_lgbm_ld128_tuned` | 0.6290 | 0.4994 | −0.0066 | 0.4911 | 50 | 434 |

**Test AP ranking:** TUNE-B0 > TUNE-AE3

### Tuning lift vs fixed/default (same feature family)

| Model line | Fixed/default Test AP | Tuned Test AP | Tuning gain |
|------------|----------------------|---------------|-------------|
| Baseline (STR-B0 → TUNE-B0) | 0.4858 | **0.5060** | +0.0202 |
| AE3 (STR-AE3 → TUNE-AE3) | 0.4892 | 0.4994 | +0.0102 |

Under fair tuned comparison, **tuned raw baseline wins**. The fixed/default AE3 advantage (+0.0034) does **not** survive Optuna tuning.

Artifact: `outputs/ae_integration_strategy_ablation_ld128/optuna/comparison.csv`

## 4. Feature importance — AE3 fixed vs AE3 tuned

Both models use original raw features + `v_ae_reconstruction_mse` + `v_ae_reconstruction_log1p_mse` only (no latent, no reconstructed V).

### STR-AE3 fixed/default (LD128)

Source: `outputs/ae_integration_strategy_ablation_ld128/AE3_reconstruction_error_augmentation/feature_importance.csv`

| Rank (gain) | Feature | importance_gain | importance_split |
|-------------|---------|-----------------|------------------|
| **1** | `v_ae_reconstruction_mse` | **830,020** | 3,175 |
| 2 | V258 | 628,548 | 106 |
| 3 | P_emaildomain | 522,286 | 4,534 |
| … | … | … | … |
| 57 | `v_ae_reconstruction_log1p_mse` | 36,799 | 532 |

`v_ae_reconstruction_mse` dominates by a wide margin under fixed/default LightGBM.

### TUNE-AE3 tuned (LD128)

Source: `outputs/ae_integration_strategy_ablation_ld128/optuna/AE3_reconstruction_error_tuned/feature_importance.csv`

| Rank (gain) | Feature | importance_gain | importance_split |
|-------------|---------|-----------------|------------------|
| **1** | `v_ae_reconstruction_mse` | **64,578** | 3,924 |
| 2 | V258 | 38,764 | 37 |
| 3 | TransactionDT | 23,170 | 8,301 |
| … | … | … | … |
| 39 | `v_ae_reconstruction_log1p_mse` | 3,575 | 942 |

`v_ae_reconstruction_mse` remains **#1 by gain** after tuning, but absolute gain drops sharply (~830k → ~65k) as the tuned model redistributes importance across raw features (V258, TransactionDT, card/identity columns). `v_ae_reconstruction_log1p_mse` stays secondary in both runs.

## 5. Final interpretation

### Integration strategy (fixed/default ablation)

| Question | LD32 answer | LD128 answer |
|----------|-------------|--------------|
| Does latent replacement help? | No (STR-AE1 below STR-B0) | No |
| Does reconstructed V replacement help? | No (STR-AE2 lowest) | No |
| Does reconstruction-error augmentation help? | Marginal; still below baseline | **Yes** — only strategy above STR-B0 |

AE is more useful as an **anomaly-signal augmentation** than as feature replacement, but only with a sufficiently expressive AE (LD128). LD32 reconstruction error did not carry enough signal.

### Fair tuned comparison

After 50-trial Optuna tuning (`final` profile):

- **TUNE-B0** achieves the highest test AP (**0.5060**) and test MCC (**0.4984**).
- **TUNE-AE3** improves over fixed/default STR-AE3 (+0.0102 test AP) but **does not beat tuned baseline** (−0.0066 test AP).
- Tuning benefits baseline roughly **twice as much** as AE3 (+0.0202 vs +0.0102).

The fixed/default STR-AE3 lead was therefore partly a **hyperparameter mismatch**: raw features respond more to tuning than AE3’s augmented matrix under the same search budget.

### Feature-importance evidence

Reconstruction MSE remains the top gain feature in both AE3 fixed and AE3 tuned models, confirming the AE contributes a real anomaly signal. After tuning, however, raw V/identity/time features absorb more relative importance and the reconstruction-error gain magnitude collapses — consistent with tuned baseline outperforming tuned AE3 on test AP.

### Thesis-facing conclusions (provisional)

1. **Do not promote Du-style latent replacement or Ding-style reconstructed V** as primary AE integration paths on IEEE-CIS under controlled LightGBM settings.
2. **Reconstruction-error augmentation (STR-AE3 + LD128)** is the only ablation variant that beat fixed/default baseline, with strong `v_ae_reconstruction_mse` importance.
3. **Under fair tuned comparison, tuned raw baseline (TUNE-B0) remains strongest** on test AP; AE3 tuned does not close the gap.
4. AE reconstruction error is best interpreted as a **supporting anomaly feature**, not a replacement for raw LightGBM or a standalone thesis candidate without further fusion or task-specific design.

Do not claim final thesis model promotion from these results alone; compare against the active thesis path (e.g. FUS-01 / LF01) and supervisor governance before narrative closure.

## Artifact index (local, gitignored)

| Stage | Path |
|-------|------|
| LD32 ablation | `outputs/ae_integration_strategy_ablation/comparison.csv` |
| LD128 ablation | `outputs/ae_integration_strategy_ablation_ld128/comparison.csv` |
| Tuned comparison | `outputs/ae_integration_strategy_ablation_ld128/optuna/comparison.csv` |
| STR-AE3 fixed importance | `outputs/ae_integration_strategy_ablation_ld128/AE3_reconstruction_error_augmentation/feature_importance.csv` |
| TUNE-AE3 importance | `outputs/ae_integration_strategy_ablation_ld128/optuna/AE3_reconstruction_error_tuned/feature_importance.csv` |