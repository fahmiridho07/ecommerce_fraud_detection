# AE Integration Strategy Ablation — Results

Completed run under `outputs/ae_integration_strategy_ablation/` (gitignored). This document records the compact summary for thesis traceability.

**Run date:** 2026-06-11  
**Shared AE artifact:** `outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32` (V-only AE, latent dim 32)  
**LightGBM settings:** fixed/default (no Optuna) across all four strategies  
**Primary metric:** validation Average Precision (PR-AUC); test AP reported descriptively only

See [`AE_INTEGRATION_STRATEGY_ABLATION.md`](AE_INTEGRATION_STRATEGY_ABLATION.md) for protocol and reproduction commands.

## Results summary

| Strategy ID | Variant | Val AP | Test AP | Δ Test AP vs STR-B0 | Test MCC | Features |
|-------------|---------|--------|---------|---------------------|----------|----------|
| **STR-B0** | `baseline_fixed` | **0.6024** | **0.4858** | — | **0.4867** | 432 |
| STR-AE3 | `reconstruction_error_augmentation` | 0.6001 | 0.4832 | −0.0025 | 0.4806 | 434 |
| STR-AE1 | `du_latent_replacement` | 0.5915 | 0.4821 | −0.0037 | 0.4769 | 125 |
| STR-AE2 | `ding_reconstructed_replacement` | 0.5827 | 0.4754 | −0.0104 | 0.4684 | 432 |

**Test AP ranking:** STR-B0 > STR-AE3 > STR-AE1 > STR-AE2

### Additional metrics

| Strategy ID | Val ROC-AUC | Test ROC-AUC | Selected threshold | Test precision | Test recall | Test F1 | Best iteration |
|-------------|-------------|--------------|--------------------|----------------|-------------|---------|----------------|
| STR-B0 | 0.9131 | 0.8752 | 0.70 | 0.6726 | 0.3706 | 0.4779 | 1062 |
| STR-AE1 | 0.9141 | 0.8788 | 0.67 | 0.6605 | 0.3629 | 0.4685 | 1466 |
| STR-AE2 | 0.9141 | 0.8793 | 0.77 | 0.6118 | 0.3804 | 0.4691 | 543 |
| STR-AE3 | 0.9126 | 0.8732 | 0.70 | 0.6561 | 0.3713 | 0.4742 | 1071 |

## Interpretation

### STR-AE1 vs STR-B0 (Du-style latent replacement)

STR-AE1 underperforms STR-B0 on both validation AP (−0.011) and test AP (−0.004). Replacing V1–V339 with 32-dimensional AE latents reduces the feature space (432 → 125 features) but does not improve ranking quality. **Latent replacement likely discards useful V-feature information** that LightGBM can exploit directly.

### STR-AE2 vs STR-B0 (Ding-style reconstructed V replacement)

STR-AE2 is the weakest variant (validation AP −0.020, test AP −0.010 vs STR-B0). Decoder-reconstructed V features in scaled space do not provide a better V representation than the originals under fixed/default LightGBM. **Ding-style reconstruction is not a viable primary integration path** in this controlled setting with AE LD32.

### STR-AE3 vs STR-B0 (reconstruction-error augmentation)

STR-AE3 is closest to STR-B0 (validation AP −0.002, test AP −0.003) while retaining all original features and adding only two reconstruction-error columns. **AE is more promising as an anomaly-signal augmentation than as feature replacement**, but the gain is still insufficient to beat the raw baseline.

### Overall conclusion

Under **fixed/default LightGBM + V-only AE LD32**, **STR-B0 (raw LightGBM) remains the strongest controlled baseline**. All three AE integration strategies underperform the baseline in test AP. This ablation supports keeping raw LightGBM as the reference integration default and treating AE-based replacement as subordinate evidence unless a different AE configuration (e.g., latent dimension, tuning, or fusion) is explored in a separate experiment.

## Artifact paths (local, gitignored)

| Strategy | Output directory |
|----------|------------------|
| Shared AE | `outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32` |
| STR-B0 | `outputs/ae_integration_strategy_ablation/B0_baseline_fixed` |
| STR-AE1 | `outputs/ae_integration_strategy_ablation/AE1_du_latent_replacement` |
| STR-AE2 | `outputs/ae_integration_strategy_ablation/AE2_ding_reconstructed_replacement` |
| STR-AE3 | `outputs/ae_integration_strategy_ablation/AE3_reconstruction_error_augmentation` |
| Comparison table | `outputs/ae_integration_strategy_ablation/comparison.csv` |

Rebuild comparison after re-running variants:

```bash
python src/build_ae_strategy_ablation_comparison.py --base-output-dir outputs/ae_integration_strategy_ablation
```