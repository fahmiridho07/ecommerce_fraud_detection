# Ding Reconstruction Alignment Experiment

## Motivation

Ding et al. integrate Autoencoder outputs with LightGBM using **decoder-reconstructed features** that preserve the original input dimensionality. Prior thesis experiments used **encoder bottleneck latent features** (V-only replacement, latent augmentation, and selected-numerical latent replacement).

Selected-numerical latent replacement compresses 387 numerical inputs into 128 latent features and validation AP fell substantially below P01. One methodological question remained: does decoder reconstruction better preserve discriminative information for LightGBM than bottleneck latents?

This experiment isolates Autoencoder **output strategy** while holding the selected-numerical input scope and frozen Autoencoder weights fixed.

## Controlled design

**Changed:**

- Autoencoder output passed to LightGBM: encoder latent (128) → decoder reconstructed output (387, scaled space).

**Fixed:**

- Selected 387 numerical Autoencoder input columns
- Train-median imputer and `StandardScaler` artifacts from `outputs/autoencoder_selected_numerical_ld128/`
- Fixed scaled clipping [-10, 10]
- Frozen trained Autoencoder weights (`autoencoder_retrained: false`)
- Chronological 60/20/20 `TransactionDT` split and row order
- Default LightGBM parameter recipe (P01)
- Average Precision validation metric and MCC threshold selection
- Test split for descriptive reporting only

## Feature construction

| Component | Count |
|-----------|------:|
| Selected numerical AE inputs (removed downstream) | 387 |
| Retained original predictors (non-AE) | 45 |
| Decoder-reconstructed numerical features | 387 |
| **Final LightGBM features** | **432** |

Retained 45 columns include `TransactionDT`, 31 categorical strings, and 13 numeric-coded categorical predictors. Reconstructed features use names `ae_reconstructed_{original_name}` in standardized/scaled decoder output space.

## Results

| Model | Validation AP | Test AP (descriptive) |
|-------|---------------|----------------------|
| P01 baseline default | 0.602433 | 0.485756 |
| AAE01 selected-numerical latent replacement LD128 | 0.525103 | 0.398658 |
| **AAE02 selected-numerical reconstructed replacement** | **0.549737** | **0.430796** |

Deltas versus P01 (validation): −0.052696. Deltas versus latent replacement (validation): +0.024634.

Evidence: `outputs/final_comparison/autoencoder_output_strategy_comparison.csv`

## Interpretation

**Rule B:** Decoder reconstruction preserves more useful information than bottleneck latent replacement, but does not outperform original-feature LightGBM.

Validation AP for reconstructed replacement (0.549737) exceeds latent replacement (0.525103) yet remains below P01 (0.602433). Test AP is descriptive only.

## Limitations

- Conceptual alignment with Ding et al., not exact replication.
- IEEE-CIS differs from datasets used in Ding.
- Selected numerical features use train-median preprocessing (differs from V-only `fillna(0)` in earlier branches).
- Autoencoder reached maximum epoch (100) during prior training; weights were not updated here.
- Reconstruction objective remains unsupervised MSE.
- Single random seed; validation reused for AE early stopping (prior run) and LightGBM early stopping.
- Historical repeated test inspection across exploratory branches.
- Reconstructed features are used in scaled space without inverse transform; this is a documented representation choice, not a searched alternative.