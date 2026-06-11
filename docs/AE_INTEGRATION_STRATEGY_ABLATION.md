# AE Integration Strategy Ablation

## Purpose

This is a **separate diagnostic ablation**, not the initial proposal literal rerun under `outputs/initial_proposal/`.

The objective is to compare how a V-only Autoencoder should be integrated with fixed/default LightGBM on IEEE-CIS, holding all other training choices constant so the effect of the AE integration strategy can be observed clearly.

The experiment asks whether AE information is better used as:

1. **Latent replacement** — encoded representation replaces original V-features.
2. **Reconstructed-feature replacement** — decoder output replaces original V-features.
3. **Reconstruction-error augmentation** — original features retained; AE reconstruction error added as an anomaly signal.

## Strategies in scope

| Strategy ID | CLI variant | Description |
|-------------|-------------|-------------|
| STR-B0 | `baseline_fixed` | Raw LightGBM fixed/default baseline. Original raw features retained. No Autoencoder features. Control. |
| STR-AE1 | `du_latent_replacement` | Du-style latent replacement. V1–V339 replaced by AE latent representation. Non-V features retained. |
| STR-AE2 | `ding_reconstructed_replacement` | Ding-style reconstructed V replacement. V1–V339 passed through AE; decoder-reconstructed V features replace originals. Non-V features retained. |
| STR-AE3 | `reconstruction_error_augmentation` | AE anomaly-style reconstruction-error augmentation. All original features retained. Adds `v_ae_reconstruction_mse` and `v_ae_reconstruction_log1p_mse`. |

## Paper grounding

- **STR-AE1 (Du-style):** Inspired by [Du et al.] — latent/encoded representation fed to LightGBM instead of high-dimensional raw V-features.
- **STR-AE2 (Ding-style):** Inspired by [Ding et al.] — decoder-reconstructed features used as a denoised V representation before LightGBM.
- **STR-AE3 (Anomaly-style):** Inspired by [Autoencoder anomaly detection literature] — reconstruction error as an anomaly signal appended to the original feature matrix.

> **TODO:** Full APA citation metadata for Du et al., Ding et al., and the autoencoder anomaly-detection references must be filled in during thesis writing.

## Why Optuna is excluded

Optuna hyperparameter tuning is intentionally excluded in this phase. Tuning would confound the comparison: improvements could come from better LightGBM settings rather than from the AE integration strategy. All four strategies use the same fixed/default LightGBM parameters and the same validation early-stopping rule so the only intended difference is how AE information is integrated.

## Output isolation

All artifacts are written under:

```text
outputs/ae_integration_strategy_ablation/
```

This path is separate from:

- `outputs/initial_proposal/` — initial proposal literal rerun (BASE-01..AE-02)
- `outputs/final_comparison/initial_proposal_comparison.csv` — proposal-only comparison table

This ablation builds its own comparison at:

```text
outputs/ae_integration_strategy_ablation/comparison.csv
```

## Controlled design

All four strategies share:

- Chronological 60/20/20 split by `TransactionDT` (deterministic tie-break on `TransactionID`)
- Fixed/default LightGBM parameters (same as baseline default)
- `scale_pos_weight` computed from training labels only
- Validation-only early stopping on average precision (100 rounds)
- Validation-only threshold selection (MCC primary, F1 tie-breaker)
- Test used only for final evaluation
- Same output artifact schema (metrics JSON, threshold table, confusion matrices, feature importance, model, preprocessing)

The **only** intended difference between STR-B0, STR-AE1, STR-AE2, and STR-AE3 is the AE integration strategy.

## Expected interpretation

| Observation | Interpretation |
|-------------|----------------|
| STR-AE1 underperforms STR-B0 | Latent replacement may be losing useful V-feature information. |
| STR-AE2 improves over STR-B0 | Ding-style reconstruction may provide a useful denoised V representation. |
| STR-AE3 improves over STR-B0 | AE may be more useful as anomaly-signal augmentation than as feature replacement. |
| All AE variants underperform STR-B0 | Raw LightGBM remains the strongest controlled default baseline under fixed settings. |

Grouped reconstruction error, behavioral features, causal features, CDV AE, task-aware AE, static FE branches, late fusion, score ensembles, and Optuna tuning are **out of scope** for this phase.

## Recommended command order

Run validation first, then train the shared V-only Autoencoder, then each LightGBM strategy, then build the comparison table.

```bash
python src/_validate_ae_strategy_ablation_pipeline.py

python src/train_autoencoder_robust.py \
--latent-dim 32 \
--output-dir outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32 \
--phase-name ae_strategy_ablation_v_autoencoder_ld32

python src/train_ae_integration_strategy_ablation.py \
--variant baseline_fixed \
--output-dir outputs/ae_integration_strategy_ablation/B0_baseline_fixed \
--phase-name STR_B0_baseline_fixed

python src/train_ae_integration_strategy_ablation.py \
--variant du_latent_replacement \
--autoencoder-output-dir outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32 \
--output-dir outputs/ae_integration_strategy_ablation/AE1_du_latent_replacement \
--phase-name STR_AE1_du_latent_replacement

python src/train_ae_integration_strategy_ablation.py \
--variant ding_reconstructed_replacement \
--autoencoder-output-dir outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32 \
--output-dir outputs/ae_integration_strategy_ablation/AE2_ding_reconstructed_replacement \
--phase-name STR_AE2_ding_reconstructed_replacement

python src/train_ae_integration_strategy_ablation.py \
--variant reconstruction_error_augmentation \
--autoencoder-output-dir outputs/ae_integration_strategy_ablation/autoencoder_robust_ld32 \
--output-dir outputs/ae_integration_strategy_ablation/AE3_reconstruction_error_augmentation \
--phase-name STR_AE3_reconstruction_error_augmentation

python src/build_ae_strategy_ablation_comparison.py \
--base-output-dir outputs/ae_integration_strategy_ablation
```