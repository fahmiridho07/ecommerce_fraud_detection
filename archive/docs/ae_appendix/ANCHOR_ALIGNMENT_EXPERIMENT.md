# Anchor Alignment Experiment

## Motivation

Anchor studies (Ding et al.; Du et al.) train Autoencoders on **broader standardized numerical predictor inputs**, whereas the executed thesis pipeline limited Autoencoder input to V1–V339. Under V-only replacement, chronological validation Average Precision did not improve over original-feature LightGBM (P01: 0.602433; AE05 LD128: 0.594149).

That negative result supports only the narrower claim: *V-only Autoencoder latent representations did not improve LightGBM under the executed pipeline.* It does not establish whether a broader, anchor-aligned numerical Autoencoder input would be more effective.

This experiment tests input-scope alignment as a **single controlled diagnostic** without opening tuning, augmentation, reconstruction-error, or architecture search branches.

## Controlled changes

**Changed:**

- Autoencoder input scope: V-only (339 features) → selected numerical predictors (387 features: V1–V339 + TransactionAmt + C1–C14 + D1–D15 + dist1–dist2 + 16 continuous `id_*` columns).

**Unchanged:**

- IEEE-CIS merged training data and chronological 60/20/20 `TransactionDT` split
- LightGBM default parameter recipe (same as P01 / AE05)
- Average Precision validation metric and MCC threshold selection on validation
- LD128 latent dimension and undercomplete dense AE architecture
- Replacement integration design (original AE-input columns removed; latent added)
- Test split used only for descriptive reporting

## Selected numerical feature policy

See `docs/SELECTED_NUMERICAL_AE_FEATURE_AUDIT.md` and `outputs/selected_numerical_ae_feature_audit/selected_numerical_features.json`.

| Group | Count | AE role |
|-------|------:|---------|
| V1–V339 | 339 | Included |
| TransactionAmt, C1–C14, D1–D15, dist1–dist2 | 32 | Included |
| Continuous `id_*` (16 columns) | 16 | Included |
| TransactionDT | 1 | Excluded from AE; retained downstream |
| Categorical strings (31 columns) | 31 | Excluded from AE; retained downstream |
| Numeric-coded categorical (13 columns) | 13 | Excluded from AE; retained downstream |

Preprocessing: train-median imputation per feature, then `StandardScaler` on train only, with fixed scaled clipping [-10, 10].

## Results

| Model | Validation AP | Test AP (descriptive) |
|-------|---------------|----------------------|
| P01 baseline default | 0.602433 | 0.485756 |
| AE05 V-only replacement LD128 | 0.594149 | 0.489417 |
| **AAE01 selected-numerical replacement LD128** | **0.525103** | **0.398658** |

Deltas versus P01 (validation): −0.077330. Deltas versus V-only LD128 (validation): −0.069046.

Evidence: `outputs/final_comparison/autoencoder_input_scope_comparison.csv`

## Interpretation

**Rule C:** Broadening the Autoencoder input does not improve the latent-replacement approach under the executed chronological protocol.

Validation AP for selected-numerical replacement (0.525103) remains below both P01 and V-only LD128. Test AP is reported descriptively only and must not be used for model selection.

## Limitations

- IEEE-CIS feature semantics differ substantially from ULB/Santander anchor datasets.
- Many predictors are anonymous; MSE reconstruction is unsupervised and may not suit all included numerical groups equally.
- One latent dimension (LD128), one random seed, one imputation policy.
- Validation split reused for AE early stopping and LightGBM early stopping.
- Historical test inspection occurred in earlier exploratory development.
- This is **conceptual alignment** with anchor methods, not exact replication of Ding/Du preprocessing or feature universes.
- Downstream LightGBM feature count drops to 173 (45 retained raw + 128 latent), changing capacity versus P01 (432) and AE05 (221); the experiment isolates input-scope broadening, not equal final feature cardinality.