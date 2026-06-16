# Task-Aware Autoencoder Experiment

## Motivation

Prior unsupervised Autoencoder integration experiments did not improve chronological validation Average Precision versus original-feature LightGBM (P01). V-only latent replacement (AE05), clean latent-only augmentation (AE17), selected-numerical latent replacement (AAE01), and decoder-reconstructed replacement (AAE02) all remained below P01 under the executed protocol.

A consistent diagnosis is that reconstruction-oriented Autoencoder training optimizes dominant transaction structure rather than low-frequency fraud-discriminative information. Recent IEEE-CIS fraud-detection literature often uses hybrid or task-aware Autoencoder systems rather than pure unsupervised latent replacement.

This experiment introduces the **smallest controlled form of supervision**: a shared encoder with joint reconstruction and fraud-classification objectives. It does not implement VAE, GAN, attention, sequence models, graph models, ensembles, SMOTE, or architecture search.

## Research question

Does adding a supervised fraud-classification objective during Autoencoder training produce a latent representation that is more useful for downstream LightGBM than the existing unsupervised selected-numerical latent representation (AAE01)?

## Controlled design

**Changed:**

- Addition of a fraud classification head on the shared latent representation.
- Joint loss: `total_loss = reconstruction_loss + lambda_classification × classification_loss`.
- Bounded validation-only lambda ablation: `lambda_classification ∈ {0.1, 0.5, 1.0}`.
- Lambda selected by downstream LightGBM validation AP only.

**Fixed:**

- Selected numerical input scope (387 features from AAE01 audit).
- Frozen train-median imputer and StandardScaler from `outputs/autoencoder_selected_numerical_ld128/`.
- Latent dimension LD128.
- Chronological 60/20/20 `TransactionDT` split.
- Downstream LightGBM default parameter recipe (same as P01 / AAE01).
- Replacement integration design: remove 387 selected numerical features; add 128 latent features; retain 45 non-AE raw predictors including `TransactionDT`.
- Average Precision validation metric and MCC threshold selection on validation.
- Test split used only once for descriptive reporting of the selected lambda.

## Architecture

| Component | Structure |
|-----------|-----------|
| Input | 387 selected numerical features (scaled) |
| Shared encoder | 387 → Dense(256, ReLU) → Dense(128, ReLU) → Latent(128, ReLU) |
| Reconstruction decoder | 128 → Dense(128, ReLU) → Dense(256, ReLU) → Dense(387, linear) |
| Classification head | 128 → Dense(64, ReLU) → Dropout(0.2) → Dense(1, sigmoid) |

**Joint objective:**

- Reconstruction loss: MSE on scaled selected numerical inputs.
- Classification loss: weighted binary cross-entropy on chronological train `isFraud` labels only.
- Positive class weight: `negative_count / positive_count` from train labels (= 28.556557).
- Total loss: `reconstruction_loss + lambda_classification × classification_loss`.

Early stopping monitors total validation loss (`val_loss`). Classification-head validation AP and ROC-AUC are logged as diagnostics only.

## Lambda selection

Selection criterion: **highest downstream LightGBM validation AP** among the three lambda candidates. Test was not used for lambda selection. Non-selected candidates have no test latent arrays.

| lambda_classification | AE best epoch | Downstream LGBM validation AP | Selected |
|----------------------:|--------------:|------------------------------:|:--------:|
| 0.1 | 6 | **0.524481** | Yes |
| 0.5 | 3 | 0.522151 | No |
| 1.0 | 1 | 0.520238 | No |

Evidence: `outputs/final_comparison/task_aware_lambda_selection.csv`

## Results

| Model | Validation AP | Test AP (descriptive) |
|-------|---------------|----------------------|
| P01 baseline default | 0.602433 | 0.485756 |
| AAE01 unsupervised selected-numerical LD128 | 0.525103 | 0.398658 |
| **TAE01 task-aware selected-numerical LD128 (λ=0.1)** | **0.524481** | **0.407953** |

Deltas versus P01 (validation): TAE01 −0.077952. Deltas versus AAE01 (validation): TAE01 −0.000621.

Evidence: `outputs/final_comparison/task_aware_ae_comparison.csv`

## Interpretation

**Rule C:** Adding the supervised classification objective does not improve the selected-numerical latent representation for downstream LightGBM under the executed protocol.

TAE01 validation AP (0.524481) does not exceed AAE01 (0.525103) or P01 (0.602433). Test AP is reported descriptively only and was not used for lambda or model selection.

## Limitations

- Task-aware AE is supervised during representation learning rather than a pure unsupervised Autoencoder.
- One architecture only; three lambda values only; one random seed (42).
- One chronological validation block reused for AE early stopping, lambda selection, and LightGBM early stopping/threshold selection.
- Historical test inspection occurred in earlier exploratory branches.
- No graph, attention, sequence, or generative component.
- Results apply only to this dataset and chronological protocol.
- TAE01 is the **final permitted AE integration experiment**; no further AE branches are authorized.

## Artifacts

| Artifact | Path |
|----------|------|
| Task-aware AE outputs | `outputs/task_aware_autoencoder_selected_numerical_ld128/` |
| Selected lambda downstream LGBM | `outputs/task_aware_ae_lgbm_ld128/selected/` |
| Lambda selection CSV | `outputs/final_comparison/task_aware_lambda_selection.csv` |
| Controlled comparison CSV | `outputs/final_comparison/task_aware_ae_comparison.csv` |
| Training scripts | `src/train_task_aware_autoencoder_selected_numerical.py`, `src/train_task_aware_ae_lgbm.py` |