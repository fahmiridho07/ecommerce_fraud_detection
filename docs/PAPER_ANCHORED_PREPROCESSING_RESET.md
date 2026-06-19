# Paper-Anchored Preprocessing Reset

Status: completed preprocessing decision record after stratified split reset.

Purpose: rebuild preprocessing around a small number of explicit paper anchors,
instead of accumulating useful but loosely justified engineering features. The
A1 branch described here has since become the final active representation for
the tuned AE-vs-SMOTE comparison; see `THESIS_SCOPE.md` and
`THESIS_RESULTS_BAB4.md` for current results.

## Why Reset

The previous `frequency_missingness_time_amount` branch performed well under
the old chronological protocol, but its components had mixed levels of direct
paper support:

- frequency encoding is explicitly supported by Alharbi et al. (2026);
- time/amount and broad feature engineering are supported by Moradi et al.
  (2025), but that paper uses a much broader pipeline than this thesis;
- identity/device normalization and rare bucketing are project diagnostics, not
  a direct paper recipe;
- preserving numeric `NaN` for LightGBM is practical, but differs from Alharbi
  et al. (2026), which uses numeric imputation and z-score scaling.

The next branch should therefore be narrow, reproducible, and easy to cite.

## Non-Negotiable Protocol Anchors

| Principle | Anchor | Thesis adaptation |
|-----------|--------|-------------------|
| Class proportions should be stable across train/test | Common fraud preprocessing practice; Alharbi et al. (2026) random/stratified evaluation context | Use stratified 60/20/20 train/validation/test holdout with `random_state=42`. |
| Sampling before split causes leakage | Kabane & Ouali (2024) | If resampling is tested, apply it only to the training split. |
| Fitted preprocessing must not see validation/test rows | Kabane & Ouali (2024); leakage audit notes | Fit imputers, encoders, scalers, frequency maps, and AE preprocessing on train only. |
| Validation/test must preserve original imbalance | Fraud evaluation literature | Never balance validation/test. |
| Temporal drift exists but is not the main S1 protocol | Dal Pozzolo et al. (2018); Lucas et al. (2019) | Discuss chronological/time-aware evaluation as limitation and future work. |

Correct thesis flow:

```text
raw -> clean -> stratified split -> fit preprocessing on train only
    -> transform validation/test with train-fitted objects
    -> optional train-only imbalance handling
    -> model training/evaluation
```

Do not use:

```text
raw -> preprocessing -> balancing -> split
```

That order can leak distributional or synthetic-sample information into the test
set.

## Candidate Anchor Branches

### A0 - Stratified Baseline Control

Role: new control baseline after reset.

Definition:

- drop `TransactionID`;
- keep original IEEE-CIS features;
- train-fitted categorical integer mapping;
- categorical missing value mapped to `__MISSING__`;
- unseen validation/test category mapped to unknown value;
- numeric `NaN` preserved for LightGBM native missing handling;
- stratified 60/20/20 split.

Use:

- establishes the new baseline under the active protocol;
- replaces the old chronological P02 as the active control.

### A1 - Alharbi-Style IEEE-CIS Preprocessing

Role: main paper-anchored preprocessing branch for the next rerun.

Primary anchor:

- Alharbi et al. (2026), IEEE-CIS section: categorical frequency encoding,
  z-score normalization for numerical attributes, median imputation for
  numerical attributes, dedicated missing category for categorical attributes,
  and augmentation/resampling procedures.

Thesis adaptation:

- use stratified 60/20/20 holdout instead of Alharbi's simpler train/test split;
- apply every fitted preprocessing object only from train to validation/test;
- initially run without generative augmentation because the thesis method is
  Autoencoder-LightGBM, not VAE-GAN data generation.

Minimum branch definition:

- drop `TransactionID`;
- categorical missing -> dedicated missing category;
- categorical frequency encoding learned on train only;
- numeric missing -> train median imputation;
- numeric scaling -> train-fitted z-score standardization;
- no target encoding;
- no identity/device manual normalization unless added as a separate diagnostic
  ablation.

Recommended experiment ID:

```text
A1 / alharbi_style_preprocessing_lgbm
```

Implementation:

```text
src/paper_preprocessing.py
src/train_paper_preprocessing_lgbm.py
```

Default command:

```bash
python src/train_paper_preprocessing_lgbm.py \
  --output-dir outputs/stratified_reset/alharbi_style_lgbm_default \
  --phase-name A1_alharbi_style_lgbm_default_stratified
```

Tuned command:

```bash
python src/tune_lgbm_optuna.py \
  --model_type alharbi_lgbm \
  --tuning_profile final \
  --n_trials 15 \
  --storage sqlite:///outputs/stratified_reset/optuna/alharbi_lgbm_tuned/study.db \
  --study_name stratified_reset_alharbi_lgbm \
  --output-dir outputs/stratified_reset/optuna/alharbi_lgbm_tuned \
  --skip-global-comparison-update
```

### A1-S - Sampling/Class-Weight Appendix

Role: appendix or robustness branch, not default mainline.

Anchors:

- Alharbi et al. (2026) for imbalance-aware augmentation motivation;
- Kabane & Ouali (2024) for avoiding pre-split sampling leakage.

Branch definition:

- start from A1;
- apply SMOTE, undersampling, or class weighting only to the training split;
- keep validation/test untouched;
- report PR-AUC/AP, F1, MCC, and confusion matrix.

### A2 - Moradi-Style Broad Feature Engineering

Role: related-work comparison or future work, not thesis mainline.

Reason:

- Moradi et al. (2025) uses much broader feature engineering, resampling, and
  ensemble modeling;
- adopting it fully would change the thesis into a broad FE/ensemble benchmark.

Allowed use:

- cite Moradi to justify that preprocessing can materially affect AUC-PR;
- run only as appendix/future work if scope is intentionally expanded.

### A3 - Temporal/Concept-Drift Evaluation

Role: future work / limitation.

Anchors:

- Dal Pozzolo et al. (2018), Lucas et al. (2019), Carcillo-style realistic fraud
  evaluation literature.

Use:

- discuss why production fraud detection benefits from time-aware validation;
- do not make it the active experiment protocol for the current S1 reset.

## What To Demote

Until separately rerun as explicit ablations, do not describe these as the main
paper-following preprocessing protocol:

- identity/device family normalization;
- rare-category bucketing;
- compact missingness summary groups;
- `TransactionAmt` cents;
- handcrafted browser/OS/screen parsing;
- broad rolling aggregates;
- target encoding.

Use the phrase:

```text
project-specific diagnostics and exploratory ablations
```

not:

```text
directly adopted from recent paper preprocessing protocols
```

## Original Rerun Ladder

This ladder is preserved as the decision path that produced the current active
result. It is not a current to-do list unless the thesis scope is deliberately
reopened.

1. S0 split check: default stratified split summary.
2. A0 baseline: original features + fixed/default LightGBM.
3. A0 tuned baseline: original features + Optuna.
4. A1 baseline: Alharbi-style preprocessing + fixed P02-like LightGBM budget.
5. A1 tuned baseline: same preprocessing + Optuna.
6. A1 AE branch: same split + AE-LightGBM integration.
7. A1-S sampling/class-weight appendix only if needed.
8. A2/A3 remain related work or future work unless the thesis scope changes.

Decision rule:

- promote A1 only if it improves or clarifies the baseline under the same
  stratified test split;
- promote any AE-integrated branch only if it beats the strongest A1 baseline
  on test AP with paired-bootstrap confidence interval above zero.

## Current Thesis Wording

> The final main experiment uses a compact, paper-anchored A1 preprocessing
> protocol based primarily on Alharbi et al. (2026), evaluated under a
> stratified train/validation/test split with train-only fitted transformations.
> On this dense representation, AE latent-space oversampling improves LightGBM
> over the matched baseline and SMOTE-NC. Temporal evaluation and concept drift
> are retained as limitations and future work.

## Source Anchors

- Alharbi et al. (2026), `docs/literature/cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md`
- Moradi et al. (2025), `docs/literature/cards/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md`
- Nguyen et al. (2022), `docs/literature/cards/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.md`
- Kabane & Ouali (2024), `docs/literature/cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md`
- Lucas et al. (2019), `docs/literature/cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md`
- Dal Pozzolo et al. (2018), `docs/literature/cards/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.md`
