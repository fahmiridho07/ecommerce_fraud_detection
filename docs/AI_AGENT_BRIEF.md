# AI Agent Brief

Status: active agent-facing guide for Bab 4 writing, 2026-06-18.

Use this file to orient future AI agents before editing code, docs, notebooks, or
thesis prose.

## Current Truth

The active thesis protocol is:

```text
split_strategy=stratified_holdout
train/validation/test = 60/20/20
random_state = 42
primary metric = Average Precision / PR-AUC
```

There is now an active thesis-facing result after the stratified reset:

- AE latent/reconstruction features lose or tie against LightGBM.
- Minority augmentation improves the baseline under stratified and chronological
  checks.
- The AE-specific contribution is conditional: AE latent-space oversampling ties
  SMOTE-NC on raw/NaN-native A0 features, but beats SMOTE-NC on dense
  Alharbi-style A1 features.
- Final tuned A1 comparison: baseline AP 0.838988, SMOTE-NC AP 0.843476,
  AE latent-SMOTE AP 0.850031. AE beats baseline by +0.011043 AP and SMOTE-NC
  by +0.006555 AP, both with paired-bootstrap support.

All chronological P01-P04, AE-05, preprocessing-ablation, and score-ensemble
numbers are historical evidence only.

Do not mix old chronological metrics with new stratified metrics in one result
table.

## Research Objective

The current goal is to write Bab 4 from the completed active results without
overclaiming. The defensible conclusion is that Autoencoder contributes as a
latent-space oversampler on dense frequency-encoded representations, not as a
feature extractor and not as a universal replacement for classical oversampling.

## Read Order

0. `docs/SKRIPSI_INTI.md` if the user needs the shortest human-facing anchor.
1. `docs/WRITING_FOCUS_AE_LATENT_OVERSAMPLING.md`
2. `docs/THESIS_SCOPE.md`
3. `docs/THESIS_RESULTS_BAB4.md`
4. `docs/AE_INTEGRATION_EXPERIMENT_RESULTS.md`
5. `docs/EXPERIMENT_REGISTRY.md`
6. `docs/EDA_AND_METHODOLOGY_AUDIT.md`
7. `docs/STRATIFIED_SPLIT_RESET.md`
8. `docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`
9. `src/README.md`
10. `docs/literature/INDEX.md`

PDFs under `../2. Reference/` are the source of truth for exact claims and
quotes. Literature cards are summaries, not citation substitutes.

## Active Result Map

The active rerun ladder is complete. Do not rerun expensive experiments unless a
new scope decision is made or an artifact is missing.

| ID | Purpose | Script |
|----|---------|--------|
| S0 | Verify split proportions and no row overlap | `python src/check_data_split.py` |
| AE-F | A0 feature/score AE integration | `src/run_ae_integration_experiment.py` |
| AE-G | A0 AE latent-space augmentation | `src/run_ae_augmentation_experiment.py` |
| AE-G-fair | A0 AE vs SMOTE-NC/random controls | `src/run_fair_augmentation_comparison.py` |
| AE-G-rep | Split-seed robustness | `src/run_repeated_split_validation.py` |
| AE-A1 | Dense A1 AE vs SMOTE-NC | `src/run_strong_baseline_augmentation.py` |
| AE-A1-TUNED | Fair tuned-vs-tuned comparison | `src/tune_a1_augmentation_optuna.py` |
| Figures | Bab 4 result figures | `src/generate_thesis_figures.py` |

Use `outputs/stratified_reset/` for active artifacts. Keep
`outputs/initial_proposal/` as historical chronological evidence.

For a Kaggle-only rerun of the locked thesis method, use
`kaggle/ieee_final_oversampling_kaggle.py`. Other Kaggle scripts are diagnostic
or exploratory unless promoted in `docs/THESIS_SCOPE.md`.

## Active Preprocessing Contract

The A1 branch is implemented in:

```text
src/paper_preprocessing.py
src/train_paper_preprocessing_lgbm.py
```

A1 does:

- categorical missing values -> dedicated missing token;
- categorical values -> train-frequency encoding;
- unseen validation/test categories -> frequency 0;
- numeric missing values -> train median imputation;
- numeric values -> train mean/std z-score scaling;
- no target encoding;
- no SMOTE/ADASYN before split;
- no validation/test fitting.

The older `src/enhanced_preprocessing.py` branch is diagnostic/archived unless
the scope is explicitly reopened.

## AE Decision Rule

An AE branch can become thesis-facing only if all are true:

- it uses the same active stratified split as the strongest baseline;
- all AE imputers, scalers, masks, latent models, and score-combination rules
  are fitted or selected without test data;
- test AP is higher than the strongest matched baseline;
- paired-bootstrap CI for AP delta supports a positive improvement;
- the result is recorded in `docs/EXPERIMENT_REGISTRY.md`.

Preferred AE framing:

- AE as a latent-space oversampler on dense representations;
- not AE as a feature extractor that is guaranteed to improve LightGBM;
- not AE as a universal replacement for SMOTE across all representations.

## Literature Anchors

Use these first:

- LightGBM classifier: `docs/literature/cards/LightGBM A Highly Efficient Gradient Boosting.md`
- PR-AUC for imbalanced evaluation: `docs/literature/cards/Saito_Rehmsmeier_2015_Precision_Recall_Plot.md`
- Leakage/sampling guardrail: `docs/literature/cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md`
- Classical oversampling control: `docs/literature/cards/Chawla_2002_SMOTE.md`
- Latent-space oversampling anchor: `docs/literature/cards/Dablain_2022_DeepSMOTE.md`
- IEEE-CIS preprocessing and AE family: `docs/literature/cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md`
- AE anomaly framing on IEEE-CIS: `docs/literature/cards/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md`
- Dataset shift limitation: `docs/literature/cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md`
- Broad feature-engineering context only: `docs/literature/cards/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md`

## Do Not

- Do not cite AE-05 or the fixed score ensemble as the active winner after the
  split reset.
- Do not compare old chronological AP directly against new stratified AP.
- Do not use broad UID, velocity, target encoding, rolling windows, or stacked
  ensembles as mainline methods without a new scope decision.
- Do not claim SOTA absolute; claim a controlled AE-vs-SMOTE contribution on A1
  dense features.
- Do not use `project/` as canonical source; it is local scratch and ignored.
- Do not edit thesis DOCX/PDF claims unless they are sourced from
  `docs/THESIS_RESULTS_BAB4.md` and the supporting registry.

## Safe Validation

Before handing off code changes:

```bash
python -m compileall -q src tests
python -m pytest -q
```

If the IEEE-CIS data is present locally, also run:

```bash
python src/check_data_split.py
```
