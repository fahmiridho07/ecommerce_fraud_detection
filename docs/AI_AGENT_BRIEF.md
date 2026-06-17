# AI Agent Brief

Status: active agent-facing guide after the 2026-06-17 stratified split reset.

Use this file to orient future AI agents before editing code, docs, notebooks, or
thesis prose.

## Current Truth

The active thesis protocol is a clean stratified holdout reset:

```text
split_strategy=stratified_holdout
train/validation/test = 60/20/20
random_state = 42
primary metric = Average Precision / PR-AUC
```

There is no active thesis-facing winner yet after this reset. All chronological
P01-P04, AE-05, preprocessing-ablation, and score-ensemble numbers are
historical evidence only until rerun under the active split.

Do not mix old chronological metrics with new stratified metrics in one result
table.

## Research Objective

The current goal is to build the strongest defensible paper-anchored
preprocessing baseline, then test whether an Autoencoder contribution improves
that baseline with statistical support.

The desired thesis conclusion may be "AE significantly improves the strongest
baseline", but the repository must not force that conclusion. If the best
paper-anchored LightGBM baseline beats the tested AE variants, document that
honestly.

## Read Order

1. `docs/THESIS_SCOPE.md`
2. `docs/STRATIFIED_SPLIT_RESET.md`
3. `docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`
4. `docs/EXPERIMENT_REGISTRY.md`
5. `src/README.md`
6. `docs/literature/INDEX.md`

PDFs under `../2. Reference/` are the source of truth for exact claims and
quotes. Literature cards are summaries, not citation substitutes.

## Active Experiment Ladder

Run from simple to complex:

| ID | Purpose | Script |
|----|---------|--------|
| S0 | Verify split proportions and no row overlap | `python src/check_data_split.py` |
| A0 | Original-feature LightGBM baseline | `python src/train_baseline_lgbm.py --output-dir outputs/stratified_reset/baseline_lgbm_default` |
| A0-T | Tuned original-feature LightGBM | `python src/tune_lgbm_optuna.py --model_type baseline_lgbm --tuning_profile final --n_trials 15 --output-dir outputs/stratified_reset/optuna/baseline_lgbm_tuned --skip-global-comparison-update` |
| A1 | Alharbi-style preprocessing baseline | `python src/train_paper_preprocessing_lgbm.py --output-dir outputs/stratified_reset/alharbi_style_lgbm_default` |
| A1-T | Tuned Alharbi-style baseline | `python src/tune_lgbm_optuna.py --model_type alharbi_lgbm --tuning_profile final --n_trials 15 --output-dir outputs/stratified_reset/optuna/alharbi_lgbm_tuned --skip-global-comparison-update` |
| A1-AE | AE-LightGBM under the same split | Run only after A1-T exists. Use the same split and train-only AE preprocessing. |
| A1-E | Feature-level or score-level AE integration | Promote only if it beats the strongest A1 baseline with paired-bootstrap support. |

Use `outputs/stratified_reset/` for new active reruns. Keep
`outputs/initial_proposal/` as historical chronological evidence.

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

- AE as representation/anomaly/complementary score signal;
- not AE as a guaranteed replacement for raw tabular features.

## Literature Anchors

Use these first:

- LightGBM classifier: `docs/literature/cards/LightGBM A Highly Efficient Gradient Boosting.md`
- PR-AUC for imbalanced evaluation: `docs/literature/cards/Saito_Rehmsmeier_2015_Precision_Recall_Plot.md`
- Leakage/sampling guardrail: `docs/literature/cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md`
- IEEE-CIS preprocessing and AE family: `docs/literature/cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md`
- AE anomaly framing on IEEE-CIS: `docs/literature/cards/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md`
- Dataset shift limitation: `docs/literature/cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md`
- Broad feature-engineering context only: `docs/literature/cards/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md`

## Do Not

- Do not cite AE-05 or the fixed score ensemble as the active winner after the
  split reset.
- Do not compare old chronological AP directly against new stratified AP.
- Do not use broad UID, velocity, target encoding, rolling windows, SMOTE, or
  stacked ensembles as mainline methods without a new scope decision.
- Do not use `project/` as canonical source; it is local scratch and ignored.
- Do not edit thesis DOCX/PDF claims before the active stratified rerun produces
  a recorded result.

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
