# EDA and Methodology Audit

Status: active audit, 2026-06-18. Root-to-tip review of the experiment pipeline
against current fraud-detection research and industry best practice.

This document records (1) the exploratory data facts the thesis should report and
(2) a methodology audit with prioritized fixes. It complements
`AE_BASELINE_GAP_DIAGNOSIS.md` and `AE_INTEGRATION_EXPERIMENT_RESULTS.md`.

## Dataset Facts (IEEE-CIS, labeled train only)

The Kaggle `test_transaction.csv` is unlabeled, so all experiments correctly use
the labeled `train_transaction.csv` (+ `train_identity.csv`, left join on
`TransactionID`) and split it internally. This matches standard academic use of
IEEE-CIS.

| Property | Value |
|----------|-------|
| Columns after merge | 434 (433 features + `isFraud`) |
| Overall fraud rate | ~3.5% (class imbalance ~27:1) |
| Object-dtype categorical columns | 31 (ProductCD, card4/6, emaildomains, M1-M9, id_12.., DeviceType/Info) |
| Numeric-coded categoricals treated as numeric | card1 (7677 uniq), card2/3/5, addr1/2 |
| Missingness | 359/434 columns have NaN; 232 columns > 50% missing; 12 columns > 90% missing |
| Time span (`TransactionDT`) | ~6 months continuous |
| Full-row duplicates (excl. ID) | none observed |
| Entity reuse | ~97% of rows have a `card1` value that appears more than once |

These facts belong in the thesis Bab 4 EDA section. The entity-reuse figure is
the single most important one because it explains the split behaviour below.

## Audit by Pipeline Stage

### Data loading and merging - OK

`data_loader.py` left-joins transaction and identity on `TransactionID`, checks
for duplicate IDs and missing targets. Correct and standard.

### Preprocessing and cleaning - mostly OK, one gap

- All imputers, encoders, scalers, and frequency maps are fitted on the train
  split only; unseen validation/test categories map to a safe value
  (`preprocessing.py`, `paper_preprocessing.py`). This is correct leakage
  control and is done better than in many comparable theses.
- Gap: `get_categorical_columns` (`preprocessing.py:38`) detects only
  object/category dtypes, so the numeric-coded categoricals card1-6 and addr1-2
  are treated as continuous numbers by the A0 baseline and the AE. card1 has
  7677 unique values; treating it as continuous is suboptimal. The
  frequency-encoding branches (`paper_preprocessing.py`, `enhanced_preprocessing.py`)
  fix this. Implication: the A0 baseline is a clean RAW control, not the
  strongest baseline. This is fine for isolating the AE contribution, but the
  thesis should either state A0 is the raw control or also test the AE on the
  strongest preprocessing baseline.

### Splitting - leakage-aware code, optimistic protocol

- Stratified holdout 60/20/20, fixed seed, train-only fitting
  (`splitting.py:181`). The split mechanics are correct and reproducible.
- The protocol choice is optimistic: with ~97% entity reuse, the same card and
  address entities appear in train and test, so the model memorises per-entity
  fraud propensity. This is why stratified PR-AUC (~0.82-0.86) is far above
  chronological PR-AUC (~0.50). The IEEE-CIS competition itself used a temporal
  split; Dal Pozzolo et al. (2018) and Lucas et al. (2019) argue for time-aware
  evaluation. The repository already documents this as a limitation. It must be
  disclosed precisely, and is strengthened by an entity-aware / temporal
  robustness check (below).

### Cross-validation - missing

- `stratified_kfold_splits` is defined (`splitting.py:235`) but never used by any
  training script. All results come from a single holdout split. The paired
  bootstrap captures test-set resampling variance but not split variance.
- Fix (highest-value): repeated stratified holdout (multiple split seeds) or
  stratified k-fold for the headline AE-vs-baseline comparison, reporting
  mean +/- std and the fraction of splits with a positive delta. Implemented in
  `src/run_repeated_split_validation.py`.

### Class imbalance and metrics - strong

- scale_pos_weight computed from train labels only; no pre-split resampling
  (Kabane & Ouali 2024 guardrail). PR-AUC / Average Precision is the primary
  metric (Saito & Rehmsmeier 2015), with ROC-AUC, F1, MCC, confusion matrices,
  and validation-only threshold selection. Paired bootstrap for significance is
  above typical S1 rigor. Keep as is.

### Reproducibility - strong

- Fixed seed 42, train-only fitting, saved run configs and split manifests. Keep.

## Documentation Inconsistency Fixed

`notebooks/notebook.ipynb` is now treated as a legacy chronological-era notebook.
Its first markdown cell has been updated with a warning that active Bab 4 figures
must come from `src/generate_thesis_figures.py` and
`outputs/stratified_reset/thesis_figures/`. The active protocol is stratified
holdout, with temporal evaluation used as robustness/limitation evidence.

## Prioritized Actions

1. Repeated stratified holdout (split-variance) for the headline comparison.
   Harness: `src/run_repeated_split_validation.py`.
2. Entity-aware / temporal robustness of the AE augmentation (does the gain hold
   under harder evaluation). Run `src/run_ae_augmentation_experiment.py
   --split-strategy chronological`.
3. Test the AE augmentation on the strongest preprocessing baseline
   (frequency-encoded A1/enhanced), not only the raw A0 control.
4. Add a documented EDA section and reconcile the stale temporal-split narrative.
5. Consider dropping or flagging raw `TransactionDT` as a feature under random
   splitting (temporal-position proxy).

Actions 1-4 are complete in the active loop. Action 5 remains optional future
work because the thesis already discloses TransactionDT/entity leakage as a
central limitation.

## TransactionDT and Entity-Leakage Evidence (rec #5, concrete)

In the staging baseline feature importance (488 features), the top features by
gain are:

| Rank | Feature | Note |
|-----:|---------|------|
| 1 | V258 | Vesta engineered |
| 2 | card1_train_count | per-card frequency = entity memorisation |
| 3 | TransactionAmt | amount |
| 4 | card1 | card identity |
| 5 | C14 | count feature |
| 6 | TransactionDT | raw timestamp = temporal-position proxy |

Two of the top six features are entity/time identifiers (`card1_train_count`,
`card1`, `TransactionDT`). Under stratified random splitting, transactions from
the same card and from nearby timestamps appear in both train and test, so these
features let the model memorise per-card and per-time-neighbourhood fraud
propensity. This quantifies why stratified PR-AUC (~0.82-0.86) is far above
chronological (~0.50): the lift is partly genuine signal and partly
entity/time memorisation that will not generalise to future, unseen cards.

Recommended handling for the thesis (defensible):
- Keep stratified as the active protocol but disclose this explicitly as the
  central limitation.
- Optionally report a `TransactionDT`-dropped ablation and/or a GroupKFold-by-card
  robustness run to show how much performance is memorisation.
- The temporal/chronological results already produced (augmentation still helps,
  +0.0176) are the realistic-protocol robustness evidence.

## Stale Narrative Reconciliation (rec #4, done)

`notebooks/notebook.ipynb` cell 1 ("Best Practice Yang Diikuti") and cell 19
(literature accountability `literature_rows`) were updated from the stale
chronological assertion ("Split temporal, bukan random split" /
"Chronological split using TransactionDT") to the active protocol: stratified
holdout is primary, temporal evaluation is a robustness/limitation. Thesis Bab 3
prose must match this.
