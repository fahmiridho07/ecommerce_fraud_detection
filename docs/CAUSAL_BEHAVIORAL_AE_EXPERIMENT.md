# Causal Behavioral and AE-Signal Experiment

## Literature motivation

Recent IEEE-CIS and related fraud-detection studies emphasize that strong tabular systems often benefit from **temporal context**, **entity aggregations**, **interaction structure**, and **relational signals** rather than from unsupervised feature compression alone. Graph-oriented work similarly suggests that relationships among transactions, cards, addresses, devices, and time carry useful fraud signal.

This experiment implements a **leakage-safe tabular approximation** of that idea: past-only behavioral features keyed on card and contact entities, without graph neural networks, stacking ensembles, or score-level fusion. It then tests whether a single CDV Autoencoder reconstruction-error scalar adds complementary value **after** those behavioral features are present.

## Research questions

1. Do causal behavioral features improve chronological LightGBM validation Average Precision versus original-feature P01?
2. Does CDV reconstruction error (`cdv_ae_reconstruction_mse`) add complementary value beyond causal behavioral features under an identical LightGBM protocol?

## Leakage-safe design

- **Chronological ordering:** 60/20/20 split by `TransactionDT` (unchanged from P01).
- **Past-only statistics:** Each row uses only transactions preceding the current row in deterministic event order (split precedence, `TransactionDT`, `TransactionID`).
- **State continuation:** Train history flows into validation; train + validation history flows into test (online inference simulation).
- **No target history:** `isFraud` never updates behavioral state.
- **No future rows:** Later transactions cannot alter earlier feature values (verified by synthetic immutability test).
- **Deterministic tie-breaking:** Rows with equal `TransactionDT` are ordered by ascending `TransactionID`; lower-ID rows are visible to higher-ID rows at the same timestamp.

Entity keys (3): `card1`, `card1+addr1`, `card1+P_emaildomain`.

Behavioral features (19): count, time-since-previous, historical mean/std amount, amount deviation from mean, and 1h/24h window counts (windows on `card1` and `card1_P_emaildomain` only).

## Experimental models

| ID | Model | Features | AE signal |
|----|-------|----------|-----------|
| **B1** | P01 original-feature LightGBM default | 432 original predictors | None |
| **B2** | Causal behavioral LightGBM default | 432 original + 19 causal behavioral | None |
| **B3** | Causal behavioral + CDV recon error | B2 + `cdv_ae_reconstruction_mse` | Exactly one reconstruction error |

B3 reuses frozen CDV Autoencoder artifacts from `outputs/behavioral_cdv_ae_experiment/autoencoder_cdv_ld128/` (368 inputs: C1–C14, D1–D15, V1–V339). No latent features, no decoder reconstructions, no Optuna tuning.

## Results

### Validation AP (primary interpretation)

| Model | Status | Validation AP | Delta vs B1 | Delta vs corrected B2 |
|-------|--------|---------------|-------------|------------------------|
| B1 (P01) | original reference | 0.602433 | — | — |
| B2 (CBA01) | provisional / superseded | 0.613738 | +0.011305 | — |
| **B2 corrected (CBA01R)** | **corrected authoritative** | **0.615122** | **+0.012689** | — |
| B3 (CBA02) | provisional / superseded | 0.600659 | −0.001774 | −0.013079 vs provisional B2 |
| **B3 corrected (CBA02R)** | **corrected authoritative** | **0.600607** | −0.001826 | **−0.014515 vs CBA01R** |

Evidence: `outputs/final_comparison/causal_behavioral_alignment_correction.csv` (authoritative); `outputs/final_comparison/causal_behavioral_ae_comparison.csv` (legacy archive)

### Descriptive test AP (not used for model selection)

| Model | Test AP |
|-------|---------|
| B1 | 0.485756 |
| B2 | 0.495350 |
| B3 | 0.484615 |

## Feature importance

Behavioral features are actively used in B2. Top behavioral gains include `cb_historical_mean_amount_before_card1` (rank 17), `cb_historical_std_amount_before_card1` (rank 19), and entity-combination amount statistics. Window features (`cb_count_in_previous_24_hours_*`) contribute moderate gain; 1-hour windows contribute less.

In B3, `cdv_ae_reconstruction_mse` ranks **#1 by gain** despite lower validation AP than B2. Behavioral means/stds remain in the top 30.

**High feature importance alone does not prove incremental performance.** CDV reconstruction error shows high gain in B3 but validation AP decreases versus B2.

## Interpretation

### B1 vs B2 corrected (Rule A)

**CBA01R validation AP (0.615122) > B1 validation AP (0.602433).**

**Conclusion:** Identity-aligned causal behavioral features improve chronological validation AP (+0.012689). Legacy CBA01 (+0.011305) is provisional.

### B2 corrected vs B3 corrected (Rule D)

**CBA02R validation AP (0.600607) < CBA01R validation AP (0.615122).**

**Conclusion:** ID-aligned CDV reconstruction error does **not** provide additional validation benefit beyond corrected causal behavioral features (−0.014515 validation AP).

## Historical reference (not directly comparable)

| Model | Validation AP | Caveat |
|-------|---------------|--------|
| EX01 train-static FE | 0.627793 | Train-fitted aggregates, not causal behavioral |
| AE15 FE + CDV recon | 0.635954 | FE-space branch, not B2 + one recon feature |

These rows are included in the comparison CSV for context only.

## Limitations

- Entity proxies (`card1`, `addr1`, `P_emaildomain`) may not uniquely identify users.
- `TransactionDT` is relative, not calendar time.
- Same-timestamp ordering is approximated using `TransactionID`.
- One random seed, one chronological validation block.
- Historical test inspection occurred in prior exploratory branches.
- Reconstruction error is unsupervised MSE on CDV features.
- Tabular entity history approximates relational modeling; this is not a graph model.
- B2 reached max boosting rounds (1999) without early stopping; B3 stopped at iteration 982.

## Alignment correction note

Legacy B2/B3 used global concat/re-sort and positional joins. Audit confirmed 16,309 within-split TransactionID mismatches with 0 split-membership changes. CBA01R/CBA02R correct this via `TransactionID`-keyed restoration. Details: `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md`.

## Stopping rule

Causal behavioral experiment family is closed at CBA02R. LF01 late fusion was executed as an explicit post-TAE01 freeze exception; see `docs/CAUSAL_BEHAVIORAL_AE_LATE_FUSION_EXPERIMENT.md`. Supervisor approval is required before promoting LF01 to a primary thesis model. No additional entities, windows, AE signals, tuning, stacking, GNNs, or Autoencoder architecture changes without approval.

## Artifacts

| Artifact | Path |
|----------|------|
| Feature audit | `docs/CAUSAL_BEHAVIORAL_FEATURE_AUDIT.md` |
| Feature definition | `outputs/causal_behavioral_feature_audit/feature_definition.json` |
| B2 outputs | `outputs/causal_behavioral_lgbm_default/` |
| B3 outputs | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` |
| Comparison CSV | `outputs/final_comparison/causal_behavioral_ae_comparison.csv` |
| Importance table | `outputs/final_comparison/causal_behavioral_feature_importance.csv` |