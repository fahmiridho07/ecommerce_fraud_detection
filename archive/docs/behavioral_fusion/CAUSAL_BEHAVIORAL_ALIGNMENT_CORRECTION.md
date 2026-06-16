# Causal Behavioral Row-Alignment Correction

## Detected risk

The chronological 60/20/20 split orders rows by `TransactionDT` only (`src/splitting.py`). The legacy causal behavioral generator then:

1. concatenated train, validation, and test;
2. re-sorted globally by `TransactionDT` and `TransactionID`;
3. recovered split slices using row counts only;
4. joined raw features, labels, and behavioral features by positional `reset_index()` concatenation in the training scripts.

When duplicate `TransactionDT` values exist within a split, the global re-sort can permute rows relative to the frozen split's original `TransactionID` order. Training then risked pairing raw features for TransactionID A with behavioral features computed for TransactionID B and label for TransactionID A.

This is a **structural alignment risk**. The audit below confirms mismatches under the legacy procedure; it does not prove every historical metric was corrupted, but provisional CBA01/CBA02 results cannot be treated as authoritative until identity-safe reruns exist.

## Audit evidence

Source: `outputs/causal_behavioral_alignment_audit/pre_fix_alignment_report.json`

| Metric | Value |
|--------|-------|
| Pre-fix mismatched row positions (all splits) | **16,309** |
| Pre-fix split-membership changes | **0** |
| Duplicate timestamp rows (full dataset) | **33,932** |
| Train/validation boundary timestamp ties | **0** |
| Validation/test boundary timestamp ties | **0** |
| Same-timestamp ID order changes (train) | 10,964 |
| Same-timestamp ID order changes (validation) | 2,948 |
| Same-timestamp ID order changes (test) | 2,615 |
| Corrected mismatch count | **0** |
| Corrected split-membership change count | **0** |

**Classification:** confirmed within-split ordering mismatch; no split-boundary membership movement observed.

Representative examples (`outputs/causal_behavioral_alignment_audit/pre_fix_mismatch_examples.csv`):

| split | position | original TransactionID | legacy TransactionID |
|-------|----------|------------------------|----------------------|
| train | 46 | 2987047 | 2987046 |
| train | 47 | 2987046 | 2987047 |
| train | 48 | 2987049 | 2987048 |

These are adjacent swaps at equal `TransactionDT` values.

## Corrected identity policy

**Frozen split membership** from `chronological_split()` is never changed.

**Deterministic event order:**

1. train split first;
2. validation split second;
3. test split third;
4. within each split: `TransactionDT` ascending, then `TransactionID` ascending.

**State continuation:** train starts from empty state; validation continues from completed train; test continues from completed train + validation. Validation/test labels never update behavioral state.

**Feature restoration:** behavioral values are keyed by `TransactionID` and restored to each split's exact input row order before joining raw features or labels.

**Assertions enforced:** unique IDs per split, no cross-split ID overlap, one-to-one ID coverage, restored order equals input order, row counts match.

Implementation: `src/causal_behavioral_features.py`.

## Same-timestamp interpretation

Each feature uses only transactions **preceding the current row in the deterministic event order** defined by split precedence, `TransactionDT`, and `TransactionID`.

- `TransactionDT` has coarse resolution; many rows share timestamps.
- `TransactionID` tie-breaking within a split is an approximation of true event order.
- Across split boundaries, split precedence is authoritative (train → validation → test).

Do not describe the policy as "strictly earlier timestamps only."

## Corrected results

Validation AP is primary. Test AP is descriptive only.

| Model | Status | Validation AP | Δ vs P01 | Test AP (descriptive) |
|-------|--------|---------------|----------|------------------------|
| P01 | original reference | 0.602433 | — | 0.485756 |
| CBA01 (B2) | provisional / superseded | 0.613738 | +0.011305 | 0.495350 |
| **CBA01R (B2 corrected)** | **corrected authoritative** | **0.615122** | **+0.012689** | 0.493838 |
| CBA02 (B3) | provisional / superseded | 0.600659 | −0.001774 | 0.484615 |
| **CBA02R (B3 corrected)** | **corrected authoritative** | **0.600607** | −0.001826 | 0.483831 |

Evidence: `outputs/final_comparison/causal_behavioral_alignment_correction.csv`

CDV reconstruction errors were regenerated with `TransactionID` keys from the frozen CDV Autoencoder. Legacy positional CDV arrays were numerically identical when splits were not reordered (`max_abs_diff = 0.0` for all splits), so B3 metric movement is driven primarily by corrected behavioral alignment.

## Effect on previous conclusions

**CBA01 improvement vs P01:** remains positive after correction. Validation AP increases from +0.011305 (provisional) to +0.012689 (corrected).

**CBA02 CDV contribution vs corrected B2:** remains negative. Validation AP delta vs CBA01R is −0.014515 (corrected), compared with −0.013079 (provisional vs provisional B2).

## Limitations

- `TransactionDT` resolution is coarse; `TransactionID` ordering is a tie-break approximation.
- Split membership at timestamp boundaries is frozen; no global re-sort across boundaries.
- One random seed (`42`), one chronological validation block.
- Historical test inspection occurred in prior exploratory branches.
- CDV Autoencoder was not retrained; only error re-keying was performed.
- Late fusion remains blocked pending supervisor approval.

## Artifacts

| Artifact | Path |
|----------|------|
| Pre-fix audit report | `outputs/causal_behavioral_alignment_audit/pre_fix_alignment_report.json` |
| Mismatch examples | `outputs/causal_behavioral_alignment_audit/pre_fix_mismatch_examples.csv` |
| Original split ID manifest | `outputs/causal_behavioral_alignment_audit/original_split_id_manifest.json` |
| ID-aligned CDV errors | `outputs/causal_behavioral_alignment_audit/cdv_reconstruction_error_*.csv` |
| CBA01R outputs | `outputs/causal_behavioral_lgbm_id_aligned/` |
| CBA02R outputs | `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/` |
| Corrected comparison | `outputs/final_comparison/causal_behavioral_alignment_correction.csv` |
| Trackable summary | `results/causal_behavioral_alignment_correction.csv` |