# Preprocessing Decision Gate

Status: archived chronological evidence note, superseded by
`docs/STRATIFIED_SPLIT_RESET.md` and
`docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`.

## Decision

The previous `frequency_missingness_time_amount` branch is no longer an active
thesis-facing preprocessing protocol. It remains useful empirical evidence from
the old chronological exploration, but it must be rerun or redesigned before it
can support the new stratified thesis claim.

Historical result block:

| Model | Old protocol | Test AP | Status |
|-------|--------------|--------:|--------|
| P02 tuned baseline | Chronological | 0.504900 | Archived historical control |
| Enhanced identity/device baseline | Chronological | 0.516590 | Archived diagnostic |
| Best preprocessing baseline | Chronological | 0.524197 | Archived diagnostic |
| Fixed 0.50 score ensemble | Chronological | 0.529114 | Archived diagnostic candidate |

Do not mix these numbers with new stratified rerun tables.

## Why Archived

The branch combined useful ideas, but not all of them have equally direct paper
anchors:

- frequency encoding has support from Alharbi et al. (2026);
- time/amount features and broad feature engineering are closer to Moradi et al.
  (2025), but the old branch was narrower than Moradi's pipeline;
- identity/device normalization, rare bucketing, and compact missingness summary
  features are project diagnostics;
- preserving numeric `NaN` for LightGBM differs from Alharbi-style median
  imputation and z-score scaling.

After the methodology reset, the active preprocessing path is A1
Alharbi-style preprocessing under stratified holdout.

## Active Replacement

Use this flow for the next thesis-facing preprocessing branch:

```text
raw -> clean -> stratified split -> fit preprocessing on train only
    -> transform validation/test -> train/evaluate model
```

Active branch definition is maintained in:

```text
docs/PAPER_ANCHORED_PREPROCESSING_RESET.md
```

## Reproducibility Note

To reproduce old chronological evidence, pass:

```bash
--split-strategy chronological
```

to split-aware scripts. The default is now `stratified_holdout`.

## Source Documents

- `docs/STRATIFIED_SPLIT_RESET.md`
- `docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`
- `archive/docs/chronological_evidence/PREPROCESSING_DIAGNOSTIC.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `archive/docs/chronological_evidence/FINAL_CANDIDATE_VALIDATION.md`
