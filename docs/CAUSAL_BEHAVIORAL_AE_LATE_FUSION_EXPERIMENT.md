# Causal Behavioral and AE–LightGBM Late Fusion

## Motivation

Feature-level Autoencoder integration repeatedly failed to improve chronological validation Average Precision versus the original-feature LightGBM baseline. Latent replacement, latent augmentation, broader selected-numerical replacement, decoder reconstruction, task-aware latent learning, and CDV reconstruction error after causal behavioral features all remained below stronger references under the executed protocol.

CDV reconstruction error (CBA02R) degraded the identity-aligned causal behavioral model. However, historical score-level ensembles in exploratory branches showed a small positive signal, suggesting that Autoencoder-derived predictions may capture a complementary ranking perspective when kept as an independent expert and fused at decision level—without altering the stronger causal behavioral feature space.

Late fusion preserves CBA01R as the primary behavioral expert while allowing the frozen P04 V-only AE–LightGBM expert to contribute at the probability level.

## Research question

Does a frozen V-only AE–LightGBM expert provide complementary fraud-ranking information that improves the identity-aligned causal behavioral LightGBM through validation-selected score-level fusion?

## Expert models

| Expert | ID | Features | LightGBM | Validation AP | Test AP (descriptive) |
|--------|-----|----------|----------|---------------|----------------------|
| CBA01R | CBA01R | 432 original + 19 identity-aligned causal behavioral | Default | 0.615122 | 0.493838 |
| P04 | P04 | 93 non-V + 128 LD128 latent (V replaced) | Optuna tuned | 0.610631 | 0.490686 |

Artifacts: `outputs/causal_behavioral_lgbm_id_aligned/`, `outputs/optuna/ae_lgbm_ld128/`, `outputs/autoencoder_robust_ld128/`.

## Identity-safe score generation

- Chronological 60/20/20 split by `TransactionDT` recreated from labeled train data.
- CBA01R scores regenerated from frozen `model.pkl` and `preprocessing.pkl` with identity-safe causal behavioral feature restoration by `TransactionID`.
- P04 scores regenerated from frozen `final_model.pkl`, `preprocessing_non_v.pkl`, and LD128 latent arrays tied to the same split row order.
- Expert scores aligned by `TransactionID` join only; no positional concatenation.
- Reference validation/test AP reproduced within tolerance before fusion.
- Neither expert retrained.

## Complementarity audit

Validation-only audit (`src/audit_causal_behavioral_ae_complementarity.py`):

| Metric | Value |
|--------|-------|
| Spearman correlation | 0.744781 |
| Pearson correlation | 0.893622 |
| Complementarity classification | moderate |

Top-k unique fraud captures (validation):

| Top fraction | Only CBA01R | Only P04 |
|--------------|-------------|----------|
| 1% | 140 | 126 |
| 3% | 282 | 225 |
| 5% | 281 | 312 |

P04 finds fraud cases missed by CBA01R at each top-k cutoff (e.g., 126 at top 1%, 312 at top 5%). Disagreement analysis: 312 fraud cases ranked in P04 top 5% but outside CBA01R top 5%.

Evidence: `outputs/causal_behavioral_ae_late_fusion/complementarity_summary.json`, `results/late_fusion_complementarity_summary.json`.

## Fusion design

**Formula:**

```
fusion_score = behavioral_weight × cba01r_score + ae_weight × p04_ae_score
```

where `ae_weight = 1 − behavioral_weight`.

**Weight grid (validation-only):** 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00.

**Selection metric:** highest validation Average Precision.

**Tie-break:** prefer larger behavioral weight when AP values differ by less than 1e-8.

**Practical improvement criterion:** validation delta ≥ 0.002 versus CBA01R for meaningful contribution.

**Threshold:** MCC with F1 tie-break on validation fusion scores only.

## Results

### Validation (primary)

| Model | Validation AP | Delta vs CBA01R | Delta vs P04 | Delta vs P02 |
|-------|---------------|-----------------|--------------|--------------|
| CBA01R | 0.615122 | — | +0.004491 | −0.008950 |
| P04 | 0.610631 | −0.004491 | — | −0.013441 |
| P02 | 0.624072 | +0.008950 | +0.013441 | — |
| **LF01 (selected)** | **0.629600** | **+0.014478** | **+0.018969** | **+0.005528** |

**Selected weights:** behavioral_weight = 0.50, ae_weight = 0.50.

**Practical result category:** strong success (ae_weight > 0; fusion validation AP ≥ CBA01R + 0.002; fusion validation AP > P02).

Evidence: `outputs/final_comparison/causal_behavioral_ae_late_fusion_weight_search.csv`, `outputs/causal_behavioral_ae_late_fusion/frozen_fusion_config.json`.

### Test (descriptive only)

| Metric | Value |
|--------|-------|
| Average Precision | 0.505543 |
| ROC-AUC | 0.891568 |
| Precision (threshold 0.37) | 0.656000 |
| Recall | 0.403543 |
| F1 | 0.499695 |
| MCC | 0.501472 |

Test did not influence weight or threshold selection.

## Bootstrap uncertainty

Paired bootstrap (1,000 resamples, seed 42):

| Split | Comparison | Mean delta | 2.5% CI | 97.5% CI | P(delta > 0) |
|-------|------------|------------|---------|----------|--------------|
| Validation | fusion − CBA01R | +0.014469 | +0.011452 | +0.017301 | 1.000 |
| Validation | fusion − P04 | +0.018949 | +0.014899 | +0.022898 | 1.000 |
| Validation | fusion − P02 | +0.005547 | +0.001339 | +0.009751 | 0.996 |
| Test | fusion − CBA01R | +0.011694 | +0.008383 | +0.014970 | 1.000 |
| Test | fusion − P04 | +0.014933 | +0.010787 | +0.019022 | 1.000 |
| Test | fusion − P02 | +0.004224 | −0.000597 | +0.009064 | 0.963 |

Evidence: `outputs/causal_behavioral_ae_late_fusion/paired_bootstrap_summary.json`.

## Industry interpretation

The measured validation gain (+0.014478 AP vs CBA01R, +0.005528 vs P02) suggests that the AE expert provides complementary ranking information not fully captured by causal behavioral features alone. Operationally, late fusion requires serving two frozen models, doubling inference latency and maintenance surface relative to CBA01R alone. The gain may justify production complexity when recall at fixed review capacity is prioritized, but deployment cost-benefit analysis remains context-specific.

## Limitations

- One chronological validation block reused for weight and threshold selection.
- Mixed tuning status: CBA01R uses default LightGBM; P04 uses Optuna-tuned LightGBM.
- P04 uses LD128 while thesis-original P03 uses LD32.
- Simple linear probability fusion without calibration.
- One random seed (42).
- Historical repeated test inspection in prior exploratory branches.
- No external dataset or online deployment evaluation.
- LF01 executed as an explicit researcher-directed exception to the post-TAE01 freeze; supervisor approval required before promoting to a primary thesis model.

## Final conclusion

Under the predefined practical success rule, LF01 achieves **strong success**: validation-selected 50/50 late fusion improves validation AP meaningfully above CBA01R (+0.014478) and above P02 (+0.005528), with moderate expert complementarity and paired-bootstrap support on validation. Test metrics are reported descriptively only.

## Artifacts

| Artifact | Path |
|----------|------|
| Fusion outputs | `outputs/causal_behavioral_ae_late_fusion/` |
| Weight search | `outputs/final_comparison/causal_behavioral_ae_late_fusion_weight_search.csv` |
| Comparison table | `outputs/final_comparison/causal_behavioral_ae_late_fusion_comparison.csv` |
| Trackable summary | `results/causal_behavioral_ae_late_fusion.csv` |
| Manifest | `results/causal_behavioral_ae_late_fusion_manifest.json` |
| Scripts | `src/run_causal_behavioral_ae_late_fusion.py`, `src/audit_causal_behavioral_ae_complementarity.py` |