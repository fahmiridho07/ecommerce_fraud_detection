# Diagnosis: Why Ding-Style AEELG Drops on IEEE-CIS

Status: supervisor-facing diagnosis, 2026-06-19.

Purpose: respond to supervisor feedback:

```text
Stick to the original proposal first. If results are not as good as expected,
analyze the causes. From the identified causes, find alternatives to improve the
method while staying aligned with the original objective.
```

## Context

The original methodological anchor is Ding et al. (2024):

```text
SMOTE -> AutoEncoder feature reconstruction -> LightGBM
```

Sanity check on Ding's ULB anchor dataset passed:

| Dataset | Result |
|---|---|
| ULB credit-card dataset | Ding reconstructed-original reaches ROC-AUC 0.967585 and F1 0.800000, close to Ding's reported AUC 0.9683 and F-measure 0.8027. |

Then the same idea was transferred to IEEE-CIS using dense A1 preprocessing.

Strict Ding-style IEEE-CIS run:

```bash
python src/run_ding_ieeecis_strict.py \
  --output-dir outputs/stratified_reset/ding_strict_ieee_cis \
  --target-fraud-rate 0.50 \
  --n-estimators 800 \
  --ae-epochs 30 \
  --n-bootstrap 1000
```

## Key Result

Selected-threshold metrics:

| Method | AP | ROC-AUC | Precision | Recall | F1 | MCC | BCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ding GOSS baseline | 0.785897 | 0.957323 | 0.881088 | 0.634648 | 0.737834 | 0.740379 | 0.815771 |
| Ding SMOTE + GOSS | 0.767718 | 0.953987 | 0.877335 | 0.602226 | 0.714204 | 0.719117 | 0.799586 |
| Ding reconstructed original + GOSS | 0.413661 | 0.834660 | 0.794791 | 0.243649 | 0.372963 | 0.430557 | 0.620684 |
| Ding reconstructed balanced + GOSS | 0.387281 | 0.830891 | 0.706838 | 0.252601 | 0.372193 | 0.411428 | 0.624401 |

Bootstrap AP deltas:

| Comparison | Delta AP | 95% CI | p(delta <= 0) |
|---|---:|---:|---:|
| SMOTE + GOSS vs GOSS baseline | -0.018179 | [-0.021843, -0.014740] | 1.000 |
| Reconstructed original + GOSS vs GOSS baseline | -0.372236 | [-0.385188, -0.359032] | 1.000 |
| Reconstructed balanced + GOSS vs GOSS baseline | -0.398616 | [-0.411799, -0.384742] | 1.000 |

The drop is therefore not caused by using a thesis-specific LightGBM helper. It
remains after switching to Ding-like GOSS LightGBM.

## Cause 1 - AE Reconstruction Damages Fraud Ranking

In fraud detection, ranking is critical. If fraud rows are no longer ranked near
the top, AP drops even if a threshold can still catch some fraud.

Top-k fraud capture on the IEEE-CIS test set:

| Method | Top 1% fraud captured | Precision@1% | Recall@1% | Top 5% fraud captured | Precision@5% | Recall@5% |
|---|---:|---:|---:|---:|---:|---:|
| Ding GOSS baseline | 1168 | 0.988992 | 0.282603 | 3248 | 0.550042 | 0.785870 |
| Ding SMOTE + GOSS | 1168 | 0.988992 | 0.282603 | 3173 | 0.537341 | 0.767723 |
| Reconstructed original + GOSS | 961 | 0.813717 | 0.232519 | 1897 | 0.321253 | 0.458989 |
| Reconstructed balanced + GOSS | 904 | 0.765453 | 0.218727 | 1825 | 0.309060 | 0.441568 |

When reviewing the same number of rows as the number of fraud cases in the test
set (`k = 4133`):

| Method | Fraud captured | Precision@k | Recall@k |
|---|---:|---:|---:|
| Ding GOSS baseline | 3003 | 0.726591 | 0.726591 |
| Ding SMOTE + GOSS | 2937 | 0.710622 | 0.710622 |
| Reconstructed original + GOSS | 1690 | 0.408904 | 0.408904 |
| Reconstructed balanced + GOSS | 1630 | 0.394387 | 0.394387 |

Interpretation:

```text
The AE-reconstructed representation loses many fraud rows that the original A1
LightGBM ranks highly. This is a ranking failure, not merely a bad threshold.
```

## Cause 2 - SMOTE Hurts Slightly, But AE Reconstruction Is the Main Damage

SMOTE-only has a small negative effect:

```text
AP baseline      = 0.785897
AP SMOTE + GOSS  = 0.767718
Delta            = -0.018179
```

AE reconstruction has a much larger negative effect:

```text
AP reconstructed original = 0.413661
Delta vs baseline         = -0.372236

AP reconstructed balanced = 0.387281
Delta vs baseline         = -0.398616
```

Therefore:

```text
Full-balance SMOTE is not ideal for IEEE-CIS, but it is not the primary reason
for failure. The major performance loss appears after replacing the A1 feature
matrix with AE-reconstructed features.
```

## Cause 3 - Reconstruction Changes the Score Ordering

Spearman rank correlation with the strong Ding GOSS baseline:

| Score pair | Spearman correlation |
|---|---:|
| Baseline vs SMOTE + GOSS | 0.8892 |
| Baseline vs reconstructed original | 0.4801 |
| Baseline vs reconstructed balanced | 0.4838 |
| Reconstructed original vs reconstructed balanced | 0.9598 |

Interpretation:

```text
SMOTE-only keeps a ranking similar to the baseline. Both AE-reconstructed
variants produce a very different ranking, and they are similar to each other.
This points to the reconstruction representation itself as the main cause.
```

Baseline top-fraud retention:

| Baseline top-k | Fraud in baseline top-k | Retained by SMOTE | Retained by reconstructed original | Retained by reconstructed balanced |
|---:|---:|---:|---:|---:|
| 1000 | 994 | 852 (85.7%) | 610 (61.4%) | 558 (56.1%) |
| 1181 (top 1%) | 1168 | 1020 (87.3%) | 733 (62.8%) | 682 (58.4%) |
| 4133 | 3003 | 2846 (94.8%) | 1635 (54.4%) | 1577 (52.5%) |

## Cause 4 - Score Distribution Becomes Less Separable

Score quantiles on the test set:

| Method | Class | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| Ding GOSS baseline | non-fraud | 0.014343 | 0.004570 | 0.054385 | 0.166404 |
| Ding GOSS baseline | fraud | 0.558340 | 0.642304 | 0.997084 | 0.999412 |
| Reconstructed original + GOSS | non-fraud | 0.025979 | 0.015708 | 0.081809 | 0.209950 |
| Reconstructed original + GOSS | fraud | 0.267253 | 0.081870 | 0.964688 | 0.989007 |
| Reconstructed balanced + GOSS | non-fraud | 0.326595 | 0.315847 | 0.695846 | 0.854452 |
| Reconstructed balanced + GOSS | fraud | 0.659021 | 0.677892 | 0.992938 | 0.995006 |

Interpretation:

- In the baseline, fraud median score (`0.642304`) is far above non-fraud P99
  (`0.166404`).
- In reconstructed-original, fraud median (`0.081870`) is almost the same as
  non-fraud P95 (`0.081809`), so many fraud rows overlap with high-scoring
  non-fraud rows.
- In reconstructed-balanced, many non-fraud rows get large scores; at default
  threshold 0.5, precision falls to 0.119919 despite recall 0.687394.

This explains why threshold tuning cannot fully fix the problem.

## Cause 5 - AE Optimizes Reconstruction, Not Fraud Discrimination

The AE training itself is not collapsing:

| Run | First validation loss | Best validation loss | Best epoch |
|---|---:|---:|---:|
| Ding-strict IEEE-CIS | 0.594488 | 0.313106 | 30 |
| Earlier Ding-style IEEE-CIS | 0.602851 | 0.310165 | 29 |

The AE improves reconstruction loss, but downstream AP remains low. Therefore:

```text
A lower reconstruction MSE does not mean the reconstructed features preserve
fraud-discriminative information needed by LightGBM.
```

This matches the conceptual risk in the Ding paper itself: AutoEncoder is
unsupervised and not directly designed for classification. On IEEE-CIS, the
unsupervised reconstruction objective appears to smooth or distort signals that
LightGBM uses directly.

## Cause 6 - Dataset Mismatch Between Ding's Anchor and IEEE-CIS

| Aspect | Ding ULB anchor | IEEE-CIS adaptation |
|---|---|---|
| Feature type | Dense numeric PCA-like features | Mixed numeric/categorical, converted to dense A1 |
| Feature count | 30 after Time removal/Hour addition | 432 dense A1 features |
| Missingness | Minimal in public ULB data | Extensive, imputed/encoded |
| Categorical features | None in ULB | Frequency-encoded categoricals |
| SMOTE target | Balanced training set | 329,528 synthetic fraud rows added |

Implication:

```text
Ding-style full-matrix reconstruction is reasonable on a small dense numeric
dataset, but becomes risky on IEEE-CIS because the reconstructed matrix must
preserve many sparse, missingness-derived, and frequency-encoded signals.
```

## Diagnosis Summary

The most evidence-supported cause is:

```text
Replacing the full IEEE-CIS A1 feature matrix with AE-reconstructed features
destroys part of the fraud ranking signal learned by LightGBM.
```

Secondary causes:

1. Full-balance SMOTE slightly harms IEEE-CIS compared with no-SMOTE baseline.
2. AE reconstruction MSE is not aligned with fraud classification.
3. IEEE-CIS is much wider and more heterogeneous than Ding's ULB anchor data.
4. The problem is not merely threshold selection, because AP and top-k ranking
   also drop sharply.

## Improvement Alternatives That Still Match the Original Objective

These alternatives do not jump to an unrelated method. They keep the original
goal: AutoEncoder + LightGBM for fraud detection.

### Alternative A - Keep Original Features, Add AE Signals

Problem addressed: full replacement loses LightGBM's useful original signals.

Fix:

```text
Original A1 features + AE reconstruction error / latent features -> LightGBM
```

Decision rule:

- Compare against the same A1 LightGBM baseline.
- Promote only if AP improves and paired-bootstrap delta is positive.

### Alternative B - AE as Latent-Space Oversampler, Not Feature Replacement

Problem addressed: Ding's reconstruction damages ranking, while the thesis still
wants AE to help with class imbalance.

Fix:

```text
Train AE on dense fraud features
-> generate/interpolate minority samples in latent space
-> decode synthetic fraud rows
-> train LightGBM on original A1 feature space plus synthetic rows
```

Why this follows from the diagnosis:

- LightGBM still sees the A1 representation instead of a fully reconstructed
  matrix.
- AE contributes to minority generation, not full feature replacement.
- This remains within the AE + LightGBM family.

### Alternative C - Reduce SMOTE Aggressiveness

Problem addressed: full 50/50 balance slightly hurts even without AE.

Fix:

```text
Compare target fraud rates such as 0.05, 0.10, 0.15, and 0.50.
```

Decision rule:

- Use matched controls: SMOTE-only vs AE-based augmentation at the same target
  fraud rate.
- Do not promote AE unless it beats the matched SMOTE-only control.

### Alternative D - Partial Reconstruction Only

Problem addressed: reconstructing all 432 A1 features may distort too many
important signals.

Fix:

```text
Reconstruct only selected numeric/high-missingness blocks, keep all other
features original.
```

This is closer to the original proposal's intuition that AE should help with
complex feature representations, without forcing it to replace the whole table.

## Supervisor-Facing Explanation

Suggested wording:

```text
Saya sudah mengikuti masukan untuk kembali ke usulan awal. Pertama, implementasi
Ding-style diuji pada dataset asal Ding dan hasilnya mendekati paper. Setelah itu
saya terapkan ke IEEE-CIS dengan adaptasi yang perlu karena IEEE-CIS mixed-type.

Hasilnya turun bukan hanya karena threshold atau parameter LightGBM. Pada run
yang lebih dekat ke Ding, baseline GOSS mendapat AP 0,7859, sedangkan AE
reconstruction hanya 0,4137 atau 0,3873. Top-k analysis menunjukkan baseline
menangkap 3003 fraud ketika mengambil 4133 transaksi teratas, sedangkan
reconstruction hanya menangkap 1690 atau 1630. Jadi masalah utamanya adalah
rekonstruksi AE mengubah ranking fraud yang sebelumnya bagus.

Penyebab paling mungkin adalah full feature reconstruction tidak mempertahankan
sinyal diskriminatif IEEE-CIS yang heterogen dan banyak missingness. SMOTE juga
sedikit menurunkan performa, tetapi penurunan terbesar muncul saat fitur asli
diganti dengan fitur rekonstruksi AE.

Alternatif perbaikannya tetap dalam tujuan awal AE + LightGBM: AE tidak dipakai
untuk mengganti seluruh fitur, tetapi sebagai sinyal tambahan atau sebagai
latent-space oversampler, lalu dibandingkan secara adil dengan baseline dan
SMOTE-only.
```
