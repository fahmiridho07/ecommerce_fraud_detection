# Ding-Style AEELG Applied to IEEE-CIS

Status: supervisor-facing transfer result, 2026-06-19.

Purpose: answer the supervisor's question after the Ding anchor sanity check:

```text
If the Ding-style method works on the anchor dataset, what happens when the same
method is applied to IEEE-CIS?
```

Short answer:

```text
The Ding-style AE reconstruction pipeline can reproduce paper-level behavior on
ULB, but it transfers poorly to IEEE-CIS. On IEEE-CIS, direct AE-reconstructed
features lose far below a matched dense LightGBM baseline.
```

## Method Applied

Anchor recipe from Ding et al. (2024):

```text
standardize data
-> split train/test
-> SMOTE on training data
-> train AutoEncoder on balanced training data
-> reconstruct train/test features
-> train LightGBM on reconstructed features
```

IEEE-CIS adaptation:

- IEEE-CIS is mixed numeric/categorical with heavy missingness, unlike Ding's
  dense numeric ULB/Santander datasets.
- Therefore, IEEE-CIS is first converted into dense numeric A1 features:
  categorical frequency encoding, numeric median imputation, and z-score
  scaling.
- All preprocessing statistics are fit on train only.
- SMOTE-style interpolation is applied to train only.
- Validation/test are never oversampled.
- The AE reconstructs the dense A1 matrix.
- LightGBM is evaluated on the reconstructed validation/test matrices.

## Command

The full run artifact already exists at:

```text
outputs/stratified_reset/ding_aieelg_ieee_cis/
```

Reproducible command:

```bash
python archive/source/ae_appendix/run_ding_aieelg_experiment.py \
  --output-dir outputs/stratified_reset/ding_aieelg_ieee_cis \
  --target-fraud-rate 0.50 \
  --n-estimators 800 \
  --ae-epochs 30 \
  --n-bootstrap 1000
```

## Run Setup

| Item | Value |
|---|---:|
| Dataset | IEEE-CIS labeled train set |
| Split | stratified holdout |
| Train/validation/test | 60/20/20 |
| Seed | 42 |
| Test prevalence | 0.03499 |
| Dense A1 feature count | 432 |
| SMOTE target fraud rate | 0.50 |
| Synthetic fraud rows | 329,528 |
| AE architecture | 432 -> 256 -> 128 -> 128 -> 432 |
| AE output activation | linear |
| AE epochs | 30 |
| AE best validation MSE | 0.310165 |
| LightGBM trees | 800 |
| Bootstrap repeats | 1000 |

## Result

| Arm | Test AP / PR-AUC | ROC-AUC | F1 | MCC | Delta AP vs baseline |
|---|---:|---:|---:|---:|---:|
| A1 dense LightGBM baseline | 0.746013 | 0.957418 | 0.709259 | 0.703010 | reference |
| A1 + SMOTE-only LightGBM | 0.734853 | 0.948133 | 0.686757 | 0.692168 | -0.011160 |
| Ding AE reconstructed original train | 0.399351 | 0.827017 | 0.390996 | 0.430844 | -0.346662 |
| Ding AE reconstructed balanced train | 0.408320 | 0.829793 | 0.383632 | 0.436215 | -0.337693 |

Bootstrap support:

| Comparison | Delta AP | 95% CI | p(delta <= 0) |
|---|---:|---:|---:|
| SMOTE-only vs baseline | -0.011160 | [-0.015295, -0.006467] | 1.000 |
| Ding reconstructed original vs baseline | -0.346662 | [-0.359829, -0.334360] | 1.000 |
| Ding reconstructed balanced vs baseline | -0.337693 | [-0.350395, -0.324887] | 1.000 |
| Ding reconstructed balanced vs SMOTE-only | -0.326533 | [-0.339521, -0.314164] | 1.000 |

## Ding-Strict Check

After auditing the first IEEE-CIS adaptation, a stricter Ding-style check was
added because the earlier run used the thesis LightGBM helper. The strict check
keeps the same unavoidable A1 dense conversion, but changes the model/evaluation
closer to Ding:

- LightGBM uses GOSS, learning rate 0.1, 32 leaves, L2 10, `is_unbalance=False`,
  and seed 2018.
- Metrics include Precision, Recall, F1/F-measure, ROC-AUC, MCC, and BCR.
- AP is still reported for thesis comparability.

Command:

```bash
python src/run_ding_ieeecis_strict.py \
  --output-dir outputs/stratified_reset/ding_strict_ieee_cis \
  --target-fraud-rate 0.50 \
  --n-estimators 800 \
  --ae-epochs 30 \
  --n-bootstrap 1000
```

Selected-threshold test results:

| Arm | AP | ROC-AUC | Precision | Recall | F1 | MCC | BCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ding GOSS baseline | 0.785897 | 0.957323 | 0.881088 | 0.634648 | 0.737834 | 0.740379 | 0.815771 |
| Ding SMOTE + GOSS | 0.767718 | 0.953987 | 0.877335 | 0.602226 | 0.714204 | 0.719117 | 0.799586 |
| Ding reconstructed original + GOSS | 0.413661 | 0.834660 | 0.794791 | 0.243649 | 0.372963 | 0.430557 | 0.620684 |
| Ding reconstructed balanced + GOSS | 0.387281 | 0.830891 | 0.706838 | 0.252601 | 0.372193 | 0.411428 | 0.624401 |

Default-threshold test results:

| Arm | AP | ROC-AUC | Precision | Recall | F1 | MCC | BCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ding GOSS baseline | 0.785897 | 0.957323 | 0.936795 | 0.563029 | 0.703340 | 0.719291 | 0.780826 |
| Ding SMOTE + GOSS | 0.767718 | 0.953987 | 0.914308 | 0.555045 | 0.690756 | 0.704952 | 0.776579 |
| Ding reconstructed original + GOSS | 0.413661 | 0.834660 | 0.807377 | 0.238326 | 0.368018 | 0.429408 | 0.618132 |
| Ding reconstructed balanced + GOSS | 0.387281 | 0.830891 | 0.119919 | 0.687394 | 0.204212 | 0.231498 | 0.752230 |

Bootstrap AP deltas:

| Comparison | Delta AP | 95% CI | p(delta <= 0) |
|---|---:|---:|---:|
| Ding SMOTE + GOSS vs Ding GOSS baseline | -0.018179 | [-0.021843, -0.014740] | 1.000 |
| Reconstructed original + GOSS vs Ding GOSS baseline | -0.372236 | [-0.385188, -0.359032] | 1.000 |
| Reconstructed balanced + GOSS vs Ding GOSS baseline | -0.398616 | [-0.411799, -0.384742] | 1.000 |
| Reconstructed balanced + GOSS vs Ding SMOTE + GOSS | -0.380437 | [-0.393917, -0.367329] | 1.000 |

Strict-check conclusion:

```text
After using Ding-like GOSS LightGBM and Ding's reported metrics, the direct
AE-reconstruction branch still loses badly on IEEE-CIS. Therefore the drop is
not caused by the earlier thesis-specific LightGBM helper or by AP-only
evaluation. The negative transfer comes from the Ding reconstruction mechanism
itself on the IEEE-CIS representation.
```

## Interpretation

This is not just a small tuning loss. Direct Ding-style reconstruction loses by
about 0.34 AP against the matched A1 LightGBM baseline. It also loses by about
0.33 AP against the SMOTE-only control. Therefore, the main source of degradation
is the AE reconstruction step, not only SMOTE. The stricter GOSS run reinforces
this: Ding-style reconstruction loses by 0.37-0.40 AP against the Ding-like GOSS
baseline.

The result is defensible because the same project now has a sanity check on
Ding's ULB dataset:

| Dataset | Ding-style result |
|---|---|
| ULB anchor dataset | Reconstructed-original AEELG reaches ROC-AUC 0.967585 and F1 0.800000, close to Ding's AUC 0.9683 and F-measure 0.8027. |
| IEEE-CIS | Reconstructed AEELG falls to AP 0.399351-0.408320, far below A1 baseline AP 0.746013. |

Therefore:

```text
The code can reproduce Ding-like behavior on the small dense numeric anchor
dataset, but the direct reconstructed-feature method does not transfer to the
larger mixed-type IEEE-CIS dataset.
```

## Why It Drops on IEEE-CIS

Likely causes:

1. IEEE-CIS has mixed categorical/numeric features and heavy missingness; Ding's
   main ULB dataset is already dense numeric with only 30 features.
2. The dense A1 representation has 432 features. Reconstructing the whole matrix
   can smooth or distort discriminative fraud signals that LightGBM uses well.
3. Full-balance SMOTE is not helpful here; even SMOTE-only is lower than the
   baseline.
4. Ding reports mainly ROC-AUC/F1/MCC/BCR, while this thesis prioritizes
   Average Precision / PR-AUC under severe imbalance.
5. A strong LightGBM baseline on IEEE-CIS can already model nonlinear tabular
   patterns directly, so AE reconstruction is not automatically beneficial.

## Supervisor-Facing Explanation

Suggested wording:

```text
Saya sudah kembali ke anchor awal sesuai arahan. Pertama, pipeline Ding-style
saya uji pada dataset asal Ding, yaitu ULB. Hasilnya mendekati paper: ROC-AUC
0,9676 dan F1 0,8000, sedangkan paper melaporkan AUC 0,9683 dan F-measure
0,8027. Jadi masalahnya bukan semata-mata kode AEELG tidak jalan.

Setelah itu metode yang sama diterapkan ke IEEE-CIS. Karena IEEE-CIS tidak
langsung numerik seperti ULB, saya memakai representasi dense train-only:
frequency encoding kategorikal, median imputation, dan z-score scaling. SMOTE
hanya diterapkan pada train. Hasilnya, LightGBM baseline mendapat AP 0,7460,
sedangkan Ding-style AE reconstruction hanya AP 0,3994 sampai 0,4083. Paired
bootstrap menunjukkan delta AP negatif besar.

Saya juga membuat pengecekan tambahan yang lebih dekat ke Ding: LightGBM diganti
ke GOSS dengan parameter Ding-like dan metrik Ding lengkap (Precision, Recall,
F1, AUC, MCC, BCR). Hasilnya tetap sama secara substansi: baseline GOSS AP
0,7859, sedangkan AE reconstruction hanya AP 0,4137 atau 0,3873.

Kesimpulan sementara: metode rekonstruksi AE Ding valid sebagai anchor awal,
tetapi tidak transfer langsung ke IEEE-CIS. Penyebab yang paling mungkin adalah
rekonstruksi seluruh fitur menghaluskan sinyal fraud pada dataset IEEE-CIS yang
lebih besar, lebih sparse, dan mixed-type. Karena itu langkah berikutnya tetap
dalam tujuan awal AE+LightGBM, yaitu mencari integrasi AE yang tidak mengganti
seluruh fitur dengan rekonstruksi, misalnya AE sebagai latent-space oversampler
atau fitur tambahan yang terkontrol.
```

## Thesis Direction

Do not frame this as "changing topic". Frame it as:

```text
Ding-style AE reconstruction is the initial anchor and negative-transfer test.
The negative result motivates a constrained variant within the same AE+LightGBM
family, not a jump to an unrelated method.
```
