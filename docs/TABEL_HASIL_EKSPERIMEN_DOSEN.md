# Tabel Hasil Eksperimen

Status: ringkasan untuk dosen pembimbing, berdasarkan hasil aktif
`stratified_holdout` 60/20/20 dengan `random_state=42`.

Catatan penting:

- Metrik utama adalah Average Precision / PR-AUC pada test set.
- Semua preprocessing, autoencoder, dan oversampling di-fit hanya pada train
  split.
- Validation dan test tidak diseimbangkan.
- Hasil chronological lama tidak dimasukkan ke tabel utama karena memakai
  protokol berbeda.

## Tabel 1. Protokol Split Data

| Split | Jumlah Baris | Jumlah Fraud | Fraud Rate |
|---|---:|---:|---:|
| Train | 354,324 | 12,398 | 3.4991% |
| Validation | 118,108 | 4,132 | 3.4985% |
| Test | 118,108 | 4,133 | 3.4993% |

## Tabel 2. Hasil Utama Tuned Pipeline A1

Representasi A1 menggunakan preprocessing paper-anchored:
categorical frequency encoding, numeric median imputation, dan z-score scaling.
Ketiga pipeline dituning secara terpisah menggunakan Optuna dengan objektif
validation AP.

| No. | Pipeline | Penanganan Imbalance pada Train | Validation AP | Test PR-AUC | Delta vs Baseline | Keterangan |
|---:|---|---|---:|---:|---:|---|
| 1 | A1 LightGBM baseline | `scale_pos_weight` dari label train | 0.835407 | 0.838988 | acuan | Baseline utama |
| 2 | A1 + SMOTE-NC + LightGBM | SMOTE-NC hanya pada train | 0.839492 | 0.843476 | +0.004488 | Pembanding oversampling klasik |
| 3 | A1 + AE latent-SMOTE + LightGBM | AE latent-space oversampling hanya pada train | 0.844317 | 0.850031 | +0.011043 | Metode usulan |

## Tabel 3. Uji Signifikansi Paired Bootstrap pada Pipeline Tuned A1

| Perbandingan | Delta Test PR-AUC | 95% CI | p(delta <= 0) | Keputusan |
|---|---:|---|---:|---|
| AE latent-SMOTE vs baseline | +0.011043 | [+0.00789, +0.01436] | 0.000 | AE menang signifikan |
| AE latent-SMOTE vs SMOTE-NC | +0.006555 | [+0.00390, +0.00922] | 0.000 | AE menang signifikan |
| SMOTE-NC vs baseline | +0.004488 | [+0.00135, +0.00783] | 0.004 | SMOTE-NC menang signifikan |

## Tabel 4. Kontrol: AE sebagai Feature Extractor pada A0

Tabel ini menunjukkan bahwa autoencoder tidak membantu ketika hanya digunakan
sebagai penambah fitur LightGBM. Karena itu, klaim akhir diarahkan pada AE
sebagai oversampler di latent space, bukan sebagai feature extractor.

| Model A0 | Test PR-AUC | Delta vs Baseline | p(delta <= 0) | Keputusan |
|---|---:|---:|---:|---|
| Baseline A0 LightGBM | 0.821840 | acuan | - | Baseline |
| + AE reconstruction error global | 0.816413 | -0.005426 | 1.000 | Turun |
| + AE reconstruction error grouped | 0.813677 | -0.008163 | 1.000 | Turun |
| + AE latent 32 dimensi | 0.800850 | -0.020990 | 1.000 | Turun |
| + grouped reconstruction + latent | 0.798149 | -0.023691 | 1.000 | Turun |
| Score ensemble baseline + AE anomaly | 0.821840 | +0.000000 | 1.000 | Seri |

## Tabel 5. Kontrol Fair Oversampling pada A0

Pada representasi A0 mentah/NaN-native, AE latent-SMOTE meningkatkan baseline,
tetapi belum mengalahkan SMOTE-NC. Ini menjadi alasan mengapa klaim akhir
dibatasi pada representasi A1 dense.

| Pipeline A0 | Test PR-AUC | Delta vs Baseline | p(delta <= 0) | Keputusan |
|---|---:|---:|---:|---|
| Baseline A0 LightGBM | 0.821840 | acuan | - | Baseline |
| Random oversampling | 0.830705 | +0.008865 | 0.000 | Menang vs baseline |
| SMOTE-NC | 0.837939 | +0.016099 | 0.000 | Menang vs baseline |
| AE latent-SMOTE | 0.837371 | +0.015531 | 0.000 | Menang vs baseline |

| Perbandingan A0 | Delta Test PR-AUC | 95% CI | p(delta <= 0) | Keputusan |
|---|---:|---|---:|---|
| AE latent-SMOTE vs random oversampling | +0.006666 | [+0.00437, +0.00906] | 0.000 | AE menang |
| AE latent-SMOTE vs SMOTE-NC | -0.000568 | [-0.00240, +0.00120] | 0.733 | Seri |

## Tabel 6. Robustness A1: AE vs SMOTE-NC pada Beberapa Split

| Split Seed | A1 Baseline | A1 + SMOTE-NC | A1 + AE latent-SMOTE | AE - SMOTE-NC | p(delta <= 0) |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.74601 | 0.75817 | 0.78406 | +0.02589 | 0.000 |
| 1 | 0.74918 | 0.76891 | 0.78531 | +0.01640 | 0.000 |
| 2 | 0.76266 | 0.77338 | 0.79556 | +0.02217 | 0.000 |
| 3 | 0.74982 | 0.76174 | 0.77893 | +0.01719 | 0.000 |
| Mean | - | - | - | +0.02041 | <0.001 |

## Ringkasan Kesimpulan

Hasil utama menunjukkan bahwa autoencoder tidak efektif sebagai feature
extractor tambahan untuk LightGBM, tetapi efektif sebagai oversampler di latent
space pada representasi A1 yang dense. Pada pipeline tuned yang adil, AE
latent-SMOTE memperoleh Test PR-AUC 0.850031, lebih tinggi daripada baseline
0.838988 dan SMOTE-NC 0.843476, dengan paired-bootstrap yang menunjukkan
peningkatan signifikan.
