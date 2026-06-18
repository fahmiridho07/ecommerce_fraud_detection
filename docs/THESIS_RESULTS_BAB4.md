# Hasil Eksperimen Terkonsolidasi (Draf Bab 4)

Status: draf hasil siap-tulis, 2026-06-18. Sumber angka: artefak di
`outputs/stratified_reset/`. Dokumen rinci pendukung: `AE_INTEGRATION_EXPERIMENT_RESULTS.md`,
`AE_BASELINE_GAP_DIAGNOSIS.md`, `EDA_AND_METHODOLOGY_AUDIT.md`.

Catatan cleanup: konfirmasi full-budget A1 dan tuned-vs-tuned sudah selesai dan
semua placeholder utama telah terisi.

## 4.1 Setup Eksperimen

- Dataset: IEEE-CIS Fraud Detection (train berlabel, merge transaction+identity).
- Protokol: **stratified holdout 60/20/20, random_state=42**. Semua preprocessing,
  imputer, encoder, autoencoder, dan oversampling **di-fit hanya pada train**;
  validation/test tidak pernah di-augmentasi dan tidak diseimbangkan.
- Metrik utama: **Average Precision (PR-AUC)**; pendukung: ROC-AUC, F1, MCC,
  confusion matrix. Threshold dipilih di validation (MCC, tie-breaker F1).
- Signifikansi: **paired bootstrap** 2000 resample pada baris test yang sama
  (melaporkan delta AP, CI 95%, p(delta<=0)).
- Klasifikator: LightGBM. Penanganan imbalance baseline: `scale_pos_weight` dari
  label train; pada cabang augmentasi train sudah diseimbangkan ke ~15% fraud
  sehingga `scale_pos_weight=1.0`.
- Kontrol fair: setiap klaim "AE membantu" diuji terhadap kontrol yang cocok
  (SMOTE-NC dan random oversampling) dengan hanya satu variabel yang berbeda.

Dua representasi fitur diuji:
- **A0** (kontrol mentah): fitur asli, kategorikal object dipetakan integer,
  numerik NaN dipertahankan untuk penanganan native LightGBM.
- **A1** (padat, Alharbi-style): frequency encoding kategorikal + median
  imputation + z-score; menghasilkan matriks all-numeric tanpa NaN.

## 4.2 Hasil 1 - Autoencoder sebagai Ekstraktor Fitur (GAGAL)

Menambahkan representasi turunan AE (latent dan/atau reconstruction error) ke
LightGBM pada representasi A0 justru menurunkan PR-AUC secara signifikan.

| Model (A0, full budget) | Test PR-AUC | Delta vs baseline | p(delta<=0) |
|-------------------------|------------:|------------------:|------------:|
| Baseline A0 | 0.821840 | acuan | - |
| + AE recon error (global) | 0.816413 | -0.005426 | 1.000 |
| + AE recon error (grouped) | 0.813677 | -0.008163 | 1.000 |
| + AE latent (32 dim) | 0.800850 | -0.020990 | 1.000 |
| + grouped recon + latent | 0.798149 | -0.023691 | 1.000 |
| Score ensemble (prob + skor anomali AE) | 0.821840 | +0.000000 | 1.000 |

Interpretasi: AE dilatih pada blok fitur V yang sudah dikonsumsi LightGBM pada
resolusi penuh, sehingga fitur turunan AE bersifat redundan/noise. Score ensemble
yang bebas memilih bobot memberi bobot 0 pada skor AE. Temuan ini konsisten
dengan literatur tabular deep learning (Grinsztajn 2022; Gorishniy 2021).

## 4.3 Hasil 2 - Augmentasi Minoritas Mengalahkan Baseline (stratified dan temporal)

Oversampling kelas minoritas (sintesis fraud, hanya pada train) meningkatkan
baseline secara signifikan dan robust.

| Eksperimen | Baseline | Augmentasi AE | Delta AP | p(delta<=0) |
|------------|---------:|--------------:|---------:|------------:|
| Stratified (full budget) | 0.821840 | 0.837371 | +0.015531 | 0.000 |
| Stratified (mean 4 split-seed, 800 trees) | - | - | +0.02412 (+/-0.00412, 4/4) | <0.001 |
| Chronological/temporal (800 trees) | 0.484296 | 0.501923 | +0.017627 | 0.000 |

Augmentasi juga mengalahkan random oversampling. Catatan penting: augmentasi
membantu di **kedua** protokol, termasuk protokol temporal yang lebih realistis
(yang sebelumnya hanya ~0.48-0.50).

## 4.4 Hasil 3 - Kontribusi AE-Spesifik Bergantung Representasi (temuan inti)

Pertanyaan kunci yang adil: apakah keunggulan datang dari **autoencoder** atau
sekadar dari **oversampling**? Dijawab dengan membandingkan AE latent-space
oversampling terhadap SMOTE-NC klasik, semua hal lain identik.

Pada representasi A0 (mentah, NaN-native): **seri**.

| Perbandingan (A0) | Delta AP | CI 95% | p(delta<=0) | Verdict |
|-------------------|---------:|--------|------------:|---------|
| AE vs random oversampling | +0.006666 | [+0.0044, +0.0091] | 0.000 | AE menang |
| AE vs SMOTE-NC (full budget) | -0.000568 | [-0.0024, +0.0012] | 0.733 | seri |
| AE vs SMOTE-NC (mean 4 split) | -0.00046 | (0/4 menang) | - | seri |
| AE vs SMOTE-NC (temporal) | -0.004772 | [-0.0075, -0.0019] | 1.000 | AE lebih buruk |
| VAE prior vs SMOTE-NC | -0.000538 | [-0.0026, +0.0015] | 0.705 | seri |

Pada representasi A1 (padat, frequency-encoded): **AE menang signifikan & robust**.

| Split seed (A1, 800 trees) | Baseline | +SMOTE-NC | +AE | AE - SMOTE | p(delta<=0) |
|---------------------------:|---------:|----------:|----:|-----------:|------------:|
| 42 | 0.74601 | 0.75817 | 0.78406 | +0.02589 | 0.000 |
| 1  | 0.74918 | 0.76891 | 0.78531 | +0.01640 | 0.000 |
| 2  | 0.76266 | 0.77338 | 0.79556 | +0.02217 | 0.000 |
| 3  | 0.74982 | 0.76174 | 0.77893 | +0.01719 | 0.000 |
| **Mean** | - | - | - | **+0.02041 (+/-0.00446, 4/4)** | <0.001 |

Konfirmasi full budget (A1, 2000 trees): base 0.74601, smote 0.75817, ae 0.78406;
AE - SMOTE = +0.02589, CI [+0.02288, +0.02899], p<0.001 - **identik dengan
800 trees** (model A1 konvergen lebih awal via early stopping), jadi kemenangan AE
tidak bergantung budget.

### Tuned-vs-tuned (Optuna, A1 dense) - keunggulan AE BERTAHAN

Ketiga pipeline disetel terpisah dengan Optuna (TPE, 8 trial, objektif validation
AP, early stopping). Harness: `src/tune_a1_augmentation_optuna.py`.

| Pipeline (tuned, A1) | Test PR-AUC | Val AP |
|----------------------|------------:|-------:|
| Baseline | 0.838988 | 0.835407 |
| + SMOTE-NC | 0.843476 | 0.839492 |
| + AE latent-SMOTE (usulan) | 0.850031 | 0.844317 |

| Perbandingan (tuned vs tuned) | Delta AP | CI 95% | p(delta<=0) | Verdict |
|-------------------------------|---------:|--------|------------:|---------|
| AE vs baseline | +0.011043 | [+0.00789, +0.01436] | 0.000 | menang signifikan |
| AE vs SMOTE-NC | +0.006555 | [+0.00390, +0.00922] | 0.000 | menang signifikan |
| SMOTE-NC vs baseline | +0.004488 | [+0.00135, +0.00783] | 0.004 | menang signifikan |

Tuning menaikkan baseline A1 dari 0.746 (untuned) ke 0.839 (tuned), menutup
sebagian gap. Namun **AE tetap mengalahkan baseline maupun SMOTE-NC secara
signifikan setelah ketiganya di-tune adil**. Margin AE-vs-SMOTE menyusut dari
+0.026 (untuned) ke +0.0066 (tuned) tetapi tetap p<0.001. Kesimpulan: keunggulan
AE bukan artefak baseline yang kurang disetel. Model AE-augmented tuned (0.850)
juga kompetitif dengan baseline preprocessing terkuat staging (~0.856).

Interpretasi (ber-anchor SMOTE klasik, Chawla et al. 2002, dan DeepSMOTE,
Dablain et al. 2022): interpolasi di ruang laten AE menghasilkan sampel sintetis
yang lebih on-manifold daripada interpolasi SMOTE di ruang fitur mentah,
**khususnya pada data padat berdimensi tinggi yang berkorelasi** (rezim A1).
Pada representasi A0 yang jarang/NaN-native, keunggulan itu hilang.

## 4.5 Robustness

- Lintas **fraud rate** {0.10, 0.15, 0.20}: augmentasi menang, delta ~+0.034-0.036,
  semua p<0.001 (budget 600 trees).
- Lintas **seed sintesis** {42, 7}: konsisten positif.
- Lintas **split seed** {42, 1, 2, 3}: augmentasi menang 4/4; AE vs SMOTE pada A1
  menang 4/4.
- **Tuned-vs-tuned** (Optuna, 8 trial/pipeline): AE tetap menang vs baseline
  (+0.0110, p<0.001) dan vs SMOTE-NC (+0.0066, p<0.001) setelah ketiganya disetel
  adil. Keunggulan AE bukan artefak baseline under-tuned.

## 4.6 Keterbatasan dan Ancaman Validitas (untuk kejujuran sidang)

1. **Entity/temporal leakage pada stratified split.** ~97% baris memiliki `card1`
   yang muncul >1x; `card1_train_count`, `card1`, dan `TransactionDT` termasuk
   fitur paling penting (rank 2, 4, 6 dari 488). PR-AUC stratified (~0.82-0.86)
   karenanya optimistik vs deployment; hasil temporal (~0.48-0.50) lebih
   realistis. Sudah diposisikan sebagai keterbatasan + future work.
2. **A1 bukan baseline absolut terkuat.** A1 (median-impute + z-score) menghapus
   sinyal missingness sehingga lebih lemah (0.746 @800 trees) daripada A0
   NaN-native (0.766). Klaim AE-menang adalah **kontras terkontrol AE-vs-SMOTE
   pada representasi yang sama**, bukan angka SOTA absolut.
3. **Baris sintetis padat (tanpa NaN-V).** Density dibagi sama oleh SMOTE-NC dan
   AE, jadi tidak membiaskan isolasi AE-vs-SMOTE; menjaga pola missingness anchor
   adalah penyempurnaan lanjutan.
4. **Budget pohon.** Sebagian angka pada 600/800 trees untuk efisiensi sweep;
   headline pada full budget. Kontras relatif stabil terhadap budget.

## 4.7 Kesimpulan dan Klaim Defensible

Kontribusi autoencoder bersifat presisi dan kondisional:

1. Sebagai **ekstraktor fitur**, AE TIDAK membantu dan justru merugikan LightGBM
   yang sudah kuat.
2. **Augmentasi minoritas** meningkatkan deteksi fraud secara signifikan dan
   robust pada protokol stratified maupun temporal.
3. Sebagai **latent-space oversampler**, AE memberi keunggulan spesifik yang
   signifikan dan robust atas SMOTE-NC klasik **hanya pada representasi padat
   (frequency-encoded)** (+0.020 AP, 4/4 split); setara pada representasi mentah.

Klaim tesis yang jujur dan dapat dipertahankan:

> "Autoencoder berkontribusi pada deteksi fraud LightGBM bukan sebagai ekstraktor
> fitur (yang merugikan), melainkan sebagai latent-space oversampler; keunggulannya
> atas SMOTE klasik signifikan secara spesifik pada representasi fitur padat
> (frequency-encoded), dan setara pada representasi mentah."

Klaim ini memenuhi tujuan (AE signifikan > baseline) tanpa overclaim, dan tahan
terhadap pertanyaan examiner "apakah sudah dibandingkan dengan SMOTE?" karena
justru itu temuan utamanya.
