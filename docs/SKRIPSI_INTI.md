# SKRIPSI_INTI - Halaman Jangkar

Halaman ini adalah peta satu layar saat kamu mulai merasa penuh. Bacaan teknis
tetap ada di dokumen lain, tapi untuk mengingat inti skripsi, mulai dari sini.

Source of truth tetap `THESIS_SCOPE.md`. Halaman ini hanya versi tenang dan
ringkasnya.

## 1. Pertanyaan Penelitian

Pada representasi A1 dense yang sudah dipreprocessing secara paper-anchored,
apakah autoencoder sebagai oversampler di latent space dapat meningkatkan
Average Precision LightGBM dibanding:

- LightGBM tanpa penyeimbangan; dan
- LightGBM dengan SMOTE-NC.

## 2. Protokol Aktif

```text
split_strategy = stratified_holdout
train/validation/test = 60/20/20
random_state = 42
primary metric = Average Precision / PR-AUC
```

Semua preprocessing, imputer, scaler, encoder, AE, dan oversampling hanya fit
di train split. Validation dan test tidak dibalance.

## 3. Inti Metode

```text
[1] A1 preprocessing
    categorical frequency encoding, numeric median + z-score

        -> [2] AE latent-space oversampling
              membuat sampel fraud sintetis hanya dari train split

        -> [3] LightGBM

        -> [4] Evaluasi test AP + paired bootstrap
```

Kalimat sidang:

> Metode utama saya memakai autoencoder bukan sebagai feature extractor, tetapi
> sebagai pembuat sampel fraud sintetis di latent space. Pada representasi A1
> yang dense, pendekatan ini meningkatkan AP LightGBM dibanding baseline dan
> SMOTE-NC dengan dukungan paired bootstrap.

## 4. Hasil Headline Aktif

Final tuned A1 comparison:

| Skenario | Test AP | Peran |
|---|---:|---|
| A1 LightGBM baseline | 0.838988 | baseline |
| A1 + SMOTE-NC | 0.843476 | pembanding klasik |
| A1 + AE latent-SMOTE | 0.850031 | metode usulan |

Delta utama:

- AE vs baseline: +0.011043 AP.
- AE vs SMOTE-NC: +0.006555 AP.
- Keduanya didukung paired-bootstrap.

## 5. Yang Masuk Narasi Utama

Masuk Bab 3/Bab 4:

- A1 paper-anchored preprocessing.
- LightGBM baseline.
- SMOTE-NC sebagai pembanding oversampling.
- AE latent-space oversampling sebagai metode usulan.
- Average Precision / PR-AUC sebagai metrik utama.
- Paired bootstrap untuk menguji delta AP.

Jadi pembatasnya sederhana:

```text
AE menang sebagai latent-space oversampler pada representasi A1 dense.
AE tidak diklaim menang sebagai feature extractor universal.
```

## 6. Bacaan Minimal

Kalau hanya ingin menulis skripsi dan tidak ingin tenggelam:

1. `THESIS_SCOPE.md`
2. `THESIS_RESULTS_BAB4.md`
3. `EXPERIMENT_REGISTRY.md`
4. `AE_INTEGRATION_EXPERIMENT_RESULTS.md`

Dokumen reset seperti `STRATIFIED_SPLIT_RESET.md` dan
`PAPER_ANCHORED_PREPROCESSING_RESET.md` adalah decision record. Baca kalau perlu
menjelaskan alasan metodologi, bukan sebagai daftar eksperimen baru.

## 7. Skrip yang Perlu Dikenali

Jangan rerun eksperimen mahal kecuali artifact hilang atau scope dibuka ulang.

| Tujuan | Skrip |
|---|---|
| Final tuned A1 comparison | `src/tune_a1_augmentation_optuna.py` |
| A1 AE vs SMOTE-NC confirmation | `src/run_strong_baseline_augmentation.py` |
| Robustness beberapa split | `src/run_repeated_split_validation.py` |
| A0 fair control, bukan headline final | `src/run_fair_augmentation_comparison.py` |
| Figur Bab 4 | `src/generate_thesis_figures.py` |

## 8. Abaikan Saat Panik

Ini bukan narasi utama skripsi:

- AE reconstruction-error features;
- AE latent features yang ditempel ke LightGBM;
- VAE prior control;
- chronological split sebagai protokol utama;
- hasil P01-P04, AE-05, dan score ensemble lama;
- broad UID, velocity, rolling window, target encoding, dan ensemble besar.

Semua itu boleh disebut sebagai diagnostik, appendix, limitation, atau future
work jika perlu. Jangan jadikan alur utama.

## 9. Checklist Anti-Overwhelm

- [ ] Bab 3 menjelaskan A1 preprocessing, train-only AE oversampling, LightGBM,
      dan evaluasi AP.
- [ ] Bab 4 menampilkan tabel final A1 baseline vs SMOTE-NC vs AE.
- [ ] Klaim AE dibatasi pada latent-space oversampling, bukan feature extractor.
- [ ] Hasil chronological lama tidak dicampur dengan tabel stratified aktif.
- [ ] Keterbatasan menyebut temporal evaluation/concept drift sebagai future
      work.
