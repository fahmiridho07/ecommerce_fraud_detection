# Peta Penulisan - AE Latent-Space Oversampling

Status: fokus penulisan aktif setelah cleanup, 2026-06-21.

Dokumen ini merangkum jalur penulisan Bab 3, Bab 4, dan Bab 5 agar skripsi
tidak melebar ke semua eksperimen eksplorasi.

## 1. Sumbu Cerita

Masalah awal: fraud detection pada IEEE-CIS bersifat sangat imbalanced, sehingga
PR-AUC lebih relevan daripada akurasi. LightGBM dipakai sebagai baseline tabular
kuat. Autoencoder diuji untuk melihat apakah representasi laten dapat membantu
LightGBM.

Temuan penting: AE tidak membantu ketika dipakai sebagai feature extractor
langsung. Peran yang terbukti bermanfaat adalah AE sebagai pembentuk ruang laten
untuk membuat sampel fraud sintetis pada train set.

Sumbu akhir:

```text
LightGBM kuat -> masalah imbalance -> kontrol SMOTE-NC -> AE latent-SMOTE
```

## 2. Bab 3 - Metodologi

Urutan metode yang ditulis:

1. Dataset IEEE-CIS dan target `isFraud`.
2. Split stratified holdout 60/20/20, random_state=42.
3. Preprocessing A1:
   - frequency encoding untuk kategorikal;
   - median imputation untuk numerik;
   - z-score scaling;
   - semua fit hanya pada train.
4. Baseline LightGBM.
5. SMOTE-NC sebagai kontrol oversampling klasik.
6. Autoencoder fraud-only untuk latent-space oversampling:
   - train AE pada data fraud train A1;
   - encode fraud ke latent space;
   - interpolasi SMOTE-style antar fraud latent;
   - decode kembali ke ruang fitur A1;
   - append synthetic fraud hanya ke train.
7. Bayesian Optimization memakai Optuna/TPE dengan validation AP sebagai
   objective.
8. Evaluasi:
   - primary metric: Average Precision / PR-AUC;
   - supporting metrics: ROC-AUC, F1, MCC;
   - threshold dipilih di validation;
   - paired bootstrap untuk delta PR-AUC pada test set.

## 3. Bab 4 - Urutan Hasil

Gunakan urutan ini supaya cerita mengalir dari kontrol ke klaim utama.

### 4.1 Setup dan Split

Tulis protokol split, ukuran train/validation/test, fraud rate, dan aturan
anti-leakage.

### 4.2 Hasil Utama Tuned A1

Tabel headline:

| Pipeline tuned A1 | Validation AP | Test PR-AUC | Peran |
|---|---:|---:|---|
| Baseline | 0.835407 | 0.838988 | acuan |
| SMOTE-NC | 0.839492 | 0.843476 | kontrol oversampling |
| AE latent-SMOTE | 0.844317 | 0.850031 | metode usulan |

Bootstrap:

| Perbandingan | Delta AP | CI 95% | p(delta <= 0) |
|---|---:|---|---:|
| AE vs baseline | +0.011043 | [+0.00789, +0.01436] | 0.000 |
| AE vs SMOTE-NC | +0.006555 | [+0.00390, +0.00922] | 0.000 |
| SMOTE-NC vs baseline | +0.004488 | [+0.00135, +0.00783] | 0.004 |

Interpretasi wajib: margin kecil, tetapi positif dan diuji terhadap baseline yang
sudah dituning.

### 4.3 Kontrol Negatif: AE sebagai Feature Extractor

Tampilkan ringkas bahwa AE latent/reconstruction features pada A0 kalah atau
seri. Tujuannya bukan mempermalukan metode, tetapi menunjukkan bahwa posisi AE
yang tepat bukan kompresi fitur langsung.

### 4.4 Kontrol Fair Oversampling A0

Tampilkan bahwa pada A0 raw/NaN-native, AE latent-SMOTE meningkatkan baseline
tetapi tidak mengalahkan SMOTE-NC. Ini membatasi klaim agar tidak overclaim.

### 4.5 Robustness A1

Tampilkan split-seed robustness A1: AE mengalahkan SMOTE-NC pada beberapa split.
Bagian ini menjawab serangan bahwa hasil hanya kebetulan di seed 42.

### 4.6 Keterbatasan

Tuliskan terbuka:

- margin AE-vs-SMOTE-NC tuned kecil;
- split stratified bisa optimistis dibanding deployment temporal;
- A1 dense bukan baseline absolut terkuat untuk semua setting;
- synthetic dense rows belum menjaga pola missingness asli;
- hasil tidak menunjukkan AE sebagai universal feature extractor.

## 4. Bab 5 - Kesimpulan

Jawaban rumusan masalah:

1. LightGBM dengan preprocessing A1 dan tuning menjadi baseline yang kuat.
2. Oversampling meningkatkan PR-AUC dibanding baseline.
3. AE latent-SMOTE memberi peningkatan tambahan atas SMOTE-NC pada representasi
   A1 dense, tetapi peningkatannya terbatas.
4. AE tidak efektif sebagai feature extractor langsung untuk LightGBM pada
   eksperimen yang diuji.

Kalimat kesimpulan:

> Autoencoder dalam penelitian ini paling tepat diposisikan sebagai mekanisme
> latent-space oversampling untuk kelas minoritas, bukan sebagai pengganti atau
> penambah fitur langsung. Pada representasi A1 dense, integrasi tersebut
> meningkatkan PR-AUC dibanding baseline dan SMOTE-NC, meskipun margin
> peningkatannya terbatas.

## 5. Jawaban Siap Sidang

Pertanyaan: "Kenapa peningkatannya kecil?"

Jawaban:

> Karena pembandingnya sudah kuat dan sama-sama dituning. Margin kecil bukan
> berarti tidak ada kontribusi; paired bootstrap menunjukkan delta positif.
> Namun saya tidak mengklaim peningkatan besar. Kontribusinya adalah menunjukkan
> kondisi spesifik ketika AE masih memberi nilai tambah, yaitu sebagai
> latent-space oversampler pada representasi dense.

Pertanyaan: "Apakah yang membantu sebenarnya SMOTE, bukan AE?"

Jawaban:

> Itu sebabnya SMOTE-NC dijadikan kontrol. Pada A0, AE memang seri dengan
> SMOTE-NC. Pada A1 dense, setelah baseline, SMOTE-NC, dan AE dituning secara
> terpisah, AE latent-SMOTE tetap lebih tinggi daripada SMOTE-NC dengan delta
> +0.006555 AP dan paired-bootstrap positif.

Pertanyaan: "Apakah AE sebagai feature extractor berhasil?"

Jawaban:

> Tidak. Hasil eksperimen menunjukkan AE feature extraction turun atau seri.
> Karena itu klaim penelitian dipersempit secara jujur: AE bermanfaat sebagai
> oversampler latent space, bukan sebagai feature extractor universal.

## 6. Jangan Masuk Jalur Utama

Jangan menjadikan ini inti Bab 3/4:

- RankGauss/swap-noise AE;
- original proposal V replacement sebagai hasil utama;
- LSTM/VAE/entity-context exploration;
- score ensemble lama;
- chronological P01-P04 lama;
- klaim SOTA atau leaderboard.

Gunakan hanya sebagai ablation, diagnostic evidence, atau future work.
