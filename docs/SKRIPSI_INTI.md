# SKRIPSI_INTI - Halaman Jangkar

Halaman satu layar saat mulai penuh. Kalau ragu, baca ini lalu kembali ke
jalur utama.

## 0. Aturan emas

Ruang lingkup penulisan sekarang dikunci pada hasil yang paling defensible:

```text
Autoencoder sebagai latent-space oversampler + LightGBM + Bayesian Optimization
```

Tidak menambah metode besar baru. Eksperimen tambahan hanya boleh dipakai jika
memperjelas narasi ini, bukan membuka cabang skripsi baru.

## 1. Judul

"Deteksi Penipuan Transaksi E-Commerce Menggunakan Integrasi Autoencoder dan
LightGBM dengan Bayesian Optimization."

Judul tetap aman karena integrasi Autoencoder tetap ada, tetapi perannya
dijelaskan lebih presisi: bukan pengganti fitur mentah, melainkan pembentuk
ruang laten untuk oversampling fraud.

## 2. Inti metode

Dataset IEEE-CIS diproses menjadi representasi A1 dense:

- categorical frequency encoding;
- numeric median imputation;
- z-score scaling;
- semua statistik fit hanya pada train.

Pipeline utama:

```text
A1 preprocessing -> AE latent-space oversampling pada train fraud
                 -> LightGBM
                 -> Optuna/TPE tuning
                 -> evaluasi PR-AUC + paired bootstrap
```

## 3. Tabel utama Bab 4

Gunakan tiga pipeline tuned A1 sebagai tabel headline:

| Pipeline | Peran |
|---|---|
| A1 LightGBM baseline | baseline kuat tanpa oversampling |
| A1 + SMOTE-NC + LightGBM | kontrol oversampling klasik |
| A1 + AE latent-SMOTE + LightGBM | metode usulan |

Angka headline:

| Pipeline tuned A1 | Test PR-AUC |
|---|---:|
| Baseline | 0.838988 |
| SMOTE-NC | 0.843476 |
| AE latent-SMOTE | 0.850031 |

Delta utama:

| Perbandingan | Delta AP |
|---|---:|
| AE latent-SMOTE vs baseline | +0.011043 |
| AE latent-SMOTE vs SMOTE-NC | +0.006555 |
| SMOTE-NC vs baseline | +0.004488 |

## 4. Klaim aman

Klaim akhir:

> Autoencoder berkontribusi pada deteksi fraud LightGBM sebagai latent-space
> oversampler pada representasi fitur dense. Peningkatannya signifikan secara
> statistik tetapi terbatas, sehingga kontribusinya bersifat kondisional, bukan
> klaim bahwa AE selalu unggul sebagai feature extractor.

Kalimat penting untuk sidang:

> Margin AE memang kecil setelah baseline dituning kuat. Itu justru menunjukkan
> evaluasi dilakukan terhadap pembanding yang kompetitif. Kontribusi penelitian
> adalah isolasi kondisi ketika AE masih memberi nilai tambah, yaitu pada
> oversampling di ruang laten dense, bukan pada kompresi fitur langsung.

## 5. Kontrol yang masuk Bab 4

Masukkan sebagai bukti pendukung, bukan headline:

- AE sebagai feature extractor pada A0 turun atau seri.
- Pada A0 raw/NaN-native, AE latent-SMOTE menang atas baseline tetapi seri
  terhadap SMOTE-NC.
- Pada A1 dense, AE latent-SMOTE mengalahkan SMOTE-NC pada split utama dan
  robustness beberapa split.
- Paired bootstrap dipakai untuk semua klaim delta AP.

## 6. Skrip yang dipakai

Canonical local:

```bash
python src/tune_a1_augmentation_optuna.py
```

Kaggle ringkas:

```text
kaggle/ieee_final_oversampling_kaggle.py
```

Output Kaggle:

```text
/kaggle/working/final_oversampling_results.json
```

## 7. Abaikan untuk penulisan utama

Jangan jadikan cabang ini sebagai metode utama:

- original proposal V-latent replacement;
- AE feature extractor/additive latent;
- reconstruction-error-only AE;
- broad AE feature ladder;
- RankGauss/swap-noise diagnostic;
- LSTM/VAE/entity-context experiments;
- chronological historical P01-P04 sebagai tabel utama.

Cabang-cabang itu boleh disebut singkat sebagai ablation, diagnosis, limitation,
atau future work jika diperlukan.
