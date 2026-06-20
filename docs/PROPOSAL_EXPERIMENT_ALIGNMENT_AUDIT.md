# Audit Kesesuaian Eksperimen Dengan Proposal Original

Status: advisor-facing diagnosis, 2026-06-19.

Sumber proposal yang dipakai untuk audit ini:

```text
../0. Skripsi/proposal/original/Draft TA_Proposal.docx
```

Dokumen ini sengaja tidak memakai `Draft_ProposalTA.docx` revisi lain sebagai
acuan, karena proposal original seminar adalah dokumen yang dimiliki dosen
pembimbing.

## Jawaban Singkat

Sebelum audit ini, eksperimen yang benar-benar literal mengikuti proposal
original belum pernah selesai dijalankan pada protokol yang sama.

Yang sudah ada sebelumnya masih berbeda dari proposal original:

- P01-P04 adalah historical evidence under the previous chronological protocol,
  sehingga tidak sesuai dengan proposal original yang memakai stratified split.
- Eksperimen AE latent-space oversampling A1 adalah pengembangan setelah
  diagnosis, bukan desain awal proposal original.
- Eksperimen Ding-style dan rekonstruksi full-feature bukan desain proposal
  original yang hanya menempatkan Autoencoder pada `V1-V339`.
- Script stratified lama `run_initial_design_replace_v_stratified.py` belum
  literal karena mengganti `V1-V339` dengan hasil rekonstruksi, bukan encoder
  latent representation, dan preprocessing V-nya tidak mengikuti kontrak
  zero-impute plus z-score secara literal.

Setelah audit ini, eksperimen literal proposal original sudah diimplementasikan
dan dijalankan melalui:

```text
src/run_original_proposal_stratified.py
outputs/stratified_reset/original_proposal_v_latent_replacement/
```

Hasilnya: desain proposal original `Autoencoder V-only latent replacement +
LightGBM` tidak mengalahkan baseline LightGBM. Penurunan terjadi pada default
dan juga setelah kedua pipeline sama-sama dituning dengan Optuna/TPE.

## Kontrak Metode Dari Proposal Original

Proposal original mendefinisikan desain berikut:

| Komponen | Kontrak proposal original |
|---|---|
| Dataset | IEEE-CIS Fraud Detection, `train_transaction` digabung `train_identity` via `TransactionID`. |
| Split | Stratified train/validation/test 60/20/20, seed tetap. |
| Baseline | LightGBM pada fitur original setelah preprocessing numerik/kategorikal. |
| Autoencoder | Dilatih unsupervised hanya pada `V1-V339`. |
| Preprocessing V untuk AE | Missing value `V1-V339` diisi 0, lalu z-score standardization. |
| Integrasi AE | Original `V1-V339` diganti dengan latent representation dari encoder. |
| Resampling | Tidak ada resampling pada train/validation/test. |
| Tuning | Bayesian Optimization / Optuna TPE untuk LightGBM baseline dan AE-LightGBM. |
| Metrik utama | Average Precision / PR-AUC, dengan ROC-AUC, F1, precision, recall sebagai pendukung. |

## Implementasi Yang Dibuat

Script baru `src/run_original_proposal_stratified.py` dibuat untuk mengunci
kontrak di atas.

Detail implementasi:

- split `stratified_holdout` 60/20/20, `random_state=42`;
- categorical features proposal dikonversi numerik dengan train-fitted ordinal
  encoding;
- baseline LightGBM menerima 432 fitur original hasil preprocessing;
- Autoencoder dilatih hanya pada 339 fitur `V1-V339`;
- missing value V diisi `0.0`, lalu `StandardScaler` fit hanya pada training V;
- arsitektur AE undercomplete: `339 -> 256 -> 128 -> latent(32) -> 128 -> 256 -> 339`;
- hidden layer memakai ReLU, loss MSE, optimizer Adam, early stopping memakai
  validation split;
- pipeline AE-LightGBM memakai 125 fitur: 93 non-V + 32 latent;
- fitur `V1-V339` original benar-benar dikeluarkan dari pipeline AE-LightGBM;
- tidak ada missingness indicator V dan tidak ada resampling;
- baseline dan AE sama-sama dituning 15 trial Optuna/TPE memakai search space
  proposal original.

Artefak utama:

```text
outputs/stratified_reset/original_proposal_v_latent_replacement/experiment_summary.json
outputs/stratified_reset/original_proposal_v_latent_replacement/data_contract.json
outputs/stratified_reset/original_proposal_v_latent_replacement/ae_training_history.csv
```

## Hasil Eksperimen

Sumber:
`outputs/stratified_reset/original_proposal_v_latent_replacement/experiment_summary.json`.

| Pipeline | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test F1 | Test MCC |
|---|---:|---:|---:|---:|---:|
| Baseline LightGBM default | 0.859456 | 0.859857 | 0.969302 | 0.821069 | 0.817946 |
| AE latent replacement default | 0.850307 | 0.849081 | 0.967288 | 0.807534 | 0.805252 |
| Baseline LightGBM tuned | 0.874589 | 0.873133 | 0.970245 | 0.834898 | 0.833308 |
| AE latent replacement tuned | 0.860905 | 0.860110 | 0.969492 | 0.816403 | 0.814836 |

Paired-bootstrap pada test PR-AUC:

| Comparison | Delta AP | 95% CI | p(delta <= 0) |
|---|---:|---|---:|
| AE default vs baseline default | -0.010776 | [-0.013058, -0.008412] | 1.000 |
| AE tuned vs baseline tuned | -0.013023 | [-0.016044, -0.010160] | 1.000 |

Interpretasi: pada desain original, AE latent replacement bukan hanya kalah
tipis secara numerik, tetapi kalah secara konsisten pada bootstrap. CI seluruhnya
negatif, sehingga penurunan tidak dapat dijelaskan sebagai noise sampling biasa.

## Diagnosis Penyebab Penurunan

### 1. Mengganti `V1-V339` membuang sinyal tabular yang kuat

Baseline tuned melihat 432 fitur, termasuk 339 fitur V original. Pipeline AE
melihat 125 fitur, karena 339 fitur V diganti menjadi 32 latent features.
Kompresi ini mengurangi dimensi, tetapi juga membuang detail granular yang dapat
dieksploitasi LightGBM.

Pada IEEE-CIS, fitur V adalah fitur numerik anonim yang sangat banyak dan
berkorelasi dengan pola transaksi. LightGBM kuat pada fitur tabular seperti ini
karena bisa mencari split non-linear, threshold lokal, dan interaksi sederhana
tanpa perlu representasi dense. Ketika 339 fitur itu diringkas menjadi 32 latent
features, sebagian variasi diskriminatif untuk fraud ikut hilang.

### 2. Objective AE tidak sama dengan objective fraud detection

Autoencoder dilatih dengan MSE untuk merekonstruksi `V1-V339`, tanpa label
`isFraud`. Dengan data yang sangat imbalanced, MSE terutama belajar struktur
mayoritas transaksi non-fraud. Pola minoritas fraud yang jarang bisa dianggap
variasi kecil dalam objective rekonstruksi, padahal variasi itu penting untuk
PR-AUC.

Jadi AE berhasil belajar rekonstruksi, tetapi representasi yang baik untuk
rekonstruksi belum tentu baik untuk ranking fraud.

### 3. Zero-imputation pada V menghilangkan informasi missingness

Proposal original mengisi missing value V dengan 0 sebelum standardization.
Ini membuat nilai missing menjadi angka biasa. Baseline LightGBM, sebaliknya,
dapat memperlakukan `NaN` secara native dan memanfaatkan pola missingness sebagai
sinyal.

Karena pipeline AE tidak membawa missingness mask, model downstream kehilangan
informasi "nilai ini memang hilang" vs "nilai ini benar-benar 0".

### 4. Latent ReLU pada data z-score bisa membatasi representasi

Input AE sudah distandardisasi dengan z-score, sehingga nilai informatif bisa
berada di sisi positif maupun negatif. Latent layer ReLU membuat representasi
tidak negatif. Ini sesuai implementasi proposal original yang memakai ReLU
hidden layers, tetapi dapat membatasi informasi arah negatif pada latent space.

Ini bukan satu-satunya penyebab, tetapi menjadi kandidat ablation yang masuk
akal: latent activation linear atau latent dimension lebih besar.

### 5. Baseline proposal-compatible ternyata jauh lebih kuat setelah stratified reset

Pada split stratified original, baseline LightGBM default sudah mencapai test
PR-AUC 0.859857. Setelah Optuna/TPE, baseline naik menjadi 0.873133.

Ini berbeda dari historical P01-P04 karena P01-P04 memakai chronological split
lama. Jadi pembanding pada proposal original ternyata jauh lebih kuat daripada
angka historical yang sebelumnya terlihat.

### 6. Tuning membantu AE, tetapi tidak menutup gap

AE default test PR-AUC 0.849081 naik menjadi 0.860110 setelah tuning. Namun
baseline juga naik dari 0.859857 ke 0.873133. Dengan pembanding yang sama-sama
dituning, gap AE vs baseline justru tetap negatif:

```text
AE tuned - baseline tuned = -0.013023 test PR-AUC
```

Artinya masalahnya bukan sekadar hyperparameter LightGBM. Ada information loss
di mekanisme integrasi AE.

## Posisi Terhadap Masukan Dosen

Masukan dosen:

```text
Stick dulu dengan usulan semula. Kalau hasilnya tidak sebagus yang diharapkan,
kaji penyebabnya. Dari penyebab yang ditemukan, cari alternatif untuk
memperbaikinya, tetap sesuai tujuan awal dan tidak melompat ke metode lain.
```

Jawaban berdasarkan audit:

1. Sudah kembali ke usulan semula dan menjalankan eksperimen literal proposal
   original.
2. Hasilnya memang tidak sebagus baseline.
3. Penyebab utama adalah latent replacement V-only menyebabkan hilangnya sinyal
   yang sebelumnya bisa dimanfaatkan LightGBM dari fitur V original dan pola
   missingness.
4. Perbaikan harus tetap dalam keluarga Autoencoder + LightGBM, bukan langsung
   mengganti ke metode lain.

## Follow-Up Perbaikan Dalam Koridor AE + LightGBM

Setelah diagnosis di atas, empat arah perbaikan yang masih sesuai tujuan awal
sudah diuji melalui:

```text
src/run_ae_feature_improvement_ladder.py
outputs/stratified_reset/ae_feature_improvement_ladder/
docs/AE_FEATURE_IMPROVEMENT_LADDER_RESULTS.md
```

Reference stop rule tetap memakai tuned LightGBM proposal original:

```text
test PR-AUC = 0.873133
```

Ringkasan hasil:

| Tahap | Alternatif | Best test PR-AUC | Delta vs tuned LightGBM | Kesimpulan |
|---|---|---:|---:|---|
| 1 | Perbesar latent dimension tetapi tetap replace V | 0.859875 | -0.013258 | Tidak memulihkan AP. |
| 2 | Concatenate V asli + AE latent | 0.864217 | -0.008917 | Lebih baik dari replace, tetapi belum menang. |
| 3 | Denoising AE + concat | 0.864170 | -0.008963 | Validasi sedikit naik, test tidak membaik. |
| 4 | Rekonstruksi sebagian V high-missing | 0.871309 | -0.001824 | Paling dekat, tetapi belum melewati baseline. |

Kesimpulan follow-up: tidak ada varian AE feature-improvement yang mengalahkan
tuned LightGBM. Namun hasilnya memperjelas penyebab penurunan. Ketika semua
`V1-V339` diganti latent AE, AP turun paling besar. Ketika fitur V asli
dipertahankan atau hanya subset V high-missing yang direkonstruksi, gap jauh
menyempit. Jadi penyebab utama tetap information loss dari full V replacement,
bukan sekadar kesalahan implementasi atau latent dimension terlalu kecil.

Diagnosis lanjutan berbasis feature importance, ranking loss, dan missingness V
ditulis di:

```text
docs/AE_FAILURE_DEEP_DIAGNOSIS_AND_FIX_PLAN.md
```

## Alternatif Perbaikan Yang Masih Sesuai Tujuan Awal

Urutan perbaikan yang paling defensible berdasarkan audit awal:

| Prioritas | Alternatif | Alasan tetap sesuai proposal |
|---|---|---|
| 1 | AE latent sebagai fitur tambahan, bukan pengganti `V1-V339` | Masih AE + LightGBM, tetapi tidak membuang fitur V original. |
| 2 | Tambahkan missingness mask V ke AE/LightGBM | Menguji apakah gap berasal dari hilangnya informasi missingness. |
| 3 | Ablation latent dimension 64/128 dan latent activation linear | Menguji apakah kompresi 32 dimensi dan ReLU terlalu sempit. |
| 4 | Tambahkan reconstruction error per row sebagai fitur pendamping | Masih memakai AE sebagai representasi/anomaly signal tanpa mengganti semua V. |
| 5 | Latent-space oversampling sebagai pengembangan terakhir | Masih AE + LightGBM, tetapi perlu dijelaskan sebagai perbaikan mekanisme integrasi dari hasil diagnosis, bukan desain original literal. |

Rekomendasi narasi untuk pembimbing:

```text
Saya sudah menjalankan ulang desain proposal original secara literal dengan
stratified split 60/20/20. Hasilnya AE-LightGBM dengan latent replacement V-only
belum mengalahkan baseline LightGBM. Setelah tuning, baseline mencapai test
PR-AUC 0.873133, sedangkan AE latent replacement 0.860110. Paired-bootstrap
menunjukkan delta AE terhadap baseline tetap negatif. Dari diagnosis, penyebab
utamanya adalah original V features dan pola missingness membawa sinyal kuat
untuk LightGBM, sedangkan AE yang dilatih unsupervised untuk rekonstruksi
mengompresi 339 fitur V menjadi 32 latent features sehingga sebagian sinyal fraud
hilang. Perbaikan berikutnya akan tetap berada dalam tujuan awal Autoencoder +
LightGBM, misalnya menjadikan latent AE sebagai fitur tambahan, menjaga
missingness mask, dan menguji latent dimension/activation, sebelum menyimpulkan
varian final yang paling defensible.
```
