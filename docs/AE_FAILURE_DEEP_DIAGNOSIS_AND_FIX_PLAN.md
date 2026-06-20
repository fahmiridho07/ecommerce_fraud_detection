# Deep Diagnosis: Why AE Feature Improvement Has Not Beaten Tuned LightGBM

Status: advisor-facing diagnosis, 2026-06-20.

Tujuan dokumen ini adalah menjawab masukan pembimbing:

```text
Stick dulu dengan usulan semula. Kalau hasilnya tidak sebagus yang diharapkan,
kaji penyebabnya. Dari penyebab yang ditemukan, cari alternatif untuk
memperbaikinya, tetap sesuai tujuan awal dan tidak melompat ke metode lain.
```

Koridor metodologi yang dipertahankan:

```text
Autoencoder memperbaiki / membentuk fitur untuk LightGBM.
```

Bukan berpindah ke model lain, bukan mengganti objective penelitian, dan bukan
mencampur hasil historical chronological protocol dengan hasil stratified reset.

## Status Implementasi Terhadap Proposal

Eksperimen yang literal terhadap proposal original sudah dilakukan melalui:

```text
src/run_original_proposal_stratified.py
outputs/stratified_reset/original_proposal_v_latent_replacement/
```

Kontrak utama yang sudah sesuai:

- dataset IEEE-CIS `train_transaction` + `train_identity`;
- split stratified holdout 60/20/20, `random_state=42`;
- metrik utama Average Precision / PR-AUC;
- Autoencoder dilatih unsupervised hanya pada `V1-V339`;
- missing value V diisi 0 lalu z-score untuk input AE;
- fitur `V1-V339` original diganti oleh latent representation encoder;
- LightGBM dipakai sebagai classifier downstream;
- baseline LightGBM dan AE-LightGBM sama-sama dituning.

Hasil proposal literal:

| Pipeline | Val AP | Test AP | Delta vs pembanding |
|---|---:|---:|---:|
| Baseline LightGBM default | 0.859457 | 0.859857 | - |
| AE latent replacement default | 0.850307 | 0.849081 | -0.010776 |
| Baseline LightGBM tuned | 0.874589 | 0.873133 | - |
| AE latent replacement tuned | 0.860905 | 0.860110 | -0.013023 |

Kesimpulan implementasi:

```text
Implementasi proposal original sudah sesuai dan hasilnya memang belum
mengalahkan baseline LightGBM. Jadi masalah utama bukan lagi "eksperimennya
belum dilakukan", tetapi "mengapa integrasi AE tersebut menurunkan AP".
```

## Bukti Dari Ladder Perbaikan

Empat tahap perbaikan yang masih dalam koridor AE + LightGBM juga sudah diuji:

```text
src/run_ae_feature_improvement_ladder.py
outputs/stratified_reset/ae_feature_improvement_ladder/ladder_summary.json
docs/AE_FEATURE_IMPROVEMENT_LADDER_RESULTS.md
```

Reference stop rule:

```text
Tuned LightGBM proposal original test AP = 0.873133233976
```

Ringkasan hasil:

| Tahap | Kandidat terbaik tahap | Test AP | Delta vs tuned LightGBM | Makna |
|---|---|---:|---:|---|
| 1 | Larger latent, tetap replace V | 0.859875 | -0.013258 | Kapasitas lebih besar tidak memperbaiki full replacement. |
| 2 | Concatenate V original + latent AE | 0.864217 | -0.008917 | Menjaga V original membantu, tetapi latent belum cukup kuat. |
| 3 | Denoising AE + concat | 0.864170 | -0.008963 | Denoising tidak memberi generalisasi test yang lebih baik. |
| 4 | Partial reconstruction high-missing V | 0.871309 | -0.001824 | Paling dekat; AE lebih masuk akal jika dibatasi pada V bermasalah. |

Tidak ada kandidat yang mengalahkan tuned LightGBM.

## Diagnosis 1 - Full Replacement Membuang Sinyal V Yang Kuat

Pada baseline tuned, fitur V original masih menyumbang porsi gain yang besar:

| Model | Kelompok fitur | Gain share |
|---|---|---:|
| Baseline tuned | Non-V original | 69.76% |
| Baseline tuned | V original | 30.24% |

Artinya, `V1-V339` bukan noise yang bebas dibuang. LightGBM benar-benar memakai
fitur V original untuk ranking fraud.

Pada desain proposal original, 339 fitur V diganti menjadi 32 latent features:

| Model | Kelompok fitur | Gain share |
|---|---|---:|
| AE latent replacement tuned | Non-V original | 76.77% |
| AE latent replacement tuned | AE latent | 23.23% |

AE latent memang dipakai oleh LightGBM, tetapi tidak cukup untuk menggantikan
detail 339 fitur V original. Ini menjelaskan kenapa performa turun walaupun
feature importance latent terlihat tinggi.

Interpretasi:

```text
Masalahnya bukan latent AE diabaikan oleh LightGBM. Masalahnya latent AE
menjadi ringkasan yang terlalu kasar untuk menggantikan sinyal split-level dari
fitur V original.
```

## Diagnosis 2 - Penurunan Terlihat Sebagai Ranking Loss, Bukan Sekadar Threshold

Average Precision sensitif terhadap urutan ranking fraud. Pada test set,
jumlah fraud adalah 4,133 baris. Jika kita lihat top 4,133 skor tertinggi:

| Model | Fraud captured di top 4,133 | Selisih TP vs baseline |
|---|---:|---:|
| Baseline tuned | 3,384 | - |
| Original AE replace LD32 | 3,314 | -70 |
| Replace LD64 | 3,325 | -59 |
| Replace LD128 | 3,309 | -75 |
| Replace LD256 | 3,289 | -95 |
| Concat LD32 | 3,332 | -52 |
| Denoising concat LD32 | 3,339 | -45 |
| Partial replace high-missing | 3,366 | -18 |
| Partial append high-missing | 3,362 | -22 |

Top 1% masih hampir sama-sama sangat presisi, tetapi AP tetap turun karena
ranking pada area setelah puncak ikut bergeser. Ini penting untuk menjelaskan
kenapa F1/precision pada threshold tertentu tidak cukup untuk membuktikan AE
lebih baik.

Interpretasi:

```text
AE tidak menghancurkan semua prediksi. Namun AE cukup menggeser urutan risiko
sehingga sebagian fraud yang baseline tempatkan tinggi menjadi turun posisinya.
Pada PR-AUC, pergeseran ranking seperti ini langsung menurunkan AP.
```

## Diagnosis 3 - Missingness V Adalah Sinyal, Tetapi AE Proposal Tidak Membawa Mask

Dari `train_transaction.csv`, pola missingness fitur V sangat tidak merata:

| Bucket missing rate V | Jumlah fitur V |
|---|---:|
| <= 1% | 86 |
| 5%-25% | 65 |
| 25%-50% | 29 |
| 75%-90% | 159 |

Tidak ada fitur V pada bucket 50%-75%, 90%-99%, atau >99% pada perhitungan ini.
Jadi struktur missingness V membentuk beberapa blok fitur yang jelas.

Beberapa fitur V penting baseline juga memiliki missingness yang berbeda tajam
antara non-fraud dan fraud:

| Fitur | Baseline gain | Missing all | Missing non-fraud | Missing fraud | Selisih fraud - non-fraud |
|---|---:|---:|---:|---:|---:|
| V258 | 45,523.95 | 77.91% | 78.90% | 50.68% | -28.23 pp |
| V201 | 5,155.60 | 76.32% | 77.41% | 46.43% | -30.98 pp |
| V219 | 664.73 | 77.91% | 78.90% | 50.68% | -28.23 pp |
| V70 | 1,824.30 | 13.06% | 12.74% | 21.82% | +9.08 pp |
| V90 | 1,422.09 | 15.10% | 14.90% | 20.69% | +5.80 pp |

Proposal original mengisi missing V dengan 0 sebelum standardization dan tidak
mengirim missingness mask ke LightGBM. Baseline LightGBM, sebaliknya, masih bisa
memanfaatkan pola `NaN`/missingness pada fitur original.

Interpretasi:

```text
Sebagian sinyal fraud pada V bukan hanya nilai numeriknya, tetapi juga apakah
nilai itu muncul atau hilang. AE proposal mengubah missingness menjadi angka
hasil imputasi dan latent dense, sehingga informasi mask hilang.
```

Ini juga menjelaskan kenapa partial reconstruction high-missing menjadi kandidat
terbaik. Ia menyasar 159 fitur pada bucket 75%-90% missing, tetapi tetap
mempertahankan fitur lain.

## Diagnosis 4 - Memperbesar Latent Dimension Tidak Menjawab Akar Masalah

Stage 1 menunjukkan:

| Candidate | Test AP | Delta |
|---|---:|---:|
| Replace LD64 | 0.859875 | -0.013258 |
| Replace LD128 | 0.856170 | -0.016963 |
| Replace LD256 | 0.854391 | -0.018742 |

Jika penyebab utama hanya "latent 32 terlalu kecil", maka LD64/128/256
seharusnya memulihkan AP. Yang terjadi justru sebaliknya.

Interpretasi:

```text
Akar masalah bukan sekadar kompresi terlalu kecil, tetapi desain full
replacement itu sendiri: V original dan missingness diganti oleh representasi
dense yang tidak label-guided.
```

## Diagnosis 5 - Concatenate Membantu, Tetapi Latent AE Masih Redundan/Noisy

Pada concat LD32, fitur V original dipertahankan dan latent AE ditambahkan:

| Model | Kelompok fitur | Gain share |
|---|---|---:|
| Concat LD32 | Non-V original | 63.14% |
| Concat LD32 | V original | 26.60% |
| Concat LD32 | AE latent | 10.27% |

Concat lebih baik daripada full replacement, tetapi belum mengalahkan baseline:

```text
Concat LD32 test AP = 0.864217
Delta vs baseline  = -0.008917
```

Interpretasi:

```text
Latent AE membawa sedikit sinyal tambahan, tetapi sebagian besar sinyal itu
redundan dengan V original atau tidak cukup aligned dengan label fraud. Ketika
latent dimension diperbesar, AP turun lagi, sehingga fitur latent tambahan mulai
lebih banyak noise daripada informasi.
```

## Diagnosis 6 - Denoising AE Tidak Cukup Mengubah Objective

Denoising concat LD32:

```text
Validation AP = 0.864072
Test AP       = 0.864170
Delta         = -0.008963
```

Denoising tidak lebih baik dari concat LD32 regular pada test set.

Interpretasi:

```text
Denoising membuat AE belajar fitur yang lebih robust terhadap noise input, tetapi
objective-nya tetap rekonstruksi unsupervised. Membersihkan input V tidak
otomatis menghasilkan ranking fraud yang lebih baik.
```

## Diagnosis 7 - Partial Reconstruction Adalah Sinyal Arah Perbaikan Terkuat

Partial reconstruction high-missing menjadi hasil terbaik:

| Model | Kelompok fitur | Gain share |
|---|---|---:|
| Partial replace high-missing | Non-V original | 68.24% |
| Partial replace high-missing | V high-missing reconstructed | 19.18% |
| Partial replace high-missing | V remaining original | 12.58% |

Hasil:

```text
Test AP = 0.871309
Delta   = -0.001824
```

Gap turun jauh dari original AE replacement:

```text
Original AE replace delta = -0.013023
Partial replace delta     = -0.001824
```

Interpretasi:

```text
AE paling berguna ketika diterapkan terbatas pada bagian fitur V yang memang
bermasalah, bukan sebagai pengganti penuh seluruh V block.
```

Namun partial replacement masih belum menang. Kemungkinan penyebabnya:

- rekonstruksi masih meratakan pola missing/anomaly yang berguna;
- mask missingness belum eksplisit diberikan ke LightGBM;
- subset high-missing dipilih hanya dari threshold missing rate, belum dari
  kombinasi missingness + feature importance + class-difference;
- AE masih memakai MSE biasa, belum masked loss atau objective yang menjaga
  sinyal fraud ranking.

## Kesimpulan Penyebab Utama

Penyebab paling kuat berdasarkan bukti:

1. Full replacement `V1-V339 -> latent` membuang sinyal granular fitur V.
2. Missingness V membawa informasi fraud, tetapi pipeline AE proposal tidak
   membawa missingness mask.
3. Objective AE MSE unsupervised tidak sama dengan objective AP/PR-AUC.
4. Memperbesar latent dimension tidak menyelesaikan akar masalah karena desain
   replacement dan objective tetap sama.
5. AE lebih masuk akal sebagai perbaikan terbatas untuk subset V noisy/missing,
   bukan sebagai pengganti seluruh V block.

## Alternatif Perbaikan Yang Masih Sesuai Tujuan Awal

Prioritas berikut disusun dari penyebab yang ditemukan, bukan loncat metode.

| Prioritas | Alternatif | Penyebab yang dijawab | Mengapa masih sesuai koridor |
|---:|---|---|---|
| 1 | Missingness-aware partial AE | Missingness V hilang pada AE proposal | AE tetap memperbaiki fitur V; LightGBM tetap classifier. |
| 2 | Masked reconstruction loss untuk V observed-only | AE belajar merekonstruksi 0 hasil imputasi | Masih Autoencoder, hanya loss dibuat lebih sesuai data missing. |
| 3 | Partial reconstruction dengan seleksi subset lebih ketat | Threshold 75% terlalu kasar | AE tetap diterapkan pada V bermasalah, bukan full replacement. |
| 4 | Reconstruction error features, bukan reconstructed values penuh | Rekonstruksi meratakan anomaly | AE memberi sinyal anomaly tambahan ke LightGBM. |
| 5 | Linear/tanh latent + regularized AE | ReLU latent pada data z-score dan latent noisy | Masih ablation arsitektur AE. |
| 6 | Supervised auxiliary AE sebagai opsi lanjutan | MSE tidak label-aligned | Tetap AE feature learning, tetapi perlu disetujui sebagai pengembangan. |

### 1. Missingness-Aware Partial AE

Desain:

- fokus pada subset high-missing V;
- input AE berisi nilai V yang diimputasi dan mask missingness;
- LightGBM menerima fitur original/reconstructed dan mask missingness;
- evaluasi tetap melawan tuned LightGBM AP 0.873133.

Alasan:

```text
Diagnosis menunjukkan missingness V berbeda antara fraud dan non-fraud. Jadi
perbaikan paling langsung adalah membawa mask tersebut, bukan menghapusnya.
```

### 2. Masked Reconstruction Loss

Desain:

- missing V tetap diimputasi untuk kebutuhan tensor;
- loss rekonstruksi dihitung hanya pada elemen observed, bukan pada missing
  yang berubah menjadi 0 artifisial;
- mask dapat juga direkonstruksi sebagai auxiliary output bila diperlukan.

Alasan:

```text
AE saat ini berisiko belajar merekonstruksi pola 0 hasil imputasi, bukan hanya
struktur nilai V yang benar-benar teramati.
```

### 3. Partial Reconstruction Dengan Subset Lebih Ketat

Desain:

- jangan langsung semua 159 fitur high-missing;
- uji subset berdasarkan kombinasi:
  - missing rate tinggi;
  - baseline feature importance tinggi;
  - perbedaan missingness fraud vs non-fraud tinggi.

Contoh kandidat:

```text
V high-missing + high importance
V high-missing + class-missingness gap besar
V high-missing dengan threshold 0.70 / 0.80 / 0.85
```

Alasan:

```text
Partial reconstruction paling mendekati baseline. Maka langkah berikutnya yang
paling defensible adalah memperbaiki subset dan mask-nya, bukan mengganti arah
metode.
```

### 4. Reconstruction Error Features

Desain:

- pertahankan semua fitur original;
- tambahkan error AE per row atau per group V sebagai fitur;
- jangan mengganti nilai V dengan rekonstruksi penuh.

Alasan:

```text
Jika fraud adalah pola yang sulit direkonstruksi oleh AE, reconstruction error
bisa menjadi sinyal anomaly tanpa merusak fitur original yang sudah kuat.
```

### 5. Latent Activation dan Regularization Ablation

Desain:

- latent activation linear atau tanh;
- dropout/sparse penalty/contractive penalty ringan;
- latent dimension kecil-menengah, misalnya 16/32/64, bukan langsung besar.

Alasan:

```text
Input AE sudah z-score, sehingga informasi negatif mungkin penting. ReLU latent
tidak negatif dapat membatasi representasi. Namun ini prioritas lebih rendah
daripada missingness-aware partial AE karena LD64/128/256 sudah tidak
memulihkan AP.
```

### 6. Supervised Auxiliary AE

Desain:

- encoder tetap belajar rekonstruksi;
- tambahkan auxiliary fraud prediction head saat training AE;
- downstream tetap LightGBM memakai fitur AE.

Catatan:

```text
Ini masih AE feature learning + LightGBM, tetapi sudah lebih jauh dari proposal
literal karena AE mulai memakai label. Jadi sebaiknya ditempatkan sebagai opsi
lanjutan setelah missingness-aware dan masked-loss AE.
```

## Yang Tidak Disarankan Dilanjutkan

Untuk menjaga eksperimen tetap efisien dan sesuai bukti, varian berikut tidak
menjadi prioritas:

- full replacement `V1-V339` dengan latent AE biasa;
- sekadar memperbesar latent dimension tanpa mask/loss baru;
- blind concat dengan latent dimension makin besar;
- full-matrix Ding-style reconstruction untuk IEEE-CIS;
- pindah ke model selain LightGBM sebagai solusi utama.

## Rekomendasi Eksperimen Berikutnya

Urutan paling defensible:

1. `AE-PARTIAL-MASK`: partial high-missing AE + missingness mask ke LightGBM.
2. `AE-PARTIAL-MASKEDLOSS`: partial AE dengan masked reconstruction loss.
3. `AE-PARTIAL-SELECT`: subset high-missing diseleksi memakai importance dan
   class-missingness gap.
4. `AE-ERROR-FEATURES`: reconstruction error per row/group sebagai fitur
   tambahan.
5. `AE-LATENT-ACT`: latent linear/tanh + regularization sebagai ablation
   arsitektur.

Stop rule tetap:

```text
Berhenti jika test AP > 0.873133233976 dan bootstrap delta AP positif.
```

Jika semua masih kalah, kesimpulan thesis-facing tetap defensible:

```text
Pada IEEE-CIS stratified protocol, LightGBM tuned dengan fitur original adalah
baseline terkuat. AE tidak terbukti meningkatkan AP untuk varian yang diuji.
Namun eksperimen menunjukkan penyebabnya secara jelas: full replacement
menghilangkan sinyal V dan missingness. Varian AE yang paling masuk akal adalah
partial/missingness-aware feature improvement, bukan pengganti seluruh fitur V.
```

## Status Setelah Rekomendasi Dijalankan

Rencana missingness-aware, masked-loss, selective subset, reconstruction-error,
dan latent-activation ablation sudah dieksekusi pada 2026-06-20 melalui:

```text
src/run_ae_diagnosis_fix_ladder.py
outputs/stratified_reset/ae_diagnosis_fix_ladder/fix_ladder_summary.json
docs/AE_DIAGNOSIS_FIX_LADDER_RESULTS.md
```

Reference tetap:

```text
Tuned LightGBM proposal original test AP = 0.873133233976
```

Hasil:

| Tahap | Kandidat | Test AP | Delta vs tuned LightGBM | Makna |
|---:|---|---:|---:|---|
| 1 | Missingness-aware partial AE replace + mask | 0.870329 | -0.002804 | Mask membantu secara konseptual, tetapi rekonstruksi masih meratakan sinyal V. |
| 2 | Masked-loss partial AE, observed-only replace | 0.871646 | -0.001487 | Terbaik; mempersempit gap tetapi belum menang. |
| 3 | Selective top-64 masked-loss partial AE | 0.871624 | -0.001509 | Seleksi subset menaikkan validation AP, tetapi test AP tidak melewati fix2. |
| 4 | Reconstruction-error features | 0.869771 | -0.003363 | Error AE masih redundant/noisy terhadap fitur V original. |
| 5 | Linear latent concat LD32 | 0.864650 | -0.008483 | Aktivasi latent bukan penyebab utama. |

Kesimpulan setelah rekomendasi dijalankan:

```text
Tidak ada alternatif perbaikan dalam keluarga AE feature improvement yang
mengalahkan tuned LightGBM. Perbaikan terbaik adalah masked-loss partial AE,
tetapi AP tetap lebih rendah. Jadi penyebab kegagalan bukan semata bug atau
ketidaksesuaian implementasi; LightGBM tuned dengan fitur original memang masih
lebih kuat untuk protocol ini.
```

## Status Setelah Broad AE Feature Ladder

Pertanyaan tambahan yang diuji:

```text
Apakah AE perlu diperluas, tidak hanya pada V feature, tetapi tetap sebagai
feature improvement untuk LightGBM?
```

Eksperimen dijalankan pada 2026-06-20 melalui:

```text
src/run_broad_ae_feature_ladder.py
outputs/stratified_reset/broad_ae_feature_ladder/
docs/BROAD_AE_FEATURE_LADDER_RESULTS.md
```

Catatan runtime:

```text
Matriks augmented LightGBM sempat native crash pada full tuned 999 estimator,
sehingga broad ladder diselesaikan dengan cap 600 estimator. Baseline referensi
tetap tuned LightGBM full test AP 0.873133233976.
```

Hasil:

| Tahap | Kandidat | Test AP | Delta vs tuned LightGBM | Makna |
|---:|---|---:|---:|---|
| 1 | All-feature AE top-192, latent + error | 0.857028 | -0.016105 | AE lintas fitur global tetap kehilangan/menambah noise. |
| 2 | Group-wise AE latent + error | 0.868076 | -0.005057 | Terbaik di broad ladder; memisahkan keluarga fitur membantu, tetapi belum menang. |
| 3 | Value + missingness-mask reconstruction | 0.857716 | -0.015418 | Rekonstruksi mask tidak cukup menjadi sinyal tambahan. |
| 4 | Normal-only AE anomaly error | 0.867750 | -0.005383 | Error anomaly berguna terbatas, tetapi masih kalah. |
| 5 | Supervised auxiliary AE | 0.848394 | -0.024739 | Auxiliary fraud head belajar sinyal, tetapi menjadi noisy/redundant untuk LightGBM. |

Kesimpulan tambahan:

```text
Membuka AE ke fitur non-V tidak menyelesaikan masalah utama. Bahkan ketika fitur
original tetap dipertahankan dan AE hanya ditambahkan sebagai latent/error/aux
features, tuned LightGBM original tetap lebih kuat. Jadi penyebab kegagalan
bukan sekadar "AE hanya dipasang pada V", melainkan representasi AE yang diuji
belum menambah informasi yang cukup di atas fitur original IEEE-CIS.
```

## Narasi Singkat Untuk Pembimbing

```text
Saya sudah kembali ke desain proposal original dan menjalankannya secara literal
dengan split stratified 60/20/20. Hasilnya AE latent replacement memang belum
mengalahkan tuned LightGBM: test AP 0.860110 vs 0.873133. Setelah dianalisis,
penyebab utamanya bukan sekadar latent_dim terlalu kecil. Baseline masih memakai
fitur V original dengan gain sekitar 30%, dan beberapa fitur V penting memiliki
pola missingness yang berbeda tajam antara fraud dan non-fraud. Pipeline AE
proposal mengisi missing dengan 0 lalu mengganti seluruh V dengan latent dense,
sehingga sinyal granular dan missingness hilang. Ladder perbaikan menunjukkan
bahwa full replacement tetap turun, concat membaik tetapi belum menang,
denoising belum membantu test, dan partial reconstruction high-missing paling
mendekati baseline dengan delta -0.001824. Jadi alternatif berikutnya tetap
dalam koridor AE + LightGBM: missingness-aware partial AE, masked reconstruction
loss, subset high-missing yang lebih selektif, dan reconstruction-error features.
Alternatif lanjutan juga sudah diuji dengan AE tidak hanya pada V: all-feature
AE, group-wise AE, value+mask reconstruction, normal-only anomaly AE, dan
supervised auxiliary AE. Hasil terbaiknya group-wise AE test AP 0.868076, masih
kalah dari tuned LightGBM 0.873133. Jadi kesimpulannya, kegagalan bukan hanya
karena AE dibatasi pada V, melainkan AE feature learning yang diuji belum
menambah informasi cukup di atas fitur original yang sudah sangat kuat untuk
LightGBM.
```
