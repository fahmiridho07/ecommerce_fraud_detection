# AE Diagnosis-Driven Fix Ladder Results

Status: advisor-facing result, 2026-06-20.

Eksperimen ini melanjutkan diagnosis kegagalan AE feature improvement pada
desain proposal original. Tujuannya tetap sama:

```text
Autoencoder memperbaiki atau melengkapi fitur untuk LightGBM.
```

Tidak ada perpindahan ke model classifier lain. Semua kandidat dibandingkan
dengan tuned LightGBM dari rerun proposal original.

## Reference

```text
Original proposal tuned LightGBM
validation AP = 0.874588541619
test AP       = 0.873133233976
```

Stop rule:

```text
Berhenti pada kandidat pertama dengan selected test AP > 0.873133233976.
```

Output utama:

```text
outputs/stratified_reset/ae_diagnosis_fix_ladder/fix_ladder_summary.json
```

Harness:

```text
src/run_ae_diagnosis_fix_ladder.py
```

## Protocol

- Dataset: IEEE-CIS Fraud Detection.
- Split: stratified holdout 60/20/20, `random_state=42`.
- Primary metric: Average Precision / PR-AUC.
- Feature family: original proposal V-block AE plus LightGBM.
- Candidate selection: tiap kandidat melatih dua fixed tuned LightGBM profiles,
  lalu memilih profile dengan validation AP tertinggi.
- Bootstrap comparison: paired bootstrap AP delta terhadap tuned LightGBM,
  `n_bootstrap=300`.

## Summary Table

| Candidate | Purpose | Selected profile | Val AP | Test AP | Delta vs tuned LightGBM | 95% bootstrap CI | p(delta<=0) |
|---|---|---|---:|---:|---:|---|---:|
| `fix1_partial_mask_recon_replace` | Partial AE high-missing V + missing mask | baseline_tuned | 0.868888 | 0.870329 | -0.002804 | [-0.005305, 0.000031] | 0.973 |
| `fix2_partial_maskedloss_observed_replace` | Masked loss + replace observed values only | baseline_tuned | 0.869367 | 0.871646 | -0.001487 | [-0.003907, 0.000963] | 0.900 |
| `fix3_select64_maskedloss_observed_replace` | Selective top-64 high-missing V by importance and class gap | baseline_tuned | 0.870663 | 0.871624 | -0.001509 | [-0.003771, 0.000845] | 0.897 |
| `fix4_recon_error_append_high_missing` | Keep V original + append AE reconstruction-error features | baseline_tuned | 0.867325 | 0.869771 | -0.003363 | [-0.005832, -0.001094] | 0.997 |
| `fix5_linear_latent_concat_ld32` | Keep V original + append linear latent LD32 | baseline_tuned | 0.863010 | 0.864650 | -0.008483 | [-0.011304, -0.006114] | 1.000 |

Tidak ada kandidat yang mengalahkan tuned LightGBM. Stop rule tidak pernah
terpenuhi (`stopped_after_stage = null`).

Best candidate:

```text
fix2_partial_maskedloss_observed_replace
test AP = 0.871645912542
delta   = -0.001487321434
```

## Stage Analysis

### 1. Missingness-Aware Partial AE

`fix1_partial_mask_recon_replace` menguji penyebab bahwa proposal original
menghapus pola missingness V. AE menerima value + missing mask, dan LightGBM
menerima reconstructed high-missing V plus mask.

Hasil:

```text
test AP = 0.870329
delta   = -0.002804
```

Interpretasi: membawa mask belum cukup. Selama nilai high-missing V tetap
diganti oleh rekonstruksi AE berbasis MSE biasa, sebagian sinyal granular dan
sinyal anomaly masih diratakan.

### 2. Masked Loss and Observed-Only Replacement

`fix2_partial_maskedloss_observed_replace` membuat loss hanya dihitung pada
elemen observed, lalu hanya mengganti nilai V yang memang observed. Missing
tetap tidak dipaksa menjadi nilai rekonstruksi penuh.

Hasil:

```text
test AP = 0.871646
delta   = -0.001487
```

Interpretasi: ini kandidat terbaik. Masked loss menjawab sebagian masalah
missing-as-zero dan observed-only replacement mengurangi kerusakan sinyal.
Namun delta masih negatif, dan bootstrap masih lebih sering mendukung baseline
(`p(delta<=0)=0.900`). Jadi perbaikan ini menutup gap, tetapi belum cukup untuk
klaim AE menang.

### 3. Selective Partial AE

`fix3_select64_maskedloss_observed_replace` memperketat subset menjadi 64 fitur
V high-missing yang dipilih dari kombinasi baseline importance, missing rate,
dan class-missingness gap.

Hasil:

```text
validation AP = 0.870663
test AP       = 0.871624
delta         = -0.001509
```

Interpretasi: seleksi subset menaikkan validation AP menjadi yang tertinggi di
antara kandidat diagnosis, tetapi test AP hampir sama dengan fix2 dan sedikit
lebih rendah. Artinya, memperketat subset membantu mengurangi noise, tetapi
sinyal tambahan AE masih belum cukup robust pada test set.

### 4. Reconstruction Error Features

`fix4_recon_error_append_high_missing` tidak mengganti V original. Semua fitur
original dipertahankan, lalu error rekonstruksi AE ditambahkan sebagai fitur
anomaly.

Hasil:

```text
test AP = 0.869771
delta   = -0.003363
```

Interpretasi: reconstruction error tidak menjadi sinyal tambahan yang cukup
berguna untuk LightGBM tuned. Karena baseline sudah melihat V original dan
missingness, error AE cenderung redundant atau noisy terhadap objective AP.

### 5. Linear Latent Concatenation

`fix5_linear_latent_concat_ld32` menguji apakah ReLU latent pada input z-score
menjadi masalah. V original dipertahankan dan latent AE LD32 dengan aktivasi
linear ditambahkan.

Hasil:

```text
test AP = 0.864650
delta   = -0.008483
```

Interpretasi: linear latent tidak membantu. Ini memperkuat diagnosis bahwa
masalah utama bukan sekadar aktivasi latent, tetapi mismatch antara objective
rekonstruksi unsupervised dan ranking fraud yang dioptimalkan oleh PR-AUC.

## Overall Diagnosis

Hasil ini konsisten dengan ladder sebelumnya:

- full V replacement adalah sumber penurunan terbesar;
- mempertahankan fitur V original membantu;
- partial/missingness-aware AE adalah arah yang paling masuk akal;
- masked loss memperbaiki gap, tetapi belum melampaui tuned LightGBM;
- reconstruction error dan latent add-on masih redundant/noisy untuk baseline
  LightGBM yang sudah kuat.

Kesimpulan advisor-facing:

```text
Saya sudah kembali ke desain proposal original, menguji implementasi literal,
lalu menguji alternatif perbaikan yang tetap berada dalam koridor AE memperbaiki
fitur untuk LightGBM. Hasil terbaik adalah masked-loss partial AE dengan test AP
0.871646, tetapi tuned LightGBM masih lebih tinggi pada 0.873133. Jadi saya
tidak mengklaim AE menang sebagai feature extractor. Temuan utamanya adalah
penyebab kegagalan dapat dijelaskan: AE replacement mereduksi sinyal granular V,
missingness, dan ranking fraud. Perbaikan partial/masked-loss mempersempit gap,
tetapi belum cukup untuk melewati baseline.
```
