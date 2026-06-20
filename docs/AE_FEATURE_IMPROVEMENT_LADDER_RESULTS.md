# AE Feature Improvement Ladder Results

Status: advisor-facing follow-up, 2026-06-19.

Tujuan eksperimen ini adalah menguji alternatif perbaikan yang tetap berada di
koridor usulan awal: Autoencoder memperbaiki atau melengkapi fitur untuk
LightGBM, bukan melompat ke metode lain.

Reference yang dipakai untuk stop rule:

```text
Original proposal tuned LightGBM
validation AP = 0.874588541619
test AP       = 0.873133233976
```

Output utama:

```text
outputs/stratified_reset/ae_feature_improvement_ladder/ladder_summary.json
```

Deep diagnosis lanjutan:

```text
docs/AE_FAILURE_DEEP_DIAGNOSIS_AND_FIX_PLAN.md
```

Harness:

```text
src/run_ae_feature_improvement_ladder.py
```

## Protocol

- Dataset: IEEE-CIS Fraud Detection.
- Split: stratified holdout 60/20/20, `random_state=42`.
- Primary metric: Average Precision / PR-AUC.
- Candidate LightGBM selection: train two fixed tuned profiles per candidate,
  then choose by validation AP.
- Stop rule: stop if selected candidate test AP is greater than tuned LightGBM
  reference test AP `0.873133233976`.
- Bootstrap: stage 1 candidates use the previously completed 1000 bootstrap
  replicates; stage 2 onward use 100 replicates for ladder screening. The final
  decision still uses direct test AP against the reference.

## Summary Table

| Candidate | Stage | Selected profile | Val AP | Test AP | Delta vs tuned LightGBM | p(delta<=0) |
|---|---:|---|---:|---:|---:|---:|
| `s1_replace_ld64` | 1 | baseline_tuned | 0.858531 | 0.859875 | -0.013258 | 1.00 |
| `s1_replace_ld128` | 1 | baseline_tuned | 0.855742 | 0.856170 | -0.016963 | 1.00 |
| `s1_replace_ld256` | 1 | ae_tuned_ld32 | 0.855562 | 0.854391 | -0.018742 | 1.00 |
| `s2_concat_ld32` | 2 | baseline_tuned | 0.863734 | 0.864217 | -0.008917 | 1.00 |
| `s2_concat_ld64` | 2 | baseline_tuned | 0.861460 | 0.861768 | -0.011365 | 1.00 |
| `s2_concat_ld128` | 2 | baseline_tuned | 0.860183 | 0.859573 | -0.013560 | 1.00 |
| `s3_denoising_concat_ld32` | 3 | baseline_tuned | 0.864072 | 0.864170 | -0.008963 | 1.00 |
| `s4_partial_recon_replace_high_missing` | 4 | baseline_tuned | 0.869678 | 0.871309 | -0.001824 | 0.89 |
| `s4_partial_recon_append_high_missing` | 4 | baseline_tuned | 0.869992 | 0.868841 | -0.004292 | 1.00 |

No candidate beat the tuned LightGBM reference.

Best candidate by test AP:

```text
s4_partial_recon_replace_high_missing
test AP = 0.871309098418
delta   = -0.001824135558
```

## Stage Analysis

### Stage 1: Larger Latent Dimension, Still Replacing V

Candidates:

- `replace_ld64`: test AP 0.859875.
- `replace_ld128`: test AP 0.856170.
- `replace_ld256`: test AP 0.854391.

Finding: memperbesar `latent_dim` tidak memulihkan AP. Bahkan semakin besar
latent dimension, performa cenderung turun.

Interpretation: penyebab penurunan bukan hanya kompresi 32 dimensi yang terlalu
kecil. Selama desainnya tetap mengganti semua `V1-V339`, LightGBM kehilangan
sinyal granular dan pola missingness dari fitur V asli. Latent yang lebih besar
juga dapat membawa noise rekonstruksi yang tidak label-guided.

### Stage 2: Concatenate, Not Replace

Candidates:

- `concat_ld32`: test AP 0.864217.
- `concat_ld64`: test AP 0.861768.
- `concat_ld128`: test AP 0.859573.

Finding: concatenate lebih baik daripada replace karena fitur V asli tetap
dipertahankan. Namun hasilnya masih di bawah tuned LightGBM. LD32 adalah varian
concat terbaik; latent yang lebih besar justru turun.

Interpretation: AE latent memang bisa menambah sedikit sinyal, tetapi sinyal
tambahan itu tidak cukup untuk mengalahkan LightGBM yang sudah melihat semua
fitur original. Penurunan LD64/LD128 menunjukkan fitur latent tambahan mulai
lebih redundan/noisy daripada informatif.

### Stage 3: Denoising Autoencoder

Candidate:

- `denoising_concat_ld32`: test AP 0.864170.

Finding: denoising sedikit menaikkan validation AP dibanding concat LD32
regular, tetapi test AP tidak membaik. Test AP denoising `0.864170` sedikit di
bawah concat LD32 regular `0.864217`.

Interpretation: denoising AE berhasil memperbaiki objective rekonstruksi, tetapi
objective itu tetap tidak sama dengan objective fraud ranking. Membersihkan
noise pada `V` tidak otomatis membuat fitur lebih diskriminatif untuk PR-AUC.

### Stage 4: Partial Reconstruction of High-Missing V

Subset partial reconstruction memilih 159 fitur `V` dengan missing rate
`>= 0.75`.

Candidates:

- replace high-missing V with AE reconstruction: test AP 0.871309.
- append high-missing V reconstructions: test AP 0.868841.

Finding: partial replace adalah perbaikan terbaik dalam ladder. Gap terhadap
tuned LightGBM turun dari sekitar `-0.0130` pada AE latent replacement original
menjadi `-0.001824`. Namun kandidat ini tetap belum mengalahkan baseline.

Interpretation: ini mendukung diagnosis bahwa masalah utama ada pada full
replacement/full compression. Ketika AE hanya dipakai pada bagian V yang lebih
bermasalah dan fitur V lain tetap dipertahankan, performa jauh membaik. Namun
AE reconstruction masih berpotensi meratakan sinyal missing/anomaly yang justru
berguna bagi LightGBM, sehingga belum melewati baseline.

Append partial punya validation AP tertinggi di antara kandidat AE
(`0.869992`), tetapi test AP turun ke `0.868841`. Ini menunjukkan fitur
rekonstruksi tambahan dapat meningkatkan validation fit, tetapi tidak cukup
robust pada test set.

## Diagnosis Updated

Hasil ladder memperkuat diagnosis awal:

1. Full V latent replacement adalah sumber penurunan terbesar.
2. Mempertahankan V asli membantu, sehingga informasi original V memang penting.
3. Menambah kapasitas latent tidak menyelesaikan masalah, sehingga bukan hanya
   isu under-compression.
4. Denoising membantu rekonstruksi/validasi kecil, tetapi tidak mengubah sinyal
   fraud ranking secara robust.
5. Partial reconstruction adalah arah paling masuk akal dalam koridor AE +
   LightGBM, tetapi masih belum cukup untuk mengalahkan tuned LightGBM.

## Advisor-Facing Conclusion

Eksperimen sudah mengikuti masukan pembimbing: kembali ke usulan semula,
menemukan penyebab penurunan, lalu menguji perbaikan yang masih sejalur dengan
tujuan awal.

Narasi yang defensible:

```text
Saya sudah menguji empat alternatif perbaikan yang tetap berada dalam koridor
Autoencoder + LightGBM. Hasilnya tidak ada kandidat AE yang mengalahkan
LightGBM tuned, sehingga saya tidak mengklaim AE menang sebagai feature
extractor. Namun ladder ini menjelaskan penyebabnya: desain awal yang mengganti
semua fitur V dengan latent AE menyebabkan information loss. Ketika fitur V asli
dipertahankan atau hanya fitur V high-missing yang direkonstruksi, performa
membaik signifikan dan gap menyempit sampai -0.001824 AP, tetapi tetap belum
melewati baseline. Jadi kesimpulan sementara adalah LightGBM tuned dengan fitur
V asli masih lebih kuat, sementara AE paling layak diposisikan sebagai perbaikan
terbatas pada fitur noisy/missing, bukan pengganti seluruh fitur V.
```
