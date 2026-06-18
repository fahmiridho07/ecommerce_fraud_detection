---
id: Dablain_2022_DeepSMOTE
priority: core
authors: "Dablain, Krawczyk & Chawla"
year: 2022
doi: "10.1109/TNNLS.2021.3136503"
dataset: "image (MNIST, Fashion-MNIST, CIFAR-10, SVHN, CelebA)"
method: "Encoder-decoder (autoencoder) + SMOTE di ruang laten + decode"
metrics: "ACSA, GM, akurasi kelas minoritas"
split: "N/A"
comparable_to_thesis: "method-anchor"
thesis_use: "Bab 2/3 - anchor langsung metode usulan: AE latent-space oversampling"
bab: "2,3"
pdf: "../../2. Reference/02_Ketidakseimbangan_Kelas/Dablain_2022_DeepSMOTE.pdf"
---

# Dablain et al. (2022) - DeepSMOTE

## Ringkasan

- DeepSMOTE = "Fusing Deep Learning and SMOTE for Imbalanced Data" (IEEE TNNLS).
  Sebuah encoder-decoder (autoencoder) belajar embedding berdimensi rendah; SMOTE
  diterapkan **di ruang laten**, lalu decoder mengembalikan sampel ke ruang fitur.
- Klaim inti: interpolasi di ruang laten yang dipelajari menghasilkan sampel
  minoritas sintetis yang lebih **on-manifold** dan realistis daripada SMOTE klasik
  di ruang fitur mentah ([[Chawla_2002_SMOTE]]).
- **Anchor metodologis persis untuk metode pemenang tesis ini**: AE encode ->
  SMOTE di latent -> decode. Tesis mengadaptasi prinsip dari domain citra ke data
  **tabular fraud terenkode-frekuensi (representasi padat A1)**.
- Konsisten dengan temuan tesis: keunggulan oversampling ruang-laten atas SMOTE
  muncul pada representasi padat berkorelasi (A1), dan menyusut/seri pada
  representasi mentah jarang (A0). Lihat juga augmentasi generatif AE pada IEEE-CIS
  [[Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS]].

## File

- PDF (source of truth): `2. Reference/02_Ketidakseimbangan_Kelas/Dablain_2022_DeepSMOTE.pdf`
  (CATATAN: PDF belum ada di folder lokal; DOI 10.1109/TNNLS.2021.3136503, IEEE TNNLS 34(9):6390-6404).
- Kartu ini untuk ringkasan; verifikasi angka/kutipan ke PDF.
