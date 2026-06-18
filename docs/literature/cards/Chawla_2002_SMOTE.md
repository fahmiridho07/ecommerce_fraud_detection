---
id: Chawla_2002_SMOTE
priority: core
authors: "Chawla et al."
year: 2002
doi: "10.1613/jair.953"
dataset: "UCI benchmarks (various)"
method: "Synthetic Minority Over-sampling Technique (interpolasi k-NN ruang fitur)"
metrics: "ROC, precision/recall"
split: "N/A"
comparable_to_thesis: "method-control"
thesis_use: "Bab 2/3 - oversampling klasik; kontrol pembanding untuk augmentasi AE"
bab: "2,3"
pdf: "../../2. Reference/02_Ketidakseimbangan_Kelas/Chawla_2002_SMOTE.pdf"
---

# Chawla et al. (2002)

## Ringkasan

- SMOTE membuat sampel kelas minoritas sintetis dengan interpolasi linier antara
  satu instance minoritas dan salah satu tetangga k-NN-nya **di ruang fitur**.
- Metode oversampling rujukan yang menjadi dasar banyak varian; pada tesis ini
  menjadi **kontrol "interpolasi ruang fitur mentah"** yang dibandingkan terhadap
  oversampling ruang-laten Autoencoder (DeepSMOTE, [[Dablain_2022_DeepSMOTE]]).
- Untuk fitur campuran numerik+kategorikal dipakai varian **SMOTE-NC** (numerik
  diinterpolasi; kategorikal diambil dari anchor/tetangga), yaitu kontrol langsung
  pada eksperimen A0/A1 tesis.
- Hanya diterapkan pada split train (mencegah leakage, lihat
  [[Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection]]).

## File

- PDF (source of truth): `2. Reference/02_Ketidakseimbangan_Kelas/Chawla_2002_SMOTE.pdf`
  (CATATAN: PDF belum ada di folder lokal; DOI 10.1613/jair.953, JAIR 16:321-357).
- Kartu ini untuk ringkasan; verifikasi angka/kutipan ke PDF.
