---
id: Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection
priority: core
authors: "Ding et al."
year: 2024
doi: "10.7717/peerj-cs.2323"
dataset: "credit-card (not IEEE-CIS)"
method: "AE + LightGBM"
metrics: "Recall, F-measure, AUC, MCC, BCR"
split: "unspecified"
comparable_to_thesis: "partial"
thesis_use: "Bab 2 — precedent arsitektur AE+LightGBM"
bab: "2"
pdf: "../../2. Reference/05_Integrasi_Autoencoder_LightGBM/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.pdf"
---

# Ding et al. (2024)

## Ringkasan

- Pipeline paling dekat dengan P03/P04: AE rekonstruksi fitur lalu LightGBM klasifikasi.
- Evaluasi pada dataset kartu kredit 31 fitur + Santander 200 fitur, bukan IEEE-CIS.
- SMOTE + model terbaik melaporkan AUC ~96.8%; gunakan sebagai motivasi metode, bukan benchmark angka.

## File

- PDF (source of truth): `2. Reference/05_Integrasi_Autoencoder_LightGBM/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.pdf`
- Kartu ini untuk ringkasan; verifikasi angka/kutipan ke PDF.
