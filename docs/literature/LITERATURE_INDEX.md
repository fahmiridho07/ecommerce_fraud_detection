# Literature Index — Tugas Akhir IEEE-CIS

Indeks 55 referensi. **Kartu ringkas** di `_cards/`; **PDF** di `2. Reference/` adalah source of truth untuk angka dan kutipan.

Status note: experiment metrics listed here are historical chronological
references unless explicitly marked as stratified rerun results.

## Thesis experiment baseline (bandingkan ke literatur)

| Model | Test AP | Catatan |
|-------|---------|---------|
| P02 BASE-02 (tuned LGBM) | **0.5049** | Protokol chronological - `archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md` |
| Moradi 2025 (literature) | 0.891 AUC-PR | FE + SMOTE; **tidak comparable** |
| P02 vs prevalence ~3.5% | — | Laporkan no-skill baseline (Williams 2021) |

## Core papers (20)

| Priority | Authors | Year | Comparable? | Bab | PDF | Card |
|----------|---------|------|-------------|-----|-----|------|
| core | Ali et al. | 2022 | background | 2 | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Ali_2022_Financial_Fraud_Detection_SLR.pdf` | `_cards/Ali_2022_Financial_Fraud_Detection_SLR.md` |
| core | Dal Pozzolo et al. | 2018 | protocol | 3 | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.pdf` | `_cards/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.md` |
| core | Thimonier et al. | 2023 | discussion | 5 | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.pdf` | `_cards/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.md` |
| core | Kabane & Ouali | 2024 | protocol | 3 | `2. Reference/02_Ketidakseimbangan_Kelas/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.pdf` | `_cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md` |
| core | Niu et al. | 2019 | discussion | 5 | `2. Reference/03_Autoencoder/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.pdf` | `_cards/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.md` |
| core | Ke et al. | 2017 | theory | 2 | `2. Reference/04_LightGBM/LightGBM A Highly Efficient Gradient Boosting.pdf` | `_cards/LightGBM A Highly Efficient Gradient Boosting.md` |
| core | Ding et al. | 2024 | partial | 2 | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.pdf` | `_cards/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.md` |
| core | Du et al. | 2023 | partial | 2 | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.pdf` | `_cards/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.md` |
| core | Prabha & Priscilla | 2024 | partial | 2 | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.pdf` | `_cards/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.md` |
| core | Akiba et al. | 2019 | method | 2,3 | `2. Reference/06_Bayesian_Optimization_Optuna/Akiba_2019_Optuna_Hyperparameter_Optimization.pdf` | `_cards/Akiba_2019_Optuna_Hyperparameter_Optimization.md` |
| core | Saito & Rehmsmeier | 2015 | metrics | 2 | `2. Reference/07_Metrik_Evaluasi/Saito_Rehmsmeier_2015_Precision_Recall_Plot.pdf` | `_cards/Saito_Rehmsmeier_2015_Precision_Recall_Plot.md` |
| core | Williams | 2021 | metrics | 2,3 | `2. Reference/07_Metrik_Evaluasi/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.pdf` | `_cards/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.md` |
| core | Alharbi et al. | 2026 | no | 2 | `2. Reference/08_Dataset_IEEE-CIS/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.pdf` | `_cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md` |
| core | Bakhtiari et al. | 2023 | partial | 2 | `2. Reference/08_Dataset_IEEE-CIS/Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS.pdf` | `_cards/Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS.md` |
| core | Carcillo et al. | 2018 | protocol | 3 | `2. Reference/08_Dataset_IEEE-CIS/Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud.pdf` | `_cards/Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud.md` |
| core | Jiang et al. | 2023 | partial | 2 | `2. Reference/08_Dataset_IEEE-CIS/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.pdf` | `_cards/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md` |
| core | Lucas et al. | 2019 | protocol | 3 | `2. Reference/08_Dataset_IEEE-CIS/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.pdf` | `_cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md` |
| core | Lucas et al. | 2019 | future-work | 2,5 | `2. Reference/08_Dataset_IEEE-CIS/Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud.pdf` | `_cards/Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud.md` |
| core | Moradi et al. | 2025 | no | 2,5 | `2. Reference/08_Dataset_IEEE-CIS/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.pdf` | `_cards/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md` |
| core | Nguyen et al. | 2022 | partial | 2 | `2. Reference/08_Dataset_IEEE-CIS/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.pdf` | `_cards/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.md` |

## Semua referensi (55)

| Folder | File | Priority | PDF | Card |
|--------|------|----------|-----|------|
| 01_Deteksi_Penipuan_E-Commerce | Ali_2022_Financial_Fraud_Detection_SLR | core | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Ali_2022_Financial_Fraud_Detection_SLR.pdf` | `_cards/Ali_2022_Financial_Fraud_Detection_SLR.md` |
| 01_Deteksi_Penipuan_E-Commerce | Ashtiani_Raahemi_2022_Intelligent_Fraud_Detection_SLR | supporting | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Ashtiani_Raahemi_2022_Intelligent_Fraud_Detection_SLR.pdf` | — |
| 01_Deteksi_Penipuan_E-Commerce | Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy | core | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.pdf` | `_cards/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.md` |
| 01_Deteksi_Penipuan_E-Commerce | Financial Fraud A Review of Anomaly Detection Techniques and Recent Advance | supporting | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Financial Fraud A Review of Anomaly Detection Techniques and Recent Advance.pdf` | — |
| 01_Deteksi_Penipuan_E-Commerce | Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments | core | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.pdf` | `_cards/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.md` |
| 02_Ketidakseimbangan_Kelas | Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection | core | `2. Reference/02_Ketidakseimbangan_Kelas/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.pdf` | `_cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md` |
| 02_Ketidakseimbangan_Kelas | Survey on deep learning with class imbalance | supporting | `2. Reference/02_Ketidakseimbangan_Kelas/Survey on deep learning with class imbalance.pdf` | — |
| 02_Ketidakseimbangan_Kelas | Zhao_2023_Improved_LightGBM_Imbalanced_Data | supporting | `2. Reference/02_Ketidakseimbangan_Kelas/Zhao_2023_Improved_LightGBM_Imbalanced_Data.pdf` | — |
| 03_Autoencoder | Hinton_Salakhutdinov_2006_Reducing_Dimensionality_Neural_Networks | supporting | `2. Reference/03_Autoencoder/Hinton_Salakhutdinov_2006_Reducing_Dimensionality_Neural_Networks.pdf` | — |
| 03_Autoencoder | Misra_2020_Autoencoder_Fraudulent_Credit_Card_Transaction | supporting | `2. Reference/03_Autoencoder/Misra_2020_Autoencoder_Fraudulent_Credit_Card_Transaction.pdf` | — |
| 03_Autoencoder | Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection | core | `2. Reference/03_Autoencoder/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.pdf` | `_cards/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.md` |
| 03_Autoencoder | Vincent_2010_Stacked_Denoising_Autoencoders | supporting | `2. Reference/03_Autoencoder/Vincent_2010_Stacked_Denoising_Autoencoders.pdf` | — |
| 04_LightGBM | An Optimized LightGBM Model for Fraud Detection | supporting | `2. Reference/04_LightGBM/An Optimized LightGBM Model for Fraud Detection.pdf` | — |
| 04_LightGBM | Credit_Card_Fraud_Detection_Using_Lightgbm_Model | supporting | `2. Reference/04_LightGBM/Credit_Card_Fraud_Detection_Using_Lightgbm_Model.pdf` | — |
| 04_LightGBM | LightGBM A Highly Efficient Gradient Boosting | core | `2. Reference/04_LightGBM/LightGBM A Highly Efficient Gradient Boosting.pdf` | `_cards/LightGBM A Highly Efficient Gradient Boosting.md` |
| 05_Integrasi_Autoencoder_LightGBM | Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection | core | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.pdf` | `_cards/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.md` |
| 05_Integrasi_Autoencoder_LightGBM | Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection | core | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.pdf` | `_cards/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.md` |
| 05_Integrasi_Autoencoder_LightGBM | Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS | core | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.pdf` | `_cards/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.md` |
| 06_Bayesian_Optimization_Optuna | Akiba_2019_Optuna_Hyperparameter_Optimization | core | `2. Reference/06_Bayesian_Optimization_Optuna/Akiba_2019_Optuna_Hyperparameter_Optimization.pdf` | `_cards/Akiba_2019_Optuna_Hyperparameter_Optimization.md` |
| 06_Bayesian_Optimization_Optuna | Bergstra_2011_Algorithms_Hyperparameter_Optimization_TPE | supporting | `2. Reference/06_Bayesian_Optimization_Optuna/Bergstra_2011_Algorithms_Hyperparameter_Optimization_TPE.pdf` | — |
| 06_Bayesian_Optimization_Optuna | Bergstra_Bengio_2012_Random_Search_Hyperparameter_Optimization | supporting | `2. Reference/06_Bayesian_Optimization_Optuna/Bergstra_Bengio_2012_Random_Search_Hyperparameter_Optimization.pdf` | — |
| 06_Bayesian_Optimization_Optuna | Lim_2024_Bayesian_Optimization_Fraud_Detection | supporting | `2. Reference/06_Bayesian_Optimization_Optuna/Lim_2024_Bayesian_Optimization_Fraud_Detection.pdf` | — |
| 06_Bayesian_Optimization_Optuna | Shahriari_2016_Bayesian_Optimization_Review | supporting | `2. Reference/06_Bayesian_Optimization_Optuna/Shahriari_2016_Bayesian_Optimization_Review.pdf` | — |
| 07_Metrik_Evaluasi | Boyd_2013_Unachievable_Region_Precision_Recall_Space | supporting | `2. Reference/07_Metrik_Evaluasi/Boyd_2013_Unachievable_Region_Precision_Recall_Space.pdf` | `_cards/Boyd_2013_Unachievable_Region_Precision_Recall_Space.md` |
| 07_Metrik_Evaluasi | Davis_Goadrich_2006_Precision_Recall_and_ROC | supporting | `2. Reference/07_Metrik_Evaluasi/Davis_Goadrich_2006_Precision_Recall_and_ROC.pdf` | — |
| 07_Metrik_Evaluasi | Saito_Rehmsmeier_2015_Precision_Recall_Plot | core | `2. Reference/07_Metrik_Evaluasi/Saito_Rehmsmeier_2015_Precision_Recall_Plot.pdf` | `_cards/Saito_Rehmsmeier_2015_Precision_Recall_Plot.md` |
| 07_Metrik_Evaluasi | Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves | core | `2. Reference/07_Metrik_Evaluasi/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.pdf` | `_cards/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.md` |
| 08_Dataset_IEEE-CIS | Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS | core | `2. Reference/08_Dataset_IEEE-CIS/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.pdf` | `_cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md` |
| 08_Dataset_IEEE-CIS | Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS | core | `2. Reference/08_Dataset_IEEE-CIS/Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS.pdf` | `_cards/Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS.md` |
| 08_Dataset_IEEE-CIS | Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud | core | `2. Reference/08_Dataset_IEEE-CIS/Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud.pdf` | `_cards/Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud.md` |
| 08_Dataset_IEEE-CIS | Gopalakrishnan_2026_SilIF_Anomaly_IEEE-CIS | supporting | `2. Reference/08_Dataset_IEEE-CIS/Gopalakrishnan_2026_SilIF_Anomaly_IEEE-CIS.pdf` | `_cards/Gopalakrishnan_2026_SilIF_Anomaly_IEEE-CIS.md` |
| 08_Dataset_IEEE-CIS | Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS | core | `2. Reference/08_Dataset_IEEE-CIS/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.pdf` | `_cards/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md` |
| 08_Dataset_IEEE-CIS | Lucas_2019_Dataset_Shift_Credit_Card_Fraud | core | `2. Reference/08_Dataset_IEEE-CIS/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.pdf` | `_cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md` |
| 08_Dataset_IEEE-CIS | Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud | core | `2. Reference/08_Dataset_IEEE-CIS/Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud.pdf` | `_cards/Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud.md` |
| 08_Dataset_IEEE-CIS | Moradi_2025_Ensemble_AUC_PR_IEEE-CIS | core | `2. Reference/08_Dataset_IEEE-CIS/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.pdf` | `_cards/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md` |
| 08_Dataset_IEEE-CIS | Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN | core | `2. Reference/08_Dataset_IEEE-CIS/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.pdf` | `_cards/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.md` |
| 08_Dataset_IEEE-CIS | Prabha_2024_LSTM autoencoder | supporting | `2. Reference/08_Dataset_IEEE-CIS/Prabha_2024_LSTM autoencoder.pdf` | — |
| 08_Dataset_IEEE-CIS | Psychoula_2021_Explainable_ML_Fraud_Detection_IEEE-CIS | supporting | `2. Reference/08_Dataset_IEEE-CIS/Psychoula_2021_Explainable_ML_Fraud_Detection_IEEE-CIS.pdf` | — |
| 08_Dataset_IEEE-CIS | Zheng_2023_Unsupervised_Fraud_LowRank_IEEE-CIS | supporting | `2. Reference/08_Dataset_IEEE-CIS/Zheng_2023_Unsupervised_Fraud_LowRank_IEEE-CIS.pdf` | `_cards/Zheng_2023_Unsupervised_Fraud_LowRank_IEEE-CIS.md` |
| 99_Arsip | A Comparative Study of Model Updating Strategies for Concept Drift | archive | `2. Reference/99_Arsip/A Comparative Study of Model Updating Strategies for Concept Drift.pdf` | — |
| 99_Arsip | A Survey On Fraud Detection Techniques in E-Commerce | archive | `2. Reference/99_Arsip/A Survey On Fraud Detection Techniques in E-Commerce.pdf` | — |
| 99_Arsip | A survey on learning from imbalanced data streams | archive | `2. Reference/99_Arsip/A survey on learning from imbalanced data streams.pdf` | — |
| 99_Arsip | A_Hybrid_Deep_Learning_Model_For_Online_Fraud_Detection | archive | `2. Reference/99_Arsip/A_Hybrid_Deep_Learning_Model_For_Online_Fraud_Detection.pdf` | — |
| 99_Arsip | ADWIN-U adaptive windowing for unsupervised drift detection on data streams | archive | `2. Reference/99_Arsip/ADWIN-U adaptive windowing for unsupervised drift detection on data streams.pdf` | — |
| 99_Arsip | An ensemble learning approach for anomaly detection in credit card data | archive | `2. Reference/99_Arsip/An ensemble learning approach for anomaly detection in credit card data.pdf` | — |
| 99_Arsip | Comparative Review of Credit Card Fraud Detection using Machine Learning and Concept Drift Techniques | archive | `2. Reference/99_Arsip/Comparative Review of Credit Card Fraud Detection using Machine Learning and Concept Drift Techniques.pdf` | — |
| 99_Arsip | Credit_card_fraud_detection_and_concept-drift_adaptation_with_delayed_supervised_information | archive | `2. Reference/99_Arsip/Credit_card_fraud_detection_and_concept-drift_adaptation_with_delayed_supervised_information.pdf` | — |
| 99_Arsip | fraud detection-ensemble learning_hybrid data sample | archive | `2. Reference/99_Arsip/fraud detection-ensemble learning_hybrid data sample.pdf` | — |
| 99_Arsip | Fraud_Detection_in_Online_Credit_Card_Transactions_Using_Deep_Learning | archive | `2. Reference/99_Arsip/Fraud_Detection_in_Online_Credit_Card_Transactions_Using_Deep_Learning.pdf` | — |
| 99_Arsip | Large-Scale Learning from Data Streams with | archive | `2. Reference/99_Arsip/Large-Scale Learning from Data Streams with.pdf` | — |
| 99_Arsip | Learning_under_Concept_Drift_A_Review | archive | `2. Reference/99_Arsip/Learning_under_Concept_Drift_A_Review.pdf` | — |
| 99_Arsip | Online Ensemble Using Adaptive Windowing for Data Streams with Concept Drift | archive | `2. Reference/99_Arsip/Online Ensemble Using Adaptive Windowing for Data Streams with Concept Drift.pdf` | — |
| 99_Arsip | PSIfinal | archive | `2. Reference/99_Arsip/PSIfinal.pdf` | — |
| 99_Arsip | Real-Time Transaction Fraud Detection Using Adaptive Hoeffding Trees for Concept-Drift Resilience | archive | `2. Reference/99_Arsip/Real-Time Transaction Fraud Detection Using Adaptive Hoeffding Trees for Concept-Drift Resilience.pdf` | — |
| 99_Arsip | systematic review-fraud detection | archive | `2. Reference/99_Arsip/systematic review-fraud detection.pdf` | — |

## Deep Research (sintesis)

| Laporan | Path |
|---------|------|
| Autoencoder–LightGBM Fraud Detection Thesis | `4. Deep Research/Autoencoder–LightGBM Fraud Detection Thesis.md` |
| Autoencoder–LightGBM pada IEEE-CIS Fraud Detection | `4. Deep Research/Autoencoder–LightGBM pada IEEE-CIS Fraud Detection.md` |
| IEEE-CIS Fraud Detection Papers | `4. Deep Research/IEEE-CIS Fraud Detection Papers.md` |
| IEEE-CIS Fraud Detection Studies That Explicitly Report PR-AUC | `4. Deep Research/IEEE-CIS Fraud Detection Studies That Explicitly Report PR-AUC.md` |
| Paper Fraud Detection Berbasis Autoencoder pada Dataset IEEE-CIS Fraud Detection | `4. Deep Research/Paper Fraud Detection Berbasis Autoencoder pada Dataset IEEE-CIS Fraud Detection.md` |

## Navigasi proyek

```text
1_TugasAkhir/
├── ecommerce_fraud_detection/   # code + docs/literature
├── 2. Reference/                # PDF resmi
├── 5. Literature Cards/     # kartu + indeks (tanpa OCR)
└── 4. Deep Research/            # analisis sintesis
```

## Source of truth

1. **Kartu `_cards/`** — ringkasan untuk agent & penulisan.
2. **PDF `2. Reference/`** — verifikasi angka dan sitasi resmi.
3. **Deep Research** — sintesis Bab 2/5.
