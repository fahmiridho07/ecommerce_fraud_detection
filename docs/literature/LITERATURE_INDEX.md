# Literature Index — Tugas Akhir IEEE-CIS

Indeks agent-friendly untuk 55 referensi. **Kartu ringkas** ada di `_cards/`; full-text OCR di file `.md` sejajar.

## Thesis experiment baseline (bandingkan ke literatur)

| Model | Test AP | Catatan |
|-------|---------|---------|
| P02 BASE-02 (tuned LGBM) | **0.5049** | Protokol chronological — `EXPERIMENT_REGISTRY.md` |
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

| Folder | File | Priority | MD | PDF |
|--------|------|----------|----|-----|
| 01_Deteksi_Penipuan_E-Commerce | Ali_2022_Financial_Fraud_Detection_SLR | core | `5. Reference (MarkDown)/01_Deteksi_Penipuan_E-Commerce/Ali_2022_Financial_Fraud_Detection_SLR.md` | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Ali_2022_Financial_Fraud_Detection_SLR.pdf` |
| 01_Deteksi_Penipuan_E-Commerce | Ashtiani_Raahemi_2022_Intelligent_Fraud_Detection_SLR | supporting | `5. Reference (MarkDown)/01_Deteksi_Penipuan_E-Commerce/Ashtiani_Raahemi_2022_Intelligent_Fraud_Detection_SLR.md` | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Ashtiani_Raahemi_2022_Intelligent_Fraud_Detection_SLR.pdf` |
| 01_Deteksi_Penipuan_E-Commerce | Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy | core | `5. Reference (MarkDown)/01_Deteksi_Penipuan_E-Commerce/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.md` | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.pdf` |
| 01_Deteksi_Penipuan_E-Commerce | Financial Fraud A Review of Anomaly Detection Techniques and Recent Advance | supporting | `5. Reference (MarkDown)/01_Deteksi_Penipuan_E-Commerce/Financial Fraud A Review of Anomaly Detection Techniques and Recent Advance.md` | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Financial Fraud A Review of Anomaly Detection Techniques and Recent Advance.pdf` |
| 01_Deteksi_Penipuan_E-Commerce | Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments | core | `5. Reference (MarkDown)/01_Deteksi_Penipuan_E-Commerce/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.md` | `2. Reference/01_Deteksi_Penipuan_E-Commerce/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.pdf` |
| 02_Ketidakseimbangan_Kelas | Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection | core | `5. Reference (MarkDown)/02_Ketidakseimbangan_Kelas/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md` | `2. Reference/02_Ketidakseimbangan_Kelas/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.pdf` |
| 02_Ketidakseimbangan_Kelas | Survey on deep learning with class imbalance | supporting | `5. Reference (MarkDown)/02_Ketidakseimbangan_Kelas/Survey on deep learning with class imbalance.md` | `2. Reference/02_Ketidakseimbangan_Kelas/Survey on deep learning with class imbalance.pdf` |
| 02_Ketidakseimbangan_Kelas | Zhao_2023_Improved_LightGBM_Imbalanced_Data | supporting | `5. Reference (MarkDown)/02_Ketidakseimbangan_Kelas/Zhao_2023_Improved_LightGBM_Imbalanced_Data.md` | `2. Reference/02_Ketidakseimbangan_Kelas/Zhao_2023_Improved_LightGBM_Imbalanced_Data.pdf` |
| 03_Autoencoder | Hinton_Salakhutdinov_2006_Reducing_Dimensionality_Neural_Networks | supporting | `5. Reference (MarkDown)/03_Autoencoder/Hinton_Salakhutdinov_2006_Reducing_Dimensionality_Neural_Networks.md` | `2. Reference/03_Autoencoder/Hinton_Salakhutdinov_2006_Reducing_Dimensionality_Neural_Networks.pdf` |
| 03_Autoencoder | Misra_2020_Autoencoder_Fraudulent_Credit_Card_Transaction | supporting | `5. Reference (MarkDown)/03_Autoencoder/Misra_2020_Autoencoder_Fraudulent_Credit_Card_Transaction.md` | `2. Reference/03_Autoencoder/Misra_2020_Autoencoder_Fraudulent_Credit_Card_Transaction.pdf` |
| 03_Autoencoder | Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection | core | `5. Reference (MarkDown)/03_Autoencoder/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.md` | `2. Reference/03_Autoencoder/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.pdf` |
| 03_Autoencoder | Vincent_2010_Stacked_Denoising_Autoencoders | supporting | `5. Reference (MarkDown)/03_Autoencoder/Vincent_2010_Stacked_Denoising_Autoencoders.md` | `2. Reference/03_Autoencoder/Vincent_2010_Stacked_Denoising_Autoencoders.pdf` |
| 04_LightGBM | An Optimized LightGBM Model for Fraud Detection | supporting | `5. Reference (MarkDown)/04_LightGBM/An Optimized LightGBM Model for Fraud Detection.md` | `2. Reference/04_LightGBM/An Optimized LightGBM Model for Fraud Detection.pdf` |
| 04_LightGBM | Credit_Card_Fraud_Detection_Using_Lightgbm_Model | supporting | `5. Reference (MarkDown)/04_LightGBM/Credit_Card_Fraud_Detection_Using_Lightgbm_Model.md` | `2. Reference/04_LightGBM/Credit_Card_Fraud_Detection_Using_Lightgbm_Model.pdf` |
| 04_LightGBM | LightGBM A Highly Efficient Gradient Boosting | core | `5. Reference (MarkDown)/04_LightGBM/LightGBM A Highly Efficient Gradient Boosting.md` | `2. Reference/04_LightGBM/LightGBM A Highly Efficient Gradient Boosting.pdf` |
| 05_Integrasi_Autoencoder_LightGBM | Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection | core | `5. Reference (MarkDown)/05_Integrasi_Autoencoder_LightGBM/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.md` | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.pdf` |
| 05_Integrasi_Autoencoder_LightGBM | Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection | core | `5. Reference (MarkDown)/05_Integrasi_Autoencoder_LightGBM/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.md` | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.pdf` |
| 05_Integrasi_Autoencoder_LightGBM | Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS | core | `5. Reference (MarkDown)/05_Integrasi_Autoencoder_LightGBM/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.md` | `2. Reference/05_Integrasi_Autoencoder_LightGBM/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.pdf` |
| 06_Bayesian_Optimization_Optuna | Akiba_2019_Optuna_Hyperparameter_Optimization | core | `5. Reference (MarkDown)/06_Bayesian_Optimization_Optuna/Akiba_2019_Optuna_Hyperparameter_Optimization.md` | `2. Reference/06_Bayesian_Optimization_Optuna/Akiba_2019_Optuna_Hyperparameter_Optimization.pdf` |
| 06_Bayesian_Optimization_Optuna | Bergstra_2011_Algorithms_Hyperparameter_Optimization_TPE | supporting | `5. Reference (MarkDown)/06_Bayesian_Optimization_Optuna/Bergstra_2011_Algorithms_Hyperparameter_Optimization_TPE.md` | `2. Reference/06_Bayesian_Optimization_Optuna/Bergstra_2011_Algorithms_Hyperparameter_Optimization_TPE.pdf` |
| 06_Bayesian_Optimization_Optuna | Bergstra_Bengio_2012_Random_Search_Hyperparameter_Optimization | supporting | `5. Reference (MarkDown)/06_Bayesian_Optimization_Optuna/Bergstra_Bengio_2012_Random_Search_Hyperparameter_Optimization.md` | `2. Reference/06_Bayesian_Optimization_Optuna/Bergstra_Bengio_2012_Random_Search_Hyperparameter_Optimization.pdf` |
| 06_Bayesian_Optimization_Optuna | Lim_2024_Bayesian_Optimization_Fraud_Detection | supporting | `5. Reference (MarkDown)/06_Bayesian_Optimization_Optuna/Lim_2024_Bayesian_Optimization_Fraud_Detection.md` | `2. Reference/06_Bayesian_Optimization_Optuna/Lim_2024_Bayesian_Optimization_Fraud_Detection.pdf` |
| 06_Bayesian_Optimization_Optuna | Shahriari_2016_Bayesian_Optimization_Review | supporting | `5. Reference (MarkDown)/06_Bayesian_Optimization_Optuna/Shahriari_2016_Bayesian_Optimization_Review.md` | `2. Reference/06_Bayesian_Optimization_Optuna/Shahriari_2016_Bayesian_Optimization_Review.pdf` |
| 07_Metrik_Evaluasi | Boyd_2013_Unachievable_Region_Precision_Recall_Space | supporting | `5. Reference (MarkDown)/07_Metrik_Evaluasi/Boyd_2013_Unachievable_Region_Precision_Recall_Space.md` | `2. Reference/07_Metrik_Evaluasi/Boyd_2013_Unachievable_Region_Precision_Recall_Space.pdf` |
| 07_Metrik_Evaluasi | Davis_Goadrich_2006_Precision_Recall_and_ROC | supporting | `5. Reference (MarkDown)/07_Metrik_Evaluasi/Davis_Goadrich_2006_Precision_Recall_and_ROC.md` | `2. Reference/07_Metrik_Evaluasi/Davis_Goadrich_2006_Precision_Recall_and_ROC.pdf` |
| 07_Metrik_Evaluasi | Saito_Rehmsmeier_2015_Precision_Recall_Plot | core | `5. Reference (MarkDown)/07_Metrik_Evaluasi/Saito_Rehmsmeier_2015_Precision_Recall_Plot.md` | `2. Reference/07_Metrik_Evaluasi/Saito_Rehmsmeier_2015_Precision_Recall_Plot.pdf` |
| 07_Metrik_Evaluasi | Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves | core | `5. Reference (MarkDown)/07_Metrik_Evaluasi/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.md` | `2. Reference/07_Metrik_Evaluasi/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.pdf` |
| 08_Dataset_IEEE-CIS | Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.pdf` |
| 08_Dataset_IEEE-CIS | Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Bakhtiari_2023_LightGBM_Ensemble_IEEE-CIS.pdf` |
| 08_Dataset_IEEE-CIS | Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud.md` | `2. Reference/08_Dataset_IEEE-CIS/Carcillo_2018_SCARFF_Streaming_Credit_Card_Fraud.pdf` |
| 08_Dataset_IEEE-CIS | Gopalakrishnan_2026_SilIF_Anomaly_IEEE-CIS | supporting | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Gopalakrishnan_2026_SilIF_Anomaly_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Gopalakrishnan_2026_SilIF_Anomaly_IEEE-CIS.pdf` |
| 08_Dataset_IEEE-CIS | Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.pdf` |
| 08_Dataset_IEEE-CIS | Lucas_2019_Dataset_Shift_Credit_Card_Fraud | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md` | `2. Reference/08_Dataset_IEEE-CIS/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.pdf` |
| 08_Dataset_IEEE-CIS | Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud.md` | `2. Reference/08_Dataset_IEEE-CIS/Lucas_2019_HMM_Feature_Engineering_Credit_Card_Fraud.pdf` |
| 08_Dataset_IEEE-CIS | Moradi_2025_Ensemble_AUC_PR_IEEE-CIS | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.pdf` |
| 08_Dataset_IEEE-CIS | Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN | core | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.md` | `2. Reference/08_Dataset_IEEE-CIS/Nguyen_2022_Card_Fraud_Detection_CatBoost_DNN.pdf` |
| 08_Dataset_IEEE-CIS | Prabha_2024_LSTM autoencoder | supporting | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Prabha_2024_LSTM autoencoder.md` | `2. Reference/08_Dataset_IEEE-CIS/Prabha_2024_LSTM autoencoder.pdf` |
| 08_Dataset_IEEE-CIS | Psychoula_2021_Explainable_ML_Fraud_Detection_IEEE-CIS | supporting | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Psychoula_2021_Explainable_ML_Fraud_Detection_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Psychoula_2021_Explainable_ML_Fraud_Detection_IEEE-CIS.pdf` |
| 08_Dataset_IEEE-CIS | Zheng_2023_Unsupervised_Fraud_LowRank_IEEE-CIS | supporting | `5. Reference (MarkDown)/08_Dataset_IEEE-CIS/Zheng_2023_Unsupervised_Fraud_LowRank_IEEE-CIS.md` | `2. Reference/08_Dataset_IEEE-CIS/Zheng_2023_Unsupervised_Fraud_LowRank_IEEE-CIS.pdf` |
| 99_Arsip | A Comparative Study of Model Updating Strategies for Concept Drift | archive | `5. Reference (MarkDown)/99_Arsip/A Comparative Study of Model Updating Strategies for Concept Drift.md` | `2. Reference/99_Arsip/A Comparative Study of Model Updating Strategies for Concept Drift.pdf` |
| 99_Arsip | A Survey On Fraud Detection Techniques in E-Commerce | archive | `5. Reference (MarkDown)/99_Arsip/A Survey On Fraud Detection Techniques in E-Commerce.md` | `2. Reference/99_Arsip/A Survey On Fraud Detection Techniques in E-Commerce.pdf` |
| 99_Arsip | A survey on learning from imbalanced data streams | archive | `5. Reference (MarkDown)/99_Arsip/A survey on learning from imbalanced data streams.md` | `2. Reference/99_Arsip/A survey on learning from imbalanced data streams.pdf` |
| 99_Arsip | A_Hybrid_Deep_Learning_Model_For_Online_Fraud_Detection | archive | `5. Reference (MarkDown)/99_Arsip/A_Hybrid_Deep_Learning_Model_For_Online_Fraud_Detection.md` | `2. Reference/99_Arsip/A_Hybrid_Deep_Learning_Model_For_Online_Fraud_Detection.pdf` |
| 99_Arsip | ADWIN-U adaptive windowing for unsupervised drift detection on data streams | archive | `5. Reference (MarkDown)/99_Arsip/ADWIN-U adaptive windowing for unsupervised drift detection on data streams.md` | `2. Reference/99_Arsip/ADWIN-U adaptive windowing for unsupervised drift detection on data streams.pdf` |
| 99_Arsip | An ensemble learning approach for anomaly detection in credit card data | archive | `5. Reference (MarkDown)/99_Arsip/An ensemble learning approach for anomaly detection in credit card data.md` | `2. Reference/99_Arsip/An ensemble learning approach for anomaly detection in credit card data.pdf` |
| 99_Arsip | Comparative Review of Credit Card Fraud Detection using Machine Learning and Concept Drift Techniques | archive | `5. Reference (MarkDown)/99_Arsip/Comparative Review of Credit Card Fraud Detection using Machine Learning and Concept Drift Techniques.md` | `2. Reference/99_Arsip/Comparative Review of Credit Card Fraud Detection using Machine Learning and Concept Drift Techniques.pdf` |
| 99_Arsip | Credit_card_fraud_detection_and_concept-drift_adaptation_with_delayed_supervised_information | archive | `5. Reference (MarkDown)/99_Arsip/Credit_card_fraud_detection_and_concept-drift_adaptation_with_delayed_supervised_information.md` | `2. Reference/99_Arsip/Credit_card_fraud_detection_and_concept-drift_adaptation_with_delayed_supervised_information.pdf` |
| 99_Arsip | fraud detection-ensemble learning_hybrid data sample | archive | `5. Reference (MarkDown)/99_Arsip/fraud detection-ensemble learning_hybrid data sample.md` | `2. Reference/99_Arsip/fraud detection-ensemble learning_hybrid data sample.pdf` |
| 99_Arsip | Fraud_Detection_in_Online_Credit_Card_Transactions_Using_Deep_Learning | archive | `5. Reference (MarkDown)/99_Arsip/Fraud_Detection_in_Online_Credit_Card_Transactions_Using_Deep_Learning.md` | `2. Reference/99_Arsip/Fraud_Detection_in_Online_Credit_Card_Transactions_Using_Deep_Learning.pdf` |
| 99_Arsip | Large-Scale Learning from Data Streams with | archive | `5. Reference (MarkDown)/99_Arsip/Large-Scale Learning from Data Streams with.md` | `2. Reference/99_Arsip/Large-Scale Learning from Data Streams with.pdf` |
| 99_Arsip | Learning_under_Concept_Drift_A_Review | archive | `5. Reference (MarkDown)/99_Arsip/Learning_under_Concept_Drift_A_Review.md` | `2. Reference/99_Arsip/Learning_under_Concept_Drift_A_Review.pdf` |
| 99_Arsip | Online Ensemble Using Adaptive Windowing for Data Streams with Concept Drift | archive | `5. Reference (MarkDown)/99_Arsip/Online Ensemble Using Adaptive Windowing for Data Streams with Concept Drift.md` | `2. Reference/99_Arsip/Online Ensemble Using Adaptive Windowing for Data Streams with Concept Drift.pdf` |
| 99_Arsip | PSIfinal | archive | `5. Reference (MarkDown)/99_Arsip/PSIfinal.md` | `2. Reference/99_Arsip/PSIfinal.pdf` |
| 99_Arsip | Real-Time Transaction Fraud Detection Using Adaptive Hoeffding Trees for Concept-Drift Resilience | archive | `5. Reference (MarkDown)/99_Arsip/Real-Time Transaction Fraud Detection Using Adaptive Hoeffding Trees for Concept-Drift Resilience.md` | `2. Reference/99_Arsip/Real-Time Transaction Fraud Detection Using Adaptive Hoeffding Trees for Concept-Drift Resilience.pdf` |
| 99_Arsip | systematic review-fraud detection | archive | `5. Reference (MarkDown)/99_Arsip/systematic review-fraud detection.md` | `2. Reference/99_Arsip/systematic review-fraud detection.pdf` |

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
├── 5. Reference (MarkDown)/     # full-text OCR + index ini
└── 4. Deep Research/            # analisis sintesis
```

## Catatan OCR

File `.md` di folder ini adalah ekstrak teks PDF (sering tanpa spasi). Untuk penulisan/skripsi:
1. Baca **kartu `_cards/`** atau **Deep Research** dulu.
2. Verifikasi angka ke **PDF**.
3. Jangan andalkan OCR untuk kutipan final.
