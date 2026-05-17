<div align="center">

# E-commerce Fraud Detection

**Undergraduate thesis project on detecting fraudulent e-commerce transactions using the IEEE-CIS Fraud Detection dataset.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Baseline-9ACD32?style=flat-square)
![Autoencoder](https://img.shields.io/badge/Robust%20AE-V--features-FFB000?style=flat-square)
![Metric](https://img.shields.io/badge/Primary%20Metric-PR--AUC-FF6B6B?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-7C3AED?style=flat-square)

</div>

## Overview

This repository contains an undergraduate thesis implementation for e-commerce fraud detection. The project compares a LightGBM baseline with an Autoencoder-enhanced LightGBM pipeline, using a temporal evaluation setup based on `TransactionDT` so the experiment better reflects real-world fraud detection over time.

## Highlights

- **Temporal split:** labeled train data is sorted by `TransactionDT` and split chronologically into train, validation, and test sets.
- **Main models:** LightGBM baseline, Robust Autoencoder for `V`-features, and AE-LightGBM.
- **Tuning:** Optuna is used for hyperparameter search, with final tuning still in progress.
- **Primary metric:** PR-AUC / Average Precision, chosen because fraud detection is highly imbalanced.
- **Leakage-aware evaluation:** preprocessing, representation learning, threshold selection, and tuning are fitted only on the allowed split for each phase.

## Project Structure

```text
.
|-- data/
|   `-- raw/                    # Local Kaggle dataset files
|-- notebooks/
|   |-- eda.ipynb
|   |-- kaggle_runner.ipynb
|   `-- ta-fraud-detection.ipynb
|-- outputs/                    # Generated experiment artifacts
|-- src/
|   |-- check_data_split.py
|   |-- train_baseline_lgbm.py
|   |-- train_autoencoder_robust.py
|   |-- train_ae_lgbm.py
|   |-- tune_lgbm_optuna.py
|   `-- compare_results.py
|-- requirements.txt
`-- README.md
```

## Run Locally

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Download the IEEE-CIS Fraud Detection files from Kaggle and place them in:

```text
data/raw/
|-- train_transaction.csv
|-- train_identity.csv
|-- test_transaction.csv
`-- test_identity.csv
```

3. Run the main pipeline scripts.

```bash
python src/check_data_split.py
python src/train_baseline_lgbm.py
python src/train_autoencoder_robust.py
python src/train_ae_lgbm.py
```

4. Optional Optuna tuning.

```bash
python src/tune_lgbm_optuna.py --model_type baseline_lgbm --tuning_profile quick
python src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128 --tuning_profile quick
```

## Kaggle Execution

For Kaggle runs, attach the **IEEE-CIS Fraud Detection** dataset to the notebook. The project automatically uses `/kaggle/input/ieee-fraud-detection` when that path exists; otherwise it falls back to local `data/raw/`.

```bash
!python src/train_baseline_lgbm.py
!python src/train_autoencoder_robust.py
!python src/train_ae_lgbm.py
```

Kaggle competition test files are **not used for metric evaluation** because they do not contain `isFraud` labels. The main reported evaluation uses the temporal train/validation/test split from the labeled Kaggle train files.

## Current Status

- LightGBM baseline pipeline: ready
- Robust Autoencoder representation learning: ready
- AE-LightGBM pipeline: ready
- Optuna final tuning: running
- Final comparison summary: pending after tuning finishes

## Author

Created and maintained by the repository owner as part of an undergraduate thesis project.
