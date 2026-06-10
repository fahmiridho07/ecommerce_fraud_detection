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
- **Tuning:** Optuna TPE tuning is implemented and completed for the primary baseline and AE-LightGBM LD128 candidates (`outputs/optuna/`).
- **Primary metric:** PR-AUC / Average Precision, chosen because fraud detection is highly imbalanced.
- **Leakage-aware evaluation:** preprocessing, representation learning, threshold selection, and tuning are fitted only on the allowed split for each phase.

## Project Structure

```text
.
|-- data/
|   `-- raw/                    # Local Kaggle dataset files
|-- notebooks/
|   `-- thesis_experiment_report.ipynb
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

## Notebook Report

The main supervisor-facing notebook is:

```text
notebooks/thesis_experiment_report.ipynb
```

It is a clean reporting notebook for thesis guidance. It loads existing artifacts from `outputs/`, summarizes the methodology and final results, and includes lightweight EDA and diagnostic visualizations. It does **not** retrain models or rerun Optuna/Autoencoder experiments.

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

- LightGBM baseline pipeline: executed (`outputs/baseline_lgbm/`)
- Robust Autoencoder representation learning: executed (`outputs/autoencoder_robust/`)
- AE-LightGBM replacement pipeline: executed (`outputs/ae_lgbm/`)
- Optuna tuning (baseline + AE LD128): executed (`outputs/optuna/`)
- Comparison summaries: available (`outputs/final_comparison/`)
- Experiment governance docs: [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md), [`docs/FINAL_EXPERIMENT_PLAN.md`](docs/FINAL_EXPERIMENT_PLAN.md), [`docs/EXPERIMENT_SCOPE_FREEZE.md`](docs/EXPERIMENT_SCOPE_FREEZE.md)

### Primary chronological finding (validation-selected)

On the frozen primary comparison, **tuned original-feature LightGBM (P02)** outperforms the **tuned AE replacement candidate (P04)** on both validation AP (0.624072 vs 0.610631) and test AP (0.501438 vs 0.490686). AE latent replacement did not improve over the original-feature baseline in the executed pipeline. The highest historical test AP in exploratory branches does **not** automatically define the final thesis model.

## Experiment Governance

Experiments are divided into four governance categories: **primary thesis experiments**, **Autoencoder diagnostics**, **methodological diagnostics**, and **exploratory archive**. Chronological evaluation on `TransactionDT` is the primary protocol; stratified holdout and stratified CV are appendix sensitivity analyses only and must not be used to choose the final thesis model.

Historical experiments remain available locally under `outputs/` for reproducibility audit. The authoritative experiment map is [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md). The frozen final scope and reporting policy are defined in [`docs/FINAL_EXPERIMENT_PLAN.md`](docs/FINAL_EXPERIMENT_PLAN.md) and [`docs/EXPERIMENT_SCOPE_FREEZE.md`](docs/EXPERIMENT_SCOPE_FREEZE.md). Small result summaries recommended for later Git tracking are listed in [`docs/RESULT_ARTIFACT_MANIFEST.md`](docs/RESULT_ARTIFACT_MANIFEST.md).

## Author

Created and maintained by the repository owner as part of an undergraduate thesis project.
