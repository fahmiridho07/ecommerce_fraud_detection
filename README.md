# E-commerce Fraud Detection Thesis

This repository contains the implementation for an undergraduate thesis:

**E-commerce transaction fraud detection using Autoencoder and LightGBM with Bayesian Optimization.**

The project uses the IEEE-CIS Fraud Detection dataset from Kaggle. The final experiment design uses a chronological split because `TransactionDT` is a relative timedelta and the dataset has temporal structure.

## Dataset

For local execution, download the Kaggle IEEE-CIS Fraud Detection files and place them in:

```text
data/raw/
  train_transaction.csv
  train_identity.csv
  test_transaction.csv
  test_identity.csv
```

Only `train_transaction.csv` and `train_identity.csv` are used for thesis metric evaluation because they contain `isFraud`. Kaggle competition test files do not contain labels, so they are not used for PR-AUC, ROC-AUC, Precision, Recall, F1, or MCC evaluation.

On Kaggle, attach the IEEE-CIS Fraud Detection dataset to the notebook. The code automatically uses `/kaggle/input/ieee-fraud-detection` when that directory exists; otherwise it uses local `data/raw`.

Kaggle notebooks can run project scripts with commands such as:

```bash
!python src/train_baseline_lgbm.py
```

## Project Structure

```text
src/
  config.py
  utils.py
  data_loader.py
  splitting.py
  preprocessing.py
  evaluation.py
  train_baseline_lgbm.py
  train_autoencoder.py
  train_ae_lgbm.py
  tune_lgbm_optuna.py
  compare_results.py

outputs/
  baseline_lgbm/
  autoencoder/
  ae_lgbm/
  optuna/
  final_comparison/

data/
  raw/

notebooks/
  kaggle_runner.ipynb
  eda.ipynb
```

## Planned Phases

Phase 0 creates the project foundation and configuration. No model training is performed.

Planned commands for later phases:

```bash
python src/check_data_split.py
python src/train_baseline_lgbm.py
python src/train_autoencoder.py
python src/train_ae_lgbm.py
python src/tune_lgbm_optuna.py
python src/compare_results.py
```

## Experiment Notes

- Sort by `TransactionDT` before splitting.
- Split chronologically: first 60% train, next 20% validation, final 20% test.
- Use PR-AUC / Average Precision as the primary metric.
- Use ROC-AUC, Precision, Recall, F1, and MCC as supporting metrics.
- Avoid data leakage: fit preprocessing, scalers, encoders, Autoencoder, thresholds, and hyperparameters only on allowed training or validation data according to the phase design.
- Treat V-features (`V1` to `V339`) as numerical Vesta-engineered features, not categorical features.
