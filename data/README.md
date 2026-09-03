# Data

This project uses the [IEEE-CIS Fraud Detection competition dataset](https://www.kaggle.com/c/ieee-fraud-detection/data).

The dataset is deliberately not committed to this GitHub repository. In
addition to the competition's access terms, `train_transaction.csv` and
`test_transaction.csv` exceed GitHub's 100 MB file-size limit.

For a local run, download the competition data from Kaggle and place these
files in `data/raw/`:

```text
train_transaction.csv
train_identity.csv
test_transaction.csv
test_identity.csv
sample_submission.csv
```

For Kaggle, add **IEEE-CIS Fraud Detection** as the notebook input. The final
Kaggle script expects it at `/kaggle/input/ieee-fraud-detection/`.
