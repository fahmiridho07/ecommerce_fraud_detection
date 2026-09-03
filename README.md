# E-commerce Fraud Detection

Final artifacts for an undergraduate thesis on fraud detection with the
IEEE-CIS Fraud Detection dataset.

This repository is intentionally narrow. It contains only the final Kaggle
implementation, the final methodology notebook, and dataset provenance. Local
experiments, development utilities, documentation drafts, outputs, models, and
historical work are deliberately excluded.

## Contents

| Path | Purpose |
| --- | --- |
| [`data/`](data/) | Dataset source and expected input files. The dataset itself is not versioned here. |
| [`kaggle/ieee_final_oversampling_kaggle.py`](kaggle/ieee_final_oversampling_kaggle.py) | Locked final method: A1 dense preprocessing, LightGBM baseline, SMOTE control, and AE latent-space oversampling. |
| [`notebooks/final_methodology_ae_latent_oversampling.ipynb`](notebooks/final_methodology_ae_latent_oversampling.ipynb) | Final, narrative implementation and evaluation notebook. |

## Run on Kaggle

1. Create a Kaggle notebook and add the **IEEE-CIS Fraud Detection** competition
   data as input.
2. Copy `kaggle/ieee_final_oversampling_kaggle.py` into the notebook and run it.
3. The script writes `final_oversampling_results.json` to `/kaggle/working/`.

The final protocol uses a stratified 60/20/20 split with `random_state=42`.
The primary metric is average precision (PR-AUC).

## Data availability

The original IEEE-CIS files are not stored in this repository: two required
CSV files exceed GitHub's 100 MB per-file limit and the source is governed by
Kaggle competition access. Download the data from the official competition
page and place the required files as described in [`data/README.md`](data/README.md).

## License

Distributed under the [MIT License](LICENSE).
