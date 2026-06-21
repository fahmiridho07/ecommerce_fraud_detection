# RankGauss + Swap-Noise AE Implementation

Status: implemented diagnostic harness, pending empirical run.

This document records the code implementation of the external deep-research
recommendations for the original Autoencoder + LightGBM corridor. It does not
replace the current thesis-facing result in `THESIS_SCOPE.md`; it adds a
proposal-consistent diagnostic branch that can be run if the scope is reopened.

## Implemented

- `src/rankgauss_ae_utils.py`
  - V-column reduction by missingness group and within-group correlation.
  - Train-only observed-value `QuantileTransformer(output_distribution="normal")`.
  - Missing values become a zero placeholder in transformed space plus an
    explicit observed-value mask.
  - Reconstruction-error features are computed on observed cells only.

- `src/run_rankgauss_swapnoise_ae_ladder.py`
  - Swap-noise denoising Autoencoder.
  - Observed-only masked MSE.
  - Two proposal-consistent fusion candidates:
    - append RankGauss swap-noise AE latent and reconstruction-error features
      while keeping original features;
    - replace only observed selected V values with inverse-transformed
      reconstructions and append missingness masks.
  - LightGBM selection remains based on validation Average Precision / PR-AUC.
  - Bootstrap comparison uses the original proposal tuned LightGBM reference.

## Run Command

Kaggle standalone script:

```bash
kaggle/ieee_rankgauss_swapnoise_ladder_KAGGLE.py
```

In Kaggle, add the IEEE-CIS Fraud Detection input, upload or paste that script,
then run it. It writes:

- `/kaggle/working/rankgauss_swapnoise_ladder_results.json`
- `/kaggle/working/rankgauss_swapnoise_ladder_summary.csv`
- `/kaggle/working/rankgauss_v_selection_report.csv`
- `/kaggle/working/rankgauss_swapnoise_test_scores.csv`

Local repo runner:

```bash
python src/run_rankgauss_swapnoise_ae_ladder.py \
  --output-dir outputs/stratified_reset/rankgauss_swapnoise_ae_ladder
```

Optional faster smoke command:

```bash
python src/run_rankgauss_swapnoise_ae_ladder.py \
  --output-dir outputs/stratified_reset/rankgauss_swapnoise_ae_ladder_smoke \
  --max-v-columns 32 \
  --ae-max-epochs 3 \
  --n-bootstrap 20
```

## Interpretation Rule

Promote this branch only if the selected candidate has higher test AP than the
matched tuned LightGBM reference and the paired-bootstrap AP delta supports a
positive improvement. If it does not, keep the existing thesis conclusion:
feature-level AE variants remain weaker than tuned LightGBM, while AE's
defensible contribution is latent-space oversampling on dense A1 features.
