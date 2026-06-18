# Ding-Style AEELG on IEEE-CIS

Status: closed and archived as negative-transfer evidence under the active
stratified protocol.

Purpose: test Ding et al. (2024) as a direct methodological anchor after
adapting the AEELG recipe to the IEEE-CIS Fraud Detection dataset.

## Anchor

Ding et al. (2024) propose an AutoEncoder enhanced LightGBM method for credit
card fraud detection. The paper recipe can be summarized as:

```text
standardize data
-> split train/test
-> apply SMOTE to training data
-> train AutoEncoder on the balanced train data
-> reconstruct train/test features with the AutoEncoder
-> train LightGBM on reconstructed features
```

Important detail: Ding-style AEELG uses **reconstructed features**, not the
bottleneck latent vector as the final LightGBM feature set. This differs from the
original proposal's latent-replacement framing.

## IEEE-CIS Adaptation

Ding's two datasets are already numeric. IEEE-CIS contains mixed categorical and
numeric features with extensive missingness, so the experiment first creates a
dense numeric representation using the active A1 preprocessing:

- categorical missing values become a dedicated category;
- categorical values are train-frequency encoded;
- numeric missing values are imputed with train medians;
- numeric values are z-score scaled using train means and standard deviations;
- every fitted statistic is learned from the train split only.

This produces an all-numeric matrix suitable for AutoEncoder reconstruction.

## Leakage Guardrails

The experiment intentionally differs from many competition-style recipes by
keeping thesis guardrails:

- split before fitting preprocessing;
- fit preprocessing, SMOTE neighbour search, and AutoEncoder on train only;
- never oversample validation or test;
- use validation only for early stopping and threshold selection;
- use test only for final evaluation and paired bootstrap.

## Experiment Arms

The Ding paper is slightly ambiguous about whether LightGBM is trained on the
reconstructed original training set or on the reconstructed SMOTE-balanced
training set. To avoid choosing the more convenient interpretation after seeing
results, the script reports both:

| ID | Description |
|----|-------------|
| `a1_baseline` | A1 dense LightGBM without SMOTE or AutoEncoder. |
| `a1_smote_only` | A1 dense LightGBM trained on SMOTE-balanced train data. |
| `ding_reconstructed_original_train` | AutoEncoder trained on SMOTE-balanced train; LightGBM trained on reconstructed original train. This follows the pseudocode literally. |
| `ding_reconstructed_balanced_train` | AutoEncoder trained on SMOTE-balanced train; LightGBM trained on reconstructed SMOTE-balanced train. This carries the SMOTE preprocessing through to the classifier. |

The key comparisons are:

- `ding_reconstructed_original_train` vs `a1_baseline`;
- `ding_reconstructed_balanced_train` vs `a1_baseline`;
- `ding_reconstructed_balanced_train` vs `a1_smote_only`.

The last comparison isolates whether AutoEncoder reconstruction adds value
beyond ordinary SMOTE on the same dense representation.

## Command

Initial Ding-style run before archival:

```bash
python src/run_ding_aieelg_experiment.py \
  --output-dir outputs/stratified_reset/ding_aieelg_ieee_cis \
  --target-fraud-rate 0.50 \
  --n-estimators 800 \
  --ae-epochs 30 \
  --n-bootstrap 1000
```

After archival, rerun from the repository root with:

```bash
python archive/source/ae_appendix/run_ding_aieelg_experiment.py \
  --output-dir outputs/stratified_reset/ding_aieelg_ieee_cis \
  --target-fraud-rate 0.50 \
  --n-estimators 800 \
  --ae-epochs 30 \
  --n-bootstrap 1000
```

The target fraud rate `0.50` is the closest adaptation to Ding's "balanced
training set" language. If this is too slow, use `0.15` for a controlled
lightweight comparison aligned with the existing augmentation experiments.

## Result Table

Run metadata:

| Item | Value |
|------|-------|
| Split | `stratified_holdout`, train/validation/test = 60/20/20 |
| Seed | 42 |
| Test prevalence | 0.03499 |
| A1 dense feature count | 432 |
| SMOTE target fraud rate | 0.50 |
| Synthetic fraud rows | 329,528 |
| AE architecture | 432 -> 256 -> 128 -> 128 -> 432 |
| AE output activation | `linear` |
| AE epochs | 30 |
| AE best validation MSE | 0.310165 |
| LightGBM trees | 800 |
| Bootstrap repeats | 1000 |

Test results:

| Arm | Test AP / PR-AUC | ROC-AUC | F1 | MCC | Bootstrap AP delta |
|-----|------------------|---------|----|-----|--------------------|
| `a1_baseline` | 0.746013 | 0.957418 | 0.709259 | 0.703010 | reference |
| `a1_smote_only` | 0.734853 | 0.948133 | 0.686757 | 0.692168 | -0.011160 vs baseline, 95% CI [-0.015295, -0.006467] |
| `ding_reconstructed_original_train` | 0.399351 | 0.827017 | 0.390996 | 0.430844 | -0.346662 vs baseline, 95% CI [-0.359829, -0.334360] |
| `ding_reconstructed_balanced_train` | 0.408320 | 0.829793 | 0.383632 | 0.436215 | -0.337693 vs baseline, 95% CI [-0.350395, -0.324887] |

The balanced Ding arm also loses directly against the SMOTE-only control:

```text
AP delta = -0.326533
95% CI  = [-0.339521, -0.314164]
p(delta <= 0) = 1.000
```

## Interpretation

Direct Ding-style reconstructed-feature AEELG does not transfer well to this
IEEE-CIS setup. The negative result is not only against the strongest A1 control;
it also loses badly against the matched SMOTE-only control. This suggests that
the reconstruction step smooths or distorts high-dimensional fraud-discriminative
signals after A1 preprocessing instead of improving class separation.

Full-balance SMOTE itself is also weaker than the A1 baseline in this run, so the
failure is not caused only by the AutoEncoder. However, the much larger drop from
SMOTE-only to reconstructed AEELG shows that AE reconstruction is the dominant
source of performance loss.

## Research-Gap Hook

Ding et al. remain useful as the starting architectural anchor for testing
whether SMOTE + AutoEncoder feature reconstruction + LightGBM transfers to a
larger, messier e-commerce fraud dataset. The isolated result supports a
negative but thesis-useful conclusion:

```text
A direct Ding-style reconstruction pipeline is not competitive on IEEE-CIS under
the active stratified protocol.
```

Because the AP gap is very large, Optuna tuning is not recommended as the next
main branch for this exact reconstructed-feature pipeline. If tuning is used as
an appendix robustness check, tune both:

- the strongest non-AE control (`a1_baseline` or `a1_smote_only`);
- the Ding-style AEELG candidate.

Do not promote AEELG unless the tuned AE candidate beats the matched tuned
control on test Average Precision and paired-bootstrap AP delta supports a
positive gain.

For a stronger thesis direction, the better tuning candidate is the separate
AE-as-augmentation branch, where the AutoEncoder is used to generate or organize
minority-class samples in latent/dense space rather than replacing the entire
feature matrix with reconstructed values.

## Artifacts

The script writes:

```text
outputs/stratified_reset/ding_aieelg_ieee_cis/experiment_summary.json
outputs/stratified_reset/ding_aieelg_ieee_cis/test_scores.csv
outputs/stratified_reset/ding_aieelg_ieee_cis/ae_training_history.csv
```
