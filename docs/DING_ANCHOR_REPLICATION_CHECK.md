# Ding Anchor Replication Check

Status: sanity-check evidence after supervisor feedback, 2026-06-19.

Purpose: test whether the Ding et al. (2024) AEELG pipeline works on Ding's
own datasets before using its negative transfer on IEEE-CIS as thesis evidence.

Supervisor-facing question:

```text
If the Ding-style code does not work on the same dataset as the anchor paper,
debug the implementation first. If it works there but fails on IEEE-CIS, analyze
why the transfer fails before proposing alternatives.
```

## Data

Local folder:

```text
../Eksperimen Ding et al/
```

Datasets used:

| Dataset | Local file | Shape | Positive rate |
|---|---:|---:|---:|
| ULB European credit-card fraud | `ULB Dataset/creditcard.csv` | 284,807 x 31 | 0.1727% |
| Santander customer transaction prediction | `Santander Dataset/train.csv` | 200,000 x 202 | 10.0490% |

## Runner

New script:

```bash
python src/run_ding_anchor_replication.py --help
```

The runner keeps guarded thesis protocol choices:

- split before fitted preprocessing;
- scaler fit on train only;
- SMOTE-style interpolation on train only;
- validation/test are never oversampled;
- test is used only for final evaluation;
- paper metrics are reported: ROC-AUC, recall, F1, MCC, BCR;
- thesis metric is also reported: Average Precision / PR-AUC.

## ULB Result

Best sanity-check run:

```bash
python src/run_ding_anchor_replication.py \
  --dataset ulb \
  --output-dir outputs/ding_anchor/ulb_ding_public_linear \
  --scaling amount_hour \
  --lgbm-preset ding \
  --hidden-dim 16 \
  --latent-dim 8 \
  --decoder-dim 8 \
  --output-activation linear \
  --ae-epochs 60 \
  --n-estimators 300
```

Paper anchor: Ding et al. report SMOTE+AEELG AUC 96.83% and F-measure 80.27%
on ULB.

Selected-threshold test metrics:

| Arm | AP | ROC-AUC | Precision | Recall | F1 | MCC | BCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline LightGBM | 0.791383 | 0.970277 | 0.829630 | 0.756757 | 0.791519 | 0.792013 | 0.878244 |
| SMOTE + LightGBM | 0.680659 | 0.958510 | 0.818182 | 0.729730 | 0.771429 | 0.772321 | 0.864724 |
| Ding reconstructed original train | 0.784617 | 0.967585 | 0.866142 | 0.743243 | 0.800000 | 0.802029 | 0.871522 |
| Ding reconstructed balanced train | 0.536060 | 0.968452 | 0.639785 | 0.804054 | 0.712575 | 0.716691 | 0.901634 |

Default 0.5-threshold metrics for the closest Ding arm:

| Arm | AP | ROC-AUC | Precision | Recall | F1 | MCC | BCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ding reconstructed original train | 0.784617 | 0.967585 | 0.840909 | 0.750000 | 0.792857 | 0.793820 | 0.874877 |

Interpretation:

- The ULB run is close to the paper-level ULB claim: ROC-AUC 0.9676 vs paper
  0.9683, and selected-threshold F1 0.8000 vs paper F-measure 0.8027.
- This supports that the AE reconstruction implementation is not fundamentally
  broken on Ding's small dense numeric dataset.
- The reconstructed-original interpretation is the plausible Ding replication.
  Carrying SMOTE-balanced reconstructed rows through to LightGBM hurts AP and
  F1.
- The strict public-code style ReLU output activation performed worse because
  the ULB PCA/standardized features contain negative values. Linear output is
  more appropriate for reconstruction.

## Santander Result

Best current Santander run:

```bash
python src/run_ding_anchor_replication.py \
  --dataset santander \
  --output-dir outputs/ding_anchor/santander_linear_wide_no_l1 \
  --scaling all \
  --lgbm-preset ding \
  --hidden-dim 256 \
  --latent-dim 64 \
  --decoder-dim 128 \
  --output-activation linear \
  --learning-rate 0.0005 \
  --l1-penalty 0 \
  --ae-epochs 80 \
  --n-estimators 300
```

Selected-threshold test metrics:

| Arm | AP | ROC-AUC | Precision | Recall | F1 | MCC | BCR |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline LightGBM | 0.567343 | 0.886614 | 0.569975 | 0.492453 | 0.528386 | 0.481503 | 0.725475 |
| SMOTE + LightGBM | 0.329638 | 0.789904 | 0.342494 | 0.415492 | 0.375478 | 0.299925 | 0.663194 |
| Ding reconstructed original train | 0.475487 | 0.847491 | 0.431915 | 0.523470 | 0.473305 | 0.410521 | 0.723279 |
| Ding reconstructed balanced train | 0.414677 | 0.829463 | 0.460647 | 0.375684 | 0.413850 | 0.357923 | 0.663273 |

Interpretation:

- Santander is not replicated yet under the guarded runner. The baseline
  LightGBM is already close to Ding's reported Santander AUC range, while AE
  reconstruction underperforms it.
- A smaller regularized AE collapsed to near-mean reconstruction on Santander;
  the wider no-L1 AE improved reconstruction but still did not make AEELG
  competitive.
- Before using Santander as strong validation evidence, Ding's exact Santander
  split, thresholding, AE architecture, sampling variant, and LightGBM settings
  need deeper reconstruction from the paper/source.

## Thesis Decision

Use the ULB result as the immediate code sanity check:

```text
The Ding-style AE reconstruction pipeline can reproduce paper-level behavior on
the smaller dense ULB dataset when implemented with leakage-safe train-only
preprocessing and a linear reconstruction output.
```

Use IEEE-CIS and Santander failures as diagnosis targets, not as a reason to jump
to unrelated methods:

- IEEE-CIS is mixed-type, high-missingness, and much wider after preprocessing.
- Full-balance SMOTE can create many false positives and reduce precision.
- Reconstructing the entire feature matrix can smooth fraud-discriminative
  signals that LightGBM uses directly.
- The Ding paper reports ROC-AUC/F1/MCC/BCR, while this thesis prioritizes
  Average Precision under severe imbalance.
- A strong LightGBM baseline can already match or exceed the reconstructed
  representation.

Next methodological move should stay within the original AE+LightGBM objective:

1. Treat direct Ding AEELG reconstruction as the initial anchor and negative
   transfer test on IEEE-CIS.
2. Analyze why reconstruction loses against LightGBM.
3. Only then justify a constrained alternative, such as using AE as a latent
   minority oversampler instead of replacing features with reconstructed values.
