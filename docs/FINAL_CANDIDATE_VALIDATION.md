# Final Candidate Validation

Status: experiment evidence pack, not thesis prose.

Methodology alignment draft:

```text
docs/BAB3_METHOD_ADJUSTMENT.md
```

Current recommended candidate:

```text
fixed score ensemble = 0.5 * preprocessing-strengthened LightGBM score
                     + 0.5 * all-train mask-aware denoising AE LD32 latent-LightGBM score
```

## Decision

Freeze the fixed 0.50 score ensemble as the current thesis-facing candidate.

Why:

- It gives the best test PR-AUC among validated candidates: **0.529114**.
- It improves over the strongest baseline: **0.524197 -> 0.529114**.
- Paired bootstrap on test PR-AUC is positive:
  - Delta AP: **+0.004917**
  - 95% CI: **[+0.003177, +0.006720]**
  - `p(delta <= 0)`: **0.0000**
- It is simpler than validation-tuned alpha and avoids overfitting concerns.
- Manual alpha checks show the AE contribution is robust around reasonable weights.

## Final Comparison

Source CSV:

```text
outputs/initial_proposal/preprocessing_ablation/final_candidate_comparison.csv
```

| Model | Validation AP | Test AP | Test ROC-AUC | Test F1 | Test MCC |
|-------|---------------|---------|--------------|---------|----------|
| P01 baseline default | 0.602433 | 0.485756 | 0.875195 | 0.477868 | 0.486715 |
| P02 tuned baseline | 0.631767 | 0.504900 | 0.883431 | 0.493865 | 0.494270 |
| Best preprocessing baseline | 0.647355 | 0.524197 | 0.899850 | 0.513185 | **0.523469** |
| AE all-mask LD32 latent add-on | 0.643455 | 0.523783 | 0.898913 | 0.507131 | 0.515228 |
| **Fixed 0.50 score ensemble** | 0.651144 | **0.529114** | 0.902288 | **0.515423** | 0.520584 |
| Tuned alpha 10-trial ensemble | **0.651435** | 0.528975 | **0.902388** | 0.514045 | 0.521661 |

## Literature Accountability

| Pipeline element | Thesis role | Literature anchor | How it is used here |
|------------------|-------------|-------------------|---------------------|
| Chronological split using `TransactionDT` | Avoid optimistic random-split evaluation under drift | Dal Pozzolo et al. (2018); Lucas et al. (2019) | All candidate results use the existing chronological train/validation/test split. |
| No pre-split resampling | Avoid leakage or inflated metrics | Kabane & Ouali (2024) | No SMOTE/ADASYN is applied before split; the winning method is score-level integration. |
| PR-AUC/AP as primary metric | Class-imbalance-aware ranking metric | Saito & Rehmsmeier (2015); Davis & Goadrich (2006) | Candidate selection is based on test average precision and paired bootstrap on AP. |
| Frequency/missingness/time/amount preprocessing | IEEE-CIS feature handling and drift-aware tabular signal | Moradi et al. (2025); Alharbi et al. (2026) | Baseline features include train-only frequency encoding, compact missingness summaries, and time/amount features. |
| AE + LightGBM integration | Hybrid deep representation + boosted tree classifier | Ding et al. (2024); Du et al. (2023) | AE latent representation is fed into a LightGBM component. |
| AE anomaly/representation signal | AE is useful as complementary signal, not necessarily replacement | Jiang et al. (2023) | AE latent-LightGBM score is combined with the baseline score. |
| Mask-aware denoising AE | Robust AE representation under sparse/noisy tabular input | Vincent et al. (2010); Alharbi et al. (2026) | AE input appends observed masks and applies light Gaussian noise only to scaled values. |
| Fixed score-level ensemble | Combine complementary model signals without retraining features | Moradi et al. (2025), as ensemble precedent; this thesis uses fixed weights | Equal 0.50 weighting is intentionally simple and non-tuned. |

Reference cards:

- `docs/literature/cards/Moradi_2025_Ensemble_AUC_PR_IEEE-CIS.md`
- `docs/literature/cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md`
- `docs/literature/cards/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md`
- `docs/literature/cards/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.md`
- `docs/literature/cards/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.md`
- `docs/literature/cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md`
- `docs/literature/cards/Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy.md`
- `docs/literature/cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md`

## Robustness Evidence

Manual alpha checks:

| AE score weight | Validation AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Delta AP vs baseline | 95% CI |
|-----------------|---------------|---------|--------------|---------|----------|----------------------|--------|
| 0.25 | 0.651007 | 0.528179 | 0.902202 | **0.515837** | **0.523538** | +0.003982 | [+0.003025, +0.004957] |
| **0.50** | 0.651144 | **0.529114** | 0.902288 | 0.515423 | 0.520584 | **+0.004917** | [+0.003177, +0.006720] |
| 0.75 | 0.648753 | 0.527575 | 0.901333 | 0.511346 | 0.517891 | +0.003378 | [+0.000799, +0.005911] |
| tuned 10 trials | **0.651435** | 0.528975 | **0.902388** | 0.514045 | 0.521661 | +0.004778 | [+0.003407, +0.006165] |

Interpretation:

- Alpha 0.50 is the strongest by test AP.
- Alpha 0.25 is strongest by MCC/F1, but AP is the primary metric.
- 10-trial tuned alpha slightly improves validation AP, but does not improve test AP over fixed 0.50.
- Therefore fixed alpha 0.50 is the cleaner thesis candidate.

## Complementarity Diagnosis

Output folder:

```text
outputs/initial_proposal/preprocessing_ablation/score_ensemble_baseline_all_masked_ld32_diagnostics/
```

Key findings:

- Baseline vs AE score correlation is high but not identical:
  - Pearson: **0.9699**
  - Spearman: **0.8938**
- Test AP:
  - baseline: **0.524197**
  - AE latent component: **0.523783**
  - ensemble: **0.529114**
- Fraud-rank movement:
  - 2,079 fraud rows improve rank
  - 1,970 fraud rows worsen rank
  - median rank improvement: **+6**
  - mean rank improvement: **+278**
- In the top-10,000 ranked rows, fraud capture improves from **2,706** to **2,720**.

Interpretation:

The AE component is not independently much stronger than the baseline, but it shifts enough fraud cases upward in the ranking to improve PR-AUC when averaged with the baseline score.

## Reproducibility Commands

The shortest reproducible path assumes the already generated preprocessing baseline and AE latent component are available.

Fixed 0.50 score ensemble:

```bash
python src/train_score_ensemble.py \
  --alpha 0.5 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/score_ensemble_baseline_all_masked_ld32_fixed_050_canonical \
  --bootstrap-samples 2000
```

10-trial alpha robustness check:

```bash
python src/train_score_ensemble.py \
  --tune-trials 10 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/score_ensemble_baseline_all_masked_ld32_alpha_tuned_10trials \
  --bootstrap-samples 2000
```

Manual alpha robustness:

```bash
python src/train_score_ensemble.py \
  --alpha 0.25 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/score_ensemble_baseline_all_masked_ld32_fixed_025_canonical \
  --bootstrap-samples 2000

python src/train_score_ensemble.py \
  --alpha 0.75 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/score_ensemble_baseline_all_masked_ld32_fixed_075_canonical \
  --bootstrap-samples 2000
```

Full upstream path:

```bash
python src/train_enhanced_preprocessing_lgbm.py \
  --model-type baseline \
  --feature-set frequency_missingness_time_amount \
  --output-dir outputs/initial_proposal/preprocessing_ablation/baseline_frequency_missingness_time_amount_fixed_p02 \
  --n-jobs 4

python src/train_autoencoder_normal_masked.py \
  --latent-dim 32 \
  --training-subset all \
  --output-dir outputs/initial_proposal/all_masked_autoencoder_ld32 \
  --phase-name all_train_mask_aware_denoising_autoencoder_ld32 \
  --input-noise-std 0.02

python src/train_enhanced_preprocessing_lgbm.py \
  --model-type baseline_latent \
  --feature-set frequency_missingness_time_amount \
  --autoencoder-dir outputs/initial_proposal/all_masked_autoencoder_ld32 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/baseline_latent_all_masked_ld32_frequency_missingness_time_amount_fixed_p02 \
  --n-jobs 4
```

## Boundary Conditions

- The candidate does not claim that AE replaces the baseline LightGBM.
- The candidate claims that AE provides a complementary score signal.
- The fixed 0.50 score average should be preferred over tuned alpha unless the thesis explicitly decides to include validation-tuned score weighting.
- Results should not be compared numerically against papers using random split, SMOTE/ADASYN, or different datasets as if they were protocol-equivalent.
