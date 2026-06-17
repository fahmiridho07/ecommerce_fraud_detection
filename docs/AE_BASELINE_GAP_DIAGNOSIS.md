# AE vs Baseline Gap Diagnosis

Status: active diagnostic note, 2026-06-18.

Purpose: record, as an official pre-execution note, why the Autoencoder-LightGBM
proposed model does not beat the LightGBM baseline under the active stratified
protocol, and which paper-anchored experiments are allowed next. This file is a
diagnosis and decision record only. It must not be cited as a thesis result.

Read this after `THESIS_SCOPE.md` and `EXPERIMENT_REGISTRY.md`. It does not
change the source-of-truth order; it explains the evidence behind the AE
decision rule already stated in those files.

## Two Problems Were Being Conflated

The investigation separated one perceived problem into two distinct problems.

| | Problem | Status |
|---|---------|--------|
| Goal 1 | "The model cannot predict fraud well" (advisor concern, chronological PR-AUC ~0.5) | Resolved by the stratified switch: baseline test PR-AUC is now 0.8557. |
| Goal 2 | "The proposed AE-LightGBM must beat the LightGBM baseline" | Not satisfied, and structurally hard under the current AE integration design. |

The advisor concern (Goal 1) was about absolute fraud-detection ability. That
concern is addressed under stratified holdout, where the baseline already scores
PR-AUC 0.8557 instead of ~0.5. The remaining issue (Goal 2) is narrower: the AE
contribution does not add value on top of an already strong tabular baseline.
Goal 2 is not solved by "raising the score as high as possible" but by fixing the
AE integration design or the comparison framing.

Caveat on Goal 1: the large jump from chronological ~0.5 to stratified ~0.86 is
substantially attributable to entity leakage across train/test under stratified
splitting (the same `card`, `addr`, and device entities appear in both sides).
This is already documented as a limitation in `THESIS_SCOPE.md` and
`PAPER_ANCHORED_PREPROCESSING_RESET.md`, and must stay a limitation, not a claim.

## Evidence

All numbers below were recomputed from the staging stratified artifacts under:

```text
outputs/initial_proposal/split_strategy_current/stratified_holdout/
```

These are staging artifacts produced before the `outputs/stratified_reset/`
boundary. They are diagnostic evidence, not canonical active results. The active
A0/A1 ladder must still be rerun before any thesis-facing table is built.

### The reported ordering is confirmed

| Model | Test PR-AUC | Delta vs baseline | Bootstrap on AP delta |
|-------|------------:|------------------:|-----------------------|
| Baseline (frequency/missingness/time/amount, fixed P02 budget) | 0.8557 | reference | reference |
| AE latent LD32 add-on (proposed) | 0.8353 | -0.0205 | not in favor of AE |
| Score ensemble, fixed 0.50 | 0.8509 | -0.0048 | p(delta <= 0) = 1.000 |
| Score ensemble, alpha-tuned | 0.8556 | -0.0001 | p(delta <= 0) = 0.903 |

The strongest AE integration (alpha-tuned score ensemble) is a statistical tie,
not a win: observed AP delta -0.000116 with a paired-bootstrap interval that
includes zero.

### Root cause: the AE produces a weaker view of information the GBDT already has

Standalone discriminative strength of the AE signals on the test split:

| Signal | ROC-AUC | PR-AUC |
|--------|--------:|-------:|
| AE reconstruction error (single score) | 0.751 | 0.159 |
| Best single latent dimension | n/a | 0.216 |
| LightGBM on raw V-features (inside the baseline) | 0.968 | 0.856 |
| Test prevalence (random reference) | 0.500 | 0.035 |

The reconstruction error does separate fraud from normal (fraud mean MSE 0.109
is about 5.5x the normal mean MSE 0.020), so the AE itself is not broken. But
the AE was fitted only on the `V1`-`V339` block, which is the exact feature block
LightGBM already consumes at full resolution. Therefore:

- in latent-replacement mode (339 features compressed to 32), the AE loses
  information;
- in latent add-on mode, the AE adds a redundant lossy copy of features the model
  already has, which behaves as noise.

Either way the AE provides no new information, only a degraded view of existing
information, so it cannot improve a model that already reads the raw features.
This matches the tabular deep-learning literature: learned embeddings rarely beat
gradient-boosted trees on the same tabular inputs (Grinsztajn et al. 2022;
Gorishniy et al. 2021). Those two references are not yet in the literature card
set and should be added if this argument is used in the thesis.

### AE configuration that drove the result

- Autoencoder training subset: all training rows (fraud and normal mixed), so the
  AE learns to reconstruct fraud reasonably well and the anomaly contrast is
  softened.
- Latent dimension 32 from 339 V-features.
- Inputs restricted to the V-block only; no non-V feature seen by the AE.

## Implications

1. Bayesian optimization will not reverse this ordering. Tuning typically moves
   PR-AUC by roughly 0.005-0.02, which is smaller than the structural redundancy
   gap, and tuning must be applied fairly to both the baseline and the proposed
   model (tuned-vs-tuned). Deferring tuning until a defensible AE branch exists is
   the correct decision.
2. The metric matters. The AE+LightGBM precedents report ROC-AUC/F1, often with
   SMOTE, on small (~30 feature) credit-card datasets, not IEEE-CIS PR-AUC. On
   ROC-AUC the gap here is small (0.9644 vs 0.9685). PR-AUC is the strictest and
   most honest metric for this imbalance and should remain primary.
3. The AE's value is highest exactly where memorization fails, that is, under the
   harder temporal protocol. This connects the AE story to the leakage limitation
   and is the basis of the honest fallback framing (Option D).

## Candidate Next Experiments

Guiding principle: the AE must contribute information the GBDT cannot already
extract by itself, or the comparison must run on a footing where the AE's
strength is the variable under study. Promotion still follows the AE decision
rule in `THESIS_SCOPE.md` and `AI_AGENT_BRIEF.md` (test AP higher than the
strongest matched baseline plus paired-bootstrap support).

### Option A - AE as a clean complementary anomaly score

Train the AE on normal-only rows (semi-supervised anomaly detection), then add
only a compact set of reconstruction-error features (global plus per-group MSE)
to the full baseline, and compare with paired bootstrap.

- Anchors: Jiang et al. (2023) UAAD-FDNet; Thimonier et al. (2023); Niu et al.
  (2019).
- Status: cheapest defensible test. Code already exists
  (`src/train_autoencoder_normal_masked.py`,
  `src/train_ae_reconstruction_error_lgbm.py`) and has not yet been run under the
  active stratified protocol.
- Expected outcome: small chance of a win on stratified, but a clean and honest
  experiment regardless of result.

### Option B - AE as generative augmentation (highest PR-AUC upside)

Reframe the integration from "AE as features" to "AE as augmentation": use an
AE/VAE to generate synthetic fraud samples on the training split only, then train
LightGBM.

- Anchors: Alharbi et al. (2026) Multi-AE generative ensemble on IEEE-CIS;
  Kabane & Ouali (2024) for train-only resampling to avoid leakage.
- Status: different mechanism that the literature actually shows improving
  minority recall/PR-AUC. Highest upside for genuinely moving the primary metric.
- Caveat: gains are not guaranteed for a strong GBDT, and the scope file keeps
  resampling as an appendix branch. A null result is still a valid honest
  ablation. This option needs a written scope gate before it becomes mainline.

### Option C - Controlled V-block representation ablation

Hold the model and the non-V features fixed and vary only the V-block
representation: raw V, PCA-k, AE-latent-k, AE-latent plus reconstruction error.
Research question: which representation of the high-dimensional, high-missingness
V-block is best for LightGBM.

- Anchors: Prabha & Priscilla (2024) AE-latent to booster on IEEE-CIS.
- Status: narrow, defensible thesis contribution even if raw V wins, because it
  characterizes when AE representation helps.

### Option D - Honest framing if AE still loses

Position the result as: the AE does not beat a strong GBDT under stratified
evaluation (because entity leakage makes memorization sufficient), but provides
complementary value under the harder temporal protocol or as augmentation. This
is explicitly permitted by `AI_AGENT_BRIEF.md` and `THESIS_SCOPE.md` and turns a
negative result into a documented finding.

## Recommended First Step

Start with Option A because the code already exists and has not been run under
the active stratified split. It is the cheapest way to test whether the AE can
add any value before larger investment.

1. Train normal-only AE under stratified holdout.
2. Build baseline plus reconstruction-error features only.
3. Paired-bootstrap against the strongest matched A1 baseline.

If Option A ties or loses, proceed to Option B (Alharbi-style augmentation),
which has the largest upside for actually moving PR-AUC, after a written scope
gate.

Methodological guardrail: do not quietly compare the AE against a weaker A1
baseline to manufacture a win. The AE must be compared against the strongest
matched baseline under the same stratified test split. That is what survives a
thesis defense.

## Thesis Wording For Now

> Diagnosis menunjukkan bahwa di bawah protokol stratified holdout, baseline
> LightGBM sudah mencapai PR-AUC tinggi sehingga keluhan awal "model kurang
> mampu memprediksi fraud" sudah teratasi. Masalah yang tersisa adalah model
> usulan integrasi Autoencoder-LightGBM belum memberi peningkatan di atas
> baseline. Penyebab utamanya adalah Autoencoder dilatih pada blok fitur V yang
> sama dengan yang sudah dipakai LightGBM secara penuh, sehingga representasi
> laten hanya menjadi versi terkompresi yang redundan. Karena itu, eksperimen
> berikutnya difokuskan pada skema integrasi di mana Autoencoder memberi
> informasi yang tidak bisa diekstrak langsung oleh LightGBM, yaitu skor anomali
> rekonstruksi dari Autoencoder normal-only, augmentasi generatif berbasis
> Autoencoder, atau studi ablasi representasi blok V. Tuning Bayesian Optimization
> ditunda sampai cabang Autoencoder yang defensible tersedia, dan akan diterapkan
> secara adil pada baseline maupun model usulan.

## Source Anchors

- Alharbi et al. (2026): `docs/literature/cards/Alharbi_2026_Multi_AE_Generative_Ensemble_IEEE-CIS.md`
- Jiang et al. (2023): `docs/literature/cards/Jiang_2023_UAAD_FDNet_Autoencoder_IEEE-CIS.md`
- Thimonier et al. (2023): `docs/literature/cards/Thimonier_2023_Anomaly_Detection_Fraud_Online_Payments.md`
- Niu et al. (2019): `docs/literature/cards/Niu_2019_Supervised_vs_Unsupervised_Fraud_Detection.md`
- Prabha & Priscilla (2024): `docs/literature/cards/Prabha_Priscilla_2024_LSTMAE_XGBoost_IEEE-CIS.md`
- Ding et al. (2024): `docs/literature/cards/Ding_2024_AutoEncoder_Enhanced_LightGBM_Fraud_Detection.md`
- Du et al. (2023): `docs/literature/cards/Du_2023_AutoEncoder_and_LightGBM_Fraud_Detection.md`
- Kabane & Ouali (2024): `docs/literature/cards/Kabane_2024_Sampling_Leakage_XGBoost_Fraud_Detection.md`
- Saito & Rehmsmeier (2015): `docs/literature/cards/Saito_Rehmsmeier_2015_Precision_Recall_Plot.md`
- Williams et al. (2021): `docs/literature/cards/Williams_2021_Effect_Class_Imbalance_Precision_Recall_Curves.md`
- Lucas et al. (2019): `docs/literature/cards/Lucas_2019_Dataset_Shift_Credit_Card_Fraud.md`
- Akiba et al. (2019): `docs/literature/cards/Akiba_2019_Optuna_Hyperparameter_Optimization.md`
- External, not yet carded (add if used): Grinsztajn et al. (2022) "Why do
  tree-based models still outperform deep learning on tabular data"; Gorishniy et
  al. (2021) "Revisiting Deep Learning Models for Tabular Data".

PDFs under `../2. Reference/` remain the source of truth for exact claims and
quotes. Literature cards are summaries, not citation substitutes.
