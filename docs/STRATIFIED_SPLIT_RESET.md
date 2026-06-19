# Stratified Split Reset

Status: completed cleanup decision record, 2026-06-17.

Purpose: document the project cleanup before rerunning experiments under the
new thesis protocol. The active rerun ladder was completed on 2026-06-18; see
`THESIS_SCOPE.md`, `THESIS_RESULTS_BAB4.md`, and `EXPERIMENT_REGISTRY.md` for
current results.

## Decision

The active evaluation protocol is now:

```text
stratified_holdout, 60% train / 20% validation / 20% test, random_state=42
```

Reason:

- The thesis needs a narrower, paper-anchored protocol that is easy to defend.
- Several IEEE-CIS related works use random/stratified evaluation or do not
  expose a strict temporal split.
- Temporal drift and concept drift will be discussed as limitation/future work,
  not treated as the main S1 experiment protocol.

## Code Cleanup

Default split strategy is centralized in:

```text
src/config.py
```

Active scripts now use `stratified_holdout` by default and write
`split_strategy` into their run configs:

- `src/check_data_split.py`
- `src/train_baseline_lgbm.py`
- `src/train_autoencoder_robust.py`
- `src/train_autoencoder_normal_masked.py`
- `src/train_ae_lgbm.py`
- `src/tune_lgbm_optuna.py`
- `src/train_enhanced_preprocessing_lgbm.py`
- `src/train_score_ensemble.py`
- AE reconstruction and bootstrap helper scripts.

`chronological` remains available through `--split-strategy chronological` for
historical reproduction only.

## Split Check Result

Command:

```bash
python src/check_data_split.py
```

Observed on the full local IEEE-CIS training data:

| Split | Rows | Fraud rate |
|-------|-----:|-----------:|
| Train | 354,324 | 3.4991% |
| Validation | 118,108 | 3.4985% |
| Test | 118,108 | 3.4993% |

This confirms that the active split preserves the class ratio across all three
sets.

## Archive Boundary

The following are archived historical evidence:

| Evidence | Old protocol | New status |
|----------|--------------|------------|
| P01-P04 proposal comparison | Chronological | Historical baseline block |
| AE-05 hybrid result | Chronological | Historical post-diagnostic candidate |
| Fixed score ensemble result | Chronological | Historical post-diagnostic candidate |
| Preprocessing diagnostic branch | Chronological | Empirical note, not active protocol |

Do not place historical chronological numbers in the same result table as new
stratified numbers.

## Original Rerun Order

This order is preserved for traceability. It is not a current to-do list unless
the scope is reopened or an artifact must be regenerated.

The original cleanup run used this order:

```bash
python src/check_data_split.py

python src/train_baseline_lgbm.py \
  --output-dir outputs/stratified_reset/baseline_lgbm_default \
  --phase-name S0_baseline_lgbm_default_stratified

python src/tune_lgbm_optuna.py \
  --model_type baseline_lgbm \
  --tuning_profile final \
  --n_trials 15 \
  --storage sqlite:///outputs/stratified_reset/optuna/baseline_lgbm_tuned/study.db \
  --study_name stratified_reset_baseline_lgbm \
  --output-dir outputs/stratified_reset/optuna/baseline_lgbm_tuned \
  --skip-global-comparison-update
```

After those pass, implement/run the A1 Alharbi-style preprocessing branch.
The A1 default and tuned entry points are now:

```bash
python src/train_paper_preprocessing_lgbm.py \
  --output-dir outputs/stratified_reset/alharbi_style_lgbm_default \
  --phase-name A1_alharbi_style_lgbm_default_stratified

python src/tune_lgbm_optuna.py \
  --model_type alharbi_lgbm \
  --tuning_profile final \
  --n_trials 15 \
  --storage sqlite:///outputs/stratified_reset/optuna/alharbi_lgbm_tuned/study.db \
  --study_name stratified_reset_alharbi_lgbm \
  --output-dir outputs/stratified_reset/optuna/alharbi_lgbm_tuned \
  --skip-global-comparison-update
```

## Thesis Wording

Use this final wording for the active scope:

> Setelah reset metodologi, eksperimen utama menggunakan stratified holdout
> 60/20/20 dengan random seed tetap. Hasil final menunjukkan bahwa autoencoder
> berkontribusi sebagai oversampler di latent space pada representasi A1 dense,
> bukan sebagai feature extractor universal. Hasil chronological sebelumnya
> dipertahankan sebagai arsip eksplorasi, sedangkan evaluasi temporal dan
> concept drift dibahas sebagai keterbatasan dan saran penelitian lanjutan.
