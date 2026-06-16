# Experiment Registry

This is the active thesis registry after cleanup. It tracks only the original proposal-scope experiments (P01–P04). Parked work is documented in [`../archive/README.md`](../archive/README.md).

## Canonical Artifact Location

The **authoritative** P01–P04 rerun (post missingness-preserving AE fix) lives under `outputs/initial_proposal/`. Legacy paths under `outputs/baseline_lgbm/`, `outputs/ae_lgbm/`, and `outputs/optuna/` remain for backward compatibility but reflect the pre-fix pipeline.

| Artifact | Path |
|----------|------|
| Four-row comparison table | `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv` |
| Missing-artifact log | `outputs/initial_proposal/final_comparison/initial_proposal_missing_artifacts.json` |

## Active Thesis Experiments

| Canonical ID | Legacy ID | Experiment | Script | Canonical output path | Status |
|--------------|-----------|------------|--------|----------------------|--------|
| BASE-01 | P01 | Baseline LightGBM default | `src/train_baseline_lgbm.py` | `outputs/initial_proposal/baseline_lgbm_default/` | Complete (post-fix rerun) |
| BASE-02 | P02 | Baseline LightGBM Optuna tuned | `src/tune_lgbm_optuna.py --model_type baseline_lgbm` | `outputs/initial_proposal/optuna/baseline_lgbm_tuned/` | Complete (post-fix rerun) |
| AE-01 | P03 | AE-LightGBM LD32 + `v_missing_*` indicators | `src/train_autoencoder_robust.py --latent-dim 32`, `src/train_ae_lgbm.py` | `outputs/initial_proposal/ae_lgbm_ld32_default/` | Complete (post-fix rerun) |
| AE-02 | P04 | AE-LightGBM LD128 + `v_missing_*`, Optuna tuned | `src/train_autoencoder_robust.py --latent-dim 128`, `src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128` | `outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned/` | Complete (post-fix rerun) |

Supporting Autoencoder artifacts:

| Step | Path |
|------|------|
| AE LD32 | `outputs/initial_proposal/autoencoder_robust_ld32/` |
| AE LD128 | `outputs/initial_proposal/autoencoder_robust_ld128/` |

## Current Metrics (Post-Fix Rerun)

Rerun date: **2026-06-16**. Full IEEE-CIS train set (`SAMPLE_SIZE=None`). Chronological split per `outputs/split_summary.json`. Primary metric: **validation/test Average Precision (PR-AUC)** at the validation-selected threshold. Optuna tuning used `--tuning_profile final` with **15 trials** (exploratory budget).

| ID | Val AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Features | Tuning trials |
|----|--------|---------|--------------|---------|----------|----------|---------------|
| P01 / BASE-01 | 0.602433 | 0.485756 | 0.875195 | 0.477868 | 0.486715 | 432 | — |
| P02 / BASE-02 | 0.631767 | **0.504900** | 0.883431 | 0.493865 | 0.494270 | 432 | 15 |
| P03 / AE-01 | 0.590998 | 0.480217 | 0.880349 | 0.477902 | 0.472922 | 464 | — |
| P04 / AE-02 | 0.606709 | 0.484527 | 0.878874 | 0.485623 | 0.496689 | 560 | 15 |

**Test PR-AUC ranking:** P02 > P01 > P04 > P03.

### Feature construction (post-fix)

| Model | Non-V | Latent | `v_missing_*` | Total |
|-------|-------|--------|---------------|-------|
| P01 / P02 | 93 | — | — (original V* kept as NaN) | 432 |
| P03 | 93 | 32 | 339 | 464 |
| P04 | 93 | 128 | 339 | 560 |

### AE reconstruction drift (masked MSE, observed cells only)

| Latent dim | Train MSE | Valid MSE | Test MSE |
|------------|-----------|-----------|----------|
| LD32 | 0.0199 | 0.0355 | 0.1516 |
| LD128 | 0.0056 | 0.0128 | 0.0629 |

LD128 reconstructs better and drifts less, but P04 still does not beat P02 on test PR-AUC.

## Thesis-Facing Finding (Post-Fix)

After correcting the AE pipeline (median imputation, masked reconstruction loss, linear latent layer, and `v_missing_*` indicators downstream), **the tuned LightGBM baseline (P02) remains the best model** under the chronological IEEE-CIS protocol.

The missingness-preserving fix did not close the gap:

- P03 test AP moved from 0.481593 (pre-fix) to 0.480217 (post-fix).
- P04 test AP moved from 0.490686 (pre-fix) to 0.484527 (post-fix).

This supports a defendable thesis narrative: the AE latent replacement strategy loses supervised signal from original `V*` values that LightGBM exploits natively, and better AE reconstruction (LD128) does not automatically yield better fraud detection.

## Pre-Fix Local Metrics (Historical)

Keep for traceability only. Generated before median imputation, masked loss, linear latent, and `v_missing_*` indicators.

| ID | Validation AP | Test AP | Features |
|----|---------------|---------|----------|
| P01 / BASE-01 | 0.602433 | 0.485756 | 432 |
| P02 / BASE-02 | 0.624072 | 0.501438 | 432 |
| P03 / AE-01 | 0.591398 | 0.481593 | 125 |
| P04 / AE-02 | 0.610631 | 0.490686 | 221 |

Legacy artifact paths: `outputs/baseline_lgbm/`, `outputs/ae_lgbm/`, `outputs/optuna/`.

## Representation Ablation (Post-Diagnostic, Not P01–P04)

Controlled fix after diagnostics showed baseline `V*` gain share is 33.4% while `v_missing_*` contributes only 0.12% in P03.

| ID | Model | Script | Output path | Purpose |
|----|-------|--------|-------------|---------|
| AE-03 | Hybrid top-25 `V*` + LD32 latent | `src/train_ae_lgbm.py --retain-top-v-features 25` | `outputs/initial_proposal/ae_lgbm_ld32_top25v_default/` | Test whether retaining high-gain `V*` values fixes the replacement bottleneck before any AE tuning |

Comparison table:

```bash
python src/build_representation_ablation_comparison.py
```

Output: `outputs/initial_proposal/representation_ablation/representation_ablation_comparison.csv`

**AE-03 result (2026-06-16, default LightGBM, no tuning):**

| Model | Val AP | Test AP | vs P01 | vs P03 |
|-------|--------|---------|--------|--------|
| P01 baseline | 0.602433 | 0.485756 | — | +0.00554 |
| P03 full replacement | 0.590998 | 0.480217 | −0.00554 | — |
| AE-03 top-25 V + latent | 0.602539 | 0.485306 | −0.00045 | +0.00509 |

Retaining the top-25 baseline `V*` features by gain nearly closes the gap to P01 and confirms the diagnostic finding: full latent replacement discards high-value supervised signal that `v_missing_*` masks cannot recover.

This ablation is **supporting evidence** for the thesis discussion. It does not replace canonical P03 unless a written scope decision says otherwise.

## Reconstruction-Error Augmentation (Post-Fix, Not P01-P04)

Archive review showed that AE reconstruction error was the strongest historical AE integration path. This post-fix rerun keeps all original baseline features and appends only two LD128 Autoencoder anomaly features from `outputs/initial_proposal/autoencoder_robust_ld128/`:

- `v_ae_reconstruction_mse`
- `v_ae_reconstruction_log1p_mse`

Command:

```bash
python src/train_ae_reconstruction_error_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --output-dir outputs/initial_proposal/ae_reconstruction_error_ld128_default \
  --initial-proposal-dir outputs/initial_proposal \
  --phase-name AE_RECON_LD128_initial_proposal_postfix
```

**AE-04 result (2026-06-16, default LightGBM, no Optuna):**

| Model | Val AP | Test AP | Test ROC-AUC | Test F1 | Test MCC | Features |
|-------|--------|---------|--------------|---------|----------|----------|
| P01 baseline default | 0.602433 | 0.485756 | 0.875195 | 0.477868 | 0.486715 | 432 |
| P02 baseline tuned | 0.631767 | **0.504900** | 0.883431 | 0.493865 | 0.494270 | 432 |
| P03 latent replacement LD32 | 0.590998 | 0.480217 | 0.880349 | 0.477902 | 0.472922 | 464 |
| P04 latent replacement LD128 tuned | 0.606709 | 0.484527 | 0.878874 | **0.485623** | **0.496689** | 560 |
| AE-04 reconstruction-error augmentation | 0.612397 | 0.495067 | 0.873653 | 0.485057 | 0.491299 | 434 |

AE-04 improves test AP over P01 (+0.009311), P03 (+0.014851), and P04 (+0.010541), but remains below P02 tuned (-0.009832). Its top feature by gain is `v_ae_reconstruction_mse`, supporting the interpretation that the Autoencoder contributes useful anomaly signal when used as augmentation rather than full `V*` replacement.

## Diagnostics

Run after the isolated rerun completes:

```bash
python src/generate_initial_proposal_diagnostics.py
```

Outputs: `outputs/initial_proposal/diagnostics/` (`diagnostic_summary.json`, `diagnostic_notes.md`, and supporting CSV tables). See `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md` for the full file list.

## Caveats

- **P03 vs P04 is not fully apples-to-apples:** P03 uses LD32 with default LightGBM params; P04 uses LD128 with Optuna tuning. No tuned LD32 AE-LightGBM path exists in the active registry.
- **Tuning budget:** The post-fix rerun used 15 Optuna trials per tuned model. Increase to 50 before treating hyperparameters as final for defense.
- **Legacy vs canonical paths:** Scripts default to legacy `outputs/` directories. Use `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md` for isolated reruns under `outputs/initial_proposal/`.
- **Archived branches** may report higher metrics but answer different research questions and are out of proposal scope.

## Archived Families

| Archive folder | Status |
|----------------|--------|
| `archive/source/ae_appendix/` | Appendix only |
| `archive/source/behavioral_fusion/` | Out of proposal scope |
| `archive/source/feature_engineering_ensembles/` | Out of proposal scope |
| `archive/source/gbdt_wip/` | WIP only |
| `archive/source/methodology_reports/` | Reporting and diagnostics only |
| `archive/docs/` | Historical governance and branch notes |
| `archive/notebooks/` | Historical notebook report |
| `archive/results/` | Compact archived summaries |
