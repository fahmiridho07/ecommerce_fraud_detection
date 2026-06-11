# Initial Proposal Rerun Guide

This guide covers **only** the original thesis proposal experiment path. It does **not** cover behavioral features, causal behavioral alignment, CDV AE, AE15, feature-engineered static FE branches, late fusion (LF01), score ensembles, or graph/temporal/fusion experiments.

## Active models (initial proposal only)

| Canonical ID | Legacy ID | Role |
|--------------|-----------|------|
| BASE-01 | P01 | Baseline LightGBM default |
| BASE-02 | P02 | Tuned baseline LightGBM |
| AE-01 | P03 | AE-LightGBM default (LD32 latent replacement) |
| AE-02 | P04 | Tuned AE-LightGBM LD128 |

## Out of scope

Do **not** use these branches when reproducing the initial proposal comparison:

- Feature-engineered static FE (`baseline_lgbm_entity_time_amount_features`, UID, velocity)
- AE augmented / AE15 / CDV AE branches
- Behavioral and causal behavioral LightGBM (`causal_behavioral_*`)
- Late fusion LF01 / FUS-01
- Score ensemble and three-model ensemble scripts
- Task-aware AE, selected-numerical AE, reconstruction-error augmentation

## Isolated output directories (recommended)

Use a dedicated tree under `outputs/initial_proposal/` so reruns do not overwrite legacy thesis artifacts:

| Step | Isolated directory |
|------|-------------------|
| BASE-01 | `outputs/initial_proposal/baseline_lgbm_default` |
| AE LD32 | `outputs/initial_proposal/autoencoder_robust_ld32` |
| AE-01 | `outputs/initial_proposal/ae_lgbm_ld32_default` |
| AE LD128 | `outputs/initial_proposal/autoencoder_robust_ld128` |
| BASE-02 | `outputs/initial_proposal/optuna/baseline_lgbm_tuned` |
| AE-02 | `outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned` |
| Comparison | `outputs/initial_proposal/final_comparison/` |

Legacy locations (`outputs/baseline_lgbm/`, `outputs/ae_lgbm/`, etc.) remain the script defaults for backward compatibility.

## Required command order (isolated rerun)

Run from the repository root after installing `requirements.txt` and placing IEEE-CIS files under `data/raw/` (or Kaggle input).

```bash
python src/check_data_split.py

python src/train_baseline_lgbm.py \
  --output-dir outputs/initial_proposal/baseline_lgbm_default \
  --phase-name 2_baseline_lgbm_initial_proposal

python src/train_autoencoder_robust.py \
  --latent-dim 32 \
  --output-dir outputs/initial_proposal/autoencoder_robust_ld32 \
  --phase-name 3B_robust_autoencoder_representation_learning_ld32

python src/train_ae_lgbm.py \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld32 \
  --output-dir outputs/initial_proposal/ae_lgbm_ld32_default \
  --phase-name 4_ae_lgbm_ld32_default_initial_proposal

python src/train_autoencoder_robust.py \
  --latent-dim 128 \
  --output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --phase-name 3B_robust_autoencoder_representation_learning_ld128

python src/tune_lgbm_optuna.py \
  --model_type baseline_lgbm \
  --tuning_profile final \
  --n_trials 50 \
  --storage sqlite:///outputs/initial_proposal/optuna/baseline_lgbm_tuned/study.db \
  --study_name initial_proposal_baseline_lgbm \
  --output-dir outputs/initial_proposal/optuna/baseline_lgbm_tuned \
  --skip-global-comparison-update

python src/tune_lgbm_optuna.py \
  --model_type ae_lgbm_ld128 \
  --autoencoder-output-dir outputs/initial_proposal/autoencoder_robust_ld128 \
  --tuning_profile final \
  --n_trials 50 \
  --storage sqlite:///outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned/study.db \
  --study_name initial_proposal_ae_lgbm_ld128 \
  --output-dir outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned \
  --skip-global-comparison-update

python src/build_initial_proposal_comparison.py \
  --baseline-default-dir outputs/initial_proposal/baseline_lgbm_default \
  --baseline-tuned-dir outputs/initial_proposal/optuna/baseline_lgbm_tuned \
  --ae-lgbm-default-dir outputs/initial_proposal/ae_lgbm_ld32_default \
  --ae-lgbm-ld128-tuned-dir outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned \
  --output-dir outputs/initial_proposal/final_comparison
```

Use `--skip-global-comparison-update` so Optuna reruns do not overwrite the broader `optuna_comparison.csv` table used by later exploratory branches.

To read legacy artifacts instead, omit the directory overrides and rely on script defaults:

```bash
python src/build_initial_proposal_comparison.py
```

## Expected output folders

| Step | Output directory | Key artifacts |
|------|------------------|---------------|
| Split check | `outputs/split_summary.json` | Chronological split summary |
| BASE-01 | `outputs/initial_proposal/baseline_lgbm_default/` | `metrics_*_selected_threshold.json`, `run_config.json` |
| AE LD32 | `outputs/initial_proposal/autoencoder_robust_ld32/` | `latent_*.npy`, `latent_split_manifest.csv`, `latent_split_manifest_summary.json` |
| AE-01 | `outputs/initial_proposal/ae_lgbm_ld32_default/` | `metrics_*_selected_threshold.json`, `model.pkl` |
| AE LD128 | `outputs/initial_proposal/autoencoder_robust_ld128/` | Same manifest + latent arrays for LD128 |
| BASE-02 | `outputs/initial_proposal/optuna/baseline_lgbm_tuned/` | `best_params.json`, `final_model.pkl`, metrics JSON |
| AE-02 | `outputs/initial_proposal/optuna/ae_lgbm_ld128_tuned/` | Same tuned artifacts |
| Comparison | `outputs/initial_proposal/final_comparison/` | `initial_proposal_comparison.csv`, `initial_proposal_missing_artifacts.json` |

## Why latent TransactionID alignment is validated

Autoencoder latent vectors are saved as NumPy arrays without embedded row keys. Row `i` in `latent_train.npy` must correspond to row `i` in the chronological train split. If split ordering changes (for example, sorting only by `TransactionDT` without a stable `TransactionID` tie-breaker), latent features can be joined to the wrong transactions without changing array shapes.

`train_autoencoder_robust.py` now writes `latent_split_manifest.csv` beside the latent arrays. `train_ae_lgbm.py` and the `ae_lgbm_ld128` path in `tune_lgbm_optuna.py` load that manifest and fail fast when `TransactionID` order does not match the current split.

## Why `initial_proposal_comparison.csv` is separate from `optuna_comparison.csv`

- **`initial_proposal_comparison.csv`** — four rows only: BASE-01, BASE-02, AE-01, AE-02. Used for the original proposal narrative and defense of the core chronological baseline vs AE replacement path.
- **`optuna_comparison.csv`** — broader Optuna sweep table that also includes feature-engineered, augmented, and other out-of-scope branches updated by `tune_lgbm_optuna.py` unless `--skip-global-comparison-update` is set.

Keeping them separate prevents initial-proposal reruns from mixing in FE, ensemble, or fusion experiments and preserves governance traceability.

## Lightweight validation (no full dataset)

```bash
python src/_validate_initial_proposal_pipeline_guards.py
python -m pytest tests/test_initial_proposal_pipeline_guards.py -q
```