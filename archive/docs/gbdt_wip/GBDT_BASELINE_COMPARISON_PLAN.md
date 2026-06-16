# GBDT Baseline Comparison Plan

## Purpose

Run a controlled shootout between **LightGBM**, **XGBoost**, and **CatBoost** on raw IEEE-CIS features (432 columns) using the same chronological split and evaluation protocol as the existing thesis experiments. If a non-LightGBM backend wins with a meaningful validation-AP margin, integrate **STR-AE3** (reconstruction-error LD128) on that winner only.

This path is isolated under `outputs/gbdt_baseline_comparison/` and does **not** modify FUS-01 / LF01 unless a later migration phase is explicitly approved.

**Current cleanup status (2026-06-16):** active WIP. `LGBM_fixed` exists locally as a raw LightGBM control, but the full shootout is not complete because XGBoost/CatBoost runs, tuned runs, `comparison.csv`, and `decision_gate.json` are not all present yet. Do not use this branch for thesis conclusions until the decision gate is complete and reviewed.

## References

| Artifact | Role |
|----------|------|
| [`docs/AE_STRATEGY_TUNING_RESULTS.md`](AE_STRATEGY_TUNING_RESULTS.md) | TUNE-B0 / TUNE-AE3 baselines for decision gate |
| [`src/train_baseline_lgbm.py`](../src/train_baseline_lgbm.py) | Fixed LightGBM parameter template |
| [`src/tune_lgbm_optuna.py`](../src/tune_lgbm_optuna.py) | Optuna `final` search space |
| [`src/train_ae_integration_strategy_ablation.py`](../src/train_ae_integration_strategy_ablation.py) | AE3 feature construction |

## Experiment IDs

### Phase 1 — Raw GBDT shootout (required)

| ID | Backend | Mode | Output subdir |
|----|---------|------|---------------|
| GBDT-LGBM-FIX | LightGBM | fixed/default | `LGBM_fixed` |
| GBDT-XGB-FIX | XGBoost | fixed/default | `XGB_fixed` |
| GBDT-CAT-FIX | CatBoost | fixed/default | `CAT_fixed` |
| GBDT-LGBM-TUNE | LightGBM | Optuna 50 (`final`) | `optuna/LGBM_tuned` |
| GBDT-XGB-TUNE | XGBoost | Optuna 50 (`final`) | `optuna/XGB_tuned` |
| GBDT-CAT-TUNE | CatBoost | Optuna 50 (`final`) | `optuna/CAT_tuned` |

### Phase 2 — AE3 on winner only (conditional)

| ID | Description | Output subdir |
|----|-------------|---------------|
| GBDT-WIN-AE3-FIX | AE3 fixed/default on winning backend | `AE3_fixed/{backend}` |
| GBDT-WIN-AE3-TUNE | AE3 + Optuna 50 on winning backend | `optuna/AE3_tuned/{backend}` |

AE artifact (frozen LD128):

`outputs/ae_integration_strategy_ablation_ld128/autoencoder_robust_ld128`

## Preprocessing

### Layer A (primary)

Library-native preprocessing via `--preprocessing-mode native` (default):

| Backend | Categorical handling | Numeric missing |
|---------|---------------------|-----------------|
| LightGBM | Integer encoding (`fit_baseline_preprocessing`) | NaN preserved |
| CatBoost | Native string categoricals | NaN preserved |
| XGBoost | Native `pd.Categorical` from train levels | NaN preserved |

### Layer B (optional sensitivity)

`--preprocessing-mode shared_lgbm` applies the existing LightGBM integer-encoding pipeline to all three backends. Use only for appendix fairness checks.

## Decision gate

Compare the **best tuned raw backend** (Phase 1) against existing **TUNE-B0**:

| Reference | Validation AP | Test AP |
|-----------|---------------|---------|
| TUNE-B0 | 0.6378 | 0.5060 |

**Gate passes** when the GBDT winner's validation AP exceeds TUNE-B0 by at least **0.003** and the test AP trend is consistent (not validation-only noise).

Phase 2 runs only if the gate passes. Compare AE3 results against **TUNE-AE3** (0.6290 val / 0.4994 test).

The comparison builder writes `decision_gate.json` automatically.

## Leakage prevention

- **Train:** preprocessing fit + model fitting
- **Validation:** early stopping, Optuna objective, threshold selection
- **Test:** final evaluation only

## Command order

Validate the pipeline first:

```powershell
python src/_validate_gbdt_baseline_pipeline.py
```

### Phase 1 — fixed/default (3 runs)

```powershell
python src/train_gbdt_baseline.py --backend lightgbm
python src/train_gbdt_baseline.py --backend xgboost
python src/train_gbdt_baseline.py --backend catboost
```

### Phase 1 — Optuna tuned (3 runs, ~4–5 h each on full data)

```powershell
python src/tune_gbdt_baseline.py --backend lightgbm --tuning-profile final --n-trials 50 --storage sqlite:///outputs/gbdt_baseline_comparison/optuna/lgbm.db
python src/tune_gbdt_baseline.py --backend xgboost --tuning-profile final --n-trials 50 --storage sqlite:///outputs/gbdt_baseline_comparison/optuna/xgb.db
python src/tune_gbdt_baseline.py --backend catboost --tuning-profile final --n-trials 50 --storage sqlite:///outputs/gbdt_baseline_comparison/optuna/cat.db
```

### Build comparison + decision gate

```powershell
python src/build_gbdt_baseline_comparison.py
```

Inspect `outputs/gbdt_baseline_comparison/decision_gate.json`. If `phase2_recommended` is true, note `winner_backend`.

### Phase 2 — AE3 on winner (example: catboost)

```powershell
python src/train_gbdt_ae3_integration.py --backend catboost
python src/tune_gbdt_baseline.py --backend catboost --feature-set ae3 --tuning-profile final --n-trials 50 --storage sqlite:///outputs/gbdt_baseline_comparison/optuna/ae3_cat.db
python src/build_gbdt_baseline_comparison.py --winner-backend catboost --include-phase2
```

## Out of scope (this phase)

- STR-AE1 / STR-AE2 on XGBoost / CatBoost
- Optuna across all AE strategies × all GBDT backends
- Rebuilding BEH-01 / causal behavioral expert on non-LightGBM
- LF01 / fusion migration (Phase 4, conditional)

## Scripts

| Script | Purpose |
|--------|---------|
| [`src/gbdt_backends.py`](../src/gbdt_backends.py) | Shared backend abstraction |
| [`src/train_gbdt_baseline.py`](../src/train_gbdt_baseline.py) | Fixed/default raw baseline |
| [`src/tune_gbdt_baseline.py`](../src/tune_gbdt_baseline.py) | Optuna tuning (`--feature-set raw\|ae3`) |
| [`src/train_gbdt_ae3_integration.py`](../src/train_gbdt_ae3_integration.py) | AE3 fixed/default |
| [`src/build_gbdt_baseline_comparison.py`](../src/build_gbdt_baseline_comparison.py) | `comparison.csv` + `decision_gate.json` |
| [`src/_validate_gbdt_baseline_pipeline.py`](../src/_validate_gbdt_baseline_pipeline.py) | Static validation |
