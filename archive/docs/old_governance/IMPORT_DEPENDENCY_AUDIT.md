# Import Dependency Audit

Flat `src/` layout uses same-directory imports (`from config import …`). **Physical refactor is not safe** without updating all import paths and documentation.

---

## Active script dependency chains

### `src/train_baseline_lgbm.py` (BASE-01 / P01)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `data_loader`, `evaluation`, `preprocessing`, `splitting`, `utils` |
| Reads | `data/raw/` train files |
| Writes | `outputs/baseline_lgbm/` |
| Move safe? | **No** |

### `src/tune_lgbm_optuna.py` (BASE-02 / P02, AE-02 / P04)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `data_loader`, `evaluation`, `feature_engineering`, `preprocessing`, `splitting`, `train_ae_augmented_lgbm`, `train_ae_lgbm`, `train_baseline_lgbm`, `utils` |
| Reads | `data/raw/`; optional AE latent dirs for `ae_lgbm_ld128` |
| Writes | `outputs/optuna/baseline_lgbm/`, `outputs/optuna/ae_lgbm_ld128/` |
| Move safe? | **No** — also pulls legacy `train_ae_augmented_lgbm` |

### `src/train_autoencoder_robust.py` (AE-02 / P04 AE training)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `data_loader`, `preprocessing`, `splitting`, `utils` |
| Reads | `data/raw/` V-features |
| Writes | `outputs/autoencoder_robust_ld128/` (and LD32/64 variants) |
| Move safe? | **No** |

### `src/train_ae_lgbm.py` (AE-02 / P04 LGBM; AE-01 / P03)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `data_loader`, `evaluation`, `preprocessing`, `splitting`, `utils` |
| Reads | `data/raw/`; `outputs/autoencoder_robust*/` latent `.npy` |
| Writes | `outputs/ae_lgbm/`, `outputs/ae_lgbm_ld128/`, etc. |
| Imported by | `late_fusion_experts.py`, `tune_lgbm_optuna.py`, ensemble scripts |
| Move safe? | **No** — fusion depends on exported functions |

### `src/train_causal_behavioral_lgbm.py` (BEH-01 / CBA01R)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `autoencoder_helpers`, `causal_behavioral_features`, `config`, `data_loader`, `evaluation`, `preprocessing`, `splitting`, `train_baseline_lgbm`, `utils` |
| Reads | `data/raw/` |
| Writes | `outputs/causal_behavioral_lgbm_id_aligned/` (`--id-aligned`) |
| Imported by | `late_fusion_experts.py` (`prepare_causal_behavioral_splits`) |
| Move safe? | **No** |

### `src/causal_behavioral_features.py` (BEH-01 / CBA01R)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `splitting` (indirect) |
| Imported by | `train_causal_behavioral_lgbm.py`, `late_fusion_experts.py`, `audit_causal_behavioral_row_alignment.py`, tests |
| Move safe? | **No** |

### `src/late_fusion_experts.py` (FUS-01 / LF01)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `causal_behavioral_features`, `config`, `preprocessing`, `train_ae_lgbm`, `train_causal_behavioral_lgbm` |
| Reads | `outputs/causal_behavioral_lgbm_id_aligned/` (`model.pkl`, `preprocessing.pkl`); `outputs/optuna/ae_lgbm_ld128/`; `outputs/autoencoder_robust_ld128/` latents |
| Writes | In-memory scores passed to fusion runner |
| Move safe? | **No** — central fusion dependency hub |

### `src/run_causal_behavioral_ae_late_fusion.py` (FUS-01 / LF01)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `audit_causal_behavioral_ae_complementarity`, `config`, `evaluation`, `late_fusion_experts`, `train_baseline_lgbm`, `utils` |
| Reads | Expert artifacts via `late_fusion_experts` |
| Writes | `outputs/causal_behavioral_ae_late_fusion/` |
| Move safe? | **No** |

### `src/build_causal_behavioral_ae_late_fusion_comparison.py` (FUS-01 / LF01)

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `utils` |
| Reads | Metric JSON from multiple `outputs/` dirs |
| Writes | `outputs/final_comparison/causal_behavioral_ae_late_fusion_comparison.csv`, `results/causal_behavioral_ae_late_fusion.csv` |
| Move safe? | **No** |

### `src/_post_execution_validation_causal_behavioral_alignment.py`

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | validation helpers, `config` |
| Reads | `outputs/causal_behavioral_lgbm_id_aligned/`, alignment audit outputs |
| Writes | `results/causal_behavioral_alignment_correction.csv` |
| Move safe? | **No** |

### `src/_post_execution_validation_late_fusion.py`

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config` |
| Reads | `outputs/causal_behavioral_ae_late_fusion/`, `results/` |
| Writes | validation stdout / optional checks |
| Move safe? | **No** |

---

## Active WIP dependency chain

### `src/gbdt_backends.py` and GBDT comparison scripts

| Dependency type | Files / paths |
|-----------------|---------------|
| Local imports | `config`, `preprocessing`, `tune_lgbm_optuna`; GBDT runners also import `data_loader`, `evaluation`, `splitting`, `train_ae_integration_strategy_ablation`, `train_gbdt_ae3_integration`, `utils` |
| Reads | `data/raw/`; optional `outputs/ae_integration_strategy_ablation_ld128/autoencoder_robust_ld128` for AE3 |
| Writes | `outputs/gbdt_baseline_comparison/` |
| Move safe? | **No** — WIP is internally consistent but still uses flat same-directory imports |

This WIP branch is isolated from FUS-01 / LF01. Do not move or package it before the GBDT comparison table and decision gate are complete.

---

## Shared core modules (all active paths)

```text
config.py ──┬── data_loader.py
            ├── splitting.py
            ├── preprocessing.py
            ├── evaluation.py
            └── utils.py
```

`feature_engineering.py` — required by `tune_lgbm_optuna.py` only among active path (FE tuning arms).

`autoencoder_helpers.py` — required by causal behavioral CDV path and AE15 legacy path.

---

## Import graph summary (active fusion path)

```text
run_causal_behavioral_ae_late_fusion.py
  └── late_fusion_experts.py
        ├── train_causal_behavioral_lgbm.py
        │     ├── causal_behavioral_features.py
        │     └── train_baseline_lgbm.py → [core stack]
        └── train_ae_lgbm.py → [core stack]
  └── audit_causal_behavioral_ae_complementarity.py
```

---

## Refactor safety assessment

| Refactor type | Safe now? | Blocker |
|---------------|-----------|---------|
| Move single legacy script to `src/legacy/` | **No** | Breaks `from X import` unless package refactor |
| Move active fusion scripts | **No** | `late_fusion_experts` cross-imports trainers |
| Add `src/experiments/` package | **No** | Requires `__init__.py`, relative imports, CI update |
| Documentation-only cleanup | **Yes** | No import changes |

**Recommendation:** Strategy A from [`docs/REPOSITORY_STRUCTURE_RECOMMENDATION.md`](REPOSITORY_STRUCTURE_RECOMMENDATION.md).

---

## Related documents

- [`docs/REPOSITORY_CLEANUP_AUDIT.md`](REPOSITORY_CLEANUP_AUDIT.md)
- [`docs/REPOSITORY_CLEANUP_PLAN.md`](REPOSITORY_CLEANUP_PLAN.md)
