# Active Experiment Map

## Purpose

This document lists **only** the final active thesis path. All experiments below remain in the repository under their legacy output directories and scripts.

**Legacy IDs are preserved for reproducibility** because output directories, scripts, and historical records use them. **Canonical IDs are used for thesis writing and final reporting.**

See [`docs/EXPERIMENT_NAMING_GUIDE.md`](EXPERIMENT_NAMING_GUIDE.md) for the full canonical registry.

## Active path overview

```text
BASE-01 / P01 ──► BASE-02 / P02
                        │
BEH-01 / CBA01R ◄──────┼──────► AE-02 / P04
         │              │              │
         └──────────────┴──────────────┘
                        │
                 FUS-01 / LF01  (thesis candidate)
```

---

## BASE-01 / P01 — Raw LightGBM Default

| Field | Value |
|-------|-------|
| **Purpose** | Establish the chronological original-feature LightGBM reference under default hyperparameters |
| **Why it remains active** | All integration and behavioral comparisons are anchored to this baseline (B1 / Rule A reference) |
| **Exact script** | `src/train_baseline_lgbm.py` |
| **Exact output path** | `outputs/baseline_lgbm/` |
| **Result summary** | Validation AP **0.602433**, Test AP **0.485756** — primary chronological reference for 432 original features |

---

## BASE-02 / P02 — Tuned Raw LightGBM

| Field | Value |
|-------|-------|
| **Purpose** | Measure the best achievable performance on original features under the frozen Optuna tuning budget |
| **Why it remains active** | Strongest tuned original-feature benchmark; fusion must beat this to claim practical improvement |
| **Exact script** | `src/tune_lgbm_optuna.py --model_type baseline_lgbm` |
| **Exact output path** | `outputs/optuna/baseline_lgbm/` |
| **Result summary** | Validation AP **0.624072**, Test AP **0.501438** — outperforms AE-02 / P04 on both splits |

---

## AE-02 / P04 — Tuned V-only AE-LightGBM LD128

| Field | Value |
|-------|-------|
| **Purpose** | Provide a frozen V-only AE representation expert for decision-level fusion |
| **Why it remains active** | Best available tuned AE replacement candidate; complementary ranking signal for FUS-01 / LF01 |
| **Exact script** | `src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128` (AE trained via `src/train_autoencoder_robust.py --latent-dim 128`) |
| **Exact output path** | `outputs/optuna/ae_lgbm_ld128/`, `outputs/autoencoder_robust_ld128/` |
| **Result summary** | Validation AP **0.610631**, Test AP **0.490686** — below BASE-02 alone, but contributes complementary fraud captures when fused |

---

## BEH-01 / CBA01R — Identity-Aligned Behavioral LightGBM

| Field | Value |
|-------|-------|
| **Purpose** | Test whether leakage-safe causal behavioral features improve chronological LightGBM over BASE-01 |
| **Why it remains active** | Authoritative corrected B2 result; primary behavioral expert in FUS-01 / LF01 |
| **Exact script** | `src/causal_behavioral_features.py`, `src/train_causal_behavioral_lgbm.py --id-aligned` |
| **Exact output path** | `outputs/causal_behavioral_lgbm_id_aligned/` |
| **Result summary** | Validation AP **0.615122** (+0.012689 vs BASE-01), Test AP **0.493838** — Rule A positive; supersedes LEGACY-01 / CBA01 |

---

## FUS-01 / LF01 — Behavioral + AE-LightGBM Late Fusion

| Field | Value |
|-------|-------|
| **Purpose** | Combine frozen BEH-01 and AE-02 scores at decision level without altering either feature space |
| **Why it remains active** | Meets predefined practical success criterion; conditional thesis-candidate pending supervisor approval |
| **Exact script** | `src/run_causal_behavioral_ae_late_fusion.py`, `src/audit_causal_behavioral_ae_complementarity.py` |
| **Exact output path** | `outputs/causal_behavioral_ae_late_fusion/` |
| **Result summary** | Validation AP **0.629600** (+0.014478 vs BEH-01, +0.005528 vs BASE-02), Test AP **0.505543** (descriptive); selected weights 50/50; category **strong success** |

## Governance note

FUS-01 / LF01 was executed as an explicit post-TAE01 freeze exception. Supervisor approval is required before promoting it to the primary thesis model. BASE-01 through AE-02 primary roles remain unchanged until that decision.

## Warning: high-AP branches excluded from active path

**AE-06 / AE15** (validation AP 0.635954) and other FE-space branches can exceed FUS-01 / LF01 (0.629600) on validation AP. They remain **outside** the active thesis path because:

| Branch | Validation AP | Why excluded despite high AP |
|--------|---------------|------------------------------|
| AE-06 / AE15 | 0.635954 | Static FE + one CDV recon error; non-causal aggregations; exploratory `behavioral_cdv_ae_experiment` Arm A; `stopping_criteria` references FE+AE ensemble scores; not comparable to governed primary experts |
| EX02 / tuned FE | 0.654316 | FE-space; answers different research question than AE-on-original-features |
| EX08 / FE+AE ensemble | 0.659935 | Post-hoc FE-space score ensemble; test-inspection risk per `docs/FINAL_REPORT_GOVERNANCE_NOTE.md` |

**High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

The governed BEH-02 / CBA02R result confirms that CDV reconstruction error does not help on the primary causal-behavioral path (−0.014515 vs BEH-01), even though the same CDV AE artifacts exist.

## Related documents

- [`docs/CAUSAL_BEHAVIORAL_AE_LATE_FUSION_EXPERIMENT.md`](CAUSAL_BEHAVIORAL_AE_LATE_FUSION_EXPERIMENT.md)
- [`docs/ABLATION_EXPERIMENT_MAP.md`](ABLATION_EXPERIMENT_MAP.md)
- [`docs/EXPERIMENT_SCOPE_FREEZE.md`](EXPERIMENT_SCOPE_FREEZE.md)