# Repository Structure Recommendation

## Recommended strategy: **A — Documentation-based cleanup only**

Keep all current file paths unchanged. Use experiment maps, cleanup audits, and README guidance to navigate the repository. Do **not** physically refactor `src/` until a future governed migration is approved.

### Why not B (light archive) or C (full refactor) now

| Factor | Finding |
|--------|---------|
| Import coupling | `late_fusion_experts.py` imports `train_ae_lgbm`, `train_causal_behavioral_lgbm`, `causal_behavioral_features` |
| Optuna coupling | `tune_lgbm_optuna.py` imports `train_ae_augmented_lgbm`, `feature_engineering` |
| Flat `src/` convention | All 76 scripts use same-directory imports (`from config import …`) |
| Movement cost | Any folder move requires updating dozens of imports + docs + `run_config.json` lineage references |
| Risk | Physical refactor adds breakage risk without improving thesis reproducibility |

See [`docs/IMPORT_DEPENDENCY_AUDIT.md`](IMPORT_DEPENDENCY_AUDIT.md) for per-script dependency detail.

---

## Why this repository stays research-heavy

1. **Scientific traceability:** Every failed or superseded experiment (AE-04, BEH-02, AE-06/AE15) documents *why* the final method was chosen.
2. **Governance evidence:** Alignment audits, pre/post validations, and freeze documents prove chronological validation policy was followed.
3. **Thesis defense:** Examiners may ask about exploratory branches with higher test AP (EX07, AE-06). Deleting them weakens the argument.
4. **Corrected reruns:** LEGACY-01/02 provisional results remain to show alignment correction impact.

---

## Directory policy recommendations

### `outputs/` — remain gitignored

- Contains model binaries (`.pkl`, `.keras`), latent `.npy`, large CSVs.
- All primary metrics are reproducible from scripts + local artifacts.
- `docs/RESULT_ARTIFACT_MANIFEST.md` defines a future narrow allow-list for small JSON/CSV only.
- **Do not** track full `outputs/` trees in Git.

### `results/` — compact summaries only

Current tracked files (~5) are appropriate:

- CSV/JSON summaries for supervisor review
- Manifests with artifact lineage
- No transaction-level predictions, no model weights

**Policy:** Add new rows only when a governed experiment produces a thesis-facing summary worth versioning.

### `src/` — flat layout preserved

Proposed future layout (from `docs/FINAL_EXPERIMENT_PLAN.md`) remains **not executed**:

```text
src/experiments/   # primary trainers
src/diagnostics/   # audits, validations
src/legacy/        # FE, ensemble, AE15
```

Defer until import audit + migration plan approved.

### `docs/` — canonical navigation layer

Primary navigation for thesis writing:

1. `docs/EXPERIMENT_NAMING_GUIDE.md` — dual notation registry
2. `docs/ACTIVE_EXPERIMENT_MAP.md` — final path
3. `docs/ABLATION_EXPERIMENT_MAP.md` — supporting evidence
4. `docs/EXPERIMENT_REGISTRY.md` — full legacy inventory

---

## Strategy comparison

| Strategy | Effort | Risk | Thesis benefit |
|----------|--------|------|----------------|
| **A. Documentation only** | Low | Low | High — clarifies without breaking reproduction |
| B. Light archive movement | Medium | Medium | Medium — cleaner tree, import churn |
| C. Full physical refactor | High | High | Low until migration governance exists |

---

## AE-06 / AE15 governance note

AE-06 / AE15 (validation AP 0.635954) must remain in **legacy archived** documentation, not active path maps. High AP reflects static FE space, not governed final-method superiority.

> **High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

---

## Related documents

- [`docs/REPOSITORY_CLEANUP_AUDIT.md`](REPOSITORY_CLEANUP_AUDIT.md)
- [`docs/REPOSITORY_CLEANUP_PLAN.md`](REPOSITORY_CLEANUP_PLAN.md)