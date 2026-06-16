# Repository Cleanup Plan

Phased actions after canonical naming and cleanup audit. **No movement or deletion in this audit pass.**

---

## Level 1 — Safe now (documentation only)

| Action | File(s) | Reason | Risk | Rollback | Validation |
|--------|---------|--------|------|----------|------------|
| Keep experiment maps | `docs/EXPERIMENT_NAMING_GUIDE.md`, `ACTIVE_EXPERIMENT_MAP.md`, `ABLATION_EXPERIMENT_MAP.md` | Canonical navigation | none | N/A | Manual review |
| Add cleanup audit suite | `docs/REPOSITORY_*.md`, `IMPORT_DEPENDENCY_AUDIT.md`, `README_CLEANUP_PROPOSAL.md` | Hygiene planning | none | `git checkout` new docs | Files exist |
| Update README from proposal | `README.md` | Stale structure; missing FUS-01 path | low | `git checkout README.md` | Visual review |
| Add `src/README.md` index | `src/README.md` | Script discovery without moving files | low | delete file | Manual review |
| Add dev requirements | `requirements-dev.txt` | Make pytest/test setup explicit | low | delete file | `pip install -r requirements-dev.txt` in Python 3.10+ env |
| Refresh cleanup audit for 90 scripts | `docs/REPOSITORY_CLEANUP_AUDIT.md` | Current repo includes AE strategy, initial-proposal, and GBDT WIP scripts | low | revert doc edit | Compare `Get-ChildItem src -Filter *.py` count |
| Keep GBDT isolated as WIP | `docs/GBDT_BASELINE_COMPARISON_PLAN.md`, `src/*gbdt*` | Active WIP; not thesis-candidate evidence | low | revert WIP docs | No change to active thesis map |
| Cross-link docs | Add links from `EXPERIMENT_REGISTRY.md` to cleanup docs | Discoverability | none | revert link lines | grep links |
| Mark AE-06 legacy in all maps | Already in NAMING_GUIDE, ABLATION_MAP | Prevent high-AP misinterpretation | none | revert doc lines | Read AE-06 rows |
| Keep all `src/` paths | entire `src/` | Import stability | none | N/A | `py_compile` active scripts |
| Keep `results/` summaries | 5 tracked files | Thesis-facing metrics | none | N/A | `git ls-files results/` |
| Keep `.gitignore` outputs rule | `.gitignore` | Prevent model binary commits | none | N/A | `outputs/` untracked |

**Validation commands (Level 1):**

```bash
python -m compileall -q src tests
python -m pytest tests/test_causal_behavioral_alignment.py -q
python src/_validate_initial_proposal_pipeline_guards.py
python src/_validate_ae_strategy_ablation_pipeline.py
python src/_validate_gbdt_baseline_pipeline.py
```

Use a Python 3.10+ environment with project dependencies installed. `requirements-dev.txt` adds `pytest` on top of runtime requirements.

---

## Level 2 — Safe after review (optional archive movement)

**Prerequisite:** Supervisor approval + import refactor plan.

| Action | File(s) | Reason | Risk | Rollback | Validation |
|--------|---------|--------|------|----------|------------|
| Create `src/legacy/` package | FE/ensemble scripts (22 files) | Cleaner tree | **medium** | `git mv` reverse + import fix | Full import grep + py_compile all |
| Update imports to package paths | All moved scripts + callers | Required for move | **high** | Git revert commit | Run fusion validation scripts |
| Narrow `outputs/` git allow-list | `.gitignore` per MANIFEST | Track metric JSON only | medium | restore `.gitignore` | `git status outputs/` |

**Do not move** without completing [`docs/IMPORT_DEPENDENCY_AUDIT.md`](IMPORT_DEPENDENCY_AUDIT.md) checklist.

**Blocked movers (never first):**

- `late_fusion_experts.py`
- `train_ae_lgbm.py`
- `train_causal_behavioral_lgbm.py`
- `causal_behavioral_features.py`
- `config.py`

---

## Level 3 — Only if confirmed (delete low-risk junk)

| Action | File(s) | Reason | Risk | Rollback | Validation |
|--------|---------|--------|------|----------|------------|
| Local delete `terminals/` | `terminals/` (untracked) | IDE noise | low | N/A | `git status` clean |
| Local delete Python caches | `src/__pycache__/`, `tests/__pycache__/` | Ignored runtime cache, not scientific evidence | low | Recreated by Python automatically | `git status --ignored` no cache entry |
| **No tracked deletes** | — | No DELETE_CANDIDATE confirmed | — | — | See DELETE_CANDIDATES doc |

**Policy:** Do not delete any `src/*.py` until listed in `docs/REPOSITORY_DELETE_CANDIDATES.md` with risk=low and supervisor sign-off.

---

## Protected files — do not move/delete without explicit approval

### Canonical groups

- BASE-01 / P01, BASE-02 / P02, AE-02 / P04, BEH-01 / CBA01R, FUS-01 / LF01

### Core helpers

`src/config.py`, `src/splitting.py`, `src/preprocessing.py`, `src/evaluation.py`, `src/utils.py`, `src/data_loader.py`

### Active trainers and fusion

Listed in [`docs/REPOSITORY_CLEANUP_AUDIT.md`](REPOSITORY_CLEANUP_AUDIT.md) protected section.

### Docs (minimum set)

All files listed in Phase 2 of the cleanup audit prompt.

### Results

`results/causal_behavioral_alignment_correction.csv`, `results/causal_behavioral_alignment_manifest.json`, `results/causal_behavioral_ae_late_fusion.csv`, `results/causal_behavioral_ae_late_fusion_manifest.json`, `results/late_fusion_complementarity_summary.json`

---

## Recommended execution order

1. **Now:** Refresh documentation/index/dev setup; keep all code paths unchanged.
2. **Level 1:** Keep GBDT as isolated WIP and update audit/registry navigation.
3. **Level 3 local-only:** Remove ignored caches/session captures if desired.
4. **Stop** unless supervisor requests physical refactor.
5. **Level 2:** Only with explicit approval and rollback tag.

---

## Related documents

- [`docs/REPOSITORY_STRUCTURE_RECOMMENDATION.md`](REPOSITORY_STRUCTURE_RECOMMENDATION.md) — recommends Strategy A
- [`docs/REPOSITORY_DELETE_CANDIDATES.md`](REPOSITORY_DELETE_CANDIDATES.md) — none found
- [`docs/REPOSITORY_ARCHIVE_CANDIDATES.md`](REPOSITORY_ARCHIVE_CANDIDATES.md)
