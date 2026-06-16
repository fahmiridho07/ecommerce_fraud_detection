# Repository Delete Candidates

**Audit date:** 2026-06-11  
**Result:** No git-tracked file meets all DELETE_CANDIDATE conditions.

Per policy, a file is DELETE_CANDIDATE only if **all** of the following hold:

- not imported by any active script
- not referenced by `docs/`
- not referenced by `results/`
- not referenced by experiment registry
- not referenced by artifact manifest
- not needed to reproduce any listed metric
- not useful as ablation/diagnostic/legacy evidence
- duplicate/temporary/debug with no unique logic

---

## Tracked files reviewed and retained

| Path | Why not DELETE_CANDIDATE |
|------|-------------------------|
| `src/compare_results.py` | Referenced by `README.md`; early comparison utility |
| `src/compare_extended_optuna.py` | EX11 in `docs/EXPERIMENT_REGISTRY.md` |
| `src/compare_next_experiments.py` | Exploratory CSV builder; registry + `outputs/final_comparison/` lineage |
| `src/_feature_audit_phase1.py` | AE-04 feature audit; `docs/SELECTED_NUMERICAL_AE_FEATURE_AUDIT.md` |
| `src/generate_final_report_assets.py` | EX12; `docs/FINAL_REPORT_GOVERNANCE_NOTE.md` |
| `src/train_autoencoder.py` | AE16 superseded; registry + scientific traceability |
| All FE/ensemble trainers | LEGACY-03 / EX branches; governance documentation |

---

## Untracked local-only item

| Path | Reason | References | Risk | Action | Rollback |
|------|--------|------------|------|--------|----------|
| `terminals/` | IDE session captures; not in git index | none in repo | **low** (local only) | Optional local delete by developer; **do not** add to git | N/A — never committed |
| `src/__pycache__/`, `tests/__pycache__/` | Python bytecode caches; ignored runtime byproduct | none as scientific artifact | **low** (local only) | Optional local delete; regenerated automatically | N/A |

This directory is **not** recommended for repository deletion workflow because it is untracked and outside the scientific artifact set.

Active untracked GBDT files (`docs/GBDT_BASELINE_COMPARISON_PLAN.md`, `src/*gbdt*`) are **not** delete candidates; they are active WIP and should be reviewed/committed or explicitly parked as a separate decision.

---

## Files explicitly protected from deletion

All scripts listed in [`docs/REPOSITORY_CLEANUP_AUDIT.md`](REPOSITORY_CLEANUP_AUDIT.md) under ACTIVE_*, THESIS_CANDIDATE, ABLATION_EVIDENCE, DIAGNOSTIC_*, and LEGACY_ARCHIVE_CANDIDATE.

All `results/*.csv` and `results/*.json` files.

All `docs/*.md` experiment and governance documents.

---

## If delete candidates appear in future audits

Required checklist before any deletion:

1. `rg <filename> src/ docs/ results/ README.md`
2. Check `docs/EXPERIMENT_REGISTRY.md` master table
3. Check `docs/RESULT_ARTIFACT_MANIFEST.md`
4. Check `docs/EXPERIMENT_NAMING_GUIDE.md` and active/ablation maps
5. Confirm no import in `late_fusion_experts.py` dependency chain
6. Delete only in separate commit with rollback tag

**Rollback plan template:** `git checkout <commit> -- <path>` or restore from commit hash documented in cleanup PR.

---

## Conclusion

**No safe delete candidates found** among tracked thesis artifacts. Default action: **keep all tracked files**; improve navigation via documentation (Strategy A).
