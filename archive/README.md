# Archive Map

This archive parks experiments that are useful for traceability but outside the cleaned thesis proposal scope. Files are preserved instead of deleted so earlier work remains auditable.

## Folders

| Folder | Contents | Use in thesis |
|--------|----------|---------------|
| `source/ae_appendix/` | Selected-numerical AE, reconstruction variants, task-aware AE, AE strategy tuning, and related validators | Appendix or future work only |
| `source/behavioral_fusion/` | Causal behavioral features, CDV reconstruction, late fusion, and related validators | Out of current proposal scope |
| `source/feature_engineering_ensembles/` | Static feature engineering, UID/velocity features, FE+AE, and score ensembles | Out of current proposal scope |
| `source/gbdt_wip/` | LightGBM/XGBoost/CatBoost backend comparison WIP | WIP only |
| `source/methodology_reports/` | Split appendix, business impact diagnostics, final defense/report utilities | Reporting support only |
| `docs/ae_appendix/` | AE appendix experiment notes | Historical documentation |
| `docs/behavioral_fusion/` | Behavioral and fusion experiment notes | Historical documentation |
| `docs/gbdt_wip/` | GBDT comparison plan | WIP documentation |
| `docs/old_governance/` | Pre-cleanup maps, audits, freeze notes, and broad registries | Historical governance |
| `notebooks/` | Historical thesis report notebook | Historical report only |
| `results/` | Compact summaries from archived branches | Historical summaries only |

## Restore Rule

Do not move files back into `src/` casually. If an archived branch becomes active again, restore the whole branch as a deliberate decision, update `docs/THESIS_SCOPE.md`, and rerun the relevant validators.
