# Final Report Governance Note

## Purpose

This note documents a governance correction for historical report metadata under `outputs/final_report/`. It does **not** change any stored metric values.

## Historical artifact warning

`outputs/final_report/final_summary.json` (generated 2026-05-19) uses fields such as:

- `best_overall_model`
- `best_standalone_model`
- `best_baseline_model`
- `ranking_metric: "test_pr_auc"`

These fields rank models by **observed chronological test AP**. That ranking is **descriptive only** and is **not** a valid thesis model-selection rule.

At generation time, the highest test AP belonged to the FE+AE score ensemble (`fe_ae_tuned_score_ensemble`, test AP 0.533935). That model is an exploratory ensemble and is **outside** the frozen primary thesis comparison.

## Corrected interpretation policy

| Label | Meaning | Valid for model selection? |
|-------|---------|----------------------------|
| Highest observed test AP | Descriptive ordering of exploratory report table | No |
| Validation-selected model | Highest validation AP within frozen primary models (P01–P04) | Yes |
| Thesis-primary model | P02 baseline tuned under current verified evidence | Yes (until freeze approved) |

Among frozen primary models, **P02 (baseline LightGBM tuned)** has the highest validation AP (0.624072) and outperforms **P04 (AE replacement tuned LD128)** on both validation and test AP.

## Generator correction

`src/generate_final_report_assets.py` was updated to emit:

- `descriptive_test_rank` instead of selection-oriented `rank` labels
- `highest_observed_test_ap_model` instead of `best_overall_model` as the primary summary label
- `thesis_primary_validation_leader` for validation-based primary comparison
- `ranking_purpose: descriptive_test_ranking_not_model_selection`

Deprecated alias fields (`best_overall_model`, etc.) may still be written for backward compatibility when the script is rerun, but documentation and thesis text must not treat them as model-selection outputs.

## Action required before thesis writing

1. Do **not** cite `final_summary.json` as the authoritative model-selection artifact.
2. Use `docs/EXPERIMENT_SCOPE_FREEZE.md` and primary metric JSON files under `outputs/baseline_lgbm/`, `outputs/optuna/`, and `outputs/ae_lgbm/`.
3. Regenerate `outputs/final_report/` only after supervisor approval of the frozen scope, and only if needed for notebook formatting — not for choosing a new model.