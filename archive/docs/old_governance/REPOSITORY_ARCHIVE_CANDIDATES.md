# Repository Archive Candidates

Non-active experiments grouped by recommended treatment. **No files moved or deleted** by this document.

> **High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

---

## 1. Ablation evidence — keep visible

| Canonical / Legacy | Script(s) | Output path(s) | Docs | Why not active | Insight | Treatment |
|--------------------|-----------|----------------|------|----------------|---------|-----------|
| AE-01 / P03 | `train_autoencoder_robust.py`, `train_ae_lgbm.py` | `outputs/ae_lgbm/`, `outputs/autoencoder_robust/` | REGISTRY, FINAL_PLAN | Superseded as fusion expert by AE-02 | Original AE replacement default; below BASE-01 | **keep_visible** |
| AE-03 / AE17 | `train_ae_latent_only_augmented_lgbm.py` | `outputs/ae_latent_only_augmented_lgbm_ld128/` | SCOPE_FREEZE | Latent augmentation failed | Fair augmentation test | **keep_visible** |
| AE-04 / AAE01 | `train_autoencoder_selected_numerical.py`, `train_selected_numerical_ae_lgbm.py` | `outputs/selected_numerical_ae_lgbm_ld128/` | ANCHOR_ALIGNMENT | Input-scope broadening failed | Rule C evidence | **keep_visible** |
| AE-05 / AAE02 | `generate_selected_numerical_reconstructed_features.py`, `train_selected_numerical_reconstructed_lgbm.py` | `outputs/selected_numerical_reconstructed_lgbm/` | DING_RECONSTRUCTION | Below BASE-01 | Rule B evidence | **keep_visible** |
| AE-07 / TAE01 | `train_task_aware_autoencoder_selected_numerical.py`, `train_task_aware_ae_lgbm.py` | `outputs/task_aware_ae_lgbm_ld128/selected/` | TASK_AWARE_AUTOENCODER | Supervised latent did not help | Closes AE search space | **keep_visible** |
| BEH-02 / CBA02R | `train_causal_behavioral_cdv_reconstruction_lgbm.py --id-aligned` | `outputs/causal_behavioral_cdv_reconstruction_lgbm_id_aligned/` | CAUSAL_BEHAVIORAL_AE | CDV recon degrades BEH-01 | Rule D; motivates fusion not feature injection | **keep_visible** |

Supporting comparison builders (keep with ablations): `build_autoencoder_input_scope_comparison.py`, `build_autoencoder_output_strategy_comparison.py`, `build_task_aware_ae_comparison.py`, `build_causal_behavioral_ae_comparison.py`, `run_latent_dim_ablation.py`, `train_reconstruction_error_lgbm.py`, `train_ae_augmented_lgbm.py`.

---

## 2. Diagnostic appendix — keep visible

| Canonical / Legacy | Script(s) | Output path(s) | Docs | Why not active | Insight | Treatment |
|--------------------|-----------|----------------|------|----------------|---------|-----------|
| APP-01 / MD01–MD06 | `compare_split_strategy_appendix.py` | `outputs/split_strategy_appendix/` | FINAL_PLAN, SCOPE_FREEZE | Protocol sensitivity only | Supports chronological choice | **keep_visible** |

Supporting diagnostics: `audit_autoencoder.py`, `generate_diagnostic_analysis.py`, `generate_business_impact_diagnostics.py`, `generate_final_defense_diagnostics.py` → **keep_visible** (MD07–MD08).

---

## 3. Legacy archived — keep for traceability

| Canonical / Legacy | Script(s) | Output path(s) | Docs | Why not active | Insight | Treatment |
|--------------------|-----------|----------------|------|----------------|---------|-----------|
| AE-06 / AE15 | `train_behavioral_cdv_autoencoder.py`, `train_fe_cdv_reconstruction_error_lgbm.py`, `compare_behavioral_cdv_ae_experiment.py` | `outputs/behavioral_cdv_ae_experiment/` | NAMING_GUIDE, ABLATION_MAP | Static FE branch; not comparable to FUS-01 | CDV AE artifacts; high AP from FE baseline | **keep_legacy** |
| LEGACY-01 / CBA01 | `train_causal_behavioral_lgbm.py` (default) | `outputs/causal_behavioral_lgbm_default/` | ALIGNMENT_CORRECTION | Superseded by CBA01R | Alignment risk demonstration | **keep_legacy** |
| LEGACY-02 / CBA02 | `train_causal_behavioral_cdv_reconstruction_lgbm.py` (default) | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` | ALIGNMENT_CORRECTION | Superseded by CBA02R | Provisional B3 record | **keep_legacy** |
| LEGACY-03 / EX05–EX08 | `run_score_ensemble.py`, `run_three_model_score_ensemble.py`, `run_fe_ae_fine_ensemble.py`, `run_fe_ae_score_ensemble.py` | `outputs/score_ensemble_*`, `outputs/fe_ae_*` | REGISTRY, GOVERNANCE_NOTE | Post-hoc FE ensembles; test-peeking risk | Hint for later governed fusion | **keep_legacy** |
| — / EX01 | `train_feature_engineered_lgbm.py` | `outputs/baseline_lgbm_entity_time_amount_features/` | REGISTRY | FE-space; not primary question | Supporting benchmark | **keep_legacy** |
| — / EX02–EX04 | `tune_lgbm_optuna.py`, `train_uid_feature_engineered_lgbm.py`, `train_historical_velocity_lgbm.py` | `outputs/optuna/baseline_lgbm_entity_*`, velocity outputs | REGISTRY | Parallel FE exploration | Higher AP branches excluded from freeze | **keep_legacy** |
| — / EX09–EX10 | `train_fe_reconstruction_error_lgbm.py`, `train_fe_ae_augmented_lgbm.py` | `outputs/fe_ae_controlled_experiments/` | REGISTRY | Controlled FE+AE arms | FE-space diagnostics | **keep_legacy** |
| — / EX11 | `compare_extended_optuna.py` | `outputs/_smoke_fe_extended/` | REGISTRY | Incomplete smoke test | Debug only | **keep_legacy** |
| — / EX12 | `generate_final_report_assets.py` | `outputs/final_report/` | GOVERNANCE_NOTE | Test-ranked historical report | Documents test-inspection bias | **keep_legacy** |
| — / AE16 | `train_autoencoder.py` | `outputs/autoencoder/` | REGISTRY | Superseded by robust AE | Historical AE training | **keep_legacy** |
| — / AE11–14 | `train_autoencoder_normal_only.py`, `train_reconstruction_error_lgbm.py` | various `outputs/baseline_lgbm_plus_*` | REGISTRY | Secondary recon paths | Recon-error family evidence | **keep_legacy** |
| — | `compare_next_experiments.py`, `compare_results.py` | `outputs/final_comparison/` (partial) | README (compare_results) | Early/exploratory utilities | Historical ranking CSVs | **keep_legacy** |

---

## 4. Possible delete candidates

**None confirmed** among git-tracked scientific files. See [`docs/REPOSITORY_DELETE_CANDIDATES.md`](REPOSITORY_DELETE_CANDIDATES.md).

Untracked `terminals/` may be removed locally by the developer; it is not part of the thesis artifact set.

---

## Related documents

- [`docs/EXPERIMENT_NAMING_GUIDE.md`](EXPERIMENT_NAMING_GUIDE.md)
- [`docs/ACTIVE_EXPERIMENT_MAP.md`](ACTIVE_EXPERIMENT_MAP.md)
- [`docs/ABLATION_EXPERIMENT_MAP.md`](ABLATION_EXPERIMENT_MAP.md)