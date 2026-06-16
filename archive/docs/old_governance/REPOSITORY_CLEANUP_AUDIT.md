# Repository Cleanup Audit

**Audit date:** 2026-06-11; refreshed 2026-06-16  
**Scope:** Git-tracked files under `src/`, `docs/`, `results/`, `tests/`, and root project files, plus active untracked GBDT WIP files. Local `outputs/` inspected by path reference only (gitignored).  
**Thesis candidate:** FUS-01 / LF01 (BEH-01 / CBA01R + AE-02 / P04 late fusion).

> **High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

## Summary counts (src/ only)

| Classification | Count |
|----------------|------:|
| CORE_FINAL | 0 |
| CORE_SUPPORT | 8 |
| ACTIVE_REFERENCE | 2 |
| ACTIVE_EXPERT | 4 |
| THESIS_CANDIDATE | 4 |
| ABLATION_EVIDENCE | 13 |
| DIAGNOSTIC_AUDIT | 32 |
| DIAGNOSTIC_APPENDIX | 1 |
| LEGACY_ARCHIVE_CANDIDATE | 20 |
| ACTIVE_WIP_GBDT | 6 |
| PROVISIONAL_SUPERSEDED | 0* |
| DELETE_CANDIDATE | 0 |

\*Provisional superseded behavior is covered by dual-mode scripts (`train_causal_behavioral_lgbm.py`, `train_causal_behavioral_cdv_reconstruction_lgbm.py`) classified under ACTIVE_EXPERT / ABLATION with legacy notes—not standalone files.

**Total `src/*.py` audited:** 90

**2026-06-16 refresh note:** the original 76-script audit remains valid for the frozen FUS-01 thesis path. Fourteen later scripts are now classified without changing active thesis conclusions: AE strategy ablation/tuning support, initial-proposal rerun guards, and isolated GBDT WIP.

---

## Protected active path (canonical IDs)

| Canonical ID | Legacy ID | Key scripts |
|--------------|-----------|-------------|
| BASE-01 | P01 | `src/train_baseline_lgbm.py` |
| BASE-02 | P02 | `src/tune_lgbm_optuna.py` |
| AE-02 | P04 | `src/train_autoencoder_robust.py`, `src/train_ae_lgbm.py`, `src/tune_lgbm_optuna.py` |
| BEH-01 | CBA01R | `src/causal_behavioral_features.py`, `src/train_causal_behavioral_lgbm.py` |
| FUS-01 | LF01 | `src/late_fusion_experts.py`, `src/run_causal_behavioral_ae_late_fusion.py`, `src/audit_causal_behavioral_ae_complementarity.py` |

---

## Protected active WIP (not thesis-candidate evidence)

| WIP branch | Status | Key scripts | Output path | Cleanup rule |
|------------|--------|-------------|-------------|--------------|
| GBDT baseline comparison | Active WIP / isolated | `src/gbdt_backends.py`, `src/train_gbdt_baseline.py`, `src/tune_gbdt_baseline.py`, `src/train_gbdt_ae3_integration.py`, `src/build_gbdt_baseline_comparison.py`, `src/_validate_gbdt_baseline_pipeline.py` | `outputs/gbdt_baseline_comparison/` | Keep together; do not promote to active thesis path until comparison + decision gate are complete |

GBDT WIP currently does not alter BASE-01, BASE-02, AE-02, BEH-01, or FUS-01. Its artifacts remain under gitignored `outputs/`.

---

## 2026-06-16 additional script classifications

| Path | Classification | Short reason | Recommended action |
|------|----------------|--------------|-------------------|
| `src/train_ae_integration_strategy_ablation.py` | ABLATION_EVIDENCE | STR-B0..STR-AE3 AE integration strategy ablation | **keep** |
| `src/tune_ae_strategy_ablation.py` | ABLATION_EVIDENCE | TUNE-B0 / TUNE-AE3 strategy tuning | **keep** |
| `src/build_ae_strategy_ablation_comparison.py` | DIAGNOSTIC_AUDIT | Builds ablation summary tables | **keep** |
| `src/build_ae_strategy_tuned_comparison.py` | DIAGNOSTIC_AUDIT | Builds tuned strategy comparison | **keep** |
| `src/_validate_ae_strategy_ablation_pipeline.py` | DIAGNOSTIC_AUDIT | Static guards for AE strategy ablation | **keep** |
| `src/_validate_ae_strategy_tuning_pipeline.py` | DIAGNOSTIC_AUDIT | Static guards for AE strategy tuning | **keep** |
| `src/build_initial_proposal_comparison.py` | DIAGNOSTIC_AUDIT | Rebuilds BASE/AE initial proposal comparison | **keep** |
| `src/_validate_initial_proposal_pipeline_guards.py` | DIAGNOSTIC_AUDIT | Guards for proposal-only rerun path | **keep** |
| `src/gbdt_backends.py` | ACTIVE_WIP_GBDT | Shared LightGBM/XGBoost/CatBoost abstraction | **keep_wip** |
| `src/train_gbdt_baseline.py` | ACTIVE_WIP_GBDT | Fixed/default raw GBDT shootout runner | **keep_wip** |
| `src/tune_gbdt_baseline.py` | ACTIVE_WIP_GBDT | Optuna tuning for raw/AE3 GBDT variants | **keep_wip** |
| `src/train_gbdt_ae3_integration.py` | ACTIVE_WIP_GBDT | Conditional AE3 integration on winning backend | **keep_wip** |
| `src/build_gbdt_baseline_comparison.py` | ACTIVE_WIP_GBDT | GBDT comparison table and decision gate builder | **keep_wip** |
| `src/_validate_gbdt_baseline_pipeline.py` | ACTIVE_WIP_GBDT | Static validation for GBDT WIP pipeline | **keep_wip** |

---

## Phase 1 — `src/` script inventory

| Path | Classification | Canonical ID | Legacy ID | Short reason | Docs? | Results/manifest? | Active import? | Safe move? | Safe delete? | Recommended action |
|------|----------------|--------------|-----------|--------------|-------|-------------------|----------------|------------|--------------|-------------------|
| `src/config.py` | CORE_SUPPORT | — | — | Output paths, constants, `OUTPUT_PATHS` registry | yes | yes | yes | no | no | **keep** |
| `src/splitting.py` | CORE_SUPPORT | — | — | Chronological 60/20/20 split | yes | yes | yes | no | no | **keep** |
| `src/preprocessing.py` | CORE_SUPPORT | — | — | Baseline preprocessing for all arms | yes | yes | yes | no | no | **keep** |
| `src/evaluation.py` | CORE_SUPPORT | — | — | AP, MCC threshold selection | yes | yes | yes | no | no | **keep** |
| `src/utils.py` | CORE_SUPPORT | — | — | Logging, JSON I/O, seeds | yes | indirect | yes | no | no | **keep** |
| `src/data_loader.py` | CORE_SUPPORT | — | — | Labeled train data loading | yes | indirect | yes | no | no | **keep** |
| `src/feature_engineering.py` | CORE_SUPPORT | — | EX01 | FE mappings; imported by `tune_lgbm_optuna.py` | yes | yes | yes | no | no | **keep** |
| `src/autoencoder_helpers.py` | CORE_SUPPORT | — | — | CDV/recon helpers; causal behavioral CDV | yes | yes | yes | no | no | **keep** |
| `src/train_baseline_lgbm.py` | ACTIVE_REFERENCE | BASE-01 | P01 | Original-feature default baseline | yes | yes | yes | no | no | **keep** |
| `src/tune_lgbm_optuna.py` | ACTIVE_REFERENCE | BASE-02, AE-02 | P02, P04 | Tuned baseline and AE LD128 expert | yes | yes | yes | no | no | **keep** |
| `src/train_autoencoder_robust.py` | ACTIVE_EXPERT | AE-02, AE-01 | P04, P03 | Robust AE training (LD32/64/128) | yes | yes | yes | no | no | **keep** |
| `src/train_ae_lgbm.py` | ACTIVE_EXPERT | AE-02, AE-01 | P04, P03 | AE-LightGBM; P04 score regeneration in fusion | yes | yes | yes | no | no | **keep** |
| `src/causal_behavioral_features.py` | ACTIVE_EXPERT | BEH-01 | CBA01R | Identity-safe causal behavioral features | yes | yes | yes | no | no | **keep** |
| `src/train_causal_behavioral_lgbm.py` | ACTIVE_EXPERT | BEH-01 | CBA01R, CBA01 | B2 corrected (`--id-aligned`) and legacy B2 | yes | yes | yes | no | no | **keep** |
| `src/late_fusion_experts.py` | THESIS_CANDIDATE | FUS-01 | LF01 | Frozen expert score regeneration | yes | yes | yes | no | no | **keep** |
| `src/run_causal_behavioral_ae_late_fusion.py` | THESIS_CANDIDATE | FUS-01 | LF01 | Main fusion orchestrator | yes | yes | no | no | no | **keep** |
| `src/audit_causal_behavioral_ae_complementarity.py` | THESIS_CANDIDATE | FUS-01 | LF01 | Complementarity audit | yes | yes | yes | no | no | **keep** |
| `src/build_causal_behavioral_ae_late_fusion_comparison.py` | THESIS_CANDIDATE | FUS-01 | LF01 | Fusion comparison CSV builder | yes | yes | no | no | no | **keep** |
| `src/_prerun_validation_late_fusion.py` | DIAGNOSTIC_AUDIT | FUS-01 | LF01 | Pre-run fusion gates | yes | no | no | no | no | **keep** |
| `src/_post_execution_validation_late_fusion.py` | DIAGNOSTIC_AUDIT | FUS-01 | LF01 | Post-run fusion validation | yes | no | no | no | no | **keep** |
| `src/audit_causal_behavioral_row_alignment.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA01R | Alignment risk audit | yes | yes | no | no | no | **keep** |
| `src/_prerun_validation_causal_behavioral_alignment.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA01R | Pre-run alignment checks | yes | no | no | no | no | **keep** |
| `src/_post_execution_validation_causal_behavioral_alignment.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA01R | Post-run alignment validation | yes | yes | no | no | no | **keep** |
| `src/_validate_causal_behavioral_alignment_fix.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA01R | Alignment fix verification | yes | no | no | no | no | **keep** |
| `src/build_causal_behavioral_alignment_correction_comparison.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA01R | Corrected CBA comparison CSV | yes | yes | no | no | no | **keep** |
| `src/regenerate_cdv_reconstruction_errors_id_aligned.py` | DIAGNOSTIC_AUDIT | BEH-02 | CBA02R | ID-keyed CDV error regeneration | yes | yes | no | no | no | **keep** |
| `src/train_ae_latent_only_augmented_lgbm.py` | ABLATION_EVIDENCE | AE-03 | AE17 | Clean latent-only augmentation | yes | yes | no | no | no | **keep** |
| `src/train_autoencoder_selected_numerical.py` | ABLATION_EVIDENCE | AE-04 | AAE01 | Selected-numerical AE training | yes | yes | no | no | no | **keep** |
| `src/train_selected_numerical_ae_lgbm.py` | ABLATION_EVIDENCE | AE-04 | AAE01 | Selected-numerical latent replacement | yes | yes | no | no | no | **keep** |
| `src/generate_selected_numerical_reconstructed_features.py` | ABLATION_EVIDENCE | AE-05 | AAE02 | Decoder reconstruction features | yes | yes | no | no | no | **keep** |
| `src/train_selected_numerical_reconstructed_lgbm.py` | ABLATION_EVIDENCE | AE-05 | AAE02 | Reconstructed replacement LGBM | yes | yes | no | no | no | **keep** |
| `src/train_task_aware_autoencoder_selected_numerical.py` | ABLATION_EVIDENCE | AE-07 | TAE01 | Task-aware AE training | yes | yes | no | no | no | **keep** |
| `src/train_task_aware_ae_lgbm.py` | ABLATION_EVIDENCE | AE-07 | TAE01 | Task-aware downstream LGBM | yes | yes | no | no | no | **keep** |
| `src/train_causal_behavioral_cdv_reconstruction_lgbm.py` | ABLATION_EVIDENCE | BEH-02 | CBA02R, CBA02 | B3 corrected + legacy B3 | yes | yes | no | no | no | **keep** |
| `src/run_latent_dim_ablation.py` | ABLATION_EVIDENCE | AE-01 | P03 | LD32/64/128 replacement sweep | yes | yes | no | no | no | **keep** |
| `src/train_ae_augmented_lgbm.py` | ABLATION_EVIDENCE | — | AE06 | Confounded augmentation reference | yes | yes | yes† | no | no | **keep** |
| `src/train_reconstruction_error_lgbm.py` | ABLATION_EVIDENCE | — | AE08 | Reconstruction-error-only path | yes | yes | no | no | no | **keep** |
| `src/compare_split_strategy_appendix.py` | DIAGNOSTIC_APPENDIX | APP-01 | MD01–MD06 | Split-strategy sensitivity | yes | yes | no | no | no | **keep** |
| `src/build_autoencoder_input_scope_comparison.py` | DIAGNOSTIC_AUDIT | AE-04 | AAE01 | Input-scope comparison CSV | yes | yes | no | no | no | **keep** |
| `src/build_autoencoder_output_strategy_comparison.py` | DIAGNOSTIC_AUDIT | AE-05 | AAE02 | Output-strategy comparison CSV | yes | yes | no | no | no | **keep** |
| `src/build_task_aware_ae_comparison.py` | DIAGNOSTIC_AUDIT | AE-07 | TAE01 | Task-aware comparison CSV | yes | yes | no | no | no | **keep** |
| `src/build_causal_behavioral_ae_comparison.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA | Behavioral family comparison | yes | yes | no | no | no | **keep** |
| `src/_post_execution_validation.py` | DIAGNOSTIC_AUDIT | AE-04 | AAE01 | AAE01 post-run validation | yes | no | no | no | no | **keep** |
| `src/_post_execution_validation_reconstructed.py` | DIAGNOSTIC_AUDIT | AE-05 | AAE02 | AAE02 post-run validation | yes | no | no | no | no | **keep** |
| `src/_post_execution_validation_causal_behavioral.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA | CBA family post-run validation | yes | no | no | no | no | **keep** |
| `src/_post_execution_validation_task_aware.py` | DIAGNOSTIC_AUDIT | AE-07 | TAE01 | TAE01 post-run validation | yes | no | no | no | no | **keep** |
| `src/_prerun_validation_causal_behavioral.py` | DIAGNOSTIC_AUDIT | BEH-01 | CBA | CBA pre-run gates | yes | no | no | no | no | **keep** |
| `src/_prerun_validation_selected_numerical.py` | DIAGNOSTIC_AUDIT | AE-04 | AAE01 | AAE01 pre-run gates | yes | no | no | no | no | **keep** |
| `src/_prerun_validation_task_aware.py` | DIAGNOSTIC_AUDIT | AE-07 | TAE01 | TAE01 pre-run gates | yes | no | no | no | no | **keep** |
| `src/_feature_audit_phase1.py` | DIAGNOSTIC_AUDIT | AE-04 | AAE01 | Selected-numerical feature audit | yes | yes | no | no | no | **keep** |
| `src/audit_autoencoder.py` | DIAGNOSTIC_AUDIT | — | MD07 | AE audit diagnostics | yes | yes | no | no | no | **keep** |
| `src/generate_diagnostic_analysis.py` | DIAGNOSTIC_AUDIT | — | MD07 | AE audit outputs | yes | yes | no | no | no | **keep** |
| `src/generate_business_impact_diagnostics.py` | DIAGNOSTIC_AUDIT | — | MD08 | Bootstrap/business diagnostics | yes | yes | no | no | no | **keep** |
| `src/generate_final_defense_diagnostics.py` | DIAGNOSTIC_AUDIT | — | MD08 | Defense diagnostics | yes | yes | no | no | no | **keep** |
| `src/check_data_split.py` | DIAGNOSTIC_AUDIT | — | — | Split integrity utility | yes (README) | no | no | yes‡ | no | **keep** |
| `src/report_notebook_utils.py` | DIAGNOSTIC_AUDIT | — | — | Notebook reporting helpers | yes (notebook) | no | no | yes‡ | no | **keep** |
| `src/train_behavioral_cdv_autoencoder.py` | LEGACY_ARCHIVE_CANDIDATE | AE-06 | AE15 | CDV AE training for FE branch | yes | yes | no | no | no | **keep_legacy** |
| `src/train_fe_cdv_reconstruction_error_lgbm.py` | LEGACY_ARCHIVE_CANDIDATE | AE-06 | AE15 | FE + CDV recon error (static FE) | yes | yes | no | no | no | **keep_legacy** |
| `src/compare_behavioral_cdv_ae_experiment.py` | LEGACY_ARCHIVE_CANDIDATE | AE-06 | AE15 | Arm A comparison vs FE refs | yes | yes | no | no | no | **keep_legacy** |
| `src/run_score_ensemble.py` | LEGACY_ARCHIVE_CANDIDATE | LEGACY-03 | EX05 | Baseline+AE score ensemble | yes | yes | no | no | no | **keep_legacy** |
| `src/run_three_model_score_ensemble.py` | LEGACY_ARCHIVE_CANDIDATE | LEGACY-03 | EX06 | Three-model ensemble | yes | yes | no | no | no | **keep_legacy** |
| `src/run_fe_ae_fine_ensemble.py` | LEGACY_ARCHIVE_CANDIDATE | LEGACY-03 | EX07 | FE+AE fine ensemble | yes | yes | no | no | no | **keep_legacy** |
| `src/run_fe_ae_score_ensemble.py` | LEGACY_ARCHIVE_CANDIDATE | LEGACY-03 | EX08 | Controlled FE+AE ensemble | yes | yes | no | no | no | **keep_legacy** |
| `src/train_feature_engineered_lgbm.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX01 | Static FE baseline | yes | yes | yes† | no | no | **keep_legacy** |
| `src/train_uid_feature_engineered_lgbm.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX03 | UID FE branch | yes | yes | no | no | no | **keep_legacy** |
| `src/train_historical_velocity_lgbm.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX04 | Historical velocity FE | yes | yes | no | no | no | **keep_legacy** |
| `src/historical_velocity_features.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX04 | Velocity feature generator | yes | yes | yes† | no | no | **keep_legacy** |
| `src/train_fe_ae_augmented_lgbm.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX10 | FE latent+recon arm | yes | yes | no | no | no | **keep_legacy** |
| `src/train_fe_reconstruction_error_lgbm.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX09 | FE recon error arm | yes | yes | no | no | no | **keep_legacy** |
| `src/compare_fe_ae_controlled_experiments.py` | LEGACY_ARCHIVE_CANDIDATE | LEGACY-03 | EX08–10 | Controlled FE+AE comparison | yes | yes | no | no | no | **keep_legacy** |
| `src/compare_next_experiments.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX | Exploratory ranking CSV | yes | yes | no | no | no | **keep_legacy** |
| `src/generate_final_report_assets.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX12 | Historical test-ranked report | yes | yes | no | no | no | **keep_legacy** |
| `src/train_autoencoder.py` | LEGACY_ARCHIVE_CANDIDATE | — | AE16 | Superseded non-robust AE | yes | yes | no | no | no | **keep_legacy** |
| `src/train_autoencoder_normal_only.py` | LEGACY_ARCHIVE_CANDIDATE | — | AE11 | Normal-only AE path | yes | yes | no | no | no | **keep_legacy** |
| `src/compare_extended_optuna.py` | LEGACY_ARCHIVE_CANDIDATE | — | EX11 | Optuna FE smoke test | yes | yes | no | no | no | **keep_legacy** |
| `src/compare_results.py` | LEGACY_ARCHIVE_CANDIDATE | — | — | Early comparison utility; README only | yes (README) | no | no | yes‡ | no | **keep_legacy** |

† Imported by non-active legacy or tuning paths (`tune_lgbm_optuna` imports `train_ae_augmented_lgbm`; FE trainers import velocity).  
‡ Physically movable only after import-path refactor; not recommended now.

---

## Phase 1 — `docs/` inventory

| Path | Role | Protected? | Recommended action |
|------|------|------------|-------------------|
| `docs/EXPERIMENT_NAMING_GUIDE.md` | Canonical ID registry | **yes** | keep |
| `docs/ACTIVE_EXPERIMENT_MAP.md` | Active thesis path | **yes** | keep |
| `docs/ABLATION_EXPERIMENT_MAP.md` | Ablation/legacy map | **yes** | keep |
| `docs/EXPERIMENT_REGISTRY.md` | Full legacy inventory | **yes** | keep |
| `docs/EXPERIMENT_SCOPE_FREEZE.md` | Freeze policy | **yes** | keep |
| `docs/FINAL_EXPERIMENT_PLAN.md` | Reporting template | **yes** | keep |
| `docs/RESULT_ARTIFACT_MANIFEST.md` | Git-tracking guidance | **yes** | keep |
| `docs/CAUSAL_BEHAVIORAL_ALIGNMENT_CORRECTION.md` | Alignment correction | **yes** | keep |
| `docs/CAUSAL_BEHAVIORAL_AE_LATE_FUSION_EXPERIMENT.md` | FUS-01 experiment doc | **yes** | keep |
| `docs/CAUSAL_BEHAVIORAL_AE_EXPERIMENT.md` | Behavioral family | **yes** | keep |
| `docs/TASK_AWARE_AUTOENCODER_EXPERIMENT.md` | AE-07 / TAE01 | **yes** | keep |
| `docs/ANCHOR_ALIGNMENT_EXPERIMENT.md` | AE-04 / AAE01 | no | keep |
| `docs/DING_RECONSTRUCTION_ALIGNMENT_EXPERIMENT.md` | AE-05 / AAE02 | no | keep |
| `docs/CAUSAL_BEHAVIORAL_FEATURE_AUDIT.md` | Behavioral audit | no | keep |
| `docs/SELECTED_NUMERICAL_AE_FEATURE_AUDIT.md` | AE input audit | no | keep |
| `docs/FINAL_REPORT_GOVERNANCE_NOTE.md` | Test-ranking warning | no | keep |
| `docs/REPOSITORY_CLEANUP_AUDIT.md` | This audit | no | keep (new) |
| Other `docs/REPOSITORY_*.md` | Cleanup planning | no | keep (new) |

---

## Phase 1 — `results/` inventory (compact summaries)

| Path | Classification | Canonical link | Docs? | Manifest? | Action |
|------|----------------|----------------|-------|-----------|--------|
| `results/causal_behavioral_alignment_correction.csv` | THESIS_CANDIDATE support | BEH-01 | yes | yes | **keep** |
| `results/causal_behavioral_alignment_manifest.json` | THESIS_CANDIDATE support | BEH-01 | yes | yes | **keep** |
| `results/causal_behavioral_ae_late_fusion.csv` | THESIS_CANDIDATE | FUS-01 | yes | yes | **keep** |
| `results/causal_behavioral_ae_late_fusion_manifest.json` | THESIS_CANDIDATE | FUS-01 | yes | yes | **keep** |
| `results/late_fusion_complementarity_summary.json` | THESIS_CANDIDATE | FUS-01 | yes | yes | **keep** |

---

## Phase 1 — `tests/` inventory

| Path | Classification | Reason | Action |
|------|----------------|--------|--------|
| `tests/test_causal_behavioral_alignment.py` | DIAGNOSTIC_AUDIT | Alignment unit tests for BEH-01 | **keep** |
| `tests/test_initial_proposal_pipeline_guards.py` | DIAGNOSTIC_AUDIT | Guards proposal-only rerun path from LF01/fusion bleed-through | **keep** |
| `tests/test_ae_strategy_ablation_pipeline.py` | DIAGNOSTIC_AUDIT | Guards AE integration strategy ablation wiring | **keep** |

---

## Phase 1 — Root / other tracked files

| Path | Classification | Reason | Action |
|------|----------------|--------|--------|
| `README.md` | DIAGNOSTIC_AUDIT | Stale structure; update from proposal | **keep; update Level 1** |
| `requirements.txt` | CORE_SUPPORT | Dependencies | **keep** |
| `requirements-dev.txt` | CORE_SUPPORT | Test/dev dependencies layered on runtime requirements | **keep** |
| `.gitignore` | CORE_SUPPORT | Excludes `outputs/*`, allows `results/` summaries | **keep** |
| `notebooks/thesis_experiment_report.ipynb` | DIAGNOSTIC_AUDIT | Supervisor reporting notebook | **keep** |
| `src/README.md` | DIAGNOSTIC_AUDIT | Source script navigation without path movement | **keep** |

---

## Untracked local items (not in git index)

| Path | Note | DELETE_CANDIDATE? |
|------|------|-------------------|
| `terminals/` | IDE/session capture; not scientific evidence | Local-only optional cleanup; **not** in tracked audit delete list |
| `src/__pycache__/`, `tests/__pycache__/` | Python bytecode caches | Local-only optional cleanup; regenerated automatically |
| `docs/GBDT_BASELINE_COMPARISON_PLAN.md`, `src/*gbdt*` | Active WIP for backend comparison | **No**; keep WIP together or explicitly park in a later decision |

---

## Related cleanup documents

- [`docs/REPOSITORY_STRUCTURE_RECOMMENDATION.md`](REPOSITORY_STRUCTURE_RECOMMENDATION.md)
- [`docs/REPOSITORY_ARCHIVE_CANDIDATES.md`](REPOSITORY_ARCHIVE_CANDIDATES.md)
- [`docs/REPOSITORY_DELETE_CANDIDATES.md`](REPOSITORY_DELETE_CANDIDATES.md)
- [`docs/IMPORT_DEPENDENCY_AUDIT.md`](IMPORT_DEPENDENCY_AUDIT.md)
- [`docs/README_CLEANUP_PROPOSAL.md`](README_CLEANUP_PROPOSAL.md)
- [`docs/REPOSITORY_CLEANUP_PLAN.md`](REPOSITORY_CLEANUP_PLAN.md)
