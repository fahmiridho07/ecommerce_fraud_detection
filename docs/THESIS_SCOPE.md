# Thesis Scope

This document is the source of truth for the cleaned repository.

## Active Research Scope

The thesis studies fraud detection on the IEEE-CIS Fraud Detection dataset using:

- LightGBM as the baseline supervised model.
- Autoencoder representation learning on anonymized numerical `V*` features.
- AE-LightGBM, where original `V*` values are replaced by learned latent features while `V*` missingness indicators (`v_missing_*`) are preserved for the downstream classifier.
- Bayesian hyperparameter optimization with Optuna.

The active comparison is P01–P04 only:

| ID | Model |
|----|-------|
| P01 / BASE-01 | Baseline LightGBM default |
| P02 / BASE-02 | Baseline LightGBM Optuna tuned |
| P03 / AE-01 | AE-LightGBM default, LD32 latent replacement plus `V*` missing indicators |
| P04 / AE-02 | AE-LightGBM Optuna tuned, LD128 latent replacement plus `V*` missing indicators |

Canonical experiment artifacts: `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv`.

## Out Of Scope For Current Thesis Claims

The following branches are parked in `archive/` and must not be used as active thesis claims without a new written decision gate:

- Feature-engineered static FE branches, UID features, velocity features, and FE+AE combinations.
- Score ensembles and three-model ensembles.
- Behavioral, causal behavioral, CDV, and late-fusion branches.
- GBDT backend shootouts with XGBoost and CatBoost.
- Selected-numerical AE, task-aware AE, Ding-style reconstruction, and other AE appendix branches.
- Historical final report generators and notebook summaries based on broad exploratory rankings.

## Source-Of-Truth Order

When documents disagree, use this order:

1. `docs/THESIS_SCOPE.md`
2. `docs/EXPERIMENT_REGISTRY.md`
3. `docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`
4. `src/README.md`
5. `archive/README.md`

## Decision Rule (Post-Fix Rerun, 2026-06-16)

After rerunning P01–P04 with the missingness-preserving AE pipeline under `outputs/initial_proposal/`, the repository supports this thesis conclusion:

> Under the chronological IEEE-CIS split used in this project, Optuna-tuned LightGBM on the original features (P02) outperforms the AE-LightGBM latent replacement branch (P03 and P04), even after preserving `V*` missingness through `v_missing_*` indicators and training the Autoencoder with median imputation, masked reconstruction loss, and a linear latent layer.

Post-fix test PR-AUC: P02 **0.5049** > P01 0.4858 > P04 0.4845 > P03 0.4802.

## Open Items (Not Blockers For Scope)

These are documented limitations, not reasons to reopen archived branches:

- P03 (LD32 default) vs P04 (LD128 tuned) is a partially asymmetric comparison.
- Post-fix Optuna runs used 15 trials; 50 trials may be needed before defense if reviewers question tuning stability.
- Diagnostics: `src/generate_initial_proposal_diagnostics.py` → `outputs/initial_proposal/diagnostics/`.
- Representation ablation AE-03 (top-25 `V*` + LD32 latent, no tuning) nearly matches P01 test AP (0.4853 vs 0.4858). See `outputs/initial_proposal/representation_ablation/`.