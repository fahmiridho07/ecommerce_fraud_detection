# Kaggle Script Index

Status: cleanup focus for thesis writing, 2026-06-21.

Use this folder as a Kaggle convenience layer only. The canonical source of
truth remains the local `src/` scripts and `docs/` registry.

## Active Thesis Script

Use this for the current locked thesis method:

```text
ieee_final_oversampling_kaggle.py
```

It implements:

- A1 dense preprocessing;
- baseline LightGBM;
- SMOTE oversampling control;
- AE latent-space oversampling;
- Optuna/TPE tuning;
- paired-bootstrap AP comparisons.

Output:

```text
/kaggle/working/final_oversampling_results.json
```

## Optional Diagnostic Only

This script is not the main thesis method:

```text
ieee_rankgauss_swapnoise_ladder_KAGGLE.py
```

Use it only if a final diagnostic is explicitly needed. It tests
RankGauss/swap-noise AE feature variants and should be reported as exploration
or future work unless it is promoted in `docs/THESIS_SCOPE.md`.

## Legacy / Exploration Scripts

Do not use these as the active Bab 4 headline unless the scope is deliberately
reopened:

- `ieee_ae_smote_lgbm_FINAL.py` - legacy feature-extractor + SMOTE framing.
- `ieee_ae_smote_additive_TWEAK.py`
- `ieee_ae_selective_compression_POV.py`
- `ieee_ae_importance_weighted_EXPLORATION.py`
- `ieee_ding_du_faithful_EXPLORATION.py`
- `ieee_ding_oneclass_grouperr_BESTSHOT.py`
- `ieee_entity_contextual_ae_ADVANCED.py`
- `ieee_lstmae_attention_POV.py`
- `ieee_remaining_levers_BATCH.py`
- `ieee_swapnoise_dae_POV.py`

These files are kept for traceability and argument-building, not for the final
method narrative.
