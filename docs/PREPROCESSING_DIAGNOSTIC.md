# Preprocessing Diagnostic

Generated after the AE-05 candidate run. This note focuses on data preparation rather than model architecture or hyperparameter search.

## Current Preprocessing Contract

- Split: chronological 60/20/20 by `TransactionDT`, with `TransactionID` only as deterministic tie-breaker.
- Baseline LightGBM:
  - drops `TransactionID`;
  - keeps numeric missing values as `NaN` for LightGBM native missing handling;
  - maps categorical strings using train-fitted integer mappings;
  - maps missing category values to `__MISSING__`;
  - maps unseen validation/test categories to `-1`.
- Autoencoder:
  - uses only `V1`-`V339`;
  - fits median imputer and scaler on train only;
  - uses masked reconstruction loss over observed `V*` cells;
  - saves `v_imputer.pkl` and `v_scaler.pkl`.
- AE-05:
  - keeps top-25 raw `V*` values;
  - uses LD32 latent features for lower-gain replaced `V*`;
  - appends global AE reconstruction-error features.

## Diagnostic Evidence

Artifacts: `outputs/initial_proposal/preprocessing_diagnostics/`.

Split composition:

| Split | Rows | Fraud rate |
|-------|------|------------|
| Train | 354,324 | 3.3833% |
| Validation | 118,108 | 3.9041% |
| Test | 118,108 | 3.4409% |

Feature inventory:

| Group | Count |
|-------|------:|
| Model features | 432 |
| Numeric features | 401 |
| Categorical object features | 31 |
| `V*` features | 339 |
| Non-`V*` features | 93 |

Missingness by family:

| Family | Train missing cells | Validation missing cells | Test missing cells |
|--------|--------------------:|-------------------------:|-------------------:|
| `V*` | 42.68% | 43.92% | 43.22% |
| `D*` | 61.76% | 59.02% | 58.15% |
| `M*` | 58.79% | 36.89% | 36.35% |
| identity `id_*` | 82.77% | 88.29% | 87.53% |
| email | 45.21% | 47.42% | 48.81% |
| distance | 77.35% | 75.94% | 75.23% |

Categorical unseen-rate issue:

| Feature | Validation unseen rate among observed | Test unseen rate among observed |
|---------|--------------------------------------:|--------------------------------:|
| `id_31` | 36.23% | 47.46% |

`id_31` is therefore the clearest categorical preprocessing weakness. It is mostly missing, but among non-missing rows a large share of browser/client strings in future periods are unseen by the train mapping.

Distribution shift:

- `TransactionDT` is intentionally shifted because the split is chronological.
- Top missingness drift includes `M7`, `M8`, `M9`, `V1`-`V11`, and `D11`.
- AE reconstruction error also drifts strongly from train/validation to test, so AE preprocessing must be evaluated under time shift, not just reconstruction loss.

## Diagnosis

The baseline preprocessing is strong for LightGBM because it avoids unnecessary numeric imputation and lets LightGBM learn missing-value routing. The previous AE failure was not mostly caused by baseline preprocessing; it came from forcing dense neural preprocessing onto highly missing `V*` columns and then replacing raw `V*` values that LightGBM used well.

AE-05 works because it fixes the representation policy rather than trying to make all preprocessing neural:

- raw high-gain `V*` values stay available;
- AE latent features summarize lower-gain `V*`;
- reconstruction error acts as an anomaly score;
- LightGBM still sees native missing values for retained raw features.

The remaining preprocessing bottleneck is likely outside raw `V*`: high-cardinality, temporally drifting categorical identity/device/browser fields and coarse temporal variables.

## Recommended Preprocessing Experiments

Priority 1 - Normalize high-cardinality identity/device categoricals:

- Parse `id_31` into browser family and browser version bucket.
- Parse `id_30` into OS family and version bucket.
- Parse `id_33` into screen width, height, and aspect bucket.
- Normalize `DeviceInfo` into brand/family tokens plus rare bucket.
- Add train-fitted rare-category bucketing before categorical mapping.

Rationale: current raw string mapping makes future browser/device variants appear as unseen categories. This is the strongest observed preprocessing gap.

Priority 2 - Add compact missingness summaries:

- row-level missing counts/rates by `V*`, `D*`, `M*`, `id_*`, email, and distance groups;
- optional high-missingness bins for `V*`;
- keep numeric `NaN` values unchanged for LightGBM.

Rationale: explicit per-column `v_missing_*` indicators had low gain, but group-level missingness can be more stable and lower-dimensional.

Priority 3 - Time preprocessing ablation:

- compare raw `TransactionDT` vs dropped `TransactionDT`;
- add hour/day/week style components from `TransactionDT`;
- test `D* - floor(TransactionDT / 86400)` style anchored deltas as a separate, clearly documented ablation.

Rationale: raw absolute time is predictive but may be period-specific. This should be measured, not assumed.

Priority 4 - AE-specific preprocessing:

- try all-train mask-aware LD32 AE, not only normal-only AE;
- compare StandardScaler vs RobustScaler/QuantileTransformer for `V*`;
- keep global reconstruction error as the main AE output feature unless an ablation proves latent-only is stronger.

Rationale: AE-05 shows reconstruction error is valuable, but the AE input representation may still be improved under temporal drift.

## Guardrails

- Do not median-impute all numeric features for LightGBM.
- Do not fit encoders, rare buckets, scalers, or imputers on validation/test.
- Do not silently replace the AE-05 result with broad feature-engineering branches; each preprocessing family should be ablated independently.
- Keep P02 tuned baseline and AE-05 as the main comparison pair for preprocessing experiments.

## Executed Ablation: Enhanced Identity/Device Preprocessing

Implemented in `src/enhanced_preprocessing.py` and `src/train_enhanced_preprocessing_lgbm.py`.

Changes:

- `id_31` -> browser family + major version, raw `id_31` dropped.
- `id_30` -> OS family + major version, raw `id_30` dropped.
- `id_33` -> screen width, height, area, aspect ratio, and size bucket, raw `id_33` dropped.
- `DeviceInfo` -> device family, raw `DeviceInfo` dropped.
- train-fitted rare-category bucketing with `rare_min_count=50`.
- numeric missing values remain `NaN` for LightGBM native missing handling.

Commands:

```bash
python src/train_enhanced_preprocessing_lgbm.py \
  --model-type baseline \
  --output-dir outputs/initial_proposal/preprocessing_ablation/baseline_enhanced_fixed_p02

python src/train_enhanced_preprocessing_lgbm.py \
  --model-type ae05 \
  --output-dir outputs/initial_proposal/preprocessing_ablation/ae05_enhanced_fixed_ae05
```

Comparison output: `outputs/initial_proposal/preprocessing_ablation/preprocessing_ablation_comparison.csv`.

Results:

| Model | Val AP | Test AP | ROC-AUC | F1 | MCC | Features |
|-------|-------:|--------:|--------:|---:|----:|---------:|
| P02 tuned baseline | 0.631767 | 0.504900 | 0.883431 | 0.493865 | 0.494270 | 432 |
| AE-05 hybrid reconstruction | 0.626124 | 0.509821 | 0.882011 | 0.504766 | 0.512071 | 466 |
| Enhanced baseline, fixed P02 params | **0.643247** | **0.516590** | **0.895311** | 0.503787 | 0.504735 | 438 |
| Enhanced AE-05, fixed AE-05 params | 0.631194 | 0.514975 | 0.889967 | **0.518194** | **0.514024** | 472 |

Interpretation:

- The preprocessing hypothesis is confirmed: identity/device normalization and rare bucketing improve both the tuned baseline and AE-05.
- Enhanced baseline has the best PR-AUC/ranking quality.
- Enhanced AE-05 has the best thresholded F1 and MCC, meaning the AE reconstruction signal still helps classification behavior after threshold selection.
- Because the enhanced baseline now exceeds enhanced AE-05 on PR-AUC, the next fair step is either tune both enhanced variants under the same Optuna budget or refine AE-specific preprocessing while keeping enhanced categorical preprocessing fixed.
