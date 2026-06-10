# Selected Numerical Autoencoder Feature Audit

## Experiment purpose

Anchor studies (Ding et al.; Du et al.) train Autoencoders on **broader standardized numerical predictor sets**, not on anonymous V-features alone. The executed thesis pipeline previously limited Autoencoder input to V1–V339 and observed that V-only latent **replacement** did not improve chronological validation Average Precision versus original-feature LightGBM (P01).

This audit defines a **single controlled broadening** of Autoencoder input to defensible numerical predictors. The experiment asks whether input-scope alignment with anchor methods explains the prior negative V-only result, without opening additional AE variants, tuning searches, or augmentation branches.

## Eligibility rules

**Include (AE_ELIGIBLE_NUMERICAL)** when all of the following hold:

- Column is a predictor in the merged IEEE-CIS training dataframe (excluding `isFraud`, `TransactionID`).
- Semantic role is amount-like, count-like, duration-like, distance-like, anonymous numerical (V), or identity-numerical measuring a continuous quantity.
- `float64` / `int64` dtype with MSE reconstruction conceptually reasonable.
- Eligibility determined from column semantics, IEEE-CIS feature groups, cardinality, missingness, and repository baseline preprocessing — **not** from validation/test model performance.

**Exclude** when any of the following hold:

- `EXCLUDED_IDENTIFIER`: `TransactionID`, target, or row keys.
- `EXCLUDED_RAW_TIME`: `TransactionDT` (retained for downstream LightGBM to preserve P01 comparability).
- `EXCLUDED_CATEGORICAL`: object/string/category columns used as categorical in baseline LightGBM.
- `EXCLUDED_NUMERIC_CODED_CATEGORICAL`: numeric dtype columns representing issuer codes, address codes, device/OS/browser codes, sparse offset codes, or binary flags — not metric quantities.

Ambiguous columns default to **conservative exclusion** from Autoencoder input while remaining available to downstream LightGBM.

## Included feature groups

| Feature group | Exact columns | Count | dtype summary | Missing-rate range | Cardinality range | Semantic reason |
|---------------|---------------|------:|---------------|-------------------:|------------------:|-----------------|
| TransactionAmt | `TransactionAmt` | 1 | float64 | 0.000–0.000 | 326,108 | Transaction amount; continuous monetary quantity |
| C1–C14 | `C1`…`C14` | 14 | float64 | 0.000–0.000 | 12–172 | Count-like transaction features |
| D1–D15 | `D1`…`D15` | 15 | float64 | 0.002–0.934 | 2–637 | Duration/time-delta features |
| V1–V339 | `V1`…`V339` | 339 | float64 | 0.000–0.861 | 2–196,004 | Anonymous Vesta engineered numerical features |
| dist1–dist2 | `dist1`, `dist2` | 2 | float64 | 0.597–0.936 | 10–637 | Distance-like features |
| id_numerical_continuous | `id_01`, `id_02`, `id_03`, `id_04`, `id_06`, `id_13`, `id_14`, `id_17`, `id_18`, `id_19`, `id_20`, `id_21`, `id_22`, `id_24`, `id_25`, `id_26` | 16 | float64 | 0.756–0.992 | 2–3,580 | Identity-linked measurable numerical quantities (screen metrics, counts, continuous scores) |

## Excluded feature groups

| Columns / pattern | Reason | Retained for downstream LightGBM |
|-------------------|--------|----------------------------------|
| `TransactionID` | Row identifier | No (always dropped) |
| `isFraud` | Target | No |
| `TransactionDT` | Raw chronological time index; excluded from AE to avoid encoding split-defining time directly | **Yes** |
| `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `M1`–`M9`, `DeviceType`, `DeviceInfo`, `id_12`, `id_15`, `id_16`, `id_23`, `id_27`–`id_31`, `id_33`–`id_38` | Categorical strings / symbols | Yes |
| `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2` | Numeric-coded issuer/address identifiers | Yes |
| `id_05`, `id_07`, `id_08`, `id_09`, `id_10`, `id_11`, `id_32` | Numeric-coded categorical / sparse discrete codes | Yes |

## Ambiguous columns

| Column | Final conservative decision |
|--------|----------------------------|
| `id_05` | Exclude from AE (binary 0/1 anomaly flag); retain downstream |
| `id_07` | Exclude from AE (sparse discrete offset code, >99% missing); retain downstream |
| `id_08` | Exclude from AE (sparse discrete offset code, >99% missing); retain downstream |
| `id_09` | Exclude from AE (timezone code); retain downstream |
| `id_10` | Exclude from AE (OS version code); retain downstream |
| `id_11` | Exclude from AE (browser version code); retain downstream |
| `id_32` | Exclude from AE (device code); retain downstream |
| `card1`–`card3`, `card5` | Exclude from AE (issuer ID codes); retain downstream |
| `addr1`, `addr2` | Exclude from AE (address region codes); retain downstream |

## Final selected feature list

- **Total feature count:** 387
- **V-feature count:** 339
- **Non-V numerical feature count:** 48
- **Ordered list:** persisted in `outputs/selected_numerical_ae_feature_audit/selected_numerical_features.json` (`feature_names`)

Non-V additions: `TransactionAmt`, `C1`–`C14`, `D1`–`D15`, `dist1`, `dist2`, and 16 continuous `id_*` columns listed above.

## Leakage review

| Check | Status |
|-------|--------|
| Target (`isFraud`) excluded from AE input | Confirmed |
| Identifier (`TransactionID`) excluded from AE input | Confirmed |
| Raw `TransactionDT` excluded from AE input | Confirmed |
| No validation/test statistics used for eligibility thresholds | Confirmed |
| No label-derived feature selection | Confirmed |
| No test-based feature selection | Confirmed |
| All 432 raw predictors accounted for (387 AE + 45 non-AE) | Confirmed |

**Audit artifact:** `outputs/selected_numerical_ae_feature_audit/selected_numerical_features.json`