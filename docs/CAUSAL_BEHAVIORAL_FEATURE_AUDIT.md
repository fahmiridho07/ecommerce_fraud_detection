# Causal Behavioral Feature Audit

## Purpose

This audit evaluates whether existing entity/time/amount and historical-velocity features in the repository are valid for **chronological, leakage-safe** fraud prediction. It inventories current feature families, classifies leakage risk, and defines the **SAFE_CAUSAL_FEATURES** set used by controlled experiments B2 and B3.

B2 requires original P01 predictors plus **past-only** behavioral aggregates. B3 adds exactly one CDV reconstruction-error feature on top of B2. Neither arm may reuse train-static full-period entity counts or label-derived statistics.

## Existing feature families

### `feature_engineering.py` — amount and time derivations

| Feature | Entity key | Time window | Statistic | Uses only prior rows? | Fit on train only? | Validation/test handling | Leakage status | Keep? |
|---------|------------|-------------|-----------|----------------------|--------------------|--------------------------|----------------|-------|
| TransactionAmt_log | — | — | log transform | N/A (row-local) | N/A | Same formula per row | Safe row-local | EX01 only |
| TransactionAmt_decimal | — | — | fractional part | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| TransactionAmt_cents | — | — | cents mod 100 | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| TransactionAmt_is_round | — | — | round flag | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| TransactionAmt_num_decimals | — | — | decimal count | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| transaction_day | — | — | DT / 86400 | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| transaction_hour | — | — | hour proxy | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| transaction_week | — | — | week proxy | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| transaction_dayofweek_proxy | — | — | day mod 7 | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| sin_hour | — | — | cyclic hour | N/A | N/A | Same formula per row | Safe row-local | EX01 only |
| cos_hour | — | — | cyclic hour | N/A | N/A | Same formula per row | Safe row-local | EX01 only |

### `feature_engineering.py` — count and frequency (17 entity groups)

| Feature pattern | Entity key | Time window | Statistic | Uses only prior rows? | Fit on train only? | Validation/test handling | Leakage status | Keep? |
|-----------------|------------|-------------|-----------|----------------------|--------------------|--------------------------|----------------|-------|
| count_{group} | card1, card2, card3, card5, addr1, addr2, P_emaildomain, R_emaildomain, DeviceInfo, ProductCD, card1_card2, card1_addr1, card1_P_emaildomain, card1_addr1_P_emaildomain, card1_DeviceInfo, ProductCD_card1, uid_card_addr | Full train period | train row count | **No** | Yes | Train counts mapped to val/test; unseen → 0 | **Train-static aggregate** | EX01 only — **exclude B2** |
| freq_{group} | Same as above | Full train period | count / train_rows | **No** | Yes | Train frequencies mapped to val/test | **Train-static aggregate** | EX01 only — **exclude B2** |

### `feature_engineering.py` — amount statistics (6 entity groups)

| Feature pattern | Entity key | Time window | Statistic | Uses only prior rows? | Fit on train only? | Validation/test handling | Leakage status | Keep? |
|-----------------|------------|-------------|-----------|----------------------|--------------------|--------------------------|----------------|-------|
| amt_mean_by_{group} | card1, card1_addr1, card1_P_emaildomain, P_emaildomain, ProductCD, addr1 | Full train period | mean(TransactionAmt) | **No** | Yes | Train stats mapped; fallback to global train mean | **Train-static aggregate** | EX01 only — **exclude B2** |
| amt_median_by_{group} | Same | Full train period | median | **No** | Yes | Train stats mapped | **Train-static aggregate** | EX01 only |
| amt_std_by_{group} | Same | Full train period | std | **No** | Yes | Train stats mapped | **Train-static aggregate** | EX01 only |
| amt_to_mean_by_{group} | Same | Full train period | ratio to train mean | **No** | Yes | Uses train-fitted mean | **Train-static aggregate** | EX01 only |
| amt_diff_mean_by_{group} | Same | Full train period | diff from train mean | **No** | Yes | Uses train-fitted mean | **Train-static aggregate** | EX01 only |
| amt_to_median_by_{group} | Same | Full train period | ratio to train median | **No** | Yes | Uses train-fitted median | **Train-static aggregate** | EX01 only |
| amt_zscore_by_{group} | Same | Full train period | z-score vs train | **No** | Yes | Uses train-fitted mean/std | **Train-static aggregate** | EX01 only |

### `feature_engineering.py` — UID branch nunique (7 relationship features)

| Feature | Entity key | Time window | Statistic | Uses only prior rows? | Fit on train only? | Validation/test handling | Leakage status | Keep? |
|---------|------------|-------------|-----------|----------------------|--------------------|--------------------------|----------------|-------|
| nunique_P_emaildomain_by_card1 | card1 | Full train | nunique(P_emaildomain) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |
| nunique_addr1_by_card1 | card1 | Full train | nunique(addr1) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |
| nunique_DeviceInfo_by_card1 | card1 | Full train | nunique(DeviceInfo) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |
| nunique_card1_by_DeviceInfo | DeviceInfo | Full train | nunique(card1) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |
| nunique_card1_by_P_emaildomain | P_emaildomain | Full train | nunique(card1) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |
| nunique_addr1_by_P_emaildomain | P_emaildomain | Full train | nunique(addr1) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |
| nunique_ProductCD_by_card1 | card1 | Full train | nunique(ProductCD) | **No** | Yes | Train mapping | **Train-static aggregate** | EX03 only |

### `historical_velocity_features.py` — causal historical family (5 entities)

| Feature pattern | Entity key | Time window | Statistic | Uses only prior rows? | Fit on train only? | Validation/test handling | Leakage status | Keep? |
|-----------------|------------|-------------|-----------|----------------------|--------------------|--------------------------|----------------|-------|
| hist_count_before_{entity} | card1, card1_addr1, card1_P_emaildomain, P_emaildomain, DeviceInfo | Cumulative prior | prior transaction count | **Yes** (prior TransactionDT groups) | N/A (online state) | State continues from train → val → test | **Safe historical** with caveat | **Review** — same-timestamp policy differs from B2 |
| hist_time_since_prev_{entity} | Same | Since last prior txn | time delta | **Yes** | N/A | State continuation | Safe historical (caveat: timestamp batching) | Review |
| hist_amt_prev_{entity} | Same | Last prior txn | amount | **Yes** | N/A | State continuation | Safe historical | Review |
| hist_amt_diff_prev_{entity} | Same | Last prior txn | amount diff | **Yes** | N/A | State continuation | Safe historical | Review |
| hist_amt_ratio_prev_{entity} | Same | Last prior txn | amount ratio | **Yes** | N/A | State continuation | Safe historical | Review |
| hist_amt_mean_before_{entity} | Same | Cumulative prior | expanding mean | **Yes** | N/A | State continuation | Safe historical | **Adapt for B2** |
| hist_amt_std_before_{entity} | Same | Cumulative prior | expanding std | **Yes** | N/A | State continuation | Safe historical | **Adapt for B2** |
| hist_count_last_1h_{entity} | card1, card1_P_emaildomain | 3600 s | rolling count | **Yes** | N/A | State continuation | Safe historical | **Adapt for B2** |
| hist_count_last_24h_{entity} | card1, card1_P_emaildomain | 86400 s | rolling count | **Yes** | N/A | State continuation | Safe historical | **Adapt for B2** |
| hist_count_last_7d_{entity} | card1, card1_P_emaildomain | 7 days | rolling count | **Yes** | N/A | State continuation | Safe historical | EX04 only — **exclude B2** (window not in B2 spec) |

**EX04 caveat:** `historical_velocity_features.py` batches same-`TransactionDT` rows so peers at identical timestamps do not see each other. B2 uses **TransactionID tie-breaking** within timestamps (stricter row causality).

## Leakage definitions

1. **Safe historical feature** — Uses only transactions strictly before the current row (by `TransactionDT`, then `TransactionID` at ties).
2. **Train-fitted static mapping** — Computed from train only and applied to validation/test. Not future leakage across splits, but **not row-causal** within train/val/test streaming inference.
3. **Full-period aggregate** — Uses future transactions from the same split or full dataset. **Not acceptable** for B2/B3.
4. **Label-derived feature** — Uses `isFraud` or fraud outcomes. **Not allowed.**

## Specific risks detected

| Risk | Location | Finding |
|------|----------|---------|
| groupby transform over full dataframe | `fit_count_frequency_mappings`, `fit_amount_stat_mappings` | Train-period aggregates — **train-static**, excluded from B2 |
| counts including current transaction | `count_{group}` in FE | Full train counts include all train rows for entity — **noncausal** for streaming |
| means including future rows | `amt_mean_by_{group}` | Train entity mean uses all train-period rows — **noncausal** |
| validation/test stats computed independently | FE mappings | Val/test use **train** maps only (no val-period fitting) — safe across splits but not causal within val |
| full-dataset frequency encoding | `freq_{group}` | Train frequency — **train-static** |
| fraud-rate / target encoding | Not found in FE or velocity scripts | **None detected** |
| rolling windows without shift | `historical_velocity_features.py` | Windows exclude current row via prior-state write — **safe** |
| expanding statistics without shift | `historical_velocity_features.py` | Expanding stats use state before row update — **safe** |
| sorting misalignment | `chronological_split` sorts by `TransactionDT` only | B2 re-sorts by `TransactionDT`, `TransactionID` for feature generation |
| category mappings fit outside train | `fit_baseline_preprocessing` | Categorical maps train-only — acceptable for original features |

## Final feature policy

### SAFE_CAUSAL_FEATURES (B2/B3 behavioral block — 19 features)

Entity definitions (3):

- `card1` → `["card1"]`
- `card1_addr1` → `["card1", "addr1"]`
- `card1_P_emaildomain` → `["card1", "P_emaildomain"]`

Per-entity base features (5 × 3 = 15):

- `cb_transaction_count_before_{entity}`
- `cb_time_since_previous_transaction_{entity}`
- `cb_historical_mean_amount_before_{entity}`
- `cb_historical_std_amount_before_{entity}`
- `cb_amount_deviation_from_historical_mean_{entity}`

Window features (2 windows × 2 entities = 4):

- `cb_count_in_previous_1_hour_{entity}` for `card1`, `card1_P_emaildomain`
- `cb_count_in_previous_24_hours_{entity}` for `card1`, `card1_P_emaildomain`

Implementation: `src/causal_behavioral_features.py`. Metadata: `outputs/causal_behavioral_feature_audit/feature_definition.json`.

### TRAIN_STATIC_FEATURES (reference / EX01–EX04 only)

All `count_*`, `freq_*`, `amt_*_by_*`, `nunique_*` from `feature_engineering.py`, plus amount/time derivations when bundled in EX01. **Not used in B2/B3.**

### EXCLUDED_NONCAUSAL_FEATURES

- Full-period entity counts and frequencies
- Train-fitted entity amount means/medians/stds used as row features
- `hist_count_last_7d_*` (EX04 window; not in B2 spec)
- AE latent, reconstructed, or multiple reconstruction transforms
- UID internal keys and label-derived aggregates

### REVIEW_REQUIRED_FEATURES

- EX04 `hist_*` family: causal but uses timestamp-batch policy instead of TransactionID ordering — **not reused verbatim**; B2 reimplements with stricter tie-breaking
- `TransactionDT` as P01 original predictor: retained in B2/B3; not a behavioral aggregate

## B1/B2/B3 existing experiment classification

| Role | Candidate | Classification | Reason |
|------|-----------|----------------|--------|
| B1 | `outputs/baseline_lgbm/` (P01) | **Existing and directly comparable** | 432 original features; chronological default LGBM |
| B2 | EX01 `baseline_lgbm_entity_time_amount_features/` | **Existing with caveat** | Uses train-static FE, not causal behavioral features |
| B2 | EX04 historical velocity | **Existing with caveat** | Mixes train-static FE + causal hist; 5 entities; timestamp-batch policy |
| B3 | AE15 `behavioral_cdv_ae_experiment/A_fe_lgbm_cdv_reconstruction_mse_default/` | **Existing with caveat** | FE + CDV recon; not B2 + one recon feature |
| B2 | `outputs/causal_behavioral_lgbm_default/` | **Missing** (pre-audit) | Required controlled arm |
| B3 | `outputs/causal_behavioral_cdv_reconstruction_lgbm_default/` | **Missing** (pre-audit) | Required controlled arm |

**Decision:** Implement corrected B2 and B3. Reuse P01 metrics for B1. Reuse CDV AE artifacts from `outputs/behavioral_cdv_ae_experiment/autoencoder_cdv_ld128/` without retraining.

## State continuation policy (B2/B3)

1. Sort all rows by `TransactionDT`, then `TransactionID`.
2. Process train rows sequentially; update entity state after each row.
3. Continue state into validation (train history visible).
4. Continue state into test (train + validation history visible).
5. Labels never update state.
6. No future rows used.