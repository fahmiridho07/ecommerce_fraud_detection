# Broad AE Feature Ladder Results

Date: 2026-06-20

Purpose: test proposal-consistent alternatives after diagnosis showed that
restricting the Autoencoder to the `V` block and replacing original features
removes useful IEEE-CIS signal. These variants still keep the original
direction: Autoencoder-derived features are used to improve the feature space
for LightGBM, not to replace LightGBM with another classifier.

## Reference

Active protocol:

```text
split_strategy = stratified_holdout
train/validation/test = 60/20/20
random_state = 42
primary metric = Average Precision / PR-AUC
```

Reference model:

```text
original proposal tuned LightGBM validation AP = 0.874588541619
original proposal tuned LightGBM test AP       = 0.873133233976
```

Reference scores:

```text
outputs/stratified_reset/original_proposal_v_latent_replacement/baseline_tuned/baseline_tuned_test_scores.csv
```

## Implementation

Harness:

```text
src/run_broad_ae_feature_ladder.py
```

Output:

```text
outputs/stratified_reset/broad_ae_feature_ladder/
```

Key implementation details:

- uses the same stratified split and target/test score order as the proposal
  rerun;
- reads IEEE-CIS with a chunked memory-light loader because full pandas/Arrow
  CSV loading repeatedly exceeded local RAM;
- keeps original proposal LightGBM features and appends AE-derived features;
- tests AE beyond only `V` features using top feature-importance columns or
  feature-family groups;
- uses `max_lgbm_estimators=600` for the augmented LightGBM profiles because
  native LightGBM repeatedly crashed locally on the augmented full 999-estimator
  matrices without Python traceback.

The 600-estimator cap is a runtime constraint, not a methodological change in
the AE design. The tuned LightGBM reference remains the full reference score
above.

## Results

| Rank | Candidate | Design | Selected profile | Feature count | Validation AP | Test AP | Delta vs tuned LightGBM | Bootstrap CI 95% | p(delta<=0) |
|---:|---|---|---|---:|---:|---:|---:|---|---:|
| 1 | `broad2_groupwise_ae_latent_error` | Group-wise AE for V, identity/device, payment, behavior, match flags; append latent + error | `baseline_tuned` | 532 | 0.867723 | 0.868076 | -0.005057 | [-0.007915, -0.002072] | 1.000 |
| 2 | `broad4_normal_only_all_feature_anomaly` | AE trained only on normal transactions; append anomaly reconstruction-error features | `baseline_tuned` | 436 | 0.866876 | 0.867750 | -0.005383 | [-0.008262, -0.002869] | 1.000 |
| 3 | `broad3_all_feature_value_mask_recon_ld64` | All-feature AE reconstructs values + observed/missing mask | `ae_tuned_ld32` | 503 | 0.859240 | 0.857716 | -0.015418 | [-0.018155, -0.012071] | 1.000 |
| 4 | `broad1_all_feature_masked_latent_error_ld64` | All-feature AE on top 192 cross-family features; append latent + error | `ae_tuned_ld32` | 500 | 0.860304 | 0.857028 | -0.016105 | [-0.019015, -0.012733] | 1.000 |
| 5 | `broad5_supervised_aux_all_feature_ld64` | All-feature AE with auxiliary fraud head; append latent + error + aux score | `baseline_tuned` | 501 | 0.849368 | 0.848394 | -0.024739 | [-0.028754, -0.020344] | 1.000 |

No candidate beat the tuned LightGBM reference.

## Interpretation

The best broad variant is the group-wise AE (`broad2`), which narrows the test
gap to `-0.005057` AP. This suggests that separating AE representation by
feature family is better than one global AE, but still not enough to surpass
the tuned LightGBM on original proposal features.

The normal-only anomaly AE (`broad4`) is close to broad2, but still negative.
This means reconstruction error can carry some fraud/anomaly information, yet
LightGBM already extracts stronger signal from the original feature matrix.

The value+mask reconstruction AE (`broad3`) and auxiliary supervised AE
(`broad5`) did not help. The auxiliary head learned a fraud-related signal, but
when appended to LightGBM it behaved as redundant/noisy information rather than
recovering AP.

## Thesis-Facing Conclusion

These results support the advisor-facing explanation:

```text
The issue is not simply that the AE was only applied to V features. Even after
expanding AE feature learning across feature families, preserving original
features, adding reconstruction error, modeling missingness, and adding a
supervised auxiliary head, the tuned LightGBM on original proposal features
remains stronger under the IEEE-CIS stratified protocol.
```

The most defensible next conclusion is therefore not to jump to an unrelated
method, but to report that the AE feature-improvement family has been tested
within the original AE + LightGBM corridor and did not outperform the tuned
LightGBM baseline. The closest AE directions remain partial/missingness-aware
V reconstruction and group-wise AE error/latent features, but both are still
negative against the tuned reference.
