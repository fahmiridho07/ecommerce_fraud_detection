---
id: THESIS_P01_P04_RESULTS
priority: core
authors: "This thesis (Ridho)"
year: 2026
dataset: "IEEE-CIS"
method: "AE-LightGBM + Optuna"
metrics: "P02 test AP 0.5049"
split: "historical chronological TransactionDT"
comparable_to_thesis: "historical only after stratified reset"
thesis_use: "Bab 4-5 historical audit trail only"
bab: "4,5"
source: "../../ecommerce_fraud_detection/archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md"
---

# Hasil Eksperimen P01-P04 (post-fix 2026-06-16)

## Ringkasan

- P02 tuned LightGBM test AP 0.5049 > P01 0.4858 > P04 0.4845 > P03 0.4802.
- Di blok proposal ini, AE latent replacement tidak mengalahkan P02.
- AE-05 (0.5098) juga historical setelah stratified reset; blok terpisah, jangan campur dengan tabel P01-P04 atau hasil stratified baru.
