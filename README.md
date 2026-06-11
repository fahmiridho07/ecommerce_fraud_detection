<div align="center">

# E-commerce Fraud Detection

**Undergraduate thesis project on detecting fraudulent e-commerce transactions using the IEEE-CIS Fraud Detection dataset.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-Baseline-9ACD32?style=flat-square)
![Autoencoder](https://img.shields.io/badge/Robust%20AE-V--features-FFB000?style=flat-square)
![Metric](https://img.shields.io/badge/Primary%20Metric-PR--AUC-FF6B6B?style=flat-square)
![Status](https://img.shields.io/badge/Thesis%20Candidate-FUS--01%2FLF01-7C3AED?style=flat-square)

</div>

## Project status

IEEE-CIS fraud detection research repository for an undergraduate thesis.

The **current thesis-candidate method** is **FUS-01 / LF01**: validation-selected score-level late fusion between:

- **BEH-01 / CBA01R** — identity-aligned causal behavioral LightGBM
- **AE-02 / P04** — tuned V-only AE-LightGBM LD128 representation expert

Historical experiments are **retained** for ablation evidence, governance traceability, and thesis defense. They are not deleted when superseded.

**Canonical IDs** are used in thesis writing; **legacy IDs** (P01, CBA01R, LF01, etc.) remain on output directories and scripts for reproducibility.

**Navigation:**

- [`docs/EXPERIMENT_NAMING_GUIDE.md`](docs/EXPERIMENT_NAMING_GUIDE.md) — canonical / legacy dual notation
- [`docs/ACTIVE_EXPERIMENT_MAP.md`](docs/ACTIVE_EXPERIMENT_MAP.md) — final active path
- [`docs/ABLATION_EXPERIMENT_MAP.md`](docs/ABLATION_EXPERIMENT_MAP.md) — ablation and legacy evidence
- [`docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`](docs/INITIAL_PROPOSAL_RERUN_GUIDE.md) — BASE-01..AE-02 proposal-only rerun path
- [`docs/AE_INTEGRATION_STRATEGY_ABLATION.md`](docs/AE_INTEGRATION_STRATEGY_ABLATION.md) — STR-B0..STR-AE3 AE integration strategy ablation
- [`docs/REPOSITORY_CLEANUP_AUDIT.md`](docs/REPOSITORY_CLEANUP_AUDIT.md) — repository hygiene audit

Initial proposal reruns (BASE-01, BASE-02, AE-01, AE-02) are separate from LF01/fusion and other out-of-scope branches. After a proposal rerun, build `outputs/final_comparison/initial_proposal_comparison.csv` (generated output under gitignored `outputs/`, not tracked in git).

The AE integration strategy ablation (STR-B0..STR-AE3) is a separate diagnostic path under `outputs/ae_integration_strategy_ablation/`; it is not the initial proposal literal rerun and does not modify LF01/fusion experiments.

## Canonical experiment IDs (active path)

| Canonical ID | Legacy ID | Role |
|--------------|-----------|------|
| BASE-01 | P01 | Raw LightGBM default |
| BASE-02 | P02 | Tuned raw LightGBM |
| AE-02 | P04 | Tuned V-only AE-LightGBM LD128 expert |
| BEH-01 | CBA01R | Identity-aligned behavioral LightGBM |
| FUS-01 | LF01 | Behavioral + AE late fusion (thesis candidate) |

## Active thesis path

| ID | Description | Validation AP | Test AP |
|----|-------------|---------------|---------|
| BASE-01 / P01 | 432 original features; default LightGBM | 0.602433 | 0.485756 |
| BASE-02 / P02 | Tuned original-feature LightGBM | 0.624072 | 0.501438 |
| AE-02 / P04 | V replaced by LD128 latent; Optuna tuned | 0.610631 | 0.490686 |
| BEH-01 / CBA01R | Original + 19 causal behavioral features | 0.615122 | 0.493838 |
| FUS-01 / LF01 | 50/50 score fusion of BEH-01 + AE-02 | 0.629600 | 0.505543 |

Supervisor approval is required before promoting FUS-01 / LF01 as the final thesis model.

## How to reproduce key summaries

Requires local `data/raw/` and trained artifacts under `outputs/` (gitignored).

```bash
# Alignment correction validation (BEH-01 / CBA01R)
python src/_post_execution_validation_causal_behavioral_alignment.py

# Late fusion (FUS-01 / LF01) — requires frozen CBA01R + P04 artifacts
python src/run_causal_behavioral_ae_late_fusion.py --overwrite

# Build fusion comparison table
python src/build_causal_behavioral_ae_late_fusion_comparison.py

# Post-run fusion validation
python src/_post_execution_validation_late_fusion.py
```

**Artifacts:**

- `outputs/` — full experiment outputs (models, metrics JSON); **gitignored**
- `results/` — compact tracked summaries (CSV/JSON) for supervisor review
- Local model pickles and latent arrays are required for full score regeneration

## Install and full pipeline (reference)

```bash
pip install -r requirements.txt
python src/check_data_split.py
python src/train_baseline_lgbm.py
python src/train_autoencoder_robust.py --latent-dim 128
python src/train_ae_lgbm.py
python src/tune_lgbm_optuna.py --model_type baseline_lgbm
python src/tune_lgbm_optuna.py --model_type ae_lgbm_ld128
python src/train_causal_behavioral_lgbm.py --id-aligned
python src/run_causal_behavioral_ae_late_fusion.py
```

Full reproduction requires all intermediate `outputs/` artifacts. See [`docs/ACTIVE_EXPERIMENT_MAP.md`](docs/ACTIVE_EXPERIMENT_MAP.md).

## Highlights

- **Temporal split:** chronological 60/20/20 by `TransactionDT`.
- **Primary metric:** Average Precision (PR-AUC); validation AP drives model-design conclusions.
- **Leakage-aware evaluation:** preprocessing, thresholds, and tuning fitted only on allowed splits.

## Ablation and legacy experiments

Not part of the final method but **kept in the repository**:

- **AE-01–AE-07, BEH-02** — integration and behavioral ablations
- **APP-01** — split-strategy appendix
- **LEGACY-01–03** — provisional CBA results, FE score ensembles
- **AE-06 / AE15** — FE + CDV recon exploratory branch (`legacy_archived`)

> **High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

AE-06 / AE15 (validation AP 0.635954) is excluded from the active path despite higher validation AP than FUS-01 because it uses static FE features and exploratory governance.

## Important caveats

- **Test AP** is descriptive only for the frozen comparison set.
- FUS-01 / LF01 uses a validation-only weight grid {0.50–1.00} and MCC threshold selection.
- Mixed tuning: BEH-01 default LightGBM vs AE-02 Optuna-tuned.
- Historical test inspection in exploratory FE/ensemble branches is documented in [`docs/FINAL_REPORT_GOVERNANCE_NOTE.md`](docs/FINAL_REPORT_GOVERNANCE_NOTE.md).

## Project structure

```text
.
├── data/raw/              # Local Kaggle files (gitignored content)
├── docs/                  # Experiment registry, maps, governance, cleanup audit
├── notebooks/             # thesis_experiment_report.ipynb
├── outputs/               # Experiment artifacts (gitignored)
├── results/               # Compact tracked summaries
├── src/                   # Training, fusion, audit scripts
├── tests/                 # Alignment unit tests
├── requirements.txt
└── README.md
```

See [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) for the full script inventory.

## Notebook report

```text
notebooks/thesis_experiment_report.ipynb
```

Loads existing artifacts from `outputs/` for supervisor-facing summaries. Does **not** retrain models.

## Kaggle execution

Attach the **IEEE-CIS Fraud Detection** dataset. The project uses `/kaggle/input/ieee-fraud-detection` when present; otherwise `data/raw/`. Competition test files are **not** used for metric evaluation.

## Experiment governance

| Document | Purpose |
|----------|---------|
| [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) | Full experiment inventory |
| [`docs/FINAL_EXPERIMENT_PLAN.md`](docs/FINAL_EXPERIMENT_PLAN.md) | Reporting template |
| [`docs/EXPERIMENT_SCOPE_FREEZE.md`](docs/EXPERIMENT_SCOPE_FREEZE.md) | Frozen narrative scope |
| [`docs/RESULT_ARTIFACT_MANIFEST.md`](docs/RESULT_ARTIFACT_MANIFEST.md) | Git-tracking recommendations |
| [`docs/REPOSITORY_CLEANUP_PLAN.md`](docs/REPOSITORY_CLEANUP_PLAN.md) | Phased cleanup plan |

Chronological evaluation on `TransactionDT` is the primary protocol. Stratified holdout and CV are appendix sensitivity analyses only.

## Author

Created and maintained by the repository owner as part of an undergraduate thesis project.