# README Cleanup Proposal

**Applied to `README.md` on 2026-06-11 (Level 1 cleanup).** Retained here as the merge reference.

---

# Project status

IEEE-CIS e-commerce fraud detection research repository for an undergraduate thesis.

The **current thesis-candidate method** is **FUS-01 / LF01**: validation-selected score-level late fusion between:

- **BEH-01 / CBA01R** — identity-aligned causal behavioral LightGBM
- **AE-02 / P04** — tuned V-only AE-LightGBM LD128 representation expert

Historical experiments are **retained** for ablation evidence, governance traceability, and thesis defense. They are not deleted when superseded.

**Canonical IDs** are used in thesis writing; **legacy IDs** (P01, CBA01R, LF01, etc.) remain on output directories and scripts for reproducibility.

Navigation:

- [`docs/EXPERIMENT_NAMING_GUIDE.md`](docs/EXPERIMENT_NAMING_GUIDE.md)
- [`docs/ACTIVE_EXPERIMENT_MAP.md`](docs/ACTIVE_EXPERIMENT_MAP.md)
- [`docs/ABLATION_EXPERIMENT_MAP.md`](docs/ABLATION_EXPERIMENT_MAP.md)

---

# Canonical experiment IDs (active path)

| Canonical ID | Legacy ID | Role |
|--------------|-----------|------|
| BASE-01 | P01 | Raw LightGBM default |
| BASE-02 | P02 | Tuned raw LightGBM |
| AE-02 | P04 | Tuned V-only AE-LightGBM LD128 expert |
| BEH-01 | CBA01R | Identity-aligned behavioral LightGBM |
| FUS-01 | LF01 | Behavioral + AE late fusion (thesis candidate) |

---

# How to reproduce key summaries

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

---

# Active thesis path

| ID | Description | Validation AP | Test AP |
|----|-------------|---------------|---------|
| BASE-01 / P01 | 432 original features; default LightGBM | 0.602433 | 0.485756 |
| BASE-02 / P02 | Tuned original-feature LightGBM | 0.624072 | 0.501438 |
| AE-02 / P04 | V replaced by LD128 latent; Optuna tuned | 0.610631 | 0.490686 |
| BEH-01 / CBA01R | Original + 19 causal behavioral features | 0.615122 | 0.493838 |
| FUS-01 / LF01 | 50/50 score fusion of BEH-01 + AE-02 | 0.629600 | 0.505543 |

Supervisor approval required before promoting FUS-01 / LF01 as the final thesis model.

---

# Ablation and legacy experiments

Not part of the final method but **kept in the repository**:

- **AE-01–AE-07, BEH-02** — integration and behavioral ablations
- **APP-01** — split-strategy appendix
- **LEGACY-01–03** — provisional CBA results, FE score ensembles
- **AE-06 / AE15** — FE + CDV recon exploratory branch (`legacy_archived`)

> **High AP alone is insufficient for final-model promotion** when the branch has weaker governance, static/non-causal feature construction, test-inspection risk, or incomplete reproducibility.

AE-06 / AE15 (validation AP 0.635954) is excluded from the active path despite higher validation AP than FUS-01 because it uses static FE features and exploratory governance.

---

# Important caveats

- **Validation AP** drives model-design conclusions; **test AP** is descriptive only for the frozen comparison set.
- FUS-01 / LF01 uses validation-only weight grid {0.50–1.00} and MCC threshold selection.
- Mixed tuning: BEH-01 default LightGBM vs AE-02 Optuna-tuned.
- Historical test inspection in exploratory FE/ensemble branches is a documented limitation (`docs/FINAL_REPORT_GOVERNANCE_NOTE.md`).
- Chronological 60/20/20 split by `TransactionDT` is the primary protocol.

---

# Project structure (updated)

```text
.
├── data/raw/              # Local Kaggle files (gitignored content)
├── docs/                  # Experiment registry, maps, governance
├── notebooks/             # thesis_experiment_report.ipynb
├── outputs/               # Experiment artifacts (gitignored)
├── results/               # Compact tracked summaries
├── src/                   # Training, fusion, audit scripts
├── tests/                 # Alignment unit tests
├── requirements.txt
└── README.md
```

See [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) for the full script inventory.

---

# Install and full pipeline (reference)

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