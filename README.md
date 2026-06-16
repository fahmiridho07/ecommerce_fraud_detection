# E-commerce Fraud Detection

Undergraduate thesis project on fraud detection using the IEEE-CIS Fraud Detection dataset.

## Project Status

This repository is scoped to the original thesis proposal path (P01–P04):

- Baseline LightGBM on the original IEEE-CIS features.
- Autoencoder representation learning on `V*` features (median imputation, masked loss, linear latent).
- AE-LightGBM with latent replacement plus `v_missing_*` indicators.
- Optuna tuning for baseline (P02) and AE-LightGBM LD128 (P04).

The source of truth is [`docs/THESIS_SCOPE.md`](docs/THESIS_SCOPE.md). Exploratory branches are parked under [`archive/`](archive/).

**Post-fix rerun completed (2026-06-16).** Proposal rerun: `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv`. Extended table (includes AE-05): `extended_proposal_comparison.csv`.

## Active Experiments

| ID | Experiment | Test PR-AUC | Role |
|----|------------|-------------|------|
| P01 / BASE-01 | Baseline LightGBM default | 0.4858 | Proposal baseline |
| P02 / BASE-02 | Baseline LightGBM Optuna tuned (15 trials) | 0.5049 | Tuned baseline (proposal) |
| P03 / AE-01 | AE-LightGBM LD32 + missing indicators | 0.4802 | Proposal AE default |
| P04 / AE-02 | AE-LightGBM LD128 Optuna tuned (15 trials) | 0.4845 | Tuned AE (LD128 caveat) |
| **AE-05** | Hybrid top-25 `V*` + LD32 latent + recon error | **0.5098** | **Best model (post-diagnostic)** |

P01–P04 answer the original proposal (latent replacement loses to P02). **AE-05** answers the refined integration path: retain high-gain `V*`, add AE latent + reconstruction-error features — beats P02 on test PR-AUC (+0.0049, bootstrap p≈0.009). See [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) and [`docs/DEFENSE_FAQ.md`](docs/DEFENSE_FAQ.md).

Legacy artifacts under `outputs/baseline_lgbm/` and `outputs/ae_lgbm/` reflect the pre-fix pipeline (125/221 features, zero-fill AE). Use `outputs/initial_proposal/` for thesis-facing numbers.

## Navigation

- [`docs/THESIS_SCOPE.md`](docs/THESIS_SCOPE.md) - active thesis scope and exclusion rules.
- [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) - experiment registry (P01-P04 + AE-05).
- [`docs/DEFENSE_FAQ.md`](docs/DEFENSE_FAQ.md) - sidang Q&A (metrik, protokol, AE-05).
- [`docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`](docs/INITIAL_PROPOSAL_RERUN_GUIDE.md) - exact rerun commands for the proposal path.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) - environment, validation, and artifact policy.
- [`src/README.md`](src/README.md) - script index for the remaining active source files.
- [`archive/README.md`](archive/README.md) - map of parked experiments.

## Reproduce The Core Path

Use Python 3.10+.

```bash
pip install -r requirements-dev.txt
python -m compileall -q src tests
python -m pytest -q
```

For a full rerun, place the IEEE-CIS files under `data/raw/` and follow [`docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`](docs/INITIAL_PROPOSAL_RERUN_GUIDE.md). The final four-row thesis table is `initial_proposal_comparison.csv`.

## Artifact Policy

- `data/raw/` is local Kaggle data and remains gitignored.
- `outputs/initial_proposal/` is the canonical thesis artifact tree; `outputs/_legacy/` holds pre-cleanup exploratory runs (not for thesis claims). All of `outputs/` remains gitignored.
- `results/` no longer carries active thesis evidence after cleanup.
- Tracked thesis-facing documentation lives in `docs/`.
- Parked exploratory scripts, docs, notebooks, and compact results live in `archive/`.

## Project Structure

```text
.
|-- archive/              # Parked exploratory and out-of-scope work
|-- data/raw/             # Local IEEE-CIS files, gitignored
|-- docs/                 # Active thesis documentation
|-- outputs/              # Local experiment artifacts, gitignored
|-- src/                  # Active proposal-scope source files
|-- tests/                # Active proposal-scope tests
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## Thesis Boundary

The cleaned repository does not promote behavioral features, late fusion, feature engineering, score ensembles, GBDT backend shootouts, or task-aware AE branches as active thesis claims. Those branches remain available in `archive/` for traceability only.

## Literature & thesis materials

Thesis PDFs and literature cards live **one level up** from this repo. Agent-friendly copies are synced into `docs/literature/`. PDF is the source of truth for citations.

- [`docs/literature/INDEX.md`](docs/literature/INDEX.md)
- [`docs/literature/PROJECT_LAYOUT.md`](docs/literature/PROJECT_LAYOUT.md)
- Sync command: `python scripts/sync_literature.py`
