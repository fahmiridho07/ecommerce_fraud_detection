# Reproducibility

## Environment

Use Python 3.10 or newer.

```bash
python --version
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `requirements.txt` plus `pytest`.

## Data

Place the Kaggle IEEE-CIS Fraud Detection training files under `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

The competition test files are not used for metric evaluation.

## Static Validation

```bash
python -m compileall -q src tests
python -m pytest -q
python src/_validate_initial_proposal_pipeline_guards.py
```

## Core Rerun

Follow [`INITIAL_PROPOSAL_RERUN_GUIDE.md`](INITIAL_PROPOSAL_RERUN_GUIDE.md) for the isolated P01-P04 rerun. The recommended outputs are under `outputs/initial_proposal/` so historical artifacts are not overwritten.

A post-fix rerun was completed on **2026-06-16**. Thesis-facing metrics are in:

- `outputs/initial_proposal/final_comparison/initial_proposal_comparison.csv`
- `docs/EXPERIMENT_REGISTRY.md` (tabular summary and caveats)

Legacy exploratory runs live under `outputs/_legacy/` (including pre-fix `baseline_lgbm/`, `ae_lgbm/`, behavioral, ensemble branches). Do not cite them as current thesis results.

## Artifact Policy

- `outputs/` is intentionally gitignored because it contains large models, arrays, metrics, and Optuna databases.
- `data/raw/` is intentionally gitignored because it contains Kaggle data.
- `archive/` is tracked source/documentation history, not active thesis evidence.
- New thesis summary tables should be generated from the active P01-P04 pipeline.

## Archived Validators

Archived validator scripts under `archive/source/` are not part of the active test suite. Run them only if you intentionally restore the corresponding archived experiment family.
