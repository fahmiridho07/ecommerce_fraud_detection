# E-commerce Fraud Detection

Undergraduate thesis project on fraud detection using the IEEE-CIS Fraud
Detection dataset.

## Project Status

Status: stratified split cleanup reset, 2026-06-17.

The active thesis protocol now uses:

```text
stratified_holdout, 60% train / 20% validation / 20% test, random_state=42
```

There is no active thesis-facing winner after this reset yet. Previous
chronological P01-P04, AE-05, preprocessing, and score-ensemble results are
archived as historical evidence and must not be mixed with new stratified
results.

The source of truth is [`docs/THESIS_SCOPE.md`](docs/THESIS_SCOPE.md). The
cleanup/rerun plan is [`docs/STRATIFIED_SPLIT_RESET.md`](docs/STRATIFIED_SPLIT_RESET.md).

## Active Rerun Plan

| ID | Experiment | Split | Status |
|----|------------|-------|--------|
| S0 | Split validation | `stratified_holdout` | Complete |
| A0 | Baseline LightGBM default | `stratified_holdout` | Pending |
| A0-T | Baseline LightGBM tuned | `stratified_holdout` | Pending |
| A1 | Alharbi-style preprocessing baseline | `stratified_holdout` | Pending |
| A1-T | Tuned Alharbi-style preprocessing baseline | `stratified_holdout` | Pending |

Historical chronological results remain in
[`archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md`](archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md).

## Navigation

- [`docs/THESIS_SCOPE.md`](docs/THESIS_SCOPE.md) - active thesis scope and exclusion rules.
- [`docs/README.md`](docs/README.md) - documentation map.
- [`docs/AI_AGENT_BRIEF.md`](docs/AI_AGENT_BRIEF.md) - quick orientation for future AI agents.
- [`docs/STRATIFIED_SPLIT_RESET.md`](docs/STRATIFIED_SPLIT_RESET.md) - cleanup decision and rerun ladder.
- [`docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`](docs/PAPER_ANCHORED_PREPROCESSING_RESET.md) - active preprocessing anchor plan.
- [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) - active stratified registry.
- [`docs/INITIAL_PROPOSAL_RERUN_GUIDE.md`](docs/INITIAL_PROPOSAL_RERUN_GUIDE.md) - historical P01-P04 reproduction guide.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) - environment, validation, and artifact policy.
- [`src/README.md`](src/README.md) - script index.
- [`archive/README.md`](archive/README.md) - map of parked experiments.

## Validate

Use Python 3.10+.

```bash
pip install -r requirements-dev.txt
python -m compileall -q src tests
python -m pytest -q
python src/check_data_split.py
```

## Artifact Policy

- `data/raw/` is local Kaggle data and remains gitignored.
- `outputs/` is local experiment output and remains gitignored.
- Use `outputs/stratified_reset/` for the next clean rerun.
- `outputs/initial_proposal/` contains historical chronological artifacts.
- Tracked thesis-facing documentation lives in `docs/`.
- Parked exploratory scripts, historical evidence docs, notebooks, and compact results live in `archive/`.

## Project Structure

```text
.
|-- archive/              # Parked exploratory and out-of-scope work
|-- data/raw/             # Local IEEE-CIS files, gitignored
|-- docs/                 # Thesis documentation and active reset plan
|-- outputs/              # Local experiment artifacts, gitignored
|-- src/                  # Active source files
|-- tests/                # Test suite
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## Literature & Thesis Materials

Thesis PDFs and literature cards live one level up from this repo. Agent-friendly
copies are synced into `docs/literature/`. PDF is the source of truth for
citations.

- [`docs/literature/INDEX.md`](docs/literature/INDEX.md)
- [`docs/literature/PROJECT_LAYOUT.md`](docs/literature/PROJECT_LAYOUT.md)
- Sync command: `python scripts/sync_literature.py`
