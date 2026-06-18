# E-commerce Fraud Detection

Undergraduate thesis project on fraud detection using the IEEE-CIS Fraud
Detection dataset.

## Project Status

Status: active Bab 4 writing baseline after stratified reset, 2026-06-18.

The active thesis protocol now uses:

```text
stratified_holdout, 60% train / 20% validation / 20% test, random_state=42
```

The active thesis result is now complete enough for Bab 4 drafting. The final
defensible claim is narrow: Autoencoder features/reconstruction hurt or tie, but
Autoencoder latent-space oversampling improves LightGBM on the dense
Alharbi-style representation and beats matched SMOTE-NC after fair tuning.

Previous chronological P01-P04, AE-05, preprocessing, and score-ensemble results
remain archived as historical evidence and must not be mixed with active
stratified result tables.

The source of truth is [`docs/THESIS_SCOPE.md`](docs/THESIS_SCOPE.md). The
write-ready Bab 4 result draft is
[`docs/THESIS_RESULTS_BAB4.md`](docs/THESIS_RESULTS_BAB4.md).

## Active Result Snapshot

| Finding | Active result |
|---------|---------------|
| Split validation | Complete, stratified 60/20/20, fraud rate ~3.499% in all splits |
| AE as feature extractor | Loses or ties; latent/reconstruction features reduce AP |
| A0 raw augmentation | Beats baseline, but AE ties SMOTE-NC on raw/NaN-native features |
| A1 dense augmentation | AE beats SMOTE-NC across 4 split seeds, mean delta AP +0.0204 |
| A1 full budget | AE beats SMOTE-NC by +0.02589 AP |
| A1 tuned-vs-tuned | AE 0.8500 AP > SMOTE-NC 0.8435 > baseline 0.8390 |

Historical chronological results remain in
[`archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md`](archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md).

## Navigation

- [`docs/THESIS_SCOPE.md`](docs/THESIS_SCOPE.md) - active thesis scope and exclusion rules.
- [`docs/README.md`](docs/README.md) - documentation map.
- [`docs/AI_AGENT_BRIEF.md`](docs/AI_AGENT_BRIEF.md) - quick orientation for future AI agents.
- [`docs/STRATIFIED_SPLIT_RESET.md`](docs/STRATIFIED_SPLIT_RESET.md) - cleanup decision and rerun ladder.
- [`docs/PAPER_ANCHORED_PREPROCESSING_RESET.md`](docs/PAPER_ANCHORED_PREPROCESSING_RESET.md) - active preprocessing anchor plan.
- [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) - active stratified registry.
- [`docs/AE_INTEGRATION_EXPERIMENT_RESULTS.md`](docs/AE_INTEGRATION_EXPERIMENT_RESULTS.md) - detailed AE results and bootstrap comparisons.
- [`docs/EDA_AND_METHODOLOGY_AUDIT.md`](docs/EDA_AND_METHODOLOGY_AUDIT.md) - EDA facts and methodology audit.
- [`docs/THESIS_RESULTS_BAB4.md`](docs/THESIS_RESULTS_BAB4.md) - consolidated Bab 4 result draft.
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
- Active experiment artifacts live under `outputs/stratified_reset/`.
- Active Bab 4 figures live under `outputs/stratified_reset/thesis_figures/`.
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
