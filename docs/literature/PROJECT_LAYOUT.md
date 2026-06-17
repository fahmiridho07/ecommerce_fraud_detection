# Project Layout - Tugas Akhir

Workspace root: `1_TugasAkhir/`. Open this folder in Cursor/Codex for full
context. Start from `../../../AGENTS.md`.

| Path | Isi |
|------|-----|
| `ecommerce_fraud_detection/` | Repo code: active stratified pipeline, archived historical evidence, tests |
| `ecommerce_fraud_detection/docs/THESIS_SCOPE.md` | Scope aktif thesis |
| `ecommerce_fraud_detection/docs/AI_AGENT_BRIEF.md` | Panduan cepat untuk AI agent |
| `ecommerce_fraud_detection/docs/EXPERIMENT_REGISTRY.md` | Registry aktif stratified reset |
| `ecommerce_fraud_detection/archive/docs/chronological_evidence/HISTORICAL_EXPERIMENT_REGISTRY.md` | Metrik P01-P04 dan evidence chronological lama |
| `ecommerce_fraud_detection/docs/literature/` | Salinan agent-friendly: cards, deep-research, index |
| `AGENTS.md` | Instruksi baca workspace untuk AI agent, lokal dan tidak di-push |
| `ecommerce_fraud_detection/scripts/sync_literature.py` | Sync cards, index, dan deep-research dari parent |
| `../../2. Reference/` | PDF resmi sidang, source of truth sitasi |
| `../../5. Literature Cards/` | Kartu `_cards/` dan `LITERATURE_INDEX.md` master |
| `../../4. Deep Research/` | Laporan sintesis master |
| `../../0. Skripsi/proposal/Draft_ProposalTA.docx` | Proposal canonical |
| `../../0. Skripsi/skripsi/TA_Achmad Fahmi Ainur Ridho_2026.docx` | Skripsi canonical |
| `../../3. Lampiran Gambar dan Tabel/` | Gambar dan tabel sidang |

## Urutan Baca Untuk Agent

1. `AGENTS.md` di workspace root.
2. `docs/THESIS_SCOPE.md`.
3. `docs/AI_AGENT_BRIEF.md`.
4. `docs/EXPERIMENT_REGISTRY.md`.
5. `docs/literature/INDEX.md`, lalu `cards/` dan `deep-research/`.
6. Verifikasi angka dan kutipan ke PDF di `../../2. Reference/`.

## Navigasi Proyek

```text
1_TugasAkhir/
|-- ecommerce_fraud_detection/   # code + docs/literature
|-- 2. Reference/                # PDF resmi
|-- 5. Literature Cards/         # kartu + indeks master
|-- 4. Deep Research/            # analisis sintesis
`-- 0. Skripsi/                  # proposal, skripsi, template
```

## Source Of Truth

1. PDF di `2. Reference/` untuk angka dan kutipan resmi.
2. Kartu `_cards/` untuk ringkasan agent dan penulisan.
3. Deep Research untuk sintesis Bab 2/5.
4. `docs/EXPERIMENT_REGISTRY.md` untuk hasil aktif.
5. `archive/docs/chronological_evidence/` untuk hasil historis.
