# Project Layout — Tugas Akhir

Workspace root: `1_TugasAkhir/` (buka folder ini di Cursor untuk konteks penuh). Mulai dari [`../../../AGENTS.md`](../../../AGENTS.md).

| Path | Isi |
|------|-----|
| `ecommerce_fraud_detection/` | **Repo code** — pipeline P01–P04, outputs, tests |
| `ecommerce_fraud_detection/docs/THESIS_SCOPE.md` | Scope aktif thesis |
| `ecommerce_fraud_detection/docs/EXPERIMENT_REGISTRY.md` | Metrik P01–P04 (source of truth) |
| `ecommerce_fraud_detection/docs/literature/` | **Salinan agent-friendly** (cards, deep-research, index) |
| `AGENTS.md` (workspace root, lokal) | Instruksi baca untuk AI agent — tidak di-push |
| `ecommerce_fraud_detection/scripts/sync_literature.py` | Sync cards + index + deep-research dari parent |
| `../../2. Reference/` | PDF resmi sidang (master) |
| `../../5. Reference (MarkDown)/` | Full-text OCR + `_cards/` + `LITERATURE_INDEX.md` (master) |
| `../../4. Deep Research/` | Laporan sintesis (5 file, master) |
| `../../0. Skripsi/proposal/Draft_ProposalTA.docx` | Proposal (canonical) |
| `../../0. Skripsi/skripsi/TA_Achmad Fahmi Ainur Ridho_2026.docx` | Skripsi (canonical) |
| `../../3. Lampiran Gambar dan Tabel/` | Gambar & tabel sidang |

## Urutan baca untuk agent

1. `AGENTS.md` (workspace root, lokal)
2. `docs/THESIS_SCOPE.md`
3. `docs/EXPERIMENT_REGISTRY.md`
4. `docs/literature/INDEX.md` → `cards/` → `deep-research/`
5. Full-text hanya jika perlu detail — `../../5. Reference (MarkDown)/`
