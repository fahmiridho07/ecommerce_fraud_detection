# DOCX Update Notes

Status: archived chronological DOCX update note. Superseded by the 2026-06-17
stratified split reset; do not treat the method-update DOCX as current thesis
prose until it is rewritten.

Date: 2026-06-17

Updated working copy:

```text
0. Skripsi/skripsi/TA_Achmad Fahmi Ainur Ridho_2026_method_update.docx
```

Original preserved:

```text
0. Skripsi/skripsi/TA_Achmad Fahmi Ainur Ridho_2026.docx
```

## Scope of Update

The working copy updates the thesis body from Bab 3 through Bab 5:

- Bab 3: methodology aligned with the final fixed score-level ensemble candidate.
- Bab 4: result summary table, bootstrap significance table, and discussion.
- Bab 5: conclusion and suggestions aligned with the current candidate.

The update follows:

- `docs/BAB3_METHOD_ADJUSTMENT.md`
- `archive/docs/chronological_evidence/FINAL_CANDIDATE_VALIDATION.md`
- `docs/EXPERIMENT_REGISTRY.md`

## Method Claim

The thesis-facing claim is:

```text
Autoencoder improves detection when used as a complementary score-level signal
to a preprocessing-strengthened LightGBM baseline.
```

The update avoids claiming that:

- AE replaces LightGBM.
- AE latent replacement alone beats the baseline.
- AE reconstruction error alone is the final improvement.

## QA

Structural QA completed:

- Required headings detected:
  - `METODOLOGI`
  - `Integrasi Skor LightGBM dan AE-LightGBM`
  - `Hasil dan Pembahasan`
  - `Kesimpulan dan Saran`
  - `DAFTAR PUSTAKA`
- Result table detected with 7 rows and 6 columns.
- Bootstrap table detected with 2 rows and 6 columns.
- Original DOCX preserved.

Visual render QA was attempted with the Documents renderer, but LibreOffice/`soffice` was not available in the environment (`where soffice` returned no executable). The DOCX should be opened in Word or LibreOffice for final visual inspection before submission.
