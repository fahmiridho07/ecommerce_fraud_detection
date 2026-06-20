# Tools

Local reporting and rendering helpers live here so `src/` can stay focused on
experiment entrypoints and reusable project code.

## Document Builders

- `build_brief_docx.js` generates `docs/BRIEF_DISKUSI_PAKARIF.docx`.
- `build_kajian_docx.js` generates `docs/KAJIAN_PENYEBAB_AE.docx`.

These scripts use the root `package.json` dependency on `docx`. Generated DOCX
files are local thesis/discussion artifacts and remain gitignored.

## Figure Renderers

The `render_*_png.py` scripts generate presentation/discussion PNGs under
`outputs/figures/`. Generated images remain gitignored with the rest of
`outputs/`.

## Visual Notebook

- `build_visual_notebook.py` builds a visual notebook/report artifact from
tracked docs and gitignored experiment outputs.
