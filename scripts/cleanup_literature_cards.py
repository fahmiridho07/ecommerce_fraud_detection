#!/usr/bin/env python3
"""Remove OCR references from literature cards and rebuild LITERATURE_INDEX."""

from __future__ import annotations

import re
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LIT_ROOT = WORKSPACE_ROOT / "5. Literature Cards"
CARDS_DIR = LIT_ROOT / "_cards"
INDEX_PATH = LIT_ROOT / "LITERATURE_INDEX.md"

FILE_SECTION_RE = re.compile(
    r"(## File\n\n)(.*?)(?=\n## |\Z)",
    re.DOTALL,
)
FULLTEXT_MD_RE = re.compile(r"^fulltext_md:.*\n", re.MULTILINE)
OCR_BLOCK_RE = re.compile(
    r"\n- Full-text MD \(OCR\):.*\n\n> Full-text MD adalah ekstrak OCR.*\n",
    re.MULTILINE,
)


def _pdf_line_from_yaml(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("pdf:"):
            raw = line.split(":", 1)[1].strip().strip('"')
            return raw.replace("../../", "")
    return None


def clean_card(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = FULLTEXT_MD_RE.sub("", original)
    updated = OCR_BLOCK_RE.sub("\n", updated)

    if "## File" in updated:
        pdf_line = _pdf_line_from_yaml(updated)
        if pdf_line:
            replacement = (
                "## File\n\n"
                f"- PDF (source of truth): `{pdf_line}`\n"
                "- Kartu ini untuk ringkasan; verifikasi angka/kutipan ke PDF.\n"
            )
            updated = FILE_SECTION_RE.sub(replacement, updated, count=1)

    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def rebuild_index() -> None:
    cards = {p.stem for p in CARDS_DIR.glob("*.md")}
    rows: list[str] = []
    in_table = False

    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("| Folder | File | Priority | MD | PDF |"):
            rows.append("| Folder | File | Priority | PDF | Card |")
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
                rows.append(line)
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) != 5 or parts[0] == "Folder":
                rows.append(line)
                continue
            folder, file_stem, priority, _md, pdf = parts
            card = f"`_cards/{file_stem}.md`" if file_stem in cards else "—"
            rows.append(f"| {folder} | {file_stem} | {priority} | {pdf} | {card} |")
            continue
        rows.append(line)

    text = "\n".join(rows)
    text = text.replace(
        "Indeks agent-friendly untuk 55 referensi. **Kartu ringkas** ada di `_cards/`; full-text OCR di file `.md` sejajar.",
        "Indeks 55 referensi. **Kartu ringkas** di `_cards/`; **PDF** di `2. Reference/` adalah source of truth untuk angka dan kutipan.",
    )
    text = text.replace(
        "├── 5. Literature Cards/     # full-text OCR + index ini",
        "├── 5. Literature Cards/     # kartu + indeks (tanpa OCR)",
    )
    text = re.sub(
        r"\n## Catatan OCR\n.*",
        "\n## Source of truth\n\n1. **Kartu `_cards/`** — ringkasan untuk agent & penulisan.\n2. **PDF `2. Reference/`** — verifikasi angka dan sitasi resmi.\n3. **Deep Research** — sintesis Bab 2/5.\n",
        text,
        flags=re.DOTALL,
    )
    INDEX_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    changed = sum(clean_card(p) for p in sorted(CARDS_DIR.glob("*.md")))
    thesis = CARDS_DIR / "THESIS_P01_P04_RESULTS.md"
    body = thesis.read_text(encoding="utf-8")
    fixed = body.replace(
        "P02 tuned LightGBM test AP 0.5049 > P04 0.4845 > P01 0.4858 > P03 0.4802.",
        "P02 tuned LightGBM test AP 0.5049 > P01 0.4858 > P04 0.4845 > P03 0.4802.",
    )
    if fixed != body:
        thesis.write_text(fixed, encoding="utf-8")
        changed += 1
    rebuild_index()
    print(f"Updated {changed} card(s) and rebuilt LITERATURE_INDEX.md")


if __name__ == "__main__":
    main()