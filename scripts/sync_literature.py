#!/usr/bin/env python3
"""Sync literature cards, index, and deep-research reports from parent workspace."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

SOURCES = {
    "cards": WORKSPACE_ROOT / "5. Reference (MarkDown)" / "_cards",
    "index": WORKSPACE_ROOT / "5. Reference (MarkDown)" / "LITERATURE_INDEX.md",
    "deep_research": WORKSPACE_ROOT / "4. Deep Research",
}

DESTINATIONS = {
    "cards": REPO_ROOT / "docs" / "literature" / "cards",
    "index": REPO_ROOT / "docs" / "literature" / "LITERATURE_INDEX.md",
    "deep_research": REPO_ROOT / "docs" / "literature" / "deep-research",
}


def _copy_markdown_files(src_dir: Path, dest_dir: Path, dry_run: bool) -> list[str]:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    actions: list[str] = []

    for src in sorted(src_dir.glob("*.md")):
        dest = dest_dir / src.name
        if dry_run:
            actions.append(f"would copy {src.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(REPO_ROOT)}")
            continue
        shutil.copy2(src, dest)
        actions.append(f"copied {src.name}")

    return actions


def _copy_file(src: Path, dest: Path, dry_run: bool) -> str:
    if not src.is_file():
        raise FileNotFoundError(f"Source file not found: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return f"would copy {src.relative_to(WORKSPACE_ROOT)} -> {dest.relative_to(REPO_ROOT)}"
    shutil.copy2(src, dest)
    return f"copied {dest.name}"


def sync_literature(dry_run: bool = False) -> int:
    missing = [name for name, path in SOURCES.items() if not path.exists()]
    if missing:
        print("Missing parent sources:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}: {SOURCES[name]}", file=sys.stderr)
        print(
            "\nRun from repo root with workspace parent at 1_TugasAkhir/.",
            file=sys.stderr,
        )
        return 1

    actions: list[str] = []
    actions.extend(_copy_markdown_files(SOURCES["cards"], DESTINATIONS["cards"], dry_run))
    actions.append(_copy_file(SOURCES["index"], DESTINATIONS["index"], dry_run))
    actions.extend(
        _copy_markdown_files(
            SOURCES["deep_research"],
            DESTINATIONS["deep_research"],
            dry_run,
        )
    )

    label = "Dry run" if dry_run else "Sync complete"
    print(f"{label}: {len(actions)} file(s)")
    for line in actions:
        print(f"  {line}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned copies without writing files.",
    )
    args = parser.parse_args()
    return sync_literature(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())