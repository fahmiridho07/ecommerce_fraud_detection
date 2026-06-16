from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_literature.py"


def test_sync_literature_dry_run_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "would copy" in result.stdout


def test_sync_literature_outputs_exist() -> None:
    cards_dir = REPO_ROOT / "docs" / "literature" / "cards"
    deep_dir = REPO_ROOT / "docs" / "literature" / "deep-research"
    index_file = REPO_ROOT / "docs" / "literature" / "LITERATURE_INDEX.md"

    assert cards_dir.is_dir()
    assert any(cards_dir.glob("*.md"))
    assert deep_dir.is_dir()
    assert (deep_dir / "IEEE-CIS Fraud Detection Papers.md").is_file()
    assert index_file.is_file()