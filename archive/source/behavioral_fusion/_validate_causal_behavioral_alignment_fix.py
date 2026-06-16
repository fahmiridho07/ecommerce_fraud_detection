"""Standalone runner for causal behavioral alignment tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    tests_path = Path(__file__).resolve().parents[1] / "tests" / "test_causal_behavioral_alignment.py"
    result = subprocess.run([sys.executable, str(tests_path)], check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()