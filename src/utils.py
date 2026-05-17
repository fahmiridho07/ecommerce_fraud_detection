"""Small shared utilities used by the experiment scripts."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: list[str | Path] | tuple[str | Path, ...]) -> None:
    """Create multiple directories."""
    for path in paths:
        ensure_dir(path)


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON with stable formatting."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def log(message: str) -> None:
    """Print a clean timestamped message."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def make_run_id(prefix: str = "run") -> str:
    """Create a compact timestamp-based run id."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"
