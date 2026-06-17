"""Generic file I/O helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_save_json(data: dict[str, Any], dest: Path) -> None:
    """Write JSON atomically: write to tmp, then rename.

    This prevents corrupt files if the process is interrupted mid-write.
    """
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    shutil.move(str(tmp), str(dest))


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file and return as dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_copy(src: Path, dst: Path) -> None:
    """Copy a file, creating destination parent dirs as needed."""
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))
