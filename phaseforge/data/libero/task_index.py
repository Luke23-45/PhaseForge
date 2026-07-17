"""Deterministic task-id assignment from LIBERO task filenames.

Problem this module solves
--------------------------
The previous implementation derived ``task_id`` via
``abs(hash(path.stem)) % 10**6``. Python's built-in ``hash()`` is salted
per process (unless ``PYTHONHASHSEED`` is fixed), so the SAME task file
got a DIFFERENT id on every run. Proven by simulation: the string
``"KITCHEN_SCENE1_open_drawer_demo"`` produced 947019, 225566, and 700893
across three processes. That silently corrupts any per-task metric
(routing stability, phase-to-expert alignment) and makes results
non-reproducible across machines.

This module replaces that with a stable, content-derived scheme:

    sorted(all task filenames in the suite)  ->  0, 1, 2, ...

The same suite, on any machine, in any process, always yields the same
name -> id mapping. The mapping is also written into the cache manifest
so it is auditable and can be reloaded without recomputation.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Filename suffix shared by every task file in the LIBERO suites.
#: Verified: 100% of files in both libero_90/ and libero_10/ end with this.
_HDF5_SUFFIX = ".hdf5"


def build_task_index(raw_suite_dir: Path) -> dict[str, int]:
    """Build a deterministic ``{task_name: int_id}`` mapping for a suite.

    The task name is the HDF5 filename stem (e.g.
    ``"KITCHEN_SCENE1_open_drawer_demo"``). Filenames are sorted
    lexicographically (``sorted`` is stable and locale-independent for
    ASCII task names), then assigned sequential ids 0, 1, 2, ...

    Args:
        raw_suite_dir: Directory containing the suite's per-task ``.hdf5``
            files, e.g. ``data/raw/libero/libero_90/``.

    Returns:
        Dict mapping each filename stem to a stable integer id.

    Raises:
        FileNotFoundError: If ``raw_suite_dir`` does not exist.
        RuntimeError: If no ``.hdf5`` files are found.
    """
    raw_suite_dir = Path(raw_suite_dir)
    if not raw_suite_dir.exists():
        raise FileNotFoundError(
            f"LIBERO suite directory not found: {raw_suite_dir}. "
            "Run `python -m phaseforge.data.scripts.download_libero` first."
        )

    hdf5_files = sorted(raw_suite_dir.glob(f"*{_HDF5_SUFFIX}"))
    if not hdf5_files:
        raise RuntimeError(
            f"No .hdf5 files found in {raw_suite_dir}. "
            "Run `python -m phaseforge.data.scripts.download_libero` first."
        )

    task_index: dict[str, int] = {
        f.stem: i for i, f in enumerate(hdf5_files)
    }
    logger.debug(
        "Built task index for %s: %d tasks (ids 0..%d)",
        raw_suite_dir,
        len(task_index),
        len(task_index) - 1,
    )
    return task_index


def task_id_for(stem: str, task_index: dict[str, int]) -> int:
    """Look up the stable id for a task filename stem.

    Args:
        stem: Filename stem (e.g. ``"KITCHEN_SCENE1_open_drawer_demo"``).
        task_index: Mapping produced by :func:`build_task_index`.

    Raises:
        KeyError: If ``stem`` is not in ``task_index`` (indicates a
            mismatch between the files present and the index — should not
            happen if both come from the same suite directory).
    """
    try:
        return task_index[stem]
    except KeyError as exc:
        raise KeyError(
            f"Task name {stem!r} not found in task index. "
            f"Known tasks: {sorted(task_index)[:5]}... ({len(task_index)} total). "
            "Rebuild the index from the current raw suite directory."
        ) from exc
