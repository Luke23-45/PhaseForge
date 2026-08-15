"""Remove leftover failed run directories from ``outputs/``.

A run directory is treated as a *failed attempt* when it contains
``logs/exception.txt`` and has **no** sibling ``<run_dir>.completed``
lifecycle marker. The runner only resolves runs that carry the completed
marker and the registry (``outputs/_runner/state.json``) only records them,
so such directories are pure leftovers from attempts that crashed before
writing a marker (e.g. the 12 pre-fix ``warmstart_moe`` /
``plain_encoder_phase_bootstrap`` dimension-mismatch failures).

Prints what it would delete by default (dry-run); pass ``--apply`` to
actually remove the directories.

Usage::

    python scripts/cleanup_failed_runs.py --outputs outputs [--apply]

Exit code 0 with nothing to clean (or after cleaning); exit code 1 only on
internal error. Never touches directories that carry a ``.completed``
marker or that lack ``logs/exception.txt``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_[^_]+)?_[0-9a-f]{8}$")


def find_failed_run_dirs(outputs: Path) -> list[Path]:
    failed: list[Path] = []
    for run_dir in outputs.rglob("*"):
        if not run_dir.is_dir():
            continue
        if not _RUN_DIR_RE.match(run_dir.name):
            continue
        if (run_dir / "logs" / "exception.txt").is_file():
            marker = run_dir.with_name(run_dir.name + ".completed")
            if not marker.is_file():
                failed.append(run_dir)
    return sorted(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the directories (default: dry-run).",
    )
    args = parser.parse_args(argv)

    outputs = args.outputs.resolve()
    if not outputs.is_dir():
        parser.error(f"outputs directory not found: {outputs}")

    failed = find_failed_run_dirs(outputs)
    print(f"found {len(failed)} failed run director{'y' if len(failed) == 1 else 'ies'}")
    for run_dir in failed:
        rel = run_dir.relative_to(outputs).as_posix()
        if args.apply:
            shutil.rmtree(run_dir, ignore_errors=False)
            print(f"  deleted {rel}")
        else:
            print(f"  would delete {rel}")

    if not args.apply and failed:
        print("\nre-run with --apply to delete these directories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
