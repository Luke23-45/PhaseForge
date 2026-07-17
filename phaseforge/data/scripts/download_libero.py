"""Reproducible LIBERO dataset download from the HuggingFace mirror.

This is a standalone CLI — it does NOT depend on Hydra or on the data FSM.
It can be run directly on a cloud box or locally, and it is also importable
by :mod:`phaseforge.data.ingestion.state_machine` so the FSM can fetch raw
data on demand.

Why a dedicated script (instead of the official
``download_libero_datasets.py``)
------------------------------------------------------------------
1. The official script uses ``box.com`` ZIP links that the maintainers
   themselves flag as "may expire soon", and it prompts via ``input()``
   which deadlocks unattended cloud jobs.
2. The official HF path passes ``local_dir_use_symlinks=False`` to
   ``snapshot_download`` — that parameter was removed from modern
   ``huggingface_hub`` (verified: not in the 1.x signature), so the
   official snippet no longer runs on current hub versions.
3. This script is non-interactive, idempotent, resumable, writes a
   provenance manifest, and verifies the official file-count integrity
   check (90 files for libero_90, 10 for libero_10).

Verified facts driving this implementation
------------------------------------------
- HF repo: ``yifengzhu-hf/LIBERO-datasets`` (public).
- It exposes two top-level folders relevant to this project:
  ``libero_90/`` (90 task HDF5 files) and ``libero_10/`` (10 task HDF5
  files, i.e. LIBERO-Long). Confirmed via the HF tree API.
- Every file in both folders ends in ``_demo.hdf5`` (100% of files,
  zero exceptions — verified on a full 90-file scan).
- ``huggingface_hub.snapshot_download`` accepts ``allow_patterns``,
  ``local_dir``, ``revision``, ``repo_type="dataset"``, ``force_download``.
  ``local_dir_use_symlinks`` is NOT a valid kwarg on hub >= 1.0.
- ``HfApi.dataset_info(...).sha`` returns the resolved commit SHA of the
  revision (verified in the DatasetInfo dataclass docstring + field list).
- The official LIBERO integrity check is by FILE COUNT, not SHA-256
  (see ``check_libero_dataset`` in ``download_utils.py``). We keep that
  as the required check and add SHA-256 only as an optional, on-demand
  supplement (off by default — it is expensive over 66 GB).

Usage
-----
::

    python -m phaseforge.data.scripts.download_libero \\
        --suites libero_90 libero_10

    # Pin to a specific revision for full reproducibility
    python -m phaseforge.data.scripts.download_libero --revision f13aa24a

    # Custom data root (overrides PHASEFORGE_DATA_DIR)
    python -m phaseforge.data.scripts.download_libero --data-root /mnt/data

    # Re-download even if files exist
    python -m phaseforge.data.scripts.download_libero --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from phaseforge.data.paths import (
    DATA_DIR_ENV_VAR,
    EXPECTED_FILE_COUNTS,
    LIBERO_LONG_DIRNAME,
    LIBERO_90_DIRNAME,
    get_data_root,
    libero_manifest_path,
    libero_raw_root,
    libero_suite_dir,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — all verified, not guessed
# ---------------------------------------------------------------------------

#: HuggingFace dataset repository id for the LIBERO mirror.
HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"

#: Default revision to download. ``None`` means "current main". For a
#: paper-grade reproducible pin, pass ``--revision <sha>``.
DEFAULT_REVISION: str | None = None

#: All suites this script knows how to fetch.
ALL_SUITES: tuple[str, ...] = (LIBERO_90_DIRNAME, LIBERO_LONG_DIRNAME)

#: Chunk size for optional SHA-256 streaming.
_SHA_CHUNK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _resolve_revision(revision: str | None) -> tuple[str, str]:
    """Resolve a revision to a concrete commit SHA and its human label.

    Args:
        revision: A branch/tag/commit, or ``None`` for the default branch.

    Returns:
        ``(resolved_sha, requested_label)`` where ``resolved_sha`` is the
        full 40-char commit the download will be pinned to, and
        ``requested_label`` is the original revision string (or ``"main"``).

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - explicit dependency error
        raise ImportError(
            "huggingface_hub is required to download from the HF mirror. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    label = revision if revision is not None else "main"
    info = HfApi().dataset_info(repo_id=HF_REPO_ID, revision=revision)
    # `info.sha` is documented as "Repo SHA at this particular revision".
    if not getattr(info, "sha", None):
        raise RuntimeError(
            f"dataset_info returned no commit SHA for {HF_REPO_ID}@{label}."
        )
    return info.sha, label


def download_suites(
    suites: list[str],
    data_root: str | Path | None = None,
    revision: str | None = None,
    force: bool = False,
) -> str:
    """Download one or more LIBERO suites from the HF mirror.

    Idempotent and resumable: ``snapshot_download`` caches partial files in
    its internal cache and skips already-fetched blobs, so re-running after
    an interruption continues where it left off. With ``force=True`` the
    local destination is cleared first and ``force_download=True`` is passed.

    Args:
        suites: Subset of :data:`ALL_SUITES` (e.g. ``["libero_90"]``).
        data_root: Override forwarded to :func:`get_data_root`.
        revision: HF revision (branch/tag/sha) or ``None`` for default.
        force: If True, re-download even if files already exist locally.

    Returns:
        The resolved commit SHA that was downloaded.

    Raises:
        ValueError: If an unknown suite is requested.
        ImportError: If ``huggingface_hub`` is not installed.
    """
    _validate_suites(suites)

    resolved_sha, label = _resolve_revision(revision)
    logger.info("HF repo %s @ %s resolved to commit %s", HF_REPO_ID, label, resolved_sha)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc

    root = get_data_root(data_root)
    libero_root = libero_raw_root(root)
    libero_root.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        suite_dir = libero_suite_dir(suite, root)
        if force and suite_dir.exists():
            logger.info("Removing existing %s (force=True)", suite_dir)
            import shutil

            shutil.rmtree(suite_dir)
        suite_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading suite '%s' -> %s (this is resumable)", suite, suite_dir
        )
        # allow_patterns targets only this suite's folder, so unrelated suites
        # in the repo (libero_goal, libero_object, libero_spatial) are skipped.
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            revision=resolved_sha,
            local_dir=str(root / "raw" / "libero"),
            allow_patterns=[f"{suite}/*"],
            force_download=force,
        )

        _verify_file_count(suite, suite_dir)

    _write_manifest(
        root=root,
        suites=suites,
        resolved_sha=resolved_sha,
        revision_label=label,
    )
    return resolved_sha


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------


def _validate_suites(suites: list[str]) -> None:
    bad = [s for s in suites if s not in ALL_SUITES]
    if bad:
        raise ValueError(
            f"Unknown suite(s): {bad}. Expected subset of {list(ALL_SUITES)}."
        )
    if not suites:
        raise ValueError("No suites requested.")


def _verify_file_count(suite: str, suite_dir: Path) -> None:
    """Apply the official LIBERO file-count integrity check.

    The official ``check_libero_dataset`` verifies that ``libero_90`` has
    exactly 90 HDF5 files and ``libero_10`` has exactly 10. We mirror that
    rather than inventing a new rule.
    """
    hdf5_files = sorted(suite_dir.glob("*.hdf5"))
    got = len(hdf5_files)
    expected = EXPECTED_FILE_COUNTS[suite]
    if got != expected:
        raise RuntimeError(
            f"Integrity check FAILED for {suite}: expected {expected} "
            f".hdf5 files, found {got} in {suite_dir}."
        )
    logger.info("Integrity OK: %s has %d .hdf5 files", suite, got)


def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file (streamed, constant memory)."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_SHA_CHUNK), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------


def _write_manifest(
    root: Path,
    suites: list[str],
    resolved_sha: str,
    revision_label: str,
) -> Path:
    """Write ``MANIFEST.json`` recording exactly what was downloaded."""
    manifest_path = libero_manifest_path(root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    suites_meta: dict[str, dict] = {}
    for suite in suites:
        suite_dir = libero_suite_dir(suite, root)
        files = sorted(suite_dir.glob("*.hdf5"))
        suites_meta[suite] = {
            "expected_files": EXPECTED_FILE_COUNTS[suite],
            "actual_files": len(files),
            "total_bytes": sum(f.stat().st_size for f in files),
            "files": [
                {
                    "name": f.name,
                    "size_bytes": f.stat().st_size,
                }
                for f in files
            ],
        }

    manifest = {
        "source": "huggingface",
        "repo_id": HF_REPO_ID,
        "revision_requested": revision_label,
        "commit_sha": resolved_sha,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "suites": suites_meta,
        "integrity_check": "file_count (mirrors official LIBERO check_libero_dataset)",
        "schema_version": 1,
    }

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)
    logger.info("Wrote manifest -> %s", manifest_path)
    return manifest_path


def load_manifest(data_root: str | Path | None = None) -> dict:
    """Load and return the download manifest as a dict.

    Raises:
        FileNotFoundError: If no manifest exists yet.
    """
    path = libero_manifest_path(data_root)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_libero",
        description=(
            "Reproducibly download LIBERO task suites from the HuggingFace "
            f"mirror ({HF_REPO_ID}). Non-interactive, resumable, integrity-"
            "checked. Data root resolves from --data-root, then the "
            f"{DATA_DIR_ENV_VAR} env var, then ./data."
        ),
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=list(ALL_SUITES),
        choices=list(ALL_SUITES),
        help=(
            "Which suites to download. Defaults to both libero_90 (train) "
            "and libero_10 (LIBERO-Long, eval)."
        ),
    )
    p.add_argument(
        "--data-root",
        default=None,
        help=(
            f"Data root directory. Overrides ${DATA_DIR_ENV_VAR} and the "
            f"./data default."
        ),
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=(
            "HF revision (branch/tag/commit SHA) to pin. Defaults to the "
            "repo's default branch. Pin a SHA for paper-grade reproducibility."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist locally.",
    )
    p.add_argument(
        "--sha256",
        action="store_true",
        help=(
            "After download, compute and record SHA-256 for every file. "
            "Expensive (~66 GB for libero_90); off by default since the "
            "official integrity check is by file count."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t0 = time.time()
    sha = download_suites(
        suites=args.suites,
        data_root=args.data_root,
        revision=args.revision,
        force=args.force,
    )

    if args.sha256:
        _augment_manifest_with_sha256(data_root=args.data_root)

    logger.info("Done in %.1fs. Commit: %s", time.time() - t0, sha)
    return 0


def _augment_manifest_with_sha256(data_root: str | Path | None = None) -> None:
    """Recompute SHA-256 for every downloaded file and update the manifest."""
    root = get_data_root(data_root)
    manifest_path = libero_manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for suite, meta in manifest["suites"].items():
        suite_dir = libero_suite_dir(suite, root)
        for entry in meta["files"]:
            entry["sha256"] = compute_sha256(suite_dir / entry["name"])
        logger.info("SHA-256 computed for all %d files in %s", len(meta["files"]), suite)

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


if __name__ == "__main__":
    sys.exit(main())
