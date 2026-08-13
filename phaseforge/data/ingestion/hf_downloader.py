"""HuggingFace provisioning of raw robomimic dataset artifacts.

The ingestion FSM stays fail-closed by default (``VALIDATE_SOURCE`` raises
when the raw directory is missing). Setting ``data.source.auto_download=true``
routes the FSM through the ``PROVISION_SOURCE`` state, which downloads the
configured artifact from the HuggingFace mirror and verifies its SHA-256
(pinned in config, else the mirror's own LFS metadata) before the ingester
ever touches it.

Idempotency contract: an existing file whose checksum matches is left
untouched; a checksum mismatch is a hard error and is never silently
overwritten.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_mirror_sha256(repo_id: str, path: str) -> str | None:
    """Return the mirror's recorded SHA-256 for a dataset file (LFS), if any."""
    try:
        from huggingface_hub import get_paths_info
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for auto-download provisioning; "
            "install it or provision the dataset manually."
        ) from exc
    try:
        infos = get_paths_info(repo_id=repo_id, paths=path, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not read HuggingFace metadata for {repo_id}/{path}: {exc}"
        ) from exc
    if not infos:
        return None
    lfs = getattr(infos[0], "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    return str(sha) if sha else None


def download_hf_file(
    repo_id: str,
    path: str,
    dest_dir: str | Path,
    pinned_sha256: str | None = None,
) -> Path:
    """Download a dataset file into ``dest_dir`` (flat basename), SHA-verified.

    The expected checksum is the pinned value when given, else the mirror's
    own LFS metadata; when neither is available the file is sanity-checked as
    a readable HDF5 archive. An existing file with a matching checksum is
    returned untouched; a checksum mismatch is a hard error (never an
    overwrite).

    Returns:
        The verified destination path.

    Raises:
        RuntimeError: Download failure, checksum mismatch, or unreadable file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required for auto-download provisioning; "
            "install it or provision the dataset manually."
        ) from exc

    dest_dir = Path(dest_dir)
    dest = dest_dir / Path(path).name
    dest_dir.mkdir(parents=True, exist_ok=True)

    expected = pinned_sha256 or fetch_mirror_sha256(repo_id, path)

    if dest.exists():
        if expected is not None:
            actual = _sha256_of(dest)
            if actual == expected:
                logger.info("  already provisioned: %s (SHA-256 verified)", dest)
                return dest
            raise RuntimeError(
                f"Existing file {dest} has SHA-256 {actual}, expected {expected}. "
                "Refusing to overwrite — remove or repair it manually."
            )
        logger.info("  already provisioned (no checksum available): %s", dest)
        return dest

    logger.info("  downloading %s/%s -> %s", repo_id, path, dest)
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=path, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download {repo_id}/{path} from HuggingFace: {exc}"
        ) from exc
    shutil.copy2(cached, dest)

    if expected is not None:
        actual = _sha256_of(dest)
        if actual != expected:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file {dest} has SHA-256 {actual}, expected "
                f"{expected}. The mirror artifact changed — refusing to ingest."
            )
        logger.info("  downloaded %s (SHA-256 verified: %s)", dest.name, actual)
    else:
        try:
            import h5py

            with h5py.File(dest, "r"):
                pass
        except Exception as exc:  # noqa: BLE001
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file {dest} is not a readable HDF5 file: {exc}"
            ) from exc
        logger.info("  downloaded %s (HDF5 sanity check passed)", dest.name)

    return dest