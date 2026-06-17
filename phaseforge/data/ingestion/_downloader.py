"""Remote file downloader with progress display and checksum verification."""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2  # seconds


def download_files(
    files: list[dict[str, Any]],
    base_url: str,
    dest_dir: Path,
) -> None:
    """Download a list of files to dest_dir if not already present.

    Args:
        files: List of dicts with keys ``filename`` and optional ``checksum_sha256``.
        base_url: URL prefix.
        dest_dir: Local directory to save files.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file_spec in files:
        filename = file_spec["filename"]
        expected_sha = file_spec.get("checksum_sha256")
        dest = dest_dir / filename

        if dest.exists():
            if expected_sha and not _verify_sha256(dest, expected_sha):
                logger.warning(
                    f"Checksum mismatch for existing {filename}. Re-downloading."
                )
                dest.unlink()
            else:
                logger.info(f"  {filename} already present. Skipping download.")
                continue

        url = base_url.rstrip("/") + "/" + filename
        _download_with_retry(url, dest, expected_sha)


def _download_with_retry(url: str, dest: Path, expected_sha: str | None) -> None:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(f"  Downloading {url} (attempt {attempt}/{_MAX_RETRIES})…")
            _stream_download(url, dest)

            if expected_sha:
                if not _verify_sha256(dest, expected_sha):
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"SHA-256 mismatch for {dest.name}. File deleted."
                    )
            logger.info(f"  Downloaded {dest.name} successfully.")
            return
        except Exception as exc:
            logger.warning(f"  Attempt {attempt} failed: {exc}")
            dest.unlink(missing_ok=True)
            if attempt < _MAX_RETRIES:
                sleep_s = _BACKOFF_BASE ** attempt
                logger.info(f"  Retrying in {sleep_s}s…")
                time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to download {url} after {_MAX_RETRIES} attempts."
    )


def _stream_download(url: str, dest: Path) -> None:
    """Stream download with a Rich progress bar."""
    tmp = dest.with_suffix(".part")
    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("download", filename=dest.name, total=None)
        with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
            total = response.headers.get("Content-Length")
            if total:
                progress.update(task, total=int(total))
            chunk_size = 1 << 20  # 1 MB
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(task, advance=len(chunk))
    tmp.rename(dest)


def _verify_sha256(path: Path, expected: str) -> bool:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest() == expected
