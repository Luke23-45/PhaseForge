"""Config-hash-based persistent cache manager with full provenance tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

#: The pinned LIBERO mirror this project consumes (see download_libero.py).
DATASET_REPO = "yifengzhu-hf/LIBERO-datasets"


def git_commit() -> str:
    """Best-effort HEAD SHA of the repo containing this module ('' when N/A)."""
    try:
        repo = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (handles multi-GB HDF5 files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CacheManager:
    """Manages the persistent on-disk cache for processed dataset.

    Cache layout::

        cache_root/
        └── {config_hash}/
            ├── manifest.json
            ├── norm_stats.pt
            ├── splits.json
            ├── task_index.json
            ├── train_tasks.txt
            ├── validation_tasks.txt
            └── trajectories/
                ├── 000000.pt   # {"state": (T,S), "action": (T,A), "phase": (T,), "task_id": int}
                └── ...

    The config_hash is the SHA-256 of the canonical YAML of the data config
    PLUS a cheap provenance context (git commit, object-index content hash,
    raw dataset file names/sizes). Any change to phase thresholds, state
    keys, split ratios, code revision, the object index, or the raw dataset
    invalidates the cache. Full content provenance (per-file SHA-256,
    schema, task names) is persisted in ``manifest.json`` so a result can
    be audited later.
    """

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    @staticmethod
    def provenance_context(data_cfg: DictConfig) -> dict:
        """Cheap inputs to the cache identity (never reads file contents).

        Folds in code revision and data identity so a stale cache cannot be
        silently reused after the raw dataset or object index changes.
        Content-level hashes (expensive) live in the manifest instead.
        """
        ctx: dict[str, Any] = {"git_commit": git_commit()}

        # Object-index content hash (small file, safe to read).
        try:
            oscfg = data_cfg.get("object_state")
            if oscfg is None or oscfg.get("enabled", True):
                from phaseforge.data.paths import resolve_object_index_path

                p = resolve_object_index_path(data_cfg)
                if p.exists():
                    ctx["object_index_sha256"] = sha256_bytes(p.read_bytes())
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not hash the object index for the cache identity — "
                "the cache key will not depend on it.",
                exc_info=True,
            )

        # Raw dataset identity: names + sizes (cheap). Content hashes are
        # recorded in the manifest during save().
        try:
            suite = data_cfg.get("libero", {}).get("suite")
            if suite:
                from phaseforge.data.paths import libero_suite_dir

                suite_dir = libero_suite_dir(str(suite))
                if not suite_dir.exists():
                    logger.warning(
                        "Raw suite dir %s not found while computing the "
                        "cache identity — the cache key will not include "
                        "the raw dataset. If the data exists elsewhere, "
                        "set PHASEFORGE_DATA_DIR.",
                        suite_dir,
                    )
                files = sorted(suite_dir.glob("*.hdf5"))
                ctx["raw_files"] = [
                    {
                        "name": p.name,
                        "size": p.stat().st_size,
                        "mtime_ns": p.stat().st_mtime_ns,
                    }
                    for p in files
                ]
        except Exception:  # noqa: BLE001
            logger.warning(
                "Could not fingerprint the raw dataset for the cache "
                "identity — the cache key will not depend on it.",
                exc_info=True,
            )
        return ctx

    @staticmethod
    def compute_hash(data_cfg: DictConfig, extra_context: dict | None = None) -> str:
        """SHA-256 of the data config YAML + provenance context (first 16 chars).

        ``extra_context`` (if given) is merged over the automatic
        :meth:`provenance_context` so callers can pin additional identity
        inputs.
        """
        context = CacheManager.provenance_context(data_cfg)
        if extra_context:
            context = {**context, **extra_context}
        payload = OmegaConf.to_yaml(data_cfg, resolve=True)
        if context:
            payload += "\n" + json.dumps(context, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def cache_dir(self, config_hash: str) -> Path:
        return self.cache_root / config_hash

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def cache_exists(self, config_hash: str) -> bool:
        """Return True only if the cache directory and a valid manifest exist."""
        manifest = self.cache_dir(config_hash) / "manifest.json"
        if not manifest.exists():
            return False
        try:
            meta = json.loads(manifest.read_text())
            return meta.get("complete", False)
        except (json.JSONDecodeError, KeyError):
            return False

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        config_hash: str,
        trajectories: list[dict[str, Any]],
        norm_stats: dict[str, torch.Tensor],
        splits: dict[str, list[int]],
        task_index: dict[str, int] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        """Atomically write all processed data to the cache.

        Writes to a tmp directory first, then renames to the final path
        to prevent partial cache corruption.

        Args:
            task_index: Optional ``{task_name: int_id}`` mapping to persist
                alongside the cache for auditability.
            provenance: Optional dict with the audit trail recorded in
                ``manifest.json``: dataset repo/commit, per-file SHA-256,
                object-index hash, code git commit, state schema,
                normalization method, phase-labeler config and split task
                names (the latter are also written as human-readable
                ``train_tasks.txt`` / ``validation_tasks.txt``).
        """
        final_dir = self.cache_dir(config_hash)
        tmp_dir = self.cache_root / f"{config_hash}_tmp"

        # Clean up any existing tmp dir
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)

        # Trajectories
        traj_dir = tmp_dir / "trajectories"
        traj_dir.mkdir()
        for idx, traj in enumerate(trajectories):
            torch.save(traj, traj_dir / f"{idx:06d}.pt")

        # Norm stats
        torch.save(norm_stats, tmp_dir / "norm_stats.pt")

        # Splits (indices into the trajectories list)
        (tmp_dir / "splits.json").write_text(json.dumps(splits, indent=2))

        # Task index (deterministic name -> id; auditable)
        if task_index is not None:
            (tmp_dir / "task_index.json").write_text(json.dumps(task_index, indent=2))

        # Human-readable split task names (Gate 7 audit)
        split_task_names = (
            (provenance or {}).get("split_task_names") if provenance else None
        )
        if split_task_names:
            for fname, key in (
                ("train_tasks.txt", "train"),
                ("validation_tasks.txt", "val"),
            ):
                names = sorted(split_task_names.get(key, []))
                (tmp_dir / fname).write_text(
                    "\n".join(names) + ("\n" if names else ""), encoding="utf-8"
                )

        # Manifest
        manifest = {
            "config_hash": config_hash,
            "num_trajectories": len(trajectories),
            "splits": {k: len(v) for k, v in splits.items()},
            "num_tasks": len(task_index) if task_index is not None else None,
            "created_at": time.time(),
            "complete": True,
        }
        if provenance:
            manifest["provenance"] = provenance
        (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Atomic rename
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.move(str(tmp_dir), str(final_dir))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        config_hash: str,
    ) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, list[int]], dict[str, int]]:
        """Load all data from the cache.

        Returns:
            ``(trajectories, norm_stats, splits, task_index)``.
            ``task_index`` is ``{}`` if the cache predates task-index
            persistence (loaded from an old cache that lacks task_index.json).
        """
        d = self.cache_dir(config_hash)
        traj_dir = d / "trajectories"

        traj_files = sorted(traj_dir.glob("*.pt"))
        trajectories = [torch.load(f, weights_only=False) for f in traj_files]
        norm_stats = torch.load(d / "norm_stats.pt", weights_only=False)
        splits = json.loads((d / "splits.json").read_text())

        task_index_path = d / "task_index.json"
        if task_index_path.exists():
            task_index = json.loads(task_index_path.read_text())
        else:
            task_index = {}

        return trajectories, norm_stats, splits, task_index
