"""Config-hash-based persistent cache manager."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


class CacheManager:
    """Manages the persistent on-disk cache for processed dataset.

    Cache layout::

        cache_root/
        └── {config_hash}/
            ├── manifest.json
            ├── norm_stats.pt
            ├── splits.json
            └── trajectories/
                ├── 000000.pt   # {"state": (T,S), "action": (T,A), "phase": (T,), "task_id": int}
                └── ...

    The config_hash is the SHA-256 of the canonical YAML of the data config.
    Any change to phase thresholds, state keys, or split ratios invalidates the cache.
    """

    def __init__(self, cache_root: Path) -> None:
        self.cache_root = Path(cache_root)

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    @staticmethod
    def compute_hash(data_cfg: DictConfig) -> str:
        """SHA-256 of the data config YAML (first 16 chars)."""
        yaml_str = OmegaConf.to_yaml(data_cfg, resolve=True)
        return hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()[:16]

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
    ) -> None:
        """Atomically write all processed data to the cache.

        Writes to a tmp directory first, then renames to the final path
        to prevent partial cache corruption.

        Args:
            task_index: Optional ``{task_name: int_id}`` mapping to persist
                alongside the cache for auditability.
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

        # Manifest
        manifest = {
            "config_hash": config_hash,
            "num_trajectories": len(trajectories),
            "splits": {k: len(v) for k, v in splits.items()},
            "num_tasks": len(task_index) if task_index is not None else None,
            "created_at": time.time(),
            "complete": True,
        }
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
