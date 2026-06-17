"""Hydra config resolution helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def config_to_yaml(cfg: DictConfig) -> str:
    """Serialize a DictConfig to canonical YAML string (deterministic)."""
    return OmegaConf.to_yaml(cfg, resolve=True)


def config_hash(cfg: DictConfig) -> str:
    """Compute a short SHA-256 hash of the config for cache keying.

    Args:
        cfg: A DictConfig subtree (typically the data config).

    Returns:
        16-character hex string uniquely identifying this config.
    """
    yaml_str = config_to_yaml(cfg)
    return hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()[:16]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    """Resolve a path, optionally relative to a base directory.

    Args:
        path: Absolute or relative path string.
        base: If provided and path is relative, join with this base.

    Returns:
        Resolved absolute Path.
    """
    p = Path(path)
    if not p.is_absolute() and base is not None:
        p = base / p
    return p.resolve()


def get_output_dir(cfg: DictConfig) -> Path:
    """Return the Hydra output directory from config."""
    return Path(cfg.project.output_dir)
