"""Hydra config resolution and output directory management."""

from __future__ import annotations

import functools
import hashlib
import json
import secrets
import subprocess
from datetime import datetime
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


@functools.lru_cache(maxsize=1)
def _project_root() -> Path:
    """Return the absolute project root, robust to Hydra cwd changes.

    Uses ``hydra.utils.get_original_cwd()`` when available (i.e. inside a
    ``@hydra.main`` function) so the path always anchors to where the user
    invoked the CLI, regardless of Hydra's ``chdir`` behaviour.
    """
    try:
        from hydra.utils import get_original_cwd

        return Path(get_original_cwd()).resolve()
    except (ImportError, ValueError):
        return Path.cwd().resolve()


def generate_run_id(length: int = 4) -> str:
    """Generate a short random hex string for collision-safe run identification."""
    return secrets.token_hex(length)


def _git_info() -> dict[str, str]:
    """Capture current git commit hash and branch, or empty strings on failure."""
    info: dict[str, str] = {"commit": "", "branch": ""}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return info


def write_run_meta(output_dir: Path, cfg: DictConfig) -> None:
    """Write a lightweight JSON metadata file for quick run inspection."""
    git = _git_info()
    meta = {
        "model_name": getattr(cfg.models, "name", cfg.models._target_.split(".")[-1]),
        "stage": cfg.train.get("stage", 1),
        "seed": cfg.project.get("seed", None),
        "device": cfg.project.get("device", None),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "config_hash": config_hash(cfg),
        "tag": cfg.project.get("tag", None),
    }
    path = output_dir / "run_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def get_output_dir(cfg: DictConfig) -> Path:
    """Construct the structured output directory path.

    Returns::

        {project_root}/outputs/{model_name}/stage{N}/{timestamp}[_{tag}]_{run_id}/

    The model name is read from ``cfg.models.name`` (falling back to
    the last component of ``cfg.models._target_``). The stage is read
    from ``cfg.train.stage``. An optional ``cfg.project.tag`` is
    inserted before the run-id suffix for user-friendly labelling.
    """
    base = _project_root() / cfg.project.output_dir
    model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
    stage = cfg.train.get("stage", 1)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = generate_run_id()
    tag = cfg.project.get("tag", None)

    if tag:
        run_dir = f"{timestamp}_{tag}_{run_id}"
    else:
        run_dir = f"{timestamp}_{run_id}"

    return (base / model_name / f"stage{stage}" / run_dir).resolve()


def find_latest_checkpoint(
    model_name: str,
    stage: int = 1,
    base: str = "outputs",
) -> Path | None:
    """Find the most recent *best* checkpoint for a model+stage combo.

    Scans run directories under ``{project_root}/outputs/{model_name}/stage{stage}/``
    in reverse-alphabetical order (newest timestamp first) and returns
    the first ``checkpoints/checkpoint_best.pt`` found, or ``None``.
    """
    base_dir = _project_root() / Path(base) / model_name / f"stage{stage}"
    if not base_dir.is_dir():
        return None

    runs = sorted(base_dir.iterdir(), reverse=True)
    for run in runs:
        if not run.is_dir():
            continue
        ckpt = run / "checkpoints" / "checkpoint_best.pt"
        if ckpt.is_file():
            return ckpt.resolve()
    return None
