"""Hydra config resolution and output directory management."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


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


def git_info() -> dict[str, str]:
    """Public accessor for the current git commit/branch (eval provenance).

    Returns ``{"commit": ..., "branch": ...}`` with empty strings when git
    is unavailable or the repo cannot be resolved.
    """
    return _git_info()


def write_run_meta(output_dir: Path, cfg: DictConfig, stage: int | None = None) -> None:
    """Write a lightweight JSON metadata file for quick run inspection.

    Args:
        stage: Effective model stage. Eval runs pass the stage restored
            from the loaded checkpoint so the metadata reflects the
            evaluated artifact, not the default ``train`` group.
    """
    git = _git_info()
    meta = {
        "model_name": getattr(cfg.models, "name", cfg.models._target_.split(".")[-1]),
        "stage": cfg.train.get("stage", 1) if stage is None else stage,
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


def get_eval_output_dir(cfg: DictConfig) -> Path:
    """Construct output directory for evaluation runs.

    Returns::

        {project_root}/outputs/eval/{model_name}/{timestamp}[_{tag}]_{run_id}/

    Separated from training outputs to avoid collisions under ``stage1/``.
    """
    base = _project_root() / cfg.project.output_dir
    model_name = getattr(cfg.models, "name", cfg.models._target_.split(".")[-1])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = generate_run_id()
    tag = cfg.project.get("tag", None)

    if tag:
        run_dir = f"{timestamp}_{tag}_{run_id}"
    else:
        run_dir = f"{timestamp}_{run_id}"

    return (base / "eval" / model_name / run_dir).resolve()


@dataclass
class CheckpointInfo:
    """Lightweight metadata about a discovered checkpoint."""

    path: Path
    """Absolute path to ``checkpoint_best.pt`` (or periodic checkpoint)."""

    model_name: str
    """Model name (e.g. ``phaseforge``, ``bc``)."""

    stage: int
    """Training stage (1 or 2)."""

    run_dir: str
    """Run directory name (e.g. ``2026-07-17_12-00-00_a1b2c3d4``)."""

    timestamp: str
    """Timestamp portion of the run directory name."""

    run_id: str
    """Collision-safe hex run ID."""

    config_hash: str | None = None
    """Config hash loaded from ``run_meta.json``, if available."""

    tag: str | None = None
    """User-provided tag from ``run_meta.json``, if available."""

    seed: int | None = None
    """Training seed from ``run_meta.json``, if available."""


def resolve_checkpoint_source(model_name: str) -> str:
    """Map a model name to the source model for Stage 1 checkpoint lookup.

    Some models share a pretrained encoder with another model and should
    therefore look for that model's Stage 1 checkpoint.  For example,
    ``warmstart_moe`` was pretrained *without* a phase head (via ``BC``),
    so its Stage 1 checkpoint lives under ``outputs/bc/stage1/``.

    The 2x2 factorial (C1) cells and the teacher-forced cell (E8) follow
    the same pattern:

    * ``warmstart_moe``, ``plain_encoder_phase_bootstrap`` -> plain BC
      checkpoint (``bc``).
    * ``phase_pretrain_random_router``, ``teacher_forced`` -> phaseforge's
      phase-supervised checkpoint (``phaseforge``) — shared pretraining,
      so only the Stage 2 supervision regime differs (locked E8 decision).

    Returns the model name to query, which may be different from the input.
    """
    alias_map: dict[str, str] = {
        "warmstart_moe": "bc",
        "plain_encoder_phase_bootstrap": "bc",
        "phase_pretrain_random_router": "phaseforge",
        "teacher_forced": "phaseforge",
    }
    return alias_map.get(model_name, model_name)


def scan_checkpoints(
    model_name: str,
    stage: int = 1,
    base: str | Path = "outputs",
) -> list[CheckpointInfo]:
    """Scan all run directories for a *model+stage* and return checkpoint info.

    Returns a list of :class:`CheckpointInfo` entries sorted newest-first
    by run directory name.  Each entry includes the checkpoint path plus
    metadata parsed from the directory structure and ``run_meta.json``.

    Returns an empty list if no matching runs exist.
    """
    base_dir = _project_root() / Path(base) / model_name / f"stage{stage}"
    if not base_dir.is_dir():
        return []

    checkpoints: list[CheckpointInfo] = []
    runs = sorted(base_dir.iterdir(), reverse=True)

    for run in runs:
        if not run.is_dir():
            continue

        ckpt_path = run / "checkpoints" / "checkpoint_best.pt"
        if not ckpt_path.is_file():
            continue

        # Parse run directory name: timestamp[_tag]_run_id
        # Format generated by get_output_dir():
        #   no tag:  YYYY-MM-DD_HH-MM-SS_XXXXXXXX
        #   with tag: YYYY-MM-DD_HH-MM-SS_<tag>_XXXXXXXX
        # where XXXXXXXX is always an 8-char hex run_id.
        run_dir_name = run.name
        tail = run_dir_name.rsplit("_", 1)
        if len(tail) == 2 and len(tail[1]) == 8:
            run_id = tail[1]
            head_parts = tail[0].split("_", 2)
            if len(head_parts) >= 2:
                timestamp = f"{head_parts[0]}_{head_parts[1]}"
                tag = "_".join(head_parts[2:]) if len(head_parts) > 2 else None
            else:
                timestamp = tail[0]
                tag = None
        else:
            # Fallback for legacy or non-standard naming
            run_id = ""
            timestamp = run_dir_name
            tag = None

        # Load metadata from run_meta.json when available
        config_hash: str | None = None
        meta_tag: str | None = tag
        meta_seed: int | None = None
        meta_path = run / "run_meta.json"
        if meta_path.is_file():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                config_hash = meta.get("config_hash")
                meta_tag = meta.get("tag") or tag
                seed_val = meta.get("seed")
                meta_seed = int(seed_val) if isinstance(seed_val, int) else None
            except (json.JSONDecodeError, OSError):
                pass

        checkpoints.append(CheckpointInfo(
            path=ckpt_path.resolve(),
            model_name=model_name,
            stage=stage,
            run_dir=run_dir_name,
            timestamp=timestamp,
            run_id=run_id,
            config_hash=config_hash,
            tag=meta_tag,
            seed=meta_seed,
        ))

    return checkpoints


def validate_checkpoint(path: str | Path) -> bool:
    """Verify that a checkpoint file is loadable and contains expected keys.

    Checks:
    * File exists and is non-empty
    * Can be loaded by ``torch.load``
    * Contains the minimum required keys (``model_state_dict``, ``epoch``)

    Returns ``True`` if the checkpoint is valid, ``False`` otherwise
    (with a warning logged).
    """
    p = Path(path)
    if not p.is_file():
        logger.warning("Checkpoint not found: %s", p)
        return False
    if p.stat().st_size == 0:
        logger.warning("Checkpoint is empty: %s", p)
        return False
    try:
        import torch

        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        required = {"model_state_dict", "epoch"}
        missing = required - set(ckpt.keys())
        if missing:
            logger.warning("Checkpoint %s missing keys: %s", p, missing)
            return False
        return True
    except Exception as exc:
        logger.warning("Failed to load checkpoint %s: %s", p, exc)
        return False


def find_latest_checkpoint(
    model_name: str,
    stage: int = 1,
    base: str | Path = "outputs",
    resolve_alias: bool = True,
    seed: int | None = None,
    require_seed: bool = False,
) -> Path | None:
    """Find the most recent *best* checkpoint for a model+stage combo.

    Delegates to :func:`scan_checkpoints` and :func:`resolve_checkpoint_source`
    so that alias handling (e.g. ``warmstart_moe`` → ``bc``) is centralised.

    Args:
        model_name: Model name (e.g. ``phaseforge``, ``bc``).
        stage: Training stage (1 or 2).
        base: Relative or absolute base output directory.
        resolve_alias: If ``True``, apply :func:`resolve_checkpoint_source`
            so that models sharing a pretrained encoder find the correct
            checkpoint.  Set to ``False`` when the caller has already
            performed resolution.
        seed: If provided, prefer a checkpoint whose ``run_meta.json``
            records this training seed (multi-seed runs). Without
            ``require_seed``, falls back to the newest run when no seed match
            exists (e.g. legacy runs written before seeds were recorded).
        require_seed: If ``True`` and ``seed`` is provided, a missing
            seed-specific checkpoint raises :class:`FileNotFoundError`
            instead of silently falling back to a different seed — the
            multi-seed protocol must never mix seeds (Stage 2 for seed 43
            silently loading Stage 1 from seed 42 corrupts the sweep).

    Returns:
        Absolute path to the latest ``checkpoint_best.pt``, or ``None``.

    Raises:
        FileNotFoundError: When ``require_seed`` is set and no checkpoint
            records the requested seed.
    """
    source = resolve_checkpoint_source(model_name) if resolve_alias else model_name
    checkpoints = scan_checkpoints(source, stage, base)
    if not checkpoints:
        return None
    if seed is not None:
        for info in checkpoints:
            if info.seed == seed:
                return info.path
        if require_seed:
            base_dir = _project_root() / Path(base) / source / f"stage{stage}"
            raise FileNotFoundError(
                f"No Stage {stage} checkpoint found for '{source}' with seed "
                f"{seed} under {base_dir}. The multi-seed protocol requires a "
                "seed-exact checkpoint; refusing to silently fall back to a "
                "different seed."
            )
    return checkpoints[0].path
