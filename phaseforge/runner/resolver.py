"""Resolve run directories and checkpoints for the runner.

Two sources of truth, in order of preference:

1. The runner state registry — the *exact* run directory/checkpoint a stage
   produced (deterministic even after later runs are added).
2. A strict seed+tag filesystem scan of the output tree (covers runs
   launched manually, outside the runner).

A run directory is an eligible artifact only if it carries the
``<run_dir>.completed`` sibling marker that ``RunWriter.mark_completed``
writes at the very end of a successful run — ``run_meta.json`` alone proves
nothing (it is written when a run *starts*, so crashed runs have it too).
The scan therefore can never select a partial run whose early checkpoints
survived a crash.

The tag disambiguates variants that share a model output tree, e.g. the
``data=robot_only`` BC cell recorded with ``project.tag=robot_only`` next to
the default BC runs under ``outputs/bc/stage1/``.

Since seeds became a directory dimension, runs live under
``{model}/stage{N}/seed{S}/{run}/``; legacy runs directly under
``{model}/stage{N}/`` are still resolved (both layouts are scanned). The
``seed`` filter is applied from ``run_meta.json`` regardless of layout, so
resolution stays seed-exact either way.

Tag semantics here are *strict*: ``tag=None`` matches only runs whose
``run_meta.json`` records no tag (the default cell must never resolve to the
``robot_only`` variant, and vice versa). This is intentionally stricter than
:func:`phaseforge.utils.config.find_latest_checkpoint`, whose ``tag=None``
means "no constraint".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from phaseforge.runner.protocol import Method
from phaseforge.runner.registry import RunnerState

_REQUIRED_CKPT_REL = "checkpoints/checkpoint_best.pt"
_SEED_DIR_RE = re.compile(r"^seed\d+$")


def is_seed_dir(path: str | Path) -> bool:
    """Return whether ``path`` is a ``seed{N}`` directory level.

    Multi-seed runs are organised as ``{model}/stage{N}/seed{S}/{run}/``;
    this predicate lets the resolver accept both that layout and the legacy
    ``{model}/stage{N}/{run}/`` layout (runs written before seeds were a
    directory dimension).
    """
    return bool(_SEED_DIR_RE.match(Path(path).name))


class CheckpointError(RuntimeError):
    """Raised when a required run directory or checkpoint cannot be resolved."""


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _tag_matches(meta_tag: Any, expected: str | None) -> bool:
    if expected is None:
        return meta_tag is None
    return meta_tag == expected


def _is_completed(run_dir: Path) -> bool:
    """A run is an eligible artifact only if the run *finished* successfully.

    ``run_meta.json`` is written when a run *starts*, so it exists for
    crashed/killed runs too; only the ``<run_dir>.completed`` sibling marker
    (written by ``RunWriter.mark_completed`` at the very end of a successful
    run) proves the run actually finished. Requiring it here guarantees the
    resolver can never select a partial run whose early checkpoints were
    saved before a crash.
    """
    return run_dir.with_name(run_dir.name + ".completed").is_file()


def _iter_runs_newest_first(search_dir: Path):
    """Collect candidate run dirs across the dual output layout, newest-first.

    Current runs live under ``stage{N}/seed{S}/{run}/``; legacy runs (written
    before seeds were a directory dimension) sit directly under ``stage{N}/``.
    Both are collected, then sorted globally by run name so the newest-first
    contract holds regardless of layout.
    """
    runs: list[Path] = []
    for child in search_dir.iterdir():
        if not child.is_dir():
            continue
        if is_seed_dir(child):
            runs.extend(sub for sub in child.iterdir() if sub.is_dir())
        else:
            runs.append(child)
    runs.sort(key=lambda p: p.name, reverse=True)
    return iter(runs)


def _find_run(
    search_dir: Path,
    model_name: str,
    kind: str,
    *,
    seed: int,
    tag: str | None,
) -> Path:
    if not search_dir.is_dir():
        raise CheckpointError(f"No {kind} runs found under {search_dir} for seed {seed}.")
    for run in _iter_runs_newest_first(search_dir):
        if not _is_completed(run):
            continue
        meta = _read_run_meta(run)
        if meta.get("seed") != seed:
            continue
        if not _tag_matches(meta.get("tag"), tag):
            continue
        return run
    raise CheckpointError(
        f"No completed {kind} run for {model_name} seed {seed}"
        + (f" with tag {tag!r}" if tag else " with no tag")
        + f" under {search_dir}."
    )


def resolve_run_dir(
    outputs_base: Path,
    model_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
) -> Path:
    """Return the newest *completed* run directory for ``model/stage`` matching
    seed+tag.

    Searches ``<outputs_base>/<model_name>/stage<stage>/`` (including
    ``seed{S}/`` sub-directories when present) newest-first and returns the
    first directory whose ``<run_dir>.completed`` marker exists and whose
    ``run_meta.json`` records the expected seed and tag. ``tag=None`` matches
    only untagged runs. Requiring the completion marker means a run that
    crashed after saving early checkpoints is never selected. Newest-first
    matters because a sweep can legitimately produce several runs for one
    ``(model, stage, seed)`` cell (e.g. a re-run after a crash); under the
    runner's "latest successful wins" policy the newest completed run is the
    intended artifact.

    Raises:
        CheckpointError: No matching run directory exists.
    """
    stage_dir = outputs_base / model_name / f"stage{stage}"
    return _find_run(
        stage_dir,
        model_name,
        f"{model_name} stage{stage}",
        seed=seed,
        tag=tag,
    )


def resolve_checkpoint_path(
    outputs_base: Path,
    method: Method,
    stage: int,
    *,
    seed: int,
    state: RunnerState | None = None,
) -> Path:
    """Resolve the absolute ``checkpoint_best.pt`` for a completed stage.

    Prefers the exact artifact recorded in the runner state; falls back to a
    strict seed+tag scan (so manually launched runs work too). The returned
    path is the one the evaluation step loads.
    """
    rel = state.get_ckpt(method.phase_key, seed, stage) if state is not None else None
    if rel:
        candidate = (outputs_base / rel).resolve()
        if candidate.is_file():
            return candidate
    run_dir = resolve_run_dir(
        outputs_base, method.model_name, stage, seed=seed, tag=method.output_tag
    )
    ckpt = run_dir / _REQUIRED_CKPT_REL
    if not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no {_REQUIRED_CKPT_REL} — the "
            "training run did not persist a best checkpoint."
        )
    return ckpt.resolve()


def resolve_stage_ckpt(
    outputs_base: Path,
    model_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
) -> Path:
    """Resolve the absolute ``checkpoint_best.pt`` of a provider run.

    This is what a stage-2 training subprocess loads via
    ``train.stage1_ckpt_path``. It reuses the strict seed+tag resolution so
    the exact, completed, *untagged* provider run is selected — never the CLI
    auto-detect (:func:`phaseforge.utils.config.find_latest_checkpoint`),
    whose ``tag=None`` means "no constraint" and can therefore pick a tagged
    sibling variant that shares the provider's output tree (e.g.
    ``bc_robot_only`` next to ``bc``), crashing the stage-2 load with a
    dimension mismatch.
    """
    run_dir = resolve_run_dir(outputs_base, model_name, stage, seed=seed, tag=tag)
    ckpt = run_dir / _REQUIRED_CKPT_REL
    if not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no {_REQUIRED_CKPT_REL} — the "
            "training run did not persist a best checkpoint."
        )
    return ckpt.resolve()


def checkpoint_exists(
    outputs_base: Path,
    model_name: str,
    stage: int,
    *,
    seed: int,
    tag: str | None = None,
) -> bool:
    """Return whether a ``checkpoint_best.pt`` exists for the exact seed+tag."""
    try:
        run_dir = resolve_run_dir(outputs_base, model_name, stage, seed=seed, tag=tag)
    except CheckpointError:
        return False
    return (run_dir / _REQUIRED_CKPT_REL).is_file()


def stage_checkpoint_relative(
    outputs_base: Path, run_dir: Path, model_name: str, stage: int
) -> str:
    """Record the run's checkpoint as a path relative to the outputs base."""
    ckpt = run_dir / _REQUIRED_CKPT_REL
    if not ckpt.is_file():
        raise CheckpointError(
            f"Run {run_dir} completed but has no {_REQUIRED_CKPT_REL} — the "
            "training run did not persist a best checkpoint."
        )
    return ckpt.relative_to(outputs_base).as_posix()


def resolve_eval_run_dir(
    outputs_base: Path,
    model_name: str,
    *,
    seed: int,
    tag: str | None = None,
) -> Path:
    """Return the newest evaluation run directory matching seed+tag.

    Evaluation runs live under ``<outputs_base>/eval/<model_name>/``.
    ``tag=None`` matches only untagged runs.
    """
    eval_dir = outputs_base / "eval" / model_name
    return _find_run(
        eval_dir,
        model_name,
        f"evaluation for {model_name}",
        seed=seed,
        tag=tag,
    )
