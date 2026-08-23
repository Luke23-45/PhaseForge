"""Run discovery: map (namespace, task, method, seed, stage) -> run directory.

Layout produced by the runner (verified against real sweeps)::

    <root>/<model_name>/stage<N>/seed<S>/<ts>_<tag>_<runid>/    train runs
    <root>/eval/<model_name>/seed<S>/<ts>_<tag>_<runid>/        eval runs

Each train/eval run carries ``run_meta.json``; completion is marked by a
``.completed`` sibling file. Discovery is fail-closed: duplicate completed
runs for one cell and unreadable metadata abort with the offending paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from studies.analysis.common import io as cio

_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<tag>.+?)_(?P<runid>[0-9a-f]{8})$")


@dataclass(frozen=True)
class TrainRun:
    namespace: str
    model_name: str
    task: str | None
    method: str
    seed: int
    stage: int
    path: Path
    git_commit: str | None
    config_hash: str | None
    tag: str | None

    @property
    def key(self) -> tuple[str, str | None, str, int, int]:
        return (self.namespace, self.task, self.method, self.seed, self.stage)


@dataclass(frozen=True)
class EvalRun:
    namespace: str
    model_name: str
    task: str | None
    method: str
    seed: int
    path: Path
    git_commit: str | None
    tag: str | None

    @property
    def key(self) -> tuple[str, str | None, str, int]:
        return (self.namespace, self.task, self.method, self.seed)


def _parse_dir_name(name: str) -> tuple[str, str] | None:
    m = _DIR_RE.match(name)
    if m is None:
        return None
    return m.group("tag"), m.group("runid")


def _task_from_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    return tag.split("__", 1)[0] or None


def _resolve_task(namespace: str, method: str, tag: str | None) -> str | None:
    """Task of a run, resolved through the manifest (name, tag) identity.

    Parsing the tag prefix alone mis-keys task-less manifests whose rows carry
    variant tags (e.g. lift_ablation's bc_robot_only with tag 'robot_only'):
    the tag is not a task there. The manifest is authoritative; the prefix
    fallback only covers stray runs no manifest row claims.
    """
    from studies.analysis.common import registry

    for m in registry.methods(namespace):
        if m.name == method and (m.tag or None) == (tag or None):
            return m.task
    return _task_from_tag(tag)


def scan_namespace(namespace: str, root: Path) -> tuple[list[TrainRun], list[EvalRun]]:
    """Discover every completed run under one output namespace."""
    if not root.is_dir():
        raise ValueError(f"Namespace root {root} does not exist (namespace {namespace!r})")

    train_runs: list[TrainRun] = []
    eval_runs: list[EvalRun] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "eval"):
        _scan_train_tree(namespace, run_dir, train_runs)
    eval_root = root / "eval"
    if eval_root.is_dir():
        for model_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
            _scan_eval_tree(namespace, model_dir, eval_runs)
    return train_runs, eval_runs


def _meta_or_skip(run_dir: Path) -> dict | None:
    """Return run_meta for a completed run, else None (incomplete runs are skipped)."""
    if not run_dir.is_dir() or not (run_dir.parent / f"{run_dir.name}.completed").exists():
        return None
    try:
        meta = cio.read_json(run_dir / "run_meta.json")
    except ValueError:
        return None
    return meta if isinstance(meta, dict) else None


def _scan_train_tree(namespace: str, model_dir: Path, out: list[TrainRun]) -> None:
    model_name = model_dir.name
    for stage_dir in sorted(model_dir.glob("stage*")):
        m = re.fullmatch(r"stage(\d+)", stage_dir.name)
        if m is None or not stage_dir.is_dir():
            continue
        stage = int(m.group(1))
        for seed_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                meta = _meta_or_skip(run_dir)
                if meta is None:
                    continue
                tag = meta.get("tag")
                out.append(
                    TrainRun(
                        namespace=namespace,
                        model_name=str(meta.get("model_name", model_name)),
                        task=_resolve_task(namespace, str(meta.get("method", "")), tag),
                        method=str(meta.get("method", "")),
                        seed=int(meta["seed"]) if "seed" in meta else -1,
                        stage=stage if "stage" in meta else int(meta.get("stage", stage)),
                        path=run_dir,
                        git_commit=meta.get("git_commit"),
                        config_hash=meta.get("config_hash"),
                        tag=tag,
                    )
                )


def _scan_eval_tree(namespace: str, model_dir: Path, out: list[EvalRun]) -> None:
    model_name = model_dir.name
    for seed_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            meta = _meta_or_skip(run_dir)
            if meta is None:
                continue
            tag = meta.get("tag")
            out.append(
                EvalRun(
                    namespace=namespace,
                    model_name=str(meta.get("model_name", model_name)),
                    task=_resolve_task(namespace, str(meta.get("method", "")), tag),
                    method=str(meta.get("method", "")),
                    seed=int(meta["seed"]) if "seed" in meta else -1,
                    path=run_dir,
                    git_commit=meta.get("git_commit"),
                    tag=tag,
                )
            )


def assert_no_duplicates(train_runs: list[TrainRun], eval_runs: list[EvalRun]) -> None:
    """Fail closed on two completed runs claiming the same cell."""
    seen: dict[tuple, Path] = {}
    for run in train_runs:
        if run.key in seen:
            raise ValueError(
                f"Duplicate completed train run for {run.key}: {seen[run.key]} and {run.path}"
            )
        seen[run.key] = run.path
    seen_eval: dict[tuple, Path] = {}
    for run in eval_runs:
        if run.key in seen_eval:
            raise ValueError(
                f"Duplicate completed eval run for {run.key}: {seen_eval[run.key]} and {run.path}"
            )
        seen_eval[run.key] = run.path
