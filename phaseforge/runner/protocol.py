"""Frozen experiment protocol: load + validate the JSON matrix, build the plan.

The protocol manifest (``experiments/lift_pilot.json``) is the single source
of truth describing every baseline method, its training stages, its Stage 1
source dependency, and whether a complete run includes the offline
evaluation. ``load_protocol`` validates it loudly so a malformed matrix
never silently produces a partial sweep.

A *plan* is the ordered list of concrete steps produced by
:func:`build_plan`: for every selected method and seed, all training stages
in order, then the evaluation of the final-stage checkpoint, with any
required pretraining stages injected first when ``with_dependencies`` is set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_VALID_STAGES = frozenset({1, 2})
_VALID_DATA = frozenset({"common", "robot_only"})


class ProtocolError(ValueError):
    """Raised when a protocol manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class Method:
    """One baseline method in the protocol."""

    index: int
    name: str
    role: str
    model: str
    data: str
    stages: tuple[int, ...]
    stage2_source: str | None
    evaluate: bool
    tag: str | None = None

    @property
    def model_name(self) -> str:
        """Filesystem model name (e.g. ``baselines/bc`` -> ``bc``)."""
        return self.model.rsplit("/", 1)[-1]

    @property
    def final_stage(self) -> int:
        """The stage whose checkpoint a complete run evaluates."""
        return self.stages[-1]

    @property
    def phase_key(self) -> str:
        """Registry key prefix for this method's phases (stable identity)."""
        return self.name


@dataclass(frozen=True)
class Protocol:
    """The validated protocol document."""

    name: str
    task: str
    description: str
    seeds: tuple[int, ...]
    defaults: tuple[str, ...]
    methods: tuple[Method, ...]

    def method_by_name(self, name: str) -> Method | None:
        for m in self.methods:
            if m.name == name:
                return m
        return None

    def method_by_index(self, index: int) -> Method | None:
        for m in self.methods:
            if m.index == index:
                return m
        return None

    def select_methods(self, refs: list[str]) -> list[Method]:
        """Resolve a list of user references to ordered :class:`Method`s.

        Each reference may be a method name or its numeric index. Unknown
        references raise a :class:`ProtocolError` listing every valid option.
        """
        selected: list[Method] = []
        seen: set[str] = set()
        for ref in refs:
            method: Method | None = None
            if ref.isdigit():
                method = self.method_by_index(int(ref))
                if method is None:
                    raise ProtocolError(
                        f"Unknown method index {ref!r}. Valid indices: "
                        f"{[m.index for m in self.methods]}."
                    )
            else:
                method = self.method_by_name(ref)
                if method is None:
                    raise ProtocolError(
                        f"Unknown method {ref!r}. Valid names: "
                        f"{[m.name for m in self.methods]}."
                    )
            if method.name not in seen:
                selected.append(method)
                seen.add(method.name)
        return sorted(selected, key=lambda m: m.index)


def _require(data: dict[str, Any], key: str, label: str) -> Any:
    if key not in data or data[key] in (None, ""):
        raise ProtocolError(f"Protocol method is missing {label} ({key!r}).")
    return data[key]


def _parse_method(raw: dict[str, Any]) -> Method:
    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol method must be a JSON object, got {type(raw).__name__}.")

    index = raw.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 1:
        raise ProtocolError(f"Method {raw.get('name', '?')!r}: 'index' must be an int >= 1.")
    name = _require(raw, "name", "a name")
    if not isinstance(name, str) or not name.replace("_", "").isalnum():
        raise ProtocolError(f"Method {index}: 'name' must be [a-z0-9_]+ (got {name!r}).")
    role = str(_require(raw, "role", "a role description"))
    model = _require(raw, "model", "a model config path")
    if not isinstance(model, str) or not model:
        raise ProtocolError(f"Method {name!r}: 'model' must be a non-empty string.")
    data = str(raw.get("data", "common"))
    if data not in _VALID_DATA:
        raise ProtocolError(
            f"Method {name!r}: 'data' must be one of {sorted(_VALID_DATA)}, got {data!r}."
        )

    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ProtocolError(f"Method {name!r}: 'stages' must be a non-empty list.")
    stages: list[int] = []
    for s in stages_raw:
        if isinstance(s, bool) or not isinstance(s, int) or s not in _VALID_STAGES:
            raise ProtocolError(
                f"Method {name!r}: stage {s!r} must be one of {sorted(_VALID_STAGES)}."
            )
        if s not in stages:
            stages.append(s)
    if stages != sorted(stages):
        raise ProtocolError(f"Method {name!r}: 'stages' must be ascending and unique.")

    stage2_source = raw.get("stage2_source")
    if stage2_source is not None:
        if stage2_source not in ("self", "bc", "phaseforge"):
            raise ProtocolError(
                f"Method {name!r}: 'stage2_source' must be null, 'self', 'bc' or 'phaseforge', "
                f"got {stage2_source!r}."
            )
        if 2 not in stages:
            raise ProtocolError(
                f"Method {name!r}: 'stage2_source' set but method has no stage 2."
            )

    evaluate = raw.get("evaluate", True)
    if not isinstance(evaluate, bool):
        raise ProtocolError(f"Method {name!r}: 'evaluate' must be a boolean.")

    tag = raw.get("tag")
    if tag is not None and (not isinstance(tag, str) or not tag):
        raise ProtocolError(f"Method {name!r}: 'tag' must be a non-empty string or null.")

    return Method(
        index=index,
        name=name,
        role=role,
        model=model,
        data=data,
        stages=tuple(stages),
        stage2_source=stage2_source,
        evaluate=evaluate,
        tag=tag,
    )


def load_protocol(path: str | Path) -> Protocol:
    """Load and strictly validate a protocol manifest.

    Raises:
        ProtocolError: The document is malformed (bad JSON, missing keys,
            duplicate indices, non-int seeds, ...).
        OSError: The file cannot be read.
    """
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Protocol {p} is not valid JSON: {exc.msg}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"Protocol {p} must be a JSON object.")

    name = _require(raw, "name", "a protocol name")
    task = str(_require(raw, "task", "a task name"))
    description = str(raw.get("description", ""))

    seeds_raw = raw.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ProtocolError(f"Protocol {name!r}: 'seeds' must be a non-empty list of ints.")
    seeds: list[int] = []
    for s in seeds_raw:
        if isinstance(s, bool) or not isinstance(s, int):
            raise ProtocolError(f"Protocol {name!r}: seed {s!r} must be an int.")
        if s not in seeds:
            seeds.append(s)

    defaults_raw = raw.get("defaults", [])
    if not isinstance(defaults_raw, list) or not all(
        isinstance(d, str) and d for d in defaults_raw
    ):
        raise ProtocolError(f"Protocol {name!r}: 'defaults' must be a list of override strings.")
    defaults = tuple(defaults_raw)

    methods_raw = raw.get("methods")
    if not isinstance(methods_raw, list) or not methods_raw:
        raise ProtocolError(f"Protocol {name!r}: 'methods' must be a non-empty list.")

    methods: list[Method] = [_parse_method(m) for m in methods_raw]
    seen_indices: dict[int, str] = {}
    seen_names: set[str] = set()
    for m in methods:
        if m.index in seen_indices:
            raise ProtocolError(
                f"Duplicate method index {m.index} ({m.name!r} vs {seen_indices[m.index]!r})."
            )
        if m.name in seen_names:
            raise ProtocolError(f"Duplicate method name {m.name!r}.")
        seen_indices[m.index] = m.name
        seen_names.add(m.name)
    methods.sort(key=lambda m: m.index)

    # Cross-checks: a stage2_source provider must be a real method, and the
    # provided stage 1 must come from a single-task "common" run (no tag) —
    # otherwise dependency injection would point at the wrong output tree.
    providers = {m.name for m in methods}
    for m in methods:
        if m.stage2_source in ("bc", "phaseforge"):
            if m.stage2_source not in providers:
                raise ProtocolError(
                    f"Method {m.name!r}: 'stage2_source' {m.stage2_source!r} is not a "
                    "method in this protocol."
                )
            provider = next(p for p in methods if p.name == m.stage2_source)
            if 1 not in provider.stages:
                raise ProtocolError(
                    f"Method {m.name!r}: provider {provider.name!r} has no stage 1."
                )
            if provider.tag is not None:
                raise ProtocolError(
                    f"Method {m.name!r}: provider {provider.name!r} must be the default "
                    "(common-data) cell, but it carries a variant tag."
                )

    return Protocol(
        name=str(name),
        task=task,
        description=description,
        seeds=tuple(seeds),
        defaults=defaults,
        methods=tuple(methods),
    )


@dataclass(frozen=True)
class Step:
    """One concrete run step: a training stage or the evaluation of a method."""

    kind: str
    method: Method
    seed: int
    stage: int | None = None
    dependency: bool = False

    @property
    def label(self) -> str:
        if self.kind == "eval":
            return f"{self.method.name} seed={self.seed} eval"
        return f"{self.method.name} seed={self.seed} stage{self.stage}"

    @property
    def registry_phase(self) -> str:
        """Registry phase name (``stage1``/``stage2``/``eval``)."""
        if self.kind == "eval":
            return "eval"
        return f"stage{self.stage}"

    @property
    def phase_key(self) -> str:
        """Registry phase key (stable identity for resume/skip logic)."""
        return f"{self.method.name}/{self.seed}/{self.registry_phase}"

    def required_checkpoint(self) -> tuple[str, int] | None:
        """The ``(model, stage)`` artifact this step must load, or ``None``.

        * A stage-2 step loads the Stage 1 checkpoint of its provider
          (``"self"`` -> this method's own stage 1).
        * An eval step loads the method's final-stage checkpoint.
        * A stage-1 step loads nothing.
        """
        if self.kind == "eval":
            return (self.method.model_name, self.method.final_stage)
        if self.kind == "train" and self.stage == 2:
            source = self.method.stage2_source
            if source == "self":
                return (self.method.model_name, 1)
            if source in ("bc", "phaseforge"):
                return (source, 1)
        return None


def _validate_seed_filter(protocol: Protocol, seeds_filter: list[int] | None) -> list[int]:
    if not seeds_filter:
        return list(protocol.seeds)
    for s in seeds_filter:
        if s not in protocol.seeds:
            raise ProtocolError(
                f"Seed {s} is not in the protocol ({list(protocol.seeds)}). "
                "Adjust the protocol manifest or choose a protocol seed."
            )
    return list(seeds_filter)


def build_plan(
    protocol: Protocol,
    methods: list[Method],
    *,
    seeds: list[int] | None = None,
    stage: int | None = None,
    eval_only: bool = False,
    skip_eval: bool = False,
    with_dependencies: bool = False,
) -> list[Step]:
    """Build the ordered plan of steps for the selected methods and seeds.

    Ordering: methods by manifest index (so a provider runs before its
    consumers in a full sweep); within a method, seeds in protocol order and
    each seed's training stages ascending, followed by the method's eval.
    When ``with_dependencies`` is set and a selected method's Stage 1
    provider is not itself selected, the provider's Stage 1 (train only,
    marked ``dependency``) is prepended so a partial selection still runs to
    completion. Stage filtering disables dependency injection (the scope was
    explicitly narrowed; a missing prerequisite then fails pre-flight).
    """
    if eval_only and stage is not None:
        raise ProtocolError("--stage and --eval-only are mutually exclusive.")
    if eval_only and skip_eval:
        raise ProtocolError("--eval-only and --skip-eval are mutually exclusive.")
    if stage is not None and stage not in (1, 2):
        raise ProtocolError(f"--stage must be 1 or 2, got {stage}.")

    seed_list = _validate_seed_filter(protocol, seeds)
    selected_names = {m.name for m in methods}

    deps: list[Step] = []
    if with_dependencies and not eval_only and stage is None:
        for m in methods:
            if m.stage2_source in ("bc", "phaseforge") and m.stage2_source not in selected_names:
                provider = protocol.method_by_name(m.stage2_source)
                if provider is None:
                    raise ProtocolError(
                        f"Method {m.name!r}: provider {m.stage2_source!r} is missing "
                        "from the protocol."
                    )
                for seed in seed_list:
                    deps.append(
                        Step(kind="train", method=provider, seed=seed, stage=1, dependency=True)
                    )
        seen: set[tuple[str, int]] = set()
        deduped: list[Step] = []
        for step in deps:
            key = (step.method.name, step.seed)
            if key not in seen:
                seen.add(key)
                deduped.append(step)
        deps = sorted(deduped, key=lambda s: (s.method.index, seed_list.index(s.seed)))

    steps: list[Step] = []
    for method in methods:
        for seed in seed_list:
            if eval_only:
                if method.evaluate:
                    steps.append(Step(kind="eval", method=method, seed=seed))
                continue
            for stage_n in method.stages:
                if stage is not None and stage_n != stage:
                    continue
                steps.append(
                    Step(kind="train", method=method, seed=seed, stage=stage_n)
                )
            if method.evaluate and not skip_eval and stage is None:
                steps.append(Step(kind="eval", method=method, seed=seed))

    return deps + steps

