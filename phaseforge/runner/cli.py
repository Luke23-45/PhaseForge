"""``phaseforge-sweep`` — the experiment runner CLI.

Runs complete experiments from the frozen protocol manifest: for every
selected method and seed it executes each training stage in order and then
the offline evaluation of that method's final-stage checkpoint, honouring
the protocol's Stage 1 source dependencies. A resumable state registry
(``<outputs>/_runner/state.json``) records what completed and the *exact*
artifact each stage produced, so evaluation never targets a stale or
seed-mixed checkpoint and an interrupted sweep resumes in place.

When a selected stage-2 method's Stage 1 provider (``bc``/``phaseforge``) is
not selected and its checkpoint is missing from the output tree, the runner
auto-trains the provider's Stage 1 as a dependency step before running the
consumer, so a partial ``--methods`` selection no longer fails pre-flight on
a missing provider. Explicitly scoped runs (``--stage``/``--eval-only``)
keep the strict pre-flight check: a missing prerequisite then fails loudly
instead of silently training.

Examples::

    phaseforge-sweep                                    # full five-task matrix x seeds
    phaseforge-sweep --methods 1,5,9                    # by method index
    phaseforge-sweep --methods bc,warmstart_moe         # by method name
    phaseforge-sweep --methods teacher_forced --with-dependencies
    phaseforge-sweep --methods 1 --seeds 42
    phaseforge-sweep --stage 1                          # train stage 1 only
    phaseforge-sweep --eval-only                        # (re)run evaluations only
    phaseforge-sweep --methods 3 --dry-run              # preview commands
    phaseforge-sweep --list                             # show the method matrix

``python -m phaseforge.runner --help`` is equivalent.

The five-task manifest contains task-specific entries for Lift, Can, Square,
Tool Hang, and Transport. Tool Hang subprocesses are routed to the dedicated
robosuite 1.5.0 interpreter; the other tasks use the current environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phaseforge.runner.executor import (
    CommandError,
    preflight_toolhang_python,
    resolve_toolhang_python,
    run_step,
    step_command,
)
from phaseforge.runner.protocol import (
    Method,
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    load_protocol,
)
from phaseforge.runner.registry import RegistryError, RunnerState
from phaseforge.runner.resolver import (
    CheckpointError,
    resolve_checkpoint_path,
    resolve_eval_run_dir,
    resolve_run_dir,
    resolve_stage_ckpt,
    stage_checkpoint_relative,
    verify_checkpoint_contract,
)
from phaseforge.utils.config import git_info

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Expert-count contract for every MoE checkpoint consumed by the final
#: protocol (the canonical method and all MoE controls are six-expert).
#: Dense checkpoints (BC family) carry no ``moe_layer.experts.*`` keys and
#: skip the check inside ``verify_checkpoint_contract``. This is the
#: fail-closed guard against pre-final artifacts that share a filesystem
#: name with the canonical method (the retired 8-expert ``phaseforge``).
FINAL_EXPERT_CONTRACT = 6

#: Default protocol manifest. The full five-task evaluation lives in
#: ``experiments/five_task.json``; ``experiments/lift_pilot.json`` is the
#: original single-task pilot that remains useful for debugging the
#: rollout pipeline.
DEFAULT_MANIFEST = "experiments/five_task.json"


def _split_list(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="phaseforge-sweep",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Protocol manifest JSON (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[],
        help="Method indices and/or names (e.g. '1,5,9' or 'bc,warmstart_moe'). "
        "Default: all methods.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=[],
        help="Protocol seeds to run (e.g. '42,44'). Default: all protocol seeds.",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        default=None,
        help="Run only this training stage (no evaluation).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only the evaluation steps (final-stage checkpoints must exist).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run all training stages but skip the evaluation steps.",
    )
    parser.add_argument(
        "--with-dependencies",
        action="store_true",
        help="Include the required Stage 1 pretraining of a shared provider in "
        "the printed plan. The runtime auto-injects it for unscoped sweeps "
        "regardless, so this flag mainly controls plan visibility.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run steps that the state registry already marks completed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed step and keep going instead of stopping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exact commands without executing anything.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the method matrix from the manifest and exit.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Output base directory, relative to the project root (default: outputs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward project.log_level=INFO to the runs (default WARNING).",
    )
    parser.add_argument(
        "--toolhang-python",
        default=None,
        help="Python interpreter for Tool Hang steps (also PHASEFORGE_TOOLHANG_PYTHON). "
        "Defaults to .venv-toolhang/bin/python or .venv-toolhang/Scripts/python.exe.",
    )
    return parser.parse_args(argv)


def _manifest_path(args: argparse.Namespace) -> Path:
    p = Path(args.manifest)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _outputs_base(args: argparse.Namespace) -> Path:
    return (PROJECT_ROOT / args.outputs).resolve()


def _build_plan(protocol: Protocol, args: argparse.Namespace) -> list[Step]:
    methods = (
        protocol.select_methods(_split_list(args.methods))
        if args.methods
        else list(protocol.methods)
    )
    seeds: list[int] | None = None
    if args.seeds:
        try:
            seeds = [int(s) for s in _split_list(args.seeds)]
        except ValueError as exc:
            raise ProtocolError(f"--seeds must be integers: {args.seeds}") from exc
    return build_plan(
        protocol,
        methods,
        seeds=seeds,
        stage=args.stage,
        eval_only=args.eval_only,
        skip_eval=args.skip_eval,
        with_dependencies=args.with_dependencies,
    )


def _require_stage2_prereq(
    step: Step, outputs_base: Path, expected_commit: str | None = None
) -> Path | None:
    """Resolve the exact Stage 1 checkpoint a stage-2 step bootstraps from.

    Returns ``None`` for steps that load nothing (a stage-1 step, or a
    stage-2 step whose model has no Stage 1 source). The returned path is
    passed to the subprocess as ``train.stage1_ckpt_path`` so it loads this
    exact artifact rather than re-running its own auto-detect
    (:func:`phaseforge.utils.config.find_latest_checkpoint`), whose
    ``tag=None`` means "no constraint" and can select a newer *tagged* sibling
    variant that shares the provider's output tree (e.g. ``bc_robot_only``
    next to ``bc``), crashing the load with a dimension mismatch. With
    ``expected_commit``, only provider checkpoints from that git revision are
    eligible.
    """
    req = step.required_checkpoint()
    if req is None:
        return None
    model, stage = req
    try:
        source_tag = (
            step.method.output_tag if step.method.stage2_source == "self" else step.method.task
        )
        ckpt = resolve_stage_ckpt(
            outputs_base,
            model,
            stage,
            seed=step.seed,
            # A Stage 2 dependency consumes either this method's own tagged
            # Stage 1 cell or the task's default provider cell.
            tag=source_tag,
            expected_commit=expected_commit,
        )
    except CheckpointError as exc:
        raise CheckpointError(
            f"{step.method.name} stage 2 needs a {model} stage 1 checkpoint for "
            f"seed {step.seed} under {outputs_base}. Run {model} stage 1 first, "
            "or re-run with --with-dependencies."
        ) from exc
    # Fail closed on contract violations (wrong model tree, wrong expert
    # count, wrong stage) before the subprocess consumes the artifact.
    verify_checkpoint_contract(
        ckpt,
        expected_model_name=model,
        expected_num_experts=FINAL_EXPERT_CONTRACT,
        expected_stage=stage,
    )
    return ckpt


def _auto_dependency_provider(
    step: Step, protocol: Protocol, args: argparse.Namespace
) -> Method | None:
    """Return the Stage 1 provider the runner should auto-train, or ``None``.

    Mirrors the plan-level dependency policy: providers are the default
    ``bc``/``phaseforge`` cells of the consumer's task, and explicit scoping
    (``--stage``, ``--eval-only``) disables injection so a deliberately
    narrowed sweep still fails pre-flight.
    """
    if args.eval_only or args.stage is not None:
        return None
    if step.kind != "train" or step.stage != 2:
        return None
    req = step.required_checkpoint()
    if req is None:
        return None
    model, stage = req
    if stage != 1 or model not in ("bc", "phaseforge"):
        return None
    return protocol.method_by_name(model, task=step.method.task)


def _run_dependency_step(
    step: Step,
    protocol: Protocol,
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    toolhang_python: Path | None,
    expected_commit: str | None = None,
) -> None:
    """Execute one auto-injected Stage 1 dependency step and record it."""
    print(f"\n[runner] auto-injecting missing dependency: {step.label}", flush=True)
    try:
        run_step(
            step,
            ckpt_path=None,
            outputs_base=outputs_base,
            defaults=protocol.defaults,
            cwd=PROJECT_ROOT,
            log_level="INFO" if args.verbose else "WARNING",
            toolhang_python=toolhang_python,
        )
    except CommandError as exc:
        raise CheckpointError(f"Auto-injected dependency {step.label} failed: {exc}") from exc
    if step.stage is None:  # pragma: no cover - guard for type checkers
        raise CheckpointError(f"Dependency step {step.label} has no stage.")
    run_dir = resolve_run_dir(
        outputs_base,
        step.method.model_name,
        step.stage,
        seed=step.seed,
        tag=step.method.output_tag,
        expected_commit=expected_commit,
    )
    ckpt_rel = stage_checkpoint_relative(
        outputs_base, run_dir, step.method.model_name, step.stage
    )
    state.mark(
        step.method.phase_key,
        step.seed,
        step.registry_phase,
        run_dir=run_dir.relative_to(outputs_base).as_posix(),
        ckpt=ckpt_rel,
    )


def _resolve_stage2_with_auto_dependency(
    step: Step,
    protocol: Protocol,
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    toolhang_python: Path | None,
    expected_commit: str | None = None,
) -> Path | None:
    """Resolve a stage-2 prerequisite, auto-training a missing provider first.

    In an unscoped sweep, a stage-2 step whose Stage 1 provider checkpoint is
    absent from the output tree no longer fails pre-flight: the runner trains
    the provider's Stage 1 (a dependency step) and then resolves the consumer
    exactly as before. A provider that already exists is never retrained, and
    providers of explicitly scoped runs (``--stage``/``--eval-only``) still
    fail loudly. Returns ``None`` for steps that load no prerequisite.
    """
    try:
        return _require_stage2_prereq(step, outputs_base, expected_commit)
    except CheckpointError:
        provider = _auto_dependency_provider(step, protocol, args)
        if provider is None:
            raise
        dep = Step(kind="train", method=provider, seed=step.seed, stage=1, dependency=True)
        _run_dependency_step(
            dep, protocol, outputs_base, state, args, toolhang_python, expected_commit
        )
        return _require_stage2_prereq(step, outputs_base, expected_commit)


def _eval_target(
    step: Step, outputs_base: Path, state: RunnerState, expected_commit: str | None = None
) -> Path:
    ckpt = resolve_checkpoint_path(
        outputs_base,
        step.method,
        step.method.final_stage,
        seed=step.seed,
        state=state,
        expected_commit=expected_commit,
    )
    # Fail closed before evaluation consumes a pre-final or cross-model
    # artifact (same filesystem name, wrong contract).
    verify_checkpoint_contract(
        ckpt,
        expected_model_name=step.method.model_name,
        expected_num_experts=FINAL_EXPERT_CONTRACT,
        expected_stage=step.method.final_stage,
    )
    return ckpt


def _should_skip_eval(
    step: Step,
    outputs_base: Path,
    state: RunnerState,
    force: bool,
    expected_commit: str | None = None,
) -> bool:
    if force:
        return False
    entry = state.get(step.method.phase_key, step.seed, "eval")
    if entry is None or entry.get("status") != "completed":
        return False
    if not state.is_complete(step.method.phase_key, step.seed, "eval"):
        return False
    try:
        target = _eval_target(step, outputs_base, state, expected_commit)
    except CheckpointError:
        return False
    return entry.get("ckpt") == target.relative_to(outputs_base).as_posix()


def _print_plan(
    plan: list[Step],
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    expected_commit: str | None = None,
) -> None:
    print(f"\n[runner] plan ({len(plan)} steps, outputs base: {outputs_base})")
    for index, step in enumerate(plan, start=1):
        tag = " [dependency]" if step.dependency else ""
        status = (
            "done"
            if _step_done(step, outputs_base, state, args.force, expected_commit)
            else "pending"
        )
        print(f"  {index:>3}. {step.label:<48} {status}{tag}")


def _step_done(
    step: Step,
    outputs_base: Path,
    state: RunnerState,
    force: bool,
    expected_commit: str | None = None,
) -> bool:
    if force:
        return False
    if step.kind == "train":
        return state.is_complete(step.method.phase_key, step.seed, step.registry_phase)
    return _should_skip_eval(
        step, outputs_base, state, force=False, expected_commit=expected_commit
    )


def _print_methods(protocol: Protocol) -> None:
    print(f"Protocol: {protocol.name} — {protocol.task} — {protocol.description}")
    for method in protocol.methods:
        stages = ",".join(f"stage{s}" for s in method.stages)
        print(
            f"  {method.index:>2}  {method.name:<32} {method.model:<40} "
            f"data={method.data:<10} {stages:<12} "
            f"eval={'yes' if method.evaluate else 'no'}"
        )
    print(f"seeds: {list(protocol.seeds)}")


def _print_summary(protocol: Protocol, state: RunnerState, counts: dict[str, int]) -> None:
    print("\n[runner] summary")
    print(f"  ran={counts['run']} skipped={counts['skip']} failed={counts['failed']}")
    for method in protocol.methods:
        phases = [f"stage{s}" for s in method.stages]
        if method.evaluate:
            phases.append("eval")
        cells: list[str] = []
        for seed in protocol.seeds:
            statuses = []
            for phase in phases:
                entry = state.get(method.phase_key, seed, phase)
                if state.is_complete(method.phase_key, seed, phase):
                    statuses.append("ok")
                elif entry is not None and entry.get("status") == "failed":
                    statuses.append("FAIL")
                else:
                    statuses.append("-")
            cells.append(f"seed{seed}=[{' '.join(statuses)}]")
        print(f"  {method.name:<32} {'  '.join(cells)}")


def _current_commit() -> str | None:
    """Return the current git commit, or ``None`` when unavailable.

    Used as the commit gate for the state registry and checkpoint
    resolution: a sweep at revision X must not reuse artifacts recorded or
    produced at revision Y (e.g. pre-fix checkpoints) — that silently
    corrupts the sweep. When git is unavailable (no repo), gating is
    disabled and resolution falls back to "newest completed" behaviour.
    """
    commit = git_info().get("commit") or ""
    return commit.strip() or None


def run(args: argparse.Namespace) -> int:
    outputs_base = _outputs_base(args)
    try:
        protocol = load_protocol(_manifest_path(args))
    except (ProtocolError, OSError) as exc:
        print(f"[runner] ERROR loading manifest: {exc}", file=sys.stderr)
        return 2

    if args.list:
        _print_methods(protocol)
        return 0

    commit = _current_commit()
    if commit is None:
        print(
            "[runner] WARNING: git unavailable — commit gating disabled; stale "
            "pre-fix checkpoints will NOT be filtered. Verify the tree is clean "
            "at the intended revision before trusting results.",
            file=sys.stderr,
        )
    else:
        print(f"[runner] commit gate: {commit}")

    try:
        state = RunnerState(RunnerState.default_path(outputs_base), expected_commit=commit)
    except RegistryError as exc:
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        plan = _build_plan(protocol, args)
    except ProtocolError as exc:
        print(f"[runner] ERROR: {exc}", file=sys.stderr)
        return 2

    if not plan:
        print("[runner] No steps match the current selection.", file=sys.stderr)
        return 0

    _print_plan(plan, outputs_base, state, args, commit)

    toolhang_python: Path | None = None
    has_toolhang = any(step.method.task == "ToolHang" for step in plan)
    if has_toolhang and not args.dry_run:
        try:
            toolhang_python = resolve_toolhang_python(PROJECT_ROOT, args.toolhang_python)
            preflight_toolhang_python(toolhang_python)
            print(f"[runner] Tool Hang interpreter: {toolhang_python}")
        except CommandError as exc:
            print(f"[runner] ERROR: {exc}", file=sys.stderr)
            return 2

    counts = {"run": 0, "skip": 0, "failed": 0}
    total = len(plan)
    for index, step in enumerate(plan, start=1):
        prefix = f"[{index}/{total}]"
        try:
            if _step_done(step, outputs_base, state, args.force, commit):
                print(f"{prefix} skip {step.label} (already completed)")
                counts["skip"] += 1
                continue

            if args.dry_run:
                _print_dry_run(step, protocol, outputs_base, state, args, commit)
                counts["run"] += 1
                continue

            ckpt_abs: Path | None = None
            if step.kind == "eval":
                ckpt_abs = _eval_target(step, outputs_base, state, commit)
            else:
                ckpt_abs = _resolve_stage2_with_auto_dependency(
                    step, protocol, outputs_base, state, args, toolhang_python, commit
                )

            run_step(
                step,
                ckpt_path=ckpt_abs,
                outputs_base=outputs_base,
                defaults=protocol.defaults,
                cwd=PROJECT_ROOT,
                log_level="INFO" if args.verbose else "WARNING",
                toolhang_python=toolhang_python,
            )

            if step.kind == "eval":
                if ckpt_abs is None:  # pragma: no cover - guard for type checkers
                    raise CheckpointError(f"Eval step {step.label} has no checkpoint.")
                run_dir = resolve_eval_run_dir(
                    outputs_base,
                    step.method.model_name,
                    seed=step.seed,
                    tag=step.method.output_tag,
                    expected_commit=commit,
                )
                state.mark(
                    step.method.phase_key,
                    step.seed,
                    "eval",
                    ckpt=ckpt_abs.relative_to(outputs_base).as_posix(),
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                )
            else:
                if step.stage is None:  # pragma: no cover - guard for type checkers
                    raise CheckpointError(f"Train step {step.label} has no stage.")
                run_dir = resolve_run_dir(
                    outputs_base,
                    step.method.model_name,
                    step.stage,
                    seed=step.seed,
                    tag=step.method.output_tag,
                    expected_commit=commit,
                )
                ckpt_rel = stage_checkpoint_relative(
                    outputs_base, run_dir, step.method.model_name, step.stage
                )
                state.mark(
                    step.method.phase_key,
                    step.seed,
                    step.registry_phase,
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                    ckpt=ckpt_rel,
                )
            counts["run"] += 1
            print(f"{prefix} OK {step.label}")
        except (CheckpointError, CommandError) as exc:
            state.mark_failed(step.method.phase_key, step.seed, step.registry_phase, str(exc))
            counts["failed"] += 1
            print(f"{prefix} FAILED {step.label}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                _print_summary(protocol, state, counts)
                return 1

    _print_summary(protocol, state, counts)
    return 0 if counts["failed"] == 0 else 1


def _print_dry_run(
    step: Step,
    protocol: Protocol,
    outputs_base: Path,
    state: RunnerState,
    args: argparse.Namespace,
    expected_commit: str | None = None,
) -> None:
    try:
        ckpt_abs: Path | None = None
        if step.kind == "eval":
            ckpt_abs = _eval_target(step, outputs_base, state, expected_commit)
        else:
            ckpt_abs = _require_stage2_prereq(step, outputs_base, expected_commit)
        cmd = step_command(
            step, ckpt_path=ckpt_abs, outputs_base=outputs_base, defaults=protocol.defaults
        )
        print(f"  [dry-run] WOULD RUN  {step.label}")
        print(f"    $ {' '.join(cmd)}")
    except CheckpointError as exc:
        provider = _auto_dependency_provider(step, protocol, args)
        if provider is not None:
            dep = Step(kind="train", method=provider, seed=step.seed, stage=1, dependency=True)
            cmd = step_command(
                dep, ckpt_path=None, outputs_base=outputs_base, defaults=protocol.defaults
            )
            print(f"  [dry-run] AUTO-INJECT dependency: {dep.label}")
            print(f"    $ {' '.join(cmd)}")
            return
        print(f"  [dry-run] BLOCKED  {step.label} — prerequisite missing: {exc}")


def main() -> None:
    args = parse_args()
    try:
        exit_code = run(args)
    except KeyboardInterrupt:
        print("\n[runner] Interrupted by user.", file=sys.stderr)
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
