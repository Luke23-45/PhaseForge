"""``phaseforge-sweep`` — the experiment runner CLI.

Runs complete experiments from the frozen protocol manifest: for every
selected method and seed it executes each training stage in order and then
the offline evaluation of that method's final-stage checkpoint, honouring
the protocol's Stage 1 source dependencies. A resumable state registry
(``<outputs>/_runner/state.json``) records what completed and the *exact*
artifact each stage produced, so evaluation never targets a stale or
seed-mixed checkpoint and an interrupted sweep resumes in place.

Examples::

    phaseforge-sweep                                    # full matrix (all 9 methods x seeds)
    phaseforge-sweep --methods 1,5,9                    # by method index
    phaseforge-sweep --methods bc,warmstart_moe         # by method name
    phaseforge-sweep --methods teacher_forced --with-dependencies
    phaseforge-sweep --methods 1 --seeds 42
    phaseforge-sweep --stage 1                          # train stage 1 only
    phaseforge-sweep --eval-only                        # (re)run evaluations only
    phaseforge-sweep --methods 3 --dry-run              # preview commands
    phaseforge-sweep --list                             # show the method matrix

``python -m phaseforge.runner --help`` is equivalent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phaseforge.runner.executor import CommandError, run_step, step_command
from phaseforge.runner.protocol import (
    Protocol,
    ProtocolError,
    Step,
    build_plan,
    load_protocol,
)
from phaseforge.runner.registry import RegistryError, RunnerState
from phaseforge.runner.resolver import (
    CheckpointError,
    checkpoint_exists,
    resolve_checkpoint_path,
    resolve_eval_run_dir,
    resolve_run_dir,
    stage_checkpoint_relative,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = "experiments/lift_pilot.json"


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
        help="Auto-run the required Stage 1 pretraining of a shared provider "
        "when a selected method needs it and the provider is not selected.",
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


def _require_stage2_prereq(step: Step, outputs_base: Path) -> None:
    """A stage-2 step must be able to load its provider's Stage 1 checkpoint."""
    req = step.required_checkpoint()
    if req is None:
        return
    model, stage = req
    # Providers are validated to be untagged, so the strict lookup is tag=None.
    if not checkpoint_exists(outputs_base, model, stage, seed=step.seed, tag=None):
        raise CheckpointError(
            f"{step.method.name} stage 2 needs a {model} stage 1 checkpoint for "
            f"seed {step.seed} under {outputs_base}. Run {model} stage 1 first, "
            "or re-run with --with-dependencies."
        )


def _eval_target(
    step: Step, outputs_base: Path, state: RunnerState
) -> Path:
    return resolve_checkpoint_path(
        outputs_base,
        step.method,
        step.method.final_stage,
        seed=step.seed,
        state=state,
    )


def _should_skip_eval(
    step: Step, outputs_base: Path, state: RunnerState, force: bool
) -> bool:
    if force:
        return False
    entry = state.get(step.method.name, step.seed, "eval")
    if entry is None or entry.get("status") != "completed":
        return False
    try:
        target = _eval_target(step, outputs_base, state)
    except CheckpointError:
        return False
    return entry.get("ckpt") == target.relative_to(outputs_base).as_posix()


def _print_plan(
    plan: list[Step], outputs_base: Path, state: RunnerState, args: argparse.Namespace
) -> None:
    print(f"\n[runner] plan ({len(plan)} steps, outputs base: {outputs_base})")
    for index, step in enumerate(plan, start=1):
        tag = " [dependency]" if step.dependency else ""
        status = "done" if _step_done(step, outputs_base, state, args.force) else "pending"
        print(f"  {index:>3}. {step.label:<48} {status}{tag}")


def _step_done(
    step: Step, outputs_base: Path, state: RunnerState, force: bool
) -> bool:
    if force:
        return False
    if step.kind == "train":
        return state.is_complete(step.method.name, step.seed, step.registry_phase)
    return _should_skip_eval(step, outputs_base, state, force=False)


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
                entry = state.get(method.name, seed, phase)
                if state.is_complete(method.name, seed, phase):
                    statuses.append("ok")
                elif entry is not None and entry.get("status") == "failed":
                    statuses.append("FAIL")
                else:
                    statuses.append("-")
            cells.append(f"seed{seed}=[{' '.join(statuses)}]")
        print(f"  {method.name:<32} {'  '.join(cells)}")


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

    try:
        state = RunnerState(RunnerState.default_path(outputs_base))
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

    _print_plan(plan, outputs_base, state, args)

    counts = {"run": 0, "skip": 0, "failed": 0}
    total = len(plan)
    for index, step in enumerate(plan, start=1):
        prefix = f"[{index}/{total}]"
        try:
            if _step_done(step, outputs_base, state, args.force):
                print(f"{prefix} skip {step.label} (already completed)")
                counts["skip"] += 1
                continue

            if args.dry_run:
                _print_dry_run(step, outputs_base, state, protocol.defaults)
                counts["run"] += 1
                continue

            ckpt_abs: Path | None = None
            if step.kind == "eval":
                ckpt_abs = _eval_target(step, outputs_base, state)
            else:
                _require_stage2_prereq(step, outputs_base)

            run_step(
                step,
                ckpt_path=ckpt_abs,
                outputs_base=outputs_base,
                defaults=protocol.defaults,
                cwd=PROJECT_ROOT,
                log_level="INFO" if args.verbose else "WARNING",
            )

            if step.kind == "eval":
                if ckpt_abs is None:  # pragma: no cover - guard for type checkers
                    raise CheckpointError(f"Eval step {step.label} has no checkpoint.")
                run_dir = resolve_eval_run_dir(
                    outputs_base,
                    step.method.model_name,
                    seed=step.seed,
                    tag=step.method.tag,
                )
                state.mark(
                    step.method.name,
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
                    tag=step.method.tag,
                )
                ckpt_rel = stage_checkpoint_relative(
                    outputs_base, run_dir, step.method.model_name, step.stage
                )
                state.mark(
                    step.method.name,
                    step.seed,
                    step.registry_phase,
                    run_dir=run_dir.relative_to(outputs_base).as_posix(),
                    ckpt=ckpt_rel,
                )
            counts["run"] += 1
            print(f"{prefix} OK {step.label}")
        except (CheckpointError, CommandError) as exc:
            state.mark_failed(step.method.name, step.seed, step.registry_phase, str(exc))
            counts["failed"] += 1
            print(f"{prefix} FAILED {step.label}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                _print_summary(protocol, state, counts)
                return 1

    _print_summary(protocol, state, counts)
    return 0 if counts["failed"] == 0 else 1


def _print_dry_run(
    step: Step, outputs_base: Path, state: RunnerState, defaults: tuple[str, ...]
) -> None:
    try:
        ckpt_abs: Path | None = None
        if step.kind == "eval":
            ckpt_abs = _eval_target(step, outputs_base, state)
        else:
            _require_stage2_prereq(step, outputs_base)
        cmd = step_command(
            step, ckpt_path=ckpt_abs, outputs_base=outputs_base, defaults=defaults
        )
        print(f"  [dry-run] WOULD RUN  {step.label}")
        print(f"    $ {' '.join(cmd)}")
    except CheckpointError as exc:
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
