"""V2-E: rollout evaluation under all four router interventions.

For each seed, load the stage-2 best checkpoint of the phaseforge method
and rollout-evaluate it on the frozen reset bank under every V2-E
evaluation-time routing intervention:

* ``learned`` — the trained router (the protocol's standard eval);
* ``sticky``  — beta-EMA over the learned top-1 choice (V2-C's stickiness
  at eval time);
* ``uniform`` — equal weight over ALL experts (the A5 uniform
  counterfactual: the V2 success criterion is learned > uniform);
* ``oracle``  — the frozen phase head routed through the soft
  phase->expert mapping M (M^T softmax(phase_head(z))).

Per-phase success rates are reported when the training cache carries the
phase-threshold calibration (``--require-phase-tracking`` makes a missing
artifact a hard failure, so pre-calibration caches cannot silently produce
runs without per-phase SR).

Outputs: four ``phaseforge-eval`` runs per seed under ``outputs/eval/...``
with ``project.tag=router_<mode>``; the per-mode SR + CI + per-phase SR are
collected into ``outputs/_findings/v2_eval.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phaseforge.runner.commands import eval_command
from phaseforge.runner.protocol import Step, load_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from smoke_matrix import (  # noqa: E402
    EVAL_QUIET,
    _execute,
    _find_run_by_meta,
    _run_dir_base,
)

FINDINGS_DIR = Path("outputs/_findings")
MODES = ("learned", "sticky", "uniform", "oracle")


def _step(method, seed: int) -> Step:
    return Step(kind="eval", method=method, seed=seed, stage=None)


def _best_ckpt(outputs: Path, method, seed: int) -> Path:
    run_dir = _find_run_by_meta(
        _run_dir_base(outputs, method, 2, seed), method.name, seed, method.output_tag
    )
    ckpt = run_dir / "checkpoints" / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise RuntimeError(
            f"stage-2 run {run_dir.name} is incomplete "
            "(missing checkpoints/checkpoint_best.pt)"
        )
    return ckpt


def _eval_mode(
    method,
    seed: int,
    ckpt_path: Path,
    mode: str,
    outputs: Path,
    defaults: tuple[str, ...],
    logs: Path,
    timeout: int,
    require_phase_tracking: bool,
) -> dict:
    quiet = list(EVAL_QUIET) + [
        f"eval.episodes.router_mode={mode}",
        f"project.tag=router_{mode}",
    ]
    if require_phase_tracking:
        quiet.append("eval.episodes.require_phase_tracking=true")
    step = _step(method, seed)
    cmd = eval_command(
        step,
        ckpt_path=ckpt_path,
        outputs_base=outputs,
        defaults=defaults + tuple(quiet),
    )
    _execute(
        ["phaseforge-eval", "project.log_level=WARNING"] + cmd[1:],
        logs / f"seed{seed}_{mode}.log",
        timeout,
    )
    run_dir = _find_run_by_meta(
        outputs / "eval" / method.model_name / f"seed{seed}",
        method.name,
        seed,
        f"router_{mode}",
    )
    summary = run_dir / "rollout_summary.json"
    data = json.loads(summary.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "mode": mode,
        "sr": metrics.get("eval/rollout/success_rate"),
        "ci_low": metrics.get("eval/rollout/wilson_ci95_low"),
        "ci_high": metrics.get("eval/rollout/wilson_ci95_high"),
        "valid_episodes": metrics.get("eval/rollout/valid_episodes"),
        "phase_tracking": metrics.get("eval/rollout/phase_tracking"),
        "per_phase_sr": metrics.get("eval/rollout/per_phase_sr"),
        "run_dir": str(run_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--outputs", default="outputs/surgical")
    parser.add_argument("--manifest", default=PROJECT_ROOT / "experiments" / "lift_ablation.json")
    parser.add_argument("--method", default="phaseforge")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--require-phase-tracking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    modes = tuple(m for m in args.modes.split(",") if m)
    unknown = set(modes) - set(MODES)
    if unknown:
        print(f"v2_eval: unknown mode(s): {sorted(unknown)}", file=sys.stderr)
        return 2
    outputs = (PROJECT_ROOT / args.outputs).resolve()
    logs = outputs / "_v2_eval" / "logs"
    protocol = load_protocol(Path(args.manifest))
    methods = {m.name: m for m in protocol.methods}
    method = methods.get(args.method)
    if method is None:
        print(
            f"v2_eval: method {args.method!r} not in manifest {args.manifest}",
            file=sys.stderr,
        )
        return 2
    if method.evaluate_mode != "rollout":
        print(
            f"v2_eval: method {args.method!r} evaluates with "
            f"{method.evaluate_mode!r}, not rollout",
            file=sys.stderr,
        )
        return 2

    print(
        f"[v2-eval] method={method.name} seeds={seeds} modes={modes} "
        f"outputs={outputs}"
    )

    results: dict[str, dict] = {}
    for seed in seeds:
        if args.dry_run:
            for mode in modes:
                quiet = list(EVAL_QUIET) + [
                    f"eval.episodes.router_mode={mode}",
                    f"project.tag=router_{mode}",
                ]
                step = _step(method, seed)
                cmd = eval_command(
                    step,
                    ckpt_path=Path("<stage2-best-checkpoint>"),
                    outputs_base=outputs,
                    defaults=protocol.defaults + tuple(quiet),
                )
                print(f"[v2-eval][dry] {' '.join(cmd[1:])}")
            continue
        ckpt_path = _best_ckpt(outputs, method, seed)
        results[str(seed)] = {"checkpoint": str(ckpt_path), "modes": {}}
        for mode in modes:
            results[str(seed)]["modes"][mode] = _eval_mode(
                method,
                seed,
                ckpt_path,
                mode,
                outputs,
                protocol.defaults,
                logs,
                args.timeout,
                args.require_phase_tracking,
            )
            print(
                f"[v2-eval] seed {seed} {mode}: "
                f"SR={results[str(seed)]['modes'][mode]['sr']}"
            )

    if args.dry_run:
        return 0

    findings = {
        "method": method.name,
        "modes": modes,
        "require_phase_tracking": args.require_phase_tracking,
        "seeds": results,
    }
    FINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    (FINDINGS_DIR / "v2_eval.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8"
    )
    print(f"[v2-eval] findings written to {FINDINGS_DIR / 'v2_eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())