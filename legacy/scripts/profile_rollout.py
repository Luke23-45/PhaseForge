"""Rollout phase profiler: where does evaluation wall-clock time go?

Runs a SMALL serial evaluation (default 2 tasks x 5 episodes) through the
real :class:`RolloutEvaluator` path and reports, per task and aggregated:

- wall-seconds per episode and steps per episode (live ETA basis)
- evaluator-side phases: ``reset`` / ``step`` / ``infer`` / ``other``
- env-side sub-phases (from ``StateOnlyLiberoEnv.timings()``):
  ``physics_step`` / ``check_success`` / ``extract_state`` /
  ``sim_reset`` (hard-reset XML rebuild) / ``set_init`` / ``wait_steps``
- episode outcome counts (success / timeout / early termination)

Use it to decide where to optimize (hard reset cost, physics substeps,
BDDL predicate, model inference, worker count) BEFORE committing to a
full multi-hour protocol run.

Usage (run from repo root, real data + checkpoint required):
    uv sync --extra rollout
    uv run python scripts/profile_rollout.py --suite libero_90 \
        --tasks 0,1 --episodes 5

Exit code 0 = profiling completed; 1 = environment/config error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import DictConfig


def _compose_cfg(overrides: list[str]) -> DictConfig:
    """Compose the same hydra config the ``phaseforge-eval`` CLI uses."""
    from hydra import compose, initialize

    # config_path is resolved relative to THIS file (scripts/), not CWD.
    with initialize(version_base="1.3", config_path="../phaseforge/config"):
        return compose(config_name="main", overrides=overrides)


def _fmt_pct(secs: float, total: float) -> str:
    return f"{100.0 * secs / total:5.1f}%" if total else "   n/a"


def _print_task(tid: int, num_tasks: int, bd: dict) -> None:
    total = bd["total_seconds"]
    phases = bd["phases"]
    env = bd["env_phases"]
    in_env = env.get("step_total", {}).get("seconds", 0.0) + env.get(
        "reset_total", {}
    ).get("seconds", 0.0)
    out = bd["outcomes"]
    print(
        f"  [{tid + 1}/{num_tasks}] {bd['description']} — "
        f"{bd['episodes']} ep, {total:.0f}s total, "
        f"{bd['mean_episode_seconds']:.1f}s/ep, {bd['mean_steps']:.0f} steps/ep"
    )
    print(
        "      phases: reset {:s} | step {:s} | infer {:s} | other {:s}".format(
            _fmt_pct(phases["reset_seconds"], total),
            _fmt_pct(phases["step_seconds"], total),
            _fmt_pct(phases["infer_seconds"], total),
            _fmt_pct(phases["other_seconds"], total),
        )
    )
    print(
        "      env:    physics {:s} | check_success {:s} | extract {:s} | "
        "sim_reset {:s} | set_init {:s} | wait {:s} (of {:.0f}s in-env)".format(
            _fmt_pct(env.get("physics_step", {}).get("seconds", 0.0), in_env),
            _fmt_pct(env.get("check_success", {}).get("seconds", 0.0), in_env),
            _fmt_pct(env.get("extract_state", {}).get("seconds", 0.0), in_env),
            _fmt_pct(env.get("reset_construct", {}).get("seconds", 0.0), in_env),
            _fmt_pct(env.get("set_init_state", {}).get("seconds", 0.0), in_env),
            _fmt_pct(env.get("wait_steps", {}).get("seconds", 0.0), in_env),
            in_env,
        )
    )
    print(
        "      outcomes: {} ok / {} timeout / {} early — running SR "
        "would need full episode set".format(
            out["success"], out["timeout"], out["early_term"]
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile where rollout-evaluation wall-clock time goes."
    )
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument(
        "--tasks", default="0,1",
        help="Comma-separated task ids to profile (default: 0,1).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--out", default="outputs/eval/profile_rollout.json",
        help="JSON output path for the structured breakdown.",
    )
    args = parser.parse_args()

    task_ids = [int(t) for t in args.tasks.split(",") if t.strip()]

    try:
        cfg = _compose_cfg(
            [
                "eval=rollout",
                f"eval.environment.suites=[{args.suite}]",
                "eval.environment.num_workers=1",
                f"eval.evaluation.num_episodes_per_task={args.episodes}",
                "eval.evaluation.episodes_per_suite=null",
            ]
        )
    except Exception as exc:  # noqa: BLE001 — top-level CLI diagnostic
        print(f"FATAL: could not compose hydra config: {exc}")
        return 1

    try:
        from phaseforge.cli import _resolve_device, build_eval_model
        from phaseforge.evaluations.envs.libero_env import SUITE_MAX_STEPS
        from phaseforge.evaluations.runners.rollout_evaluator import (
            SUITE_N_TASKS,
            RolloutEvaluator,
        )
        from phaseforge.utils.seed import set_seed
    except ImportError as exc:
        print(f"FATAL: phaseforge not importable: {exc}")
        return 1

    if args.suite not in SUITE_N_TASKS:
        print(f"FATAL: unknown suite {args.suite!r}")
        return 1

    set_seed(cfg.project.seed)
    print(f"Suite {args.suite}: profiling tasks {task_ids} x {args.episodes} "
          f"episodes (max {SUITE_MAX_STEPS[args.suite]} steps/ep)…")

    try:
        device = _resolve_device(cfg)
        model = build_eval_model(cfg)
        model.to(device)
        evaluator = RolloutEvaluator(cfg=cfg, model=model, device=device)
    except RuntimeError as exc:
        print(f"FATAL: could not build model/evaluator: {exc}")
        return 1

    num_tasks = SUITE_N_TASKS[args.suite]
    episodes = list(range(args.episodes))
    tasks_out: list[dict] = []

    print("\nTask breakdown:")
    for tid in task_ids:
        try:
            _, _, _ = evaluator._evaluate_task(
                args.suite, tid, episodes, num_tasks
            )
        except Exception as exc:  # noqa: BLE001 — per-task diagnostic
            print(f"  [{tid}] FAILED: {exc}")
            continue
        bd = evaluator._last_task_breakdown
        tasks_out.append(bd)
        _print_task(tid, num_tasks, bd)

    if not tasks_out:
        print("No tasks profiled successfully — nothing to aggregate.")
        return 1

    # Aggregated summary across profiled tasks.
    n_eps = sum(t["episodes"] for t in tasks_out)
    total = sum(t["total_seconds"] for t in tasks_out)
    agg_phases = {
        k: sum(t["phases"][k] for t in tasks_out)
        for k in ("reset_seconds", "step_seconds", "infer_seconds", "other_seconds")
    }
    print(
        f"\nAggregated over {n_eps} episode(s) ({total:.0f}s):"
    )
    print(
        f"  mean {total / n_eps:.1f}s/ep | "
        f"{sum(t['mean_steps'] * t['episodes'] for t in tasks_out) / n_eps:.0f} "
        "steps/ep"
    )
    print("  phases: reset {:s} | step {:s} | infer {:s} | other {:s}".format(
        _fmt_pct(agg_phases["reset_seconds"], total),
        _fmt_pct(agg_phases["step_seconds"], total),
        _fmt_pct(agg_phases["infer_seconds"], total),
        _fmt_pct(agg_phases["other_seconds"], total),
    ))

    summary = {
        "suite": args.suite,
        "tasks": task_ids,
        "episodes_per_task": args.episodes,
        "max_steps": SUITE_MAX_STEPS[args.suite],
        "num_episodes": n_eps,
        "total_seconds": total,
        "mean_episode_seconds": total / n_eps,
        "mean_steps": sum(
            t["mean_steps"] * t["episodes"] for t in tasks_out
        ) / n_eps,
        "phases": {k: v for k, v in agg_phases.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "tasks": tasks_out}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
