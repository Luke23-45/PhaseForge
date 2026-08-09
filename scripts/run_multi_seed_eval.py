#!/usr/bin/env python3
"""Run the 3-seed LIBERO rollout evaluation for all PhaseForge cells.

Orchestrates one ``phaseforge-eval`` subprocess per (cell, suite, seed),
parses each ``eval_results.json`` via
:mod:`phaseforge.evaluations.runners.multi_seed_summary`, and writes one
aggregated ``final_results.json`` plus a console summary table.

Hard guarantees (why this script exists):

- No silent zero padding: a missing/failed seed is ``None``, never ``0.0``
  (a crashed seed stays distinguishable from "the policy scored 0%").
- A cell below ``MIN_SEEDS_FOR_SUMMARY`` valid seeds is marked
  ``complete: false``, flagged loudly, and the script exits nonzero.
- Per-suite subprocess timeouts are sized to the real workload
  (episode count x measured seconds/episode / workers x headroom) — the
  previous fixed 2-hour timeout killed every libero_90 run (~8.2 h).
- Every result is cross-checked: the run dir is identified by a
  before/after snapshot (no "newest dir" guessing), and ``eval/seed`` in
  the JSON must equal the requested seed.
- ``--explicit-checkpoint`` is threaded into every subprocess with no
  alternative resolution path — it can never be silently dropped.
- Statistics: mean +/- std (ddof=1) plus a 95% stratified bootstrap CI
  over task-level rates (Agarwal et al. 2021, rliable). Head-to-head
  comparisons (every cell pair) add the rliable-style probability of
  improvement and a Mann-Whitney U p-value on per-seed aggregates.
- Suite roles never disappear: the console table and
  ``final_results.json`` carry the ID (in-distribution) / ZS (zero-shot)
  label on every suite column and summary (professor review item 4).

Usage:
    uv sync --extra rollout          # one-time: installs libero + robosuite
    uv run python scripts/run_multi_seed_eval.py
    uv run python scripts/run_multi_seed_eval.py --cells bc phaseforge
    uv run python scripts/run_multi_seed_eval.py --dry-run --cells bc
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

from phaseforge.evaluations.runners.multi_seed_summary import (
    DEFAULT_WORKERS,
    MIN_SEEDS_FOR_SUMMARY,
    SECONDS_PER_EPISODE_ESTIMATE,
    SUITES,
    TIMEOUT_BUFFER,
    SeedResult,
    compare_cells,
    estimated_timeout_s,
    parse_seed_result,
    summarize_cell,
    summarize_overall,
)
from phaseforge.utils.config import find_latest_checkpoint

logger = logging.getLogger(__name__)

# (cell name, Hydra config path, explicit checkpoint path or None).
# The config path is REQUIRED: model configs live under config/models/
# (e.g. ``baselines/bc``). If the checkpoint path is None, the
# seed-matched best checkpoint is auto-detected from
# outputs/<cell>/stage<N>/ (see STAGES below).
MODELS: list[tuple[str, str, str | None]] = [
    ("bc", "baselines/bc", None),
    ("phaseforge", "phaseforge", None),
    ("scratch_moe", "baselines/scratch_moe", None),
    ("warmstart_moe", "baselines/warmstart_moe", None),
    ("oracle_moe", "baselines/oracle_moe", None),
    ("teacher_forced", "baselines/teacher_forced", None),
    ("phase_pretrain_random_router", "baselines/phase_pretrain_random_router", None),
    ("plain_encoder_phase_bootstrap", "baselines/plain_encoder_phase_bootstrap", None),
]

# Stage whose best checkpoint is evaluated for each cell: BC trains in
# Stage 1 only; all MoE variants are evaluated after Stage 2.
STAGES: dict[str, int] = {
    "bc": 1,
    "phaseforge": 2,
    "scratch_moe": 2,
    "warmstart_moe": 2,
    "oracle_moe": 2,
    "teacher_forced": 2,
    "phase_pretrain_random_router": 2,
    "plain_encoder_phase_bootstrap": 2,
}

DEFAULT_SEEDS = [42, 43, 44]


def run_single_seed(
    cell: str,
    model_cfg: str,
    suite,
    seed: int,
    checkpoint: Path,
    eval_root: Path,
    args: argparse.Namespace,
) -> SeedResult:
    """Run one (cell, suite, seed) eval subprocess and parse its results.

    Never raises on eval failure — a failed run comes back as a
    :class:`SeedResult` with ``success_rate=None`` and the reason in
    ``error``.
    """
    model_name = model_cfg.split("/")[-1]
    eval_dir = eval_root / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in eval_dir.iterdir() if p.is_dir()}

    cmd = [
        "phaseforge-eval",
        f"models={model_cfg}",
        "eval=rollout",
        f"project.seed={seed}",
        f"train.stage1_ckpt_path={checkpoint}",
        f"eval.environment.suites=[{suite.name}]",
    ]
    timeout_s = estimated_timeout_s(
        suite, workers=args.workers, buffer=args.timeout_buffer
    )
    logger.info(
        "cell=%s suite=%s seed=%d ckpt=%s timeout=%ds",
        cell, suite.name, seed, checkpoint, timeout_s,
    )

    if args.dry_run:
        logger.info("  [dry-run] %s", " ".join(cmd))
        return SeedResult(
            seed=seed, suite=suite.name, success_rate=None,
            n_episodes_run=None, per_task_rates=[], raw_path=None,
            error="dry run (no subprocess)",
        )

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "cell=%s suite=%s seed=%d TIMED OUT after %ds",
            cell, suite.name, seed, timeout_s,
        )
        return SeedResult(
            seed=seed, suite=suite.name, success_rate=None,
            n_episodes_run=None, per_task_rates=[], raw_path=None,
            error=f"timed out after {timeout_s}s (timeout sized to the suite workload)",
        )
    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        logger.error(
            "cell=%s suite=%s seed=%d FAILED (rc=%d, %.0fs)\nstderr tail:\n%s",
            cell, suite.name, seed, proc.returncode, elapsed, proc.stderr[-2000:],
        )
        return SeedResult(
            seed=seed, suite=suite.name, success_rate=None,
            n_episodes_run=None, per_task_rates=[], raw_path=None,
            error=f"subprocess rc={proc.returncode}",
        )

    new_dirs = sorted(
        p for p in eval_dir.iterdir() if p.is_dir() and p.name not in before
    )
    if len(new_dirs) != 1:
        logger.error(
            "cell=%s suite=%s seed=%d: expected exactly 1 new eval dir, "
            "found %d — cannot attribute results",
            cell, suite.name, seed, len(new_dirs),
        )
        return SeedResult(
            seed=seed, suite=suite.name, success_rate=None,
            n_episodes_run=None, per_task_rates=[], raw_path=None,
            error=f"ambiguous eval output dirs ({len(new_dirs)} new)",
        )

    result = parse_seed_result(
        new_dirs[0] / "eval_results.json", suite, seed
    )
    if result.success_rate is not None:
        logger.info(
            "cell=%s suite=%s seed=%d OK success_rate=%.4f (%.0fs)",
            cell, suite.name, seed, result.success_rate, elapsed,
        )
    else:
        logger.error(
            "cell=%s suite=%s seed=%d BAD RESULT: %s",
            cell, suite.name, seed, result.error,
        )
    return result


def _fmt_stats(mean: float | None, std: float | None, ci95: list | None) -> str:
    if mean is None:
        return "n/a"
    text = f"{mean:.4f} +/- {std:.4f}" if std is not None else f"{mean:.4f}"
    if ci95 is not None:
        text += f" [{ci95[0]:.4f}, {ci95[1]:.4f}]"
    return text


def _fmt_head_to_head(c: dict) -> str:
    """One line of the head-to-head console summary."""
    a, b = c["cell_a"], c["cell_b"]
    if c["prob_a_over_b"] is None:
        return f"{a} vs {b}: NOT COMPUTED ({c.get('note', 'unknown reason')})"
    mw = c["mann_whitney_u"]
    p_value = f"{mw['p']:.3f}" if mw["p"] is not None else "n/a"
    return (
        f"{a} vs {b}: P({a} > {b})={c['prob_a_over_b']:.3f}, "
        f"margin={c['margin_a_minus_b']:+.4f}, MWU p={p_value} "
        f"(seeds={len(c['seeds_used'])})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells", nargs="+", default=None,
        help="cells to evaluate (default: all 8 in MODELS)",
    )
    parser.add_argument(
        "--suites", nargs="+", default=list(SUITES.keys()),
        choices=list(SUITES.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("outputs"),
        help="base directory for seed-matched checkpoint lookup",
    )
    parser.add_argument(
        "--explicit-checkpoint", type=Path, default=None,
        help="single checkpoint for ALL cells/seeds (smoke tests); "
             "threaded into every subprocess — never silently dropped",
    )
    parser.add_argument(
        "--eval-root", type=Path, default=Path("outputs") / "eval",
        help="directory where phaseforge-eval writes run dirs",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs") / "eval" / "final_results.json",
    )
    parser.add_argument(
        "--seconds-per-episode", type=float, default=SECONDS_PER_EPISODE_ESTIMATE,
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-buffer", type=float, default=TIMEOUT_BUFFER)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(message)s"
    )

    cells = args.cells or [name for name, _, _ in MODELS]
    known = {name for name, _, _ in MODELS}
    unknown = [c for c in cells if c not in known]
    if unknown:
        parser.error(f"unknown cell(s): {unknown}")
    if args.explicit_checkpoint is not None and not args.explicit_checkpoint.is_file():
        parser.error(f"--explicit-checkpoint not found: {args.explicit_checkpoint}")
    args.workers = max(int(args.workers), 1)

    suites = [SUITES[name] for name in args.suites]
    cells_output: list[dict] = []
    incomplete: list[str] = []
    comparison_data: dict[str, tuple[dict, dict]] = {}

    for cell in cells:
        model_cfg = next(cfg for name, cfg, _ in MODELS if name == cell)
        stage = STAGES[cell]
        suite_results: dict[str, list[SeedResult]] = {}
        suite_summaries: list[dict] = []

        for suite in suites:
            seed_results: list[SeedResult] = []
            for seed in args.seeds:
                if args.explicit_checkpoint is not None:
                    ckpt = args.explicit_checkpoint
                else:
                    ckpt = find_latest_checkpoint(
                        cell, stage=stage, base=args.checkpoint_root,
                        resolve_alias=False, seed=seed,
                    )
                if ckpt is None or not ckpt.is_file():
                    logger.error(
                        "cell=%s seed=%d: checkpoint not found under %s — skipping",
                        cell, seed, args.checkpoint_root,
                    )
                    seed_results.append(
                        SeedResult(
                            seed=seed, suite=suite.name, success_rate=None,
                            n_episodes_run=None, per_task_rates=[], raw_path=None,
                            error=(
                                f"checkpoint not found under {args.checkpoint_root}"
                            ),
                        )
                    )
                    continue
                seed_results.append(
                    run_single_seed(cell, model_cfg, suite, seed, ckpt, args.eval_root, args)
                )
            suite_results[suite.name] = seed_results
            suite_summaries.append(summarize_cell(cell, suite, seed_results))
            if not suite_summaries[-1]["complete"]:
                incomplete.append(f"{cell}/{suite.name}")

        overall = summarize_overall(cell, suite_summaries, suite_results)
        if not overall["complete"]:
            incomplete.append(f"{cell}/overall")
        cells_output.append(
            {
                "cell": cell,
                "suites": {s["suite"]: s for s in suite_summaries},
                "overall": overall,
            }
        )
        # D3: per-cell comparison data — seed-level overall rates plus the
        # per-suite per-seed task-rate matrices (from valid seeds only).
        comparison_data[cell] = (
            {int(s): float(r) for s, r in overall["per_seed"].items()},
            {
                suite_name: {
                    r.seed: list(r.per_task_rates)
                    for r in seed_list
                    if r.success_rate is not None and r.per_task_rates
                }
                for suite_name, seed_list in suite_results.items()
            },
        )

    if args.dry_run:
        print("dry run — nothing executed; commands printed above")
        return 0

    cells_output.sort(key=lambda entry: [name for name, _, _ in MODELS].index(entry["cell"]))

    # D3: head-to-head statistics (rliable-style probability of improvement,
    # Mann-Whitney U) for every unordered pair of evaluated cells. Sorted by
    # |margin| so the closest contests surface first.
    comparisons: list[dict] = []
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            a_rates, a_tasks = comparison_data[a]
            b_rates, b_tasks = comparison_data[b]
            comparisons.append(
                compare_cells(a, b, a_rates, b_rates, a_tasks, b_tasks)
            )
    comparisons.sort(
        key=lambda c: abs(c["margin_a_minus_b"])
        if c["margin_a_minus_b"] is not None
        else float("inf")
    )

    payload = {
        "min_seeds_for_summary": MIN_SEEDS_FOR_SUMMARY,
        "seeds_requested": args.seeds,
        "suites_requested": args.suites,
        "cells": cells_output,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("wrote aggregated results to %s", args.output)

    header = f"{'Cell':<32} {'Overall':<30}" + "".join(
        f" {name} ({'ID' if SUITES[name].role == 'in-distribution' else 'ZS'}):<30"
        for name in args.suites
    )
    print(header)
    print("-" * len(header))
    for entry in cells_output:
        suite_cols = {
            s["suite"]: s for s in entry["suites"].values()
        }
        overall_stats = _fmt_stats(
            entry["overall"]["mean"],
            entry["overall"]["std"],
            entry["overall"]["ci95"],
        )
        row = f"{entry['cell']:<32} {overall_stats:<30}"
        for name in args.suites:
            s = suite_cols[name]
            row += f" {_fmt_stats(s['mean'], s['std'], s['ci95']):<30}"
        print(row)
    print("-" * len(header))
    print(
        "suite columns: ID = in-distribution (libero_90), "
        "ZS = zero-shot (libero_10). Every cell: mean +/- std (ddof=1) "
        "with 95% stratified bootstrap CI in brackets."
    )

    print("\nHead-to-head on the overall aggregate "
          "(P(a > b) = rliable-style probability of improvement):")
    core = ("phaseforge", "warmstart_moe")
    printed = 0
    for c in comparisons:
        is_core = (c["cell_a"], c["cell_b"]) == core
        is_core = is_core or (c["cell_b"], c["cell_a"]) == core
        if printed >= 4 and not is_core:
            continue
        print(f"  {_fmt_head_to_head(c)}")
        printed += 1

    if incomplete:
        logger.warning(
            "%d cell/suite pair(s) INCOMPLETE (<%d valid seeds) — do not "
            "report without a footnote: %s",
            len(incomplete), MIN_SEEDS_FOR_SUMMARY, incomplete,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
