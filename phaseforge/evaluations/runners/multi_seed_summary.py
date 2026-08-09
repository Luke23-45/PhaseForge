"""Pure aggregation and statistics for multi-seed rollout evaluation.

This module deliberately contains no LIBERO/GPU/CLI code. It is the single
source of truth for:

- the suite protocol spec (Decision 2, issues register A2; episode counts
  locked at E5): ``libero_90`` in-distribution, ``libero_10`` zero-shot;
- per-suite subprocess timeout sizing (the previous fixed 7200 s timeout
  killed every ``libero_90`` run: 4500 episodes at ~13 s/ep with 2 workers
  is ~8.2 h);
- parsing a single seed's ``eval_results.json`` into a :class:`SeedResult`
  where a missing/failed value is ``None`` — never a fabricated ``0.0``
  (a crashed seed must stay distinguishable from "the policy scored 0%");
- per-cell aggregation with honest statistics: ``mean +/- std`` (ddof=1)
  plus a 95% percentile bootstrap CI over task-level rates, following the
  few-run guidance of Agarwal et al. 2021 ("Deep RL at the Edge of the
  Statistical Precipice", rliable).

The orchestration script ``scripts/run_multi_seed_eval.py`` is a thin
subprocess layer on top of this module; tests import this module directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SuiteSpec:
    """Protocol spec for one LIBERO suite.

    ``episodes_per_task`` is what ``rollout.yaml`` currently names
    ``episodes_per_suite``: it is episodes per TASK within the suite
    (libero_90: 50 eps x 90 tasks = 4500; libero_10: 10 x 10 = 100).
    """

    name: str
    n_tasks: int
    episodes_per_task: int
    max_steps: int
    role: str  # "in-distribution" | "zero-shot"

    @property
    def n_episodes(self) -> int:
        return self.n_tasks * self.episodes_per_task


# Decision 2 (issues register A2) + E5: only these two suites are evaluated.
SUITES: dict[str, SuiteSpec] = {
    "libero_90": SuiteSpec(
        name="libero_90",
        n_tasks=90,
        episodes_per_task=50,
        max_steps=400,
        role="in-distribution",
    ),
    "libero_10": SuiteSpec(
        name="libero_10",
        n_tasks=10,
        episodes_per_task=10,
        max_steps=520,
        role="zero-shot",
    ),
}

MIN_SEEDS_FOR_SUMMARY = 3  # B5: paper tables require 3 seeds.

# Timeout sizing. Measured on free Colab T4 with 2 workers: ~13.1 s/episode
# (F3 register: libero_spatial 500 eps / 6539 s). Independent reference:
# vla-eval (arXiv 2603.13966) reports ~14 h for 2000 *vision* episodes;
# state-only policies are at the lower end. The 50% buffer absorbs
# subprocess startup and simulator variance.
SECONDS_PER_EPISODE_ESTIMATE = 13.0
DEFAULT_WORKERS = 2
TIMEOUT_BUFFER = 1.5


def estimated_timeout_s(
    suite: SuiteSpec,
    workers: int = DEFAULT_WORKERS,
    buffer: float = TIMEOUT_BUFFER,
) -> int:
    """Size a subprocess timeout to the suite's real workload."""
    raw = suite.n_episodes * SECONDS_PER_EPISODE_ESTIMATE / max(int(workers), 1)
    return int(raw * buffer)


@dataclass
class SeedResult:
    """Parsed outcome of one (cell, suite, seed) evaluation.

    ``success_rate`` is ``None`` for any missing/failed run — never ``0.0``.
    """

    seed: int
    suite: str
    success_rate: float | None
    n_episodes_run: int | None
    per_task_rates: list[float]  # aligned per-task rates, task order
    raw_path: Path | None
    error: str | None = None


def parse_seed_result(
    results_path: Path, suite: SuiteSpec, expected_seed: int
) -> SeedResult:
    """Read one seed's ``eval_results.json`` into a :class:`SeedResult`.

    Every failure mode (missing file, malformed JSON, missing rate key,
    ``eval/seed`` mismatch, episode-count mismatch) returns
    ``success_rate=None`` with the reason in ``error``.
    """
    base = SeedResult(
        seed=expected_seed,
        suite=suite.name,
        success_rate=None,
        n_episodes_run=None,
        per_task_rates=[],
        raw_path=results_path,
    )
    if not results_path.is_file():
        base.error = f"no eval_results.json at {results_path}"
        return base

    try:
        payload = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        base.error = f"malformed eval_results.json: {exc}"
        return base

    rate_key = f"eval/success_rate/{suite.name}"
    rate = payload.get(rate_key)
    if rate is None:
        base.error = f"missing {rate_key!r} in {results_path.name}"
        return base
    try:
        rate = float(rate)
    except (TypeError, ValueError):
        base.error = f"{rate_key!r} is not numeric in {results_path.name}"
        return base
    if not 0.0 <= rate <= 1.0:
        base.error = f"{rate_key!r}={rate} out of [0, 1] in {results_path.name}"
        return base

    seed_val = payload.get("eval/seed")
    if seed_val != expected_seed:
        base.error = (
            f"eval/seed={seed_val!r} does not match requested seed "
            f"{expected_seed} in {results_path.name}"
        )
        return base

    n_eps = payload.get(f"eval/total_episodes/{suite.name}")
    if n_eps != suite.n_episodes:
        base.n_episodes_run = int(n_eps) if isinstance(n_eps, int) else None
        base.error = (
            f"ran {n_eps!r}/{suite.n_episodes} episodes — incomplete "
            f"({results_path.name})"
        )
        return base

    per_task = payload.get(f"eval/per_task/{suite.name}", {}) or {}
    per_task_rates: list[float] = []
    for stats in per_task.values():
        episodes = stats.get("episodes", 0)
        if episodes:
            per_task_rates.append(float(stats.get("successes", 0)) / float(episodes))

    base.success_rate = rate
    base.n_episodes_run = int(n_eps)
    base.per_task_rates = per_task_rates
    return base


def _bootstrap_suite_samples(
    per_suite_task_lists: dict[str, list[list[float]]],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Per-suite arrays of n_boot aggregate means (stratified bootstrap).

    Mirrors rliable's ``stratified_bootstrap`` (Agarwal et al. 2021):
    each replicate resamples BOTH axes of the per-seed task-rate matrix
    with replacement — ``num_runs`` runs and ``num_tasks`` tasks, i.e.
    ``arr[np.ix_(run_idx, task_idx)].mean()``. Resampling only the run
    axis would preserve the value multiset and produce zero-variance
    replicates, which is precisely the undercoverage trap this procedure
    exists to avoid.

    A suite is only present when at least 2 seeds contribute per-task
    data; per-task lists are truncated to the shortest one so every seed
    covers the same task grid.
    """
    samples: dict[str, np.ndarray] = {}
    for name, seed_task_lists in per_suite_task_lists.items():
        usable = [t for t in seed_task_lists if t]
        if len(usable) < 2:
            continue
        n_tasks = min(len(t) for t in usable)
        arr = np.asarray([t[:n_tasks] for t in usable], dtype=float)
        n_seeds, _ = arr.shape
        boot = np.empty(n_boot)
        for i in range(n_boot):
            run_idx = rng.integers(0, n_seeds, size=n_seeds)
            task_idx = rng.integers(0, n_tasks, size=n_tasks)
            boot[i] = arr[np.ix_(run_idx, task_idx)].mean()
        samples[name] = boot
    return samples


def _overall_samples(samples: dict[str, np.ndarray]) -> np.ndarray:
    """Per-replicate overall aggregate: unweighted mean of suite means.

    Empty when no suite produced samples (caller treats that as NaN).
    """
    arrays = list(samples.values())
    if not arrays:
        return np.array([])
    return np.stack(arrays).mean(axis=0)


def _percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    lo = float(np.percentile(samples, (1 - ci) / 2 * 100))
    hi = float(np.percentile(samples, (1 + ci) / 2 * 100))
    return (lo, hi)


def bootstrap_cis(
    per_task_by_seed: dict[str, list[list[float]]],
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng_seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Joint stratified bootstrap CIs over task-level success rates.

    ``per_task_by_seed`` maps suite name -> list (one entry per valid
    seed) of per-task success rates, aligned by task order within each
    seed.

    Runs (seeds) and tasks are both resampled with replacement per
    replicate (rliable's stratified bootstrap, Agarwal et al. 2021);
    the suite rate and the overall rate (unweighted mean of suite rates)
    are recomputed per replicate.

    Returns ``{suite_name: (lo, hi), "overall": (lo, hi)}``. Pairs are
    NaN when fewer than 2 seeds contribute per-task data for a suite.
    """
    rng = np.random.default_rng(rng_seed)
    samples = _bootstrap_suite_samples(per_task_by_seed, n_boot, rng)
    overall = _overall_samples(samples)

    out: dict[str, tuple[float, float]] = {}
    for name in per_task_by_seed:
        arr = samples.get(name)
        out[name] = _percentile_ci(arr, ci) if arr is not None else (float("nan"), float("nan"))
    out["overall"] = (
        _percentile_ci(overall, ci) if overall.size else (float("nan"), float("nan"))
    )
    return out


def probability_of_improvement(
    a_per_suite: dict[str, list[list[float]]],
    b_per_suite: dict[str, list[list[float]]],
    n_boot: int = 10_000,
    rng_seed: int = 0,
) -> float:
    """rliable-style probability of improvement: P(aggregate(A) > aggregate(B)).

    Both cells are resampled from the same stratified-bootstrap grid
    (rliable's procedure, see :func:`_bootstrap_suite_samples`) and their
    overall aggregates (unweighted mean of suite means) compared
    replicate-by-replicate. The two cells draw from INDEPENDENT streams
    so that two identical cells yield P ~ 0.5 rather than a degenerate
    0 or 1; results stay deterministic for a fixed ``rng_seed``.
    This is the few-runs comparison statistic of Agarwal et al. 2021
    (rliable's probability of improvement), and the honest answer to
    "is cell A actually better than cell B" at n=3.

    Returns NaN when either side has fewer than 2 usable seeds.
    """
    rng_a = np.random.default_rng(rng_seed)
    rng_b = np.random.default_rng(rng_seed + 1_000_003)
    a_samples = _overall_samples(
        _bootstrap_suite_samples(a_per_suite, n_boot, rng_a)
    )
    b_samples = _overall_samples(
        _bootstrap_suite_samples(b_per_suite, n_boot, rng_b)
    )
    if a_samples.size == 0 or b_samples.size == 0:
        return float("nan")
    return float(np.mean(a_samples > b_samples))


def mann_whitney_u(
    a_rates: list[float], b_rates: list[float]
) -> dict[str, float | None]:
    """Two-sided Mann-Whitney U on per-seed aggregate rates (scipy).

    Note: with 3 vs 3 seeds the test is weak (minimum attainable p is
    0.1); it is reported for completeness alongside the bootstrap
    probability of improvement, never instead of it.
    """
    if len(a_rates) < 2 or len(b_rates) < 2:
        return {"u": None, "p": None}
    from scipy import stats  # scipy is a declared dependency (transitively pinned)

    res = stats.mannwhitneyu(a_rates, b_rates, alternative="two-sided")
    return {"u": float(res.statistic), "p": float(res.pvalue)}


def compare_cells(
    cell_a: str,
    cell_b: str,
    a_seed_rates: dict[int, float],
    b_seed_rates: dict[int, float],
    a_task_rates_by_suite: dict[str, dict[int, list[float]]],
    b_task_rates_by_suite: dict[str, dict[int, list[float]]],
    n_boot: int = 10_000,
    rng_seed: int = 0,
) -> dict:
    """Head-to-head comparison of two cells on the overall aggregate.

    ``a_seed_rates`` / ``b_seed_rates`` map seed -> per-seed overall
    rate; ``*_task_rates_by_suite`` map suite -> {seed: per-task rates}.
    Only seeds valid in EVERY suite of BOTH cells are used — a crashed
    seed must never tilt a comparison. ``prob_a_over_b`` is the
    rliable-style probability of improvement; ``mann_whitney_u`` is the
    U test on the per-seed aggregate rates. Both are ``None`` when fewer
    than 2 common seeds exist.
    """
    common = sorted(set(a_seed_rates) & set(b_seed_rates))
    result: dict = {
        "cell_a": cell_a,
        "cell_b": cell_b,
        "seeds_used": common,
        "mean_a": None,
        "mean_b": None,
        "margin_a_minus_b": None,
        "prob_a_over_b": None,
        "prob_b_over_a": None,
        "mann_whitney_u": {"u": None, "p": None},
    }
    if len(common) < 2:
        result["note"] = (
            "fewer than 2 seeds valid in every suite of both cells — "
            "comparison not computed"
        )
        return result

    a_rates = [a_seed_rates[s] for s in common]
    b_rates = [b_seed_rates[s] for s in common]
    result["mean_a"] = float(np.mean(a_rates))
    result["mean_b"] = float(np.mean(b_rates))
    result["margin_a_minus_b"] = result["mean_a"] - result["mean_b"]

    shared_suites = sorted(set(a_task_rates_by_suite) & set(b_task_rates_by_suite))
    a_per_suite = {
        name: [
            a_task_rates_by_suite[name][s]
            for s in common
            if s in a_task_rates_by_suite[name] and a_task_rates_by_suite[name][s]
        ]
        for name in shared_suites
    }
    b_per_suite = {
        name: [
            b_task_rates_by_suite[name][s]
            for s in common
            if s in b_task_rates_by_suite[name] and b_task_rates_by_suite[name][s]
        ]
        for name in shared_suites
    }
    if all(len(v) >= 2 for v in a_per_suite.values()) and all(
        len(v) >= 2 for v in b_per_suite.values()
    ):
        result["prob_a_over_b"] = probability_of_improvement(
            a_per_suite, b_per_suite, n_boot=n_boot, rng_seed=rng_seed
        )
        p = result["prob_a_over_b"]
        result["prob_b_over_a"] = 1.0 - p if p == p else None
    else:
        result["note"] = (
            "fewer than 2 seeds with per-task data in some suite — "
            "probability of improvement not computed"
        )
    result["mann_whitney_u"] = mann_whitney_u(a_rates, b_rates)
    return result


def summarize_cell(
    cell: str,
    suite: SuiteSpec,
    results: list[SeedResult],
    min_seeds: int = MIN_SEEDS_FOR_SUMMARY,
) -> dict:
    """Aggregate a cell's seed results into one suite summary dict.

    Missing seeds stay absent (``per_seed`` only lists valid seeds,
    ``seed_errors`` records every failure reason). Stats are ``None`` when
    no valid seeds exist, and a cell below ``min_seeds`` is marked
    ``complete: False`` — it must be flagged in any table, not silently
    reported as the 3-seed statistic.
    """
    valid = [r for r in results if r.success_rate is not None]
    n_valid = len(valid)

    summary: dict = {
        "cell": cell,
        "suite": suite.name,
        "suite_role": suite.role,
        "seeds_requested": len(results),
        "seeds_valid": n_valid,
        "complete": n_valid >= min_seeds,
        "mean": None,
        "std": None,
        "ci95": None,
        "per_seed": {},
        "seed_errors": {},
    }
    for r in results:
        if r.success_rate is None:
            summary["seed_errors"][str(r.seed)] = r.error or "failed"
        else:
            summary["per_seed"][str(r.seed)] = float(r.success_rate)

    if n_valid == 0:
        logger.error("cell=%s suite=%s has ZERO valid seeds", cell, suite.name)
        return summary
    if n_valid < min_seeds:
        logger.warning(
            "cell=%s suite=%s has only %d/%d valid seeds — flag this cell "
            "explicitly in any table",
            cell, suite.name, n_valid, len(results),
        )

    rates: list[float] = []
    for r in valid:
        if r.success_rate is not None:
            rates.append(float(r.success_rate))
    summary["mean"] = float(np.mean(rates))
    summary["std"] = float(np.std(rates, ddof=1)) if n_valid > 1 else None

    complete_task_lists = [r.per_task_rates for r in valid if r.per_task_rates]
    if n_valid > 1 and len(complete_task_lists) >= 2:
        cis = bootstrap_cis({suite.name: complete_task_lists})
        summary["ci95"] = list(cis[suite.name])
    elif n_valid > 1:
        logger.warning(
            "cell=%s suite=%s: bootstrap CI skipped — per-task data missing "
            "for %d seed(s)",
            cell, suite.name, n_valid - len(complete_task_lists),
        )
    return summary


def summarize_overall(
    cell: str,
    suite_summaries: list[dict],
    per_suite_seed_results: dict[str, list[SeedResult]],
    min_seeds: int = MIN_SEEDS_FOR_SUMMARY,
) -> dict:
    """Across-suite cell summary: unweighted mean of suite means.

    A seed contributes to the overall only if it has a valid rate in
    EVERY suite (never padded with 0.0); ``seeds_valid`` counts exactly
    those seeds. The overall CI is the joint stratified bootstrap across
    all suites.
    """
    suite_names = [s["suite"] for s in suite_summaries]
    seeds_requested = max(
        (s["seeds_requested"] for s in suite_summaries), default=0
    )

    per_seed_rates: dict[int, dict[str, float]] = {}
    for suite_name in suite_names:
        for r in per_suite_seed_results[suite_name]:
            if r.success_rate is not None:
                per_seed_rates.setdefault(r.seed, {})[suite_name] = float(
                    r.success_rate
                )
    complete_seeds = sorted(
        s for s, rates in per_seed_rates.items() if len(rates) == len(suite_names)
    )

    seed_errors: dict[str, list[str]] = {}
    for suite_name in suite_names:
        for r in per_suite_seed_results[suite_name]:
            if r.success_rate is None:
                seed_errors.setdefault(str(r.seed), []).append(
                    f"{suite_name}: {r.error or 'failed'}"
                )

    summary: dict = {
        "cell": cell,
        "suite": "overall",
        "suite_role": "unweighted mean of suite means",
        "seeds_requested": seeds_requested,
        "seeds_valid": len(complete_seeds),
        "complete": len(complete_seeds) >= min_seeds,
        "mean": None,
        "std": None,
        "ci95": None,
        "per_seed": {
            str(s): float(
                np.mean([per_seed_rates[s][sn] for sn in suite_names])
            )
            for s in complete_seeds
        },
        "seed_errors": {
            s: " | ".join(errs) for s, errs in sorted(seed_errors.items())
        },
    }
    if not complete_seeds:
        return summary
    if len(complete_seeds) < min_seeds:
        logger.warning(
            "cell=%s overall has only %d/%d complete seeds — flag explicitly",
            cell, len(complete_seeds), seeds_requested,
        )

    rates = list(summary["per_seed"].values())
    summary["mean"] = float(np.mean(rates))
    summary["std"] = float(np.std(rates, ddof=1)) if len(rates) > 1 else None

    task_data = {
        sn: [
            r.per_task_rates
            for r in per_suite_seed_results[sn]
            if r.success_rate is not None
            and r.seed in complete_seeds
            and r.per_task_rates
        ]
        for sn in suite_names
    }
    if len(rates) > 1 and all(len(v) >= 2 for v in task_data.values()):
        summary["ci95"] = list(bootstrap_cis(task_data)["overall"])
    return summary
