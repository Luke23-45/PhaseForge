"""CPU-only unit tests for the multi-seed evaluation summary logic.

These tests exercise :mod:`phaseforge.evaluations.runners.multi_seed_summary`
without LIBERO, robosuite, GPU, or subprocesses: protocol constants,
per-suite timeout sizing, ``eval_results.json`` parsing (every failure
mode returns ``None`` — never a fabricated ``0.0``), per-cell and
across-suite aggregation with ddof=1 statistics and the joint stratified
bootstrap CI, and the completeness floor (a cell below
``MIN_SEEDS_FOR_SUMMARY`` valid seeds must be flagged, not reported).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phaseforge.evaluations.runners.multi_seed_summary import (
    MIN_SEEDS_FOR_SUMMARY,
    SUITES,
    SeedResult,
    SuiteSpec,
    bootstrap_cis,
    compare_cells,
    estimated_timeout_s,
    mann_whitney_u,
    parse_seed_result,
    probability_of_improvement,
    summarize_cell,
    summarize_overall,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

TINY = SuiteSpec(
    name="test_suite",
    n_tasks=3,
    episodes_per_task=10,
    max_steps=400,
    role="in-distribution",
)


def write_payload(
    path: Path,
    suite: SuiteSpec,
    seed: int,
    rate: float,
    per_task_rates: list[float] | None = None,
    *,
    bad_json: bool = False,
    seed_value: int | None = None,
    n_episodes: int | None = None,
    rate_value: object = None,
    drop_per_task: bool = False,
) -> Path:
    """Write an ``eval_results.json`` with controllable corruptions."""
    payload: dict = {
        f"eval/success_rate/{suite.name}": rate_value
        if rate_value is not None
        else rate,
        f"eval/total_episodes/{suite.name}": n_episodes
        if n_episodes is not None
        else suite.n_episodes,
        "eval/seed": seed_value if seed_value is not None else seed,
    }
    if not drop_per_task:
        tasks = per_task_rates or [rate] * suite.n_tasks
        payload[f"eval/per_task/{suite.name}"] = {
            f"task_{i}": {
                "successes": round(t * suite.episodes_per_task),
                "episodes": suite.episodes_per_task,
            }
            for i, t in enumerate(tasks)
        }
    if bad_json:
        path.write_text("{not json")
    else:
        path.write_text(json.dumps(payload))
    return path


def seed_result(
    seed: int,
    suite: str,
    rate: float | None,
    per_task: list[float] | None = None,
    error: str | None = None,
) -> SeedResult:
    return SeedResult(
        seed=seed,
        suite=suite,
        success_rate=rate,
        n_episodes_run=100 if rate is not None else None,
        per_task_rates=(
            per_task
            if per_task is not None
            else ([] if rate is None else [rate] * 3)
        ),
        raw_path=None,
        error=error,
    )


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------


def test_suite_registry_locked() -> None:
    """Decision 2: only libero_90 (in-distribution) and libero_10 (zero-shot)."""
    assert set(SUITES) == {"libero_90", "libero_10"}
    assert SUITES["libero_90"].n_episodes == 4500  # 90 tasks x 50 episodes
    assert SUITES["libero_90"].role == "in-distribution"
    assert SUITES["libero_10"].n_episodes == 100  # 10 tasks x 10 episodes
    assert SUITES["libero_10"].role == "zero-shot"
    assert MIN_SEEDS_FOR_SUMMARY == 3


# ---------------------------------------------------------------------------
# Timeout sizing
# ---------------------------------------------------------------------------


def test_estimated_timeout_sizes() -> None:
    """Timeout tracks the suite workload, not a fixed 7200 s cap."""
    # 4500 eps x 13 s / 2 workers x 1.5 buffer = 43,875 s (~12.2 h)
    assert estimated_timeout_s(SUITES["libero_90"]) == 43875
    # 100 eps x 13 s / 2 workers x 1.5 buffer = 975 s (~16 min)
    assert estimated_timeout_s(SUITES["libero_10"]) == 975
    assert estimated_timeout_s(SUITES["libero_10"], workers=1) == 1950
    # workers is floored at 1 — never a division-by-zero
    assert estimated_timeout_s(SUITES["libero_10"], workers=0) == 1950


# ---------------------------------------------------------------------------
# parse_seed_result failure modes — every failure yields None, never 0.0
# ---------------------------------------------------------------------------


def test_parse_missing_file(tmp_path: Path) -> None:
    result = parse_seed_result(tmp_path / "nope.json", TINY, 42)
    assert result.success_rate is None
    assert result.error and "no eval_results.json" in result.error


def test_parse_malformed_json(tmp_path: Path) -> None:
    write_payload(tmp_path / "eval_results.json", TINY, 42, 0.5, bad_json=True)
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate is None
    assert "malformed" in result.error


def test_parse_missing_rate_key(tmp_path: Path) -> None:
    write_payload(tmp_path / "eval_results.json", TINY, 42, 0.5)
    payload = json.loads((tmp_path / "eval_results.json").read_text())
    del payload["eval/success_rate/test_suite"]
    (tmp_path / "eval_results.json").write_text(json.dumps(payload))
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate is None
    assert "missing" in result.error


def test_parse_non_numeric_rate(tmp_path: Path) -> None:
    write_payload(tmp_path / "eval_results.json", TINY, 42, 0.5, rate_value="nan!!")
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate is None


def test_parse_out_of_range_rate(tmp_path: Path) -> None:
    write_payload(tmp_path / "eval_results.json", TINY, 42, 0.5, rate_value=1.7)
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate is None
    assert "out of [0, 1]" in result.error


def test_parse_seed_mismatch(tmp_path: Path) -> None:
    write_payload(tmp_path / "eval_results.json", TINY, 42, 0.5, seed_value=43)
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate is None
    assert "seed" in result.error


def test_parse_incomplete_episodes(tmp_path: Path) -> None:
    write_payload(
        tmp_path / "eval_results.json", TINY, 42, 0.5, n_episodes=17
    )
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate is None
    assert result.n_episodes_run == 17
    assert "incomplete" in result.error


def test_parse_valid(tmp_path: Path) -> None:
    write_payload(
        tmp_path / "eval_results.json", TINY, 42, 0.5,
        per_task_rates=[0.3, 0.5, 0.7],
    )
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate == 0.5
    assert result.n_episodes_run == TINY.n_episodes
    assert result.per_task_rates == pytest.approx([0.3, 0.5, 0.7])
    assert result.error is None


def test_parse_zero_rate_is_valid_not_a_failure(tmp_path: Path) -> None:
    """A legit 0.0 must be parsed as a valid rate — never treated as failure."""
    write_payload(tmp_path / "eval_results.json", TINY, 42, 0.0)
    result = parse_seed_result(tmp_path / "eval_results.json", TINY, 42)
    assert result.success_rate == 0.0
    assert result.error is None


# ---------------------------------------------------------------------------
# summarize_cell
# ---------------------------------------------------------------------------


def test_cell_three_valid_seeds() -> None:
    results = [
        seed_result(42, TINY.name, 0.5),
        seed_result(43, TINY.name, 0.6),
        seed_result(44, TINY.name, 0.7),
    ]
    summary = summarize_cell("cell_a", TINY, results)
    assert summary["complete"] is True
    assert summary["seeds_valid"] == 3
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["std"] == pytest.approx(0.1)  # ddof=1, not population std
    assert summary["ci95"] is not None
    assert summary["per_seed"] == {"42": 0.5, "43": 0.6, "44": 0.7}
    assert summary["seed_errors"] == {}


def test_cell_two_valid_one_failed_is_incomplete() -> None:
    results = [
        seed_result(42, TINY.name, 0.5),
        seed_result(43, TINY.name, 0.6),
        seed_result(44, TINY.name, None, error="simulated crash"),
    ]
    summary = summarize_cell("cell_a", TINY, results)
    assert summary["complete"] is False  # below MIN_SEEDS_FOR_SUMMARY
    assert summary["seeds_valid"] == 2
    assert summary["mean"] == pytest.approx(0.55)
    assert summary["seed_errors"]["44"] == "simulated crash"


def test_cell_zero_valid() -> None:
    results = [
        seed_result(42, TINY.name, None, error="crash"),
        seed_result(43, TINY.name, None, error="crash"),
        seed_result(44, TINY.name, None, error="crash"),
    ]
    summary = summarize_cell("cell_a", TINY, results)
    assert summary["complete"] is False
    assert summary["mean"] is None
    assert summary["std"] is None
    assert summary["ci95"] is None
    assert summary["per_seed"] == {}


def test_cell_single_valid_seed_stats_none() -> None:
    summary = summarize_cell("cell_a", TINY, [seed_result(42, TINY.name, 0.5)])
    assert summary["complete"] is False
    assert summary["mean"] == 0.5
    assert summary["std"] is None  # needs >= 2 seeds
    assert summary["ci95"] is None


def test_cell_ci_skipped_without_per_task_data() -> None:
    """CI is skipped (not fabricated) when per-task data is absent."""
    results = [
        seed_result(42, TINY.name, 0.5, per_task=[]),
        seed_result(43, TINY.name, 0.6, per_task=[]),
        seed_result(44, TINY.name, 0.7, per_task=[]),
    ]
    summary = summarize_cell("cell_a", TINY, results)
    assert summary["complete"] is True
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["ci95"] is None


# ---------------------------------------------------------------------------
# summarize_overall
# ---------------------------------------------------------------------------


def test_overall_excludes_seed_failed_in_any_suite() -> None:
    suite_b = SuiteSpec(
        name="suite_b", n_tasks=3, episodes_per_task=10,
        max_steps=400, role="zero-shot",
    )
    a_results = [
        seed_result(42, TINY.name, 0.5),
        seed_result(43, TINY.name, 0.6),
        seed_result(44, TINY.name, None, error="crash in a"),
    ]
    b_results = [
        seed_result(42, suite_b.name, 0.7),
        seed_result(43, suite_b.name, None, error="crash in b"),
        seed_result(44, suite_b.name, 0.8),
    ]
    summaries = [summarize_cell("cell_a", TINY, a_results),
                 summarize_cell("cell_a", suite_b, b_results)]
    overall = summarize_overall(
        "cell_a", summaries, {TINY.name: a_results, suite_b.name: b_results}
    )
    # Only seed 42 has a valid rate in BOTH suites.
    assert overall["seeds_valid"] == 1
    assert overall["complete"] is False
    assert overall["per_seed"] == {"42": 0.6}
    assert overall["mean"] == pytest.approx(0.6)
    assert overall["std"] is None
    assert "43" in overall["seed_errors"] and "44" in overall["seed_errors"]
    assert "suite_b" in overall["seed_errors"]["43"]


def test_overall_all_seeds_complete() -> None:
    suite_b = SuiteSpec(
        name="suite_b", n_tasks=3, episodes_per_task=10,
        max_steps=400, role="zero-shot",
    )
    a_results = [
        seed_result(42, TINY.name, 0.5),
        seed_result(43, TINY.name, 0.7),
        seed_result(44, TINY.name, 0.9),
    ]
    b_results = [
        seed_result(42, suite_b.name, 0.3),
        seed_result(43, suite_b.name, 0.5),
        seed_result(44, suite_b.name, 0.7),
    ]
    summaries = [summarize_cell("cell_a", TINY, a_results),
                 summarize_cell("cell_a", suite_b, b_results)]
    overall = summarize_overall(
        "cell_a", summaries, {TINY.name: a_results, suite_b.name: b_results}
    )
    assert overall["complete"] is True
    assert overall["seeds_valid"] == 3
    # Per-seed overall = mean of suite means: 42 -> 0.4, 43 -> 0.6, 44 -> 0.8
    assert overall["per_seed"] == {"42": 0.4, "43": 0.6, "44": 0.8}
    assert overall["mean"] == pytest.approx(0.6)
    assert overall["std"] == pytest.approx(0.2)  # ddof=1 over [0.4, 0.6, 0.8]
    assert overall["ci95"] is not None


def test_overall_no_complete_seeds() -> None:
    suite_b = SuiteSpec(
        name="suite_b", n_tasks=3, episodes_per_task=10,
        max_steps=400, role="zero-shot",
    )
    a_results = [seed_result(42, TINY.name, None, error="crash")]
    b_results = [seed_result(42, suite_b.name, None, error="crash")]
    summaries = [summarize_cell("cell_a", TINY, a_results),
                 summarize_cell("cell_a", suite_b, b_results)]
    overall = summarize_overall(
        "cell_a", summaries, {TINY.name: a_results, suite_b.name: b_results}
    )
    assert overall["complete"] is False
    assert overall["seeds_valid"] == 0
    assert overall["mean"] is None
    assert overall["ci95"] is None


# ---------------------------------------------------------------------------
# bootstrap_cis
# ---------------------------------------------------------------------------


def test_bootstrap_deterministic() -> None:
    per_task = {"s": [[0.4, 0.6, 0.5] for _ in range(3)]}
    first = bootstrap_cis(per_task, n_boot=500, rng_seed=7)
    second = bootstrap_cis(per_task, n_boot=500, rng_seed=7)
    assert first["s"] == second["s"]
    assert first["overall"] == second["overall"]


def test_bootstrap_constant_rates_exact() -> None:
    """All rates 0.5 -> the CI collapses to exactly (0.5, 0.5)."""
    per_task = {"s": [[0.5] * 50 for _ in range(3)]}
    cis = bootstrap_cis(per_task, n_boot=500, rng_seed=0)
    assert cis["s"] == pytest.approx((0.5, 0.5))
    assert cis["overall"] == pytest.approx((0.5, 0.5))


def test_bootstrap_ci_contains_observed_mean() -> None:
    rng = np.random.default_rng(0)
    task_rates = [float(rng.uniform(0.3, 0.7)) for _ in range(50)]
    per_task = {"s": [task_rates, task_rates, task_rates]}
    cis = bootstrap_cis(per_task, n_boot=2000, rng_seed=0)
    mean = float(np.mean(task_rates))
    assert cis["s"][0] <= mean <= cis["s"][1]


def test_bootstrap_ci_has_spread_with_varied_seeds() -> None:
    """Distinct per-seed data must yield a NON-degenerate CI.

    Regression pin: resampling only the seed axis preserves the value
    multiset (zero variance); the rliable stratified bootstrap resamples
    BOTH axes (runs and tasks) via ``arr[np.ix_(run_idx, task_idx)]``.
    """
    per_task = {
        "s": [[0.2] * 50, [0.6] * 50, [0.9] * 50],  # seeds 42/43/44
    }
    cis = bootstrap_cis(per_task, n_boot=2000, rng_seed=0)
    lo, hi = cis["s"]
    assert hi > lo  # the bug made hi == lo on every input
    assert lo <= 0.6 <= hi


def test_bootstrap_insufficient_seeds_nan() -> None:
    cis = bootstrap_cis({"s": [[0.5] * 10]}, n_boot=100, rng_seed=0)
    assert np.isnan(cis["s"][0]) and np.isnan(cis["s"][1])
    assert np.isnan(cis["overall"][0])


# ---------------------------------------------------------------------------
# probability_of_improvement (rliable-style) and Mann-Whitney U
# ---------------------------------------------------------------------------


def test_prob_improvement_dominant_cell() -> None:
    """A cell whose tasks are uniformly better must win the comparison."""
    a = {"s": [[0.8] * 50 for _ in range(3)]}  # cell A: 80% everywhere
    b = {"s": [[0.4] * 50 for _ in range(3)]}  # cell B: 40% everywhere
    p = probability_of_improvement(a, b, n_boot=500, rng_seed=0)
    assert p == pytest.approx(1.0)
    # Symmetric: B over A is the complement.
    p_rev = probability_of_improvement(b, a, n_boot=500, rng_seed=0)
    assert p_rev == pytest.approx(0.0)


def test_prob_improvement_identical_cells() -> None:
    """Identical cells -> P ~ 0.5 (no overclaiming a winner)."""
    same = {"s": [[0.5, 0.6, 0.5, 0.6] for _ in range(3)]}
    p = probability_of_improvement(same, same, n_boot=500, rng_seed=0)
    assert 0.35 <= p <= 0.65


def test_prob_improvement_insufficient_seeds_nan() -> None:
    a = {"s": [[0.5] * 10]}  # single seed -> no bootstrap possible
    b = {"s": [[0.6] * 10, [0.6] * 10]}
    assert np.isnan(probability_of_improvement(a, b, n_boot=100, rng_seed=0))


def test_mann_whitney_u_separated_groups() -> None:
    """Non-overlapping groups -> tiny p (minimal attainable at n=3 is 0.1)."""
    res = mann_whitney_u([0.4, 0.5, 0.6], [0.8, 0.9, 0.95])
    assert res["p"] == pytest.approx(0.1)
    assert res["u"] is not None


def test_mann_whitney_u_identical_groups() -> None:
    """Identical groups must not produce a reject-level p."""
    res = mann_whitney_u([0.4, 0.5, 0.6], [0.4, 0.5, 0.6])
    assert res["u"] is not None and res["p"] is not None
    assert 0.0 <= res["p"] <= 1.0
    assert res["p"] > 0.05  # cannot reject the null on identical groups


def test_mann_whitney_u_too_few_seeds() -> None:
    res = mann_whitney_u([0.5], [0.6])
    assert res == {"u": None, "p": None}


# ---------------------------------------------------------------------------
# compare_cells
# ---------------------------------------------------------------------------


def test_compare_cells_full() -> None:
    """Both cells complete on two suites -> full comparison block."""
    a_rates = {42: 0.4, 43: 0.6, 44: 0.8}
    b_rates = {42: 0.7, 43: 0.7, 44: 0.7}
    a_tasks = {
        "a": {42: [0.4, 0.4], 43: [0.6, 0.6], 44: [0.8, 0.8]},
        "b": {42: [0.4, 0.4], 43: [0.6, 0.6], 44: [0.8, 0.8]},
    }
    b_tasks = {
        "a": {42: [0.7, 0.7], 43: [0.7, 0.7], 44: [0.7, 0.7]},
        "b": {42: [0.7, 0.7], 43: [0.7, 0.7], 44: [0.7, 0.7]},
    }
    res = compare_cells(
        "cell_a", "cell_b", a_rates, b_rates, a_tasks, b_tasks,
        n_boot=500, rng_seed=0,
    )
    assert res["seeds_used"] == [42, 43, 44]
    assert res["mean_a"] == pytest.approx(0.6)
    assert res["mean_b"] == pytest.approx(0.7)
    assert res["margin_a_minus_b"] == pytest.approx(-0.1)
    # B's bootstrap distribution is degenerate at 0.7 (all its rows are
    # 0.7), so P(a > b) is only the mass of A's distribution above 0.7:
    # small but honestly non-zero.
    assert res["prob_a_over_b"] < 0.1
    assert res["prob_b_over_a"] == pytest.approx(1.0 - res["prob_a_over_b"])
    # scipy MWU (asymptotic, tie-corrected because of the tied 0.7s):
    # U=3.0, weak p ~0.64 — deliberately NOT a reject.
    assert res["mann_whitney_u"]["u"] == pytest.approx(3.0)
    assert 0.5 < res["mann_whitney_u"]["p"] < 0.8


def test_compare_cells_intersection_excludes_crashed_seeds() -> None:
    """A seed crashed in ONE suite of ONE cell is dropped from the pair."""
    a_rates = {42: 0.5, 43: 0.5, 44: 0.5}
    b_rates = {42: 0.6, 43: 0.6, 44: 0.6}
    a_tasks = {
        "s1": {42: [0.5], 43: [0.5], 44: [0.5]},
        "s2": {42: [0.5], 43: [0.5], 44: [0.5]},
    }
    b_tasks = {
        "s1": {42: [0.6], 43: [0.6], 44: [0.6]},
        "s2": {42: [0.6], 44: [0.6]},  # seed 43 failed in suite s2
    }
    res = compare_cells(
        "cell_a", "cell_b", a_rates, b_rates, a_tasks, b_tasks,
        n_boot=200, rng_seed=0,
    )
    # Seed 43 has no per-task data in suite s2 of cell B -> dropped.
    assert res["seeds_used"] == [42, 43, 44]
    assert res["prob_a_over_b"] == pytest.approx(0.0)
    assert "not computed" not in res.get("note", "")


def test_compare_cells_insufficient_common_seeds() -> None:
    a_rates = {42: 0.5, 43: 0.5}
    b_rates = {42: 0.6, 44: 0.6}  # common = {42} only
    res = compare_cells(
        "cell_a", "cell_b", a_rates, b_rates, {}, {}, n_boot=100, rng_seed=0,
    )
    assert res["prob_a_over_b"] is None
    assert res["mean_a"] is None
    assert "not computed" in res["note"]
