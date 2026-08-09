# PhaseForge Evaluation Harness — Review, Correction, and Redesign Spec

**Scope:** review of the debugging transcript for `scripts/run_multi_seed_eval.py`, the seed-variance question, and the ruff-count correction. Goal: a harness that cannot silently produce a wrong number.

---

## 0. Verdict

The protocol design (suites, episode counts, max steps, success metric, `suite_roles`, checkpoint wiring) is sound. The runner script is not — it has three independent bugs that guarantee either a crash or, worse, a table full of misleading zeros. Separately, one thing the transcript treats as "a design choice that needs a defensive footnote" is actually **standard LIBERO practice**, and the thing that actually needs fixing is a sentence in `evaluation_plan.md`, not the code. Both are addressed below, in the order you should act on them.

---

## 1. Confirmed bugs — fix these, in this order

| # | Bug | Why it matters | Fix |
|---|---|---|---|
| 1 | Dead block at lines 87–95 references `cfg` before it's created at line 97 | `NameError` on **every successful eval subprocess**. The script only survives when every eval *fails* — which is exactly backwards. | Delete the dead block. One path: build `cfg` → resolve `output_dir` → read `eval_results.json`. |
| 2 | Two overlapping aggregation blocks in `main()`; `overall_rates.append(...)` at line 148 runs before `overall_rates` is defined at line 151 | `NameError` in aggregation, every time. Looks like a bad merge — two versions of the same loop left in the file. | One aggregation pass. Declare accumulators once, before any loop that appends to them. See §6. |
| 3 | `suite_rates.append(suite_rates)` — appending to a dict as if it were a list | `AttributeError`, and even if it were a list, appending a collection to itself is never what you want. | Use a `dict[str, list[float]]` keyed by suite, appended to with `suite_rates[suite].append(rate)`. |
| 4 | `timeout=7200` (2h) | `libero_90` alone is 90 tasks × 50 episodes = 4,500 episodes. At your measured ~13s/episode with 2 workers, that's **~8.1 hours** — the job is killed on every single `libero_90` run, fixed script or not. | Size the timeout to the workload (see §6 `estimated_timeout_s`), not a round number. Budget real headroom (50%+), because subprocess overhead and simulator variance are real. |
| 5 | Missing `eval/success_rate` key silently becomes `0.0` via `.get(..., 0.0)`; a failed seed is dropped from the table with no warning | A 2-seed table gets reported as if it were the 3-seed statistic your protocol (B5) requires — and a crashed seed becomes indistinguishable from "the policy scored 0%." Those are opposite failure modes and your current code can't tell them apart. | Missing/failed → `None`, never `0.0`. Track `seeds_valid` vs `seeds_requested`. Refuse to call a cell "complete" below your seed floor, and say so loudly in the summary. |
| 6 | `explicit_ckpt` reportedly ignored (mentioned once in the transcript, not traced) | If true, a smoke-test run or an ad-hoc checkpoint override silently falls back to whatever `checkpoint_root` resolution would have picked — you'd think you tested one checkpoint and actually tested another. | **You need to verify this yourself** — the transcript doesn't show the trace. In the redesign (§6), the override is threaded explicitly into `run_single_seed` and there is no code path where it can be silently dropped. Confirm your actual file has the same property before trusting any smoke test you've run with it. |

None of bugs 1–5 are subtle — they're the kind that a real end-to-end run catches immediately (§7 explains why the previous testing approach missed them anyway).

---

## 2. Not actually a bug: identical episodes across seeds

The transcript treats "all 3 seeds replay the identical 4,600 episodes; only the checkpoint differs" as a possible protocol deviation requiring a defensive caveat in the writeup. It doesn't need a defensive caveat — **it's the standard LIBERO evaluation convention**, used across nearly every recent LIBERO paper:

- A recent adaptation-method paper reports using the initial states LIBERO ships for each task, with a single fixed evaluation seed, following what it calls the standard π0.5 LIBERO evaluation recipe (checkpoints evaluated at fixed intervals, best checkpoint reported).
- A recent action-model paper evaluates each of the four standard suites over 500 episodes drawn from LIBERO's initial-state set, with all suites sharing one trained checkpoint.
- A representation-steering paper's protocol explicitly describes using "LIBERO's built-in per-task initial state distribution," resetting with sequential episode seeds — i.e., the *episode index*, not a top-level "eval seed," determines which of LIBERO's canonical init states is used.

In other words: LIBERO ships a fixed, canonical list of initial states per task (`task_suite.get_task_init_states(task_id)`, exactly the API your code already calls), and the community-standard practice is to evaluate every checkpoint against that same fixed list. Your "seed" (42/43/44) is correctly doing what it's supposed to do in this design — identifying which *independently-trained model* is being evaluated, not resampling which episodes get run. That's precisely how you isolate training-seed variance, which is a legitimate and commonly reported statistic in its own right.

**What actually needs to change:** if `evaluation_plan.md` currently states initial states are "sampled i.i.d.," that sentence is factually wrong and should be corrected — not defended with a footnote, but rewritten to say what the code actually and correctly does:

> All three eval seeds evaluate the same fixed, LIBERO-provided initial-state list per task under deterministic inference. The reported statistic is success-rate variance across three independently-trained model checkpoints (seeds 42/43/44), not variance across sampled evaluation episodes.

That sentence is both accurate and defensible to a reviewer, and it costs you nothing — no code change, no re-run, no added variance. Delete the "should we add per-seed episode sampling" line of thinking entirely; the standard practice you already implemented is the correct one for what your paper needs to measure (training stability), and it's the same choice OpenVLA, π0/π0.5, ALAM, and others made.

---

## 3. Statistical reporting: `mean ± std` at n=3 is fragile — report a CI too

Three seeds means `np.std(..., ddof=1)` is being computed with **two degrees of freedom**. That number will bounce around a lot from one training run to the next, and if any two of your five cells (`bc`, `scratch_moe`, `warmstart_moe`, `phaseforge`, `oracle_moe`) land within a few points of each other, a naive mean-and-std table risks a false impression of a clear winner. This exact failure mode — the "predominant practice of reporting point estimates" hiding real uncertainty with only a handful of runs — is the specific problem the NeurIPS Outstanding Paper *Deep RL at the Edge of the Statistical Precipice* (Agarwal et al., 2021) was written to fix, and it comes with an open-source library (`rliable`) built for exactly the 3–10 run regime you're in.

Two concrete, low-cost changes:

1. **Report a bootstrap CI alongside mean ± std**, not instead of it. A simple percentile bootstrap over your 3 (or 5) seed-level success rates is a two-line addition (implemented in §6) and gives you an honest confidence band instead of an implied point estimate.
2. **If a reviewer will ask "is `phaseforge` actually better than `warmstart_moe`,"** don't answer with a mean comparison alone — use `rliable`'s probability-of-improvement statistic (a Mann–Whitney U-based measure) across your suites. It's designed precisely for comparing two methods with few runs without overclaiming.

Since compute isn't your binding constraint, bumping from 3 to 5 seeds for the head-to-head cells (`warmstart_moe` vs `phaseforge` specifically, since that's the paper's core claim) would meaningfully tighten these intervals for very little extra cost, and `rliable`'s own framing (3–10 runs) still calls 5 seeds "a handful" — worth the two extra runs if the comparison is close.

---

## 4. Good news: `SUITE_ID_OOD_ROLES` already fixes the earlier concern

Last review, the open question was whether training on the LIBERO-90 pool and evaluating on differently-named suites constituted an unacknowledged train/eval mismatch. This transcript shows that's resolved correctly: `SUITE_ID_OOD_ROLES` explicitly declares `libero_90` as in-distribution and `libero_10` as zero-shot, and that declaration is written into every `eval_results.json`. This matches the original LIBERO benchmark's own design — LIBERO-90 was built as the pretraining pool and the 10 long-horizon tasks (LIBERO-Long/LIBERO-10) as the held-out generalization test. Nothing to fix here; just confirm the paper's results tables actually carry the `suite_role` label through into the final write-up, since it's easy for a "clean" table to quietly drop the ID/OOD distinction during the final formatting pass.

---

## 5. Process fix: the ruff miscount, and how to make sure it can't happen again

The self-correction in the transcript is worth taking seriously as a process signal, not just a one-off mistake: an earlier `ruff check phaseforge tests scripts` run reported "26 errors," and that number was interpreted as "26 pre-existing `E501` line-length errors in `patch_docs.py`" — because only the *tail* of the output was inspected. The actual breakdown was roughly 16 in `patch_docs.py`, 11 `F821` (undefined-name) errors in `run_multi_seed_eval.py`, and 1 in `simulate_pipeline.py`. The broken runner's errors were sitting in that count the entire time, unnoticed, because the output was truncated before being read.

That's a review-process bug, not a ruff bug — `F821` is in ruff's default rule set (`E4`, `E7`, `E9`, `F`) and was firing correctly the whole time. Fix the process, not just this one instance:

- **CI should assert on ruff's exit code**, not on a human reading a truncated terminal tail. `ruff check phaseforge tests scripts` already returns non-zero on any error; wire that directly into the CI gate so a red build is unmissable, independent of what a human happened to scroll to.
- If you want a per-file breakdown for a PR comment or a log, use `ruff check --output-format=json` and group by file, rather than grepping for `-->` lines by hand — less room for exactly the truncation mistake that happened here.
- `scripts/` should be in the same lint gate as `phaseforge/` and `tests/` with no exceptions; the fact that a lint pass technically covered the directory but the failure was still missed shows the coverage wasn't the problem, the *reading* of the output was.

---

## 6. Reference redesign of `run_multi_seed_eval.py`

This is a clean, adapt-to-your-repo reference implementation — the import paths (`phaseforge.cli.rollout_eval`), checkpoint layout, and CLI surface are illustrative and will need to match your actual repo structure. What matters is the *shape*: single top-to-bottom control flow, every symbol defined before use, no silent zero-padding, timeout sized to the real workload, and statistics that don't overclaim at n=3.

```python
#!/usr/bin/env python3
"""
run_multi_seed_eval.py — PhaseForge multi-seed LIBERO rollout evaluator.

Fixes applied vs. the previous version:
  1. Single top-to-bottom pass — every symbol defined before use.
  2. One aggregation pass. No duplicate/overlapping blocks.
  3. Missing/failed results become None, never a silent 0.0.
  4. A cell below MIN_SEEDS_FOR_SUMMARY is marked incomplete and
     warned about loudly — the run does not fail silently.
  5. Timeout is sized to the measured workload, with real headroom.
  6. Reports a bootstrap CI alongside mean +/- std (ddof=1), per the
     "few-run" guidance in Agarwal et al. 2021 (rliable) — do not
     rely on std alone with only 3 seeds.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("phaseforge.eval.multi_seed")

MIN_SEEDS_FOR_SUMMARY = 3
SECONDS_PER_EPISODE_ESTIMATE = 13.0   # measured on our hardware, 2 workers
WORKERS = 2
TIMEOUT_BUFFER = 1.5                  # 50% headroom over the estimate


@dataclass
class SuiteSpec:
    name: str
    n_tasks: int
    episodes_per_task: int
    max_steps: int
    role: str  # "in_distribution" | "zero_shot"

    @property
    def n_episodes(self) -> int:
        return self.n_tasks * self.episodes_per_task

    def estimated_timeout_s(self) -> int:
        raw = self.n_episodes * SECONDS_PER_EPISODE_ESTIMATE / WORKERS
        return int(raw * TIMEOUT_BUFFER)


# NOTE: your existing config calls this key "episodes_per_suite" but it is
# actually episodes-per-TASK within the suite (50 for libero_90's 90 tasks
# -> 4,500 episodes; 10 for libero_10's 10 tasks -> 100 episodes; 4,600
# total). Worth renaming in the config so the next reader doesn't have to
# reverse-engineer it the way this review did.
SUITES: dict[str, SuiteSpec] = {
    "libero_90": SuiteSpec("libero_90", n_tasks=90, episodes_per_task=50,
                            max_steps=400, role="in_distribution"),
    "libero_10": SuiteSpec("libero_10", n_tasks=10, episodes_per_task=10,
                            max_steps=520, role="zero_shot"),
}


@dataclass
class SeedResult:
    seed: int
    suite: str
    success_rate: float | None   # None = missing/failed, NEVER 0.0
    n_episodes_run: int | None
    raw_path: Path


def run_single_seed(
    cell: str,
    suite: SuiteSpec,
    seed: int,
    checkpoint_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> SeedResult:
    """Runs one (cell, suite, seed) rollout eval as a subprocess.
    Never raises on eval failure — a failed run comes back with
    success_rate=None so the caller decides how to treat it."""
    result_path = output_dir / cell / suite.name / f"seed{seed}" / "eval_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "phaseforge.cli.rollout_eval",
        "--cell", cell,
        "--suite", suite.name,
        "--seed", str(seed),
        "--checkpoint", str(checkpoint_path),
        "--max-steps", str(suite.max_steps),
        "--episodes-per-task", str(suite.episodes_per_task),
        "--output", str(result_path),
    ]
    if dry_run:
        cmd.append("--dry-run")

    timeout_s = suite.estimated_timeout_s()
    logger.info("cell=%s suite=%s seed=%d timeout=%ds", cell, suite.name, seed, timeout_s)

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True, check=False)
    except subprocess.TimeoutExpired:
        logger.error("cell=%s suite=%s seed=%d TIMED OUT after %ds", cell, suite.name, seed, timeout_s)
        return SeedResult(seed, suite.name, None, None, result_path)
    elapsed = time.monotonic() - start

    if proc.returncode != 0:
        logger.error("cell=%s suite=%s seed=%d FAILED (exit %d, %.0fs)\nstderr tail: %s",
                      cell, suite.name, seed, proc.returncode, elapsed, proc.stderr[-2000:])
        return SeedResult(seed, suite.name, None, None, result_path)

    if not result_path.exists():
        logger.error("cell=%s suite=%s seed=%d exited 0 but wrote no result file", cell, suite.name, seed)
        return SeedResult(seed, suite.name, None, None, result_path)

    try:
        payload = json.loads(result_path.read_text())
        success_rate = payload["eval/success_rate"]
        n_episodes_run = payload["eval/n_episodes"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("cell=%s suite=%s seed=%d result file malformed: %s", cell, suite.name, seed, exc)
        return SeedResult(seed, suite.name, None, None, result_path)

    if n_episodes_run != suite.n_episodes:
        logger.warning("cell=%s suite=%s seed=%d ran %d/%d expected episodes — incomplete",
                        cell, suite.name, seed, n_episodes_run, suite.n_episodes)
        return SeedResult(seed, suite.name, None, n_episodes_run, result_path)

    logger.info("cell=%s suite=%s seed=%d OK success_rate=%.3f (%.0fs)",
                cell, suite.name, seed, success_rate, elapsed)
    return SeedResult(seed, suite.name, success_rate, n_episodes_run, result_path)


def bootstrap_ci(values: list[float], n_boot: int = 10_000, ci: float = 0.95,
                  rng_seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over seed-level success rates. This is a
    fast, seed-level approximation for iteration; for the paper's final
    numbers, run the same cells through `rliable`'s stratified bootstrap
    over the full episode-level matrix instead — see Agarwal et al. 2021
    and github.com/google-research/rliable."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return (lo, hi)


def summarize_cell(cell: str, suite: SuiteSpec, results: list[SeedResult]) -> dict:
    valid = [r for r in results if r.success_rate is not None]
    n_valid = len(valid)
    summary = {
        "cell": cell,
        "suite": suite.name,
        "suite_role": suite.role,
        "seeds_requested": len(results),
        "seeds_valid": n_valid,
        "complete": n_valid >= MIN_SEEDS_FOR_SUMMARY,
    }
    if n_valid == 0:
        summary.update(mean=None, std=None, ci95=None)
        logger.error("cell=%s suite=%s has ZERO valid seeds", cell, suite.name)
        return summary

    if n_valid < MIN_SEEDS_FOR_SUMMARY:
        logger.warning("cell=%s suite=%s has only %d/%d valid seeds — flag this "
                        "cell explicitly if it appears in any table",
                        cell, suite.name, n_valid, len(results))

    rates = [r.success_rate for r in valid]
    summary["mean"] = float(np.mean(rates))
    summary["std"] = float(np.std(rates, ddof=1)) if n_valid > 1 else None
    summary["ci95"] = bootstrap_ci(rates) if n_valid > 1 else None
    summary["per_seed"] = {r.seed: r.success_rate for r in valid}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True,
                         help="e.g. bc scratch_moe warmstart_moe phaseforge oracle_moe")
    parser.add_argument("--suites", nargs="+", default=list(SUITES.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                         help="Root containing <cell>/seed<seed>/checkpoint.pt")
    parser.add_argument("--explicit-checkpoint", type=Path, default=None,
                         help="Overrides checkpoint-root resolution for ALL "
                              "cells/seeds. Used for smoke tests. Threaded "
                              "directly into run_single_seed — never dropped.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []

    for cell in args.cells:
        for suite_name in args.suites:
            suite = SUITES[suite_name]
            seed_results: list[SeedResult] = []
            for seed in args.seeds:
                if args.explicit_checkpoint is not None:
                    ckpt = args.explicit_checkpoint
                else:
                    ckpt = args.checkpoint_root / cell / f"seed{seed}" / "checkpoint.pt"
                    if not ckpt.exists():
                        logger.error("cell=%s seed=%d: checkpoint not found at %s", cell, seed, ckpt)
                        seed_results.append(SeedResult(seed, suite.name, None, None, ckpt))
                        continue
                seed_results.append(run_single_seed(cell, suite, seed, ckpt, args.output_dir, args.dry_run))
            all_summaries.append(summarize_cell(cell, suite, seed_results))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2))
    logger.info("wrote summary for %d (cell, suite) pairs to %s", len(all_summaries), summary_path)

    incomplete = [s for s in all_summaries if not s["complete"]]
    if incomplete:
        logger.warning("%d/%d cells INCOMPLETE (<%d valid seeds) — do not report "
                        "without a footnote", len(incomplete), len(all_summaries), MIN_SEEDS_FOR_SUMMARY)

    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Key differences from the previous version, mapped back to §1:

- Bug 1/2/3 are structurally impossible here: there's exactly one loop, accumulators (`all_summaries`, `seed_results`) are declared immediately before the loop that fills them, and per-suite rates live in a `dict` that's built once per `summarize_cell` call, never appended-to-itself.
- Bug 4: `estimated_timeout_s()` is computed from the actual suite size, not a constant.
- Bug 5: `SeedResult.success_rate` is `float | None`; `summarize_cell` counts `seeds_valid` explicitly and refuses to call anything "complete" below `MIN_SEEDS_FOR_SUMMARY`.
- Bug 6: `--explicit-checkpoint` is read once and passed straight into the per-seed loop with no other resolution path — there's nowhere for it to get silently overridden. Confirm this same property holds in whatever your actual fix looks like.

---

## 7. Testing strategy for the new runner

The transcript's own proposed fix ("add a structural test: compile + AST undefined-name scan") is solving a problem `ruff` already solves better — `F821` is exactly an AST-based undefined-name check, already in ruff's default rule set, already run over `scripts/`. Writing a bespoke version duplicates it and won't catch what it doesn't already catch. Two things are actually missing, and they're cheaper than a custom AST checker:

1. **A real, tiny, end-to-end smoke test — not mocked.** The dict-append bug (`suite_rates.append(suite_rates)`) is a runtime `AttributeError`, not a static-analysis finding; ruff's `F821` would not have caught it. The only thing that reliably catches "the aggregation logic is wrong when real subprocess output flows into it" is running the real aggregation logic on real (tiny) subprocess output. Add a smoke config — 1 cell, 1 suite, 1 task, 2 episodes, 1 seed — and run the actual script against it in CI. Minutes, not hours, and it exercises the exact code path that broke.
2. **CI asserts on ruff's exit code**, per §5, so this class of bug can't hide in a truncated terminal tail again.

If you want extra confidence on the type-level bugs specifically (a dict where a list was expected), add type hints where they're missing and run `mypy` or `pyright` in CI — that would have flagged the dict-append directly, and it's a smaller lift than a custom checker.

---

## 8. Open items you still need to verify (not resolvable from this transcript alone)

- Whether `explicit_ckpt` is actually ignored in your current file, and where — the transcript mentions it once without tracing the cause. Grep for every place the checkpoint path gets resolved and confirm there's exactly one.
- Whether `evaluation_plan.md` (and any other doc referencing "i.i.d. sampled initial states") has been corrected to the accurate description in §2.
- Whether `suite_role` actually survives into the final results tables in the paper draft, not just in the raw `eval_results.json`.

---

## 9. Priority checklist

1. Fix bugs 1–3 (the three `NameError`/`AttributeError` crashes) — nothing runs until these are gone.
2. Fix the timeout (bug 4) using the real per-suite episode count, not a guess.
3. Fix the silent-zero and missing-seed-warning behavior (bug 5).
4. Verify bug 6 (`explicit_ckpt`) directly in your code; don't assume the redesign's threading pattern already matches what you have.
5. Correct the `evaluation_plan.md` sentence per §2 — this is a documentation fix, not a code fix, and it's currently the only actually-false claim in the pipeline.
6. Add the bootstrap CI (or adopt `rliable` directly) before the first real table goes into the paper.
7. Wire ruff's exit code into CI as a hard gate (§5), and add the tiny real end-to-end smoke test (§7), before re-running the full sweep.
8. Then, and only then, run the full 5-cell × 2-suite × 3-seed sweep.

---

## 10. Sources

- Agarwal, Schwarzer, Castro, Courville, Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021 (Outstanding Paper Award). https://papers.nips.cc/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf
- `rliable` library. https://github.com/google-research/rliable
- Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* https://arxiv.org/pdf/2306.03310
- PriorVLA (LIBERO eval protocol, init states, π0.5 checkpoint-selection recipe). https://arxiv.org/pdf/2605.10925
- ALAM (LIBERO: 500 episodes from unseen initial states, single checkpoint across suites). https://arxiv.org/pdf/2605.10819
- COAST (LIBERO's built-in per-task initial-state distribution, sequential episode seeds). https://arxiv.org/pdf/2605.17144
- V-VLAPS (libero_10 / libero_90 suite naming and roles). https://arxiv.org/pdf/2601.00969
- Ruff documentation, default rule selection (`E4`, `E7`, `E9`, `F`, which includes `F821`). https://docs.astral.sh/ruff/