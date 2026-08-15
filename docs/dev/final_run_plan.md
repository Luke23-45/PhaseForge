uv sync --extra dev                                     # 1. install
# 2. provision dataset at $PHASEFORGE_DATA_DIR/raw/robomimic/lift/  (fail-closed, no auto-download)
uv run python -m phaseforge.runner --list               # 3. sanity: 9 methods + seeds [42,43,44]
uv run python -m phaseforge.runner --dry-run            # 4. pre-flight: 57-step plan, read-only
uv run python -m phaseforge.runner --continue-on-error  # 5. THE full sweep (all methods x seeds 42,43,44)
uv run python -m phaseforge.runner --dry-run            # 6. verify: every step "done"
uv run python scripts/summarize_train.py --outputs outputs --baseline phaseforge   # 7. train tables
uv run python scripts/summarize_eval.py  --outputs outputs --baseline phaseforge   # 8. eval tables


# Final Run Plan — Lift pilot sweep → paper tables

Run this once, top to bottom, on the machine that will produce the paper
results. Every command was verified against the current code (runner,
manifest `experiments/lift_pilot.json`, `scripts/summarize_*.py`).

> Current workspace state (2026-08-14): `outputs/` contains only stale
> legacy-format run dirs — no `checkpoints/checkpoint_best.pt`, no
> `<run_dir>.completed` markers, no `outputs/_results/`, no
> `outputs/_runner/state.json`. The runner ignores those legacy dirs, so
> the sweep below starts clean. A completed sweep is what makes the paper
> tables possible: `_results/results.jsonl` (eval rows) +
> `_results/training_summary.jsonl` (train rows) → `_summaries/*`.

---

## 1. One-time setup

### 1.1 Install the package

```bash
uv sync --extra dev
```

(If `uv` is unavailable: `pip install -e ".[dev]"`.)

### 1.2 Provision the dataset (fail-closed — the pipeline will NOT download it)

Place the official robomimic **Lift Proficient-Human low-dim** HDF5
file(s) under:

```
{PHASEFORGE_DATA_DIR}/raw/robomimic/lift/
```

Default root if the variable is unset: `./data/raw/robomimic/lift`.
Set it for the whole session (PowerShell):

```powershell
$env:PHASEFORGE_DATA_DIR = "D:\phaseforge-data"
```

Record the dataset revision and checksum in the external data manifest
before training. The cache (`{root}/processed/cache`) builds automatically
on first use; a later cache hit only re-verifies.

### 1.3 Sanity: the manifest loads

```bash
uv run python -m phaseforge.runner --list
```

Expect 9 methods and `seeds: [42, 43, 44]`.

---

## 2. Pre-flight (read-only — executes nothing)

```bash
uv run python -m phaseforge.runner --dry-run
```

Expect `plan (57 steps, ...)`: all 9 methods x 3 seeds, training stages in
dependency order, then each method's offline evaluation. Per-step lines:
stage-1 and eval of a finished method print `WOULD RUN`; a stage-2 step whose
provider Stage 1 does not exist yet prints `BLOCKED ... prerequisite missing`
— expected on a fresh tree, and exactly what the sweep needs to build first.

---

## 3. Run the full final sweep

```bash
uv run python -m phaseforge.runner --continue-on-error
```

Notes:
- **No `--seeds`/`--methods` flags.** The manifest drives seeds `[42, 43, 44]`
  and all 9 methods — seeds are config-driven, not typed by hand.
- `--continue-on-error` records a failed step in the registry and keeps going;
  re-run the failed cell afterwards (section 5).
- **Resumable**: a completed step is skipped on re-run (state registry +
  `<run_dir>.completed` marker). Safe to interrupt and restart.
- Output layout: `outputs/{model}/stage{N}/seed{S}/{ts}[_{tag}]_{runid}/`
  (eval under `outputs/eval/{model}/seed{S}/`).

---

## 4. Verify completion

```bash
uv run python -m phaseforge.runner --dry-run
```

Expect every step line `done` (no `pending`, no `failed`). Confirm the
ledgers exist:

```powershell
Get-Item outputs\_results\results.jsonl, outputs\_results\training_summary.jsonl
```

---

## 5. Re-run a failed cell (only if needed)

Fix the cause first, then re-run the specific cell with `--force`:

```bash
# e.g. PhaseForge (method 1), seed 42, stage 2 only, ignoring recorded status:
uv run python -m phaseforge.runner --methods 1 --seeds 42 --stage 2 --force
```

`--force` re-runs the selected steps regardless of registry status; omit it
to respect the registry. For a partial selection that needs a provider's
Stage 1, add `--with-dependencies`.

> **Known fixed bug (2026-08-14):** stage-2 steps for `warmstart_moe` (5) and
> `plain_encoder_phase_bootstrap` (7) crashed with an encoder input-dimension
> mismatch because the subprocess auto-detected the newer `bc_robot_only`
> checkpoint (state dim 23) instead of the untagged BC checkpoint (state dim
> 19). The runner now passes the exact untagged provider checkpoint as
> `train.stage1_ckpt_path`. To pick up the fix, `git pull` and re-run the
> failed cells for all seeds:
>
> ```bash
> uv run python -m phaseforge.runner --methods 5 --seeds 42,43,44 --stage 2 --force
> uv run python -m phaseforge.runner --methods 7 --seeds 42,43,44 --stage 2 --force
> # then their evals (or let the full sweep re-run them):
> uv run python -m phaseforge.runner --continue-on-error
> ```

---

## 6. Generate paper tables (idempotent — safe to re-run after any sweep)

Training side:

```bash
uv run python scripts/summarize_train.py --outputs outputs --baseline phaseforge
```

Writes under `outputs/_summaries/`:
- `training_aggregates.csv` — per (model, stage) mean/std/n over seeds
- `training_cost.csv` — wall time, epochs, optimizer steps, params
  (wall time = `summary.json` `wall_seconds`, training-loop only, **not**
  `timings.json`'s full-lifecycle value — see `training_summaries.py`)
- `training_curves.csv` — per (model, stage, epoch) curve means (plot source)
- `rollout_success.csv` / `rollout_comparisons.csv` — written ONLY when
  `episodes.jsonl` exists; rollout evaluation is intentionally blocked, so
  these will not appear for the current pilot.

Evaluation side:

```bash
uv run python scripts/summarize_eval.py --outputs outputs --baseline phaseforge
```

Writes under `outputs/_summaries/`:
- `aggregates.csv` — per (model, stage) mean/std/n per metric
- `bootstrap_ci.csv` — percentile bootstrap 95% CIs
- `paired_wilcoxon.csv` — two-sided Wilcoxon vs `phaseforge`, paired on
  (stage, seed)
- `metrics.json` — per (model, stage) per-metric means for the paper text

---

## 7. Reminder of what is NOT part of this plan

- No manual per-method `phaseforge-train`/`phaseforge-eval` commands — the
  runner emits those internally with exact seed/ckpt plumbing.
- No `scripts/run_multi_seed_train.py` — removed (superseded by the runner).
- Rollout evaluation is blocked by design (`eval.mode=rollout` rejected) until
  the simulator adapter and paired test runner exist; offline metrics are the
  diagnostic sanity check only.
