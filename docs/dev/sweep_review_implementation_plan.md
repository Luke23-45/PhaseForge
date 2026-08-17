# Sweep Output Review — Findings & Implementation Plan

**Date:** 2026-08-14 — after the full Lift pilot sweep (57/57 cells, 9 methods × seeds 42/43/44).
**Source of truth reviewed:** `outputs/_results/results.jsonl`, `outputs/_results/training_summary.jsonl`, `outputs/_summaries/*`, `outputs/_runner/state.json`, per-run `metrics/summary.json` + `timings.json`.

## Verdict

The results themselves are **valid, complete, and reproducible** (no failed cells in the registry; every row schema-validated; checkpoint paths resolve to `.completed` runs). No number in the report is wrong. There is **one real tooling bug** (P1) that produced *wrong aggregate tables*, plus a few medium/low quality items. Each is fixable without re-running any training or evaluation.

---

## P1 — [BUG, HIGH] `bc` and `bc_robot_only` are merged in every summary table

> **STATUS: IMPLEMENTED (2026-08-14).** `tag`/`method` added to the result + training schemas and write paths; summarizers group by `(model, tag, stage)`; legacy ledgers migrated with `scripts/backfill_tags.py` (reads `tag` from each run's `run_meta.json`); verified — `bc` and `bc`/`robot_only` now separate rows in `aggregates.csv`/`training_aggregates.csv`/`training_cost.csv`/`metrics.json`. Gates: 331 tests, ruff clean, mypy clean.

### Symptom
`bc` (default 19-dim) and `bc_robot_only` (23-dim robot-only negative control) share `model_name == "bc"`. `results.jsonl` and `training_summary.jsonl` rows record only `model`, `stage`, `seed` — never the data-variant `tag`. Every summarizer groups by `(model, stage)`, so all six BC rows (3 default + 3 robot-only) collapse into one `bc` group:

- `aggregates.csv`: `bc,1,n_seeds=3,n_rows=6`, action_mse_mean **0.02386** — a wrong BC floor. The true numbers are default BC **0.02796 ± 0.00052** and robot-only **0.01975 ± 0.00059**.
- `metrics.json`, `bootstrap_ci.csv`, `paired_wilcoxon.csv`, `training_aggregates.csv`, `training_cost.csv`: same merge; the robot-only negative control is silently hidden.

The report (`docs/dev/lift_pilot_offline_report.md`) already separates them manually; the *tooling* must be fixed so the paper tables are correct automatically.

### Fix
1. **Schema** (`phaseforge/outputs_writer/schema.py`):
   - Add optional `tag: str | None` (and `method: str | None`) to the results schema and the training-summary schema. Missing → `None` (untagged), so old rows stay readable.
2. **Write side**:
   - Eval row write (`phaseforge/cli.py` eval path): include `tag = cfg.project.get("tag")` and `method = cfg.project.get("method")`.
   - Training summary write (`phaseforge/trains/callbacks/persistence.py`): same two fields from the resolved config.
   - `project.method` is a new lightweight identity override the runner can pass (the runner knows the protocol method name); fall back to model name when absent.
3. **Summarizers** (`phaseforge/outputs_writer/tables.py`, `summarize.py`, `training_summaries.py`):
   - Group key becomes `(model, tag, stage)`; emit a `tag` column; keep the untagged rows grouped under `tag=""`/`None`.
4. **Migration for existing rows** — new `scripts/backfill_tags.py`:
   - For each row missing `tag`: read `run_meta.json` from the run dir (available via `run_dir` on training rows; derivable from `ckpt_path`'s parent chain on eval rows) and set `tag` from `run_meta.tag`; rewrite the jsonl in place.
5. **Tests** (`tests/test_outputs_writer.py`, `tests/test_runner.py`):
   - Schema accepts/missing `tag`; grouping with tagged+untagged rows stays separate; migration backfills correctly; `paired_wilcoxon` pairs only same-`(model, tag, stage)`.
6. **Verify:** re-run both summarizers on the existing outputs; expect `bc` default = 0.02796 and a separate `bc`/`robot_only` = 0.01975; `n_rows` back to 3 per group.

---

## P2 — [MEDIUM] Stage-1 checkpoint selection & phase-label imbalance

> **STATUS: MONITOR FIX IMPLEMENTED + VALIDATED (2026-08-18).** The stage-1 checkpoint monitor was changed from `val/loss_total` to `val/loss_action` in `phaseforge/config/train/stage1.yaml`, matching the predeclared plan rule (`best val/loss_action`, state_only_rollout_implementation_plan.md §4.6 / §6) and `stage2.yaml`. **Supervisor decision was made (2026-08-18, "do not let it lose to others … try every possible thing") to apply fix #4**, reversing the earlier deferral. Local CPU validation (Lift, seed 42, `a07dd2c`): stage-1 best epoch moved from **2** (val/loss_total 0.836) to **41** (val/loss_action **0.0264**, better than BC's 0.0277); stage-2 warmed from the corrected checkpoint reached best loss_action 0.0286 with NMI **0.463** (vs 0.393 buggy) and 0% collapse. BC is unaffected (no phase head ⇒ `loss_total == loss_action`). A `lambda_phase=0.1` probe was slightly *worse* (0.0279), so the protocol value 1.0 is retained. The earlier best-epoch phase metrics analysis (§5 of `lift_pilot_offline_report.md`) still stands as the diagnostic; #1/#2/#3 (best-epoch logging, phase-label stats) remain follow-ups.

### Symptom
- `phaseforge` stage-1 selects `best_epoch = 4` via `val/loss_total` (monitor in `config/train/stage1.yaml`): best val 0.84 at epoch 4 vs **final** val loss 2.63. The phase CE grows from ≈0.8 to **2.60**, i.e. *worse than a uniform predictor* (ln 6 ≈ 1.79) — a classic signature of noisy/imbalanced phase labels and late-stage overfitting. BC's stage-1 loss stays flat (0.0275 → 0.0334), so this is specific to the phase head.
- The stage-2 bootstrap consumes the epoch-4 phase head (phase CE ≈ 0.8 there — reasonable), but **the phase metrics at the selected epoch are never logged**, so head quality at the checkpoint actually used is unverifiable.

### Fix (no retraining needed)
1. `phaseforge/trains/callbacks/persistence.py`: record a `best_val` block in `summary.json` mirroring `final_val` (phase acc, balanced acc, NMI at the *selected* epoch).
2. `phaseforge/outputs_writer/training_summaries.py`: surface best-epoch phase metrics in `training_aggregates.csv`.
3. Add a phase-label sanity report (Gate-3-style) to `scripts/summarize_train.py` (or a small `scripts/phase_label_stats.py`): per-phase counts, durations, balance — feeds the paper appendix and explains the NMI ceiling.
4. **Do not** change the stage-1 monitor without a supervisor decision: `val/loss_total` is a stable validation rule and the bootstrap head is early (fine); `val/phase_balanced_acc` is the alternative but noisy at n=3. Record the trade-off, don't silently switch.

---

## P3 — [LOW] Wall-time definitions disagree

> **STATUS: IMPLEMENTED (2026-08-14).** Definition documented in `training_summaries.py` (`training_cost.csv` wall time = `summary.json` `wall_seconds`, training-loop only, ≠ `timings.json`) and in `docs/dev/final_run_plan.md`; chose the documentation option over a new `total_wall_seconds` column.

`timings.json` (full run lifecycle, e.g. 69.7 s) ≠ `metrics/summary.json` `wall_seconds` (training loop only, 56.6 s) for the same run; `training_cost.csv` uses `summary.json`. Not a bug, but confusing.
- Document the definition in `training_summaries.py` + `docs/dev/final_run_plan.md`, or add a `total_wall_seconds` column sourced from `timings.json`.

---

## P4 — [LOW] Leftover failed-run artifacts

> **STATUS: IMPLEMENTED (2026-08-14).** The `.failed` markers were already gone; `scripts/cleanup_failed_runs.py` (dry-run default, `--apply` to delete) removed the **10** leftover run dirs (`warmstart_moe` 5 + `plain_encoder_phase_bootstrap` 5, stage2) that carried `logs/exception.txt` without a `.completed` marker. Post-cleanup: 57 run dirs = 57 `.completed` markers = 57 registry cells, 0 `exception.txt`, 0 failed. Registry had no failed entries to prune.

Pre-fix attempts left `.failed` markers + run dirs (`warmstart_moe`/`plain_encoder_phase_bootstrap` stage2, 1–3 per seed) with `logs/exception.txt` recording the old dimension-mismatch. Harmless (resolver requires `.completed`) but clutter.
- Add `scripts/cleanup_failed_runs.py` (delete `*.failed` siblings + their run dirs; optionally prune failed entries in `outputs/_runner/state.json`). Run before finalizing paper artifacts.

---

## P5 — [INFO, optional] Provenance hygiene

`results.jsonl` mixes two commits: 45 rows `f2fd35b`, 6 re-run rows `2abafd5`. Verified `git diff f2fd35b..2abafd5` touches **only** runner plumbing, docs, and tests — no training/eval/model code — so the 6 rows are comparable. No action required; if uniform provenance is desired, re-run the whole sweep under one commit (~40 min of GPU).

---

## Explicitly ruled out (not bugs)
- `action_l2_threshold_rate` = NaN everywhere — **by design**, `enabled: false` in `phaseforge/config/eval/metrics.yaml`.
- `config_hash` differs per seed — by design (full-config identity incl. seed); the seed-invariant `data_config_hash` is consistent across all rows (`a5034f0298c2cb25`).
- Wilcoxon p-values in `{0.25, 0.5, 0.75, 1.0}` — inherent to n=3 seeds; report states no significance.
- `git_sha` mix — provenance, see P5.
- Empty `rollout_*.csv` — rollout evaluation is intentionally blocked.

---

## Order of work
1. **P1** (schema + write + summarizers + migration + tests) — **DONE**: unblocked correct paper tables from the existing outputs; the migrated + regenerated tables are in `outputs/_summaries/`.
2. **P2** (best-epoch logging + phase-label stats) — finding quantified in report §5; `best_val` logging + stats script deferred (future runs only; monitor change waits on supervisor).
3. **P3, P4** (doc + cleanup) — **DONE** (2026-08-14).
4. **P5** — only if the supervisor asks for single-commit provenance.
