# GPU Re-Run Runbook (post-fix, commit-pinned)

**Status:** ready to execute on the GPU machine.

**Why a re-run is required:** the cloud part-3 sweep (git `a07dd2c`) trained
stage-1 with the buggy monitor `val/loss_total`. Because `loss_phase`
explodes (val/loss_phase 2.59 > random ln 6 ≈ 1.79), the buggy monitor
selected epoch 1–2 checkpoints with near-random encoders; stage-2 then
bootstrapped routers from those centroids, and rollouts evaluated the
degenerate policies. Only three methods consume phaseforge's stage-1
(`phaseforge`, `phase_pretrain_random_router`, `teacher_forced`); the rest
(`bc`, `bc_robot_only`, `warmstart_moe`, `plain_encoder_phase_bootstrap`,
`scratch_moe`, `oracle_moe`, `bc_rnn`) were unaffected **training-side** but
still need a full re-run for a consistent artifact set at one git revision
(their part-3 eval rows target old checkpoints).

**Fix commit:** `3cd510f` (stage-1 monitor now `val/loss_action`, matching the
predeclared rule). All code below must run at exactly this revision (or a
descendant with no further training-affecting change).

---

## 0. Preflight checklist (on the GPU machine)

1. `git checkout 3cd510f` (or a descendant); `git status` clean.
2. `uv run python scripts/preflight_configs.py` → must print
   `all 165 train cell(s) and 150 eval cell(s) passed.` (315 cells).
   - This composes every (method, task, stage, seed) + every eval cell via
     Hydra and validates: data task match, `models.name` resolution alias,
     `num_phases` consistency, checkpoint monitor rule, `freeze_encoder`,
     scheduler `T_max`, `eval` group/mode consistency.
3. `uv run pytest -q` → 509 passed (baseline at this revision).
4. Verify the dataset/cache is present:
   - Processed cache under the shared data root (hash `4b06f5c2b28ebc9f` for
     Lift) OR the raw HDF5 files per task. The pipeline auto-builds from raw
     if the cache is absent (correct cache key `b47dd41ff6dc2192` for
     `oracle_moe` differs only because cache provenance keys include file
     mtimes — same content, different machine).
   - `CacheManager.compute_hash` output for every task's data config must be
     printable — the preflight script's eval-cell pass already exercises the
     config side; the data side is validated by one training cell per task.
5. Reset bank: frozen bank SHA-256 already verified in-repo; the rollout eval
   loads it from the cache (or re-generates deterministically from the pinned
   seed). Expected bank ID `a7d3953c0afcf560`.
6. ToolHang: dedicated interpreter with robosuite **1.5.0**
   (`--toolhang-python` gate at runner start); Lift/Can/Square/Transport run
   under robosuite **1.5.1**. Do NOT run ToolHang with 1.5.1 or the preflight
   gate aborts the sweep before any cell runs (by design).

## 1. Command

```bash
# From the repo root, with the venv active:
phaseforge-sweep \
  --manifest experiments/five_task.json \
  --outputs outputs_rerun \
  --with-dependencies \
  --continue-on-error
```

- `--with-dependencies` ensures BC stage-1 providers are trained before the
  stage-2 methods that consume them (all 5 tasks).
- The runner prints `[runner] commit gate: <sha>` on start; verify it equals
  the intended revision. If git is unavailable it prints a warning and
  disables gating — do not trust results from that state.
- `--continue-on-error` is optional; without it the sweep aborts on the first
  failing cell (recommended for a first pass so a config/data issue surfaces
  immediately and is fixable before burning GPU on the rest).
- Fresh `outputs_rerun` base avoids any interaction with stale cloud
  artifacts. The runner's commit gating would refuse to reuse pre-fix
  checkpoints anyway (`state.json` entries and filesystem scans are both
  filtered by `git_commit`), but a clean base is simplest.

## 2. What will run (50 method rows × 3 seeds)

| Rows | Method | Stages | Stage-1 source |
|---|---|---|---|
| 1,10,19,28,37 | phaseforge | 1→2→eval | self |
| 2,11,20,29,38 | bc | 1→eval | — |
| 3,12,21,30,39 | bc_robot_only | 1→eval | — |
| 4,13,22,31,40 | scratch_moe | 2→eval | — (random init) |
| 5,14,23,32,41 | warmstart_moe | 2→eval | bc stage-1 |
| 6,15,24,33,42 | phase_pretrain_random_router | 2→eval | phaseforge stage-1 |
| 7,16,25,34,43 | plain_encoder_phase_bootstrap | 2→eval | bc stage-1 |
| 8,17,26,35,44 | teacher_forced | 2→eval | phaseforge stage-1 |
| 9,18,27,36,45 | oracle_moe | 2→eval | — (offline-only) |
| 46–50 | bc_rnn | 1→eval | — |

Eval mode: rollout for all except `oracle_moe` (offline metrics).

## 3. Monitoring

- Watch `outputs_rerun/_runner/state.json`: every step transitions to
  `completed` (with `git_commit` set) or `failed` (with the error).
- The per-run `metrics/summary.json` records `best_epoch`, `best_val_monitor`
  and `source_stage1` — for the three phaseforge-consuming methods, stage-2
  `source_stage1.git_commit` must equal the fix revision.
- Stage-1 sanity (Lift, matching the local CPU validation): best
  `val/loss_action` ≈ 0.024–0.026 with `best_epoch` ≈ 25–41 (never 1–2).
  Stage-2 NMI final ≈ 0.44–0.46, top-1 collapse 0%.
- If a cell fails: fix the cause, then re-run the same command — the runner
  resumes from `state.json`, skipping completed cells, and the commit gate
  keeps artifacts consistent.

## 4. Post-run verification

1. `outputs_rerun/_runner/state.json` has no `failed` entries.
2. Every eval row in `outputs_rerun/_results/*.jsonl` records a
   `ckpt_path` under `outputs_rerun/...`.
3. Spot-check `git_commit` in a sample of `run_meta.json` files = fix revision.
4. `phaseforge-sweep --outputs outputs_rerun --dry-run` prints all steps as
   `skip (already completed)` — nothing to re-run.
5. Re-run `uv run python scripts/preflight_configs.py` (still 315 passed).

## 5. Local validation reference (CPU, for comparison)

Stage-1 (fixed monitor, 3 seeds): best `val/loss_action` 0.0264/0.0240/0.0261
@ epochs 41/36/25 (buggy: 0.0451/0.0404/0.0659 @ epochs 2/2/1).
Stage-2 (warm start from those): action 0.0301/0.0279/0.0308; NMI final
0.449/0.457/0.436; collapse 0%. Runs live under `outputs_local_train/`
(tag `Lift_fixed_v3_stage1`/`Lift_fixed_v3_stage2`).

## 6. Notes / cautions

- Do NOT mix `outputs/` cloud artifacts into this run; the resolver scans the
  configured `--outputs` base only, so keeping the rerun separate is enough.
- `train.phase_class_weight` defaults to `"none"` (plain CE, the frozen
  protocol). The `"balanced"` option is implemented and probed (fixes the
  phase head: val/loss_phase 2.54→1.38, balanced acc 0.568→0.643) but drops
  stage-2 NMI 0.449→0.415 on seed 42 — decision: keep protocol default
  `"none"`. Do not enable it for the rerun.
- Resume after a kill is supported but the per-epoch DataLoader shuffle order
  is NOT bit-exact after resume (generator seeded from `project.seed`, not
  checkpointed) — resume only to salvage an interrupted cell, then re-verify
  its summary.