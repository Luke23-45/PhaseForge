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

**Fix commits:** `3cd510f` (stage-1 monitor now `val/loss_action`, matching the
predeclared rule). All code below must run at exactly the final revision (or a
descendant with no further training-affecting change).

**Adopted Phase-1 fix (2026-08-18, CPU-validated on 3 Lift seeds):**
the **monitor restoration** — stage-1 best-checkpoint selection by
`val/loss_action` exactly as the protocol predeclared. Local validation
(tags `fixed`/`tiebreak_v1` reference): best epochs 41/36/25, action loss
0.0264/0.0240/0.0261 (spread 0.0024 vs 0.0255 buggy); stage-2 NMI
0.449/0.457/0.436 (spread 0.021 vs 0.069 buggy) at equal level (0.447).

**Deliberately NOT enabled — the λ-decay refinement:** linear λ decay
(1.0 → 0.0) was locally validated (stage-2 NMI spread 0.021 → 0.010, means
identical; tags `lambdav1`/`lambdav1_stage2`) but is a **protocol deviation
affecting only the phase-supervised arms**. Fairness decision 2026-08-18:
the official comparison runs the predeclared λ = 1.0 for every method;
baselines are untouched and unchanged. Diagnosis that motivated both changes:
grad-cosine measurement cos(∇L_action, ∇L_phase) ≈ 0 (no gradient conflict →
no PCGrad/CAGrad/Du-adaptive weighting); val/loss_phase explosion is
late-training degradation on the flat action plateau (shared-encoder drift).
The tie-break re-selection was also tested and rejected (spread 0.044
standalone, 0.039 on top of λ-decay): its criterion selects early epochs
whose router-bootstrap centroids are consistently worse.

The λ schedule machinery remains in the code (inert by default: `constant`
type = bit-identical to no schedule) and is documented as the fallback if
the Gate 2 spread criterion fails.

---

## 0. Preflight checklist (on the GPU machine)

1. `git checkout 3cd510f` (or a descendant); `git status` clean.
2. `uv run python scripts/preflight_configs.py` → must print
   `all 165 train cell(s) and 150 eval cell(s) passed.` (315 cells).
   - This composes every (method, task, stage, seed) + every eval cell via
     Hydra and validates: data task match, `models.name` resolution alias,
     `num_phases` consistency, checkpoint monitor rule, `freeze_encoder`,
     scheduler `T_max`, `eval` group/mode consistency.
3. `uv run pytest -q` → 547 passed (baseline at this revision).
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

### 1a. No manifest edit — the fix is in the committed config

The adopted fix is the **monitor restoration** (`val/loss_total` → `val/loss_action`,
commit `3cd510f`), already in the codebase; `experiments/five_task.json` needs **no change**
to its `defaults` (only the predeclared `train.early_stopping.enabled=false`). Stage-1 runs
with λ = 1.0 constant, exactly the predeclared protocol.

**Deliberately NOT enabled — the λ-decay refinement:** linear λ decay (1.0 → 0.0) was locally
validated (stage-2 NMI spread 0.021 → 0.010, means identical; tags `lambdav1`/`lambdav1_stage2`
under `outputs_local_train/`) but is a **protocol deviation affecting only the phase-supervised
arms** (phaseforge, phase_pretrain_random_router, teacher_forced via phaseforge's stage-1).
Fairness decision 2026-08-18: the official comparison runs the predeclared λ = 1.0 for every
method; baselines are untouched and unchanged. The refinement remains available: re-adding

```json
"train.lambda_schedule.type=linear",
"train.lambda_schedule.start=1.0",
"train.lambda_schedule.end=0.0"
```

to `defaults` (plus the inert `lambda_schedule` block already present in `stage2.yaml`)
is the full activation, and is the documented fallback if the Gate 2 spread criterion
fails. Protocol-compliance check on final runs: `train/lambda_phase` must be pinned at 1.0
on every stage-1 curve.

### 1b. Phase-2 subset first (professor §7): Lift, the two stage-1-affecting methods

```bash
# From the repo root, with the venv active:
phaseforge-sweep \
  --manifest experiments/five_task.json \
  --outputs outputs_rerun \
  --methods phaseforge,phase_pretrain_random_router \
  --seeds 42,43,44 \
  --with-dependencies \
  --continue-on-error
```

Success criteria (professor §7 Phase 2): per-seed rollout spread drops while
`val/loss_action` stays on its plateau; target PhaseForge spread 0.24 →
≈0.04–0.10 (plain-encoder control ≈0.04). Local CPU proxy already achieved
with the official pipeline: stage-2 NMI spread 0.021 (λ-decay refinement:
0.010 — documented fallback if this criterion fails). If achieved, the
result upgrades from "highest mean, highest variance" to
**"highest mean, low variance"**.

### 1c. Full sweep only after Gate 2 passes

```bash
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
- Stage-1 snapshots: keep `train.checkpoint.every_n_epochs` at the stage-1
  default (10) for the re-run — the tie-break is NOT part of the adopted
  fix, so per-epoch snapshots are unnecessary disk. Re-selection scripts
  (`scripts/tie_break_selector.py`) remain available for post-hoc audits.

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

**Official pipeline — fixed monitor, λ = 1.0 constant (3 seeds):**
stage-1 best `val/loss_action` 0.0264/0.0240/0.0261 @ epochs 41/36/25
(buggy: 0.0451/0.0404/0.0659 @ epochs 2/2/1); stage-2 final action
0.0301/0.0279/0.0308, NMI **0.449/0.457/0.436 (spread 0.021)**, collapse 0%.

**Refinement only — fixed monitor + λ-decay (tag `lambdav1`, not adopted):**
stage-1 best `val/loss_action` 0.0266/0.0241/0.0260 @ epochs 41/36/25;
stage-2 (tag `lambdav1_stage2`) final action 0.0337/0.0275/0.0312, NMI
**0.450/0.450/0.440 (spread 0.010)**; final `val/loss_phase` 2.28–2.47
(shared-encoder drift, no longer phase-head overfitting).

For reference, stage-2 from tie-break checkpoints NMI 0.440/0.411/0.395
(spread 0.044); from tie-break-on-λ checkpoints NMI 0.434/0.414/0.394
(spread 0.039). All runs live under `outputs_local_train/`.

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