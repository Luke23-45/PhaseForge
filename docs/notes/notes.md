# PhaseForge — Final Experiment Command Reference

Authoritative source: `docs/dev/final_run_plan.md`. This document restates its
commands in execution order. Do not deviate from it without a ledger entry.

**Invocation contract**

- Always launch the runner from the rollout venv interpreter:
  `.venv-rollout/Scripts/python -m phaseforge.runner`
- NEVER substitute `uv run` — it resolves the mujoco-less dev environment
  (`final_run_plan.md` §3). Tool Hang steps route automatically to
  `.venv-toolhang` with a version-pin preflight; no special handling needed.
- Main matrix namespace: `outputs_final` (mandatory). Ablation namespace:
  `outputs_ablation`. Never widen or merge namespaces.
- Every scoped run carries `--expect-steps N` — the runner refuses to start
  unless the built plan has exactly N steps (guard against selection drift).
- Scoped (`--methods ...`) selections that consume a Stage-1 provider must
  pass `--with-dependencies`; the runtime only auto-injects dependencies for
  unscoped sweeps.
- Resumable by design: completed steps are skipped (state registry +
  git-commit gating). Safe to interrupt and restart. Use `--force` only to
  deliberately re-run a cell.
- Exit codes: 0 success · 1 failed steps (with `--continue-on-error`) ·
  2 pre-flight/config error · 130 interrupted.

---

## 0. Pre-flight validation (read-only, run before anything)

```powershell
# 0.1 Confirm the method matrix loads: expect 50 rows (10 per task),
#     seeds [42, 43, 44], phaseforge rows marked stage1,stage2.
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json `
  --list

# 0.2 Dry-run the full sweep on a fresh namespace: expect exactly
#     315 steps, every one `pending`.
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json `
  --outputs outputs_final `
  --expect-steps 315 `
  --dry-run

# 0.3 Environment gates — all five must pass before the sweep.
.venv-rollout/Scripts/phaseforge-gates data=lift eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=can eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=square eval=rollout
.venv-toolhang/Scripts/phaseforge-gates data=tool_hang eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=transport eval=rollout
```

---

## 1. Baseline Runs — Five-Task Matrix (`experiments/five_task.json`)

Frozen matrix: 10 methods x 5 tasks (Lift, Can, Square, ToolHang, Transport)
x 3 seeds (42, 43, 44) = 50 cells = **315 runner steps**
(21 per task-seed: 11 training + 10 evaluations).

### 1.1 Primary protocol — single unscoped sweep (recommended)

```powershell
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json `
  --outputs outputs_final `
  --expect-steps 315 `
  --continue-on-error
```

This is the canonical invocation: dependencies are auto-injected and each
provider trains exactly once. Prefer it over any per-method decomposition.

### 1.2 Per-method decomposition (equivalent, guarded)

Use only when scheduling constraints require splitting the sweep. Step
counts below are enforced by `--expect-steps` and hold regardless of prior
completion (completed steps remain in the plan and are skipped at runtime).
Consumers of a shared Stage-1 provider include `--with-dependencies`
(+15 provider steps = 5 tasks x 3 seeds, trained once then reused).

| # | Method | Role | Stages | Steps |
|---|--------|------|--------|-------|
| 1 | `phaseforge` | proposed (6 experts, centroid router, 50% partial warm-start) | 1+2 | 45 |
| 2 | `bc` | structured-state BC floor | 1 | 30 |
| 3 | `bc_large` | parameter-matched dense capacity control | 1 | 30 |
| 4 | `bc_rnn` | temporal comparator (10-step history) | 1 | 30 |
| 5 | `bc_robot_only` | robot-only negative control (descriptive) | 1 | 30 |
| 6 | `scratch_moe` | MoE architecture control (random init, no Stage 1) | 2 | 30 |
| 7 | `warmstart_moe` | warm-start control (plain encoder x random router) | 2 (+bc S1) | 45 |
| 8 | `phase_pretrain_random_router` | phase-representation control (H1) | 2 (+pf S1) | 45 |
| 9 | `plain_encoder_phase_bootstrap` | centroid-init control (H2) | 2 (+bc S1) | 45 |
| 10 | `teacher_forced` | privileged-training diagnostic (E8, descriptive) | 2 (+pf S1) | 45 |

```powershell
# 1 — Proposed method (Stages 1 & 2, self-provided)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods phaseforge --expect-steps 45 --continue-on-error

# 2 — Standard behavior cloning floor
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods bc --expect-steps 30 --continue-on-error

# 3 — Parameter-matched dense baseline
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods bc_large --expect-steps 30 --continue-on-error

# 4 — Temporal history comparator
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods bc_rnn --expect-steps 30 --continue-on-error

# 5 — Robot-only negative control
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods bc_robot_only --expect-steps 30 --continue-on-error

# 6 — Scratch MoE (no provider dependency)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods scratch_moe --expect-steps 30 --continue-on-error

# 7 — Warm-start MoE (requires bc Stage-1 provider: 30 + 15 = 45)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods warmstart_moe --with-dependencies --expect-steps 45 --continue-on-error

# 8 — Phase pretraining / random router (requires phaseforge Stage-1: 45)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods phase_pretrain_random_router --with-dependencies --expect-steps 45 --continue-on-error

# 9 — Plain encoder / centroid router (requires bc Stage-1: 45)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods plain_encoder_phase_bootstrap --with-dependencies --expect-steps 45 --continue-on-error

# 10 — Privileged training diagnostic (requires phaseforge Stage-1: 45)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json --outputs outputs_final `
  --methods teacher_forced --with-dependencies --expect-steps 45 --continue-on-error
```

### 1.3 Verify completion (main matrix)

```powershell
# Re-run the dry-run: every step must read `done` (no pending, no failed).
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json `
  --outputs outputs_final `
  --expect-steps 315 `
  --dry-run

# Ledger checks: results.jsonl must hold 150 rows (50 cells x 3 seeds);
# training_summary.jsonl must hold 165 rows (55 train cells x 3).
```

---

## 2. Ablation Runs — Lift Suite (`experiments/lift_ablation.json`)

27 cells (9 Lift replicas of the main matrix, `bc_rnn` excluded, plus 18
ablation-only cells) x 3 seeds = **165 runner steps**, under the isolated
`outputs_ablation` namespace. Start **only after** the main matrix is
complete and verified (§1.3).

### 2.1 Primary protocol — single unscoped suite (recommended)

```powershell
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json `
  --outputs outputs_ablation `
  --expect-steps 165 `
  --continue-on-error
```

The canonical provider (EXP-101 `phaseforge` Stage 1) is auto-injected once
per seed and shared by all ablation cells.

### 2.2 Per-group decomposition (guarded)

Each `pf_*` cell costs 2 steps/seed (Stage-2 training + rollout evaluation)
= 6 steps per method; scoped selections add the canonical-provider Stage-1
via `--with-dependencies` (+3 steps, trained once then reused).

```powershell
# ---- Group A: Router initialization controls (5 methods: 30 + 3 = 33 steps)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json --outputs outputs_ablation `
  --methods pf_spherical_kmeans pf_kmeans pf_phase_head pf_random_random pf_centroid_random `
  --with-dependencies --expect-steps 33 --continue-on-error

# ---- Group B: Expert initialization & warm-start drop sweep (6 methods: 36 + 3 = 39 steps)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json --outputs outputs_ablation `
  --methods pf_full_warm pf_drop00 pf_drop25 pf_drop75 pf_drop100 pf_one_warm_plus_random `
  --with-dependencies --expect-steps 39 --continue-on-error

# ---- Group C: Representation, fine-tuning & phase noise (5 methods: 30 + 3 = 33 steps)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json --outputs outputs_ablation `
  --methods pf_spherical pf_ft pf_corrupt_25 pf_corrupt_50 pf_shuffle_control `
  --with-dependencies --expect-steps 33 --continue-on-error

# ---- Group D: Capacity & expert scaling, K sweep (2 methods: 12 + 3 = 15 steps)
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json --outputs outputs_ablation `
  --methods pf_k3 pf_k12 `
  --with-dependencies --expect-steps 15 --continue-on-error
```

Single-cell recovery example (any `pf_*` cell alone: 6 + 3 = 9 steps):

```powershell
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json --outputs outputs_ablation `
  --methods pf_drop25 --with-dependencies --expect-steps 9 --continue-on-error
```

### 2.3 Verify completion (ablation)

```powershell
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json `
  --outputs outputs_ablation `
  --expect-steps 165 `
  --dry-run
```

Every step `done`, no `pending`, no `failed`.

---

## 3. Failed-cell re-run recipe (only if needed)

Fix the root cause first, then force the exact cell — never widen the
namespace:

```powershell
# Example: re-run the proposed method on Lift, seed 42, Stage 2 only.
.venv-rollout/Scripts/python -m phaseforge.runner `
  --manifest experiments/five_task.json `
  --outputs outputs_final `
  --methods phaseforge@Lift --seeds 42 --stage 2 --force
```

Add `--with-dependencies` when a partial selection needs its provider's
Stage 1. Useful filters: `--tasks Lift Can`, `--seeds 42,43`,
`--stage {1,2}`, `--eval-only`, `--skip-eval`.

---

## 4. Paper tables (final namespaces ONLY — after both suites verify clean)

```powershell
uv run python scripts/analysis/summarize_train.py --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/summarize_eval.py  --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/stratified_stats.py --root outputs_final --json outputs_final/_summaries/stratified_stats.json
.venv-rollout/Scripts/phaseforge-rollout-report outputs_final
```

(`stratified_stats.py` reports per task, never pooled; `--root` fully
replaces the default so historical `outputs/` rows cannot leak in.)

Fairness/protocol sanity:

```powershell
uv run python scripts/analysis/fairness_accounting.py
uv run python scripts/protocol/preflight_configs.py
```

---

## 5. Prohibited in the final protocol (`final_run_plan.md` §10)

- No manual `phaseforge-train` / `phaseforge-eval` invocations — the runner
  emits them with exact seed/checkpoint plumbing and fail-closed contract
  checks (model identity, stage, 6-expert contract).
- No merging pre-final `outputs/` results into final tables.
- No oracle/teacher-forced *evaluation* until D10 is resolved
  (`teacher_forced` training diagnostic itself is unaffected).
- Retired identities stay gone: 8-expert `phaseforge`, `phaseforge_r50`,
  `lar_moe_state_only`, `pf_random_warm`, `phaseforge_e6`, `pf_jitter_*`,
  `warmstart_r50`.
