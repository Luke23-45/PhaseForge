# PhaseForge — Final Experiment Command Reference (Cloud)

All commands use `uv run python -m phaseforge.runner` — the cloud image's
default `uv` environment already has the rollout stack (`torch`, `robosuite`
1.5.1, `mujoco` 3.2.7) installed. ToolHang steps are routed automatically to
the ToolHang interpreter (`PHASEFORGE_TOOLHANG_PYTHON` / `.venv-toolhang`);
no manual env switch is needed.

Two experiment types only: **Baseline Matrix** (`outputs_final`, 315 steps)
and **Ablation Suite** (`outputs_ablation`, 165 steps). Both are resumable
(completed steps are skipped via `outputs/_runner/state.json` + commit gating).

---

## 0. Pre-flight (read-only)

```bash
# List the frozen matrix: 50 rows (10/method × 5 tasks), seeds [42,43,44]
uv run python -m phaseforge.runner --manifest experiments/five_task.json --list

# Dry-run the full baseline sweep: must be exactly 315 steps, all `pending` on a fresh namespace
uv run python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final \
  --dry-run

# Dry-run the full ablation suite: must be exactly 165 steps
uv run python -m phaseforge.runner \
  --manifest experiments/lift_ablation.json \
  --outputs outputs_ablation \
  --dry-run
```

---

## 1. Baseline Runs — Five-Task Matrix (`experiments/five_task.json`)

Frozen: 10 methods × 5 tasks (Lift, Can, Square, ToolHang, Transport) ×
3 seeds (42, 43, 44) = 50 cells = **315 steps** (21 per task-seed).

### 1.1 Full sweep — single command (recommended)

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final \
  --continue-on-error
```

Runtime auto-injects missing Stage-1 providers; each provider trains once.

### 1.2 Per-method decomposition (guarded, equivalent)

Use only when splitting across jobs. Providers consumed via
`stage2_source` need `--with-dependencies` (+15 steps = 5 tasks × 3 seeds).

| # | Method | Role | Stages | Total Steps |
|---|--------|------|--------|-------------|
| 1 | `phaseforge` | proposed — 6 experts, centroid router, 50% partial warm-start | 1, 2 (self) | 45 |
| 2 | `bc` | structured-state BC floor | 1 | 30 |
| 3 | `bc_large` | parameter-matched dense capacity control | 1 | 30 |
| 4 | `bc_rnn` | temporal comparator (10-step history) | 1 | 30 |
| 5 | `bc_robot_only` | robot-only negative control | 1 | 30 |
| 6 | `scratch_moe` | MoE architecture control, random init | 2 (none) | 30 |
| 7 | `warmstart_moe` | warm-start MoE (plain encoder × random router, needs `bc` S1) | 2 → `bc` | 45 |
| 8 | `phase_pretrain_random_router` | H1 control (needs `phaseforge` S1) | 2 → `phaseforge` | 45 |
| 9 | `plain_encoder_phase_bootstrap` | H2 control (needs `bc` S1) | 2 → `bc` | 45 |
| 10 | `teacher_forced` | privileged-training diagnostic, E8 (needs `phaseforge` S1) | 2 → `phaseforge` | 45 |

```bash
# 1 — Proposed (self-provided, no extra provider)
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods phaseforge --continue-on-error

# 2 — BC floor
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc --continue-on-error

# 3 — Parameter-matched dense
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc_large --continue-on-error

# 4 — Temporal history comparator
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc_rnn --continue-on-error

# 5 — Robot-only negative control
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc_robot_only --continue-on-error

# 6 — Scratch MoE (no provider)
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods scratch_moe --continue-on-error

# 7 — Warm-start MoE (needs bc S1: 30 + 15 = 45)
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods warmstart_moe --with-dependencies --continue-on-error

# 8 — Phase pretraining / random router (needs phaseforge S1: 45)
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods phase_pretrain_random_router --with-dependencies --continue-on-error

# 9 — Plain encoder / centroid router (needs bc S1: 45)
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods plain_encoder_phase_bootstrap --with-dependencies --continue-on-error

# 10 — Privileged training diagnostic (needs phaseforge S1: 45)
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods teacher_forced --with-dependencies --continue-on-error
```

Verify after:

```bash
uv run python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --dry-run
# every step `done`; then results.jsonl = 150 rows, training_summary.jsonl = 165 rows
```

---

## 2. Ablation Runs — Lift Suite (`experiments/lift_ablation.json`)

27 cells (Lift only) × 3 seeds = **165 steps** under `outputs_ablation`.
Nine are Lift replicas of the main matrix (`bc_rnn` excluded); 18 are
ablation-only (`pf_*`). The canonical provider is EXP-101 `phaseforge` S1,
shared by all `pf_*` cells (auto-injected).

### 2.1 Full suite — single command (recommended)

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/lift_ablation.json \
  --outputs outputs_ablation \
  --continue-on-error
```

Run only after the baseline matrix verifies clean (independent namespace,
but keeps provider semantics identical).

### 2.2 Per-group decomposition (guarded)

Each `pf_*` cell = Stage-2 train + eval = 2 steps/seed = 6 steps/method;
scoped selections add the canonical `phaseforge` S1 via
`--with-dependencies` (+3 steps).

```bash
# Group A — Router initialization controls (5 methods: 30 + 3 = 33)
uv run python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_spherical_kmeans pf_kmeans pf_phase_head pf_random_random pf_centroid_random --with-dependencies --continue-on-error

# Group B — Expert initialization & warm-start drop sweep (6 methods: 36 + 3 = 39)
uv run python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_full_warm pf_drop00 pf_drop25 pf_drop75 pf_drop100 pf_one_warm_plus_random --with-dependencies --continue-on-error

# Group C — Representation, fine-tuning & phase noise (5 methods: 30 + 3 = 33)
uv run python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_spherical pf_ft pf_corrupt_25 pf_corrupt_50 pf_shuffle_control --with-dependencies --continue-on-error

# Group D — Capacity & expert scaling, K sweep (2 methods: 12 + 3 = 15)
uv run python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_k3 pf_k12 --with-dependencies --continue-on-error
```

Single-cell recovery (any `pf_*` alone: 6 + 3 = 9):

```bash
uv run python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_drop25 --with-dependencies --continue-on-error
```

Verify after:

```bash
uv run python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --dry-run
```

---

## 3. Failed-cell re-run (only if needed)

```bash
# Example: re-run proposed method on Lift, seed 42, Stage 2 only
uv run python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final \
  --methods phaseforge@Lift --seeds 42 --stage 2 --force

# Add --with-dependencies if the cell needs its provider's S1
```

Useful filters: `--tasks Lift Can`, `--seeds 42,43`, `--stage {1,2}`, `--eval-only`, `--skip-eval`.

---

## 4. Paper tables (final namespaces ONLY, after verification)

```bash
uv run python scripts/analysis/summarize_train.py --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/summarize_eval.py  --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/stratified_stats.py --root outputs_final --json outputs_final/_summaries/stratified_stats.json
uv run python -m phaseforge.evaluations.rollout.report_cli outputs_final
```

`--root` fully replaces the default so historical `outputs/` rows cannot leak in.
