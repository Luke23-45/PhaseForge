# Final Run Plan — Five-Task Sweep → Paper Tables

Run once, top to bottom, on the machine that produces the paper results.
Manifest `experiments/five_task.json`, namespace `outputs_final/` only.

**Preconditions:** professor approval recorded in the ledger; D2
multiplicity correction confirmed (draft: Holm step-down over the five
primary comparisons per task); D10 resolved before any oracle /
teacher-forced *diagnostic* evaluation (blocks nothing else).

---

## 1. Method matrix (frozen)

Ten methods per task × five tasks (Lift, Can, Square, ToolHang, Transport)
× seeds 42/43/44 = 50 cells, 315 runner steps.

| Method | Role | Model config | Data | Stages | S2 source |
|---|---|---|---|---|---|
| `phaseforge` | proposed (6 experts, centroid router, 50% partial warm-start) | `phaseforge` | `<task>` | 1, 2 | self |
| `bc` | structured-state floor | `baselines/bc` | `<task>` | 1 | — |
| `bc_large` | parameter-matched dense capacity control | `baselines/bc_large` | `<task>` | 1 | — |
| `bc_rnn` | temporal comparator (10-step, same schema; not history-matched) | `baselines/bc_rnn` | `<task>_rnn` | 1 | — |
| `bc_robot_only` | robot-only negative control (descriptive) | `baselines/bc` | `robot_only_<task>` | 1 | — |
| `scratch_moe` | MoE architecture control (random init, no Stage 1) | `baselines/scratch_moe` | `<task>` | 2 | none |
| `warmstart_moe` | warm-start control (plain encoder × random router) | `baselines/warmstart_moe` | `<task>` | 2 | bc |
| `phase_pretrain_random_router` | phase-representation control | `baselines/phase_pretrain_random_router` | `<task>` | 2 | phaseforge |
| `plain_encoder_phase_bootstrap` | centroid-init control | `baselines/plain_encoder_phase_bootstrap` | `<task>` | 2 | bc |
| `teacher_forced` | privileged-training diagnostic (E8, descriptive) | `baselines/teacher_forced` | `<task>` | 2 | phaseforge |

D2 primary family: `phaseforge` vs `bc`, `scratch_moe`, `warmstart_moe`,
`phase_pretrain_random_router`, `plain_encoder_phase_bootstrap` — five per
task. `bc_large`/`bc_rnn` reported with intervals but outside the corrected
family; `bc_robot_only`/`teacher_forced` descriptive.

The router controls `pf_spherical_kmeans`/`pf_kmeans`/`pf_phase_head` are
not in this matrix (D1): they run in the Lift ablation (section 9) on the
same canonical provider + 50% partial warm-start — an exact-match
factorial.

---

## 2. One-time setup

### 2.1 Environments

Rollout environments must be **Python 3.11** (`mujoco==3.2.7` has no
wheels for ≥ 3.13); the dev environment runs tests only. On POSIX use
`bin/` instead of `Scripts/` throughout this document.

```bash
# main rollout env (robosuite 1.5.1 / mujoco 3.2.7)
uv venv --python 3.11 .venv-rollout
uv pip install --python .venv-rollout -e ".[rollout]"
cp .venv-rollout/Lib/site-packages/mujoco/mujoco.dll \
   .venv-rollout/Lib/site-packages/robosuite/utils/mujoco.dll

# Tool Hang env (robosuite 1.5.0 / mujoco 3.2.7)
uv venv --python 3.11 .venv-toolhang
uv pip install --python .venv-toolhang -e .
uv pip install --python .venv-toolhang "robosuite==1.5.0" "mujoco==3.2.7"
cp .venv-toolhang/Lib/site-packages/mujoco/mujoco.dll \
   .venv-toolhang/Lib/site-packages/robosuite/utils/mujoco.dll
```

(The DLL copy: robosuite 1.5.x on Windows loads a DLL from its own utils
dir that the PyPI wheel does not bundle.)

### 2.2 Datasets (fail-closed — no download)

The five official robomimic **PH low-dim v1.5** HDF5 sets under
`{PHASEFORGE_DATA_DIR}/raw/robomimic/{lift,can,square,tool_hang,transport}/`
(default `./data/raw/robomimic/`). Record revisions/checksums in the
external data manifest. Caches build on first use, strictly hash-gated on
reuse.

### 2.3 Reset banks (frozen artifacts — travel with the `data/` root)

| Task | bank_id | robosuite |
|---|---|---|
| Lift | `a7d3953c0afcf560` | 1.5.1 |
| Can | `310d9cfd3fa5e843` | 1.5.1 |
| Square | `e16288589f5f69c2` | 1.5.1 |
| ToolHang | `db5b4c2a5e6519d0` | 1.5.0 |
| Transport | `c6683cf0dbb23876` | 1.5.1 |

50 cases each, bank seed 2026, SHA-256-verified. A machine lacking them
regenerates byte-identical banks during section 3's gates
(`auto_generate: true`).

### 2.4 Sanity: manifest loads

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json --list
```

Expect 50 method rows (10 per task), `seeds: [42, 43, 44]`, `phaseforge`
rows with `stage1,stage2`.

---

## 3. Environment gates (before any training)

```bash
.venv-rollout/Scripts/phaseforge-gates data=lift eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=can eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=square eval=rollout
.venv-toolhang/Scripts/phaseforge-gates data=tool_hang eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=transport eval=rollout
```

Every gate must pass (parity, bank verification, action contract, native
success predicate, sanity sweeps, demo replay). A failed gate blocks the
sweep (plan §11 gate 9). Never substitute `uv run` — it resolves the
mujoco-less dev environment.

---

## 4. Pre-flight (read-only)

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --dry-run
```

Expect exactly **315 steps** = 21 per (task, seed): 11 training
(`phaseforge` trains Stages 1+2, the other nine methods their final stage
only) + 10 evaluations; × 5 tasks × 3 seeds. Every step `pending` on a
fresh namespace.

Preview caveat: the `$` echoes on a fresh namespace show `AUTO-INJECT`
dependency previews (provider stage-1 commands may repeat 2–4×) and
`BLOCKED` evals — execution trains each provider once and reuses it. The
authoritative signal is the numbered `done`/`pending` list.

---

## 5. Run the sweep

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final \
  --continue-on-error
```

- Launch from the rollout venv as shown; Tool Hang steps (train and eval)
  route automatically to `.venv-toolhang` with a version-pin preflight.
- `--outputs outputs_final` is mandatory; the runner fail-closes on
  wrong-contract artifacts (6-expert + config-hash gates).
- `--continue-on-error` records failures and keeps going (section 7).
- Resumable: completed steps are skipped (state registry + `.completed`
  markers); safe to interrupt and restart.
- Layout: `outputs_final/{model}/stage{N}/seed{S}/{ts}[_{tag}]_{runid}/`,
  eval under `outputs_final/eval/{model}/seed{S}/`.

---

## 6. Verify completion

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --dry-run
```

Every step `done` (no `pending`, no `failed`), then:

- `outputs_final/_results/results.jsonl` holds **150 rows** (50 cells × 3
  seeds); `training_summary.jsonl` holds **165 rows** (55 train cells × 3).
- Every cache used by final evals contains `phase_thresholds.json` — a
  missing artifact silently nulls per-phase SR (S8b.6); verify none are
  null before reporting.

Parameter/fairness tables (the committed `fairness_accounting.md` still
shows the retired 8-expert row):

```bash
uv run python scripts/analysis/fairness_accounting.py   # Lift-schema table, no args
uv run python scripts/protocol/preflight_configs.py     # per-task ±2% match; must end
                                                        # "all 165 train ... 150 eval ... passed"
```

---

## 7. Re-run a failed cell (only if needed)

Fix the cause first, then:

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --methods 1 --seeds 42 --stage 2 --force
```

`--force` re-runs regardless of registry status; add
`--with-dependencies` when a partial selection needs a provider's Stage 1.
Never widen the outputs namespace.

---

## 8. Paper tables (final namespace ONLY)

```bash
uv run python scripts/analysis/summarize_train.py --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/summarize_eval.py  --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/stratified_stats.py --root outputs_final --json outputs_final/_summaries/stratified_stats.json
.venv-rollout/Scripts/phaseforge-rollout-report outputs_final
```

`stratified_stats.py` reports per task (never pooled): seed-level means,
bootstrap 95% CIs, pairwise P(X > Y). `--root` fully replaces the default
— historical `outputs/` rows cannot leak in.

Contents per `state_only_rollout_implementation_plan.md` §5:

- per-episode success/validity/reset case/seed/horizon outcome/failure category;
- success rate per model × task × seed with 95% Wilson intervals over
  valid episodes; mean ± sample std across seeds; the five-task
  macro-average as a secondary aggregate only;
- paired PhaseForge-minus-baseline differences on identical reset cases
  per seed (exact McNemar + Newcombe CI; no pooling); across-seed
  direction consistency reported descriptively;
- the D2 correction over the five primary comparisons per task — three
  seeds are descriptive, no population-level significance claims;
- infrastructure failures excluded from the denominator and listed;
  policy-caused invalid actions/NaNs counted as failures (strict metric)
  and labeled;
- per-task parameter counts (`bc_large` match +0.8%…+1.8% deployed),
  routing diagnostics, training cost, provenance hashes as secondary
  diagnostics.

---

## 9. Ablation suite (after the main matrix — D8)

`experiments/lift_ablation.json`: 27 cells, Lift only, seeds 42/43/44, own
namespace `outputs_ablation/`, 165 steps. Nine Lift replicas of the main
matrix (`bc_rnn` excluded; self-contained) plus 18 ablation-only cells:

- **Router init** (canonical provider + 50% partial warm): `pf_spherical_kmeans`,
  `pf_kmeans`, `pf_phase_head`, `pf_random_random`, `pf_centroid_random`;
- **Expert init**: drop-rate `pf_drop00/25/75/100` (50% = canonical),
  `pf_full_warm`, `pf_one_warm_plus_random`;
- **Representation/training**: `pf_spherical`, `pf_ft`, `pf_corrupt_25/50`,
  `pf_shuffle_control`;
- **Capacity**: `pf_k3`, `pf_k12` (top-1 variant never included).

Removed at migration (final): `pf_random_warm`, `phaseforge_e6` (D3,
redundant); `pf_jitter_00/10` (D4, superseded by the drop-rate sweep);
`warmstart_r50` (would recreate the canonical method).

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/lift_ablation.json \
  --outputs outputs_ablation \
  --continue-on-error
```

Start only after the main matrix is complete and verified (section 6).

---

## 10. Not part of this plan

- No manual per-method `phaseforge-train`/`phaseforge-eval` — the runner
  emits them with exact seed/checkpoint plumbing and contract checks.
- No oracle/teacher-forced evaluation until D10 is resolved.
- No external re-run baselines (D9) — robomimic-study and Diffusion-Policy
  numbers appear only as literature context in the paper.
- Pre-final results under `outputs/` are never merged into the final tables.
- Retired identities stay gone: the old 8-expert `phaseforge` config, the
  `phaseforge_r50` config and alias (ledger Phases 1–2), and
  `lar_moe_state_only` (D11). No `phaseforge_r50` row, namespace, or paper
  label exists.
