# Final Run Plan — Five-Task Sweep → Paper Tables

Run once, top to bottom, on the machine that produces the paper results.
Namespace: `outputs_final/` only. Manifest: `experiments/five_task.json`.

**Preconditions (do not start the sweep without them):**

- Professor approval of the final baseline plan recorded in the ledger.
- D2 multiplicity correction confirmed (draft: Holm step-down over the five
  primary comparisons per task — `state_only_rollout_implementation_plan.md` §5).
- D10 (oracle dispatch on centroid init) resolved before any
  oracle/teacher-forced *diagnostic* evaluation; it blocks nothing else.

---

## 1. Method matrix (frozen)

Ten methods per task, three seeds (42, 43, 44), five tasks
(Lift, Can, Square, ToolHang, Transport) = 50 cells, 315 runner steps.

| Method | Role | Model config | Data config | Stages | Stage-2 source |
|---|---|---|---|---|---|
| `phaseforge` | proposed method (6 experts, centroid router, 50% partial warm-start) | `phaseforge` | `<task>` | 1, 2 | self |
| `bc` | structured-state floor | `baselines/bc` | `<task>` | 1 | — |
| `bc_large` | parameter-matched dense capacity control | `baselines/bc_large` | `<task>` | 1 | — |
| `bc_rnn` | temporal-history comparator (10-step, same schema; not history-matched) | `baselines/bc_rnn` | `<task>_rnn` | 1 | — |
| `bc_robot_only` | robot-only negative control | `baselines/bc` | `robot_only_<task>` | 1 | — |
| `scratch_moe` | MoE architecture control (random init, no Stage 1) | `baselines/scratch_moe` | `<task>` | 2 | none |
| `warmstart_moe` | warm-start control (plain encoder × random router) | `baselines/warmstart_moe` | `<task>` | 2 | bc |
| `phase_pretrain_random_router` | phase-representation control | `baselines/phase_pretrain_random_router` | `<task>` | 2 | phaseforge |
| `plain_encoder_phase_bootstrap` | centroid-init control | `baselines/plain_encoder_phase_bootstrap` | `<task>` | 2 | bc |
| `teacher_forced` | privileged-training diagnostic (E8; descriptive) | `baselines/teacher_forced` | `<task>` | 2 | phaseforge |

Primary comparison family (D2): `phaseforge` vs `bc`, `scratch_moe`,
`warmstart_moe`, `phase_pretrain_random_router`,
`plain_encoder_phase_bootstrap` — five per task. `bc_large` and `bc_rnn`
are reported with intervals but sit outside the corrected family;
`bc_robot_only` and `teacher_forced` are descriptive.

The mechanism controls `pf_spherical_kmeans`, `pf_kmeans`, `pf_phase_head`
are **not** in this matrix (ledger D1): they run in the Lift ablation
suite (section 9).

---

## 2. One-time setup

### 2.1 Environments (pinned simulator tracks)

The rollout environment must be **Python 3.11** — `mujoco==3.2.7` has no
wheels for Python ≥ 3.13. The dev environment only runs tests.

```bash
# main rollout env (robosuite 1.5.1 / mujoco 3.2.7)
uv venv --python 3.11 .venv-rollout
uv pip install --python .venv-rollout -e ".[rollout]"
# robosuite 1.5.x on Windows loads a DLL from its own utils dir that the
# PyPI wheel does not bundle — copy it from the mujoco package:
cp .venv-rollout/Lib/site-packages/mujoco/mujoco.dll \
   .venv-rollout/Lib/site-packages/robosuite/utils/mujoco.dll

# Tool Hang env (robosuite 1.5.0 / mujoco 3.2.7)
uv venv --python 3.11 .venv-toolhang
uv pip install --python .venv-toolhang -e .
uv pip install --python .venv-toolhang "robosuite==1.5.0" "mujoco==3.2.7"
cp .venv-toolhang/Lib/site-packages/mujoco/mujoco.dll \
   .venv-toolhang/Lib/site-packages/robosuite/utils/mujoco.dll
```

On POSIX use `bin/` instead of `Scripts/` throughout this document.

### 2.2 Datasets (fail-closed — the pipeline will NOT download)

All five official robomimic **Proficient-Human low-dim v1.5** HDF5 sets under
`{PHASEFORGE_DATA_DIR}/raw/robomimic/{lift,can,square,tool_hang,transport}/`
(default root `./data/raw/robomimic/`). Record dataset revisions/checksums
in the external data manifest. Caches build automatically on first use and
are strictly hash-gated on reuse.

### 2.3 Reset banks

The five frozen reset banks already exist (50 cases each, bank seed 2026,
SHA-256-verified) and travel with the `data/` root:

| Task | bank_id | robosuite |
|---|---|---|
| Lift | `a7d3953c0afcf560` | 1.5.1 |
| Can | `310d9cfd3fa5e843` | 1.5.1 |
| Square | `e16288589f5f69c2` | 1.5.1 |
| ToolHang | `db5b4c2a5e6519d0` | 1.5.0 |
| Transport | `c6683cf0dbb23876` | 1.5.1 |

If a machine lacks them, section 3's gate run regenerates byte-identical
banks via the same code path (`auto_generate: true`).

### 2.4 Sanity: the manifest loads

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json --list
```

Expect 50 method rows (10 per task × 5 tasks), `seeds: [42, 43, 44]`, and
the `phaseforge` rows with `stage1,stage2`.

---

## 3. Environment gates (checkpoint-free — run BEFORE any training)

```bash
.venv-rollout/Scripts/phaseforge-gates data=lift eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=can eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=square eval=rollout
.venv-toolhang/Scripts/phaseforge-gates data=tool_hang eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=transport eval=rollout
```

Every gate must pass (simulator parity, reset-bank verification, action
contract, native success predicate, random/no-op sanity, demo replay). A
failed gate blocks the sweep (plan §11 gate 9). Never substitute
`uv run` here — it resolves the mujoco-less dev environment.

---

## 4. Pre-flight (read-only)

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --dry-run
```

Expect exactly **315 steps** = 21 per (task, seed) — 11 training steps
(`phaseforge` trains Stage 1 and Stage 2; the other nine methods train
their final stage only) + 10 evaluation steps — × 5 tasks × 3 seeds. On a
fresh namespace every step is `pending`.

Preview semantics: the `$` command echoes against a fresh namespace show
`AUTO-INJECT dependency` previews for stage-2 cells (identical provider
stage-1 commands may repeat 2–4×) and `BLOCKED` for every eval — at
execution the providers are trained once and reused, never retrained. The
authoritative signal is the numbered `done`/`pending` list.

---

## 5. Run the full final sweep

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final \
  --continue-on-error
```

- Launch from the rollout venv as shown. Every subprocess except Tool Hang
  resolves its console scripts from the launching environment; Tool Hang
  steps (train and eval) are routed automatically to `.venv-toolhang`
  (robosuite 1.5.0) and pin-preflighted.
- `--outputs outputs_final` is mandatory. The runner fail-closes on
  wrong-contract artifacts (6-expert check, config-hash gate) in both the
  stage-2 provider and evaluation funnels.
- `--continue-on-error` records a failed step and keeps going; re-run
  failures afterwards (section 7).
- Resumable: completed steps are skipped on re-run (state registry +
  `<run_dir>.completed` markers). Safe to interrupt and restart.
- Output layout: `outputs_final/{model}/stage{N}/seed{S}/{ts}[_{tag}]_{runid}/`
  (eval under `outputs_final/eval/{model}/seed{S}/`).

---

## 6. Verify completion

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --dry-run
```

Every step `done` (no `pending`, no `failed`), and:

```powershell
Get-Item outputs_final\_results\results.jsonl, outputs_final\_results\training_summary.jsonl
```

- `results.jsonl` must hold **150 rows** (50 evaluated cells × 3 seeds);
  `training_summary.jsonl` must hold **165 rows** (55 train cells × 3 seeds).
- Per-phase success rates: every cache used by the final evals must contain
  `phase_thresholds.json` — with `require_phase_tracking: false` a missing
  artifact silently nulls per-phase SR (ledger S8b.6). Verify none are null
  before reporting.

Parameter/fairness tables (reproduce before reporting; the committed
`fairness_accounting.md` still shows the retired 8-expert row):

```bash
uv run python scripts/analysis/fairness_accounting.py
uv run python scripts/protocol/preflight_configs.py
```

`fairness_accounting.py` takes no arguments and prints the Lift-schema
fairness table (Markdown + LaTeX). The per-task deployed-parameter match
(±2% vs PhaseForge per task) is enforced by the preflight script, which
must end `all 165 train cell(s) and 150 eval cell(s) passed`.

---

## 7. Re-run a failed cell (only if needed)

Fix the cause first, then re-run the specific cell with `--force`:

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --methods 1 --seeds 42 --stage 2 --force
```

`--force` re-runs selected steps regardless of registry status. For a
partial selection that needs a provider's Stage 1, add
`--with-dependencies`. Never widen the outputs namespace.

---

## 8. Generate paper tables (final namespace ONLY)

```bash
uv run python scripts/analysis/summarize_train.py --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/summarize_eval.py  --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/stratified_stats.py --root outputs_final --json outputs_final/_summaries/stratified_stats.json
.venv-rollout/Scripts/phaseforge-rollout-report outputs_final
```

`stratified_stats.py` reports per task (never pooled): seed-level means +
percentile bootstrap 95% CIs and the pairwise P(X > Y) matrix. `--root`
fully replaces the default — historical `outputs/` rows can never leak in.

Report contents per `state_only_rollout_implementation_plan.md` §5:

- per-episode success/validity/reset case/seed/horizon outcome/failure category;
- success rate per model × task × training seed, with 95% Wilson intervals
  over valid episodes;
- mean ± sample std across the three seeds per task; the unweighted
  macro-average over the five tasks as a secondary aggregate only;
- paired PhaseForge-minus-baseline differences on identical reset cases,
  per training seed (exact McNemar + Newcombe 95% CI; no pooling across
  seeds); direction consistency across seeds reported descriptively;
- the declared multiplicity correction (D2) over the five primary
  comparisons per task; three seeds are descriptive — no population-level
  significance claims;
- infrastructure failures excluded from the denominator and listed
  separately; policy-caused invalid actions/NaNs counted as failures
  (strict metric) and labeled;
- per-task parameter counts (bc_large match is per-task: +0.8%…+1.8%
  deployed), routing diagnostics, training cost, configuration/provenance
  hashes as secondary diagnostics.

---

## 9. Ablation suite (AFTER the main matrix — ledger D8)

`experiments/lift_ablation.json`: 27 cells, Lift only, seeds 42/43/44, own
namespace `outputs_ablation/` (165 runner steps). Contains the matched
mechanism controls
(`pf_spherical_kmeans`, `pf_kmeans`, `pf_phase_head`), the router/init
factorial (`pf_random_random`, `pf_centroid_random`), top-k and
partial-warm-rate sweeps (`pf_k3`/`pf_k12`, `pf_drop00/25/75/100`,
`pf_full_warm`, `pf_one_warm_plus_random`), phase-supervision corruption
(`pf_corrupt_25/50`, `pf_shuffle_control`), and fine-tune/representation
variants (`pf_ft`, `pf_spherical`).

```bash
.venv-rollout/Scripts/python -m phaseforge.runner \
  --manifest experiments/lift_ablation.json \
  --outputs outputs_ablation \
  --continue-on-error
```

Start it only after the main matrix is complete and verified (section 6).

---

## 10. Not part of this plan

- No manual per-method `phaseforge-train`/`phaseforge-eval` — the runner
  emits those internally with exact seed/checkpoint plumbing and contract
  checks.
- No oracle/teacher-forced evaluation until D10 is resolved.
- Pre-final results under `outputs/` are never merged into the final tables.
- No external re-run baselines (D9): robomimic-study and Diffusion-Policy
  numbers appear only as literature context in the paper.
