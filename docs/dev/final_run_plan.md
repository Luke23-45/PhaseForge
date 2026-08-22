# Final Run Plan — Five-Task Sweep → Paper Tables (canonical R50 protocol)

Run this once, top to bottom, on the machine that will produce the paper
results. Verified against the migrated codebase (canonical `phaseforge` =
R50 contract, 10-method matrix in `experiments/five_task.json`, checkpoint
contract gating, fresh `outputs_final/` namespace).

> **Preconditions (implementation ledger Phase 0/§11):** professor approval
> of `final_baselines_plan.md` recorded, and the D2 multiplicity correction
> confirmed (draft: Holm step-down over the five primary comparisons per
> task — see `state_only_rollout_implementation_plan.md` §5). Do not start
> the sweep until both are recorded in the ledger's Progress Log.

---

## 1. One-time setup

### 1.1 Environments (pinned simulator tracks)

The dev environment may run any modern Python (tests only). The **rollout
environment must be Python 3.11** — `mujoco==3.2.7` has no wheels for
Python ≥3.13:

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

### 1.2 Datasets (fail-closed — the pipeline will NOT download)

All five official robomimic **Proficient-Human low-dim v1.5** HDF5 sets under
`{PHASEFORGE_DATA_DIR}/raw/robomimic/{lift,can,square,tool_hang,transport}/`
(default root `./data/raw/robomimic/`). Record dataset revisions/checksums
in the external data manifest. The cache builds automatically on first use.

### 1.3 Sanity: the manifest loads

```bash
uv run python -m phaseforge.runner --manifest experiments/five_task.json --list
```

Expect 50 methods (10 per task × 5 tasks), `seeds: [42, 43, 44]`, and the
proposed `phaseforge` rows with `stage1,stage2`.

---

## 2. Environment gates (checkpoint-free — run BEFORE any training)

```bash
.venv-rollout/Scripts/phaseforge-gates data=lift eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=can eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=square eval=rollout
.venv-toolhang/Scripts/phaseforge-gates data=tool_hang eval=rollout
.venv-rollout/Scripts/phaseforge-gates data=transport eval=rollout
```

Every gate must pass (simulator parity, reset bank, action contract, native
success predicate). A failed gate blocks the sweep (plan §11 gate 9).

---

## 3. Pre-flight (read-only)

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --dry-run
```

Expect exactly **315 steps** (hand-computed: 19/task/seed × 5 tasks + 10
BC-RNN rows = 105/seed × 3 seeds). On a fresh namespace every step is
`pending`; the dependency graph must contain no stale provider names.

---

## 4. Run the full final sweep

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final \
  --continue-on-error
```

Notes:
- **`--outputs outputs_final` is mandatory** — the fresh final namespace.
  Never point the final sweep at the historical `outputs/` tree; the runner
  additionally fail-closes on wrong-contract artifacts (6-expert check) in
  both the stage-2 provider and evaluation funnels.
- Tool Hang subprocesses are routed to `.venv-toolhang` automatically
  (robosuite 1.5.0); everything else uses the active environment, so run
  the sweep with the rollout venv's Python or ensure `.venv-rollout` is
  importable — simplest: `.venv-rollout/Scripts/python -m phaseforge.runner ...`.
- `--continue-on-error` records a failed step and keeps going; re-run
  failed cells afterwards (section 6).
- **Resumable**: completed steps are skipped on re-run (state registry +
  `<run_dir>.completed` markers). Safe to interrupt and restart.
- Output layout: `outputs_final/{model}/stage{N}/seed{S}/{ts}[_{tag}]_{runid}/`
  (eval under `outputs_final/eval/{model}/seed{S}/`).

---

## 5. Verify completion

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/five_task.json \
  --outputs outputs_final --dry-run
```

Expect every step `done` (no `pending`, no `failed`), and confirm:

```powershell
Get-Item outputs_final\_results\results.jsonl, outputs_final\_results\training_summary.jsonl
```

Reproduce the fairness/parameter table against the migrated canonical
method before reporting (the committed `fairness_accounting.md` still shows
the retired 8-expert row):

```bash
uv run python scripts/analysis/fairness_accounting.py --outputs outputs_final
```

---

## 6. Re-run a failed cell (only if needed)

Fix the cause first, then re-run the specific cell with `--force`:

```bash
uv run python -m phaseforge.runner --manifest experiments/five_task.json \
  --outputs outputs_final --methods 1 --seeds 42 --stage 2 --force
```

`--force` re-runs selected steps regardless of registry status. For a
partial selection that needs a provider's Stage 1, add
`--with-dependencies`.

---

## 7. Generate paper tables (all against the final namespace ONLY)

```bash
uv run python scripts/analysis/summarize_train.py --outputs outputs_final --baseline phaseforge
uv run python scripts/analysis/summarize_eval.py  --outputs outputs_final --baseline phaseforge
.venv-rollout/Scripts/phaseforge-rollout-report outputs_final
```

Report contents per `state_only_rollout_implementation_plan.md` §5: per-task
per-seed success with Wilson intervals, paired PhaseForge-minus-baseline
differences on identical resets (McNemar + Newcombe), the declared
multiplicity correction over the five primary comparisons, per-task
parameter counts (bc_large match is per-task: +0.8%…+1.8% deployed),
routing diagnostics, training cost, and provenance hashes. Three seeds are
descriptive only.

---

## 8. What is NOT part of this plan

- No manual per-method `phaseforge-train`/`phaseforge-eval` — the runner
  emits those internally with exact seed/ckpt plumbing and contract checks.
- No ablation-suite runs — `experiments/lift_ablation.json` is executed
  separately after the main matrix (ledger D8), into its own namespace.
- No oracle/teacher-forced evaluation until D10 is resolved (the oracle
  dispatch path requires the soft-mapping decision).
- Pre-final results under `outputs/` are never merged into the final tables.
