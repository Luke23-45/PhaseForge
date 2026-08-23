# Rollout Performance Review — Historical Findings and Rejected Optimization

> **Current implementation status (2026-08-23):** The canonicalized soft-reset
> path described below was reverted. The evaluation adapter again follows the
> dataset-compatible legacy reset path (`hard_reset` is not overridden,
> construction uses the normal robosuite RNG path, and state restore performs
> `sim.forward()` only). The soft-reset bank revision is no longer part of the
> active protocol. The measurements and recommendations below are retained as
> historical review material and must not be used to describe the active final
> experiment.

**Historical status:** patches were implemented and gated on 2026-08-22;
corrections covered the lite_physics fact, residual attribution,
parallel-cells reclassification, and gate redefinition.
**Date:** 2026-08-22
**Scope:** state-only rollout evaluation path (`phaseforge/evaluations/`), pinned rollout stack (robosuite 1.5.1 / mujoco 3.2.7 / torch 2.13+cpu, py3.11), motivated by the five-task cloud sweep.
**Method:** every claim below was measured on the actual installed stack or verified in the installed robosuite/mujoco source; none are assumed. Harnesses: [`scripts/dev/bench_rollout_hotpath.py`](../../../scripts/dev/bench_rollout_hotpath.py), [`scripts/dev/ab_reset_equivalence.py`](../../../scripts/dev/ab_reset_equivalence.py) — both now run on the **dataset-pinned** environment (composed from the real data config), not robosuite constructor defaults.

---

## 1. Headless verification

Yes — the adapter is fully headless by construction. `_FORCED_ENV_KWARGS`
(`robosuite_adapter.py`) forces:

```python
has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False
```

Verified against the installed robosuite 1.5.1 source: render contexts
(`MjRenderContextOffscreen`) are only constructed when those flags are true.
Consequently **no OpenGL context is ever created**, so cloud machines need no
EGL/OSMesa/MUJOCO_GL configuration for this protocol.

`lite_physics`: the **dataset pins `lite_physics=False`** in the recorded
env_kwargs (verified live on the composed protocol env), overriding
robosuite's `True` constructor default. This does not change the 25-substep
count — it selects the exact per-substep execution data collection used
(`sim.forward()` + `sim.step()` per substep, vs lite's split
`step1()`/`step2()`). Keep it False: enabling True would change numerics
relative to the collection protocol (a parity break), not just speed.

## 2. Measured cost structure (dev box medians, protocol env)

| Component | Median | Share of a timeout-heavy episode |
|---|---|---|
| `env.step` (one policy step = **25 physics substeps**) | ~17.8 ms | dominant per-step cost |
| policy inference batch=1 (canonical R50, CPU) | ~1.0–1.5 ms | minor |
| causal phase labeler per step | 0.06 ms | negligible |
| `validate_action` | 0.007 ms | negligible (run twice, by design — §4 P1b) |
| `extract_state` | 0.009 ms | negligible |
| **`reset_to` (retired hard reset)** | **~530–770 ms/episode** | was ~60–80 % of wall-clock on mixed runs |
| **`reset_to` (protocol: soft + canonicalized)** | **~2.7 ms/case** | adopted (§3.2, §7) |
| episodes.jsonl append (FileLock + fsync) | ~2.7 ms × 50 = ~133 ms/run | keep |

Key structural facts, verified in the installed robosuite source:

1. `MujocoEnv.step` runs `int(control_timestep/model_timestep)` = `int((1/20)/0.002)`
   = **25 MuJoCo substeps per policy step**. `control_freq=20` and the
   timestep come from the dataset's recorded env_kwargs; changing either
   would break contract parity with data collection. This cost is intrinsic.
2. robosuite's default `hard_reset=True` re-generates the arena XML and
   recompiles it through `mujoco.MjModel.from_xml_string` on *every* reset —
   cProfile of 20 resets showed **17.25 s of 20.3 s inside
   `from_xml_string` (~0.86 s per reset)**. State restore itself
   (`set_state_from_flattened` + `forward`) costs ~0.07 ms.
3. Bank cases store `xml=null` and `ep_meta=null` on disk (verified across
   all five frozen banks), so `reset_to`'s XML-reload branch never triggers;
   the measured hard-reset cost was entirely robosuite's internal recompile.

## 3. Reset-path engineering history

### 3.0 The pre-revision protocol was itself episode-order-dependent

Before any optimization, the **stock hard-reset pipeline leaked hidden state
across episodes**, so the "identical resets" promise held only for the
serialized `(time, qpos, qvel)`. Measured on Lift with identical bank cases
and identical deterministic action sequences:

- the same case run twice in a row on one env diverged (max state diff
  **9.6e-2** over 120 steps);
- with another case rolled in between, **3.8e-1**.

Root causes, each verified by direct field diffs: MuJoCo's
`qacc_warmstart` solver hint (measured at O(10) after an episode vs 0 for a
fresh sim), the OSC part controllers' cached references/goals and
`initial_joint` (the reset-time joint sample used for nullspace actions),
stale observable caches, and construction-time geometry drawn from the
global RNG (robosuite's `BoxObject` sizes sample `np.random.uniform` inside
`make`). The revision below fixes this defect — this is the primary
ratification argument, ahead of the speedup.

### 3.1 Naive soft branch — measured and rejected

`hard_reset=False` alone is ~200× faster at reset but is NOT
bit-equivalent: post-restore fields matched at t=0 only up to solver
residue, and trajectories diverged to O(0.1–1) within tens of steps
(chaotic contact amplification). Zeroing `qacc_warmstart` alone did not
restore equality. Rejected without the canonicalization of §3.2.

### 3.2 Canonicalized soft reset — implemented and gated (2026-08-22)

The adapter now combines three mechanisms:

1. **Deterministic construction seeding** — `robosuite.make` runs under a
   task-derived fixed seed with the caller's RNG state restored afterwards,
   so geometry is a pure function of the task, identical across methods,
   seeds, and processes.
2. **Soft branch forced** (`hard_reset=False` in `_FORCED_ENV_KWARGS`) —
   the compiled MJCF model is reused across resets (~2.7 ms vs ~530–770 ms).
3. **Hidden-state canonicalization after every restore** — zero
   `qacc_warmstart`, force-refresh every part controller
   (`update(force=True)`), then `update_initial_joints(restored joints)`
   (which for OSC also resets the goal to the achieved pose), and
   force-refresh observables. Order matters: reading `joint_pos` before the
   forced update would canonicalize to the *previous* episode's joints.

**The protocol property this delivers:** within the adopted path, the same
reset case produces a **bitwise-identical trajectory** regardless of
episode history, environment construction, method, or process. Proven by
the standing gate ([`ab_reset_equivalence.py`](../../scripts/dev/ab_reset_equivalence.py))
on **all five tasks** — history and construction arms BITWISE EQUAL on
every case (robosuite 1.5.1 and 1.5.0; Appendix B).

The retired hard branch deviates from the protocol arm by up to ~5e-2
(first control step, non-amplifying over 80-step probes; Appendix B). This
is **fully attributed**: each hard reset re-samples object placements from
the global RNG and recompiles the model before the serialized state is
restored, so the solver's path to the same restored state differs. It is
not unexplained residue, and it does not gate anything — the hard branch is
retired. Cross-machine bitwise equality is likewise not claimed: the
protocol pins seeds and code paths, not CPU floating-point behavior; do not
"validate" by diffing rollouts across different machines.

**Status:** adopted in code as a protocol revision. Consequences that must
be stated wherever results are reported: (a) all cells of the five-task
sweep share one reset path, so within-sweep comparisons are internally
valid; (b) numbers produced before this change (e.g. the pre-final Lift
rollout report) used hard-reset semantics and are not cross-version
comparable — and were themselves order-dependent per §3.0; (c) supervisor
ratification of the revision is required before sweep numbers enter the
paper (ledger D12).

### 3.3 Refuted alternative — "keep hard reset for bank generation" (external review, 2026-08-22)

An external review claimed a critical regression: with `hard_reset=False`
forced, `generate_reset_bank`'s `adapter.env.reset()` would take the soft
branch and "not re-sample placements", producing 50 duplicate states and an
`InfrastructureError` from the duplicate check. **Refuted twice over:**

1. **Source:** robosuite's `MujocoEnv.reset()` runs
   `self._reset_internal()` unconditionally in *both* the hard and soft
   branches — the hard branch only additionally recompiles the model. The
   placement sampling lives in `_reset_internal` (e.g. `Lift._reset_internal`
   calls `placement_initializer.sample()` and writes fresh object qpos), so
   soft resets re-sample placements exactly like hard ones.
2. **Measurement:** through the exact eval-path factory
   (`_adapter_from_config`, patched soft adapter), consecutive soft resets
   differ by L2 = 1.20, and 5-case generation is distinct on every task —
   min pairwise L2: Lift 0.135, ToolHang 0.096, Square 0.209, Transport
   0.102 (duplicate threshold 1e-3). No exceptions, nothing written to the
   frozen banks.

Institutionalized as the `--bank-smoke` mode of the standing harness (§8),
so the claim stays refuted under any future change. The same review's
other recommendations match the implemented state (tolerance ownership,
threads pinning, keep the intentional duplicate validation, balance-loss
skip stays rejected); its proposal to re-anchor the gate to the retired
hard branch (H-vs-S ≤ 1.8e-3) and to weaken the adapter docstring from
"bitwise-identical" to "~1e-3-bounded" were **rejected**: the adopted-path
bitwise property is proven (§3.2, Appendix B), and the ~1e-3 number belongs
to the retired branch only.

### 3.4 Bank invalidation after the reset-protocol revision

The reset-protocol change also invalidates every bank generated before this
revision. The bank identity now includes the explicit revision
`soft-reset-canonical-v1`, and every manifest stores the same revision. A
manifest with a missing or different revision is rejected by `ResetBank.load`
and cannot reach rollout evaluation. Consequently, the previously recorded
bank IDs are historical artifacts only: they must not be used for the final
sweep. The five current banks must be generated once through the final
evaluation path, their new IDs recorded in `final_run_plan.md` and the ledger,
and then SHA-256-verified on every subsequent load.

## 4. Adopt now (semantics-preserving) — implementation status

- **P0 — Parallel cells as OS processes. NOT ADOPTED for the final sweep.**
  Concurrent runner processes on one namespace are unsafe:
  `RunnerState.save()` persists its whole in-memory registry with no
  cross-process lock or merge — two processes lose each other's marks
  (last-writer-wins), breaking resumability and the "every step done"
  verification. (`results.jsonl` appends are FileLock+fsync safe, but that
  is insufficient.) The final sweep runs **sequentially**; with the reset
  fix the whole 150-cell evaluation is ~2–3 h of wall-clock while training
  dominates the schedule anyway. A `--jobs` worker pool inside a single
  runner process (one registry owner) would be the correct design if
  parallelism is ever needed — deferred unless the timeline demands it.
- **P0 — Pin the eval device explicitly. Already present** (`_resolve_device`
  in `phaseforge/cli.py` falls back to CPU when CUDA is unavailable and
  writes the effective device back into `cfg.project.device` before
  artifacts are written). Rollout venvs stay CPU-torch; training keeps GPU.
- **P1 — Action-contract tolerance harmonization. IMPLEMENTED.** The
  adapter owns the configured tolerance (`action_tolerance` constructor
  field fed from `eval.episodes.action_tolerance` via
  `_adapter_from_config`); `validate_action(tolerance=None)` resolves to
  it, so the runner's pre-step check and `step`'s own guard enforce the
  same contract. Default `1e-4` preserves prior behavior; gates pass
  explicit tolerances unchanged.
- **P1b — Remove the duplicated per-step validation. REJECTED with reason.**
  The runner-side check runs *before* the phase labeler consumes the state;
  deleting it would move the failure point of invalid-action episodes past
  one additional labeler transition, changing the recorded `max_phase` on
  failure rows — a behavioral delta for ~7 µs/step saved. Both checks now
  use identical values (P1), so the duplicate is harmless.
- **P2 — `torch.set_num_threads(1)` in rollout processes. IMPLEMENTED** at
  the top of `run_rollout_evaluation` (logged when applied; dedicated eval
  subprocess, so the process-global setting has no other consumers).
  Disclosure: thread count changes bitwise inference results (reduction
  order) — applied uniformly to every final evaluation, so within-sweep
  comparisons are unaffected; recorded as part of the protocol revision.
- **P2 — Skip the router's balance-loss branch at inference. REJECTED with
  reason.** Outputs were bitwise identical when skipped, but the stage-2
  trainer shares the code path, and the saving was below measurement noise.
- Keep unchanged: per-episode fsync (~133 ms/run, audit durability), bank
  SHA-256 verification (~0.05 s/run), success-predicate double evaluation
  (µs-scale; touching it risks parity for zero gain).

## 5. Expected wall-clock (post-adoption, dev-box numbers)

The five-task matrix is **150 evaluation cells** (50 method-task cells × 3
seeds). Grounded in §2's measurements (re-measure on the target machine
with the bench harness before scheduling):

| Scenario | Estimate |
|---|---|
| One eval cell (50 eps × ≤500 steps, protocol reset) | ~40–70 s |
| All 150 eval cells, sequential | ~2–3 h total |
| Retired path for contrast (hard resets) | +~27 s/cell ≈ +1.1 h |

Transport (horizon 700) adds ~1.4× stepping relative to 500-step tasks.
Training (165 runs) dominates the sweep schedule on the GPU box; the
evaluation half fits comfortably in a sequential run.

## 6. Cloud runbook deltas (Lightning.ai / Colab)

1. **No GL setup required** — the protocol never renders (§1). Do not
   install EGL/OSMesa packages for it; they are dead weight here.
2. **Rollout venv stays CPU-torch** (`torch+cpu`): smaller install, and
   batch-1 inference has no GPU upside. Train in the CUDA env; evaluate
   from the rollout env with `project.device=cpu` pinned in the manifests.
3. **Tool Hang exception stands:** its bank/dataset pin requires the
   separate robosuite 1.5.0 environment; its subprocesses route there
   automatically.
4. Re-run `scripts/dev/bench_rollout_hotpath.py` (and the §8 gate) on the
   target machine and recompute §5 before committing to a sweep schedule.

## 7. Reset-path protocol revision — ratification status

The **canonicalized soft reset** (§3.2) is implemented, tested, and gated.
It (a) cuts reset cost ~200×, (b) makes environment geometry
construction-seed-pure, and (c) **removes the episode-order dependence and
cross-method condition variance that the stock hard-reset path had (§3.0)**
— the adopted path is bitwise-deterministic per reset case.

Governance: this is a **protocol revision**, not an engineering fix — the
supervisor must ratify it (ledger D12) before sweep numbers enter the
paper. Required statements wherever results are reported: single reset path
for all sweep cells (internally valid); pre-revision results are not
cross-version comparable and were themselves order-dependent; the revision
*improves* determinism by removing placement-RNG coupling from evaluation;
inference thread count is pinned to 1 in eval processes (P2 disclosure).

## 7b. Legacy note (superseded candidate text)

An earlier version of §7 framed the deterministic reset as a professor-gated
future option and judged the adopted path by bitwise equality against the
hard branch (~1e-3 residual). Superseded: the hard branch is retired, and
the property that gates the protocol is within-path bitwise determinism
(proven, §3.2). Retained only so prior references resolve.

## 8. Standing gate (redefined)

[`ab_reset_equivalence.py`](../../scripts/dev/ab_reset_equivalence.py) gates
the **adopted** reset path on the **dataset-pinned** environment:

```bash
.venv-rollout/Scripts/python.exe scripts/dev/ab_reset_equivalence.py            # lift,can,square,transport
.venv-toolhang/Scripts/python.exe scripts/dev/ab_reset_equivalence.py --tasks tool_hang
```

PASS = exit code 0 with every **history** and **construction** arm BITWISE
EQUAL on every requested task. A retired hard-reset arm is printed as INFO
only (expected small first-step deviation; §3.2). Any future reset-path
change must re-run this gate and pass before adoption.

`--bank-smoke` (same script) gates generation diversity: bank generation
through the patched soft-reset adapter must produce pairwise-distinct
cases on every requested task (§3.3); nothing is written to disk:

```bash
.venv-rollout/Scripts/python.exe scripts/dev/ab_reset_equivalence.py --bank-smoke
.venv-toolhang/Scripts/python.exe scripts/dev/ab_reset_equivalence.py --bank-smoke --tasks tool_hang
```

On Linux shells use `python scripts/dev/...` with the activated rollout env.

---

## Appendix A — raw measurements (dev box, 2026-08-22; pre/at-canonicalization mix)

```
# bench harness (protocol env, lite_physics=False, 25 substeps/step)
physics substeps per policy step : 25            (control_freq 20, dt 0.002)
env.step median                  : 17.0–17.8 ms  p90 ~23–24
_get_observations                :  0.11 ms      (post-step obs dict rebuild)
_check_success                   :  0.002 ms
R50 forward batch=1 thr=2 / thr=1: ~1.0–1.5 / ~1.0 ms
phase_labeler.step               :  0.06 ms
validate_action                  :  0.007 ms
extract_state                    :  0.009 ms
reset_to RETIRED (hard)          : 526–770 ms median across runs
reset_to PROTOCOL (soft+canon.)  : 2.1–2.7 ms median
cProfile 20x env.reset (retired) : 17.25 s of 20.30 s in mujoco from_xml_string (2 compiles/reset)
append+fsync                     : 2.1–2.7 ms/episode
bank load+verify (50 cases)      : 0.05 s

# pre-canonicalization divergence (stock pipeline, identical cases+actions)
same case twice, one env         : max diff 9.6e-2 (120 steps)
same case, other case in between : max diff 3.8e-1
post-restore qacc_warmstart      : O(10) vs 0 for a fresh sim
balance-loss skip at inference   : outputs bitwise identical (max|diff| = 0)
```

## Appendix B — standing-gate output (2026-08-22, post-adoption)

`ab_reset_equivalence.py` — 3 cases × 80 steps per task, protocol arms vs
retired-hard INFO arm:

```
lift      (robosuite 1.5.1): GATE history BITWISE EQUAL ×3 | GATE construction BITWISE EQUAL ×3 | INFO retired-hard ≤ 8.6e-04
can       (robosuite 1.5.1): GATE history BITWISE EQUAL ×3 | GATE construction BITWISE EQUAL ×3 | INFO retired-hard 0.0
square    (robosuite 1.5.1): GATE history BITWISE EQUAL ×3 | GATE construction BITWISE EQUAL ×3 | INFO retired-hard 0.0
transport (robosuite 1.5.1): GATE history BITWISE EQUAL ×3 | GATE construction BITWISE EQUAL ×3 | INFO retired-hard ≤ 5.0e-02
tool_hang (robosuite 1.5.0): GATE history BITWISE EQUAL ×3 | GATE construction BITWISE EQUAL ×3 | INFO retired-hard 0.0
GATE PASSED (exit 0) on all five tasks.
```

Historical context: an earlier A/B run comparing against the then-current
hard-reset reference measured worst-case H-vs-S deviations of ~1e-3–1.8e-3
on 80-step probes after canonicalization — consistent with the retired
branch's per-reset placement re-sampling (§3.2), and irrelevant to the
adopted path, which is the bitwise-gated one.
