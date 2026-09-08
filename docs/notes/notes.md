# PhaseForge — Final Experiment Command Reference

This file is the operator reference for the locked final experiment matrix.
The authoritative protocol is
`experiments/final_causal_matrix.json`; the scientific rules are in
`docs/final/baselines/PRECISION_RESIDUAL_BASELINE_PROTOCOL.md`.

The retired research manifests and their output trees have been removed. Use
only `experiments/final_causal_matrix.json` and a fresh `outputs_final/`
namespace for the locked comparison.

## Current readiness status

The implementation has been audited and the full repository test suite passes
(984 tests). The final manifest gates also pass:

- 10 method identities × 5 tasks × 3 seeds;
- 45 Stage 1 steps, 135 Stage 2 steps, and 150 evaluation steps;
- 330 total planned steps;
- explicit provider ordering and final-family model identities;
- no `_pw` method or phase-weighting override in the final matrix.

Training is not cleared until both operational prerequisites are satisfied:

1. Commit the reviewed implementation and run the matrix at that frozen
   commit. The current working tree must not remain dirty for final results.
2. Generate and verify the required `topo_pelt_k6` cache artifacts. Every
   topology-consuming task/seed must contain a valid
   `topo_artifact/topo_manifest.json`, including its label mapping and checksums.

The final output namespace must be fresh: `outputs_final/` must not contain
previous final run markers, checkpoints, or runner state.

## 1. Read-only pre-flight

Run these commands after committing the implementation and before training:

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/final_causal_matrix.json \
  --outputs outputs_final \
  --verify-gates
```

The report must show:

```text
methods=50 seeds=[42, 43, 44] steps=330
DRY-02_plan_expands: pass
DRY-03_provider_ordering: pass
DRY-04_command_contract: pass
DRY-05_historical_isolation: pass
namespace_fresh=True
```

Then inspect the complete command plan without executing training:

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/final_causal_matrix.json \
  --outputs outputs_final \
  --expect-steps 330 \
  --dry-run
```

Do not proceed if the plan is not exactly 330 steps, if the namespace is not
fresh, or if any provider/topology prerequisite is missing.

## 2. Locked final matrix

The matrix uses tasks `Lift`, `Can`, `Square`, `ToolHang`, and `Transport`,
with seeds `42`, `43`, and `44`.

| Identity | Role | Stage 1 source | Evaluation |
|---|---|---|---|
| `precision_residual_phaseforge` | Sole proposed method | self | rollout |
| `bc` using `final_aligned_bc` | normalized BC floor | self | rollout |
| `precision_residual_plain_encoder` | plain-representation control | `final_aligned_bc_stage1` | rollout |
| `precision_residual_phase_random_router` | random-router control | `precision_residual_phaseforge_stage1` | rollout |
| `precision_residual_scratch_moe` | random-expert control | `precision_residual_phaseforge_stage1` | rollout |
| `precision_residual_factorial_floor` | plain representation × random router | `final_aligned_bc_stage1` | rollout |
| `final_aligned_softmax_top1` | registered softmax router-package comparison | `precision_residual_phaseforge_stage1` | rollout |
| `final_aligned_static_rule` | rule-label integrated comparison | self | rollout |
| `precision_residual_teacher_forced` | privileged training diagnostic | `precision_residual_phaseforge_stage1` | separate diagnostic rollout |
| `precision_residual_oracle` | privileged offline diagnostic | `precision_residual_phaseforge_stage1` | offline only |

The final method uses normalized encoder latents, six experts, hard top-1
prototype routing, topology-derived prototypes, 50% partial expert
warm-start, and `beta=0.0`. The residual branch is therefore not a claimed
source of improvement in this matrix.

`phase_topo` is a train-only topology-derived label. It must never enter the
deployable policy's rollout input. The Static Rule comparison uses `phase`
consistently instead.

The teacher-forced and oracle rows are not pooled with ordinary deployable
success rates. The oracle must never be evaluated through the normal
state-only rollout interface.

## 3. Full final sweep

After the pre-flight gates pass and the topology artifacts are verified:

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/final_causal_matrix.json \
  --outputs outputs_final \
  --expect-steps 330
```

The runner resolves the explicit Stage 1 providers and records the exact
checkpoint consumed by each Stage 2 row. Do not use `--no-commit-gate` for
the final run. Do not use `--continue-on-error` for the locked publication
matrix; a failure should stop the sweep and be investigated.

Expected completed records are:

- 45 Stage 1 training records;
- 135 Stage 2 training records;
- 150 evaluation result rows;
- 330 total runner steps, including provider dependencies.

## 4. Recovery of an individual failed cell

Use the exact final identity and task facet. For example, to rerun Stage 2
for the proposed method on Lift, seed 42:

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/final_causal_matrix.json \
  --outputs outputs_final \
  --methods precision_residual_phaseforge@Lift \
  --seeds 42 \
  --stage 2 \
  --force
```

The required Stage 1 provider must already exist at the same committed
revision. For a missing provider, run the corresponding provider Stage 1
explicitly first; do not point the consumer at a historical checkpoint.

Useful filters are `--tasks`, `--seeds`, `--stage`, `--eval-only`, and
`--skip-eval`. Use `--eval-only` only when the exact final checkpoint already
exists and passes the checkpoint contract.

## 5. After the sweep

Verify the completed namespace before analysis:

```bash
uv run python -m phaseforge.runner \
  --manifest experiments/final_causal_matrix.json \
  --outputs outputs_final \
  --expect-steps 330 \
  --dry-run
```

Then inspect the final output ledgers and provenance. Every topology-consuming
run must carry the topology provenance metadata, provider checkpoint identity,
dataset/cache hash, commit, resolved-config hash, and evaluation reset-bank
identity. Do not combine these rows with any pre-final output tree.

The oracle remains an offline diagnostic, and the teacher-forced row remains a
separate privileged diagnostic. Neither supports the primary deployable
success-rate claim.
