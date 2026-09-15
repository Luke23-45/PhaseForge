# Router-initialization hypothesis and evidence

- **Status:** Evidence review after the completed focused ablation
- **Run:** `outputs_router_ablation_can_square`
- **Code revision recorded by the run:** `f83d096`
- **Tasks:** Can and Square
- **Seeds:** 42, 43, 44

## 1. Research question

The focused experiment asks:

> When the same memoryless, beta-zero, direct-action MoE is trained with the
> same phase-aware representation provider, does initializing the hard
> prototype router from topology-derived prototypes improve performance or
> routing organization relative to random or phase-derived initialization?

This is narrower than asking whether the complete PhaseForge method is better
than every baseline. The complete method also contains an objective-level
choice, including the Stage 2 margin loss. The focused ablation disabled that
loss so that router initialization could be compared under a matched Stage 2
objective.

## 2. Research hypotheses

### H1 — Routing-organization hypothesis

Topology-derived prototype initialization will produce more structured and
stable routing than random initialization. The observable indicators are the
logged phase-expert NMI and routing-switch rate.

**Result:** Supported by the logged routing diagnostics.

### H2 — Can performance hypothesis

Topology-derived initialization will improve Can rollout success relative to
phase-derived initialization and random initialization.

**Result:** Partially supported. It improves the pooled Can result relative to
phase initialization, but the advantage over random initialization is small
and not consistent across the three seeds.

### H3 — Cross-task performance hypothesis

Topology-derived initialization will improve rollout success consistently on
both Can and Square.

**Result:** Not supported by this run. Topology initialization is best among
the three initialization arms on Can but is worst on Square when compared with
phase initialization, representation BC, and the softmax-router control.

### H4 — Hard-routing package hypothesis

Hard prototype top-1 routing with topology initialization will outperform the
softmax-router control.

**Result:** Not supported. The softmax-router control has the highest pooled
success rate and wins on all three Square seeds.

### H5 — Phase-aware representation hypothesis

The phase-aware representation provider will improve rollout success relative
to the representation-BC control in a task-independent way.

**Result:** Not established. The topology arm is better on Can but worse on
Square. The evidence is task-dependent rather than uniformly positive.

## 3. Experimental contract actually executed

The runner ledger records 30 Stage 2 evaluations:

- 5 evaluated methods;
- 2 tasks;
- 3 training seeds;
- 50 rollout episodes per method/task/seed;
- 500-step rollout horizon.

The five arms were:

| Method token | Model package | Experimental role |
|---|---|---|
| `router_init_random` | `precision_residual_phase_random_router` | Random router initialization |
| `router_init_phase` | `precision_residual_phaseforge` | Centroids from `phase` |
| `router_init_topology` | `precision_residual_phaseforge` | Centroids from `phase_topo` |
| `representation_bc` | `precision_residual_plain_encoder` | Representation-BC control |
| `routing_softmax_top1` | `final_aligned_softmax_top1` | Softmax-router control with top-1 deployment |

The resolved Stage 2 configurations were checked across all 30 Stage 2
training runs:

- `beta: 0.0` in all 30 configurations;
- `train.margin.enabled: false` in all 30 configurations;
- the same action-loss contract and shared Stage 1 provider where applicable;
- no top-1 routing collapse was reported;
- all evaluations used reset seed `2026`;
- all methods within Can used reset bank `310d9cfd3fa5e843`;
- all methods within Square used reset bank `e16288589f5f69c2`.

The reset bank is therefore matched across methods within each task, but not
shared between the two tasks.

Therefore, this run evaluates a memoryless direct-action configuration. It is
not evidence about a nonzero residual impedance or feedback controller.

## 4. Rollout results

The primary metric is rollout success. Percentages below are pooled over the
three training seeds, with 50 episodes per seed. The raw counts are included
to avoid hiding the small sample size.

| Method | Can | Square | Can + Square |
|---|---:|---:|---:|
| `router_init_random` | 110/150 = 73.3% | 42/150 = 28.0% | 152/300 = 50.7% |
| `router_init_phase` | 103/150 = 68.7% | 47/150 = 31.3% | 150/300 = 50.0% |
| `router_init_topology` | 114/150 = 76.0% | 40/150 = 26.7% | 154/300 = 51.3% |
| `representation_bc` | 101/150 = 67.3% | 48/150 = 32.0% | 149/300 = 49.7% |
| `routing_softmax_top1` | 110/150 = 73.3% | 53/150 = 35.3% | 163/300 = 54.3% |

### Per-seed results

| Method | Can seed 42 | Can seed 43 | Can seed 44 | Square seed 42 | Square seed 43 | Square seed 44 |
|---|---:|---:|---:|---:|---:|---:|
| `router_init_random` | 34/50 (68%) | 35/50 (70%) | 41/50 (82%) | 7/50 (14%) | 20/50 (40%) | 15/50 (30%) |
| `router_init_phase` | 34/50 (68%) | 37/50 (74%) | 32/50 (64%) | 12/50 (24%) | 18/50 (36%) | 17/50 (34%) |
| `router_init_topology` | 33/50 (66%) | 41/50 (82%) | 40/50 (80%) | 11/50 (22%) | 14/50 (28%) | 15/50 (30%) |
| `representation_bc` | 34/50 (68%) | 37/50 (74%) | 30/50 (60%) | 21/50 (42%) | 13/50 (26%) | 14/50 (28%) |
| `routing_softmax_top1` | 33/50 (66%) | 37/50 (74%) | 40/50 (80%) | 16/50 (32%) | 19/50 (38%) | 18/50 (36%) |

The matched-seed topology differences are:

| Comparison: topology minus comparator | Can seed deltas | Square seed deltas | Interpretation |
|---|---:|---:|---|
| Random initialization | -2, +12, -2 percentage points | +8, -12, 0 points | No consistent advantage over random |
| Phase initialization | -2, +8, +16 points | -2, -8, -4 points | Can advantage; Square disadvantage |
| Softmax-router control | 0, +8, 0 points | -10, -10, -6 points | Softmax is consistently better on Square |
| Representation-BC control | -2, +8, +20 points | -20, +2, +2 points | Task-dependent representation effect |

Only three training seeds are available. The stored evaluation files provide
aggregate success counts, not per-episode paired outcomes. Accordingly, this
document does not claim statistical significance for the observed differences.
The per-seed pattern is more informative than the pooled Can+Square number.

## 5. Routing diagnostics

The training summaries provide evidence that initialization changes the
internal routing organization, even when it does not improve task success.
The following values are means across the three Stage 2 seeds.

| Method | Can phase-expert NMI | Square phase-expert NMI | Can switch rate | Square switch rate |
|---|---:|---:|---:|---:|
| `router_init_random` | 0.07 | 0.09 | 0.11 | 0.10 |
| `router_init_phase` | 0.08 | 0.09 | 0.11 | 0.10 |
| `router_init_topology` | 0.67 | 0.51 | 0.04 | 0.07 |
| `representation_bc` | 0.41 | 0.47 | 0.06 | 0.06 |
| `routing_softmax_top1` | 0.72 | 0.61 | 0.04 | 0.05 |

All methods reported a zero top-1 collapse rate. Thus, the topology arm did
not fail because of complete expert collapse. Its Square performance loss
occurs despite more structured routing.

The task-level failure totals also differ substantially: across all five
methods and three seeds, Can had 212 timeouts out of 750 episodes, while
Square had 520 timeouts out of 750. No other failure category was recorded.
This confirms that Square is the harder and more discriminating task in this
run; it should not be treated as a saturated task like Lift.

## 6. Interpretation for the proposed method

The broad claim that topology initialization improves performance is not
supported. It is contradicted by the Square results and by the fact that
topology initialization does not consistently beat random initialization on
matched seeds.

The narrower claim is supported:

> Topology-derived prototype initialization imposes a more phase-aligned and
> temporally stable routing organization in a memoryless hard prototype MoE.

The performance claim must be conditional:

> That routing organization improves Can performance in this experiment, but
> does not improve Square performance under the same beta-zero, margin-disabled
> training contract.

The current data also do not support making hard top-1 prototype routing the
default package. The softmax-router control achieves 54.3% pooled success
across Can and Square versus 51.3% for topology-initialized hard routing, and
the softmax control wins all three Square seeds.

## 7. Important comparability limitation

The earlier full `precision_residual_phaseforge` experiment is not a clean
direct causal comparison with this ablation. Its resolved configuration had
the Stage 2 margin loss enabled with `lambda_margin: 0.05`, whereas every arm
in this focused ablation has the margin loss disabled. The earlier and current
results do use the same task-specific reset banks, so the reset bank is not a
reason to dismiss that comparison. However, the earlier runs were not created
as the explicitly matched margin-attribution experiment defined below.

Consequently, the earlier full-method success rates cannot be used by
themselves to claim that topology initialization, rather than the margin
objective, caused the difference.

There is a second technical reason to avoid reactivating the old margin setup
without correction: topology prototype IDs are derived from `phase_topo` and
are not guaranteed to have the same semantic ID ordering as `phase`. A margin
loss using the wrong label vocabulary would confound the initialization test.

## 8. Recommended next experiment

Run one objective-attribution experiment before presenting the complete method
as a final result:

1. Reuse the existing `router_init_topology` / margin-disabled arm as the
   reference.
2. Add a matched topology-initialized arm with the margin loss enabled at the
   locked `lambda_margin: 0.05`.
3. Set the margin label field to `phase_topo` for topology-initialized
   prototypes, so the target labels use the same six-class vocabulary as the
   prototype construction.
4. Keep task, seeds, Stage 1 provider, `beta=0`, expert initialization,
   optimizer settings, and reset bank fixed.
5. Run only Can and Square with seeds 42, 43, and 44.

This is the minimum additional experiment that can separate the effect of
topology initialization from the effect of the margin objective. If the
margin-enabled topology arm recovers Square performance, the margin/objective
component—not topology initialization alone—must receive the credit. If it
does not, the proposed method should be reported as a routing-structure result
with task-dependent performance, not as a generally superior policy.

If the research question is limited strictly to router initialization, no
additional performance claim should be added: the present ablation already
answers that question, and the answer is mixed rather than universally
positive.

## 9. Reproducibility action required

The copied result directory contains complete metadata, evaluation summaries,
configuration files, and artifact manifests. However, the actual
`checkpoints/checkpoint_best.pt` files are absent from the local copy even
though the manifests mark them as present. The recorded metrics can be
reviewed, but the weights cannot currently be independently reloaded from
this archive.

Before sending the results as a final artifact, recover or re-export the 42
training checkpoints from the cloud result workspace, or preserve the original
cloud workspace containing them. This is an archival/reproducibility issue;
it does not change the recorded rollout counts.

## 10. Evidence locations

- Protocol: `experiments/router_initialization_ablation.json`
- Protocol explanation: `docs/plan/router_initialization_ablation.md`
- Runner ledger: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/_ledger/`
- Runner state: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/_runner/state.json`
- Evaluation index: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/_results/results.jsonl`
- Per-evaluation metrics: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/eval/`
- Per-training diagnostics: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/*/stage2/`
