# MoE Routing Organization and Closed-Loop Success

**To:** Research Director / Principal Investigator
**From:** PhaseForge Project Team
**Date:** September 15, 2026  
**Status:** Post-audit evidence review for supervisor discussion
**Primary focused run:** `outputs_router_ablation_can_square`
**Focused-run code revision:** `f83d096`
**Earlier full-matrix run revision:** `e948b73`

## Executive conclusion

The present evidence does **not** support claiming that topology initialization
is a generally superior manipulation policy, a state-of-the-art method, or a
universal improvement over the controls.

The evidence does support a narrower and potentially publishable empirical
claim:

> In the evaluated memoryless, direct-action, hard top-1 MoE, topology-derived
> prototype initialization produces substantially more phase-aligned routing
> and lower within-validation-trajectory expert switching than random or rule-based
> initialization. That routing organization does not translate into a
> consistent task-success improvement: it is favorable on Can and unfavorable
> on Square.

This is a **task-dependent routing-organization/performance decoupling** result.
It is not yet evidence that topology initialization causes contact failure, nor
that it improves training-time MoE stability in the sense used by StableMoE.
Those are testable follow-up hypotheses, not completed findings.

The appropriate decision is therefore to retain the project as an empirical
study of structured routing and its limitations, while removing unsupported
causal and universal-performance language from the manuscript.

## 1. Scope and terminology

The focused ablation evaluates a memoryless direct-action configuration. The
final experiment configurations do not use the feedback-controller class. All
30 focused Stage 2 configurations have `beta: 0.0` and have the Stage 2 margin
loss disabled. The results should therefore be described as results for this
specific configuration, not as results for every PhaseForge component.

The word “stability” must be qualified:

| Term | Meaning | Status in this project |
|---|---|---|
| Training-time routing stability | The same input remains assigned to the same expert as training progresses | **Not measured by the focused ablation** |
| Expert-utilization stability | Experts avoid collapse or pathological under-use | No persistent final-validation collapse was observed; initialization collapse occurred in some arms, and fuller utilization claims require the balance statistics |
| Within-rollout temporal coherence | Adjacent observations in an evaluation trajectory produce fewer expert switches | **Measured; supported for topology initialization** |
| Closed-loop control stability | The physical system converges reliably under environment dynamics and contact | **Not established; task success is mixed** |

StableMoE uses “routing fluctuation” primarily for training-time changes in the
assignment of the same input during optimization. The current
`routing_switch_rate` metric is different: it counts adjacent top-1 expert
changes within validation trajectories. The two measurements should not be
treated as interchangeable. See [StableMoE](https://aclanthology.org/2022.acl-long.489/)
and [Switch Transformers](https://www.jmlr.org/papers/volume23/21-0998.html)
for the broader MoE stability context.

The implementation contract was also checked directly in the source. In
`phaseforge/models/components/prototype_router.py`, hard top-1 routing is the
deterministic nearest-prototype assignment (`argmin` over `torch.cdist`), and
trajectory identifiers are accepted only for call-site compatibility and are
not used as router state. In
`phaseforge/models/components/impedance_expert.py`, `beta == 0.0` returns the
warm-started base action before the residual branch is evaluated. Thus,
“memoryless” and “direct-action” are code-level descriptions of the recorded
configuration, not assumptions inferred from the success rates.

## 2. Experimental contract

The umbrella protocol manifest lists child manifests for Lift, Can, and
Square. The recorded runner plan explicitly selected only Can and Square, so
the focused result archive contains 5 method arms × 2 tasks × 3 training seeds
× 50 evaluation episodes: 30 evaluations and 1,500 evaluation episodes in
total. Lift was not selected for this run because it is saturated in the full
matrix; ToolHang and Transport are intentionally excluded from the protocol.
Running all three child manifests would be a different 108-step protocol
(18 Stage 1 provider trainings + 45 Stage 2 trainings + 45 evaluations). The
recorded Can/Square plan contains 72 completed steps (12 + 30 + 30), and its
manifest SHA-256 matches the local protocol file.

The evaluated arms were:

| Method token | Experimental role |
|---|---|
| `router_init_random` | Random prototype initialization |
| `router_init_phase` | Rule-based centroids from `phase` |
| `router_init_topology` | Topology-derived centroids from `phase_topo` |
| `representation_bc` | Plain BC latent-representation control |
| `routing_softmax_top1` | Learned softmax gating with top-1 deployment |

The primary attribution is the three-way initialization comparison among
`router_init_random`, `router_init_phase`, and `router_init_topology`. Those
arms use the same PhaseForge Stage 1 provider, beta-zero expert family,
partial-warm expert initialization, Stage 2 objective, and seeds; their
registered router initialization source is the intended varying factor.
`representation_bc` changes the Stage 1 representation provider and therefore
is a representation comparison, not a router-only control. The softmax arm is
a registered router-package comparison, not a pure one-variable gate test: it
uses a `TopKRouter` with `noise_std: 0.1` during training,
`normalize_input: true`, and `balance_coeff: 0.01`, whereas the prototype arms
use deterministic `PrototypeRouter` routing with `balance_coeff: 0.0001`.

The following matching conditions were verified from the run metadata:

- `beta: 0.0` in all 30 Stage 2 configurations.
- `train.margin.enabled: false` in all 30 Stage 2 configurations.
- Same task, seed set, Stage 1 provider where applicable, and action-loss
  contract across the ablation arms.
- All evaluations used reset seed `2026`.
- All Can methods used reset bank `310d9cfd3fa5e843`.
- All Square methods used reset bank `e16288589f5f69c2`.
- The final validation summaries report `val/top1_collapse_rate: 0.0` for all
  30 Stage 2 runs.
- All 30 evaluations contain 50 valid episodes, zero policy failures, and zero
  invalid attempts; all 30 exported traces are present and parseable.
- The evaluation index contains 30 unique method–task–seed records, each
  matching exactly one rollout summary.
- Present artifacts matched their declared hashes. The 42 declared training
  checkpoint paths are the exception: the checkpoint files are absent from
  the copied archive.
- The rollout result rows have `eval/action_mse: NaN`; no action-MSE claim is
  made in this report. Rollout success is the primary outcome.

This last statement is not true of initialization. The initialization
diagnostics report nonzero `t0_collapse_rate` for the random arm on both tasks
and for the phase arm on some Square seeds. The observed ranges across seeds
are:

| Arm | Can initialization collapse range | Square initialization collapse range | Final validation collapse |
|---|---:|---:|---:|
| `router_init_random` | 0.33–0.50 | 0.17–0.50 | 0.0 for all seeds |
| `router_init_phase` | 0.0 | 0.0–0.33 | 0.0 for all seeds |
| `router_init_topology` | 0.0 | 0.0 | 0.0 for all seeds |
| `representation_bc` | 0.0 | 0.0 | 0.0 for all seeds |
| `routing_softmax_top1` | 0.0 | 0.0 | 0.0 for all seeds |

The defensible statement is therefore “no persistent final-validation
collapse was observed,” not “no collapse occurred.”

The reset bank is matched between methods within each task, but it is not the
same bank across Can and Square. The checkpoint binaries are absent from the
copied result archive even though artifact manifests say they are present.
The recorded metrics can be audited, but the archive is not independently
reloadable until the 42 training checkpoints are recovered or re-exported.

## 3. Rollout results

The percentages below are pooled over the three training seeds. The training
seed, not the 50 episodes generated by one checkpoint, is the primary
independent experimental unit. With only three training seeds and no paired
per-episode outcomes in the stored summaries, these results should be reported
descriptively rather than as statistically significant effects.

| Method | Can | Square | Can + Square |
|---|---:|---:|---:|
| `router_init_random` | 110/150 = 73.3% | 42/150 = 28.0% | 152/300 = 50.7% |
| `router_init_phase` | 103/150 = 68.7% | 47/150 = 31.3% | 150/300 = 50.0% |
| `router_init_topology` | 114/150 = 76.0% | 40/150 = 26.7% | 154/300 = 51.3% |
| `representation_bc` | 101/150 = 67.3% | 48/150 = 32.0% | 149/300 = 49.7% |
| `routing_softmax_top1` | 110/150 = 73.3% | 53/150 = 35.3% | **163/300 = 54.3%** |

The underlying per-seed success counts are:

| Method | Can s42 / s43 / s44 | Square s42 / s43 / s44 |
|---|---:|---:|
| `router_init_random` | 34 / 35 / 41 | 7 / 20 / 15 |
| `router_init_phase` | 34 / 37 / 32 | 12 / 18 / 17 |
| `router_init_topology` | 33 / 41 / 40 | 11 / 14 / 15 |
| `representation_bc` | 34 / 37 / 30 | 21 / 13 / 14 |
| `routing_softmax_top1` | 33 / 37 / 40 | 16 / 19 / 18 |

The matched-seed pattern is mixed:

- Topology initialization is not consistently better than random
  initialization on either task.
- It is the strongest of the three initialization arms on Can, but it is the
  weakest of those arms on Square.
- The softmax control wins on all three Square seeds and has the highest pooled
  result across the two tasks.
- The topology-versus-random Can difference is only 4 episodes out of 150,
  and the per-seed deltas change sign.

The failure totals across the focused run were 212 timeouts out of 750 Can
episodes and 520 timeouts out of 750 Square episodes; no other failure
category was recorded. This makes Square the more discriminating task in this
run. It should not be described as saturated.

## 4. Routing-organization results

The training summaries report the following means across the three Stage 2
seeds:

| Method | Can phase-expert NMI | Square phase-expert NMI | Can switch rate | Square switch rate |
|---|---:|---:|---:|---:|
| `router_init_random` | 0.07 | 0.09 | 0.11 | 0.10 |
| `router_init_phase` | 0.08 | 0.09 | 0.11 | 0.10 |
| `router_init_topology` | **0.67** | **0.51** | **0.04** | **0.07** |
| `representation_bc` | 0.41 | 0.47 | 0.06 | 0.06 |
| `routing_softmax_top1` | 0.72 | 0.61 | 0.04 | 0.05 |

These measurements are means of the final validation diagnostics over the
three Stage 2 seeds. They support the structural part of the hypothesis:
topology initialization changes the expert partition so that it is more aligned
with the logged phase labels and has lower within-trajectory switching than the
random and rule-based initialization arms. The softmax control shows that this
organization is not unique to topology initialization.

The correct wording is “phase-expert NMI” and “validation-trajectory switch
rate.” It
is not correct to call the NMI a direct measurement of topological correctness,
or to call the switch rate a measurement of training-time routing fluctuation.
The validation switch metric and the independent rollout-trace switch audit in
Section 5 use different data and aggregations; their numerical values should
not be compared as if they were the same statistic.

### 4.1 Supplementary CPU diagnosis: phase observability is a plausible bottleneck

I ran the repository's trajectory-grouped, state-only observability audit and
the held-out phase-mean action diagnostic on the local Can and Square caches.
This is supplementary evidence, not a replacement for the focused cloud run:
the raw HDF5 files are not present locally, and the local caches are not
asserted to be byte-identical to the archived cloud caches. The Can state,
action, and demo-key order was verified identical between its phase and
topology-label caches. No local Square `phase_topo` cache was available, so no
Square topology-label conclusion is reported.

| Task | State-only phase macro-F1 | Held-out action-MSE reduction from phase means | All-data action variance explained by phase | State-only `phase_topo` macro-F1 |
|---|---:|---:|---:|---:|
| Can | 0.502 | 9.4% | 13.2% | 0.950 |
| Square | 0.409 | 12.7% | 14.7% | Not available |

Both rule-based phase labels fail the repository's 0.60 observability gate,
and Square is less inferable from the instantaneous state than Can. This makes
“phase/state ambiguity contributes to the Square degradation” a credible
hypothesis. It does not establish that phase learning is the sole cause.
The phase labels still explain some action variation on both tasks, and the
Square value is slightly higher. Most action variation remains within phase,
so a phase-organized router cannot by itself resolve the fine-grained action
differences required for precise contact control.

The Can cache also shows that `phase` and `phase_topo` are materially different
partitions: their normalized mutual information is 0.058 and adjusted Rand
index is 0.021. In the archived focused configurations, Stage 1 SupCon uses
`phase`, while the topology arm initializes prototypes from `phase_topo`.
That is a real representation/initialization mismatch, not an interchangeable
label naming detail. The corresponding Square agreement cannot be estimated
from the local cache and must not be invented.

There is a second implementation fact that changes how the old logs must be
read. The focused Stage 1 configurations enable SupCon but do not explicitly
set `train.supcon.zero_ce`. The trainer defaults that field to `true`, so
`_effective_lambda_phase()` returns zero; all six PhaseForge Stage 1 logs
(Can/Square × three seeds) have `loss_phase: 0.0`. Their logged phase accuracy
values are therefore diagnostics from a retained phase head, not evidence that
the phase-classification CE objective trained that head. The latent is shaped
by SupCon, but the phase-head accuracy must not be used as a measure of the
learned representation under this contract.

The correct causal framing is consequently:

> Square may be disadvantaged by poorer instantaneous phase observability,
> label-partition mismatch, and/or loss of within-phase action detail. These
> are competing explanations. The current evidence does not justify the
> stronger statement that rollout success is directly proportional to phase
> classification accuracy.

## 5. Action-jump audit: corrected status

The report previously presented topology trace numbers that do not match the
exported `trace.jsonl` files. The values below are the corrected pooled
recomputation over adjacent timesteps within episodes. For each adjacent pair,
the action jump is the L2 norm of the difference between `final_action` values;
the rows partition pairs by whether the selected top-1 expert changed.

| Method | Task | Switch pairs / adjacent pairs | Mean jump at switch | Mean jump at non-switch | Ratio |
|---|---|---:|---:|---:|---:|
| `router_init_topology` | Can | 817 / 33,547 (2.44%) | 0.2154 | 0.0438 | 4.92× |
| `router_init_topology` | Square | 2,276 / 62,646 (3.63%) | 0.1903 | 0.0229 | 8.29× |
| `router_init_phase` | Can | 1,948 / 38,314 (5.08%) | 0.2393 | 0.0378 | 6.33× |
| `router_init_phase` | Square | 2,320 / 61,580 (3.77%) | 0.1110 | 0.0220 | 5.03× |
| `router_init_random` | Can | 1,636 / 34,973 (4.68%) | 0.2062 | 0.0418 | 4.94× |
| `router_init_random` | Square | 2,387 / 62,998 (3.79%) | 0.1364 | 0.0237 | 5.76× |
| `routing_softmax_top1` | Can | 1,082 / 34,693 (3.12%) | 0.3072 | 0.0459 | 6.69× |
| `routing_softmax_top1` | Square | 1,467 / 60,323 (2.43%) | 0.2147 | 0.0236 | 9.09× |

The table is a pooled recomputation from the 30 exported traces (24 traces
contain a selected expert; the six plain-encoder traces have
`selected_expert: null`). The plain-encoder control is therefore correctly
excluded from expert-switch calculations. The random and softmax rows are consistent with
the earlier audit. The earlier topology rows are not: the earlier report gave
3.85% and 3.70% with different means and ratios. Those earlier topology values
must not be used.

The corrected audit establishes an association between expert switches and
larger action jumps. It does **not** establish any of the following:

- that the switch causes a timeout;
- that the jump occurs during insertion or contact;
- that it causes wedging, jamming, or a limit cycle;
- that the jump is uniquely responsible for the topology arm’s Square result;
- that the effect is catastrophic in a control-theoretic sense.

In fact, the softmax Square control has the largest corrected action-jump ratio
in this table while achieving higher Square success than topology
initialization. Therefore, a simple “larger switch ratio causes lower success”
explanation is contradicted by the current data. Action discontinuity remains a
plausible mechanism to investigate, not a completed mechanistic discovery.

## 6. Full-matrix context

The full-matrix manifest contains 50 task-specific method rows: 10 method
identities instantiated once for each of 5 tasks. With 3 seeds, this defines
150 method–task–seed cells. The local archive contains 135 rollout summaries
for nine rollout-capable identities and 15 oracle `eval_results.json` files
that are offline-only; the oracle cells do not have rollout summaries.
The table below reports selected rollout-capable methods, with 150 episodes
per method/task cell:

| Task | BC | Plain encoder | Softmax top-1 | Phase-random | PhaseForge |
|---|---:|---:|---:|---:|---:|
| Lift | 150/150 (100%) | 150/150 (100%) | 150/150 (100%) | 150/150 (100%) | 150/150 (100%) |
| Can | 94/150 (62.7%) | 54/150 (36.0%) | 103/150 (68.7%) | 93/150 (62.0%) | 117/150 (78.0%) |
| Square | 51/150 (34.0%) | **75/150 (50.0%)** | 58/150 (38.7%) | 58/150 (38.7%) | 61/150 (40.7%) |
| ToolHang | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) |
| Transport | 0/150 (0.0%) | **1/150 (0.7%)** | 0/150 (0.0%) | 0/150 (0.0%) | **1/150 (0.7%)** |

The full matrix provides context, not a clean attribution of topology
initialization. The earlier full `precision_residual_phaseforge` runs used
the Stage 2 margin loss with `lambda_margin: 0.05`, while all focused arms
disabled that loss. The old result therefore cannot isolate topology
initialization from the objective change. The full matrix also does not show a
universal performance advantage. The local full-matrix copy, like the focused
copy, contains metadata and hashes but no actual checkpoint binaries.

## 7. What the current evidence permits us to claim

### Supported claims

1. Topology-derived prototypes produce more phase-aligned expert assignments
   than random or rule-based prototypes under the evaluated contract.
2. They produce lower within-validation-trajectory top-1 switching than the
   random and rule-based initialization arms in the reported diagnostics.
3. Structural routing organization and rollout success are decoupled in this
   experiment: topology is favorable on Can but unfavorable on Square.
4. Hard top-1 prototype routing is not established as the best-performing
   package; the softmax control is stronger on the pooled focused result and on
   every Square seed.

### Claims that must be removed or labeled as hypotheses

- “Topology initialization solves MoE temporal/training instability.”
- “The exact physical mechanism has been identified.”
- “Switches cause contact jamming” or “cause catastrophic failure.”
- “Over 80% of switches occur at phase boundaries.” No boundary-partition
  analysis has been completed.
- “Topology prototypes enforce a wide margin and lower perturbation flips.” No
  perturbation experiment has been completed.
- “The data are fully reproducible.” The copied focused and full-matrix
  archives contain metadata and hashes but no checkpoint binaries.
- “All ToolHang and Transport models achieve zero.” Transport has one success
  for the plain and PhaseForge entries in the full matrix.
- “980+ tests” as a repository fact. The current collection check reported 875
  tests.
- Statistical significance or “statistically indistinguishable” claims based
  only on three training seeds and aggregate counts.

The balance coefficients shown in configurations (`0.0001` and `0.01`) are
hyperparameters, not measured evidence of utilization balance. Report the
actual utilization statistic if that claim is needed.

## 8. Recommended paper framing

The defensible working title is:

> **Decoupling Routing Organization from Closed-Loop Success: An Empirical
> Study of Structured Mixture-of-Experts in Robotic Manipulation**

The central thesis should be:

> We find that topology-derived prototype initialization produces more
> phase-consistent and temporally coherent routing in a beta-zero, memoryless
> hard-routing MoE. This structural organization does not consistently improve
> rollout success: it helps Can but hurts Square. The results therefore support
> a task-dependent decoupling between routing organization and task success.
> Whether switch-associated action discontinuities are causal for contact
> failure remains an open hypothesis requiring targeted temporal and
> perturbation analyses.

This framing does not require the method to achieve state-of-the-art success.
The contribution is an empirical finding about an architectural property and
its limitation. The paper must make the scope, negative result, and unresolved
causal mechanism explicit.

## 9. Required follow-up work

The minimum work needed before making a causal stability claim is:

1. **Boundary partitioning.** Use the same `phase_topo` changepoints and a
   pre-registered window to measure the fraction of switches near boundaries
   versus within phases. Do not state an expected “over 80%” result before it
   is measured.
2. **Switch/contact temporal alignment.** Compare switch events and action
   jumps with contact state, insertion state, distance-to-goal, and timeout
   proximity. This is needed to test, rather than assume, the contact-failure
   mechanism.
3. **Observation perturbation test.** Inject calibrated perturbations and
   report top-1 flip probability as a function of noise scale, with identical
   states and seeds across arms. Treat this as router sensitivity, not as a
   complete control-stability proof.
4. **Training-time routing test.** If the manuscript uses the StableMoE term
   “training-time routing stability,” hold fixed a set of observations and
   measure assignment changes across epochs/checkpoints. The current rollout
   switch metric cannot substitute for this test.
5. **Phase-loss attribution on CPU.** Run a small, explicitly logged Stage 1
   screening matrix on Can and Square with the same cache, split, optimizer,
   and seeds: (a) SupCon with CE suppressed, matching the archived contract;
   (b) SupCon with `train.supcon.zero_ce: false`; and (c) CE-only with SupCon
   disabled. Compare held-out phase macro-F1/balanced accuracy, state-only
   probe performance, action MSE, and latent phase clustering. This diagnoses
   whether the phase head, SupCon, or neither is responsible before using GPU
   budget for a new rollout matrix. It is a screening experiment, not a new
   performance claim.
6. **Objective attribution.** If the complete PhaseForge package remains in
   scope, run the matched topology-initialized margin-on arm with the margin
   label field set to `phase_topo`, `lambda_margin: 0.05`, and all other
   settings fixed. Otherwise, report the current focused ablation as the
   router-initialization study and do not attribute the old full-matrix result
   to topology alone.
7. **Archive the weights.** Recover or re-export the focused run’s 42 training
   checkpoints, and preserve the full-matrix weights if those results remain
   in the paper. Record all recovered-file checksums with the result archive.

## 10. Final decision

Proceed with the project as a **routing-organization and task-dependent
decoupling study**. Do not present it yet as a generally superior policy, a
proof of closed-loop control stability, or a completed causal explanation of
Square failure.

The next scientifically valuable result is not another broad benchmark run.
It is a matched, logged analysis that determines whether organized routing is
aligned with phase boundaries, whether switches coincide with contact failure,
and whether the router is stable under controlled observation perturbations.

## Evidence locations

- Focused protocol: `experiments/router_initialization_ablation.json`
- Protocol explanation: `docs/plan/router_initialization_ablation.md`
- Focused runner ledger: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/_ledger/`
- Focused runner state: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/_runner/state.json`
- Focused evaluation index: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/_results/results.jsonl`
- Focused evaluation traces and metrics: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square/eval/`
- Routing metric implementation: `phaseforge/evaluations/metrics/routing_stability.py`
- CPU phase/router diagnosis: `scripts/analysis/phase_router_diagnosis.py`
- CPU diagnosis output: `outputs_cpu_debug/phase_router_diagnosis.json`

## References

- Shen et al., “StableMoE: Stable Routing Strategy for Mixture of Experts,” ACL 2022: [ACL Anthology](https://aclanthology.org/2022.acl-long.489/).
- Fedus et al., “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” JMLR 2022: [JMLR](https://www.jmlr.org/papers/volume23/21-0998.html).
