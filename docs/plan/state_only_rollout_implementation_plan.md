# PhaseForge — State-Only Rollout Implementation Plan

**Status:** implementation plan; no code or experiments are changed by this document

**Scope:** non-visual, structured low-dimensional robot state only

**Supersedes:** the required/optional classification for the next implementation
step. The existing offline pilot remains valid as an offline diagnostic.

## 1. Correct claim boundary

The immediate paper claim is:

> In pinned robomimic/robosuite low-dimensional tasks, phase-informed MoE
> router initialization is evaluated using structured state, offline routing
> diagnostics, and closed-loop simulator success.

The policy receives no RGB image, image embedding, language input, or VLA
feature. Its input is the same declared structured state used by the offline
pipeline: robot state plus task-relevant object state, with the current
single-step history contract.

The offline action and routing metrics remain important diagnostics. They are
not task success. A closed-loop rollout is required only for a claim about
manipulation behavior; it is not required because the project lacks vision.

## 2. What is required now

The required next experiment is a state-only rollout study on all five
robomimic manipulation tasks: Lift, Can, Square, Tool Hang, and Transport.
Use Lift first as an engineering smoke test because it is the simplest task to
debug. Lift-only results are not the final cross-task evaluation.

1. pin a dataset/environment-compatible robosuite release;
2. implement the shared state-only simulator adapter;
3. validate the adapter on Lift with a scripted or state-oracle controller;
4. run the same validation gates for every task;
5. evaluate the matched model matrix separately on all five tasks in closed loop;
6. persist per-episode results and uncertainty summaries;
7. interpret per-task and aggregate rollout success together with the existing
   offline diagnostics.

The core Lift matrix is:

- structured-state BC-MLP;
- structured-state BC-RNN or an equivalent declared history baseline;
- Scratch MoE;
- Warm-Start MoE;
- Phase-Pretrain Random-Router;
- Plain-Encoder Phase-Bootstrap;
- PhaseForge;
- Teacher-Forced and Ground-Truth Routing only as explicitly labeled
  privileged diagnostics;
- robot-only BC only as a separate negative control.

Use the existing training seeds 42, 43, and 44 for the first complete
five-task matrix.
This three-seed result is descriptive. If the compute budget permits stronger
inferential evidence, add seeds 45 and 46 before producing the final paper
table; do not silently mix pilot and final results. Use the same frozen rollout
cases, horizon, action contract, checkpoint rule, and state schema for every
model and seed.

## 3. What is not required for this paper version

These are not automatic gates for the five-task, single-task-per-environment
state-policy paper:

- RGB observations or vision models;
- LIBERO or a VLA benchmark;
- a multitask policy;
- an official Diffusion Policy reproduction.

The five tasks are evaluated as separate single-task policies. This is not a
multitask-training claim. A shared multitask policy is a later experiment with
its own task-conditioning protocol.

## 4. Implementation work

### 4.1 Freeze simulator and dataset parity first

The current low-dimensional artifact is from the robomimic v1.5 track. The
repository's current rollout extra pins `robosuite==1.4.0`, so rollout work must
not begin with that environment unchanged.

Choose one complete compatible pair and record it in the protocol and run
provenance. The preferred path is the current v1.5.1-compatible robomimic
artifact together with its matching robosuite/MuJoCo environment. The older
`offline_study` or v1.4.1 track is valid only if the matching dataset and
environment are deliberately selected together. The robomimic documentation
warns that these tracks may not reproduce one another.

Record:

- dataset artifact revision and SHA-256;
- robomimic revision;
- robosuite version and revision;
- MuJoCo version;
- environment name and serialized `env_args`;
- state-key order and dimensions;
- action convention, range, and gripper convention.

If the versions do not match the dataset metadata, stop before training or
rollout evaluation.

### 4.2 Add the robosuite state-only adapter

Add the adapter under `phaseforge/evaluations/envs/` with a small explicit
interface:

```text
make(task, pinned_env_metadata) -> environment
reset(reset_case) -> structured_state
step(normalized_action) -> structured_state, done, success, info
close()
```

The adapter must:

- disable image observations and rendering;
- construct the task from the pinned dataset environment metadata;
- extract exactly the declared low-dimensional keys in the declared order;
- apply the training normalizer without refitting it;
- assert the expected state dimension before the first action;
- convert model output using the frozen action contract;
- enforce action range and finite-value checks;
- expose the environment's task success predicate and termination reason;
- use the configured task horizon;
- fail closed on schema, action, version, or checkpoint mismatches.

Do not implement task success as an action-MSE threshold, distance proxy, or
offline agreement metric. The task's simulator success predicate is the
behavioral outcome.

### 4.3 Freeze reset cases correctly

Do not rely on an undocumented arbitrary random seed bank. Before final
training-checkpoint selection, generate or obtain a fixed set of 50 valid
evaluation reset cases **for each task** from that task's pinned reset
distribution. Store the serialized initial-state payload, reset seed, task,
and environment hash.

The cases must be disjoint from training demonstrations, validation data,
checkpoint selection, and phase-label calibration. Every model and seed must
run the identical cases in identical order. A seed may be recorded for
reproducibility, but the serialized reset state is the authoritative paired
evaluation input when the environment supports it.

### 4.4 Implement the rollout runner

Add a rollout evaluator and wire `eval.mode=rollout` in `phaseforge/cli.py`.
The runner must:

- load a final checkpoint and its normalizer/provenance;
- run deterministic inference under the single-step state contract;
- execute each frozen reset case until success or the fixed horizon;
- record one validated row per attempted episode;
- hash and record the evaluated checkpoint;
- distinguish valid task failures from infrastructure failures, invalid
  actions, exceptions, and simulator errors;
- exclude only infrastructure failures from the success denominator;
  policy-generated NaNs, invalid actions, or safety violations are reported
  as policy failures under a strict metric (labeled separately), never
  silently removed;
- preserve enough information to rerun or pair every case.

The existing episode schema and rollout summary utilities may be reused. The
missing piece is the simulator adapter and the runner that produces real
`episodes.jsonl` records.

### 4.5 Validate before any full sweep

Run the following gates in order:

1. environment construction and state-schema self-test;
2. reset-and-replay of one recorded demonstration action sequence;
3. action range, normalization, and gripper-convention test;
4. scripted/state-oracle controller on the frozen reset cases for each task;
5. random/no-op controller sanity check;
6. one checkpoint through 10 smoke episodes.

The scripted controller must solve the supported reset cases for each task. If it fails,
repair the environment, reset cases, action mapping, or success predicate
before evaluating learned policies. The 10-episode run is a smoke test only,
never a final result.

### 4.6 Evaluate the matched five-task matrix

After the gates pass, run 50 episodes per task/reset bank for each of the three
training seeds and each required model. Keep the following fixed within and
across tasks:

- state schema and normalization;
- action convention and horizon;
- reset cases and order;
- checkpoint selection rule (predeclared `best val/loss_action`; the selected
  checkpoint is evaluated on the separate frozen evaluation reset bank);
- deterministic inference rule;
- simulator and MuJoCo versions;
- training budget and model-specific configuration;
- provenance and artifact hashing.

The final report must separate:

- rollout success rate and uncertainty;
- offline action reproduction;
- routing alignment, balance, entropy, and collapse;
- privileged diagnostic rows;
- robot-only negative-control rows.

## 5. Final statistical report

For every task, report:

- every episode's success, validity, reset case, seed, horizon outcome, and
  failure category;
- success rate for each model and training seed;
- a 95% Wilson interval over valid episodes for each task and seed;
- mean and sample standard deviation across the three training seeds for each
  task;
- the unweighted macro-average of the five task-level success rates as a
  secondary aggregate; never replace the per-task table with the macro-average;
- paired PhaseForge-minus-baseline differences using identical reset cases, per
  training seed (exact McNemar + Newcombe 95% CI; no pooling across seeds);
  every seed's paired difference and its direction consistency are reported,
  and the final claim is descriptive — three seeds cannot support
  population-level significance;
- a declared multiplicity correction for the five primary matched comparisons
  (PhaseForge versus BC-MLP, Scratch MoE, Warm-Start MoE,
  Phase-Pretrain Random-Router, and Plain-Encoder Phase-Bootstrap). Privileged
  routing and robot-only negative-control rows are descriptive, not part of the
  primary comparison family;
- infrastructure failures separately (excluded from the success denominator);
  policy-caused invalid actions/NaNs/safety violations counted as failures
  under a strict metric and labeled separately;
- the offline routing/action metrics as secondary diagnostics.

Do not call a routing improvement a manipulation improvement. Interpret the
outcome using this decision table:

| Rollout result | Correct conclusion |
|---|---|
| Scripted controller fails | evaluator/environment contract is not validated |
| Scripted controller succeeds, BC fails | state/action/temporal/training issue; do not judge MoE yet |
| BC succeeds, PhaseForge and controls match | routing hypothesis is not supported behaviorally |
| PhaseForge improves routing only | mechanism-level offline result; no behavior claim |
| PhaseForge improves rollout success consistently across the five tasks with matched controls | behavioral evidence for the five-task state-only claim |

## 6. Optional extensions after the core result

Add these only if the paper claim is intentionally broadened:

### Single-task history control

Use the same declared history contract for BC-RNN and the PhaseForge policy
matrix. The history baseline is required for the five-task benchmark claim;
it is not evidence that PhaseForge itself is history-dependent unless the
PhaseForge model consumes the same history.

### Multitask extension

Train a shared task-conditioned policy only after the single-task state-only
pipeline is validated. A task ID must be consumed by the model; merely storing
it in the dataset is not task conditioning.

### Checkpoint-selection ablation

Keep the predeclared `best val/loss_action` rule for the first study. As a
later ablation, compare it against two-bank rollout-based selection
(`save_on_best_rollout_success_rate`-style, with a selection bank disjoint from
the evaluation bank); robomimic Lesson 3 documents a 50–100% gap for val-loss
selection in their setting, so the trade-off for ours must be measured, not
assumed.

## 7. Publication boundary

The current offline Lift report can support an offline mechanism/diagnostic
report if it is presented with that narrow claim.

The five-task state-only rollout implementation described here is required
before claiming that PhaseForge improves manipulation behavior across the
project's task family. It does not require images, a vision encoder, LIBERO, or
comparison against VLA models.

No implementation, training run, or rollout has been performed as part of
writing this plan.

## 8. Literature basis

- The robomimic study evaluates low-dimensional and image observations,
  includes BC and BC-RNN, and documents that the training objective can differ
  from the rollout evaluation objective: [robomimic study](https://robomimic.github.io/study/).
- The robomimic documentation provides the low-dimensional task artifacts and
  warns that the v1.5.1, older `offline_study`, and v1.4.1 tracks are not
  interchangeable: [dataset documentation](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html).
- Diffusion Policy exposes separate state-based resources and includes the
  robomimic Lift, Can, Square, Tool Hang, and Transport simulation tasks:
  [official project](https://diffusion-policy.cs.columbia.edu/).
- State-only imitation has been published as a distinct setting:
  [State-Only Imitation Learning for Dexterous Manipulation](https://arxiv.org/abs/2004.04650).
