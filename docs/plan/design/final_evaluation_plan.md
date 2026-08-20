# PhaseForge — Final State-Based Evaluation and Research Plan

**Status:** **SUPERSEDED** — [state_only_rollout_implementation_plan.md](../specs/state_only_rollout_implementation_plan.md) is now the authoritative protocol for the current five-task state-only paper. This file is retained for reference; its earlier required/optional classifications must not override the authoritative plan.

**Research definition:** [research_definition.md](../specs/research_definition.md)

**Research area:** non-visual robot manipulation from structured low-dimensional observations

**Primary benchmark family:** robomimic datasets with robosuite simulation

**Excluded from this project:** RGB observations, vision encoders, language models, vision-language-action models, and foundation-model leaderboard comparisons

---

## 1. Research question

PhaseForge studies one narrow question:

> Does initializing a mixture-of-experts router from phase structure in a pretrained low-dimensional policy improve manipulation behavior and expert specialization compared with ordinary behavioral cloning, a randomly initialized MoE, and a generic warm-started MoE?

This is a training-strategy study. It is not a perception study, a vision benchmark, or a foundation-model study.

The central comparison is:

```text
PhaseForge vs. Scratch MoE vs. Warm-Start MoE
```

Behavioral cloning is the basic control floor. A teacher-forced phase-routing model is a diagnostic reference, not the deployable method.

The main scientific claim is supported only if PhaseForge improves rollout behavior and the improvement is consistent with better phase–expert specialization. A routing metric without a task-success improvement is not sufficient evidence.

---

## 2. Literature-grounded design decisions

### 2.1 “State-based” does not mean robot joints alone

The accepted low-dimensional manipulation setting uses task-relevant structured observations. Depending on the task, these may include end-effector pose, gripper state, object pose, goal information, and a short history of observations.

The relevant published precedents are:

- Diffusion Policy provides a separate state-based benchmark track, with
  low-dimensional experiment artifacts and the same robomimic task family.
  [Official project](https://diffusion-policy.cs.columbia.edu/)
- robomimic studies low-dimensional observations, image observations, BC, BC-RNN, and hierarchical BC on simulated and real manipulation tasks. It reports that history-dependent policies are highly effective and that validation loss is not a reliable substitute for rollout evaluation. [Study](https://robomimic.github.io/study/)
- robomimic extracts low-dimensional observations from simulator states and evaluates task completion with simulator success termination. [Dataset protocol](https://robomimic.github.io/docs/v0.4/datasets/robosuite.html)
- Meta-World documents structured state containing robot, object, and goal information rather than bare robot proprioception. [State-space specification](https://metaworld.farama.org/benchmark/state_space/)
- CIMER’s state-only setting models both robot-hand and object motion, then refines the behavior through emulation. [ICRA 2025 paper](https://star-lab.cc.gatech.edu/papers/han-cimer-icra/)

Therefore, PhaseForge must not define its main policy as a 23-D robot-only controller. That representation is retained only as a deliberately limited negative-control baseline.

### 2.2 Primary benchmark

Use the established robomimic/robosuite low-dimensional manipulation setting with these five simulated tasks:

- Lift
- Can
- Square
- Tool Hang
- Transport

These tasks provide a useful range of pick-and-place, insertion, contact, and multi-stage behavior while remaining within the same simulator and data ecosystem.

The primary results must use the benchmark’s low-dimensional observation extraction and simulator reset/evaluation conventions. Do not silently substitute a project-specific state representation and call it the published benchmark protocol.

### 2.3 No cross-benchmark fallback

This plan intentionally uses one benchmark family. No second benchmark is part of the core protocol.

Any future transfer experiment must be written as a separate protocol with its own observation schema, task split, baselines, and claims. It must not be added to the primary results table merely to increase the number of benchmarks.

### 2.4 Evaluation evidence from published state-based studies

The evaluation design follows the parts of the published low-dimensional
literature that are relevant to this project:

- [robomimic's study protocol](https://robomimic.github.io/study/) defines
  task reset distributions and evaluates policies in the
  simulator rather than treating validation loss as the task result;
  the authors explicitly report that the best validation checkpoint can be
  substantially worse in rollout performance;
- robomimic identifies history-dependent BC-RNN as a strong baseline, so a
  single-step BC model cannot be the only behavioral control;
- [Diffusion Policy's official project page](https://diffusion-policy.cs.columbia.edu/)
  reports the same five robomimic simulation tasks in its
  state-based benchmark track, which supports using Lift, Can, Square, Tool
  Hang, and Transport as a recognizable state-policy comparison set;
- published simulation evaluations commonly repeat training with at least
  three seeds and report rollout success over fixed evaluation trials; for
  example, the simulation protocol in the [PerAct supplementary
  evaluation](https://proceedings.mlr.press/v205/nasiriany23a/nasiriany23a-supp.pdf)
  evaluates three seeds and repeated rollouts. This protocol therefore treats
  training seeds and paired rollout initial states as separate sources of
  variation.

These precedents do not make PhaseForge a reproduction of Diffusion Policy or
robomimic. They define the evaluation standard: closed-loop task success is the
primary outcome, validation action loss is diagnostic, and all methods must be
compared under identical reset conditions.

---

## 3. Observation and task definition

### 3.1 Main observation

The main policy input is the benchmark’s low-dimensional structured observation:

```text
robot state
+ task-relevant object state
+ task goal information, where provided by the task interface
+ short temporal history
```

The exact observation keys and dimensions must be recorded in a versioned schema before training. At minimum, the schema must state whether it contains:

- end-effector position and orientation;
- gripper state;
- robot joint state;
- object positions and orientations;
- goal or target information;
- task identity;
- observation history.

Object and goal information is structured simulator state. It is privileged relative to a physical robot that would need perception, so the paper must call it **privileged structured state** or **low-dimensional simulator state**. It must never be described as raw proprioception.

### 3.2 Task conditioning

The policy must know which task it is solving.

There are two valid modes:

1. **Single-task mode:** train and evaluate one policy per task. No task ID is needed.
2. **Multitask mode:** train one policy over multiple tasks and provide an explicit task ID or goal embedding.

The first implementation must use single-task mode because it is the cleanest standard baseline and removes task-conditioning confounds. Multitask training is a second experiment after the single-task pipeline is validated.

Accordingly, the primary results are reported separately for each of the five tasks, with a predeclared cross-task summary only after all task-level results are complete. A pooled multitask result is a secondary experiment and must not replace the per-task results.

If multitask mode is implemented, `task_id` must be consumed by the model through a learned embedding or equivalent conditioning path. Carrying `task_id` in the data loader while ignoring it in the model is not valid task-conditioned learning.

### 3.3 Temporal context

The current single-step MLP is insufficient for a claim about long-horizon or history-dependent behavior.

All main models must receive the same fixed history window, or the project must explicitly narrow its claim to instantaneous control. The initial choice should be a short fixed window, such as 5–10 observations, with the same window and padding rules for BC, MoE, and PhaseForge.

A recurrent BC baseline is required because robomimic reports strong benefits from history-dependent policies.

### 3.4 Negative-control observation

Retain a robot-only observation condition containing proprioception without object or goal state.

Its role is diagnostic:

```text
robot-only state → information-ceiling control
```

It is not the main PhaseForge setting and must not be used to claim that the method solves general manipulation without perception.

---

## 4. Data protocol

### 4.1 Dataset handling

Use the official robomimic low-dimensional dataset artifacts where available. If observations are regenerated from raw simulator states, record:

- robosuite version;
- MuJoCo version;
- dataset revision;
- observation-extraction command and configuration;
- action convention;
- success/done convention;
- train/validation/test split files.

Do not mix observations extracted with one simulator version and rollouts executed with another without a documented parity test.

The release track must be frozen before Gate 1. The [current robomimic v0.1
dataset documentation](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html)
provides low-dimensional artifacts based on robosuite v1.5.1
and explicitly warns that the older `offline_study` and v1.4.1 dataset tracks
may not reproduce the same results. The repository's optional rollout extra
pins the v1.5.1 environment for Lift, Can, Square, and Transport; it is not
approval to mix that environment with Tool Hang, whose embedded metadata
requires v1.5.0. Before rollout work, select the matching per-task
environment and record it in `MANIFEST.json`.

### 4.2 Splits

Split demonstrations by trajectory, never by individual timesteps.

For each task:

- training trajectories are used for fitting;
- validation trajectories are used for hyperparameter selection;
- test initial states are held out from all training and checkpoint-selection decisions.

The reset distribution must be documented. The same test initial states must be used for every model and every training seed so that comparisons are paired.

### 4.3 Action and normalization contract

Freeze one action representation for the complete study. Record:

- action dimensionality;
- absolute versus delta end-effector control;
- gripper convention;
- action range expected by robosuite;
- training normalization;
- evaluation de-normalization.

The evaluator must fail loudly when action dimensions, ranges, or normalization metadata do not match the checkpoint.

### 4.4 Phase labels

Phase labels may be generated from demonstration state/action trajectories, but they are auxiliary training labels only.

There is no standard robomimic six-phase ground-truth annotation that this project
may claim to reproduce. The current six-phase rule labeler is a project-specific
heuristic. It must therefore be treated as an ablation-sensitive preprocessing
choice, and any phase-based claim requires the validation checks below.

Before the main experiment, verify:

- label distribution per task;
- minimum and median phase duration;
- transition locations;
- sensitivity to the label thresholds;
- phase-prediction accuracy from permitted observations;
- absence of deployment leakage: labels must not be supplied to the policy at
  inference, and any learned phase predictor must use only its declared
  permitted observation history. The current adaptive percentile calibration
  uses the complete demonstration, so the labels are offline annotations and
  must not be described as causal online labels.

At inference, PhaseForge must not receive the ground-truth phase label.

---

## 5. Models and baselines

Every model in a comparison must use the same observation schema, history, data split, optimizer budget, and evaluation seeds.

### Required models

1. **BC-MLP** — direct action regression from the structured low-dimensional input.
2. **BC-RNN/history BC** — temporal baseline using the same observation window as the MoE models.
3. **Scratch MoE** — randomly initialized router and experts.
4. **Warm-Start MoE** — pretrained action encoder, randomly initialized router, and warm-started experts.
5. **PhaseForge** — phase-supervised pretraining followed by phase-centroid router initialization.
6. **Teacher-Forced Phase Routing** — expert assignment uses ground-truth phases during training and a learned phase predictor at inference; this is a diagnostic reference and must be labeled as privileged-training.
7. **Robot-only BC** — proprioception-only negative control.

### Optional reference

An official low-dimensional Diffusion Policy reproduction may be added after the required models are working. It is a reference point, not a required component of the PhaseForge causal comparison.

### Fairness requirements

For the central PhaseForge comparison, hold constant:

### Reproducibility and seeding policy

Three orthogonal seed sources are used in a single training run, and they
must not be conflated:

1. **`data.split.seed`** (default `42`) — seeds the **train/val split**
   via `np.random.default_rng`. It is intentionally **independent of
   the training seed** below: a model-seed change must not re-shuffle
   the train/val boundary, which would otherwise invalidate prior val
   curves and create a hidden coupling between the seed we report and
   the data we hold out. Across the protocol seeds `[42, 43, 44]`
   every cell sees the *same* train/val trajectories.
2. **`project.seed`** (`42`, `43`, or `44`) — the **training seed**.
   Passed to `set_seed(...)` at the very top of `train()` so every
   RNG (Python `random`, `numpy`, `torch` CPU+CUDA) starts from the
   same state. It also seeds `cudnn.deterministic=True` and
   `cudnn.benchmark=False`. The train DataLoader uses an explicit CPU
   `torch.Generator` derived from this seed so the per-epoch sample
   order is reproducible from the project seed alone — it does not
   rely on the global torch RNG state at DataLoader construction
   time.
3. **`project.seed` again** — seeds model initialisation (handled by
   the same `set_seed` call). The reset bank for evaluation has its
   own dedicated seed (`10000`–`10049` per task), independent of all
   training seeds.

This three-source decomposition is enforced in code (the
`_train_sampler_generator` method builds the DataLoader generator from
`project.seed` and the `_build_task_level_splits` method reads
`data.split.seed` only) and verified by regression tests
(`tests/test_state_machine.py::TestTrainSamplerGenerator`,
`::TestSplitSeedIndependence`).
- parameter budget, as closely as practical;
- history length;
- encoder width and depth;
- optimizer and learning-rate schedule;
- number of training updates;
- normalization;
- batch construction;
- task split;
- checkpoint selection rule;
- evaluation initial states.

Only the routing initialization and phase-supervision strategy should change in the primary factorial comparison.

---

## 6. Training stages and gates

### Gate 0 — protocol freeze

Before any large run, commit a protocol file containing:

- benchmark and task list;
- observation keys and dimensions;
- task-conditioning mode;
- history window;
- action contract;
- phase-label definition;
- train/validation/test split;
- seeds;
- evaluation reset-seed bank and episode count;
- primary metrics;
- confidence-interval and aggregation rules;
- checkpoint-selection rule.

No result is included in the final comparison if it was produced under a different schema or split.

### Gate 1 — simulator and evaluator validation

Run, in order:

1. demonstration-action replay;
2. state extraction parity checks;
3. action-scale and normalization checks;
4. native success-predicate availability (task-independent probe);
5. a no-op/random-action baseline.

The simulator must answer the success-predicate question on a pinned reset
case and the action-contract / state-restore / parity probes must all pass
before training learned policies. If any of these fail, stop and repair
the environment or evaluator.

### Gate 2 — BC floor

Train BC-MLP and BC-RNN on a small pilot subset first.

Proceed only when structured-state BC produces clearly nonzero rollout success on held-out initial states and outperforms the robot-only negative control where object information is necessary.

If structured-state BC fails, investigate task conditioning, observation
extraction, temporal context, action scaling, and checkpoint loading. Do
not respond by increasing the MoE training budget.

### Gate 3 — phase validity

Proceed to PhaseForge only if:

- phases are present in every task;
- no phase is absent or overwhelmingly dominant without explanation;
- phase labels are predictable from permitted inputs;
- phase boundaries correspond to meaningful behavior changes;
- the phase-labeling code passes held-out trajectory checks.

If this gate fails, revise the phase definition before testing router bootstrapping.

### Gate 4 — controlled model matrix

Run the required models with three independent training seeds. First run one task end-to-end, then expand to all five tasks.

For the primary study, each task has its own task-specific policy and data split. The same architecture, training budget, and evaluation protocol are reused across tasks. Only after this study is complete may one shared multitask policy be trained with explicit task conditioning.

Do not rent additional compute or run the full matrix until the single-task pipeline passes Gates 1–3.

### Gate 5 — final evaluation

Use the frozen final checkpoints selected only through the validation protocol. Run the complete paired test evaluation for every required model and seed.

---

## 7. Rollout evaluation

### 7.1 Episodes

- 10 episodes per task: smoke test only; use this only to catch adapter,
  checkpoint, action-scale, or success-predicate failures.
- 50 episodes per task for each training seed: final reported evaluation.
- 3 independent training seeds: minimum final matrix, fixed as 42, 43, 44.
- Use one frozen evaluation initial-state bank per task, containing 50
  deterministic reset seeds, fixed as 10000 through 10049. Every model and
  every training seed receives the same bank in the same order. The bank must
  be disjoint from all training, validation, checkpoint-selection, and
  phase-label calibration decisions.
- Evaluate in inference mode with deterministic action selection. If a model
  is intrinsically stochastic, its evaluation RNG and sampling rule must be
  fixed before the final run and recorded.
- The task horizon, reset distribution, and success predicate come from the
  pinned robosuite/robomimic environment. Do not replace task success with an
  action-error threshold or a hand-written distance cutoff.

The 10-episode run must never be presented as the final result when a
50-episode result is required for the protocol.

### 7.1.1 Robosuite version split (ToolHang vs the other four tasks)

The five robomimic PH low-dim datasets split into two robosuite version
pins in their embedded `env_args`:

- Lift, Can, Square, Transport → `robosuite==1.5.1`
- ToolHang → `robosuite==1.5.0`

The parity gate validates the installed robosuite against the dataset's
recorded `env_version` (exact match). The per-task pins are also declared
in `data/<task>.yaml` as `source.robosuite_requirement`. ToolHang will
hard-fail under the 1.5.1 environment used for the other four tasks; the
operator must run ToolHang in a separately pinned 1.5.0 venv. The
dataset's recorded version is authoritative and is never bypassed.
For Tool Hang, create a separate environment without the v1.5.1 rollout
extra, then install the recorded exception explicitly:

```bash
uv venv --python 3.11 .venv-toolhang
source .venv-toolhang/bin/activate
uv pip install -e '.[dev]'
uv pip install 'robosuite==1.5.0' 'mujoco==3.2.7'
```

On PowerShell, activate the same environment with
`.venv-toolhang\\Scripts\\Activate.ps1` instead.

Do not run `uv sync --extra rollout` in that environment afterward, because it
would restore robosuite 1.5.1 and the parity gate must reject that mismatch.

### 7.2 Primary reporting

Report:

- binary success per episode;
- success rate per task;
- mean task success across tasks;
- aggregate episode success, clearly distinguished from mean task success;
- per-seed success rate and mean ± sample standard deviation across the
  three training seeds;
- a 95% Wilson interval for each per-seed task success rate over its 50
  episodes;
- the paired PhaseForge-minus-baseline difference for each task and seed;
- a 95% interval for the cross-seed paired difference, with the three seeds
  treated as the independent training replicates;
- number of valid seeds and failed/incomplete runs.

Every result table must identify:

- task;
- observation condition;
- history length;
- model;
- training seed;
- number of evaluation episodes;
- checkpoint-selection rule.

Missing or failed runs are incomplete, never silently converted to zero success.

Do not use a bootstrap over only three seed-level observations as the primary
confidence interval. It is too small to provide a reliable nonparametric
sampling distribution. Episode-level intervals describe reset uncertainty;
seed-level variation describes training uncertainty. Report both separately.

### 7.3 Statistical summary and aggregation

The primary cross-task number is the unweighted macro-average of the five
per-task success rates, giving each task equal weight. Also report the pooled
episode rate as a descriptive number, because it weights tasks by the number
of valid episodes. The macro-average is the primary summary; the pooled rate
must not replace it.

For pairwise comparisons, use the same task and evaluation seed bank for both
methods and report paired differences. With only three training seeds, treat
formal p-values as secondary and do not call a result significant solely
because of a single favorable seed. A strong PhaseForge claim requires a
positive paired effect on the macro-average, replication across seeds, and no
unexplained collapse on an individual task.

### 7.4 Checkpoint selection

Checkpoint selection must be fixed before the final test evaluation. It must use validation data or a predeclared training rule, never the final test success rate.

Because offline loss and rollout success can disagree, final checkpoint selection must be justified independently of the held-out test episodes.

### 7.5 Videos and failure analysis

Record representative rollouts for:

- successful PhaseForge behavior;
- PhaseForge failure;
- baseline failure;
- routing collapse;
- phase-transition failure;
- object-contact failure.

At least one failure taxonomy must be reported. Success rate alone does not explain whether the method failed during approach, grasp, transport, placement, or termination.

---

## 8. Metrics

### Primary outcome

- task success rate from the simulator’s success predicate.

### Secondary behavioral outcomes

- per-task success;
- success by task stage only when the pinned simulator exposes a validated
  stage predicate;
- episode length/time-to-termination, with timeout reported separately from
  task failure;
- boundary-action discontinuity on held-out demonstration trajectories;
- failure-stage distribution from a predeclared taxonomy;
- optional simulator-native progress signals, clearly labeled as secondary
  and never substituted for binary task success.

### Mechanism diagnostics

- phase-classification accuracy;
- phase–expert normalized mutual information;
- the phase–top-1-expert contingency matrix and per-phase assignment purity;
- phase coverage and duration statistics for the offline auxiliary labels;
- pre-top-k routing entropy;
- top-1 and top-k expert load balance and collapse rates;
- routing switch rate, computed within trajectories only;
- time to stable routing plus the fraction of trajectories that stabilize;
- per-phase action error on held-out trajectories, when phase labels are
  available for that diagnostic.

Offline action loss is diagnostic only. It is not the main performance metric.

Mechanism metrics are evidence about the proposed training mechanism, not
independent performance objectives. In particular, high NMI or high balance
without improved rollout success must be reported as a mechanistic result,
not as task improvement. Ground-truth/oracle routing is a metric sanity check;
it is not a deployable upper bound unless the paper explicitly defines it as
such.

NMI is interpreted differently by model:

- for Scratch MoE, Warm-Start MoE, and PhaseForge, it measures emergent alignment;
- for Teacher-Forced Routing, it primarily measures the learned phase predictor and is not evidence of emergent specialization;
- for any ground-truth routing reference, it is a sanity check, not a learned result.

---

## 9. Decision rules

### Positive result

The PhaseForge hypothesis receives support only if most of the following hold:

- PhaseForge improves rollout success over Scratch MoE and Warm-Start MoE;
- the improvement is replicated across tasks and seeds;
- phase–expert alignment is higher or more stable;
- expert collapse is not increased;
- load balance is not achieved by making all experts behaviorally redundant;
- the advantage remains under the predeclared evaluation protocol.

### Mechanistic but not behavioral result

If PhaseForge improves NMI or routing stability but not task success, report a routing effect without claiming a manipulation-performance improvement.

### Negative result

If PhaseForge does not outperform the matched baselines, report the controlled negative result. Do not change the claim after seeing the result.

### SR = 0 result

If all learned models obtain SR = 0:

- native predicate / parity / state-restore gate fails → evaluator or environment problem;
- gates pass and structured BC fails → learning, observation, task-conditioning, or action-contract problem;
- structured BC succeeds and robot-only BC fails → expected information ceiling;
- BC succeeds and all MoEs fail → MoE implementation or optimization problem;
- PhaseForge succeeds but does not beat baselines → phase-bootstrap hypothesis is unsupported.

SR = 0 is not a reason by itself to train longer or purchase more compute.

---

## 10. Required implementation changes

Before the final matrix, the codebase must implement and test:

1. robomimic/robosuite dataset ingestion;
2. the exact low-dimensional observation schema;
3. trajectory-level train/validation/test splits;
4. single-task training and evaluation;
5. a common history-window interface;
6. BC-MLP and BC-RNN baselines;
7. explicit task conditioning for any multitask extension;
8. action normalization and de-normalization checks;
9. demonstration replay and simulator parity tests;
10. native success-predicate availability probe (task-independent);
11. paired fixed-initial-state rollout evaluation;
12. per-task success and failure-stage reporting;
13. checkpoint-selection safeguards;
14. per-episode rollout records containing task, evaluation seed, reset seed,
    checkpoint, success, termination reason, horizon/timeout status, and
    failure category;
15. rollout-time routing traces so mechanism diagnostics are computed on the
    same held-out episodes as behavioral evaluation;
16. Wilson episode intervals and paired cross-seed summary statistics;
17. complete provenance in every result file.

The current repository does not yet satisfy this list: it has the Lift HDF5
ingestion pilot, offline single-step data path, simulator rollout adapter,
and task-independent gates (parity + state restore + action contract +
native success-predicate probe), but not the history-aware dataset/model
path or the complete three-seed × five-task configuration matrix. No final
success-rate claim is valid until the three-seed matrix finishes
successfully across all five tasks.

The existing PhaseForge router, phase-labeling, routing diagnostics, checkpointing, and evaluation instrumentation may be reused only after they satisfy this benchmark contract.

---

## 11. Final claims allowed in the paper

The paper may claim:

> We study phase-aware mixture-of-experts training for structured low-dimensional robot manipulation. Under a controlled robomimic/robosuite evaluation protocol, we compare phase-centroid router bootstrapping with matched behavioral-cloning and MoE baselines using simulator task success and routing diagnostics.

The paper may not claim:

- state-of-the-art vision performance;
- superiority to vision-language-action foundation models;
- perception capability;
- general open-world manipulation;
- success from bare proprioception on tasks requiring object information;
- that routing metrics alone prove better manipulation.

This scope is deliberately narrow. It is also the scientifically defensible version of the project.

---

## 12. Source record

- Chi et al., **Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**, RSS 2023. [Project and state-based resources](https://diffusion-policy.cs.columbia.edu/)
- Mandlekar et al., **What Matters in Learning from Offline Human Demonstrations for Robot Manipulation**, CoRL 2021. [Study and lessons](https://robomimic.github.io/study/)
- robomimic, **robosuite dataset and low-dimensional observation extraction protocol**. [Documentation](https://robomimic.github.io/docs/v0.4/datasets/robosuite.html)
- robomimic, **multimodal and low-dimensional observation interfaces**. [Documentation](https://robomimic.github.io/docs/tutorials/observations.html)
- Yu et al., **Meta-World: A Benchmark for Multi-Task and Meta Reinforcement Learning**, 2019. [Project](https://meta-world.github.io/) and [state-space documentation](https://metaworld.farama.org/benchmark/state_space/)
- Han et al., **CIMER: Combining Imitation and Emulation to Learn Prehensile Dexterity from State-only Observations**, ICRA 2025. [Paper page](https://star-lab.cc.gatech.edu/papers/han-cimer-icra/)
