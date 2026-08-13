# PhaseForge — Final State-Based Evaluation and Research Plan

**Status:** final protocol; current code is a Lift low-dimensional ingestion pilot, not a complete evaluation implementation

**Research definition:** [research_definition.md](research_definition.md)

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

- Diffusion Policy provides separate state-based and vision-based experiments, with reproducible low-dimensional experiment logs and three-seed evaluation. [Official project](https://diffusion-policy.cs.columbia.edu/)
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
may not reproduce the same results. The repository's current optional rollout
extra is still a temporary v1.4.0 pilot pin; it is not approval to mix that
environment with the current released artifacts. Before rollout work, either
select the v1.5.1 artifact/environment pair or deliberately select an older
matching pair and record it in `MANIFEST.json`.

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
- absence of future-state leakage.

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
- primary metrics;
- checkpoint-selection rule.

No result is included in the final comparison if it was produced under a different schema or split.

### Gate 1 — simulator and evaluator validation

Run, in order:

1. demonstration-action replay;
2. state extraction parity checks;
3. action-scale and normalization checks;
4. success-predicate checks;
5. a no-op/random-action baseline;
6. a scripted or state-oracle controller.

The scripted controller must solve the task instances used by the evaluation harness. If it fails, stop and repair the environment or evaluator before training learned policies.

### Gate 2 — BC floor

Train BC-MLP and BC-RNN on a small pilot subset first.

Proceed only when structured-state BC produces clearly nonzero rollout success on held-out initial states and outperforms the robot-only negative control where object information is necessary.

If structured-state BC fails while the scripted controller succeeds, investigate task conditioning, observation extraction, temporal context, action scaling, and checkpoint loading. Do not respond by increasing the MoE training budget.

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

- 10 episodes per task: smoke test only.
- 50 episodes per task: final reported evaluation.
- 3 independent training seeds: minimum final matrix.

The 10-episode run must never be presented as the final result when a 50-episode result is required for the protocol.

### 7.2 Primary reporting

Report:

- binary success per episode;
- success rate per task;
- mean task success across tasks;
- aggregate episode success, clearly distinguished from mean task success;
- mean ± standard deviation across training seeds;
- paired confidence intervals or bootstrap intervals over task-level results;
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

### 7.3 Checkpoint selection

Checkpoint selection must be fixed before the final test evaluation. It must use validation data or a predeclared training rule, never the final test success rate.

Because offline loss and rollout success can disagree, final checkpoint selection must be justified independently of the held-out test episodes.

### 7.4 Videos and failure analysis

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
- success by task stage or phase;
- action smoothness;
- boundary-action discontinuity;
- time to completion;
- failure-stage distribution.

### Mechanism diagnostics

- phase-classification accuracy;
- phase–expert normalized mutual information;
- expert assignment purity by phase;
- routing entropy;
- expert load balance;
- collapsed-expert count;
- routing switch rate;
- time to stable routing;
- per-phase action error.

Offline action loss is diagnostic only. It is not the main performance metric.

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

- scripted controller fails → evaluator or environment problem;
- scripted controller succeeds and structured BC fails → learning, observation, task-conditioning, or action-contract problem;
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
10. scripted/state-oracle evaluator validation;
11. paired fixed-initial-state rollout evaluation;
12. per-task success and failure-stage reporting;
13. checkpoint-selection safeguards;
14. complete provenance in every result file.

The current repository does not yet satisfy this list: it has the Lift HDF5
ingestion pilot and offline single-step data path, but not the history-aware
dataset/model path, simulator rollout adapter, scripted-controller gate, or
the five-task configuration matrix. No final success-rate claim is valid until
those missing components are implemented and Gate 1 has passed.

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
