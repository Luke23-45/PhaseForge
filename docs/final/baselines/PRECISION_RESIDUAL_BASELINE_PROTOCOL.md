# Precision-Residual PhaseForge: Final Baseline and Ablation Execution Plan

**Status:** Final research-definition plan. This document is the execution authority for the final baseline matrix. It does not authorize training, deletion, or movement of artifacts until the validation gates pass.

**Proposed-method identity:** `precision_residual_phaseforge`

**Scope:** This plan consolidates the five professor reports and the repository audit. It fixes the label taxonomy, action-path semantics, baseline identities, checkpoint provenance, implementation work, validation gates, training sequence, and reporting rules.

## 1. Final decisions

The project is now locked to one proposed-method identity: `precision_residual_phaseforge`.

The paper's primary claim is **not** a residual-compliance claim. The current final configuration sets `ResidualImpedanceExpert.beta = 0.0`; the residual branch therefore does not affect the action and receives no action-loss gradient. The paper will describe the executed method as a memoryless, topology-initialized, hard prototype-routed direct-action MoE with a phase/topology-supervised representation. The residual expert remains an architectural capability, not a demonstrated source of improvement in this paper.

The final label policy is canonical:

| Function | Proposed method and topology controls | Static-rule comparison |
|---|---|---|
| Stage 1 SupCon | `phase_topo` | `phase` |
| Stage 1 auxiliary phase classification | `phase_topo` | `phase` |
| Stage 2 prototype initialization | `phase_topo` | `phase` |
| Stage 2 margin-routing loss | `phase_topo` | `phase` |
| Routing diagnostics/targets | the declared method label field | the declared method label field |

This is an explicit project decision. It is not a claim that `phase_topo` is ground-truth annotation. `phase_topo` is a train-only, topology-derived regime label produced by the repository's PELT pipeline. It must be called a topology-derived or privileged discovered label in the paper.

All final tasks use `beta = 0.0`: Lift, Can, Square, ToolHang, and Transport. No beta schedule is allowed in the final matrix. Activating beta is a separate future experiment with a new identity and new ablations.

## 2. Repository facts that constrain the plan

The current final model configuration is [precision_residual_phaseforge.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/models/precision_residual_phaseforge.yaml). Its relevant contract is:

- normalized `StateEncoder` output;
- six experts and a hard top-1 `PrototypeRouter`;
- topology prototype source when `router_init.prototype_source=topo`;
- `ResidualImpedanceExpert` with `beta: 0.0`;
- 50% partial expert warm-start;
- Stage 2 encoder fine-tuning with the configured encoder learning-rate scale;
- SupCon and margin losses enabled only by explicit manifest overrides.

The residual expert returns its `base_expert` action immediately at beta zero. The final action path is therefore direct-action, although the residual module is present in the model graph.

The current training code still hardcodes `batch["phase"]` in several places. The final plan therefore requires configurable label fields, not only manifest overrides:

- Stage 1 phase-classification loss currently uses `batch["phase"]` in [stage1_loop.py:165](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/trains/loops/stage1_loop.py:165).
- Stage 1 SupCon already has a configurable label field in [stage1_loop.py:274](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/trains/loops/stage1_loop.py:274).
- Stage 2 margin loss currently uses `batch["phase"]` in [stage2_loop.py:342](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/trains/loops/stage2_loop.py:342).
- The current plain encoder control uses the old `TopKRouter` and reads `batch["phase"]` for bootstrap in [plain_encoder_phase_bootstrap.py:204](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/baselines/plain_encoder_phase_bootstrap.py:204).

The current runner's protocol schema accepts only `stage2_source` values `self`, `bc`, and `phaseforge`; this is enforced in [protocol.py:241](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/runner/protocol.py:241). Final provider identities must therefore be added to the existing runner rather than written into a manifest and assumed to work.

The existing runner already has strict seed-aware run resolution in [resolver.py:259](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/runner/resolver.py:259). The final implementation must extend and reuse that mechanism.

## 3. Label and data contract

### 3.1 Label definitions

`phase`/`phase_rule` is the existing human-defined or rule-derived phase vocabulary.

`phase_topo` is produced by the PELT topology pipeline. The final protocol uses `topo_pelt_k6`, so the expected topology regime count is six. The artifact is train-only for representation and router initialization; it must not enter the proposed policy's deployable rollout input.

The final data configuration must include the topology artifact and labels for every task used by the final matrix. The topology configuration is [topo_pelt_k6.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/topo/topo_pelt_k6.yaml), which declares six regimes and `train_label_field: phase_topo`.

Before training, validate that every required split contains `phase_topo`, that labels are integer-valued, contiguous after the repository's documented remapping, and compatible with the six-class phase head and six experts. A K-sweep or a different topology regime count is outside this locked matrix.

### 3.2 Canonical loss-field configuration

Add explicit fields to the existing Stage 1/Stage 2 configuration structure. Do not invent a nested `train.stage1`/`train.stage2` schema because the repository selects `train=stage1` and `train=stage2`.

The required resolved fields are:

```yaml
# train=stage1
phase_label_field: "phase_topo"
supcon:
  enabled: true
  label_field: "phase_topo"

# train=stage2
phase_label_field: "phase_topo"
margin:
  enabled: true
  label_field: "phase_topo"
```

For the Static Rule MoE, the corresponding fields are `phase`.

The trainer must fail closed when a requested label field is absent. The resolved label fields must be written to run metadata and included in the configuration hash.

The phase-classification loss, SupCon loss, prototype initialization, Stage 2 margin loss, and routing diagnostics must use the resolved label field appropriate to the method. Phase-specific optional losses such as release loss remain disabled in this matrix; if enabled in future, they require their own declared label-field contract.

## 4. Action-path contract

The final matrix uses `ResidualImpedanceExpert` with `beta=0.0` for every MoE row. At beta zero:

- action output is the warm-started direct base action;
- residual pose heads do not affect the output;
- residual heads receive no action-loss gradient;
- the gripper channel remains the base expert's direct channel;
- no task-state feedback impedance controller is active.

The separate `ImpedanceExpert` path used by `bc_impedance` is not an exact control for the latent-conditioned residual module and is not part of the core causal matrix. It may be retained as a secondary action-space reference with separate labeling.

The final resolver must reject any final-matrix configuration whose resolved beta is not exactly zero or whose beta schedule is non-null. The validation belongs at the experiment/manifest layer, not inside the expert class, because the model class does not know the experiment identity.

## 5. Final experimental matrix

All final rows use the same task data, state/action conventions, training/evaluation protocol, six-expert capacity, and beta-zero action contract unless explicitly stated otherwise.

### 5.1 Proposed method

| Identity | Stage 1 provider | Stage 1 labels | Router | Router init | Expert init | Role |
|---|---|---|---|---|---|---|
| `precision_residual_phaseforge` | self | `phase_topo` for phase CE and SupCon | `PrototypeRouter`, top-1 | topology prototypes from `phase_topo` | 50% partial warm-start | Sole proposed method |

This is the only row that may be called the proposed method.

### 5.2 Single-factor causal controls

| Identity | Stage 1 provider | Representation | Router/init | Expert init | Factor tested |
|---|---|---|---|---|---|
| `precision_residual_plain_encoder` | final-aligned normalized BC | no SupCon and no phase head; BC representation | `PrototypeRouter`, top-1; prototypes from `phase_topo` in the BC latent space | 50% partial warm-start | Representation |
| `precision_residual_phase_random_router` | proposed method Stage 1 | `phase_topo` | `PrototypeRouter`, top-1; deterministic random initialization | 50% partial warm-start | Router initialization |
| `precision_residual_scratch_moe` | proposed method Stage 1 | `phase_topo` | topology-initialized `PrototypeRouter`, top-1 | independent random experts | Expert initialization |

The plain encoder is not allowed to reuse prototypes computed from the SupCon encoder. Its bootstrap must compute prototypes from its own BC latent vectors and the declared `phase_topo` labels.

### 5.3 Factorial corner

| Identity | Stage 1 provider | Representation | Router/init | Expert init | Factors tested |
|---|---|---|---|---|---|
| `precision_residual_factorial_floor` | final-aligned normalized BC | no SupCon and no phase head; BC representation | `PrototypeRouter`, top-1; random initialization | 50% partial warm-start | Representation + router initialization |

This row is intentionally a two-factor 2×2 corner. It must not be described as a single-factor ablation.

### 5.4 Major external baselines

| Identity | Stage 1 provider | Router | Expert | Role |
|---|---|---|---|---|
| `bc` | itself | none | direct action head | Dense imitation floor |
| `final_aligned_softmax_top1` | proposed method Stage 1 | learned softmax-scored top-1 `TopKRouter` | same residual expert, beta zero | Pre-registered router-package comparison |
| `final_aligned_static_rule` | its own Stage 1 | `PrototypeRouter`, top-1 | same residual expert, beta zero | Integrated rule-label versus topology-label comparison |

The Static Rule MoE uses `phase` consistently for SupCon, phase classification, prototype initialization, and margin targets. It is an integrated comparison, not a router-only ablation.

The final `PrototypeRouter` configuration uses `balance_coeff=0.0001`, as specified by the locked proposed-model configuration. This is a small auxiliary dead-expert penalty; topology initialization and the margin objective remain the primary prototype-router mechanisms. The softmax control uses `balance_coeff=0.01`, as specified below, because the learned softmax gate has a separate registered anti-collapse regularizer. This coefficient difference is deliberate and must be reported. Consequently, `final_aligned_softmax_top1` is a pre-registered router-package comparison, not a mathematically pure one-factor gate-only ablation. A pure gate-only claim would require an additional matched-balance study and is outside this locked matrix.

The softmax row uses these pre-registered settings:

```yaml
router:
  _target_: phaseforge.models.components.router.TopKRouter
  top_k: 1
  noise_std: 0.1
  normalize_input: true
  balance_coeff: 0.01
```

Noise is active only during training through the router's existing train/eval behavior. No post-hoc tuning of `balance_coeff` is permitted. The primary label is **learned softmax-scored top-1 router**, not a true top-2 soft mixture. A top-2 soft mixture, if ever run, is a separate secondary comparison and must not replace the primary control.

Because the final method uses a prototype margin loss, the softmax router must expose a compatible `margin_loss_from_logits` interface using the same multiclass margin formula on gate logits. The margin loss must use deterministic pre-exploration logits, or the implementation must explicitly document an identical noise policy for both routers; training noise must not silently affect only the softmax margin targets. Without this interface, the softmax row would differ in both router mechanism and training objective and would not be a valid pre-registered router-package comparison. The implementation must test the common loss numerically on both router classes.

### 5.5 Privileged diagnostics

| Identity | Training routing | Evaluation routing | Status |
|---|---|---|---|
| `precision_residual_teacher_forced` | `phase_topo` labels | predicted `phase_topo` phase-head class | Privileged training diagnostic; separate table |
| `precision_residual_oracle` | `phase_topo` labels | recorded `phase_topo` labels supplied by a privileged offline evaluator | Non-deployable reference; not an ordinary rollout row |

`teacher_forced` and `oracle` are distinct. Teacher-forced training uses labels but evaluation uses the learned phase head. The oracle uses labels at both training and evaluation.

The oracle must not fabricate `phase_topo` from a state-only `get_action()` call. The standard rollout interface does not provide the offline PELT label. The oracle is therefore evaluated offline or through an explicitly privileged evaluator that supplies the recorded label. It must reject normal deployable rollout mode.

The oracle is an offline routing/action diagnostic, not an ordinary rollout success-rate baseline. Do not report an oracle rollout success rate unless a separately validated privileged labeler supplies labels on the policy-generated state trajectory; recorded demonstration labels alone cannot be assumed to label a different rollout trajectory.

The final-aligned oracle is restricted to offline diagnostics on held-out demonstration data. Permissible metrics are: validation action MSE on held-out demonstration states; routing agreement between the supplied labels and the learned phase head on those states; and per-phase action error under label-directed routing. A rollout success rate is out of scope for this matrix because no validated labeler for arbitrary policy-generated states is included.

The existing historical `oracle_moe` is not automatically a final-aligned oracle: its rollout path falls back to a router that was not trained for oracle deployment. It must remain historical unless replaced by the final privileged evaluator.

## 6. Final-aligned model implementations

### 6.1 Shared implementation principles

Final-family models must share code for genuinely common behavior. They must not be four copied models with drifting defaults.

The shared implementation must support:

- normalized and non-normalized encoders as explicit configuration;
- Stage 1 action head and optional phase head;
- `PrototypeRouter` and `TopKRouter`;
- topology, rule, random, and K-Means initialization modes where explicitly registered;
- configurable label fields for every loss and bootstrap operation;
- partial warm-start and random expert initialization;
- `ResidualImpedanceExpert` beta-zero behavior;
- seed- and config-recorded initialization metadata;
- deployment metadata recording router, expert, label fields, and beta;
- memoryless inference for all deployable rows.

### 6.2 Plain encoder and factorial floor

Create a final-aligned plain MoE model or generalize the old plain model. It must:

1. load a normalized BC Stage 1 checkpoint;
2. use `PrototypeRouter`, not the old `TopKRouter`;
3. accept a configured bootstrap label field;
4. compute prototypes from its own BC latent vectors;
5. use `phase_topo` for prototype and margin targets in the final plain control;
6. use the residual expert with beta zero;
7. support random router initialization for the factorial floor;
8. use the same six experts, action dimensions, partial warm-start, and Stage 2 optimization as the proposed method.

Unit tests must prove that changing the label field changes only the intended bootstrap/target source and that no proposed-method latent or checkpoint is reused.

### 6.3 Softmax router

Use the existing `TopKRouter` implementation with the locked settings above. Add the common margin-loss method required by Section 5.4 and expose deterministic pre-exploration gate logits for that loss. The test must verify:

- top-1 selection;
- normalized input;
- training noise and evaluation determinism;
- balance-loss coefficient;
- compatible margin-loss output and gradients;
- identical beta-zero action-path contract.

Do not tune the balance coefficient after observing final utilization.

### 6.4 Teacher-forced diagnostic

Create or generalize the teacher-forced model so its final-aligned version has:

- normalized final encoder;
- topology-labeled Stage 1 checkpoint;
- final expert class with beta zero;
- declared partial warm-start policy;
- `phase_topo` labels during Stage 2 training;
- predicted phase-head routing during evaluation;
- explicit privileged-diagnostic metadata.

The old direct-action teacher-forced implementation remains historical and must not be silently relabeled.

### 6.5 Privileged oracle

Implement a final-aligned oracle evaluator that receives recorded `phase_topo` labels from the offline evaluation data. It must:

- use the final encoder/expert contract;
- route by the supplied topology label at training and evaluation;
- be marked non-deployable;
- refuse ordinary state-only rollout mode;
- record privileged-label access in metadata;
- validate label range and six-expert mapping.

Its standard result is an offline diagnostic on held-out data (for example, action loss or routing agreement). A simulation success rate is admissible only if the evaluator has a documented, validated privileged label source for the states actually visited during that rollout.

## 7. Loss and label implementation

Implement the following explicit configuration fields in the existing Hydra stage configs:

```yaml
# train=stage1
phase_label_field: "phase_topo"
supcon:
  label_field: "phase_topo"

# train=stage2
phase_label_field: "phase_topo"
margin:
  enabled: true
  label_field: "phase_topo"
```

The Static Rule MoE resolves all three fields to `phase`. Plain and factorial controls have no Stage 1 phase/SupCon loss, but their Stage 2 prototype and margin fields resolve to `phase_topo`.

Update the trainers so that:

- Stage 1 CE reads the resolved `phase_label_field`;
- Stage 1 SupCon reads `supcon.label_field`;
- Stage 2 margin reads `margin.label_field`;
- validation routing accuracy and related diagnostics use the declared field;
- missing fields fail closed with the method, task, and requested field in the error;
- label cardinality and contiguity are checked before training.

Where a phase head is retained for checkpoint compatibility, its output dimension remains six under the locked `topo_pelt_k6` protocol. A future K change requires a new protocol, new phase-head dimensions, and new identities.

## 8. Checkpoint and runner provenance

Do not hardcode absolute checkpoint paths in the manifest. Add explicit provider identities to the existing runner protocol or extend `stage2_source` to named providers. The runner must use the existing strict seed-aware resolution machinery and pass the resolved checkpoint as `train.stage1_ckpt_path` to the subprocess.

Required provider identities include:

- `precision_residual_phaseforge_stage1` — final topology/SupCon Stage 1 provider;
- `final_aligned_bc_stage1` — normalized BC Stage 1 provider, produced by an explicit final-aligned BC configuration or override with the required normalized encoder contract;
- `final_aligned_static_rule_stage1` — rule-label Stage 1 provider.

The exact names may be implemented as method identities, but they must be explicit in the protocol and cannot resolve through the historical aliases `phaseforge`, `bc`, or old baseline names.

For every Stage 2 consumer, resolution must require:

- matching task;
- matching training seed;
- completed provider run;
- expected provider commit;
- expected provider resolved-config hash;
- valid `checkpoint_best.pt`.

The run metadata must record:

- provider identity;
- resolved absolute checkpoint path;
- checkpoint SHA-256;
- provider commit;
- provider resolved-config hash;
- consumer resolved-config hash;
- dataset/cache hash;
- topology artifact hash when used;
- evaluation reset-bank hash;
- environment versions.

No final-aligned control may load a historical `phaseforge` Stage 1 checkpoint. The runner must fail before training if a seed-exact provider is unavailable.

For every task and training seed, all rows that consume `phase_topo` must resolve the same topology-artifact identity and hash. The artifact's label mapping/remapping must also be recorded. A method-specific topology artifact, or a silent regeneration with a different seed/configuration, is a protocol violation because it changes the target labels between compared rows.

### 8.1 Locked checkpoint and optimization policy

All Stage 1 and Stage 2 final-family checkpoints are selected by minimum `val/loss_action` (validation action MSE) on the held-out validation split. Rollout success, routing diagnostics, and final evaluation-bank results must never select a checkpoint. The selected epoch, validation split identity, monitor name, monitor mode, and monitor value must be recorded in run metadata. Early stopping remains disabled for the locked full-length runs.

The locked optimization values are the resolved values in [stage1.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/train/stage1.yaml), [stage2.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/train/stage2.yaml), and the per-task data configs:

| Setting | Stage 1 | Stage 2 |
|---|---:|---:|
| Epochs | 100 | 200 |
| Batch size | 256 | 256 |
| Optimizer | AdamW | AdamW |
| Learning rate | `3e-4` | `1e-4` |
| Weight decay | `1e-4` | `1e-4` |
| Betas / epsilon | `(0.9, 0.999)` / `1e-8` | `(0.9, 0.999)` / `1e-8` |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Scheduler `T_max` / `eta_min` | `100` / `1e-6` | `200` / `1e-7` |
| Gradient clipping norm | `1.0` | `1.0` |
| Checkpoint monitor | `val/loss_action`, min | `val/loss_action`, min |

Final-family Stage 2 rows that fine-tune the encoder use `encoder_lr_scale=0.1`; the value is part of the resolved configuration hash. Per-method loss switches and declared initialization factors are not silent hyperparameter changes: they must appear explicitly in the manifest and metadata. Any change to the values in this table requires a new protocol revision.

## 9. Manifest structure

Create a new manifest such as `experiments/final_causal_matrix.json`. Do not edit `experiments/five_task.json` in place for this final matrix.

The manifest must:

- contain task rows for Lift, Can, Square, ToolHang, and Transport;
- use training seeds 42, 43, and 44;
- declare final model identities, providers, label fields, and overrides explicitly;
- pin `topo@_global_=topo_pelt_k6` for rows that require topology labels;
- set `beta=0.0` explicitly for every residual-expert row;
- set `train.supcon.label_field=phase_topo` for topology representation rows;
- set `train.phase_label_field` and `train.margin.label_field` explicitly;
- include the Static Rule MoE with `phase` fields;
- include deployable rows in rollout evaluation only;
- include the oracle only in its privileged offline evaluator;
- use a fresh output namespace so historical artifacts cannot be overwritten.

The old `five_task.json` remains historical. Existing `phase`-trained results are not included in the final matrix, but they are preserved with their original method identity, commit, config hash, provider, and label policy.

## 10. Validation gates

No final training begins until every gate passes.

### 10.1 Configuration gates

1. Every final manifest composes successfully for all five tasks.
2. Every final row explicitly declares its label fields.
3. Topology rows use `topo_pelt_k6` and `phase_topo`.
4. Static Rule MoE uses `phase` consistently.
5. All final residual-expert rows resolve beta exactly to `0.0`.
6. No beta schedule is present in the final matrix.
7. All final rows use six experts and the declared top-k.
8. Config hashes differ only in declared factor fields and explicitly documented model-role fields.

### 10.2 Label/data gates

1. `phase_topo` exists in every required batch split.
2. `phase_topo` labels are valid six-class integer labels after documented remapping.
3. Stage 1 CE, SupCon, Stage 2 margin, bootstrap, and diagnostics resolve to the declared fields.
4. No topology label enters deployable rollout input.
5. Static-rule rows do not accidentally consume `phase_topo` for their declared rule-label losses.
6. All topology-consuming rows for a given task and seed use the same topology-artifact hash and label mapping.

### 10.3 Model-contract gates

1. Proposed and deployable controls are memoryless at inference.
2. No ground-truth or topology label enters proposed-policy rollout.
3. Teacher-forced evaluation uses the phase-head prediction, not the label.
4. Oracle evaluation is rejected by the ordinary rollout path.
5. Plain prototypes are computed from plain latents.
6. No final control reuses prototypes or Stage 2 checkpoints from another method.
7. All action outputs satisfy the task action dimension and range contract.

### 10.4 Router/loss gates

1. Prototype router is hard top-1.
2. Softmax router is top-1 with the exact registered settings.
3. Both routers expose the common margin-loss interface used by their corresponding rows.
4. `PrototypeRouter.balance_coeff=0.0001` and `TopKRouter.balance_coeff=0.01` are fixed before training.
5. The coefficient asymmetry is reported as part of the softmax router-package comparison; it is not described as a pure gate-only causal effect.
6. Neither balance coefficient is tuned after final results are observed.
7. Router utilization, transition counts, entropy, margin, and collapse diagnostics are recorded using the definitions in Section 12.

### 10.5 Provenance gates

1. Every provider is seed-exact.
2. Every provider has the expected commit and resolved-config hash.
3. Every Stage 2 run records the exact provider checkpoint and SHA-256.
4. Missing providers fail before subprocess launch.
5. Historical aliases cannot satisfy a final provider request.

## 11. Dry-run and test sequence

### Phase 0 — no-write inspection

1. Verify repository commit and working-tree ownership.
2. Enumerate existing historical outputs without moving or deleting them.
3. Confirm the topology artifact and label fields for every task.
4. Confirm the final output namespace is fresh.

### Phase 1 — implementation

1. Add configurable label fields to Stage 1 and Stage 2 trainers.
2. Add label-field validation and cardinality checks.
3. Implement the final-aligned plain/factorial model.
4. Generalize the teacher-forced model to the final contract.
5. Implement the privileged oracle evaluator, not a fake state-only oracle action method.
6. Add the common router margin-loss interface.
7. Extend the existing runner provider resolution with explicit identities and seed/commit/config checks.
8. Add beta-zero validation at the resolved manifest layer.

### Phase 2 — unit and composition tests

1. Compose every final model for all five tasks.
2. Verify every declared label field is present in representative batches.
3. Verify label counts and remapping.
4. Verify plain prototypes come from plain latents.
5. Verify random and topology router initialization hashes.
6. Verify partial warm-start drop hashes are seed-deterministic.
7. Verify beta-zero outputs equal the base action path and residual heads do not contribute.
8. Verify both routers compute the common margin loss.
9. Verify teacher-forced GT train/predicted eval behavior.
10. Verify oracle rejection in ordinary rollout mode.
11. Verify missing or wrong-seed providers fail before training.
12. Run the existing unit/integration/manifest test suites.

### Phase 3 — dry-run

1. Build the full plan from `final_causal_matrix.json`.
2. Verify provider dependency ordering.
3. Verify Stage 1 providers run before Stage 2 consumers for each seed.
4. Verify all commands carry the intended data, task, seed, model, and label overrides.
5. Verify no final row resolves to historical `phaseforge` or old baseline aliases.
6. Perform a small implementation-only pilot; do not use pilot scores to select methods or tune final settings.

### Phase 4 — final training and evaluation

Run every declared deployable row at one frozen commit with:

- seeds 42, 43, and 44;
- Lift, Can, Square, ToolHang, and Transport;
- identical task data and, within each task/seed, identical topology artifact and label mapping for every topology-consuming row;
- identical frozen reset banks per task;
- rollout mode for deployable rows;
- 50 reset-bank cases as configured by the evaluation protocol;
- task-recorded horizons, currently 500 for Lift/Can/Square/ToolHang and 700 for Transport;
- fresh output namespace;
- complete metadata and hashes.

Run teacher-forced only as its separate diagnostic. Run the oracle only through its privileged offline/evaluation path. Do not pool either diagnostic with deployable success rates.

## 12. Analysis and paper reporting

Report separate tables for:

1. external baselines: BC and final-aligned softmax top-1;
2. integrated comparison: Static Rule MoE;
3. single-factor causal controls: plain representation, random router initialization, scratch experts;
4. factorial corner: BC representation plus random prototype initialization;
5. privileged diagnostics: teacher-forced and oracle;
6. historical development results in an appendix only.

For every deployable method report:

- per-task success rate;
- per-seed success rate;
- pooled numerator and denominator;
- Wilson intervals;
- paired method differences on identical reset cases where applicable;
- action-loss metrics;
- phase/topology metrics using the declared label field;
- router utilization, entropy, margin, transition counts, and collapse diagnostics;
- training and inference cost;
- failure categories and infrastructure-failure accounting;
- configuration, checkpoint, data, topology, and bank hashes.

Routing diagnostics use these fixed definitions:

- **Utilization:** fraction of evaluation steps assigned to each expert by top-1 routing. For any future top-k row, also report the all-selected-experts fraction separately; do not mix the two definitions.
- **Entropy:** mean Shannon entropy of the full pre-top-k softmax distribution over all six experts, normalized by `log(6)`, matching the repository's `routing_entropy` metric.
- **Margin:** for a prototype router, `d_(2) - d_(1)` where `d_(1)` and `d_(2)` are the smallest and second-smallest prototype distances; for a softmax router, the pre-exploration-logit difference `logit_(1) - logit_(2)`. Report the mean and per-seed values.
- **Transition count/rate:** count adjacent in-trajectory pairs whose top-1 expert changes, using matching `trajectory_id` and consecutive `trajectory_position`; never count episode or trajectory boundaries as transitions.
- **Collapse:** an expert is collapsed when its top-1 utilization is `< 1/(5E)`, matching the final Stage 2 `expert_utilization.collapse_rate` implementation (`E=6` here). The older initialization diagnostic uses a different `< 0.01/E` threshold and must be labeled separately if reported; the two metrics must not be pooled.

Do not claim that the method solves boundary chattering or load collapse. State that the design is intended to reduce them, then report the measured transition and utilization evidence.

Do not claim a residual-compliance improvement. The residual branch is inactive at beta zero in this paper.

Do not claim that the factorial floor isolates one factor. It is a two-factor corner.

Do not call `phase_topo` ground truth. Call it topology-derived or privileged discovered labeling.

## 13. Historical artifact policy

Historical implementations and outputs are preserved until the final research record is complete. They must retain their original identity, commit, configuration hash, provider, and label policy.

They must not be:

- renamed to `precision_residual_phaseforge`;
- pooled with final-generation results;
- used as final-family Stage 1 providers;
- edited in place to appear final-aligned;
- deleted before the final record and provenance audit are complete.

After publication materials and the final provenance archive are complete, obsolete active configs may be archived or removed in a dedicated, reviewable cleanup change. Git history and the final protocol remain available.

## 14. Completion criterion

The final baseline program is complete only when:

1. `precision_residual_phaseforge` is the sole proposed-method identity.
2. The canonical `phase_topo` label policy is implemented for all declared representation/routing losses.
3. The plain, random-router, scratch, factorial, softmax, and static-rule controls have exact declared differences.
4. The final-aligned teacher-forced diagnostic exists.
5. The oracle is privileged and non-deployable by construction.
6. All providers resolve seed-exactly with commit/config/hash checks.
7. Beta is zero and validated for every final residual-expert row.
8. The same-generation matrix is complete across all five tasks and three seeds.
9. Historical and final results are separated by identity, commit, config hash, label policy, and data/topology hashes.
10. The paper claims match the executed method and measured evidence.

Until all ten conditions hold, the project remains in research-definition/implementation phase. Do not start final training, publication tables, hard deletion, or result relabeling.
