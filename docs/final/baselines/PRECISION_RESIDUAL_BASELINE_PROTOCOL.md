# Precision-Residual PhaseForge: Final Baseline and Ablation Research Protocol

**Status:** Design specification — no final baseline training or deletion is authorized until the gates in this document pass.

**Scope:** The final proposed method is `precision_residual_phaseforge`. This document defines the controls required to explain why it works, how each control must differ, which historical implementations are no longer valid for the final comparison, and the exact sequence from implementation through reporting.

## 1. Executive decision

The four existing cells below must not be deleted as research questions:

1. `warmstart_moe`
2. `plain_encoder_phase_bootstrap`
3. `phase_pretrain_random_router`
4. `teacher_forced`

They are necessary because the proposed method is an MoE and its claimed contribution is decomposable. However, their current implementations were written for the older direct-action `phaseforge` family. They cannot be used unchanged as final-method ablations.

The correct treatment is:

- preserve their historical code, checkpoints, and results as historical records;
- create final-aligned versions with explicit identities;
- keep the original research questions;
- never silently relabel old outputs as results for `precision_residual_phaseforge`;
- retain `teacher_forced` as a privileged diagnostic, not as a deployable baseline;
- do not delete the ablation family until the final comparison and provenance record are complete.

The final method is the only proposed method. The controls are not competing PhaseForge versions; they are controlled experiments that remove or replace one component of the final method.

## 2. Current implementation facts

The current final configuration is [precision_residual_phaseforge.yaml](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/config/models/precision_residual_phaseforge.yaml). It specifies:

- normalized state encoder output;
- a six-expert `PrototypeRouter`;
- hard top-1 routing;
- topology-based prototype initialization;
- a `ResidualImpedanceExpert`;
- 50% partial warm-start from the Stage 1 action head;
- Stage 2 encoder fine-tuning with `encoder_lr_scale: 0.1`;
- SupCon and margin-routing losses enabled by the confirmation manifest.

The current old controls do not share that complete contract:

| Cell | Current implementation | Current mismatch |
|---|---|---|
| `warmstart_moe` | `TopKRouter` + direct `ExpertMLP` | Old top-2/direct-action path; random router and standard warm-start semantics |
| `plain_encoder_phase_bootstrap` | BC encoder + centroid `TopKRouter` + direct experts | Old router/expert path; no final residual expert or final prototype geometry |
| `phase_pretrain_random_router` | `WarmStartMoEModel` with direct experts | Its provider resolves to old `phaseforge` through [config.py](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/utils/config.py) |
| `teacher_forced` | Direct experts and `TopKRouter` structural parity | Its provider resolves to old `phaseforge`; encoder is frozen under its old diagnostic contract |

The old cells therefore answer historical questions about the old architecture. They do not yet answer the corresponding questions about the final architecture.

## 3. The critical semantic gate: is the residual branch active?

Before implementing final-aligned controls, the final method's semantic identity must be resolved.

The current final config sets:

```yaml
expert:
  _target_: phaseforge.models.components.impedance_expert.ResidualImpedanceExpert
  beta: 0.0
```

The current [ResidualImpedanceExpert](C:/Users/Hellx/Documents/Programming/python/Project/Neryva/PhaseForge/phaseforge/models/components/impedance_expert.py) returns the direct base action immediately when `beta == 0.0`:

```python
if self.beta == 0.0:
    return base_action
```

Therefore, at `beta: 0.0`:

- the residual compliance output does not affect the action;
- the residual branch receives no action-loss gradient;
- the final policy behaves as a direct-action expert MoE with prototype routing;
- calling the method “precision-residual” describes an installed component, not an active learned mechanism.

This is not a minor implementation detail. It determines what the baselines must isolate.

### Required decision

Choose exactly one of the following before the final baseline run:

**Decision A — active residual method.** The residual branch is part of the claim. Then the implementation must define and test a nonzero or scheduled `beta`, pass task state correctly, and demonstrate that residual parameters receive gradients. The schedule or value must be fixed before final evaluation.

**Decision B — zero-residual direct method.** `beta: 0.0` is intentional and the final claim is actually normalized prototype-routed direct-action MoE. Then the paper and manifests must describe the method that is executed, and the residual branch must not be presented as a demonstrated source of improvement.

Both decisions are scientifically valid. They are not interchangeable. No baseline matrix can be finalized until this decision is recorded.

### Required implementation checks for Decision A

If the residual branch is active, the following must be fixed before training:

1. `PhaseBootstrappedMoE._uses_impedance_experts()` must recognize `ResidualImpedanceExpert` when its forward path requires task state.
2. `MoELayer` must pass the normalized task state to residual experts.
3. `ResidualImpedanceExpert` must expose a testable residual contribution and gradient path.
4. `beta` initialization and any beta schedule must be stored in resolved configuration and run metadata.
5. A unit test must prove that `beta > 0` changes the pose action while leaving the gripper channel on the declared direct path.
6. A training-step test must prove nonzero gradients reach `delta_head` and `gain_head` when the residual path is enabled.

The existing zero-initialization idea is compatible with a safe recovery schedule, but “initialized at zero” and “zero for the entire run” are different experiments.

## 4. Research question and causal decomposition

The final method can be represented as:

```text
F = P + T + R + W + U
```

where:

- `P` = phase/SupCon-shaped representation;
- `T` = topology-derived prototype routing;
- `R` = residual impedance action representation;
- `W` = 50% partial warm-start expert initialization;
- `U` = final Stage 2 encoder update policy, currently low-rate unfreezing.

The baseline program must test these factors without changing unrelated factors.

The central causal chain is:

```text
phase supervision
  -> structured latent geometry
  -> useful prototype routing
  -> expert specialization
  -> precision-preserving action adaptation
  -> closed-loop success
```

This is why deleting the controls would weaken the research. A single success-rate table can show that the final method wins; it cannot show which part of the method caused the win.

MoE studies treat routing, expert specialization, and representation as coupled but separable design dimensions. The relevant literature also distinguishes dense-to-MoE initialization from routing and expert behavior: [Towards Understanding Mixture of Experts](https://arxiv.org/abs/2208.02813), [Sparse Upcycling](https://arxiv.org/abs/2212.05055), and [Drop-Upcycling](https://arxiv.org/abs/2502.19261).

## 5. Final-aligned factorial matrix

The following matrix is the required design after the residual semantic gate is resolved.

| Method identity | Representation | Router | Expert/action path | Initialization | Scientific question |
|---|---|---|---|---|---|
| `precision_residual_phaseforge` | final phase/SupCon encoder | topology prototypes, learned routing | residual impedance expert | partial warm | Proposed method |
| `precision_residual_plain_encoder` | normalized BC encoder | topology prototypes, learned routing | same residual expert | partial warm | Does phase-supervised representation help? |
| `precision_residual_phase_random_router` | final phase/SupCon encoder | random prototype router | same residual expert | partial warm | Does topology/centroid router initialization help? |
| `precision_residual_warmstart_moe` | normalized BC encoder | random prototype router | same residual expert | partial warm | Plain 2×2 factorial corner; does the phase pathway matter jointly with routing? |
| `precision_residual_teacher_forced` | final phase/SupCon encoder | ground-truth phase during training; predicted phase at evaluation | same residual expert | partial warm | Privileged routing upper bound; not a deployable method |
| `phaseforge_prototype` / `pr_direct_action` | final phase/SupCon encoder | topology prototypes, learned routing | direct `ExpertMLP` | partial warm | Does the residual action representation help? |
| `bc_impedance` / `pr_bc_impedance` | normalized BC encoder | no router | single impedance expert | ordinary BC training | Does action representation help without MoE routing? |
| `bc` | ordinary BC encoder | none | direct action head | ordinary BC | Dense imitation floor |
| `bc_large` | wider dense encoder | none | direct action head | ordinary BC | Capacity control |
| `precision_residual_scratch_moe` | normalized scratch encoder | random prototype router | same residual expert | random | Is partial warm-start necessary? Secondary control |

The four requested cells are not redundant:

- `precision_residual_plain_encoder` and the final method differ in representation.
- `precision_residual_phase_random_router` and the final method differ in router initialization.
- `precision_residual_warmstart_moe` completes the representation × router factorial.
- `precision_residual_teacher_forced` estimates the value of privileged phase routing and must be reported separately.

## 6. Exact definition of each requested control

### 6.1 `precision_residual_warmstart_moe`

**Purpose:** Test the final MoE/action machinery without phase-supervised representation or phase-informed router initialization.

Required properties:

- normalized BC Stage 1 encoder;
- no phase-supervised Stage 1 loss;
- same latent dimension, expert count, action scale, residual expert, and Stage 2 optimizer as the final method;
- random final-family prototype router;
- same 50% partial warm-start from the BC action head;
- same Stage 2 encoder update policy as the final method unless the final decision explicitly defines frozen encoders;
- no ground-truth phase at inference.

It must not use the old direct `ExpertMLP` if the final method's active claim is residual action control.

### 6.2 `precision_residual_plain_encoder`

**Purpose:** Isolate the contribution of the phase/SupCon representation while holding final routing, expert structure, and initialization fixed.

Required properties:

- normalized BC Stage 1 encoder trained with the same state and action protocol;
- same topology prototype router as the final method;
- same number of experts and same top-k;
- same residual expert and beta policy;
- same partial warm-start and Stage 2 optimization;
- no phase head or phase loss in the BC pretraining path;
- topology labels may be used for router bootstrap only if that use is explicitly part of the registered control and is held identical to the final router bootstrap.

The plain encoder must be trained with normalization enabled. Loading an old unnormalized BC checkpoint and merely toggling normalization at Stage 2 is invalid because the representation was not trained under the same mapping.

### 6.3 `precision_residual_phase_random_router`

**Purpose:** Isolate topology/prototype router initialization.

Required properties:

- Stage 1 checkpoint supplied by the final `precision_residual_phaseforge` provider;
- same normalized encoder, SupCon labels, phase head, residual expert, partial warm-start, and Stage 2 optimization as the final method;
- same prototype-router class and dimensions;
- router parameters left at a deterministic random initialization;
- no topology prototypes installed;
- no phase labels supplied during rollout.

The current historical `phase_pretrain_random_router` resolves to old `phaseforge` and therefore cannot be reused as this control without a new Stage 1 run.

### 6.4 `precision_residual_teacher_forced`

**Purpose:** Measure the gap between privileged phase routing and learned autonomous routing.

Required properties:

- final Stage 1 checkpoint;
- final normalized representation;
- same residual experts, partial warm-start, action scale, and Stage 2 optimization;
- ground-truth phase dispatch during Stage 2 training;
- predicted phase dispatch during evaluation;
- no ground-truth phase supplied to the proposed method;
- explicit privileged label in all reports.

This is not an ordinary baseline. It must not be included as evidence that the deployable method works without privileged information.

## 7. Historical versus final-aligned identities

The old names must not be silently reused for new definitions.

| Historical identity | Final-aligned identity | Treatment |
|---|---|---|
| `warmstart_moe` | `precision_residual_warmstart_moe` | Preserve old artifacts; create new final-family control |
| `plain_encoder_phase_bootstrap` | `precision_residual_plain_encoder` | Preserve old artifacts; create new final-family control |
| `phase_pretrain_random_router` | `precision_residual_phase_random_router` | Change Stage 1 provider to final method and rerun |
| `teacher_forced` | `precision_residual_teacher_forced` | Preserve old privileged diagnostic; create final-family diagnostic |
| `scratch_moe` | `precision_residual_scratch_moe` if needed | Do not compare old direct-action scratch MoE as a final residual control |

Old results remain valid only as historical results for the old architecture. They cannot be pooled with the final-family results.

## 8. Required implementation architecture

The final-aligned controls should share code where behavior is genuinely common. They should not be four copied implementations with drifting defaults.

### 8.1 Shared final-family control base

Create or generalize a shared control implementation that supports:

- `StateEncoder` with explicit `normalize_output`;
- `ActionHead` for Stage 1 checkpoint compatibility;
- optional `PhaseClassificationHead`;
- `TopKRouter` only where a legacy diagnostic explicitly requires it;
- `PrototypeRouter` for final-family controls;
- `ExpertMLP` for the direct-action control;
- `ResidualImpedanceExpert` for the final action path;
- random, topology/prototype, centroid, and teacher routing modes;
- partial warm-start through the wrapped `base_expert`;
- task-state extraction for residual action computation;
- the final encoder freeze/unfreeze policy;
- `ModelOutput.latent` and `ModelOutput.info` for training diagnostics;
- deployment metadata recording router and expert types.

The implementation must fail closed when a residual expert is configured but task state is absent.

### 8.2 Checkpoint providers

The checkpoint source map must become explicit and final-family-specific:

```text
precision_residual_phase_random_router -> precision_residual_phaseforge Stage 1
precision_residual_teacher_forced     -> precision_residual_phaseforge Stage 1
precision_residual_plain_encoder      -> final-aligned normalized BC Stage 1
precision_residual_warmstart_moe      -> final-aligned normalized BC Stage 1
```

No final-aligned control may silently load an old `phaseforge` checkpoint.

Stage 2 must always start from the provider's Stage 1 checkpoint. It must never start from the proposed method's Stage 2 checkpoint.

### 8.3 Configuration inheritance and drift prevention

All final-family configs must pin, rather than inherit accidentally, the following fields:

- state/action dimensions through the task data config;
- latent dimension and encoder widths;
- `normalize_output`;
- expert count;
- router class and top-k;
- margin and balance coefficient;
- topology/prototype source;
- action scales;
- residual beta policy;
- partial-warm drop rate and seed source;
- Stage 2 encoder update policy;
- loss enablement and weights.

The resolved config hash must be written to every run. A control is valid only when its differences from the final method match a declared allowlist.

## 9. Fairness contract

The final method and its mechanism controls must share:

- the same task-specific dataset and split;
- the same state schema and normalization statistics;
- the same action convention and action scale;
- the same number of training epochs;
- the same optimizer and scheduler unless the factor under study is the optimizer;
- the same Stage 1 and Stage 2 checkpoint-selection rules;
- the same seeds;
- the same frozen evaluation reset bank;
- the same episode count and horizon;
- the same simulator and environment version;
- the same success predicate;
- the same rollout mode and no history/oracle intervention for deployable rows.

Only the declared factor may differ.

For example, `precision_residual_plain_encoder` may differ in Stage 1 representation source, but it may not also switch from a residual expert to a direct expert, from top-1 prototype routing to top-2 soft routing, or from partial warm-start to full warm-start.

## 10. Validation and test gates

No final-family training begins until these tests pass.

### Configuration gates

1. Every final-family model composes for Lift, Can, Square, ToolHang, and Transport.
2. All four requested controls resolve to the intended model class.
3. All phase-supervised controls resolve Stage 1 from `precision_residual_phaseforge`.
4. All plain-encoder controls resolve Stage 1 from the final-aligned normalized BC provider.
5. No final-family manifest contains `model: phaseforge` as a Stage 1 provider.
6. Resolved config hashes are different only in the declared factor fields.

### Model-contract gates

1. All final-family policies are memoryless at inference.
2. No ground-truth phase enters proposed-method rollout.
3. Teacher-forced ground-truth routing is rejected in deployable evaluation mode.
4. Residual experts receive task state when and only when the active residual path requires it.
5. Direct experts never receive task state.
6. All action outputs satisfy the declared action range and dimension.

### Initialization gates

1. Partial warm-start copies the action head into the residual expert's `base_expert` only.
2. The dropped-neuron index set is shared across experts and hash-recorded.
3. The same seed produces the same dropped-neuron hash.
4. A different seed produces a different hash.
5. Final-aligned controls use the same drop rate and seed convention as the proposed method.

### Residual gates

If Decision A is selected:

1. `beta > 0` or the declared schedule must be present in resolved metadata.
2. The residual branch must produce a nonzero controlled contribution on a synthetic test input.
3. The residual parameters must receive nonzero gradients during a training step.
4. The gripper channel must follow the declared direct path.
5. The beta schedule must be fixed before reading final evaluation results.

## 11. Training and evaluation sequence

### Phase 0 — freeze definitions

1. Decide active versus zero residual behavior.
2. Freeze the final config and final-family control table.
3. Assign final identities and historical identities.
4. Define the intentional-difference allowlist for each control.
5. Freeze seeds, tasks, reset banks, and evaluation protocol.

### Phase 1 — implement and test

1. Add the shared final-family control implementation.
2. Add final-aligned normalized BC Stage 1 provider if required.
3. Update checkpoint-source resolution.
4. Add composition and contract tests.
5. Add residual-gradient tests if Decision A is selected.
6. Run unit, integration, and manifest tests on CPU.

### Phase 2 — dry-run and pilot

1. Dry-run every final-family manifest.
2. Verify dependency order: Stage 1 provider before Stage 2 consumer.
3. Run a small pilot only for implementation failures and numerical stability.
4. Do not select the final method or remove controls based on the pilot.

### Phase 3 — final training

Run all declared methods at one frozen commit:

- training seeds 42, 43, and 44;
- Lift, Can, Square, ToolHang, and Transport where the protocol allows;
- identical evaluation banks per task;
- fresh output namespace;
- resolved config, provider checkpoint, commit, dataset, bank hash, and environment recorded.

### Phase 4 — analysis

Report:

- per-task and per-seed rollout success;
- pooled success only with denominators shown;
- Wilson intervals;
- paired method-minus-control differences on identical reset cases;
- offline action MSE and phase metrics;
- router margin, entropy, utilization, and collapse diagnostics;
- residual contribution magnitude and residual-gradient diagnostics;
- total and active parameter counts;
- training time and inference cost;
- failure categories and timeout accounting;
- configuration and checkpoint hashes.

Do not pool old-generation artifacts with final-generation artifacts.

## 12. Interpretation rules

| Outcome | Valid conclusion |
|---|---|
| Final method beats BC only | The method improves over a dense imitation floor; MoE-specific causality is not established |
| Final method beats scratch/warm-start MoE | Evidence that the final initialization/representation/routing combination matters |
| Plain encoder control loses to final method | Evidence that phase/SupCon representation contributes |
| Random-router control loses to final method | Evidence that topology/prototype initialization contributes |
| Direct-action control loses to final method | Evidence that the residual action path contributes, but only if residual beta is active |
| Teacher-forced control is better | There is remaining phase-prediction or autonomous-routing loss; this is not evidence against the final method by itself |
| Teacher-forced control is worse | Ground-truth routing is not automatically sufficient; inspect expert allocation and training mismatch |
| Final method wins on one seed only | Exploratory evidence, not a robust method claim |
| Residual beta is zero throughout | Do not claim a residual-action improvement |

Three seeds provide a descriptive robustness check. They do not justify treating seed means as a large-sample population test.

## 13. Disposition of historical code and outputs

Historical implementations and outputs must be retained until the final research record is complete because they document the development path and explain why the final architecture was selected.

They must be clearly labeled as:

```text
historical / pre-final / old direct-action PhaseForge family
```

They must not be:

- renamed to `precision_residual_phaseforge`;
- combined with final-family results;
- used as evidence for final-family causal ablations;
- loaded as Stage 1 providers for final-family controls.

After the final report is complete, obsolete active configs may be hard-deleted from the working tree in a dedicated cleanup commit. Git history, immutable output directories, and this protocol must remain available. Rewriting Git history is not part of the research cleanup.

## 14. Final acceptance criterion

The baseline program is complete only when:

1. `precision_residual_phaseforge` is the sole proposed method identity.
2. The four requested research questions have final-aligned controls.
3. Every final-aligned control changes only its declared factor.
4. The residual branch's active/inactive semantics are explicitly resolved.
5. No final-family control loads an old `phaseforge` checkpoint.
6. The same-generation run matrix is complete.
7. Historical and final results are separated by identity, commit, and config hash.
8. The final report can explain not only that the method wins, but why.

Until these conditions hold, the project is in research-definition phase and should not proceed to irreversible cleanup or publication claims.

