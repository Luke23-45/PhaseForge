# `lar_moe_state_only` Implementation Plan

**Status:** pre-implementation design; no code changes authorized by this document  
**Purpose:** define a faithful state-only adaptation of LAR-MoE before implementation  
**Primary source:** Rodriguez et al., “LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning,” arXiv:2603.08476, <https://arxiv.org/html/2603.08476>

## 1. Decision and naming

Implementing the method without vision is technically feasible. It must be named
`lar_moe_state_only` and described as a **state-only adaptation**, not as an exact
reproduction of the published vision-language LAR-MoE.

The adaptation preserves LAR-MoE’s algorithmic mechanism:

1. student–teacher representation pretraining using future actions;
2. frozen student representation during policy training;
3. soft expert routing from the student latent;
4. latent–routing distance-consistency regularization;
5. entropy and group-sparsity regularization.

Only the observation encoder is changed from the published visual-language input
to the repository’s structured state input. PhaseForge-specific phase labels,
phase-centroid initialization, top-k routing, and partial expert warm-starting
are excluded from this method.

## 2. Scientific role

`lar_moe_state_only` is a secondary mechanism-level comparator for PhaseForge. It
must not replace the primary PhaseForge baselines or be merged into the PhaseForge
ablation factorial.

The comparison is:

| Method | Representation source | Routing mechanism |
|---|---|---|
| PhaseForge | Phase-supervised state representation | One-time phase-geometry transfer into a sparse router |
| `lar_moe_state_only` | Unsupervised state–future-action representation | Continuous latent–routing alignment during policy training |

The method must not consume phase labels in pretraining, policy training,
checkpoint selection, or inference. Phase labels may be retained in evaluation
artifacts only for post-hoc diagnostics, never as an input or training target.

## 3. Fidelity boundary

### 3.1 Required LAR-MoE mechanisms

The following components are required for the method to be called a LAR-MoE
adaptation:

- A **student encoder** receives only the current state (s_t).
- A **teacher encoder** receives the current state and a future action window
  (a_{t:t+H-1}) during pretraining.
- A teacher decoder reconstructs the future action window from the teacher
  latent.
- The student latent is trained to match the teacher latent.
- The student encoder is frozen before policy training.
- The policy router computes a soft distribution over all experts from the frozen
  student latent.
- The policy output is the probability-weighted combination of expert outputs.
- Policy training includes action loss, distance-consistency loss, entropy
  regularization, and group-sparsity regularization.

### 3.2 Explicitly excluded PhaseForge mechanisms

The following must not be used in `lar_moe_state_only`:

- phase classification loss;
- phase-centroid or phase-prototype router initialization;
- `TopKRouter` sparse top-2 dispatch;
- `partial_warm`, `warmstart`, `one_warm`, or action-head expert copying;
- PhaseForge’s soft phase-to-expert mapping;
- PhaseForge’s load-balancing loss as a substitute for LAR’s regularizers;
- privileged phase labels as a teacher or routing target.

Adding any of these would create a hybrid method and require a different name.

### 3.3 What “without vision” changes

The following published components are replaced by state-compatible equivalents:

- visual-language observation encoder → structured-state MLP encoder;
- visual-language context inside each action expert → state-policy context;
- visual action-chunk decoder implementation → state-compatible action-chunk
  decoder.

The future-action teacher signal and the action-chunk objective must remain. If
future actions are removed and the method becomes a single-step state policy, it
is no longer a faithful LAR-MoE adaptation; it becomes only LAR-inspired.

## 4. Mathematical specification

Let (s_t) be the current normalized state and let
(A_t = a_{t:t+H-1}) be a valid future action window from the same trajectory.

### 4.1 Representation pretraining

The student produces:

\[
\hat z_t = f_s(s_t).
\]

The teacher receives the privileged future-action window:

\[
z_t = f_t(s_t, A_t).
\]

The teacher decoder reconstructs the action window:

\[
\hat A_t = g(z_t).
\]

The paper defines the two pretraining losses as:

\[
\mathcal L_{student} = \operatorname{MSE}(\hat z_t, z_t),
\]

\[
\mathcal L_{teacher} = \operatorname{MSE}(\hat A_t, A_t).
\]

Before implementation, confirm from the complete source and any official
implementation whether these losses are summed or separately weighted. If no
weight is specified, use the unit-sum rule and record that as an explicit
state-only adaptation choice. Do not silently copy PhaseForge’s phase-loss
weight because the objectives are different.

### 4.2 Policy routing

After pretraining, freeze (f_s). The router computes:

\[
p_t = \operatorname{softmax}(T \cdot \operatorname{MLP}(\hat z_t)),
\]

where (T) is a learnable temperature initialized according to the LAR-MoE
paper specification. Routing is soft over all experts; there is no top-k
selection.

Each expert predicts an action window (\hat A_{t,n}), and the policy output is:

\[
\hat A_t = \sum_{n=1}^{N} p_{t,n}\hat A_{t,n}.
\]

### 4.3 Policy objective

For a batch of student latents (Z) and routing distributions (P), compute
pairwise cosine-distance matrices (D^{(Z)}) and (D^{(P)}). The required
distance-consistency term is:

\[
\mathcal L_{DC} = \frac{1}{B^2}
\left\|D^{(Z)} - D^{(P)}\right\|_F^2.
\]

The policy objective is:

\[
\mathcal L_{policy} = \mathcal L_{action}
 + \lambda_{DC}\mathcal L_{DC}
 + \lambda_H\mathcal L_H
 + \lambda_G\mathcal L_G.
\]

The entropy and group-sparsity terms must follow the paper’s definitions. Their
weights, the temperature initialization, latent dimension, expert count, and
future-action horizon must be explicit resolved configuration fields.

## 5. Repository changes required

This is not a configuration-only baseline. The current main data contract uses
`sequence_length: 1`, and the existing `MoELayer` rejects sequence-aware latents.
The implementation must therefore be isolated from the PhaseForge top-k path.

### 5.1 Data layer

Add a dedicated future-window dataset or dataset mode, for example:

`phaseforge/data/common/future_action_dataset.py`

Required behavior:

- return (s_t), (A_t), trajectory identifier, and source position;
- construct only complete windows of length (H);
- never cross trajectory boundaries or train/validation boundaries;
- preserve the existing state/action normalization contract;
- retain no phase field as a training input for LAR-MoE;
- expose deterministic indexing and seed-independent window construction;
- provide a validity test for every returned window.

The existing temporal collator may be reused only after verifying that its output
shape is appropriate for the new model. Do not change the existing PhaseForge
single-step contract merely to support this baseline.

### 5.2 Model layer

Add a separate model implementation, for example:

`phaseforge/models/baselines/lar_moe_state_only.py`

The model should contain separate modules for:

- state student encoder;
- state-plus-future-action teacher encoder;
- teacher action-window decoder;
- soft LAR router;
- action-window experts;
- weighted expert aggregation.

Do not subclass `PhaseBootstrappedMoE` unless the subclass fully disables its
phase head, centroid bootstrap, top-k router, soft mapping, and expert warm-start
logic. A separate implementation is safer and makes contamination tests possible.

### 5.3 Training layer

Add separate trainers or clearly separate training modes:

- `LARPretrainTrainer` for student–teacher representation learning;
- `LARPolicyTrainer` for frozen-student soft-MoE policy learning.

The existing `Stage1Trainer` and `Stage2Trainer` encode PhaseForge assumptions:
phase classification, PhaseForge bootstrap transitions, top-k routing, and
action-plus-balance loss. Reusing them is allowed only after their assumptions
are isolated behind tested, method-specific interfaces.

### 5.4 Configuration

Add a dedicated model configuration:

`phaseforge/config/models/baselines/lar_moe_state_only.yaml`

Add dedicated training configurations for pretraining and policy training, or a
single explicit method configuration with an unambiguous stage selector. The
resolved configuration must record at least:

- `future_horizon: H`;
- `num_experts: N`;
- latent dimension;
- student and teacher encoder dimensions;
- router temperature initialization and learnability;
- all four policy-loss coefficients;
- pretraining loss aggregation rule and any coefficients;
- optimizer and scheduler;
- random seed;
- checkpoint identity;
- whether the student is frozen.

No field may inherit from `phaseforge.yaml` if that inheritance could introduce
phase supervision, centroids, top-k routing, or partial warm-starting.

## 6. Experimental protocol

### 6.1 Primary comparator

Use `N=6` for the main state-only comparator to match the canonical PhaseForge
expert count. This is a capacity-matched adaptation, not the paper’s reported
default configuration.

If an additional LAR-style expert-count study is desired, run it separately and
label it as an adaptation sensitivity study. Do not mix it into the primary
PhaseForge causal matrix.

### 6.2 Data and training fairness

Use the same task, train/validation split, normalization, seed set, checkpoint
selection rule, and frozen rollout reset bank as the primary PhaseForge study.

The LAR adaptation must not use PhaseForge phase labels during training. If phase
labels are retained for analysis, that use must be recorded as post-hoc only.

Because LAR-MoE predicts action chunks, define the evaluation rule before
training. The recommended rule is to execute the first predicted action and
advance the policy by one environment step, while reporting the action-horizon
configuration. This preserves the existing closed-loop rollout contract without
silently treating a chunk policy as a single-step policy.

### 6.3 Required LAR ablations

To verify that the implementation is actually using LAR’s mechanism, include at
least the following ablations on the same state-only protocol:

1. naive soft MoE without student freezing and without alignment regularization;
2. frozen student without alignment regularization;
3. unfrozen student with alignment regularization;
4. full state-only LAR adaptation with frozen student and all regularizers.

These are implementation-validation ablations, not replacements for PhaseForge’s
primary baselines.

## 7. Tests and acceptance criteria

### 7.1 Data tests

- Every future window has exactly (H) actions.
- No window crosses a trajectory boundary.
- The teacher receives future actions; the student does not.
- Validation and test windows are constructed independently of training windows.
- Phase fields are absent from the LAR training batch or are provably ignored.

### 7.2 Model tests

- Student output shape is `(B, latent_dim)`.
- Teacher reconstruction shape is `(B, H, action_dim)`.
- Router probabilities have shape `(B, N)` and sum to one.
- All experts receive gradients through soft aggregation.
- Student parameters receive no policy-stage gradients after freezing.
- Policy output is invariant to the presence of phase labels in the batch.
- No PhaseForge centroid, top-k, soft-mapping, or partial-warm initialization is
  invoked.

### 7.3 Loss tests

- Student–teacher latent loss decreases on a deterministic toy batch.
- Teacher action reconstruction loss decreases on a deterministic toy batch.
- Distance-consistency loss is near zero when latent and routing distances match.
- Entropy and group-sparsity terms are finite for uniform and near-one-hot routing.
- Pairwise distance computation has deterministic behavior under fixed seeds.
- The total loss is exactly the configured weighted sum of its components.

### 7.4 Protocol tests

- Resolved configuration contains no phase-training target.
- The policy-stage checkpoint records the frozen student identity.
- Inference succeeds when phase labels are removed from the batch entirely.
- Repeated runs with the same seed produce identical initialization metadata.
- A dry run produces the correct two-stage dependency graph and does not reuse a
  PhaseForge Stage-1 checkpoint.

## 8. Reporting and claim boundaries

Permitted claims:

- “We implement a state-only adaptation of the LAR-MoE training mechanism.”
- “The adaptation preserves student–teacher future-action pretraining, frozen
  student routing, soft gating, and latent-alignment regularization.”
- “The comparison isolates initialization-based privileged geometry transfer
  from unsupervised latent-alignment-based routing under a state-only protocol.”

Prohibited claims:

- “We reproduce the published LAR-MoE model.”
- “We reproduce the published LAR-MoE results.”
- “PhaseForge outperforms LAR-MoE” when LAR-MoE is not evaluated under the same
  protocol.
- “LAR-MoE is vision-dependent” as a definition of its contribution; vision is
  an input realization, while latent-aligned routing is the mechanism.
- “The adaptation is faithful” if future-action teacher training or alignment
  regularization has been removed.

The paper should call the method `lar_moe_state_only` throughout the code,
manifest, metadata, and tables. The related-work text should state that it is an
algorithmic state-only adaptation, not the original vision-language system.

## 9. Implementation order

1. Freeze (H), (N), latent dimension, output horizon, and all loss weights in
   the resolved design document.
2. Add and test deterministic future-action windows.
3. Implement and unit-test student–teacher pretraining.
4. Implement and unit-test soft routing and the three LAR regularizers.
5. Implement the frozen-student policy stage.
6. Add configuration, registry, checkpoint metadata, and manifest support.
7. Run the four required LAR mechanism ablations.
8. Run a small deterministic overfit test and a single-task pilot.
9. Audit resolved configurations for PhaseForge contamination.
10. Obtain approval before adding the method to the final multi-task experiment.

## 10. Go/no-go criteria

Proceed to implementation only if:

- the professor approves treating this as a secondary state-only adaptation;
- the future-action horizon is explicitly selected;
- the action-chunk evaluation rule is accepted;
- the method remains separate from the canonical PhaseForge implementation;
- the four required mechanism ablations fit the available experiment budget.

Stop and revise the design if the implementation requires phase labels,
PhaseForge centroid initialization, top-k routing, partial warm-starting, or a
single-step-only objective. Those changes would no longer constitute a faithful
LAR-MoE adaptation.
