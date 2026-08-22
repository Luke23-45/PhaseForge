# `lar_moe_state_only` Implementation Plan

**Status:** pre-implementation design; no code changes authorized by this document
**Revision:** 2 (2026-08-22) — cross-checked line-by-line against the full paper
text (arXiv:2603.08476v1). Revision 1 gaps fixed: expert architecture made a
required mechanism (R4), expert conditioning path pre-registered (§4.4),
temperature init pinned to the paper's value, optimizer pinned to AdamW,
pretraining stop-gradient semantics recorded, entropy sign semantics stated,
group-sparsity grid semantics fixed for N=6, hyperparameter provenance and
tuning-parity rule added (no official implementation exists as of 2026-08-22).
**Purpose:** define a faithful state-only adaptation of LAR-MoE before implementation
**Primary source:** Rodriguez et al., "LAR-MoE: Latent-Aligned Routing for
Mixture of Experts in Robotic Imitation Learning," arXiv:2603.08476,
<https://arxiv.org/abs/2603.08476> (IROS 2026 submission). All equation
references below are to that paper.

## 0. Verified paper facts this plan is built on

| Fact | Paper reference |
|---|---|
| Two stages: (1) unsupervised skill discovery via student–teacher co-training on future actions; (2) policy learning with routing regularized onto the learned latent structure | Abstract, §III |
| Student ẑ_t = φ_s(o_t) — current observation only; teacher z_t = φ_t(o_t, a_{t:t+H}) — observation + future action chunk | §III-A, Eq. (1) |
| L_s = MSE(ẑ_t, z_t); teacher decoder ψ reconstructs the chunk, L_t = MSE(â_{t:t+H}, a_{t:t+H}) | §III-A, Eqs. (1)–(2) |
| Stage 2 freezes the student; router p_t = softmax(T · MLP(ẑ_t)); **T learnable, initialized to 100** | §III-B, Eq. (3) |
| N action experts ψ_n are **transformer decoders following the simple architecture of ACT, with learned embedding tokens, predicting the whole action chunk at one step** | §III-B, Fig. 2 |
| Each expert receives a context vector c_t (paper: EdgeNeXt18 visual encodings ⊕ frozen MiniLM-L6 language); final output â = Σ_n (p_t)_n ψ_n(c_t) | §III-B, Eq. (4) |
| L_MSE over the predicted chunk; distance consistency over the batch: cosine-distance matrices of latents and of routing distributions, L_DC = (1/B²)‖D(Z) − D(P)‖²_F | §III-C1, Eq. (5) |
| Entropy regularization L_H = −Σ_n (p_t)_n log(p_t)_n to encourage specialization (adding +λ_H·L_H to the minimized objective MINIMIZES entropy → sharpens routing) | §III-C2 |
| Group-sparse regularization L_G = Σ_{i,j} sqrt((F_σ ∗ reshape(p_t)²)_{ij}) — Gaussian lowpass over the routing probabilities reshaped into an approximately square expert grid (kang2025) | §III-C3, Fig. 3 |
| Total: L = L_MSE + λ_DC L_DC + λ_H L_H + λ_G L_G; optimizer AdamW | §III-C |
| Paper default configuration: **16 experts** (LAR-MoE16); ablations toggle student freezing (F) and regularization (R); 3 seeds | §IV, Fig. 4 |
| **No official code repository exists** (searched 2026-08-22); λ weights, σ, chunk length H, latent/MLP dimensions are not stated in the paper | — |

## 1. Decision and naming

Implementing the method without vision is technically feasible. It must be
named `lar_moe_state_only` and described as a **state-only adaptation**, not
as an exact reproduction of the published vision-language LAR-MoE.

The adaptation preserves LAR-MoE's algorithmic mechanism:

1. student–teacher representation pretraining using future actions;
2. frozen student representation during policy training;
3. soft expert routing from the student latent;
4. latent–routing distance-consistency regularization;
5. entropy and group-sparsity regularization;
6. **ACT-style transformer-decoder action experts with learned embedding
   tokens producing the full action chunk per forward pass.**

Only the observation encoder is changed from the published visual-language
input to the repository's structured state input (the language stream is
dropped entirely: our tasks are single-task policies with no instruction
input). PhaseForge-specific phase labels, phase-centroid initialization,
top-k routing, and partial expert warm-starting are excluded from this
method.

## 2. Scientific role

`lar_moe_state_only` is a secondary mechanism-level comparator for PhaseForge.
It must not replace the primary PhaseForge baselines or be merged into the
PhaseForge ablation factorial.

The comparison is:

| Method | Representation source | Routing mechanism |
|---|---|---|
| PhaseForge | Phase-supervised state representation | One-time phase-geometry transfer into a sparse router |
| `lar_moe_state_only` | Unsupervised state–future-action representation | Continuous latent–routing alignment during policy training |

The method must not consume phase labels in pretraining, policy training,
checkpoint selection, or inference. Phase labels may be retained in evaluation
artifacts only for post-hoc diagnostics, never as an input or training target.

## 3. Fidelity boundary

### 3.1 Required LAR-MoE mechanisms (R1–R8)

- **R1** A **student encoder** receives only the current state (s_t).
- **R2** A **teacher encoder** receives the current state and a future action
  window (a_{t:t+H−1}, H actions — the paper's a_{t:t+H} notation is
  H-inclusive; we fix the convention to H actions and record it) during
  pretraining; a teacher decoder reconstructs the window from the teacher
  latent.
- **R3** The student latent is trained to match the teacher latent (MSE);
  the teacher is trained to reconstruct the action window (MSE).
- **R4** The **N action experts are transformer decoders following the
  simple architecture of ACT**: a stack of transformer-decoder layers that
  cross-attend to an encoded context and process **H learned embedding
  (query) tokens**, one per predicted action step, emitting the whole action
  chunk in one forward pass. **No CVAE / style latent variable** — the paper
  follows ACT's simple (deterministic) decoder, not its variational variant.
  MLP experts are NOT a faithful adaptation; if they are ever chosen for
  compute reasons, the method must be renamed (e.g. `lar_moe_mlp_state`).
- **R5** The student encoder is frozen before policy training; the router
  computes a soft distribution over ALL experts from the frozen student
  latent, p_t = softmax(T · MLP(ẑ_t)), with **T learnable, initialized to
  100** (paper value). There is no top-k selection.
- **R6** Each expert receives the observation context c_t (see §4.4) and the
  policy output is the probability-weighted combination of expert chunk
  outputs.
- **R7** Policy training minimizes
  L = L_MSE(chunk) + λ_DC L_DC + λ_H L_H + λ_G L_G with the paper's
  definitions (§4.3), optimized with **AdamW**.
- **R8** The teacher network is **discarded after pretraining** — inference
  uses only the frozen student, router, and experts.

### 3.2 Explicitly excluded PhaseForge mechanisms

The following must not be used in `lar_moe_state_only`:

- phase classification loss;
- phase-centroid or phase-prototype router initialization;
- `TopKRouter` sparse top-2 dispatch;
- `partial_warm`, `warmstart`, `one_warm`, or action-head expert copying;
- PhaseForge's soft phase-to-expert mapping;
- PhaseForge's load-balancing loss as a substitute for LAR's regularizers;
- privileged phase labels as a teacher or routing target.

Adding any of these would create a hybrid method and require a different name.

### 3.3 What "without vision" changes

The following published components are replaced by state-compatible
equivalents:

- visual-language observation encoder → structured-state MLP encoder (the
  student; a separate state tower feeds the teacher);
- visual-language context inside each action expert → the state context c_t
  (§4.4; the MiniLM language stream is dropped — single-task protocol);
- the visual implementation of the transformer-decoder experts and their
  token queries is unchanged in structure (R4); only the context encoding
  input changes.

The future-action teacher signal and the action-chunk objective must remain.
If future actions are removed and the method becomes a single-step state
policy, it is no longer a faithful LAR-MoE adaptation; it becomes only
LAR-inspired. The same is true if the transformer-decoder experts (R4) are
replaced by per-step MLPs.

## 4. Mathematical specification

Let (s_t) be the current normalized state and let
(A_t = a_{t:t+H−1}) be a valid future action window of H actions from the
same trajectory.

### 4.1 Representation pretraining

The student produces ẑ_t = f_s(s_t). The teacher receives the privileged
future-action window: z_t = f_t(s_t, A_t). The teacher decoder reconstructs
the window: Â_t = g(z_t). The paper's two pretraining losses:

- L_student = MSE(ẑ_t, z_t)  (paper Eq. 1)
- L_teacher = MSE(Â_t, A_t)  (paper Eq. 2)

**Recorded adaptation choices (unspecified by the paper; no official code):**

- **AC1 — stop-gradient:** z_t is treated as a target in L_student
  (stop-gradient through the teacher in the student loss). The teacher is
  updated only by L_teacher. This is the standard distillation reading of
  "co-training" and avoids degenerate collapse where the teacher chases the
  student.
- **AC2 — joint optimization:** both losses are optimized simultaneously in
  one stage-1 run on the same batches (L = L_student + L_teacher, unit
  weights; the paper never combines them into one weighted sum).
- **AC3 — teacher disposal:** the teacher and its decoder are deleted at the
  end of stage 1 (R8).

### 4.2 Policy routing

After pretraining, freeze f_s. The router computes
p_t = softmax(T · MLP(ẑ_t)), with T learnable and **initialized to 100**
(paper §III-B). Routing is soft over all N experts; there is no top-k
selection and no load-balancing auxiliary loss.

### 4.3 Policy objective

For a batch of student latents Z = {ẑ_t} and routing distributions
P = {p_t}, compute pairwise cosine-distance matrices
D^(Z)_{ij} = 1 − cos(ẑ_i, ẑ_j) and D^(P)_{ij} = 1 − cos(p_i, p_j). The
distance-consistency term (paper Eq. 5):

L_DC = (1/B²) ‖D^(Z) − D^(P)‖²_F.

Entropy regularization (paper §III-C2): L_H = −Σ_n (p_t)_n log (p_t)_n
summed over the batch. **Sign semantics:** with +λ_H L_H inside the
minimized objective, training MINIMIZES routing entropy — this is the
intended specialization pressure; do not flip the sign.

Group-sparse regularization (paper §III-C3, Fig. 3, after kang2025):
L_G = Σ_{i,j} sqrt((F_σ ∗ reshape(p_t²))_{ij}) — each sample's squared
routing distribution is reshaped into an approximately square expert grid
(**for N=6: a fixed 2×3 grid in expert-index order 0..5**), convolved with a
Gaussian lowpass F_σ, square-rooted, and summed. **The grid ordering is a
pre-registered constant** — identical across seeds, ablations, and runs.
σ is an adaptation hyperparameter (paper does not state it).

The policy objective is:

L_policy = L_MSE + λ_DC L_DC + λ_H L_H + λ_G L_G,

optimized with **AdamW** (paper). L_MSE is the MSE of the predicted action
chunk versus the demonstrated chunk.

### 4.4 Expert conditioning path (pre-registered decision)

The paper routes with the frozen student latent ẑ_t but conditions experts
on a context c_t (visual + language encodings); whether the expert context
encoder is the frozen student itself or a separate tower is not stated
explicitly. State-only adaptation decision:

- **Primary (faithful, pre-registered): c_t = ẑ_t** — the frozen student
  latent is the single observation representation: it routes AND conditions
  the experts (the paper's language stream concatenation disappears with the
  language modality). This is the strictest reading of "frozen student",
  maximizes parameter efficiency (the paper's stated strength), and makes
  L_DC semantically coherent: routing distances are aligned with the very
  features the experts consume.
- **Recorded deviation (only if the pilot is degenerate):** a separate
  trainable state encoder producing c_t. Choosing this requires renaming to
  `lar_moe_state_only_tctx` and disclosing that "frozen" applies only to the
  routing pathway. It must not be switched to mid-experiment.

## 5. Repository changes required

This is not a configuration-only baseline. The current main data contract
uses `sequence_length: 1`, and the existing `MoELayer` rejects sequence-aware
latents. The implementation must therefore be isolated from the PhaseForge
top-k path.

### 5.1 Data layer

Add a dedicated future-window dataset or dataset mode, for example:
`phaseforge/data/common/future_action_dataset.py`

Required behavior:

- return (s_t), (A_t), trajectory identifier, and source position;
- construct only complete windows of length H (paper convention fixed to H
  actions; record the off-by-one resolution of the paper's a_{t:t+H} notation);
- never cross trajectory boundaries or train/validation boundaries;
- preserve the existing state/action normalization contract;
- retain no phase field as a training input for LAR-MoE;
- expose deterministic indexing and seed-independent window construction;
- provide a validity test for every returned window.

Add per-task data configs (e.g. `lift_lar`, …) selecting the future-window
mode and H, registered in the runner's `_VALID_DATA` set. Do not change the
existing PhaseForge single-step contract merely to support this baseline.
The existing temporal collator may be reused only after verifying that its
output shape is appropriate for the new model.

### 5.2 Model layer

Add a separate model implementation, for example:
`phaseforge/models/baselines/lar_moe_state_only.py`

The model should contain separate modules for:

- state student encoder;
- state-plus-future-action teacher encoder (stage 1 only; discarded after);
- teacher action-window decoder (stage 1 only);
- soft LAR router (MLP + learnable temperature init 100);
- N transformer-decoder action experts with H learned embedding tokens each
  (R4), consuming the context c_t;
- weighted soft aggregation of expert chunks.

Do not subclass `PhaseBootstrappedMoE` unless the subclass fully disables its
phase head, centroid bootstrap, top-k router, soft mapping, and expert
warm-start logic. A separate implementation is safer and makes contamination
tests possible.

### 5.3 Training layer

Add separate trainers or clearly separate training modes:

- `LARPretrainTrainer` for student–teacher representation learning (AC1–AC3);
- `LARPolicyTrainer` for frozen-student soft-MoE policy learning with the
  four-term objective (AdamW).

The existing `Stage1Trainer` and `Stage2Trainer` encode PhaseForge
assumptions: phase classification, PhaseForge bootstrap transitions, top-k
routing, and action-plus-balance loss. Reusing them is allowed only after
their assumptions are isolated behind tested, method-specific interfaces.

### 5.4 Configuration

Add a dedicated model configuration:
`phaseforge/config/models/baselines/lar_moe_state_only.yaml`

Add dedicated training configurations for pretraining and policy training, or
a single explicit method configuration with an unambiguous stage selector.
The resolved configuration must record at least:

- `future_horizon: H` (default proposal: 16, the ACT-chunk convention;
  final value pre-registered before training);
- `num_experts: N`;
- latent dimension; student/teacher encoder dimensions; transformer-decoder
  depth/width/heads; number of query tokens (= H);
- router temperature initialization (100) and learnability;
- all four policy-loss coefficients (λ_DC, λ_H, λ_G, and the implicit 1.0 on
  L_MSE) **and their provenance** (see §6.4);
- group-sparsity σ and the expert-grid shape (2×3, index order fixed);
- pretraining loss aggregation rule (AC1–AC3);
- optimizer (AdamW) and scheduler;
- random seed; checkpoint identity; whether the student is frozen;
- the expert-context decision (§4.4 primary).

No field may inherit from `phaseforge.yaml` if that inheritance could
introduce phase supervision, centroids, top-k routing, or partial
warm-starting.

## 6. Experimental protocol

### 6.1 Primary comparator

Use **N=6** for the main state-only comparator to match the canonical
PhaseForge expert count (the paper's default is 16). This is a
capacity-matched adaptation, not the paper's reported configuration, and is
disclosed as such.

If an additional LAR-style expert-count study is desired, run it separately
and label it an adaptation sensitivity study. Do not mix it into the primary
PhaseForge causal matrix.

### 6.2 Data and training fairness

Use the same task, train/validation split, normalization, seed set,
checkpoint-selection rule (best validation action loss on the chunk MSE for
stage 2), and frozen rollout reset bank as the primary PhaseForge study.

The LAR adaptation must not use PhaseForge phase labels during training. If
phase labels are retained for analysis, that use must be recorded as
post-hoc only.

Because LAR-MoE predicts action chunks, the evaluation rule is fixed before
training: **receding horizon — execute the first predicted action of each
chunk and advance one environment step**, reporting the action-horizon
configuration. No temporal ensemble (the paper does not use one). The
comparator row must disclose, like BC-RNN: chunk policy (temporal-output
advantage over the single-step PhaseForge contract) and the parameter count.

### 6.3 Required LAR ablations (mirror the paper's ±F/±R, Fig. 4a)

To verify that the implementation is actually using LAR's mechanism, include
at least the following ablations on the same state-only protocol:

1. naive soft MoE without student freezing and without alignment
   regularization (−F −R);
2. frozen student without alignment regularization (+F −R);
3. unfrozen student with alignment regularization (−F +R);
4. full state-only LAR adaptation with frozen student and all regularizers
   (+F +R).

These are implementation-validation ablations, not replacements for
PhaseForge's primary baselines.

### 6.4 Hyperparameter provenance and tuning parity

The paper does not publish λ_DC, λ_H, λ_G, σ, H, or the dimension choices,
and **no official implementation exists (verified 2026-08-22)**. Every such
value is therefore an adaptation choice recorded in the resolved config with
its provenance. To keep the comparison fair in both directions:

- pre-register the values BEFORE the comparator run and freeze them;
- give the adaptation the **same hyperparameter search budget** the
  PhaseForge method received during its development (a documented, bounded
  pilot on Lift only — e.g. one small grid over the three λ's at three
  orders of magnitude — then frozen for all tasks and seeds);
- record the searched grid and the selected point in the run metadata.

## 7. Tests and acceptance criteria

### 7.1 Data tests

- Every future window has exactly H actions.
- No window crosses a trajectory boundary.
- The teacher receives future actions; the student does not.
- Validation and test windows are constructed independently of training
  windows.
- Phase fields are absent from the LAR training batch or are provably
  ignored.

### 7.2 Model tests

- Student output shape is (B, latent_dim).
- Teacher reconstruction shape is (B, H, action_dim).
- Router probabilities have shape (B, N) and sum to one; temperature is
  learnable and initialized to exactly 100.
- Each expert emits (B, H, action_dim) via H learned query tokens;
  the transformer-decoder structure is asserted (attention modules present;
  no CVAE components).
- All experts receive gradients through soft aggregation.
- Student parameters receive no policy-stage gradients after freezing
  (verified on a backward pass).
- The teacher receives no gradient from the student loss (AC1 stop-gradient)
  and is absent from the stage-2 state_dict (AC3).
- Policy output is invariant to the presence of phase labels in the batch.
- No PhaseForge centroid, top-k, soft-mapping, or partial-warm initialization
  is invoked (contamination audit on the resolved config).

### 7.3 Loss tests

- Student–teacher latent loss decreases on a deterministic toy batch.
- Teacher action reconstruction loss decreases on a deterministic toy batch.
- Distance-consistency loss is exactly zero when latent and routing distance
  matrices coincide, and matches a hand-computed value on a fixed toy batch.
- Entropy term decreases the total loss when routing sharpens (sign check).
- Group-sparsity term is finite for uniform and near-one-hot routing, and
  its reshape uses the fixed 2×3 expert grid (shape assertion).
- Pairwise distance computation is deterministic under fixed seeds.
- The total loss is exactly the configured weighted sum of its components.

### 7.4 Protocol tests

- Resolved configuration contains no phase-training target.
- The policy-stage checkpoint records the frozen student identity.
- Inference succeeds when phase labels are removed from the batch entirely.
- Repeated runs with the same seed produce identical initialization metadata
  (including the temperature init and expert grid ordering).
- A dry run produces the correct two-stage dependency graph and does not
  reuse a PhaseForge Stage-1 checkpoint (`resolve_checkpoint_source` maps
  `lar_moe_state_only` to itself; no alias into the phaseforge tree).

## 8. Reporting and claim boundaries

Permitted claims:

- "We implement a state-only adaptation of the LAR-MoE training mechanism."
- "The adaptation preserves student–teacher future-action pretraining,
  frozen-student routing, soft gating with a learnable temperature,
  ACT-style transformer-decoder chunk experts, and latent-alignment
  regularization."
- "The comparison isolates initialization-based privileged geometry transfer
  from unsupervised latent-alignment-based routing under a state-only
  protocol."

Prohibited claims:

- "We reproduce the published LAR-MoE model" (the observation modality,
  expert count, and several hyperparameters differ).
- "We reproduce the published LAR-MoE results."
- "PhaseForge outperforms LAR-MoE" when LAR-MoE is not evaluated under the
  same protocol.
- "LAR-MoE is vision-dependent" as a definition of its contribution; vision
  is an input realization, while latent-aligned routing is the mechanism.
- "The adaptation is faithful" if any of the following were removed:
  future-action teacher training, alignment regularization,
  transformer-decoder chunk experts, or soft routing.

The paper should call the method `lar_moe_state_only` throughout the code,
manifest, metadata, and tables. The related-work text should state that it
is an algorithmic state-only adaptation, not the original vision-language
system, and should cite the published results (95.2% LIBERO average, 150M
parameters; surgical task matching the supervised MoE baseline without phase
annotations) as context, never as comparators.

## 9. Implementation order

1. Freeze H, N, latent dimension, expert architecture dims, all loss weights,
   σ, and the expert grid ordering in the resolved design document (§6.4
   provenance rule).
2. Add and test deterministic future-action windows.
3. Implement and unit-test student–teacher pretraining (AC1–AC3).
4. Implement and unit-test soft routing (T₀=100), the transformer-decoder
   experts (R4), and the three LAR regularizers.
5. Implement the frozen-student policy stage (§4.4 primary context path).
6. Add configuration, registry, checkpoint metadata, and manifest support.
7. Run the four required LAR mechanism ablations (±F/±R).
8. Run a small deterministic overfit test and a single-task Lift pilot
   (including the bounded λ search of §6.4).
9. Audit resolved configurations for PhaseForge contamination.
10. Obtain approval before adding the method to the final multi-task
    experiment.

## 10. Go/no-go criteria

Proceed to implementation only if:

- the professor approves treating this as a secondary state-only adaptation;
- the future-action horizon H is explicitly selected;
- the action-chunk evaluation rule (receding horizon, first action, no
  temporal ensemble) is accepted;
- the expert-context decision (§4.4 primary) is accepted;
- the method remains separate from the canonical PhaseForge implementation;
- the four required mechanism ablations and the bounded tuning pilot fit the
  available experiment budget.

Stop and revise the design if the implementation requires phase labels,
PhaseForge centroid initialization, top-k routing, partial warm-starting, a
single-step-only objective, or per-step MLP experts. Those changes would no
longer constitute a faithful LAR-MoE adaptation.
