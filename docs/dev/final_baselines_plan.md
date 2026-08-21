# PhaseForge Final Research Protocol: Canonical Method, Baselines, and Ablations

**Status:** Implementation plan for professor review

**Scope:** Five-task, state-only, closed-loop imitation-learning study on Lift, Can, Square, Tool Hang, and Transport.

## 1. Executive decision

The final paper will present one proposed method: **PhaseForge**.

The active `phaseforge` configuration will be rewritten to contain the exact implementation and logic currently defined by `phaseforge_r50`:

- six experts;
- top-2 routing;
- cosine-normalized centroid router initialization;
- 50% partial expert warm-start;
- seed-dependent, deterministic partial reinitialization;
- frozen Stage 1 encoder during Stage 2;
- soft mapping disabled.

The old 8-expert, soft-mapping `phaseforge` configuration will be hard-deleted from the active source tree. The `phaseforge_r50` configuration will be absorbed into the canonical `phaseforge` configuration and then removed as a separate active model identity. The final experiment, runner, checkpoints, reports, and publication tables will use `phaseforge` only.

This is a method replacement, not a relabeling of results. Results produced by the earlier 8-expert configuration are pre-final engineering results and are excluded from the final evidence table. The prior R50 Lift confirmation is a selection check for the implementation direction; it is not the five-task final result.

## 2. Canonical implementation migration

### 2.1 Source-of-truth configuration

The canonical file after migration will be:

```text
phaseforge/config/models/phaseforge.yaml
```

Its contents must be identical in behavior to the current:

```text
phaseforge/config/models/phaseforge_r50.yaml
```

The migration must be performed by copying the complete R50 configuration into `phaseforge.yaml`, changing only the model name from `phaseforge_r50` to `phaseforge`, and then removing the obsolete `phaseforge_r50.yaml` file. The old 8-expert `phaseforge.yaml` must not survive in any active location.

The migration must not be implemented as Hydra overrides on the old configuration. The final file must be self-contained so that obsolete defaults cannot reintroduce eight experts, soft mapping, or the previous expert initialization.

### 2.2 Required canonical settings

The resolved final configuration must satisfy all of the following:

| Component | Required value |
|---|---|
| Model target | `phaseforge.models.phase_moe.PhaseBootstrappedMoE` |
| Experts | 6 |
| Router top-k | 2 |
| Router input normalization | enabled |
| Router initialization | centroid |
| Expert initialization | partial warm-start |
| Partial drop rate | 0.5 |
| Partial-init seed | training seed |
| Partial-init jitter | 0.0 |
| Soft mapping | disabled |
| Stage 2 encoder | frozen |

The implementation must retain the shared `PhaseBootstrappedMoE` class. The class is the model implementation used by the canonical method and by initialization ablations; it is not the obsolete 8-expert method.

### 2.3 Naming and provenance

The final active identity is:

```text
method name: phaseforge
model config: phaseforge
paper name: PhaseForge
implementation lineage: R50 configuration, promoted and renamed
```

The string `phaseforge_r50` may appear only in migration documentation, the pre-final selection record, or immutable provenance describing the source of the canonical implementation. It must not appear as a final method row, final checkpoint namespace, or publication method name.

## 3. Research question and claims

The final study tests whether privileged phase information available during training can shape a latent representation and initialize an MoE router so that the resulting policy improves closed-loop manipulation success without phase labels at inference.

The causal chain is:

```text
phase supervision
  -> phase-discriminative latent geometry
  -> phase-informed router initialization
  -> autonomous expert specialization
  -> closed-loop task success
```

The primary behavioral claim is permitted only if PhaseForge improves rollout success consistently across the registered tasks and matched controls. Offline action loss, phase accuracy, NMI, routing entropy, and expert balance are mechanism diagnostics; none is a substitute for task success.

The final policy must receive only the declared structured state at inference. Ground-truth phase labels must never be supplied to the proposed method during rollout.

## 4. Final publication matrix

The final five-task matrix should distinguish ordinary baselines, causal mechanism controls, and privileged diagnostics. They must not be merged into one undifferentiated ranking.

### 4.1 Proposed method

| Method | Active identity | Description |
|---|---|---|
| PhaseForge | `phaseforge` | Phase-supervised Stage 1; centroid-initialized six-expert MoE; 50% partial expert warm-start; frozen encoder in Stage 2 |

### 4.2 Primary task baselines

These baselines belong in the main behavioral comparison table.

| Method | Manifest name | Implementation | Purpose | Matching status |
|---|---|---|---|
| BC-MLP | `bc` | `baselines/bc` | Standard structured-state behavior cloning floor | Lower-capacity dense baseline |
| BC-Large | `bc_large` | `baselines/bc_large` | Dense capacity control near the MoE total parameter count | Parameter-matched dense control |
| BC-RNN | `bc_rnn` | `baselines/bc_rnn` | History-dependent behavior cloning using the declared RNN data variant | Strong temporal comparator; not history-matched to the single-state PhaseForge policy |
| Scratch MoE | `scratch_moe` | `baselines/scratch_moe` | Tests whether the MoE architecture alone explains the result | Six experts; trainable random encoder and random experts |
| Warm-Start MoE | `warmstart_moe` | `baselines/warmstart_moe` | Tests dense-to-MoE upcycling without phase geometry | BC encoder; random router; standard copied experts |

BC-Large is required because a success difference between PhaseForge and a small BC model could otherwise be attributed to capacity. BC-RNN is required because history-dependent policies are an established strong comparator in robomimic-style manipulation. Its larger temporal model must be reported with parameter count and with the explicit limitation that it does not share PhaseForge's single-step observation contract.

### 4.3 Primary mechanism controls

These controls are required to support the causal interpretation of the proposed method. They belong in the main paper or a designated mechanism table, not in an optional appendix.

| Control | Intended contrast with PhaseForge | Required implementation |
|---|---|---|
| Phase-Pretrain Random-Router | Router initialization: phase-supervised encoder held fixed, router changed from centroid to random | R50-matched partial expert warm-start |
| Plain-Encoder Phase-Bootstrap | Representation: centroid router and expert initialization held fixed, phase supervision removed from Stage 1 | R50-matched partial expert warm-start |
| Phase-Supervised Spherical-KMeans | Privileged phase prototypes versus generic directional clustering | Same phase-supervised encoder, six experts, and R50 expert initialization |
| Phase-Supervised Euclidean-KMeans | Prototype initialization versus Euclidean activation clustering | Same phase-supervised encoder, six experts, and R50 expert initialization |
| Phase-Head Router | Prototype centroids versus discriminative phase-classifier directions | Same phase-supervised encoder, six experts, and R50 expert initialization |

The first two controls are the decisive H1/H2 comparisons. The clustering and phase-head controls test whether the proposed prior is specifically phase-structured rather than an incidental consequence of any router initialization.

The operational final manifest therefore contains nine rows per task: `phaseforge`, `bc`, `bc_robot_only`, `scratch_moe`, `warmstart_moe`, `phase_pretrain_random_router`, `plain_encoder_phase_bootstrap`, `teacher_forced`, and `bc_rnn`. The first six rows form the proposed-method and primary-comparison family; the last three are a negative control, a privileged diagnostic, and a temporal comparator, respectively.

## 5. Required implementation corrections for valid causal controls

The existing control names are present, but their current expert initialization is not fully matched to the promoted R50 method.

The current implementations of `warmstart_moe`, `phase_pretrain_random_router`, and `plain_encoder_phase_bootstrap` use standard warm-start experts. The proposed method uses 50% partial warm-start experts. Consequently, the current versions cannot be described as exact R50 factorial controls.

Before the final experiment, implement and test the following R50-matched controls:

1. `phase_pretrain_random_router`: phase-supervised encoder, random router, six experts, 50% partial warm-start.
2. `plain_encoder_phase_bootstrap`: BC encoder, centroid router, six experts, 50% partial warm-start.
3. `pf_spherical_kmeans`: phase-supervised encoder, spherical K-Means router, six experts, 50% partial warm-start.
4. `pf_kmeans`: phase-supervised encoder, Euclidean K-Means router, six experts, 50% partial warm-start.
5. `pf_phase_head`: phase-supervised encoder, phase-head router, six experts, 50% partial warm-start.

The controls may share the generic `PhaseBootstrappedMoE` implementation if the configuration and checkpoint-loading logic are made correct. They must not silently inherit the old standard-warm-start path. Each run must persist router type, expert-init type, drop rate, dropped-neuron hash, training seed, and resolved Stage 1 provider.

`plain_encoder_phase_bootstrap` is structurally different because its Stage 1 checkpoint has no phase head. It must either be extended in its own model class to accept the same expert-initialization contract or be replaced by a state-dict-compatible implementation. It must not be converted by simply loading a plain BC checkpoint into a phase-head model and ignoring missing parameters.

If the matched controls are not implemented, the final report must downgrade those comparisons to non-isolated behavioral comparisons and must not claim a clean H1, H2, H3, or H4 causal result.

## 6. Separate ablation program

The following ablations are separate from the primary baseline matrix. They should be run on Lift first and extended to other tasks only when the paper claim requires cross-task mechanism evidence.

### 6.1 Router initialization ablations

All cells in this suite use the canonical PhaseForge encoder, six experts, frozen Stage 2 encoder, and identical 50% partial expert warm-start unless the row explicitly changes that factor.

- Centroid router — canonical condition.
- Random router — router-init control.
- Random router with standard warm-start experts — legacy router/expert diagnostic represented by `pf_random_warm`; not an R50-matched H1 control.
- Spherical centroid router — effect of normalized spherical prototype averaging.
- Spherical K-Means router — generic directional clustering.
- Euclidean K-Means router — generic Euclidean clustering.
- Phase-head router — discriminative classifier directions.

### 6.2 Expert initialization ablations

All cells in this suite use the canonical phase-supervised encoder and centroid router.

- 50% partial warm-start — canonical condition.
- Full standard warm-start — all expert parameters copied from the Stage 1 action head with the declared small jitter.
- Fully random experts — no expert weights copied from the action head.
- One warm expert plus random experts — diagnostic for whether one generalist expert is sufficient.
- Partial drop-rate sweep — at minimum 0%, 25%, 50%, 75%, and 100%, with the same deterministic initialization procedure.

The existing `pf_random_random`, `pf_centroid_random`, `warmstart_r50`, and `pf_one_warm_plus_random` cells cover parts of this suite, but they must be migrated to the new canonical `phaseforge` provider and checked for exact expert-count, router, and seed parity.

`pf_random_warm` must also be retained as a separately labeled standard-warm-start router diagnostic or replaced by an R50-matched random-router/partial-warm cell. `phaseforge_e6` is not a final ablation: after the R50 migration, its six-expert centroid condition is the canonical PhaseForge condition and the duplicate diagnostic row is retired.

### 6.3 Representation and training ablations

- Phase-supervised encoder versus plain BC encoder.
- Frozen Stage 2 encoder versus controlled low-rate Stage 2 encoder fine-tuning.
- Clean phase labels versus 25%, 50%, and permutation-shuffled bootstrap labels.
- Phase-head auxiliary-loss schedule only as a predeclared secondary ablation; it must not be introduced after inspecting final results.

The existing `pf_ft`, `pf_corrupt_25`, `pf_corrupt_50`, and `pf_shuffle_control` cells are candidates for this suite, but their source provider and configuration must be changed to the canonical `phaseforge` method and their label-corruption semantics must be verified before use.

### 6.4 Capacity and routing-scale ablations

- Three experts, fewer than the six phase labels.
- Six experts, canonical condition.
- Twelve experts, more than the six phase labels.
- Top-1 versus top-2 routing only if the change is explicitly registered and applied consistently.

The existing `pf_k3` and `pf_k12` cells are usable candidates after they are converted from the old provider identity and pinned to the canonical R50 initialization. Expert count changes total parameter count and must be reported as a capacity/routing-scale ablation, not as evidence for the core causal claim.

### 6.5 Privileged and negative-control diagnostics

These rows are not ordinary baselines and must be labeled separately.

- Robot-only BC: tests whether object state is necessary.
- Teacher-Forced MoE: uses ground-truth phase to partition experts during training and a learned phase predictor at inference.
- Oracle routing: routes a fixed trained expert set with ground-truth phase during evaluation; it is an evaluation intervention, not an independently trained policy.

Teacher-Forced and Oracle results must not be included in the primary multiplicity family or described as deployable privileged-free methods.

### 6.6 Disposition of existing ablation cells

| Existing cell | Final disposition |
|---|---|
| `pf_random_warm` | Retain as a supplementary standard-warm-start random-router diagnostic, or replace with the R50-matched random-router/partial-warm control |
| `pf_random_random` | Retain as the fully random expert/router control after provider and configuration migration |
| `pf_centroid_random` | Retain as the centroid-router/random-expert control after provider and configuration migration |
| `pf_spherical` | Retain as the spherical-centroid prototype ablation |
| `pf_spherical_kmeans` | Retain as the generic directional-clustering ablation |
| `pf_kmeans` | Retain as the Euclidean-clustering ablation |
| `pf_phase_head` | Retain as the discriminative phase-head-direction ablation |
| `pf_ft` | Retain as the controlled Stage 2 encoder-fine-tuning ablation |
| `pf_k3`, `pf_k12` | Retain as the expert-count/routing-scale ablation |
| `pf_corrupt_25`, `pf_corrupt_50`, `pf_shuffle_control` | Retain as phase-bootstrap-label corruption ablations after semantics are verified |
| `pf_jitter_00`, `pf_jitter_10` | Retain only as standard-warm-start jitter diagnostics; replace with the partial-warm drop-rate sweep for R50-specific analysis |
| `pf_one_warm_plus_random` | Retain as the one-warm-expert diagnostic |
| `warmstart_r50` | Absorb into canonical `phaseforge`; do not publish as a duplicate method |
| `phaseforge_e6` | Retire; its six-expert centroid condition is superseded by canonical `phaseforge` |

## 7. Methods that are not required in the final publication matrix

The following should not be added to the main five-task sweep merely to increase the number of rows:

- duplicate historical `phaseforge` configurations;
- a separate `phaseforge_r50` publication row after migration;
- both old and R50 versions of the same proposed method;
- unregistered router or expert variants selected after observing results;
- additional vision, multitask, diffusion-policy, or VLA methods outside the declared state-only question.

The older `phaseforge` configuration is hard-deleted from the active source tree. Its pre-final results are not a final baseline and are not reproduced in the publication table.

## 8. Manifest and runner migration

### 8.1 Final manifest identity

Every task must contain exactly one proposed-method row:

```json
{
  "name": "phaseforge",
  "role": "proposed method",
  "model": "phaseforge",
  "stages": [1, 2],
  "stage2_source": "self",
  "evaluate": true,
  "evaluate_mode": "rollout"
}
```

The final manifest must contain no `phaseforge_r50` method row and no old 8-expert `phaseforge` variant.

All phase-supervised Stage 2 consumers must use:

```text
stage2_source: phaseforge
```

The provider now means the canonical R50 implementation under the final `phaseforge` identity.

### 8.2 Checkpoint resolution

The runner must:

1. Resolve all final PhaseForge Stage 1 and Stage 2 checkpoints under the `phaseforge` namespace.
2. Remove `phaseforge_r50` from active final aliases and provider resolution.
3. Resolve `phase_pretrain_random_router`, teacher-forced, clustering, phase-head, and other phase-supervised controls from the canonical `phaseforge` Stage 1 provider.
4. Pass the exact provider checkpoint path to every Stage 2 subprocess.
5. Reject checkpoints whose resolved configuration does not satisfy the six-expert, centroid, partial-warm R50 contract.
6. Persist the resolved configuration and provider identity in metadata and artifact manifests.

The old 8-expert checkpoint namespace must not be eligible for a final run. A stale checkpoint with the same filesystem name must fail closed through commit, configuration, state-schema, and model-shape checks.

### 8.3 Hard-deletion scope

After approval and before final training:

- replace `phaseforge.yaml` with the exact R50 configuration and set `name: phaseforge`;
- delete `phaseforge_r50.yaml` as an obsolete active configuration;
- remove active references to `phaseforge_r50` from manifests, aliases, and runner logic;
- remove the old 8-expert/soft-mapping configuration from source control;
- update tests, documentation, and configuration snapshots to the canonical identity;
- use a fresh final output namespace and do not reuse pre-final checkpoints.

The shared `PhaseBootstrappedMoE` implementation and shared initialization utilities are retained. They are required by the canonical method and the ablation suite and are not the obsolete method configuration.

## 9. Final evaluation protocol

The approved matrix will use:

- Lift, Can, Square, Tool Hang, and Transport as separate single-task policies;
- training seeds 42, 43, and 44 for the first complete matrix;
- 50 frozen evaluation reset cases per task and training seed;
- identical reset cases and order across methods;
- a separate evaluation bank disjoint from training, validation, and checkpoint selection;
- the predeclared best validation action-loss checkpoint rule;
- deterministic state-only inference;
- pinned simulator, MuJoCo, dataset, state schema, normalization, action convention, and horizon;
- per-episode records with failure categories and infrastructure-failure handling.

The final report must provide per-task and per-seed success rates, Wilson intervals, paired PhaseForge-minus-baseline differences on identical resets, offline action metrics, routing diagnostics, parameter counts, active capacity, training cost, and all configuration/provenance hashes.

Three seeds are descriptive. No population-level significance claim may be based on three seed means. The primary comparison family and multiplicity correction must be declared before the final run.

## 10. Interpretation rules

| Result | Permitted conclusion |
|---|---|
| PhaseForge does not exceed BC or matched MoE controls | The proposed behavioral claim is unsupported under the declared protocol |
| PhaseForge improves routing diagnostics only | Mechanism-level evidence without a manipulation-performance claim |
| PhaseForge improves success on some tasks only | Task-conditional evidence; no universal five-task claim |
| PhaseForge improves success consistently across tasks and matched controls | Evidence supporting the state-only PhaseForge claim |
| All methods fail rollout | Report training and routing diagnostics only; do not claim manipulation success |

The report must not convert a routing metric into a task-success claim, and must not use an oracle or teacher-forced result as evidence of deployable performance.

## 11. Pre-run acceptance gates

The final sweep may begin only after all gates pass:

1. The canonical `phaseforge.yaml` resolves exactly to the R50 contract.
2. `phaseforge_r50.yaml` and the old 8-expert configuration are absent from the active configuration set.
3. The final manifest contains exactly one proposed `phaseforge` method per task.
4. No final manifest row uses `model: phaseforge_r50`.
5. All phase-supervised consumers resolve their Stage 1 provider to canonical `phaseforge`.
6. R50-matched H1/H2/H3/H4 controls either pass implementation tests or are explicitly excluded from causal claims.
7. The runner dry-run has the expected dependency graph and no stale provider names.
8. Baseline and ablation configuration tests pass.
9. Simulator, reset-bank, action-contract, and native-success-predicate gates pass for every task.
10. The final output namespace is fresh and all checkpoint hashes are recorded.

## 12. References

The baseline and evaluation design follows the reproducibility and low-dimensional-policy considerations in:

- Mandlekar et al., “What Matters in Learning from Offline Human Demonstrations for Robot Manipulation,” CoRL 2021: [paper](https://arxiv.org/abs/2108.03298), [robomimic study](https://robomimic.github.io/study/).
- robomimic low-dimensional dataset documentation, including BC and BC-RNN reproduction guidance: [documentation](https://robomimic.github.io/docs/datasets/robomimic_v0.1.html).
- Komatsuzaki et al., “Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints,” ICLR 2023: [paper](https://arxiv.org/abs/2212.05055).
- Fedus et al., “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity,” JMLR 2022: [paper](https://arxiv.org/abs/2101.03961).

No external benchmark result is treated as a pass/fail target. The final claims are determined only by the declared PhaseForge protocol and its matched evaluation artifacts.
