# PhaseForge Final Baselines and Canonical-Method Migration Plan

**Status:** Proposed for professor review; no final sweep should start until this plan is approved.

**Decision requested:** Promote `phaseforge_r50` to the canonical proposed method for the final experiment and remove the default `phaseforge` method from the final evaluation protocol.

## 1. Decision

Yes. The final experiment should use `phaseforge_r50` as the only proposed PhaseForge method.

The old `phaseforge` configuration is not a required final comparator, dependency, or publication method. The final experiment and final paper must not use it as the proposed method. Previous `phaseforge` results are pre-final engineering context only; they must not be pooled with R50 results, used as final evidence, or presented as the published PhaseForge method.

If the old configuration is retained in the repository, it is only an optional archival snapshot. It must not remain on the active final execution path. Removing or archiving old files is a separate repository-cleanup decision and is not required to approve the final experimental design.

The final method identity will be:

```text
method name: phaseforge_r50
model config: phaseforge_r50
role: canonical proposed method
```

For paper-facing text, the method may be called **PhaseForge (R50)**. The code and artifact identity must remain `phaseforge_r50`; it must not be silently renamed back to the historical `phaseforge` identifier.

The published method name will be **PhaseForge**, with the exact implementation/artifact identity `phaseforge_r50`. The old name must not appear as a final method label.

## 2. Reason for the migration

The original `phaseforge` configuration is not the method selected for the final study. It uses the older 8-expert and soft-mapping configuration. The promoted R50 configuration is self-contained and explicitly specifies:

- 6 experts;
- top-2 routing;
- centroid router initialization;
- 50% partial expert warm-start;
- seed-dependent partial reinitialization;
- `soft_mapping.enabled: false`.

The R50 configuration is defined in `phaseforge/config/models/phaseforge_r50.yaml`.

The existing R50 confirmation manifest records a prior Lift engineering result of 0.56 / 0.84 / 0.72 across seeds 42 / 43 / 44, with mean 0.707. This supports selecting R50 for the final test, but it is not the final five-task result and must not be described as final validation.

The migration must be explicit. Silently changing the model behind the name `phaseforge` would mix the pre-final engineering run with the final experiment. The final experiment must use the R50 model/configuration directly.

## 3. Final baseline matrix

The final five-task experiment will contain the following methods for each task and seed.

### Primary comparison family

1. `phaseforge_r50` — proposed method.
2. `bc` — structured-state dense behavior-cloning floor.
3. `scratch_moe` — MoE trained without Stage 1 pretraining.
4. `warmstart_moe` — BC encoder, random router, warm-start experts.
5. `phase_pretrain_random_router` — phase-supervised encoder, random router, warm-start experts.
6. `plain_encoder_phase_bootstrap` — BC encoder, phase-centroid router, warm-start experts.

The five comparator methods are retained because they test the dense floor, MoE architecture, warm-starting, router initialization, and phase-supervised representation separately.

**Matching limitation that must be resolved:** the existing `warmstart_moe`, `phase_pretrain_random_router`, and `plain_encoder_phase_bootstrap` implementations use standard warm-start experts, while R50 uses 50% partial warm-start experts. Therefore, those existing cells are useful behavioral comparators, but they are not exact R50 factorial controls. They must not be described as isolating router initialization or representation effects for R50 unless matched partial-warm variants are added.

The recommended final mechanism matrix is therefore:

- keep the existing comparators for continuity and behavioral comparison;
- add R50-matched variants for the specific H1/H2 claims, with the same 6-expert, centroid/random-router, and partial-warm settings as R50;
- if those matched variants are not approved or implemented, report the existing cells as non-isolated secondary comparisons and restrict the final claim to R50 versus the registered behavioral baselines.

The R50-matched variants are implementation work, not existing methods. They must not be presented as already implemented.

### Secondary and diagnostic methods

- `bc_rnn` — temporal-history comparator. It uses the declared RNN data variant and is not a history-matched PhaseForge comparison.
- `bc_robot_only` — negative control using the robot-only observation schema.
- `teacher_forced` — privileged-training diagnostic. Ground-truth phase is used during Stage 2 training; inference uses the learned phase predictor. It must not be presented as a standard non-privileged baseline.
- Oracle routing — evaluation-time routing intervention on a fixed trained expert set. It is not a separately trained final method and must not be counted as a deployable policy.

## 4. Methods excluded from the final five-task matrix

The following remain supplementary Lift ablations or historical experiments and are not required in the final five-task matrix:

- `bc_large`;
- `pf_spherical_kmeans`;
- `pf_kmeans`;
- `pf_phase_head`;
- `pf_spherical`;
- `pf_centroid_random`;
- `pf_random_random`;
- `pf_ft`;
- `pf_k3` and `pf_k12`;
- `pf_jitter_00` and `pf_jitter_10`;
- `pf_corrupt_25`, `pf_corrupt_50`, and `pf_shuffle_control`;
- `warmstart_r50`;
- `pf_one_warm_plus_random`.

These cells answer additional initialization, capacity, noise, or expert-initialization questions. Including all of them in the final five-task matrix would expand the experiment beyond the registered primary question and increase the multiple-comparison burden.

## 5. Manifest migration

The final manifest must be changed from the current default method to R50.

For every task, the proposed-method entry must become equivalent to:

```json
{
  "name": "phaseforge_r50",
  "role": "canonical proposed method",
  "model": "phaseforge_r50",
  "stages": [1, 2],
  "stage2_source": "self",
  "evaluate": true,
  "evaluate_mode": "rollout"
}
```

All final-manifest consumers that currently identify the phase-supervised Stage 1 provider as `phaseforge` must be changed to use `phaseforge_r50` for the final protocol.

The final manifest must not contain an entry with:

```text
name = phaseforge
model = phaseforge
```

Pre-final manifests and reports must not be rewritten as if they had used R50. They are not part of the final results table and must not be silently relabeled.

## 6. Runner and checkpoint-source changes

The current runner only accepts `self`, `bc`, and `phaseforge` as explicit Stage 2 providers. The final protocol requires `phaseforge_r50` to be a valid provider identifier.

Implementation work must therefore:

1. Allow `stage2_source: "phaseforge_r50"` in the protocol validator.
2. Validate that the `phaseforge_r50` provider exists for the same task and provides Stage 1.
3. Resolve final PhaseForge Stage 1 checkpoints from the `phaseforge_r50` output tree.
4. Update final-method aliases so R50 never falls back to the old `phaseforge` Stage 1 tree.
5. Remove `phaseforge` from the active final provider/alias path. Any optional legacy support must be isolated from the final manifest and clearly marked non-final.
6. Record `phaseforge_r50` as the resolved provider in run metadata, provenance, and summaries.
7. Add tests proving that a final R50 Stage 2 run cannot auto-detect an old `phaseforge` checkpoint.

The R50 configuration is self-contained. The implementation must not recreate it through overrides on the old `phaseforge` configuration, because that could silently inherit 8-expert or soft-mapping defaults.

## 7. Checkpoint and output isolation

The final run must use a fresh output namespace. Existing old `phaseforge` and prior R50 confirmation outputs must not be reused as final-run checkpoints.

Required safeguards:

- use a fresh output directory or clean final-run registry;
- require the expected commit in checkpoint resolution;
- record the resolved model name as `phaseforge_r50`;
- record the resolved configuration hash;
- verify `num_experts=6`;
- verify `router_init.type=centroid`;
- verify `expert_init.type=partial_warm`;
- verify `expert_init.drop_rate=0.5`;
- verify `soft_mapping.enabled=false`;
- verify the checkpoint state schema and task match the requested task;
- fail closed if an old `phaseforge` checkpoint is selected.

## 8. Scientific interpretation

The final report must distinguish these claims:

- Earlier `phaseforge` results are pre-final engineering context and are excluded from the final evidence table.
- `phaseforge_r50` is the only proposed final method and will be published under the method name PhaseForge.
- The final five-task experiment tests whether R50 improves closed-loop task success over the registered controls.
- The earlier Lift R50 confirmation is a pre-final selection check, not a substitute for the five-task matrix.
- Teacher-forced and oracle results are diagnostics, not evidence of deployable privileged-free performance.
- Three seeds remain descriptive unless the approved protocol adds more seeds.

No method may be declared superior from a single seed or offline action MSE alone. Closed-loop success on the frozen paired reset bank remains the primary behavioral metric.

## 9. Validation gates before the final sweep

Before training the final matrix:

1. Validate the updated manifest with the runner parser.
2. Confirm that exactly one proposed method exists per task: `phaseforge_r50`.
3. Confirm that no final-manifest method uses `model: "phaseforge"`.
4. Confirm that all phase-supervised Stage 2 consumers resolve to the R50 Stage 1 provider.
5. Run the runner dry-run and verify the expected step count and dependency graph.
6. Run configuration-resolution tests for `phaseforge_r50`.
7. Run checkpoint-source tests that reject legacy `phaseforge` artifacts for final R50 steps.
8. Run all environment gates before any learned-policy rollout.
9. Confirm the fixed reset bank, task/environment versions, state schema, action contract, and checkpoint rule.

The final sweep must not start if any of these checks fail.

## 10. Approval boundary

This document is an implementation plan for review. Until professor approval:

- do not start the final five-task training sweep;
- do not include the old `phaseforge` method in the final experiment or publication table;
- do not relabel pre-final results as R50;
- do not merge the final-manifest migration as completed;
- do not claim that R50 has been validated on all five tasks.

After approval, implementation should proceed in this order: runner/provider support, manifest migration, tests, dry-run review, environment gates, then the final training and rollout sweep.
