# Router-initialization ablation

This protocol is a focused follow-up to the final causal matrix. It tests the
memoryless direct-action hypothesis without changing the existing final
matrix.

Manifest:

```text
experiments/router_initialization_ablation.json
```

The protocol covers `Lift`, `Can`, and `Square`. `ToolHang` and `Transport`
are intentionally excluded from this focused run because the existing matrix
does not provide a useful discriminating performance signal for them.

## Conditions

There are two shared Stage 1 providers per task:

- `precision_residual_phaseforge`: phase-aware representation provider.
- `bc`: normalized BC representation provider.

The five evaluated Stage 2 conditions are:

| Method | Purpose | Stage 1 source | Router/prototype condition |
|---|---|---|---|
| `router_init_random` | Primary initialization arm | PhaseForge | Random prototypes |
| `router_init_phase` | Primary initialization arm | PhaseForge | Centroids from `phase` |
| `router_init_topology` | Primary arm and reference | PhaseForge | Centroids from `phase_topo` |
| `representation_bc` | Representation ablation | BC | Topology centroids |
| `routing_softmax_top1` | Routing ablation | PhaseForge | Softmax top-1 with topology initialization |

All evaluated conditions use the same beta-zero direct-action expert contract,
the same phase-aware Stage 1 provider where applicable, the same Stage 2
action-loss contract, and the same three seeds. The Stage 2 margin loss is
explicitly disabled in this focused protocol. This is intentional: topology
prototype IDs are remapped from `phase_topo` and are not guaranteed to equal
the `phase` class IDs, so an active margin loss would confound initialization
with an unverified label-to-expert mapping. The topology arm is reused as the
reference for the two secondary comparisons; it is not retrained under a
duplicate name.

The primary comparison is:

```text
random prototypes  vs  phase centroids  vs  topology centroids
```

Only router initialization is allowed to vary within that comparison. In
particular, the Stage 1 checkpoint, phase-aware representation, action-loss
settings, disabled margin setting, expert initialization, beta value, and
seeds must remain identical.

## Runner command

Use a new output namespace. The runner refuses to mix this protocol with
existing artifacts:

```text
phaseforge-sweep --manifest experiments/router_initialization_ablation.json --outputs outputs_router_initialization_ablation --with-dependencies
```

The protocol expands to 108 steps:

- 18 shared-provider Stage 1 training steps,
- 45 Stage 2 training steps,
- 45 rollout evaluations.

Do not use the existing `outputs_final` directory for this run.

## Interpretation

- `router_init_topology > router_init_random` supports a topology-initialization benefit.
- `router_init_phase > router_init_random` supports a phase-initialization benefit.
- `router_init_topology > router_init_phase` supports topology labels adding value beyond phase labels.
- `router_init_topology > representation_bc` supports phase-aware representation training.
- `router_init_topology > routing_softmax_top1` supports the hard prototype routing package over the registered softmax package.

The primary result must be reported with per-seed success counts. A final
claim should not be based on one task or on a pooled episode count alone.
