# Trajectory-Regime Initialization for Prototype-Routed Manipulation Policies: A Matched Study of Routing Organization and Closed-Loop Outcomes

# 1. Introduction

Manipulation tasks often combine approach, contact, transport, and terminal-placement behaviors that require different local state-to-action mappings. Mixture-of-experts (MoE) policies accommodate this heterogeneity by assigning observations to specialized experts, but their performance depends on how the routing function partitions the behavioral space.

Recent robotic MoE methods incorporate structure from demonstrations through supervised phase routing, routing regularization toward learned skill representations, or semantic skill routing [Mazza et al., 2026; Rodriguez et al., 2026; Deng et al., 2026]. We study a narrower intervention: whether trajectory-derived kinematic regimes can initialize the prototype geometry of a hard-routing manipulation policy and influence its final routing organization after joint fine-tuning.

The regimes are used to construct initial prototype locations from a Stage-1 latent representation. Stage 1 is trained with action prediction and rule-derived phase supervision; trajectory-derived regimes are not themselves the source of that supervision.

To isolate initialization, we compare trajectory-derived, rule-based, and random prototype placement under matched prototype-routing conditions. We measure phase-expert alignment and routing-switch rates on validation demonstrations, and closed-loop rollout success separately.

Trajectory-derived initialization produces more phase-aligned, lower-switch routing on the validation distribution in the observed runs. Across three training seeds, closed-loop success comparisons between regime and matched control initializations are not statistically resolved.

We make three contributions:

1. **Trajectory-derived regime initialization.** We use change-point segmentation and clustering to derive trajectory regimes, then initialize MoE routing prototypes from regime-conditioned latent centroids.

2. **Matched evaluation of routing organization.** Under identical prototype-routing conditions, we show that trajectory-derived regime initialization produces higher alignment with rule-derived phase labels (NMI) and lower routing-switch rates on validation demonstrations relative to matched random and rule-based controls.

3. **Bounded outcome analysis.** In the matched Can/Square ablation, trajectory-derived initialization yields higher final offline routing organization than the matched initialization controls, while the associated closed-loop comparisons across three training seeds are not statistically resolved.
