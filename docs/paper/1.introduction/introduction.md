# How Does Trajectory Regime Initialization Shape MoE Routing? An Empirical Study of Policy Specialization

# 1. Introduction

Manipulation tasks often combine approach, contact, transport, and terminal-placement behaviors that require different local state-to-action mappings. Mixture-of-experts (MoE) policies accommodate this heterogeneity by assigning observations to specialized experts, but their performance depends on how the routing function partitions the behavioral space. Standard end-to-end training learns this partition jointly with the representation and experts, without explicitly using demonstration structure to set its initial geometry.

We investigate whether trajectory-derived kinematic regimes can provide such an initialization prior. We segment demonstrations, cluster the resulting regimes, and initialize routing prototypes from the corresponding Stage-1 latent centroids before joint fine-tuning. This poses two questions: does regime initialization change routing organization, and does a more organized partition predict closed-loop performance?

To isolate initialization, we compare trajectory-derived, rule-based, and random prototype placement under matched prototype-routing conditions. We measure phase-expert alignment and routing-switch rates on validation demonstrations, and closed-loop rollout success separately.

Trajectory-derived initialization produces more phase-aligned, lower-switch routing on the validation distribution in the observed runs. Its relationship to task success differs by task: it leads the matched prototype arms on Can but trails them on Square. Thus, routing organization and closed-loop control quality are empirically dissociated in the evaluated setting.

We make three contributions:

1. **Trajectory-derived regime initialization.** We use change-point segmentation and clustering to derive trajectory regimes, then initialize MoE routing prototypes from regime-conditioned latent centroids.

2. **Matched evaluation of routing organization.** Under identical prototype-routing conditions, we show that trajectory-derived regime initialization produces higher alignment with rule-derived phase labels (NMI) and lower routing-switch rates on validation demonstrations relative to matched random and rule-based controls.

3. **Routing--control dissociation.** On Can and Square, we find that higher offline routing organization does not consistently coincide with higher closed-loop success.
