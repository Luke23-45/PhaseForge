# 1. Introduction

A manipulation policy may encounter substantially different control regimes over the course of a single task. Approaching an object, establishing contact, transporting it, and completing the final placement can require different local mappings from state to action. Mixture-of-experts models provide a direct mechanism for representing such heterogeneity: rather than forcing a single network to explain the entire behavioral distribution with one set of parameters, they partition the policy among several experts and use a router to determine which expert is active.

The effectiveness of this decomposition depends not only on the capacity of the experts, but also on how the behavioral space is partitioned among them. A useful partition can isolate locally coherent action regimes, whereas an arbitrary partition may leave individual experts responsible for incompatible portions of the trajectory. In standard end-to-end training, this partition emerges jointly with the representation, router, and expert parameters. The demonstrations therefore contain structure that may be relevant to specialization, but the policy is not explicitly initialized to exploit it.

Demonstration trajectories provide one source of such structure. Rather than viewing the demonstrations solely as state-action pairs, we consider the kinematic regimes that can be extracted from their temporal organization and use those regimes to initialize the locations of routing prototypes. The resulting prior does not prescribe the final specialization of the experts; it sets the starting partition from which optimization proceeds.

This raises a more specific question: **to what extent does the regime structure of the demonstrations influence the organization of a learned expert partition, and does a more structured partition necessarily produce a better controller?**

We examine this question using a prototype-routed mixture-of-experts policy for robot manipulation with a fixed expert count $K = E = 6$ across all tasks. Trajectory-derived regime prototypes are compared with phase-derived and random initializations under matched training conditions. The analysis deliberately separates two outcomes that are often conflated: the organization of the routing function and the success of the resulting closed-loop policy. Routing organization is evaluated through phase-expert alignment (NMI) and routing-switch rates on held-out validation demonstrations, while policy quality is evaluated through closed-loop rollout success.

The central result is an empirical dissociation between these quantities in the evaluated setting. Trajectory-derived regime initialization produces substantially more structured routing than the unstructured initialization conditions, yet the corresponding advantage in task success depends on the task. The result indicates that the kinematic regime structure of the demonstrations can act as a meaningful prior over expert routing organization, while also showing that a coherent routing partition is not synonymous with a superior controller.

In summary, this work provides three contributions:

1. **Trajectory-derived regime initialization.** A procedure that extracts discrete kinematic regimes from demonstration trajectories via change-point segmentation and clustering, and uses the resulting cluster centers to initialize routing prototypes in a mixture-of-experts policy with a fixed expert count ($K = E = 6$).

2. **Controlled evaluation of routing organization.** Empirical evidence, measured on held-out validation demonstrations, that trajectory-derived prototype initialization increases routing alignment with rule-derived phase labels (NMI) and is associated with lower routing-switch rates relative to random and rule-based controls under matched training conditions.

3. **Empirical dissociation from closed-loop success.** A controlled causal ablation across Can and Square demonstrating that more structured, phase-aligned routing partitions do not consistently translate to higher closed-loop task success, establishing that offline partition coherence and closed-loop control quality are empirically separable in this setting.
