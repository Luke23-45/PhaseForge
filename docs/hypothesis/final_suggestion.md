## Abstract

Mixture-of-experts policies provide a natural way to represent heterogeneous behavior by assigning different regions of the policy space to different expert networks. In manipulation, however, the resulting partition is learned together with the policy, leaving the relationship between the structure present in demonstrations and the organization of the experts largely implicit. We investigate whether this structure can instead be introduced through the initialization of the expert partition.

Our approach derives prototype locations from the topology of demonstration trajectories and uses these prototypes to initialize a hard-routing mixture-of-experts policy. The resulting experts are trained jointly, allowing the initial partition to change during optimization. We compare topology-based initialization with random and phase-based initialization under a matched training procedure, and separately examine the resulting routing structure and closed-loop behavior.

Topology-based initialization produces substantially more phase-aligned routing than random or phase-based initialization in the focused Can/Square ablation, with normalized mutual information of 0.67 and 0.51 and measured routing-switch rates of 0.04 and 0.07 for Can and Square, respectively. This structural effect does not translate uniformly into task success. The topology-initialized policy achieves 76.0% success on Can and 26.7% on Square, compared with 68.7% and 31.3% for phase initialization and 73.3% and 35.3% for the softmax control.

The results show that the geometry used to initialize an expert partition can substantially shape the organization of a learned modular policy, while that organization is not by itself sufficient to determine closed-loop performance. This separates the role of an initialization prior in structuring expert specialization from its downstream effect on control.


## 1. Introduction

A manipulation policy may encounter substantially different control regimes over the course of a single task. Approaching an object, establishing contact, transporting it, and completing the final placement can require different local mappings from state to action. Mixture-of-experts models provide a direct mechanism for representing such heterogeneity: rather than forcing a single network to explain the entire behavioral distribution with one set of parameters, they partition the policy among several experts and use a router to determine which expert is active.

The effectiveness of this decomposition depends not only on the capacity of the experts, but also on how the behavioral space is partitioned among them. A useful partition can isolate locally coherent action regimes, whereas an arbitrary partition may leave individual experts responsible for incompatible portions of the trajectory. In standard end-to-end training, this partition emerges jointly with the representation, router, and expert parameters. The demonstrations therefore contain structure that may be relevant to specialization, but the policy is not explicitly initialized to exploit it.

Demonstration trajectories provide one source of such structure. Rather than viewing the demonstrations solely as state-action pairs, we consider their organization as a geometric object and use that structure to initialize the locations of routing prototypes. The resulting prior does not prescribe the final specialization of the experts; it determines only the starting partition from which optimization proceeds.

This raises a more specific question: **to what extent does the geometry of the demonstrations influence the organization of a learned expert partition, and does a more structured partition necessarily produce a better controller?**

We examine this question using a prototype-routed mixture-of-experts policy for robot manipulation. Topology-derived prototypes are compared with phase-derived and random initializations under matched training conditions. The analysis deliberately separates two outcomes that are often conflated: the organization of the routing function and the success of the resulting closed-loop policy. Routing is evaluated through phase-expert alignment and temporal switching, while policy quality is evaluated through task success.

The central result is a separation between these quantities. Topology-derived initialization produces substantially more structured routing than the unstructured initialization conditions, yet the corresponding advantage in task success depends on the task. The result indicates that the geometry of the demonstrations can act as a meaningful prior over expert specialization, while also showing that a coherent routing structure is not synonymous with a superior control policy.
