# 7. Conclusion

This paper tested whether the geometry of demonstration trajectories can be used to initialize the routing partition of a mixture-of-experts policy, and whether the resulting organization predicts closed-loop control performance.

The answer to the first question is yes. Topology-derived prototype placement produces a routing partition that remains substantially more phase-aligned than random or rule-based initialization after joint fine-tuning, with NMI values of 0.51–0.67 compared with 0.07–0.09 for unstructured conditions under matched training. The initialization geometry has a persistent effect on the learned modular structure — where the prototypes start determines the character of the expert specialization that emerges.

The answer to the second question is no, or at least not uniformly. In the five-task sweep, the topology-initialized policy leads on Can (78% vs. 63% for BC), but in the matched ablation on Square, it trails the softmax control (27% vs. 35%), where the most phase-aligned prototype partition produces the lowest success rate. A more structured routing partition does not reliably produce a better controller.

The separation between these two outcomes is the main finding. Routing organization and task success respond to the same intervention — changing the initialization geometry — but they are not equivalent quantities, and treating one as a proxy for the other would produce misleading conclusions. This separation holds regardless of the specific system used to demonstrate it, and it bears on any modular policy architecture in which expert specialization is assumed to benefit from structural alignment with the task.

The result also identifies a concrete limitation of prototype-based routing: when the behaviorally relevant transitions in a task do not coincide with the kinematic regime boundaries discovered by the topology pipeline, a phase-aligned partition may assign the wrong expert to the precision-critical region. Whether this limitation can be addressed by richer trajectory representations, adaptive segmentation, or alternative routing mechanisms is an open question.
