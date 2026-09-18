# 6. Discussion and Limitations

The results establish a clear positive finding and a clear negative one. In the evaluated runs, topology-derived prototype initialization shapes the organization of a learned modular policy: the starting prototype placement is associated with whether the final routing partition aligns with the behavioral phase structure of the demonstrations. At the same time, a more phase-aligned partition does not consistently produce a better controller. The separation between these two outcomes is the central contribution, and this section interprets what it does and does not establish.


## 6.1 What initialization geometry controls

The ablation shows that the starting positions of routing prototypes have an observed effect on the routing partition that persists through joint fine-tuning. Topological initialization produces final NMI values of 0.51–0.67, compared with 0.07–0.09 for random and rule-based starts, despite identical training objectives and identical pre-trained representations in the matched prototype subset. The effect is consistent across the two evaluated tasks.

This persistence is not trivial. The prototypes and encoder are both trainable during Stage 2, and the optimization could in principle reorganize the partition entirely. The retained association between initialization and final routing is consistent with initialization-dependent optimization, but these experiments do not identify the underlying loss-landscape mechanism or establish the existence of a particular attraction basin.

The rule-based condition sharpens this interpretation. Phase-rule centroids are formed from the same Stage 1 latent representations as the topology centroids, but grouped using heuristic kinematic phase labels. They produce routing organization similar to random initialization in the observed runs (NMI 0.08–0.09). The contrast between the topology and rule-based conditions is therefore consistent with the possibility that the relationship between the initialization labels and the learned latent geometry matters, rather than merely whether the starting point is non-random.


## 6.2 What routing organization does not control

The Square ablation is the sharpest evidence that routing organization and task success are separable. Among the prototype-based conditions, topology initialization produces the highest NMI (0.51) and the lowest success rate (27%). The learned softmax condition, which also achieves high NMI (0.61), succeeds at 35%. On Can, by contrast, topology initialization leads on both NMI and success.

Several factors could account for the task dependence, though the current measurements do not isolate which ones are operative:

- **Phase labels may not carve Square at the right joints.** The topology discovery pipeline identifies kinematic regime boundaries from trajectory statistics. If the contact transitions that matter for peg insertion do not coincide with the statistically salient changes in the task-variable signal — for example, if the critical phase is a short alignment maneuver within a longer transport segment — then a phase-aligned partition may assign the wrong expert to the precision-critical region.

- **Routing stability may have different costs on different tasks.** The topology-initialized partition produces the lowest switch rate on both tasks (0.04 on Can, 0.07 on Square). Stable routing is desirable when each behavioral regime is well-modeled by a single expert, but if a task requires rapid adjustments near contact — fine corrections that span Voronoi boundaries — then low switch rate could mean the policy fails to recruit the appropriate local specialist.

- **The action-continuity trade-off may favor flexibility over coherence.** Hard top-1 routing produces action discontinuities at expert transitions. A more structured partition concentrates these transitions at phase boundaries. Whether that is better or worse depends on whether the task tolerates discontinuities at those boundaries or requires smooth transitions that a more flexible routing scheme can provide.

These are candidate explanations, not established mechanisms. The data show *that* routing organization and success dissociate on Square; they do not show *why*.


## 6.3 The role of end-to-end gating

The learned softmax condition achieves the highest phase-expert NMI on both tasks (0.72 Can, 0.61 Square) and the highest success on Square; on Can, its success rate ties the random prototype condition but remains below topology initialization. It does so without a topology-initialization prior. This raises a natural question about the necessity of topological seeding: if end-to-end optimization can discover a similarly organized partition, what does the initialization add?

Two observations bear on this question. First, the softmax condition uses a fundamentally different routing mechanism — a parameterized gating network rather than a fixed-form Voronoi partition — so the comparison conflates initialization with architecture. The NMI agreement may be coincidental rather than reflecting equivalent routing dynamics. Second, the softmax condition in the ablation operates without margin loss, a setting that may favor learned gating over prototype routing. Whether the comparison holds under the full training objective (including margin regularization) is untested in the ablation.

The comparison establishes that a non-prototype learned gate can produce high phase-expert NMI without topological seeding. It does not isolate initialization because the softmax condition uses a different routing mechanism, and its NMI is higher than the topology condition on both tasks. The result is therefore a diagnostic comparison, not a lower or upper bound on the effect of prototype initialization.


## 6.4 Limitations

**Statistical power.** All comparisons are based on three training seeds. The paired sign tests produce no significant results after Holm correction ($p = 1.0$ throughout Table A15). The observed differences in success rate — including the 15-percentage-point advantage on Can — are not resolved by the available seed-level sample. The results describe the observed runs, not a population-level effect.

**Task coverage.** Two of the five benchmark tasks (ToolHang and Transport) are unsolved by all methods, providing no evidence about the initialization hypothesis in high-complexity or multi-agent regimes. The informative comparisons are restricted to Can and Square, both single-arm tasks with moderate state dimensionality.

**Memoryless policy constraint.** The evaluation is conducted entirely under a memoryless policy contract — no recurrent state, no action chunking, no observation history. Whether topology-derived initialization produces different effects under history-conditioned or sequence-prediction architectures is unknown.

**Phase-label validity.** The topology discovery pipeline identifies behavioral regimes from kinematic trajectory statistics. These labels are verified for observability from instantaneous state, but their correspondence to the dynamically relevant contact transitions is assumed, not measured. The teacher-forced diagnostic uses ground-truth regime labels during Stage 2 training but phase-head predictions during rollout evaluation; its poor performance therefore reflects a training–evaluation routing mismatch and does not test an oracle routing policy.

**Unmeasured quantities.** The current evaluation does not measure contact forces, friction-cone satisfaction, grasp stability, or the physical consequences of action discontinuities at expert transitions. The action-jump metric ($\|a_t - a_{t-1}\|_2$) is a kinematic proxy; its relationship to contact-level failure modes is not established.
