# 6. Discussion and Limitations

The results establish a clear positive finding and a clear negative one. Topology-derived prototype initialization reliably shapes the organization of a learned modular policy: where the prototypes start determines whether the final routing partition aligns with the behavioral phase structure of the demonstrations. At the same time, a more phase-aligned partition does not reliably produce a better controller. The separation between these two outcomes is the central contribution, and this section interprets what it does and does not establish.


## 6.1 What initialization geometry controls

The ablation demonstrates that the starting positions of routing prototypes have a persistent effect on the routing partition that survives joint fine-tuning. Topological initialization produces final NMI values of 0.51–0.67, compared with 0.07–0.09 for random and rule-based starts, despite identical training objectives and identical pre-trained representations. The effect is large and consistent across both tasks.

This persistence is not trivial. The prototypes and encoder are both trainable during Stage 2, and the optimization could in principle reorganize the partition entirely. That it does not — that the final routing structure retains a signature of the initialization geometry — indicates that the loss landscape around a topology-aligned partition is locally attractive, or at least that gradient-based optimization does not easily escape the basin established by the initial placement.

The rule-based condition sharpens this interpretation. Phase-rule centroids, computed from heuristic kinematic thresholds in raw task-variable space, produce routing organization indistinguishable from random initialization (NMI 0.08–0.09). The topology-derived centroids, by contrast, are computed in the same representation space that the Stage 1 contrastive objective was trained to organize. The alignment between the initialization geometry and the latent manifold appears to be what matters — not merely having a non-random starting position.


## 6.2 What routing organization does not control

The Square ablation is the sharpest evidence that routing organization and task success are separable. Among the prototype-based conditions, topology initialization produces the highest NMI (0.51) and the lowest success rate (27%). The learned softmax condition, which also achieves high NMI (0.61), succeeds at 35%. On Can, by contrast, topology initialization leads on both NMI and success.

Several factors could account for the task dependence, though the current measurements do not isolate which ones are operative:

- **Phase labels may not carve Square at the right joints.** The topology discovery pipeline identifies kinematic regime boundaries from trajectory statistics. If the contact transitions that matter for peg insertion do not coincide with the statistically salient changes in the task-variable signal — for example, if the critical phase is a short alignment maneuver within a longer transport segment — then a phase-aligned partition may assign the wrong expert to the precision-critical region.

- **Routing stability may have different costs on different tasks.** The topology-initialized partition produces the lowest switch rate on both tasks (0.04 on Can, 0.07 on Square). Stable routing is desirable when each behavioral regime is well-modeled by a single expert, but if a task requires rapid adjustments near contact — fine corrections that span Voronoi boundaries — then low switch rate could mean the policy fails to recruit the appropriate local specialist.

- **The action-continuity trade-off may favor flexibility over coherence.** Hard top-1 routing produces action discontinuities at expert transitions. A more structured partition concentrates these transitions at phase boundaries. Whether that is better or worse depends on whether the task tolerates discontinuities at those boundaries or requires smooth transitions that a more flexible routing scheme can provide.

These are candidate explanations, not established mechanisms. The data show *that* routing organization and success dissociate on Square; they do not show *why*.


## 6.3 The role of end-to-end gating

The learned softmax condition achieves the highest phase-expert NMI on both tasks (0.72 Can, 0.61 Square) and ties for the highest or achieves the highest success — without any initialization prior. This raises a natural question about the necessity of topological seeding: if end-to-end optimization can discover a similarly organized partition, what does the initialization add?

Two observations bear on this question. First, the softmax condition uses a fundamentally different routing mechanism — a parameterized gating network rather than a fixed-form Voronoi partition — so the comparison conflates initialization with architecture. The NMI agreement may be coincidental rather than reflecting equivalent routing dynamics. Second, the softmax condition in the ablation operates without margin loss, a setting that may favor learned gating over prototype routing. Whether the comparison holds under the full training objective (including margin regularization) is untested in the ablation.

What the comparison does establish is a lower bound on what end-to-end optimization can achieve without structural priors. The topology-derived initialization exceeds this bound in routing organization on Can (NMI 0.67 vs 0.72 for softmax, but with prototype rather than learned routing), though not on Square or in task success.


## 6.4 Limitations

**Statistical power.** All comparisons are based on three training seeds. The paired sign tests produce no significant results after Holm correction ($p = 1.0$ throughout Table A15). The observed differences in success rate — including the 15-percentage-point advantage on Can — cannot be distinguished from seed-level variability at this sample size. The results describe the observed runs, not a population-level effect.

**Task coverage.** Two of the five benchmark tasks (ToolHang and Transport) are unsolved by all methods, providing no evidence about the initialization hypothesis in high-complexity or multi-agent regimes. The informative comparisons are restricted to Can and Square, both single-arm tasks with moderate state dimensionality.

**Memoryless policy constraint.** The evaluation is conducted entirely under a memoryless policy contract — no recurrent state, no action chunking, no observation history. Whether topology-derived initialization produces different effects under history-conditioned or sequence-prediction architectures is unknown.

**Phase-label validity.** The topology discovery pipeline identifies behavioral regimes from kinematic trajectory statistics. These labels are verified for observability from instantaneous state, but their correspondence to the dynamically relevant contact transitions is assumed, not measured. The teacher-forced diagnostic — in which ground-truth phase routing produces catastrophic failure — demonstrates that the phase labels as currently defined do not correspond to an independently useful routing partition at test time, though this may reflect the distribution shift between training-time and oracle routing rather than the labels themselves.

**Unmeasured quantities.** The current evaluation does not measure contact forces, friction-cone satisfaction, grasp stability, or the physical consequences of action discontinuities at expert transitions. The action-jump metric ($\|a_t - a_{t-1}\|_2$) is a kinematic proxy; its relationship to contact-level failure modes is not established.
