# 5. Results

The evaluation separates two quantities: how the initialization geometry organizes the routing partition, and whether that organization translates into closed-loop task success. We report the five-task benchmark sweep first, then the matched ablation that isolates the initialization prior.


## 5.1 Five-Task Benchmark

Table 1 reports rollout success rates across all deployable methods. Three task-level patterns structure the comparison.

**Lift is saturated.** All methods except Scratch MoE and Factorial Floor reach 100% success. Lift provides no discriminative signal for distinguishing modular architectures.

**Can separates methods.** PhaseForge achieves 78% [71, 84], compared with 63% [55, 70] for BC, 69% [61, 76] for Softmax Top-1, and 62% [54, 69] for Phase-Random. The paired difference against BC is $\Delta = +0.153$, though the per-seed standard deviation is 0.192 — BC seed 42 succeeds at 78% while seed 44 drops to 42% — and the Holm-adjusted sign test does not reject the null ($p = 1.0$; Table A15). PhaseForge also outperforms Static Rule ($\Delta = +0.253$, std 0.031) and Factorial Floor ($\Delta = +0.420$, std 0.159) by wider margins.

**Square does not follow the same ranking.** Plain Encoder leads at 50% [42, 58], followed by PhaseForge at 41% [33, 49] and Softmax Top-1 and Phase-Random at 39% each. The paired difference between PhaseForge and Plain Encoder is *negative*: $\Delta = -0.093$ (std 0.058). A method that omits phase structuring entirely outperforms the proposed method on this task.

**ToolHang and Transport are unsolved.** Every method, including PhaseForge, scores 0% on ToolHang. Transport yields at most one successful episode across 150 trials for any method. These tasks lie beyond the capability frontier of the memoryless policies tested here and provide no evidence for or against the initialization hypothesis.

The five-task sweep thus reveals a task-dependent pattern: PhaseForge's advantage concentrates on Can, where it leads all comparators, while Square — a contact-rich task with tighter tolerances — does not favor the topology-initialized partition. Macro-averages (PhaseForge 0.44 ± 0.42 vs. BC 0.39 ± 0.41) are dominated by Lift saturation and the ToolHang/Transport floor, and should not be read as evidence of a broad advantage.


## 5.2 Controlled Initialization Ablation

The focused Can/Square ablation holds the representation, training objective, and Stage 2 configuration constant across five initialization conditions (Table 2; margin loss disabled in all arms). This isolates the effect of the starting partition geometry on two measurable outcomes: routing organization and task success.

### Initialization geometry shapes routing organization

Topological prototype placement produces the highest phase-expert NMI among the three prototype-based conditions: 0.67 on Can and 0.51 on Square, compared with 0.07–0.09 for both random and rule-based initialization. The corresponding switch rates are 0.04 and 0.07 for topology, versus 0.10–0.11 for the unstructured conditions. Where prototypes begin determines where they end: starting aligned with the latent regime clusters yields a phase-coherent, temporally stable partition after training; starting at random or at rule-based centroids does not.

The rule-based condition is informative. Phase-rule initialization places prototypes at the centroids of heuristic kinematic labels, yet the resulting NMI (0.08–0.09) is indistinguishable from random placement. The initial geometry is not the only factor — it must interact with the representation structure established in Stage 1. Topology-derived centroids align with the contrastive geometry of the pre-trained latent space; rule-based centroids, computed from a different feature space, do not.

At initialization (Table A7), the topology condition starts with NMI 0.89–0.92 and no dead experts. By the end of training, NMI has decreased to 0.51–0.67 as the joint optimization reshapes both prototypes and representations. The initial alignment provides a structured starting point, not a frozen partition, and the final organization reflects both the initialization prior and the gradient-driven adaptation.

### Routing organization does not determine task success

On Can, topology initialization (76%) leads the prototype-based arms, with random (73%) and softmax (73%) close behind and rule-based (69%) and BC latent (67%) lower. The ranking is loosely consistent with routing coherence: the more phase-aligned conditions tend to succeed more often.

On Square, this correspondence breaks. Topology initialization has the highest NMI among prototype methods (0.51) but the *lowest* success rate (27%). The learned softmax condition achieves the highest success (35%) and the highest NMI (0.61). BC latent, with NMI 0.47, reaches 32%. The most phase-aligned prototype partition is the least successful controller on this task.

The separation is the central observation of the ablation: initialization geometry reliably controls routing organization, but routing organization is not sufficient to predict closed-loop performance. A more structured partition can coincide with higher success (Can) or lower success (Square) depending on the task.

### The softmax control

The learned softmax condition achieves the highest NMI across both tasks (0.72 Can, 0.61 Square) and the highest or tied-highest success on both (73% Can, 35% Square) — without any prototype initialization prior. Its routing is organized end-to-end through gradient descent on the gating network. This condition demonstrates that phase-aligned routing can emerge without topological seeding, and that the gating mechanism itself — not only the initialization — contributes to routing organization.

### Teacher-Forced diagnostic

The teacher-forced condition, which routes using ground-truth phase labels at test time, performs catastrophically: 41% on Lift, 1% on Can, 2% on Square. Providing the "correct" routing partition at deployment is worse than random performance. This reflects a distribution shift: the policy is trained under learned routing dynamics and evaluated under a routing function it has never seen. The failure does not indicate that phase labels are wrong; it indicates that the correspondence between phase labels and useful expert assignments depends on the co-adaptation of router and experts during training.
