# 5. Results

The evaluation separates two quantities: how the initialization geometry organizes the routing partition, and whether that organization translates into closed-loop task success. We report the five-task benchmark sweep first, then the matched ablation that isolates the initialization prior.


## 5.1 Five-Task Benchmark

Table 1 reports rollout success rates across all deployable methods. Three task-level patterns structure the comparison.

**Lift is saturated.** All methods except Scratch MoE and Factorial Floor reach 100% success. Lift provides no discriminative signal for distinguishing modular architectures.

**Can separates methods.** PhaseForge achieves 78% [71, 84], compared with 63% [55, 70] for BC, 69% [61, 76] for Softmax Top-1, and 62% [54, 69] for Phase-Random; the brackets are 95% Wilson intervals. The paired difference against BC is $\Delta = +0.153$, though the per-seed standard deviation is 0.192 — BC seed 42 succeeds at 78% while seed 44 drops to 42% — and the Holm-adjusted sign test does not reject the null ($p = 1.0$; Table A15). The widest margins are against Static Rule ($\Delta = +0.253$, std 0.031) and Plain Encoder ($\Delta = +0.420$, std 0.159).

**Square does not follow the same ranking.** Plain Encoder leads at 50% [42, 58], followed by PhaseForge at 41% [33, 49] and Softmax Top-1 and Phase-Random at 39% each; the brackets are 95% Wilson intervals. The paired difference between PhaseForge and Plain Encoder is *negative*: $\Delta = -0.093$ (std 0.058). A method that omits phase structuring entirely outperforms the proposed method on this task.

**ToolHang and Transport are unsolved.** Every method, including PhaseForge, scores 0% on ToolHang. Transport yields 0–2 successful episodes across 150 trials, depending on the method. These tasks lie beyond the capability frontier of the memoryless policies tested here and provide no evidence for or against the initialization hypothesis.

The five-task sweep thus reveals a task-dependent pattern: PhaseForge's advantage concentrates on Can, where it leads all comparators in the observed three-seed runs, while Square — a contact-rich task with tighter tolerances — does not favor the regime-initialized partition. Macro-averages (PhaseForge 0.44 ± 0.42 vs. BC 0.39 ± 0.41) are dominated by Lift saturation and the ToolHang/Transport floor, and should not be read as evidence of a broad advantage.


## 5.2 Controlled Initialization Ablation

The focused Can/Square ablation holds the representation, training objective, and Stage 2 configuration constant across five initialization conditions (Table 2; margin loss disabled in all arms). This isolates the effect of the starting partition geometry on two measurable outcomes: routing organization and task success.

### Initialization geometry shapes routing organization

On held-out validation demonstrations, trajectory-derived regime prototype placement produces the highest phase-expert NMI among the three matched prototype-initialization conditions: 0.67 on Can and 0.51 on Square, compared with 0.07–0.09 for both random and rule-based initialization. The corresponding routing-switch rates on the same validation demonstrations are 0.04 and 0.07 for the regime-initialized condition, versus 0.10–0.11 for the unstructured conditions. In the observed runs, regime-derived initialization was associated with a phase-coherent, temporally stable partition after training, whereas random and rule-based starts were not.

The rule-based condition is informative. Phase-rule initialization places prototypes at centroids formed by grouping the same Stage 1 latent representations with heuristic kinematic labels, yet the resulting NMI (0.08–0.09) is similar to random placement in the observed runs. The result suggests that a non-random label source alone is insufficient; the relationship between the initialization labels and the learned representation may matter.

The initialization provides a structured starting point, not a frozen partition. The final organization reflects both the initialization prior and gradient-driven adaptation of the trainable encoder, prototypes, and experts.

### Routing organization does not determine task success

On Can, regime initialization (76%) leads the prototype-based arms in closed-loop rollout success, with random (73%) and softmax (73%) close behind and rule-based (69%) and BC latent (67%) lower. The ranking is loosely consistent with routing coherence: the more phase-aligned conditions tend to succeed more often.

On Square, this correspondence breaks. Regime initialization has the highest NMI among prototype methods (0.51 on validation demonstrations) but the *lowest* rollout success rate (27%). The learned softmax condition achieves the highest success (35%) and the highest NMI (0.61). BC latent, with mean NMI 0.46, reaches 32%. The most phase-aligned prototype partition is the least successful controller on this task.

The dissociation is the central observation of the ablation: initialization geometry reliably shapes routing organization on held-out validation demonstrations, but routing organization measured offline is not sufficient to predict closed-loop performance. A more structured partition can coincide with higher success (Can) or lower success (Square) depending on the task.

### The softmax control

The learned softmax condition achieves the highest NMI across both tasks (0.72 Can, 0.61 Square, measured on validation demonstrations) and the highest rollout success on Square (35%); on Can, its 73% success ties the random prototype condition but remains below regime initialization at 76%. It does so without a regime-initialization prior. Its routing is organized end-to-end through gradient descent on the gating network. This condition demonstrates that phase-aligned routing can emerge without regime-derived seeding, and that the gating mechanism itself — not only the initialization — contributes to routing organization.

### Teacher-Forced diagnostic

The teacher-forced condition, which uses ground-truth regime labels for expert dispatch during Stage 2 training but the frozen phase-head prediction during rollout evaluation, performs poorly: 41% on Lift, 1% on Can, and 2% on Square. This is a training–evaluation routing mismatch rather than a test of an oracle routing policy. The result shows that the experts and routing signal must remain compatible across training and deployment; it does not establish that the underlying phase labels are an independently useful routing policy.
