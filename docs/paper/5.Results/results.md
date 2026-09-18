# 5. Results

The evaluation separates two quantities: how the initialization geometry organizes the routing partition, and whether that organization translates into closed-loop task success. We report the five-task benchmark sweep first, then the matched ablation that isolates the initialization prior.

We note that the five-task benchmark in §5.1 evaluates the full PhaseForge pipeline (including active Stage 2 margin loss $\lambda_m > 0$), whereas the controlled ablation in §5.2 deliberately disables margin regularization ($\lambda_m = 0$) across all arms to isolate the starting prototype placement without confounding margin dynamics. Consequently, baseline and proposed success rates differ slightly across the two sections (e.g., Can 78% vs. 76.0%; Square 41% vs. 26.7%).


## 5.1 Five-Task Benchmark

Table 1 reports rollout success rates across all deployable methods. Three task-level patterns structure the comparison.

**Lift is saturated.** All methods except Scratch MoE and Factorial Floor reach 100% success. Lift provides no discriminative signal for distinguishing modular architectures.

**Can separates methods.** PhaseForge records an observed success rate of 78% [71, 84], compared with 63% [55, 70] for BC, 69% [61, 76] for Softmax Top-1, and 62% [54, 69] for Phase-Random; brackets denote 95% Wilson score intervals on pooled episodes (150 trials). The paired difference against BC is $\Delta = +0.153$, though the per-seed standard deviation is 0.192 — BC seed 42 succeeds at 78% while seed 44 drops to 42% — and the Holm-adjusted sign test does not reject the null ($p = 1.0$; Table A15). The widest margins are against Static Rule ($\Delta = +0.253$, std 0.031) and Plain Encoder ($\Delta = +0.420$, std 0.159).

**Square does not follow the same ranking.** Plain Encoder leads in observed success at 50% [42, 58], followed by PhaseForge at 41% [33, 49] and Softmax Top-1 and Phase-Random at 39% each; brackets denote 95% Wilson intervals. The paired difference between PhaseForge and Plain Encoder is negative: $\Delta = -0.093$ (std 0.058). A method that omits phase structuring entirely records a higher observed success rate than the proposed method on this task.

**ToolHang and Transport are unsolved.** Every method, including PhaseForge, scores 0% on ToolHang. Transport yields 0–2 successful episodes across 150 trials, depending on the method. Both tasks remain unresolved under the evaluated memoryless policies and training budget, and provide no discriminative evidence regarding the initialization hypothesis.

The five-task sweep thus reveals a task-dependent pattern: PhaseForge records a higher observed success rate on Can across the three evaluated seeds, while Square — a contact-rich task with tight clearance tolerances — does not favor the regime-initialized partition.


## 5.2 Controlled Initialization Ablation

The focused Can/Square ablation holds the representation, training objective, and Stage 2 configuration constant across the three matched prototype-initialization arms (Table 2; margin loss disabled in all arms). This isolates the effect of the starting partition geometry on two measurable outcomes: routing organization and task success.

### Initialization geometry is associated with routing organization

On the validation split demonstrations, trajectory-derived regime prototype placement produces the highest phase-expert NMI among the three matched prototype-initialization conditions: 0.67 on Can and 0.51 on Square, compared with 0.07–0.09 for both matched random and rule-based prototype initialization. The corresponding routing-switch rates on the same validation demonstrations are 0.04 and 0.07 for the regime-initialized condition, versus 0.10–0.11 for the unstructured prototype conditions. In the observed runs, regime-derived initialization is associated with a phase-coherent, temporally stable partition on validation demonstrations, whereas random and rule-based starts are not.

The rule-based condition is informative. Phase-rule initialization places prototypes at centroids formed by grouping the same Stage 1 latent representations with heuristic kinematic labels, yet the resulting NMI (0.08–0.09) is similar to random placement in the observed runs. The result suggests that a non-random label source alone is insufficient; the relationship between the initialization labels and the learned representation may matter.

The initialization provides a structured starting point, not a frozen partition. The final organization reflects both the initialization prior and gradient-driven adaptation of the trainable encoder, prototypes, and experts.

### Routing organization does not determine task success

On Can, regime initialization (76.0%) leads the three matched prototype-based arms in closed-loop rollout success, followed by matched random prototype initialization (73.3%) and matched rule-based initialization (68.7%). Among the separate diagnostic controls, learned softmax top-1 gating achieves 73.3% and Plain Encoder reaches 67.3%.

On Square, this relationship breaks. Regime initialization achieves the highest NMI among the matched prototype methods (0.51 on validation demonstrations) but the lowest rollout success rate (26.7%), trailing matched random prototype initialization (28.0%) and matched rule-based initialization (31.3%). Among the diagnostic controls, learned softmax gating achieves 35.3% success (NMI 0.61) and Plain Encoder reaches 32.0% (NMI 0.47). The most phase-aligned prototype partition is the least successful controller on this task.

The dissociation is the central observation of the ablation: initialization geometry is associated with routing organization on validation demonstrations, but routing organization measured offline is not sufficient to predict closed-loop performance. A more structured partition can coincide with higher observed success (Can) or lower observed success (Square) depending on the task.

### The softmax control

The learned softmax condition achieves the highest NMI across both tasks (0.72 Can, 0.61 Square, measured on validation demonstrations) and the highest rollout success on Square (35.3%); on Can, its 73.3% success matches the random prototype condition but remains below regime initialization at 76.0%. It does so without a regime-initialization prior. Its routing is organized end-to-end through gradient descent on the gating network. This condition serves as an architectural diagnostic, demonstrating that phase-aligned routing can emerge without regime-derived seeding, and that the gating mechanism itself — not only the initialization — contributes to routing organization.

### Teacher-Forced diagnostic

The teacher-forced condition, which uses offline-generated regime labels (`phase_topo`) for expert dispatch during Stage 2 training but the frozen phase-head prediction during rollout evaluation, performs poorly: 41% on Lift, 1% on Can, and 2% on Square. This reflects a training–evaluation routing mismatch rather than a test of an oracle routing policy. The result shows that the experts and routing signal must remain compatible across training and deployment; it does not establish that the underlying phase labels are an independently useful routing policy.
