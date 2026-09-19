# 6. Discussion and Limitations

The matched ablation supports two observations. Trajectory-derived regime initialization changes offline routing organization after joint fine-tuning. That organization does not consistently predict closed-loop success across Can and Square. The result concerns a memoryless, hard-routing MoE under the evaluated training configuration.

## 6.1 Initialization and Routing Organization

Among the three matched prototype-initialization arms, trajectory-derived initialization produces higher NMI and lower routing-switch rates than rule-based and random initialization on validation demonstrations. The encoder, prototypes, and experts remain trainable during Stage 2, so this association persists after joint adaptation rather than reflecting the initial prototype positions alone.

The experiment does not identify an optimization mechanism. It does not establish a particular loss-landscape basin, nor does it show why the rule-based initialization produces NMI values similar to random initialization. The rule-based result only shows that non-random prototype placement is not sufficient to produce high phase-expert alignment in the observed runs.

Hard routing further bounds the interpretation. The action loss updates the selected expert and encoder but does not directly update prototype locations through the discrete routing decision. In the matched ablation, margin loss is disabled, so prototypes receive direct gradient signal only through the balance term. The ablation therefore characterizes an initialization-sensitive hard prototype router, not a routing mechanism whose prototypes are directly optimized by task loss.

## 6.2 Limits of Offline Routing Organization

The Square ablation shows that offline routing organization and rollout success need not rank methods in the same order. Among the matched prototype arms, trajectory-derived initialization has the highest NMI on validation demonstrations and the lowest rollout success on Square. On Can, the same initialization has the highest observed NMI and success among the matched prototype arms.

NMI and rollout success measure different quantities on different state distributions. NMI measures association between expert assignments and rule-derived phase labels on validation demonstrations. Success measures closed-loop behavior from rollout states. The observed reversal on Square shows that offline phase alignment is not sufficient to predict control quality in this setting.

The Square reversal is best interpreted as a boundary on the offline metrics rather than evidence for an unmeasured failure mechanism. Phase--expert NMI and routing-switch rate characterize assignment structure on held-out demonstrations, whereas rollout success also depends on the actions produced by the selected expert along states visited during execution. Because this study does not intervene on routing while holding expert behavior fixed, it cannot identify the mechanism behind the reversal on Square. It establishes that offline routing organization is not, by itself, a sufficient proxy for closed-loop control quality in the evaluated setting.

## 6.3 Softmax as an Architectural Diagnostic

The Learned Softmax Top-1 condition reaches high NMI without trajectory-derived prototype initialization. It uses a learned gating network rather than a nearest-prototype Voronoi partition. This result shows that phase-aligned routing can emerge without regime-derived prototype seeding, but it does not isolate the source of the difference because routing mechanism and architecture both change. Softmax is therefore an architectural diagnostic, not an initialization control.

## 6.4 Limitations

**Inference and task coverage.** All comparisons use three training seeds. The paired tests do not establish seed-level performance differences after Holm correction. The results describe the observed runs rather than a population-level effect [Henderson et al., 2018; Agarwal et al., 2021]. ToolHang and Transport remain unresolved under the evaluated policy class and training budget, so they provide no discriminative evidence about initialization.

**Supervision and full-pipeline coupling.** The matched ablation evaluates prototype initialization on a representation already shaped by rule-derived phase classification and supervised contrastive learning. It does not establish whether trajectory-derived initialization alone can organize routing without that supervision. The full five-task benchmark additionally includes the fixed index-based margin coupling between rule-derived phase labels and prototype indices defined in §3.4. The benchmark therefore evaluates the combined system, not prototype initialization in isolation. This coupling is absent from the matched ablation because \(\lambda_m=0\).

**Architecture scope.** The evaluation fixes \(K=E=6\), uses low-dimensional state observations, and restricts policies to memoryless direct-action control. The study does not determine how the result changes with adaptive expert counts, image observations, action chunking, or history-conditioned policies.

**Regime and metric validity.** Trajectory-derived regimes pass the specified observability probe, but their correspondence to dynamically relevant contact events is not measured. NMI measures association with rule-derived phase labels on validation demonstrations; it does not measure alignment with trajectory-derived regimes during rollout. Teacher-Forced uses trajectory-derived labels during training and predictions from a rule-phase head during deployment, so its poor performance reflects that label-and-dispatch mismatch rather than oracle routing quality.

**Unmeasured physical and routing quantities.** The study does not measure contact forces, grasp stability, rollout-time switching, action discontinuities, or the correspondence between routing boundaries and contact events. Segmentation also uses an unweighted concatenation of heterogeneous physical variables. Variable scale can influence the squared-Euclidean segmentation cost, but this sensitivity was not evaluated.