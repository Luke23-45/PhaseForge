# 4. Experimental Setup

We distinguish two experimental roles. The five-task benchmark characterizes the behavior of the full method and its controls. A focused Can/Square ablation holds the prototype-routing architecture and training configuration fixed to evaluate prototype initialization.

## 4.1 Tasks and Data

We evaluate on five Robomimic manipulation tasks using Proficient-Human demonstrations [Mandlekar et al., 2021]. The environments use simulated Franka Panda robot systems under operational-space control at 20 Hz [Zhu et al., 2020; Todorov et al., 2012]. Transport uses a bimanual configuration. Actions are bounded to \([-1,1]^A\).

| Task | \(D\) | \(A\) | Contact character |
| :--- | :---: | :---: | :--- |
| Lift | 19 | 7 | Single grasp and vertical transport |
| Can | 23 | 7 | Pick-and-place across bins |
| Square | 23 | 7 | Peg insertion with tight clearance |
| ToolHang | 53 | 7 | Multi-stage assembly with long-horizon contact dependencies |
| Transport | 59 | 14 | Bimanual handover and placement |

Each task contains 200 demonstrations, split into 180 training and 20 validation trajectories. Observations are low-dimensional state vectors containing end-effector pose, gripper state, and object coordinates. We normalize observations with training-split z-score statistics.

All policies are deterministic and memoryless: the action at time \(t\) depends only on \(x_t\). All modular conditions use a fixed regime and expert count of \(K=E=6\).

We report the complete five-task sweep and focus the matched initialization analysis on Can and Square, the two tasks with outcome variation in the reported full-suite matrix. This focused analysis is not a broad five-task performance estimate.

## 4.2 Comparative Controls

The full benchmark contains methods that differ in representation training, prototype initialization, expert initialization, or routing mechanism. It is therefore a contextual benchmark rather than a fully matched factorial comparison.

| Condition | Primary difference from regime-initialized MoE | Role |
| :--- | :--- | :--- |
| Regime-initialized MoE | Trajectory-derived regime centroids initialize prototypes | Proposed configuration |
| Plain Encoder | Action-only representation pre-training | Representation diagnostic |
| Phase-Random | Random prototype initialization | Prototype-initialization diagnostic |
| Scratch MoE | Random expert initialization | Expert-seeding diagnostic |
| Factorial Floor | Action-only representation and random prototypes | Low-structure reference |
| Monolithic BC | No expert partition or router | Unpartitioned policy reference |
| Learned Softmax Top-1 | Learned gating network replaces prototype routing | Architectural diagnostic |
| Static Rule | Hard-coded kinematic thresholds dispatch experts | Rule-dispatch reference |
| Teacher-Forced | Offline labels dispatch experts during Stage 2 training | Privileged diagnostic |

The Static Rule thresholds are reported in the appendix. The full-suite comparisons should not be interpreted as isolated estimates of a single design factor.

Teacher-Forced uses trajectory-derived regime labels to dispatch experts during Stage 2 training. During rollout, it dispatches with predictions from a frozen phase head trained on rule-derived phase labels. Its training and deployment routes differ in both dispatch source and label vocabulary. It is excluded from comparative rankings.

## 4.3 Focused Router-Initialization Ablation

The focused ablation isolates prototype initialization on Can and Square. The three matched arms use the same phase-aware Stage-1 representation, expert initialization, Stage-2 optimizer, and Stage-2 objective. Margin loss is disabled in every arm:

\[
\lambda_m=0.
\]

The arms differ only in the initial prototype locations:

1. **Trajectory-derived regime initialization:** normalized latent centroids grouped by trajectory-derived regime labels.
2. **Rule-based initialization:** normalized latent centroids grouped by rule-derived phase labels.
3. **Random initialization:** standard small random prototype parameters.

Plain Encoder and Learned Softmax Top-1 are reported as architectural diagnostics. Plain Encoder changes representation learning, and Softmax changes the routing mechanism. Neither is part of the matched prototype-initialization comparison.

## 4.4 Evaluation and Metrics

Each task-method-seed condition is evaluated from a frozen bank of initial simulator states shared across methods. We run 50 rollout episodes per seed over three training seeds, yielding 150 episodes per task-method condition.

**Task success.** Success is the fraction of episodes satisfying the environment’s native success predicate before timeout. Wilson score intervals [Wilson, 1927] summarize pooled rollout episodes; they do not quantify variation across independently trained seeds. Where paired comparisons are reported, they use within-seed success differences on identical reset states. Paired sign tests and Holm adjustment [Holm, 1979] are reported with the corresponding results.

**Phase-expert alignment.** We measure normalized mutual information [Vinh et al., 2010] between top-1 expert assignments \(k_t^*\) and rule-derived phase labels \(y_t^{\mathrm{phase}}\):

\[
\operatorname{NMI}(k^*,y^{\mathrm{phase}})
=
\frac{2I(k^*;y^{\mathrm{phase}})}
{H(k^*)+H(y^{\mathrm{phase}})}.
\]

Higher NMI indicates a stronger association between expert assignments and rule-derived phase labels. NMI is computed for each training seed over the concatenated samples from all 20 validation trajectories and is then averaged across seeds. The validation split is not used for checkpoint selection. NMI is not measured on rollout states and is not computed against trajectory-derived regime labels.

**Routing-switch rate.** For each training seed, routing-switch rate is the fraction of valid adjacent timestep pairs across all 20 validation trajectories for which the selected expert changes; transitions between trajectories are excluded. The per-seed rates are then averaged across seeds. It characterizes routing behavior on the validation-demonstration distribution rather than on rollout states.

Phase–expert NMI measures association between top-1 expert assignments and rule-derived phase labels on held-out validation demonstrations. Routing-switch rate measures changes in top-1 expert assignment between adjacent timesteps within those demonstrations. These are offline routing diagnostics; they do not measure action quality, recovery behavior, or closed-loop task success.

Because Stage 1 uses rule-derived phase supervision, and both rule phases and trajectory-derived regimes are functions of demonstration kinematics, NMI should be interpreted as alignment with this study’s rule-based kinematic phase vocabulary. It is not an annotation-free or universal measure of semantic specialization.

Task success, NMI, and routing-switch rate measure different quantities on different data sources. Success measures closed-loop control; NMI and switch rate measure offline routing organization. The analysis does not treat either routing metric as a proxy for task success.

The appendix reports reset-bank provenance, seed identifiers, software and dataset versions, static-rule thresholds, hyperparameters, and the full statistical tables.
