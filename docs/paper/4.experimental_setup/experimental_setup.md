# 4. Experimental Setup

We distinguish two experimental roles. The five-task benchmark characterizes the behavior of the full method and its controls. A focused Can/Square ablation holds the prototype-routing architecture and training configuration fixed to evaluate prototype initialization.

## 4.1 Tasks and Data

We evaluate on five Robomimic manipulation tasks using Proficient-Human demonstrations. The environments use simulated Franka Panda robot systems under operational-space control at 20 Hz. Transport uses a bimanual configuration. Actions are bounded to \([-1,1]^A\).

| Task | \(D\) | \(A\) | Contact character |
| :--- | :---: | :---: | :--- |
| Lift | 19 | 7 | Single grasp and vertical transport |
| Can | 23 | 7 | Pick-and-place across bins |
| Square | 23 | 7 | Peg insertion with tight clearance |
| ToolHang | 53 | 7 | Multi-stage assembly with long-horizon contact dependencies |
| Transport | 59 | 14 | Bimanual handover and placement |

Each task contains 200 demonstrations, split into 180 training and 20 validation trajectories. Observations are low-dimensional state vectors containing end-effector pose, gripper state, and object coordinates. We normalize observations with training-split z-score statistics.

All policies are deterministic and memoryless: the action at time \(t\) depends only on \(x_t\). All modular conditions use a fixed regime and expert count of \(K=E=6\).

Can and Square form the focused initialization ablation because both require sequential manipulation behavior while imposing different contact demands. Can is a clearance-tolerant pick-and-place task; Square requires precise alignment during insertion.

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

**Task success.** Success is the fraction of episodes satisfying the environment’s native success predicate before timeout. Wilson score intervals summarize pooled rollout episodes; they do not quantify variation across independently trained seeds. Where paired comparisons are reported, they use within-seed success differences on identical reset states. Paired sign tests and Holm adjustment are reported with the corresponding results.

**Phase-expert alignment.** We measure normalized mutual information between top-1 expert assignments \(k_t^*\) and rule-derived phase labels \(y_t^{\mathrm{phase}}\):

\[
\operatorname{NMI}(k^*,y^{\mathrm{phase}})
=
\frac{2I(k^*;y^{\mathrm{phase}})}
{H(k^*)+H(y^{\mathrm{phase}})}.
\]

Higher NMI indicates a stronger association between expert assignments and rule-derived phase labels. NMI is computed on the 20 validation demonstrations, averaged across trajectories and training seeds. The validation split is not used for checkpoint selection. NMI is not measured on rollout states and is not computed against trajectory-derived regime labels.

**Routing-switch rate.** The switch rate is the fraction of adjacent timestep pairs within a demonstration for which the selected expert changes. It is computed on validation demonstrations, averaged across trajectories and seeds, and excludes transitions between trajectories. It characterizes routing behavior on the validation-demonstration distribution rather than on rollout states.

These are offline structural diagnostics of routing assignments on held-out demonstrations. They do not measure the action quality of the selected expert, recovery behavior, or closed-loop task success.

Task success, NMI, and routing-switch rate measure different quantities on different data sources. Success measures closed-loop control; NMI and switch rate measure offline routing organization. The analysis does not treat either routing metric as a proxy for task success.

The appendix reports reset-bank provenance, seed identifiers, software and dataset versions, static-rule thresholds, hyperparameters, and the full statistical tables.
