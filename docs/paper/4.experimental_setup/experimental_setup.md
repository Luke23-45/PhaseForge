# 4. Experimental Setup

The central experimental question — whether the kinematic regime structure of demonstration trajectories can shape expert specialization, and whether that organization translates into closed-loop control — requires comparisons that isolate specific factors of the modular architecture. This section describes the evaluation testbed, the controls used to isolate those factors, and the measurements that separate routing organization from task performance.


## 4.1 Tasks and Data

We evaluate on five manipulation tasks from the Robomimic benchmark, using the Proficient-Human demonstration sets. All tasks use a simulated Franka Panda arm under operational-space control at 20 Hz with actions in $[-1, 1]^A$.

| Task | $D$ | $A$ | Contact character |
| :--- | :---: | :---: | :--- |
| Lift | 19 | 7 | Single grasp, vertical transport |
| Can | 23 | 7 | Pick-and-place across bins; distinct kinematic phases |
| Square | 23 | 7 | Peg insertion; tight clearance tolerances |
| ToolHang | 53 | 7 | Multi-stage assembly; long-horizon contact dependencies |
| Transport | 59 | 14 | Bimanual handover and placement |

Each task provides 200 human demonstrations (180 train, 20 validation). The observation $x_t \in \mathbb{R}^D$ is a low-dimensional structured state vector — end-effector pose, gripper state, and object coordinates — normalized by z-score statistics from the training split. All policies are deterministic and memoryless: the action at time $t$ depends only on $x_t$.

The regime count is fixed at $K = E = 6$ across all five tasks. The method does not adapt the number of experts per task.

The five tasks span a range of contact complexity and kinematic diversity. Can and Square are used for the focused ablation (§4.3) because both exhibit clear sequential phase structure — approach, grasp, transport, and task-specific terminal placement or insertion — while differing in the precision required at contact: Can involves a clearance-tolerant pick-and-place, whereas Square demands tight peg alignment.


## 4.2 Comparative Controls

The comparison suite is organized by the factor each condition is intended to probe, while making explicit which implementation details differ. The modular conditions use six experts, but the representation, router, expert initialization, and auxiliary objectives are not identical across the full suite. Three quantities are manipulated independently within matched subsets:

**Representation.** Whether the encoder receives phase-contrastive pre-training (§3.3) or is trained on action prediction alone.
- *Plain Encoder*: action-only pre-training, regime-derived prototypes. Isolates the contribution of regime-structured representation to downstream expert specialization.

**Prototype initialization.** Whether the initial Voronoi partition is placed using trajectory-derived regime centroids, phase-rule centroids, or random assignment.
- *Phase-Random*: full phase-aware representation, but randomly initialized prototype parameters. Within the matched prototype subset, this isolates the geometric placement of the starting partition from the representation quality.

**Expert seeding.** Whether expert weights start from the pre-trained action head or from random initialization.
- *Scratch MoE*: regime-derived prototypes and structured representation, but random expert weights. Isolates the role of pre-trained action competence in each expert.

**Two-factor corner.** *Factorial Floor*: unstructured representation combined with random prototypes, testing whether the absence of both representation structuring and regime-derived initialization is recoverable through end-to-end training.

**External baselines.**
- *Monolithic BC*: a single feedforward policy with a matched encoder trunk (three hidden layers, same width), trained on mean squared error. Provides the performance floor of an unpartitioned policy.
- *Learned Softmax Top-1*: replaces prototype routing with a learned gating network, executing the argmax expert. Tests whether the prototype-based partition mechanism itself is a relevant factor.
- *Static Rule*: hard-coded kinematic thresholds determine expert assignment (thresholds tabulated in Appendix Table A1). Tests whether human-specified phase rules outperform learned or regime-derived partitions.

**Privileged diagnostic.** *Teacher-Forced*: uses the offline-generated regime labels (`phase_topo`) to dispatch experts during Stage 2 training, while rollout evaluation dispatches using the frozen phase-head prediction. Because the training route is label-dependent and differs from the evaluation route, this condition is excluded from comparative rankings and reported separately.

The five-task benchmark (§5.1) characterizes broad policy capability as contextual evidence. The matched Can/Square prototype ablation (§4.3) isolates the effect of prototype initialization within an otherwise identical architecture.


## 4.3 Focused Router-Initialization Ablation

The five-task sweep characterizes broad policy capability, but the comparison is not fully matched: different methods may use different learning rates, epoch counts, auxiliary-loss configurations, representations, or routing mechanisms. To isolate the specific effect of prototype initialization, we therefore compare three matched prototype arms on Can and Square. These arms use the same phase-aware Stage 1 representation, expert initialization, Stage 2 optimizer, and Stage 2 objective with margin loss disabled ($\lambda_m = 0$):

1. *Trajectory-derived regime prototype placement* (the proposed initialization)
2. *Rule-based centroid placement* (from heuristic kinematic phase labels)
3. *Randomly initialized prototypes*

Two additional controls are included for context but are not part of this matched initialization comparison:

4. *Plain Encoder* (action-only representation with prototype routing)
5. *Learned softmax top-1 gating* (changes the routing mechanism and has no prototypes)

Differences among the first three conditions isolate prototype initialization under the stated matched configuration. Comparisons involving the Plain Encoder and softmax controls also reflect their intentionally different representation or routing mechanism and are interpreted as architectural diagnostics, not as isolated initialization effects.


## 4.4 Evaluation and Metrics

Each policy is evaluated by closed-loop rollout from a frozen set of initial simulator states, fixed across all methods and seeds. Performance and routing organization are measured separately, and the two measurement domains use different data sources.

**Task success** is the fraction of evaluation episodes in which the environment's native success predicate is satisfied before timeout. Success is measured via closed-loop rollouts (50 episodes $\times$ 3 seeds = 150 episodes per cell). Reported uncertainty intervals are Wilson score intervals describing pooled rollout episodes (150 trials), rather than variation across independently trained seeds. To assess the paired effect of each method relative to BC, we compute within-seed success differences on identical initial conditions.

**Phase-expert alignment** is quantified by normalized mutual information (NMI) between the top-1 expert assignment $k_t^*$ and the canonical rule-derived phase label $y_t^{\mathrm{phase}}$:

$$\operatorname{NMI}(k^*, y^{\mathrm{phase}}) = \frac{2\, I(k^*;\, y^{\mathrm{phase}})}{H(k^*) + H(y^{\mathrm{phase}})}$$

High NMI indicates that individual experts specialize in distinct rule-derived behavioral phases; low NMI indicates that the routing partition does not correspond to the phase structure. NMI is evaluated on the 20 demonstration trajectories of the validation split (which is not used for checkpoint selection) and averaged across demonstration trajectories and training seeds. NMI is not evaluated against the regime-discovered `phase_topo` labels, nor is it measured during closed-loop rollouts.

**Routing stability** is the step-to-step switch rate — the fraction of adjacent timestep pairs within a trajectory at which the active expert changes. It is computed across the validation split demonstrations and averaged across trajectories and seeds, excluding transitions across trajectory boundaries. The reported switch rates therefore describe the router's behavior on the validation demonstration distribution, not on the rollout state distribution.

**Action continuity** can be defined as the Euclidean norm of consecutive action differences $\|a_t - a_{t-1}\|_2$, separated for timesteps where the expert switches and where it does not. We note that action continuity is an auxiliary diagnostic of the kinematic consequences of expert transitions and is not analyzed as empirical evidence in the present results.

These quantities are deliberately distinct. Phase-expert alignment and switch rate characterize the routing organization on the validation demonstration split; task success characterizes closed-loop control on rollout episodes. The central analysis depends on not treating one as a proxy for the other.

Evaluation details — episode counts, frozen reset-bank provenance, statistical intervals, and multiplicity corrections — are reported alongside the results and tabulated in the appendix (Tables A1, A15).
