# 4. Experimental Setup

The central experimental question — whether the geometry of demonstration trajectories can shape expert specialization, and whether that organization translates into closed-loop control — requires comparisons that isolate specific factors of the modular architecture. This section describes the evaluation testbed, the controls used to isolate those factors, and the measurements that separate routing organization from task performance.


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

The five tasks span a range of contact complexity and kinematic diversity. Can and Square are used for the focused ablation (§4.3) because both exhibit clear sequential phase structure — approach, grasp, transport, insertion — while differing in the precision required at contact: Can involves a clearance-tolerant pick-and-place, whereas Square demands tight peg alignment.


## 4.2 Comparative Controls

The comparison suite is organized by the factor each condition is intended to probe, while making explicit which implementation details differ. The modular conditions use six experts, but the representation, router, expert initialization, and auxiliary objectives are not identical across the full suite. Three quantities are manipulated independently within matched subsets:

**Representation.** Whether the encoder receives phase-contrastive pre-training (§3.3) or is trained on action prediction alone.
- *Plain Encoder*: action-only pre-training, topology-derived prototypes. Isolates the contribution of regime-structured representation to downstream expert specialization.

**Prototype initialization.** Whether the initial Voronoi partition is placed using topology, phase rules, or random assignment.
- *Phase-Random*: full phase-aware representation, but randomly initialized prototype parameters. Within the matched prototype subset, this isolates the geometric placement of the starting partition from the representation quality.

**Expert seeding.** Whether expert weights start from the pre-trained action head or from random initialization.
- *Scratch MoE*: topology-derived prototypes and structured representation, but random expert weights. Isolates the role of pre-trained action competence in each expert.

**Two-factor corner.** *Factorial Floor*: unstructured representation combined with random prototypes, testing whether the absence of both representation structuring and topology-derived initialization is recoverable through end-to-end training.

**External baselines.**
- *Monolithic BC*: a single feedforward policy with a matched encoder trunk (three hidden layers, same width), trained on mean squared error. Provides the performance floor of an unpartitioned policy.
- *Learned Softmax Top-1*: replaces prototype routing with a learned gating network, executing the argmax expert. Tests whether the prototype-based partition mechanism itself is a relevant factor.
- *Static Rule*: hard-coded kinematic thresholds determine expert assignment. Tests whether human-specified phase rules outperform learned or topology-derived partitions.

**Privileged diagnostic.** *Teacher-Forced*: uses the configured ground-truth regime labels to dispatch experts during Stage 2 training, while rollout evaluation dispatches using the frozen phase-head prediction. Because the training route is label-dependent and differs from the evaluation route, this condition is excluded from comparative rankings and reported separately.


## 4.3 Focused Router-Initialization Ablation

The five-task sweep (§4.2) characterizes broad policy capability, but the comparison is not fully matched: different methods may use different learning rates, epoch counts, auxiliary-loss configurations, representations, or routing mechanisms. To isolate the causal effect of prototype initialization, we therefore compare three matched prototype arms on Can and Square. These arms use the same phase-aware Stage 1 representation, expert initialization, Stage 2 optimizer, and Stage 2 objective with margin loss disabled ($\lambda_m = 0$):

1. *Topological prototype placement* (the proposed initialization)
2. *Rule-based centroid placement* (from heuristic kinematic phase labels)
3. *Randomly initialized prototypes*

Two additional controls are included for context but are not part of this matched initialization comparison:

4. *Prototypes from a plain BC encoder* (changes the representation)
5. *Learned softmax top-1 gating* (changes the routing mechanism and has no prototypes)

Differences among the first three conditions can be attributed to prototype initialization under the stated matched configuration. Comparisons involving the plain-BC and softmax controls also reflect their intentionally different representation or routing mechanism and are interpreted as diagnostic, not as isolated initialization effects.


## 4.4 Evaluation and Metrics

Each policy is evaluated by closed-loop rollout from a frozen set of initial simulator states, fixed across all methods and seeds. Performance and routing organization are measured separately.

**Task success** is the fraction of evaluation episodes in which the environment's native success predicate is satisfied before timeout. To assess the paired effect of each method relative to BC, we compute within-seed success differences on identical initial conditions.

**Phase-expert alignment** is quantified by normalized mutual information (NMI) between the top-1 expert assignment $k_t^*$ and the behavioral regime label $r_t$:

$$\operatorname{NMI}(k^*, r) = \frac{2\, I(k^*;\, r)}{H(k^*) + H(r)}$$

High NMI indicates that individual experts specialize in distinct behavioral regimes; low NMI indicates that the routing partition does not correspond to the phase structure in the demonstrations.

**Routing stability** is the step-to-step switch rate — the fraction of adjacent timestep pairs at which the active expert changes. Frequent switching indicates that the policy oscillates between Voronoi cells, producing rapid alternation in the active action function.

**Action continuity** can be computed from full rollout traces as the Euclidean norm of consecutive action differences $\|a_t - a_{t-1}\|_2$, separated for timesteps where the expert switches and where it does not. It is a secondary diagnostic for action variation arising from the modular transition mechanism; it is not used to rank task success or support the present quantitative claims.

These quantities are deliberately distinct. Phase-expert alignment and switch rate characterize the routing organization; task success characterizes closed-loop control; action continuity is an optional diagnostic of the kinematic consequences of expert transitions. The central analysis depends on not treating one as a proxy for another (F1, F3).

Evaluation details — episode counts, frozen reset-bank provenance, statistical intervals, and multiplicity corrections — are reported alongside the results and tabulated in the appendix (Tables A1, A10, A15).
