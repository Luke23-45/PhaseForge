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

The comparison suite is organized by the factor each condition isolates, not by the implementation details that differ. All modular conditions share the same six-expert architecture and the same Stage 2 training objective; what varies is the representation, the prototype placement, or the expert initialization. Three quantities are manipulated independently:

**Representation.** Whether the encoder receives phase-contrastive pre-training (§3.3) or is trained on action prediction alone.
- *Plain Encoder*: action-only pre-training, topology-derived prototypes. Isolates the contribution of regime-structured representation to downstream expert specialization.

**Prototype initialization.** Whether the initial Voronoi partition is placed using topology, phase rules, or random assignment.
- *Phase-Random*: full phase-aware representation, but prototypes placed uniformly on $\mathbb{S}^{d-1}$. Isolates the geometric placement of the starting partition from the representation quality.

**Expert seeding.** Whether expert weights start from the pre-trained action head or from random initialization.
- *Scratch MoE*: topology-derived prototypes and structured representation, but random expert weights. Isolates the role of pre-trained action competence in each expert.

**Two-factor corner.** *Factorial Floor*: unstructured representation combined with random prototypes, testing whether the absence of both representation structuring and topology-derived initialization is recoverable through end-to-end training.

**External baselines.**
- *Monolithic BC*: a single feedforward policy with a matched encoder trunk (three hidden layers, same width), trained on mean squared error. Provides the performance floor of an unpartitioned policy.
- *Learned Softmax Top-1*: replaces prototype routing with a learned gating network, executing the argmax expert. Tests whether the prototype-based partition mechanism itself is a relevant factor.
- *Static Rule*: hard-coded kinematic thresholds determine expert assignment. Tests whether human-specified phase rules outperform learned or topology-derived partitions.

**Privileged diagnostic.** *Teacher-Forced*: routes using ground-truth phase labels at test time. Because these labels require temporal context unavailable to an autonomous agent, this condition is non-deployable. It is excluded from all comparative rankings and reported separately.


## 4.3 Focused Router-Initialization Ablation

The five-task sweep (§4.2) characterizes broad policy capability, but the comparison is not fully matched: different methods may use different learning rates, epoch counts, or auxiliary-loss configurations. To isolate the causal effect of the initialization prior alone, we run a controlled ablation on Can and Square with five conditions that share identical pre-trained representations, identical Stage 2 optimizers, and no margin loss ($\lambda_m = 0$):

1. *Topological prototype placement* (the proposed initialization)
2. *Rule-based centroid placement* (from heuristic kinematic phase labels)
3. *Random placement on $\mathbb{S}^{d-1}$*
4. *Prototypes from a plain BC encoder* (unstructured representation)
5. *Learned softmax top-1 gating* (no prototypes; gating weights optimized end-to-end)

Any difference in routing structure or task success across these five conditions is attributable to the starting partition geometry, not to representation quality or objective-function design.


## 4.4 Evaluation and Metrics

Each policy is evaluated by closed-loop rollout from a frozen set of initial simulator states, fixed across all methods and seeds. Performance and routing organization are measured separately.

**Task success** is the fraction of evaluation episodes in which the environment's native success predicate is satisfied before timeout. To assess the paired effect of each method relative to BC, we compute within-seed success differences on identical initial conditions.

**Phase-expert alignment** is quantified by normalized mutual information (NMI) between the top-1 expert assignment $k_t^*$ and the behavioral regime label $r_t$:

$$\operatorname{NMI}(k^*, r) = \frac{2\, I(k^*;\, r)}{H(k^*) + H(r)}$$

High NMI indicates that individual experts specialize in distinct behavioral regimes; low NMI indicates that the routing partition does not correspond to the phase structure in the demonstrations.

**Routing stability** is the step-to-step switch rate — the fraction of adjacent timestep pairs at which the active expert changes. Frequent switching indicates that the policy oscillates between Voronoi cells, producing rapid alternation in the active action function.

**Action continuity** is the Euclidean norm of consecutive action differences $\|a_t - a_{t-1}\|_2$, reported separately for timesteps where the expert switches and where it does not. This separates action variation arising from the modular transition mechanism from variation within a single expert's smooth output.

These four quantities are deliberately distinct. Phase-expert alignment and switch rate characterize the routing organization; task success characterizes closed-loop control; action continuity characterizes the kinematic consequences of expert transitions. The central analysis depends on not treating one as a proxy for another (F1, F3).

Evaluation details — episode counts, frozen reset-bank provenance, statistical intervals, and multiplicity corrections — are reported alongside the results and tabulated in the appendix (Tables A1, A10, A15).
