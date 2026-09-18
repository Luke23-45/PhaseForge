# Abstract

Mixture-of-experts policies provide a natural way to represent heterogeneous behavior by assigning different regions of the policy space to different expert networks. In manipulation, however, the resulting partition is learned together with the policy, leaving the relationship between the structure present in demonstrations and the organization of the experts largely implicit. We investigate whether this structure can instead be introduced through the initialization of the expert partition.

Our approach extracts discrete kinematic regimes from demonstration trajectories via change-point segmentation and centroid-based clustering, and uses the Stage-1 latent centroids grouped by these trajectory-derived regime labels to initialize the routing prototypes of a mixture-of-experts policy. The resulting experts are trained jointly, allowing the initial partition to adapt during optimization. We compare trajectory-derived regime initialization with matched random and rule-based prototype initializations under an identical prototype-routing procedure, alongside an architectural diagnostic using learned softmax gating, and separately examine the resulting routing structure and closed-loop behavior.

Under matched prototype-routing conditions, trajectory-derived regime initialization produces substantially more phase-aligned routing than random or rule-based prototype initialization in the focused Can/Square ablation, yielding mean normalized mutual information against rule-derived phase labels of 0.67 and 0.51 and mean routing-switch rates of 0.04 and 0.07 across seeds on validation demonstrations for Can and Square, respectively. This organizational effect is not consistently reflected in closed-loop task success. In rollout evaluations, the regime-initialized policy achieves 76.0% success on Can and 26.7% on Square, compared with 73.3% and 28.0% for matched random prototype initialization, 68.7% and 31.3% for matched rule-based prototype initialization, and 73.3% and 35.3% for the separate learned softmax gating diagnostic.

The results show that the geometry used to initialize an expert partition can shape the organization of the learned routing partition on validation demonstrations, while that organizational change is not by itself sufficient to predict closed-loop performance across the evaluated tasks. This separates the role of an initialization prior in structuring expert routing from its downstream effect on control.

---

# When Does Trajectory Regime Structure Guide Policy Specialization? An Empirical Study of Mixture-of-Experts Routing Initialization

# 1. Introduction

A manipulation policy may encounter substantially different control regimes over the course of a single task. Approaching an object, establishing contact, transporting it, and completing the final placement can require different local mappings from state to action. Mixture-of-experts models provide a direct mechanism for representing such heterogeneity: rather than forcing a single network to explain the entire behavioral distribution with one set of parameters, they partition the policy among several experts and use a router to determine which expert is active.

The effectiveness of this decomposition depends not only on the capacity of the experts, but also on how the behavioral space is partitioned among them. A useful partition can isolate locally coherent action regimes, whereas an arbitrary partition may leave individual experts responsible for incompatible portions of the trajectory. In standard end-to-end training, this partition emerges jointly with the representation, router, and expert parameters. The demonstrations therefore contain structure that may be relevant to specialization, but the policy is not explicitly initialized to exploit it.

Demonstration trajectories provide one source of such structure. Rather than viewing the demonstrations solely as state-action pairs, we consider the kinematic regimes that can be extracted from their temporal organization and use those regimes to initialize the locations of routing prototypes. The resulting prior does not prescribe the final specialization of the experts; it sets the starting partition from which optimization proceeds.

This raises a more specific question: **to what extent does the regime structure of the demonstrations influence the organization of a learned expert partition, and does a more structured partition necessarily produce a better controller?**

We examine this question using a prototype-routed mixture-of-experts policy for robot manipulation with a fixed expert count $K = E = 6$ across all tasks. Trajectory-derived regime prototypes are compared with rule-based phase-derived and random initializations under matched prototype-routing conditions. The analysis deliberately separates two outcomes that are often conflated: the organization of the routing function and the success of the resulting closed-loop policy. Routing organization is evaluated through phase-expert alignment (NMI) against rule-derived phase labels and routing-switch rates on validation demonstrations, while policy quality is evaluated through closed-loop rollout success.

The central result is an empirical dissociation between these quantities in the evaluated setting. Trajectory-derived regime initialization produces substantially more structured routing than the unstructured prototype initialization conditions, yet the corresponding advantage in task success depends on the task. The result indicates that the kinematic regime structure of the demonstrations can act as an effective prior over expert routing organization, while also showing that a coherent routing partition is not synonymous with a superior controller.

In summary, this work provides three contributions:

1. **Trajectory-derived regime initialization.** A procedure that extracts discrete kinematic regimes from demonstration trajectories via change-point segmentation and clustering, and uses the Stage-1 latent centroids grouped by these trajectory-derived regime labels to initialize routing prototypes in a mixture-of-experts policy with a fixed expert count ($K = E = 6$).

2. **Controlled evaluation of routing organization.** Empirical evidence, measured on validation demonstrations, that trajectory-derived prototype initialization increases routing alignment with rule-derived phase labels (NMI) and is associated with lower routing-switch rates relative to matched random and rule-based prototype controls under identical training conditions.

3. **Empirical dissociation from closed-loop success.** A matched initialization ablation across Can and Square isolating prototype initialization within the stated configuration, showing that more structured, phase-aligned routing partitions on validation demonstrations do not consistently translate to higher closed-loop rollout success, showing an empirical dissociation between offline partition organization and closed-loop control quality in the evaluated setting.

---

# 2. Related Work

This paper sits at the intersection of three lines of work: mixture-of-experts policies for behavioral decomposition, trajectory segmentation as a source of behavioral structure, and the role of initialization in shaping learned modular architectures. We organize the discussion around how each line bears on the specific question this paper addresses — whether the kinematic regime structure of demonstrations can be used to initialize expert routing, and whether the resulting organization predicts closed-loop performance.


## 2.1 Mixture-of-experts in policy learning

Mixture-of-experts architectures partition a model's computation among specialized sub-networks, with a gating function determining which expert processes each input [Jacobs et al., 1991; Jordan and Jacobs, 1994]. In large-scale language modeling, sparse MoE layers with learned gating have become a standard mechanism for scaling model capacity without proportional increases in per-sample compute [Shazeer et al., 2017; Fedus et al., 2022; Lepikhin et al., 2021]. The routing dynamics in these models — load balancing, expert collapse, and the sensitivity of specialization to initialization — are active research topics [Zoph et al., 2022; Zhou et al., 2022].

In robot learning, the MoE structure has a different motivation: behavioral regimes within a single task or across tasks may require qualitatively different state-to-action mappings, and a modular architecture can assign each regime to a dedicated expert. Prior work has used mixture formulations for multi-modal action distributions [Zhao et al., 2023], hierarchical skill decomposition [Kipf et al., 2019; Shankar et al., 2020], and option-conditioned policies [Bacon et al., 2017; Zhang et al., 2019]. In most of these approaches, the expert partition emerges jointly with the policy through end-to-end training. The question of whether the initial partition — not just the final one — matters for downstream behavior has received comparatively little attention in the manipulation setting.


## 2.2 Trajectory segmentation and behavioral phases

Decomposing continuous trajectories into discrete behavioral segments has a long history in robotics. Dynamic movement primitives [Ijspeert et al., 2013] and probabilistic movement primitives [Paraschos et al., 2018] represent trajectories as sequences of parameterized motion segments. Changepoint detection methods — including Bayesian online detection [Adams and MacKay, 2007], kernel-based tests [Harchaoui and Cappé, 2007], and penalized cost approaches [Killick et al., 2012; Truong et al., 2020] — identify temporal boundaries where the generating process changes.

In imitation learning, trajectory segmentation has been used to discover sub-goals [Niekum et al., 2012; Konidaris et al., 2012], to define options for hierarchical policies [Krishnan et al., 2017; Fox et al., 2017], and to provide auxiliary supervision for representation learning [Shiarlis et al., 2018]. These methods produce behavioral labels — segments, primitives, or phase indices — but the labels are typically consumed as training targets or as inputs to a separate planning layer. The possibility of using them to *initialize* the geometry of a routing partition, rather than as targets or planning abstractions, is the connection this paper explores.


## 2.3 Initialization, prototypes, and routing geometry

The effect of initialization on neural network training is well-studied in the supervised learning literature [Glorot and Bengio, 2010; He et al., 2015; Mishkin and Matas, 2016], and recent work has examined how initialization interacts with the loss landscape geometry of deep networks [Li et al., 2018; Fort et al., 2019]. In the MoE context specifically, the initial placement of routing parameters can determine which experts receive gradient signal early in training, creating a path dependence that persists through optimization [Lewis et al., 2021; Roller et al., 2021].

Prototype-based classification — where predictions are made by proximity to learned class representatives in an embedding space — provides the routing mechanism used in this work. Prototypical networks [Snell et al., 2017] and related metric-learning methods [Vinyals et al., 2016; Sung et al., 2018] learn prototypes jointly with representations. The connection to Voronoi partitions of the latent space is explicit: each prototype defines a cell, and nearest-prototype assignment partitions inputs into regions. Whether the initial locations of these prototypes — derived from an external source of structure rather than learned from scratch — affect the properties of the converged partition is the specific empirical question this paper tests.


## 2.4 The gap this paper addresses

The three lines above converge on a specific intersection. Mixture-of-experts policies provide modular architectures whose routing can, in principle, reflect the phase structure of a task. Trajectory segmentation methods can discover that phase structure from demonstrations. Initialization is known to affect neural network optimization and expert specialization in MoE models. However, regarding the specific question of whether demonstration-derived kinematic regime structure changes the organization of an MoE routing partition when used as an initialization prior, and whether that change predicts task performance, we are not aware of a direct evaluation in the setting studied here.

This paper provides that test, with a controlled comparison that separates the structural effect (routing organization) from the performance effect (closed-loop success) under matched training conditions.

---

# 3. Method

A manipulation trajectory typically traverses several kinematically distinct regimes — approaching, grasping, transporting, placing — each of which requires a different local mapping from state to action. Rather than learning a single monolithic policy that must internally partition its capacity among these regimes, we structure the policy as a mixture of experts whose routing partition is seeded from the kinematic regime structure of the demonstration data itself.

This section defines the four components of the approach: the modular policy architecture (§3.1), the offline procedure that discovers behavioral regimes from demonstration trajectories (§3.2), the phase-aware representation pre-training that shapes the latent space (§3.3), and the prototype initialization and joint fine-tuning that anchors the routing partition to the discovered structure (§3.4).


## 3.1 Policy Architecture

At each timestep $t$, the environment presents a state $x_t \in \mathbb{R}^D$ and the policy returns a bounded action $a_t \in [-1, 1]^A$. The policy is deterministic and memoryless: the action depends only on the current observation $x_t$, with no recurrent state or trajectory history.

### Shared encoder

A feedforward encoder $f_\phi$ maps the normalized observation into a $d$-dimensional latent vector, followed by projection onto the unit hypersphere:

$$z_t = \frac{f_\phi(x_t)}{\| f_\phi(x_t) \|_2}$$

The resulting representation $z_t \in \mathbb{S}^{d-1}$ is shared by the router and all experts. Projecting onto the unit sphere ensures that distances between representations reflect directional relationships rather than activation magnitude, and provides a natural domain on which to define prototype-based routing.

### Prototype router

The routing function maintains $E$ trainable prototype vectors $\{c_k\}_{k=1}^E$. In the regime-derived and rule-based centroid conditions, the prototypes are initialized as normalized latent centroids; the random control uses the router's standard small random initialization. The router selects the expert whose prototype is nearest to the current representation:

$$k_t^* = \arg\min_{k \in \{1, \ldots, E\}} \| z_t - c_k \|_2$$

For a normalized centroid initialization, both $z_t$ and $c_k$ lie on the unit sphere, so the distance rule is equivalent at that point to selecting the prototype with maximum cosine similarity: $\| z_t - c_k \|_2^2 = 2 - 2\, z_t^\top c_k$. The prototypes remain trainable and are not constrained to remain normalized during Stage 2; after fine-tuning, routing is defined by the stated Euclidean-distance rule.

Hard top-1 selection partitions the normalized latent representation space into a nearest-prototype Voronoi partition:

$$\mathcal{V}_k = \bigl\{ z \in \mathbb{S}^{d-1} : \| z - c_k \|_2 \le \| z - c_j \|_2 \;\; \forall\, j \ne k \bigr\}$$

in which each cell $\mathcal{V}_k$ maps exclusively to expert $k$. The partition is determined by prototype placement: moving a prototype reshapes the cell boundaries and thereby changes which states each expert is responsible for. Because prototypes are trainable and need not remain normalized, the post-training cells should not be interpreted as a Voronoi tessellation whose sites all lie on the unit sphere.

**Optimization of hard dispatch.** The top-1 selection operator $k_t^* = \arg\min_k \|z_t - c_k\|_2$ is piecewise constant and non-differentiable. Consequently, the action reconstruction loss $\mathcal{L}_{\text{act}}$ does not backpropagate gradients into the router prototypes $\{c_k\}$ or through the discrete routing assignment (no straight-through estimator is used). Instead, the prototypes $\{c_k\}$ are optimized exclusively via the auxiliary objectives in Stage 2: specifically, the soft load-balancing loss $\mathcal{L}_{\text{bal}}$ (which computes differentiable softmax probabilities over negative distances) and, when active, the margin loss $\mathcal{L}_{\text{margin}}$ (which differentiates directly through pairwise distance differences). When margin loss is disabled ($\lambda_m = 0$), the soft balance loss serves as the sole gradient channel for updating prototype locations.

### Expert networks and action execution

Each expert $e_k : \mathbb{S}^{d-1} \to (-1, 1)^A$ is an independent feedforward network with a $\tanh$ output nonlinearity. At execution time, only the selected expert produces the action:

$$a_t = e_{k_t^*}(z_t)$$

In the reported configuration, the optional residual coefficient is fixed at $\beta=0$, so the residual feedback branch is inactive and each expert reduces to its direct-action base network.

The policy is therefore piecewise-defined across the Voronoi cells: within each cell, the action is a smooth function of the representation, but a discontinuity can occur at cell boundaries where the active expert changes.


## 3.2 Trajectory Regime Discovery

The routing partition described above is parameterized by the prototype locations $\{c_k\}$. In standard mixture-of-experts training, these would be initialized randomly and learned end-to-end. The central methodological choice in this work is instead to derive the initial prototype placement from the kinematic regime structure of the demonstration trajectories. This subsection describes the offline pipeline that extracts that structure.

The number of regimes $K$ is fixed at $K = 6$ across all tasks. The method does not discover the number of experts; it segments and clusters into a predetermined count. The number of experts $E$ is set equal to $K$, so $K = E = 6$ throughout.

### Task-variable signal

From each demonstration, we extract a kinematic signal $s_t$ that captures end-effector configuration and its spatial relationship to the manipulated object:

$$s_t = \bigl[ \, p_t, \;\; q_t, \;\; g_t, \;\; o_t, \;\; (p_t - o_{t,\, 0\!:\!3}), \;\; \alpha_t \, \bigr]$$

where $p_t \in \mathbb{R}^3$ is the end-effector position, $q_t \in \mathbb{S}^3$ is the orientation quaternion (sign-canonicalized to non-negative scalar part), $g_t$ denotes the gripper joint state, $o_t$ contains the object's proprioceptive variables, and $\alpha_t$ is a scalar gripper aperture. The relative displacement $(p_t - o_{t,\, 0\!:\!3})$ isolates object-relative motion from workspace-absolute motion.

**Signal scaling and physical units.** Prior to segmentation, observations are denormalized back to physical coordinates (meters for end-effector and object Cartesian positions, sign-canonical unit quaternions on $\mathbb{S}^3$ for orientation, and joint positions for the gripper). The concatenated signal $s_t$ is unweighted, so the subsequent cost function operates directly on these mixed physical dimensions.

### Change-point segmentation

Each trajectory is partitioned into contiguous intervals of locally homogeneous task-variable statistics. Change-points $0 = \tau_0 < \tau_1 < \cdots < \tau_M = T$ are obtained by solving an exact dynamic program that minimizes the total within-segment variance under a complexity penalty:

$$\min_{\{\tau_j\}_{j=0}^M} \; \sum_{j=0}^{M-1} C(s_{\tau_j : \tau_{j+1}}) \;+\; \lambda_{\mathrm{cp}} \, (M - 1), \qquad \text{subject to} \quad \tau_{j+1} - \tau_j \ge L_{\min}$$

where the segment cost measures squared deviation from the segment mean:

$$C(s_{i:j}) = \sum_{t=i}^{j-1} \| s_t - \bar{s}_{i:j} \|_2^2$$

The penalty $\lambda_{\mathrm{cp}}$ controls the granularity of the decomposition, and the minimum-duration constraint $L_{\min}$ prevents over-fragmentation at transient sensor fluctuations. The optimization is deterministic and exact.

### Regime clustering

Individual trajectory segments are summarized by their first and second moments:

$$\phi_j = \bigl[ \operatorname{mean}(s_{\tau_j : \tau_{j+1}}), \;\; \operatorname{var}(s_{\tau_j : \tau_{j+1}}) \bigr]$$

These summary vectors are pooled across all training demonstrations and partitioned into $K = 6$ discrete behavioral regimes via centroid-based clustering. Each timestep inherits the regime label of its enclosing segment, producing a per-timestep assignment $r_t \in \{1, \ldots, K\}$.

The result is a decomposition of the demonstration data into regimes that are intended to represent kinematically coherent intervals. Labels such as approach, contact, transport, and placement are interpretations of the resulting clusters, not supervision supplied to the discovery procedure.

### Label vocabularies

The pipeline produces two distinct per-timestep label artifacts, which are not identical:

- **`phase` (rule-derived labels).** Hand-engineered, task-specific heuristic boundaries based on physical thresholds — gripper aperture, end-effector height relative to the object, and contact state. These are deterministic and identical for a given trajectory.

- **`phase_topo` (regime-derived labels).** Unsupervised labels produced by the change-point segmentation and clustering procedure described above. These depend on the segmentation penalty $\lambda_{\mathrm{cp}}$, the clustering seed, and the training-split segment pool.

The two label sets have different temporal boundaries for the same trajectory and assign different integer labels to the same timestep. In the proposed configuration, `phase_topo` labels determine prototype initialization (§3.4), while `phase` labels supervise Stage 1 classification and contrastive losses (§3.3) and the Stage 2 margin loss (§3.4).

### Observability verification

The regime labels are derived from trajectory-level segmentation, which has access to temporal context. For these labels to be usable as a routing prior in a memoryless policy, they must be recoverable from instantaneous state alone. We verify this by training a linear classifier to predict the regime label from a single normalized observation $x_t$, evaluated under trajectory-grouped cross-validation. The regime artifact is accepted for routing only if the probe exceeds a minimum classification threshold (accuracy $\ge 0.70$) and each regime meets a minimum occupancy requirement ($\ge 0.05$); otherwise, the configured fail-closed gate rejects the artifact. Specific validation thresholds and split protocols are listed in Appendix Table A1.


## 3.3 Representation Pre-Training (Stage 1)

Before the modular architecture is instantiated, the encoder $f_\phi$ is pre-trained to produce a representation that is simultaneously informative about the demonstrated actions and organized with respect to the canonical behavioral phase labels. This stage trains the encoder together with two auxiliary heads — an action prediction head and a phase-classification head — under a composite objective:

$$\mathcal{L}_1 = \mathcal{L}_{\text{act}} + \lambda_{\text{phase}} \, \mathcal{L}_{\text{phase}} + \lambda_{\text{sc}} \, \mathcal{L}_{\text{SupCon}}$$

**Action loss.** The action head predicts a deterministic action $a_i^{\text{gen}}$ from the latent representation and is trained by mean squared error against the demonstration target $a_i^*$:

$$\mathcal{L}_{\text{act}} = \frac{1}{|B|\, A} \sum_{i \in B} \sum_{d=1}^{A} \bigl( a_{i,d}^{\text{gen}} - a_{i,d}^* \bigr)^2$$

**Phase classification loss.** A linear classifier maps $z_i$ to logits over the $K$ phase classes and is trained with cross-entropy:

$$\mathcal{L}_{\text{phase}} = - \frac{1}{|B|} \sum_{i \in B} \log \frac{\exp(\ell_{i, y_i})}{\sum_{q=1}^{K} \exp(\ell_{i, q})}$$

where $y_i \in \{1, \ldots, K\}$ is the phase label used by the representation-training configuration. In the proposed configuration, this is the canonical rule-derived `phase` field. The regime-discovered `phase_topo` labels are used to initialize routing prototypes (§3.4) and are not the targets of the Stage 1 classification or supervised-contrastive losses.

**Supervised contrastive loss.** To impose metric structure on $\mathbb{S}^{d-1}$, a supervised contrastive objective pulls representations with the same phase label toward each other while pushing apart representations from different phases:

$$\mathcal{L}_{\text{SupCon}} = \frac{1}{|I|} \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i^\top z_p \,/\, \tau)}{\sum_{a \ne i} \exp(z_i^\top z_a \,/\, \tau)}$$

Here, $P(i) = \{p \in B \setminus \{i\} : y_p = y_i\}$ is the set of batch elements sharing the phase label of anchor $i$, and $I = \{i : |P(i)| > 0\}$ excludes singletons. The temperature $\tau$ controls the sharpness of the similarity distribution.

The combined objective is intended to map observations with the same canonical phase label into neighboring regions of $\mathbb{S}^{d-1}$, while maintaining enough action-predictive information to seed the downstream experts.


## 3.4 Prototype Initialization and Joint Fine-Tuning (Stage 2)

After Stage 1, the auxiliary heads are detached, and the modular architecture — router and experts — is instantiated for joint training. The central step is the initialization of the routing prototypes from the latent geometry established in Stage 1.

### Prototype initialization

For each discovered regime $k$, the routing prototype $c_k$ is placed at the $L_2$-normalized centroid of the Stage 1 representations assigned to that regime:

$$\tilde{c}_k = \frac{1}{N_k} \sum_{i:\, r_i = k} z_i, \qquad c_k = \frac{\tilde{c}_k}{\| \tilde{c}_k \|_2}$$

where $z_i$ are the Stage 1 latent vectors and $N_k$ is the number of samples in regime $k$. In the proposed configuration, membership is determined by the regime-discovered `phase_topo` labels, whereas Stage 1 is trained with the rule-derived `phase` labels. Regime initialization therefore uses the learned latent geometry grouped by trajectory-derived regime labels; it does not imply that the representation was trained on those same regime labels. The centroid construction supplies a structured initial partition; it does not freeze that partition.

### Expert initialization

Each expert network is seeded from the Stage 1 action head. To break symmetry between experts while preserving baseline action-prediction competence, a fixed proportion of hidden-layer weights ($0.20$ of weights in the final two layers) is re-initialized independently per expert. The remaining weights retain their pre-trained values.

### Joint fine-tuning objective

The encoder, prototypes, and expert parameters are optimized jointly under:

$$\mathcal{L}_2 = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{bal}} + \lambda_m \, \mathcal{L}_{\text{margin}}$$

**Action loss.** Identical in form to Stage 1, but now evaluated on the output of the selected expert $e_{k_i^*}(z_i)$ rather than the monolithic action head.

**Balance loss.** A soft load-balancing penalty prevents degenerate partitions in which a subset of experts captures the entire data distribution:

$$\mathcal{L}_{\text{bal}} = \lambda_{\text{bal}} \, E \sum_{k=1}^{E} f_k \, p_k$$

where $f_k = \frac{1}{|B|} \sum_{i \in B} \mathbf{1}[k_i^* = k]$ is the hard assignment fraction and $p_k = \frac{1}{|B|} \sum_{i \in B} \operatorname{softmax}(-d_i)_k$ is the mean soft routing probability for expert $k$. The product $f_k \, p_k$ is large only when expert $k$ both receives many hard assignments and has high average soft affinity — penalizing concentration on both axes simultaneously. Differentiating through $p_k$ provides a smooth gradient signal directly to the prototypes $\{c_k\}$.

**Margin loss.** An explicit distance margin $m$ separates the target-regime prototype from all alternatives:

$$\mathcal{L}_{\text{margin}} = \frac{1}{|B|} \sum_{i \in B} \sum_{j \ne \pi(y_i)} \bigl[\, m - (d_{i,j} - d_{i, \pi(y_i)}) \,\bigr]_+$$

where $d_{i,k} = \| z_i - c_k \|_2$, $y_i \in \{0, \dots, K-1\}$ is the rule-derived integer phase label, and $\pi$ denotes the mapping from phase label IDs to prototype indices. In the implementation, $\pi$ is the identity mapping $\pi(y_i) = y_i$; no bipartite matching or semantic permutation is solved between the rule-derived `phase` IDs and the trajectory-derived `phase_topo` cluster IDs. Crucially, the matched Can/Square initialization ablation (§4.3) disables this margin term ($\lambda_m = 0$), eliminating any cross-vocabulary indexing assumption and isolating prototype initialization under an identical objective.

### What adapts during Stage 2

The encoder, prototypes, and experts remain trainable: the encoder adapts at a reduced learning rate, while prototypes and experts optimize at the base rate. The auxiliary Stage 1 heads are detached before Stage 2. The initial regime-derived partition therefore provides a structured starting point — not a frozen constraint — and the final partition reflects the combined influence of the initialization geometry and task-driven gradient updates.

---

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

---

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

---

# 6. Discussion and Limitations

The results establish a clear positive finding and a clear negative one. In the observed runs, trajectory-derived regime initialization shapes the organization of a learned modular policy: the starting prototype placement is associated with whether the final routing partition aligns with the behavioral phase structure of the validation demonstrations. At the same time, a more phase-aligned partition does not consistently produce a better controller. The empirical dissociation between these two outcomes in this setting is the central contribution, and this section interprets what it does and does not establish.


## 6.1 What initialization geometry controls

The ablation shows that the starting positions of routing prototypes have an observed effect on the routing partition that persists through joint fine-tuning. Regime-derived initialization produces final NMI values against rule-derived phase labels of 0.51–0.67 on validation demonstrations, compared with 0.07–0.09 for matched random and rule-based prototype starts, despite identical training objectives and identical pre-trained representations in the matched prototype subset. The effect is consistent across the two evaluated tasks.

This persistence is not trivial. The prototypes and encoder are both trainable during Stage 2, and the optimization could in principle reorganize the partition entirely. The retained association between initialization and final routing is consistent with initialization-dependent optimization, but these experiments do not identify the underlying loss-landscape mechanism or establish the existence of a particular attraction basin.

The rule-based condition sharpens this interpretation. Phase-rule centroids are formed from the same Stage 1 latent representations as the regime centroids, but grouped using heuristic kinematic phase labels. They produce routing organization similar to random initialization in the observed runs (NMI 0.08–0.09). The contrast between the regime-derived and rule-based conditions is therefore consistent with the possibility that the relationship between the initialization labels and the learned latent geometry matters, rather than merely whether the starting point is non-random.


## 6.2 What routing organization does not control

The Square ablation is the sharpest evidence that routing organization and task success are empirically dissociable in this setting. Among the matched prototype-based conditions, regime initialization produces the highest NMI on validation demonstrations (0.51) and the lowest rollout success rate (26.7%). The learned softmax diagnostic condition, which also achieves high NMI (0.61), succeeds at 35.3%. On Can, by contrast, regime initialization leads the matched prototype arms on both NMI and success.

Several factors could account for the task dependence, though the current measurements do not isolate which ones are operative. These are untested hypotheses, not established mechanisms:

- **The regime boundaries may not align with the contact-critical transitions on Square.** The discovery pipeline identifies kinematic regime boundaries from trajectory statistics. If the contact transitions that matter for peg insertion do not coincide with the statistically salient changes in the task-variable signal — for example, if the critical phase is a short alignment maneuver within a longer transport segment — then a phase-aligned partition may assign the wrong expert to the precision-critical region. This remains an untested hypothesis; the current experiments do not measure the correspondence between regime boundaries and the dynamically relevant insertion phases.

- **Routing stability may carry different costs on different tasks.** The regime-initialized partition is associated with the lowest routing-switch rates on validation demonstrations on both tasks (0.04 on Can, 0.07 on Square). Stable routing on validation data is consistent with expert coherence, but if a task requires rapid adjustments near contact — fine corrections that span Voronoi boundaries — then low switch rates on the demonstration distribution could indicate that the policy does not recruit the appropriate local specialist.

- **The action-continuity trade-off may favor flexibility over coherence.** Hard top-1 routing can produce action discontinuities at expert transitions. A more structured partition may concentrate these transitions at phase boundaries. Whether that is better or worse depends on whether the task tolerates discontinuities at those boundaries or requires smooth transitions that a more flexible routing scheme can provide.

The data show *that* routing organization and success dissociate on Square in the observed runs; they do not show *why*.


## 6.3 The role of end-to-end gating

The learned softmax condition achieves the highest phase-expert NMI across both tasks (0.72 Can, 0.61 Square, on validation demonstrations) and the highest rollout success on Square (35.3%); on Can, its success rate matches the random prototype condition (73.3%) but remains below regime initialization (76.0%). It does so without a regime-initialization prior. This raises a natural question about the necessity of regime-derived seeding: if end-to-end optimization can discover a similarly organized partition, what does the initialization add?

Two observations bear on this question. First, the softmax condition uses a fundamentally different routing mechanism — a parameterized gating network rather than a fixed-form Voronoi partition — so the comparison conflates initialization with architecture. The NMI agreement may be coincidental rather than reflecting equivalent routing dynamics. Second, the softmax condition in the ablation operates without margin loss, a setting that may favor learned gating over prototype routing. Whether the comparison holds under the full training objective (including margin regularization) is untested in the ablation.

The comparison establishes that a non-prototype learned gate can produce high phase-expert NMI without regime-derived seeding. It does not isolate initialization because the softmax condition uses a different routing mechanism, and its NMI is higher than the regime condition on both tasks. The result is therefore an architectural diagnostic, not an isolated measurement of prototype initialization.


## 6.4 Limitations

**Statistical power.** All comparisons are based on three training seeds. The paired sign tests produce no significant results after Holm correction ($p = 1.0$ throughout Table A15). The observed differences in success rate — including the 15-percentage-point advantage on Can in the sweep — are not resolved by the available seed-level sample. The results describe the observed runs, not a population-level effect.

**Prior phase supervision in representation.** The Stage 1 representation is pre-trained using rule-derived `phase` classification and supervised contrastive objectives. The matched initialization ablation therefore isolates the specific contribution of prototype initialization on top of a latent space that is already structured by external phase heuristics. Our study does not establish whether trajectory-derived regime initialization alone can induce routing organization without this prior phase supervision.

**Fixed expert count.** The regime and expert counts are fixed a priori at $K = E = 6$ across all tasks. The pipeline does not adaptively determine the optimal number of experts, which may under-parameterize long-horizon assembly tasks (e.g., ToolHang) or over-partition simpler reaching motions.

**Observation modality.** All evaluations are conducted exclusively with low-dimensional physical state vectors. Visual observations (RGB or depth images) are not included. This constitutes an architectural scope boundary; whether image-based representations interact differently with prototype initialization remains untested, and the absence of visual input should not be interpreted as an explanation for task performance differences.

**Task coverage.** Two of the five benchmark tasks (ToolHang and Transport) remain unresolved by all evaluated methods under the given memoryless policy contract and training budget, providing no discriminative evidence regarding the initialization hypothesis in multi-stage or bimanual regimes.

**Memoryless policy constraint.** The evaluation is conducted entirely under a memoryless policy contract — no recurrent state, no action chunking, no observation history. Whether regime-derived initialization produces different effects under history-conditioned or sequence-prediction architectures is unknown.

**Phase-label validity and contract separation.** The regime discovery pipeline produces kinematic regime labels (`phase_topo`) from trajectory statistics, whereas representation pre-training and NMI evaluation rely on heuristic rule-derived labels (`phase`). While `phase_topo` labels are verified for observability from instantaneous state, their correspondence to physical contact dynamics is assumed, not measured. Furthermore, the teacher-forced diagnostic uses offline-generated `phase_topo` labels during Stage 2 training but phase-head predictions during rollout evaluation; its poor performance reflects this deployment routing mismatch rather than an oracle evaluation.

**Sensitivity to task-variable scaling.** Trajectory segmentation operates on unweighted concatenations of heterogeneous physical dimensions (meters, quaternions, joint positions). Because the squared-Euclidean cost does not normalize by variable variance, dimensions with larger numeric ranges exert disproportionate influence on change-point locations. Investigating scale-invariant or metric-normalized segmentation costs is left for future work.

**Unmeasured quantities.** The current evaluation does not measure contact forces, friction-cone satisfaction, grasp stability, or the physical consequences of action discontinuities at expert transitions. The action-jump metric ($\|a_t - a_{t-1}\|_2$) is an auxiliary kinematic diagnostic; its relationship to contact-level failure modes is not established.

---

# 7. Conclusion

This paper tested whether the kinematic regime structure of demonstration trajectories can be used to initialize the routing partition of a mixture-of-experts policy, and whether the resulting organization predicts closed-loop control performance.

The answer to the first question is yes within the evaluated setting. Trajectory-derived regime prototype placement produces a routing partition that remains substantially more phase-aligned than random or rule-based initialization after joint fine-tuning, with mean NMI values against rule-derived phase labels of 0.51–0.67 on validation demonstrations compared with 0.07–0.09 for unstructured prototype initializations under matched training. The initialization geometry has an observed persistent effect on the learned modular structure — the starting prototype placement is associated with the character of the expert specialization that emerges.

Regarding the second question, the empirical evidence does not establish uniform improvement in closed-loop control. In the five-task benchmark, the regime-initialized policy records a higher observed success rate on Can (78% vs. 63% for BC in closed-loop rollouts), but in the matched ablation on Square, it trails the matched random and rule-based prototype initializations (26.7% vs. 28.0% and 31.3%) as well as the separate learned softmax diagnostic (35.3%), where the most phase-aligned prototype partition produces the lowest rollout success rate. A more structured routing partition does not reliably produce a superior controller.

The empirical dissociation between these two outcomes in this setting is the main finding. Within the matched prototype ablation, both outcomes are evaluated under changes to initialization geometry, but they are not equivalent quantities, and treating offline routing organization as a proxy for closed-loop control quality would produce misleading conclusions. Within the evaluated memoryless direct-action MoE setting, this result shows that structural alignment of the routing partition is not sufficient to guarantee improved control. Whether the same relationship holds for history-conditioned or other modular policy architectures remains to be tested.

The result also identifies a concrete open question for prototype-based routing: when the behaviorally relevant transitions in a task do not coincide with the kinematic regime boundaries discovered by the segmentation pipeline, a phase-aligned partition may assign the wrong expert to the precision-critical region. Whether this limitation can be addressed by richer trajectory representations, adaptive segmentation, or alternative routing mechanisms remains an open question for future investigation.
