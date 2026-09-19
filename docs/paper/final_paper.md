Abstract
Mixture-of-experts (MoE) policies can represent the distinct control regimes within a manipulation task, but their routing partitions are usually learned without using the kinematic structure present in demonstrations. We test whether trajectory-derived kinematic regimes can serve as an initialization prior: demonstrations are segmented and clustered, and regime-conditioned centroids of a Stage-1 latent representation initialize hard-routing prototypes before joint fine-tuning. Stage 1 is trained with action prediction and rule-derived phase supervision.

Under a matched prototype-routing ablation, we compare trajectory-derived initialization with random and rule-based prototype initialization. We measure routing alignment with rule-derived phase labels and routing-switch rates on validation demonstrations, separately from closed-loop rollout success; a learned softmax gate serves as an architectural diagnostic.

Trajectory-derived initialization yields higher final NMI (0.67 on Can; 0.51 on Square) than random and rule-based initialization (0.07–0.09), with lower switch rates (0.04–0.07 versus 0.10–0.11). Its rollout success is 76.0% on Can, compared with 73.3% and 68.7% for random and rule-based initialization, and 26.7% on Square, compared with 28.0% and 31.3%. These comparisons use three independently trained seeds and are not statistically resolved after Holm adjustment. Thus, under matched conditions, trajectory-derived initialization changes final offline routing organization, but the study cannot determine whether those differences translate into closed-loop performance gains.

---

# Trajectory-Regime Initialization for Prototype-Routed Manipulation Policies: A Matched Study of Routing Organization and Closed-Loop Outcomes

# 1. Introduction

Manipulation tasks often combine approach, contact, transport, and terminal-placement behaviors that require different local state-to-action mappings. Mixture-of-experts (MoE) policies accommodate this heterogeneity by assigning observations to specialized experts, but their performance depends on how the routing function partitions the behavioral space.

Recent robotic MoE methods incorporate structure from demonstrations through supervised phase routing, routing regularization toward learned skill representations, or semantic skill routing [Mazza et al., 2026; Rodriguez et al., 2026; Deng et al., 2026]. We study a narrower intervention: whether trajectory-derived kinematic regimes can initialize the prototype geometry of a hard-routing manipulation policy and influence its final routing organization after joint fine-tuning.

The regimes are used to construct initial prototype locations from a Stage-1 latent representation. Stage 1 is trained with action prediction and rule-derived phase supervision; trajectory-derived regimes are not themselves the source of that supervision.

To isolate initialization, we compare trajectory-derived, rule-based, and random prototype placement under matched prototype-routing conditions. We measure phase-expert alignment and routing-switch rates on validation demonstrations, and closed-loop rollout success separately.

Trajectory-derived initialization produces more phase-aligned, lower-switch routing on the validation distribution in the observed runs. Across three training seeds, closed-loop success comparisons between regime and matched control initializations are not statistically resolved.

We make three contributions:

1. **Trajectory-derived regime initialization.** We use change-point segmentation and clustering to derive trajectory regimes, then initialize MoE routing prototypes from regime-conditioned latent centroids.

2. **Matched evaluation of routing organization.** Under identical prototype-routing conditions, we show that trajectory-derived regime initialization produces higher alignment with rule-derived phase labels (NMI) and lower routing-switch rates on validation demonstrations relative to matched random and rule-based controls.

3. **Bounded outcome analysis.** In the matched Can/Square ablation, trajectory-derived initialization yields higher final offline routing organization than the matched initialization controls, while the associated closed-loop comparisons across three training seeds are not statistically resolved.

---

# 2. Related Work

This work connects mixture-of-experts routing, trajectory segmentation in robot learning, and prototype-based initialization. The central question is whether kinematic structure extracted from demonstrations can initialize an MoE routing partition, and whether the resulting routing organization predicts closed-loop control quality.

## 2.1 Mixture-of-experts in policy learning

Mixture-of-experts architectures divide computation among specialized subnetworks and use a gating function to select or weight expert outputs [Jacobs et al., 1991; Jordan and Jacobs, 1994]. In large-scale models, sparse routing enables increased capacity without evaluating every expert for every input [Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2022]. This literature also studies expert utilization, load balancing, and collapse, all of which affect whether experts develop distinct roles [Zoph et al., 2022; Zhou et al., 2022].

Recent robotic MoE systems use explicit structure to organize expert routing. MoE-ACT uses supervised structure for phase-structured surgical manipulation [Mazza et al., 2026]. LAR-MoE regularizes routing toward a learned skill representation obtained from demonstrations [Rodriguez et al., 2026]. SMoDP uses VLM-derived semantic skill information to route a diffusion policy [Deng et al., 2026]. These approaches differ in policy architecture, supervision, and routing objective from the setting studied here.

We examine whether kinematic regimes extracted from demonstration trajectories can initialize the prototype geometry of a memoryless hard-routing policy. In the focused matched ablation, the Stage-2 margin loss is disabled for every primary prototype-initialization arm; the arms therefore differ in prototype initialization rather than in continued phase- or regime-alignment supervision. The full five-task configuration is reported separately and retains its stated rule-phase-indexed margin objective.

Cluster-informed expert initialization has also appeared in adjacent MoE work. PADD initializes experts from clusters of teacher neurons within an LLM distillation pipeline [Peng et al., 2026]. This differs from our setting, which derives clusters from manipulation trajectories and initializes routing prototypes rather than expert weights.

## 2.2 Trajectory segmentation and behavioral phases

Trajectory segmentation identifies intervals with distinct motion or task-variable statistics. Movement-representation methods, including dynamic and probabilistic movement primitives, model trajectories through structured temporal components [Ijspeert et al., 2013; Paraschos et al., 2018]. Change-point methods identify boundaries at which the generating process changes, using online Bayesian inference, kernel-based tests, or penalized cost objectives [Adams and MacKay, 2007; Harchaoui and Cappé, 2007; Killick et al., 2012; Truong et al., 2020].

In robot learning, segmentation has supported subgoal discovery, option construction, primitive learning, and auxiliary representation objectives [Niekum et al., 2012; Konidaris et al., 2012; Krishnan et al., 2017b; Shiarlis et al., 2018]. These methods commonly use segments as skills, planning abstractions, or supervisory targets.

Our use of segmentation is narrower. Trajectory-derived regime labels group Stage-1 latent representations when constructing the initial routing prototypes. They do not define a fixed skill sequence or replace the separate rule-derived phase labels used for representation supervision. The intervention is therefore the initialization of routing geometry from trajectory-derived regimes.

## 2.3 Initialization, prototypes, and routing geometry

Initialization affects optimization by determining the parameter region from which learning begins [Glorot and Bengio, 2010; He et al., 2015; Mishkin and Matas, 2016]. Analyses of neural-network geometry further show that training trajectories can depend on the initial parameterization [Li et al., 2018; Fort et al., 2019]. In an MoE, early routing assignments influence which expert parameters receive updates [Dai et al., 2022]. Initial router geometry can therefore influence later specialization without fixing the final partition.

Prototype-based methods provide the routing mechanism used here. Prototypical networks and related metric-learning methods represent classes or groups by vectors in an embedding space and assign examples according to proximity [Vinyals et al., 2016; Snell et al., 2017; Sung et al., 2018]. Nearest-prototype assignment induces a Voronoi partition: each prototype defines the region routed to its associated expert. This paper adapts that geometry to hard expert dispatch in a control policy. Rather than learning prototype locations from an arbitrary initial state, it initializes them from latent centroids grouped by trajectory-derived regimes and then allows the encoder, prototypes, and experts to adapt jointly.

## 2.4 Position of This Work

The relevant gap is not whether trajectory segmentation, phase supervision, or MoE routing can each support robot learning; prior work establishes each independently. The question studied here is whether trajectory-derived regime structure can initialize a prototype-routing partition in a way that changes its learned organization after fine-tuning. To our knowledge, this question has not been directly evaluated in the robot-manipulation setting studied here.

We examine this question through a matched comparison of trajectory-derived, rule-based, and random prototype initialization on Can and Square. Routing alignment and switch rate are measured on validation demonstrations, while success is measured in closed-loop rollouts. This design separates the structural effect of initialization on routing organization from its effect on task performance.

---

# 3. Method

We study a deterministic, memoryless mixture-of-experts (MoE) policy. A trajectory-derived regime-discovery procedure initializes the routing partition, after which the encoder, routing prototypes, and experts are jointly fine-tuned.

## 3.1 Policy Architecture

At time \(t\), the policy maps an observation \(x_t \in \mathbb{R}^{D}\) to a bounded action \(a_t \in [-1,1]^A\). It has no recurrent state or access to trajectory history.

A feedforward encoder maps the normalized observation to a unit-length latent representation:

\[
z_t = \frac{f_\phi(x_t)}{\|f_\phi(x_t)\|_2},
\qquad z_t \in \mathbb{S}^{d-1}.
\]

The router maintains \(E\) trainable prototypes \(\{c_k\}_{k=1}^{E}\) and selects the nearest prototype:

\[
k_t^* = \arg\min_{k \in \{1,\ldots,E\}} \|z_t-c_k\|_2.
\]

When both \(z_t\) and \(c_k\) are unit-normalized, this rule is equivalent to maximum cosine similarity:

\[
\|z_t-c_k\|_2^2 = 2 - 2z_t^\top c_k.
\]

This equivalence applies to normalized centroid initialization. During Stage 2, prototypes remain trainable and are not constrained to stay on the unit sphere.

For a fixed encoder, the prototypes define a Voronoi partition of latent space:

\[
\mathcal{V}_k =
\left\{
z \in \mathbb{S}^{d-1} :
\|z-c_k\|_2 \leq \|z-c_j\|_2
\quad \forall j \neq k
\right\}.
\]

Each cell is assigned to one expert. The encoder and prototypes both change during fine-tuning, so the induced partition of observation space can change throughout training.

Each expert \(e_k:\mathbb{S}^{d-1}\rightarrow(-1,1)^A\) is a feedforward network with a \(\tanh\) output layer. At rollout, only the selected expert produces the action:

\[
a_t = e_{k_t^*}(z_t).
\]

The residual-feedback coefficient is fixed at \(\beta=0\) in the reported configuration. The evaluated policy is therefore a direct-action MoE rather than a feedback-residual controller. Hard routing can produce action discontinuities when the active expert changes.

## 3.2 Trajectory Regime Discovery

The number of trajectory regimes and experts is fixed before training:

\[
K=E=6.
\]

The method does not select the number of regimes or experts adaptively.

For each demonstration, we construct a task-variable signal

\[
s_t =
\left[
p_t,\;
q_t,\;
g_t,\;
o_t,\;
(p_t-o_{t,0:3}),\;
\alpha_t
\right],
\]

where \(p_t\in\mathbb{R}^3\) is end-effector position, \(q_t\in\mathbb{S}^3\) is a sign-canonicalized orientation quaternion, \(g_t\) is raw gripper joint position, \(o_t\) contains object proprioceptive variables, and \(\alpha_t=\max_j |g_{t,j}|\) is scalar gripper-aperture excursion magnitude. The relative displacement term represents end-effector position relative to the object.

Before segmentation, observations are denormalized into physical coordinates. The resulting signal concatenates Cartesian positions, unit quaternions, and joint variables without additional feature weighting.

Each trajectory is segmented into intervals with locally homogeneous task-variable statistics. Change-points

\[
0=\tau_0 < \tau_1 < \cdots < \tau_M=T
\]

minimize a penalized within-segment cost:

\[
\min_{\{\tau_j\}_{j=0}^{M}}
\sum_{j=0}^{M-1} C(s_{\tau_j:\tau_{j+1}})
+
\lambda_{\mathrm{cp}}(M-1),
\qquad
\text{subject to }
\tau_{j+1}-\tau_j \geq L_{\min},
\]

where

\[
C(s_{i:j}) =
\sum_{t=i}^{j-1}
\|s_t-\bar{s}_{i:j}\|_2^2.
\]

The penalty \(\lambda_{\mathrm{cp}}\) controls segmentation granularity, and \(L_{\min}\) prevents short segments.

Each segment is represented by its first and second moments:

\[
\phi_j =
\left[
\operatorname{mean}(s_{\tau_j:\tau_{j+1}}),\;
\operatorname{var}(s_{\tau_j:\tau_{j+1}})
\right].
\]

Segment summaries from the training demonstrations are clustered into \(K=6\) groups. Each timestep inherits its enclosing segment’s cluster assignment, yielding a regime label \(r_t\in\{1,\ldots,K\}\). The clusters describe statistically distinct kinematic regimes; names such as approach, contact, transport, and placement are post hoc descriptions rather than supervision used by the discovery procedure.

### Label contract

Two distinct label vocabularies are used:

| Label vocabulary | Source | Use |
|---|---|---|
| Rule-derived phase labels (`phase`) | Task-specific kinematic heuristics | Stage 1 phase classification, supervised contrastive learning, full-pipeline margin loss, and NMI evaluation |
| Trajectory-derived regime labels (`phase_topo`) | Change-point segmentation and segment clustering | Grouping Stage-1 latents for prototype initialization |

The two vocabularies are generated independently and need not share temporal boundaries or label identities.

Trajectory-derived regime labels (`phase_topo`) are computed from demonstration kinematics without manual phase annotation and are used to group Stage-1 latent representations for prototype initialization. Rule-derived phase labels (`phase`) supervise the Stage-1 phase-classification and supervised-contrastive objectives and define the reference vocabulary for NMI evaluation. Thus, regime discovery is annotation-free, but the representation used to construct prototypes is not phase-unsupervised.

The causal interpretation of prototype initialization is restricted to the focused matched ablation, where the Stage-2 margin coefficient is zero for every primary arm. The full five-task configuration retains its stated phase-indexed margin objective; its results are not interchangeable with those of the matched ablation.

Because deployment uses a memoryless router, we require the trajectory-derived regime labels to be predictable from an instantaneous observation. A linear probe predicts regime labels from normalized \(x_t\) under trajectory-grouped cross-validation. The regime artifact is accepted only when macro-F1 is at least \(0.60\) and every regime has occupancy of at least \(0.01\); otherwise, the configured gate rejects it. The complete validation protocol is reported in the appendix.

## 3.3 Representation Pre-Training

Before the MoE is instantiated, the encoder is trained with an action-prediction head and a rule-phase classification head. The Stage-1 objective is

\[
\mathcal{L}_1 =
\mathcal{L}_{\mathrm{act}}
+
\lambda_{\mathrm{phase}}\mathcal{L}_{\mathrm{phase}}
+
\lambda_{\mathrm{sc}}\mathcal{L}_{\mathrm{SupCon}}.
\]

The action head predicts the demonstration action and is trained by mean squared error:

\[
\mathcal{L}_{\mathrm{act}}
=
\frac{1}{|B|A}
\sum_{i\in B}
\sum_{d=1}^{A}
\left(a_{i,d}^{\mathrm{gen}}-a_{i,d}^{*}\right)^2.
\]

Let \(y_i\in\{1,\ldots,K\}\) denote the one-based mathematical representation of the rule-derived phase label. A linear head predicts logits \(\ell_i\) and is trained with cross-entropy:

\[
\mathcal{L}_{\mathrm{phase}}
=
-\frac{1}{|B|}
\sum_{i\in B}
\log
\frac{\exp(\ell_{i,y_i})}
{\sum_{q=1}^{K}\exp(\ell_{i,q})}.
\]

The supervised contrastive term pulls together latent representations with the same rule-derived phase label and separates representations with different labels [Khosla et al., 2020]. Thus, Stage 1 makes the latent space action-predictive while organizing it according to rule-derived phases.

Trajectory-derived regime labels are not targets of the Stage-1 classification or contrastive objectives.

## 3.4 Prototype Initialization and Joint Fine-Tuning

After Stage 1, the auxiliary heads are detached. The router and experts are instantiated, and the encoder, prototypes, and experts are jointly optimized.

For trajectory-derived initialization, prototype \(c_k\) is the normalized centroid of Stage-1 latent representations assigned to regime \(k\):

\[
\tilde{c}_k =
\frac{1}{N_k}
\sum_{i:r_i=k} z_i,
\qquad
c_k =
\frac{\tilde{c}_k}{\|\tilde{c}_k\|_2}.
\]

Rule-based initialization uses the same construction with rule-derived phase labels in place of \(r_i\). The random control uses the router’s standard small random initialization. These conditions differ only in the labels or distribution used to set the initial prototype locations.

Each expert is initialized from the Stage-1 action head. To break symmetry, a fixed subset of hidden-layer parameters is independently reinitialized for each expert; the remaining parameters retain their Stage-1 values.

Stage 2 optimizes

\[
\mathcal{L}_2 =
\mathcal{L}_{\mathrm{act}}
+
\mathcal{L}_{\mathrm{bal}}
+
\lambda_m\mathcal{L}_{\mathrm{margin}}.
\]

The action loss has the same form as in Stage 1, but uses the action produced by the selected expert \(e_{k_i^*}(z_i)\).

The balance term discourages concentration on a small subset of experts:

\[
\mathcal{L}_{\mathrm{bal}}
=
\lambda_{\mathrm{bal}}E
\sum_{k=1}^{E} f_kp_k,
\]

where

\[
f_k =
\frac{1}{|B|}
\sum_{i\in B}\mathbf{1}[k_i^*=k],
\qquad
p_k =
\frac{1}{|B|}
\sum_{i\in B}
\operatorname{softmax}(-d_i)_k,
\]

and \(d_{i,k}=\|z_i-c_k\|_2\). The hard-assignment fraction \(f_k\) measures expert usage, while \(p_k\) is a differentiable soft affinity. Gradients through \(p_k\) update prototype locations.

The full configuration also includes a margin loss:

\[
\mathcal{L}_{\mathrm{margin}}
=
\frac{1}{|B|}
\sum_{i\in B}
\sum_{j\neq\pi(y_i)}
\left[
m-\left(d_{i,j}-d_{i,\pi(y_i)}\right)
\right]_+.
\]

Here, \(\pi(y_i)=y_i\) in one-based mathematical notation. Implementation arrays use zero-based indices, with the corresponding offset applied before indexing prototype columns. The mapping is fixed by label index; no semantic matching is learned between rule-derived phase labels and trajectory-derived regime-cluster labels.

The matched Can/Square initialization ablation sets \(\lambda_m=0\) for every arm. It therefore compares prototype initialization without the margin-based coupling between the two label vocabularies.

The hard routing decision is non-differentiable. No straight-through estimator is used. The action loss updates the selected expert and encoder within the current routing assignment, but does not directly update prototype locations through the discrete \(\arg\min\) operation. Prototype updates arise from the differentiable balance term and, when enabled, the margin loss. The encoder, prototypes, and experts remain trainable throughout Stage 2, so initialization defines a starting partition rather than a fixed expert assignment.

---

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

---

# 5. Results

The five-task benchmark and the focused ablation use different Stage-2 objectives. The full benchmark uses the regime-initialized configuration with active margin loss \((\lambda_m>0)\). The matched Can/Square ablation disables margin loss in every prototype arm \((\lambda_m=0)\). Their absolute success rates should therefore not be compared directly.

## 5.1 Five-Task Benchmark

Table 1 reports closed-loop success for all deployable methods. Brackets denote 95% Wilson score intervals over pooled rollout episodes. These intervals describe episode-level uncertainty, not variation across training seeds.

Lift is saturated: all methods except Scratch MoE and Factorial Floor reach 100% success. It does not distinguish the modular architectures.

On Can, the full regime-initialized configuration records 78% success \([71,84]\), compared with 63% \([55,70]\) for BC, 69% \([61,76]\) for Learned Softmax Top-1, and 62% \([54,69]\) for Phase-Random. Its paired difference from BC is \(+0.153\) across the three seeds. The Holm-adjusted paired sign test does not reject the null (\(p=1.0\); Table A15), so this ordering is descriptive.

Square has a different ranking. Plain Encoder records 50% success \([42,58]\), followed by the full regime-initialized configuration at 41% \([33,49]\), with Learned Softmax Top-1 and Phase-Random at 39%. The paired difference between the full regime-initialized configuration and Plain Encoder is \(-0.093\). The action-only representation control records a higher observed success rate than the full regime-initialized configuration on this task.

ToolHang remains unsolved: every evaluated method has 0% success. Transport produces 0–2 successful episodes out of 150, depending on the method. These tasks remain unresolved under the evaluated memoryless policies and training budget, so they do not distinguish the initialization conditions.

The full five-task sweep provides contextual coverage rather than a causal estimate of prototype initialization. Several tasks are saturated or near the success floor, and the full configuration differs from the matched ablation through its active margin objective.

## 5.2 Controlled Initialization Ablation

Table 2 reports the focused Can/Square comparison. The three matched prototype arms use the same phase-aware Stage-1 representation, expert initialization, Stage-2 optimizer, and Stage-2 objective. They differ only in prototype initialization:

1. trajectory-derived regime centroids;
2. rule-derived phase centroids;
3. random prototype parameters.

Margin loss is disabled in every matched arm.

### Offline routing organization

Routing metrics are computed on validation demonstrations. Trajectory-derived initialization has the highest mean NMI among the matched prototype arms on both tasks: 0.67 on Can and 0.51 on Square. Random and rule-based initialization produce NMI values between 0.07 and 0.09.

Trajectory-derived initialization also has the lowest mean routing-switch rate: 0.04 on Can and 0.07 on Square, compared with 0.10–0.11 for the random and rule-based conditions. These values describe routing assignments on the validation-demonstration distribution, not on rollout states.

Rule-based prototype initialization produced final NMI values similar to random initialization under the matched configuration. The endpoint measurements do not identify whether this reflects initial prototype geometry, prototype scale or separation, expert utilization, or subsequent training dynamics.

### Closed-loop success

On Can, trajectory-derived initialization records the highest observed success among the matched prototype arms: 76.0%, compared with 73.3% for random initialization and 68.7% for rule-based initialization.

On Square, the ordering reverses. Trajectory-derived initialization records 26.7% success, compared with 28.0% for random initialization and 31.3% for rule-based initialization.

The matched arms differed substantially in final offline routing diagnostics. The observed rollout orderings on Can and Square are descriptive and do not establish a task-by-initialization interaction. Across three training seeds, the closed-loop comparisons were not statistically resolved after Holm adjustment.

### Architectural diagnostics

Learned Softmax Top-1 is not a matched initialization control because it replaces prototype routing with a learned gating network. It reaches NMI values of 0.72 on Can and 0.61 on Square, with rollout success of 73.3% and 35.3%, respectively. This condition shows that high phase-expert alignment can arise without trajectory-derived prototype initialization. It does not isolate the effect of gating or prototype initialization.

Plain Encoder is also diagnostic rather than matched because it changes representation pre-training. Its values are reported in Table 2 but do not estimate the isolated effect of prototype initialization.

Teacher-Forced dispatches experts with trajectory-derived regime labels during Stage-2 training, then uses predictions from a frozen rule-phase head during rollout. It records 41% success on Lift, 1% on Can, and 2% on Square. This condition evaluates a label-and-dispatch mismatch between training and deployment. It is not an oracle-routing experiment and does not establish that either label vocabulary is independently useful or ineffective.

---

# 6. Discussion and Limitations

The matched ablation supports two observations. Trajectory-derived regime initialization changes offline routing organization after joint fine-tuning. That organization does not consistently predict closed-loop success across Can and Square. The result concerns a memoryless, hard-routing MoE under the evaluated training configuration.

## 6.1 Initialization and Routing Organization

Among the three matched prototype-initialization arms, trajectory-derived initialization produces higher NMI and lower routing-switch rates than rule-based and random initialization on validation demonstrations. The encoder, prototypes, and experts remain trainable during Stage 2. The reported difference is therefore a final-checkpoint observation after joint adaptation; without post-initialization and training-trajectory diagnostics, the study does not identify how that difference emerged.

The experiment does not identify an optimization mechanism. It does not establish a particular loss-landscape basin, nor does it show why the rule-based initialization produces NMI values similar to random initialization. The rule-based result only shows that non-random prototype placement is not sufficient to produce high phase-expert alignment in the observed runs.

Hard routing further bounds the interpretation. The action loss updates the selected expert and encoder but does not directly update prototype locations through the discrete routing decision. In the matched ablation, margin loss is disabled, so prototypes receive direct gradient signal only through the balance term. The ablation therefore characterizes an initialization-sensitive hard prototype router, not a routing mechanism whose prototypes are directly optimized by task loss.

## 6.2 Limits of Offline Routing Organization

The Square ablation shows that offline routing organization and rollout success need not rank methods in the same order. Among the matched prototype arms, trajectory-derived initialization has the highest NMI on validation demonstrations and the lowest rollout success on Square. On Can, the same initialization has the highest observed NMI and success among the matched prototype arms.

NMI and rollout success measure different quantities on different state distributions. NMI measures association between expert assignments and rule-derived phase labels on validation demonstrations. Success measures closed-loop behavior from rollout states. The observed ordering on Square shows that offline phase alignment is not sufficient to predict control quality in this setting.

The observed Square ordering is best interpreted as a boundary on the offline metrics rather than evidence for an unmeasured failure mechanism. Phase--expert NMI and routing-switch rate characterize assignment structure on held-out demonstrations, whereas rollout success also depends on the actions produced by the selected expert along states visited during execution. Because this study does not intervene on routing while holding expert behavior fixed, it cannot identify the mechanism behind the observed Square ordering. With three training seeds, the study cannot determine whether the observed offline routing differences translate into closed-loop control gains.

## 6.3 Softmax as an Architectural Diagnostic

The Learned Softmax Top-1 condition reaches high NMI without trajectory-derived prototype initialization. It uses a learned gating network rather than a nearest-prototype Voronoi partition. This result shows that phase-aligned routing can emerge without regime-derived prototype seeding, but it does not isolate the source of the difference because routing mechanism and architecture both change. Softmax is therefore an architectural diagnostic, not an initialization control.

## 6.4 Limitations

**Inference and task coverage.** All comparisons use three training seeds. The paired tests do not establish seed-level performance differences after Holm correction. The results describe the observed runs rather than a population-level effect [Henderson et al., 2018; Agarwal et al., 2021]. ToolHang and Transport remain unresolved under the evaluated policy class and training budget, so they provide no discriminative evidence about initialization.

**Supervision and full-pipeline coupling.** The matched ablation evaluates prototype initialization on a representation already shaped by rule-derived phase classification and supervised contrastive learning. It does not establish whether trajectory-derived initialization alone can organize routing without that supervision. The full five-task benchmark additionally includes the fixed index-based margin coupling between rule-derived phase labels and prototype indices defined in §3.4. The benchmark therefore evaluates the combined system, not prototype initialization in isolation. This coupling is absent from the matched ablation because \(\lambda_m=0\).

**Policy capacity and task coverage.** Absolute completion was limited on Square and near zero on ToolHang and Transport in the reported sweep. The evaluated policy is memoryless and direct-action, with fixed expert count and low-dimensional state observations. The unresolved initialization comparisons may reflect limitations of routing, policy class, expert expressiveness, optimization, or task coverage; this study does not distinguish among these possibilities.

**Architecture scope.** The evaluation fixes \(K=E=6\), uses low-dimensional state observations, and restricts policies to memoryless direct-action control. The study does not determine how the result changes with adaptive expert counts, image observations, action chunking, or history-conditioned policies.

**Regime and metric validity.** Trajectory-derived regimes pass the specified observability probe, but their correspondence to dynamically relevant contact events is not measured. NMI measures association with rule-derived phase labels on validation demonstrations; it does not measure alignment with trajectory-derived regimes during rollout. Teacher-Forced uses trajectory-derived labels during training and predictions from a rule-phase head during deployment, so its poor performance reflects that label-and-dispatch mismatch rather than oracle routing quality.

**Unmeasured physical and routing quantities.** The study does not measure contact forces, grasp stability, rollout-time switching, action discontinuities, or the correspondence between routing boundaries and contact events. Segmentation also uses an unweighted concatenation of heterogeneous physical variables. Variable scale can influence the squared-Euclidean segmentation cost, but this sensitivity was not evaluated.

---

# 7. Conclusion

We evaluated whether trajectory-derived kinematic regimes can initialize the routing partition of a hard prototype-routed MoE policy. In the matched Can/Square ablation, regime-derived prototype initialization was associated with higher phase-expert alignment and lower routing-switch rates on validation demonstrations than rule-based or random prototype initialization.

The matched ablation produced large differences in final offline routing diagnostics, while the corresponding three-seed rollout comparisons were not statistically resolved. The study therefore cannot determine whether the observed routing differences translate into closed-loop control gains. It does not establish a universal performance benefit from regime-derived initialization, a mechanism for the observed Square ordering, or a conclusion about history-conditioned or vision-based policies. Future work should test whether the observed relationship changes with alternative routing mechanisms, observation modalities, and policy architectures.
