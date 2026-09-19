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
a_t
\right],
\]

where \(p_t\in\mathbb{R}^3\) is end-effector position, \(q_t\in\mathbb{S}^3\) is a sign-canonicalized orientation quaternion, \(g_t\) is raw gripper joint position, \(o_t\) contains object proprioceptive variables, and \(a_t = \max_j |g_{t,j}|\) is scalar gripper aperture excursion magnitude. The relative displacement term represents end-effector position relative to the object.

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