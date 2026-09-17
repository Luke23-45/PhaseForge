# 3. Method

A manipulation trajectory typically traverses several kinematically distinct regimes — approaching, grasping, transporting, placing — each of which requires a different local mapping from state to action. Rather than learning a single monolithic policy that must internally partition its capacity among these regimes, we structure the policy as a mixture of experts whose routing partition is seeded from the geometric organization of the demonstration data itself.

This section defines the four components of the approach: the modular policy architecture (§3.1), the offline procedure that discovers behavioral regimes from demonstration trajectories (§3.2), the multi-task representation pre-training that shapes the latent space (§3.3), and the prototype initialization and joint fine-tuning that anchors the routing partition to the discovered structure (§3.4).


## 3.1 Policy Architecture

At each timestep $t$, the environment presents a state $x_t \in \mathbb{R}^D$ and the policy returns a bounded action $a_t \in [-1, 1]^A$. The policy is deterministic and memoryless: the action depends only on the current observation $x_t$, with no recurrent state or trajectory history.

### Shared encoder

A feedforward encoder $f_\phi$ maps the normalized observation into a $d$-dimensional latent vector, followed by projection onto the unit hypersphere:

$$z_t = \frac{f_\phi(x_t)}{\| f_\phi(x_t) \|_2}$$

The resulting representation $z_t \in \mathbb{S}^{d-1}$ is shared by the router and all experts. Projecting onto the unit sphere ensures that distances between representations reflect directional relationships rather than activation magnitude, and provides a natural domain on which to define prototype-based routing.

### Prototype router

The routing function maintains $E$ trainable prototype vectors $\{c_k\}_{k=1}^E$, each initialized on $\mathbb{S}^{d-1}$. The router selects the expert whose prototype is nearest to the current representation:

$$k_t^* = \arg\min_{k \in \{1, \ldots, E\}} \| z_t - c_k \|_2$$

Because both $z_t$ and $c_k$ lie on the unit sphere, this is equivalent to selecting the prototype with maximum cosine similarity: $\| z_t - c_k \|_2^2 = 2 - 2\, z_t^\top c_k$.

Hard top-1 selection partitions the latent sphere into a Voronoi tessellation

$$\mathcal{V}_k = \bigl\{ z \in \mathbb{S}^{d-1} : \| z - c_k \|_2 \le \| z - c_j \|_2 \;\; \forall\, j \ne k \bigr\}$$

in which each cell $\mathcal{V}_k$ maps exclusively to expert $k$. The partition is determined entirely by prototype placement: moving a prototype reshapes the cell boundaries and thereby changes which states each expert is responsible for.

### Expert networks and action execution

Each expert $e_k : \mathbb{S}^{d-1} \to (-1, 1)^A$ is an independent feedforward network with a $\tanh$ output nonlinearity. At execution time, only the selected expert produces the action:

$$a_t = e_{k_t^*}(z_t)$$

The policy is therefore piecewise-defined across the Voronoi cells: within each cell, the action is a smooth function of the representation, but a discontinuity can occur at cell boundaries where the active expert changes.


## 3.2 Trajectory Topology Discovery

The routing partition described above is parameterized by the prototype locations $\{c_k\}$. In standard mixture-of-experts training, these would be initialized randomly and learned end-to-end. The central methodological choice in this work is instead to derive the initial prototype placement from the geometric structure of the demonstration trajectories. This subsection describes the offline pipeline that extracts that structure.

### Task-variable signal

From each demonstration, we extract a kinematic signal $s_t$ that captures end-effector configuration and its spatial relationship to the manipulated object:

$$s_t = \bigl[ \, p_t, \;\; q_t, \;\; g_t, \;\; o_t, \;\; (p_t - o_{t,\, 0\!:\!3}), \;\; \alpha_t \, \bigr]$$

where $p_t \in \mathbb{R}^3$ is the end-effector position, $q_t \in \mathbb{S}^3$ is the orientation quaternion (sign-canonicalized to non-negative scalar part), $g_t$ denotes the gripper joint state, $o_t$ contains the object's proprioceptive variables, and $\alpha_t$ is a scalar gripper aperture. The relative displacement $(p_t - o_{t,\, 0\!:\!3})$ isolates object-relative motion from workspace-absolute motion. This signal operates in the physical task-variable space, not in the learned latent space.

### Change-point segmentation

Each trajectory is partitioned into contiguous intervals of stationary kinematic behavior. Change-points $0 = \tau_0 < \tau_1 < \cdots < \tau_M = T$ are obtained by solving an exact dynamic program that minimizes the total within-segment variance under a complexity penalty:

$$\min_{\{\tau_j\}_{j=0}^M} \; \sum_{j=0}^{M-1} C(s_{\tau_j : \tau_{j+1}}) \;+\; \beta \, (M - 1), \qquad \text{subject to} \quad \tau_{j+1} - \tau_j \ge L_{\min}$$

where the segment cost measures squared deviation from the segment mean:

$$C(s_{i:j}) = \sum_{t=i}^{j-1} \| s_t - \bar{s}_{i:j} \|_2^2$$

The penalty $\beta$ controls the granularity of the decomposition, and the minimum-duration constraint $L_{\min}$ prevents over-fragmentation at transient sensor fluctuations. The optimization is deterministic and exact.

### Regime clustering

Individual trajectory segments are summarized by their first and second moments:

$$\phi_j = \bigl[ \operatorname{mean}(s_{\tau_j : \tau_{j+1}}), \;\; \operatorname{var}(s_{\tau_j : \tau_{j+1}}) \bigr]$$

These summary vectors are pooled across all training demonstrations and partitioned into $K$ discrete behavioral regimes via centroid-based clustering. Each timestep inherits the regime label of its enclosing segment, producing a per-timestep assignment $r_t \in \{1, \ldots, K\}$.

The result is a decomposition of the demonstration data into regimes that correspond to kinematically coherent intervals — approach, contact, transport, placement — identified purely from the statistical structure of the task-variable trajectories.

### Observability verification

The regime labels are derived from trajectory-level segmentation, which has access to temporal context. For these labels to be usable as a routing prior in a memoryless policy, they must be recoverable from instantaneous state alone. We verify this by training a linear classifier to predict the regime label from a single normalized observation $x_t$, evaluated under trajectory-grouped cross-validation. Regimes are retained only if the probe exceeds a minimum classification threshold and each regime meets a minimum occupancy requirement.


## 3.3 Representation Pre-Training (Stage 1)

Before the modular architecture is instantiated, the encoder $f_\phi$ is pre-trained to produce a representation that is simultaneously informative about the demonstrated actions and geometrically organized with respect to the discovered regimes. This stage trains the encoder together with two auxiliary heads — an action prediction head and a regime classification head — under a composite objective:

$$\mathcal{L}_1 = \mathcal{L}_{\text{act}} + \lambda_{\text{phase}} \, \mathcal{L}_{\text{phase}} + \lambda_{\text{sc}} \, \mathcal{L}_{\text{SupCon}}$$

**Action loss.** The action head predicts a deterministic action $a_i^{\text{gen}}$ from the latent representation and is trained by mean squared error against the demonstration target $a_i^*$:

$$\mathcal{L}_{\text{act}} = \frac{1}{|B|\, A} \sum_{i \in B} \sum_{d=1}^{A} \bigl( a_{i,d}^{\text{gen}} - a_{i,d}^* \bigr)^2$$

**Regime classification loss.** A linear classifier maps $z_i$ to logits over the $K$ discovered regimes and is trained with cross-entropy:

$$\mathcal{L}_{\text{phase}} = - \frac{1}{|B|} \sum_{i \in B} \log \frac{\exp(\ell_{i, y_i})}{\sum_{q=1}^{K} \exp(\ell_{i, q})}$$

where $y_i \in \{1, \ldots, K\}$ is the regime label assigned by topology discovery.

**Supervised contrastive loss.** To impose metric structure on $\mathbb{S}^{d-1}$, a supervised contrastive objective pulls representations with the same regime label toward each other while pushing apart representations from different regimes:

$$\mathcal{L}_{\text{SupCon}} = \frac{1}{|I|} \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i^\top z_p \,/\, \tau)}{\sum_{a \ne i} \exp(z_i^\top z_a \,/\, \tau)}$$

Here, $P(i) = \{p \in B \setminus \{i\} : y_p = y_i\}$ is the set of batch elements sharing the regime label of anchor $i$, and $I = \{i : |P(i)| > 0\}$ excludes singletons. The temperature $\tau$ controls the sharpness of the similarity distribution.

The combined objective ensures that by the end of Stage 1, the encoder maps demonstrations from the same behavioral regime into neighboring regions of $\mathbb{S}^{d-1}$, while maintaining enough action-predictive information to seed the downstream experts.


## 3.4 Prototype Initialization and Joint Fine-Tuning (Stage 2)

After Stage 1, the auxiliary heads are detached, and the modular architecture — router and experts — is instantiated for joint training. The central step is the initialization of the routing prototypes from the latent geometry established in Stage 1.

### Prototype initialization

For each regime $k$, the routing prototype $c_k$ is placed at the $L_2$-normalized centroid of all Stage 1 representations belonging to that regime:

$$\tilde{c}_k = \frac{1}{N_k} \sum_{i:\, r_i = k} z_i, \qquad c_k = \frac{\tilde{c}_k}{\| \tilde{c}_k \|_2}$$

where $z_i$ are the frozen Stage 1 latent vectors and $N_k$ is the number of samples in regime $k$. This places the initial Voronoi boundaries at the natural cluster interfaces on $\mathbb{S}^{d-1}$: at initialization, the routing partition aligns with the regime structure that the representation was explicitly trained to separate.

### Expert initialization

Each expert network is seeded from the Stage 1 action head. To break symmetry between experts while preserving baseline action-prediction competence, a fixed proportion of hidden-layer weights is re-initialized independently per expert. The remaining weights retain their pre-trained values.

### Joint fine-tuning objective

The encoder, prototypes, and expert parameters are optimized jointly under:

$$\mathcal{L}_2 = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{bal}} + \lambda_m \, \mathcal{L}_{\text{margin}}$$

**Action loss.** Identical in form to Stage 1, but now evaluated on the output of the selected expert $e_{k_i^*}(z_i)$ rather than the monolithic action head.

**Balance loss.** A soft load-balancing penalty prevents degenerate partitions in which a subset of experts captures the entire data distribution:

$$\mathcal{L}_{\text{bal}} = \lambda_{\text{bal}} \, E \sum_{k=1}^{E} f_k \, p_k$$

where $f_k = \frac{1}{|B|} \sum_{i \in B} \mathbf{1}[k_i^* = k]$ is the hard assignment fraction and $p_k = \frac{1}{|B|} \sum_{i \in B} \operatorname{softmax}(-d_i)_k$ is the mean soft routing probability for expert $k$. The product $f_k \, p_k$ is large only when expert $k$ both receives many hard assignments and has high average soft affinity — penalizing concentration on both axes simultaneously.

**Margin loss.** An explicit distance margin $m$ separates the target-regime prototype from all alternatives:

$$\mathcal{L}_{\text{margin}} = \frac{1}{|B|} \sum_{i \in B} \sum_{j \ne y_i} \bigl[\, m - (d_{i,j} - d_{i, y_i}) \,\bigr]_+$$

where $d_{i,k} = \| z_i - c_k \|_2$ and $y_i$ is the regime label used as the margin target. This term encourages each representation to lie closer to its target prototype by at least margin $m$ than to any other prototype, sharpening the Voronoi boundaries without prescribing which expert ultimately captures which region.

### What adapts during Stage 2

All components remain trainable: the encoder adapts at a reduced learning rate, while prototypes and experts optimize at the base rate. The initial topology-derived partition therefore provides a structured starting point — not a frozen constraint — and the final partition reflects the combined influence of the initialization geometry and task-driven gradient updates.
