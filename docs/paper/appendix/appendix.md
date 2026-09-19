# Supplementary Material: PhaseForge Appendices

This supplementary document provides complete mathematical derivations, architectural specifications, training configurations, raw empirical tables, and reproducibility artifacts for **PhaseForge: Bootstrapped Mixture-of-Experts for Trajectory Regime Alignment in Robot Manipulation**.

---

# Appendix A — Regime Discovery and Label Contract

This appendix documents the mathematical formulation, signal transformations, hyperparameters, and validation protocols for the unsupervised trajectory regime discovery pipeline and its contract with the rule-derived phase labels.

---

## A.1 Task-Variable Construction and Physical-Unit Transformation

The trajectory segmentation pipeline operates on a kinematically grounded task-variable signal $s_t$ extracted from each demonstration timestep. The signal isolates end-effector kinematics, gripper state, and object-relative spatial relationships:

\[
s_t = \left[\, p_t,\; q_t,\; g_t,\; o_t,\; (p_t - o_{t,0:3}),\; a_t \,\right]
\]

where:
- $p_t \in \mathbb{R}^3$: Cartesian position of the robot end-effector in meters ($\mathrm{m}$).
- $q_t \in \mathbb{S}^3$: Orientation quaternion of the end-effector. To eliminate the double-cover antipodal ambiguity of $\mathbb{S}^3$ (where $q$ and $-q$ represent identical physical rotations), quaternions are sign-canonicalized to possess a non-negative scalar component:
  \[
  q_t \leftarrow \operatorname{sign}(q_{t,0}) \cdot q_t, \qquad q_{t,0} \ge 0.
  \]
- $g_t = [g_{t,1}, \dots, g_{t,J}] \in \mathbb{R}^J$: Raw gripper joint positions (e.g., Franka Panda finger joint positions $g_{t,1}, g_{t,2} \in [-0.04, 0.04]\,\mathrm{m}$).
- $o_t$: Proprioceptive state of the primary manipulated object, including its Cartesian position and orientation.
- $(p_t - o_{t,0:3}) \in \mathbb{R}^3$: Spatial displacement vector from the manipulated object to the robot end-effector, isolating object-relative approach and alignment from workspace-absolute translation.
- $a_t \in \mathbb{R}$: Scalar gripper aperture excursion magnitude, derived directly from the gripper finger joint positions:
  \[
  a_t = \max_j |g_{t,j}|.
  \]
  This formulation derives aperture strictly from the physical finger joint excursions rather than the end-effector orientation quaternion, and is invariant to antisymmetric finger coordinate conventions where $\operatorname{mean}(g_{t,1}, g_{t,2}) \approx 0$ regardless of grasp state.

### Physical-Unit Transformation and Unweighted Scaling

Prior to segmentation, normalized simulator states are denormalized back to their physical coordinates:
- Cartesian positions are represented in meters ($\mathrm{m}$).
- Rotations are represented as sign-canonicalized unit quaternions on $\mathbb{S}^3$.
- Gripper finger coordinates are represented in joint coordinates ($\mathrm{m}$ or $\mathrm{rad}$).

The concatenated signal $s_t$ is unweighted: no artificial dimension-wise scaling or variance normalization is applied prior to change-point optimization. The squared-Euclidean distance metric in the segmentation cost operates directly on this heterogeneous physical representation.

---

## A.2 Segmentation and Clustering Configuration

### Change-Point Detection via Exact PELT

Each demonstration trajectory of length $T$ is partitioned into $M$ contiguous intervals defined by change-points $0 = \tau_0 < \tau_1 < \dots < \tau_M = T$. The change-points are computed by solving an exact dynamic program under the Pruned Exact Linear Time (PELT) formulation [Killick et al., 2012]:

\[
\min_{\{\tau_j\}_{j=0}^M} \;\; \sum_{j=0}^{M-1} C(s_{\tau_j:\tau_{j+1}}) \;+\; \lambda_{\mathrm{cp}}(M-1), \qquad \text{subject to } \tau_{j+1} - \tau_j \ge L_{\min}
\]

where the segment cost function $C(s_{i:j})$ measures the total sum of squared deviations from the empirical segment mean $\bar{s}_{i:j} = \frac{1}{j-i} \sum_{t=i}^{j-1} s_t$:

\[
C(s_{i:j}) = \sum_{t=i}^{j-1} \| s_t - \bar{s}_{i:j} \|_2^2.
\]

The hyperparameters are fixed across all tasks:
- **Complexity penalty:** $\lambda_{\mathrm{cp}} = 10.0$, balancing segmentation granularity against model complexity.
- **Minimum segment duration:** $L_{\min} = 5$ timesteps ($250\,\mathrm{ms}$ at $20\,\mathrm{Hz}$ control frequency), preventing spurious over-fragmentation from high-frequency sensor noise.

### Segment Moment Summarization and K-Means Clustering

Each discovered segment $j$ spanning interval $[\tau_j, \tau_{j+1})$ is compressed into its first and second temporal moments:

\[
\phi_j = \left[\, \operatorname{mean}(s_{\tau_j:\tau_{j+1}}),\; \operatorname{var}(s_{\tau_j:\tau_{j+1}}) \,\right] \in \mathbb{R}^{2 \dim(s_t)}.
\]

The summary vectors $\{\phi_j\}$ from all training demonstration trajectories are pooled into a single dataset and partitioned into $K = 6$ discrete behavioral clusters using standard $K$-means clustering:
- **Cluster count:** $K = 6$ (fixed *a priori* across all tasks).
- **Initialization:** $K$-means++ initialization with $n_{\mathrm{init}} = 10$ restarts.
- **Convergence tolerance:** $10^{-4}$ on cluster inertia.
- **Random seed:** Pinned to the experiment seed (`seed=42`).

Every timestep $t$ in a demonstration inherits the discrete cluster label of its enclosing segment, generating a per-timestep trajectory regime label $r_t \in \{0, \dots, K-1\}$.

---

## A.3 Rule-Derived versus Trajectory-Derived Label Contract

The PhaseForge framework maintains two completely distinct label vocabularies. The two vocabularies are generated through independent pipelines and fulfill strictly separated functions:

| Dimension | Rule-Derived Phase Labels (`phase`) | Trajectory-Derived Regime Labels (`phase_topo`) |
| :--- | :--- | :--- |
| **Generation source** | Hand-engineered kinematic state machine | Unsupervised PELT segmentation and $K$-means clustering |
| **Inputs inspected** | End-effector pose and adaptive gripper aperture | Full concatenated task-variable vector $s_t$ |
| **Temporal boundaries** | Exact kinematic threshold transitions | Minimum-variance change-points ($\tau_j$) |
| **Stage 1 role** | Supervised classification ($\mathcal{L}_{\mathrm{phase}}$) and contrastive learning ($\mathcal{L}_{\mathrm{SupCon}}$) | None (excluded from Stage 1 losses) |
| **Stage 2 role** | Margin loss target ($\mathcal{L}_{\mathrm{margin}}$, active in benchmark sweep only) | Grouping Stage-1 latents to seed prototype centroids ($c_k$) |
| **Evaluation role** | Ground-truth alignment target for validation NMI | None (NMI is not evaluated against `phase_topo`) |

### Adaptive Gripper Aperture Calibration for Rule Labels

The rule-based labeler (`RuleBasedPhaseLabeler`) implements an adaptive kinematic state machine. To avoid brittle task-specific thresholds across varying object geometries, the open and closed gripper thresholds are calibrated dynamically per demonstration:
1. **Aperture span:** The 5th percentile ($lo$) and 95th percentile ($hi$) of $a_t$ are extracted across the trajectory, defining span $\Delta = hi - lo$.
2. **Hysteresis bands:** Hysteresis switching levels are established 30% inside the observed span:
   \[
   \text{closed\_level} = lo + 0.3\,\Delta, \qquad \text{open\_level} = hi - 0.3\,\Delta.
   \]
3. **Fallback limits:** For demonstrations with near-zero aperture variation ($\Delta \le 10^{-3}\,\mathrm{m}$), the system defaults to fixed absolute thresholds: $a_{\mathrm{closed}} = 0.02\,\mathrm{m}$ and $a_{\mathrm{open}} = 0.04\,\mathrm{m}$.
4. **Kinematic filtering:** Phase transitions require an end-effector velocity threshold of $0.01\,\mathrm{m/s}$, a minimum dwell duration of $5$ timesteps, and a causal median filter window of size $7$.

---

## A.4 Observability Gate and Per-Task Audit

Because PhaseForge deploys a memoryless reactive policy $a_t = \pi(x_t)$, the trajectory-derived regime labels $r_t$ (which were discovered using temporal segment context) must be predictable from an instantaneous observation $x_t$. If a regime cannot be recovered from instantaneous state, seeding a memoryless spatial Voronoi partition from it is ill-posed.

### Validation Protocol

Before any regime artifact is accepted for policy training, it must pass a fail-closed linear probe audit implemented in `phaseforge/data/topo/observability.py`:
1. **Classifier:** A multi-class logistic regression probe is trained on instantaneous normalized observations $x_t \in \mathbb{R}^D$ to predict regime label $r_t \in \{0, \dots, K-1\}$.
2. **Trajectory-Aware Cross-Validation:** Grouped cross-validation (`GroupKFold`) with $n_{\mathrm{splits}} = \min(3, n_{\mathrm{groups}})$ over demonstration trajectories. Whole trajectories are assigned to folds, guaranteeing zero temporal overlap or adjacent-step frame leakage between train and validation splits.
3. **Configured Acceptance Thresholds:**
   - **Macro-F1:** Cross-validated macro-averaged F1 score must satisfy $\text{macro-F1} \ge 0.60$, ensuring balanced instantaneous inferability across all regimes.
   - **Minimum Regime Occupancy:** Every regime must satisfy an empirical occupancy fraction $\min_k \text{occupancy}(k) \ge 0.01$ ($1\%$), preventing degenerate, near-empty clusters from being routed on.

### Per-Task Audit Results

All five Robomimic benchmark datasets were audited against this gate prior to Stage-2 policy training. All tasks comfortably satisfied the fail-closed observability and occupancy criteria:

| Task | State Dim $D$ | Cross-Validated Macro-F1 | Minimum Regime Occupancy | Gate Status |
| :--- | :---: | :---: | :---: | :---: |
| **Lift** | 19 | 0.912 | 0.082 | **PASSED** |
| **Can** | 23 | 0.847 | 0.076 | **PASSED** |
| **Square** | 23 | 0.781 | 0.064 | **PASSED** |
| **ToolHang** | 53 | 0.742 | 0.058 | **PASSED** |
| **Transport** | 59 | 0.718 | 0.052 | **PASSED** |

Because all tasks satisfied the fail-closed gate ($\text{macro-F1} \ge 0.60$, $\text{min occupancy} \ge 0.01$), the trajectory-derived regime partitions were admitted for Stage-2 prototype initialization.

---

# Appendix B — Model, Training, and Control Definitions

This appendix specifies the complete neural network architectures, training objectives, optimization schedules, baseline definitions, and computational capacity accounting for all evaluated methods.

---

## B.1 Architecture and Parameterization

All policy models operate under a deterministic, memoryless contract mapping observation $x_t \in \mathbb{R}^D$ to bounded action $a_t \in [-1, 1]^A$.

### Shared Encoder
- **Trunk Architecture:** A 3-hidden-layer feedforward Multi-Layer Perceptron (MLP) with layer dimensions $[D \to 256 \to 256 \to 256 \to 128]$, implemented in `phaseforge/models/components/encoder.py`.
- **Activations and Regularization:** Gaussian Error Linear Unit (`GELU`) activations and a dropout rate of $0.10$ applied after each 256-dimensional hidden layer.
- **Residual Connection:** A residual shortcut bridges the input to the output. Because input dimension $D \ne 128$, a learned linear projection layer (`res_proj = nn.Linear(D, 128)`) projects the input to the latent dimension:
  \[
  h = \operatorname{Hidden}(x_t), \qquad \tilde{z}_t = W_{\mathrm{out}} h + W_{\mathrm{res}} x_t.
  \]
- **Hypersphere Projection:** The output representation is projected onto the 128-dimensional unit hypersphere $\mathbb{S}^{127}$ via $L_2$ normalization:
  \[
  z_t = \frac{\tilde{z}_t}{\| \tilde{z}_t \|_2} \in \mathbb{S}^{127}.
  \]
  This unit-norm representation is shared by both the routing mechanism and all expert networks.

### Prototype Router
- **Source Module:** Implemented in `phaseforge/models/components/prototype_router.py`.
- **Parameterization:** The router maintains $E = 6$ trainable prototype vectors $\{c_k\}_{k=1}^E$, where each $c_k \in \mathbb{R}^{128}$.
- **Initialization:** In the proposed regime-derived condition (`topo`) and rule-based condition (`rule`), prototypes are initialized as normalized latent centroids of Stage-1 representations for each cluster. In the random prototype control, prototypes are initialized as standard normal random vectors scaled to unit length.
- **Hard Top-1 Dispatch:** The router assigns input representation $z_t$ to the nearest prototype under Euclidean distance:
  \[
  k_t^* = \arg\min_{k \in \{1, \dots, E\}} \| z_t - c_k \|_2.
  \]
  During Stage 2 fine-tuning, prototype vectors are free parameters in $\mathbb{R}^{128}$, allowing both radial and directional adaptation.

### Expert Networks
- **Individual Architecture:** Implemented in `phaseforge/models/components/expert.py` and `phaseforge/models/components/impedance_expert.py`. Each expert is configured with a single hidden layer of width 256 (`hidden_dims=[256]`), mapping $128 \to 256 \to A$.
- **Activation and Output Squash:** Intermediate activation is `GELU`; the output layer is squashed with a hyperbolic tangent ($\tanh$) nonlinearity, enforcing the strict action contract $a_t \in (-1, 1)^A$. This single-hidden-layer structure matches the Stage-1 ActionHead architecture (`Linear(128, 256) -> GELU -> Linear(256, A) -> tanh`).
- **Residual Branch Setting:** In the evaluated confirmation configuration, the residual compliance feedback coefficient is set to $\beta = 0.0$. Experts therefore operate as direct-action networks:
  \[
  a_t = e_{k_t^*}(z_t).
  \]
- **Expert Symmetry Breaking via Partial Warm-Start:** Each expert is initialized from the Stage-1 ActionHead using Drop-Upcycling partial reinitialization [Nakamura et al., 2025] (`partial_reinit_experts_from_action_head`). The ActionHead weights are copied into each expert, and then an independent fraction ($\text{drop\_rate} = 0.50$) of each expert's intermediate hidden neurons are reinitialized using Kaiming uniform initialization [He et al., 2015] with fixed seed. The remaining 50% of the neurons retain their pre-trained ActionHead weights bit-exactly. This breaks functional symmetry across experts while guaranteeing functional initialization from the pre-trained generalist.

---

## B.2 Stage-1 and Stage-2 Objectives and Hyperparameters

### Stage 1: Representation Pre-Training

Stage 1 optimizes the shared encoder $f_\phi$ alongside an action-prediction head $g_{\mathrm{act}}$ and a linear phase-classification head $g_{\mathrm{phase}}$ under the composite objective:

\[
\mathcal{L}_1 = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{phase}} \, \mathcal{L}_{\mathrm{phase}} + \lambda_{\mathrm{sc}} \, \mathcal{L}_{\mathrm{SupCon}}
\]

1. **Action Prediction Loss ($\mathcal{L}_{\mathrm{act}}$):** Mean squared error against demonstration actions $a_i^*$:
   \[
   \mathcal{L}_{\mathrm{act}} = \frac{1}{|B|\, A} \sum_{i \in B} \sum_{d=1}^A \left( a_{i,d}^{\mathrm{gen}} - a_{i,d}^* \right)^2.
   \]
2. **Phase Classification Loss ($\mathcal{L}_{\mathrm{phase}}$):** Cross-entropy loss over rule-derived phase labels $y_i \in \{0, \dots, K-1\}$:
   \[
   \mathcal{L}_{\mathrm{phase}} = - \frac{1}{|B|} \sum_{i \in B} \log \frac{\exp(\ell_{i, y_i})}{\sum_{q=0}^{K-1} \exp(\ell_{i, q})}.
   \]
3. **Supervised Contrastive Loss ($\mathcal{L}_{\mathrm{SupCon}}$):** Encourages clustering of timesteps sharing identical phase labels [Khosla et al., 2020]:
   \[
   \mathcal{L}_{\mathrm{SupCon}} = \frac{1}{|I|} \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i^\top z_p \,/\, \tau)}{\sum_{a \ne i} \exp(z_i^\top z_a \,/\, \tau)}
   \]
   where $P(i) = \{p \in B \setminus \{i\} : y_p = y_i\}$, $I = \{i : |P(i)| > 0\}$, and temperature $\tau = 0.07$.
- **Loss weights:** $\lambda_{\mathrm{phase}} = 1.0$, $\lambda_{\mathrm{sc}} = 1.0$ (when enabled in configuration).

### Stage 2: Joint Modular Fine-Tuning

In Stage 2, auxiliary classification heads are removed. The encoder, prototypes, and experts are fine-tuned under:

\[
\mathcal{L}_2 = \mathcal{L}_{\mathrm{act}} + \mathcal{L}_{\mathrm{bal}} + \lambda_m \, \mathcal{L}_{\mathrm{margin}}
\]

1. **Modular Action Loss ($\mathcal{L}_{\mathrm{act}}$):** Evaluated exclusively on the action of the dispatched expert $e_{k_i^*}(z_i)$:
   \[
   \mathcal{L}_{\mathrm{act}} = \frac{1}{|B|\, A} \sum_{i \in B} \sum_{d=1}^A \left( e_{k_i^*}(z_i)_d - a_{i,d}^* \right)^2.
   \]
2. **Differentiable Balance Loss ($\mathcal{L}_{\mathrm{bal}}$):** Penalizes expert starvation using soft routing probabilities:
   \[
   \mathcal{L}_{\mathrm{bal}} = \lambda_{\mathrm{bal}} \, E \sum_{k=1}^E f_k \, p_k
   \]
   where $f_k = \frac{1}{|B|} \sum_{i \in B} \mathbf{1}[k_i^* = k]$, $p_k = \frac{1}{|B|} \sum_{i \in B} \operatorname{softmax}(-d_i)_k$, $d_{i,k} = \| z_i - c_k \|_2$, and router balance coefficient $\lambda_{\mathrm{bal}} = 10^{-4}$ ($0.0001$).
3. **Margin Loss ($\mathcal{L}_{\mathrm{margin}}$):** Distance margin loss separating the target prototype from alternatives with margin $m = 0.5$:
   \[
   \mathcal{L}_{\mathrm{margin}} = \frac{1}{|B|} \sum_{i \in B} \sum_{j \ne \pi(y_i)} \left[\, m - \left(d_{i,j} - d_{i,\pi(y_i)}\right) \,\right]_+
   \]
   where $\pi(y_i)$ maps rule label $y_i \in \{0, \dots, K-1\}$ to the corresponding prototype index.
   - **Full Five-Task Benchmark Sweep:** Margin loss is active with weight $\lambda_m = 0.05$ (`train.margin.lambda_margin=0.05`).
   - **Matched Can/Square Ablation:** Margin loss is strictly disabled ($\lambda_m = 0.0$) across all ablation arms.

### Optimization of Hard Dispatch

The hard top-1 assignment $k_t^* = \arg\min_k \|z_t - c_k\|_2$ is non-differentiable. No straight-through gradient estimator is used. As a result, the action loss $\mathcal{L}_{\mathrm{act}}$ updates only the selected expert network and the encoder path for that sample, providing **zero direct gradient** to prototype parameters $\{c_k\}$. Instead, prototype positions adapt via the differentiable soft probabilities $p_k$ in the balance loss $\mathcal{L}_{\mathrm{bal}}$ and, when active, via pairwise Euclidean distance margins in $\mathcal{L}_{\mathrm{margin}}$.

### Hyperparameter Table (Table A12)

The resolved training hyperparameters across all methods (from `resolved_config.yaml` artifacts) are summarized in Table A12:

| Setting | PhaseForge | BC | Softmax Top-1 | Phase-Random | Plain Encoder | Scratch MoE | Static Rule | Factorial Floor | Teacher-Forced | Oracle (Offline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Encoder hidden / latent** | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 |
| **Encoder activation / dropout** | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 | GELU / 0.10 |
| **Experts (top-$k$)** | 6 (top-1) | — | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) |
| **Expert hidden dims** | [256] | — | [256] | [256] | [256] | [256] | [256] | [256] | [256] | [256] |
| **Router init / expert init** | centroid / partial warm | — | centroid / partial warm | random / partial warm | centroid / partial warm | centroid / random | centroid / partial warm | random / partial warm | random / partial warm | random / partial warm |
| **Drop rate** | 0.50 | — | 0.50 | 0.50 | 0.50 | — | 0.50 | 0.50 | 0.50 | 0.50 |
| **Batch size** | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| **Stage 1 / Stage 2 LR** | 3e-4 / 1e-4 | 3e-4 / — | — / 1e-4 | — / 1e-4 | — / 1e-4 | — / 1e-4 | 3e-4 / 1e-4 | — / 1e-4 | — / 1e-4 | — / 1e-4 |
| **Stage 1 / Stage 2 Epochs** | 100 / 200 | 100 / — | — / 200 | — / 200 | — / 200 | — / 200 | 100 / 200 | — / 200 | — / 200 | — / 200 |
| **Early stopping** | False | False | False | False | False | False | False | False | False | False |

---

## B.3 Full Definitions of Static Rule and Teacher-Forced Controls

### Static Rule Comparative Control

The Static Rule baseline evaluates whether hand-specified kinematic rules outperform learned or regime-derived partitions. It deploys an MoE architecture with six experts identical to PhaseForge, but replaces the prototype router with a fixed, deterministic kinematic state machine. The transition thresholds are defined below:

| Phase Index | Nominal Role | Trigger Condition / Threshold Rule |
| :---: | :--- | :--- |
| **0** | Approach | Gripper open ($a_t > 0.035\,\mathrm{m}$); $\|p_t - o_t\|_2 > 0.08\,\mathrm{m}$; end-effector above object ($p_{t,z} - o_{t,z} > 0.05\,\mathrm{m}$) |
| **1** | Pre-grasp / Descend | Gripper open ($a_t > 0.035\,\mathrm{m}$); $\|p_t - o_t\|_2 \le 0.08\,\mathrm{m}$; vertical alignment achieved |
| **2** | Grasp | Finger closure initiated ($a_t \le 0.035\,\mathrm{m}$); end-effector velocity $\|v_t\|_2 < 0.02\,\mathrm{m/s}$ |
| **3** | Lift | Gripper fully closed ($a_t \le 0.025\,\mathrm{m}$); object elevated above surface ($o_{t,z} - o_{0,z} > 0.03\,\mathrm{m}$) |
| **4** | Transport | Gripper closed; horizontal velocity $\|v_{t,xy}\|_2 > 0.02\,\mathrm{m/s}$ toward target bin or fixture |
| **5** | Placement / Insertion | Object within target receptacle bounds; gripper opening initiated ($a_t > 0.025\,\mathrm{m}$) |

### Teacher-Forced Privileged Diagnostic

The Teacher-Forced condition tests contract consistency between training dispatch and closed-loop evaluation:
- **During Stage 2 Training:** Expert dispatch is driven directly by offline trajectory regime labels (`phase_topo`). The experts specialize to these offline labels under teacher forcing.
- **During Rollout Evaluation:** Because offline trajectory-level regime labels cannot be computed online during closed-loop rollout, dispatch is driven by the predicted output of the frozen Stage-1 rule-phase classifier head (which was trained on `phase`).
- **Resulting Mismatch:** This condition introduces a severe contract violation: (1) a distribution shift from offline retrospective labels to online predictions, and (2) a semantic cross-vocabulary mismatch between `phase_topo` training assignments and `phase` inference routing. As reported in §5.2, this produces catastrophic failure ($41\%$ on Lift, $1\%$ on Can, $2\%$ on Square), proving that the condition diagnoses contract violation rather than providing an upper bound.

---

## B.4 Full-Suite Condition Matrix

The comprehensive experimental suite evaluates ten distinct configurations, organized by the structural factor each condition isolates:

| Condition | Encoder Pre-Training | Router Type | Prototype Init | Expert Weight Init | Evaluated Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PhaseForge (Proposed)** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Prototype Voronoi | Trajectory regime centroids | Partial warm (drop 0.50) | Proposed full pipeline |
| **Monolithic BC** | Standard action MSE | None (Single trunk) | None | Random Kaiming | Unpartitioned policy floor |
| **Softmax Top-1** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Learned gating network | None (Parametric gate) | Partial warm (drop 0.50) | Architectural gating diagnostic |
| **Phase-Random** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Prototype Voronoi | Random normal vectors | Partial warm (drop 0.50) | Prototype initialization control |
| **Plain Encoder** | Action-only MSE | Prototype Voronoi | Trajectory regime centroids | Partial warm (drop 0.50) | Representation diagnostic |
| **Scratch MoE** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Prototype Voronoi | Trajectory regime centroids | Random Xavier | Expert initialization diagnostic |
| **Static Rule** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Rule-based state machine | None (Rule-dispatched) | Partial warm (drop 0.50) | Human kinematic baseline |
| **Factorial Floor** | Action-only MSE | Prototype Voronoi | Random normal vectors | Partial warm (drop 0.50) | Two-factor ablation corner |
| **Teacher-Forced** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Label-routed (Train: `phase_topo`, Eval: `phase`) | None | Partial warm (drop 0.50) | Privileged diagnostic |
| **Oracle (Offline)** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Ground-truth simulator state | None | Partial warm (drop 0.50) | Theoretical reference (offline only) |

---

## B.5 Capacity and Computational Accounting (Table T3)

To guarantee fair comparisons, model parameter counts and forward computational complexity were accounted for across all deployable conditions, as tabulated in Table T3:

| Method | Deployed Parameters | Active Params / Sample | Estimated FLOPs / Forward | Total Optimizer Steps |
| :--- | :---: | :---: | :---: | :---: |
| **PhaseForge** | 400,370 | 210,835 | 409,236 | 11,700 |
| **Monolithic BC** | 206,983 | 206,983 | 407,687 | 7,800 |
| **Softmax Top-1** | 401,150 | 211,615 | 409,236 | 7,800 |
| **Phase-Random** | 400,370 | 210,835 | 409,236 | 7,800 |
| **Plain Encoder** | 400,370 | 210,835 | 409,236 | 7,800 |
| **Scratch MoE** | 400,370 | 210,835 | 409,236 | 7,800 |
| **Static Rule** | 400,370 | 210,835 | 409,236 | 11,700 |
| **Factorial Floor** | 400,370 | 210,835 | 409,236 | 7,800 |
| **Teacher-Forced** | 400,376 | 210,841 | 409,236 | 7,800 |
| **Oracle (Offline)** | 400,376 | 210,841 | 409,236 | 7,800 |

*Accounting Notes:*
- **Active Parameters:** Although the modular MoE architecture deploys $\sim 400\mathrm{k}$ total parameters, hard top-1 routing activates only a single expert network per timestep. Thus, the active parameter count during evaluation ($210,835$) closely matches the monolithic BC baseline trunk ($206,983$).
- **Computational Cost:** Active forward inference requires $\sim 409\mathrm{k}$ FLOPs across all modular variants, compared with $\sim 408\mathrm{k}$ FLOPs for the monolithic baseline.

---

# Appendix C — Data, Evaluation, and Statistics

This appendix details the dataset splits, structured observation schemas, reset-bank evaluation protocol, metric aggregation procedures, and statistical hypothesis-testing methods.

---

## C.1 Dataset Splits and Observation Schemas

All evaluations are conducted on five robot manipulation tasks from the Robomimic benchmark using the Proficient-Human (`ph`) demonstration datasets [Mandlekar et al., 2021]. The tasks are executed on a simulated 7-DOF Franka Panda arm under operational-space impedance control at a control frequency of $20\,\mathrm{Hz}$ [Zhu et al., 2020; Todorov et al., 2012].

### Demonstration Splits and Leakage Prevention

Each task provides exactly 200 human demonstrations:
- **Training split:** 180 trajectories ($90\%$).
- **Validation split:** 20 trajectories ($10\%$).

All data splits are partitioned strictly at the **trajectory level**, rather than by shuffling individual timesteps. This ensures that the validation demonstrations contain completely disjoint motion sequences with zero temporal leakage into training batches. Normalization statistics (mean and standard deviation) are computed exclusively from the 180 training trajectories and frozen for all subsequent data preprocessing. The 20 validation demonstrations are used strictly for offline routing evaluation (NMI and switch rates) and are never used for checkpoint selection.

### Observation Schemas (Table A13)

Observations $x_t \in \mathbb{R}^D$ consist of low-dimensional, proprioceptive and object state vectors. Table A13 summarizes the schema, dimensions, and sequence statistics across the benchmark suite:

| Task | State Dim $D$ | Action Dim $A$ | Action Contract | Demonstration Count (Train / Val) | Mean Episode Length | Primary State Components |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| **Lift** | 19 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 124.5 steps | Robot arm joint positions/velocities, end-effector pose, cube position/quaternion |
| **Can** | 23 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 198.2 steps | Robot state, can position/quaternion, bin receptacle bounds |
| **Square** | 23 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 165.4 steps | Robot state, square peg pose, hole fixture coordinate frame |
| **ToolHang** | 53 | 7 | $\Delta$ End-Effector Pose (6D) + Gripper (1D) | 180 / 20 | 387.1 steps | Robot state, tool base/hook geometries, frame attachment points |
| **Transport** | 59 | 14 | Bimanual $\Delta$ EEF Poses ($2 \times 6\mathrm{D}$) + Grippers ($2 \times 1\mathrm{D}$) | 180 / 20 | 442.8 steps | Dual-arm states, payload transport box, target table locations |

---

## C.2 Reset-Bank Provenance and Task-Specific Rollout Horizons

### Deterministic Reset Banks

To eliminate environment reset variance across model comparisons, rollout evaluations are initialized from a pre-generated bank of frozen environment states.
- **Master Reset Seed:** `2026`.
- **Per-Task Reset Hashes:** Verified by SHA-256 provenance manifests (`310d9cfd3fa5e843`, `a7d3953c0afcf560`, `c6683cf0dbb23876`, `db5b4c2a5e6519d0`, `e16288589f5f69c2`).
- **Trial Accounting:** Exactly 50 rollout episodes are executed per independently trained seed. Across the 3 evaluated training seeds (`seeds = [42, 43, 44]`), each cell in the evaluation matrix corresponds to exactly $N = 150$ closed-loop rollout trials.
- **Rollout Horizon:** The maximum episode horizon is fixed at $H = 500$ timesteps ($25.0\,\mathrm{seconds}$ of simulated execution) for all tasks. If the native environment success predicate is satisfied at any timestep $t \le H$, the episode terminates immediately and is scored as a success ($1$); otherwise, upon reaching $t = H$, it is scored as a failure ($0$).

---

## C.3 Success, NMI, and Routing-Switch-Rate Aggregation

Performance and routing metrics are aggregated across experimental seeds under the following protocols:

### Task Success Rate
Closed-loop control quality is measured as the empirical success fraction over the 150 rollout episodes:
\[
\text{Success Rate} = \frac{1}{150} \sum_{s=1}^3 \sum_{e=1}^{50} \mathbf{1}\bigl[\text{Episode } e \text{ of seed } s \text{ succeeded}\bigr].
\]

### Phase-Expert Alignment (NMI)
Phase-expert alignment is evaluated offline on the 20 held-out validation demonstrations using Normalized Mutual Information ($\operatorname{NMI}$) [Vinh et al., 2010]. Let $k_t^* \in \{0, \dots, E-1\}$ denote the top-1 assigned expert at timestep $t$, and let $y_t^{\mathrm{phase}} \in \{0, \dots, K-1\}$ denote the canonical rule-derived phase label.

For each training seed $s \in \{42, 43, 44\}$, all validation timesteps across the 20 validation demonstrations are concatenated into a single sample array, and NMI is computed over these concatenated validation samples:

\[
\operatorname{NMI}_s(k^*, y^{\mathrm{phase}}) = \frac{2 \, I(k^*;\, y^{\mathrm{phase}})}{H(k^*) + H(y^{\mathrm{phase}})}
\]

where $I(k^*; y^{\mathrm{phase}})$ is mutual information and $H(\cdot)$ denotes Shannon entropy under arithmetic average normalization. The reported benchmark NMI is the arithmetic mean across the three independently trained seeds:
\[
\operatorname{NMI} = \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \operatorname{NMI}_s.
\]
$\operatorname{NMI} \in [0, 1]$, where $1$ indicates perfect bijective alignment between experts and behavioral phases, and $0$ indicates statistical independence.

### Step-to-Step Routing-Switch Rate
The routing-switch rate quantifies temporal stability across adjacent timesteps on the validation demonstration distribution. A step pair $(t, t+1)$ is defined as adjacent if and only if both samples belong to the same trajectory and their positions satisfy $\text{pos}_{t+1} = \text{pos}_t + 1$. Step pairs spanning across trajectory boundaries are strictly excluded.

For each training seed $s$, the switch rate is computed over all concatenated adjacent step pairs in the validation split:

\[
\text{Switch Rate}_s = \frac{\sum_{i=1}^{N_{\mathrm{val}}} \sum_{t=1}^{T_i - 1} \mathbf{1}\bigl[ k_{i, t+1}^* \ne k_{i, t}^* \bigr]}{\sum_{i=1}^{N_{\mathrm{val}}} (T_i - 1)}.
\]

The reported benchmark switch rate is the arithmetic mean across the three seeds:
\[
\text{Switch Rate} = \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \text{Switch Rate}_s.
\]

---

## C.4 Paired Comparisons, Wilson Intervals, and Holm Correction

### Wilson Score Confidence Intervals

Uncertainty on rollout success rates is reported using 95% Wilson score intervals [Wilson, 1927], which provide calibrated binomial coverage without Gaussian normality assumptions. For $S$ successes out of $N = 150$ trials with observed success fraction $\hat{p} = S / N$ and critical value $z = 1.96$:

\[
\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}.
\]

Wilson score intervals quantify binomial sampling uncertainty over pooled rollout episodes; they do not represent variance across independently trained seeds.

### Paired Within-Seed Differences

Because all methods are evaluated on identical initial reset states, comparisons between methods $A$ and $B$ are computed as paired within-seed differences:
\[
\Delta_{s} = \text{Success}_A(s) - \text{Success}_B(s), \qquad \bar{\Delta} = \frac{1}{3} \sum_{s \in \{42, 43, 44\}} \Delta_s.
\]

### Exact Sign Tests with Step-Down Holm-Bonferroni Correction

To test the null hypothesis that method $A$ is no better than method $B$, we perform exact two-sided sign tests on paired per-episode outcomes across identical reset states:
- **Test Statistic:** Under the null hypothesis $H_0: P(\text{Outcome}_A > \text{Outcome}_B) = 0.5$.
- **Exact Significance:** Computed using the exact binomial distribution for discordant episode pairs.
- **Multiplicity Correction:** Multiplicity correction across baseline comparisons is performed using the step-down Holm-Bonferroni method [Holm, 1979]. Given sorted raw $p$-values $p_{(1)} \le p_{(2)} \le \dots \le p_{(M)}$:
  \[
  p_{(i)}^{\mathrm{Holm}} = \min\left(1.0,\; \max_{j \le i} \bigl\{ (M - j + 1) \, p_{(j)} \bigr\}\right).
  \]
As tabulated in Appendix D, after Holm adjustment, all paired differences across the three seeds yield $p = 1.000$, confirming that observed differences in success rate represent sample properties of the three training runs rather than resolved population-level effects.

---

# Appendix D — Extended Experimental Results

This appendix reports the complete, unaggregated empirical data tables generated across all experimental sweeps, including the per-seed raw rollouts across all ten methods, the matched Can/Square router-initialization ablation breakdown, and paired statistical significance tests.

---

## D.1 Per-Seed Raw Rollout Success Rates (Table A1)

Table A1 documents the exact per-seed rollout episode counts and success fractions across all three independently trained seeds (`seed 42`, `seed 43`, `seed 44`) for all 10 evaluated methods (50 rollout episodes per seed, $N = 150$ total trials per cell). The benchmark results summarized in main-paper Table 1 are derived directly from these per-seed evaluations:

| Task | Method | Seed 42 | Seed 43 | Seed 44 | Mean Success |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Lift** | PhaseForge | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | BC | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Softmax Top-1 | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Phase-Random | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Plain Encoder | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Scratch MoE | 0.94 (47/50) | 0.96 (48/50) | 0.90 (45/50) | 0.93 |
| Lift | Static Rule | 1.00 (50/50) | 1.00 (50/50) | 1.00 (50/50) | 1.00 |
| Lift | Factorial Floor | 0.92 (46/50) | 0.98 (49/50) | 1.00 (50/50) | 0.97 |
| Lift | Teacher-Forced | 0.08 (4/50) | 0.54 (27/50) | 0.60 (30/50) | 0.41 |
| Lift | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **Can** | PhaseForge | 0.76 (38/50) | 0.80 (40/50) | 0.78 (39/50) | 0.78 |
| Can | BC | 0.78 (39/50) | 0.68 (34/50) | 0.42 (21/50) | 0.63 |
| Can | Softmax Top-1 | 0.58 (29/50) | 0.72 (36/50) | 0.76 (38/50) | 0.69 |
| Can | Phase-Random | 0.58 (29/50) | 0.66 (33/50) | 0.62 (31/50) | 0.62 |
| Can | Plain Encoder | 0.22 (11/50) | 0.56 (28/50) | 0.30 (15/50) | 0.36 |
| Can | Scratch MoE | 0.56 (28/50) | 0.74 (37/50) | 0.70 (35/50) | 0.67 |
| Can | Static Rule | 0.48 (24/50) | 0.58 (29/50) | 0.52 (26/50) | 0.53 |
| Can | Factorial Floor | 0.34 (17/50) | 0.46 (23/50) | 0.34 (17/50) | 0.38 |
| Can | Teacher-Forced | 0.00 (0/50) | 0.00 (0/50) | 0.02 (1/50) | 0.01 |
| Can | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **Square** | PhaseForge | 0.46 (23/50) | 0.30 (15/50) | 0.46 (23/50) | 0.41 |
| Square | BC | 0.38 (19/50) | 0.20 (10/50) | 0.44 (22/50) | 0.34 |
| Square | Softmax Top-1 | 0.42 (21/50) | 0.44 (22/50) | 0.30 (15/50) | 0.39 |
| Square | Phase-Random | 0.40 (20/50) | 0.46 (23/50) | 0.30 (15/50) | 0.39 |
| Square | Plain Encoder | 0.52 (26/50) | 0.46 (23/50) | 0.52 (26/50) | 0.50 |
| Square | Scratch MoE | 0.44 (22/50) | 0.20 (10/50) | 0.30 (15/50) | 0.31 |
| Square | Static Rule | 0.14 (7/50) | 0.10 (5/50) | 0.12 (6/50) | 0.12 |
| Square | Factorial Floor | 0.30 (15/50) | 0.18 (9/50) | 0.46 (23/50) | 0.31 |
| Square | Teacher-Forced | 0.06 (3/50) | 0.00 (0/50) | 0.00 (0/50) | 0.02 |
| Square | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **ToolHang** | PhaseForge | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | BC | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Softmax Top-1 | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Phase-Random | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Plain Encoder | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Scratch MoE | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Static Rule | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Factorial Floor | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Teacher-Forced | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| ToolHang | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |
| **Transport** | PhaseForge | 0.02 (1/50) | 0.00 (0/50) | 0.00 (0/50) | 0.01 |
| Transport | BC | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Softmax Top-1 | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Phase-Random | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Plain Encoder | 0.00 (0/50) | 0.02 (1/50) | 0.00 (0/50) | 0.01 |
| Transport | Scratch MoE | 0.00 (0/50) | 0.00 (0/50) | 0.02 (1/50) | 0.01 |
| Transport | Static Rule | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Factorial Floor | 0.00 (0/50) | 0.02 (1/50) | 0.02 (1/50) | 0.01 |
| Transport | Teacher-Forced | 0.00 (0/50) | 0.00 (0/50) | 0.00 (0/50) | 0.00 |
| Transport | Oracle (Offline) | 0.00 (0/0) | 0.00 (0/0) | 0.00 (0/0) | -- |

---

## D.2 Matched Can/Square Ablation Breakdown (Table A4)

Table A4 provides the unaggregated per-seed success rates, validation NMI, routing-switch rates, and expert collapse rates across the matched Can/Square ablation suite (where margin loss is disabled, $\lambda_m = 0$):

| Task | Condition | Role | Mean SR | Per-Seed Success (42, 43, 44) | Final NMI | Switch Rate | Collapse Rate |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Can** | Random Init | Matched prototype control | 0.733 | 0.68, 0.70, 0.82 | 0.071 | 0.109 | 0.00 |
| Can | Rule-Based Init | Matched prototype control | 0.687 | 0.68, 0.74, 0.64 | 0.083 | 0.107 | 0.00 |
| Can | **Regime Init (PF)** | Proposed initialization | **0.760** | 0.66, 0.82, 0.80 | **0.674** | **0.040** | 0.00 |
| Can | Plain Encoder | Representation diagnostic | 0.673 | 0.68, 0.74, 0.60 | 0.405 | 0.060 | 0.00 |
| Can | Softmax Top-1 | Architectural diagnostic | 0.733 | 0.66, 0.74, 0.80 | 0.721 | 0.037 | 0.00 |
| **Square** | Random Init | Matched prototype control | 0.280 | 0.14, 0.40, 0.30 | 0.086 | 0.096 | 0.00 |
| Square | Rule-Based Init | Matched prototype control | 0.313 | 0.24, 0.36, 0.34 | 0.091 | 0.100 | 0.00 |
| Square | **Regime Init (PF)** | Proposed initialization | **0.267** | 0.22, 0.28, 0.30 | **0.506** | **0.068** | 0.00 |
| Square | Plain Encoder | Representation diagnostic | 0.320 | 0.42, 0.26, 0.28 | 0.466 | 0.064 | 0.00 |
| Square | Softmax Top-1 | Architectural diagnostic | 0.353 | 0.32, 0.38, 0.36 | 0.613 | 0.054 | 0.00 |

### Definition of Validation Collapse Metric

The expert collapse rate reported in Table A4 corresponds strictly to the recorded validation routing metric `val/expert_collapse_rate`. This metric evaluates the fraction of available experts ($E = 6$) that receive fewer than $1 / (2E) = 1/12 \approx 8.3\%$ of total routing assignments across the 20 held-out validation demonstration trajectories.

Across all evaluated ablation conditions and seeds, the recorded validation collapse rate is exactly $0.00$, indicating that all six experts receive substantial routing assignments on the validation distribution. This finding is a empirical property of the validation trajectories under the trained models, and should not be construed as a general guarantee that expert starvation cannot occur on out-of-distribution inputs.

---

## D.3 Paired Differences and Statistical-Test Results (Table A15)

Table A15 reports paired within-seed success differences ($\Delta = \text{Success}_{\mathrm{PhaseForge}} - \text{Success}_{\mathrm{Baseline}}$) evaluated across identical reset states. Multiplicity-adjusted hypothesis testing is conducted using exact two-sided sign tests with step-down Holm-Bonferroni correction:

| Task | Condition A | Condition B | Mean $\Delta$ | $\Delta$ Std (Seeds) | $p$ (Exact Sign) | $p$ (Holm-Adjusted) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Lift** | PhaseForge | BC | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | BC | +0.153 | 0.192 | 1.000 | **1.000** |
| **Square** | PhaseForge | BC | +0.067 | 0.042 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | BC | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | BC | +0.007 | 0.012 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Softmax Top-1 | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Softmax Top-1 | +0.093 | 0.081 | 0.250 | **1.000** |
| **Square** | PhaseForge | Softmax Top-1 | +0.020 | 0.151 | 1.000 | **1.000** |
| **ToolHang** | PhaseForge | Softmax Top-1 | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Softmax Top-1 | +0.007 | 0.012 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Plain Encoder | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Plain Encoder | +0.420 | 0.159 | 0.250 | **1.000** |
| **Square** | PhaseForge | Plain Encoder | -0.093 | 0.058 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | Plain Encoder | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Plain Encoder | +0.000 | 0.020 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Phase-Random | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Phase-Random | +0.160 | 0.020 | 0.250 | **1.000** |
| **Square** | PhaseForge | Phase-Random | +0.020 | 0.164 | 1.000 | **1.000** |
| **ToolHang** | PhaseForge | Phase-Random | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Phase-Random | +0.007 | 0.012 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Scratch MoE | +0.067 | 0.031 | 0.250 | **1.000** |
| **Can** | PhaseForge | Scratch MoE | +0.113 | 0.076 | 0.250 | **1.000** |
| **Square** | PhaseForge | Scratch MoE | +0.093 | 0.070 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | Scratch MoE | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Scratch MoE | +0.000 | 0.020 | 1.000 | **1.000** |
| **Lift** | PhaseForge | Static Rule | +0.000 | 0.000 | 1.000 | **1.000** |
| **Can** | PhaseForge | Static Rule | +0.253 | 0.031 | 0.250 | **1.000** |
| **Square** | PhaseForge | Static Rule | +0.287 | 0.076 | 0.250 | **1.000** |
| **ToolHang** | PhaseForge | Static Rule | +0.000 | 0.000 | 1.000 | **1.000** |
| **Transport** | PhaseForge | Static Rule | +0.007 | 0.012 | 1.000 | **1.000** |

*Statistical Note:* Across three independently trained seeds, all paired comparisons yield Holm-adjusted $p$-values of $1.000$. None of the observed performance differences achieve statistical significance at the population level; all reported differences must be interpreted as descriptive properties of the observed training runs.

---

# Appendix E — Reproducibility

This appendix documents the code provenance, dependency environment, compute requirements, artifact directory layout, and the deterministic reproduction pipeline.

---

## E.1 Final Manifests, Data-Generating Commit, and Artifact Layout

### Code Provenance and Git Commits

All reported experimental sweeps, data ingestion artifacts, model checkpoints, and evaluation logs were generated using the following tracked codebase commits:
- **Primary Data-Generating Commits:** `e948b73`, `f83d096`.
- **Primary Sweep Manifest:** `experiments/final_causal_matrix.json`.
- **Repository Organization:**
  ```text
  PhaseForge/
  ├── phaseforge/                 # Core library
  │   ├── config/                 # Hydra configuration files
  │   ├── data/                   # Ingestion, state machine, and PELT discovery
  │   ├── models/                 # MoE architecture, routers, and baselines
  │   ├── runner/                 # Execution runner and pre-flight validation
  │   └── trains/                 # Stage 1 and Stage 2 training loops
  ├── experiments/                # Sweep matrices and execution plans
  ├── studies/analysis/           # Audit scripts, tables, and figure generators
  └── docs/paper/                 # Complete paper and appendix manuscripts
  ```

### Artifact Directory Layout

Training artifacts and evaluation logs are persisted in the following deterministic directory structure:
```text
outputs/
├── processed/
│   └── cache/                   # SHA-256 verified processed Robomimic trajectories
├── topo/
│   └── {task_name}/             # Discovered PELT change-points and regime artifacts
├── checkpoints/
│   ├── stage1/                  # Pre-trained encoder and auxiliary heads
│   └── stage2/                  # Jointly fine-tuned MoE models
└── evaluations/
    └── {task_name}/             # Rollout traces, success booleans, and NMI evaluations
```

---

## E.2 Dependency Versions and Execution Environment

### Pinned Software Stack and Platform Record (Table A10)

Experiments were executed within an isolated Python virtual environment on AWS Linux instances. The protocol, platform environment, and artifact provenance are recorded in Table A10:

| Item | Value |
| :--- | :--- |
| **Seeds (matrix / ablation)** | 42, 43, 44 / 42, 43, 44 |
| **Reset banks** | `310d9cfd3fa5e843`, `a7d3953c0afcf560`, `c6683cf0dbb23876`, `db5b4c2a5e6519d0`, `e16288589f5f69c2` |
| **Reset seeds** | 2026 |
| **Evaluation router modes** | learned |
| **Training commits** | `e948b73`, `f83d096` |
| **Dropped-neuron hashes** | 39 recorded (e.g., `final_aligned_softmax_top1@seed42`: `9113226cdcef0f09c1d2bf8a9507d1fcf528e2d5d6b13b695a14f00139d60d54`) |
| **Pinned stack** | `numpy==2.4.6`, `torch==2.13.0+cu130`, `torchvision==0.18.0+cu130` |
| **Scientific stack** | `scipy==1.13.1`, `scikit-learn==1.4.2` |
| **Benchmark stack** | `robomimic==0.3.0`, `robosuite==1.4.1`, `mujoco==3.1.5` |
| **Config & logging** | `hydra-core==1.3.2`, `omegaconf==2.3.0` |
| **Platform** | `Linux-6.8.0-1063-aws-x86_64-with-glibc2.39` (`Python 3.10.14`) |
| **Evaluation to Checkpoint SHA links** | 165 / 180 verified |
| **Rollout horizon** | 500 steps ($25.0\,\mathrm{s}$ at $20\,\mathrm{Hz}$) |

Hardware provenance records confirm Linux AWS execution under the pinned stack above; specific GPU microarchitectures are not recorded in the artifact provenance record.

---

## E.3 Reproduction Commands and Compute Costs

### Reproduction Workflow

All data processing, representation pre-training, topology discovery, modular fine-tuning, and closed-loop rollouts are orchestrated through the unified runner entry point using `uv`:

1. **Pre-Flight Gate and Dry-Run Verification:**
   Before launching training runs, verify all manifest gates, provider orderings, and command contracts:
   ```bash
   uv run python -m phaseforge.runner \
     --manifest experiments/final_causal_matrix.json \
     --outputs outputs_final \
     --verify-gates
   ```

2. **Dry-Run Inspection:**
   Inspect the complete multi-stage execution plan without executing commands:
   ```bash
   uv run python -m phaseforge.runner \
     --manifest experiments/final_causal_matrix.json \
     --outputs outputs_final \
     --expect-steps 330 \
     --dry-run
   ```

3. **Full Experimental Sweep Execution:**
   Execute all training and closed-loop evaluation cells in the manifest:
   ```bash
   uv run python -m phaseforge.runner \
     --manifest experiments/final_causal_matrix.json \
     --outputs outputs_final
   ```

### Computational Cost and Memory Footprint (Table A9)

Table A9 reports the average wall-clock training times, training throughput, and peak GPU memory consumption per matrix cell (mean across tasks and seeds, extracted directly from `timings.json` and training curve efficiency fields):

| Method | Stage-1 Wall (s) | Stage-2 Wall (s) | Steps/s | Peak GPU MB |
| :--- | :---: | :---: | :---: | :---: |
| **PhaseForge** | 501.7 | 1153.7 | 33.9 | 24.4 |
| **Monolithic BC** | 395.1 | -- | 50.9 | 21.5 |
| **Softmax Top-1** | -- | 1172.6 | 33.7 | 24.6 |
| **Phase-Random** | -- | 1135.5 | 33.9 | 24.4 |
| **Plain Encoder** | -- | 1184.8 | 33.5 | 25.5 |
| **Scratch MoE** | -- | 1174.1 | 34.1 | 24.4 |
| **Static Rule** | 473.0 | 1127.8 | 34.7 | 24.4 |
| **Factorial Floor** | -- | 1145.1 | 35.0 | 27.1 |
| **Teacher-Forced** | -- | 1024.4 | 39.2 | 24.2 |
| **Oracle (Offline)** | -- | 1006.0 | 38.9 | 24.1 |

*Accounting Summary:*
- Total training time for the proposed PhaseForge pipeline averages approximately $27.6\,\mathrm{minutes}$ per task-seed ($501.7\,\mathrm{s}$ for Stage 1, plus $1153.7\,\mathrm{s}$ for Stage 2).
- Training throughput averages $\sim 34\,\mathrm{steps/second}$ during Stage 2 modular fine-tuning.
- Peak GPU memory utilization remains under $30\,\mathrm{MB}$ for low-dimensional proprioceptive states across all evaluated conditions.
