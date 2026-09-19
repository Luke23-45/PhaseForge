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
