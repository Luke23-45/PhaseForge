# Appendix B — Model, Training, and Control Definitions

This appendix specifies the complete neural network architectures, training objectives, optimization schedules, baseline definitions, and computational capacity accounting for all evaluated methods.

---

## B.1 Architecture and Parameterization

All policy models operate under a deterministic, memoryless contract mapping observation $x_t \in \mathbb{R}^D$ to bounded action $a_t \in [-1, 1]^A$.

### Shared Encoder
- **Trunk Architecture:** A 3-layer feedforward Multi-Layer Perceptron (MLP) with layer dimensions $[D \to 256 \to 256 \to 256]$.
- **Normalization and Activations:** Layer Normalization (`LayerNorm`) followed by Rectified Linear Unit (`ReLU`) nonlinearities after each linear hidden layer.
- **Hypersphere Projection:** The final linear layer maps to a latent space of dimension $d = 128$. Representations are projected onto the unit hypersphere $\mathbb{S}^{d-1} = \mathbb{S}^{127}$:
  \[
  z_t = \frac{f_\phi(x_t)}{\| f_\phi(x_t) \|_2} \in \mathbb{S}^{127}.
  \]
  This projection is shared by both the routing mechanism and all expert networks.

### Prototype Router
- **Parameterization:** The router maintains $E = 6$ trainable prototype vectors $\{c_k\}_{k=1}^E$, where each $c_k \in \mathbb{R}^{128}$.
- **Initialization:** In the regime-derived (proposed) and rule-based conditions, prototypes are initialized as normalized latent centroids of Stage-1 representations. In the random control, prototypes are initialized using standard normal initialization scaled to unit length.
- **Hard Top-1 Dispatch:** The router assigns the input representation $z_t$ to the nearest prototype under Euclidean distance:
  \[
  k_t^* = \arg\min_{k \in \{1, \dots, E\}} \| z_t - c_k \|_2.
  \]
  During Stage 2, prototypes are not constrained to the unit sphere, allowing both radial and angular repositioning.

### Expert Networks
- **Individual Architecture:** Each expert $e_k : \mathbb{S}^{127} \to (-1, 1)^A$ is an independent 3-layer MLP $[128 \to 256 \to 256 \to A]$.
- **Output Layer:** Linear output followed by a hyperbolic tangent ($\tanh$) nonlinearity, enforcing the strict action bounds $[-1, 1]^A$.
- **Residual Branch Setting:** In the evaluated configuration, the residual feedback coefficient is fixed at $\beta = 0$. Experts therefore operate as direct-action networks rather than residual correctors:
  \[
  a_t = e_{k_t^*}(z_t).
  \]
- **Expert Symmetry Breaking:** Each expert is initialized from the Stage-1 action head. To break functional symmetry between experts while preserving baseline action competence, an independent perturbation is applied: exactly $20\%$ ($0.20$) of the weights in the final two linear layers are independently re-initialized using standard Xavier uniform initialization, while the remaining $80\%$ retain their pre-trained values.

---

## B.2 Stage-1 and Stage-2 Objectives and Hyperparameters

### Stage 1: Representation Pre-Training

Stage 1 trains the shared encoder $f_\phi$ alongside an action-prediction head $g_{\mathrm{act}}$ and a linear phase-classification head $g_{\mathrm{phase}}$ under the composite objective:

\[
\mathcal{L}_1 = \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{phase}} \, \mathcal{L}_{\mathrm{phase}} + \lambda_{\mathrm{sc}} \, \mathcal{L}_{\mathrm{SupCon}}
\]

1. **Action Prediction Loss ($\mathcal{L}_{\mathrm{act}}$):** Evaluates mean squared error against demonstration actions $a_i^*$:
   \[
   \mathcal{L}_{\mathrm{act}} = \frac{1}{|B|\, A} \sum_{i \in B} \sum_{d=1}^A \left( a_{i,d}^{\mathrm{gen}} - a_{i,d}^* \right)^2.
   \]
2. **Phase Classification Loss ($\mathcal{L}_{\mathrm{phase}}$):** Cross-entropy loss over canonical rule-derived phase labels $y_i \in \{1, \dots, K\}$:
   \[
   \mathcal{L}_{\mathrm{phase}} = - \frac{1}{|B|} \sum_{i \in B} \log \frac{\exp(\ell_{i, y_i})}{\sum_{q=1}^K \exp(\ell_{i, q})}.
   \]
3. **Supervised Contrastive Loss ($\mathcal{L}_{\mathrm{SupCon}}$):** Encourages latent clustering of timesteps sharing identical rule phase labels:
   \[
   \mathcal{L}_{\mathrm{SupCon}} = \frac{1}{|I|} \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i^\top z_p \,/\, \tau)}{\sum_{a \ne i} \exp(z_i^\top z_a \,/\, \tau)}
   \]
   where $P(i) = \{p \in B \setminus \{i\} : y_p = y_i\}$, $I = \{i : |P(i)| > 0\}$, and temperature $\tau = 0.07$.
- **Loss weights:** $\lambda_{\mathrm{phase}} = 1.0$, $\lambda_{\mathrm{sc}} = 0.1$.

### Stage 2: Joint Modular Fine-Tuning

In Stage 2, the auxiliary heads are discarded, and the encoder, prototypes, and experts are optimized jointly under:

\[
\mathcal{L}_2 = \mathcal{L}_{\mathrm{act}} + \mathcal{L}_{\mathrm{bal}} + \lambda_m \, \mathcal{L}_{\mathrm{margin}}
\]

1. **Modular Action Loss ($\mathcal{L}_{\mathrm{act}}$):** Evaluated exclusively on the action of the dispatched expert $e_{k_i^*}(z_i)$:
   \[
   \mathcal{L}_{\mathrm{act}} = \frac{1}{|B|\, A} \sum_{i \in B} \sum_{d=1}^A \left( e_{k_i^*}(z_i)_d - a_{i,d}^* \right)^2.
   \]
2. **Differentiable Balance Loss ($\mathcal{L}_{\mathrm{bal}}$):** Penalizes expert collapse using soft affinities:
   \[
   \mathcal{L}_{\mathrm{bal}} = \lambda_{\mathrm{bal}} \, E \sum_{k=1}^E f_k \, p_k
   \]
   where $f_k = \frac{1}{|B|} \sum_{i \in B} \mathbf{1}[k_i^* = k]$, $p_k = \frac{1}{|B|} \sum_{i \in B} \operatorname{softmax}(-d_i)_k$, and $d_{i,k} = \| z_i - c_k \|_2$. Coefficient $\lambda_{\mathrm{bal}} = 0.01$.
3. **Margin Loss ($\mathcal{L}_{\mathrm{margin}}$):** An explicit distance margin $m = 0.1$ separating the phase-indexed prototype from alternatives:
   \[
   \mathcal{L}_{\mathrm{margin}} = \frac{1}{|B|} \sum_{i \in B} \sum_{j \ne \pi(y_i)} \left[\, m - \left(d_{i,j} - d_{i,\pi(y_i)}\right) \,\right]_+
   \]
   where $\pi(y_i) = y_i$ in 1-based indexing.
   - **Full Five-Task Benchmark Sweep:** Margin loss is active ($\lambda_m = 0.05$).
   - **Matched Can/Square Ablation:** Margin loss is strictly disabled ($\lambda_m = 0$) across all arms.

### Optimization of Hard Dispatch

The discrete routing decision $k_t^* = \arg\min_k \|z_t - c_k\|_2$ is non-differentiable. No straight-through estimator or surrogate gradient is applied to the discrete assignment. Consequently, gradients from $\mathcal{L}_{\mathrm{act}}$ update the selected expert network and the encoder within the selected cell, but provide **zero direct gradient** to the prototype vectors $\{c_k\}$. Instead, prototypes adapt exclusively via the differentiable soft probabilities $p_k$ in the balance loss $\mathcal{L}_{\mathrm{bal}}$ and, when active, via the pairwise distance differentials in $\mathcal{L}_{\mathrm{margin}}$.

### Hyperparameter Table (Table A12)

The resolved training hyperparameters across all methods are summarized in Table A12:

| Setting | PhaseForge | BC | Softmax Top-1 | Phase-Random | Plain Encoder | Scratch MoE | Static Rule | Factorial Floor | Teacher-Forced |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Encoder hidden / latent** | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 | [256, 256, 256] / 128 |
| **Experts (top-$k$)** | 6 (top-1) | — | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) | 6 (top-1) |
| **Router init** | regime centroids | — | learned gate | random | regime centroids | regime centroids | rule centroids | random | offline labels |
| **Expert init** | partial warm (0.20) | — | partial warm (0.20) | partial warm (0.20) | partial warm (0.20) | random | partial warm (0.20) | partial warm (0.20) | partial warm (0.20) |
| **Dropout rate** | 0.5 | — | 0.5 | 0.5 | 0.5 | — | 0.5 | 0.5 | 0.5 |
| **Batch size** | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 256 |
| **Learning rate** | 0.0003 | 0.0003 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0003 | 0.0001 | 0.0001 |
| **Epochs** | 100 | 100 | 200 | 200 | 200 | 200 | 100 | 200 | 200 |
| **Early stopping** | False | False | False | False | False | False | False | False | False |

---

## B.3 Full Definitions of Static Rule and Teacher-Forced Controls

### Static Rule Comparative Control

The Static Rule baseline tests whether human-specified kinematic rules outperform learned or regime-derived partitions. It uses an MoE architecture with six experts matching PhaseForge, but replaces the prototype router with a fixed, deterministic kinematic state machine. The thresholds are defined below:

| Phase Index | Nominal Role | Trigger Condition / Threshold Rule |
| :---: | :--- | :--- |
| **0** | Approach | Gripper open ($\alpha_t > 0.035\,\mathrm{m}$); $\|p_t - o_t\|_2 > 0.08\,\mathrm{m}$; end-effector above object ($p_{t,z} - o_{t,z} > 0.05\,\mathrm{m}$) |
| **1** | Pre-grasp / Descend | Gripper open ($\alpha_t > 0.035\,\mathrm{m}$); $\|p_t - o_t\|_2 \le 0.08\,\mathrm{m}$; vertical alignment achieved |
| **2** | Grasp | Finger closure initiated ($\alpha_t \le 0.035\,\mathrm{m}$); end-effector velocity $\|v_t\|_2 < 0.02\,\mathrm{m/s}$ |
| **3** | Lift | Gripper fully closed ($\alpha_t \le 0.025\,\mathrm{m}$); object elevated above surface ($o_{t,z} - o_{0,z} > 0.03\,\mathrm{m}$) |
| **4** | Transport | Gripper closed; horizontal velocity $\|v_{t,xy}\|_2 > 0.02\,\mathrm{m/s}$ toward target bin or fixture |
| **5** | Placement / Insertion | Object within target receptacle bounds; gripper aperture opening ($\alpha_t > 0.025\,\mathrm{m}$) |

### Teacher-Forced Privileged Diagnostic

The Teacher-Forced condition demonstrates the critical necessity of contract consistency between training and inference:
- **During Stage 2 Training:** Expert dispatch is hard-routed according to the offline trajectory regime labels (`phase_topo`). The experts specialize to these offline labels under teacher forcing.
- **During Rollout Evaluation:** Because offline regime labels are unavailable in a closed-loop simulator rollout, dispatch is driven by the predicted output of the frozen Stage-1 rule-phase classifier head (which was trained on `phase`).
- **Resulting Mismatch:** This condition introduces a severe double-mismatch: (1) an architectural distribution shift from offline labels to online predictions, and (2) a semantic cross-vocabulary mismatch between `phase_topo` training routes and `phase` deployment routes. As reported in §5.2, this produces catastrophic failure ($41\%$ on Lift, $1\%$ on Can, $2\%$ on Square), confirming that the condition diagnoses contract violation rather than measuring an oracle upper bound.

---

## B.4 Full-Suite Condition Matrix

The comprehensive experimental suite evaluates ten distinct configurations, organized by the specific structural factor each condition isolates:

| Condition | Encoder Pre-Training | Router Type | Prototype Init | Expert Weight Init | Evaluated Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PhaseForge (Proposed)** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Prototype Voronoi | Trajectory regime centroids | Warm + 20% reinit | Proposed full pipeline |
| **Monolithic BC** | Standard action MSE | None (Single trunk) | None | Random | Unpartitioned policy floor |
| **Softmax Top-1** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Learned gating network | None (Parametric gate) | Warm + 20% reinit | Architectural gating diagnostic |
| **Phase-Random** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Prototype Voronoi | Random normal vectors | Warm + 20% reinit | Prototype initialization control |
| **Plain Encoder** | Action-only MSE | Prototype Voronoi | Trajectory regime centroids | Warm + 20% reinit | Representation diagnostic |
| **Scratch MoE** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Prototype Voronoi | Trajectory regime centroids | Random Xavier | Expert initialization diagnostic |
| **Static Rule** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Rule-based state machine | None (Rule-dispatched) | Warm + 20% reinit | Human kinematic baseline |
| **Factorial Floor** | Action-only MSE | Prototype Voronoi | Random normal vectors | Warm + 20% reinit | Two-factor ablation corner |
| **Teacher-Forced** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Label-routed (Train: `phase_topo`, Eval: `phase`) | None | Warm + 20% reinit | Privileged diagnostic |
| **Oracle (Offline)** | Phase-aware ($\mathcal{L}_{\mathrm{phase}} + \mathcal{L}_{\mathrm{sc}}$) | Ground-truth simulator state | None | Warm + 20% reinit | Theoretical upper bound reference |

---

## B.5 Capacity and Computational Accounting (Table T3)

To ensure fair comparisons, model parameter counts and forward computational complexity were rigorously controlled and accounted for across all deployable conditions, as tabulated in Table T3:

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
