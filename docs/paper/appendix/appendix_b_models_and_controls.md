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
- **Expert Symmetry Breaking via Partial Warm-Start:** Each expert is initialized from the Stage-1 ActionHead using Drop-Upcycling partial reinitialization (`partial_reinit_experts_from_action_head`). The ActionHead weights are copied into each expert, and then an independent fraction ($\text{drop\_rate} = 0.50$) of each expert's intermediate hidden neurons are reinitialized using Kaiming uniform initialization with fixed seed. The remaining 50% of the neurons retain their pre-trained ActionHead weights bit-exactly. This breaks functional symmetry across experts while guaranteeing functional initialization from the pre-trained generalist.

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
3. **Supervised Contrastive Loss ($\mathcal{L}_{\mathrm{SupCon}}$):** Encourages clustering of timesteps sharing identical phase labels:
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
