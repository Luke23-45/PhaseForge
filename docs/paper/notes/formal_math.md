# PhaseForge: Implemented Mathematical Specification

**Status:** implementation reference for the current repository. This note defines the current production method, and separates implemented optional components from the production path. It is not a specification of an unimplemented controller, a simulator dynamics model, or a stability theorem.

The source of truth is the code and resolved configuration, particularly:

- "phaseforge/models/phase_moe.py";
- "phaseforge/models/components/{encoder,action_head,phase_head,prototype_router,moe_layer,impedance_expert}.py";
- "phaseforge/trains/loops/{stage1_loop,stage2_loop}.py";
- "phaseforge/data/{common/normalizer.py,topo/{task_vars,pelt,cluster,observability}.py}";
- "experiments/final_causal_matrix.json".

## 1. Scope, data, and notation

For one task,

\[
  x_t\in\mathbb{R}^{D},\qquad a_t\in[-1,1]^A
\]

are the state presented to the policy and the environment action. The policy is trained and evaluated one timestep at a time. Trajectory provenance is stored as (trajectory_id, trajectory_position), but the production prototype router does not use it as an input.

| Task | State layout | \(D\) | \(A\) | Horizon |
|---|---|---:|---:|---:|
| Lift | robot end-effector (3,4,2) + object (10) | 19 | 7 | 500 |
| Can | robot end-effector (3,4,2) + object (14) | 23 | 7 | 500 |
| Square | robot end-effector (3,4,2) + object (14) | 23 | 7 | 500 |
| ToolHang | robot end-effector (3,4,2) + object (44) | 53 | 7 | 500 |
| Transport | two end-effectors (3,4,2) each + object (41) | 59 | 14 | 700 |

The ordering is defined by data.state_keys. The current data path uses sequence_length=1 and stride=1. The registered table includes all benchmark contracts. The topology task-variable extractor uses the required robot0 and object keys; for the Transport layout, robot1-specific variables are not included in its canonical signal.

### 1.1 Normalization

Statistics are computed on the training split only using Welford's algorithm. With training count \(n\), mean \(\mu\), and accumulator \(M_2\),

\[
  \sigma_d =
  \begin{cases}
    1, & n<2,\\
    \sqrt{M_{2,d}/(n-1)}+10^{-6}, & n\ge2.
  \end{cases}
\]

The model input is

\[
  \bar{x}_t=(x_t-\mu)/\sigma,
\]

and inverse normalization is \(x_t=\bar{x}_t\odot\sigma+\mu\). The same frozen training statistics are used for validation and rollout. In the rest of this note, \(x_t\) means the model input unless raw/physical state is explicit.

## 2. Shared representation and Stage 1

### 2.1 Encoder

The production encoder is an MLP with hidden widths [256,256,256], GELU, dropout 0.1 during training, a learned residual projection, output width 128, and L2-normalized output. With \(h_{0,t}=x_t\), its hidden recurrence is:

\[
\begin{aligned}
  h_{\ell,t} &= \operatorname{Dropout}_{0.1}
        (\operatorname{GELU}(W_\ell h_{\ell-1,t}+b_\ell)),
        \quad \ell\in\{1,2,3\},\\
  h_t &= h_{3,t},\\
  r_t &= W_rx_t+b_r,\\
  \tilde z_t &= W_oh_t+b_o+r_t,\\
  z_t &= \tilde z_t/\max(\|\tilde z_t\|_2,\varepsilon).
\end{aligned}
\]

Dropout is inactive in evaluation mode. Thus \(z_t\in\mathbb{R}^{128}\) is unit-normalized up to numerical precision. Linear layers use Kaiming-uniform initialization and zero biases.

### 2.2 Stage 1 heads

The deterministic action head is

\[
  a_t^{\rm gen}
  =\tanh\!\left(W_a\operatorname{GELU}(W_hz_t+b_h)+b_a\right)
  \in(-1,1)^A.
\]

The phase head is a single linear classifier

\[
  \ell_t^{\rm phase}=W_pz_t+b_p\in\mathbb{R}^{6}.
\]

It returns raw logits. The Stage 1 action and phase heads are frozen after bootstrap for production Stage 2.

## 3. Stage 1 objective

The final production row uses

\[
  \mathcal{L}_1
  =\mathcal{L}_{\rm act}
   +\lambda_{\rm phase}\mathcal{L}_{\rm phase}
   +\lambda_{\rm sc}\mathcal{L}_{\rm SupCon},
\]

with \(\lambda_{\rm phase}=1\), \(\lambda_{\rm sc}=1\), \(\tau=0.07\), and constant phase weight over 100 epochs.

Stage 1 uses AdamW with learning rate \(3\times10^{-4}\), weight decay
\(10^{-4}\), and cosine annealing to the configured minimum learning rate.

With action target \(a_i^*\),

\[
  \mathcal{L}_{\rm act}
  =\frac{1}{|B|A}\sum_{i\in B}\sum_{d=1}^{A}
    (a_{i,d}^{\rm gen}-a_{i,d}^{*})^2.
\]

For phase target \(y_i\in\{0,\ldots,5\}\),

\[
  \mathcal{L}_{\rm phase}
  =-\frac{1}{|B|}\sum_{i\in B}
  \log\frac{\exp(\ell_{i,y_i}^{\rm phase})}
  {\sum_{q=0}^{5}\exp(\ell_{i,q}^{\rm phase})}.
\]

Class weighting, label smoothing, and padding-mask handling are implemented options; they are disabled and unnecessary for the current single-step production row.

For \(P(i)=\{p\ne i:y_p=y_i\}\), \(s_{ip}=z_i^\mathsf{T}z_p/\tau\), and \(I=\{i:|P(i)|>0\}\),

\[
  \mathcal{L}_{\rm SupCon}
  =\frac{1}{|I|}\sum_{i\in I}
  -\frac{1}{|P(i)|}\sum_{p\in P(i)}
  \log\frac{\exp(s_{ip})}{\sum_{a\ne i}\exp(s_{ia})}.
\]

Self-similarities are excluded. Singleton anchors are omitted; an all-singleton batch returns zero.

## 4. Offline topology discovery

Topology discovery is fit on training demonstrations and packaged as phase_topo. Validation labels are assigned using the frozen training result.

### 4.1 Task-variable signal

For each single-arm trajectory, the state is inverse-normalized when statistics are available. The task-variable matrix is

\[
  s_t=[\mathrm{eef\_pos}_t,\mathrm{eef\_quat}_t,\mathrm{gripper}_t,
       \mathrm{object}_t,
       \mathrm{eef\_pos}_t-\mathrm{object}_{t,0:3},
       \mathrm{gripper\_aperture}_t].
\]

The quaternion is normalized and sign-canonicalized to non-negative scalar component. The aperture is

\[
  \mathrm{gripper\_aperture}_t=\max(|q_{t,0}|,|q_{t,1}|).
\]

The object-position proxy is the first three coordinates of the declared object block. This is a feature construction, not a claim that object coordinates have identical semantics across tasks. The extractor requires robot0 end-effector keys and an object block at least three coordinates wide; additional declared keys are not included unless listed in the canonical signal above.

### 4.2 Segmentation

The implemented l2 routine solves the exact optimal-partitioning objective

\[
  \min_{0=\tau_0<\cdots<\tau_M=T}
  \sum_{j=0}^{M-1}C(s_{\tau_j:\tau_{j+1}})
  +\beta(M-1),
  \qquad \tau_{j+1}-\tau_j\ge L_{\min},
\]

where

\[
  C(s_{i:j})=\sum_{t=i}^{j-1}\|s_t-\bar s_{i:j}\|_2^2.
\]

The code calls this function run_pelt, but it uses an unpruned exact dynamic program, not the asymptotic pruning behavior of classical PELT. The production topology configuration uses \(\beta=10\), \(L_{\min}=5\), and cost l2. Complexity is \(O(T^2D_s)\), CPU-only, and deterministic.

### 4.3 Segment clustering and assignment

Each segment is represented by component-wise mean and variance:

\[
  \phi_j=[\operatorname{mean}(s_{\tau_j:\tau_{j+1}}),
           \operatorname{var}(s_{\tau_j:\tau_{j+1}})].
\]

Action statistics are not routing-label features. Production uses six-cluster scikit-learn K-means with n_init=10 and the configured seed. Validation segments are assigned to the nearest training centroid. The resulting timestep labels are phase_topo.

The repository also implements spherical K-means, agglomerative clustering, and the helper

\[
  K^*=\arg\max_K[O(K)+A(K)+S(K)-C(K)],
\]

where the terms are caller-supplied. This helper does not measure the terms or prove that a selected K is optimal.

### 4.4 Observability audit

Before enforced topology labels are accepted, the code tests inferability from instantaneous normalized state alone. It reports occupancy, duration statistics, a trajectory-grouped Logistic Regression probe, confusion matrix, optional action-variance reduction, and merge candidates. With at least two trajectories, cross-validation uses at most three GroupKFold splits; one trajectory uses resubstitution and is not a cross-trajectory test.

The production defaults require macro-F1 \(\ge0.6\) and minimum occupancy \(\ge0.01\). Passing is an empirical observability check, not proof of causal correctness or rollout benefit.

## 5. Bootstrap

Bootstrap scans the training loader, computes Stage 1 latents, and selects the declared source. Production prototypes use phase_topo; Stage 1 phase loss, SupCon, and Stage 2 margin use phase. Topology labels are remapped to contiguous IDs using the ordered vocabulary returned by torch.unique, and that mapping is recorded in metadata.

For six regimes and six experts, let \(r_i\) be the contiguous regime ID:

\[
  \tilde c_k=\frac{1}{N_k}\sum_{i:r_i=k}z_i,
  \qquad
  c_k=\tilde c_k/\max(\|\tilde c_k\|_2,\varepsilon).
\]

The final centroid normalization occurs even with spherical=false; that option controls input normalization before averaging. Every required regime must be present.

Production uses ResidualImpedanceExpert with one 256-unit hidden layer. Its direct base expert matches the Stage 1 action head. partial_warm copies that head into every base expert and reinitializes a shared 50% subset of hidden neurons independently per expert with deterministic Kaiming-uniform draws. Residual heads are initialized separately. After bootstrap, Stage 1 heads are frozen; the production encoder is trainable at learning-rate scale 0.1, and prototypes and experts remain trainable.

## 6. Production Stage 2

### 6.1 Router

For latent \(z_t\) and trainable prototype \(c_k\),

\[
  d_{t,k}=\|z_t-c_k\|_2,\qquad g_{t,k}=-d_{t,k}.
\]

With top_k=1,

\[
  k_t^*=\arg\min_k d_{t,k},\qquad
  w_{t,k}=\mathbf{1}[k=k_t^*].
\]

The reported gate logits are \(g_t\). Unit-normalized latent and initialized prototype vectors make this equivalent to maximum cosine similarity at initialization. Prototypes are trainable after bootstrap and are not EMA-updated in production.

The optional top_k>1 branch uses nearest experts and

\[
  w_{t,k}=\frac{\exp(-d_{t,k})}
  {\sum_{j\in\mathcal{K}_t}\exp(-d_{t,j})},
  \qquad k\in\mathcal{K}_t.
\]

It is not the production configuration.

### 6.2 Balance loss

For a batch of size B,

\[
  f_k=\frac{1}{B}\sum_i\mathbf{1}[k_i^*=k],
  \qquad
  p_k=\frac{1}{B}\sum_i\operatorname{softmax}(-d_i)_k,
\]

and the router emits the already-scaled term

\[
  \mathcal{L}_{\rm bal}
  =\lambda_{\rm bal}E\sum_{k=1}^{E}f_kp_k,
  \qquad \lambda_{\rm bal}=10^{-4}.
\]

The hard assignment used for \(f_k\) and soft distribution used for \(p_k\) differ intentionally.

### 6.3 Direct-action production path

For expert k's direct base action,

\[
  b_k(z_t)=\tanh\!\left(W_{o,k}
  \operatorname{GELU}(W_{h,k}z_t+b_{h,k})+b_{o,k}\right).
\]

The final configuration sets \(\beta=0\). The residual expert therefore returns exactly \(e_k(z_t)=b_k(z_t)\), and

\[
  a_t=\sum_{k=1}^{E}w_{t,k}e_k(z_t)=e_{k_t^*}(z_t).
\]

The residual branch is not evaluated and no task-space feedback error is computed. The precise production description is: a memoryless hard prototype router over direct-action experts, with topology-derived prototype initialization and a phase-aware Stage 1 representation objective.

### 6.4 Stage 2 objective

The final Stage 2 objective is

\[
  \mathcal{L}_2
  =\mathcal{L}_{\rm act}
   +\mathcal{L}_{\rm bal}
   +\lambda_m\mathcal{L}_{\rm margin},
  \qquad \lambda_m=0.05,\quad m=0.5.
\]

With clean logits \(g_{i,k}=-d_{i,k}\) and target \(y_i\),

\[
  \mathcal{L}_{\rm margin}
  =\frac{1}{B}\sum_i\sum_{j\ne y_i}
    [m-(d_{i,j}-d_{i,y_i})]_+.
\]

The production margin target is phase. The term encourages separation under those labels; it does not establish low-density boundaries, temporal stability, or success.

The final row uses 200 Stage 2 epochs and AdamW with base learning rate
\(10^{-4}\), weight decay \(10^{-4}\), gradient clipping at 1.0, and cosine
annealing. The encoder learning rate is \(10^{-5}\) because its scale is 0.1;
the prototypes and experts use the base rate. The row disables Stage 2 phase
loss, SupCon, stickiness, teacher KL, release loss, Lipschitz penalty, gain
regularization, and action dimension/phase weighting.

## 7. Optional impedance controller

The repository contains a distinct ImpedanceExpert; it is not the production residual expert at beta=0. For supported single-arm input, its task state is

\[
  y_t=\psi(x_t)=[p_t,q_t,g_t]\in\mathbb{R}^{8},
\]

after inverse normalization when statistics are supplied, quaternion canonicalization, and aperture extraction. It predicts target \(T_k(z_t)\in\mathbb{R}^8\) and positive gains

\[
  \kappa_k=\operatorname{clip}(\operatorname{softplus}(G_kh_k+b_{G,k}),
  \kappa_{\min},\kappa_{\max}).
\]

Its error and adapted action are

\[
  e_k=[T_{k,p}-p_t,\;
       \operatorname{Log}(R(T_{k,q})R(q_t)^{-1}),\;
       T_{k,g}-g_t]\in\mathbb{R}^{7},
  \qquad
  a_{t,k}=\tanh((\kappa_k\odot e_k)/s).
\]

Quaternion subtraction is not used. This path requires a supported single-arm width and is inactive in the final production method; final runs therefore do not support claims about impedance feedback, force regulation, or contact stability.

## 8. Deployment and diagnostics

The learned production policy is deterministic and memoryless:

\[
  a_t=\pi_\theta(\bar{x}_t),
  \qquad
  \Pr(a_t\mid\bar{x}_{0:t},\mathrm{metadata})
  =\delta_{\pi_\theta(\bar{x}_t)}(a_t).
\]

Trajectory fields are accepted for interface and offline grouping but ignored by PrototypeRouter. Sticky EMA, oracle, uniform, and teacher-routing modes are separate interventions, not the learned production policy.

For N episodes with binary success \(S_n\),

\[
  \mathrm{SR}=N^{-1}\sum_n S_n.
\]

NMI uses phase labels and top-1 choices:

\[
  \mathrm{NMI}(Y,K)=\frac{2I(Y;K)}{H(Y)+H(K)},
\]

with scikit-learn arithmetic normalization. A row-normalized top-1 contingency matrix is

\[
  C_{p,k}=
  \frac{\#\{i:y_i=p,\;k_i^*=k\}}{\#\{i:y_i=p\}},
\]

and absent phases produce zero rows. Utilization diagnostics may count all selected top-k indices and are not identical to NMI or the balance loss.

For valid adjacent within-trajectory pairs \(\mathcal{A}\),

\[
  \mathrm{SwitchRate}
  =\frac{\sum_{(i,j)\in\mathcal{A}}\mathbf{1}[k_i^*\ne k_j^*]}
  {|\mathcal{A}|}.
\]

No valid pair yields zero. Entropy uses the full pre-top-k softmax:

\[
  H_t=-\sum_{k=1}^{E}p_{t,k}\log p_{t,k},
  \qquad p_t=\operatorname{softmax}(g_t).
\]

The normalized metric divides mean entropy by \(\log E\). For the prototype router this is a certainty diagnostic for softmax(-distance), not entropy of a hard assignment. Entropy variance is within-window sample variance (window 100 by default), and time-to-stable requires consecutive windows with variance strictly below threshold. The structured result distinguishes observed stabilization from right-censoring at trajectory end.

For a trace-level action-jump diagnostic,

\[
  J_t=\|a_t-a_{t-1}\|_2
\]

is an observed output difference, not a force impulse, controller derivative, or proof of a contact transition.

## 9. Interpretation and limits

With fixed prototypes, hard routing partitions latent space into

\[
  \mathcal{V}_k=\{z:\|z-c_k\|_2\le\|z-c_j\|_2\;\forall j\}.
\]

The policy is piecewise defined by the selected expert. If expert outputs differ at a cell boundary, a hard switch can produce a nonzero action jump. This establishes an architectural possibility, not that a task crosses a boundary or fails because of it.

The defensible implementation-level description is:

1. PhaseForge combines phase-aware Stage 1 representation learning, offline topology-derived prototype initialization, trainable nearest-prototype hard top-1 routing, and direct-action experts in the final beta-zero path.
2. Topology initialization changes the router's starting geometry and can change routing organization and temporal behavior.
3. Whether that organization improves success is empirical and task-dependent; phase alignment or lower switch rate is not equivalent to higher success.

The current implementation and evidence do not establish global closed-loop stability, a contraction guarantee, force-impulse reduction, friction-cone satisfaction, universal performance improvement, or a causal success benefit from topology initialization. Such claims require additional measurements and/or a different controller and analysis.
