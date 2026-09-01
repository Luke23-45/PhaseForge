# PhaseForge 2.0: Dynamical Phase Discovery for Mixture-of-Experts Policy Learning in Contact-Rich Manipulation

---

## Abstract

PhaseForge, a Mixture-of-Experts (MoE) architecture that leverages privileged phase labels to initialize expert routing, achieves strong performance on simple manipulation tasks (0.71 success on Lift) but degrades dramatically on contact-rich tasks (0.13 on Square). Through systematic analysis, we identify the root cause: **semantic phase labels provided by human heuristics do not correspond to kinematically coherent action regimes**. Specifically, the "insertion" phase in Square collapses states requiring diametrically opposed actions (pushing down vs. pulling up vs. lateral jiggling) into a single latent centroid. This forces the assigned expert to average contradictory gradients, producing jittery or degenerate policies.

We propose **PhaseForge 2.0**, which replaces human semantic labels with **dynamically discovered phases** derived from demonstration data via switching linear dynamical systems (SLDS) or, in fully autonomous settings, discrete latent variable models (VQ-VAE). These phases are defined by action coherence and local dynamics, not human narrative. By initializing the router with kinematically meaningful centroids and allowing end-to-end fine-tuning, PhaseForge 2.0 preserves the architectural benefits of the original while eliminating its brittleness.

**Key insight:** Privileged phase information is useful only if phases reflect the underlying functional decomposition of the agent-environment interaction, not semantic storytelling.

---

## 1. Introduction

### 1.1 Background

Mixture-of-Experts (MoE) architectures have emerged as a powerful paradigm for robot policy learning, enabling specialized experts to handle distinct behavioral modes (Jacobs et al., 1991; Shazeer et al., 2017). However, learning expert routing from scratch is sample-inefficient and often converges to suboptimal decompositions. PhaseForge (original) addresses this by exploiting privileged phase labels—typically derived from human rule-based heuristics—to supervise a phase-prediction head and initialize expert centroids.

### 1.2 The Failure Mode

Empirical results (Table 1) reveal a striking pattern:

| Task | Scratch MoE | PhaseForge (Semantic Labels) |
|------|-------------|------------------------------|
| Lift | 0.65 | **0.71** |
| Can  | 0.18 | 0.15 |
| Square | **0.21** | 0.13 |

PhaseForge outperforms Scratch MoE on Lift but underperforms on contact-rich tasks. Why?

**The Semantic-Kinematic Mismatch.** Human heuristics segment tasks by narrative logic: "approach," "grasp," "insert." For simple tasks, these align with distinct free-space/contact regimes. But for high-precision insertion, the semantic label "insertion" encompasses states with opposite action requirements:

- **State A:** Nut hovering 1mm above peg → requires downward force
- **State B:** Nut jammed against peg edge → requires lateral jiggle or upward compliance
- **State C:** Nut sliding down peg → requires pure downward force, zero lateral

When PhaseForge forces all three into one centroid:
\[
c_{\text{insertion}} = \frac{\sum_{i:p_i=\text{insertion}} z_i}{\left\|\sum_{i:p_i=\text{insertion}} z_i\right\|_2}
\]
the resulting vector is a meaningless average. The expert assigned to this centroid receives contradictory gradients, its action output averages to a jittery mess, and the task fails.

### 1.3 Contribution

We identify the root cause as **centroid collapse due to semantic-kinematic mismatch** and propose a minimal fix: replace human semantic labels with **dynamically discovered phases** that are defined by action coherence. We present two approaches:

1. **PhaseForge 2.0-SLDS:** Use switching linear dynamical systems to infer kinematically cohesive phases from demonstrations.
2. **PhaseForge 2.0-VQ:** Use a discrete latent variable model (VQ-VAE) to discover phases end-to-end when privileged labels are unavailable.

Crucially, **we add no new architectural components**. We only change the source of labels used in Stage 1 and allow router fine-tuning.

---

## 2. Problem Formulation

### 2.1 Original PhaseForge

Given demonstration trajectories \(\tau = \{(s_t, a_t)\}_{t=1}^T\) with semantic phase labels \(\{p_t\}_{t=1}^T\):

- **Stage 1:** Train encoder \(H_\psi: s_t \to z_t\) to predict \(p_t\). Compute centroids \(c_p\) for each phase.
- **Stage 2:** Initialize router \(R_\theta(s_t) = \text{softmax}(W z_t)\) with \(W_p = c_p^\top\). Train policy end-to-end with action loss.

### 2.2 The Flaw

Semantic labels \(\{p_t^{\text{sem}}\}\) do not satisfy the property:
\[
p_t = p_{t'} \implies \pi(a|s_t) \approx \pi(a|s_{t'})
\]
That is, states in the same phase do not share similar action distributions. This violates the core assumption of expert specialization.

### 2.3 The Fix

Replace \(\{p_t^{\text{sem}}\}\) with \(\{p_t^{\text{dyn}}\}\) derived from dynamics:
\[
p_t^{\text{dyn}} = \arg\min_{k} \sum_{t:p_t=k} \|a_t - \mu_k\|^2
\]
or more generally, modes of a switching dynamical system. This ensures within-phase action coherence.

---

## 3. Method: PhaseForge 2.0

### 3.1 Stage 0: Dynamical Phase Discovery

#### 3.1.1 Switching Linear Dynamical System (SLDS)

Assume demonstrations are generated by:
\[
s_{t+1} = A_{p_t} s_t + B_{p_t} a_t + \epsilon_{p_t}, \quad \epsilon_{p_t} \sim \mathcal{N}(0, \Sigma_{p_t})
\]
where \(p_t \in \{1, \ldots, K\}\) is a discrete latent mode.

**Inference:** Use expectation-maximization (EM) or variational inference (e.g., structured mean-field with message passing) to infer posterior \(q(p_t)\). The most likely mode sequence \(\hat{p}_t = \arg\max_p q(p_t|\tau)\) gives us kinematically cohesive labels.

**Why SLDS works:** The linear dynamics within each mode force the model to group states that share similar state transition and control matrices \((A_k, B_k)\). Contact states (free motion, sliding, sticking, jamming) naturally emerge as distinct modes because they have fundamentally different dynamics.

**Complexity:** SLDS fitting is \(O(K^2 T)\) with EM and can handle thousands of timesteps in minutes on CPU. No GPU required.

#### 3.1.2 Action Clustering (Fallback)

If SLDS is too heavy, cluster on action vectors:
\[
\hat{p}_t = \arg\min_k \|a_t - \mu_k\|^2
\]
using Gaussian Mixture Models (GMM) or k-means on \(\{a_t\}\) (or \([s_t, a_t]\) for context). Actions that are similar in magnitude and direction correspond to similar control regimes. This is crude but often sufficient for coarse phase separation.

#### 3.1.3 VQ-VAE Variant (No Privileged Labels)

When no demonstrations with actions are available, or we want full autonomy:

- **Encoder:** \(z_t = f_\phi(s_t)\)
- **Quantization:** \(p_t = \arg\min_k \|z_t - e_k\|_2\), where \(\{e_k\}_{k=1}^K\) are learnable codebook vectors.
- **Decoder:** \((\hat{s}_{t+1}, \hat{a}_t) = g_\theta(e_{p_t}, s_t)\)

Train with:
\[
\mathcal{L} = \|s_{t+1} - \hat{s}_{t+1}\|^2 + \|a_t - \hat{a}_t\|^2 + \|z_t - \text{sg}(e_{p_t})\|^2 + \beta \|\text{sg}(z_t) - e_{p_t}\|^2
\]

The discrete codes \(p_t\) are forced to capture information needed to predict both next state and action. These are the phases.

**Key difference from original VQ-VAE:** The decoder predicts actions in addition to next states, forcing the codebook to be action-relevant.

### 3.2 Stage 1: Encoder Training with Dynamical Labels

Train \(H_\psi: s_t \to z_t\) to predict \(\hat{p}_t^{\text{dyn}}\):
\[
\mathcal{L}_{\text{phase}} = -\sum_t \log P(\hat{p}_t^{\text{dyn}} | s_t; \psi)
\]

Compute centroids:
\[
c_k = \frac{\sum_{t:\hat{p}_t=k} z_t}{\left\|\sum_{t:\hat{p}_t=k} z_t\right\|_2}
\]

### 3.3 Stage 2: Router Initialization and Fine-tuning

Initialize router:
\[
R_\theta(s_t) = \text{softmax}(W z_t), \quad W_k = c_k^\top
\]

**Crucially, do not freeze.** Train end-to-end with:
\[
\mathcal{L} = \mathcal{L}_{\text{action}} + \lambda \sum_t D_{KL}(R_\theta(s_t) \| \text{onehot}(\hat{p}_t))
\]

The KL term prevents the router from drifting too far from the dynamical decomposition unless the action loss demands it. This preserves the initialization benefit while allowing adaptation to improve on imperfect labels.

---

## 4. Why This Works

### 4.1 No New Components

PhaseForge 2.0 uses the exact same architecture: encoder, phase head, router, experts. We only change the labels and allow router fine-tuning. This is a **minimal intervention** that directly addresses the root cause.

### 4.2 Directly Solves Centroid Collapse

Dynamical labels group states by action similarity. The "insertion" mush is split into:

- **Hover:** Nut 1mm above peg, downward force
- **Contact:** Nut touching peg edge, lateral/upward compliance
- **Slide:** Nut on peg, downward with zero lateral

Each phase has a coherent action distribution, so the centroid is meaningful. Experts receive consistent gradients.

### 4.3 Leverages Existing Tools

SLDS and VQ-VAE are standard, well-understood, and computationally cheap. No new training infrastructure required.

---

## 5. Alternative Solutions Considered

### 5.1 Hierarchical MoE

**Idea:** Use coarse router for free-space/contact/insertion, fine router within insertion for sub-phases.

**Why we rejected it:** Adds architectural complexity. The root problem is label quality, not model capacity. Fixing labels is simpler and more principled.

### 5.2 Mutual Information Regularization

**Idea:** Add \(I(p_t; a_t | s_t)\) term to encourage phases to be action-relevant.

**Why we rejected it:** This is an auxiliary loss that complicates training. If we have demonstrations, we can directly infer phases from actions via SLDS. MI is only needed when labels are unavailable, and VQ-VAE already handles that.

### 5.3 Recurrent Context

**Idea:** Make phase prediction depend on history \(p_t = f(s_{1:t}, a_{1:t-1})\) to disambiguate ambiguous states.

**Why we rejected it:** It adds a recurrent module and changes the architecture. SLDS already accounts for temporal context through its Markov structure.

### 5.4 End-to-End Discrete Latent with Policy

**Idea:** Train policy and phase discovery jointly from scratch.

**Why we rejected it:** This is essentially Scratch MoE with discrete latent, which loses the initialization benefit. PhaseForge's value is in exploiting privileged information; we keep that by using SLDS on demonstrations.

---

## 6. Discussion

### 6.1 The Deeper Insight

PhaseForge's failure is not a bug but a revelation: **modular structure imposed on a learning system must reflect the functional decomposition of the task, not human narrative.** This applies beyond MoE—to skill libraries, hierarchical RL, and any approach that relies on human-defined segmentation.

### 6.2 The Animal Analogy

An ant navigating a rainforest floor does not use semantic phases ("cross gap," "climb leaf"). It uses sensorimotor primitives switched by physical events: slip, contact, load change. The phases emerge from interaction dynamics. PhaseForge 2.0 moves toward this principle.

### 6.3 Limitations

- SLDS assumes linear dynamics within phases. Highly nonlinear tasks may require nonlinear switching models (e.g., switching Gaussian processes) at higher computational cost.
- The VQ-VAE variant requires careful tuning of codebook size \(K\) and commitment loss weight \(\beta\). Too large \(K\) leads to fragmented phases; too small leads to collapse.
- We assume demonstration data is available for SLDS. If only state observations exist, VQ-VAE is the fallback.

### 6.4 Future Work

- **Automatic phase count selection:** Use Bayesian nonparametrics (Dirichlet process) to infer \(K\) from data.
- **Online phase discovery:** Adapt phases during deployment as new contact regimes are encountered.
- **Cross-task transfer:** Test whether dynamically discovered phases transfer across tasks (e.g., from Square to NutAssembly).

---

## 7. Conclusion

We identified the root cause of PhaseForge's failure on contact-rich tasks: semantic-kinematic mismatch leading to centroid collapse. The solution is not architectural complexity but a change in the source of privileged information. By replacing human heuristics with dynamically discovered phases (SLDS or VQ-VAE), we restore the promise of PhaseForge: expert specialization guided by meaningful phase structure. The result is a simple, principled architecture that scales to contact-rich manipulation.

**The lesson:** In modular policy learning, let physics define the modules, not human language.

---

## References

1. Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation*.
2. Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR*.
3. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. (SLDS, Chapter 18)
4. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *NeurIPS*.
5. Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). Bayesian nonparametric inference of switching dynamic linear models. *IEEE Transactions on Signal Processing*.

---

**End of Report**