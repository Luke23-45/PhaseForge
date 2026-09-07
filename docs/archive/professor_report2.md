**MEMORANDUM**

**TO:** Lead Engineer / Antigravity AI Engineering Team
**FROM:** Principal Investigator / Research Director
**DATE:** September 6, 2026
**SUBJECT:** Final Code-Aligned Baseline & Ablation Protocol (Post-Audit Correction)

Your code audit is exemplary. You have correctly identified that my previous report relied on theoretical ablation design rather than the actual implementation state of the repository. In top-tier empirical research, the codebase is the ground truth. If the theoretical design contradicts the code, the theory must be revised, not the other way around. 

I have reviewed your findings against the specific file paths and line numbers you provided. Your corrections regarding the plain encoder centroid computation, the existing `TopKRouter`, the hardcoded `beta=0.0`, and the historical checkpoint aliases are factually accurate and scientifically binding.

Below are the direct answers to your four questions, followed by the final, 100% code-aligned experimental matrix. This document supersedes all previous baseline recommendations.

---

### Part 1: Direct Answers to the Engineering Team's Questions

#### 1. Plain-Encoder Ablation: Topology Labels vs. K-Means
**Decision:** Use **topology labels mapped to the plain encoder’s own BC latent space**.
**Rationale:** As you correctly noted in `plain_encoder_phase_bootstrap.py` (lines 204, 237), the code already computes centroids from the BC latent vectors using the topology labels. This perfectly isolates the *representation* factor. It tests the hypothesis: *"Given the exact same topological semantic initialization, does the SupCon representation yield better routing geometry than a standard BC representation?"* 
Using K-Means would confound the ablation by simultaneously removing the label-based semantic initialization. Keep the topology labels; just ensure they are mapped to the BC latent space as the code currently does.

#### 2. The Residual Branch and Beta Narrative
**Decision:** **Keep `beta=0.0`. Do not claim a residual-action improvement for the Can/Lift tasks.**
**Rationale:** Your audit of `precision_residual_phaseforge.yaml` (line 53) and `impedance_expert.py` (line 251) is definitive. Because `beta` is hardcoded to zero and no annealing schedule exists, the residual branch is mathematically inert. It receives no gradients and contributes nothing to the action. 
**Action:** The paper will describe the method as a **Topology-Initialized Prototype-Routed Direct-Action MoE**. The `ResidualImpedanceExpert` class will be described as an architectural extension that supports compliance, but we will explicitly state that for the free-space precision tasks evaluated in this paper, the compliance coefficient $\beta$ is set to 0 to prevent release variance. We will **not** ablate the residual branch on Can. The direct-vs-residual ablation is deferred to future work on contact-rich insertion (e.g., Square) where $\beta > 0$ is actually implemented and active.

#### 3. Softmax-MoE Baseline Architecture
**Decision:** The Softmax-MoE baseline must use the **exact same final-aligned architecture**, changing *only* the router mechanism.
**Rationale:** You are correct that comparing a Softmax router with a Direct MLP against a Prototype router with a Residual Expert confounds two variables. 
**Action:** The `Final-Aligned Softmax MoE` will use the SupCon encoder, the `ResidualImpedanceExpert` (with `beta=0.0`), and the 50% partial warm-start. The *only* difference is that the `PrototypeRouter` is replaced by the existing `TopKRouter` (`router.py` lines 46, 330) with its standard Gaussian noise and auxiliary load-balancing loss. This perfectly isolates the routing mechanism.

#### 4. Static Rule-Based MoE Definition
**Decision:** It must use the **exact same final-aligned architecture**, changing *only* the label source for prototype initialization.
**Rationale:** To prove that dynamic topological discovery (PELT) is superior to human heuristics, we must hold the architecture constant.
**Action:** The `Static Rule-Based MoE` will use the SupCon encoder, the `PrototypeRouter`, and the `ResidualImpedanceExpert` (`beta=0.0`). The only difference is that the initial prototypes ($c_k$) are computed using the 6 human-defined rule labels instead of the PELT-derived topology labels. This isolates the value of data-driven regime discovery.

---

### Part 2: The Final, Code-Aligned Experimental Matrix

This matrix is strictly derived from the current codebase capabilities. It separates Major Baselines (external validity) from Causal Ablations (internal decomposition) and Diagnostics. 

#### Table A: Major Baselines (External Validity)
These methods answer: *"Why use PhaseForge instead of standard methods?"*

| Method Identity | Representation | Router Mechanism | Router Init Source | Expert Path (`beta`) | Scientific Question |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Standard BC`** | Standard BC | None | None | Direct MLP | Dense imitation floor. |
| **`Final-Aligned Softmax MoE`** | SupCon / Phase | **`TopKRouter`** (Softmax + Load Bal.) | Random | `ResidualImpedanceExpert` (`beta=0.0`) | *Is Prototype routing better than standard NLP/CV MoE gating for control?* |
| **`Static Rule-Based MoE`** | SupCon / Phase | `PrototypeRouter` (Hard Top-1) | **Human Rule Labels** | `ResidualImpedanceExpert` (`beta=0.0`) | *Is dynamic topological discovery better than static human heuristics?* |

#### Table B: Core Causal Ablations (Internal Decomposition)
These methods answer: *"Which specific component of PhaseForge drives the performance?"* Every row changes **exactly one** factor from the proposed method.

| Method Identity | Representation | Router Mechanism | Router Init Source | Expert Path (`beta`) | Scientific Question |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PhaseForge (Ours)`** | SupCon / Phase | `PrototypeRouter` (Hard Top-1) | Topology (PELT) | `ResidualImpedanceExpert` (`beta=0.0`) | **Full Proposed Method** |
| **`No SupCon (Plain Encoder)`** | **Standard BC** | `PrototypeRouter` (Hard Top-1) | Topology (mapped to BC latent) | `ResidualImpedanceExpert` (`beta=0.0`) | *Does phase-aware representation matter?* |
| **`Random Init Router`** | SupCon / Phase | `PrototypeRouter` (Hard Top-1) | **Random** | `ResidualImpedanceExpert` (`beta=0.0`) | *Does Topology initialization prevent routing collapse?* |
| **`Factorial Floor (Warmstart)`**| **Standard BC** | `PrototypeRouter` (Hard Top-1) | **Random** | `ResidualImpedanceExpert` (`beta=0.0`) | *The 2x2 floor. No phase, no topology. Does the joint pathway matter?* |
| **`Scratch Init Experts`** | SupCon / Phase | `PrototypeRouter` (Hard Top-1) | Topology (PELT) | **Random Scratch** | *Is the 50% partial warm-start necessary for convergence?* |

#### Table C: Privileged Diagnostics
These are not deployable baselines. They establish theoretical bounds.

| Method Identity | Configuration | Scientific Question |
| :--- | :--- | :--- |
| **`Oracle Router`** | SupCon + **Ground-Truth Phase Routing** (Train & Eval) | What is the theoretical upper bound of expert specialization? |
| **`Historical Baselines`** | Old `phaseforge` direct-action models (from `five_task.json`) | Documenting the engineering evolution of the project. Reported in an appendix, strictly separated from the final matrix. |

---

### Part 3: Strict Implementation & Provenance Directives

To ensure zero mistakes and absolute reproducibility, the following engineering directives are now in effect:

1. **Checkpoint Isolation (`config.py` & `five_task.json`):**
   You correctly identified that historical aliases still resolve to the old `phaseforge` provider. 
   * **Directive:** Create a new manifest strictly for the final matrix (e.g., `final_causal_matrix.json`). In this manifest, explicitly hardcode the Stage 1 checkpoint paths for the final-aligned controls. Do not rely on alias resolution. 
   * **Gate:** The `plain_encoder` and `warmstart_moe` controls in the final matrix **must** load a newly trained, final-aligned normalized BC Stage 1 checkpoint. They must never load the historical `phaseforge` Stage 1 checkpoint.

2. **The `beta=0.0` Contract:**
   * **Directive:** Add a configuration validation hook in the `ResidualImpedanceExpert` initialization. If the task is Can, Lift, or Transport, the code must assert `beta == 0.0` and log a warning if it is not. This prevents accidental activation of the residual branch during hyperparameter sweeps.

3. **Router Load-Balancing for Softmax MoE:**
   * **Directive:** When implementing the `Final-Aligned Softmax MoE`, ensure the auxiliary load-balancing loss weight is tuned so that expert utilization is roughly uniform. If the Softmax MoE suffers from expert collapse despite the load-balancing loss, this is a highly valuable empirical finding that strengthens the paper's argument for Prototype Routing.

4. **Paper Narrative Alignment:**
   * **Directive:** The abstract and introduction will be updated to reflect the `beta=0.0` reality. We will frame the contribution as: *"A memoryless, topology-initialized prototype routing mechanism that solves boundary chattering and load-collapse in continuous control MoEs, paired with a phase-aware contrastive representation."* The residual compliance branch will be mentioned as an architectural capability, but not as the primary contribution of this specific paper.

### Final Sign-Off

Your refusal to accept a theoretically pleasing but empirically flawed protocol is exactly the standard required for this project. By forcing the baseline matrix to align perfectly with the actual codebase (`plain_encoder_phase_bootstrap.py`, `router.py`, `impedance_expert.py`), you have eliminated the risk of Reviewer 2 finding a fatal methodological flaw.

The matrix in Part 2 is now mathematically sound, code-aligned, and scientifically defensible. 

Proceed with the implementation of the `final_causal_matrix.json` manifest. I expect the dry-run provenance audit to confirm that no historical checkpoints are leaking into the final-aligned controls. 

Excellent work. Let's execute.