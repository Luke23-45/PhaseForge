**MEMORANDUM**

**TO:** Lead Engineer / Antigravity AI Engineering Team
**FROM:** Principal Investigator / Research Director
**DATE:** September 6, 2026
**SUBJECT:** Final Execution-Ready Protocol: Resolution of All Remaining Ambiguities

Your audit is exceptional. You have correctly identified that Report 2, while directionally improved, still contained implementation ambiguities, post-hoc tuning risks, and conflated diagnostics. In a top-tier empirical paper, ambiguity is indistinguishable from error. 

I have reviewed your critique line-by-line. Below are the definitive, rational, and final answers to your five questions, followed by the corrected, execution-ready protocol. This document supersedes all previous guidance.

---

### Part 1: Direct Answers to the Five Unresolved Questions

#### 1. Label Source: SupCon vs. Router Initialization
**Decision:** **Both Stage 1 SupCon and Router Initialization must use `phase_topo`.**
**Rationale:** Your audit correctly identified a critical mismatch. If SupCon trains on human-defined `phase` labels but the router initializes prototypes using data-driven `phase_topo` labels, the latent space is optimized for a different partition than the router is trying to exploit. This confounds the representation ablation.
*   **Action:** The final method (`precision_residual_phaseforge`) will use `phase_topo` for both SupCon contrastive learning and prototype initialization. 
*   **Action:** The `Static Rule-Based MoE` baseline will use human-defined `phase` labels for both SupCon and prototype initialization. 
*   **Why:** This creates a clean, scientifically valid comparison: *Topology-driven representation & routing* vs. *Rule-driven representation & routing*.

#### 2. Exact `TopKRouter` Settings for the Softmax-MoE Baseline
**Decision:** The Softmax-MoE baseline must be pre-registered with the following exact settings. **No post-hoc tuning is permitted.**
*   **`top_k`:** **1**. 
    *   *Rationale:* The proposed method uses Hard Top-1 Prototype Routing. To isolate the routing mechanism, the Softmax baseline must also use Top-1. Comparing Top-1 Prototype vs. Top-2 Softmax would confound routing mechanism with routing sparsity.
*   **`noise_std`:** **0.1** (during training), **0.0** (during evaluation).
    *   *Rationale:* Standard sparse MoE practice (Shazeer et al.) uses noise for exploration. A small standard deviation prevents early expert collapse without destabilizing continuous control.
*   **`normalize_input`:** **True**.
    *   *Rationale:* The proposed method uses normalized latents. The Softmax gate must receive the same normalized input to ensure fair comparison.
*   **`balance_coeff`:** **0.01**.
    *   *Rationale:* This is a standard, conservative load-balancing coefficient used in Switch Transformers and related literature. 
    *   *Critical Directive:* **Do not tune this coefficient to force uniform utilization.** If the Softmax MoE suffers from expert collapse at `0.01`, that is a valid empirical result demonstrating the instability of standard softmax routing in continuous control compared to prototype routing.

#### 3. Oracle vs. Teacher-Forced Diagnostics
**Decision:** **Keep both. They are distinct diagnostics with different scientific purposes.**
*   **`teacher_forced` (Privileged Training Diagnostic):**
    *   *Configuration:* Ground-truth phase routing during Stage 2 training; predicted-phase routing during evaluation.
    *   *Purpose:* Measures the upper bound of *training signal quality*. If the experts had perfect routing during training, how well would they learn? This isolates the quality of the expert specialization from the router's inference accuracy.
*   **`oracle_moe` (Non-Deployable Reference):**
    *   *Configuration:* Ground-truth phase routing during both training and evaluation.
    *   *Purpose:* Measures the theoretical upper bound of the *entire architecture*. If the router never made a mistake, what is the maximum possible success rate? 
*   **Action:** Report both in a dedicated "Diagnostics" table. Clearly label them as non-deployable and privileged. Do not merge them.

#### 4. Beta Contract Scope
**Decision:** **`beta=0.0` is locked for ALL tasks in the final matrix, including Square and ToolHang.**
**Rationale:** The paper's core claim is about topology-initialized prototype routing for direct-action MoEs. Introducing active residual compliance on Square would require a separate set of ablations, hyperparameter schedules, and physical arguments. 
*   **Action:** The final manifest will explicitly set `beta=0.0` for Can, Lift, Square, ToolHang, and Transport.
*   **Action:** The validation hook must be moved to the **Manifest/Experiment Resolver layer**, not the model layer. The resolver must assert `expert.beta == 0.0` for every task in the final matrix. If a future researcher wants to activate beta, they must create a new manifest.

#### 5. Checkpoint Resolution Strategy
**Decision:** **Use explicit provider identities with seed-exact resolution. Do not hardcode paths.**
**Rationale:** Hardcoding paths breaks multi-seed reproducibility. Seed 42 must load the Stage 1 checkpoint trained with Seed 42.
*   **Action:** The manifest will define a `provider_identity` (e.g., `precision_residual_phaseforge_stage1`). The resolution logic will dynamically construct the path based on the requested seed.
*   **Action:** The run metadata must record the **resolved absolute path** and the **SHA-256 hash** of the loaded checkpoint. If the seed-matched checkpoint does not exist, the run must fail immediately.

---

### Part 2: The Final, Execution-Ready Protocol

This protocol is now 100% aligned with your code audit. It resolves the plain-encoder label mismatch, locks the softmax configuration, and enforces seed-safe checkpoint resolution.

#### A. Final Configuration Directives

1. **Plain Encoder Control (`precision_residual_plain_encoder`):**
   * **Code Change Required:** Modify `plain_encoder_phase_bootstrap.py` or the manifest injection to explicitly pass `phase_topo` as the label source for centroid computation.
   * **Logic:** Compute centroids from the plain BC latent vectors using `phase_topo` labels. Do not use `phase` labels. Do not reuse SupCon prototypes.
2. **SupCon Label Source:**
   * **Final Method:** `supcon_label_source = "phase_topo"`.
   * **Static Rule Baseline:** `supcon_label_source = "phase"`.
3. **Softmax-MoE Baseline (`final_aligned_softmax_moe`):**
   * `router_class = "TopKRouter"`
   * `top_k = 1`
   * `noise_std = 0.1` (train), `0.0` (eval)
   * `normalize_input = True`
   * `balance_coeff = 0.01`
   * `expert_path = "ResidualImpedanceExpert"` (with `beta=0.0`)
4. **Beta Validation:**
   * Implement a validation hook in the `ManifestResolver`:
     ```python
     if experiment_id in FINAL_MATRIX_EXPERIMENTS:
         assert config.expert.beta == 0.0, f"Beta must be locked to 0.0 for {experiment_id}"
     ```

#### B. The Corrected Experimental Matrix

| Method Identity | Representation Label | Router Init Label | Router Mechanism | Expert (`beta`) | Factor Changed vs. Ours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PhaseForge (Ours)`** | `phase_topo` | `phase_topo` | Prototype (Top-1) | `Residual` (0.0) | **Baseline** |
| **`No SupCon (Plain)`** | None (BC) | `phase_topo` | Prototype (Top-1) | `Residual` (0.0) | Representation only |
| **`Random Init Router`** | `phase_topo` | Random | Prototype (Top-1) | `Residual` (0.0) | Router Init only |
| **`Softmax MoE`** | `phase_topo` | Random | **Softmax Top-K** | `Residual` (0.0) | Router Mechanism only |
| **`Static Rule MoE`** | `phase` | `phase` | Prototype (Top-1) | `Residual` (0.0) | Label Source (Rule vs Topo) |
| **`Factorial Floor`** | None (BC) | Random | Prototype (Top-1) | `Residual` (0.0) | *Two factors (Rep + Init)* |

*Note: The `Factorial Floor` is explicitly marked as a 2x2 factorial corner, not a single-factor ablation.*

#### C. Corrected Paper Claims

Do not overclaim. Use the following precise language:

*   **Incorrect:** "PhaseForge solves boundary chattering and load-collapse."
*   **Correct:** "PhaseForge is designed to reduce routing boundary chattering and expert load collapse through hard prototype routing and topology-based initialization. We empirically measure routing transitions and expert utilization to validate this design."

---

### Part 3: Final Engineering Directives

1. **Manifest Implementation:** Create `final_causal_matrix.json`. Use provider identities, not hardcoded paths.
2. **Plain Encoder Fix:** Submit a PR to ensure `plain_encoder_phase_bootstrap` uses `phase_topo` for centroid computation.
3. **Softmax Baseline:** Implement the `TopKRouter` baseline with the exact pre-registered settings (Top-1, noise=0.1, balance=0.01).
4. **Diagnostics:** Keep `teacher_forced` and `oracle_moe` separate. Report them in the diagnostics table.
5. **Validation:** Add the manifest-level beta assertion.

Your rigor has ensured that this protocol is now scientifically sound and implementation-ready. There are no remaining ambiguities. 

Proceed with the implementation of the final manifest. I expect the dry-run to confirm seed-exact checkpoint resolution and the beta validation hook. 

Excellent work. Let's execute.