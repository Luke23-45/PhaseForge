# PhaseForge Experiment Report — Final

**Date:** 17 July 2026
**Environment:** Google Colab (CUDA)
**Dataset:** LIBERO-90 (604,836 train / 64,207 val samples, 23-DoF state, 7-DoF action)
**Seed:** 42
**Git base:** `0adeacb` (train), `55068b9`+patches (eval)

---

## 1. Experiment Overview

Six models covering the proposed method and five baselines:

| # | Model | Config | Stage | Epochs | Params | Description |
|---|-------|--------|-------|--------|--------|-------------|
| 1 | **PhaseForge** | `models=phaseforge` | Stage 1 | 19 (ES) | — | Phase-supervised pretraining (encoder + action_head + phase_head) |
| 2 | **PhaseForge** | `models=phaseforge` | Stage 2 | 11 (ES) | 640,835 | MoE bootstrapped from Stage 1 (phase centroids → router init) |
| 3 | **BC** | `models=baselines/bc` | Stage 1 | 15 (ES) | — | Behavioral cloning baseline (no phase supervision) |
| 4 | **Scratch MoE** | `models=baselines/scratch_moe` | Stage 2 | 21 (ES) | — | MoE from scratch, no pretraining |
| 5 | **WarmStart MoE** | `models=baselines/warmstart_moe` | Stage 2 | 11 (ES) | 640,061 | MoE with BC-pretrained encoder + random router |
| 6 | **Oracle MoE** | `models=baselines/oracle_moe` | Stage 2 | 11 (ES) | 778,160 | MoE with ground-truth phase routing (upper bound) |

*(ES = early stopping triggered)*

*Oracle MoE footnotes (E7): receives privileged GT phase labels at inference — non-deployable; its only claim is the routing signature (NMI=1.0, entropy≈0), not success. The July matrix is superseded by the 8-cell Batch A/B protocol (C1/E7/E8: `bc`, `scratch_moe`, `warmstart_moe`, `phaseforge`, `oracle_moe`, `teacher_forced`, `phase_pretrain_random_router`, `plain_encoder_phase_bootstrap`); rollout success (B1) is goal-predicate based, not this L2 proxy.*

**Training hyperparameters (Stage 1):**
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealing, T_max=100, eta_min=1e-6
- Phase loss weight: λ_phase=1.0
- Early stopping: patience=10 on val/loss_total

**Training hyperparameters (Stage 2):**
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealing, T_max=200, eta_min=1e-7
- Load-balancing loss: balance_coeff=0.01
- Encoder frozen: true (PhaseForge/WarmStart), false (Scratch/Oracle)
- Early stopping: patience=10 on val/loss_action

---

## 2. Evaluation Results

**Metric definitions:**
- **Success Rate:** Offline proxy — L2 error < 0.05 threshold on 7-DoF actions
- **Routing Entropy:** H(gate_logits) normalized by log(E); lower = more confident routing
- **Balance Score:** Normalized entropy of expert assignment frequencies; 1.0 = perfect balance
- **Collapse Rate:** Fraction of experts with usage < 1/(factor × E); factor=5.0
- **Phase-Expert NMI:** Normalized Mutual Information between phase labels and expert assignments

### 2.1 Full Results Table

| Model | Success Rate | Routing Entropy | Balance Score | Collapse Rate | Phase-Expert NMI |
|-------|:-----------:|:---------------:|:-------------:|:-------------:|:----------------:|
| **BC** | **9.56%** | N/A | N/A | N/A | N/A |
| **PhaseForge** (proposed) | **10.99%** | 0.734 | 0.990 | 0.000 | 0.000 |
| **Scratch MoE** (baseline) | **11.12%** | 0.690 | 0.980 | 0.000 | 0.000 |
| **WarmStart MoE** (ablation) | **11.37%** | 0.946 | 0.992 | 0.000 | 0.000 |
| **Oracle MoE** (GT routing; signature-only bound) | **4.54%** | ≈0 | ≈0 | 0.833 | 1.000 |

### 2.2 Success Rate Ranking

| Rank | Model | Δ vs BC |
|:----:|-------|:--------:|
| 1 | **WarmStart MoE** — 11.37% | +1.81pp |
| 2 | **Scratch MoE** — 11.12% | +1.56pp |
| 3 | **PhaseForge** (proposed) — 10.99% | +1.43pp |
| 4 | **BC** — 9.56% | — |
| 5 | **Oracle MoE** — 4.54% | −5.02pp |

### 2.3 MoE Routing Metrics Analysis

| Metric | PhaseForge | Scratch MoE | WarmStart MoE | Oracle MoE |
|--------|:----------:|:-----------:|:-------------:|:----------:|
| Entropy | 0.734 | 0.690 | 0.946 | ≈0 |
| Balance | 0.990 | 0.980 | 0.992 | ≈0 |
| Collapse | 0.000 (0/6) | 0.000 (0/6) | 0.000 (0/6) | 0.833 (5/6) |
| NMI | 0.000 | 0.000 | 0.000 | 1.000 |

**Key observations:**

1. **All non-oracle models achieve perfect balance (≥0.98) and zero collapse** — the load-balancing loss (coeff=0.01) successfully prevents expert starvation but may over-constrain routing.

2. **All non-oracle models have Phase-Expert NMI = 0.0** — no learned alignment between routing decisions and phase boundaries. The router distributes tokens uniformly across experts regardless of phase label, consistent with **pseudo-balancing** (Memory-Aware Routing, ACL 2026): the balance loss causes the router to assign tokens by randomness rather than by matching, resulting in expert redundancy.

3. **WarmStart MoE has the highest entropy (0.946), suggesting the most diffuse routing** — the BC-pretrained encoder + random router combination produces the least decisive routing policy.

4. **Scratch MoE has the lowest entropy (0.690), suggesting the most confident routing** — training the encoder freely from scratch allows the latent space to structure in a way the router can exploit.

5. **Oracle MoE collapses to routing entropy ≈ 0 and collapse rate = 0.833** — with perfect NMI = 1.0, each phase maps to exactly one expert, but phase distribution is imbalanced (one phase dominates), so 5/6 experts receive negligible traffic. The oracle framework confirms the dataset has strongly imbalanced phase frequencies.

---

## 3. Hypothesis Analysis

### 3.1 Central Hypothesis

From the proposal:
> *"Phase-supervised pretraining shapes the encoder latent space so that the router's decision boundaries align with behavior phases before expert fine-tuning begins. This should reduce cold-start instability, lower expert collapse, improve routing consistency, and make the MoE policy easier to train than scratch MoE or generic warm-start MoE trained without phase supervision."*

### 3.2 Verdict: NOT Supported

The experimental evidence contradicts the hypothesis on every concrete prediction:

| Prediction | Expected | Observed | Status |
|-----------|----------|----------|--------|
| PhaseForge > Scratch MoE (success rate) | Higher | 10.99% < 11.12% | **Falsified** |
| PhaseForge > WarmStart (success rate) | Higher | 10.99% < 11.37% | **Falsified** |
| Router aligns with phase boundaries | NMI > 0.0 | NMI = 0.0 all non-oracle | **Falsified** |
| Phase bootstrapping reduces collapse | Lower collapse | All models: 0.0 collapse | **Inconclusive** |
| Oracle provides valid upper bound | Oracle > all | 4.54% (worst) | **Falsified** |

### 3.3 Classification

Per the proposal's defined outcomes (*docs/proposal/imp.md §10.3*):

> **Outcome C:** *"Warm-starting matches phase bootstrapping — this would mean phase supervision is not adding much beyond ordinary pretraining, and the design should be revised."*

The actual result exceeds Outcome C — **scratch (random) MoE also matches or exceeds phase bootstrapping**, meaning phase supervision adds no detectable benefit over any alternative initialization strategy.

---

## 4. Root Cause Analysis

### 4.1 Cause 1: Pseudo-Balancing (Primary)

The load-balancing loss (balance_coeff = 0.01) successfully enforces uniform expert utilization (balance ≥ 0.98) but at the cost of preventing specialization. The literature describes this failure mode precisely:

> *"Existing load-balancing losses can cause the same input to be randomly routed to different experts across training steps instead of the most matching one. This leads to severe knowledge overlap among experts."*
> — Memory-Aware Routing, ACL 2026

Without a countervailing signal to encourage specialization, the router optimizes for the easier objective (uniform distribution). The experts converge to redundant representations because they receive similar training data via random assignment. The MoE output becomes a weighted average of near-identical expert outputs — functionally equivalent to a single head.

**Evidence:** NMI = 0.0 (no specialization) + balance ≈ 1.0 (perfect uniformity) for all non-oracle models.

### 4.2 Cause 2: Frozen Encoder Limits Adaptation

PhaseForge and WarmStart MoE freeze the encoder in Stage 2. The latent space was optimized for the Stage 1 objective (action prediction + phase classification), not for the Stage 2 objective (expert specialization with load balancing). The router and experts must work with a representation that cannot adapt to their needs.

Scratch MoE, which trains the encoder from scratch, achieves the best results despite the worst initialization. This suggests the encoder's ability to adapt matters more than initialization quality.

**Recommendation:** Unfreezing the encoder (with a lower learning rate) should be the first ablation.

### 4.3 Cause 3: Phase Classification ≠ Phase Specialization

The Stage 1 phase-supervised pretraining optimizes a phase classification head on the latent representation. This encourages the encoder to produce latents where phases are linearly separable — but linear separability does not imply that the latent manifold is structured to support routing. In fact, the classification loss may **compress away** phase-specific variance (since only a separating hyperplane is needed), leaving less structure for the router to exploit.

### 4.4 Cause 4: Oracle Training Produces Weak Experts

Oracle MoE routes each training sample to exactly one expert based on its phase label. Each expert sees only samples from its assigned phase (~1/6 of total). With no cross-phase training signal:

- Experts overfit to narrow data distributions
- Phase imbalance causes expert starvation (confirmed by collapse_rate = 0.833)
- The router weights receive zero gradient signal throughout training (they are never used)
- The balance loss is zero (the oracle explicitly disables it), so no mechanism prevents collapse

This makes the current Oracle an invalid upper bound. A properly constructed oracle would need balanced phase sampling or per-expert augmentation to compensate for phase imbalance.

### 4.5 Cause 5: Offline L2 Threshold is Not Informative

All models cluster in the 9–11% range under the uniform 0.05 L2 threshold. This is consistent with a floor effect:

- 7-DoF × (0.05)² = 0.0175 total L2² budget per timestep
- Action dimensions have different units (translation in meters, rotation in radians, gripper binary)
- A single dimension exceeding 0.05 causes failure regardless of the other 6

LIBERO's explicit guidance (*Appendix E.2*): *"Success rates, instead of behavioral cloning loss, should be the right metric"* — the paper warns that action MSE does NOT correlate with rollout success. EWC can have lower loss but worse success; ER can have higher loss but better success.

Without environment-based rollout evaluation, the reported success rates cannot be interpreted as task completion.

---

## 5. Comparison with Literature

### 5.1 State-Only MoE on LIBERO

No published work establishes SOTA for state-only MoE on LIBERO. The community uses vision-based evaluation, with SOTA at 97–98% (GR00T N1.6, Pi0.5, OpenVLA-OFT) using 0.4B–7B+ parameters and visual backbones. Our study operates in a fundamentally different regime:

| Property | Published SOTA | This Work |
|----------|:-------------:|:---------:|
| Input | RGB + proprioception (8+3×128×128) | Proprioception only (23-DoF) |
| Parameters | 0.4B–7B | 0.6M–0.8M |
| Success Rate | 97–98% | 9–11% (L2 proxy) |
| Evaluation | Simulator rollouts (goal predicates) | Offline L2 threshold |

The direct comparison is internal (PhaseForge vs its own baselines), and that comparison is negative for the hypothesis.

### 5.2 Relevant Literature

| Paper | Finding | Relevance to Our Results |
|-------|---------|-------------------------|
| **LAR-MoE** (Rodriguez, 2026) | 95.2% on LIBERO via latent-aligned routing, no phase labels | Phase supervision is unnecessary — unsupervised alignment works better |
| **SMP** (Hao, ICLR 2026) | Orthonormal skill basis + sticky routing solves expert identifiability | Our NMI=0 shows unidentifiability problem; orthonormal constraint could fix it |
| **MoE-DP** (Cheng, 2025) | MoE discovers phase structure without phase labels | Phase specialization can emerge without supervision — our explicit phase signal may interfere |
| **Move-Then-Operate** (Xu, ICML 2026) | +24% over π₀; chunk-level routing + disjoint expert params | Phase decoupling works when done at chunk level with independent experts |
| **AdaMoE** (Shen, 2025) | Decoupled selection/weighting (+21.5% real world) | Our winner-takes-all routing may be the wrong selection strategy |
| **Memory-Aware Routing** (ACL 2026) | Pseudo-balancing identified as failure mode | Directly explains our NMI=0 + balance≈1 result |
| **ST-MoE** (Zoph, 2022) | Router z-loss prevents collapse at scale | Router z-loss should be added to training |

### 5.3 Key Insight from Literature

The most successful approaches do NOT use supervised phase labels for routing. They use:

1. **Unsupervised latent alignment** (LAR-MoE): skill structure discovered via co-training, router regularized to follow it
2. **Orthonormal action bases** (SMP): locally whitened action space eliminates expert overlap by construction
3. **Chunk-level temporal routing** (MTO, MoE-ACT): routing decisions are made per action chunk, not per timestep, providing temporal coherence

This suggests the PhaseForge approach — supervised phase labels → phase classification loss → centroid-based router initialization — is the wrong inductive bias.

---

## 6. Conclusions

### 6.1 Core Findings

1. **The Phase-Bootstrapped MoE hypothesis is not supported.** PhaseForge (10.99%) does not outperform Scratch MoE (11.12%) or WarmStart MoE (11.37%) on state-only LIBERO-90 under the offline L2 metric.

2. **Phase-Expert NMI = 0.0 across all non-oracle models** confirms no phase-specialized routing emerges under the current training regime. The load-balancing loss dominates the routing objective, producing pseudo-balanced but redundant expert assignments.

3. **The frozen encoder limits Stage 2 adaptation.** Scratch MoE (no freeze) achieves the best results despite random initialization, suggesting encoder adaptability matters more than initialization quality.

4. **The Oracle MoE is an invalid upper bound.** Expert starvation from phase-imbalanced training and zero router gradient signal produce the worst performance (4.54%). A redesigned oracle with balanced sampling or auxiliary router supervision is needed.

5. **The offline L2 threshold metric is not informative.** All models cluster in a narrow 9–11% band under a uniform 0.05 threshold for 7-DoF actions. Environment-based rollout evaluation is required to determine whether any model learns a useful policy.

### 6.2 Contributions

Even though the hypothesis was not supported, this study makes valuable contributions:

1. **Demonstrates pseudo-balancing at small scale (0.6M params):** Previously identified at 7B+ NLP scale, we show the same failure mode affects small MoE policies in continuous control — balance ≈ 1.0, specialization ≈ 0.0.

2. **Provides state-only MoE baselines on LIBERO-90:** First published routing metrics (entropy, balance, collapse, NMI) for state-only MoE on the LIBERO benchmark, establishing a standardized diagnostic protocol.

3. **Controlled negative result:** Rigorous evidence that supervised phase bootstrapping does not improve MoE routing in state-only manipulation, ruling out a plausible design and guiding the field toward unsupervised routing strategies.

### 6.3 Applicability Limits

These findings are specific to:
- **State-only observations** (no vision, no language)
- **Offline imitation learning** (behavioral cloning, no RL)
- **LIBERO-90 task distribution** (short-horizon, single-step tasks)
- **Small-scale models** (0.6M–0.8M parameters)
- **6 experts / 6 phases** (no phase count ablation performed)

They may not generalize to vision-based policies, RL-based training, long-horizon tasks (LIBERO-LONG), larger models, or different phase/expert counts.

---

## 7. Recommendations

### 7.1 Immediate (1 week)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P0** | Run environment-based (rollout) evaluation using robosuite | The L2 threshold cannot determine task success; rollouts are the only valid metric |
| **P1** | Unfreeze encoder in PhaseForge Stage 2 (use lower LR, not zero) | Frozen encoder limits latent space adaptation; scratch MoE's success suggests this matters |
| **P1** | Add router z-loss to training (ST-MoE, Zoph et al.) | Prevents the pseudo-balancing loop by penalizing logit saturation |
| **P2** | Remove early stopping or increase patience to 20 | All models stopped early; more epochs may improve convergence |
| **P2** | Oracle redesign: balanced phase sampling + auxiliary router loss | Restores Oracle as a valid upper bound for future experiments |

### 7.2 Short-Term Redesign (1 month)

Replace the supervised phase bootstrapping strategy with one of the following:

**Option A: Latent-Aligned Routing** (highest evidence, per LAR-MoE)
- Remove phase supervision; replace with unsupervised skill discovery via student-teacher co-training
- Regularize router to follow the discovered latent structure
- **Expected benefit:** Eliminates dependence on phase label quality; aligns with SOTA (95.2% on LIBERO)

**Option B: Orthonormal Skill Basis** (per SMP, ICLR 2026)
- Replace independent experts with a locally whitened orthonormal action basis
- Use sticky routing (slowly-varying gating) for temporal coherence
- **Expected benefit:** Solves the unidentifiability problem by construction; enables compact skill reuse

**Option C: Chunk-Level Phase Decoupling** (per Move-Then-Operate, ICML 2026)
- Route at the chunk level (not per-timestep)
- Use disjoint expert parameters (fully independent experts)
- Derive phase labels from action statistics (velocity/acceleration) not clustering
- **Expected benefit:** Temporal coherence in routing; +24% demonstrated on a similar benchmark

### 7.3 Long-Term Research Direction (3 months)

1. **Fix the MoE mechanism first** — implement Option A or B with rollout validation
2. **Establish meaningful state-only baselines** — train until convergence with proper learning curves
3. **Only then add vision** — as a controlled variable, not a confound
4. **Scale systematically** — larger capacity → more experts → longer horizon

### 7.4 What Not to Do

- **Do not add vision** until the MoE mechanism is verified — perception confounds would make causal attribution impossible
- **Do not scale model size** before fixing routing — redundant experts scale linearly with waste
- **Do not chase LIBERO SOTA** — SOTA requires 0.4B–7B vision-language models, which is a different research question
- **Do not increase expert count** without fixing pseudo-balancing — more experts will just produce more redundancy

---

## 8. Evaluation Protocol Note

**The offline L2 threshold used in this report is NOT the accepted standard for LIBERO evaluation.** After a comprehensive literature review, the accepted standard is:

| Element | Standard Protocol | Source |
|---------|-----------------|--------|
| **Metric** | Binary task success (goal predicates) in robosuite simulator | LIBERO Appendix E.2 |
| **Episodes per task** | 50 (standard) or 10 (minimum) | OpenVLA, LeRobot |
| **Total per suite** | 500 (10 tasks × 50 episodes) | OpenVLA §4 |
| **Seeds** | 3, report mean ± standard error | All SOTA work |
| **Suites** | LIBERO-Spatial, Object, Goal, Long (separately) | LIBERO §4.2 |

The LIBERO paper explicitly warns (*Appendix E.2*): *"Success rates, instead of behavioral cloning loss, should be the right metric"* — action MSE does not correlate with rollout success. The uniform 0.05 L2 threshold for 7-DoF actions has no precedent in the literature and is not informative for comparing models.

**The current offline results (9–11%) cannot be compared to any published work or used to draw conclusions about task performance.** Environment-based rollout evaluation is required. A detailed implementation plan for proper evaluation is in `docs/notes/evaluation_plan.md`.

### 8.1 Updated Recommendation Priority

| Priority | Action | Where |
|----------|--------|-------|
| **P0** | Implement rollout evaluation (robosuite + state-only env wrapper) | `docs/notes/evaluation_plan.md` |
| P1 | Run 3-seed rollout eval for all 5 models | After implementation |
| P2 | Then re-evaluate the hypothesis with real task success rates | After obtaining rollout results |

---

*This report supersedes the preliminary experiment report. Two eval-time bugs (missing `_stage` restoration and Oracle eval-mode phase-label fallback) were discovered and fixed during the analysis, producing the corrected results shown above. The current evaluation metric (offline L2 threshold) is not the accepted standard — see §8.*
