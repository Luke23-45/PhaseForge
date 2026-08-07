# PhaseForge — Novelty Claim & Positioning

**Purpose:** expand section D of the issues register (`docs/notes/issues_register.md`) into the full statement of what PhaseForge claims, what prior work already covers (verified online), what is genuinely ours, and the empirical program required to defend it. This document drives the paper's contribution statement and the reply to the professor.

**As of:** 2026-08-07 · Rev. 2: teacher-forced routing cell (E8) added — decomposable oracle (§3.2), prediction 4 (§4), experiment + decision rules (§5), positioning (§6), risk row (§7). · Cross-references: `issues_register.md` (D1, D4, C1, C3), `REPORT_to_professor_2.md`

---

## 1. The research question (restated)

PhaseForge is a **training-strategy study**, not a new-architecture or new-perception study:

> Does bootstrapping a MoE router from phase structure discovered in a *frozen, phase-supervised latent* produce more stable routing, cleaner expert specialization, and better long-horizon policy behavior than (a) no structure (scratch), (b) generic warm-starting (BC encoder, random router), (c) supervised routing without initialization (MEAT/PAMAE/MTO-style), or (d) unsupervised latent alignment (LAR-MoE-style)?

Two-stage design: **Stage 1** trains a generalist BC policy with an auxiliary phase-classification head (6 rule-based phases). **Stage 2** freezes the encoder, initializes 6 experts from the stage-1 action head, **initializes the router weights from phase centroids computed in the stage-1 latent space**, and continues training with top-2 routing under an auxiliary load-balancing loss. Inference is state-only; phase labels are never an input.

Five baselines (two to be added): `bc`, `scratch_moe`, `warmstart_moe`, `oracle_moe` (ground-truth phase routing, upper bound), and the proposed `phaseforge`; plus `phase_pretrain_random_router` and `plain_encoder_phase_bootstrap` to complete the 2×2 (issues register C1).

---

## 2. The verified prior-art landscape (what is taken)

All of the following were verified online (2026-08-07). The general claim — *"phase/skill structure helps MoE specialize in manipulation"* — is already made by many:

| Paper | Venue/ID | Core mechanism | Overlaps with us |
|---|---|---|---|
| **MEAT** (Mazza et al.), "MoE-ACT: Improving Surgical IL Policies through Supervised MoE" | arXiv 2601.21971 | Supervised MoE on ACT; gating trained with phase-label CE; experts per phase | Phase-supervised routing (no init strategy) |
| **MoE-ACT** (Guo et al.), bimanual | arXiv 2603.15265 | MoE FFN in ACT encoder; language-conditioned; FiLM | MoE + ACT family (task-level, not phase-level) |
| **SMP** (Hao, Zhai, Liu, Soh) | ICLR 2026, arXiv 2601.21251 | Diffusion MoE; state-adaptive orthonormal skill basis; sticky Dirichlet-Markov gating; distilled state-only router | Phase-consistent activation; identifiability fix for exactly our NMI=0 symptom |
| **LAR-MoE** | arXiv 2603.08476 | Unsupervised student–teacher latent co-training; routing regularized to latent; no labels | Two-stage; challenges our supervised-phase premise |
| **AdaMoE** (Shen et al.) | arXiv 2510.14300 | Decouples expert selection from weighting (scale adapter); resolves balance-vs-specialization conflict | The C3 dilemma, solved at router level |
| **Memory-Aware Routing** (Hou et al.) | Findings ACL 2026, pp. 17320–17337 | Identifies **pseudo-balancing**; memory buffers for consistent routing | Our dry-run signature (NMI=0, balance ≥ 0.98), at LLM scale |
| **Move-Then-Operate** (Lei et al.) | ICML 2026, arXiv 2604.23620 | Dual-expert (move/operate); chunk-level phase router; teacher-forcing; MLLM phase labels | Phase-supervised routing + label generation |
| **PAMAE** | arXiv 2606.27144 | Phase-aware router + phase head; two-stage (warmup → supervised routing, annealed) | Closest two-stage structure — still no centroid bootstrap |
| **CoRDE** | arXiv 2606.21935 | Frozen concept encoder → variational expert responsibility; LoRA experts | Semantic priors instead of phase labels |

**What this means:** the general finding is not publishable on its own. The contributions must be (i) the specific initialization mechanism, (ii) the evaluation framework, (iii) the controlled-study design — and (iv) any empirical finding (e.g., small-scale pseudo-balancing) that is new by scale/domain.

---

## 3. The gap analysis — what is actually ours

### 3.1 Mechanism: phase-centroid router bootstrap (primary claim)

No verified paper initializes router **weights** from phase centroids computed in a frozen pretrained latent. The field's approaches to phase-supervised routing:

- **Supervise routing gradients directly** (MEAT: CE on gate; MTO: teacher-forcing; PAMAE: routing-alignment loss) — the router learns from scratch under supervision.
- **Align routing to unsupervised structure** (LAR-MoE: latent distances; CoRDE: concept prior).
- **Distill a gate posterior** (SMP: state-only router matched to amortized posterior).
- **Fix routing mechanics** (AdaMoE: decoupled weighting; MAR: memory buffers).

PhaseForge instead treats phase structure as an **initialization prior**: stage-1 phase supervision shapes the latent, phase centroids are computed there, and the router starts at those centroids. The phase structure is injected *before* any routing optimization, then refined by the balance loss. This is a distinct, unclaimed training strategy. Its falsifiable prediction: **the bootstrap survives load balancing** — i.e., NMI between phase labels and expert assignment > 0 at convergence, unlike warmstart/scratch, *without* sacrificing balance or inducing collapse.

### 3.2 Evaluation framework: decomposable oracle + NMI diagnostics (methodology claim)

The `oracle_moe` baseline (GT-phase routing, expected signature NMI=1.0, entropy≈0, balance≈0 — observed in the dry run) is a faithful upper bound on phase-aligned specialization, and the routing-metric battery (time-to-stable-routing, entropy, balance, collapse, phase-expert NMI) is an interpretable diagnostic instrument. No verified manipulation-MoE paper uses this combination (SMP reports gate patterns; PAMAE reports dominance purity — neither has an oracle bound).

**Decomposable oracle (new cell, E8).** A teacher-forced variant — experts partitioned by GT phase during training, routing at inference by a learned phase predictor (`argmax` of the stage-1 phase head) — turns the oracle into a *decomposable instrument* that splits the failure budget:

```
oracle (GT routing)            = ceiling with perfect phase knowledge
oracle + predicted phases      = ceiling with *learnable* phase knowledge (teacher-student)
phaseforge                     = implicit, balance-constrained routing
Gap 1 (oracle − predicted)     = phase-predictability loss: is the phase signal inferable from the state?
Gap 2 (predicted − phaseforge) = strategy loss: what the bootstrap + balance loss costs
```

This is the professor-endorsed teacher-student pattern: privileged *training*, label-free *inference*. The cell is no longer an oracle and must be renamed (e.g., "teacher-forced routing") in all artifacts. Its success becomes a valid bound again (still footnoted as privileged-training per the honesty rule), and Gap 1 answers the C4 question — are the rule-based phases grounded in the 23-DoF state? — quantitatively. Caveat (issues register B2/B3): expert imbalance/starvation still applies to the GT partition — keep balanced sampling or an auxiliary router loss on the table for this cell.

### 3.3 Experimental design: controlled state-only factorial (design claim)

A 2×2 factorial over (encoder init: phase-supervised vs plain) × (router init: centroid-bootstrap vs random), plus `bc` and the oracle, at 0.6–0.8M parameters, on LIBERO-90 with the accepted rollout protocol — this is the first controlled study isolating the *training strategy* variable in phase-structured MoE. All vision/VLA-scale work bundles architecture + perception + scale; we deliberately hold those constant (state-only, small scale) so that any measured difference is attributable to the strategy.

### 3.4 Empirical finding: pseudo-balancing in small-scale continuous control (secondary claim)

MAR documents pseudo-balancing at LLM scale. Our dry run shows the same signature at 0.6M parameters in continuous control (balance ≥ 0.98, NMI = 0.0, zero collapse — redundancy with balance). If this holds after the evaluation fixes (A2/B6/C3), it is a transferable, small-scale demonstration of the failure mode and a clean motivation for the bootstrap.

---

## 4. The sharpened novelty claim (wording for the paper)

> We study the **training strategy** of phase-structured MoE policies for manipulation. Prior work supervises routing directly (MEAT; Move-Then-Operate; PAMAE), aligns routing to unsupervised latents (LAR-MoE), or fixes routing mechanics (AdaMoE; Memory-Aware Routing). We show that **bootstrapping the router from phase centroids in a frozen, phase-supervised latent** is a distinct initialization strategy that (prediction 1) yields phase-expert alignment (NMI > 0) that survives load-balancing, unlike random- or plain-warm-started routers, (prediction 2) improves routing stability and rollout success over the same architecture without the bootstrap, and (prediction 3) is bounded above by an oracle-routing policy whose NMI=1.0 signature validates the diagnostics. (prediction 4) The teacher-forced routing cell — explicit phase supervision on expert assignment, route by predicted phase at inference — lies between the oracle and PhaseForge: Gap 1 quantifies phase predictability from the state, Gap 2 quantifies the strategy loss. If PhaseForge matches or beats the teacher-forced cell, the bootstrap beats direct supervision; if not, the claim falls back to a controlled negative result.

**Explicitly NOT claimed:** vision-level performance; LIBERO leaderboard numbers; the general finding "phase structure helps MoE" (taken); perception capability of any kind (stages 2–3 are separate).

**Metric-meaning rules (honesty):** NMI is an emergent-specialization test only for `phaseforge`, `scratch_moe`, `warmstart_moe` — for the teacher-forced cell NMI is *prediction quality* (trained with CE to match GT), and for the oracle NMI=1.0 is a sanity check. The teacher-forced cell receives privileged labels during *training* only (teacher-student, footnoted); the oracle receives them at inference too (non-deployable, footnoted).

---

## 5. Empirical program to defend the claim

| # | Experiment | Gate / decision |
|---|---|---|
| E1 | **Evaluation fixes first** (issues A2, B6): in-distribution suite decided (libero_90 as primary; spatial/object/goal/10 as labeled zero-shot); state-replay consistency test passes | Blocking — nothing below is interpretable without it |
| E2 | **2×2 factorial** (C1): add `phase_pretrain_random_router`, `plain_encoder_phase_bootstrap`; train all 7 models full-length (100/200 epochs, no truncating early stop) | Isolates encoder-init vs router-init effects |
| E3 | **Full-length training** (C5) on the object-state channel (Stage 1 of professor plan) | Real ceilings, not dry-run floors |
| E4 | **Balance-vs-NMI logging + balance-weight sweep** (C3): 0 / 0.01 / 0.1 | Tests the pseudo-balancing mechanism; if balance kills NMI at all weights, the bootstrap claim needs the orthogonal-basis direction (SMP) or decoupling (AdaMoE) |
| E5 | **Rollout protocol**: 5 suites × 50 episodes/task × 3 seeds, per-suite + per-task breakdowns; oracle footnoted as non-deployable (B3) | Predictions 1–3 testable; zero-shot suites reported as such |
| E6 | **Phase-label spot-check** (C4) on real trajectories | Protects the bootstrap, NMI, and oracle from label-error propagation |
| E7 | **Oracle redesign decision** (B2): balanced sampling or auxiliary router loss, OR relabel as signature-only bound | Keeps the upper-bound claim honest |
| E8 | **Teacher-forced routing cell** (oracle + learned phase predictor; route at inference by `argmax` of predicted phase) | Fills the supervision-regime axis; quantifies Gap 1 (phase predictability) and Gap 2 (strategy loss); head-to-head vs the MEAT/MTO/PAMAE recipe |

**Decision rules** (from the proposal, updated for the factorial):

- Supported: PhaseForge ≥ warmstart ≥ scratch on success AND routing stability, with NMI > 0 for phaseforge at convergence.
- Mechanism-level support: the 2×2 shows the *encoder* effect (phase vs plain) and the *router* effect (bootstrap vs random) separately; a win on either axis is attributable.
- Rejected: PhaseForge ≤ warmstart ⇒ phase structure adds nothing beyond pretraining; NMI stays 0 for all ⇒ bootstrap does not survive balance ⇒ pursue the SMP/AdaMoE directions (E4) or revise the hypothesis.
- Caveat: if `oracle_moe` success ≤ learned models (as in the dry run), the oracle cannot bound success — only the routing signature (NMI=1.0) is claimed (3.2).
- Teacher-forced cell (E8): if Gap 1 (oracle − predicted) is large ⇒ phases are not grounded in the state (C4 is the root cause, not the model); if Gap 2 (predicted − phaseforge) is large ⇒ explicit supervision beats the bootstrap (mechanism claim dropped, controlled comparison stands); if phaseforge ≥ predicted ⇒ the bootstrap beats direct supervision (strongest positive outcome).

---

## 6. Positioning strategy

- **Comparison discipline (D2):** state-oracle/state-only numbers are an internal architecture sanity check. Never placed next to OpenVLA/π0.5/ACT numbers; protocol declared (50 episodes/task; in-distribution vs zero-shot per suite).
- **Related-work framing:** position against the five routing strategies in §3.1 — supervised (MEAT/MTO/PAMAE), unsupervised-aligned (LAR-MoE), posterior-distilled (SMP), mechanical-fix (AdaMoE/MAR) — with our init-prior strategy as the new cell. The teacher-forced cell (E8) is our *controlled representative of the supervised family*, making the comparison head-to-head within one setting rather than across papers. Cite both MoE-ACT papers disambiguated; correct citations per issues register D4.
- **Labelling:** oracle footnoted as non-deployable privileged bound; zero-shot rows labeled; stage-1 numbers never called "LIBERO results."
- **Stages 2–3 (per professor plan):** the novelty claim lives in stages 1–2. Cached-embedding (stage 2) and end-to-end vision (stage 3) are perception questions, orthogonal to the strategy claim; they extend the study only after the strategy question is answered.

---

## 7. Risks to the claim and mitigations

| Risk | Source | Mitigation |
|---|---|---|
| Unsupervised latent alignment beats supervised bootstrap | LAR-MoE (verified) | 2×2 + E4; if confirmed, report as controlled negative result — still a valid contribution (first state-only factorial) |
| Our NMI=0 is the identifiability problem SMP solves by construction (orthogonal basis) | SMP (verified) | E4 sweep decides; if balance always kills NMI, adopt orthonormal-basis or decoupled-weighting variant as follow-up, not in this paper |
| Load balancing itself is the enemy | AdaMoE, MAR (verified) | Sweep balance weight; log balance-vs-NMI per epoch (C3) |
| Supervised routing (MEAT/MTO/PAMAE) makes "phase helps" unnovel | verified | Claims are init-mechanism + evaluation-framework + controlled-study, not the general finding |
| Zero-shot confound resurfaces and is misread as "fix failed" | issues A2 | In-distribution suite primary; zero-shot labeled; gates in E1 |
| Oracle invalid as success bound | dry run (collapse 0.833) | Signature-only claim or redesign (E7) |
| Supervised routing (teacher-forced cell, E8) beats the bootstrap | MEAT, MTO, PAMAE (verified) | Head-to-head cell makes the outcome falsifiable; if confirmed, drop the mechanism claim and report the controlled comparison itself as the contribution |

---

## 8. References (verified 2026-08-07)

1. Mazza et al., "MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts" (MEAT), arXiv:2601.21971.
2. Guo, Liu, Sun, Zhao, Zhou, Ma, "MoE-ACT: Scaling Multi-Task Bimanual Manipulation with Sparse Language-Conditioned Mixture-of-Experts Transformers", arXiv:2603.15265.
3. Hao, Zhai, Liu, Soh, "Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies" (SMP), ICLR 2026, arXiv:2601.21251.
4. "LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Manipulation", arXiv:2603.08476 (authors unverified — confirm before citing).
5. Shen et al., "Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning" (AdaMoE), arXiv:2510.14300.
6. Hou et al., "From Pseudo-Balancing to True Specialization: Memory-Aware Routing for Mixture-of-Experts", Findings of ACL 2026, pp. 17320–17337, DOI 10.18653/v1/2026.findings-acl.857.
7. Lei, Gu, Tang, Chen, Wang, "Move-Then-Operate: Behavioral Phasing for Human-Like Robotic Manipulation", ICML 2026, arXiv:2604.23620.
8. PAMAE: "Phase-Aware-MoE Action Experts Towards Reliable Flow-Matching VLA Policies", arXiv:2606.27144.
9. CoRDE: "Concept-prior Routed Diffusion Experts for Structural Generalization in Robot Manipulation", arXiv:2606.21935.
10. Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning", NeurIPS 2023 (Datasets & Benchmarks).
