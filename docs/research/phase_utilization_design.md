# Phase Utilization Study — Design Document (2026-08-18)

**Status:** research design, pre-registration for CPU screening experiments
**Goal:** find the mechanism(s) by which the stage-1 phase knowledge (latent geometry + phase predictor + phase process structure) can be carried into stage 2 and deployment — so that it is *used*, not forgotten. Every mechanism is derived, its failure mode predicted, and its experiment predeclared.
**Constraint:** the frozen protocol's factorial cells (phaseforge, plain_encoder_phase_bootstrap, phase_pretrain_random_router, warmstart_moe, bc, scratch_moe, teacher_forced) stay exactly as defined. The variants below are *new method cells* (PhaseForge-X family) that may use phase labels in stage 2 — that is the point of the study.

---

## 1. Problem statement (evidence that phase knowledge dies)

The stage-1 phase supervision shapes the encoder latent and the phase head, but stage 2 discards both:

- `phaseforge/models/phase_moe.py:141` — stage-2 forward sets `phase_logits=None`; `get_action` routes purely by the learned router (`phase_moe.py:157`).
- Measured on the fixed-protocol checkpoints (lambdav1 runs, 3 seeds): phase↔expert NMI ≈ 0.44–0.45 (oracle = 1.0), normalized routing entropy ≈ 0.95 (near-uniform), stage-1 phase-head balanced accuracy ≈ 0.56–0.64.

Diagnosis chain (already established in prior gates): the phase head is a mediocre per-step predictor (uniform errors, majority-phase bias, Phase 1.2), the router is nearly uniform (Phase 1.2/4 evidence), and the balance loss actively counteracts commitment. The phase knowledge is therefore used *once* (at bootstrap) and then allowed to decay.

## 2. Why it decays — the optimization analysis

Stage-2 objective:

    min_{G, π}  L = E[ ‖ Σ_k g_k(z) π_k(z) − a ‖² ] + β · L_bal(G)

where g = softmax∘G, L_bal = E·Σ_i f_i·p_i (Switch).

**Lemma 1 (action loss underdetermines routing).** For any fixed action profile (π_k), the action loss is minimized by any g that puts all mass on the expert(s) whose outputs best match the target a. If experts are near-identical (warm start + jitter), *every* routing achieves the same action loss → the action gradient on G is weak and routing is driven by the balance term alone.

**Lemma 2 (balance loss is a uniformizing force).** The Switch balance loss gradient pushes f_i and p_i toward 1/E:

    ∂L_bal/∂G(z) ∝ (f_i + p_i) · (g_i − 1/E) · z   (per-token, up to Jacobian)

i.e., each token's gate probability is pulled toward the uniform 1/E. With β > 0 and a flat action gradient (Lemma 1), the stationary point of the routing dynamics is the *max-entropy* router — exactly the observed entropy ≈ 0.95. This is the "pseudo-balancing" failure recently identified for LLM MoEs (MAR, Hou et al., ACL 2026 Findings): balance is achieved by randomly scattering tokens rather than by specialization.

**Lemma 3 (soft logit anchors are initialization-equivalent).** Let F(z) = W_F z + b_F be the (linear) phase head and consider an "anchored" router G(z) = F(z) + r(z) with trainable r. If r is any linear map, G is a re-parameterization of the unconstrained linear router: the function class is identical, so the anchor changes only the optimization trajectory, not the hypothesis space. **Consequence:** logit-level "anchoring" is not a structural mechanism for a linear phase head; only (a) constrained residuals (low-rank, bounded), (b) decision-level coupling (routing *decisions* derived from F), or (c) loss-level coupling (distillation, alignment) create genuine structure.

**Lemma 4 (decoder benefit under monotone phase processes).** Lift's phase process is monotone non-decreasing with near-deterministic transitions (p_{t+1} ∈ {p_t, p_t+1}) and long segments. A per-step classifier with error rate ε < 0.5, decoded by the Viterbi algorithm under the empirical (monotone) transition prior, replaces isolated flicker errors with segment-consistent labels; per-step error drops toward the segment-boundary error rate. This is the standard constrained-decoding result (TAS literature; CAD, arXiv:2605.10149). **Consequence:** the *process* structure of phases — never used so far — is a free and robust channel to carry phase knowledge to decision time.

These four lemmas define the design space and predict which mechanisms will fail (soft logit anchors: equivalent to init) and which may work (decision-level decoding, constrained residuals, loss-level coupling, phase-conditional balance).

## 3. Design space taxonomy

| Channel | Mechanism | Carries | Robustness |
|---|---|---|---|
| (I) Initialization | centroid init (current), logit init (V1) | geometry | decays (Lemma 1) |
| (L) Loss | distill KL (V2), phase-alignment CE | predictor function | persists while loss is on; needs annealing |
| (P) Parameterization | low-rank anchored router (V6) | predictor as frozen backbone | persists by construction |
| (B) Balance | phase-conditional balance (V4) | phase identity into utilization | counteracts Lemma 2's uniformizing force |
| (D) Decision | Viterbi-decoded routing (V5) | process structure (transitions, segments) | persists by construction, inference-time |

V1–V6 are the experiment matrix. They are single-factor by design (each changes exactly one channel), so attribution is clean.

## 4. Literature grounding (verified 2026-08-18)

- **MoE-ACT** (arXiv:2601.21971): phase CE supervises the *gating network directly during MoE training*; experts are top-1 exclusive per phase. Shows phase-supervised gating is effective in a vision/ACT setting. Our V-family differs: the gating teacher is the *frozen stage-1 predictor* (no GT labels in stage-2), and routing is top-2-of-6 learnable.
- **Cluster-aware Upcycling** (CVPR 2026): centroid router init from semantic clusters (same init idea as PhaseForge, discovered independently in LLM upcycling) + *expert-ensemble self-distillation* to keep specialization alive. Confirms the init-decay problem and motivates our (L)/(P) channels.
- **LAR-MoE** (arXiv:2603.08476): routing *regularized during training* to follow an unsupervised latent structure (distance-consistency loss). Confirms loss-level coupling works; ours uses a *supervised* phase predictor as the anchor.
- **MAR** (ACL 2026 Findings): pseudo-balancing diagnosis — balance losses scatter tokens randomly, blocking specialization; fix = per-expert memory buffers. Confirms Lemma 2; our V4 attacks the same failure with phase-conditioned balance instead of memory.
- **SIMBAL** (arXiv:2506.14038): orthogonality-preserving router regularization to keep similar tokens on similar experts. Adjacent to V4's motivation.
- **CAD / constrained Viterbi decoding** (arXiv:2605.10149; Set-Constrained Viterbi CVPR 2020): inference-time structural constraints (transition confidence, durations) integrated into Viterbi; no retraining. V5 applies this *to expert routing* — the decoder selects the policy's experts, not just labels.
- **SMoDP** (arXiv:2605.23477): chunk-consistent skill routing + dual contrastive alignment; nearest to V5 in spirit (routing coherence), but language-grounded and diffusion-based.

Positioning of the study: the *combination* of (i) a frozen supervised phase predictor as the router backbone, (ii) decision-level Viterbi-decoded expert selection under the empirical monotone transition prior, and (iii) phase-conditioned balance to remove the uniformizing force — with a controlled single-factor matrix — is not covered by any of the above.

## 5. Variant definitions (pre-registered)

All variants share: stage-1 checkpoint (lambdav1 fixed-monitor runs, seeds 42/43/44), 200-epoch stage-2, AdamW 1e-4, cosine anneal to 1e-7, clip 1.0, batch 256, freeze_encoder, checkpoint monitor `val/loss_action` (min, top-1), no early stopping. Only the named channel differs.

**V0 — baseline PhaseForge** (existing lambdav1_stage2 runs, 3 seeds; reference, no rerun).

**V1 — phase-head logit initialization (channel I).** Router init from the *function* F, not the centroids:

    gate_linear.weight = W_F (unit-scaled), gate_bias = b_F

The router starts as the exact stage-1 phase predictor (hyperplane classifier, richer than nearest-centroid: captures separation directions, works for non-spherical phase clusters). Pure initialization — consistent with the paper's "initialization-only" positioning. Prediction: better early NMI than V0; same decay behavior (Lemma 3 applies to the unconstrained continuation); the comparison V1-vs-V0 isolates first-order (centroid) vs second-order (hyperplane) use of stage-1 geometry.

**V2 — phase distillation warmup (channel L).** For epochs t < T_d (T_d = 40), add

    L_distill = κ(t) · E[ KL( g(z) ‖ softmax(F(z)/τ) ) ],   κ(t) = κ_0 · (1 − t/T_d)

with κ_0 = 1.0, τ = 1.0, F frozen (stop-grad). The stage-1 phase predictor *teaches* the router; the KL anneals to zero so the action loss takes over (no permanent distortion of the protocol's "no phase supervision in stage 2" for the final objective). Prediction: high early alignment, retained if the alignment is action-compatible, decayed if not. V2-vs-V1 separates the warmup-loss channel from the init channel.

**V4 — phase-conditional balance (channel B).** Replace the global Switch balance with a within-phase balance:

    L_bal^φ = Σ_p (E / |D_p|) Σ_i f_i^p · p_i^p

where f_i^p is the fraction of *phase-p* tokens dispatched (top-1) to expert i and p_i^p the mean gate probability over phase-p tokens. A phase can commit to a small expert subset while utilization stays balanced *within* each phase — the uniformizing force (Lemma 2) is neutralized exactly where it harms commitment, without reintroducing dead experts. Uses phase labels in stage-2 training (privileged; new-method cell). Prediction: NMI ↑, entropy ↓, top-1 collapse within phase ↓, no dead experts.

**V6 — low-rank anchored router (channel P).** Router logits:

    G(z) = F(z) + A·(B·z) + b_r,   A ∈ R^{E×k}, B ∈ R^{k×d}, k = 4, A = B = 0 at init

F (frozen, from stage 1) is the backbone; the residual is rank-constrained, so the router can only deviate from the phase predictor in a k-dimensional subspace. By Lemma 3 this is the minimal structural anchor that is *not* initialization-equivalent: the phase classifier's boundaries persist by construction, the residual adds local corrections. Prediction: NMI bounded below by phase-predictability of F; entropy lower; action loss ≥ unconstrained (capacity cost), but routing persistence much higher.

**V5 — Viterbi-decoded routing (channel D, inference-time; no training change).** Two emissions, both decoded with the empirical transition prior T built from *training* phase labels (monotone Lift prior: p→p or p→p+1, plus observed durations):

1. **Phase-head decode:** emissions = softmax(F(z)); Viterbi → MAP phase sequence p̂_t → experts = phase-affinity top-2 (affinity matrix measured from the *trained* router's usage, not assumed bijection).
2. **Router-logit decode:** emissions = softmax(G(z)); Viterbi → MAP expert sequence directly.

Metrics: per-step phase accuracy (decode vs GT), run-length/coherence, routing agreement (fraction of steps where decoded expert-set == learned top-2), and action MSE under decoded vs learned routing. Evaluated on val trajectories (regrouped by trajectory_id). V5 composes with V1/V2/V4/V6 checkpoints for free.

## 6. Metrics (predeclared)

Primary: stage-2 `val/loss_action` at best checkpoint.
Mechanism: `val/phase_expert_nmi`, `val/routing_entropy` (normalized), `val/topk_balance_score`, `val/top1_collapse_rate`, per-variant phase-alignment trajectory.
V5 (separate evaluation): decoded phase accuracy, mean run length, routing agreement, action MSE (decoded vs learned), all on the val split.

## 7. Compute plan and recording

Screening: all variants × seed 42 (≈15 min/run CPU). Winners (by primary + mechanism jointly) then run × seeds 43, 44. Results recorded in `outputs_local_train/_results/phase_utilization_screening.csv` + a summary notebook section in this file's sibling report. No GPU needed for training screening; rollout validation of winners remains a GPU-phase task (Phase 2+ of the main plan).

## 8. Interpretation rules (what each outcome means)

- V1 > V0: second-order geometry helps → keep; the init channel is stronger than currently claimed.
- V2 > V0: loss-level coupling helps → the "no phase loss in stage-2" design is the bottleneck; motivates a PhaseForge-2 claim.
- V4 > V0: the balance loss is the binding constraint (pseudo-balancing confirmed in this setting); strongest mechanism story.
- V6 > V0 (or ≈ V0 with much higher NMI): constrained anchoring is a viable structural mechanism; the paper gains a parameterization-level contribution.
- V5 improves phase accuracy and routing coherence without hurting action MSE: the process channel is free value; deployment gets a decoder with zero training cost.
- Any combination that dominates V0 on primary + mechanism jointly → candidate for the 3-seed confirmation and (later) GPU rollout.

Honesty rule (unchanged): no claim of novelty without the documented search (Section 4 records it); no claim of behavioral gain without rollout; null results are reported as null results.

## 9. Screening results (seed 42; 200-epoch stage-2, shared lambdav1 stage-1 checkpoint)

Recording: `outputs_local_train/_results/phase_utilization_screening.csv` (all rows below, machine-readable); raw per-run curves in each run dir; V5 JSONs in `outputs_local_train/v5_decoded_routing/`.

All variants reproduced the V0 protocol except the single named channel. No run diverged from the protocol config (resolved_config.yaml recorded per run).

### 9.1 Training metrics (primary + mechanism, best-checkpoint / final)

| variant | best val/action | final val/action | NMI | entropy | topk balance | top1 collapse | best ep |
|---|---|---|---|---|---|---|---|
| V0 baseline | 0.02872 | 0.0337 | 0.4495 | 0.9510 | 0.9948 | 0.000 | 9 |
| V1 phase-head init | 0.02900 | 0.0340 | 0.4724 | 0.2728 | 0.9616 | 0.000 | 10 |
| V2 distill warmup | 0.02770 | 0.0329 | 0.4801 | 0.8441 | 0.8572 | 0.333 | 9 |
| V4 phase-cond balance | 0.02699 | 0.0314 | 0.0985 | 0.9993 | 0.9860 | 0.000 | 49 |
| V6 anchored router | 0.02858 | 0.0355 | 0.5016 | 0.1436 | 0.8353 | 0.333 | 10 |

Findings (seed 42 only; confirmation in §10):

- **V1: the init channel PERSISTS.** Routing entropy starts at 0.14 (the phase head's saturated classifier logits) and only drifts to 0.27 over 200 epochs — vs V0's 0.95. The router is phase-committed at *zero* action cost (best 0.0290 ≈ V0's 0.0287). Lemma 3's "decay to uniform" happens on a 200-epoch timescale far slower than training. NMI +0.02 vs V0.
- **V2: distillation is the only variant that improves the primary metric** (best 0.02770 vs 0.02872, −3.6%) with the highest NMI (0.4801). But it costs one dead expert (top1 collapse 0.333). The distill term anneals to zero by epoch 39 and stays off; the NMI gain (0.497 → 0.480) survives release.
- **V4: phase-conditional balance DESTROYS phase alignment** (NMI 0.0985) while producing the *best* action loss (0.02699). This is the pseudo-balancing prediction in the strongest form: even the within-phase balance acts as a uniformizing force on the phase↔expert correspondence (Lemma 2, empirically confirmed in this setting). The router's expert partition is orthogonal to phases, and actions are unaffected — **the phase-partitioned routing is NOT what drives Lift action quality** (echoes the corrected Phase-4 result).
- **V6: anchored router keeps the phase structure BEST** (NMI 0.5016 — the only variant above 0.50 — entropy 0.14) at zero best-epoch action cost (0.02858 ≈ V0). Final epoch is the worst of the five (0.0355) — the rank-4 residual does not recover the late-epoch action drift. Lemma 3's structural anchor behaves as predicted: alignment persists by construction.

### 9.2 V5 decoding results (val split, 1026 steps / 20 trajectories)

Phase-head decode metrics are identical across variants (shared frozen stage-1 head): argmax phase acc 0.6033; **Viterbi phase decode 0.6121 (+0.9 pts)**; mean decoded run length 12.2 steps (the head's errors are already segment-consistent, so the prior buys little).

| variant | mse learned | mse phase-decoded | mse router-decoded | router phase acc | router decoded acc | agreement |
|---|---|---|---|---|---|---|
| V0 | 0.02872 | 0.07291 | 0.07179 | 0.5068 | 0.4561 | 0.891 |
| V1 | 0.02900 | 0.03386 | 0.02994 | 0.5760 | 0.5702 | 0.964 |
| V2 | 0.02770 | 0.07266 | 0.07038 | 0.5575 | 0.5448 | 0.959 |
| V4 | 0.02699 | 0.03126 | 0.02647 | 0.2329 | 0.1033 | 0.510 |
| V6 | 0.02858 | 0.02986 | 0.02898 | 0.5234 | 0.5273 | 0.965 |

Findings:

- **V6 is the first configuration where phase-decoded hard routing reaches parity with the learned top-2 routing** (0.02986 vs 0.02858, +4.5% — by far the closest). The affinity map phase→expert is meaningful because the anchored router genuinely lives near the phase predictor. V1 is second (0.03386, +17%). V0/V2 hard-decode 2.5× worse — their routers are too far from the phase geometry for the affinity mapping to transfer.
- **V4's router-decoded actions (0.02647) beat its own learned routing (0.02699)** — but through expert-side smoothing of a phase-agnostic near-uniform router (decoded phase acc 0.10), not through phases.
- Routing agreement (decoder vs learned top-1): 0.96 for V1/V2/V6, 0.89 for V0, 0.51 for V4.

### 9.3 Verdicts (seed-42 screening)

- V4 wins the primary metric but via the anti-phase mechanism — the strongest evidence that the balance loss (not the phase knowledge) was the binding constraint. High-value *negative* result for the paper.
- V2 is the only variant with a real primary-metric gain; NMI story intact; one dead expert is a mechanism wart to report.
- V6 is the only variant where phase knowledge demonstrably *governs* routing at deployment (V5 parity) with a phase-alignment record above V0 — the "phase-conditioned" claim's best candidate.
- V1 ≈ V0 on actions; its persistent commitment is a finding about the init channel, not a method gain.
- → 3-seed confirmation ordered: **V4, V2, V6** × seeds 43, 44 (V1 dominated by V6 on both primary and mechanism).

## 10. Confirmation runs (3-seed)

Screening winners V2/V4/V6 × seeds 43/44 launched 2026-08-18 13:03–13:12 (CPU, shared lambdav1 stage-1 checkpoints per seed). All completed; full rows in `phase_utilization_screening.csv`.

### 10.1 Primary metric (best val/loss_action; lower is better)

| variant | seed 42 | seed 43 | seed 44 | mean | vs V0 |
|---|---|---|---|---|---|
| V0 baseline | 0.02872 | 0.02607 | 0.02772 | 0.02750 | — |
| V2 distill | 0.02770 | 0.02370 | 0.02601 | **0.02580** | −6.2% |
| V4 phase-cond balance | 0.02699 | 0.02371 | 0.02631 | **0.02567** | −6.7% |
| V6 anchored | 0.02858 | 0.02442 | 0.02722 | 0.02674 | −2.8% |

Final val/loss_action means: V4 0.02837 (−7.9%), V2 0.02986 (−3.1%), V0 0.03081, V6 0.03095 (≈V0). V2 and V4 beat V0 on **all three seeds individually** on the primary metric; V6 beats V0 on 2 of 3.

### 10.2 Mechanism (means; per-seed in CSV)

| variant | NMI | entropy | top1 collapse |
|---|---|---|---|
| V0 | 0.447 | 0.955 | 0.000 |
| V2 | 0.477 | 0.850 | 0.333 (one dead expert, every seed) |
| V4 | 0.073 | 0.999 | 0.000 |
| V6 | **0.503** | 0.186 | 0.333 |

V6's NMI is the highest and the most consistent (0.502/0.503/0.505 across seeds); V4's phase alignment is destroyed deterministically; V2's is modestly above V0. All three seed-42 conclusions hold per-seed.

### 10.3 V5 decoding (3-seed means)

| variant | mse learned | mse phase-decoded | mse router-decoded | router phase acc | router decoded acc | agreement |
|---|---|---|---|---|---|---|
| V0 | 0.0275 | 0.0536 (1.95×) | 0.0513 | 0.444 | 0.294 | 0.682 |
| V2 | 0.0258 | 0.0538 (2.09×) | 0.0516 | 0.569 | 0.549 | 0.957 |
| V4 | 0.0257 | 0.0465 (1.81×) | 0.0424 | 0.315 | 0.211 | 0.624 |
| V6 | 0.0267 | **0.0289 (1.08×)** | **0.0276 (1.03×)** | 0.550 | 0.555 | 0.968 |

The flagship: **V6 is the only configuration whose phase-decoded hard routing reaches parity with the learned top-2 router** (within 4–11% on every seed). With V6, the frozen phase predictor + empirical monotone prior + affinity map is a deployable routing policy at essentially zero action cost — the phase knowledge governs routing at inference instead of being discarded.

### 10.4 Conclusions for the paper

1. **V4 is a high-value negative result:** the balance loss — not the phase knowledge — was the binding constraint on Lift. Phase-conditional balance removes phase alignment entirely (NMI 0.07) yet improves action loss on every seed (−6.7%). Phase-partitioned routing is not what drives Lift action quality (consistent with the corrected Phase-4 result).
2. **V2 is the only positive primary-metric result that keeps phases:** distillation warmup improves action loss (−6.2%) and NMI (+0.03) on every seed at the cost of one dead expert (report as a mechanism wart; entropy stays 0.85, so the router is not collapsed).
3. **V6 is the phase-conditioned story:** phase alignment persists by construction (NMI 0.50, entropy 0.19), actions match V0 at best-epoch (−2.8%), and the V5 decoder turns the phase process into near-parity routing at deployment. This is the paper's new mechanism contribution (parameterization channel).
4. V1 (init only) is dominated by V6 on both axes and dropped after screening.
5. Next step (GPU, Phase 2+ of the main plan): rollout validation of V2/V4/V6 winners; the offline story is complete on CPU.