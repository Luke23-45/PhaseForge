Here's the verified picture of the prior-art landscape (all papers found online, real titles/venues), then the detailed comparison against our implementation.

## The papers, verified

| Paper | Venue/ID | Mechanism | Key result |
|---|---|---|---|
| **MEAT** — "MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts" (Mazza et al.) | arXiv 2601.21971, Jan 2026 | Supervised MoE added to **ACT**; one expert per task phase; gating network **supervised with phase labels** (CE loss); ordered surgical phases (grasp → retract) | ACT 50%→85% end-to-end; 60%→85% grasping; data-efficient (<150 demos) |
| **MoE-ACT** (Guo et al.) — the one our proposal cites | arXiv 2603.15265, Mar 2026 | MoE FFN in ACT **encoder**, language-conditioned, FiLM decoding, multi-scale cross-attn; bimanual RoboTwin 2.0 | +33% over vanilla ACT (22%→55%) |
| **SMP** — "Abstracting Robot Manipulation Skills via MoE Diffusion Policies" (Hao, Zhai, Liu, Soh) | **ICLR 2026**, arXiv 2601.21251 | Diffusion MoE with **state-adaptive orthonormal skill basis** (Stiefel/QR) + **sticky Dirichlet-Markov gating**; state-only router distilled at inference; sparse top-k activation | 0.54 avg on RoboTwin-2 vs 0.48 next best; ~30% active params |
| **LAR-MoE** — "Latent-Aligned Routing for MoE in Robotic..." | arXiv 2603.08476 | Two-stage: **unsupervised** student–teacher co-training of obs+future-action latent, then routing regularized to follow that latent; **no phase labels**; ACT-style experts; soft routing + entropy reg | Structured specialization without annotations; zero-shot ex vivo |
| **AdaMoE** (Shen et al.) | arXiv 2510.14300, Oct 2025 | MoE in π0's action expert; **decouples expert selection (router) from weighting (scale adapter)** — directly attacks the load-balance-vs-specialization conflict | +1.8% LIBERO, +9.3% RoboTwin, +21.5% real world |
| **Memory-Aware Routing (MAR)** (Hou et al.) | Findings ACL 2026 | Names the **pseudo-balancing** failure (same input routed randomly to satisfy balance → knowledge overlap, no specialization); per-expert memory buffers guide consistent routing | +35% specialization (KED), 2–25% accuracy, half the experts |
| **Move-Then-Operate (MTO)** (Lei, Gu, Tang, Chen, Wang — *not* Xu) | ICML 2026, arXiv 2604.23620 | Dual-expert VLA (move/operate), **chunk-level phase router**, teacher-forcing with GT labels, hard routing, MLLM-generated phase labels | +24.1% over π0 on RoboTwin2 (68.9%) |
| **PAMAE** | arXiv 2606.27144 | Phase-aware MoE action module for flow-matching VLA; phase-aware router + lightweight phase head + routing alignment; **two-stage training** (warmup experts, then supervised routing, aux losses annealed 30%) | +9.2% over VLA baselines |
| **CoRDE** | arXiv 2606.21935 | Frozen concept encoder guides variational expert responsibility via soft mapping; LoRA experts on frozen backbone; reduces routing collapse | LIBERO + D3IL gains |

## Where we overlap with each (the professor's point, confirmed)

The general claim — **"phase/skill structure helps MoE specialize in manipulation"** — is already made by MEAT, SMP, MTO, PAMAE, and implicitly LAR-MoE/CoRDE. On the specific dimensions:

- **Phase-supervised routing during training:** MEAT (CE on gate), MTO (teacher-forcing router), PAMAE (routing-alignment loss). All three are "our idea minus the bootstrap."
- **Two-stage training:** PAMAE (warmup→supervised routing), LAR-MoE (unsupervised pretrain→latent-aligned routing), CoRDE (offline distillation→online). Two-stage is now a standard recipe.
- **State-only router at inference:** SMP (distills state-only router), CoRDE (online state-only execution). So "router sees only state" is *not* a differentiator either.
- **Balance-vs-specialization:** AdaMoE and MAR both exist *because* of the exact dilemma we hit (dry run: balance ≥ 0.98, NMI = 0.0 — textbook pseudo-balancing). Our C3 issue is a known, named problem.
- **Temporal routing stability:** SMP (sticky gates), TRACT (monotone chunk routing) — our `time_to_stable_routing` metric is our answer to a problem they solve in the architecture.

## What remains genuinely ours (candidate novelty)

1. **Phase-centroid router bootstrap on a frozen stage-1 latent.** Nobody initializes router *weights* from phase centroids computed in a pretrained latent. The closest are PAMAE/MTO/MEAT, which supervise routing gradients directly; LAR-MoE, which aligns to *unsupervised* latents; SMP, which distills an amortized posterior. Centroid-bootstrap-as-initialization is unclaimed — this is the sharpest mechanism claim.
2. **The oracle-routing upper bound + NMI phase-expert alignment diagnostics.** SMP measures phase consistency via gate patterns, PAMAE via dominance purity — but the *oracle* (GT-label routing as an upper bound, with NMI=1.0 as the instrumented sanity signature) is our evaluation-framework contribution.
3. **The controlled state-only factorial at 0.6M params.** Everyone else is vision/VLA-scale. Nobody runs the *training-strategy question* as a controlled study (bc/scratch/warmstart/phase-bootstrap/oracle + the planned 2×2). That framing is defensible — but only if we add the 2×2 (C1), or the comparison is still uninterpretable.
4. **Pseudo-balancing demonstrated in small-scale continuous control.** MAR found it at LLM scale; our dry run (NMI=0 with balance≈1.0 at 0.6M params) would be the small-scale evidence — a genuine transferable finding if the diagnosis holds after the eval fixes.

## Corrections to our existing citations (found during verification)

- Our experiment report cites "Move-Then-Operate (Xu, ICML 2026)" — actual authors are **Lei Lei, Jie Gu, Chu Tang, Jingmin Chen, Ruiqi Wang**.
- Our proposal's "MoE-ACT (Guo et al., 2026)" is **real** (bimanual paper, arXiv 2603.15265) — but there's a **second, different** MoE-ACT (surgical, Mazza et al.) which is the "MEAT" the professor means. Both must be cited and disambiguated.
- MAR is verified: Hou et al., Findings ACL 2026, pages 17320–17337.
- LAR-MoE authors could not be confirmed from the PDF snippet ("Rodriguez et al." unverified) — flag for verification.
- MoE-DP citation (Cheng, 2025) unverified; the SMP paper's closest baseline is the MoE diffusion policy of Wang et al., 2024 — likely the correct reference.

## What this changes in the paper positioning

The sharpened contribution (per professor D1) should read something like:

> "We isolate the *training strategy* variable in phase-structured MoE: is bootstrapping the router from phase centroids in a frozen, phase-supervised latent a better initialization than supervised routing (MEAT/PAMAE/MTO), unsupervised latent alignment (LAR-MoE), or no structure (scratch/warmstart)? — measured with an oracle-routing upper bound and NMI-based phase-expert diagnostics, in a controlled state-only setting."

That claim is narrow enough to be novel and directly testable with the planned 2×2. The general finding, pseudo-balancing at small scale, is a secondary contribution that also needs the balance-vs-NMI sweep (C3) to be credible.

