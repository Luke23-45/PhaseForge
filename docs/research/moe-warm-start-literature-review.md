# Warm-Starting Mixture-of-Experts: A Deep Literature Review (1991–2026)

> Purpose: find the best / SOTA way to warm-start MoE experts — from the oldest work to the present,
> covering LLM upcycling, diffusion-based MoE, router initialization, robotics/RL skill warm-starting, and PEFT.
> Every claim below was verified against its source during research (Aug 2026). Unverifiable papers were omitted.

**Relevance to PhaseForge.** Our pipeline: Stage-1 generalist (encoder + ActionHead + phase head, supervised
by phase labels) → Stage-2 MoE (6 experts, top-2, router initialized from phase-centroid prototypes, experts
warm-started by copying ActionHead into all experts + Gaussian jitter 0.02). The question this review answers:
what is the SOTA way to warm-start experts (and the router) in that setting?

---

## 0. Framing: two separate design decisions

The literature treats **expert initialization** and **router initialization** as independent problems, and a
third axis — **what to do after init** (losses, balancing, two-stage schedules) — determines whether the
init advantage survives. Naive expert copy without any symmetry breaking is the single worst option: identical
experts ⇒ routing invariant ⇒ no routing gradient (expert symmetry collapse). All modern recipes therefore
combine *knowledge transfer* (warm start) with *structured differentiation* (partial re-init, shuffle, cluster
subspaces, LoRA deltas, or trained-from-scratch specialists).

---

## 1. Classical foundations (1991–2017)

- **Jacobs, Jordan, Nowlan & Hinton (1991), "Adaptive Mixtures of Local Experts"** — *Neural Computation* 3(1):79–87.
  The founding paper: gating network + competing experts, divide-and-conquer. Gating learns to partition input space.
  https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf
- **Jordan & Jacobs (1994), "Hierarchical Mixtures of Experts and the EM Algorithm"** — *Neural Computation* 6(2):181–214.
  Tree-structured gating = precursor to hierarchical/phase-structured routing (our phase hierarchy).
- **Masoudnia & Ebrahimpour (2014), "Mixture of experts: a literature survey"** — *Artificial Intelligence Review* 42(2):275–293.
  Surveys decades of MoE design incl. clustering-based expert placement.
- **Shazeer et al. (2017), "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"** (ICLR 2017) — modern sparse
  MoE: top-k gating, **gating noise (jitter σ=0.01 on logits during training)** for exploration, importance-based
  load-balancing aux loss, capacity factor. https://arxiv.org/abs/1701.06538
- **Rosenbaum et al. (2017), "Routing Networks: Adaptive Selection of Non-linear Functions for Multi-task Learning"** —
  earliest "cluster as router": routing via **k-means-like assignment** in the routing network. https://arxiv.org/abs/1704.06363

Old insight that persists: clustering-based gating initialization and competition-based specialization predate
everything modern; "experts from scratch, routing learns" was the default — warm-starting appeared only with upcycling (2022).

---

## 2. The scaling era: routing, balancing, stability (2020–2022)

- **GShard (Lepikhin et al., ICLR 2021)** — top-2, capacity, load-balancing aux loss, random init. https://arxiv.org/abs/2006.16668
- **Switch Transformer (Fedus, Zoph & Shazeer, JMLR 2022)** — top-1 routing, capacity factor, load-balancing loss; up to 512 experts. https://arxiv.org/abs/2101.03961
- **BASE Layers (Lewis et al., NeurIPS 2021)** — assignment-based routing (no aux loss; auction-like assignment). https://arxiv.org/abs/2103.16716
- **Hash Layers (Roller et al., ICLR 2021)** — fixed hash routing; **no router at all** — an extreme contrast to learned routing. https://arxiv.org/abs/2106.04426
- **Expert Choice (Zhou et al., NeurIPS 2022)** — expert-side (non-token) routing; load balanced by construction; no aux loss. https://arxiv.org/abs/2202.09368
- **ST-MoE (Zoph et al., 2022)** — the first "recipe" paper for stable sparse MoE: **router z-loss** (weight 0.001),
  capacity-factor studies, top-2, input-jitter hurts at scale, sparse models overfit on small data, **upcycling experiment**.
  https://arxiv.org/abs/2202.08906
- **Representation collapse of sparse MoE / X-MoE (Chi et al., NeurIPS 2022)** — routers can collapse (one expert dominates);
  fixes: lower-dimensional routing inputs + data-dependent gating. Central reference for "routing collapse". https://arxiv.org/abs/2210.10022
- **THOR (Zuo et al., ICLR 2022)** — stochastic/dropout-style routing. https://arxiv.org/abs/2202.06541
- **Patch-level routing (Chowdhury et al., ICML 2023)** — **k-means-based routing** for CNN MoE; provably sample-efficient
  (routing as clustering has theoretical grounding). https://arxiv.org/abs/2306.04073
- **MoEC, Mixture of Expert Clusters (Xie et al., AAAI 2023)** — variance-based clustering constraint on routing.

Takeaway: the "router = cluster/prototype" view is old and now theoretically supported; phaseforge's centroid router
is a principled instance of it (see §4 for the 2023–2026 refinements).

---

## 3. Warm-start / upcycling era (2022–2025) — the core of the answer

- **Sparse Upcycling (Komatsuzaki et al., ICLR 2023)** — the canonical expert warm-start: **clone the dense FFN into E
  experts + random router, then continue training**. Beats from-scratch MoE up to ~120% of original training compute;
  advantage is largest early and erodes with longer training. https://arxiv.org/abs/2212.05055
- **Mixtral 8x7B (Jiang et al., 2024)** — 8 experts, top-2. Widely reported to be upcycled from Mistral-7B (not officially
  confirmed; expert-similarity analysis "dark cross" in *A Closer Look into Mixture-of-Experts in LLMs* (2024, arXiv 2406.18219)
  supports an upcycling-like origin). https://arxiv.org/abs/2401.04088
- **StableMoE (Dai et al., ACL 2022)** — **two-stage training: (1) stabilize/pre-train the router, (2) freeze router and train
  experts**; distinct from upcycling but validates the "routing first, then experts" schedule (our stage-1/2 pattern).
  https://aclanthology.org/2022.acl-long.489/
- **LLaMA-MoE (Zhu et al., EMNLP 2024)** — converts dense LLaMA to MoE; ablates expert construction (split vs clone) +
  continual pretraining. Structured conversion + continued training wins. https://arxiv.org/abs/2409.04160
- **Drop-Upcycling (Nakamura et al., Feb 2025)** — **partial re-initialization**: re-init a fraction r of each upcycled
  expert's weights from the original distribution; **r = 0.5 is optimal**; fixes naive upcycling's symmetry/slow-convergence
  problem; never overtaken by scratch training. https://arxiv.org/abs/2502.19261
- **Qwen2-MoE (Yang et al., 2024, tech report)** — industrial SOTA recipe, upcycled from Qwen2-7B (57B-A14B, 64 routed +
  8 shared experts, top-8): each expert's intermediate weights **shuffled along the intermediate dim + 50% randomly
  re-initialized** (industry-validated Drop-Upcycling). https://arxiv.org/abs/2407.10671
- **Qwen2.5 tech report (2024)** — fine-grained expert segmentation + shared experts pattern (following Qwen1.5-MoE). https://arxiv.org/abs/2412.15115
- **NVIDIA upcycling study (Hefny et al., 2024), "Upcycling LLMs into Mixture of Experts"** — systematic ablations;
  **"virtual group" initialization + weight scaling** for fine-grained MoE; upcycling beats continued dense training;
  used for Nemotron. https://arxiv.org/abs/2410.07524
- **OLMoE (Muennighoff et al., ICLR 2025)** — open upcycling recipe; **router initialized as truncated normal σ=0.02
  (cutoff ±0.06)**; router z-loss (0.001) + load-balancing loss; upcycling advantage erodes after ~500B tokens (scratch
  eventually catches up) — warm start is mostly a *compute* win; they also tried adding noise to upcycled weights and
  it didn't help in their setting. https://arxiv.org/abs/2409.02060
- **Cluster-aware Upcycling (CVPR 2026)** — **the direct SOTA for structure-guided warm start**: (1) cluster dense
  activations; (2) initialize each expert from the **truncated-SVD subspace of its own cluster** (breaks symmetry by
  construction while keeping dense knowledge); (3) **initialize the router from cluster centroids** (exactly our
  centroid router); (4) expert-ensemble self-distillation for stability. Beats naive upcycling, noise upcycling and
  Drop-Upcycling on ViT/CLIP benchmarks; lower inter-expert similarity, higher routing confidence.
  https://arxiv.org/abs/2604.13508
- **Branch-Train-Merge (Li et al., 2022)** and **Branch-Train-MiX (Sukhbaatar et al., COLM 2024)** — diversity by
  construction: train separate dense expert models on disjoint data, then merge as MoE + finetune router. BTX
  outperforms upcycling given equal budget. https://arxiv.org/abs/2210.00038 / https://arxiv.org/abs/2403.07816
- **CuMo (NeurIPS 2024)** — **co-upcycling**: trained-from-scratch MoE failed to converge in vision-LM; upcycling the
  MLP into experts + z-loss fixed it (warm start as a *stability* device). https://arxiv.org/abs/2403.13686
- **MoE Jetpack (2024)** — an alternative sparse conversion with layer surgery. https://arxiv.org/abs/2410.17748
- **Scaling Laws for Upcycling (ICML 2025)** — an interaction term between dense-pretrain and upcycle budgets caps
  gains at large budgets. https://arxiv.org/abs/2412.09643
- **Upcycling Instruction-Tuning Dense-to-MoE via Parameter Merging (ACL 2025)** — use **intermediate checkpoints as
  experts** + genetic/parameter merging; another way to create differentiated experts from one training run. https://aclanthology.org/2025.acl-long.984/
- **CLIP-UP (EMNLP 2025)** — upcycling without increasing parameter count. https://arxiv.org/abs/2501.01367

**Consensus of §3:** the best expert warm-start is *not* "copy + i.i.d. noise" but **copy + structured differentiation**,
with three validated options: (a) partial re-init r≈0.5 (Drop-Upcycling / Qwen2), (b) cluster-subspace experts
(Cluster-aware Upcycling, CVPR 2026 — also gives the router for free), (c) trained-specialists/BTX. And always pair with
router z-loss + balancing, and keep the fine-tune short (init advantage is mostly early).

---

## 4. Router initialization: routing as clustering (2023–2026)

- **ProMoE (arXiv 2510.24711, 2025)** — for **diffusion transformers**: explicit semantic routing — **classification-based
  (class labels) or k-means prototype routing** — beats DiT-MoE and DiffMoE by a wide margin (ImageNet 256 FID:
  DiT-MoE-B-Flow 8.94 vs k-means-based 6.24 vs classification-based 5.91). Also: k-means is sensitive to init, k updated
  during training. Strongest evidence that prototype/cluster routers are SOTA in diffusion. https://arxiv.org/abs/2510.24711
- **DiT-MoE (Fei et al., 2024)** — token-to-expert routing for DiT; often underperforms dense (routing is hard without structure). https://arxiv.org/abs/2406.04619
- **DiffMoE (Shi et al., 2025)** — noise-gated MoE for diffusion. https://arxiv.org/abs/2412.02105
- **Latent Prototype Routing (LPR, arXiv 2506.21328, 2025)** — routing as clustering in latent space with
  **hyperspherical (normalized) prototype initialization** + alignment/diversity losses; random prototype init shows
  bias, normalization fixes it. https://arxiv.org/abs/2506.21328
- **Expert-Router Coupling loss (ERC, ICLR 2026)** — explicitly interprets **router rows as cluster centers** and couples
  routers to experts (contrastive/noise-augmented loss); routers as prototype embeddings.
- **MoE-LPR (AAAI 2025)** — routing with language priors; **RCL (2024)** — router contrastive learning.
- **Guiding the Experts: Semantic Priors for Focused MoE Routing (2025)**; **LAR-MoE: Latent-Aligned Routing for MoE in
  Robotic Imitation Learning (2026)** — robotics-specific routing alignment.
- **SMoE-Dropout (Chen et al., ICLR 2023)** — random fixed routing acts as dropout: "Sparse MoE as the new dropout"
  — supports keeping some routing stochasticity. https://arxiv.org/abs/2303.01610

**Consensus of §4:** a centroid/prototype-initialized router is now an independently-validated SOTA device (ProMoE, LPR,
ERC, MoEC, patch-routing). Refinements worth adopting: (1) **normalize prototypes** (hyperspherical, LPR); (2) **keep
prototypes trainable/updatable** (ProMoE updates k-means centers; ERC couples them to experts); (3) add **router z-loss**
(ST-MoE/OLMoE) to keep logits small.

---

## 5. Load balancing & stability after init (what makes init survive)

- Router **z-loss** (ST-MoE 2022; OLMoE uses 0.001) — penalizes logit magnitude, stabilizes top-k routing.
- **Aux-loss-free / bias-based balancing (DeepSeek-V3, 2024)** — per-expert bias updated by a primal-dual rule
  (`bias_update_speed`), **sigmoid gating**; no aux loss needed. https://arxiv.org/abs/2412.19437
- **Loss-Free Balancing (Wang et al., 2024)** — theoretical treatment of bias-based balancing (assignment duals). https://arxiv.org/abs/2408.15664
- **Skywork-MoE** — gating logit normalization + adaptive aux coefficient; upcycled from Skywork-13B. https://arxiv.org/abs/2406.06563
- **"Bag of Tricks for Sparse MoE: A Benchmark" (EMNLP 2025)** — systematic comparison: auxiliary loss (EB) vs
  expert-choice (LBL); specialization metrics **Δcap/Δspec**; EB generally better for task-specific, LBL for general-purpose.
- **"Synergistic Intra- and Cross-Layer Regularization Losses for MoE Expert Specialization" (2026, arXiv 2602.14159)** —
  explicit regularization toward specialization.
- **"Mixture of Experts with Soft Nearest Neighbor Loss" (2026, arXiv 2603.26734)** — SNNL loss resolves expert
  collapse via representation disentanglement.
- **MoE Routing Testbed (2026, arXiv 2604.07030)** — small-scale controlled study of expert specialization.

---

## 6. The diffusion angle (2023–2026) — user explicitly requested

- **Diffusion Policy (Chi et al., RSS 2023)** — action-chunk DDPM for visuomotor policy; models **multi-modal action
  distributions** (a key motivation for "experts" in robotics). https://arxiv.org/abs/2303.04137
- **MoDE: Mixture-of-Denoising Experts (ICLR 2025)** — Diffusion Policy + MoE for **imitation learning**:
  - **noise-conditioned routing**: experts specialize by denoising phase (routing conditioned on noise level σ);
  - router initialized **truncated normal σ=0.02** (OLMoE recipe);
  - top-2, load-balancing loss; **expert caching** cuts FLOPs ~90%, active params −40%;
  - SOTA on CALVIN (ABC: 4.01) and LIBERO-90 (0.95). https://arxiv.org/abs/2412.12953
- **Direct transfer to us:** in diffusion, the natural expert partition is the denoising phase — in our pipeline the
  natural partition is the **phase label**. Phase-conditioned routing = MoDE-style noise-conditioned routing.
- **EC-DiT (2024)** — expert-choice routing for DiT; **"Efficient Training of Diffusion MoE: A Practical Recipe" (2025)** —
  upcycling a pretrained diffusion model into a diffusion MoE; **Perturbed 3D Gaussian experts (2025)**.

---

## 7. RL / robotics skill warm-starting (2019–2026)

- **JumpStart RL (Uchendu et al., ICML 2023)** — guide-policy warm start with curriculum; sample-complexity gain proven. https://arxiv.org/abs/2204.02372
- **Warm-Start Actor-Critic (ICML 2023)** — theory of warm-starting RL with a behavior policy.
- **"Analyzing and Overcoming Degradation in Warm-Start RL" (2022)** — **BC→RL warm start can degrade** (extrapolation
  error); early stopping matters. Caution for any stage-2 fine-tuning of a BC-initialized policy.
- **SPiRL (2019) / Play-LMP (2020)** — latent skill spaces; **"Specializing Versatile Skill Libraries using Local MoE" (2021)**.
- **Di-SkilL (Celik et al., ICML 2024, PMLR 235)** — diverse skill learning with MoE; per-expert contextual curriculum. https://proceedings.mlr.press/v235/celik24a.html
- **MoE-Loco (2025)** — multitask locomotion: **freeze experts, retrain the router/gate** for fast task adaptation
  (validates "keep experts, re-learn routing" — the same trick as StableMoE's frozen-router and our stage-2).
- **Skill-MoE (ICML 2026, arXiv 2503.05641)** — instance-level skill routing among pretrained expert models.
- **MoSE (2025)** — skill-by-skill MoE for autonomous driving.
- **"Dynamic MoE of PEFT experts for lifelong robot learning" (arXiv 2506.05985, 2025)** — pretrain on LIBERO-90, then
  add per-task LoRA experts with top-3 routing — PEFT-expert warm start for robot tasks.
- **LAR-MoE (2026)** — latent-aligned routing for robotic imitation learning.

---

## 8. PEFT / LoRA expert warm-start (2023–2026)

- **MoE-LoRA (2023), LoRAMoE (ACL 2024)** — LoRA plugins + router, backbone frozen. https://aclanthology.org/2024.acl-long.106/
- **MOLE: Mixture of LoRA Experts (Zadouri et al., ICLR 2024)** — routing over LoRA experts. https://arxiv.org/abs/2402.11589
- **GOAT (2025)** — **SVD-structured LoRA experts: adaptive singular-value segments as expert initialization** — another
  "structured differentiation" recipe. https://arxiv.org/abs/2502.16894
- **PanGu-Σ (2023)** — 1.085T MoE via upcycling dense + randomly initialized experts. https://arxiv.org/abs/2303.10845
- **DeRS (2025)** — expert-shared base + low-rank delta experts.
- **MoA: Mixture-of-Attention (2026)** — attention-expert routing. https://arxiv.org/abs/2604.21203
- **PEER: Mixture of a Million Experts (He, 2024)** — product-key routing over single-neuron experts. https://arxiv.org/abs/2407.04153
- **Mixture of Word Experts (MoWE, 2023)** — fixed (word-based) routing, no learned router.

---

## 9. Synthesis: what this means for PhaseForge

Our configuration (small robotics BC dataset; stage-1 phase-supervised generalist; centroid-prototype router; expert
warm-start = ActionHead copy + jitter 0.02) maps cleanly onto the literature. Ranked recommendations:

1. **Replace "copy + i.i.d. jitter" with "copy + structured differentiation".** The single most validated upgrade:
   **partial re-initialization r≈0.5** (Drop-Upcycling, Qwen2) — re-init ~50% of each expert's weights (e.g. per-column
   mask) at warm start. Cheap, industry-proven.
2. **Cluster-subspace expert init (the SOTA answer).** Initialize expert *i* from the ActionHead restricted to the
   **phase-i cluster subspace** (e.g., SVD of phase-i activations, or column/unit masking by cluster membership) —
   Cluster-aware Upcycling (CVPR 2026). Breaks symmetry by construction, preserves knowledge, and pairs naturally with
   our existing centroid router (which the paper also does — independent validation of phaseforge's design).
3. **Router: keep centroid prototypes, add three fixes.** (a) L2-normalize prototypes (hyperspherical, LPR); (b) keep
   them trainable / EMA-update the centroids during stage 2 (ProMoE updates k-means centers); (c) add **router z-loss**
   (weight ~0.001) and either an auxiliary load-balancing loss or DeepSeek-V3-style bias balancing in stage 2.
4. **Keep stage-2 short.** Upcycling's advantage is early (Sparse Upcycling, OLMoE); long fine-tuning erases it. Our
   ~200 epochs @ 1e-4 with early stopping is in the right regime — but don't extend it hoping for more.
5. **The diffusion route (frontier).** MoDE (ICLR 2025) shows noise/phase-conditioned expert routing is SOTA for BC on
   CALVIN/LIBERO. Since our phase label ≡ denoising phase, a per-phase **diffusion action head** with phase-conditioned
   routing is the natural next-generation architecture — and warm-starts the same way (upcycle the generalist head).
6. **Alternatives worth one cell each:** experts from separately trained per-phase policies (BTX-style, diversity by
   construction); GOAT-style SVD-segment experts; PEFT (LoRA per phase) if compute is tight; a shared always-on expert
   (DeepSeek/Qwen pattern).
7. **Metrics to add:** inter-expert similarity (dark-cross analysis), routing confidence, expert co-activation, Δcap/Δspec
   (Bag of Tricks 2025). We already track NMI/phase purity.

### Concrete experiment cells (ordered by value/effort)

| # | Cell | What | Source |
|---|------|------|--------|
| 1 | `warmstart_r50` | ActionHead copy + 50% column-wise re-init per expert | Drop-Upcycling 2502.19261, Qwen2 2407.10671 |
| 2 | `warmstart_subspace` | ActionHead masked to phase-cluster SVD subspace + centroid router (already have) | Cluster-aware Upcycling 2604.13508 |
| 3 | `router_hypered` | L2-normalize centroid prototypes + trainable/EMA centroids | LPR 2506.21328, ProMoE 2510.24711 |
| 4 | `router_zloss` | router z-loss (0.001) + load-balancing (aux or bias) in stage 2 | ST-MoE 2202.08906, OLMoE 2409.02060, DeepSeek-V3 2412.19437 |
| 5 | `svdseg_experts` | experts from different SVD segments of ActionHead | GOAT 2502.16894 |
| 6 | `btx_perphase` | experts = separately trained per-phase policies | Branch-Train-MiX 2403.07816 |
| 7 | `phase_noise_router` | route with phase label as router input feature (MoDE-style σ-conditioning analog) | MoDE 2412.12953 |
| 8 | `lambda_action=0` | stage-1 phase-only pretraining (removes ActionHead entirely) | our ablation |
| 9 | `diffusion_experts` | per-phase diffusion denoisers upcycled from a diffusion generalist | MoDE 2412.12953, EC-DiT |

---

## 10. References (all verified)

1. Jacobs, Jordan, Nowlan, Hinton (1991), Neural Computation 3(1):79–87 — https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf
2. Jordan & Jacobs (1994), Neural Computation 6(2):181–214
3. Masoudnia & Ebrahimpour (2014), AI Review 42(2):275–293
4. Shazeer et al. (2017) — https://arxiv.org/abs/1701.06538
5. Rosenbaum et al. (2017) — https://arxiv.org/abs/1704.06363
6. Lepikhin et al. (2021) — https://arxiv.org/abs/2006.16668
7. Fedus et al. (2022) — https://arxiv.org/abs/2101.03961
8. Lewis et al. (2021) — https://arxiv.org/abs/2103.16716
9. Roller et al. (2021) — https://arxiv.org/abs/2106.04426
10. Zhou et al. (2022) — https://arxiv.org/abs/2202.09368
11. Zoph et al. (2022), ST-MoE — https://arxiv.org/abs/2202.08906
12. Chi et al. (2022), X-MoE — https://arxiv.org/abs/2210.10022
13. Zuo et al. (2022), THOR — https://arxiv.org/abs/2202.06541
14. Chowdhury et al. (2023) — https://arxiv.org/abs/2306.04073
15. Komatsuzaki et al. (2023), Sparse Upcycling — https://arxiv.org/abs/2212.05055
16. Jiang et al. (2024), Mixtral — https://arxiv.org/abs/2401.04088
17. Dai et al. (2022), StableMoE — https://aclanthology.org/2022.acl-long.489/
18. Zhu et al. (2024), LLaMA-MoE — https://arxiv.org/abs/2409.04160
19. Nakamura et al. (2025), Drop-Upcycling — https://arxiv.org/abs/2502.19261
20. Yang et al. (2024), Qwen2 — https://arxiv.org/abs/2407.10671
21. Qwen2.5 tech report — https://arxiv.org/abs/2412.15115
22. Hefny et al. (2024), Upcycling LLMs into MoE — https://arxiv.org/abs/2410.07524
23. Muennighoff et al. (2025), OLMoE — https://arxiv.org/abs/2409.02060
24. Cluster-aware Upcycling (2026) — https://arxiv.org/abs/2604.13508
25. Li et al. (2022), BTM — https://arxiv.org/abs/2210.00038 ; Sukhbaatar et al. (2024), BTX — https://arxiv.org/abs/2403.07816
26. CuMo (2024) — https://arxiv.org/abs/2403.13686 ; MoE Jetpack — https://arxiv.org/abs/2410.17748
27. Scaling Laws for Upcycling (2025) — https://arxiv.org/abs/2412.09643
28. Parameter-merging upcycling (ACL 2025) — https://aclanthology.org/2025.acl-long.984/
29. CLIP-UP (2025) — https://arxiv.org/abs/2501.01367
30. ProMoE (2025) — https://arxiv.org/abs/2510.24711 ; DiT-MoE — https://arxiv.org/abs/2406.04619 ; DiffMoE — https://arxiv.org/abs/2412.02105
31. LPR (2025) — https://arxiv.org/abs/2506.21328
32. DeepSeek-V3 — https://arxiv.org/abs/2412.19437 ; Loss-Free Balancing — https://arxiv.org/abs/2408.15664
33. Skywork-MoE — https://arxiv.org/abs/2406.06563
34. Chi et al. (2023), Diffusion Policy — https://arxiv.org/abs/2303.04137
35. MoDE (ICLR 2025) — https://arxiv.org/abs/2412.12953
36. JumpStart RL (2023) — https://arxiv.org/abs/2204.02372
37. Di-SkilL (2024) — https://proceedings.mlr.press/v235/celik24a.html
38. Skill-MoE (2026) — https://arxiv.org/abs/2503.05641
39. Dynamic MoE of PEFT experts (2025) — https://arxiv.org/abs/2506.05985
40. LoRAMoE (ACL 2024) — https://aclanthology.org/2024.acl-long.106/ ; MOLE (ICLR 2024) — https://arxiv.org/abs/2402.11589
41. GOAT (2025) — https://arxiv.org/abs/2502.16894 ; PanGu-Σ — https://arxiv.org/abs/2303.10845
42. PEER (2024) — https://arxiv.org/abs/2407.04153 ; SMoE-Dropout — https://arxiv.org/abs/2303.01610
43. MoA (2026) — https://arxiv.org/abs/2604.21203

*Note: a "CARBON" paper on cluster-based routing could not be verified and is deliberately excluded.*