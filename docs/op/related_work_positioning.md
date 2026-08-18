# PhaseForge — Related-Work Positioning Statement (2026-08-18)

**Purpose:** record the full reads of the closest 2026 papers (per professor review §5) and the explicit positioning of PhaseForge against them. Summaries below are grounded in the papers' own abstracts/text, fetched directly from arXiv.

---

## 1. Closest concurrent work (verified against primary sources)

### MoE-ACT — Mazza et al., RSS 2026 (arXiv:2601.21971, v2 2026-05-09)
**"Supervised Mixture-of-Experts for Surgical Grasping and Retraction"**

- Supervised MoE layer added on top of an ACT policy: one expert per phase, gating supervised directly by phase labels during training (phase CE term in the objective; experts specialize per task phase). Trained on stereo endoscopic images, <150 demos, bowel grasping + retraction; strong OOD robustness and zero-shot ex vivo transfer.
- **Overlap with PhaseForge:** phase-structured MoE for manipulation; phase labels supervise expert specialization.
- **Differences:** (a) MoE-ACT supervises the gating network directly with phase CE during MoE training; PhaseForge's stage 2 has **zero phase supervision** (lambda_phase = 0) — phase structure enters only via the *initialization* of the router (centroids) from a frozen phase-supervised encoder. (b) MoE-ACT is top-1 exclusive (one expert per phase); PhaseForge is top-2-of-6 with a learnable router. (c) Vision/ACT setting vs low-dim state MLP. (d) No controlled factorial isolating encoder-supervision from router-init; no seed-variance/checkpoint-selection analysis.

### LAR-MoE — Rodriguez et al., arXiv:2603.08476 (2026-03-09, submitted to iROS 2026)
**"Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning"**

- Two-stage: (1) student–teacher co-training learns a joint observation–future-action latent (unsupervised skill discovery); (2) the student encoder is **frozen** and a soft MoE is trained with routing **regularized** to follow the latent structure — distance-consistency loss (routing-distribution distances match latent distances), entropy regularization, group-sparse regularization. LIBERO: 95.2% avg success with 150M params; surgical bowel grasping + retraction matches a supervised-MoE baseline (MoE-ACT) without phase annotations.
- **Overlap with PhaseForge:** two-stage; frozen pretrained encoder; routing anchored to latent structure; goal of structured expert specialization without runtime phase labels.
- **Differences:** (a) LAR-MoE's pretraining is *unsupervised* (student–teacher, no phase labels) and the alignment is a *loss during training*; PhaseForge's stage 1 is *phase-supervised* and the alignment is a *one-time initialization* of the router gate (centroids as unit-norm gate weights, cosine gate) with no structural loss in stage 2. (b) No factorial ablation isolating encoder-supervision vs router-init. (c) No checkpoint-selection / seed-variance analysis.

### SMoDP — Deng et al., arXiv:2605.23477 (2026-05-22)
**"Semantically Structured Mixture-of-Experts for Compositional Robotic Manipulation"**

- VLM-aided offline skill abstraction annotates demonstrations with verb–noun skills; a lightweight inference-time skill predictor routes action chunks to phase-specialized experts in a diffusion MoE policy; dual contrastive alignment (inter-modal: skill embeddings vs frozen language-encoder embeddings; intra-modal: routing consistency across semantically similar skills) plus load-balancing loss. Multi-task benchmarks, compositional transfer via parameter-efficient fine-tuning.
- **Overlap with PhaseForge:** skill/phase-conditioned routing for manipulation; structure-aware router specialization.
- **Differences:** (a) skill semantics come from a VLM + language encoder at training time and a learned skill predictor at inference; PhaseForge uses a cheap deterministic rule-based labeler and a phase head — no language models. (b) SMoDP applies structural *losses* (contrastive) during training; PhaseForge uses pure *initialization*. (c) Diffusion-policy/vision multi-task setting vs low-dim state MLP. (d) No factorial ablation of encoder-supervision × router-init.

### Adjacent (read, lower proximity)
- **CoRDE** (arXiv:2606.21935): concept-prior routed diffusion experts (frozen concept encoder + soft mapping matrix) — shares the "frozen structure source guides routing" idea in a diffusion setting; still loss-based, no initialization-based bootstrap, no factorial.
- **MATE** (arXiv:2606.01047): cross-modal cosine router for trajectory MoE — uses cosine similarity routing (as PhaseForge's normalize_input gate does) for scale-invariance; no phase/skill structure.

---

## 2. Positioning statement (2–3 sentences for the paper)

While several concurrent works (MoE-ACT, LAR-MoE, SMoDP) converge on structure-aware routing for manipulation MoEs, none of them: (i) use a **centroid initialization** of the router gate (rather than training-time regularization or direct gating supervision) as the mechanism that injects phase structure; (ii) run a **controlled 2×2 factorial** isolating the phase-supervised-encoder effect from the router-initialization effect; or (iii) characterize the **checkpoint-selection / flat-plateau lottery** as a mechanism of seed variance in this class of method. PhaseForge's contribution rests on these three distinctives: the *initialization-only* injection of structure (no structural losses in stage 2), the *factorial evidence* that separates representation benefit from router-init benefit, and a *methodological diagnosis* (negative transfer of an auxiliary phase head + underspecified checkpoint selection on a flat monitor plateau) that generalizes beyond PhaseForge itself.

---

## 3. Caveats

- The 2026 papers (2601.21971, 2603.08476, 2605.23477, 2606.21935, 2606.01047) were read via abstract + full-text HTML where available; for precise claims (e.g., exact loss weights, benchmark numbers) re-verify against the PDF before the paper's related-work section is finalized.
- LAR-MoE and MoE-ACT share authors (Dresden group); both are preprints under review (RSS 2026 accepted for MoE-ACT).