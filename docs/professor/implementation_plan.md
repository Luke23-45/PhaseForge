# PhaseForge — Implementation Plan for Professor Report 1 (literature audit)

**Source:** `docs/professor/report1.md` (professor literature audit, 2026-08-18)
**Status:** to be executed top-down; every phase has an explicit gate
**Grounding rule:** every professor suggestion below was verified against the repo (and, for the cited papers, against primary sources) before being written. Where the report assumes something not available on disk, the plan says so and adjusts. Existing protocol cells are NOT modified; new experiments are new cells with explicit tags.

---

## 0. Verified facts that shape the plan

### 0.1 Literature (verified against primary sources this session)

| Professor's claim | Verified reality | Consequence |
|---|---|---|
| Royer et al. cluster embeddings to init an MoE gate + experts from pretrained base model (2022/2023) | **Confirmed** (BMVC 2022, arXiv:2304.05497). Per-sample **clustering-based initialization**; base-model branch reused as early-exit/regularizer; **single-gate** MoE, vision domain, **no privileged labels**, no controlled encoder×router factorial. | The generic "cluster → router init" claim is indeed not novel; positioning must stress the **privileged phase supervision** source of the clusters and the **factorial** design. |
| Cluster-aware Upcycling (CVPR 2026) clusters activations, inits experts from cluster subspaces and router from cluster centroids, uses spherical K-means | **Confirmed** (CVPR 2026 pp. 11283–11292; arXiv:2604.13508). Router weights set to **cluster centroids**; experts via truncated SVD per cluster; adds self-distillation loss; CLIP ViT zero/few-shot. Abstract mentions semantic clustering, not explicitly spherical K-means — treat "spherical K-means" as the professor's interpretation of direction-sensitive clustering; still a valid ablation for us (our router is cosine). | Same as above; additionally gives us an exact named precedent for "router = centroids", so the paper must cite it and position the **phase-supervised** variant + control setting as the difference. |
| LAR-MoE (2026) = closest high-level conceptual comparator; latent-aligned routing, no phase labels | **Confirmed** (arXiv:2603.08476, iROS 2026 submission). Two-stage; **unsupervised** student–teacher latent; **frozen** student encoder; routing **regularized** (distance-consistency + entropy + group-sparse) to the latent; LIBERO 95.2% @ 150M; matches a supervised-MoE baseline without phase annotations. | The professor's §7/§24 positioning (unsupervised latent structure vs privileged phase-supervised structure) is real and must be discussed. |
| Professor's own table cites "Phase-conditioned IL, 2026" with ref [13] = ResearchGate 233806999 | **Mislabeled**: ref [13] is Jacobs et al. 1991 *Adaptive Mixtures of Local Experts* (the classical MoE paper, correctly cited two rows up as such). The row label does not match the citation. | **Fix in the paper's related-work table** before it reaches a reviewer: either find the intended 2026 phase-conditioned IL paper (e.g., MoE-ACT, already in `docs/op/related_work_positioning.md`) or drop the row. |

### 0.2 Architecture / parameter fairness (measured on this machine)

| Model | Total params | Notes |
|---|---|---|
| BC-MLP (current floor) | **206,983** | 3×256 encoder + action head |
| PhaseForge | **382,646** (1.849× BC) | 6 experts (34,823 each) + router (1,548) + encoder |
| PhaseForge stage-2 **trainable** | **210,486** | encoder frozen; ≈ BC's entire budget |
| PhaseForge **active params/sample** | **71,194** (0.344× BC) | 2 of 6 experts active |
| Scratch MoE | 382,646 | same arch, no stage 1 — an existing capacity-matched-but-unpretrained control |
| BC-large candidate `hidden [372,372,372]` | **385,855** (+0.8% vs PhaseForge) | proposed parameter-matched dense baseline |

Key fairness nuance for the paper: PhaseForge has 1.85× total params of BC but only 0.34× active params per sample, and its stage-2 *trainable* count ≈ BC's total. All three numbers (total / trainable-in-stage-2 / active-per-sample) plus FLOPs must be reported (professor §15).

### 0.3 Existing code coverage of the professor's matrix

| Professor suggestion | Repo status |
|---|---|
| 2×2 factorial (encoder × router init) | **Exists** — 4 cells: phaseforge (phase-sup × centroid), phase_pretrain_random_router (phase-sup × random), plain_encoder_phase_bootstrap (plain × centroid), warmstart_moe (plain × random). All trained at `c09270a`/`289b3c3`; rollout results recorded (V0 0.633 mean at `289b3c3`). Current estimates favor PhaseForge, but the present sample does not establish a statistically reliable advantage (CIs overlap). |
| 3×3 factorial (+ unsupervised cluster, + phase-head init) | **Partial** — phase-head init was prototyped on CPU as uncommitted local patch `V1_phase_head_init` (seed 42 only: NMI 0.472 vs V0 0.449, entropy 0.273 — far more peaked than V0's 0.951). K-means router: **missing** (scikit-learn ≥1.4 already a dependency). |
| Four-way init ablation (random/random, centroid/random, random/warmstart, centroid/warmstart) on the **phaseforge stage-1 encoder** | **2 of 4 cells exist**: D = phaseforge ✓, C = phase_pretrain_random_router ✓. Missing: A (random router + random experts), B (centroid router + random experts). `scratch_moe` is not a substitute (no stage-1 encoder at all). |
| Parameter-matched dense baseline (BC-large) | **Missing** — needs new config + manifest row only (no new code). |
| Expert specialization matrix M_{z,e} (MSE per phase × per expert) | **Missing** — offline evaluator already collects expert_indices/gate_logits/phases; needs a new metric module + evaluator wiring (professor §16). |
| K ∈ {3, 6, 12} experts | **Config-ready** — `router.num_experts` exists; `bootstrap_moe` already handles E≠P (warns, maps 1:1 for first P, rest random, `phase_moe.py:194-199,259-262`). New cells only. |
| Jitter σ sweep | **Needs one knob** — `warm_start_experts_from_action_head(..., jitter_std=0.02)` is a hardcoded default (`expert.py:95`); not config-wired. |
| Spherical vs ordinary centroids | **Small code** — current `bootstrap_moe` averages raw latents then normalizes (`phase_moe.py:240,254`); spherical = normalize latents *before* averaging. |
| Phase-head-weight router init | **Code exists as prototype only** — V1 was an uncommitted patch; must be re-implemented cleanly. |
| Phase-label corruption | **Implemented (bootstrap-label variant)** — seeded post-labeling transform on the train split; cells EXP-205..207 corrupt the labels used for prototype computation on the **clean** phaseforge stage-1 encoder (see `research_definition.md` §6). |
| Fine-tuned encoder variant (PhaseForge-FT) | **Half-ready** — `train.freeze_encoder: false` exists; "small encoder LR" needs optimizer param groups (small code). |
| Teacher-forced (H4) + oracle MoE cells | **Fully implemented + tested, never run** (`teacher_forced.py`, `oracle_moe.py`; manifest rows exist in `lift_pilot.json`/`five_task.json`). Executing them closes the professor's §17/§18 gap decomposition. |
| Contingency matrix / entropy trajectories / balance curves | `build_contingency_matrix` exists (`phase_alignment.py:72`) but is **never logged**; entropy/NMI/balance/collapse are already logged per-epoch in stage-2 curves (trajectory plots = scripts only). |
| Router entropy/NMI trajectories across stage 2 | **Already logged** (`val/phase_expert_nmi`, `val/routing_entropy`, balance/collapse per epoch) — professor §31 is a plotting/analysis task. |
| Fairness accounting (steps, examples, FLOPs, active params) | **Missing** — new accounting script (param counts already computable). |
| Novelty reframing ("Privileged Regime Geometry Transfer") | Docs `research_definition.md` §3 (candidate contribution), `related_work_positioning.md` need updating with Royer + Cluster-aware Upcycling + LAR-MoE full reads and the reframed claim (professor §8, §25, §36). |

### 0.4 Repo / protocol state (verified)

- `master` @ `e25b646` (rollback of λ-decay to the frozen constant-λ protocol + gitignore). Variant branch `phase-utilization-experiments` @ `289b3c3` carries only inert config keys; the V1/V2/V4/V6 implementations were **local patches, not committed** — do not rely on them; re-implement cleanly on master.
- `docs/professor/` is untracked (report not yet committed).
- Current authoritative rollout results (Lift, 3 seeds, constant λ, `289b3c3`): **V0 (phaseforge) 0.633** (0.68/0.72/0.50); variant cells V4 0.600, V2 0.540, V6 0.440. All Wilson/pooled CIs overlap → current estimates favor PhaseForge, but the present sample does not establish a statistically reliable advantage (see `scripts/analysis/stratified_stats.py`, `f02a48b`).
- GPU stage-1/2 checkpoints were never synced to this machine (only metrics/metadata). Any GPU cell that needs a stage-1 checkpoint must either re-run stage 1 on GPU or sync checkpoints from Colab — plan for re-running (cheap, deterministic, commit-gated).
- Rigor machinery in place: commit-gated runner, fail-closed metadata, NaN monitor guard, 509 tests, `scripts/protocol/preflight_configs.py` validates all config cells.

---

## 1. Reframed contribution (professor §8, §36, §42 — adopt before any new compute)

- **Research question** becomes: *Can privileged regime information available during training be converted into useful latent geometry and transferred into an MoE routing prior, enabling specialized control without the privileged phase signal at inference?* ("Privileged Regime Geometry Transfer", §8).
- **Three-layer structure** (§36): (1) phase supervision shapes latent geometry; (2) phase prototypes transfer into the router; (3) phase labels disappear in stage 2 and the MoE autonomously settles its own decomposition.
- **Claims not yet permitted** (§42): do not claim "PhaseForge discovers phase-specialized experts" until the expert × phase behavioral matrix (§16) is measured; wording = "initializes routing from phase-structured latent geometry and enables subsequent expert specialization".
- Actions: update `docs/plan/research_definition.md` §1/§3/§7 accordingly; extend `docs/op/related_work_positioning.md` with full reads of Royer et al., Cluster-aware Upcycling, LAR-MoE (abstracts verified this session; keep the caveat to re-verify loss-weight-level claims against PDFs before the paper); fix the mislabeled "Phase-conditioned IL, 2026" row in the professor's own table (§0.1).

**Gate 0:** claim reframed in the docs; related-work table fixed; no compute until then.

---

## 2. Phase 1 — Code infrastructure (no GPU)

All new code lands on `master` with tests; defaults must remain **bit-identical** to the frozen protocol (determinism check in Gate 1).

### 1.1 Initialization-mode refactor (`bootstrap_moe`)
Turn the two initialization decisions into config fields, keeping current behavior as the default:
- `models.router_init: {type: centroid | random | phase_head | kmeans | spherical_centroid | spherical_kmeans}` (default `centroid`)
  - `random`: skip centroid assignment (current PhasePretrainRandomRouter behavior).
  - `phase_head`: `W_R^{(i)} = W_{p,i}/|W_{p,i}|` from the frozen linear phase head (professor §12) — re-implements V1 cleanly.
  - `kmeans` / `spherical_kmeans`: sklearn KMeans (k = P) on the frozen latents; centroids replace phase centroids (professor §10/§13; sklearn already a dependency).
  - `spherical_centroid`: normalize latents before averaging (professor §13).
- `models.expert_init: {type: warmstart | random, jitter_std: 0.02}` (default warmstart/0.02). `random` = fresh Kaiming draws, no warm start (four-way ablation cells A/B). `jitter_std` exposed for the σ sweep (professor §32) instead of the hardcoded default in `expert.py:95`.
- Wiring point: `phaseforge/cli.py:528-569` calls `model.bootstrap_moe(...)`; the config fields are read inside each model's `bootstrap_moe`. Existing model classes (`phase_pretrain_random_router`, `warmstart_moe`, `plain_encoder_phase_bootstrap`, `teacher_forced`) stay untouched — new cells are **config-only cells** on the phaseforge model with overrides, plus their own YAML under `config/models/baselines/`.

### 1.2 New cells (config + manifest rows in a new `experiments/lift_ablation.json`; never touch `five_task.json` defaults)
Professor's priority order:

| Priority | Cell | model config | stage-1 source | What it answers |
|---|---|---|---|---|
| 1 | `bc_large` | new `baselines/bc_large.yaml`, hidden `[372,372,372]` (385,855 params, +0.8%) | own stage 1 | parameter-matched dense (professor §14) |
| 1 | `pf_random_random` | phaseforge + router_init=random + expert_init=random | phaseforge | four-way ablation A (professor §33) |
| 1 | `pf_centroid_random` | phaseforge + expert_init=random | phaseforge | four-way ablation B (professor §33) |
| 1 | `pf_kmeans` | phaseforge + router_init=kmeans | phaseforge | unsupervised-cluster router (professor §10) |
| 1 | `pf_phase_head` | phaseforge + router_init=phase_head | phaseforge | phase centroid vs phase-head init (professor §12, §41) |
| 1 | `teacher_forced` | **existing** | phaseforge | gap decomposition (professor §17-18). Oracle is **not a training cell**: it is an eval-time routing intervention applied to the fixed trained expert set (`scripts/oracle_routing_diagnostic.py`, locked per professor §58). |
| 2 | `pf_k3`, `pf_k12` | `pf_spherical` + num_experts=3/12 (+ top_k=2 ≤ E) | phaseforge | expert-count sweep (professor §20-21); explicit prototype construction for arbitrary K via `clustering.py` (E<P super-prototypes, E>P intra-phase spherical K-means) |
| 2 | `pf_jitter_00`, `pf_jitter_10` | phaseforge + expert_init.jitter_std∈{0.0,0.1} (0.02 = current) | phaseforge | jitter ablation (professor §32) |
| 2 | `pf_corrupt_25`, `pf_corrupt_50`, `pf_shuffle_control` | phaseforge + `data.phase_corruption_rate`∈{0.25,0.5,1.0(+shuffle)} | shared clean phaseforge | label corruption, bootstrap-label variant (professor §19) |
| 3 | `pf_spherical` | phaseforge + router_init=spherical_centroid | phaseforge | spherical centroids (professor §13) |
| 3 | `pf_ft` | phaseforge + freeze_encoder=false + encoder LR group | phaseforge | fine-tuned variant (professor §28) |

### 1.3 Direct expert specialization measurement (professor §16, §23)
New metric module `phaseforge/evaluations/metrics/specialization.py` + evaluator wiring (`offline_evaluator.py`):
- `eval/expert_phase_mse_matrix`: run **each expert independently** over all val latents → 6×6 matrix M_{z,e} = MSE(π_e(x_z), a_z) (needs a per-expert forward, bypassing the router; MoE layer already holds `experts[e]`).
- `eval/phase_expert_contingency`: log `build_contingency_matrix` output (exists, never wired).
- `eval/expert_pairwise_divergence`: mean pairwise L2 distance between expert outputs on shared inputs.
- Persist matrices as JSON sidecar files (not curves), with schema entries in `outputs_writer/schema.py` and definitions in the `eval/metric_definitions` payload.

### 1.4 Label corruption (professor §19)
Seeded post-labeling transform in the data pipeline (`data.common.dataset`), config `data.phase_corruption_rate` + `data.phase_corruption_seed` (per training seed) + `data.phase_shuffle_control`. **Implemented semantics (EXP-205..207):** corruption applies to the train-split phase labels that feed prototype computation at bootstrap, on the clean phaseforge stage-1 encoder; validation labels stay clean; teacher_forced/oracle cells are rejected by preflight if corruption is set (done).

### 1.5 PhaseForge-FT (professor §28)
`freeze_encoder: false` exists; add optional `train.encoder_lr_scale` (e.g. 0.1×) via optimizer param groups so the encoder adapts at a small LR.

### 1.6 Fairness accounting script (professor §15)
`scripts/analysis/fairness_accounting.py`: per cell — total params, stage-2 trainable params, active params/sample (K×expert+router), total optimizer steps, examples seen (steps×batch), approximate training FLOPs (fwd+bwd, from resolved config) and inference FLOPs. Run on every cell at Gate 3 and included in the results table.

**Gate 1:** default configs bit-identical (CPU rerun of the V0 reference reproduces its recorded curves); 509+ tests pass; ruff/mypy clean; `preflight_configs.py` validates every new cell (K≤E top-k, corruption rejected on teacher_forced/oracle, `bc_large` |ratio−1|≤1.5%, freeze/FT consistency, oracle training rejected).

---

## 3. Phase 2 — CPU validation (per new cell, seed 42 first)

Same discipline as the earlier variant screening (stage-1 ~6 min/seed on CPU):
- Sanity: loss decreases, no NaN, routing metrics in plausible ranges, no silent collapse anomalies; determinism of the new code paths.
- Qualitative checks against the CPU V0 reference: e.g., V1-style phase-head init should reproduce the peaked-entropy signature (0.27 vs 0.95) already seen; kmeans router should sit between random (NMI≈0.18) and phase-centroid (NMI≈0.45).
- Corruption sweep on CPU first (5 levels × 1 seed) to choose GPU levels; σ sweep likewise.
- Also validates `bc_large` trains (deeper-wide MLP, same budget).

**Gate 2:** all new cells pass CPU sanity; questionable cells fixed before GPU spend.

---

## 4. Phase 3 — GPU run (Lift, 3 seeds, frozen protocol rules)

Wave 1 (professor's highest priority, first): the 16 cells of `experiments/lift_ablation.json` EXP-101..116 — phaseforge, bc, bc_large, bc_robot_only, scratch_moe, warmstart_moe, phase_pretrain_random_router, plain_encoder_phase_bootstrap, pf_spherical_kmeans, pf_kmeans, pf_phase_head, pf_random_random, pf_centroid_random, pf_spherical, pf_ft, teacher_forced. 13 stage-2 methods × 3 seeds; shared stage-1 providers: phaseforge, bc, bc_large, bc_robot_only (4 × 3 runs). Oracle = post-hoc eval-time intervention on the phaseforge stage-2 checkpoints (no training cell).

Wave 2 (second priority): `pf_k3`, `pf_k12`, `pf_jitter_00`, `pf_jitter_10`, `pf_corrupt_25`, `pf_corrupt_50`, `pf_shuffle_control` (7 × 3 seeds, no stage-1 — all reuse the clean phaseforge stage-1).

Same eval protocol: 50 paired episodes, reset bank `a7d3953c0afcf560`, horizon 500, commit-gated runner, `action_mse NaN` in results.jsonl expected and harmless.

**Gate 3:** every cell completes with commit-gated metadata; `fairness_accounting.py` table produced.

---

## 5. Phase 4 — Analysis and reporting

- **Mechanism story (professor §31):** per-cell trajectory plots — NMI, routing entropy, balance, collapse, action MSE vs epoch (data already logged).
- **Specialization evidence (professor §16):** M_{z,e} matrices for every MoE cell; NMI alone is never cited as specialization.
- **Gap decomposition (professor §17-18):** oracle (GT-routed eval-time intervention on the trained expert set) − phaseforge = routing-quality gap; teacher_forced − phaseforge = privileged-training strategy gap; phase predictability is assessed directly via phase-head accuracy vs GT (H5).
- **Statistics:** existing `scripts/analysis/stratified_stats.py` (seed-stratified bootstrap + PoI) on the new matrix; M=1-task caveat kept.
- **Fairness table (professor §15):** params/steps/examples/FLOPs/active-per-sample for every row of the final comparison.
- **Decision points for supervisor (professor §43-44):** does PhaseForge beat all four comparators (random-router, plain+centroid, unsupervised-cluster, param-matched dense)? Is oracle only modestly above PhaseForge? Does K=12 reveal sub-phase structure? Only then update the claim wording.

**Gate 4:** analysis write-up lands next to the professor report (`docs/professor/` or `docs/plan/`), with old claims explicitly superseded.

---

## 6. Cost estimate (GPU stage-2 runs on Lift)

| Wave | Cells × seeds | Stage-2 runs | Extra stage-1 runs | Eval runs |
|---|---|---|---|---|
| 1 | 13 stage-2 cells × 3 | 39 | 12 (phaseforge, bc, bc_large, bc_robot_only × 3 each) | 16 × 3 rollouts + 3 oracle diagnostics (post-hoc) |
| 2 | 7 × 3 | 21 | 0 (shared clean phaseforge stage-1) | 21 |

Oracle no longer trains (professor §57-58): the "upper bound" is an eval-time GT-routed pass over the phaseforge stage-2 checkpoints, so it costs 3 offline evals, not 9 training runs.

CPU screening removes the cheapest failures before GPU; waves are independent if compute must be staged.

---

## 7. What we will NOT do (professor §26)

No architectural additions (router attention, recurrence, contrastive losses, dynamic experts, extra regularizers). The mechanism stays: phase-supervised stage-1 → centroid-style router init → frozen-encoder stage-2 with top-2 noisy gating + balance loss. The plan adds *evidence*, not complexity.

## 8. Honesty commitments

- No "specialization" claim until M_{z,e} is measured (§42).
- No novelty claim without the fixed related-work table and the two 2026 papers cited (§44's second-pass warning is folded into Gate 0).
- Null results stay null; current estimates favor PhaseForge, but the present sample does not establish a statistically reliable advantage until the new matrix is analyzed.
- Protocol deviations (e.g., λ-decay) remain documented-refinement-only, exactly as recorded in `docs/op/implementation_plan.md` (Gate 1 result).
