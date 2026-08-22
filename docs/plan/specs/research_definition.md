# PhaseForge — Research Definition, Causal Framework, and Falsifiable Hypotheses

**Status:** finalized research specification (incorporating literature audit & professor feedback)

**Scope:** non-visual robot manipulation from privileged structured low-dimensional simulator state

**Benchmark:** robomimic v0.1 low-dimensional demonstrations with a single, explicitly pinned robosuite release track

---

## 1. What the project is about: Privileged Regime Geometry Transfer

PhaseForge investigates the following core scientific question:
> **Can privileged regime information available during training be converted into useful latent geometry and transferred into an MoE routing prior, enabling specialized control without requiring the privileged phase signal at inference?**

The causal framework is structured as four distinct, testable links:

$$\boxed{\text{Phase Supervision} \xrightarrow{(1)} \text{Phase-Discriminative Geometry} \xrightarrow{(2)} \text{Routing Prior Initialization} \xrightarrow{(3)} \text{Autonomous Expert Specialization} \xrightarrow{(4)} \text{Superior Control Performance}}$$

### The Three-Layer Mechanism:
1. **Stage 1 (Representation Shaping):** Generalist policy pretraining with an auxiliary phase-classification head shapes the latent space to reflect behavioral regimes.
2. **Transfer (Bootstrap Prior):** Latent centroids/prototypes transfer into the MoE router weights, and the action head warm-starts the experts.
3. **Stage 2 (Autonomous Specialization):** Privileged phase labels disappear. The MoE policy trains with the action loss and load balancing, autonomously settling its own routing decomposition.
4. **Inference Deployment:** Deployed without privileged phase labels; receives only the structured state.

---

## 2. Literature Positioning & Prior Art

PhaseForge sits at the intersection of four research traditions:
1. **Classical Mixture-of-Experts:** Jordan & Jacobs (1991), Jacobs et al. (1991) — partitioning complex dynamical state spaces into simpler local experts.
2. **Modern Sparse / Noisy Top-k MoE:** Shazeer et al. (2017), Switch Transformer (Fedus et al. 2022) — noisy top-k dispatch and load-balancing auxiliary losses.
3. **Dense-to-MoE Upcycling:** Sparse Upcycling (Komatsuzaki et al. 2022) — initializing MoE experts from pretrained dense generalists.
4. **Cluster-Based Router Initialization & Latent Routing:**
   - **Royer et al. (BMVC 2022):** Proposes clustering pretrained embeddings to initialize an MoE gate and experts from a base model.
   - **Cluster-aware Upcycling (CVPR 2026):** Clusters activation spaces via spherical K-means and initializes router weights from cluster centroids.
   - **LAR-MoE (March 2026):** Latent-aligned routing in robotic imitation learning using unsupervised representations.
5. **Phase / Subtask-Conditioned Manipulation:** PAMAE (Yang et al. 2026), SMP (Hao et al. 2026).

**Differentiator:** PhaseForge investigates **privileged regime geometry transfer** under a controlled factorial design, comparing privileged phase prototypes against generic unsupervised clustering (Spherical K-means) and discriminative classification directions.

---

## 3. Falsifiable Hypotheses

### H1 — Router-Initialization Effect
Holding the encoder and expert warm-start fixed, phase-centroid initialization will produce higher initial routing alignment ($t=0$), faster specialization, and lower action error than random router initialization (`PhaseForge` vs `Phase-Pretrain Random-Router`).

### H2 — Phase-Representation Effect
Holding router initialization and expert warm-start fixed, a phase-supervised encoder will produce more useful regime-conditioned routing priors than an unsupervised BC encoder (`PhaseForge` vs `Plain-Encoder Phase-Bootstrap`).

### H3 — Privileged Transfer vs Generic Clustering
Privileged phase-supervised prototypes will provide more control-relevant routing priors than generic unsupervised activation clustering (`PhaseForge` vs `PS-Spherical-KMeans`).

### H4 — Prototype vs Discriminative Classifier Initialization
Class prototype vectors (mean latent directions) provide a more effective routing prior than discriminative classifier weights (`PhaseForge` vs `PS-Phase-Head`).

### H5 — Routing Quality on Fixed Experts
> **Oracle MoE is an evaluation-time routing intervention applied to a fixed trained expert set.**

The routing gap measures how much performance the autonomous learned router leaves on the table compared to oracle (ground-truth phase-directed) dispatch on the **same trained experts**:

$$\text{Routing Gap} = \text{Oracle} - \text{PhaseForge}$$

A small gap indicates the learned router is already near-optimal for the expert specializations it has induced. A large gap indicates routing is the binding constraint, not expert quality.

The oracle baseline is **not** a separate trained model — it is an eval-time diagnostic that replaces learned gating with deterministic phase-directed dispatch ($e = \text{phase} \pmod{E}$) on PhaseForge's trained expert set.

### H6 — Rollout Success (Primary Performance Hypothesis)
PhaseForge rollout success rate exceeds the BC baseline (dense single-stage) and WarmStart-MoE (random router + BC encoder) under frozen evaluation seeds, establishing that privileged geometry transfer yields a deployable advantage.

> **Decision tree (SR = 0 across all methods):** If no method achieves SR > 0 on Lift, the protocol pivots to reporting **MSE-only** comparisons across the full matrix. No rollout-success claims are permitted; the paper reports "training converged but no policy transferred to closed-loop control."

---

## 4. Claims Permitted / Prohibited

| Condition | Permitted Claims | Prohibited Claims |
|---|---|---|
| PhaseForge SR > BC SR (significant) | "Privileged geometry transfer improves rollout success" | — |
| PhaseForge SR ≈ BC SR | Report MSE advantage only, if any | "PhaseForge improves control" |
| All SR = 0 | MSE comparisons, specialization metrics, routing quality | Any rollout-success claim |
| PhaseForge MSE > BC MSE | "The transfer mechanism did not improve action prediction" | "PhaseForge is better" |
| PhaseForge top seed mean but 95% CIs overlap (n = 3 seeds) | Report point estimates with CIs; note directional evidence | "PhaseForge significantly outperforms" |

---

## 5. Primary Confirmatory Matrix (Wave 1)

> **Canonical-method lineage (2026-08-22 migration):** the proposed method is
> the promoted **R50 configuration** under the canonical `phaseforge`
> identity — six experts, top-2 routing, centroid router initialization,
> **50% partial expert warm-start** (Drop-Upcycling-style, seed-dependent),
> soft mapping disabled. The H1–H4 mechanism controls are R50-matched: they
> share the same partial-warm expert initialization, so each isolates exactly
> its declared factor. Pre-final results produced by the retired 8-expert /
> soft-mapping configuration are engineering context only and are excluded
> from the final evidence table.

| EXP | Cell Name | Encoder Source | Router Init | Expert Init | Causal Contrast / Purpose |
|---|---|---|---|---|---|
| EXP-101 | `phaseforge` | Phase-supervised | Phase Centroid | Partial Warm (50%) | **Proposed Method (Privileged Geometry Transfer; canonical R50)** |
| EXP-102 | `bc` | Plain (BC) | — | — | Behavior floor |
| EXP-103 | `bc_large` | Plain (BC-Large) | — | — | Capacity control (per-task deployed match, +0.8%…+1.8%) |
| EXP-104 | `bc_robot_only` | Plain (BC) | — | — | Negative control (no object state) |
| EXP-105 | `scratch_moe` | Random | Random | Random | No-pretraining baseline |
| EXP-106 | `warmstart_moe` | Plain (BC) | Random | Warmstart (0.02) | Warm-start MoE (BC encoder × random router; behavioral baseline, deliberately standard warm-start) |
| EXP-107 | `phase_pretrain_random_router` | Phase-supervised | Random | Partial Warm (50%) | H1: Router init holding representation & experts fixed |
| EXP-108 | `plain_encoder_phase_bootstrap` | Plain (BC) | Phase Centroid | Partial Warm (50%) | H2: Phase supervision holding router & experts fixed |
| EXP-109 | `pf_spherical_kmeans` | Phase-supervised | Spherical KMeans | Partial Warm (50%) | H3: Generic clustering control (vs Cluster-aware Upcycling) |
| EXP-110 | `pf_kmeans` | Phase-supervised | Euclidean KMeans | Partial Warm (50%) | Euclidean KMeans router initialization control |
| EXP-111 | `pf_phase_head` | Phase-supervised | Linear Phase Head | Partial Warm (50%) | H4: Prototype vs discriminative classifier direction init |
| EXP-112 | `pf_random_random` | Phase-supervised | Random | Random | 4-Way Matrix Cell A: Expert-init control |
| EXP-113 | `pf_centroid_random` | Phase-supervised | Phase Centroid | Random | 4-Way Matrix Cell B: Router-init control |
| EXP-114 | `pf_spherical` | Phase-supervised | Spherical centroid avg | Partial Warm (50%) | Spherical vs Euclidean centroid averaging ablation |
| EXP-115 | `pf_ft` | Phase-supervised | Phase Centroid | Partial Warm (50%) | PhaseForge-FT: encoder unfrozen (LR scale 0.1) |
| EXP-116 | `teacher_forced` | Phase-supervised | GT (train) / PhaseHead (eval) | Warmstart (0.02) | Privileged routing diagnostic & gap decomposition (locked E8 decision; not R50-matched by design) |

> **Oracle MoE** (EXP-117) is not listed as a training cell. It is an eval-time routing intervention on PhaseForge's trained expert set (see §3 H5), implemented as the `oracle_moe` baseline: phase-directed dispatch via `eval_mode="oracle"` in `phaseforge/models/phase_moe.py` (routes by $M^T \operatorname{softmax}(\text{phase\_head}(z))$ on the trained experts). *Open item D10: on the canonical centroid-initialized config the soft-mapping buffer is empty, so the oracle dispatch path requires a decision (identity mapping for E==P, direct phase-mod-E dispatch, or dropping the H5 diagnostic) before any oracle evaluation is run.*

## 6. Wave 2 — Sensitivity & Scaling Ablations

| EXP | Cell Name | Override | Purpose |
|---|---|---|---|
| EXP-201 | `pf_k3` | `models.router.num_experts=3, top_k=2` | K sweep: super-prototype reduction (E=3 < P=6) |
| EXP-202 | `pf_k12` | `models.router.num_experts=12, top_k=2` | K sweep: intra-phase sub-prototype scaling (E=12 > P=6) |
| EXP-205 | `pf_corrupt_25` | `data.phase_corruption_rate=0.25` | Phase noise sensitivity: 25% label corruption |
| EXP-206 | `pf_corrupt_50` | `data.phase_corruption_rate=0.50` | Phase noise sensitivity: 50% label corruption |
| EXP-207 | `pf_shuffle_control` | `data.phase_corruption_rate=1.0, phase_shuffle_control=true` | Phase noise: 100% permutation shuffle control |

> **Removed (2026-08-22):** EXP-203 `pf_jitter_00` / EXP-204 `pf_jitter_10` —
> jitter is inert under the canonical `partial_warm` expert init (exact copy,
> no jitter), and the Wave-4 drop-rate sweep subsumes both endpoints
> (drop_rate 0.0 = exact copy; 1.0 = fully reinitialized).

> **Corruption semantics (EXP-205..207):** corruption applies to the **bootstrap-label signal** — the phase labels of the Stage-2 training split used to compute the router prototypes (forced-different replacement: z' = (z + U(1..P-1)) mod P). These cells reuse the **clean** phaseforge Stage-1 encoder and Stage-1 supervision (no stage-1 rerun per level); they isolate the sensitivity of the routing prior to privileged-label noise at bootstrap time. Validation labels remain clean, so routing diagnostics stay interpretable. The 100% shuffle control is a bijective permutation (preserves marginal phase counts) rather than i.i.d. noise.

## 6b. Waves 3–4 — Expert-Initialization Suite (Lift)

All cells run on the canonical phase-supervised encoder and centroid router;
only the expert initialization varies.

| EXP | Cell Name | Expert Init | Purpose |
|---|---|---|---|
| EXP-211 | `pf_one_warm_plus_random` | one warm generalist + (n−1) random, warm_idx rotated by seed | Diagnostic: is one generalist expert sufficient? |
| EXP-212 | `pf_full_warm` | Full standard warmstart (0.02) | The pre-final initialization preserved as an ablation of the canonical partial warm-start |
| EXP-213 | `pf_drop00` | Partial warm, drop_rate 0.0 (exact copy) | Drop-rate sweep lower endpoint |
| EXP-214 | `pf_drop25` | Partial warm, drop_rate 0.25 | Drop-rate sweep |
| EXP-215 | `pf_drop75` | Partial warm, drop_rate 0.75 | Drop-rate sweep |
| EXP-216 | `pf_drop100` | Partial warm, drop_rate 1.0 (fully reinitialized) | Drop-rate sweep upper endpoint |

> The 50% point of the sweep is the canonical proposed method itself
> (EXP-101). **Removed as redundant after the canonical migration:**
> `warmstart_r50` and `phaseforge_e6` (both recreated the canonical method
> via now-obsolete overrides) and `pf_random_warm` (equals
> `phase_pretrain_random_router` at partial-warm expert init).

---

## 7. Direct Behavioral Specialization Evidence ($M_{z,e}$)

Expert specialization is measured directly, not inferred from NMI alone:
1. **Behavioral Matrix ($M_{z,e}$):** $M_{z,e} = \text{MSE}(\pi_e(x_z), a_z)$ computed by running each expert $e$ independently over validation samples of phase $z$.
2. **Optimal Selection ($e^*(z)$):** $e^*(z) = \arg\min_e M_{z,e}$, comparing theoretical minimum error $M_{z,e^*(z)}$ with routed error $M_{z,\text{routed}}$.
3. **Expert Divergence ($D(e_i, e_j)$):** Mean pairwise $L_2$ distance between expert predictions $D(e_i, e_j) = \mathbb{E}[\|\pi_i(x) - \pi_j(x)\|_2]$ on shared inputs.
4. **Routing Contingency ($\mathcal{C}_{p,e}$):** Normalized probability matrix $P(\text{expert } e \mid \text{phase } p)$.

## 8. t=0 Routing Diagnostics

Immediately after `bootstrap_moe()` and before any Stage 2 gradient update, the following diagnostics are computed and persisted to `metadata/init_routing.json`:

| Metric | Definition |
|---|---|
| `nmi_phase_top1` | Normalized Mutual Information between phase labels and top-1 expert assignments |
| `routing_entropy_mean` | Mean Shannon entropy of routing distributions across validation samples |
| `routing_entropy_normalized` | Mean entropy normalized by $\log(E)$ |
| `top1_cv` | Coefficient of variation of top-1 expert usage counts |
| `topk_cv` | Coefficient of variation of top-k expert usage counts |
| `collapse_rate` | Fraction of experts receiving < 1% of total dispatch weight |
| `dead_expert_count` | Number of experts with zero top-1 assignments |
| `phase_head_accuracy` | Phase head accuracy on validation set (Stage 1 quality check) |
| `phase_head_agreement` | Agreement between phase head predictions and top-1 expert assignments |
