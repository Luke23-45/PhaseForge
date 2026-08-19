# PhaseForge v2 — design from all accumulated evidence

## Part 1: The design laws our own experiments impose

Every number below is from our verified findings (seed 42 surgical, 3-seed real matrix where noted).

| Law | Evidence | Consequence |
|---|---|---|
| L1: **Initialization is the dominant lever** | cent_warm 0.72 vs cent_reset 0.38 / rand_warm 0.44 / rand_reset 0.44 | Keep bootstrap pipeline (warm-start experts + centroid router) exactly. Any redesign that changes this block is a regression risk |
| L2: **Privileged phase supervision is the only structure that won end-to-end** | phaseforge 0.640 vs kmeans-routers 0.51/0.52 vs bc_large (matched params) 0.447 | Keep phase-supervised stage 1; kill generic clustering as a competitor |
| L3: **Per-step learned routing is the weak link** | A5: learned 0.0600 vs uniform 0.0366 (per-step MSE); rn=0.5 → 0.40; B2 divergence dip at e30 | Routing must gain temporal structure and a better training objective |
| L4: **The 6-way phase decomposition is partially mis-specified** | phases 2/3 confusable (inter 3.41 < intra ~7.2); phase counts [161,**15**,385,352,**108**,5] | Soft/weighted labels; experts must not be hard-bound to data-starved phases |
| L5: **The rest is not the bottleneck** | checkpoint sweep flat (0.62–0.78), balance_coeff flat (0.72–0.78) | Don't design around selection/balance tuning |

## Part 2: SOTA blocks worth stealing (all verified this session)

1. **LAR-MoE** (arXiv 2603.08476): routing *regularized to follow latent structure*, router/policy learning decoupled — validation of our centroid-bootstrap idea; their latent-alignment is our privileged version.
2. **SMP** (arXiv 2601.21251): **sticky routing** (slowly-varying gates) → phase-consistent behavior; orthogonal skill basis. Directly answers L3.
3. **CoRDE** (arXiv 2606.21935): **router trained by KL to a responsibility posterior, decoupled from the score-matching gradient**; EM-updated soft concept→expert mapping; LoRA expert pool. Answers L3 + L4.
4. **FiLM phase-conditioning** (Chen et al. 2026, arXiv 2605.29407): phase injected via FiLM into one policy beats token-level conditioning; a real-time phase predictor closes the loop. Validates the critique's monolith idea.
5. **Zang (NeurIPS 2023)**: no transition-based/bisimulation machinery in a 200-demo fixed dataset — we don't build any.

## Part 3: PhaseForge v2 — "Soft-Regime MoE with Teacher-Distilled Sticky Routing"

Same skeleton (stage 1 → bootstrap → stage 2 → frozen eval), five changed blocks, each mapped to a measured failure:

**V2-A. Soft regime targets (fixes L4).** Stage-1 phase head trains on *soft targets*: hard one-hot blended with a precomputed phase-similarity prior (pairwise centroid affinity in normalized state space, computed once from demos), plus inverse-frequency weighting. Phases 2/3 stop being punished for their genuine similarity; phases 1/5 stop being over-committed. Implementable: one offline precompute + a soft-label mode in the ingester/`state_machine`, weighted CE in `stage1_loop`.

**V2-B. Soft phase→expert mapping, K>P experts (fixes L4).** 8 experts, no hard phase ownership. `bootstrap_moe` gains a `soft` router-init mode: hierarchical prototypes (existing `compute_hierarchical_phase_prototypes`) plus a P×E affinity matrix M (Dirichlet-smoothed). Data-starved phases 1/5 route *through* M to shared experts. M can stay fixed in v2.0 (EM updates = v2.1).

**V2-C. History-conditioned router (fixes L3).** Router input becomes `[z_t, z_{t-1}]` (zero-padded at demo start; the collator already produces trajectory-aware batches — `padding_mask` exists). Routing decisions gain temporal context at phase boundaries — the exact place A4/C1 show confusion. Cost: one extra `latent_dim` on the gate linear.

**V2-D. Teacher-distilled routing with annealing (fixes L3 + L5, the heart of v2).** Stage-2 loss becomes:

```
L = L_action + β·L_balance + λ(t)·L_KL(softmax(gate_linear([z_t, z_{t-1}])) ‖ Mᵀ·softmax(phase_head(z_t))) + γ·L_sticky
```

- The **frozen stage-1 phase head becomes a routing teacher** during stage 2 (plumbing already exists — `teacher_forced` emits `phase_logits` in stage-2 mode); λ(t) anneals to 0 late in training.
- **L_sticky** penalizes consecutive-step gate drift (SMP's stickiness, as a loss on demo trajectories).
- Evidence for this block: B1 showed the centroid prior is worth +0.28 over random *at init* — we currently abandon it the moment stage 2 starts, and A5 shows the unaided router never recovers. L2's winning supervision is kept at train time and **dropped at inference** — the autonomy claim survives (inference = encoder + router only), and H5 (routing gap vs oracle) becomes measurable at every annealing point.

**V2-E. Honest eval protocol (unchanged premise, better reporting).** Rollout eval on the frozen bank reports four router *interventions* (learned / sticky-smoothed window / uniform / phase-head oracle — all eval-time only, per the professor's definition) and **per-phase SR**, so a timeout cluster in one phase is visible instead of hidden in the aggregate (closes the C2 gap as far as the bank allows).

## Part 4: Risks, stated honestly

1. **V2-D may collapse to the `teacher_forced` cell** (GT-route at train / phase-head at eval). If teacher_forced ≈ phaseforge in the real matrix, the distillation value is smaller — but soft annealing ≠ hard forcing, and V2-C/D together are a strictly different object. We should check the teacher_forced result from part4 before committing.
2. **New hyperparameters**: λ, γ, α (label blend), K. Mitigation: the evidence bounds them — rn/balance flatness says small aux weights are safe; sweep only λ schedule, keep everything else at v1 values.
3. **V2 degrades gracefully**: if the professor's specialist test later kills the phase premise, v2's eval-time `uniform` mode + soft labels mean it still contains the fallback (uniform ensemble was the best per-step predictor in A5) — no design bet is irreversible.
4. **Capacity**: history input + M matrix ≈ +1–2% params over v1; bc_large matching (0.8%) stays the accounting baseline.

## Part 5: Codebase mapping (all existing machinery)

- `phaseforge/data/ingestion/state_machine.py` + `phase_labeler.py` — soft-label mode (V2-A)
- `phaseforge/trains/loops/stage1_loop.py` — weighted CE + soft targets (V2-A)
- `phaseforge/models/phase_moe.py::bootstrap_moe` — `soft` init + M matrix (V2-B)
- `phaseforge/models/components/router.py` — history input; `stage2_loop.py` — L_KL + L_sticky + λ schedule (V2-C/D)
- `phaseforge/evaluations/rollout/runner.py` — router-mode flags + per-phase SR (V2-E)
- Config: one new model (`phaseforge_v2`) + overrides; eval determinism machinery already proven (bit-identical reruns)

---

**Bottom line**: v2 keeps the two blocks the data proved (phase-supervised representation, centroid+warm-start bootstrap), and replaces the one block the data falsified (unaided per-step router) with a temporally-consistent, teacher-distilled, soft-mapped router. It answers the professor's objection (routing doesn't decompose control usefully → so we supervise the decomposition properly during training and verify it at eval-time), and it survives either outcome of the decisive specialist experiment.

Want me to (a) write this up as `docs/dev/findings/phaseforge_v2_design.md`, (b) verify the `teacher_forced` cell result from part4 first (recommended — it's the one result that could derail V2-D), or both?