# PhaseForge v2 — Design Report 1 (revised after external review)

**Status:** revised 2026-08-19; every number below was re-verified against the raw findings in
`phaseforge_studies/part2/outputs/surgical/_findings/` and `outputs/part*/` before being written.

---

## Part 1: The design laws our experiments impose (revised)

**L1 — Router-init × expert-init is an interaction, not a single "init" lever.**
From the surgical four-way (`four_way_init.json`, seed 42, 50-episode frozen bank):

| router init | experts warm | experts reset |
|---|---|---|
| centroid | 0.72 [0.583, 0.825] | 0.38 [0.259, 0.518] |
| random | 0.44 [0.312, 0.577] | 0.44 [0.312, 0.577] |

- Warm-start under centroid: +0.34, **disjoint CIs** — significant.
- Centroid vs random under warm: +0.28, CIs just barely non-overlapping (0.577 < 0.583) — significant but marginal.
- Under a random router, expert init is irrelevant (0.44 / 0.44, identical binary outcomes).
- Centroid+reset (0.38) vs random router (0.44): overlapping CIs — *not* distinguishable.

Consequence: **centroid routing only pays when combined with warm-started experts**; without
warm-start it is not better than a random router (and is not worse than one, once CIs are
respected). This interaction was measured on the v1 scheme (6 experts, hard centroid prototypes,
single seed). It must be re-verified under the V2-B soft scheme (3 seeds) before it is load-bearing
(gating experiment G4).

**L2 — Privileged phase supervision is the only structure with an end-to-end edge.**
Best-known cells (3-seed means; full roster in Part 7): phaseforge 0.640, bc 0.540, bc_large
0.447, teacher_forced 0.527, pf_kmeans 0.513, pf_spherical_kmeans 0.520. **Caveat verified this
revision (the sharpest limit on our evidence):** the real-matrix cells span **six different
waves** — different `data_config_hash` (a2da6ba3 / 6e529fe8 / 89464860 / 09a68c4c / cb69d88f /
88d7ae5b) across two commits (c09270a, 282947d); bank identity is not recorded in `run_meta`.
**phaseforge and the dense cells (bc, bc_large) never share a wave**, so "phaseforge beats dense"
is directional only, pending G3. What *is* same-wave:
- Group A (a2da6ba3, c09270a): phaseforge 0.640 > scratch_moe 0.587 > warmstart_moe 0.513; teacher_forced 0.527.
- Group B (6e529fe8, c09270a): plain_encoder_phase_bootstrap 0.600 > bc 0.540 > phase_pretrain_random_router 0.520.
- Group C (89464860, 282947d): pf_spherical_kmeans 0.520 ≥ pf_kmeans 0.513 ≈ pf_random_random 0.507 > bc_large 0.447.
- Group E (cb69d88f, 282947d, wave-2 sensitivity cells — cross-group vs phaseforge 0.640, directional):
  corrupt25 0.600 ≈ corrupt50 0.580 ≈ jitter00 0.593; jitter10 0.380 (drop); shuffle 0.500 (labels matter).
All four groups agree directionally: the phase-supervised / MoE-structured cells sit at or above
the dense and random-init cells. The magnitude is not yet established at same-bank resolution.

**L3 — The per-step learned router is net-harmful, not merely suboptimal.**
From `routing_counterfactuals.json` (val-best stage-2 checkpoint — `checkpoint_best.pt`,
best_epoch=3 — seed 42, n=1026 validation samples; deterministic: same experts, only the router
distribution differs):

- learned **0.06003** · uniform 0.03664 · random 0.03568 · oracle_true 0.07250 · oracle_pred 0.07391
- Learned routing is *worse than not routing at all* (uniform), and worse than the true-phase oracle.

Pinned scales (the two ambiguous numbers, resolved):
- "rn=0.5 → 0.40" is **rollout SR** on the 50-episode surgical bank: `ablation_grid.json` rn
  (router `noise_std`) 0.0 → 0.76, 0.1 → 0.72, 0.5 → **0.40 [0.276, 0.538]** — disjoint CI vs 0.72/0.76, a significant drop.
- "B2 divergence dip at e30" is the **pairwise expert-output divergence** D(e_i,e_j) = mean L2
  distance between expert action predictions on shared validation latents (`expert_diversity.json`,
  research_definition §7.3): off-diag mean e10 0.297 → **e30 0.139** → e100 0.214 → e200 0.231,
  coinciding with the e30 SR dip (0.62).

Consequence: **the success criterion for the V2 router is to beat uniform (0.0367 offline; the
uniform mode at rollout), not merely to beat the old learned router.** (Observation, hypothesis-
generating only: learned-routing damage concentrates in phase 0 — per-phase MSE 0.159 vs uniform
0.047 — the well-sampled phase with 161 samples; not yet explained.)

**L4 — Two distinct facts, one of which the review mis-attributed.**
From `latent_geometry.json` (n=1026, sigma=1.497), phase indices 0-indexed:

- (a) **Phases 2↔3 are genuinely confusable — this is well-sampled, not thin ice.** Both phases
  have ample data (n_per_phase = {0:161, **1:15**, 2:385, 3:352, 4:108, **5:5**}); inter-phase
  distance 2–3 = **3.410** vs intra-phase 7.329 / 7.210; silhouette 0.069 / 0.091 (mean 0.278).
  The review's "computed off a 15-sample cluster" concern applies to fact (b), not (a).
- (b) **Phases 1 and 5 are data-starved (15 and 5 samples).** Their silhouettes (0.590, 0.272)
  are individually unreliable — a 15-sample centroid is not a centroid.

Consequence: the cheap phase-merge check (gating experiment G2) tests **decomposition
granularity** (should phases 2/3 be one regime?), not statistical thinness. Soft labels remain
motivated for fact (b) (starved phases must not own experts) and for (a) only if G2 says the
2↔3 boundary is worth keeping at all.

**L5 — Checkpoint selection and balance tuning are not the bottleneck.**
A1 sweep: 0.72/0.74/0.62/0.78/0.78 (e3/e10/e30/e100/e200, all CIs overlap). Grid balance_coeff:
0.0 → 0.74, 0.01 → 0.72, 0.1 → 0.78 (flat). Confirmed unchanged this revision.

---

## Part 2: SOTA blocks (verified this revision)

1. **TGR-MoE (Kada et al., CVPR 2026)** — verified; the single most relevant external result for
   V2-D. A frozen dense teacher's representations drive a lightweight teacher router
   (trained on load-balancing + entropy only); the student router is KL-distilled to it.
   Findings that directly shape V2-D:
   - **Early-only distillation wins**: first-half distillation + task in second half = 78.13 vs
     77.81 (distill+task full) vs 77.39 (VMoE baseline). Our λ(t)→0 schedule is the right shape,
     independently confirmed.
   - **Teacher-routed inference is the upper bound** (80.19) and *student-router-only inference
     under pure distillation is worst* (74.84) — routing must keep task gradients; and the
     teacher-routed bound means our H5 oracle gap may legitimately persist.
   - Teacher features must be **layer-aligned with the student router** (last-layer teacher
     features: 75.83, worse than baseline). For us this is automatic: the phase head and router
     consume the same latent z_t.
2. **LAR-MoE (arXiv 2603.08476)** — verified; precision on its result: it **matches, not beats**,
   the phase-supervised MoE baseline on the surgical task. Validates annotation-free routing at
   parity, not superiority. Not evidence for beating 0.640.
3. **SMP (arXiv 2601.21251)** — sticky routing + orthogonal skill basis; the stickiness block
   (V2-C's L_sticky) is the transferable idea.
4. **CoRDE (arXiv 2606.21935)** — router trained by KL to a responsibility posterior, decoupled
   from the generative gradient; EM-updated soft concept→expert mapping. V2-B's M matrix and
   V2-D's decoupled router objective follow this design.
5. **FiLM phase-conditioning (Chen et al., IEEE/ASME T-Mech 2026)** — verified accepted; the
   "FiLM beats token-level conditioning" ablation could not be independently confirmed from the
   pulled text — treat that specific claim as plausible, not double-checked.
6. **Zang et al. (NeurIPS 2023)** — verified; no transition/bisimulation machinery anywhere in
   v2. Reason stands.

---

## Part 3: PhaseForge v2 — "Soft-Regime MoE with Teacher-Distilled Sticky Routing"

Skeleton unchanged (stage 1 → bootstrap → stage 2 → frozen eval). Five blocks, revised:

**V2-A. Soft regime targets, with class-balanced reweighting (fixes L4).**
Stage-1 phase head trains on soft targets (one-hot blended with a precomputed phase-similarity
prior) with **effective-number-of-samples reweighting** (Cui et al., CVPR 2019:
w ∝ (1−β)/(1−β^n)) — raw inverse frequency on the 5-vs-385 imbalance is a 77× ratio and is
rejected (would destabilize training on its own). Scope of this block is gated on G2: if the
merge test shows phases 2/3 should be one regime, a 5-phase decomposition replaces the soft
labels instead.

**V2-B. Soft phase→expert mapping, K=8 experts (fixes L4).**
`bootstrap_moe` gains a `soft` mode: hierarchical prototypes plus a P×E affinity matrix M
(Dirichlet-smoothed). **M is asserted right-stochastic (rows sum to 1) with a unit test** —
Mᵀ·softmax(phase_head(z)) is a valid target distribution for the KL term only then. Data-starved
phases 1/5 route through M to shared experts. M fixed in v2.0 (EM updates = v2.1).
**Gated on G4**: the L1 warm/centroid interaction was measured on the hard 6-centroid scheme and
must be re-verified under this block before L1's prescription is trusted.

**V2-C. History-conditioned router + sticky loss (fixes L3).**
Router input [z_t, z_{t-1}] (zero-padded at demo start; collator is trajectory-aware —
`padding_mask` exists); L_sticky penalizes consecutive-step gate drift. Cost: +latent_dim params.

**V2-D. Teacher-distilled routing, TGR-shaped schedule (fixes L3; the heart of v2).**

```
L = L_action + β·L_balance + λ(t)·L_KL(softmax(gate_linear([z_t, z_{t-1}])) ‖ Mᵀ·softmax(phase_head(z_t))) + γ·L_sticky
```

- Teacher = frozen stage-1 phase head on the **same latent z_t** as the router (layer-aligned,
  per TGR-MoE); λ(t) = λ0 for the **first half of stage 2, annealed to 0** for the second half
  (TGR's best schedule: early-only distillation).
- Router keeps its task gradient throughout (TGR: pure-distillation students underperform).
- Inference never uses the phase head — the autonomy claim is preserved; H5 (routing gap vs
  oracle) is measured at each λ schedule point.

**V2-E. Honest eval protocol.**
Rollout eval reports four router interventions (learned / sticky-smoothed window / uniform /
phase-head oracle — all evaluation-time only) plus per-phase SR. **Success criterion: the learned
mode must beat the uniform mode** (L3), not just the old router.

---

## Part 4: Gating experiments — run before writing v2 code

| # | Experiment | Cost | Decides |
|---|---|---|---|
| G1 | **`teacher_forced` clean re-check**: re-run teacher_forced + phaseforge in ONE wave on the same bank/commit (part3's existing teacher_forced = 0.52/0.66/0.40, mean 0.527 — below phaseforge 0.64, but part3's own `bc` collapsed to 0.02/0.00/0.02, so part3 cells are not trustworthy) | ~half day cloud | Whether V2-D is needed at all (TGR's teacher-routed upper bound makes this the highest-information check) |
| G2 | **Phase-merge / separability check** (offline, local, no training): separability of phases 2↔3 (and 1, 5) in normalized state space; does merging 2+3 remove the confusability? | afternoon, free | V2-A's scope: soft labels vs simpler merged decomposition |
| G3 | **Same-wave, same-bank re-baseline**: phaseforge + bc_large (recomputed param match) + kmeans routers in one wave, one commit | ~1–2 h cloud | L2's directional-vs-established status; also re-verifies the 0.640-vs-0.447 edge |
| G4 | **Warm-vs-reset under the V2-B soft scheme**, 3 seeds (after V2-B exists) | included in v2 runs | L1's interaction transfers to the new object |

---

## Part 5: Risks (revised)

1. **V2-D redundancy** — G1 is now runnable from existing data only partially: teacher_forced
   (0.527) does *not* reproduce phaseforge (0.64), but part3's broken `bc` cells make that
   evidence weak in both directions. G1 decides; if teacher_forced ≈ phaseforge in a clean wave,
   the fix is "route with the phase head at train+eval" and most of V2-A..D is dropped.
2. **L1 transfer** — the warm/centroid interaction is single-seed and hard-scheme; G4 re-checks it.
3. **Cross-part comparisons** (new, this revision): phaseforge/bc_large/kmeans means span
   different commits and data-config hashes; only same-wave deltas are established.
4. **New hyperparameters** λ0, γ, α, K — bounded by evidence: rn/balance flatness says small aux
   weights are safe; sweep only λ0; everything else at v1 values.
5. **Capacity** — recompute the matched dense baseline against v2's actual parameter count
   (history input + M ≈ +1–2% over v1), in the same wave as v2 (G3).
6. **V2 degrades gracefully** — the uniform eval mode remains the fallback (best per-step
   predictor in A5); no design bet is irreversible.

---

## Part 6: Codebase mapping (unchanged from v1, plus fixes)

- `phaseforge/data/ingestion/state_machine.py` + `phase_labeler.py` — soft-label mode (V2-A)
- `phaseforge/trains/loops/stage1_loop.py` — class-balanced CE (effective-number reweighting) + soft targets (V2-A)
- `phaseforge/models/phase_moe.py::bootstrap_moe` — `soft` init + right-stochastic M with unit test (V2-B)
- `phaseforge/models/components/router.py` — history input; `stage2_loop.py` — L_KL + L_sticky + TGR-shaped λ schedule (V2-C/D)
- `phaseforge/evaluations/rollout/runner.py` — router-mode flags + per-phase SR (V2-E)
- Config: one new model (`phaseforge_v2`) + overrides; eval determinism machinery already proven

---

## Part 7: Complete experiment inventory (verified this revision)

Every cell/number below was re-read from the raw JSONs on 2026-08-19. Real-matrix means are
3-seed means (seeds 42/43/44) of rollout SR; surgical cells are seed 42.

### 7.1 Real-matrix cells (the `experiments/lift_ablation.json` 23-method manifest)

| Wave (data_config_hash / commit) | Cell | SR per seed | Mean | Status |
|---|---|---|---|---|
| a2da6ba3 / c09270a | phaseforge | 0.68 / 0.74 / 0.50 | **0.640** | evaluated |
| a2da6ba3 / c09270a | scratch_moe | 0.58 / 0.66 / 0.52 | 0.587 | evaluated |
| a2da6ba3 / c09270a | warmstart_moe | 0.58 / 0.56 / 0.40 | 0.513 | evaluated |
| a2da6ba3 / c09270a | bc (part3) | 0.02 / 0.00 / 0.02 | **broken — do not cite** | evaluated, invalid |
| a2da6ba3 / c09270a | teacher_forced | 0.52 / 0.66 / 0.40 | 0.527 | evaluated |
| 6e529fe8 / c09270a | bc | 0.60 / 0.48 / 0.54 | 0.540 | evaluated |
| 6e529fe8 / c09270a | phase_pretrain_random_router | 0.56 / 0.52 / 0.48 | 0.520 | evaluated |
| 6e529fe8 / c09270a | plain_encoder_phase_bootstrap | 0.58 / 0.62 / 0.60 | 0.600 | evaluated |
| 89464860 / 282947d | bc_large | 0.46 / 0.52 / 0.36 | 0.447 | evaluated (EXP-103, param-matched +0.8%) |
| 89464860 / 282947d | pf_kmeans | 0.50 / 0.64 / 0.40 | 0.513 | evaluated |
| 89464860 / 282947d | pf_random_random | 0.56 / 0.58 / 0.38 | 0.507 | evaluated |
| 89464860 / 282947d | pf_spherical_kmeans | 0.60 / 0.58 / 0.38 | 0.520 | evaluated |
| 09a68c4c / 282947d | pf_centroid_random | 0.46 / 0.92 / 0.60 | 0.660 | evaluated — seed43 = 0.92 is a flagged outlier |
| 09a68c4c / 282947d | pf_spherical | 0.54 / 0.76 / 0.54 | 0.613 | evaluated |
| 09a68c4c / 282947d | pf_k3 | 0.56 / 0.72 / 0.54 | 0.607 | evaluated (K sweep) |
| 09a68c4c / 282947d | pf_k12 | 0.54 / 0.64 / 0.36 | 0.513 | evaluated (K sweep) |
| cb69d88f / 282947d | jitter00 | 0.70 / 0.62 / 0.46 | 0.593 | evaluated (wave-2) |
| cb69d88f / 282947d | jitter10 | 0.52 / 0.40 / 0.22 | 0.380 | evaluated (wave-2) |
| cb69d88f / 282947d | corrupt25 | 0.52 / 0.76 / 0.52 | 0.600 | evaluated (wave-2) |
| cb69d88f / 282947d | corrupt50 | 0.54 / 0.74 / 0.46 | 0.580 | evaluated (wave-2) |
| cb69d88f / 282947d | shuffle | 0.36 / 0.72 / 0.42 | 0.500 | evaluated (wave-2) |
| 88d7ae5b / 282947d | bc_large (part5) | 0.46 / 0.52 / 0.36 | 0.447 | evaluated — identical to part4/1 (determinism check) |
| 89464860 / 282947d | pf_phase_head | — | — | **incomplete** — stage2 dir exists, no `checkpoint_best.pt`, no summary |
| 09a68c4c / 282947d | pf_ft | — | — | **incomplete** — stage2 dir exists, no `checkpoint_best.pt`, no summary |
| — | bc_robot_only | — | — | **never run** — no outputs at all |

Manifest = 23 methods; 20 have valid eval means, 1 evaluated-but-invalid (part3 bc), 2 incomplete,
1 never run. Note: the wave-2 cells (jitter/corrupt/shuffle) reuse the clean phaseforge Stage-1
encoder with bootstrap overrides, per `research_definition.md`; jitter10 is the only wave-2 cell
below the group's dense-ish level (0.380), consistent with L1's init sensitivity.

### 7.2 Surgical study (seed 42, own frozen bank, all internally same-bank)

| Experiment | Key result | Status |
|---|---|---|
| A1 checkpoint sweep | e3/e10/e30/e100/e200 → 0.72 / 0.74 / 0.62 / 0.78 / 0.78; best_epoch=3; all CIs overlap; peak 0.78 | done (reconstructed from 5 eval summaries; `checkpoint_sweep.json` existed on VM, not copied) |
| A2 val-SR correlation | corr_all **+0.380** (5 pts); corr_early −1.0 (2 pts, degenerate); corr_late null (2 pts); sr_at_selected 0.72, peak 0.78 | done (`sr_val_corr.json`) |
| A3 validation banks | 4 banks, all SR 0.72 [0.583, 0.825], bit-identical val curves | **VOID** — HDF5 dataset filters short-circuit split RNG (`state_machine.py:654-677`); fix committed 41fc3bc, not rerun |
| A4 specialization matrix | diag_true **0.698**, diag_pred 0.776 (n=1026); dominant expert per phase = [0, 1, 3, 3, 4, 5]; phases 2/3 share expert 3 (0.502 / 0.600) — corroborates C1 | done (`specialization_matrix.json`) |
| A5 routing counterfactuals | learned 0.06003 > uniform 0.03664 > random 0.03568; oracle_true 0.07250, oracle_pred 0.07391 | done (L3) |
| B1 router-init × expert-init | 2×2 with CIs — interaction, not a single lever (L1) | done (four_way_init.json) |
| B2 expert diversity | off-diag D(e_i,e_j): e10 0.297 → e30 0.139 → e100 0.214 → e200 0.231 | done (L3 pinned scale) |
| B3/B4 grid | balance_coeff flat 0.74/0.72/0.78; rn 0.0→0.76, 0.1→0.72, 0.5→0.40 [0.276,0.538] | done (ablation_grid.json) |
| C1 latent geometry | phases 2↔3 confusable (inter 3.410 vs intra 7.329/7.210); n_per = {0:161, 1:15, 2:385, 3:352, 4:108, 5:5} | done (latent_geometry.json) |
| C2 failure-phase attribution | not produced — per-phase failure attribution unavailable | not done |
| Stage-2 training curves | balance → 0.985, entropy 0.93–0.95, collapse 0 after e10, NMI ~0.44–0.47, val loss 0.0270→0.0342 | done |
| Seed expansion | seed43 stage1 died at 43/100 (no checkpoints); seed44 never started; all surgical results are seed 42 | incomplete |

### 7.3 Integrity notes

- **Determinism holds**: part1 vs part2 surgical findings bit-identical; part4/1 vs part5 bc_large SR identical (0.46/0.52/0.36).
- **Cross-wave comparisons**: 6 distinct data-config hashes across 2 commits; bank identity not recorded in run_meta → only same-wave deltas are established (see L2).
- **Invalid cells**: part3 `bc` (0.02/0.00/0.02 — broken run; do not cite); `pf_centroid_random` seed43 0.92 (outlier, single-run flag).
- **Incomplete cells**: `pf_phase_head` (part4/1), `pf_ft` (part4/2) — stage2 training dirs without checkpoints; `bc_robot_only` never run.
- **A2's `checkpoint_sweep.json`** is referenced by `sr_val_corr.json` but was not copied from the VM; A1 numbers reconstructed from the 5 per-epoch eval summaries (identical values).

---

## Bottom line

The review's structural verdict stands: every block is tied to a measured failure. Its concrete
fixes are adopted (interaction restated with CIs, class-balanced reweighting, right-stochastic M,
capacity re-baselining, TGR-MoE-shaped schedule, teacher_forced as the first check). Two of its
inferences were corrected against the data: the 2↔3 confusability is well-sampled (not thin ice),
and the "rn=0.5" / "B2" numbers are now pinned to their exact scales. Two findings added this
revision: the real-matrix cells span six waves (phaseforge vs dense is directional only — G1/G3
must run before L1/L2 become load-bearing, G2 is free), and the full 23-method manifest is
inventoried in Part 7 — 20 evaluated cells, 1 invalid (part3 bc), 2 incomplete (pf_phase_head,
pf_ft), 1 never run (bc_robot_only), with the surgical study A1–A5/B1–B4/C1 complete except A3
(VOID, fix not rerun) and C2 (not produced).
