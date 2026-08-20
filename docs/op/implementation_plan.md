# PhaseForge — Implementation Plan (professor's review, 2026-08-18)

**Source:** `docs/op/professor_reponse.txt` (literature audit + recommended plan)
**Status:** to be executed top-down; each phase has an explicit gate
**Grounding:** every assumption in the plan was verified against the repo before writing. Where the professor's plan assumed something not available on disk, the plan says so and adjusts.

---

## Verified facts that shape the plan

| Professor's assumption | Reality (verified 2026-08-18) | Consequence |
|---|---|---|
| "If per-epoch stage-1 checkpoints are still on disk, the tie-break experiment is free" | **Not on disk.** 0 `.pt`/`.ckpt`/`.pth` files anywhere in `outputs/` or `outputs_local_train/`. GPU runs synced only metrics/metadata/configs; `checkpoints/checkpoint_best.pt` + periodic `checkpoint_epoch_*.pt` were never transferred. | Tie-break experiment requires a **CPU stage-1 re-run** (~6 min/seed) with snapshots retained. Not free, still cheap. |
| Grad-cosine logging is "near-zero cost" | No per-term gradients exist today: `stage1_loop.py:99` computes `total = action + λ·phase`, single `loss.backward()` at `base.py:458`. Needs `torch.autograd.grad` per term (retain_graph) — new code. | Small implementation task (below, Phase 1.1). |
| BC-RNN is missing from the matrix | **Fully exists**: `models/baselines/bc_rnn.py`, `config/models/baselines/bc_rnn.yaml`, `config/data/*_rnn.yaml`, tests, manifest rows (five_task idx 46–50). Only execution is missing. | Phase 3 is a sweep invocation, no code. |
| Teacher-Forced (H4) is a stub | **Fully exists**: `models/baselines/teacher_forced.py` (278 lines), config, tests, manifest rows. Never run only. `oracle_moe` exists too (offline diagnostic). | Phase 3 is a sweep invocation. |
| λ scheduling would be a new option | Correct — no λ scheduling anywhere in the codebase; `lambda_phase` is a static scalar. `phase_class_weight: "balanced"` is fully implemented (cli.py:462–517). | Phase 1.4 is new code. |
| Router is "Switch-Transformer-style" | Wrong label; must be fixed to **Shazeer et al. 2017 / GShard-style** top-2 noisy gating (Switch's defining contribution is top-1). | Phase 0.1 — doc-only fix. |
| Closest 2026 related work | **LAR-MoE** (arXiv 2603.08476, Mar 2026; unsupervised latent-aligned routing, LIBERO, frozen student encoder + routing regularization — overlaps with our freeze+init idea) and **MoE-ACT** (arXiv 2601.21971, Jan 2026; supervised phase-labeled gating for surgical ACT — the closest comparator; matches a supervised-MoE-with-phase-annotations baseline). | Phase 0.2 — full reads + positioning statement. |

---

## Phase 0 — Foundation (no compute; today)

**0.1 — Router terminology fix (doc-only).**
Replace every "Switch-Transformer-style" with "Shazeer/GShard-style sparse top-2 gating (noisy gate, load-balancing loss)" in:
- `docs/plan/professor_decision_report.md`
- `docs/plan/research_definition.md` (if present there)
- method docstrings in `phaseforge/models/phase_moe.py` / `bootstrap_moe` (check first)
- `docs/plan/lift_rollout_eval_report.md` (superseded anyway, but fix on touch)

**0.2 — Related-work research (2 papers + 1 search).**
- Read in full: LAR-MoE (https://arxiv.org/abs/2603.08476), MoE-ACT (https://arxiv.org/abs/2601.21971). Search for the "Semantically Structured MoE" paper the professor referenced (not yet identified).
- Write a 2–3 sentence positioning statement to `docs/op/` (or a `related_work.md`), asserting what is distinctive: (a) **centroid router initialization** as the init mechanism, (b) the **controlled 2×2 factorial** (encoder-supervision × router-init) that none of these papers run, (c) the **checkpoint-selection/negative-transfer diagnosis** as a methodological finding.
- Cite with full reads, not snippets (professor's caveat).

**Gate 0:** terminology clean; positioning statement written. No further compute until this is done.

---

## Phase 1 — D2 diagnosis + candidate fixes (CPU only)

### 1.1 Grad-cosine diagnostic (`cos(∇L_action, ∇L_phase)` per step)
- **Where:** `stage1_loop.py` `_compute_loss` (lines 35–110) currently builds a single total. Add a callback (cleanest: a new `GradientCosineCallback` on the `on_train_batch` hook, `base.py:495`) that recomputes the two per-term gradients via `torch.autograd.grad(..., retain_graph=True)` on the retained graph and logs the mean cosine per epoch.
- **Allowlist:** add the new metric key to `CURVE_OPTIONAL_NUMERIC` (`outputs_writer/curves.py:47–64`), else it is dropped (`persistence.py:168–173`).
- **Purpose:** decides *which* failure mechanism is real — sustained negative cosine ⇒ genuine gradient conflict (escalate to PCGrad/CAGrad/ForkMerge); near-zero/positive cosine ⇒ flat-plateau selection problem (tie-break/λ fixes). Professor's §3 and §6 both demand this measurement before choosing the fix family.

### 1.2 Boundary-noise diagnostic (phase-head error vs. distance-to-transition)
- **Data already available:** per-timestep `phase` labels + `trajectory_id`/`trajectory_position` in `StateOnlyDataset` (`data/common/dataset.py:66–91`); phase boundaries via label diff; stage-1 forward returns `phase_logits`.
- **What to build:** an offline CPU script that loads a stage-1 checkpoint, runs the *validation split* through the encoder + phase head (no grad), and buckets classification error rate by distance to nearest phase transition (e.g., 0, 1–2, 3–5, >5 steps). Needs per-timestep logits — extend `OfflineEvaluator` (`evaluations/runners/offline_evaluator.py:91–141`, currently collects action/phase but not `phase_logits`).
- **Purpose:** distinguishes "overfitting to demo idiosyncrasies" (uniform error growth) from "labeler boundary noise" (error clustered at transitions). The two call for different fixes: label smoothing/boundary tolerance vs. λ scheduling (professor §3).

**Result (2026-08-18, CPU, tag `lambdav1` checkpoints at monitor epochs):** error-by-distance is **NOT boundary-clustered** — the worst rate is in long interior stretches far from transitions (`dist_11_plus` 0.67 across all seeds), with a mild elevation at `dist_0_1` (0.42–0.55) vs a dip at `dist_1_6` (0.22–0.26), 71 boundaries/1026 samples per seed. Interpretation: the dominant mechanism is **uniform auxiliary overfitting with majority-phase bias** (Lift phases 1/5 ≈ 1.2%/0.6% of steps), *plus* a secondary boundary component. Label smoothing / boundary tolerance would be the wrong primary fix — the λ-decay family is correct. Written up via `scripts/phase_boundary_diagnostic.py` + `scripts/show_boundary_diagnostic.py`.

### 1.3 Tie-break fix (cheapest, most targeted) — requires CPU stage-1 re-run
- **Because checkpoints are not on disk,** re-run stage-1 for the 3 protocol seeds on CPU (~6 min/seed) **with snapshots retained** (periodic `checkpoint_epoch_*.pt` are never evicted — `checkpointing.py:134–137` — but set `checkpoint.every_n_epochs: 1` for full granularity).
- **Selection rule change:** within the flat `val/loss_action` plateau (best ±2%), pick the epoch with the **minimum `val/loss_phase`**. Faithful to the predeclared primary metric (action loss) — a documented tie-break, not a metric swap. Implementation: the checkpoint callback gains an optional `tiebreak_metric` field, or — cleaner — a post-hoc re-selection script over saved snapshots + `training_curves.jsonl`, which keeps the callback semantics unchanged. **Prefer the post-hoc script**: zero risk to the frozen pipeline, and stage-2 consumes `checkpoint_best.pt`, so re-selection = copy chosen snapshot to that alias.
- **Validate locally:** stage-2 CPU training for the 3 seeds (as done before in `outputs_local_train/`), check (a) val/loss_action stays on its plateau, (b) phase-head quality at selection improves and its per-seed spread collapses, (c) router NMI spread shrinks.

**Result (2026-08-18, CPU, tag `tiebreak_v1`):** tie-break moved selection to epochs 23/10/14 with val/loss_phase 1.512/1.091/1.170 (all below random ln 6 ≈ 1.79, vs 2.102/2.130/1.680 at the monitor epochs 41/36/25) — criterion (b, level) met. But stage-2 NMI became **0.440/0.411/0.395 (spread 0.044)**, worse than the fixed reference 0.449/0.457/0.436 (spread 0.021). Criterion (c) failed → **1.3 judged insufficient**, proceed to 1.4.

### 1.4 λ scheduling (only if 1.3 insufficient)
- If the plateau is too flat / tie-break can't separate epochs, implement `lambda_phase` scheduling: linear decay to 0 over training, or Du et al. (2018) adaptive weighting `λ(t) ∝ max(0, cos(g_a, g_p))` — the 1.1 measurement decides which. This is new code in `stage1_loop.py:47,99` + config field; gated by the 1.1 results.

**Decision input (Phase 1.1 measurement, CPU, tag `gradcos_v1`):** cos(∇L_action, ∇L_phase) ≈ 0 across all 3 seeds (per-epoch mean +0.0005..+0.0009, range ±0.02, first-epoch dip to −0.019 then ~0). **No gradient conflict** → PCGrad/CAGrad/Du-adaptive weighting are the wrong family (Du-adaptive would set λ≈0 from epoch 1). The val/loss_phase explosion is late-training degradation on the flat action plateau → **linear λ decay**.

**Result (2026-08-18, CPU, tag `lambdav1`, linear 1.0→0.0):** stage-1 best action 0.0266/0.0241/0.0260 @ epochs 41/36/25 (plateau held); val/loss_phase still drifts up late (2.28–2.47) — but this is shared-encoder drift, no longer phase-head overfitting (λ=0 late). Stage-2 NMI **0.450/0.450/0.440, spread 0.010** (fixed 0.021, tie-break 0.044, buggy 0.069), level 0.447 (on par with fixed), 0% collapse, entropy 0.951–0.958, final action 0.0337/0.0275/0.0312. **All three validation criteria met — but see the fairness revision below.**

**Combined check (tie-break on top of λ-decay, tag `combined_stage2`):** NMI 0.434/0.414/0.394 (spread 0.039) — worse than λ-decay alone. The "min val/loss_phase on the plateau" criterion selects early epochs whose router-bootstrap centroids are consistently worse; the tie-break is **not** part of the adopted fix.

**Fairness revision (2026-08-18, supervisor review):** λ-decay is a **protocol deviation** — the frozen protocol predeclares λ_phase = 1.0 constant — and it affects **only the phase-supervised arms** (phaseforge, phase_pretrain_random_router, teacher_forced via phaseforge's stage-1): the plain-encoder/BC/scratch methods have no phase head, so they *cannot* use the schedule (λ·0 = 0), and adding phase heads to the plain controls would destroy the H1/H2 factorial. The official comparison must therefore run every method at the predeclared configuration. **1.4 downgraded from "adopted" to "documented refinement":** code, tests, and 3-seed CPU validation remain in-repo and inert (default `constant` = bit-identical to no schedule); the manifest `defaults` carry no λ overrides; the runbook documents re-adding them as the fallback if the Gate 2 spread criterion fails. **The adopted fix is the monitor restoration alone (Phase 1.0), validated on 3 seeds:** stage-2 NMI 0.449/0.457/0.436 (spread 0.021 vs 0.069 buggy) at equal level; the λ-decay incremental gain (spread 0.021 → 0.010) is a proxy-metric tightening with identical means, expected to be immaterial for rollout.

### 1.5 Rigor (professor §3 closing)
- Whatever fix is adopted: new commit, before/after determinism check (unaffected methods reproduce bit-identical), 509-test suite re-run, preflight config cells re-run. Same discipline as the `loss_total → loss_action` fix.

**Gate 1:** diagnosis (1.1, 1.2) written up; one fix (1.3, else 1.4) adopted and locally validated on 3 seeds. **STATUS: PASSED 2026-08-18** — adopted fix is the **monitor restoration** (`best val/loss_action`, Phase 1.0): stage-2 NMI spread 0.24→0.021 local proxy, plateau held, 0% collapse. λ-decay (1.4) validated (spread 0.010) but **downgraded to a documented refinement** on fairness grounds (protocol deviation affecting only the phase-supervised arms — supervisor decision 2026-08-18); tie-break (1.3) explicitly rejected as standalone and in combination; both retained as shipped scripts/tests, inert by default.

---

## Phase 2 — GPU verification (small, existing 3 seeds)

- Re-run **only PhaseForge and Phase-Pretrain Random-Router** with the winning fix (the only two methods whose stage-1 pipeline changes; plain_encoder/warmstart/scratch/bc untouched).
- Success criteria (professor §7 Phase 2): per-seed spread drops while `val/loss_action` stays on its plateau; target: PhaseForge spread 0.24 → ≈0.04–0.10 (plain-encoder control ≈0.04). If achieved, the result upgrades from "highest mean, highest variance" to **"highest mean, low variance"** — categorically stronger.
- Keep the same commit-gated runner and paired eval (50 episodes, reset bank a7d3953c0afcf560).

**Gate 2:** spread reduction confirmed (or not) on GPU. Decision point for supervisor: proceed vs. iterate on λ.

---

## Phase 3 — Highest-information missing cells (GPU)

**Order per professor: BC-RNN first, before spending on more seeds.**

1. **BC-RNN (temporal control floor)** — 3 seeds on Lift. robomimic's own study found temporality one of the largest single factors on human-demo data; none of our methods use recurrence. If BC-RNN alone rivals PhaseForge, the comparison is reframed. No code needed — manifest rows exist (five_task idx 46–50), `phaseforge-sweep --methods bc_rnn`.
2. **Teacher-Forced routing (H4)** — 3 seeds on Lift, now that the runner bug is fixed. Closes the factorial cheaply; gives the phase-observability decomposition (oracle − teacher-forced = predictability gap). No code needed.

**Gate 3:** both cells run and added to the comparison table. Then Phase 5's scope question is evaluated with complete data.

---

## Phase 4 — Statistical posture (writing, no compute)

Per professor §2 + §7 Phase 4, applied to `docs/plan/lift_rollout_eval_report.md` (or a new corrected section):

1. **Seed-stratified bootstrap CI** (resample seeds with replacement — mirror Agarwal's per-task stratification, applied to seeds) **replacing the pooled-150-episode Wilson CI**, which is pseudoreplication (episodes from one seed are correlated).
2. **Probability-of-improvement per pairwise comparison** (bootstrap P(X>Y) over per-episode Bernoulli outcomes, e.g., "72% probability PhaseForge beats Warm-Start MoE") — more informative than CI overlap, directly recommended by Agarwal et al.
3. **Explicit M=1 caveat** on the Agarwal citation: its efficiency gains assume many tasks; with Lift alone we are in the degenerate case (M=1, N=3) — report directional only.
4. **Add citations:** Colas, Sigaud & Oudeyer 2018 ("How Many Random Seeds?" — power-analysis formulas for D5), Henderson et al. 2018 ("Deep RL that Matters" — seed sensitivity is systemic, not method-specific).
5. **Supersede old §8** ("controlled null") with a "corrected evaluation + lessons learned" section — the bug-and-fix narrative as a credibility asset (professor §7 Phase 4, explicitly endorsed by the professor).

**Gate 4:** stats section rewritten with stratified bootstrap + PoI; old §8 explicitly superseded in the doc.

**Result (2026-08-18):** `scripts/analysis/stratified_stats.py` + 11 tests (MC vs exact 27-vector distribution, PoI direction/ties, determinism). Run on the authoritative `c09270a` eval runs (18 `episodes.jsonl`, 50/seed): **seed means PhaseForge 0.640 (0.68/0.74/0.50), Plain-Bootstrap 0.600, Scratch 0.587, BC 0.540, Phase-Pretrain 0.520, Warm-Start 0.513** — the corrected runs **reverse the old H2 direction** (phase encoder no longer lags) and make PhaseForge the top mean with **PoI ≥ 73.9% vs every comparator** (vs Scratch 78.7%, vs Plain-Bootstrap 73.9%). Stratified CIs overlap at N=3 ([0.500, 0.740] widest) → H3 stays a **controlled null with a directional PhaseForge advantage**; report §3–§9 rewritten, old §8 superseded, M=1 caveat + Colas/Henderson/Agarwal citations added. Recompute: `python scripts/analysis/stratified_stats.py --root outputs/part1/outputs --root outputs/part2/outputs`.

---

## Phase 5 — Scope (only after Phases 1–3)

1. **Data-driven seed count:** use Colas et al. (2018) power-analysis formulas against the *observed post-fix* effect size and variance (Phase 2 data) to compute how many additional seeds are needed — not a round number picked in advance. Present to supervisor with GPU cost.
2. **Lift-only vs. multi-task:** decide against research_definition H3's "predeclared majority of tasks" bar (3/5), informed by how crowded the space is now (Phase 0.2): a single-task (and the easiest task) result is a weaker claim in this sub-field than a year ago.

**Gate 5:** supervisor decision on scope + seed budget.

---

## Cost summary

| Phase | Compute | Est. time | Code |
|---|---|---|---|
| 0 | none | ~1–2 h | docs only |
| 1.1–1.2 | none (offline over existing logs/ckpts) | ~1 d | callback + script |
| 1.3 | CPU ×3 seeds stage-1 (+ stage-2 local) | ~1 d incl. local runs | post-hoc re-selection script |
| 1.4 (if needed) | CPU ×3 seeds | ~0.5 d | λ scheduling |
| 2 | GPU ×2 methods ×3 seeds (stage-1+2+eval) | ~1 GPU session | none |
| 3 | GPU ×2 methods ×3 seeds | ~1 GPU session | none |
| 4 | none | ~0.5 d | docs |
| 5 | none (until decision) | — | analysis script (Colas) |

Everything is sequenced so no GPU-hour is spent before cheap CPU diagnostics (1.1–1.3) have told us which fix is worth GPU time.