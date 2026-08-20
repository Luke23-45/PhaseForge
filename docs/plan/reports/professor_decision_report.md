# PhaseForge — Status Report and Decision Request

**To:** Supervisor (professor)
**From:** PhaseForge project team
**Date:** 2026-08-18
**Purpose:** Provide full context on the PhaseForge method, the evaluation protocol, a discovered-and-fixed evaluation bug, the corrected GPU re-run results, and an open statistical/design question (per-seed variance). The supervisor decides the path forward; the team will execute it.

---

## 1. What PhaseForge is

PhaseForge is a two-stage method for learning closed-loop manipulation policies from offline demonstrations, built on the robomimic `Lift` task (low-dimensional privileged state, 19-dim state → 7-dim actions).

### Stage 1 — Phase-supervised pretraining

1. **Phase labels from a rule-based labeler.** The state machine segments each demonstration trajectory into 6 phases using physical thresholds (gripper open/closed, end-effector velocity, minimum phase duration, median filtering). These labels are cheap, interpretable, and task-agnostic heuristics — not ground-truth semantics.
2. **Joint training.** A shared encoder (3×256 MLP, GELU, residual connections, latent dim 128, dropout 0.1) is trained with two heads simultaneously:
   - an **action head** (deterministic MLP, MSE loss), and
   - a **phase classification head** (CE loss, weighted `λ_phase = 1.0`).
   Total loss: `L = L_action + λ_phase · L_phase`.
3. **Checkpoint selection.** The best stage-1 checkpoint is selected on `val/loss_action` (the predeclared rule in the research definition), frozen, and passed to stage 2.

### Stage 2 — Phase-bootstrapped mixture of experts (MoE)

1. **Encoder is frozen.** Stage 2 keeps the stage-1 encoder fixed; only the router, experts, and action head are trained.
2. **Centroid router initialization.** The router is a sparse top-2-of-6 expert router (Shazeer et al. 2017 / GShard-style top-k noisy gating — noisy gate, auxiliary load-balancing loss; *not* Switch Transformer, whose defining contribution is top-1 routing). Its gate weights are initialized to **unit-normalized latent centroids of each phase** (computed by running the frozen encoder over the training data and averaging latents per phase). With L2-normalized inputs this makes the initial gate logits true cosine similarities between the latent and each phase centroid — each expert starts as "the router for one phase".
3. **Fine-tuning.** The full MoE (router + 6 experts + action head) is trained for 200 epochs with `λ_phase = 0` (no phase supervision in stage 2), LR 1e-4, cosine schedule, checkpointed on `val/loss_action`.

### Inference

At rollout time, the policy is: `state → frozen encoder → router picks top-2 experts → weighted expert action`. Routing is deterministic at eval (noise is train-only). The method's claim is that phase-supervised pretraining + centroid bootstrap gives the router a meaningful initialization that improves behavioral learning — without ever needing ground-truth phase labels at runtime.

### The controlled comparison matrix

The method is evaluated against a 2×2 factorial plus baselines:

| Cell | Encoder (stage-1) | Router init (stage-2) | Role |
|---|---|---|---|
| **PhaseForge** | phase-supervised | centroid | proposed method |
| Phase-Pretrain Random-Router | phase-supervised | random | H1: router-init effect |
| Plain-Encoder Phase-Bootstrap | plain BC | centroid | H2: phase-representation effect |
| Warm-Start MoE | plain BC | random | matched baseline |
| Scratch MoE | none (train from scratch) | random | behavioral baseline |
| BC-MLP | — | — | control floor |

Hypotheses (from the research definition): **H1** centroid router init helps; **H2** the phase-supervised encoder helps; **H3** PhaseForge improves task success over matched controls; plus a negative control (BC robot-only, ~0.01 success) showing the task is not solvable from proprioception alone.

### Evaluation protocol (frozen)

- 3 training seeds (42, 43, 44) × 50 paired eval episodes (all methods share the *same* 50 reset states; `reset_bank a7d3953c0afcf560`).
- Closed-loop robosuite rollouts, horizon 500; success = grasp + lift. Only failure category is `task_timeout` (zero policy failures / invalid attempts in all runs).
- Per-seed Wilson 95% CIs; across-seed spread is the training-uncertainty measure.
- All training checkpoints selected on `val/loss_action` (both stages) per the predeclared protocol.

---

## 2. What went wrong, and what was fixed

### The bug (discovered during the first evaluation)

The research definition predeclares checkpoint selection on **`best val/loss_action` for both stages**. The **stage-1 configs actually monitored `val/loss_total`** (action + phase + routing terms). Because the phase loss *explodes* during training (below), the `val/loss_total` monitor selected stage-1 checkpoints at epochs 1–2 — a barely-trained encoder — for the two methods that consume PhaseForge's stage-1 checkpoint (PhaseForge and Phase-Pretrain Random-Router). Stage 2 then froze that poor encoder and bootstrapped the router from its centroids, so the damage propagated. This was the root cause of:

- low means for the two affected methods, and
- the large per-seed rollout spread (stage-1 best-checkpoint action loss 0.0451/0.0404/0.0659 at epochs 2/2/1).

### The fixes (commit `c09270a`, pushed to `origin/master`)

1. **Monitor corrected** to `val/loss_action` in stage-1 configs (matches the predeclared rule and stage-2).
2. **Commit-gating** on the GPU runner — every run records and verifies the exact git commit; stale/mismatched pipelines are refused. All rerun cells carry `c09270a`.
3. **Fail-closed pinned metadata** — dataset/environment metadata must resolve to exact pinned versions; unresolved metadata refuses to run rather than silently defaulting.
4. **NaN monitor guard** — a NaN validation metric can no longer be selected as "best".
5. Reliability engineering: 509 passing tests (up from 502), ruff/mypy clean relative to baseline, and a preflight script validates all 165 training + 150 eval config cells compose and satisfy protocol rules (checkpoint monitor, phase counts, freeze flags, task matching, scheduler alignment).

### Determinism control

The 4 methods whose pipelines were *not* affected by the monitor bug (BC-MLP, Scratch MoE, Warm-Start MoE, Plain-Encoder Bootstrap) reproduce their old numbers **exactly** (e.g., Plain-Encoder 0.580/0.620/0.600 → 0.580/0.620/0.600). Only the 2 affected methods moved. This confirms the rerun machinery is deterministic and that the fix changed only what it was supposed to change.

---

## 3. Results — corrected GPU re-run at commit `c09270a`

### Rollout success (held-out, 50 paired episodes per seed)

| Method | s42 | s43 | s44 | mean | spread | old (buggy) mean | old spread |
|---|---|---|---|---|---|---|---|
| **PhaseForge** | 0.68 | 0.74 | 0.50 | **0.640** | 0.24 | 0.567 | 0.30 |
| Plain-Encoder Bootstrap | 0.58 | 0.62 | 0.60 | 0.600 | 0.04 | 0.600 | 0.04 |
| Scratch MoE | 0.58 | 0.66 | 0.52 | 0.587 | 0.14 | 0.587 | 0.14 |
| BC-MLP (floor) | 0.60 | 0.48 | 0.54 | 0.540 | 0.12 | 0.540 | 0.12 |
| Phase-Pretrain Random-Router | 0.56 | 0.52 | 0.48 | 0.520 | 0.08 | 0.460 | 0.32 |
| Warm-Start MoE | 0.58 | 0.56 | 0.40 | 0.513 | 0.18 | 0.513 | 0.18 |

95% Wilson CIs (pooled 150 episodes): PhaseForge **[0.561, 0.712]**, Plain-Encoder [0.520, 0.675], Scratch [0.507, 0.662], BC [0.460, 0.618], Phase-Pretrain [0.441, 0.598], Warm-Start [0.434, 0.592]. **All intervals overlap.** No pairwise difference is statistically significant at 3 seeds × 50 episodes.

### What the fix changed

- **PhaseForge: 0.567 → 0.640** (+0.073); **Phase-Pretrain: 0.460 → 0.520** (+0.060). The two bug-affected methods improved as predicted; all four unaffected methods are bit-identical to their old numbers.
- PhaseForge now ranks **first by mean** and is directionally ahead on every comparison: vs BC floor +0.10, vs Plain-Encoder +0.04, vs Scratch +0.05, vs Warm-Start +0.13, vs Phase-Pretrain +0.12.
- The old report's conclusion (§8: "controlled null, do not claim better manipulation") was based on pre-fix rollouts and is **no longer the appropriate conclusion**; it should be replaced by a directional-but-not-significant statement pending this review.

### Stage-2 routing quality (final validation, per seed)

| Method | NMI (42/43/44) | entropy | top1-balance | collapse |
|---|---|---|---|---|
| PhaseForge | 0.457 / 0.403 / 0.421 | 0.957 | 0.98 | 0% |
| Plain-Encoder Bootstrap | 0.403 / 0.355 / 0.338 | 0.956 | — | 0% |
| Scratch MoE | 0.258 / 0.255 / 0.297 | 0.915 | — | 0% |
| Warm-Start MoE | 0.221 / 0.235 / 0.230 | 0.872 | — | 0% |
| Phase-Pretrain Random-Router | 0.181 / 0.146 / 0.301 | 0.907 | — | 0% |

PhaseForge's routing structure is **stable across seeds and clearly the most phase-aligned** of all learned routers. Note: the old report's §8 conclusion was written against the buggy data; the routing/behavioral picture has changed materially.

---

## 4. The open question: per-seed variance of PhaseForge

The team's main concern (shared with the supervisor): PhaseForge's per-seed spread (0.24) is the largest of the six methods, even though its mean is highest. In robotics, a user cannot trust a policy whose success depends on the training seed, so this must be understood before any conclusion is drawn. The team investigated the artifacts thoroughly; here is what the evidence says.

### What the variance is NOT

1. **Not eval noise.** All methods share the identical 50 reset states; routing noise is disabled at eval; duplicate runs agree to ≤0.02. Seed-44's 0.50 is real policy behavior.
2. **Not router instability.** Routing metrics are near-identical across seeds (NMI 0.40–0.46, entropy ~0.95, balance ~0.97–0.99, collapse 0% for all three seeds). The phase structure is the *stable* part.
3. **Not the offline monitor.** Stage-2 best `val/loss_action` does not predict rollout (s42 0.0267→0.68, s43 0.0231→0.74, s44 0.0256→0.50). Offline metrics and closed-loop success are decoupled in this regime (documented in the previous report §6 as well).
4. **Not a catastrophically bad seed.** Only 4 of 50 episodes are "hard" (failed by all seeds). Seed-44 fails ~21 near-threshold episodes that others solve — but it also *wins* 2 episodes (9, 31) that both other seeds lose. Successes take ~45 steps; failures time out at 500. Success is a knife-edge: small policy differences flip many episodes.

### What the evidence shows it IS

Two compounding causes, both traceable to **stage 1**:

**(a) The phase head degrades during stage-1 training (negative transfer).**
In all three seeds, `val/loss_phase` *rises* over training from ~0.7 to **~2.5–2.7 — worse than random guessing** for a 6-way classifier (random CE = ln 6 ≈ 1.79) — while train phase accuracy reaches 0.96. The phase head overfits and its generalization is destroyed by the joint action+phase objective (`λ_phase = 1.0` puts a CE of magnitude ~1–2.6 against an MSE of ~0.03). This is the classic auxiliary-task conflict pattern documented in the multi-task literature (gradient conflict between main and auxiliary losses; see e.g. Du et al. 2018 gradient-similarity weighting, CAGrad, ForkMerge).

**(b) The checkpoint monitor sits on a flat plateau, so which phase-head quality gets frozen is a per-seed lottery.**
Stage-1 `val/loss_action` is within 2% of its best across epochs 11–38 (s43) / 31–38 (s42) — selection among those is effectively noise — but the phase-head loss at those epochs swings between 1.08 and 1.90. Each seed therefore freezes a different-quality phase structure into stage-2 (phase loss at the selected checkpoint: s43 1.083 → best rollout 0.74; s42 1.895 → 0.68; s44 1.443 → 0.50), and the centroid router inherits it.

The controls confirm this is **not inherent to the method**:
- Same phase-supervised encoder but **random** router init (Phase-Pretrain): spread 0.24 → 0.08 (the centroid init is what transmits the lottery).
- Same centroid bootstrap but a **plain, stably-trained** encoder (Plain-Encoder Bootstrap): spread 0.04 (a stable encoder removes the lottery).
- So the variance amplifier is the combination *phase-supervised encoder × centroid init* under a flat checkpoint-selection plateau — not the router mechanism, not the eval, not MoE training.

### Literature context

- **Agarwal et al. 2021 (NeurIPS outstanding), *Deep RL at the Edge of the Statistical Precipice***: with few runs, means are unreliable; recommended practice is reporting per-run results, interquartile mean (IQM), and stratified bootstrap CIs; ~5–10 seeds are typically needed to claim differences. By that standard, 3 seeds support at most a *directional* statement.
- **Auxiliary-task literature (Du et al. 2018; Yu et al. 2020 PCGrad; CAGrad; ForkMerge 2023)**: main/auxiliary gradient conflicts can degrade the auxiliary head and make the solution seed-dependent; the auxiliary weight λ is a first-order control on this.

---

## 5. Status of the evaluation matrix

| Cell | Status |
|---|---|
| PhaseForge + 5 baselines (Lift, 3 seeds) | ✅ run at `c09270a` (this report) |
| BC robot-only negative control | ✅ run previously at `d127980` (~0.01; unaffected by the bug) |
| Teacher-Forced routing (H4) | ❌ not run (previously failed on a now-fixed runner issue) |
| BC-RNN (temporal control) | ❌ not run |
| Oracle MoE | exists only as offline diagnostic (not a deployable method by design) |
| Other 4 tasks (Can, Square, Transport, ToolHang) | ❌ not evaluated |

---

## 6. Decision points for the supervisor

The team will execute whatever is decided. Options with honest trade-offs:

**D1 — Statistical posture.**
- (a) Report Lift as *directional, not significant* at 3 seeds (Agarwal-style reporting: per-seed table + IQM + bootstrap CIs), with the seed-variance analysis as a caveat.
- (b) Run additional seeds (e.g., 3 more, or up to 10) before any conclusion — costs GPU hours per seed × 6 methods.
- (c) Report as controlled null (the old §8 conclusion) — the team advises against this: it was derived from pre-fix data and the corrected rerun contradicts it.

**D2 — The seed-variance finding (stage-1 phase-head degradation).**
This is a genuine quality defect in the current training recipe, independent of the evaluation question. Options:
- (a) Accept as-is and document it.
- (b) Address it before any further compute: e.g., select stage-1 checkpoints on a phase-inclusive criterion, tune `λ_phase` down (auxiliary-conflict literature says λ is the first lever), or regularize the phase head. Any of these changes the recipe and therefore the numbers; they would need a small local validation before another GPU run (stage-1 on CPU is ~6 min/seed).
- (c) Investigate further before deciding (e.g., gradient-conflict measurement between the two stage-1 losses).

**D3 — Completeness of the matrix.**
- Run the missing cells (Teacher-Forced, BC-RNN) on the existing 3 seeds to close the factorial, and/or
- accept Lift-only as the scope of the claim (the research definition ties the behavioral claim to a majority of tasks).

**D4 — Reporting.**
The previous report (§8) should be superseded by this document's §3 numbers with the supervisor's D1 decision encoded.

---

## 7. Artifacts (all in-repo)

- This analysis's evidence: `outputs/part1/outputs/`, `outputs/part2/outputs/` (runner state `_runner/state.json` — all cells `completed` at `c09270a`; per-episode logs, `rollout_summary.json`, `training_curves.jsonl`).
- Analysis scripts: `scripts/compare_rerun_report.py`, `scripts/seed_dependence_deepdive.py`, `scripts/episode_crosscheck.py`, `scripts/phase_head_at_selection.py`, `scripts/collect_rerun_results.py`.
- Previous report (superseded numbers): `docs/plan/reports/lift_rollout_eval_report.md`.
- Research definition and protocol: `docs/plan/specs/research_definition.md`, `docs/plan/specs/state_only_rollout_implementation_plan.md`.
- Fix commit: `c09270a` (monitor fix, commit gating, fail-closed metadata, NaN guard; 509 tests passing).
- GPU runbook: `docs/plan/specs/gpu_rerun_runbook.md`.