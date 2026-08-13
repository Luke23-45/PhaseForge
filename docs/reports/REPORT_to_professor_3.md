# PhaseForge — Decision Request: Episode Budget for the Evaluation Matrix

**To:** Prof. [Name]
**From:** [Student]
**Date:** August 13, 2026
**Subject:** Request for sign-off on the episode budget for the `libero_90` evaluation matrix — proposal to adopt a 10-episode/task protocol (Option A) with a two-stage escalation tier, per the locked-protocol rule that no change to episode counts happens without your explicit decision.

> **Revision (2026-08-13, per your review):** LeRobot citation scoped precisely (its 10-eps recommendation covers its four standard suites, not `libero_90`); Flow-Matching-LIBERO citation removed (unverifiable); seed-lottery source verified (arXiv:2606.13856 resolves to the paper — your earlier resolution failure likely coincided with arXiv's Aug 4–5 maintenance window) with the regime caveat added; escalated pairs are now specified as clearly labeled follow-up comparisons, never pooled with the base-matrix statistics (interim-analysis rule).

---

## 1. Executive Summary (the ask)

The locked protocol (`evaluation_plan.md` §1) fixes `libero_90` at 50 episodes/task. We request a decision on that budget. This report presents verified measurements, a statistical argument, and published precedent showing that for **our** research question — a controlled comparison of 8 training-strategy cells that explicitly does not claim leaderboard-comparable numbers — lowering the in-distribution suite to **10 episodes/task** (a 10-episode protocol: LeRobot's published recommendation uses 10 eps/task for its four standard suites; we extend the same episode count to `libero_90` for our internal matrix only) does not materially change any decision the protocol is meant to support, while cutting evaluation wall-time ~4.5×. We recommend:

- **Option A (recommended):** matrix at 10 eps/task × 3 seeds; any head-to-head that lands within ~3 pp is escalated to 20 eps × 5 seeds for that pair only, reported as a clearly labeled follow-up comparison (never pooled with the base-matrix statistics).
- **Option B:** conservative middle — 20 eps/task × 3 seeds (SE nearly equal to 50 eps; 2.4× cheaper).
- **Option C:** status quo — 50 eps/task × 3 seeds.

Everything else in the locked protocol (suites, max steps, hard reset, suite-role labels, seeds for the matrix, mean ± std + bootstrap CI + probability-of-improvement reporting) stays exactly as you approved it.

---

## 2. What We Are Testing (research recap)

PhaseForge is a **training-strategy study**, not a new-architecture or new-perception study: *does bootstrapping a MoE router from phase centroids in a frozen, phase-supervised latent produce more stable routing, cleaner specialization, and better policy behavior than scratch MoE, warm-start MoE, or supervised routing?*

- **Stage 1:** generalist BC with an auxiliary rule-based phase head (training-only signal; never an input at inference).
- **Stage 2:** frozen encoder, 6 experts initialized from the stage-1 action head, router weights initialized from phase centroids in the stage-1 latent.
- **Matrix:** 8 cells — `bc`, `scratch_moe`, `warmstart_moe`, `phaseforge` (proposed), `oracle_moe` (signature-only, non-deployable), `teacher_forced` (privileged-training bound), `phase_pretrain_random_router`, `plain_encoder_phase_bootstrap`.

**Where the paper's claims sit (novelty_claim.md §3, REPORT_to_professor_2 §1.1):** primarily on the **MoE routing diagnostics** — phase–expert NMI, routing entropy, expert balance, collapse rate, time-to-stable-routing — with rollout success rate as the secondary axis. Two consequences matter for the protocol decision:

1. The routing diagnostics are trajectory-level statistics computed over *every control step* of every rollout. At 10 eps/task, one task yields up to 10 × 400 = **4,000 state samples**; suite-level routing statistics are over ~360k samples. Episode count barely affects these — they are the paper's primary claims.
2. The evaluation is a **paired comparison**: episodes are deterministic functions of the fixed LIBERO init-state list (`get_task_init_states`), so every cell sees the *identical* episode instances. This is the community-standard design (identical episodes across checkpoints — see §6) and it means differences between cells are measured on the same test set, not on independent samples.

---

## 3. Where We Are With Perception (the vision path) — and why it removes the need for the 50-episode protocol

Your staged plan (REPORT_to_professor_2 §4) is exactly what we are executing:

- **Stage 1 — object-state channel (implemented, in current eval runs).** The 23-DoF proprioceptive state is now extended with the per-task object-state channel (positions + quaternions + occupancy mask, `k_slots=8`). The eval env reads the same census-built index as ingestion, so train/eval schemas match by construction. The current harness logs confirm the channel is live ("Object-state channel enabled … 4 object(s)"). This is a **state-oracle sanity check** of the MoE architecture — explicitly labeled, never reported as a leaderboard number.
- **Stage 2 — cached frozen visual embeddings (next).** A small frozen encoder runs over the dataset images once; the MoE architecture trains against cached vectors. Perception becomes one-time preprocessing instead of a recurring cost.
- **Stage 3 — end-to-end vision (later, only if needed).** Only for the winning cell, and only if we ever need numbers comparable to the published leaderboard.

The decisive point for this protocol question: **the paper explicitly does not claim vision-level performance or LIBERO leaderboard numbers** (novelty_claim.md §4, §6). The 50-episode protocol exists to make numbers *comparable to published results* (OpenVLA/Pi0.5 report 500 trials/suite). We never compare against those numbers — our comparison is internal, across 8 cells under one identical protocol. The most expensive episode count in the standard protocol is priced for a claim we have deliberately excluded.

We also state the current status honestly — and it is important for interpreting any current number: **the model has not been trained with vision at all.** Every checkpoint evaluated so far was trained and run with the object-state oracle channel (§3, Stage 1) as the perception input; visual input does not exist anywhere in the training or evaluation path yet. A state-oracle policy is, by construction, **known not to pass the LIBERO evaluation** as a vision benchmark — we expect it to fail, and we say so up front. The current runs (0–5% SR on the first tasks of the last full run; the phase-profile run used a randomly-initialized checkpoint) are therefore **harness sanity checks and training-sanity checks, not performance signals** — they validate that episodes, per-task schema alignment, and phase instrumentation are correct, and that training runs end-to-end. They say nothing about the architecture's ceiling. Any SR the object-state cells produce is a deliberate state-oracle sanity bound, explicitly labeled, never reported as a benchmark number (novelty_claim.md §6). Clearing a meaningful floor is a **training** question that only becomes answerable when the cached-embedding/vision path (§3, Stage 2) lands; it is independent of the protocol decision. The decision you are making now is about the cost of the comparison once real perception input is in the loop.

---

## 4. Verified Facts: What the Harness Measures

We instrumented the rollout path (per-phase wall-clock timers in the env, per-task phase summary in the evaluator, and a standalone profiler `scripts/profile_rollout.py`). Measured on free Colab T4 (2 vCPU, 2 workers):

| Phase | Cost per episode | Share of 14.5 s (serial) |
|---|---|---|
| Physics (MuJoCo, 25 substeps/step, 400 steps) | ~11.2 s | 78 % |
| Hard reset (sim XML rebuild + compile) | ~2.4 s | 17.5 % |
| Settling steps (10 dummy steps) | ~0.26 s | 1.8 % |
| Model inference | ~0.45 s | 3.1 % |
| Success check / state extraction / init-state set | < 0.05 s | < 0.5 % |

- Serial: **14.5 s/episode**. Two workers (CPU contention): **~25 s/episode per worker → ~12.5 s of wall-clock per episode**. These match the 13 s/episode estimate already in `run_multi_seed_eval.py`.
- Full matrix at 50 eps: 8 cells × 3 seeds × 4,600 eps (4,500 + 100) = 110,400 episodes ≈ **16 eval-days wall**; plus ~4 days training ≈ 20 days total — consistent with the 2–4 week estimate you were given.
- The dominant cost (physics) is irreducible without breaking the benchmark; the harness itself adds < 0.5 % overhead. This is not a "we can't afford it" argument — it is a cost–benefit argument, quantified below.

---

## 5. The Statistical Argument: Why Lowering Episodes Does Not Materially Change Any Decision

### 5.1 The uncertainty that matters is the training seed, not the episode count

The protocol's statistic is the **mean over 3 training seeds**. Its standard error has two components:

```
SE(3-seed mean) = sqrt( σ²_seed / 3  +  σ²_eval / 3 )
```

- `σ_eval` — suite-mean sampling noise at p = 0.5: **1.67 pp at 10 eps/task** (900 trials), 1.18 pp at 20 eps, **0.75 pp at 50 eps** (4,500 trials).
- `σ_seed` — success-rate spread across independently-trained checkpoints.

Measured seed spread in this literature: the seed-lottery study (**arXiv:2606.13856**, Sam & Tsetserukou, 2026 — verified 2026-08-13: the ID resolves to the paper; your earlier resolution failure likely coincided with arXiv's Aug 4–5 maintenance window) reports **std ≈ 7.5 pp across 13 training seeds on LIBERO-Object** (65.2–94.2 %, one collapsed seed, 29 pp span), evaluated at **500 episodes/suite** (10 tasks × 50 trials, per their evaluation protocol). **Regime caveat (per your review):** that study fine-tunes a pretrained foundation VLA (VLA-JEPA, frozen encoders) end-to-end on a single RTX 5090 — a different regime from PhaseForge's from-scratch, bootstrapped MoE cells; we use it as a conservative external anchor for how large seed variance can get, not as a measurement of our setup. Even if our small-scale cells were an order of magnitude more seed-stable than that (σ_seed ≈ 0.7 pp), the arithmetic is:

| Training-seed std σ_seed | SE of 3-seed mean @ 10 eps | SE of 3-seed mean @ 50 eps | Gain from 5× episodes |
|---|---|---|---|
| 0.5 pp (unrealistically stable) | 1.01 pp | 0.52 pp | 0.49 pp |
| 1 pp | 1.12 pp | 0.72 pp | 0.40 pp |
| 3 pp | 1.98 pp | 1.79 pp | 0.19 pp |
| 7 pp (measured in seed-lottery study) | 4.15 pp | 4.06 pp | 0.09 pp |

**Five times the compute buys at most half a percentage point on the statistic the protocol actually reports.** Above ~1 pp of real seed variance, the gain is negligible — and the CI widths your table format (mean ± std + bootstrap over 3 seeds) will show are seed-dominated either way.

### 5.2 What the protocol is deciding, and the precision it needs

The matrix supports three decisions: (a) which cells clear a floor, (b) the phaseforge vs warmstart/scratch/teacher-forced head-to-heads, (c) the oracle signature sanity. At 10 eps:

- **Suite means**: SE ≤ 1.7 pp (p = 0.5). A rule of thumb — treat suite-level gaps < ~3 pp as statistically indistinguishable at this stage. At 50 eps the rule tightens to ~1.5 pp — but per §5.1, that tightening rarely survives the seed component of the CI.
- **Per-task breakdowns** (the protocol's mandated report): at 10 eps, per-task rates quantize to 10 pp bins (11 levels). Adequate for flagging catastrophic tasks (0/10) vs solvable tasks, coarse for fine per-task claims. This is the honest cost of Option A, mitigated by the escalation tier (§7).
- **Head-to-head tests** (probability-of-improvement, Mann–Whitney U over per-task rates): rank resolution is reduced by quantization — the same mitigation applies.

### 5.3 The cost asymmetry

| Option | Episodes/cell-seed | Matrix (8×3) | Eval-days | Training | Total |
|---|---|---|---|---|---|
| **A: 10 eps** | 1,000 | 24,000 | **~3.5** | ~4 d | **~7.5 d** |
| **B: 20 eps** | 1,900 | 45,600 | **~6.6** | ~4 d | **~10.5 d** |
| **C: 50 eps** | 4,600 | 110,400 | **~16** | ~4 d | **~20 d** |

---

## 6. Published Precedent

- **LeRobot (HuggingFace), the reference LIBERO evaluation stack:** their docs recommend **10 episodes per task × 3 seeds** with same-episode paired comparisons (`--seed`, `--env.init_states=true`), matching their published-results protocol — **but explicitly scoped to their four standard suites (Spatial, Object, Goal, Long = 400 episodes); `libero_90` is not among them.** We extend the same episode count to our in-distribution `libero_90` matrix as an internal, non-published comparison. LeRobot is precedent for the episode count and the paired design, not a blessing of this specific suite choice.
- **OpenVLA / Pi0.5-style 50 eps** is the default in their eval scripts (`num_trials_per_task: int = 50`) — and is required when comparing against published numbers. Community reproduction guides state this explicitly: *"Do not pass `num-trials-per-task 10` for official comparison."* We are not making official comparisons.
- **Your own locked protocol table** (`evaluation_plan.md` §1) already lists "**50 (standard) or 10 (minimum)**", and the locked config already runs `libero_10` at **10 eps/task**. Option A extends the same logic you already accepted for the zero-shot row to the in-distribution core.
- **Seed noise as the dominant term** is documented in the "seed lottery" study (arXiv:2606.13856, verified 2026-08-13 — see §5.1 for the regime caveat: fine-tuning a pretrained VLA-JEPA, used here as a conservative anchor only) and is the reason your own accepted statistics mandate bootstrap CIs and probability-of-improvement rather than point estimates (rliable, Agarwal et al. 2021) — those safeguards are unaffected by the episode budget.

---

## 7. The Escalation Tier (what we lose, and how we keep it from mattering)

Nothing is permanently lost by starting at 10 eps. The full 50-eps protocol remains available and the harness is unchanged — the decision only sets the default. Concretely:

- Any head-to-head (e.g., `phaseforge` vs `warmstart_moe`) whose suite-mean gap is **≤ 3 pp** at 10 eps is escalated to **20 eps × 5 seeds** for that pair only: 2 cells × 5 seeds × 1,900 eps ≈ **1.6 eval-days**, paid only if needed. **Per the interim-analysis review, escalated pairs are reported as clearly labeled follow-up comparisons — their own CI and probability of improvement, flagged in the table — and are never pooled into the base-matrix statistics.**
- Any per-task claim (e.g., "cell A solves the drawer tasks, B does not") is escalated to 20 eps (5 pp bins) or 50 eps for the tasks in question, under the same follow-up-labeling rule.
- If a future claim ever requires leaderboard-comparable numbers (Stage 3 vision), the full 50-eps protocol is run at that time for that cell. Option A does not waive that requirement; it defers it to the point where a claim actually needs it.
- The routing-metric claims (primary) are unaffected at any episode count.

**What does NOT change under any option:** suites (`libero_90` ID + `libero_10` ZS), max steps per suite, `hard_reset: true`, `num_steps_wait: 10`, suite-role labels in every results file, 3 seeds for the base matrix, the base-matrix statistics and reporting format (mean ± std ddof=1, bootstrap CI, probability-of-improvement, Mann–Whitney U, per-task breakdowns), the checkpoint wiring, and the workload-sized timeouts. Implementation is config-only (`num_episodes_per_task` + the `SuiteSpec` episode-count table, which resizes the per-suite timeouts automatically: libero_90 ≈ 2.4 h at 10 eps) — no harness code changes.

---

## 8. Honest Caveats

1. **Current floor effect.** At 0–5 % SR, no episode count discriminates between cells. We are not asking the protocol to fix that; we flag it so the decision is made with full information. The episode budget only becomes observable once training clears the floor.
2. **Per-task coarseness at 10 eps** — documented in §5.2, mitigated by escalation.
3. **Small-margin blindness.** If two cells differ by ~1–2 pp, Option A will call it a tie; Option C may or may not separate it depending on seed luck. Under the seed-noise analysis in §5.1, both options produce CIs wide enough that we would escalate the pair anyway — the escalation tier is the honest home of that decision, and it exists in all three options.
4. **This report's measurements** are from the profiled harness on the free-T4 2-worker setup; absolute times will shift on different hardware, but the *ratios* (physics-dominated, reset 17 %, seed-dominated uncertainty) are hardware-independent.

---

## 9. Decision Requested

Per the locked-protocol rule, we request your explicit sign-off on one of:

- **Option A (recommended):** `libero_90` at **10 eps/task × 3 seeds**, `libero_10` unchanged at 10 eps/task; any ≤ 3 pp head-to-head or per-task claim escalated to 20 eps × 5 seeds for that pair/tasks only, **reported as clearly labeled follow-up comparisons (own CI and probability of improvement, never pooled with the base-matrix statistics — interim-analysis rule)**. Total ≈ 7.5 days wall on free Colab.
- **Option B:** 20 eps/task × 3 seeds everywhere. Total ≈ 10.5 days wall. Suite-mean SE within ~0.4 pp of the full protocol.
- **Option C:** status quo, 50 eps/task × 3 seeds. Total ≈ 20 days wall.

All three options preserve the locked suites, statistics, reporting, and labeling rules; the differences are the episode default and the escalation tier. We will implement the approved option as a config-only change and report the matrix with the declared protocol in every table. Per your review, the decision and date are logged in `evaluation_plan.md` (decision log, §1); the citations above were corrected accordingly (LeRobot scoped, Flow-Matching-LIBERO removed, seed-lottery verified with regime caveat); and your item B answer is recorded: **base matrix stays at 3 seeds as approved; catastrophic-seed risk is monitored training-side (routing/loss collapse diagnostics) and the escalation tier provides 5-seed coverage at the decision boundary.**

---

## 10. Sources

- LeRobot LIBERO documentation (HuggingFace): 10 episodes/task × 3 seeds standard, same-episode paired comparison — recommendation scoped to their four standard suites (Spatial, Object, Goal, Long); `libero_90` is not among them. https://huggingface.co/docs/lerobot/libero
- OpenVLA `run_libero_eval.py`: `num_trials_per_task: int = 50`; suite max steps incl. libero_90 = 400. https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
- OpenPI/LIBERO reproduction guide: full 500-episode protocol required for official comparison. https://github.com/Jinfeng50/openpi-libero-reproduction/blob/main/docs/baseline.md
- Sam, J. & Tsetserukou, D., *Output-Level Regularization Eliminates the Seed Lottery in Single-GPU VLA Fine-Tuning*, arXiv:2606.13856 (cs.RO, submitted 2026-06-11; verified resolving 2026-08-13). LIBERO-Object: 13 seeds, 65.2–94.2 % (std 7.5 pp, one collapsed seed), 500 eps/suite. Regime: pretrained VLA-JEPA fine-tuning on one RTX 5090 — conservative external anchor only for our from-scratch MoE cells.
- Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021 (rliable; already cited in the accepted protocol).
- PhaseForge internal: `evaluation_plan.md` (locked protocol + decision log), `REPORT_to_professor_2.md` (staged plan), `novelty_claim.md` (claim scope), measured profile `outputs/eval/profile_rollout.json`.
