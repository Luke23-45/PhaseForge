# PhaseForge — Big Decisions & Resolved Plan

**Purpose:** the locked decisions behind the 0%-to-fix plan — the reference for professor consultation and the permanent record. Nothing below is pending; every item is decided, with evidence and provenance.
**Date:** 2026-08-07 · Rev. 2 (amendments from critique: C2 stance, E8 decisions, compute budget, stage-naming). Stage naming: **P-Stage** = professor remediation plan (P-Stage 1 object-state channel, P-Stage 2 cached embeddings, P-Stage 3 end-to-end vision); **stage-1/stage-2** = PhaseForge training stages. · Cross-refs: `issues_register.md` (A1, A2, C1, C2, C7, B2, B3, C3, C4, B6), `novelty_claim.md` (Rev. 4), `REPORT_to_professor_2.md`

---

## 1. Why we are here — two independent, confirmed root causes

| Cause | Status | What it means |
|---|---|---|
| A1 observability gap | Confirmed | Policy sees 23-DoF proprioception only; object/visual info stripped in train (`VisionStripper`) and eval (`KEPT_OBSERVABLE_NAMES`, `render_observations: false`). Structural — no training-side fix exists. |
| A2 task-pool mismatch | Confirmed | Train = `libero_90`; eval listed spatial/object/goal/10/90. Spatial/Object/Goal were zero-shot on unseen tasks; their near-0% would be misread as "fix failed" after an A1 fix. |

## 2. Decision 1 — Vision removed permanently (state-only study)

- Study domain: **state-only + robosuite object-state channel** (P-Stage 1). The object-state channel is ground-truth sim state — a state-oracle, honestly labeled, never compared to vision baselines.
- Cameras stay disabled in train and eval. Vision would add perception confounds to a training-strategy question; our 0.6–0.8M state-only numbers are not leaderboard-comparable by design.
- P-Stage 2 (cached frozen embeddings) and P-Stage 3 (optional end-to-end vision) are deferred. Perception is explicitly outside the paper's claim.

## 3. Decision 2 — `libero_90` is the sole core dataset

Verified 2026-08-07 (LIBERO GitHub, LeRobot docs): LIBERO = 130 tasks in 5 disjoint suites — Spatial (10), Object (10), Goal (10), LIBERO-90 (90), LIBERO-10 (10). Official semantics: LIBERO-100 = LIBERO-90 (pretraining) + LIBERO-10 (downstream test); Spatial/Object/Goal are separate transfer-study benchmarks, not eval suites for a libero_90-trained agent.

| Scope | Decision |
|---|---|
| Train | `libero_90` only — 81/9 task split; source `yifengzhu-hf/LIBERO-datasets` (HDF5 mirror, official 90-file integrity check) |
| Eval, in-distribution | `libero_90`: 50 eps/task × 90 tasks × 3 seeds, per-task breakdowns |
| Eval, zero-shot (only row) | `libero_10`: 10 eps/task (LeRobot protocol), labeled as official downstream transfer test — zero-shot by design |
| Removed from core eval | Spatial / Object / Goal (separate benchmarks; zero-shot results would carry no mechanism information) |

Rejected alternatives: train on all 130 tasks (destroys the benchmark's transfer semantics; +44% ingest/train compute); keep the 5-suite eval (resurfaces the A2 misread risk).

## 4. Locked consequences

- **Eval config:** `rollout.yaml` suites → `libero_90` + `libero_10` (P-Stage 1 diff, issue E3).
- **Cache:** single 66 GB `libero_90` re-ingestion with object-state `state_keys` (cache hash change; images excluded).
- **Model matrix (8), full-length (100/200 epochs), 3 seeds:**
  - 2×2 factorial: `phaseforge` (phase encoder × centroid router), `phase_pretrain_random_router` (phase × random), `plain_encoder_phase_bootstrap` (plain × centroid), `warmstart_moe` (plain × random)
  - plus `scratch_moe`, `bc`, `oracle_moe` (GT-routing; signature-only bound), `teacher_forced` (E8: GT-partitioned experts, predicted-phase routing at inference; privileged-training, deployable after training)
- **Protocol honesty:** declared as our own protocol — never called "LIBERO results"; oracle/teacher-forced footnoted as privileged; NMI meaning stated per model (emergent specialization vs prediction quality); per-task breakdowns reported.
- **Balance sweep (C3):** balance weight 0 / 0.01 / 0.1 on `phaseforge`.
- **C2 (frozen vs unfrozen encoder): DEFERRED, explicitly** — the core 2×2 holds the encoder frozen (per the proposal's stage-2 design); unfrozen-encoder variants stay a register-listed sensitivity ablation, not part of the headline matrix.
- **E8 implementation decisions (locked):** (i) phase predictor = stage-1 phase head of the SAME phase-supervised checkpoint that `phaseforge` uses — shared pretraining, so only the stage-2 supervision regime differs; (ii) top-k asymmetry is footnoted, not hidden: oracle/`teacher_forced` route top-1 (exclusive GT phase partition), `phaseforge` routes top-2 (method hyperparameter) — stated in every comparison table; (iii) core runs use NATURAL sampling for all cells (parity) — balanced sampling is NOT in the core protocol; if `teacher_forced`/oracle starve a phase, that is a reported diagnostic, and a labeled balanced-sampling sensitivity run follows only if it materializes.
- **Compute budget (re-estimated 2026-08-07):** training ≈ 3.5–4 days (8 models × 3 seeds × 200 epochs at ~50 s/epoch + balance sweep); ID rollout ≈ 110k episodes (90×50 + 10×10, × 3 seeds × 8 models) at ~13 s+/episode serial ≈ **2–4 weeks wall-clock on free T4 (2 workers)**. Decision: full protocol for all 8 models — NO silent reductions; if compute binds, reductions are decided only at the B6 gate and recorded here.
- **Gates:** state-replay consistency test (B6) before P-Stage 1; phase-label spot-check (C4) before the bootstrap.

## 5. For professor review

1. The two decisions (objections welcome on protocol declarations and state-oracle labeling).
2. **P-Stage 1 diff** — train/eval lockstep: object-state observables per suite, `state_keys` ↔ `KEPT_OBSERVABLE_NAMES`, `rollout.yaml` trim (parity checked first).
3. **P-Stage 2 cache design** — deferred, but design submitted per the professor's request.

**Next step after review:** implement the P-Stage 1 diff, re-ingest (single 66 GB pass), then run the 8-model full-length sweep.
