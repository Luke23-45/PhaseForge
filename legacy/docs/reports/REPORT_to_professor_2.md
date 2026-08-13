# PhaseForge — Objectives, Current State, and Revised Plan

**To:** Prof. [Name]
**From:** [Student]
**Date:** August 7, 2026
**Subject:** Response to your analysis of the 0% rollout results — our objectives, the verified failure analysis, and our revised three-stage plan

> **Revision (round 3, 2026-08-07):** this version corrects two claims of the round-2 text: (i) the 0% result is now attributed to **two** confirmed causes — A1 observability (as before) **and A2 task-pool mismatch** (the evaluated spatial/object/goal suites are zero-shot task pools; see §2.4); (ii) the oracle baseline is relabeled **signature-only / non-deployable** (E7) and a **teacher-forced cell** (E8, privileged-training, label-free inference) joins the matrix, which grew from 5 models to **8 cells** (Batch A/B). The full-length training claim is now true: early stopping is explicitly disabled in the protocol runners (`643674a`); tests: 117/117.

---

## 1. Our Objectives (with the Context Behind Them)

### 1.1 What PhaseForge is testing

PhaseForge is a controlled study of a **training-strategy** question for mixture-of-experts policies in long-horizon manipulation:

> Does phase-supervised pretraining plus phase-bootstrapped router initialization produce a more stable, more specialized, and easier-to-train MoE policy than scratch MoE or generic warm-start MoE?

The design is deliberately two-staged:

- **Stage 1:** a generalist policy is trained with behavior cloning while jointly classifying a rule-based *skill phase* (6 phases: idle/reach/pick/carry/place/other, median-filtered). The phase loss is training-only; it is never an input at inference.
- **Stage 2:** the stage-1 encoder is frozen and reorganized into a top-2-of-6 MoE. Each expert is initialized from the stage-1 action head; the router is bootstrapped from phase centroids in latent space; training continues under an auxiliary load-balancing loss.

The experimental matrix (round-2 dry run: five models — `bc`, `scratch_moe`, `warmstart_moe`, `phaseforge` (proposed), and `oracle_moe`). Round 3 extended it to eight cells: the five above, `teacher_forced` (E8; GT-partitioned experts, predicted-phase routing at inference — privileged *training*, label-free *inference*, deployable after training, footnoted), `phase_pretrain_random_router`, and `plain_encoder_phase_bootstrap` (2×2 completion). The oracle is relabeled a **signature-only, non-deployable bound** (its claim is the routing signature, not success). Evaluation: simulator rollouts, goal predicates, per-task breakdowns, 50 episodes/task on the in-distribution suite, 3 seeds. The paper's claims rest on MoE-specific metrics (time-to-stable-routing, routing entropy, expert balance, collapse rate, phase–expert NMI) *and* rollout success rates, not on loss curves alone.

### 1.2 Why we chose state-only (and what we missed)

The state-only decision was a **causal-control** choice, not primarily a compute choice: vision would have added confounds (encoder choice, pretraining, resolution, camera placement) and made it impossible to attribute any measured gain to the MoE strategy itself. That reasoning is sound as far as it goes — and your point is that it does not go far enough: the LIBERO evaluation protocol (goal predicates over per-episode randomized object placements) makes pure proprioception **structurally** insufficient, for any architecture. We did not see the separation between the two questions at the time; your reframe is exactly right, and we have internalized it:

1. **Q1 — does our MoE routing/capacity design work?** This is our paper's question. It is answerable without vision.
2. **Q2 — can we afford perception?** A separate question, and per your staged plan it has a cheap answer (object-state channel now, cached frozen embeddings next).

The two were conflated when we picked an input modality that cannot carry the task-relevant signal. We now treat them as independent axes.

---

## 2. Current State (Verified Facts, Not Impressions)

### 2.1 Pipeline validation (dry run, completed)

All six runs (5 models × 2 stages) and all evaluations completed cleanly on real LIBERO-90 data: early stopping and checkpointing behaved correctly, stage-1 → stage-2 bootstrapping verified in logs, and the oracle produced exactly the expected upper-bound signature (NMI = 1.0, routing entropy ≈ 0, balance ≈ 0). All trained MoE variants were healthy — balance ≥ 0.98, zero collapsed experts. The offline L2 proxy (which we never treat as a result, per LIBERO Appendix E.2: *success rates, not BC loss*) placed all four trained models in a 9.6–11.4% band — a floor effect, as the dry-run report stated explicitly.

### 2.2 Rollout evaluation (the real metric)

We built the full rollout harness (`StateOnlyLiberoEnv` wrapper over `OffScreenRenderEnv`, rollout evaluator, multi-seed runner) and hardened it in the process: the BDDL success predicate is now evaluated once per step, `hard_reset` semantics are verified, and silent `strict=False` state-dict mismatches are now surfaced loudly at both load sites. On the actual simulator, results are **0% success: libero_spatial 0/500, libero_object 0/500** (libero_goal was in progress at last check), with episodes running the full horizon. Round-3 analysis (A2) added a second cause: those suites are **zero-shot task pools** — their tasks were never in the training set — so 0% is also consistent with task-pool mismatch, not observability alone (see §2.4).

### 2.3 Why we concluded observability, not a bug

Before concluding, we ran the diagnostics the literature's own 0%-success post-mortems demand, against our code:

| Documented failure class | Evidence in our code | Verdict |
|---|---|---|
| Missing gradient clipping (OpenVLA issue class, 0%→72% after fix) | `grad_clip_norm: 1.0` in both stages | Ruled out |
| Action normalization mismatch (openpi issue class) | Actions raw everywhere — ingestion, losses, eval passthrough | Ruled out |
| Action-space mismatch (LeRobot / xVLA issue class) | Model 7-DoF, env OSC_POSE 7-DoF | Ruled out |
| Eval-env / checkpoint-load bugs (LeRobot issue class) | Load path audited; mismatch logging added; predicate evaluated once per step | Ruled out |
| Undertrained / **under-observable** policy | 23-DoF proprioception, zero object information | **Consistent with all evidence** |
| **Zero-shot task pool (A2)** | spatial/object/goal suites are unseen tasks; eval was zero-shot, never labeled as such | **Consistent (added round 3)** |

The primary explanation remains the information ceiling: our 23-DoF state (`robot0_joint_pos/vel`, `robot0_eef_pos/quat`, `robot0_gripper_qpos`) describes the arm and gripper only. Object placement is drawn fresh every episode and is not a function of joint angles. A policy that cannot know where the bowl is cannot pick it up.

### 2.4 Second confirmed cause: task-pool mismatch (A2)

Round-3 analysis showed the 0% was **over-attributed to observability alone**. The raw LIBERO-90 pool mixes feature/spatial (2×), object (2×) and goal (1×) tasks, and our dem100 training subset was drawn with a 5:1 object:spatial ratio — a different distribution from the eval suites. The spatial/object suites are **zero-shot** evaluation: their tasks were never trained on. 0/500 is therefore consistent with two stacked causes, not one.

**Decision 2 (locked):** evaluation is restricted to `libero_90` as the in-distribution core (90 tasks × 50 eps × 3 seeds, per-task breakdowns) plus `libero_10` as a labeled zero-shot row (10 tasks × 10 eps); spatial/object/goal suites are dropped from the protocol. Results JSONs declare `eval/suite_roles` (in-distribution / zero-shot) so the split can never be silently conflated again.

---

## 3. Response to Your Analysis

1. **Your outside evidence confirms our read.** The π0 ablation you cite (proprioception removal costs ≈1.4 points of average success when vision is present — 94.2% → 92.8%) quantifies what our failure analysis suggested qualitatively: the task-relevant signal lives almost entirely in the object/visual channel. Take the images away too, as we did, and only a fraction of a point's worth of signal channel remains to solve the whole task. Similarly, the object-perturbation stress tests — >90% models collapsing under object-position changes — underline how sensitive these tasks are to object location. We agree fully: on these tasks a model with zero object information is *structurally* unable to succeed, for any architecture.
2. **We accept the reframe of the false binary.** "Full VLM/diffusion perception stack" and "no object information at all" are not the only two options. In the robotics MoE literature (AdaMoE, MENTOR, semantically-structured MoE variants), MoE is a routing/capacity layer applied *inside or alongside* an existing perception-equipped backbone — it inherits perception; it never replaces it. A router can only select among experts using what is in its inputs; no amount of expert specialization invents information that never entered the network. That is an information-theoretic ceiling, not an expressivity problem, and we will not spend compute trying to defeat it.
3. **Agreed: no further work on the 23-dim arm-only setup.** We will not attempt to rescue it with a larger or cleverer MoE. It is retired as an evaluation configuration. Evaluation scope is locked by Decision 2 (§2.4): `libero_90` in-distribution + `libero_10` labeled zero-shot only.
4. **We adopt your staging** (Section 4), including the honesty constraints: state-oracle results are labeled as such, never compared against vision-based leaderboard numbers, and reported internally as an architecture sanity-check.

---

## 4. Revised Plan (Adopting Your Staged Plan)

### Stage 1 — Re-enable the robosuite object-state channel (current)

- **What:** restore the low-dim `object-state` observables robosuite already computes at every physics step — absolute positions of each task object and object positions relative to the end-effector. Nothing is rendered; the cost is a few extra floats per step. This is the channel classic state-based manipulation pipelines (robomimic-style baselines and the imitation-learning line LIBERO descends from) have always paired with proprioception.
- **Concrete changes in PhaseForge:**
  - *Training side:* extend the state schema (`phaseforge/config/data/common.yaml`, `state_keys`) and the ingestion stripper so the object-state keys enter the training HDF5 features; the cache hash changes, triggering re-ingestion; retrain stage 1 + stage 2 for all eight cells (Batch A/B matrix, §1.1).
  - *Eval side:* enable the matching observables in `KEPT_OBSERVABLE_NAMES` and `_extract_state` (`phaseforge/evaluations/envs/libero_env.py`) so train and eval states are exactly the same schema — parity being the place where silent drift has bitten us before.
- **Verification gates before we call Stage 1 done:**
  1. State-replay consistency test passes (env state matches recorded demo state under demo actions).
  2. In-distribution success on `libero_90` (per-task breakdowns, 3 seeds) clears the floor; the `libero_10` zero-shot row is reported labeled.
  3. Routing metrics (entropy, balance, collapse, NMI) remain healthy and interpretable — we re-open the phase–expert alignment question now that the policy can actually see the objects.
- **Honest labeling:** this is a state-oracle sanity check of the MoE architecture. These numbers will not be reported as LIBERO leaderboard results and will not be compared against OpenVLA, π0.5, or ACT.

### Stage 2 — Cached frozen visual embeddings (next)

- **What:** run a small frozen encoder (lightweight ViT / DINO / CLIP tower) over the dataset images **once**, cache the resulting embeddings, and let the MoE architecture search train against those cached vectors instead of raw pixels — switching to a live encoder only at deployment time. This is exactly the decoupling pattern in recent efficiency-focused work (train-on-cached-embeddings, decoupled vision/language encoders), and it converts vision compute from a recurring cost into a one-time preprocessing cost — precisely the axis our budget cares about, since every MoE hyperparameter re-run stops re-paying perception.
- **Budget lever if needed:** downsize inputs (e.g., 84×84, single camera), the same trick used to make offline visuomotor training cheap.
- **Validation discipline:** frozen encoders are not automatically as good as end-to-end vision (R3M/VC-1-class features have produced jitterier trajectories in at least one study). So we validate on **one LIBERO suite first**: if a handful of MoE configs place reasonably with decent object localization, we scale the sweep; if they all struggle even with good localization, we unfreeze or upgrade the encoder *before* attributing anything to the MoE design.
- **Note:** Stage 1 and Stage 2 are not mutually exclusive — Stage 1's object-state channel also serves as the "object localization" sanity signal for Stage 2, and Stage 2 may later absorb Stage 1's proprioception+object state as the low-dim branch of the visuomotor policy.

### Stage 3 — Optional end-to-end vision fine-tuning (later, only if needed)

- Only on whichever MoE configuration wins stages 1–2, and only if we actually need numbers comparable to the published leaderboard. The MoE research question (Q1) is fully answerable in stages 1–2; stage 3 is a separate, later decision about benchmark comparability.

---

## 5. Response to Your Offer

Thank you — and yes. We will implement the Stage 1 schema/observable changes and the Stage 2 embedding-cache script ourselves, but we would value your review of (a) the Stage 1 diff (state schema + observable handling in both ingestion and the eval env) *before* we re-train, since train/eval state parity is where we have found silent drift repeatedly, and (b) the Stage 2 cache design (encoder choice, embedding format, normalization) before we run the one-time preprocessing. We will send both for comment as soon as they are drafted.

---

## 6. Timeline (Estimates)

| Stage | Work | Estimate |
|---|---|---|
| Stage 1 | Schema + env changes (done), re-ingestion, retrain 8 cells, rollout eval (`libero_90` + `libero_10`, 3 seeds) | ~110k eps (8 cells × 13,800 eps); ~2–4 weeks wall on free Colab T4 (2 workers, ~13 s+/ep); training ≈ 3.5–4 days (register F3) |
| Stage 2 | Embedding-cache script, one-suite validation, then sweep | ~1 week |
| Stage 3 | Decision point after stages 1–2 results | TBD |

Compute is not the constraint: a full 5-model × 3-seed sweep plus rollout evaluation fits in roughly one compute day on our hardware. The plan's critical path is code correctness and train/eval parity.

---

## 7. Conclusion

We accept your diagnosis in full and have converted it into a staged plan: (1) object-state channel now — a state-oracle architecture sanity check with clean labeling; (2) cached frozen embeddings next — perception as a one-time preprocessing cost; (3) end-to-end vision later only if benchmark-comparable numbers are required. The MoE research question — the paper's actual question — is answered by stages 1–2, and the conflation that produced the 0% is removed. We will report back with the Stage 1 diff and results.
