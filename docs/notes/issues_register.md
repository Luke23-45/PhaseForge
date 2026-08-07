# PhaseForge — Issues Register

**Purpose:** single inventory of every known problem, confound, risk, and pending decision identified so far (0%-rollout investigation, professor feedback rounds 1–2, code audits, literature cross-checks). The resolution plan will be built directly on the OPEN items listed here.

**Status legend**
- `CONFIRMED` — verified by evidence (code, configs, logs, primary sources)
- `OPEN` — genuine, unresolved problem/decision
- `RULED-OUT` — hypothesized cause, disproven with evidence (kept for audit)
- `FIXED` — resolved; commit or mechanism noted
- `PARTIAL` — mitigated but not fully resolved

**As of:** 2026-08-07 · Repo HEAD: `c3fbb1d` (pushed to origin/master) · Tests: 69/69, ruff clean

**Latest round (Patch 1 + Batch A/B, this session):** P-Stage 1 object-state channel implemented (state_dim 151, object index census + B6 gate); 2×2 cells + teacher-forced cell (C1/C7/E8) implemented with tests; C3 per-epoch NMI-vs-balance logging + sweep script; B4 ID/OOD declaration. Tests: **113/113**. Statuses below updated to reflect code-level completion; Colab execution steps remain OPEN.

---

## A. 0% Rollout Success — Root-Cause Inventory

### A1. Observability gap (information ceiling) — `FIXED (code)` ⚠ effect pending full-length training
- Training state = 23-DoF, all arm/gripper proprioception: `robot0_joint_pos` (7) + `robot0_joint_vel` (7) + `robot0_eef_pos` (3) + `robot0_eef_quat` (4) + `robot0_gripper_qpos` (2) (`phaseforge/config/data/common.yaml`). Zero object/table information.
- LIBERO tasks are object manipulation with per-episode randomized placement; object location exists only in images or object-state sensors — both stripped from training (`VisionStripper`) and pruned in eval (`KEPT_OBSERVABLE_NAMES`, `render_observations: false`).
- Evidence: 0/500 (libero_spatial), 0/500 (libero_object); community literature (π0 proprioception ablation ≈1.4pp; perturbation stress tests; no published arm-only policy). Professor round 1 confirmed: *structural*, not expressivity, not fixable by a bigger/smarter MoE.
- **Fix (Patch 1, implemented):** P-Stage 1 object-state channel — state layout `[proprio 23 | objects K×7 | mask K]`, K=k_slots=16, `state_dim=151`; world-frame pos(3)+quat(4) per object decoded from the robosuite flatten (free-joint qpos slice / hinge / slide / fixed closed-form); `ObjectIndex` JSON from the census script with B6 state-replay gate; train side (`state_machine.py`, `vision_stripper.py`, `phase_labeler.py`, `normalizer.py`) and eval side (`libero_env.py` `_extract_state` + `_disable_unused_observables`) in lockstep. **Effect verification pending** full-length training on real data (Batch C, Colab).

### A2. Train/eval task-pool mismatch (zero-shot confound) — `CONFIRMED` ⚠ highest priority
- **Fact (LIBERO primary sources):** 130 tasks in disjoint pools — Spatial (10), Object (10), Goal (10) are separate transfer-testing suites; LIBERO-100 (100) is split into LIBERO-90 (pretraining) + LIBERO-Long (10). 10+10+10+90+10 = 130; no overlap.
- **Our setup:** training = `libero_90` (90 HDF5 files, `role: train`, source `yifengzhu-hf/LIBERO-datasets`); rollout eval = `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` (`phaseforge/config/eval/rollout.yaml:13-18`).
- **Consequence:** the Spatial/Object/Goal rollouts were **zero-shot evaluation on unseen tasks/objects/instructions**. The 0% is therefore confounded by TWO independent causes (A1 + A2). After the A1 fix, Spatial/Object/Goal could remain near-0% purely from zero-shot — and would be misread as "the fix failed."
- Consistency check: `docs/notes/evaluation_plan.md:26` correctly said "evaluate on LIBERO-90 (our training tasks)" — but the implemented `rollout.yaml` added spatial/object/goal without this rationale; discrepancy between plan doc and config.
- **Decision (RESOLVED 2026-08-07, big-decision session):** `libero_90` is the SOLE core dataset — train AND eval (in-distribution). `libero_10` retained as the ONLY labeled zero-shot row (it is the *official* downstream transfer test of LIBERO-90 pretraining per the LIBERO paper/GitHub) at LeRobot's 10 eps/task. Spatial/Object/Goal dropped from core eval entirely (they are separate transfer-study benchmarks, not eval suites for a libero_90-trained agent). Verified online 2026-08-07: LIBERO GitHub (BENCHMARK list = SPATIAL/OBJECT/GOAL/90/10; "LIBERO-100 further split into LIBERO-90 for pretraining and LIBERO-10 for testing"); LeRobot docs (standard 4-suite eval is for vision models). Consequence: `rollout.yaml` suites list shrinks to `libero_90` + `libero_10` (part of Stage 1 diff, E3); single 66 GB cache re-ingestion (object-state keys, Decision 1) instead of multi-suite.
- Decision 1 (same session): **vision permanently removed** from the core study — state-only + robosuite object-state channel (professor Stage 1); cameras stay disabled in train (VisionStripper) and eval (KEPT_OBSERVABLE_NAMES, `render_observations: false`). Claim excludes perception; state-only numbers are never leaderboard-comparable by design. Stage 3 (optional end-to-end vision) deferred, not part of this paper.

### A3. Ruled-out failure classes (audit record) — `RULED-OUT`
| Hypothesized cause | Evidence against | Source lesson |
|---|---|---|
| Missing gradient clipping | `grad_clip_norm: 1.0` in both stages | openvla #299/#333/#334 (0%→72% after fix) |
| Action normalization mismatch | actions raw everywhere: ingestion, losses, eval passthrough (no normalizer on actions) | openpi #849 |
| Action-space mismatch | model 7-DoF == env OSC_POSE 7-DoF | lerobot #2623/#2630, xvla #3401 |
| Eval-env / checkpoint-load bug | load path audited; `strict=False` mismatches now loud; predicate once per step | lerobot #2850/#2832/#3247 |

### A4. Undertrained runs (residual contributor, still open) — `OPEN`
- Dry run early-stopped at 2–21 epochs of 100/200 (`patience=10, min_delta=0.001`). Full-length training never run. In-distribution ceiling unknown.

---

## B. Evaluation Methodology

### B1. Offline L2 proxy is not a valid success metric — `CONFIRMED` (superseded)
- LIBERO Appendix E.2: success rates, not BC loss. Uniform 0.05 L2 threshold on 7-DoF actions has no precedent; floor effect (9.6–11.4% for all four trained models). Never report as task success. Metric of record = rollout success rate (A-verified).
- Status: superseded by rollout harness; keep offline only for internal action-consistency signal (`eval/action_mse`, see G6).

### B2. Oracle MoE is not a valid upper bound as built — `OPEN`
- `oracle_moe.py`: router bypassed, GT phase labels select the expert; each expert sees only its phase's samples (~1/6), phase imbalance → starvation (collapse_rate 0.833 in dry run), zero router gradient, balance loss disabled.
- Options: (a) redesigned oracle (balanced phase sampling / auxiliary router supervision), or (b) keep but relabel as "perfect phase-alignment bound, not deployable" (see B3).

### B3. Oracle privileged-information labeling (reporting integrity) — `OPEN`
- GT phases used at training; at inference falls back to the untrained router when no labels are fed (`oracle_moe.py:49`) — its privilege is training-time only unless the harness feeds labels. Exact inference behavior in rollout eval must be stated explicitly.
- Action: footnote oracle in every success table as non-deployable upper bound; add to reporting template NOW (before results arrive / anchoring bias).

### B4. Zero-shot vs in-distribution evaluation not declared — `FIXED (code)`
- No protocol statement labeling which suites are in-distribution vs zero-shot transfer. Must be stated in writeup + reporting template (see A2).
- **Fix:** `eval_results.json` now declares `eval/suite_roles` (libero_90 = in-distribution, libero_10 = zero-shot (labeled); anything else = "unclassified") — `rollout_evaluator.py`; `final_results.json` carries the same declaration; `run_multi_seed_eval.py` aggregates only the Decision-2 suites. Reporting-template footnote pending (B3/D3).

### B5. Multi-seed statistics — `OPEN` (infra ready)
- Runner exists (`scripts/run_multi_seed_eval.py`, seeds 42/43/44); full 3-seed sweep never run. Required for paper tables (mean ± std).

### B6. State-replay consistency test — `FIXED (code)` ⚠ execution pending (Colab)
- Env state vs recorded HDF5 demo state under demo actions must match within MuJoCo tolerance before trusting any rollout (evaluation_plan §3.1). Professor: "unusually careful practice" — mandatory gate for Stage 1.
- **Fix:** `scripts/…/build_object_index.py` (patch-0 census) replays every task's init state with `set_init_state` and compares against the recorded demo state as the B6 gate; used to build the per-task `ObjectIndex`. Must still be RUN on the real data cache before re-ingestion/rollouts (Batch C).

### B7. Train/eval state parity (recurring silent-drift risk) — `OPEN` (partially mitigated)
- `strict=False` mismatches now logged loudly (G1), but parity must be re-verified whenever the state schema changes — i.e., for every Stage-1/E2 change, on both ingestion and `_extract_state` sides. Professor will check the diff first on exactly this.

---

## C. Experiment Design & Causal Attribution

### C1. Confounded core comparison (bundled interventions) — `FIXED (code)` ⚠ training pending
- `phaseforge` bundles TWO interventions: (1) phase-supervised stage-1 pretraining, (2) phase-centroid router bootstrap. A win over `warmstart_moe` is unattributable.
- Professor's fix (compute is cheap, ~50 s/epoch): partial factorial → 2×2.
  - Existing cells: `scratch_moe` = random encoder × random router; `warmstart_moe` = plain (BC) encoder × random router; `phaseforge` = phase encoder × centroid-bootstrapped router.
  - **Two new cells required:** `phase_pretrain_random_router` (phase encoder × random router), `plain_encoder_phase_bootstrap` (plain encoder × centroid-bootstrapped router).
- **Fix (implemented):** `phaseforge/models/baselines/phase_pretrain_random_router.py` (shares phaseforge stage-1 ckpt via `resolve_checkpoint_source`; random router preserved by bootstrap) and `plain_encoder_phase_bootstrap.py` (shares bc stage-1 ckpt; centroid bootstrap on the plain latent). Configs + tests in `tests/test_model_baselines.py`; wired into `run_multi_seed_train.py`/`run_multi_seed_eval.py`. Training pending (Batch C).

### C2. Frozen encoder in Stage 2 — `OPEN` (stance: explicitly DEFERRED 2026-08-07)
- `scratch_moe` (unfrozen) beat frozen-encoder models in the dry run; frozen latent cannot adapt to the routing objective. Ablation (frozen vs unfrozen with lower LR) from proposal §13 never run.
- Stance (resolved plan Rev. 2): the core 2×2 holds the encoder frozen per the proposal's stage-2 design; unfrozen variants remain a register-listed sensitivity ablation, not part of the headline matrix.

### C3. Specialization–balance dilemma — `FIXED (code)` ⚠ sweep execution pending
- Balance loss (coeff 0.01) enforces balance ≥ 0.98 but NMI = 0.0 for all learned models — the documented "pseudo-balancing" pattern; balance masks washed-out phase signal.
- Action: sweep balance weight (e.g., 0 / 0.01 / 0.1); log balance score vs NMI over training.
- **Fix (implemented):** `Stage2Trainer` now logs `val/phase_expert_nmi`, `val/balance_score`, `val/collapse_rate`, `val/routing_entropy` every epoch (single forward pass, reused outputs); `scripts/run_balance_sweep.py` runs phaseforge stage-2 at balance weight 0 / 0.01 / 0.1 × 3 seeds. NOTE: the effective knob is `models.router.balance_coeff` (applied inside `TopKRouter`); `train.balance_coeff` in `stage2.yaml` is dead config. Sweep execution pending (Batch C).

### C4. Phase-label quality unvalidated — `FIXED (code)` ⚠ real-data run pending
- Rule-based labeler (`RuleBasedPhaseLabeler`: gripper 0.02/0.04, eef-vel 0.01, min-duration 5, median-7) feeds: stage-1 phase loss, centroid bootstrap, NMI, and the oracle itself. Errors at transitions (e.g., carry→place in drawer tasks) propagate everywhere.
- Action: spot-check labels on a handful of real trajectories before trusting the bootstrap.
- **Fix (implemented):** `scripts/spot_check_phase_labels.py` — C4 gate over real HDF5 trajectories using the production wiring; exit code 1 on labeler violations. Must be RUN on the real data cache (Batch C) before re-ingestion/rollouts.

### C5. Full-length training never run — `OPEN`
- 100/200 epoch schedules with early stop truncating everything; the loss/success ceiling is unknown. Required before any hypothesis claim (dry-run numbers are machinery-validation only).

### C6. Unrun ablations (proposal §13) — `OPEN`
- Phase supervision vs none (C1), frozen vs unfrozen encoder (C2), hard vs soft routing, expert/phase count, phase-label noise, data fraction. Not required all at once — decide priority with the professor's 2×2 first.

### C7. Teacher-forced routing cell (decomposable oracle) — `FIXED (code)` ⚠ training pending
- Motivation: the GT-routing oracle (B2) is an invalid success bound, and its inference is label-fed. A teacher-forced variant — experts partitioned by GT phase during training, routing at inference by `argmax` of a learned phase predictor (stage-1 phase head) — becomes a *decomposable instrument*:
  - `oracle (GT routing) − predicted` = **phase-predictability loss** (answers C4 quantitatively: are the rule-based phases grounded in the 23-DoF state?)
  - `predicted − phaseforge` = **strategy loss** (what bootstrap + balance cost vs the field's supervised-routing recipe)
- Privileged-*training* only, label-free inference → deployable after training, teacher-student pattern (professor-endorsed); footnoted per honesty rule.
- Rename in all artifacts (no longer "oracle"): "teacher-forced routing."
- NMI for this cell = prediction quality (CE-trained), NOT emergent specialization — metric-meaning rules added to `novelty_claim.md` §4.
- Imbalance/starvation caveat of B2 still applies to its GT partition — **stance LOCKED (2026-08-07)**: core runs use natural sampling for all cells (parity); balanced sampling only as a labeled sensitivity run if starvation materializes.
- **Fix (implemented):** `phaseforge/models/baselines/teacher_forced.py` — GT-partitioned top-1 dispatch in training, `argmax(phase_head)` at inference (eval mode + `get_action`); shares the phaseforge stage-1 checkpoint (alias); phase head frozen with the encoder (only experts train); router kept for structural parity, never consulted. Config `config/models/baselines/teacher_forced.yaml`; tests cover train/eval routing split, label-free inference, freeze semantics. Registered in `novelty_claim.md` as E8.

---

## D. Novelty & Positioning

### D1. General claim no longer novel — `CONFIRMED` (verified online)
The general claim "phase/skill structure helps MoE specialize in manipulation" is already made by multiple 2025–2026 papers (all verified online):

| Paper (verified) | Venue/ID | Mechanism | Relation to us |
|---|---|---|---|
| **MEAT** = "MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts" (Mazza et al.) | arXiv 2601.21971 | Supervised MoE on ACT; one expert per phase; gating supervised with phase labels (CE); ordered phases (grasp→retract) | Same phase-supervision idea, **no centroid bootstrap, no frozen-encoder init** |
| **MoE-ACT** (Guo et al.) — *different paper*, the one our proposal cites | arXiv 2603.15265 | MoE FFN in ACT encoder; language-conditioned; FiLM + multi-scale cross-attn; bimanual RoboTwin 2.0 | Task-level (not phase-level) routing |
| **SMP** (Hao, Zhai, Liu, Soh) | **ICLR 2026**, arXiv 2601.21251 | Diffusion MoE; state-adaptive orthonormal skill basis (Stiefel/QR) + sticky Dirichlet-Markov gating; state-only router distilled at inference | Phase-consistent skills without labels; orthogonal basis solves identifiability — directly relevant to our NMI=0 |
| **LAR-MoE** | arXiv 2603.08476 | Two-stage: unsupervised student–teacher latent co-training, then routing regularized to latent; **no phase labels** | Direct challenge to our supervised-phase premise |
| **AdaMoE** (Shen et al.) | arXiv 2510.14300 | Decouples expert selection (router) from weighting (scale adapter); attacks load-balance-vs-specialization conflict | The C3 dilemma, solved at the router level |
| **Memory-Aware Routing** (Hou et al.) | Findings ACL 2026, pp. 17320–17337 | Names **pseudo-balancing** (balance loss → random routing → knowledge overlap, no specialization); memory buffers for consistent routing | Our dry-run signature (NMI=0, balance≥0.98) is exactly this, at LLM scale |
| **Move-Then-Operate** (Lei et al.) | ICML 2026, arXiv 2604.23620 | Dual-expert VLA (move/operate); chunk-level phase router; teacher-forcing with GT labels; hard routing; MLLM-generated phase labels | Phase-supervised routing + label generation |
| **PAMAE** | arXiv 2606.27144 | Phase-aware MoE for flow-matching VLA; phase router + lightweight phase head; **two-stage training** (warmup → supervised routing, aux losses annealed) | Closest to our two-stage structure — but no centroid bootstrap |
| **CoRDE** | arXiv 2606.21935 | Frozen concept encoder guides variational expert responsibility; LoRA experts on frozen backbone; reduces routing collapse | Semantic priors instead of phase labels |

**Overlap map (what we share with whom):** phase-supervised routing — MEAT, MTO, PAMAE; two-stage training — PAMAE, LAR-MoE, CoRDE; state-only router at inference — SMP, CoRDE; balance-vs-specialization — AdaMoE, MAR. None of these are our differentiator.

**Genuine differentiators (candidate novelty):**
1. **Phase-centroid router bootstrap** on a frozen stage-1 latent — initialization-by-construction; nobody does this (closest: PAMAE/MTO supervise routing gradients; LAR-MoE aligns to unsupervised latents; SMP distills an amortized posterior).
2. **Oracle-routing upper bound + NMI phase-expert alignment diagnostics** — the oracle baseline (GT-label routing, NMI=1.0 sanity signature) is our evaluation-framework contribution.
3. **Controlled state-only factorial at 0.6M params** — everyone else is vision/VLA-scale; the training-strategy question as a controlled study is unclaimed — but only defensible WITH the 2×2 (C1).
4. **Pseudo-balancing in small-scale continuous control** — MAR found it at LLM scale; our dry-run is small-scale evidence, IF the diagnosis survives the eval fixes (A2/B6).

**Action:** read MEAT + MoE-ACT (Guo) + SMP + LAR-MoE + MTO + PAMAE in full before drafting related work. Sharpen contribution to: centroid bootstrap + oracle/NMI evaluation framework + controlled state-only factorial — not the general finding.

### D2. Comparison discipline vs vision baselines — `CONFIRMED` policy
- State-only/state-oracle numbers must never be compared with OpenVLA/π0.5/ACT leaderboard numbers (97–98%). Stage-1 numbers are an internal architecture sanity check. LeRobot protocol: 10 episodes/task × 4 suites (400 eps); ours: 50/task (OpenVLA standard) — fine, but state the protocol.

### D3. Report #2 contains now-stale claims — `OPEN`
- `docs/notes/REPORT_to_professor_2.md` attributes 0% to observability alone (true but incomplete) and sets gates ("per-suite success rates clear the floor") that are invalid for zero-shot suites. Must be revised in the reply to the professor (round 3) with the confirmed A2 finding.

### D4. Citation errors found during online verification — `OPEN` (fix in paper draft)
- Our experiment report cites "Move-Then-Operate (Xu, ICML 2026)" — actual authors: **Lei Lei, Jie Gu, Chu Tang, Jingmin Chen, Ruiqi Wang** (arXiv 2604.23620).
- Two DIFFERENT papers named "MoE-ACT": our proposal's (Guo et al., arXiv 2603.15265, bimanual) vs the professor's "MEAT" (Mazza et al., arXiv 2601.21971, surgical). Cite both, disambiguated.
- MAR citation verified: Hou et al., Findings of ACL 2026, pp. 17320–17337 (DOI 10.18653/v1/2026.findings-acl.857).
- LAR-MoE authors unverified ("Rodriguez et al." could not be confirmed from arXiv 2603.08476 snippet) — verify before citing.
- MoE-DP citation (Cheng, 2025) unverified; SMP's closest MoE-diffusion baseline is Wang et al., 2024 — likely the correct reference.
- Field note: PAMAE, CoRDE, TRACT, FocalPolicy are additional 2026 phase/expert-routing works — the field is crowded; novelty must be narrow (see D1).

---

## E. Data Pipeline (Stage 1 object-state channel)

### E1. State schema change requires full re-ingestion + retrain — `OPEN` (execution)
- Cache hash `a4c74be17f117a4b` binds state_keys/phase labeler config; changing `state_keys` → new hash → re-ingestion of ~66 GB libero_90 + retrain stages 1+2 for all models (8 cells: 5 prior + 3 new).
- Code is complete (object-state keys, index, labeler, normalizer); the config hash has already changed, so re-ingestion is triggered automatically by the pipeline. Execution pending on Colab (Batch C).

### E2. Exact object-state key names unverified — `FIXED (code)` ⚠ real-data verification pending
- Must enumerate the object-state observables (absolute object positions + positions relative to eef) available in the suite HDF5 files AND the robosuite observable names per suite before touching `state_keys` / `KEPT_OBSERVABLE_NAMES`. Train and eval names must match exactly.
- **Fix:** the patch-0 census (`phaseforge/data/scripts/build_object_index.py`) enumerates per-suite object sets at runtime and writes `raw/libero/object_index.json`; the ingest pipeline validates config↔index handshake (k_slots, dim_per_object, include_mask, state_dim) and fails loudly on mismatch; eval `_extract_state` decodes the identical index. Final verification on the real cache pending (Batch C).

### E3. Eval env must enable the same object observables — `FIXED (code)` ⚠ real-data verification pending
- `libero_env.py` currently keeps 5 keys and prunes object sensors (`render_observations: false`). Extending training without the eval side (or vice versa) recreates B7 drift. Both sides change together; `_extract_state` and `KEPT_OBSERVABLE_NAMES` updated in lockstep with `state_keys`.
- **Fix:** eval `_extract_state` decodes the same ObjectIndex layout as training (object states via `_disable_unused_observables(env, keep_extra=...)`); layout asserted equal (151-dim) in tests. Real-data verification pending (Batch C).

### E4. Action-space/normalization re-audit on schema change — `RESOLVED`
- Actions are raw (no normalizer) — verified and ruled out (A3) — but every schema/ingestion change should re-verify action parity (normalization, dtype float64 passthrough, order).
- **Audit done (Patch 1):** action path untouched by the object-state channel (only the state/observation side changed); regression tests cover raw float64 action passthrough; normalizer `ignore_dims` excludes the mask dims from stats.

---

## F. Rollout Infrastructure & Compute

### F1. `sim.forward()` before each `mj_step` — `OPEN` (unpatched)
- Candidate physics-correctness fix; no community precedent found; requires a Colab benchmark (e.g., replay-sanity) to decide. Do not ship without evidence.

### F2. `hard_reset` semantics — `CONFIRMED` (handled)
- `hard_reset: true` = official benchmark default (bit-identical); soft resets NOT bit-identical after settling steps (LeRobot docs). Config documents both. Keep hard_reset for published numbers.

### F3. Incomplete evaluation runs — `OPEN`
- Colab logs: libero_spatial done (500 eps, 6539 s), libero_object done (500 eps, 4579 s), libero_goal partial (ended at task 4/10), libero_10 + libero_90 not run. **Locked protocol (2026-08-07, resolved plan):** libero_90 ID (90 tasks × 50 eps) + libero_10 (10 × 10 eps) × 3 seeds × 8 models ≈ 110k eps ≈ **2–4 weeks wall-clock on free T4 (2 workers)** at ~13 s+/episode; training ≈ 3.5–4 days. Reductions only at the B6 gate, recorded in the resolved plan.

### F4. Headless/physics environment risks — `OPEN` (monitored)
- robosuite/MuJoCo need a display/EGL (xvfb/egl on headless Colab); pin robosuite/mujoco versions so physics matches training demo generation (evaluation_plan §5).

### F5. Local env lacks libero/robosuite — `CONFIRMED` constraint
- Verification limited to unit tests (113/113) + Colab logs; no local replay/debug loop for sim issues.

---

## G. Fixed / Closed (audit trail)

| # | Issue | Resolution |
|---|---|---|
| G1 | `strict=False` silent state-dict mismatches (checkpoint/eval load) | `_log_state_dict_mismatch` at both load sites (cli.py) — `c3fbb1d` |
| G2 | BDDL success predicate evaluated 3×/step ($$$ + possible inconsistency) | reward stub + done-as-success probe with fallback; predicate once per step — `0f0bfb3` |
| G3 | Normalizer mean/std on wrong device after `.cuda()` | pinned to model device in `_load_normalizer` — `0f0bfb3` |
| G4 | `time_to_stable_routing` Python loop (Ctrl+C hotspot at `consecutive = 0`) | vectorized via `unfold` sliding sums; reference-loop equivalence tests — `c3fbb1d` |
| G5 | Offline eval ran to completion without ever printing `eval/action_mse` (only after routing metrics) | `eval/action_mse` + RMSE logged immediately after forward pass; `eval/success_rate` as soon as computed — `c3fbb1d` |
| G6 | Top-level `import wandb` crashed without wandb installed | lazy import in `train()` — `c3fbb1d` |
| G7 | Cosmetic metadata: `scratch_moe`/`oracle_moe` missing `stage=2` attr; eval `run_meta` writes `stage: 1` | **FIXED** — `stage = 2` attribute present in both classes (asserted by `test_model_baselines`); eval `run_meta` writes the restored stage |
| G8 | Offline L2 threshold as reported number | superseded (B1) |

---

## H. Risks & Unknowns (watch list)

- **Frozen-encoder jitter** (Stage 2, cached embeddings): R3M/VC-1-style frozen features can produce jitterier trajectories than joint training — validate on ONE suite before committing the sweep.
- **Cached-embedding equivalence** (Stage 2): cached ≠ end-to-end vision quality; plan a compare vs unfrozen encoder on a single suite.
- **LIBERO-PRO overestimation**: standard protocol overestimates generalization (perturbation collapse) — note as limitation in writeup.
- **Dataset revision pinning**: `lerobot`/HF datasets can be re-uploaded — pin revisions for reproducibility (already manifest-pinned by our downloader).
- **Re-ingestion cost**: ~66 GB libero_90 re-ingest on Colab — time and disk budget needed; HDF5s already downloaded (no re-download if cache kept).

---

## Summary — OPEN Items by Priority

| # | Issue | Blocker? | Next action |
|---|---|---|---|
| A2 | Task-pool mismatch (zero-shot confound) | **RESOLVED** 2026-08-07 | libero_90 sole core (train+eval); libero_10 only labeled zero-shot row; spatial/object/goal dropped; `rollout.yaml` updated (Patch 1) |
| B6 | State-replay consistency test | YES — Stage 1 gate | Implemented in census (`build_object_index.py`); **RUN on real cache** (Batch C) |
| E2/E3 | Object-state keys + train/eval parity | YES — Stage 1 diff | Implemented in lockstep (index handshake + eval decode); **verify on real cache** (Batch C) |
| C1 | Confounded 2×2 factorial | YES — design decision | **Implemented** — `phase_pretrain_random_router`, `plain_encoder_phase_bootstrap` + configs + tests; train (Batch C) |
| C7 | Teacher-forced routing cell (decomposable oracle) | YES — design decision (adopted) | **Implemented** as E8 (`teacher_forced.py` + config + tests); train (Batch C) |
| C4 | Phase-label validation | YES — before bootstrap | `spot_check_phase_labels.py` ready; **RUN on real trajectories** (Batch C) |
| B3/D3 | Oracle labeling + stale Report #2 | YES — reporting integrity | Revise Report #2; footnote oracle/teacher-forced in template (next doc round) |
| A4/C5 | Full-length training | High | Run 100/200-epoch schedules (Batch C) |
| C3 | Balance-vs-NMI logging | Medium | **Implemented** (per-epoch diagnostics + `run_balance_sweep.py`); run sweep (Batch C) |
| B4 | ID/OOD declaration | Medium | **Implemented** (`eval/suite_roles` in results); template/writeup pending |
| B2 | Oracle upper-bound validity | **RESOLVED** (2026-08-07) | GT-oracle signature-only reference; teacher-forced (C7/E8) is the valid bound |
| D1 | Novelty (MEAT/SMP + 7 more verified) | Medium | Read full papers; sharpen to centroid bootstrap + oracle/NMI framework |
| D4 | Citation errors in existing docs | Low | Fix authors/IDs in paper draft (verified list in D1 table) |
| F1 | `sim.forward()` question | Low | Colab replay benchmark if pursued |
| B5/F3 | Multi-seed + full-suite runs | High | Schedule full protocol (Batch C) |
| G7 | Metadata cosmetic fix | **FIXED** | stage=2 attributes asserted by tests |
