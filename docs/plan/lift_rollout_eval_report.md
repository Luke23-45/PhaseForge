# PhaseForge — Lift Rollout Evaluation Report

**Status:** final evaluation report, Lift task only
**Scope:** held-out simulator task success for the PhaseForge controlled matrix (research_definition.md §4–§5)
**Prepared:** 2026-08-18, for supervisor review

---

## 1. Executive summary

On the **Lift** task (sole completed task of the five-task matrix), all evaluated methods reach
held-out success rates of **0.42–0.72 per seed** with one exception: the robot-only negative control,
which reaches **~0.01**. The proposed method (PhaseForge) does **not** beat its matched controls.

- Pooled means (150 episodes, 3 training seeds × 50 paired eval episodes): **PhaseForge 0.567**,
  Scratch MoE 0.587, BC-MLP floor 0.540, Warm-Start MoE 0.513, Phase-Pretrain Random-Router 0.460,
  Plain-Encoder Phase-Bootstrap 0.600, BC robot-only 0.013.
- All 95% Wilson intervals overlap. **No pairwise difference is statistically significant** at
  n = 3 seeds × 50 episodes.
- Factorial direction (H1): centroid-router cells beat matched random-router cells in the means
  (PhaseForge 0.567 > Phase-Pretrain 0.460; Plain-Bootstrap 0.600 > Warm-Start 0.513), but the
  gaps are within across-seed spread.
- Factorial direction (H2): the phase-supervised encoder does **not** help in the means
  (PhaseForge 0.567 < Plain-Bootstrap 0.600; Phase-Pretrain 0.460 < Warm-Start 0.513).
- **Conclusion on the behavioral hypothesis H3: controlled null result (H0) on Lift.** PhaseForge
  does not improve task success over matched controls. The routing-mechanism story (H1) is
  directionally suggestive but not significant; there is no evidence of a phase-representation
  benefit (H2).
- **Do not claim better manipulation.** Per research_definition.md §6, this is a controlled null
  result, or at most a directional routing-mechanism result without behavioral support.

---

## 2. Protocol

Frozen per research_definition.md §4 (H3) and the state-only rollout plan:

- **Task:** robomimic `Lift`, low-dimensional structured state (privileged simulator state).
- **Eval:** 50 paired simulator episodes per training seed, eval reset seeds 10000–10049,
  per-episode 95% Wilson intervals; training uncertainty across seeds 42, 43, 44.
- **All runs** used commit `d127980` (tanh action-contract fix) or `a07dd2c` (runner auto-inject
  fix); see §5. Every eval reports **0 `policy_failures` and 0 `invalid_attempts`** — the only
  failure category is `task_timeout` (500-step limit).

---

## 3. Results — held-out rollout success on Lift

| Method | Code cell / role | s42 | s43 | s44 | pooled mean | pooled Wilson 95% CI* |
|---|---|---|---|---|---|---|
| PhaseForge | phase encoder + centroid router (proposed) | 0.56 | 0.72 | 0.42 | 0.567 | [0.487, 0.643] |
| Scratch MoE | additional baseline | 0.58 | 0.66 | 0.52 | 0.587 | [0.507, 0.662] |
| BC-MLP (19-dim) | control floor | 0.60 | 0.48 | 0.54 | 0.540 | [0.460, 0.618] |
| Plain-Encoder Phase-Bootstrap | plain enc + centroid router | 0.58 | 0.62 | 0.60 | 0.600 | [0.520, 0.675] |
| Warm-Start MoE | plain enc + random router | 0.58 | 0.56 | 0.40 | 0.513 | [0.434, 0.592] |
| Phase-Pretrain Random-Router | phase enc + random router | 0.60 | 0.50 | 0.28 | 0.460 | [0.382, 0.540] |
| BC robot-only (23-dim) | information-ceiling negative control | 0.02 | 0.00 | 0.02 | 0.013 | [0.004, 0.047] |

\* Wilson 95% CI over the 150 pooled episodes; **diagnostic only**. The protocol's declared
per-episode uncertainty is the per-seed Wilson interval, and training uncertainty is the
across-seed spread (per-seed intervals below).

Per-seed success, Wilson 95% CI, and timeouts:

| Method | s42 (CI, tmo) | s43 (CI, tmo) | s44 (CI, tmo) |
|---|---|---|---|
| PhaseForge | 0.56 [0.423, 0.688], 22 | 0.72 [0.583, 0.825], 14 | 0.42 [0.294, 0.558], 29 |
| Scratch MoE | 0.58 [0.442, 0.706], 21 | 0.66 [0.522, 0.776], 17 | 0.52 [0.385, 0.652], 24 |
| BC-MLP | 0.60 [0.462, 0.724], 20 | 0.48 [0.348, 0.615], 26 | 0.54 [0.404, 0.670], 23 |
| Plain-Enc. Bootstrap | 0.58 [0.442, 0.706], 21 | 0.62 [0.482, 0.741], 19 | 0.60 [0.462, 0.724], 20 |
| Warm-Start MoE | 0.58 [0.442, 0.706], 21 | 0.56 [0.423, 0.688], 22 | 0.40 [0.276, 0.538], 30 |
| Phase-Pretrain Rnd | 0.60 [0.462, 0.724], 20 | 0.50 [0.366, 0.634], 25 | 0.28 [0.175, 0.417], 36 |
| BC robot-only | 0.02, 49 | 0.00, 50 | 0.02, 49 |

Every seed–method pair: `pfail = 0`, `invalid = 0`. All remaining failures are `task_timeout`.

---

## 4. Hypothesis evaluation

### H1 — Router-initialization effect (PhaseForge vs Phase-Pretrain Random-Router)

- Means: 0.567 vs 0.460. Direction favors centroid initialization (~+0.11), but per-seed Wilson
  intervals overlap and the across-seed spread of each method (PhaseForge s44 0.42; Phase-Pretrain
  s44 0.28) exceeds the gap.
- Primary H1 evidence per research_definition §4 is the **routing-alignment trajectory** (NMI,
  entropy, load, collapse), which the rollout harness does not emit. Offline-only routing metrics
  exist (see §6) and cannot substitute for paired rollout success.
- **Verdict: not significant on Lift; mechanism unverified by rollout.**

### H2 — Phase-representation effect (PhaseForge vs Plain-Encoder Phase-Bootstrap)

- Means: 0.567 vs 0.600. Direction favors the **plain** encoder, i.e., **no benefit from the
  phase-supervised encoder**. Consistent in the random-router pair too (0.460 vs 0.513).
- Phase-head quality is weak: stage-1 best checkpoint (seed 42) has `val/phase_acc ≈ 0.595`,
  `val/phase_balanced_acc ≈ 0.563`, final `loss_phase ≈ 2.59`; the best epoch was selected at
  epoch 2 because `val/loss_total` (the configured monitor) rises as `loss_phase` explodes. The
  encoder therefore carries little usable phase structure, which may explain the null.
- **Verdict: no evidence of phase-representation benefit.**

### H3 — Behavioral effect (PhaseForge vs Warm-Start MoE and Scratch MoE)

- Means: PhaseForge 0.567 vs Warm-Start 0.513 vs Scratch 0.587. PhaseForge is above Warm-Start in
  the mean, below Scratch, and not distinct from either. It does not beat the matched baselines and
  does not exceed the BC-MLP floor (0.540).
- **Verdict: H3 not supported on Lift → controlled null result (H0) for the behavioral claim.**

### H4 — Phase observability (Teacher-Forced vs ground-truth routing)

- **Not evaluable.** The Teacher-Forced run did not complete (see §7). Ground-Truth (Oracle MoE)
  routing exists only as an offline diagnostic.

### Negative control (BC robot-only)

- ~0.01 success vs ~0.5 for all structured-state methods confirms the task is **not solvable from
  proprioception alone**; the structured object state carries the learning signal. The negative
  control behaves as intended.

---

## 5. Provenance, commits, and reliability engineering

- **`d127980` — tanh action-contract fix.** Prior runs produced `policy_failures` (out-of-range
  actions); all evals cited here are post-fix and report zero policy failures and zero invalid
  attempts. This makes the success numbers directly comparable.
- **`a07dd2c` — runner auto-inject fix.** Methods 5 (Warm-Start MoE) and 6 (Phase-Pretrain
  Random-Router) previously failed pre-flight because their stage-1 providers lived in a different
  output tree and `--with-dependencies` is opt-in. The patch auto-trains a missing stage-1 provider
  for unscoped sweeps. The 19:xx rerun (part3 nested tree, commit `a07dd2c`) auto-injected
  `bc` stage 1 (19:07) and `phaseforge` stage 1 (19:33) and completed both methods. Verified by
  502 passing tests, ruff, and mypy.

### Run provenance (commit per eval)

| Method | commit | eval artifact (rollout_summary.json) |
|---|---|---|
| PhaseForge | `d127980` | `outputs/part1/outputs/eval/phaseforge/seed{42,43,44}/2026-08-17_16-0{6,15,22}_*` |
| Scratch MoE | `d127980` | `outputs/part1/outputs/eval/scratch_moe/seed{42,43,44}/2026-08-17_16-3{1,8,5}_*` |
| BC-MLP | `d127980` | `outputs/part2/outputs/eval/bc/seed{42,43,44}/2026-08-17_16-0{8,15,21}_*` |
| Plain-Enc. Bootstrap | `d127980` | `outputs/part2/outputs/eval/plain_encoder_phase_bootstrap/seed{42,43,44}/2026-08-17_16-2{8,5,2}_*` |
| Warm-Start MoE | `a07dd2c` | `outputs/part3/outputs/part1/eval/warmstart_moe/seed{42,43,44}/2026-08-17_19-{11,19,27}_*` |
| Phase-Pretrain Rnd | `a07dd2c` | `outputs/part3/outputs/part1/eval/phase_pretrain_random_router/seed{42,43,44}/2026-08-17_19-{36,44,53}_*` |
| BC robot-only | `d127980` | `outputs/part3/outputs/eval/bc/seed{42,43,44}/2026-08-17_17-{27,36,46}_*Lift__robot_only*` |
| Oracle MoE (offline) | `d127980` | `outputs/part3/outputs/_results/results.jsonl` (run_ids `5fc2c10e`, `0adee31c`, `a041c106`) |

Runner state/ledger: `outputs/{part1,part2,part3}/outputs/_runner/state.json`,
`outputs/part3/outputs/part1/_runner/state.json`, and `_ledger/runs.jsonl`.

---

## 6. Offline diagnostics vs. rollout success

The offline pilot report (`docs/dev/lift_pilot_offline_report.md`, commit before `d127980`) favored
PhaseForge on offline metrics: action MSE **0.02767** (best), phase-expert **NMI 0.430** (best of
learned routers), top-k balance **0.991**, zero collapse. The rollout results do **not** reproduce
that ordering. Offline MSE/NMI therefore did **not** translate into held-out success; this is a
caution against using offline metrics as the decision metric.

Oracle MoE (ground-truth routing, offline diagnostic): `phase_expert_nmi = 1.0` (perfect
phase–expert alignment by construction), `routing_entropy ≈ 0` (fully peaked, deterministic),
`topk_balance_score ≈ 0.754`, `topk_collapse_rate = 0.333`, `action_mse ≈ 0.031` — consistent
across seeds. It is a diagnostic reference only, not a deployable method, and its task success is
**not** a PhaseForge upper bound (research_definition §4).

---

## 7. Caveats and missing cells

- **Statistical power:** n = 3 seeds. All pairwise Wilson intervals overlap; the three largest
  apparent gaps (PhaseForge vs Phase-Pretrain, PhaseForge vs Warm-Start, Plain-Bootstrap vs
  Warm-Start) are within across-seed spread. No significance test is predeclared beyond the Wilson
  intervals and seed spread specified in the protocol.
- **Checkpoint-rule deviation (root cause; FIXED 2026-08-18):** stage-1 configs monitored `val/loss_total`
  while the plan declares `best val/loss_action`; combined with exploding `loss_phase` (val/loss_phase
  ≈ 0.79 → 2.59, worse than random 6-way CE ln 6 ≈ 1.79), stage-1 best checkpoints were selected very
  early (epoch 1–2 across seeds) with a poorly-trained encoder/action head, handicapping the PhaseForge
  and Phase-Pretrain warm-starts. The monitor is now `val/loss_action` (matching the predeclared rule
  and `stage2.yaml`). **This also explains the per-seed rollout spread** (PhaseForge 0.56/0.72/0.42):
  stage-1 best-checkpoint quality varied per seed (action loss 0.0451/0.0404/0.0659 at epochs 2/2/1),
  and stage-2 freezes that encoder and bootstraps the router from its centroids, so the variance
  propagated into warm-start quality (0.0411/0.0372/0.0513) and final routing NMI (0.393/0.377/0.324).
  Control proof: `plain_encoder_phase_bootstrap` consumes BC's *consistently good* stage-1
  (0.0277/0.0270/0.0261) and has the smallest spread (0.04), while both methods consuming PhaseForge
  stage-1 have the largest (0.30/0.32); eval noise is negligible (duplicate eval batches per seed
  agree to ≤0.02), and the data split is identical across training seeds (split.seed=42 fixed).
  **3-seed local CPU validation with the fix (`outputs_local_train/`, git `3cd510f`):**
  stage-1 best epoch moved to 41/36/25 with action loss **0.0264/0.0240/0.0261** (spread 0.0024 vs
  0.0255 buggy); stage-2 warm-start action loss tightened to **0.0301/0.0279/0.0308** (spread 0.0029
  vs 0.0141 buggy, on par with plain-encoder bootstrap 0.003); stage-2 NMI held at
  **0.449/0.457/0.436** (spread 0.021 vs 0.069 buggy), 0% collapse in all seeds, final action loss
  0.0286/0.0259/0.0276. **The rollouts in this report were produced before this fix and must be
  re-run on GPU before the fixed pipeline is judged.**
- **Missing cells:**
  - Teacher-Forced Routing (method 8, H4): failed pre-flight on the pre-patch runner
    (`d127980`, 2026-08-17 17:57, `outputs/part3/outputs/_runner/state.json`). Runnable now under
    `a07dd2c`; not yet rerun.
  - BC-RNN (temporal control floor, index 46): not run.
  - Other four tasks (Can, Square, Transport, Tool Hang): not evaluated. H3 requires a predeclared
    majority of tasks to be claimed.
  - Paired reset-seed rollout per training seed is in place but cross-seed paired hypothesis tests
    were not predeclared.
- **Parameter budgets** are not equalized (BC-MLP ≈ 0.21M, MoE ≈ 0.21–0.38M params; see pilot
  report §3.3). The MoE cost premium is real but not the decisive issue here since all MoEs land
  near the BC floor.

---

## 8. Recommendation for the final decision

1. **Report Lift as a controlled null (H0) for the behavioral claim.** PhaseForge matches its
   matched controls and the BC floor within noise; it does not beat Scratch MoE or Plain-Encoder
   Bootstrap.
2. **Do not claim better manipulation.** At most, report a directional router-initialization effect
   (H1) as a *mechanism hypothesis only*, explicitly labeled not significant on rollout success.
3. **Complete the matrix before any conclusion:** rerun Teacher-Forced with the patched runner
   (`a07dd2c`) to close H4; run BC-RNN; run the remaining tasks if the claim is to be
   multi-task; consider more seeds (or a predeclared paired test) before claiming any effect.
4. **Stage-1 checkpointing fixed** (monitor now `val/loss_action`, matching the predeclared plan rule).
   The `loss_phase` explosion (val/loss_phase 2.59 > random 1.79) remains a phase-head overfitting
   concern worth a follow-up (class weighting / earlier phase-head regularization), but the primary
   selection bug is resolved and locally validated.
5. **Use rollout success as the decision metric.** Offline MSE/NMI ordering did not reproduce in
   rollout; §6 documents this explicitly.

---

## 9. References (in-repo artifacts)

- `docs/plan/research_definition.md` — hypotheses H0–H4, required matrix, interpretation rules.
- `docs/plan/state_only_rollout_implementation_plan.md` — rollout/reset/checkpoint protocol.
- `docs/dev/lift_pilot_offline_report.md` — offline action-MSE/NMI pilot (pre-fix commit).
- `outputs/part1/outputs/`, `outputs/part2/outputs/`, `outputs/part3/outputs/` — runner state,
  ledgers, checkpoints, and `eval/*/seed*/…/rollout_summary.json` for every cited number.
- Commits: `d127980` (tanh action fix), `a07dd2c` (runner auto-inject fix).
- `phaseforge/runner/cli.py`, `phaseforge/runner/protocol.py`, `tests/test_runner.py` — the
  `a07dd2c` change surface.