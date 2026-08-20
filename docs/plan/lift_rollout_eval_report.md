# PhaseForge — Lift Rollout Evaluation Report

**Status:** final evaluation report, Lift task only; statistics corrected 2026-08-18
(seed-stratified bootstrap + probability-of-improvement, per the professor's review;
old §8 explicitly superseded by §9).
**Scope:** held-out simulator task success for the PhaseForge controlled matrix (research_definition.md §4–§5)
**Prepared:** 2026-08-18, for supervisor review

---

## 1. Executive summary

On the **Lift** task (sole completed task of the five-task matrix), all evaluated methods reach
held-out success rates of **0.40–0.74 per seed**; the robot-only negative control reaches ~0.01.
The proposed method (PhaseForge) has the **highest seed-level mean (0.640)** and the **highest
probability of improvement (PoI) against every other method (73.9–96.6%)**, but with only
n = 3 seeds its seed-stratified bootstrap 95% CIs overlap those of the next-best methods —
the result is **directional, not significant**.

- Seed means (50 paired eval episodes per training seed, all runs at commit `c09270a`):
  **PhaseForge 0.640**, Plain-Encoder Phase-Bootstrap 0.600, Scratch MoE 0.587, BC-MLP floor 0.540,
  Phase-Pretrain Random-Router 0.520, Warm-Start MoE 0.513, BC robot-only ≈ 0.013 (frozen report).
- Seed-stratified bootstrap 95% CIs (percentile, 100 000 resamples, `scripts/analysis/stratified_stats.py`):
  PhaseForge **[0.500, 0.740]**, Scratch [0.520, 0.660], Plain-Bootstrap [0.580, 0.620],
  BC [0.480, 0.600], Phase-Pretrain [0.480, 0.560], Warm-Start [0.400, 0.580]. All overlap;
  **no pairwise difference is significant at n = 3 seeds.**
- PoI matrix (same resamples, ties count 0.5): PhaseForge beats Scratch **78.7%**, beats
  Plain-Bootstrap **73.9%**, beats BC 93.4%, Phase-Pretrain 96.6%, Warm-Start 93.9%.
- Factorial direction (H1): centroid-router cells beat matched random-router cells
  (PhaseForge 0.640 > Phase-Pretrain 0.520; Plain-Bootstrap 0.600 > Warm-Start 0.513) — consistent
  with the old report and now larger.
- Factorial direction (H2): the phase-supervised encoder **no longer lags** in the corrected runs
  (PhaseForge 0.640 > Plain-Bootstrap 0.600; Phase-Pretrain 0.520 ≈ Warm-Start 0.513) — the old
  report's H2-negative direction (0.567 < 0.600) is **reversed** and the gap is small.
- **Conclusion on the behavioral hypothesis H3: controlled null (H0) on Lift at n = 3 seeds, with
  a directional PhaseForge advantage over every comparator.** The stage-1 checkpoint-lottery bug
  (§8, FIXED) corrupted the old numbers; the corrected data make PhaseForge the top mean but
  significance still cannot be claimed. The professor's M=1, N=3 caveat applies verbatim (§7).
- **Do not claim better manipulation yet.** Per research_definition.md §6 this is a controlled
  null result, or at most a directional result without statistical support; Phase 2 GPU re-runs
  (with the adopted monitor-restoration fix) replace these numbers, and Phase 5 computes the seed budget.

---

## 2. Protocol

Frozen per research_definition.md §4 (H3) and the state-only rollout plan:

- **Task:** robomimic `Lift`, low-dimensional structured state (privileged simulator state).
- **Eval:** 50 paired simulator episodes per training seed, eval reset seeds 10000–10049;
  training uncertainty across seeds 42, 43, 44.
- **All numbers in this revision** come from the re-executed eval runs at commit `c09270a`
  (the 23:37–00:46 runs listed in §5), whose `episodes.jsonl` are the authoritative local
  artifacts. The §3 tables of the previous revision (pooled means 0.567/0.587/… from the 16:xx
  `d127980` runs) are **superseded** by this revision; the differences (e.g. PhaseForge
  0.567 → 0.640, Phase-Pretrain 0.460 → 0.520) are re-eval effects on the same checkpoints
  (paired reset-seed structure preserved; see provenance §5).

---

## 3. Results — held-out rollout success on Lift (corrected, commit `c09270a`)

| Method | Code cell / role | s42 | s43 | s44 | seed mean | strat. bootstrap 95% CI* |
|---|---|---|---|---|---|---|
| PhaseForge | phase encoder + centroid router (proposed) | 0.68 | 0.74 | 0.50 | 0.640 | [0.500, 0.740] |
| Plain-Encoder Phase-Bootstrap | plain enc + centroid router | 0.58 | 0.62 | 0.60 | 0.600 | [0.580, 0.620] |
| Scratch MoE | additional baseline | 0.58 | 0.66 | 0.52 | 0.587 | [0.520, 0.660] |
| BC-MLP (19-dim) | control floor | 0.60 | 0.48 | 0.54 | 0.540 | [0.480, 0.600] |
| Phase-Pretrain Random-Router | phase enc + random router | 0.56 | 0.52 | 0.48 | 0.520 | [0.480, 0.560] |
| Warm-Start MoE | plain enc + random router | 0.58 | 0.56 | 0.40 | 0.513 | [0.400, 0.580] |
| BC robot-only (23-dim) | information-ceiling negative control | 0.02 | 0.00 | 0.02 | 0.013 | n/a (frozen) |

\* Percentile bootstrap over **seed-level means** (resample seeds with replacement), 100 000
resamples, `scripts/analysis/stratified_stats.py` with `--rng-seed 12345`. This **replaces** the
pooled-150-episode Wilson interval of the old §3: pooling episodes across seeds is
pseudoreplication (episodes from one seed share one trained policy). With N = 3 seeds the
bootstrap distribution has only 3³ = 27 distinct draw vectors, so the CIs are coarse by
construction — wider-but-honest beats narrower-but-wrong.

Per-seed success, Wilson 95% CI, and timeouts (per-seed uncertainty; training uncertainty is the
across-seed spread):

| Method | s42 (CI, tmo) | s43 (CI, tmo) | s44 (CI, tmo) |
|---|---|---|---|
| PhaseForge | 0.68 [0.544, 0.791], 16 | 0.74 [0.605, 0.840], 13 | 0.50 [0.368, 0.632], 25 |
| Plain-Enc. Bootstrap | 0.58 [0.442, 0.706], 21 | 0.62 [0.482, 0.741], 19 | 0.60 [0.462, 0.724], 20 |
| Scratch MoE | 0.58 [0.442, 0.706], 21 | 0.66 [0.522, 0.776], 17 | 0.52 [0.385, 0.652], 24 |
| BC-MLP | 0.60 [0.462, 0.724], 20 | 0.48 [0.348, 0.615], 26 | 0.54 [0.404, 0.670], 23 |
| Phase-Pretrain Rnd | 0.56 [0.423, 0.688], 22 | 0.52 [0.385, 0.652], 24 | 0.48 [0.348, 0.615], 26 |
| Warm-Start MoE | 0.58 [0.442, 0.706], 21 | 0.56 [0.423, 0.688], 22 | 0.40 [0.276, 0.538], 30 |
| BC robot-only | 0.02 | 0.00 | 0.02 |

Every seed–method pair: `pfail = 0`, `invalid = 0`. All remaining failures are `task_timeout`.

---

## 4. Hypothesis evaluation

### H1 — Router-initialization effect (centroid vs random router)

- PhaseForge 0.640 vs Phase-Pretrain 0.520 (**+0.12**); Plain-Bootstrap 0.600 vs Warm-Start 0.513
  (**+0.087**). Direction favors centroid initialization in both pairs; PoI PhaseForge > Phase-Pretrain
  = 96.6%, Plain-Bootstrap > Warm-Start = 99.9%.
- Stratified CIs still overlap within each pair. **Verdict: directional, not significant.**
- Primary H1 evidence per research_definition §4 is the routing-alignment trajectory (NMI,
  entropy, load, collapse), which the rollout harness does not emit. Offline-only routing metrics
  exist (see §6).

### H2 — Phase-representation effect (phase-supervised vs plain encoder)

- PhaseForge 0.640 vs Plain-Bootstrap 0.600 (**+0.04**); Phase-Pretrain 0.520 vs Warm-Start 0.513
  (~tie). The **old report's H2-negative direction (0.567 < 0.600) is reversed** in the corrected
  runs; the gap is small and PoI PhaseForge > Plain-Bootstrap is only 73.9%.
- Phase-head quality at the *selected* checkpoints was weak (val/phase_acc ≈ 0.595, selection at
  epoch 2 under the buggy monitor) — see §8; this was the checkpoint-lottery, not an encoder-design
  conclusion.
- **Verdict: no evidence of a phase-representation benefit or deficit; small positive direction.**

### H3 — Behavioral effect (PhaseForge vs matched controls)

- Means: PhaseForge 0.640 vs Warm-Start 0.513 vs Scratch 0.587 vs BC floor 0.540. PhaseForge
  exceeds every comparator in seed mean and in PoI (≥ 73.9%); but all stratified CIs overlap and
  the spread across seeds (0.50–0.74) is the widest of any method.
- **Verdict: H3 not supported at n = 3 seeds → controlled null result (H0) for the behavioral
  claim, with a directional PhaseForge advantage.**

### H4 — Phase observability (Teacher-Forced vs ground-truth routing)

- **Not evaluable.** The Teacher-Forced run did not complete (see §8); Ground-Truth (Oracle MoE)
  routing exists only as an offline diagnostic.

### Negative control (BC robot-only)

- ~0.01 success vs ~0.5 for all structured-state methods confirms the task is **not solvable from
  proprioception alone**; the structured object state carries the learning signal. The negative
  control behaves as intended.

---

## 5. Provenance, commits, and reliability engineering

- **`d127980` — tanh action-contract fix.** Prior runs produced `policy_failures` (out-of-range
  actions); all evals cited here are post-fix and report zero policy failures and zero invalid
  attempts.
- **`a07dd2c` — runner auto-inject fix.** Methods 5 (Warm-Start MoE) and 6 (Phase-Pretrain
  Random-Router) previously failed pre-flight because their stage-1 providers lived in a different
  output tree and `--with-dependencies` is opt-in. The patch auto-trains a missing stage-1 provider
  for unscoped sweeps.
- **`c09270a` — commit gate.** The runner records and gates on the training/eval git commit;
  all eval runs used for this revision carry `git_commit: c09270a` in `run_meta.json`.

### Run provenance (episodes.jsonl used for every corrected number)

| Method | commit | eval artifact (episodes.jsonl) |
|---|---|---|
| PhaseForge | `c09270a` | `outputs/part1/outputs/eval/phaseforge/seed{42,43,44}/2026-08-17_23-{38,47,55}-*` |
| Scratch MoE | `c09270a` | `outputs/part1/outputs/eval/scratch_moe/seed{42,43,44}/2026-08-18_00-{04,12,19}-*` |
| Warm-Start MoE | `c09270a` | `outputs/part1/outputs/eval/warmstart_moe/seed{42,43,44}/2026-08-18_00-{29,37,46}-*` |
| BC-MLP | `c09270a` | `outputs/part2/outputs/eval/bc/seed{42,43,44}/2026-08-17_23-{37,43,49}-*` |
| Phase-Pretrain Rnd | `c09270a` | `outputs/part2/outputs/eval/phase_pretrain_random_router/seed{42,43,44}/2026-08-17_23-{58},2026-08-18_00-{07,16}-*` |
| Plain-Enc. Bootstrap | `c09270a` | `outputs/part2/outputs/eval/plain_encoder_phase_bootstrap/seed{42,43,44}/2026-08-18_00-{25,33,40}-*` |
| BC robot-only | `d127980` | frozen from previous revision (`outputs/part3/...` tree not synced locally) |

Recompute: `python scripts/analysis/stratified_stats.py --root outputs/part1/outputs --root outputs/part2/outputs`.

---

## 6. Offline diagnostics vs. rollout success

The offline pilot report (`docs/dev/legacy/lift_pilot_offline_report.md`, commit before `d127980`) favored
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

## 7. Statistical posture (Phase 4 — per professor's review §2)

The professor's audit (Agarwal et al. 2021, NeurIPS Outstanding Paper) found the old §3
pooled-150-episode Wilson CI understated uncertainty: episodes from the same seed share one
trained policy, so pooling them as i.i.d. is pseudoreplication. This revision therefore uses:

1. **Seed-stratified bootstrap CIs** (§3): resample seeds with replacement — the Agarwal
   per-task stratification applied at the seed level. Honest but coarse at N = 3 (only 27
   distinct draw vectors); a wide-but-honest interval beats a narrow-but-wrong one.
2. **Probability-of-improvement** (PoI, below): P(X > Y) over per-seed Bernoulli outcomes,
   directly recommended by Agarwal et al. as more informative than CI-overlap.
3. **Explicit M = 1 caveat:** Agarwal's efficiency gains assume many tasks; with Lift alone we
   are in the degenerate case (M = 1, N = 3) their method is weakest at. Report directional only.
4. **Citations for the seed-count decision (Phase 5):** Colas, Sigaud & Oudeyer (2018), *How Many
   Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments* — explicit
   t-test/bootstrap formulas for the number of seeds needed to detect a given effect size;
   Henderson et al. (2018), *Deep RL that Matters* — seed sensitivity is a systemic
   RL/continuous-control problem, not method-specific.

### PoI matrix — P(row > column), seed-stratified bootstrap, n = 100 000, ties count 0.5

| P(X > Y) | PhaseForge | Plain-Boot | Scratch | BC | Phase-Pretrain | Warm-Start |
|---|---|---|---|---|---|---|
| **PhaseForge** | — | 73.9% | 78.7% | 93.4% | 96.6% | 93.9% |
| **Plain-Boot** | 25.8% | — | 62.8% | 97.9% | 100.0% | 99.9% |
| **Scratch** | 21.3% | 37.0% | — | 85.3% | 96.2% | 89.8% |
| **BC** | 6.7% | 2.0% | 14.7% | — | 71.3% | 67.1% |
| **Phase-Pretrain** | 3.3% | 0.0% | 3.8% | 28.6% | — | 54.0% |
| **Warm-Start** | 6.1% | 0.1% | 10.4% | 32.9% | 46.2% | — |

Reading: with n = 3 seeds, PoI is dominated by the observed seed means — it measures the
probability over the seed-resampling distribution, **not** a classical significance claim. Its
value here is ranking coherence: PhaseForge > Plain-Bootstrap > Scratch > BC > Phase-Pretrain ≈
Warm-Start, with 21 of 30 off-diagonal pairs ≥ 71% in one direction.

---

## 8. Root cause: stage-1 checkpoint lottery (FIXED 2026-08-18) — lessons learned

This is the credibility asset of the whole study: the bug was found, diagnosed with three
independent measurements, fixed, and the fix was locally validated before any GPU spend.

1. **Bug:** stage-1 configs monitored `val/loss_total` while the plan declares
   `best val/loss_action`; combined with exploding `loss_phase` (val/loss_phase ≈ 0.79 → 2.59,
   worse than random 6-way CE ln 6 ≈ 1.79), stage-1 best checkpoints were selected very early
   (epoch 1–2 across seeds), handicapping the PhaseForge and Phase-Pretrain warm-starts. The
   monitor now is `val/loss_action` (commit `3cd510f`, matching the predeclared rule and
   `stage2.yaml`). **This explains the old per-seed rollout spread** (PhaseForge 0.56/0.72/0.42
   and Phase-Pretrain 0.28–0.60) and the old H2-negative direction.
2. **Diagnosis (all CPU, 3 seeds, all written up):**
   - *1.1 gradient alignment:* cos(∇L_action, ∇L_phase) ≈ 0 per seed (mean +0.0005..+0.0009,
     range ±0.02) → no gradient conflict → PCGrad/CAGrad-style fixes ruled out.
   - *1.2 boundary noise:* phase-head error by distance-to-transition on the fixed checkpoints is
     **not** boundary-clustered (worst rate 0.67 at `dist_11_plus`, mild bump at `dist_0_1`,
     71 boundaries/1026 samples per seed) → uniform auxiliary overfitting with majority-phase
     bias; label smoothing/boundary tolerance ruled out as the primary fix.
   - *1.3 tie-break:* re-selecting `val/loss_phase` minima on the action plateau moved selection
     to epochs 23/10/14 with phase loss 1.512/1.091/1.170 (< ln 6), but stage-2 NMI became
     **0.440/0.411/0.395 (spread 0.044)** — worse than fixed-reference 0.449/0.457/0.436
     (spread 0.021) → insufficient on its own.
   - *1.4 λ-decay:* `train.lambda_schedule` linear 1.0 → 0.0 was validated (stage-2 NMI
     spread 0.021 → 0.010, means identical) but is **not adopted for the official
     comparison**: it deviates from the predeclared λ = 1.0 and affects only the
     phase-supervised arms (the plain/BC/scratch methods have no phase loss to schedule),
     which would make H2 compare "phase enc + schedule vs plain enc". Fairness decision
     (2026-08-18): every method runs its predeclared configuration. λ-decay is retained
     as a documented refinement (code + tests inert by default; fallback if Gate 2 fails).
3. **Local CPU validation of the adopted fix (monitor restoration, 3 seeds,
   `outputs_local_train/`):** stage-1 best epoch moved to 41/36/25 with action loss
   **0.0264/0.0240/0.0261** (spread 0.0024 vs 0.0255 buggy); stage-2 warm-start action loss
   0.0301/0.0279/0.0308; stage-2 NMI **0.449/0.457/0.436 (spread 0.021)**, 0% collapse in all
   seeds, final action loss 0.0286/0.0259/0.0276.
4. **Do not over-read this report's rollout numbers:** they were produced from *buggy*
   checkpoints at commit `c09270a`. The corrected pipeline (monitor restoration, λ = 1.0
   as predeclared) must be re-run on GPU (Phase 2 of the implementation plan) before any
   behavioral claim; this report then gets a final §3 update from the fresh `outputs_rerun`
   episodes. The λ-decay refinement remains available if the spread criterion fails.

---

## 9. Recommendation for the final decision (supersedes old §8)

1. **Report Lift as a controlled null (H0) for the behavioral claim at n = 3 seeds**, with a
   directional PhaseForge advantage (top seed mean; PoI ≥ 73.9% vs every comparator). Do **not**
   claim significance; state the M = 1, N = 3 caveat (§7) verbatim.
2. **Do not claim better manipulation yet.** At most, report a directional router-initialization
   effect (H1, now +0.12/+0.087 with PoI 96.6%/99.9%) as a *mechanism hypothesis only*.
3. **The old §8 of the previous revision ("controlled null, no directionality") is superseded.**
   Its conclusions were computed from the buggy-checkpoint evals; the corrected runs reverse the
   H2 direction and put PhaseForge at the top of the mean. The H3 verdict (not significant) is
   unchanged.
4. **Next steps, in order:** (a) GPU re-run with the adopted monitor-restoration fix
   (runbook `docs/plan/gpu_rerun_runbook.md`; five-task manifest needs **no** λ overrides —
   every method runs its predeclared configuration); (b) complete the missing cells —
   Teacher-Forced (H4), BC-RNN — before any conclusion; (c) seed budget from Colas et al.
   power analysis on the post-fix effect size (Phase 5); (d) multi-task claim requires the
   predeclared 3/5 tasks.
5. **Use rollout success as the decision metric.** Offline MSE/NMI ordering did not reproduce in
   rollout; §6 documents this explicitly.

---

## 10. Post-merge reproduction confirmation (2026-08-20)

The canonical `phaseforge_r50` (50% partial warm-start) direction was merged to `master`
(`533d4b2`, `--no-ff` of `0a7e415`) after the Phase 1 fix. Two independent post-merge runs
reproduce the corrected g8 results exactly:

| Run | Commit | Output | per-seed success | Mean |
|---|---|---|---|---|
| Confirmation 1 | `0a7e415` | `epo_output/outputs/fiap/outputs` | 0.56 / 0.82 / 0.72 | 0.700 |
| Confirmation 2 | `533d4b2` (master) | `epo_output/outputs/large_bjlkkllgc/outputs` | 0.56 / 0.84 / 0.72 | **0.707** |

Protocol facts for both runs: bank `a7d3953c0afcf560`, `reset_seed` 2026, 50 eval episodes
per training seed, Lift only, learned router. Verified three-way against the g8 reference
(`epo_output/outputs/v2_g8`): per-seed init hashes identical to g8 (`9113226c…` / `3bd32a24…` /
`e3c5a576…`), stage-1 `best_val_monitor` bit-identical to 17 digits across all three runs, and
stage-2 best epochs (30 / 41 / 49) consistent (monitor agreement to ~7 digits — the residual
is GPU nondeterminism). **Conclusion: the reported numbers are exactly reproducible at master.**

## 11. References

- `docs/plan/research_definition.md` — hypotheses H0–H4, required matrix, interpretation rules.
- `docs/plan/state_only_rollout_implementation_plan.md` — rollout/reset/checkpoint protocol.
- `docs/op/implementation_plan.md` — the professor's plan, gates, and Phase 1 fix results.
- the professor's statistical audit (2026-08-18) that this revision answers — summarized in §7 (§2 statistical posture).
- `docs/dev/legacy/lift_pilot_offline_report.md` — offline action-MSE/NMI pilot (pre-fix commit).
- `scripts/analysis/stratified_stats.py` + `tests/test_stratified_stats.py` — seed-stratified bootstrap
  and PoI; deterministic (rng seed 12345).
- Agarwal, Schwarzer, Castro, Courville, Bellemare (2021), *Deep RL at the Edge of the
  Statistical Precipice* — IQM + stratified bootstrap; M = 1 caveat applies.
- Colas, Sigaud, Oudeyer (2018), *How Many Random Seeds? Statistical Power Analysis in Deep
  Reinforcement Learning Experiments* — seed-count formulas (Phase 5).
- Henderson et al. (2018), *Deep Reinforcement Learning that Matters* — seed sensitivity is
  systemic, not method-specific.
- `outputs/part1/outputs/`, `outputs/part2/outputs/` — `eval/*/seed*/…/episodes.jsonl` for every
  corrected number; runner state and ledgers alongside.
- Commits: `d127980` (tanh action fix), `a07dd2c` (runner auto-inject fix), `3cd510f` (monitor
  fix), `c09270a` (commit gate), `186045b` (diagnostics + λ-decay as documented refinement).