# Detailed Analysis — PhaseForge Oracle-MoE surgical investigation (Waves A–C)

Status: working draft. Supersedes the earlier informal chat review. Every number in this
document was read directly from the run artifacts copied to
`phaseforge_studies/part2/outputs/surgical/` (cloud run of 2026-08-19, reruns included).
Nothing below is inferred beyond what the files state; where the files do not support a
claim, the document says so explicitly.

## 1. Scope and provenance

- Branch `surgical-cpu-analysis`, commit `af46184` (analysis scripts + fixes).
- Training protocol: stage 1 = 100 epochs, stage 2 = 200 epochs, 10-epoch checkpoint
  cadence, best checkpoint selected by validation action-loss monitor (`best_epoch`).
- Evaluation protocol: 50 rollout episodes, horizon 500, deterministic reset bank
  (`a7d3953c0`, `reset_seed=2026`), Wilson 95% CI reported.
- Coverage: A1, A2, A4, A5, B1, B2, B3/B4, C1 — all seed 42 only. A3 is void (see §3.1).
  C2 was not run (part4/part5 artifacts are not on the VM). The seed-43 expansion is
  incomplete: the seed-43 stage-1 run died at epoch 43/100 with no checkpoints written,
  and no stage-2/seed-43 runs exist.

## 2. Results

### 2.1 A1 — checkpoint sweep (seed 42, 50 episodes per checkpoint)

| epoch | SR | CI95 | val loss | NMI | entropy | balance | collapse |
|---|---|---|---|---|---|---|---|
| 3 (val-selected best) | 0.72 | 0.583–0.825 | 0.0270 | 0.469 | 0.932 | 0.869 | 0.167 |
| 10 | 0.74 | 0.604–0.841 | 0.0279 | 0.458 | 0.942 | 0.907 | 0.0 |
| 30 | 0.62 | 0.481–0.741 | 0.0308 | 0.436 | 0.953 | 0.975 | 0.0 |
| 100 | 0.78 | 0.648–0.872 | 0.0337 | 0.467 | 0.953 | 0.987 | 0.0 |
| 200 | 0.78 | 0.648–0.872 | 0.0342 | 0.469 | 0.952 | 0.985 | 0.0 |

Correlation (val loss vs SR, from `sr_val_corr.json`): +0.38 over all five points;
−1.0 over the early pair (e10, e30) — two points only; null over the late pair
(e100, e200) because both SRs are identical.

What this does and does not show:

- All five CIs overlap (0.481–0.872). There is no significant peak; the val-selected
  checkpoint (e3, SR 0.72) is not significantly worse than the observed peak (0.78).
- The sweep therefore provides no evidence that checkpoint selection is a dominant
  variance source, and no evidence for an optimal epoch.
- It does not rule out a small real trend; 50 episodes per checkpoint and a single seed
  cannot resolve differences of the observed size (~0.06–0.16).

### 2.2 A3 — validation banks: VOID

The four bank runs are four separate training runs (checkpoint hashes differ, val losses
differ in float-noise decimals only) but all four share the **identical validation fold**:
`data.split.seed` is inert when the HDF5 train/valid filters are enabled
(`phaseforge/data/ingestion/state_machine.py`, dataset-filter branch short-circuits the
split RNG). All four produce SR 0.72 with identical CI. This result tests nothing and
must not be cited. The script was fixed (filters disabled) but not re-run.

### 2.3 B1 — four-way init (seed 42)

| cell | SR | CI95 | best_epoch (stage-2 summary) |
|---|---|---|---|
| cent_warm (default) | 0.72 | 0.583–0.825 | 3 |
| cent_reset (random experts) | 0.38 | 0.259–0.518 | 19 |
| rand_warm (random router) | 0.44 | 0.312–0.577 | 4 |
| rand_reset (both) | 0.44 | 0.312–0.577 | 39 |

- cent_warm vs cent_reset: CIs are disjoint → the drop is significant *at this seed*.
- cent_warm vs rand_warm: CIs are adjacent but non-overlapping (0.583 vs 0.577) →
  borderline at this seed.
- cent_reset vs rand_warm / rand_reset: CIs overlap → the evidence cannot order
  "expert reset is worse than router reset" (0.38 vs 0.44).
- The findings payloads do not record `best_epoch`; values above come from the stage-2
  run summaries.

### 2.4 B3/B4 — ablation grid (seed 42)

| cell | SR | CI95 |
|---|---|---|
| bc=0.0 | 0.74 | 0.604–0.841 |
| bc=0.01 (default) | 0.72 | 0.583–0.825 |
| bc=0.1 | 0.78 | 0.648–0.872 |
| rn=0.0 | 0.76 | 0.626–0.857 |
| rn=0.1 (default) | 0.72 | 0.583–0.825 |
| rn=0.5 | 0.40 | 0.276–0.538 |

- balance_coeff: all three CIs overlap → no effect detected at this seed.
- router noise 0.5: CI is disjoint from the other five cells → harmful *at this seed*.
- Note: the bc=0.0 run trains without any balance term yet lands in the same CI band —
  consistent with the balance coefficient having little measurable effect here.

### 2.5 A4 — phase×expert specialization matrix (seed 42, n=1026)

- Diagonal (true router) 0.698; diagonal (predicted router) 0.776.
- Per-phase samples: [161, 15, 385, 352, 108, 5]. Dominant expert per phase (true): [0, 1, 3, 3, 4, 5].
- The largest confusion is phase 2 → expert 3 (row 2 puts 0.502 on expert 3); phases 1
  and 5 are nearly empty in this sample.
- Implication (limited): the router's specialization does not separate phases 2 and 3
  in the trained matrix.

### 2.6 A5 — routing counterfactuals (action MSE, seed 42)

| variant | action MSE |
|---|---|
| learned | 0.0600 |
| oracle_true | 0.0725 |
| oracle_pred | 0.0739 |
| uniform | 0.0366 |
| random | 0.0357 |

- Learned routing is worse than both uniform and random on validation action MSE; the
  two oracle variants are worse still.
- Phase 0 carries most of the learned deficit (0.159 vs 0.047 uniform).
- Caveats: this is action MSE on the validation actions — it is not a rollout SR, and no
  counterfactual SR was measured. It shows the router hurts action prediction, not that
  it hurts task success.
- The selection caveat applies: the same validation split was used for checkpoint
  selection, so these numbers are optimistic by an unknown amount.

### 2.7 B2 — expert divergence over training (seed 42)

| epoch | 10 | 30 | 100 | 200 |
|---|---|---|---|---|
| mean off-diagonal | 0.297 | 0.139 | 0.214 | 0.231 |

- Divergence dips sharply at e30 and partially recovers. The dip coincides with the SR
  dip at e30 (0.62), but e10 has the highest divergence and only mid SR (0.74) — so the
  data support no monotonic "more divergence → better SR" claim.

### 2.8 C1 — latent geometry (seed 42, n=1026)

- Mean silhouette 0.278; per-phase: [0.461, 0.590, 0.069, 0.091, 0.187, 0.272].
- Phases 2 and 3 are not separable: inter-centroid distance 3.41 is smaller than the
  intra-phase spread (7.33 / 7.21). Phases 1 and 5 are nearly empty (15 / 1026 and
  5 / 1026 samples).
- Consistent with A4 (phase 2 → expert 3 confusion) and A5 (phase 0 dominates the
  routing deficit).

## 3. Integrity notes

1. **A3 is void** (§2.2). Do not cite the four identical 0.72 SRs.
2. `checkpoint_sweep.json` is absent from the local copies; the A1 table was
   reconstructed from the five eval `rollout_summary.json` files and the stage-2
   `training_curves.jsonl`. `sr_val_corr.json` (which read the sweep findings on the VM)
   is quoted directly.
3. C2 (failure-by-phase) is unavailable. In every recorded eval in this run the only
   failure category is `task_timeout` (13/14/11/12/19/28/30/31 of 50 episodes depending
   on the cell) — i.e., failed episodes ran to the horizon without success. No per-phase
   failure attribution exists for this run.
4. The seed-43 expansion is incomplete (stage 1 died at epoch 43/100; no stage 2).
5. Reproducibility: fresh evaluation processes on the rerun reproduced bit-identical
   SR + CI for all B1/grid cells. This proves evaluation determinism, not
   seed-robustness.
6. The "0.72 cluster": seven distinct runs (cp_ebest, four vbank, cent_warm, bc=0.01,
   rn=0.1) produced the identical 50-case outcome pattern. All are best-epoch-3
   checkpoints from the same stage-1 init; the 50-case bank cannot discriminate
   near-identical policies. SR differences of ≤0.06 among these runs are noise.

## 4. What the evidence supports (and does not)

Supported, single-seed only:

- Checkpoint selection is not the dominant variance source (A1; corroborated by the
  0.72 cluster).
- Router/expert initialization matters more than the balance/noise hyperparameters:
  resets drop SR to 0.38–0.44 while bc∈{0, 0.01, 0.1} is flat (B1, B3).
- Router noise 0.5 is harmful (B4).
- The trained router does not route phases to distinct experts usefully: phases 2/3 are
  confusable in the matrix and the latent space (A4, C1), and routing is worse than
  uniform/random on action prediction (A5).

Not supported:

- That phaseforge (0.72, 1 seed) improves over the real-matrix phaseforge baseline
  (0.640, seeds 42/43/44): the 0.72 CI [0.583, 0.825] overlaps that baseline.
- That expert reset is worse than router reset (CIs overlap).
- Any ordering among checkpoints e10/e100/e200 (CIs overlap).
- That the action-MSE deficit of the learned router translates to an SR deficit
  (no counterfactual rollouts).
- Any claim across seeds: everything here is seed 42.

## 5. Open work

- Seed expansion (A1 for 43/44, then B1 + grid at 42/43/44) before any of §4's claims
  can be stated as multi-seed. Note the crashed seed-43 stage-1 run is now detected and
  retrained (fix `af46184`).
- Optional: A3 rerun with the fixed script if bank-level variance is wanted.
- C2 requires the part4/part5 artifacts on the VM.