# PhaseForge — Lift Pilot: Offline Diagnostic Results

**Report to supervisor** — date 2026-08-14
**Status:** offline single-step diagnostic pilot only; simulator/rollout evaluation is not yet implemented (see §7).

---

## 1. Executive summary

We ran the full 9-method × 3-seed (42/43/44) matrix on robomimic **Lift** (low-dimensional, proficient-human demos) and evaluated every final-stage checkpoint offline. All 57 cells completed and every result row was validated against the frozen provenance schema.

Central comparison **PhaseForge vs. Scratch MoE vs. Warm-Start MoE** (offline action MSE, mean ± s.d. over 3 seeds):

| Method | action MSE | top-k balance | phase–expert NMI | top-1 collapse |
|---|---|---|---|---|
| **PhaseForge** | **0.02767 ± 0.00107** | **0.991** | **0.430** | 0.00 |
| Scratch MoE | 0.02856 ± 0.00099 | 0.990 | 0.306 | 0.00 |
| Warm-Start MoE | 0.02869 ± 0.00037 | 0.988 | 0.243 | 0.00 |

PhaseForge achieves the lowest action error of every MoE variant, marginally better than the default behavior-cloning floor (0.02796 ± 0.00052), together with the best expert-load balance, zero expert collapse, and the highest phase–expert alignment (NMI 0.43) of the *learned* routers.

**Honest framing:** these are offline diagnostics. With only 3 seeds no difference is statistically significant (all paired Wilcoxon p ≥ 0.25), and — per our own evaluation plan — a routing/offline improvement without simulator success is *not* sufficient evidence of a task improvement. The mechanistic signal is real and consistent, but the behavioral claim awaits the rollout evaluator.

---

## 2. What was actually run (scope)

- **Task:** Lift, single-task, low-dimensional structured state, single-step policy (no temporal history).
- **Data:** robomimic Lift proficient-human low-dim HDF5; trajectory-level split; validation used for checkpoint selection (`best val/loss_action` for stage 2).
- **Models (9):** PhaseForge; Scratch MoE; Warm-Start MoE; Phase-Pretrain Random Router; Plain-Encoder Phase Bootstrap; Teacher-Forced Routing (privileged diagnostic); Oracle MoE (privileged diagnostic); BC-MLP (19-dim floor); **BC-robot-only** (23-dim proprioception-only negative control).
- **Seeds:** 42, 43, 44. Stage 1 = 100 epochs / 3,300 steps; Stage 2 = 200 epochs / 6,600 steps. Same budget, architecture, and evaluation seeds across methods.
- **Outcome metrics available in this pilot:** offline action MSE, boundary action smoothness, and the routing diagnostics (entropy, entropy variance, NMI, top-1/top-k balance and collapse, time-to-stable routing). **Task success is not yet measurable** (§7).

Two datasets were produced under different observation schemas and must not be merged: the 19-dim main observation (robot + object state) and the 23-dim robot-only negative control. The pilot's automatic summary tables (`outputs/_summaries/aggregates.csv`) group by `model_name`, which merges these two BC variants under `bc`; **all numbers in this report separate them explicitly** (verified row-by-row from `outputs/_results/results.jsonl`, which records exact checkpoint paths).

---

## 3. Results

### 3.1 Action reproduction (offline) — mean ± s.d., n = 3 seeds

| Method | stage | action MSE (↓) | boundary smoothness (↓) |
|---|---|---|---|
| BC (default, 19-dim) | 1 | 0.02796 ± 0.00052 | 0.07870 ± 0.00074 |
| BC (robot-only, 23-dim) | 1 | 0.01975 ± 0.00059 | 0.10393 ± 0.00053 |
| **PhaseForge** | 2 | **0.02767 ± 0.00107** | 0.08707 ± 0.00100 |
| Plain-Encoder Phase Bootstrap | 2 | 0.02825 ± 0.00017 | 0.08123 ± 0.00019 |
| Scratch MoE | 2 | 0.02856 ± 0.00099 | 0.08406 ± 0.00108 |
| Warm-Start MoE | 2 | 0.02869 ± 0.00037 | 0.08378 ± 0.00209 |
| Phase-Pretrain Random Router | 2 | 0.02952 ± 0.00112 | 0.09766 ± 0.00339 |
| Teacher-Forced Routing* | 2 | 0.03050 ± 0.00193 | 0.11591 ± 0.00349 |
| Oracle MoE* | 2 | 0.03260 ± 0.00143 | 0.12336 ± 0.00419 |

\* privileged-training diagnostics (expert assigned from phase labels / true phase).

### 3.2 Routing mechanism — mean over 3 seeds

| Method | entropy | entropy var. | NMI | top-k bal. | top-1 bal. | top-1 collapse | time-to-stable (epoch) |
|---|---|---|---|---|---|---|---|
| **PhaseForge** | 0.957 | 0.00087 | **0.430** | **0.991** | 0.979 | 0.00 | 13.8 |
| Scratch MoE | 0.951 | 0.00099 | 0.306 | 0.990 | 0.970 | 0.00 | 12.7 |
| Warm-Start MoE | 0.981 | 0.00014 | 0.243 | 0.988 | 0.987 | 0.00 | 12.0 |
| Plain-Enc. Bootstrap | 0.955 | 0.00047 | 0.412 | 0.983 | 0.963 | 0.00 | 12.0 |
| Pretrain Random Router | 0.870 | 0.00700 | 0.296 | 0.977 | 0.978 | 0.00 | 18.7 |
| Teacher-Forced* | ~0 | 0 | 0.490 | 0.701 | 0.701 | 0.333 | 12.0 |
| Oracle* | ~0 | 0 | 1.000 | 0.754 | 0.754 | 0.333 | 12.0 |

### 3.3 Cost (all seeds completed; per-seed wall time)

| Method | stage | trainable params | wall time (s/seed) |
|---|---|---|---|
| PhaseForge | 1 | 418,243 | 57 |
| PhaseForge | 2 | 210,486 | 145 |
| Scratch MoE | 2 | 382,646 | 163 |
| Oracle MoE | 2 | 381,872 | 143 |
| Teacher-Forced | 2 | 209,712 | 132 |
| Warm-Start MoE | 2 | 210,486 | 149 |
| Plain-Enc. Bootstrap | 2 | 210,486 | 149 |
| Pretrain Random Router | 2 | 210,486 | 153 |
| BC (default, 19-dim) | 1 | 206,983 | 53 |
| BC (robot-only, 23-dim) | 1 | 208,519 | 57 |

---

## 4. Findings and interpretation

1. **PhaseForge is the best offline MoE policy.** Lowest action MSE of all MoE variants (0.02767), best top-k expert-load balance (0.991), zero collapse, and the highest phase–expert NMI among learned routers (0.43). It also edges out the default BC floor on action MSE — a strong offline signal for a routing strategy, not a learned head replacement.

2. **Phase supervision is what drives alignment, not warm-starting alone.** Warm-Start MoE spreads load most uniformly (entropy 0.981, zero collapse) but recovers the *least* phase structure (NMI 0.243). PhaseForge's phase-centroid initialization reaches NMI 0.430 with near-identical balance — i.e. it does not sacrifice load balance to gain alignment. Plain-Encoder Phase Bootstrap (no phase head) lands close to PhaseForge on NMI (0.412), isolating the phase-head/centroid bootstrap as the main alignment contributor.

3. **Hard expert assignment degenerates utilization (sanity check).** Teacher-Forced and Oracle MoE have ~zero routing entropy and 1/3 collapse — exactly what a deterministic single-expert assignment produces. Their higher action MSE (0.0305 / 0.0326) and poorer boundary smoothness confirm that hard routing is a diagnostic reference, not a deployable strategy.

4. **Random-router pretraining is the least stable.** Phase-Pretrain Random Router has the highest routing variance (0.0070), the slowest stabilization (18.7 epochs), and the worst learned balance. Initializing from phases rather than randomness is more stable.

5. **Robot-only BC is a negative control, not a performance result.** Its lower action MSE (0.01975) reflects the reduced, uninformative-object observation space on Lift; it is not a valid comparison against the 19-dim main setting and cannot be claimed as a general result.

---

## 5. Statistical honesty

- **n = 3 seeds.** The paired one-sided Wilcoxon (vs. PhaseForge) can only yield p ∈ {0.25, 0.5, 0.75, 1.0}; **no pairwise difference is significant at any conventional level.** All comparisons above are descriptive.
- Bootstrap 95% CIs (n = 3) are reported in `outputs/_summaries/bootstrap_ci.csv`; e.g. PhaseForge action MSE CI [0.0265, 0.0286] overlaps all other MoE CIs. They quantify seed variation, not significance.
- `action_l2_threshold_rate` is intentionally disabled in the eval config (`metrics.yaml`, `enabled: false`); it is an offline action-agreement proxy and is **not** task success. It is omitted here.
- **The stage-1 phase head used for the PhaseForge bootstrap is only weakly informative at its selected checkpoint.** At the best epoch (chosen on `val/loss_total`), validation phase CE is 0.71–0.80 (vs. 1.79 for a random 6-way guess) and raw phase accuracy 0.62–0.65, but **balanced phase accuracy ≈ 0.49–0.51 and phase–expert NMI ≈ 0** across seeds — consistent with a class-imbalanced offline phase-label set in which the head approximates majority-class prediction. The high alignment reported in §4 (NMI 0.43) is measured on the **stage-2 router**, not this head; do not over-read the stage-1 head quality as evidence of phase structure.

---

## 6. Provenance

- Every run carries `run_meta.json`, `resolved_config.yaml`, `metadata/{environment,artifact_manifest,data_provenance}.json`, `metrics/summary.json` + curves, `timings.json`, and a `<run_dir>.completed` lifecycle marker; rows in `_results/results.jsonl` and `_results/training_summary.jsonl` are schema-validated.
- 45 of 57 cells ran under commit `f2fd35b`; the remaining 12 (`warmstart_moe`, `plain_encoder_phase_bootstrap` — stage 2 + eval, 3 seeds) failed mid-run from a checkpoint auto-detect bug (the stage-2 subprocess picked the newer `bc_robot_only` checkpoint, causing an input-dimension mismatch), were fixed in `2abafd5` (runner now passes the exact untagged provider checkpoint as `train.stage1_ckpt_path`), and were re-run cleanly. No numbers mix the two commits; re-run rows carry their own `git_sha`.
- Data-variant separation: `results.jsonl` and `training_summary.jsonl` now carry `tag`/`method`; the summarizers group by `(model, tag, stage)`, so `bc` and `bc_robot_only` appear as separate rows (verified after migration: `bc` action MSE `0.02796 ± 0.00052`, `bc`/`robot_only` `0.01975 ± 0.00059`). Legacy ledgers were migrated with `scripts/backfill_tags.py` (reads `tag` from each run's `run_meta.json`; no numbers changed).

---

## 7. Limitations and next steps

This pilot **does not yet support any task-success claim**. Missing per the frozen evaluation plan (`docs/plan/final_evaluation_plan.md`):

- **Simulator rollout evaluation** (primary outcome) — the rollout adapter, success predicate, and fixed 50-episode reset bank (seeds 10000–10049) are not implemented; rollout evaluation is deliberately blocked in `cli.py`.
- **Temporal history + BC-RNN baseline** — current models are single-step MLPs.
- **The remaining four tasks** (Can, Square, Tool Hang, Transport) and the five-task macro-average.
- **Multi-task / task-conditioned** setting is out of scope for now.

Recommended order: implement Gate 1 (simulator + scripted-oracle validation) → rollout evaluation on Lift with the paired reset bank → BC-RNN/history baseline → remaining tasks. Only then can the offline PhaseForge advantage be tested against task success, and only then would §4 become behavioral claims rather than mechanistic ones.