# PhaseForge — Lift Pilot: Offline Diagnostic Results

**Report to supervisor** — date 2026-08-14
**Status:** offline single-step diagnostic pilot only; simulator/rollout evaluation is not yet implemented (see §7). This document is an **offline pilot / technical report**; it does not claim task-success improvement.

---

## 1. Executive summary

We ran the full 9-method × 3-seed (42/43/44) matrix on robomimic **Lift** (low-dimensional, proficient-human demos) and evaluated every final-stage checkpoint offline. All **27 evaluated method-seed cells** completed (57 run records total: 27 offline evaluations + 30 training runs) and every result row was validated against the frozen provenance schema.

Central comparison **PhaseForge vs. Scratch MoE vs. Warm-Start MoE** (offline action MSE, mean ± s.d. over 3 seeds):

| Method | action MSE | top-k balance | phase–expert NMI | top-1 collapse |
|---|---|---|---|---|
| **PhaseForge** | **0.02767 ± 0.00107** | **0.991** | **0.430** | 0.00 |
| Scratch MoE | 0.02856 ± 0.00099 | 0.990 | 0.306 | 0.00 |
| Warm-Start MoE | 0.02869 ± 0.00037 | 0.988 | 0.243 | 0.00 |

Routing values above are means over the 3 seeds; per-seed values are in §3.2.

PhaseForge achieves the lowest action error of every learned-routing variant, marginally better than the default behavior-cloning floor (0.02796 ± 0.00052), together with the best expert-load balance, zero expert collapse, and the highest phase–expert alignment (NMI 0.43) of the *learned* routers.

**Full 8-method head-to-head (Δ vs PhaseForge on every key metric, plus cost) is in §3.4** — the table to scan for decision-making; §4 interprets each baseline's role and what it does/does not establish. Routing values there are means; per-seed values are in §3.2. The robot-only BC negative control is reported **separately** in §3.5 and is excluded from the main comparison.

**Honest framing:** these are offline diagnostics. With only 3 seeds no difference is statistically significant (the paired two-sided Wilcoxon in this pipeline has minimum p = 0.25 at n = 3), and — per our own evaluation plan — an offline routing improvement without simulator success is *not* sufficient evidence of a task improvement. The mechanism-level pattern is internally consistent across seeds, but it is **not statistically established** and the behavioral claim awaits the rollout evaluator.

---

## 2. What was actually run (scope)

- **Task:** Lift, single-task, low-dimensional structured state, single-step policy (no temporal history).
- **Data:** robomimic Lift proficient-human low-dim HDF5; trajectory-level split; validation used for checkpoint selection (`best val/loss_action` for stage 2).
- **Models (9):** PhaseForge; Scratch MoE; Warm-Start MoE; Phase-Pretrain Random Router; Plain-Encoder Phase Bootstrap; Teacher-Forced Routing (privileged diagnostic); Oracle MoE (privileged diagnostic); BC-MLP (19-dim floor); **BC-robot-only** (23-dim proprioception-only negative control, §3.5).
- **Seeds:** 42, 43, 44. Stage 1 = 100 epochs / 3,300 steps; Stage 2 = 200 epochs / 6,600 steps. Same budget, architecture, and evaluation seeds across methods.
- **Outcome metrics available in this pilot:** offline action MSE, boundary action smoothness, and the routing diagnostics (entropy, entropy variance, NMI, top-1/top-k balance and collapse, time-to-stable routing). **Task success is not yet measurable** (§7).

Two datasets were produced under different observation schemas and must not be merged: the 19-dim main observation (robot + object state) and the 23-dim robot-only negative control. The summary tables (`outputs/_summaries/*.csv`) group by `(model, tag, stage)` and keep these two BC variants as separate rows (`bc` and `bc`/`robot_only`); all numbers in this report use that split explicitly.

---

## 3. Results

### 3.1 Action reproduction (offline) — mean ± s.d., n = 3 seeds

| Method | stage | action MSE (↓) | boundary smoothness (↓) |
|---|---|---|---|
| **PhaseForge** | 2 | **0.02767 ± 0.00107** | 0.08707 ± 0.00100 |
| BC (default, 19-dim) | 1 | 0.02796 ± 0.00052 | 0.07870 ± 0.00074 |
| Plain-Encoder Phase Bootstrap | 2 | 0.02825 ± 0.00017 | 0.08123 ± 0.00019 |
| Scratch MoE | 2 | 0.02856 ± 0.00099 | 0.08406 ± 0.00108 |
| Warm-Start MoE | 2 | 0.02869 ± 0.00037 | 0.08378 ± 0.00209 |
| Phase-Pretrain Random Router | 2 | 0.02952 ± 0.00112 | 0.09766 ± 0.00339 |
| Teacher-Forced Routing* | 2 | 0.03050 ± 0.00193 | 0.11591 ± 0.00349 |
| Oracle MoE* | 2 | 0.03260 ± 0.00143 | 0.12336 ± 0.00419 |

\* privileged-training diagnostics (expert assigned from phase labels / true phase).

### 3.2 Routing mechanism — per-seed values (42 / 43 / 44) plus mean

Per-seed values are reported so that small differences are not mistaken for stable effects. Collapse is shown as exact per-seed values (0 / 0 / 0 for all learned routers — no variance, so no interval is implied). Means are computed from unrounded per-seed values; displayed per-seed values are rounded.

| Method | NMI (42 / 43 / 44) | NMI mean | top-k balance (42 / 43 / 44) | balance mean | top-1 collapse (42 / 43 / 44) | entropy | time-to-stable (epoch) |
|---|---|---|---|---|---|---|---|
| **PhaseForge** | 0.464 / 0.448 / 0.380 | **0.430** | 0.995 / 0.984 / 0.996 | **0.991** | 0 / 0 / 0 | 0.957 | 13.8 |
| Scratch MoE | 0.309 / 0.263 / 0.345 | 0.306 | 0.992 / 0.997 / 0.981 | 0.990 | 0 / 0 / 0 | 0.951 | 12.7 |
| Warm-Start MoE | 0.211 / 0.282 / 0.236 | 0.243 | 0.991 / 0.979 / 0.995 | 0.988 | 0 / 0 / 0 | 0.981 | 12.0 |
| Plain-Enc. Bootstrap | 0.433 / 0.421 / 0.382 | 0.412 | 0.983 / 0.976 / 0.990 | 0.983 | 0 / 0 / 0 | 0.955 | 12.0 |
| Pretrain Random Router | 0.350 / 0.205 / 0.335 | 0.296 | 0.970 / 0.992 / 0.969 | 0.977 | 0 / 0 / 0 | 0.870 | 18.7 |
| Teacher-Forced* | 0.482 / 0.496 / 0.490 | 0.490 | 0.703 / 0.684 / 0.717 | 0.701 | 0.333 / 0.333 / 0.333 | ~0 | 12.0 |
| Oracle* | 1.000 / 1.000 / 1.000 | 1.000 | 0.754 / 0.754 / 0.754 | 0.754 | 0.333 / 0.333 / 0.333 | ~0 | 12.0 |

Note the seed spread: e.g. PhaseForge NMI ranges 0.380–0.464 and Pretrain Random Router ranges 0.205–0.350, so per-method differences of <0.05 in the means are within the across-seed spread. Entropy and time-to-stable are means over the 3 seeds.

### 3.3 Cost (all seeds completed; per-seed wall time)

| Method | stage | trainable params | wall time (s/seed) |
|---|---|---|---|
| PhaseForge | 1 | 418,243 | 55 |
| PhaseForge | 2 | 210,486 | 142 |
| Scratch MoE | 2 | 382,646 | 159 |
| Oracle MoE | 2 | 381,872 | 139 |
| Teacher-Forced | 2 | 209,712 | 125 |
| Warm-Start MoE | 2 | 210,486 | 144 |
| Plain-Enc. Bootstrap | 2 | 210,486 | 140 |
| Pretrain Random Router | 2 | 210,486 | 142 |
| BC (default, 19-dim) | 1 | 206,983 | 54 |
| BC (robot-only, 23-dim) | 1 | 208,519 | 54 |

### 3.4 Head-to-head comparison vs PhaseForge (decision summary)

Δ = PhaseForge minus baseline on action MSE (**negative → PhaseForge better**); lower action MSE ↓. n = 3 seeds; routing metrics are means over seeds (per-seed values in §3.2). NMI = phase–expert alignment, top-k bal. = expert-load balance, top-1 collapse = fraction of collapsed experts.

| Baseline | role | Δ action MSE | action MSE | NMI | top-k bal. | collapse | params | wall (s/seed) |
|---|---|---|---|---|---|---|---|---|
| **PhaseForge** | proposed method | — | **0.02767 ± 0.00107** | **0.430** | **0.991** | 0.00 | 210,486 | 142 |
| BC-MLP (19-dim) | strong-action floor | +0.00029 | 0.02796 ± 0.00052 | — | — | — | 206,983 | 54 |
| Plain-Encoder Bootstrap | centroid init, no phase head | +0.00058 | 0.02825 ± 0.00017 | 0.412 | 0.983 | 0.00 | 210,486 | 140 |
| Scratch MoE | random init, no Stage 1 | +0.00089 | 0.02856 ± 0.00099 | 0.306 | 0.990 | 0.00 | 382,646 | 159 |
| Warm-Start MoE | encoder warm start, random router | +0.00102 | 0.02869 ± 0.00037 | 0.243 | 0.988 | 0.00 | 210,486 | 144 |
| Phase-Pretrain Random Router | random-router pretraining | +0.00185 | 0.02952 ± 0.00112 | 0.296 | 0.977 | 0.00 | 210,486 | 142 |
| Teacher-Forced* | privileged diagnostic | +0.00283 | 0.03050 ± 0.00193 | 0.490 | 0.701 | 0.33 | 209,712 | 125 |
| Oracle MoE* | privileged diagnostic | +0.00493 | 0.03260 ± 0.00143 | 1.000 | 0.754 | 0.33 | 381,872 | 139 |

\* hard single-expert assignment (diagnostic reference, **not deployable**). The robot-only BC negative control is reported separately in §3.5.

**Decision reads** (all differences descriptive; see §5 for significance):

- **The only credible competition is the BC-MLP floor (+0.00029 on error).** PhaseForge edges out BC on offline action reproduction while adding routing structure — a promising offline signal, but the margin is ~1% of the error value and not significant at n = 3. This is the comparison that decides whether the routing strategy is worth pursuing.
- **The results are consistent with phase-informed initialization helping more than warm-starting alone.** Plain-Encoder Bootstrap (centroid init, no phase head) is the closest learned-routing competitor on both error (+0.00058) and NMI (0.412 vs 0.430); Warm-Start MoE (warm encoder, random router) recovers the *least* phase structure (NMI 0.243). The PhaseForge–Plain-Encoder difference is small and within the across-seed spread (§3.2); the phase-head/centroid bootstrap is the candidate mechanism to test in rollout evaluation.
- **Scratch MoE is 1.8× the parameters of PhaseForge** (382,646 vs 210,486) with no observed offline gain — the phase-pretraining pathway is also the cheaper architecture.
- **Privileged diagnostics are sanity checks, not targets.** Oracle/Teacher-Forced have perfect or near-perfect alignment (NMI 1.00/0.49) but collapse 1/3 of experts and lose on error — confirming hard assignment is a diagnostic reference, not a deployable strategy.

### 3.5 Negative control — robot-only BC (appendix; **not** part of the main comparison)

The robot-only BC policy observes **proprioception only** (23-dim, no object state), i.e. a **different observation schema with different information content** than the 19-dim main setting. Its results are therefore **not a valid direct comparison** against any 19-dim method and must not be read as a performance result.

| Method | stage | obs dims | action MSE | boundary smoothness | params | wall (s/seed) |
|---|---|---|---|---|---|---|
| BC (robot-only) | 1 | 23 (proprioception only) | 0.01975 ± 0.00059 | 0.10393 ± 0.00053 | 208,519 | 54 |

This cell provides a descriptive negative-control reference; because the observation schema differs, it cannot isolate the contribution of object-state information.

---

## 4. Findings and interpretation

1. **PhaseForge is the strongest learned-routing variant in this offline diagnostic.** Lowest action MSE of the learned MoE variants (0.02767), best top-k expert-load balance (0.991), zero collapse, and the highest phase–expert NMI among learned routers (0.43). It also edges out the default BC floor on action MSE — a signal worth carrying into rollout evaluation, not a learned-head replacement claim.

2. **The results are consistent with phase-informed initialization contributing to alignment more than warm-starting alone.** Warm-Start MoE spreads load most uniformly (entropy 0.981, zero collapse) but recovers the *least* phase structure (NMI 0.243). PhaseForge's phase-centroid initialization reaches NMI 0.430 with near-identical balance — i.e. it does not sacrifice load balance to gain alignment. Plain-Encoder Phase Bootstrap (no phase head) lands close to PhaseForge on NMI (0.412); the PhaseForge–Plain-Encoder gap is small and within the across-seed spread, so this is **consistent with** (not proof of) the phase-head/centroid bootstrap being the main alignment contributor.

3. **Hard expert assignment degenerates utilization (sanity check).** Teacher-Forced and Oracle MoE have ~zero routing entropy and 1/3 collapse — exactly what a deterministic single-expert assignment produces. Their higher action MSE (0.0305 / 0.0326) and poorer boundary smoothness confirm that hard routing is a diagnostic reference, not a deployable strategy.

4. **Random-router pretraining is the least stable.** Phase-Pretrain Random Router has the highest routing variance (0.0070), the slowest stabilization (18.7 epochs), and the worst learned balance. Initializing from phases rather than randomness appears more stable.

5. **Robot-only BC is a negative control, not a performance result.** Its lower action MSE (0.01975) reflects a **different observation schema** (proprioception-only, 23-dim, no object state) with different information content; it is not a valid comparison against the 19-dim main setting and cannot be claimed as a general result (§3.5).

---

## 5. Statistical honesty

- **n = 3 seeds.** The paired Wilcoxon in this pipeline is **two-sided** (exact, `alternative="two-sided"`); at n = 3 pairs its p-values are restricted to {0.25, 0.5, 0.75, 1.0}, i.e. 0.25 is the minimum achievable (a one-sided test could reach 0.125, but the pipeline does not use one). **No pairwise difference is significant at any conventional level.** All comparisons above are descriptive.
- Bootstrap 95% CIs (n = 3) are reported in `outputs/_summaries/bootstrap_ci.csv`; e.g. PhaseForge action MSE CI [0.0265, 0.0286] overlaps all other MoE CIs. They quantify seed variation, not significance.
- `action_l2_threshold_rate` is intentionally disabled in the eval config (`metrics.yaml`, `enabled: false`); it is an offline action-agreement proxy and is **not** task success. It is omitted here.
- **The stage-1 phase head used for the PhaseForge bootstrap is only weakly informative at its selected checkpoint.** At the best epoch (chosen on `val/loss_total`), validation phase CE is 0.71–0.80 (vs. 1.79 for a random 6-way guess) and raw phase accuracy 0.62–0.65, but **balanced phase accuracy ≈ 0.49–0.51 and phase–expert NMI ≈ 0** across seeds — consistent with a class-imbalanced offline phase-label set in which the head approximates majority-class prediction. The higher alignment in §4 (NMI 0.43) is measured on the **stage-2 router**, not this head; do not over-read the stage-1 head quality as evidence of phase structure.

---

## 6. Provenance

- Every run carries `run_meta.json`, `resolved_config.yaml`, `metadata/{environment,artifact_manifest,data_provenance}.json`, `metrics/summary.json` + curves, `timings.json`, and a `<run_dir>.completed` lifecycle marker; rows in `_results/results.jsonl` and `_results/training_summary.jsonl` are schema-validated.
- The full sweep ran under a **single commit** `6eb38e8`: 27 evaluated cells + 30 training runs = 57 run records, all carrying that `git_sha`. This is the first complete sweep after the summary-tooling review; the earlier partial `f2fd35b`/`2abafd5` runs are superseded and no numbers mix commits.
- Data-variant separation: `results.jsonl` and `training_summary.jsonl` carry `tag`/`method` written natively by the eval/training paths (`project.tag` from the method's data variant, `project.method` from the runner). The summarizers group by `(model, tag, stage)`, so `bc` and `bc_robot_only` appear as separate rows (verified: `bc` action MSE `0.02796 ± 0.00052`, `bc`/`robot_only` `0.01975 ± 0.00059`).

---

## 7. Limitations and next steps

This pilot **does not yet support any task-success claim**. Missing per the frozen evaluation plan (`docs/plan/final_evaluation_plan.md`):

- **Simulator rollout evaluation** (primary outcome) — the rollout adapter, success predicate, and fixed 50-episode reset bank (seeds 10000–10049) are not implemented; rollout evaluation is deliberately blocked in `cli.py`.
- **Temporal history + BC-RNN baseline** — current models are single-step MLPs.
- **The remaining four tasks** (Can, Square, Tool Hang, Transport) and the five-task macro-average.
- **Multi-task / task-conditioned** setting is out of scope for now.

Recommended order: implement Gate 1 (simulator + scripted-oracle validation) → rollout evaluation on Lift with the paired reset bank → BC-RNN/history baseline → remaining tasks. Only then can the offline PhaseForge pattern be tested against task success, and only then would §4 become behavioral claims rather than mechanistic ones.