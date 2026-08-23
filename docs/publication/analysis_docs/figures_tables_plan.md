# Publication Figures & Tables Plan

**Status:** planning (pre-sweep) — frozen asset list for the paper
**Venue convention (for now):** NeurIPS — main text keeps only load-bearing
figures/tables; everything else goes to the appendix. Swap styles at
submission; the asset list itself is venue-agnostic.
**Inputs:** `outputs_final/` (five-task matrix), `outputs_ablation/`
(ablations), summaries from `scripts/analysis/`.
**Claim rules:** every asset below maps to a registered hypothesis or a
pre-declared diagnostic (H1–H6, `research_definition.md` §3–§4;
`final_baselines_plan.md` §9–§10). No asset may convert a routing metric
into a task-success claim, and teacher-forced/oracle rows are never shown as
deployable methods.

---

## 1. Data inventory — what each experiment emits

| Artifact | Fields | Feeds |
|---|---|---|
| `eval_results.json` (per method×task×seed) | `success_rate`, `successes`/`valid_episodes`, Wilson CI low/high, horizon, reset bank id, action MSE | T1, T2, F2, A1, A2 |
| `episodes.jsonl` (per episode) | success, validity, reset-case index, seed, steps-to-outcome, failure category | F2 pairing, A5, A6 |
| `training_summary.jsonl` (per epoch) | train/val action loss, phase loss/acc; Stage-2 routing: `val/phase_expert_nmi`, `val/routing_entropy`, `val/top1_collapse_rate`, expert utilization, switch rate | F3, A3, A7, A8, A11 |
| `metadata/init_routing.json` | t=0 NMI, entropy, expert frequencies, phase-head accuracy (bootstrap instant) | A7, H1 evidence |
| `metadata/init_expert.json` | router type, dropped-neuron set hash, expert-init fingerprint | A10 provenance |
| `run_meta.json` / metadata | git commit, config hash, seeds, package versions, device | A10 |
| `timings.json` | wall-clock per phase | A9 |
| `fairness_accounting.py` output | total vs deployed parameters per method, epoch/step accounting | T3 |
| `stratified_stats.py` output (`_summaries/stratified_stats.json`) | per-task seed means, bootstrap CIs, pairwise P(X>Y) | all statistical overlays |

**Statistical presentation rules (all figures/tables):**
success is a proportion at n=50/seed → always **Wilson 95% intervals**, never
bare bars; with 3 seeds, **plot individual seed points** plus the mean (never
mean-only); the primary comparison family gets the pre-declared multiplicity
correction (Holm, ledger D2 draft); per-episode pairing on identical
reset-case indices is the strongest allowed comparison and is preferred for
PhaseForge-vs-baseline deltas.

---

## 2. Main-content figures

| ID | Figure | Encoding | Source | Evidences | Pri |
|---|---|---|---|---|---|
| **F1** | Method overview: Stage-1 phase-supervised generalist → bootstrap (centroid router, 50% partial warm-start) → Stage-2 autonomous specialization → label-free rollout | Schematic (no data) | — | framing | P0 |
| **F2** | **Headline:** per-task paired success delta (PhaseForge − baseline) with 95% CIs, one row per task, panels for BC / Warm-Start MoE / H1 control (pprr) / H2 control (pepb) | Dot-and-interval (forest) plot; paired per-episode on identical reset cases; zero line = parity | `stratified_stats.json` + `episodes.jsonl` pairing | **H6** | P0 |
| **F3** | Specialization dynamics: `val/phase_expert_nmi` and `val/routing_entropy` vs Stage-2 epoch, PhaseForge vs pprr vs pepb vs scratch_moe (Lift + one harder task) | Line + per-seed thin lines; t=0 marked (bootstrap instant) | `training_summary.jsonl`, `init_routing.json` | **H1/H2 mechanism** | P0 |
| **F4** | Partial warm-start sweep: SR and final NMI vs drop rate (0/25/50/75/100 + full-warm + one-warm points), canonical 50% annotated | Point+CI line, dual axis (SR left, NMI right) | ablation drop-sweep cells | design justification | P1 |
| **F5** | Phase×expert routing heatmaps at end of training: phaseforge vs pf_centroid_random vs pf_random_random | 3 small heatmaps, shared scale | expert utilization / routing records | specialization structure | P1 |

## 3. Main-content tables

| ID | Table | Columns | Source | Evidences | Pri |
|---|---|---|---|---|---|
| **T1** | **Five-task success matrix** — the result table | 10 methods × 5 tasks: mean SR ± Wilson CI (seed points in A1); macro-average column marked secondary; PhaseForge row bolded; paired Δ vs BC with Holm markers in a footer band | `eval_results.json` | **H6** | P0 |
| **T2** | **Causal mechanism controls** (Lift, ablation namespace) | PhaseForge vs pprr (H1), pepb (H2), pf_spherical_kmeans (H3), pf_kmeans, pf_phase_head (H4): SR ± CI, paired Δ, t=0 NMI, final NMI | ablation cells + `init_routing.json` | **H1–H4** | P0 |
| **T3** | Capacity & fairness accounting | per method: total params, deployed params, epochs/steps, BC-RNN 1.16M disclosure, "not history-matched" footnote | `fairness_accounting.py` | capacity defense | P0 |

## 4. Appendix figures & tables

| ID | Asset | Encoding | Source | Pri |
|---|---|---|---|---|
| A1 | Per-seed raw SR for every matrix cell (10×5×3) | full numeric table + seed points | `eval_results.json` | P0 |
| A2 | Offline action MSE (val) matrix | heatmap + numeric | `training_summary.jsonl` | P1 |
| A3 | Training curves, all methods × tasks (val loss; phase loss/acc for phase-supervised rows) | small-multiples lines | `training_summary.jsonl` | P1 |
| A4 | **Full ablation table:** factorial corners (random_random, centroid_random), pf_spherical, pf_ft, pf_k3/k12, corruption (25/50/shuffle), one-warm, full-warm, drop sweep — SR ± CI + final NMI | one grouped table | ablation cells | P0 |
| A5 | Episode outcome/failure-category breakdown per method × task | 100% stacked bars | `episodes.jsonl` | P2 |
| A6 | Steps-to-success ECDFs per method (per task small-multiple or Lift focus) | ECDF with individual ticks | `episodes.jsonl` | P2 |
| A7 | **t=0 routing alignment table** (H1 initial condition): NMI, entropy, expert balance at bootstrap instant, all router-init variants | numeric table + mini-bars | `init_routing.json` | P1 |
| A8 | Expert utilization over Stage-2 (per task, PhaseForge) | stacked area | `training_summary.jsonl` | P2 |
| A9 | Compute cost: wall-clock per phase, total sweep | table | `timings.json` | P2 |
| A10 | **Protocol & provenance:** seeds, frozen reset-bank IDs, commit/config hashes, pinned simulator/robosuite/MuJoCo/torch versions, Holm family definition, dropped-neuron hashes, reproducibility statement (incl. the evaluation-protocol note from the attribution record) | table + text | run metadata | P0 |
| A11 | Router-init family dynamics (Lift): entropy/NMI trajectories for centroid vs kmeans vs spherical-kmeans vs phase-head vs random | line plot | ablation cells | P2 |

Teacher-forced appears **only** in T1/A1 with its diagnostic label (never in
F2's deployable comparisons). Oracle (H5) is optional and currently blocked
by the open D10 fix — if run, it goes to the appendix as an eval-time
intervention, never a method row.

---

## 5. Encoding rules by data type (the "best representation" choices)

| Data type | Chosen encoding | Rejected alternatives & why |
|---|---|---|
| Success proportion, n=50×3 | Dot-and-interval / table with Wilson CI; paired-delta forest plot for comparisons | Bare bars (no uncertainty); SE-of-mean bars (wrong for proportions) |
| 3-seed replicates | Individual points + mean marker | Mean±std only (hides n=3) |
| Comparisons on identical resets | Paired per-episode deltas (forest) | Unpaired group bars (throws away pairing) |
| Training dynamics | Line + thin per-seed lines, t=0 annotated | Smoothed single lines (hide seed spread) |
| Composition over time (expert use) | Stacked area; terminal phase×expert heatmap | Pie charts (no dynamics, poor comparison) |
| Parameter sweeps | Point+CI vs sweep variable, canonical marked | Bar-per-condition (no order) |
| Method×task overview | Annotated heatmap beside the full numeric table | Heatmap alone (imprecise) |
| Outcome/failure categories | 100% stacked bars per method | Grouped bars (categories not shares) |
| Step-count distributions | ECDF | Violin (n≈150 too small, hides tail) |

**Style:** Okabe–Ito colorblind-safe palette; one fixed method→color mapping
across every figure; PhaseForge always first/accent; vector PDF; NeurIPS
column widths (5.5 in text, 2.25 in margin-mini); ≥8 pt type; no chart junk.

## 6. Claims-to-evidence map (submission checklist)

| Claim | Primary evidence | Support |
|---|---|---|
| H6 rollout success (PhaseForge > BC, > Warm-Start) | **F2**, **T1** | A1, A10 |
| H1 router-init effect | **T2** (Δ vs pprr), **F3**, A7 | A11 |
| H2 phase-representation effect | **T2** (Δ vs pepb), **F3** | A3 |
| H3/H4 prior must be phase-structured | **T2** (vs kmeans family, phase-head) | A11 |
| Expert specialization mechanism | **F3**, F5, A8 | A7 |
| R50 design choice (50% partial warm) | F4 | A4 |
| Capacity is not the driver | **T3** | A2 |
| Protocol validity & reproducibility | A10 | attribution record |
| Negative: robot-only fails, teacher-forced = diagnostic only | T1 row labels, A5 | — |

**Open dependencies before production:** D2 multiplicity confirmation
(professor), professor protocol ratification, oracle/D10 decision, and the
completed `outputs_final/` sweep. Plotting scripts to be written under
`scripts/analysis/figures/` — one generator per figure ID, reading only
`_summaries/` + `episodes.jsonl`, no manual data entry anywhere.
