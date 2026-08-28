# Publication Figures & Tables Plan — v1.0 (finalized)

**Status:** finalized pre-sweep asset plan; field-level coverage audited
against the actual artifact schemas (verified on real run outputs).
**Venue convention (for now):** NeurIPS — main text keeps only load-bearing
figures/tables; everything else in the appendix. Swap styles at submission.
**Inputs:** `outputs_final/` (five-task matrix), `outputs_ablation/`
(ablations), summaries from `scripts/analysis/`.
**Claim rules:** every asset maps to a registered hypothesis or pre-declared
diagnostic (H1–H6). No asset may convert a routing metric into a task-success
claim; teacher-forced/oracle rows are never shown as deployable methods.

---

## 1. Data inventory — every generated artifact and field (verified schemas)

| Artifact | Fields (exact, from live runs) | Consumed by |
|---|---|---|
| `eval_results.json` | action MSE; `rollout/success_rate, successes, valid_episodes, policy_failures, invalid_attempts, wilson_ci95_low/high, horizon, reset_bank` | T1, T2, F2, A1, A2 |
| `rollout_summary.json` | run_id, model, tag, task, training_seed, **reset_seed, reset_bank, checkpoint_sha256, router_mode**, horizon, episodes, **failure_categories{}**, metrics{} | A5, A10 (eval↔checkpoint linkage) |
| `episodes.jsonl` | run_id, model, checkpoint_sha256, task, seeds, episode_index, **valid_episode, steps, success, timed_out, termination_reason, extra.max_phase** | F2 pairing, A5, A6, A14 |
| `training_curves.jsonl` (per run) | epoch, global_step, epoch_wall_seconds, **peak_gpu_memory_mb, train_steps_per_second**, checkpoint_monitor(+value); train: **loss_action, loss_balance, loss_sticky, loss_teacher_kl, loss_total, lr, teacher_lambda**; val: **loss_action, loss_total, phase_expert_nmi, routing_entropy, routing_switch_rate, top1/topk_balance_score, top1/topk_collapse_rate** | F3, F4, A2, A3, A8, A9, A11 |
| `summary.json` (stage-1 runs) | final-epoch stage-1 metrics incl. phase supervision | A3, A7 context |
| `metadata/init_routing.json` | t0 NMI, mean/normalized entropy, top1/topk CoV, top1 expert frequencies, collapse rate, dead experts, phase-head accuracy | A7, T2 |
| `metadata/init_expert.json` | router init type/seed, expert-init type, drop rate, dropped-neuron count/list/hash, training seed | A4 context, A10 |
| `metadata/environment.json` | git branch/commit, hostname, platform, python, **full package versions**, device | A10 |
| `metadata/data_provenance.json` | data config hash, state schema keys/dims, normalization, action contract | A10, A13 |
| `metadata/artifact_manifest.json` | per-artifact sha256 + sizes | A10 integrity statement |
| `run_meta.json`, `resolved_config.yaml`, `timings.json` | identity, full resolved hyperparameters, wall-clock per phase | A9, A10, A12 |
| Sweep level: `_runner/plan.json`, `_runner/state.json`, `_gates/gates_report.json`, `_summaries/` (stratified_stats.json, paired_wilcoxon.csv, train/eval summaries), fairness output | selection provenance, completion registry, gate results, statistics, parameter accounting | A10, A15, T3 |

**Statistical presentation rules (all assets):** success is a proportion at
n=50/seed → always **Wilson 95% intervals**; with 3 seeds, always plot
**individual seed points** plus the mean; the primary comparison family gets
the pre-declared Holm correction (D2); per-episode pairing on identical
reset-case indices is the strongest allowed comparison form.

---

## 2. Main-content figures

| ID | Figure | Encoding | Source | Evidences | Pri |
|---|---|---|---|---|---|
| **F1** | Method overview: Stage-1 phase-supervised generalist → bootstrap (centroid router, 50% partial warm-start) → Stage-2 autonomous specialization → label-free rollout | Schematic | — | framing | P0 |
| **F2** | **Headline:** per-task paired success delta (PhaseForge − baseline), 95% CIs; panels for BC / Warm-Start MoE / H1 control / H2 control | Forest (dot-and-interval), paired per-episode on identical reset cases; zero = parity | stratified_stats + episodes.jsonl | **H6** | P0 |
| **F3** | Specialization dynamics: `val/phase_expert_nmi`, `val/routing_entropy`, **and `val/routing_switch_rate`** vs Stage-2 epoch; PhaseForge vs pprr vs pepb vs scratch (Lift + one harder task) | Line + thin per-seed lines; t=0 marked | training_curves.jsonl, init_routing.json | **H1/H2 mechanism** | P0 |
| **F4** | Partial warm-start sweep: SR and final NMI vs drop rate (0/25/50/75/100 + full-warm + one-warm); canonical 50% annotated | Point+CI line, dual axis | ablation drop-sweep cells | design justification | P1 |
| **F5** | Phase×expert routing heatmaps at end of training: phaseforge vs pf_centroid_random vs pf_random_random | 3 small heatmaps, shared scale | expert utilization records | specialization structure | P1 |

## 3. Main-content tables

| ID | Table | Columns | Source | Evidences | Pri |
|---|---|---|---|---|---|
| **T1** | **Five-task success matrix** | 9 methods × 5 tasks: mean SR ± Wilson CI; macro-average marked secondary; PhaseForge bolded; paired Δ vs BC with Holm markers (full tests in A15) | eval_results.json | **H6** | P0 |
| **T2** | **Causal mechanism controls** (Lift) | PhaseForge vs pprr (H1), pepb (H2), pf_spherical_kmeans (H3), pf_kmeans, pf_phase_head (H4): SR ± CI, paired Δ, t=0 NMI, final NMI | ablation cells + init_routing.json | **H1–H4** | P0 |
| **T3** | Capacity & fairness accounting | total/deployed params, epochs/steps, and capacity-control disclosures | fairness_accounting.py | capacity defense | P0 |

## 4. Appendix figures & tables

| ID | Asset | Encoding | Source | Pri |
|---|---|---|---|---|
| A1 | Per-seed raw SR, every matrix cell (9×5×3) | numeric + seed points | eval_results.json | P0 |
| A2 | Offline action MSE matrix (**eval-time MSE** primary, val loss secondary) | heatmap + numeric | eval_results + curves | P1 |
| A3 | Training curves, all methods × tasks: val action loss; stage-1 phase loss/acc; **stage-2 loss decomposition (balance, sticky, teacher-KL) and lr schedule** | small multiples | training_curves.jsonl | P1 |
| A4 | **Full ablation table:** factorial corners, pf_spherical, pf_ft, k3/k12, corruption 25/50/shuffle, one-warm, full-warm, drop sweep — SR ± CI, final NMI, **final collapse rate** | grouped table | ablation cells | P0 |
| A5 | Episode outcome breakdown: success / **failure categories (`failure_categories`, `termination_reason`)** / invalid / infra, per method × task | 100% stacked bars | rollout_summary + episodes.jsonl | P1 |
| A6 | Steps-to-outcome ECDFs per method (success and timeout episodes separated) | ECDF with ticks | episodes.jsonl | P2 |
| A7 | **t=0 routing alignment table** (all router-init variants): NMI, entropy, CoV, expert frequencies, dead experts, phase-head accuracy | table + mini-bars | init_routing.json | P1 |
| A8 | Expert utilization over Stage-2 per task (PhaseForge) + **top1/topk balance-score trajectories** | stacked area + line | training_curves.jsonl | P2 |
| A9 | Compute cost: wall-clock per phase (`timings.json`), **throughput (`train_steps_per_second`), peak GPU memory** | table | timings + curves | P2 |
| A10 | **Protocol & provenance:** seeds; reset-bank IDs + reset_seed; **checkpoint_sha256 ↔ eval linkage**; router_mode per eval; git commit + config hashes; artifact-manifest checksums; pinned simulator/robosuite/MuJoCo/torch versions; `_runner/plan.json` + `state.json`; gates reports; Holm family; dropped-neuron hashes; reproducibility statement incl. the evaluation-protocol note from the attribution record | table + text | run metadata, sweep artifacts | P0 |
| A11 | Router-init family dynamics (Lift): entropy / NMI / **switch-rate / collapse** trajectories, centroid vs kmeans vs spherical-kmeans vs phase-head vs random | line plots | ablation curves | P2 |
| A12 | **Hyperparameter & configuration table**: every resolved setting per method family (dims, top-k, noise, balance coeff, drop rate, lr, batch, epochs, seeds, early-stopping state) | table | resolved_config.yaml | P0 |
| A13 | **Dataset & phase-label statistics per task**: demonstrations, episode lengths, phase distribution from the labeler, state/action schema dims | table | data_provenance + raw HDF5 stats | P1 |
| A14 | **Phase-depth analysis**: distribution of `extra.max_phase` (deepest phase reached) per method × task — *where* policies fail along the task | stacked/ECDF per method | episodes.jsonl | P1 |
| A15 | **Paired statistical tests**: Wilcoxon signed-rank table (`paired_wilcoxon.csv`) + pairwise P(X>Y) from stratified stats, Holm-adjusted | table | _summaries | P0 |

Teacher-forced appears **only** in T1/A1 with its diagnostic label. Oracle
(H5) is optional and blocked on the open D10 fix; if ratified, its routing-gap
decomposition (Gap 1: oracle − teacher-forced; Gap 2: teacher-forced −
PhaseForge) becomes an appendix table — never a method row.

---

## 5. Encoding rules by data type

| Data type | Chosen encoding | Rejected alternatives & why |
|---|---|---|
| Success proportion, n=50×3 | Dot-and-interval / table with Wilson CI; paired-delta forest for comparisons | Bare bars (no uncertainty); SE-of-mean bars (wrong for proportions) |
| 3-seed replicates | Individual points + mean marker | Mean±std only (hides n=3) |
| Comparisons on identical resets | Paired per-episode deltas | Unpaired group bars (discard pairing) |
| Training dynamics | Line + thin per-seed lines, t=0 annotated | Smoothed single lines (hide spread) |
| Composition over time (expert use) | Stacked area; terminal phase×expert heatmap | Pie charts |
| Parameter sweeps | Point+CI vs sweep variable, canonical marked | Bar-per-condition |
| Method×task overview | Annotated heatmap beside full numeric table | Heatmap alone |
| Outcome/failure categories | 100% stacked bars | Grouped bars |
| Step/depth distributions | ECDF | Violin (n too small, hides tail) |

**Style:** Okabe–Ito colorblind-safe palette; one fixed method→color mapping
across all figures; PhaseForge first/accent; vector PDF; NeurIPS widths
(5.5 in text, 2.25 in margin); ≥8 pt type; no chart junk.

## 6. Claims-to-evidence map

| Claim | Primary | Support |
|---|---|---|
| H6 rollout success | **F2**, **T1** | A1, A15, A10 |
| H1 router-init effect | **T2**, **F3** | A7, A11 |
| H2 phase-representation effect | **T2**, **F3** | A3 |
| H3/H4 prior must be phase-structured | **T2** | A11 |
| Expert specialization mechanism | **F3**, F5 | A7, A8 |
| R50 design choice (50% partial warm) | F4 | A4 |
| Capacity is not the driver | **T3** | A2 |
| Failure characterization | A5, **A14** | A6 |
| Protocol validity & reproducibility | A10 | A12, A13, attribution record |
| Negative/limits: robot-only fails; teacher-forced diagnostic-only | T1 labels, A5 | — |

## 7. Coverage audit — generated-data → asset checklist

Every emitted field class has at least one asset (triple-checked):

- [x] eval_results (SR, CIs, counts, failures, bank) → T1/T2/F2/A1/A2
- [x] rollout_summary (router_mode, checkpoint_sha256, failure_categories, reset_seed) → A5/A10
- [x] episodes.jsonl (steps, termination, validity, **max_phase**) → F2/A5/A6/**A14**
- [x] training_curves: action/total losses → A2/A3; balance/sticky/teacher-KL/lr → **A3**; NMI/entropy/switch → F3/A11; balance/collapse scores → **A8/A11**; efficiency fields → **A9**
- [x] init_routing (all t0 metrics) → A7/T2; init_expert fingerprints → A4/A10
- [x] environment/data_provenance/artifact_manifest → A10/**A13**
- [x] resolved_config → **A12**; run_meta/timings → A9/A10
- [x] sweep level: plan.json/state.json/gates/_summaries/paired_wilcoxon → A10/**A15**, T3
- [x] stage-1 summary.json → A3/A7
- [x] fairness output → T3

Not publication data (excluded deliberately): per-epoch checkpoint snapshots
(provenance only, covered by A10 checksums), wandb local logs, `.completed`
markers, dev/bisect runs (`outputs_*` outside the final namespaces).

**Open dependencies:** D2 multiplicity confirmation, professor protocol
ratification, oracle/D10 decision, completed `outputs_final/` sweep.
Plotting: one generator per asset under `scripts/analysis/figures/`, reading
only `_summaries/` + run artifacts — no manual data entry anywhere.
