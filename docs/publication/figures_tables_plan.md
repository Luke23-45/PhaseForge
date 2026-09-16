# PhaseForge Publication Figures and Tables Generation Plan

**Document version:** 5.1
**Date:** 2026-09-16
**Status:** Frozen generation specification
**Output root:** `studies/analysis/outputs`

## 1. Scope

This document specifies only the publication figures and tables generated from existing PhaseForge artifacts.

It defines:

- which figures and tables are generated;
- which files and metrics supply them;
- formatting and statistical-display conventions;
- output names and locations;
- generation and verification requirements.

It does not define the research hypothesis, model architecture, training protocol, baseline design, new experiments, or manuscript conclusions. Those belong in the research and experiment documents. This document only prevents the generation pipeline from misrepresenting those externally defined results.

## 2. Authoritative input namespaces

The generator reads these frozen experiment manifests and output namespaces:

- Final manifest: `experiments/final_causal_matrix.json`.
- Final results: `final_experiments_results/final_experiments_results`.
- Ablation manifest: `experiments/router_initialization_ablation_can_square.json`.
- Ablation results: `final_experiments_results/abalation_final/teamspace/studios/this_studio/PhaseForge/outputs_router_ablation_can_square`.

The generator must derive task names, method names, seeds, stages, and expected cells from the manifests. It must not hard-code those values in individual asset modules.

### 2.1 Data audit relevant to asset generation

The current files contain:

- 150 final evaluation records: 135 rollout records and 15 offline-oracle records.
- 30 ablation rollout records covering five arms, two tasks, and three seeds.
- 50 valid episodes per rollout evaluation cell.
- Matching episode-index sets within every task/seed pairing group.
- 15 final full-trace evaluations and 30 ablation full-trace evaluations.
- Non-finite `eval/action_mse` in rollout records; finite action MSE is available only for the offline-oracle records.

These are audit observations, not hard-coded generator inputs. The generator must revalidate them when it runs.

The final results tree also contains repeated dependency copies of shared Stage-1 runs. Raw file counts must not be used as independent-run counts. Run selection must use manifest identity and provenance.

## 3. Global generation rules

### 3.1 Evidence integrity

- Every displayed number must come from a validated artifact or a deterministic calculation from validated artifacts.
- Missing, malformed, ambiguous, or stale inputs must fail generation or be displayed as `N/A`.
- No synthetic trace, fallback number, “golden value,” or hard-coded experimental measurement is permitted.
- `NaN`, positive infinity, and negative infinity are unavailable values and must never be formatted as measurements.
- Offline-only evaluation must be separated from rollout-success evaluation.
- Duplicate dependency copies must be resolved through an explicit canonical identity; an unrestricted newest-file rule is not acceptable.

### 3.2 Statistical display

- Report success counts, valid-episode denominators, and per-seed rates.
- A pooled Wilson score interval may be displayed only after validating the denominator definition and consistency across seeds.
- A pooled Wilson interval describes episode-level outcomes; it is not an uncertainty interval over three training seeds.
- Show individual seed values and, where useful, their descriptive mean, sample standard deviation, or observed range.
- Do not call a three-seed range a confidence interval or standard error.
- Pairwise comparisons require the same reset bank and matching episode/case indices.
- A Wilson interval for a binomial proportion must not be labeled as an interval for a paired success-rate difference.
- Any multiplicity-adjusted p-values must identify the tested family and must be labeled exploratory when based on three seeds.

### 3.3 Visual style

- Export vector PDF and 300-DPI PNG for every generated figure.
- Use the configured Okabe–Ito colorblind-safe palette.
- Use redundant marker or line-style encodings in addition to color where methods are compared.
- Use booktabs-style LaTeX tables without vertical rules.
- Keep captions descriptive and evidence-bound. Do not add claims of universal improvement, causality, or statistical significance unless the supplied data and test artifact support them.

### 3.4 Figure-selection principles

The figure set is chosen by the question each asset must answer, not by visual variety.

- Use a schematic only for the computation that cannot be recovered from a numerical table.
- Use absolute outcome tables for the measured levels and paired-effect plots for method-to-method differences. Do not make one plot perform both jobs.
- Prefer individual seed trajectories or seed points when there are only three seeds. A shaded envelope may be used only when its meaning is explicit and it does not conceal the seed paths.
- Use distribution summaries for high-volume timestep data. Do not plot thousands of points as a cloud, and do not replace a distribution with one illustrative trace.
- Use the simplest mark that preserves the comparison: aligned positions, common axes, zero references, direct labels, and redundant line or marker styles.
- Do not use 3-D plots, radar charts, pie charts, dual y-axes, decorative gradients, or smoothed curves. They either obscure the measured quantities or add an unsupported visual interpretation.

These choices follow the established principle that quantitative graphics should make the intended comparison perceptually direct, and that effect estimates must be separated from their uncertainty or variability display. A forest-style layout is appropriate for effect estimates, but the interval label must match the interval actually computed. With three training seeds, the seed range is descriptive variability, not inferential uncertainty. Methodological references are [Cleveland and McGill on graphical perception](https://faculty.washington.edu/aragon/classes/hcde511/s12/readings/cleveland84.pdf), the [Cochrane guidance on forest plots and effect estimates](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-iii), and [Nature Biomedical Engineering's guidance to show individual data points when practical](https://www.nature.com/articles/s41551-017-0079).

## 4. Main-text assets

The main-text generation contract contains exactly four figures and three tables: `F1–F4` and `T1–T3`.

### F1 — Method and routing formulation

**Output:** `figures/main/F1_overview.pdf`, `figures/main/F1_overview.png`

Show the reported PhaseForge computation accurately:

- phase-aware representation training, with the Stage-2 encoder trainability and learning-rate scale taken from the resolved PhaseForge configuration;
- topology-derived prototype construction;
- six-expert hard top-1 Voronoi assignment;
- direct expert action output, with the configured `beta=0.0` residual branch inactive in the reported run.

The figure may display `beta=0.0` as a configured setting, but must not depict a residual contribution as active. It must not depict top-2 routing or a frozen encoder unless the resolved run configuration supports those descriptions.

### F2 — Paired closed-loop success differences

**Output:** `figures/main/F2_paired_deltas.pdf`, `figures/main/F2_paired_deltas.png`

Use a paired-effect dot-and-range plot of PhaseForge minus each declared comparator across the five final tasks. Arrange the four comparator panels as a 2x2 grid with shared x-limits and a common zero reference. Each task row contains:

- the mean of the three seed-level paired differences;
- one marker for each seed-level paired difference;
- a horizontal observed min-to-max seed range;
- the paired episode count and comparator identity in the source table or caption.

Pairing requires identical reset-bank identity and episode/case indices. The plot is forest-plot-like in layout, but the horizontal range must be labeled `observed seed range`, not `95% CI`, because it is not a confidence interval. The absolute success rates remain in T1; F2 is retained because it answers the different question of direction and magnitude of the within-seed comparison. A heatmap is not selected for F2: it is more compact but makes signed differences, zero, and the three paired observations less precise to read.

Observed min/max seed ranges must not be labeled confidence intervals. Statistical annotations may be added only from a validated paired-testing artifact.

### F3 — Routing organization dynamics

**Output:** `figures/main/F3_specialization.pdf`, `figures/main/F3_specialization.png`

Use Can and Square Stage-2 trajectories for:

- phase–expert NMI;
- routing switch rate;
- routing entropy.

Plot all available seed trajectories as thin, semi-transparent lines and overlay their arithmetic seed mean as a heavier line. Do not use a ribbon as the primary seed-variation display: with three seeds it hides the actual trajectories. Show each validated t=0 value as a seed-level marker; do not place the value from one arbitrarily selected seed on the mean curve. Use a shared epoch axis within each metric and clearly label the normalization of entropy. This figure is a routing-diagnostics display and must not imply that any displayed routing metric is sufficient to predict task success.

### F4 — Action-transition analysis

**Output:** `figures/main/F4_action_discontinuity.pdf`, `figures/main/F4_action_discontinuity.png`

Use the controlled ablation traces for the following method identities:

- `router_init_topology`;
- `router_init_phase`;
- `router_init_random`;
- `routing_softmax_top1`.

Panel A uses a two-task small-multiple layout with box-and-whisker summaries of the validated per-timestep action-jump distributions at expert switches and non-switch steps. Show the median, interquartile range, and a stated whisker rule, with the number of observations for every condition. Overlay the per-seed means as separate markers so the large timestep sample is not mistaken for three independent training replicates. Use a common y-axis, and use a log y-axis only if the validated distribution requires it and the zero/near-zero handling is stated.

Panel B reports the expert-switch rate for the same task/method cells as seed-level points with a descriptive mean and observed range. Include `representation_bc` as a zero-switch reference in this panel, but do not create a switch-conditioned jump distribution for it. This panel is necessary because the consequence of a switch depends on both jump magnitude and how often switches occur.

This replaces the proposed single Square micro-trace in the main figure. One trace is an example, not an estimate of the distribution, and is therefore not sufficient as primary evidence. The current 22-asset contract does not generate a trace panel; if a trace is later added as a separately approved appendix asset, its selection rule, outcome, and full provenance must be recorded, and a fallback trace is prohibited.

The BC representation control has no expert switches in the current traces and must not receive a fabricated switch distribution or a zero-denominator ratio.

Only fields present and validated in `trace.jsonl` may be plotted. Unsupported end-effector-height, insertion-progress, or other tracks must be omitted. The action-jump calculation must be performed from consecutive `final_action` records within an episode and must never use a hard-coded reference level or fabricated observation.

### 4.1 Main-figure design review

| Figure | Mark type | Information carried | Decision and reason |
|---|---|---|---|
| F1 | Method schematic | The sequence from representation to prototype assignment to direct action | Retain. The computation is structural and is not readable from a result table. Keep it literal; it is not evidence of performance. |
| F2 | Paired-effect dots with observed seed ranges | Sign and magnitude of each within-seed comparison, the three seed observations, and the no-difference reference | Retain, but use a 2x2 layout and the term `observed seed range`. This preserves task heterogeneity, including saturated and floor tasks, without presenting a pooled average as the result. A heatmap would be denser but less precise for signed effects and seed-level variation. |
| F3 | Small-multiple trajectories with seed lines and mean | The direction, timing, and between-seed variation of routing organization during Stage 2 | Retain and replace the ribbon-only emphasis with visible seed paths. A bar chart would erase the training dynamics; a single mean line would erase the only available replicate-level evidence. |
| F4 | Distribution summaries plus seed-level switch-rate points | The conditional action-jump distribution and the frequency of expert transitions | Retain in the revised form. The distribution is the evidence for discontinuity; switch rate supplies exposure. A single micro-trace is appendix-level illustration, not a population-level estimate. |

F2 therefore does not need to be replaced with a different chart family. Its underlying question is an effect-size question, for which an aligned dot-and-range display is appropriate. The important correction is semantic and statistical: the seed range must not resemble or be labeled as an inferential confidence interval. T1 supplies the absolute rates, while F2 supplies paired contrasts; omitting either would make the result harder to interpret.

## 5. Main-text tables

### T1 — Primary rollout-success matrix

**Output:** `tables/T1_success_matrix.tex`, `tables/T1_success_matrix.md`

Report rollout-evaluable methods across Lift, Can, Square, ToolHang, and Transport. Exclude the offline oracle entirely from this table. If the privileged teacher-forced diagnostic is retained, place it in a clearly separated diagnostic block, exclude it from deployable-method ranking and macro-averages, and label it as non-deployable.

Each task cell contains a valid success estimate, denominator, and Wilson interval when the denominator condition is satisfied. The macro-average is a secondary descriptive aggregate over task/seed estimates.

### T2 — Controlled Can/Square router-initialization ablation

**Output:** `tables/T2_causal_controls.tex`, `tables/T2_causal_controls.md`

Report the five ablation arms:

- `router_init_topology`;
- `router_init_phase`;
- `router_init_random`;
- `representation_bc`;
- `routing_softmax_top1`.

Display Can and Square success, phase–expert NMI, routing switch rate, and top-1 collapse/coverage diagnostics. Seeds, episode counts, and denominators must be derived from the ablation artifacts.

### T3 — Capacity and compute accounting

**Output:** `tables/T3_capacity.tex`, `tables/T3_capacity.md`

Display deployed parameters, active parameters per sample, forward-pass FLOPs, and timing/efficiency fields only when measured or deterministically derived from resolved configurations. Distinguish total deployed capacity from active top-1 computation. Do not state equality with BC without a matched calculation.

## 6. Appendix assets

The appendix generation contract contains `A1–A15`.

| Asset | Rendering form | Information and rationale |
|---|---|---|
| A1 | Table | Per-seed rollout counts and rates; offline-only cells are `N/A`. The table is the audit trail for T1, not a second performance figure. |
| A2 | Table | Eval-time action MSE is reported only for finite offline evaluations; rollout eval-time MSE is `N/A`. Training validation action loss may be shown in separate columns as a secondary diagnostic, but it must not be presented as rollout performance or pooled with offline eval MSE. |
| A3 | Small-multiple line plots | Training action loss and expert-balance curves. Lines preserve optimization time; seed paths and a mean expose replicate variation. |
| A4 | Table | Full Can/Square ablation table with per-seed success and routing diagnostics. The number of arms and metrics is too large for a readable single plot. |
| A5 | 100% stacked horizontal bars | Episode outcome and validated failure-category breakdown. This is appropriate for mutually exclusive termination categories, but zero-count categories are suppressed or explicitly reported in the caption rather than shown as if observed. |
| A6 | Empirical CDF step plots | Conditional completion-step distributions among successful episodes. ECDFs show the full observed distribution without arbitrary bins; each curve reports its successful-episode count, and timeout episodes are not silently interpreted as completed episodes. |
| A7 | Table | Bootstrap routing-alignment results with the actual selected seed recorded. A table is preferable because the selected seed and resampling details are provenance, not a visual trend. |
| A8 | Small-multiple line plots | Stage-2 expert-balance trajectories for declared tasks and methods. The temporal path is the quantity of interest; use seed lines and a mean. |
| A9 | Table | Compute-cost accounting using measured timing and efficiency fields. Exact values and units matter more than visual encoding. |
| A10 | Table | Protocol, environment, commit, seed, checkpoint, and artifact provenance. This is reproducibility metadata and should remain searchable. |
| A11 | Small-multiple line plots | Can/Square router-family dynamics across initialization controls. This isolates the controlled ablation from the final-method comparison in F3. |
| A12 | Table | Hyperparameters from resolved configurations with representative-run identity. A table prevents visual equivalence between settings that differ materially. |
| A13 | Table with schema counts | Dataset and phase-label schema statistics from provenance artifacts. Counts, ranges, and missingness are more exact in tabular form. |
| A14 | Histogram or discrete bar chart | Deepest observed phase distribution with the phase range validated against metadata. A discrete count display matches the phase variable; the caption must state timeout/censoring if depth is trajectory-limited. |
| A15 | Table | Paired statistical tests and multiplicity correction, marked exploratory for three seeds. Exact estimands, tested families, and correction method must be readable rather than encoded by symbols alone. |

The appendix is deliberately not a collection of alternative decorative plots. Each asset either exposes replicate-level evidence, documents a distribution that cannot fit in the main text, or records a reproducibility quantity that should remain exact in a table.

`F5_initial_routing.py` and `F4_drop_sweep.py` are development assets, not part of this generation contract. Existing files from those modules must not be treated as verified publication outputs.

## 7. Generation and verification contract

The implementation is ready for generation only when:

1. The registry and verifier agree exactly on `F1–F4`, `T1–T3`, and `A1–A15` — 22 assets total.
2. The generator uses the two namespaces and manifest identities specified in Section 2.
3. Missing, malformed, non-finite, stale, or ambiguous source artifacts are rejected or shown as `N/A` according to the rules above.
4. Rollout and offline diagnostics are separated in all relevant assets.
5. The verifier checks exact asset identity, declared outputs, output presence, and SHA-256 consistency.
6. The generation manifest records source manifests, source hashes, selected assets, output hashes, and generation time.
7. Figures are inspected as PDF and PNG, and tables are inspected as both LaTeX and Markdown.

The current implementation must satisfy these gates before it is called compliant with this frozen plan:

- remove the stale `F5` expectation from the verifier so the registry, verifier, and plan all contain exactly 22 assets;
- make F1 match the resolved PhaseForge configuration: hard top-1 routing, six prototypes, trainable Stage-2 encoder when configured, and inactive `beta=0.0` residual contribution;
- implement F2 as the specified paired dot-and-observed-range display, without confidence-interval labeling;
- implement F3 with visible seed trajectories and seed-specific t=0 markers;
- replace all F4 fallback values and the illustrative trace with validated distribution and switch-rate calculations;
- exclude offline-oracle rows from T1, isolate any privileged teacher-forced row, and keep offline MSE separate from rollout success;
- remove hard-coded diagnostic values and derive A5 category presence and A14 phase range from validated artifacts.

## 8. Execution order

1. Freeze this generation plan and the experiment manifests.
2. Update only the analysis-generation implementation to satisfy this contract.
3. Run static checks and analysis tests.
4. Run a no-render coverage and provenance check.
5. Generate the selected figures and tables from the frozen result namespaces.
6. Run post-generation verification and inspect all rendered outputs.
7. Use verified assets as the sole source for manuscript figures, tables, and numerical captions.
