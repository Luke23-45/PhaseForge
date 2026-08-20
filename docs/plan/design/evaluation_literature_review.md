# Evaluation Techniques for State-Only (Non-Vision) Policy Learning — Literature Review

**Date:** 2026-08-15 (expanded after second, deeper research pass)
**Purpose:** literature review of evaluation techniques for state-based (non-vision) robot manipulation policies. Companion to `state_only_rollout_implementation_plan.md`.
**Scope:** policies consuming structured low-dimensional state (no RGB, no embeddings), single-step or short-horizon, evaluated in closed-loop simulation (robosuite/MuJoCo) and — where noted — real robots.

---

## 1. Method of this review

Two research passes. Pass 1 covered the benchmark protocols of the major state-based works. Pass 2 dug into: (a) the exact evaluation sections of the primary papers (robomimic study and Diffusion Policy), (b) the **offline policy selection (OPS)** literature (what to do when you cannot rollout for checkpoint selection), (c) the **statistics** literature for binary outcomes with few runs (Colas 2018, Henderson/Agarwal 2021, McNemar/Dietterich), and (d) related MoE-in-robotics work using phase/skill-specialized experts with a router. Claims were checked against the cited sources; URLs in §12. Several related works are recent or emerging arXiv preprints; they are context only, not authoritative state-only benchmark standards.

---

## 2. Evaluation practices in the cited studies (summary)

| Element | Common practices in the cited studies | Sources |
|---|---|---|
| Primary outcome | **Task success rate** from the simulator's success predicate (binary), over a fixed horizon | robomimic study; Diffusion Policy; ACT; Consistency Policy |
| Rollouts per evaluation checkpoint | **50 episodes** in the robomimic study; 50–200 in recent work (Consistency Policy: 200); ≈20 in robosuite benchmarks | robomimic; Diffusion Policy; Consistency Policy; robosuite |
| Evaluation cadence | **Every 50 epochs**, checkpoints saved every 50 epochs (robomimic; Diffusion Policy) | robomimic; Diffusion Policy |
| Training seeds | **3 seeds** in robomimic and Diffusion Policy; **5 seeds** in SOIL/CIMER; statistics literature: 3 seeds detect only large effects (see §5.4) | robomimic; Diffusion Policy; SOIL/CIMER; Colas 2018 |
| Reset cases | **Fixed serialized reset cases**, identical and ordered for every method/seed (paired design; basis for paired analysis §5.2) | implementation plan; RoboTwin2 "Clean" split |
| Checkpoint reporting | Some studies report **max over training** AND/OR **average of last 10 checkpoints**; rule fixed up front | robomimic (max over training); Diffusion Policy (max)/(avg last 10) |
| Significance testing | Mixed: mean ± s.d. over 3 seeds without tests (robomimic); exact McNemar for matched binary outcomes (Dietterich 1998); t-tests on ≥30 aggregated points in some benchmark comparisons | robomimic; Dietterich |
| Sanity gates | Task-independent gates (parity + state restore + action contract + native-predicate probe) before learned-policy judgment — a protocol choice, not a universal requirement | ACT; implementation plan |
| Secondary diagnostics | Offline metrics (action MSE, task progress, routing statistics) kept **secondary**, never primary | robomimic C4; CI-MSE |

These are protocol choices made by the cited studies — reasonable and supported by selected precedents, but not universal standards.

---

## 3. Benchmark protocols in detail

### 3.1 robomimic study — "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation" (CoRL 2021)

The reference protocol for state-based offline IL on Lift/Can/Square/Transport/Tool Hang. Exact protocol (from the paper's §3.3 and Appendix C):

- **Training/eval schedule:** each agent trained `N` epochs (each epoch = `M` gradient steps), evaluated every `E` epochs. Low-dim: **N=2000, M=100, E=50**; image: N=600, M=500, E=20. **50 rollouts per evaluation checkpoint**, success rate over a maximum horizon (horizon = average trajectory length of the dataset).
- **Reported number:** the **maximum success rate over the course of training**, averaged over **3 seeds**. 90% train / 10% validation split.
- **Real world:** 30 rollouts per final checkpoint, identical hyperparameters to sim (their real-world results transfer sim insights; Lift (Real) 96.7% vs 100% sim).
- **Lesson 3 / challenge C4 — the canonical citation for "offline metrics ≠ outcome":** epoch-to-epoch rollout success varies drastically **even with 50 rollouts per checkpoint**; the **best-validation-loss policy is 50–100% worse than the best-performing policy**; success rate can keep climbing while validation loss increases. "Each policy checkpoint needs to be tried directly on the robot."
- **Learning curves:** they publish success-rate-vs-epoch curves (3 seeds) precisely because single point estimates hide the variance.
- **Reference numbers for Lift (PH, low-dim):** BC ≈ 100% success (their best BC row); BC-RNN ≈ 100%. These are contextual reference values; direct comparison requires matching the full dataset, environment, action, reset, and checkpoint protocols.
- **Framework support:** robomimic natively supports saving checkpoints on **best rollout success rate / best rollout return** (`save_on_best_rollout_success_rate`, `save_on_best_rollout_return` in `robomimic/utils/train_utils.py`) — i.e., rollout-based checkpoint selection is a supported implementation option; val-loss checkpointing is what the study proved unreliable.

Sources: https://robomimic.github.io/study/ · https://arxiv.org/abs/2108.03298 · https://proceedings.mlr.press/v164/mandlekar22a/mandlekar22a.pdf · https://github.com/ARISE-Initiative/robomimic

### 3.2 Diffusion Policy (RSS 2023; IJRR 2024) — state-policy benchmark table

- **Protocol:** success rates averaged over **3 training seeds × 50 environment initializations (150 total)**. Reported in the format **max performance / average of last 10 checkpoints**, checkpoints saved every 50 epochs, 4500 epochs for state-based tasks. Evaluation every 50 epochs with success rate logged (`test/mean_score`).
- **Honesty precedent:** a bug restricted robomimic evals to 22 env initializations; they disclosed it and noted all baselines were compared under identical conditions. This is the standard for disclosure (we should do the same for any infra anomaly).
- **State-policy table (robomimic PH/MH, low-dim):** LSTM-GMM (BC-RNN): Lift 1.00/0.96, Can 1.00/0.88, Square 0.82/0.59, Transport 0.88/0.62, ToolHang 0.68/0.49 — contextual reference values for Lift ≈ 1.00.
- Also evaluates a **robustness axis** (Push-T: perturbation and visual distraction robustness) — secondary axes beyond nominal success are a recommended protocol option for this project.

Sources: https://arxiv.org/abs/2303.04137 · https://arxiv.org/html/2303.04137v5 · https://github.com/real-stanford/diffusion_policy · https://journals.sagepub.com/doi/10.1177/02783649241273668

### 3.3 Consistency Policy / RecFlow (state-based diffusion successors)

- Consistency Policy (RSS 2024): **average success over 200 rollouts** with standard errors (mean ± SE), on robomimic Lift/Can/Square/Tool Hang + Push-T — the current upper bound on episode-count rigor for state-based work.
- RecFlow (2025): 10 rollouts per task — an example of the weak lower bound; treat such numbers as low-evidence.

Source: https://www.roboticsproceedings.org/rss20/p071.pdf

### 3.4 ACT / ALOHA (RSS 2023) and ALOHA Unleashed

- Sim: best-validation-checkpoint selection, 50-rollout reporting norm; real: 20–100 trials per task.
- **Temporal ensembling** is inference-time averaging that changes rollout behavior — i.e., the exact inference rule (single-step vs ensembled) must be fixed and reported (also stressed by CI-MSE, §3.8).

Sources: https://arxiv.org/abs/2304.13705 · https://arxiv.org/html/2410.13126v1

### 3.5 robosuite framework

- Standardized env layer (success predicates, placement samplers, OSC controller). Benchmark conventions: 20 evaluation episodes per checkpoint, 500-step horizon, success rate + mean return ± std over 5 seeds in their RL benchmarks. The **placement initializer** (seeded object poses) is the source of reset-case stochasticity; **fixed init states** make evaluation paired.

Sources: https://arxiv.org/abs/2009.12293 · https://robosuite.ai/docs/algorithms/benchmarking.html

### 3.6 Dexterous, state-only IL — SOIL (ICRA 2021) and CIMER (ICRA 2025; arXiv 2024)

Closest precedent for "no vision, pure state" papers:

- **SOIL:** state-only demos; dexterous tasks (Relocate, Pen, Door, Hammer); **mean-return learning curves**; explicit **upper/lower-bound structure** (DAPG state-action upper bound, pure RL lower bound).
- **CIMER:** **success ± std over 5 random seeds**; robustness: **200 episodes** under perturbed physics parameters (mass/damping); **zero-shot generalization to 17 novel objects**.

Sources: https://arxiv.org/abs/2004.04650 · https://par.nsf.gov/servlets/purl/10621141

### 3.7 Move-Then-Operate: Behavioral Phasing for Human-Like Robotic Manipulation (2026, preprint; venue not verified)

**A closely related recent method**: dual-expert policy (move expert, operate expert) with a **learnable phase selector/router**.

- Evaluation: 8 tasks on RoboTwin2; primary metric **average success rate**; ablations over the phase-labeling pipeline; reports **+24.1% absolute over monolithic Pi0**; data-efficiency claim (peak performance in 40% fewer steps).
- Phase labels auto-generated (MLLM + velocity cues) — the field treats **phase label quality as an ablable component** evaluated through downstream success, not only label accuracy.

Implication: a phase-routed dual-expert design with success-rate-as-primary evaluation is an established pattern in recent work; our contribution differs by being state-only, offline-first, MLP-based.

Source: https://arxiv.org/html/2604.23620v1

### 3.8 CI-MSE — Critical Interval MSE: Toward Reliable Offline Validation (2026, preprint; venue not verified)

Directly quantifies the offline-metric problem:

- Across 27 VLA variants: **raw validation MSE correlates with rollout success only ρ=−0.61 (Spearman)**; CI-MSE (task-critical intervals + matching inference-time ensembling) reaches ρ=−0.87.
- For the data-scale variant family, **raw MSE is positively correlated with success** (wrong ordering) — raw MSE can actively misrank policies.

Conclusion: our offline action-MSE ranking of PhaseForge vs baselines is a *diagnostic*; its agreement with rollout success must be measured, not assumed.

Sources: https://arxiv.org/html/2606.29898v1 · https://ci-mse.github.io/

### 3.9 PhAIL (2026, preprint; venue not verified) — rigorous real-robot evaluation methodology (template)

- Critiques the norm: binary success at fixed timeout, N≤25 rollouts, **no CIs, no paired tests** — "an order of magnitude below the budget the binary tests they implicitly use require".
- Holds up the **LBM examination** as the template: N=50–200 per condition, exact paired tests for binary outcomes, Welch's t-tests for scalar scores, Bayesian posterior violins, **Bonferroni-corrected significance grouping**. (Note: for the same reset case evaluated by two policies, the standard matched-pairs test is exact McNemar — see §5.2.)
- Worked power: **Wilson 95% CI at N=50 on p̂=0.70 ≈ ±12 pp**; ±5 pp needs N≈380; at field-modal N∈[10,20] the CI on 70% is ~[0.40,0.89].

Source: https://arxiv.org/html/2605.29710v1

### 3.10 Published reference numbers for robomimic Lift (PH, low-dim)

Contextual reference values to interpret our own results (from the robomimic study, Diffusion Policy table, and the SAQ paper's robomimic table):

| Method | Lift (PH) success |
|---|---|
| BC (robomimic study / SAQ table "Robomimic BC") | ≈ 100% |
| BC-RNN / LSTM-GMM (DP table, max/avg-last-10) | 1.00 / 0.96 |
| IQL | ≈ 58% |
| CQL | ≈ 64% |
| SAQ-discretized variants | ≈ 90–100% |

These are **contextual reference values, not pass/fail anchors**: direct numerical comparison requires matching the full dataset, environment, action convention, reset, and checkpoint protocols. Differences in robosuite version, dataset revision, action convention, checkpoint selection, and reset conditions can change results.

Source: https://proceedings.mlr.press/v229/luo23a/luo23a.pdf

---

## 4. Checkpoint selection and Offline Policy Selection (OPS) — the layer we must get right

This is the single most consequential evaluation decision because the current pipeline selects on `best val/loss_action`. The literature documents the issue clearly:

### 4.1 The problem (robomimic Lesson 3)

- Best-validation-loss policy is **50–100% worse** than the best-performing policy; validation loss can increase while success keeps climbing; the train objective (action regression) ≠ eval objective (task success) — the mismatch is **structural, not a tuning issue**. (robomimic study C4; §3.1.)

### 4.2 Offline policy evaluation (OPE) methods as a substitute for rollouts

- **FQE (Fitted Q Evaluation)**: one possible OPE method whose reliability is task-dependent; in the cited studies it ranked candidate policies well (Paine et al. 2020; model-selection study on sepsis task; Fu et al. 2021 OPE benchmark uses **Spearman correlation** between OPE estimate and true performance as its metric). **TD errors and FQI values are poor validation metrics** (overestimation; Irpan 2019, Paine 2020).
- **Importance sampling (WIS/AM)**: more variable across tasks than FQE.
- **Hardness result:** OPS inherits OPE's worst-case hardness — no method can be sample-efficient in the worst case; FQE works well for **top-k selection** in practice (arXiv 2312.02355).
- Practical takeaway for a small sim study: OPE proxies add complexity and are mainly justified when rollouts are expensive; they are not for us (~2.5 min per 50 episodes). For the first study we keep a predeclared validation-based checkpoint rule and evaluate the selected checkpoint on a separate frozen rollout bank (§4.4); directly comparing the two selection procedures (val-loss vs rollout-based) is deferred to a later ablation.

### 4.3 Repeated data splitting (when you must select offline)

- A single train/validation split can systematically select worse policies; **repeated random sub-sampling (K splits, Monte-Carlo cross-validation)** with overlap partitioning is the robust alternative (NeurIPS 2022 "Data-Efficient Pipeline for Offline RL"). We could apply this to our val-loss selection cheaply.

### 4.4 The practical menu (evidence-backed)

1. **Recommended for the first PhaseForge rollout study — predeclared validation-based rule + separate frozen evaluation bank:** keep the fixed `best val/loss_action` rule (predeclared before the runs), evaluate the selected checkpoint on a **completely separate frozen rollout bank**, and report the rule clearly. This avoids test-set checkpoint selection while keeping the protocol simple.
2. **Rollout-based selection on a separate selection bank** — the robomimic/DP practice (`save_on_best_rollout_success_rate` is built into robomimic). Requires a *selection* reset bank disjoint from the *evaluation* reset bank. Whether it is better than the val-loss rule in our setting must be measured, not assumed (robomimic Lesson 3 shows a 50–100% gap for val-loss selection in their setting); add it as a **later ablation** comparing the two procedures.
3. **Report both** (DP-style: max / avg-last-10) so readers see the selection sensitivity — optional for the first study.
4. **OPE (FQE)** — only if we want an offline-only story; not recommended as primary for a sim study with cheap rollouts.
5. **Repeated data splitting** — cheap robustness add-on for any offline selection rule.

---

## 5. Statistical techniques

### 5.1 Wilson score interval (the standard CI for success rates)

- Wilson (1927); recommended over the Wald/normal interval (fails for small n, extreme p). Endorsed by NIST/SEMATECH handbook and the statistics literature (Brown, Cai, DasGupta 2001).
- **Worked 95% widths** (verified by computation):
  - p̂=0.70: N=10 → ±19 pp; N=25 → ±17 pp; N=50 → **±12 pp**; N=100 → ±9 pp; N=200 → ±6.5 pp; ±5 pp needs ≈380.
  - p̂=0.50: N=50 → ±14 pp.
- For this project, 50 fixed cases is a deliberate protocol choice supported by robomimic and Diffusion Policy precedents, not a universal statistical minimum. Use 100–200 only if a tighter predeclared interval is needed.

Source: https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm

### 5.2 Paired comparisons — the correct way to compare two policies on the same episodes

- Identical reset cases in identical order turn episodes into **paired data**.
- **McNemar's test** (McNemar 1947; Dietterich 1998): compares only the **discordant pairs** (A succeeds, B fails) vs (A fails, B succeeds); **use the exact binomial version when b+c < 25**; far more powerful than comparing two independent Wilson intervals because it controls for episode difficulty.
- **Newcombe (1998) method 10**: score CI for the difference of paired proportions — the right interval to report for the delta.
- **Cochran's Q**: an optional omnibus test for more than two paired models; it is not required for the primary three-seed analysis.
- **Dietterich (1998) warning:** the resampled paired t-test (pooling episodes across seeds) has a **high false-positive rate** — do not pool episodes as if independent; the 5×2cv paired t-test is the recommended algorithm-comparison test when training sets vary.
- **Do not use Barnard's exact test for this comparison:** Barnard's test is formulated for independent two-sample binary outcomes; for the same reset case evaluated by two policies, the standard matched-pairs test is exact McNemar (above; cf. the McNemar paired-binary reference below).
- **Per-seed paired differences:** compute the paired success-rate difference per task and (model, seed) on the identical reset cases; the final claim is made at the **seed level** (e.g., task-level differences followed by an unweighted macro-average), never by pooling episodes as independent training replicates.
- **Our existing two-sided exact Wilcoxon** on per-seed aggregates is the right test *at the seed level* — with n=3 seeds the minimum attainable p is 0.25, so it can only reject if all three seeds agree in direction; this must be stated (our report already does).

Sources: https://www.statstest.com/paired-evaluation-mcnemar-test-before-after-classification · https://stresearch-dev.github.io/eval-stats-toolkit · https://sebastianraschka.com/pdf/lecture-notes/stat479fs18/11_eval-algo_slides.pdf · https://pmc.ncbi.nlm.nih.gov/articles/PMC2902578/ · McNemar paired-binary reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/

### 5.3 Bootstrap CIs and the rliable toolkit (NeurIPS 2021 Outstanding Paper)

- Point estimates (mean/median) with few runs are unreliable; advocate **interval estimates of aggregate performance** and **performance profiles** (tail distributions with confidence bands), plus more robust aggregates: **Interquartile Mean (IQM)**, optimality gap, probability of improvement. **Stratified bootstrap CIs** work "with a handful of runs".
- For us: one task × 3 seeds × 50 episodes — rliable reduces to per-seed episode-bootstrap CIs and performance-profile-style per-seed bars; still worth reporting.
- **Caveat (Colas):** the *difference* bootstrap test is unreliable below ~20 samples (see §5.4).

Sources: https://arxiv.org/abs/2108.13264 · https://github.com/google-research/rliable · https://research.google/blog/rliable-towards-reliable-evaluation-reporting-in-reinforcement-learning/

### 5.4 How many seeds? (Colas et al. 2018 — with concrete numbers)

- Framing: the number of seeds is a **statistical power** question (type-II error β = probability of missing a real difference).
- **Worked example (Half-Cheetah, DDPG variants):** with **N=5 seeds, β≈0.51** — a 51% chance of missing the observed difference; to reach the conventional β=0.2 they needed **N=10 seeds**.
- **Bootstrap difference test: do not use below N≈20 samples** (empirical false-positive rate ≈10% vs the 5% claimed).
- **Empirical type-I estimation:** split 2N runs into two random halves and test — a permutation-style check we can run with our 3-seed data.
- **Under-estimation of s.d. at small n under-estimates required N** → choose N *larger* than the power analysis prescribes.
- Henderson (2017) context: all deep-RL papers of that era used ≤5 seeds; the same algorithm with 10 seeds, averaged over two 5-seed splits, produced curves from "different statistical distributions".
- Synthesis for us: **3 seeds detects only large effects**. The first matrix therefore uses descriptive reporting; adding seeds 45 and 46 is the predeclared stronger-evidence extension.

Source: https://ar5iv.labs.arxiv.org/html/1806.08295 (code: https://github.com/flowersteam/rl-difference-testing)

### 5.5 Episode-level vs seed-level inference — avoid the pooling trap

- **Correct layering:** (1) per task and (model, seed): Wilson CI on the 50 episodes; (2) per task and model: mean ± s.d. across seeds; (3) pairwise: **paired analysis on identical reset cases** — per-task, per-seed paired differences, exact McNemar per seed, and a Newcombe 95% CI for the paired difference; do not pool episodes across seeds as independent replicates (Dietterich 1998). Report the unweighted macro-average across task-level rates only as an aggregate summary.
- **Wrong:** treating all episodes as independent for a t-test across methods (Dietterich's resampled t-test warning; Colas's bootstrap warning) — inflates significance.

### 5.6 Multiple comparisons

- The full diagnostic matrix contains many possible pairwise tests; use **Bonferroni-Holm** (discrete variant preferred for exact McNemar; PMC 2902578) on the five pre-registered primary comparisons only. Privileged and negative-control rows are descriptive and are not part of that family. Colas also recommends multiplicity control for multiple experiments.

---

## 6. MoE / routing-specific evaluation — now with robotics precedents

### 6.1 The LLM lineage (metrics we already report, plus what's missing)

| Metric | Definition / standard | Source |
|---|---|---|
| **Load-balancing loss** | `L_aux = α·N·Σᵢ fᵢ·pᵢ` (α≈0.01) | Switch Transformer; GShard; GLaM |
| **Capacity factor** | tokens/expert cap; overflow tokens **dropped** — dropped-token rate is a measured quantity | GShard; Switch; ST-MoE |
| **Expert collapse / utilization** | fraction of experts receiving <ε of tokens; collapse = most inputs route to 1–2 experts (a *failure mode* to report) | Switch; MoE reviews |
| **Load imbalance factor / CV / router entropy** | earliest warning signals for collapse | practitioner MoE monitoring standards |
| **Router z-loss** | `(1/B)Σ(log Σ exp(z))²` — training-stability metric | ST-MoE |
| **Specialization quality** | expert–token-group alignment (per-expert histograms; our phase–expert NMI) | ST-MoE; ours |
| **Aux-free balancing** | bias terms instead of aux loss (DeepSeek-V3) | DeepSeek-V3 (via HF load-balancing review) |

### 6.2 Related robotics MoE work

- **MoE-DP (arXiv 2511.05007, 2025; preprint, venue not verified):** an MoE layer inside a diffusion policy for a **single long-horizon task**, with experts specializing in **semantic phases ("approaching", "grasping", "placing")** — related to PhaseForge in its use of phase/skill-specialized experts with a router. Their key empirical finding:
  - **"Using only load-balancing loss resulted in poor specialization despite uniform expert utilization, while entropy loss alone led to severe router collapse. The combination of both losses proved essential."**
  - This is relevant evidence for our design: fraction-to-top-expert, balance coefficient, and entropy are the quantities this literature tracks, and the combination (balance + entropy) helps diagnose collapse *and* homogenization.
- **DiTEA (AAAI 2026):** MoE for VLA with a modified balance loss.
- **"Advancing Expert Specialization for Better MoE" (OpenReview iydmH9boLb):** load balancing can cause **expert homogenization** — balancing and specialization are in tension; the field's answer is to monitor both (this is why we report NMI/collapse *and* balance).
- **MoE survey (Zhang et al. 2025):** modern assessment focuses on **expert utilization balance, calibration, and inference-time aggregation behavior** rather than parameter count.

Sources: https://www.alphaxiv.org/overview/2511.05007v1 · https://arxiv.org/html/2605.23477v1 · https://ojs.aaai.org/index.php/AAAI/article/view/38902/42864 · https://openreview.net/forum?id=iydmH9boLb · https://huggingface.co/blog/NormalUhr/moe-balance

### 6.3 What this means for the PhaseForge routing diagnostics

- What we already report (fraction to top expert, balance coefficient, entropy, collapse) **matches the LLM standard**.
- **NMI vs phase labels** is the alignment diagnostic used in this project; related robotics papers use other specialization analyses (e.g., MoE-DP's qualitative expert–phase association study).
- **Additions to consider** (each with a literature hook): router entropy over time (collapse early-warning), per-phase expert load (specialization), time-to-stable routing (training-stability), and — if we ever train with the balance term — reporting the L_aux coefficient (α) for reproducibility.

---

## 7. Phase-label / phase-head evaluation

- **Balanced accuracy** is appropriate for the class-imbalanced phase labels used by this project.
- **Temporal segmentation metrics** (TacUMI and the phase-segmentation literature): **frame-wise accuracy** and **boundary accuracy** are complementary to overall accuracy — relevant if we evaluate the stage-1 phase head in isolation.
- **Downstream validation is the primary arbiter:** related phase-routing work evaluates its phase-labeling pipeline through downstream task success. The same hierarchy applies here: phase metrics are secondary and simulator success is primary.

Sources: https://api.emergentmind.com/topics/tacumi · https://arxiv.org/html/2604.23620v1 · https://arxiv.org/html/2605.23477v1

---

## 8. Offline/diagnostic metrics: what they can and cannot claim

- **Action MSE / boundary smoothness / routing statistics are diagnostics, not outcomes.** Evidence they can misrank: CI-MSE's raw-MSE ρ=−0.61 (and positive correlation under data-scale changes); robomimic C4; the BC compounding-error literature.
- **Legitimate offline uses:** checkpoint selection (with the §4 caveats), hyperparameter/ablation screening, mechanism analysis (routing behavior, phase alignment), and — after rollout evidence exists — supporting diagnostics. This is exactly the hierarchy the implementation plan proposes.

---

## 9. Negative controls and sanity gates (recommended protocol for this project)

1. **Native success-predicate probe** — task-independent: confirms that `adapter.check_success()` and `env._check_success()` are callable and return bool/dict on the frozen bank before any learned policy is trusted. Validates env adapter + action contract + success predicate chain (plan §4.5).
2. **Random / no-op control** — lower bound sanity check.
3. **Observation-schema control** — our robot-only BC (23-dim): a **negative control reported separately, never in the main table** because it has different information content. This is consistent with the role of upper/lower-bound controls in state-only imitation studies.
4. **Privileged diagnostics** — teacher-forced / oracle routing rows labeled privileged, excluded from decisions (matches oracle-bound practice in SOIL/CIMER).
5. **Robustness axes (optional, after core result)** — physics-parameter sweeps with 200 episodes (CIMER precedent); perturbation robustness (DP Push-T precedent); RoboTwin2-style "Clean vs Randomized" split (robosuite-based benchmark).

---

## 10. Synthesis — recommended protocol for the PhaseForge five-task state-only rollout

| Protocol element | Recommendation | Justified by |
|---|---|---|
| Environment/dataset freeze | Pin robosuite version pair with dataset metadata; record hashes | plan §4.1; robomimic track warning |
| Adapter validation | Native-predicate probe + state-restore + parity gates must pass on the frozen reset bank before learned policies | §9; ACT; plan §4.5 |
| Reset cases | **50 fixed serialized initial states per task**, disjoint from training/val, identical order for all models and seeds within each task | implementation plan (reset-case protocol); paired-design statistics §5.2 |
| Episodes per (task, model, seed) | **50 selected evaluation episodes** (Wilson ±12–14 pp) — a defensible precedent, not a universal statistical minimum. Increasing to 100–200 is an optional predeclared precision extension. | §5.1; robomimic; Diffusion Policy |
| Seeds | Use 3 trained seeds for the first descriptive matrix. If stronger inferential evidence is required, add seeds 45 and 46 before the final paper table; do not silently mix pilot and final results. | §5.4 Colas |
| Checkpoint rule | **Predeclared validation-based rule** (`best val/loss_action`), evaluated on a **completely separate frozen rollout bank**; rule reported clearly. Rollout-based selection (separate selection bank; robomimic/DP practice) only as a later ablation comparing the two procedures | §4.4; §4.1 (robomimic C4) |
| Inference rule | Deterministic single-step, fixed action contract; no ensembling unless declared | ACT temporal ensembling; CI-MSE |
| Primary outcome | Success rate per task, model, and seed with **Wilson 95% CIs**; mean ± s.d. across seeds per task; unweighted macro-average across the five task-level rates as a secondary aggregate | §5.1; §5.3; robomimic |
| Pairwise comparison | **Paired on identical reset cases**: per-seed paired differences; exact McNemar (binomial when b+c<25) + Newcombe 95% CI for the paired delta; **no pooling across seeds**; final claim at seed level and remains descriptive (n=3 cannot support population-level significance); paired two-sided exact Wilcoxon across seed means as a secondary check (min p=0.25 at n=3, stated) | §5.2; §5.5; Dietterich 1998 |
| Multiple comparisons | Declare PhaseForge versus the five matched, non-privileged comparators as the primary family and apply Bonferroni-Holm (or another declared correction); privileged and negative-control rows remain descriptive | §5.6; PMC 2902578 |
| Invalid episodes | Infrastructure failures excluded from the success denominator; policy-caused NaNs/invalid actions/safety violations reported as policy failures under a strict metric, labeled separately | §3.9; plan §4.4 |
| Routing diagnostics | Report fraction-to-top-expert, balance coefficient, entropy (over time), collapse, NMI vs phase labels — as diagnostics; note balance-vs-specialization tension (MoE-DP: balance-only → homogenization, entropy-only → collapse) | §6 |
| Secondary metrics | Action MSE, smoothness, task progress — labeled diagnostic-only | §8 |
| Controls | Robot-only BC separate appendix; teacher-forced/oracle labeled privileged; native-predicate + random gates | §9 |
| Claim boundary | "State-only offline diagnostic + closed-loop success across five separate robomimic tasks"; no vision or multitask claims | plan §1 |
| Contextual references | Published Lift (PH) values (BC ≈ 100%, BC-RNN 1.00/0.96) as context only; direct comparison requires matching the full dataset/environment/action/reset/checkpoint protocols | §3.10 |

---

## 11. Locked protocol decisions

1. **Seeds:** use seeds 42, 43, and 44 for the first complete matrix. Treat the three-seed result as descriptive and report every seed. If the compute budget permits stronger inferential evidence, add seeds 45 and 46 before producing the final paper table; do not silently mix pilot and final results.
2. **Episode budget:** use 50 fixed evaluation cases per model and training seed. Increase to 100 or 200 only as a predeclared precision extension, not after inspecting which method wins.
3. **Checkpoint selection:** use the predeclared `best val/loss_action` checkpoint and evaluate it on a separate frozen rollout bank. A two-bank rollout-selection comparison is a later ablation, not part of the primary claim.
4. **Aggregation and tests:** report per-seed Wilson intervals and paired PhaseForge-minus-baseline differences on identical reset cases. Use exact McNemar/Newcombe within each seed, do not pool episodes across seeds, and keep the three-seed conclusion descriptive. Do not use a t-test on three seed means.
5. **Multiple comparisons:** the primary comparison family is PhaseForge versus the five matched, non-privileged comparators: BC-MLP, Scratch MoE, Warm-Start MoE, Phase-Pretrain Random-Router, and Plain-Encoder Phase-Bootstrap. Apply a declared multiplicity correction to those comparisons. Teacher/oracle routing and robot-only BC are diagnostic or negative-control rows, not primary baselines.
6. **MoE regularization:** do not add a new load-balance auxiliary loss to the primary experiment. Keep the current PhaseForge objective fixed and report balance, entropy, collapse, and specialization diagnostics. Any new regularizer is a separate ablation.
7. **Robustness:** defer physics perturbation and randomized-reset robustness until the nominal five-task evaluator is validated and the core result is complete.
8. **Phase metrics:** retain NMI, balanced accuracy, boundary statistics, and routing health as secondary mechanism diagnostics. They cannot replace rollout success.
9. **History and references:** include BC-RNN as the declared temporal control baseline for the five-task comparison. Published Lift numbers remain contextual references only; they are not pass/fail targets for any task.

---

## 12. References and scope notes

1. Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation* (CoRL 2021). https://robomimic.github.io/study/ · https://arxiv.org/abs/2108.03298 · https://proceedings.mlr.press/v164/mandlekar22a/mandlekar22a.pdf
2. robomimic framework — rollout-based checkpoint saving. https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/train_utils.py
3. Chi et al., *Diffusion Policy* (RSS 2023 / IJRR 2024). https://arxiv.org/abs/2303.04137 · https://arxiv.org/html/2303.04137v5 · https://journals.sagepub.com/doi/10.1177/02783649241273668
4. Zhao et al., *ACT* (RSS 2023). https://arxiv.org/abs/2304.13705 · ALOHA Unleashed: https://arxiv.org/html/2410.13126v1
5. LIBERO and VLA benchmarks are intentionally excluded from the PhaseForge protocol; they are not used as evidence for the core state-only evaluation.
6. Zhu et al., *robosuite*. https://arxiv.org/abs/2009.12293 · https://robosuite.ai/docs/algorithms/benchmarking.html
7. Radosavovic et al., *SOIL* (ICRA 2021). https://arxiv.org/abs/2004.04650
8. Han et al., *CIMER* (ICRA 2025; arXiv preprint 2024) — 5 seeds, 200-ep robustness. https://par.nsf.gov/servlets/purl/10621141 · official page: https://star-lab.cc.gatech.edu/papers/han-cimer-icra/
9. Xu et al., *Move-Then-Operate: Behavioral Phasing* (2026; preprint, venue not verified). https://arxiv.org/html/2604.23620v1
10. *Critical Interval MSE* (2026; preprint, venue not verified). https://arxiv.org/html/2606.29898v1
11. Arkhangelskiy, *PhAIL* (2026; preprint, venue not verified). https://arxiv.org/html/2605.29710v1
12. Luo et al., *Action-Quantized Offline RL (SAQ)* (CoRL 2023) — robomimic PH reference table. https://proceedings.mlr.press/v229/luo23a/luo23a.pdf
13. Colas et al., *How Many Random Seeds?* (2018). https://ar5iv.labs.arxiv.org/html/1806.08295 · code: https://github.com/flowersteam/rl-difference-testing
14. Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (NeurIPS 2021, Outstanding Paper). https://arxiv.org/abs/2108.13264 · rliable: https://github.com/google-research/rliable
15. Wilson (1927); NIST/SEMATECH handbook. https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
16. McNemar (1947); Dietterich (1998); Demšar (2006); Newcombe (1998). https://www.statstest.com/paired-evaluation-mcnemar-test-before-after-classification · https://stresearch-dev.github.io/eval-stats-toolkit · Raschka STAT 479 slides: https://sebastianraschka.com/pdf/lecture-notes/stat479fs18/11_eval-algo_slides.pdf · multiple McNemar / Bonferroni-Holm: https://pmc.ncbi.nlm.nih.gov/articles/PMC2902578/ · McNemar paired-binary reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC3716987/
17. OPS/OPE: FQE model selection (PMC 9190764); repeated data splitting (NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/file/5ee7ed60a7e8169012224dec5fe0d27f-Paper-Conference.pdf); OPS hardness (https://arxiv.org/html/2312.02355v2); offline-RL workflow (https://huggingface.co/papers/2109.10813)
18. Cheng et al., *MoE-DP* (2025; preprint, venue not verified). https://www.alphaxiv.org/overview/2511.05007v1
20. Li et al., *DiTEA* (AAAI 2026). https://ojs.aaai.org/index.php/AAAI/article/view/38902/42864
21. Guo et al., *Advancing Expert Specialization for Better MoE*. https://openreview.net/forum?id=iydmH9boLb
22. MoE lineage: Switch Transformers https://arxiv.org/abs/2101.03961 · GShard https://arxiv.org/abs/2006.16668 · load-balancing review incl. DeepSeek-V3: https://huggingface.co/blog/NormalUhr/moe-balance · https://huggingface.co/blog/moe
23. TacUMI — phase segmentation evaluation. https://api.emergentmind.com/topics/tacumi
24. Consistency Policy (RSS 2024), 200-rollout eval. https://www.roboticsproceedings.org/rss20/p071.pdf
