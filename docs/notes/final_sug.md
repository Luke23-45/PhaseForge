# Status Report: Trajectory-Regime Prototype Initialization for MoE Manipulation Policies (Can/Square)

Prepared 20 Sep 2026. Sources are this conversation, the 6-seed report from your experiment agent, and the manuscript excerpts you pasted. I have not seen the full manuscript or the raw evaluation files. Each figure below carries one of three labels:
- **[R]**: recomputed by me from the reported per-seed values.
- **[A]**: reported by the experiment agent and not independently verified.
- **[O]**: an open item.

## 1. Timing correction

Today is Sep 20, 2026. Earlier I described the abstract deadline as possibly still open, and that was wrong. Under the official ICLR 2027 dates (abstract Sep 18 AOE, i.e. Sep 19 11:59 UTC; full paper Sep 25 AOE), abstract registration has closed. One guidelines page lists earlier dates still, so check OpenReview directly. **[O] Confirm whether your abstract was registered.** If it wasn't, the stated rules give no accommodation, and ICLR main track is closed for this cycle.

## 2. Study scope (agreed)

The study is a bounded empirical analysis of routing initialization, not a state-of-the-art method claim. It examines a memoryless, not vision-based, hard prototype-routed, direct-action MoE policy. Trajectory-derived kinematic regimes come from change-point segmentation and clustering, and the regime-conditioned centroids of a Stage-1 latent initialize the routing prototypes before joint fine-tuning.

- **Central question:** does initialization change final routing organization, and does that organization predict closed-loop success?
- **Comparison with other methods:** an empirical baseline such as LAR-MoE is not required, since the study makes no claim to a better method. Prior work must still be cited and distinguished (section 8).
- **Stage-1 supervision:** Stage 1 uses action prediction plus rule-derived phase supervision (cross-entropy and SupCon). Regime discovery needs no manual annotation, but the representation is not phase-unsupervised.
- **Matched ablation:** the three primary arms (regime, random, rule-based) share architecture, expert initialization, and Stage-2 objective, with the margin loss off (λ_m=0). The full five-task configuration has an active rule-phase margin loss, so the two configurations aren't interchangeable.
- **Task selection:** Lift was near saturation, and ToolHang and Transport had near-zero success, so Can and Square were the informative tasks.
- **[O] Method description:** you described Stage 1 as a "dynamic topological method" for learning phases, while the manuscript excerpts say change-point detection plus clustering. Make sure the paper describes what you actually did.

## 3. Protocol

- **Seeds:** the original 42–44, extended with 45–47 for every arm (6 total, fixed before the runs), with the same rollout protocol and hyperparameters. The extension covered 10 methods × 2 tasks × 3 seeds **[A]**.
- **Rollouts:** 50 per seed per condition.
- **Metrics:** phase–expert NMI against rule-derived labels, and within-trajectory switch rate, both on held-out demonstrations. Success is measured over 50 rollouts per seed.
- **Statistics:** pooled Wilson intervals, seed-paired differences, and exact sign tests with Holm adjustment.

## 4. Results

**Routing organization.** On seeds 42–44, regime initialization gave final NMI of 0.674 on Can and 0.506 on Square, versus 0.071–0.091 for the matched controls. Its switch rates were 0.040 and 0.068, versus 0.096–0.109. On seeds 45–47, an example run (Can, seed 45) shows NMI 0.847 and switch rate 0.030, versus NMI 0.08–0.15 and switch 0.10–0.14 for the random and plain-encoder arms **[A]**. Preliminary t=0 diagnostics from seeds 45–47 suggest rule-based initialization starts near 0.45 NMI and decays, while regime starts near 0.78 and stays high **[A]**. The agent's blanket range of 0.65–0.85 conflicts with Square's 0.506 on seeds 42–44, so recompute per task, per arm, and per seed before quoting a range.

**Success, seeds 42–44 (150 rollouts per arm).** Holm-adjusted p=1.00 for all comparisons.

| Task | Regime | Random | Rule-based | Softmax |
|---|---|---|---|---|
| Can | 76.0% | 73.3% | 68.7% | — |
| Square | 26.7% | 28.0% | 31.3% | 35.3% |

**Success, six seeds** **[A]**, with paired differences and sign tests checked by me **[R]**:

| Task | Regime per seed (42–47) | Random per seed (42–47) | Regime mean | Random mean |
|---|---|---|---|---|
| Can | .66 .82 .80 .76 .64 .66 | .68 .70 .82 .56 .56 .68 | 0.723 | 0.667 |
| Square | .22 .28 .30 .34 .42 .16 | .14 .40 .30 .32 .36 .30 | 0.287 | 0.303 |

Six-seed means for the other reported arms were:

| Task | plain_encoder | static_rule |
|---|---|---|
| Can | 0.560 | 0.533 |
| Square | 0.413 | 0.097 |

## 5. Statistical read [R]

- **Regime vs random, Can:** mean difference +5.7 points, 95% t-interval [−4, +15], 3/6 wins, sign test p=1.00.
- **Regime vs random, Square:** −1.7 points, [−11, +8], 3 wins, 2 losses, 1 tie, p=1.00.
- **Regime vs static_rule:** Can +19 points (5 wins, 1 tie; p=0.22 counting the tie as a loss, or 0.0625 with ties dropped, so state your convention). Square +19 points, 6/6 wins, p=0.031, about 0.125 after Holm across four tests.
- **Regime vs plain_encoder, Square:** −12.7 points, and plain_encoder wins on 4 of 6 seeds. Regime leads on Can, 72.3% to 56.0%.
- Six seeds narrow the estimates but can't establish equivalence or exclude effects of about 10–15 points.

## 6. Supported and unsupported claims

**Supported:**
- Regime initialization produces markedly higher final NMI and lower switch rates than the matched controls.
- No closed-loop gain was detected against random initialization on either task.

**Not supported, so avoid:**
- "Dissociation," "task-dependent effect," and "does not imply/doesn't buy performance." Use "not detected."
- "Persists from initialization," until the t=0 and training-curve diagnostics are verified across all seeds.
- Any statement that NMI is an annotation-free measure of semantic specialization.

## 7. Open items before any submission

1. **[O] Matched rule-based arm.** Your 3-seed abstract reports the matched rule-based arm at 68.7% on Can and 31.3% on Square. The report's `static_rule` scores 52.7% and 12.0% on the same seeds and comes from the sweep directory, so it appears to be a different, non-matched arm. Confirm the matched arm was run on seeds 45–47. Until then, don't report the "6/6, p=0.031" result as a primary finding.
2. **[O] Data integrity.** Verify the tables against the `eval_results.json` files, and confirm that seeds 42–44 and 45–47 used the same code, configuration, and reset banks. Explain why "de-duplicated (latest eval per seed)" was needed, and confirm no rerun was chosen based on its result.
3. **[O] Diagnostics.** Produce NMI at t=0 and through Stage 2, expert utilization, and prototype norms and distances for all arms. State the coverage if older seeds lack logs. Explain the rule-init trend.
4. **[O] plain_encoder.** Define the arm and explain why it exceeds regime on Square. Fix the primary Holm family in advance, for example regime vs random and regime vs matched rule-based on both tasks.

## 8. Related-work positioning

Cite and distinguish the following; these were verified earlier in the conversation:
- **Mazza et al.**, arXiv:2601.21971 (supervised phase-structured MoE for surgical imitation; retitled, so cite the version you used, and disambiguate it from the unrelated bimanual MoE-ACT).
- **LAR-MoE**, Rodriguez et al., arXiv:2603.08476.
- **SMoDP**, Deng et al., RSS 2026 (2026, not 2025).
- **PADD**, Peng et al., ICML 2026 (cluster-derived expert initialization in a language-model distillation setting, which is outside robotics).

Not verified: MoE-DP's routing details, and the July 2026 emergent-compositional-skills paper (I saw only an abstract). Don't make a "first to" novelty claim without a documented search.

## 9. Assessment

The honest claim is narrow and defensible: large, consistent routing-organization differences, and no detectable closed-loop gain. As a bounded study it's suitable for a preprint or workshop. For ICLR main track (only if abstract registration succeeded), I'd still expect reviewers to press on two informative tasks, novelty, and metric circularity, and that is my judgment, not a prediction. Proceed to a full submission on Sep 25 only if items 1–3 in section 7 resolve cleanly by Sep 24, the text patches are applied, and every seed is reported. Otherwise post the preprint and target a later venue.

I can export this as a .docx or .md file if you want it to send on.