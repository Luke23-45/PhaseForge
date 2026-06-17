I checked the benchmark and MoE literature, and my final recommendation is:

**Use LIBERO-100 as the data source, with LIBERO-90 for bootstrapping/pretraining and LIBERO-LONG for downstream evaluation.** That split is already built into the benchmark paper: LIBERO-100 is divided into 90 short-horizon tasks for pretraining and 10 long-horizon tasks for downstream evaluation, and the benchmark provides 50 human-teleoperated demonstrations per task. The paper also says success rate is more reliable than training loss for manipulation, and it explicitly defines FWT, NBT, and AUC as its main lifelong-learning metrics. 

I would **not** make Open X-Embodiment / RT-X the primary dataset for this paper. It is excellent for scale, but it is also deliberately heterogeneous: 1M+ real-robot trajectories from 22 robots across 21 institutions. That is the opposite of what you want for a controlled architectural study, because it adds cross-embodiment and cross-dataset variation that can hide what the MoE is actually doing. ([arXiv][1])

I would also **not** make CALVIN the primary dataset here. CALVIN is a language-conditioned long-horizon benchmark with flexible sensor suites, including RGB, depth, tactile, and proprioception, so it is useful, but it is still built around language-conditioned continuous control and a richer sensor stack than you want right now. For this paper, that would add modality noise and make it harder to isolate the MoE architecture itself. ([arXiv][2])

So the cleanest setup for your project is:

**Primary data:** LIBERO-90 for Stage 1 / bootstrapping.
**Primary evaluation:** LIBERO-LONG for Stage 2 / downstream testing.
**Optional later add-on:** CALVIN only if you want an external long-horizon stress test after the core paper is already stable. 

This also matches how recent work treats LIBERO: it is widely used as a controlled benchmark for lifelong learning and pretraining effects, and recent MoE work such as LAR-MoE evaluates on LIBERO with a standard protocol, reporting mean performance over three seeds and using ablations on encoder freezing, latent-alignment regularization, and expert count. That is a good template for your own evaluation discipline. ([arXiv][3])

For the **state-only** part: that is a valid choice for this paper. Recent manipulation work reports proprioceptive state on LIBERO and CALVIN, so a state-only protocol is consistent with current practice; it is not a strange deviation. The key benefit is that it isolates the MoE effect from perception/backbone effects. ([arXiv][4])

### Data preparation plan

1. Download the official LIBERO benchmark and use the benchmark’s built-in split: **LIBERO-90** for bootstrapping and **LIBERO-LONG** for downstream evaluation. 
2. Use only the **state/proprioception stream** for this paper; do not train any visual encoder now. ([arXiv][4])
3. Normalize all state features using training-set statistics only.
4. Derive phase labels from robot state and task event signals: gripper closure/opening, object contact, object motion thresholds, and end-effector movement.
5. Manually inspect a small subset of trajectories per task to verify that the automatic phase rules are sensible.
6. Keep task splits and random seeds fully separated so no evaluation task leaks into bootstrapping or phase-rule tuning.
7. Run sample-efficiency curves on the bootstrapping set at **25% / 50% / 75% / 100%** of LIBERO-90. That is my recommendation for the paper, because the benchmark itself was designed to study pretraining effects and knowledge transfer. 

### Validation and evaluation plan

Use **task success rate** as the main downstream score, but do **not** stop there. LIBERO’s own paper argues that success rate is more reliable than training loss for manipulation policies, and it evaluates forward transfer, negative backward transfer, and AUC across tasks. That is exactly the style of evaluation you want for a system-building MoE paper. 

For the MoE-specific part, track these metrics during training:

* **routing stability**: when expert assignment stops fluctuating strongly,
* **expert utilization balance**: whether all experts are used,
* **collapse rate**: whether one expert monopolizes routing,
* **routing entropy**: how sharp the router’s decisions are,
* **phase-to-expert alignment**: whether the router respects the phase structure,
* **seed variance**: whether the pattern survives across runs.

These metrics are justified by the MoE literature: recent survey work says training stability and load balancing are among the central MoE problems, because imbalance can lead to model collapse, and robotics MoE papers explicitly add load-balancing and orthogonal losses to keep experts both used and distinct. ([arXiv][5])

### What to report in the paper

The final report should show, side by side:

* **Scratch MoE**
* **Warm-start MoE**
* **Phase-bootstrapped MoE**
* **Oracle-phase MoE**

and then report:

* LIBERO-Long average success,
* FWT / NBT / AUC across the LIBERO sequence,
* routing stability curves,
* expert-utilization curves,
* collapse rate,
* and the phase-ablation results. 

### The key decision rule

If phase bootstrapping beats warm-starting on routing stability and downstream success on LIBERO-Long across multiple seeds, keep the method. If it only helps routing but not success, the mechanism is real but the system still needs work. If warm-start matches phase bootstrapping, phase supervision is not the right lever and the design should change. That is the correct way to make a publication-level decision from the experiment. ([arXiv][3])

My final recommendation is therefore very specific: **make LIBERO-90 → LIBERO-LONG the official data backbone of the paper, keep the study state-only, and evaluate with both standard lifelong-learning metrics and MoE-specific routing metrics.** That gives you the cleanest controlled setup and the most defensible conclusions for the professor and for publication. 

[1]: https://arxiv.org/abs/2310.08864 "[2310.08864] Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
[2]: https://arxiv.org/abs/2112.03227 "[2112.03227] CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks"
[3]: https://arxiv.org/html/2603.08476v1 "LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning"
[4]: https://arxiv.org/html/2602.06575v1?utm_source=chatgpt.com "Embodied Visual Reasoning for VLA Manipulation"
[5]: https://arxiv.org/pdf/2503.07137 "A Comprehensive Survey of Mixture-of-Experts: Algorithms, Theory, and Applications"
