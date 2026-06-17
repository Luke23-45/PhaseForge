# Pure Hypothesis File

## Phase-Bootstrapped Mixture-of-Experts for Long-Horizon Manipulation

### State-Only Controlled Study, Literature Review, Data Plan, and Implementation Blueprint

### 1. Document Purpose

This document is the **core hypothesis file** for the project. It is written to support both the later publication and the implementation work.

Its purpose is to state, as precisely as possible:

* what the project is trying to prove,
* why the experiment is designed the way it is,
* which dataset is used and why,
* what is intentionally excluded,
* how the model will be trained,
* and how success or failure will be judged.

The document is intentionally narrow. It does **not** try to solve every robotics problem at once. It is a controlled study of whether **phase-supervised bootstrapping** improves MoE routing and specialization in long-horizon manipulation when all unnecessary variables are removed.

---

### 2. Project Idea in One Sentence

We test whether **supervised phase structure**, injected before MoE specialization, produces a more stable and more useful routing geometry than random initialization or generic warm-starting in long-horizon robot manipulation.

---

### 3. What the Paper Is and Is Not

This paper is:

* a controlled system-building study,
* a mechanism-first MoE paper,
* a state-only manipulation study,
* and a training-strategy paper rather than a new architecture paper.

This paper is not:

* a vision paper,
* a VLM paper,
* a diffusion-policy paper,
* a large-scale foundation-model scaling paper,
* or a benchmark-chasing paper whose only goal is to report the highest success rate.

Vision, if it is added later, is a **future extension**, not part of the current proof. That separation is deliberate because the perception stack would add confounds and make it unclear whether any gain comes from the MoE strategy itself or from the vision backbone.

---

### 4. Final Hypothesis

The central hypothesis is:

**Phase-supervised pretraining shapes the encoder latent space so that the router’s decision boundaries align with behavior phases before expert fine-tuning begins. This should reduce cold-start instability, lower expert collapse, improve routing consistency, and make the MoE policy easier to train than scratch MoE or generic warm-start MoE trained without phase supervision.**

This hypothesis is about **architecture behavior and training dynamics**, not only final task success.

The claim is falsifiable. If phase-supervised bootstrapping does not improve routing stability relative to warm-starting, then the hypothesis is wrong or incomplete.

---

### 5. Why the Study Must Stay State-Only

The first publication should remain **state-only / proprioception-only**.

The reason is scientific control.

If vision is added now, then improvements may come from:

* the image encoder,
* the visual backbone pretraining,
* the visual resolution and camera placement,
* the dataset size required by vision,
* the inductive bias of the perception model,
* or the routing strategy itself.

That makes causal attribution weak. The current paper is supposed to identify the effect of the MoE design choice, not the effect of the perception stack.

A state-only setup is therefore the cleanest way to verify the architecture-level hypothesis first. Vision can be added later once the mechanism is verified.

---

### 6. Literature Review

#### 6.1 Sparse MoE: the core engineering problem

Sparse MoE models are attractive because they can scale capacity while keeping computation per input limited. But the literature consistently shows that **routing instability, load imbalance, and expert collapse** are central difficulties.

Switch Transformers explicitly addresses MoE training instability and uses balancing techniques to make sparse routing trainable. ST-MoE similarly focuses on stability and transferability, showing that sparse expert models need careful stabilization to perform well across tasks. More recent MoE work continues to treat load balancing, expert utilization, and routing diversity as major design issues rather than solved problems.

The lesson for this project is simple: **the router is not a minor detail**. It is often the bottleneck.

#### 6.2 MoE specialization vs. uniform balance

A major theme in the MoE literature is the tension between two objectives:

* using all experts roughly evenly,
* and letting experts specialize meaningfully.

Strong balancing can prevent collapse, but too much balancing can force overly uniform routing and reduce specialization. Recent work also suggests that simple load balancing does not always preserve meaningful expert diversity. That is directly relevant here, because the project is about phase-structured specialization rather than uniform routing.

#### 6.3 Robotics MoE: what recent work shows

Recent robotics papers show that MoE is becoming a serious design option for manipulation, but they also show that routing strategy matters.

* **LAR-MoE** uses latent-aligned routing in robotic imitation learning and reports strong performance on LIBERO without phase annotations. This shows that structured routing is valuable, and it provides a direct comparison point for our work.
* **AdaMoE** scales manipulation with MoE in a VLA-style setting and uses decoupled expert selection/weighting. This shows that expert routing is now being treated as a first-class design problem in manipulation systems.
* **MoE-ACT** integrates sparse expert routing into an ACT-based manipulation system, again showing that the field is actively using MoE to improve multi-task manipulation.

These papers matter because they confirm the same overall message: sparse expert routing is promising, but the right way to initialize, stabilize, and specialize experts is still an open design question.

#### 6.4 Why not start from the strongest vision-heavy systems

Vision-language-action systems, diffusion-based visuomotor policies, and large-scale robot foundation models are important results, but they are not the right starting point for this project.

Diffusion Policy is a strong visuomotor baseline and has demonstrated broad performance gains. Open X-Embodiment / RT-X aggregates very large heterogeneous robot datasets and is designed for large-scale cross-embodiment learning. Those systems are impressive, but they are solving a broader perception-plus-control problem.

Our current goal is narrower: **test whether phase bootstrapping is a good MoE training strategy before adding perception complexity.**

That makes the current paper cleaner and more defensible.

---

### 7. Final Dataset Choice

The most appropriate dataset backbone for this project is:

**LIBERO-100, using LIBERO-90 for training/bootstrapping and LIBERO-LONG for downstream evaluation.**

This is the best choice because the LIBERO benchmark was specifically designed for lifelong robot learning and knowledge transfer. The benchmark paper splits LIBERO-100 into **90 short-horizon tasks (LIBERO-90)** and **10 long-horizon tasks (LIBERO-LONG)**, using the former as the pretraining source and the latter as downstream evaluation.

This split is exactly aligned with our project structure:

* **LIBERO-90** is the training and bootstrapping source,
* **LIBERO-LONG** is the evaluation target.

Why this is better than Open X-Embodiment or CALVIN for the current paper:

* Open X-Embodiment is huge and heterogeneous, with over one million real robot trajectories across 22 embodiments, which is excellent for scaling but too confounded for a mechanism paper.
* CALVIN is a long-horizon benchmark too, but its language-conditioning and multimodal sensor setup make it better suited to a perception-rich study than a pure state-only routing study.

So the cleanest current choice is **LIBERO**.

---

### 8. Data Preparation Plan

#### 8.1 Observation format

Use **state / proprioceptive inputs only** for the current study.

Do not train a visual encoder in this stage.

#### 8.2 Task split

* Pretraining / bootstrapping: **LIBERO-90**
* Downstream evaluation: **LIBERO-LONG**

#### 8.3 Data normalization

* Normalize state variables using statistics computed only on the training split.
* Keep normalization fixed across all methods.
* Do not allow evaluation data to influence normalization.

#### 8.4 Phase label construction

Phase labels should be derived automatically from state signals such as:

* gripper aperture,
* end-effector motion,
* contact transitions,
* object motion thresholds,
* task event boundaries.

The point is not to manually annotate every frame. The point is to construct a repeatable phase signal from state information.

#### 8.5 Split hygiene

* No leakage between train and test tasks.
* No tuning on evaluation tasks.
* Same task order for all methods.
* Same seeds where possible.
* Same compute budget for all methods.

---

### 9. Model Design

#### Stage 1: Phase-supervised generalist

Train a shared backbone with:

* imitation/action loss,
* auxiliary phase classification loss.

Objective:

* **L_total = L_action + λ · L_phase**

The phase loss is used only during training. It is not a test-time input.

The goal of Stage 1 is to force the latent representation to reflect phase structure.

#### Stage 2: Bootstrapped MoE specialization

Use the Stage 1 representation to initialize the router and then specialize experts on phase-partitioned data.

This stage tests whether the phase-aware initialization gives:

* cleaner routing,
* fewer dead experts,
* lower collapse,
* and better long-horizon behavior.

#### Inference

At inference, the model receives only state input. No phase labels are provided.

---

### 10. Baselines

The baselines must isolate the effect of phase bootstrapping.

1. **Behavior Cloning (BC)**

   * no experts
   * no routing
   * no phase supervision

2. **Scratch MoE**

   * same architecture and parameter count
   * random router initialization
   * no phase supervision

3. **Warm-Start MoE**

   * pretrained generalist without phase supervision
   * same architecture and parameter count
   * tests whether any pretraining helps

4. **Phase-Bootstrapped MoE**

   * proposed method
   * phase-supervised pretraining and router initialization

5. **Oracle-Phase MoE**

   * uses ground-truth phase labels during training and routing reference
   * upper bound on phase-aware specialization

Optional if time and code allow:

* **Latent-alignment baseline** such as LAR-style routing.

The most important comparison is **Phase-Bootstrapped MoE vs. Warm-Start MoE**.

---

### 11. What to Measure

The evaluation must include both **MoE-specific metrics** and **task metrics**.

#### 11.1 MoE-specific metrics

These are the metrics that directly test the hypothesis:

* **Time to stable routing**: how many steps until the routing distribution stops fluctuating strongly.
* **Routing entropy**: how confident the router is.
* **Routing entropy variance**: whether the router is stable or noisy during training.
* **Expert utilization balance**: whether experts are used evenly enough.
* **Collapse rate**: whether one or a few experts dominate.
* **Phase-to-expert alignment**: whether phase boundaries map to expert switches.
* **Seed stability**: whether the result is consistent over multiple random initializations.

#### 11.2 Task metrics

These show whether routing quality matters for actual performance:

* **Success rate** on each task and average across tasks.
* **Completion rate** for long-horizon sequences.
* **Sample efficiency** at 25%, 50%, 75%, and 100% of the data.
* **Boundary smoothness** near phase transitions.
* **Phase-wise prediction error** if the model predicts state or action chunks.

#### 11.3 Why these are the right metrics

MoE literature repeatedly shows that training can look good while routing is poor. Therefore, training loss alone is not enough. The paper should measure the router directly.

---

### 12. Evaluation Protocol

A strong evaluation protocol is part of the contribution.

#### Fixed protocol across all methods

* same dataset split,
* same training steps,
* same architecture family,
* same number of experts,
* same optimization budget,
* same seeds,
* same evaluation episodes.

#### Reported outputs

* mean and standard error over seeds,
* learning curves,
* routing curves,
* collapse curves,
* sample-efficiency curves,
* final success rates.

#### Decision rule

The hypothesis is supported if:

1. phase bootstrapping improves routing stability over warm-start,
2. routing stability correlates with better long-horizon performance,
3. the result is consistent across seeds,
4. and the gain survives ablations.

---

### 13. Ablation Plan

The ablations must answer the “why” question.

1. **Phase supervision vs no phase supervision**

   * core test of the hypothesis

2. **Phase supervision vs warm-start only**

   * separates phase structure from ordinary pretraining

3. **Frozen encoder vs jointly fine-tuned encoder**

   * tests whether the bootstrapped representation should remain fixed

4. **Hard routing vs soft routing**

   * tests whether sharp specialization or smooth mixing is better

5. **Number of experts / phases**

   * tests whether the decomposition is too coarse or too fine

6. **Phase-label noise**

   * tests robustness of the derived phase signal

7. **Data fraction**

   * tests whether the method helps most in low-data settings

---

### 14. Implementation Blueprint

#### Step 1: Prepare the dataset

* load LIBERO-90 and LIBERO-LONG,
* build state-only observations,
* compute normalization,
* derive phase labels,
* verify a small set of trajectories manually for sanity.

#### Step 2: Build the Stage 1 model

* shared encoder,
* action head,
* phase head,
* train on LIBERO-90.

#### Step 3: Build the MoE stage

* initialize router from the Stage 1 representation,
* clone or initialize experts,
* specialize on phase-partitioned demonstrations.

#### Step 4: Run baselines

* BC,
* Scratch MoE,
* Warm-Start MoE,
* Phase-Bootstrapped MoE,
* Oracle-Phase MoE.

#### Step 5: Log all metrics

* routing stability,
* expert usage,
* collapse,
* success,
* sample efficiency,
* phase alignment.

#### Step 6: Decide

Use the results to decide whether the phase-bootstrapped design should be kept as the system backbone or revised.

---

### 15. Risks and What They Mean

#### Risk 1: Phase bootstrapping does not beat warm-starting

Then phase labels are not adding enough beyond generic pretraining, and the hypothesis should be revised.

#### Risk 2: Routing improves but success does not

Then the mechanism exists, but the downstream policy stack still needs work.

#### Risk 3: The number of phases is wrong

Then the phase segmentation is too coarse or too fine, and the phase construction rules need adjustment.

#### Risk 4: The model collapses to one expert

Then the router is not learning meaningful specialization, and the balancing strategy must be changed.

Each risk is useful because it tells us what to fix.

---

### 16. Final Positioning Statement

This project is a **controlled state-only MoE study** built to test whether supervised phase bootstrapping improves expert routing and specialization in long-horizon manipulation.

The paper is intentionally minimal:

* no vision,
* no perception backbone,
* no foundation-model detour,
* no unnecessary benchmarking drift.

The chosen dataset backbone is **LIBERO-90 for bootstrapping and LIBERO-LONG for evaluation**, because that split already matches the intended training/evaluation logic and is widely used for lifelong robot learning research.

If the method works, the result becomes a strong foundation for later scaling with vision and larger backbones. If it does not work, the result still matters because it tells us the architecture choice is not yet right.

That is the point of the paper: to produce a clear, defensible answer that can guide the next stage of system building.

---

### 17. Key References to Cite in the Paper Draft

* Switch Transformers (Fedus et al., 2022)
* ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022)
* Diffusion Policy (Chi et al., 2023)
* LIBERO benchmark (Liu et al., 2023)
* Open X-Embodiment / RT-X (O'Neill et al., 2023)
* LAR-MoE (Rodriguez et al., 2026)
* AdaMoE (Shen et al., 2025)
* MoE-ACT (Guo et al., 2026)

These are the core papers that should shape the later implementation and the final public
