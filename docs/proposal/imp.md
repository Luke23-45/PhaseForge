# Proposal Report

## Phase-Bootstrapped Mixture-of-Experts for Long-Horizon Manipulation

### State-Only Controlled Study for System Development

### 1. Project Summary

This project studies whether **phase-supervised bootstrapping** can improve a sparse Mixture-of-Experts (MoE) policy for long-horizon robot manipulation. The purpose is to determine whether the proposed training strategy produces better **routing stability**, **expert specialization**, and **data efficiency** under controlled conditions.

This is **not** a vision-centric study at this stage. Vision, VLMs, diffusion backbones, and other perception modules are intentionally excluded from the current paper because they would introduce additional variables that obscure the effect of the MoE training strategy itself. The current paper is therefore a **controlled mechanism-and-system study**.

The goal is to build a stronger manipulation policy through evidence-driven design choices, not to claim that the final system is complete today. The project establishes the architectural foundation first; richer perception can be added later once the MoE training behavior is understood.

---

### 2. Research Objective

The central question is:

**When all non-essential variables are held fixed, does phase-supervised bootstrapping produce a better MoE routing structure than random initialization or generic warm-starting for long-horizon manipulation?**

The answer to this question determines whether phase bootstrapping is a useful core design principle for the final system. The paper is therefore about **how the architecture functions**, not about proving that one full end-to-end robot stack is globally best.

---

### 3. Why the Study Should Be State-Only

The study should remain **state-only / proprioception-only** for the current publication.

The reason is scientific control. If vision is included now, the result may depend on the choice of visual encoder, image resolution, dataset size, augmentation strategy, backbone pretraining, and benchmark-specific appearance. That would make it hard to tell whether any gain comes from the MoE idea itself or from the perception stack.

The state-only setting gives a cleaner causal test. It lets us isolate:

* the effect of phase supervision,
* the effect of warm-starting,
* the effect of router initialization,
* and the effect of expert specialization.

This is exactly the right choice if the goal is to test the architecture first and scale later.

---

### 4. Overall Research Strategy

The project should follow a four-step experimental ladder:

1. **Controlled state-only benchmark** to isolate the MoE mechanism.
2. **Matched baselines and ablations** to identify what actually causes improvement.
3. **Scaling within the same state-only setting** to test whether the effect survives more data, more tasks, or more capacity.
4. **Vision extension later** only after the architectural mechanism is verified.

This order is important. Adding vision too early would make the conclusions noisy and ambiguous.

---

### 5. Core Method

The method has two stages.

#### Stage 1: Phase-Supervised Generalist Pretraining

A shared encoder-policy backbone is trained with two objectives:

* imitation/action loss,
* auxiliary phase-classification loss.

The phase labels are derived automatically from robot state signals such as gripper aperture, end-effector motion, contact transitions, and object movement. The purpose of Stage 1 is to organize the latent representation so that phase structure is explicitly encoded before MoE specialization begins.

#### Stage 2: MoE Bootstrapping and Specialization

The Stage 1 model initializes a sparse MoE policy. The router is bootstrapped from the phase-aware latent structure, and the experts are specialized on phase-partitioned data. The key test is whether this supervised initialization yields a more stable and useful routing geometry than a random router or a generic warm start.

At inference, only state/proprioception is used. No ground-truth phase labels are provided.

---

### 6. Benchmark Choice

The primary benchmark should be a **long-horizon manipulation environment with derivable phase structure**, ideally a controlled or instrumented environment where phase boundaries can be computed directly from state rather than annotated manually.

The benchmark should include sequential behaviors such as:

* reach,
* grasp,
* lift,
* transport,
* place,
* retract.

This is the correct setting because the model is explicitly designed to exploit phase structure.

A public benchmark can be used later as a follow-up, but the first paper should prioritize causal clarity over breadth.

---

### 7. Model Family and Control Conditions

All conditions must use the same backbone, parameter budget, training schedule, data splits, and evaluation protocol. Only the training strategy should change.

#### Main conditions

1. **Behavior Cloning (BC)**

   * single policy
   * no experts
   * no phase supervision

2. **Scratch MoE**

   * same architecture and parameter count as the proposed model
   * random router initialization
   * no phase supervision

3. **Warm-Start MoE**

   * router initialized from a generalist model trained only with action imitation
   * no phase labels during pretraining
   * tests whether generic pretraining alone is sufficient

4. **Phase-Bootstrapped MoE**

   * the proposed method
   * phase-supervised pretraining and bootstrapped routing

5. **Oracle-Phase MoE**

   * phase labels used during training and as routing reference
   * upper bound for phase-aware specialization

#### Optional comparison

6. **Latent-alignment baseline**

   * include only if a fair implementation is practical
   * useful for testing whether unsupervised latent routing can match the supervised phase signal

The most important comparison is **Phase-Bootstrapped MoE vs. Warm-Start MoE**.

---

### 8. Evaluation Metrics

The evaluation must separate mechanism quality from task quality.

#### 8.1 Mechanism metrics

* **Time to stable routing**: gradient steps until assignment patterns stop fluctuating strongly.
* **Routing entropy**: measures confidence and decisiveness of expert selection.
* **Expert utilization balance**: whether all experts receive meaningful traffic.
* **Collapse rate**: fraction of experts that receive negligible traffic.
* **Phase-to-expert alignment**: whether routing boundaries correspond to the intended phases.
* **Seed stability**: whether the same behavior appears across random seeds.

#### 8.2 Task metrics

* **Success rate** on the manipulation task.
* **Completion rate** for long-horizon trajectories.
* **Phase-wise error** in action or state prediction.
* **Boundary smoothness** near phase transitions.
* **Sample efficiency** at 25%, 50%, 75%, and 100% of the dataset.

Mechanism metrics are primary because the paper is about how the system functions. Task metrics are secondary because they show whether the mechanism matters for behavior.

---

### 9. Required Ablations

The proposal must include ablations that can falsify the hypothesis if needed.

1. **Phase supervision vs. no phase supervision**

   * tests whether phase labels matter at all

2. **Phase supervision vs. warm-start only**

   * separates phase structure from generic pretraining

3. **Frozen encoder vs. jointly fine-tuned encoder**

   * tests whether the bootstrapped representation should remain fixed

4. **Hard routing vs. soft routing**

   * tests whether sharp specialization or smoother mixing is better

5. **Number of experts / phases**

   * tests whether the decomposition is too coarse or too fine

6. **Phase label noise**

   * tests robustness to imperfect phase derivation from state signals

These ablations are necessary because a positive result is only meaningful if we know why it happened.

---

### 10. Expected Outcomes and Interpretation

There are three meaningful outcomes.

#### Outcome A: Phase bootstrapping improves routing and task performance

This would support the claim that supervised phase structure is a useful design choice for MoE manipulation policies.

#### Outcome B: Phase bootstrapping improves routing but not task performance

This would show that the mechanism is real, but the policy stack still needs stronger control modeling or more capacity.

#### Outcome C: Warm-starting matches phase bootstrapping

This would mean phase supervision is not adding much beyond ordinary pretraining, and the design should be revised.

All three outcomes are useful because the contribution is an evidence-based system design, not a pre-committed claim.

---

### 11. Why This Study Does Not Need Vision Yet

The current paper does not need vision to be scientifically valuable.

The reason is simple: the paper is not trying to solve perception. It is trying to determine whether the MoE strategy itself is worth scaling. A minimal state-only setup gives cleaner conclusions and avoids the noise introduced by visual backbone choice, visual pretraining, and data-hungry perception modules.

If the MoE strategy is already unstable or unhelpful in the state-only setting, adding vision later will not fix the underlying routing problem. If it works in the state-only setting, then vision can be added as a later extension rather than as a confound.

That is the correct engineering logic: first verify the mechanism, then scale the interface.

---

### 12. Relation to Existing Work

The proposal should be positioned relative to existing robotics work as follows.

Large-scale robotics systems such as Open X-Embodiment / RT-X are strongly data-scaled and multi-embodiment. Visuomotor policies such as Diffusion Policy and action-chunking systems such as ACT focus on perception-plus-control pipelines. Recent MoE robotics papers such as AdaMoE and MoE-ACT show that sparse specialization is promising, but they also confirm that routing and expert assignment remain difficult design problems.

This proposal is different because it intentionally removes vision and focuses on the routing question itself. That gives the paper a clear methodological identity: it tests whether phase-supervised bootstrapping is a valid architectural foundation before the system is expanded.

---

### 13. Final Positioning Statement

This project proposes a **state-only controlled study** of phase-bootstrapped MoE training for long-horizon manipulation.

The paper’s goal is to determine whether supervised phase structure improves routing stability, expert specialization, and data efficiency under matched conditions. Vision is excluded from the first paper because it would add confounds that obscure the mechanism under study. The experimental design is intentionally minimal so that the conclusions can be trusted.

If the study succeeds, it establishes a validated foundation for later scaling with vision, stronger backbones, and larger datasets. If the study fails, that failure is still valuable because it tells us the phase-bootstrapping idea is not the right foundation for the final scalable system.

That is the purpose of the paper: to produce a rigorous answer that can guide the next stage of system building.
