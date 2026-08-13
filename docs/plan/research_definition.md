# PhaseForge — Research Definition and Falsifiable Hypotheses

**Status:** finalized research definition; implementation and empirical gates remain pending

**Scope:** non-visual robot manipulation from privileged structured low-dimensional simulator state

**Benchmark:** robomimic v0.1 low-dimensional demonstrations with a single, explicitly pinned robosuite release track

This document defines what PhaseForge is testing. It is intentionally more conservative than a novelty claim: the mechanism is a candidate contribution until the controlled experiments and a systematic prior-art search support that claim.

## 1. What the project is about

PhaseForge is a training-strategy study for a small manipulation policy. It asks whether phase structure can be used as an initialization prior for a mixture-of-experts router.

The proposed pipeline is:

1. Train a low-dimensional behavioral-cloning policy with an auxiliary phase-classification head.
2. Keep the learned encoder fixed for the specialization stage.
3. Compute one latent centroid per auxiliary phase from training demonstrations.
4. Initialize the MoE router from those centroids and initialize the experts from the pretrained action head.
5. Continue training the MoE with the same action objective and the declared routing-balance objective.
6. Deploy without phase labels. At inference, the policy receives only the declared structured state and its history once history support is implemented.

The central variable is therefore router initialization, not perception, model scale, language grounding, or a new simulator.

The primary scientific comparison is the controlled 2×2 matrix:

| Encoder initialization | Router initialization | Code cell |
|---|---|---|
| plain BC encoder | random | Warm-Start MoE |
| phase-supervised encoder | random | Phase-Pretrain Random-Router |
| plain BC encoder | phase-centroid | Plain-Encoder Phase-Bootstrap |
| phase-supervised encoder | phase-centroid | PhaseForge |

BC-MLP and BC-RNN are control floors. Scratch MoE is an additional baseline, not one of the four factorial cells. Teacher-Forced Routing and Ground-Truth Routing are diagnostic references and must be labeled as privileged-training or non-deployable, respectively.

## 2. What is already established by prior work

The following points are not PhaseForge contributions:

- Low-dimensional robot manipulation with task-relevant object state is an established evaluation setting. robosuite documents separate robot-proprioceptive and task-specific object-state observations, and robomimic treats low-dimensional state as a first-class observation modality.
- The robomimic CoRL 2021 study provides released low-dimensional datasets and BC/BC-RNN baselines on Lift, Can, Square, Transport, and Tool Hang. Its model-zoo results are approximate references and require matching software/data versions; they are not numbers PhaseForge may copy into its own results table.
- Temporal context is an established part of strong imitation-learning baselines. A single-step MLP is therefore a pilot baseline, not evidence for a history-dependent manipulation claim.
- Phase-aware or skill-aware expert specialization is not a new general idea. Recent work such as PAMAE and SMP uses phase/skill structure or phase-consistent routing in substantially different VLA/diffusion and multi-task settings.
- State-only evaluation does not imply bare proprioception. The main PhaseForge input includes structured object state and is privileged simulator state. The robot-only condition is a negative control, not the main research setting.

The relevant primary sources are listed in Section 8. These sources justify the benchmark and comparison discipline; they do not establish the PhaseForge mechanism as novel.

## 3. Candidate contribution, stated cautiously

The candidate contribution is:

> A controlled test of whether initializing an MoE router from phase-conditioned centroids in a frozen, phase-supervised low-dimensional latent produces more persistent phase–expert alignment and better simulator task success than matched random-router and plain-encoder controls.

This wording is deliberately narrower than “the first phase-aware MoE” or “a novel state-only manipulation method.” Those broader claims are not permitted. Phase-aware MoE, staged training, expert balancing, and structured state inputs already have relevant precedents.

Before using “novel,” “first,” or “state of the art” in a paper, perform a systematic search covering phase-aware MoE, skill-conditioned MoE, router initialization, latent-centroid routing, manipulation imitation learning, and low-dimensional robomimic experiments. Record search dates, databases, inclusion criteria, and excluded papers.

## 4. Falsifiable hypotheses

### H1 — Router-initialization effect

Holding the encoder initialization, expert initialization, data split, optimizer, parameter budget, and evaluation protocol fixed, phase-centroid initialization will produce stronger early phase–expert alignment and better alignment retention than random router initialization.

The cleanest test is:

```text
PhaseForge vs Phase-Pretrain Random-Router
```

Primary evidence: predeclared routing-alignment trajectory, final alignment, routing entropy, expert load, and collapse rate. A single final NMI value is insufficient because it can be produced by an imbalanced or collapsed router.

### H2 — Phase-representation effect

Holding router initialization and the MoE architecture fixed, a phase-supervised encoder will produce more useful phase-conditioned latent structure than a plain BC encoder.

The cleanest test is:

```text
PhaseForge vs Plain-Encoder Phase-Bootstrap
```

If this comparison is not run, PhaseForge cannot separate the value of phase-supervised representation learning from the value of centroid initialization.

### H3 — Behavioral effect

PhaseForge will improve the predeclared aggregate held-out simulator success over the matched Warm-Start MoE and Scratch MoE baselines, with the direction replicated on a predeclared majority of tasks, without reducing the structured-state BC floor.

This is the main performance hypothesis. The aggregate, task-level aggregation rule, paired test initial states, seed handling, and interval procedure must be frozen before the final test episodes. Offline action loss, phase accuracy, and routing metrics cannot substitute for paired rollout success.

### H4 — Phase observability

The declared phase labels must be predictable from the policy’s permitted input. If a teacher-forced model using the learned phase predictor performs far below a ground-truth-routing reference, the phase signal is not sufficiently observable from the deployed state and the PhaseForge mechanism is not well grounded.

Ground-truth routing is a diagnostic reference only. It is not a success upper bound: its experts, optimization, and routing rule differ from the learned policy, so its task success may be lower or higher for reasons unrelated to phase information.

### H0 — Null hypothesis

After controlling the protocol, PhaseForge does not improve task success or routing persistence over the matched controls. This is a valid outcome. It means the proposed initialization prior is not supported under this setting; it does not mean the benchmark or BC baseline is invalid.

## 5. Required experiment matrix

Every primary comparison must use the same task, low-dimensional schema, history window, trajectory split, normalization, optimizer budget, checkpoint rule, evaluation initial states, and training seeds.

Required rows:

1. Robot-only BC — information-ceiling negative control.
2. Structured-state BC-MLP — instantaneous control floor.
3. Structured-state BC-RNN or equivalent history baseline — temporal control floor.
4. Scratch MoE — architecture/training baseline without Stage 1 pretraining.
5. Warm-Start MoE — plain encoder plus random router.
6. Phase-Pretrain Random-Router — phase encoder plus random router.
7. Plain-Encoder Phase-Bootstrap — plain encoder plus centroid router.
8. PhaseForge — phase encoder plus centroid router.
9. Teacher-Forced Routing — privileged-training diagnostic; no ground-truth phase at deployment.
10. Ground-Truth Routing — non-deployable routing diagnostic only.

The four factorial rows are the causal core. BC-RNN and rollout validation are required because robomimic’s published protocol is not a single-step offline-loss exercise.

## 6. How results will be interpreted

### Positive mechanism result

The initialization hypothesis is supported only when the centroid-router comparison improves routing persistence or specialization under matched controls, and the effect is accompanied by a nontrivial task-success result. Balanced routing alone is not specialization.

### Mechanism without behavior

If PhaseForge improves NMI, routing stability, or expert utilization but not task success, report a routing-mechanism result only. Do not claim better manipulation.

### Behavior without mechanism

If PhaseForge improves task success but does not improve the declared routing diagnostics, report a behavioral result and treat the proposed mechanism as unverified. Do not infer causality from success alone.

### No difference

If PhaseForge matches the controls, report a controlled null result. This still identifies whether phase-centroid initialization is useful under the chosen state-based protocol.

### SR = 0

Use the evaluator decision tree from the final evaluation plan:

- scripted controller fails → environment/evaluator problem;
- scripted controller succeeds but structured BC fails → observation, action, temporal, or learning problem;
- structured BC succeeds but all MoEs fail → MoE implementation or optimization problem;
- PhaseForge succeeds but does not beat matched controls → the bootstrap hypothesis is unsupported.

Never convert SR = 0 into a claim of model failure until the evaluator and structured-state BC gates pass.

## 7. Claims permitted and prohibited

Permitted after the required gates pass:

> We study whether phase-centroid router initialization improves expert specialization and manipulation success in a controlled, privileged low-dimensional robomimic/robosuite setting.

Prohibited:

- “state of the art” against vision or VLA models;
- perception or visual-understanding claims;
- general manipulation claims from a simulator state oracle;
- claims that bare proprioception solves object-dependent tasks;
- claims that phase labels are official benchmark annotations;
- claims that routing metrics alone prove better control;
- “first” or “novel” claims without a documented systematic prior-art review;
- final success claims before the rollout adapter, reset protocol, scripted controller, BC floor, and history baseline are validated.

## 8. Literature record

- Mandlekar et al., **What Matters in Learning from Offline Human Demonstrations for Robot Manipulation**, CoRL 2021: [paper](https://arxiv.org/abs/2108.03298), [official dataset and reproduction documentation](https://robomimic.github.io/docs/v0.4/datasets/robomimic_v0.1.html).
- robomimic, **Robosuite Datasets**: [official observation extraction and train/validation filter documentation](https://robomimic.github.io/docs/v0.4/datasets/robosuite.html).
- robomimic, **Multimodal Observations**: [official low-dimensional modality documentation](https://robomimic.github.io/docs/tutorials/observations.html).
- robosuite, **Environments**: [official structured observation and success-predicate documentation](https://robosuite.ai/docs/modules/environments.html).
- Chi et al., **Diffusion Policy**, RSS 2023: [official project](https://diffusion-policy.cs.columbia.edu/), which exposes separate state-based and vision-based resources and uses the same robomimic task family for part of its simulation evaluation.
- Yang et al., **PAMAE**, 2026: [arXiv paper](https://arxiv.org/abs/2606.27144), relevant because it demonstrates that phase-aware MoE routing is already an active research direction, although its setting is VLA/action-generation rather than this state-only study.
- Hao et al., **SMP**, ICLR 2026: [paper](https://openreview.net/pdf?id=VSWjHIveqZ), relevant because it studies skill-mixture routing and phase-consistent expert activation in a diffusion-policy setting.

The literature supports the benchmark choice and warns against overstating the contribution. It does not, by itself, prove that PhaseForge’s specific centroid initialization is novel or effective; those remain empirical questions for the controlled matrix above.
