# Topology-Initialized Routing in a Memoryless Mixture-of-Experts Policy

## Abstract

This study examines whether topology-derived prototype initialization improves
the organization of routing in a memoryless, direct-action mixture-of-experts
(MoE) policy for robot manipulation. The policy encodes the observed state,
routes the latent representation to an expert using prototype distances, and
produces the action directly without recurrence or task-state feedback. The
central hypothesis is that topology-derived prototypes can organize experts
around phase-relevant regions of the demonstration manifold, producing more
phase-aligned and temporally coherent routing.

The evidence supports the structural part of the hypothesis. In a matched
Can/Square initialization ablation, topology-derived prototypes produced
higher phase-expert alignment and lower measured routing-switch rates than
random and phase-derived initialization. The performance consequence was
task-dependent: topology initialization achieved the strongest pooled Can
result among the hard-routing initialization arms, but it did not improve
Square success and was below the softmax and plain-encoder controls. The
results therefore support a conditional routing-organization hypothesis, not
a claim of universal performance superiority. Quantitatively, topology
initialization achieved 76.0% Can success and 26.7% Square success in the
focused ablation; the corresponding phase-initialized results were 68.7% and
31.3%, while the softmax control achieved 73.3% and 35.3%.

## 1. Research question

Mixture-of-Experts models divide a policy into specialized experts and use a
router to select or combine them. This modular structure is attractive when a
task contains distinct behavioral regimes, but routing quality has two separate
dimensions: the organization of the expert partition and the quality of the
closed-loop policy produced by that partition. These dimensions should not be
treated as interchangeable.

Prior MoE research has identified routing fluctuation and training instability
as important design concerns. StableMoE, for example, separates the learning
of a routing strategy from later routing decisions to reduce routing
fluctuation; Switch Transformers likewise identify training stability and
expert utilization as practical concerns in sparse MoE systems ([Dai et al.,
2022](https://aclanthology.org/2022.acl-long.489/); [Fedus et al.,
2022](https://www.jmlr.org/papers/volume23/21-0998.html)). Those results motivate
the present question but do not establish that a routing strategy developed for
language models will improve robotic control.

The research question is:

> Does topology-derived prototype initialization produce a more phase-aligned
> and temporally coherent router in a memoryless direct-action MoE, and does
> that structural organization translate consistently into closed-loop task
> success?

## 2. Method and hypothesis

Let $x_t$ denote the observed robot state and let $z_t = E(x_t)$ be the state
encoder output. Given prototype vectors $c_1,\ldots,c_K$, the hard router
selects

$$
r_t = \arg\min_k \lVert z_t-c_k \rVert_2^2,
$$

and the selected expert produces the direct action

$$
u_t = f_{r_t}(z_t).
$$

The evaluated configuration uses `beta=0` residual experts, so the deployed
policy is memoryless and does not use a feedback-controller state. The
topology condition initializes the prototypes from topology-derived phase
segments. The comparison conditions use random or phase-derived prototype
initialization under the same matched training contract. A softmax router and a
plain-encoder policy provide additional controls.

The hypothesis is:

> Topology-derived initialization encourages phase-aligned expert
> specialization and lower routing variability in a memoryless hard-routing
> MoE. Its effect on task success is conditional on whether the learned
> topology aligns with the action regimes required by the task.

This hypothesis makes two distinct predictions. The first concerns routing
organization and is evaluated with phase-expert NMI and routing-switch rate.
The second concerns closed-loop performance and is evaluated with rollout
success. The second prediction does not follow automatically from the first.

## 3. Experimental protocol

The final benchmark contains five robosuite manipulation tasks, three training
seeds, and 50 rollout episodes per task-seed cell. Lift, Can, Square, and
ToolHang use a 500-step horizon; Transport uses a 700-step horizon. The
focused initialization ablation evaluates Can and Square with matched reset
banks within each task. All focused arms use `beta=0` and
the same margin-disabled Stage 2 objective; the objective is held fixed so
that the comparison isolates initialization and routing-package effects.

The complete PhaseForge protocol is evaluated separately with its specified
margin objective. Consequently, the focused initialization numbers are used
to answer the initialization question, while the complete-method numbers are
used to characterize the final PhaseForge configuration. They are not treated
as interchangeable causal comparisons.

## 4. Results

### 4.1 Final five-task benchmark

The final benchmark results are pooled over three seeds and 50 episodes per
seed:

| Task | BC | Plain encoder | Softmax top-1 | Phase-random router | PhaseForge |
|---|---:|---:|---:|---:|---:|
| Lift | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) |
| Can | 94/150 (62.7%) | 54/150 (36.0%) | 103/150 (68.7%) | 93/150 (62.0%) | 117/150 (78.0%) |
| Square | 51/150 (34.0%) | 75/150 (50.0%) | 58/150 (38.7%) | 58/150 (38.7%) | 61/150 (40.7%) |
| ToolHang | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) | 0/150 (0.0%) |
| Transport | 0/150 (0.0%) | 1/150 (0.7%) | 0/150 (0.0%) | 0/150 (0.0%) | 1/150 (0.7%) |

The benchmark demonstrates task dependence. PhaseForge performs best among
the listed methods on Can, while plain behavioral cloning performs best on
Square. Lift is saturated, and ToolHang and Transport provide little positive
signal for this memoryless policy family.

### 4.2 Focused initialization ablation

The focused ablation uses the same `beta=0` direct-action configuration across
the initialization arms:

| Method | Can pooled | Can mean ± SD | Square pooled | Square mean ± SD | Can + Square |
|---|---:|---:|---:|---:|---:|
| Random initialization | 110/150 (73.3%) | 73.3 ± 7.6% | 42/150 (28.0%) | 28.0 ± 13.1% | 152/300 (50.7%) |
| Phase initialization | 103/150 (68.7%) | 68.7 ± 5.0% | 47/150 (31.3%) | 31.3 ± 6.4% | 150/300 (50.0%) |
| Topology initialization | 114/150 (76.0%) | 76.0 ± 8.7% | 40/150 (26.7%) | 26.7 ± 4.2% | 154/300 (51.3%) |
| Representation-BC control | 101/150 (67.3%) | 67.3 ± 7.0% | 48/150 (32.0%) | 32.0 ± 8.7% | 149/300 (49.7%) |
| Softmax top-1 control | 110/150 (73.3%) | 73.3 ± 7.0% | 53/150 (35.3%) | 35.3 ± 3.1% | 163/300 (54.3%) |

The mean and standard deviation are computed across the three training seeds;
the standard deviations are reported in percentage points.

Topology initialization has the strongest pooled Can result among the three
initialization arms. On Square, it is below phase initialization, the
representation-BC control, and the softmax control. The matched-seed results
show the same task dependence rather than a consistent cross-task advantage.

### 4.3 Routing organization

The training diagnostics support a structural effect. NMI denotes normalized
mutual information between the phase labels and selected experts; switch rate
denotes the fraction of adjacent evaluated timesteps assigned to different
experts:

| Method | Can phase-expert NMI | Square phase-expert NMI | Can switch rate | Square switch rate |
|---|---:|---:|---:|---:|
| Random initialization | 0.07 | 0.09 | 0.11 | 0.10 |
| Phase initialization | 0.08 | 0.09 | 0.11 | 0.10 |
| Topology initialization | 0.67 | 0.51 | 0.04 | 0.07 |
| Representation-BC control | 0.41 | 0.47 | 0.06 | 0.06 |
| Softmax top-1 control | 0.72 | 0.61 | 0.04 | 0.05 |

Topology initialization changes the organization of the latent expert
partition. It is associated with higher phase-expert alignment and fewer
measured routing switches than the random and phase-initialized arms. The
softmax control also exhibits strong routing diagnostics, so topology
initialization is not shown to be uniquely optimal on these measures.

## 5. Interpretation

The results support the following claim:

> Topology-derived prototype initialization can organize a memoryless MoE
> around phase-relevant structure, producing more phase-aligned and lower-switch
> routing than unstructured initialization under the matched ablation
> protocol.

The results do not support the stronger claim that this organization guarantees
better task success. The Square results provide the clearest counterexample:
the topology arm has more structured routing than the random and phase arms,
but lower rollout success than several controls. This is not a contradiction
of the structural result. It shows that routing organization and closed-loop
control quality are distinct properties.

### Commanded-action transitions

For recorded traces, the commanded action difference

$$
\Delta u_t = \lVert u_t-u_{t-1}\rVert_2
$$

is larger at expert switches than at non-switch steps. In the focused traces,
the reported switch-to-non-switch jump ratios are approximately 5.7--6.7 for
the topology arm. This is evidence of a discontinuity in the commanded policy
output at some expert transitions.

It is not evidence, by itself, of a particular physical failure mechanism. The
available traces do not contain force, contact, or actual end-effector velocity
measurements. Moreover, the softmax control has a larger reported jump ratio
in the same audit while achieving higher Square success. Therefore, action
discontinuity is a plausible factor for future investigation, not a sufficient
explanation of Square failure.

## 6. Scope and limitations

The conclusions are limited to the evaluated memoryless direct-action policy
and the reported benchmark protocol.

1. Three training seeds and 50 episodes per seed provide useful comparative
   evidence but do not establish universal statistical superiority; no formal
   significance claim is made from pooled episode counts.
2. The focused ablation isolates initialization under a fixed margin-disabled
   objective; it does not attribute the complete PhaseForge result to
   initialization alone.
3. Routing NMI and switch rate measure organization, not physical stability or
   task success.
4. The trace archive lacks the physical measurements required to attribute
   timeouts to contact, force, jamming, action direction, or state coverage.
5. The current local result archive contains metrics and configurations, but
   not all training checkpoints. Independent weight reloading therefore
   requires preserving or recovering the original training artifacts.

## 7. Conclusion

PhaseForge provides a topology-informed way to organize a phase-conditioned
memoryless MoE policy. The experiments show that topology-derived prototypes
substantially alter the learned routing structure and can improve performance
on Can. They do not show a universal success advantage: Square favors other
controls, and the softmax router is strongest in the focused pooled comparison.

The appropriate conclusion is therefore conditional:

> Topology-informed initialization is a meaningful structural mechanism for
> organizing expert routing in memoryless manipulation policies. Whether that
> organization improves closed-loop performance depends on the task and on the
> compatibility between the topology-derived partition and the task’s action
> regimes.

This conclusion reports both the contribution and its boundary without
identifying routing organization with control performance.

External references motivate the MoE routing question; all numerical results
and implementation claims in this report come from the local experiment
artifacts listed below.

## References

1. R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive
   Mixtures of Local Experts,” *Neural Computation*, 1991. [Publisher page](https://direct.mit.edu/neco/article/3/1/79/5560/Adaptive-Mixtures-of-Local-Experts).
2. D. Dai, L. Dong, S. Ma, B. Zheng, Z. Sui, B. Chang, and F. Wei,
   “StableMoE: Stable Routing Strategy for Mixture of Experts,” *ACL*, 2022.
   [ACL Anthology](https://aclanthology.org/2022.acl-long.489/).
3. W. Fedus, B. Zoph, and N. Shazeer, “Switch Transformers: Scaling to
   Trillion Parameter Models with Simple and Efficient Sparsity,” *Journal of
   Machine Learning Research*, 2022. [JMLR](https://www.jmlr.org/papers/volume23/21-0998.html).

## Local evidence

- Final benchmark protocol: `experiments/final_causal_matrix.json`
- Focused initialization protocol: `experiments/router_initialization_ablation.json`
- Focused ablation metrics: `final_experiments_results/abalation_final/`
- Final PhaseForge Square runs: `debug_run/phaseforge_square/`
- Routing and commanded-action audits: `outputs_cpu_debug/`
