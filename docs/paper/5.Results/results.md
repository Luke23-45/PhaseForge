# 5. Results

The five-task benchmark and the focused ablation use different Stage-2 objectives. The full benchmark uses the regime-initialized configuration with active margin loss \((\lambda_m>0)\). The matched Can/Square ablation disables margin loss in every prototype arm \((\lambda_m=0)\). Their absolute success rates should therefore not be compared directly.

## 5.1 Five-Task Benchmark

Table 1 reports closed-loop success for all deployable methods. Brackets denote 95% Wilson score intervals over pooled rollout episodes. These intervals describe episode-level uncertainty, not variation across training seeds.

Lift is saturated: all methods except Scratch MoE and Factorial Floor reach 100% success. It does not distinguish the modular architectures.

On Can, the full regime-initialized configuration records 78% success \([71,84]\), compared with 63% \([55,70]\) for BC, 69% \([61,76]\) for Learned Softmax Top-1, and 62% \([54,69]\) for Phase-Random. Its paired difference from BC is \(+0.153\) across the three seeds. The Holm-adjusted paired sign test does not reject the null (\(p=1.0\); Table A15), so this ordering is descriptive.

Square has a different ranking. Plain Encoder records 50% success \([42,58]\), followed by the full regime-initialized configuration at 41% \([33,49]\), with Learned Softmax Top-1 and Phase-Random at 39%. The paired difference between the full regime-initialized configuration and Plain Encoder is \(-0.093\). The action-only representation control records a higher observed success rate than the full regime-initialized configuration on this task.

ToolHang remains unsolved: every evaluated method has 0% success. Transport produces 0–2 successful episodes out of 150, depending on the method. These tasks remain unresolved under the evaluated memoryless policies and training budget, so they do not distinguish the initialization conditions.

The full five-task sweep provides contextual coverage rather than a causal estimate of prototype initialization. Several tasks are saturated or near the success floor, and the full configuration differs from the matched ablation through its active margin objective.

## 5.2 Controlled Initialization Ablation

Table 2 reports the focused Can/Square comparison. The three matched prototype arms use the same phase-aware Stage-1 representation, expert initialization, Stage-2 optimizer, and Stage-2 objective. They differ only in prototype initialization:

1. trajectory-derived regime centroids;
2. rule-derived phase centroids;
3. random prototype parameters.

Margin loss is disabled in every matched arm.

### Offline routing organization

Routing metrics are computed on validation demonstrations. Trajectory-derived initialization has the highest mean NMI among the matched prototype arms on both tasks: 0.67 on Can and 0.51 on Square. Random and rule-based initialization produce NMI values between 0.07 and 0.09.

Trajectory-derived initialization also has the lowest mean routing-switch rate: 0.04 on Can and 0.07 on Square, compared with 0.10–0.11 for the random and rule-based conditions. These values describe routing assignments on the validation-demonstration distribution, not on rollout states.

Rule-based prototype initialization produced final NMI values similar to random initialization under the matched configuration. The endpoint measurements do not identify whether this reflects initial prototype geometry, prototype scale or separation, expert utilization, or subsequent training dynamics.

### Closed-loop success

On Can, trajectory-derived initialization records the highest observed success among the matched prototype arms: 76.0%, compared with 73.3% for random initialization and 68.7% for rule-based initialization.

On Square, the ordering reverses. Trajectory-derived initialization records 26.7% success, compared with 28.0% for random initialization and 31.3% for rule-based initialization.

The matched arms differed substantially in final offline routing diagnostics. The observed rollout orderings on Can and Square are descriptive and do not establish a task-by-initialization interaction. Across three training seeds, the closed-loop comparisons were not statistically resolved after Holm adjustment.

### Architectural diagnostics

Learned Softmax Top-1 is not a matched initialization control because it replaces prototype routing with a learned gating network. It reaches NMI values of 0.72 on Can and 0.61 on Square, with rollout success of 73.3% and 35.3%, respectively. This condition shows that high phase-expert alignment can arise without trajectory-derived prototype initialization. It does not isolate the effect of gating or prototype initialization.

Plain Encoder is also diagnostic rather than matched because it changes representation pre-training. Its values are reported in Table 2 but do not estimate the isolated effect of prototype initialization.

Teacher-Forced dispatches experts with trajectory-derived regime labels during Stage-2 training, then uses predictions from a frozen rule-phase head during rollout. It records 41% success on Lift, 1% on Can, and 2% on Square. This condition evaluates a label-and-dispatch mismatch between training and deployment. It is not an oracle-routing experiment and does not establish that either label vocabulary is independently useful or ineffective.