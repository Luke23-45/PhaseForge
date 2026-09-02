# PhaseForge 2.0 — Final Implementation Plan

**Status:** Proposed implementation plan  
**Scope:** Dynamics-aware phase discovery for contact-rich state-only manipulation  
**Primary objective:** Reduce the failure of PhaseForge on Can and Square while preserving the existing PhaseForge study as an immutable baseline.

## 1. Decision

Proceed with PhaseForge 2.0 as a new method and new experimental condition. Do not relabel or overwrite the existing PhaseForge results.

The completed five-task results establish the motivation:

| Task | PhaseForge | BC | Best relevant comparator | Interpretation |
|---|---:|---:|---:|---|
| Lift | 0.71 | 0.54 | PhaseForge | The method can help on a simple task. |
| Can | 0.30 | 0.48 | Warm-Start MoE 0.49 | PhaseForge's phase prior is harmful. |
| Square | 0.13 | 0.13 | Scratch/Warm-Start MoE 0.21 | The MoE architecture can work, but the PhaseForge prior hurts. |
| ToolHang | 0.00 | 0.00 | all 0.00 | No method currently solves this task. |
| Transport | 0.00 | 0.00 | all approximately 0.00 | No method currently solves this task. |

These values are from the completed analysis matrix in `studies/analysis/outputs/tables/T1_success_matrix.md`. The implementation below treats the observed failure as real and task-dependent. The proposed mechanism—coarse rule-based phase labels producing poor contact-regime geometry—is a testable working hypothesis, not an unsupported claim of proof.

## 2. Research question

The PhaseForge 1.0 question was:

> Can coarse privileged phase information be transferred into a useful MoE routing prior?

The PhaseForge 2.0 question is:

> Does replacing coarse rule-based phase geometry with action-conditioned dynamical regimes improve routing and closed-loop imitation on contact-rich tasks?

The key comparison is not simply “semantic labels versus unsupervised labels.” It is a controlled comparison of the representation and bootstrap sources:

| Stage-1 representation | Stage-2 router bootstrap | Purpose |
|---|---|---|
| Existing rule-based phases | Existing phase prototypes | Existing PhaseForge baseline |
| Existing rule-based phases | Dynamic-regime prototypes | Router-bootstrap effect |
| Dynamic regimes | Existing phase prototypes | Representation effect |
| Dynamic regimes | Dynamic-regime prototypes | Proposed PhaseForge 2.0 method |

All four cells must use the same expert count, top-k, expert initialization, optimizer, epochs, seeds, reset bank, and evaluation protocol.

## 3. Design principles

1. Preserve the canonical PhaseForge 1.0 artifacts and results.
2. Change only the source of phase/regime labels in the first implementation.
3. Keep the deployed policy label-free. Dynamic labels are training-time artifacts only.
4. Fit all discovery models using training demonstrations only.
5. Evaluate behavior using closed-loop rollout success, not offline MSE or routing alignment alone.
6. Start with the smallest dynamics-aware method that can falsify the hypothesis.
7. Defer VQ-VAE until a simpler dynamics-based method demonstrates value.

## 4. PhaseForge 2.0 method

### 4.1 Stage 0 — Dynamic-regime discovery

For each task, construct an action-conditioned transition feature for every valid demonstration step:

\[
\phi_t = [x_t, a_t, \Delta x_t], \qquad \Delta x_t = x_{t+1} - x_t.
\]

Here, `x_t` is the normalized structured state and `a_t` is the normalized demonstration action. The model must use the train-split normalization statistics already defined by the dataset pipeline. Raw and normalized feature choices must be recorded in the provenance metadata.

The primary implementation is a sticky switching linear dynamical model:

\[
x_{t+1} = A_{z_t}x_t + B_{z_t}a_t + b_{z_t} + \epsilon_t,
\]

where `z_t` is a discrete dynamic regime. A transition prior favors persistence but allows regime changes at contact events. Fit the model with EM or variational inference, then decode each trajectory with Viterbi or maximum-posterior state inference.

The first production setting uses **six dynamic regimes** so that the primary PhaseForge 2.0 condition remains matched to the canonical six-expert router. The number-of-regimes sweep is a later ablation, not part of the first claim.

Persist the following artifacts per task and training split:

- fitted transition matrices and intercepts;
- covariance or residual-scale parameters;
- transition matrix and persistence prior;
- regime count and minimum-duration setting;
- decoded training labels;
- validation-label inference procedure and labels;
- train-only normalization fingerprint;
- fit seed and implementation version;
- per-regime occupancy, duration, action variance, and transition residual variance.

### 4.2 Discovery quality checks

Dynamic labels are accepted for the pilot only when all of these checks pass:

- no regime has fewer than 2% of valid training steps;
- no trajectory is assigned one regime for its entire duration unless the trajectory is genuinely single-regime;
- decoded labels are temporally plausible under the configured persistence/minimum-duration rule;
- held-out transition negative log-likelihood is finite and better than the corresponding single-dynamics model;
- on Can and Square, within-regime transition-residual variance is lower than or comparable to the existing phase labels;
- regimes are not explained solely by trajectory ID, demonstration length, or action magnitude;
- validation labels are inferred from the fitted train-only model rather than refit on validation data.

If these checks fail, do not train the MoE. Report the discovery failure and revise the discovery model before spending GPU time.

### 4.3 Stage 1 — Dynamic-regime representation learning

Extend the existing phase-label pipeline so that the target field is generic—`regime` or `phase` with a declared vocabulary size—rather than hard-coded to six labels.

For the dynamic representation condition:

- retain the existing state encoder and action head;
- replace the rule-based phase target with the decoded dynamic-regime target;
- retain the auxiliary classification head with six outputs in the primary experiment;
- retain the existing action-plus-auxiliary objective initially;
- select the checkpoint using the canonical `val/loss_action` rule;
- record action loss, regime loss, micro accuracy, balanced accuracy, and per-regime recall.

The dynamic labeler must not see evaluation reset states, rollout states, policy actions, or any validation/test data while fitting. It may infer validation labels after fitting from the validation trajectory using the frozen discovery model.

### 4.4 Stage 2 — Dynamic router bootstrap

Use the same canonical Stage-2 contract as PhaseForge 1.0:

- six experts;
- top-2 routing;
- centroid/prototype router initialization;
- 50% partial expert warm-start;
- frozen encoder for the primary condition;
- no privileged labels during Stage-2 learning or deployment;
- the existing action, load-balance, and routing diagnostics;
- checkpoint selection on `val/loss_action`;
- deterministic learned routing at evaluation.

For the dynamic-bootstrap condition, compute one normalized latent prototype per dynamic regime from the frozen Stage-1 encoder and initialize router rows from those prototypes.

The primary PhaseForge 2.0 condition is therefore:

\[
\text{dynamic-regime Stage 1} \rightarrow \text{dynamic prototypes} \rightarrow \text{autonomous MoE Stage 2}.
\]

No dynamic regime label is required by `get_action()` or by the rollout evaluator.

## 5. Required implementation changes

### 5.1 New discovery package

Add a focused package under `phaseforge/data/`:

```text
phaseforge/data/dynamics/
    __init__.py
    features.py          # x, a, delta-x construction and validation
    switching_linear.py  # fit, decode, held-out scoring
    diagnostics.py       # occupancy, duration, residual/action coherence
    artifacts.py         # versioned serialization and provenance
```

The discovery implementation must be deterministic for a fixed task, data fingerprint, and seed.

### 5.2 Dataset and ingestion

Update the ingestion/cache path so a task cache can carry both:

- existing rule-based labels, stored as `phase_semantic` or `phase_rule`; and
- dynamic labels, stored as `phase_dynamic`, with a discovery-artifact reference.

The canonical existing `phase` field must remain unchanged for all PhaseForge 1.0 experiments. PhaseForge 2.0 configs explicitly select the dynamic field.

### 5.3 Model and training generalization

Update the model/trainer interfaces only where needed to support a declared regime vocabulary:

- remove assumptions that every label field is always the legacy six-phase label;
- preserve six as the default for all existing configurations;
- validate that the target vocabulary, classification head, prototype count, and router mapping agree;
- preserve frozen-encoder behavior and the existing checkpoint-monitor contract;
- preserve all existing baseline behavior bit-for-bit where practical.

### 5.4 Router mapping

The initial implementation requires `num_dynamic_regimes == num_experts == 6`, avoiding an unvalidated many-to-one phase-to-expert mapping.

The general mapping path must use a finite cross-entropy or `KL(target || router_distribution)` objective if a teacher signal is later added. Never use `KL(router_distribution || onehot_target)` as written in the earlier proposal; that direction is ill-defined against a one-hot distribution.

### 5.5 Configuration

Add explicit configuration groups rather than changing the canonical defaults:

```text
phaseforge/config/dynamics/
    disabled.yaml
    switching_linear_k6.yaml
```

Add model/config variants:

```text
phaseforge/config/models/
    phaseforge_dynamic.yaml
    baselines/phaseforge_rule_encoder_dynamic_router.yaml
    baselines/phaseforge_dynamic_encoder_rule_router.yaml
```

Suggested names for the four-factorial cells:

- `pf_rule_rule` — existing PhaseForge;
- `pf_rule_dynamic` — existing representation, dynamic router prototypes;
- `pf_dynamic_rule` — dynamic representation, existing phase prototypes;
- `pf_dynamic_dynamic` — proposed PhaseForge 2.0.

The existing `phaseforge` identity remains the 1.0 canonical method. Do not replace it with the dynamic method until the new study is complete and separately documented.

## 6. Experimental protocol

### 6.1 Pilot

Run the four factorial cells on:

- Lift;
- Can;
- Square;

using seeds 42, 43, and 44, the existing paired reset banks, and the same 50 evaluation episodes per seed.

The pilot should reuse completed PhaseForge 1.0 results where the configuration and provenance are exactly matched. New conditions must be evaluated on the same reset cases as their comparator.

### 6.2 Required comparisons

For every pilot cell, record:

- per-seed rollout success and pooled Wilson intervals;
- paired per-episode PhaseForge 2.0 minus baseline deltas;
- failure category and deepest phase/regime reached;
- offline action MSE, clearly marked diagnostic-only;
- Stage-1 regime accuracy and balanced accuracy;
- Stage-2 initial and final routing alignment;
- routing entropy, switch rate, balance, and collapse;
- expert behavioral matrix and pairwise expert divergence;
- parameter count, active capacity, wall-clock time, and peak memory;
- full configuration, dataset, discovery-artifact, and checkpoint hashes.

### 6.3 Progression gate

Proceed to the full five-task matrix only if the pilot shows all of the following:

1. `pf_dynamic_dynamic` improves over the existing PhaseForge condition on at least one of Can or Square by a meaningful descriptive margin, targeted at ≥0.05 absolute success rate.
2. It does not regress on the other two pilot tasks by more than 0.05 absolute success rate.
3. The improvement is not explained by a different expert count, expert initialization, evaluation bank, checkpoint rule, or data leakage.
4. Dynamic regimes pass the discovery quality checks and show lower or more coherent transition residuals on the affected tasks.
5. The paired-delta and per-seed results are directionally consistent; with three seeds, report them as descriptive evidence rather than significance claims.

If the gate fails, stop the full rollout and diagnose whether the issue is discovery quality, representation learning, router bootstrap, expert capacity, or simulator/task solvability.

### 6.4 Full evaluation

If the pilot passes, run the selected dynamic method and its matched controls on all five tasks:

- Lift;
- Can;
- Square;
- ToolHang;
- Transport.

ToolHang and Transport must remain in the evaluation for completeness, but no success claim should be made for them unless a baseline also demonstrates non-trivial solvability.

## 7. Ablations after the primary pilot

Only after the primary pilot passes, add:

1. Dynamic regime count: 4, 6, and 8, with expert capacity reported explicitly.
2. Discovery model: switching linear model versus residual-feature clustering.
3. Persistence/minimum-duration sensitivity.
4. Frozen versus fine-tuned encoder.
5. Dynamic labels used only for router bootstrap versus used for both Stage 1 and bootstrap.
6. Dynamic labels with action-only, transition-only, and state-action-transition features.

The action-only clustering fallback is a diagnostic baseline, not the preferred method. Similar actions do not guarantee similar control regimes because the same action can have different effects under different contact states.

## 8. Explicit non-goals

The first PhaseForge 2.0 implementation will not include:

- VQ-VAE or a learned discrete autoencoder;
- images or a vision encoder;
- online phase discovery during deployment;
- a new hierarchical MoE architecture;
- a recurrent policy;
- evaluation-time privileged dynamic labels;
- claims of solving ToolHang or Transport merely because the discovery model exists.

VQ-VAE is deferred because it introduces a new representation-learning system, decoder, codebook-collapse risks, and additional tuning variables. It should be considered only if train-only dynamic discovery is unavailable or demonstrably insufficient.

## 9. Verification and tests

Add tests for:

- feature construction and shape/dtype validation;
- train-only fitting and validation decoding;
- deterministic fit/decode for a fixed seed;
- no empty or collapsed regimes;
- minimum-duration and transition-prior behavior;
- artifact save/load round trips and hash stability;
- label-field selection without changing legacy PhaseForge behavior;
- dynamic head/router/prototype dimension agreement;
- no dynamic-label access in Stage-2 rollout inference;
- matched four-cell configuration composition;
- paired reset-bank and provenance compatibility;
- failure-closed behavior for missing or stale discovery artifacts.

Run the existing test suite before and after the change. Existing PhaseForge 1.0 configurations must retain their current outputs and protocol semantics.

## 10. Reporting plan

The final report should separate three conclusions:

### Behavioral conclusion

Whether dynamics-aware phase discovery improves Can and Square rollout success relative to the completed PhaseForge 1.0 baseline and matched controls.

### Mechanistic conclusion

Whether dynamic regimes produce more coherent transition/action-conditioned latent groups and a more useful initial router. Routing metrics alone must not be presented as manipulation success.

### Scope conclusion

Whether the method is task-conditional. The current data already indicates that PhaseForge 1.0 is beneficial on Lift but harmful or ineffective on Can, Square, ToolHang, and Transport. PhaseForge 2.0 should not be advertised as universally superior unless the full matrix supports that claim.

The existing results remain the PhaseForge 1.0 reference table. New results receive separate method names, separate provenance, and separate statistical summaries.

## 11. Final acceptance criteria

PhaseForge 2.0 is considered successfully implemented only when:

- the dynamic discovery artifacts are reproducible and train-only;
- legacy PhaseForge results remain unchanged;
- all four factorial cells compose and run under the same protocol;
- the primary dynamic method completes the Lift/Can/Square pilot;
- the pilot passes the progression gate or is rejected with a documented diagnosis;
- rollout success, paired deltas, routing diagnostics, and failure categories are all persisted;
- no claim exceeds what the completed five-task evidence supports.

The implementation should end with a decision between:

1. **Adopt PhaseForge 2.0:** dynamic regimes improve the contact-rich tasks and survive matched controls.
2. **Retain PhaseForge 1.0:** dynamic discovery does not improve behavior, despite improving offline regime metrics.
3. **Reject the phase-routing hypothesis:** failures persist across both rule-based and dynamic decompositions, indicating that the main limitation lies elsewhere—such as state observability, temporal dependence, action horizon, or simulator/task difficulty.
