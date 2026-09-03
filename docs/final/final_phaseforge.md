# Final Revised Report — Memoryless Impedance-Switched PhaseForge  
**For:** Professor review  
**Date:** 2026-09-03  
**Project:** PhaseForge dynamic mixture-of-experts policy for robotic manipulation  
**Status:** Final revised direction after triple-check  

---

## 1. Final decision and scope

This report finalizes the revised PhaseForge direction under a strict constraint:

> **The PhaseForge policy must remain memoryless at deployment: one state in, one action out. No recurrent hidden state, no observation history, no action chunking, no oracle/sticky test-time routing, and no temporal crutches.**

The purpose of this revision is therefore not to make PhaseForge easier by adding temporal context. The purpose is to determine whether the core PhaseForge idea — dynamic regime discovery plus mixture-of-experts routing — can solve contact-rich manipulation when forced to operate under the same instantaneous-state contract as the baseline BC policy.

The final proposed architecture is:

> **Impedance-Switched PhaseForge (IS-PhaseForge):**  
> an observation-consistent, memoryless, switched mixture-of-experts policy using topological regime discovery, contrastive latent alignment, large-margin routing, and impedance-parameterized expert action heads.

This direction deliberately withdraws temporal/history-based solutions as the primary contribution. Temporal models may remain external diagnostics only, not the proposed PhaseForge method.

---

## 2. Triple-checked diagnosis of the current failure

After re-checking the architecture against the Can low-dimensional state space, the rollout failures, and the offline discovery pipeline, the current memoryless dynamic PhaseForge has four fundamental weaknesses.

### 2.1 Offline discovery is not observation-consistent

The current dynamic regime discovery uses a sticky Switching Linear Dynamical System fitted on privileged transition features:

\[
\phi_t = [x_t, a_t, \Delta x_t]
\]

but the deployed router sees only:

\[
x_t
\]

This creates an ill-posed inference problem. The router is asked to predict a regime defined by future transition information using only the current state. If two different dynamic regimes can occur at the same instantaneous state — for example, before/after contact, stuck/sliding, or approaching/retracting — the router cannot reliably infer the correct regime.

**Correction:** regime discovery must produce labels that are observable from the instantaneous state. Offline label generation may use trajectory information, but any label used for routing must pass an observability audit from \(x_t\) alone.

### 2.2 Top-k averaging of direct 7D actions is unsafe in contact-rich tasks

The current MoE blends expert actions in the robosuite action space:

\[
\hat{a}(x) = \sum_{e \in S(x)} w_e(x) f_e(z(x))
\]

If one expert produces a high-stiffness pulling command and another produces a compliant holding command, the weighted average can produce an action that belongs to neither behavior. In contact-rich manipulation, this can fail to overcome static friction, produce incorrect gripper behavior, or create unstable contact.

**Correction:** experts should not output raw absolute actions that are directly averaged. They should output parameters of a local feedback/impedance-style controller. Blending mechanical control parameters is much safer than blending raw kinematic commands.

### 2.3 Softmax routing has no spatial hysteresis

A memoryless router evaluates each state independently. Near a regime boundary, small observation noise can cause rapid switching between experts. This produces action chattering. In switched control systems, high-frequency switching between non-commuting vector fields can induce limit cycles and destabilize contact.

**Correction:** routing must be trained with explicit spatial margins so that decision boundaries lie in low-density or task-irrelevant regions of the state space. The router should behave approximately like a hard Voronoi partition, not a soft probabilistic average.

### 2.4 The action head has no local stability bias

A deterministic MSE action head merely imitates local action means. It does not encourage the expert to pull the system back toward a valid local manifold when small compounding errors occur.

**Correction:** each expert should be regularized to behave like a locally contractive or locally stabilizing controller. This is not a substitute for environment-level stability analysis, but it provides a strong architectural bias against compounding error.

---

## 3. Final architecture: Impedance-Switched PhaseForge

The revised architecture consists of five coupled components.

```text
demonstrations
    → topological/contact-consistent regime discovery
    → observability audit
    → contrastive regime-aligned encoder
    → large-margin prototype router
    → impedance-parameterized expert controllers
    → memoryless closed-loop manipulation
```

The deployment graph is strictly:

```text
x_t → normalize → encoder z_t → router → selected expert(s)
    → impedance/target parameters → action adapter → environment
```

No hidden state is stored across environment steps.

---

## 4. Component 1: Observation-consistent topological regime discovery

### 4.1 Replace transition-based SLDS as the primary routing signal

The current SLDS labels are useful as an offline diagnostic, but they should not be the primary router supervision signal if they are not inferable from \(x_t\).

Instead, PhaseForge should discover regimes from **topological/task-space variables** that are directly observable in the instantaneous state.

For the Can task, candidate variables include:

```text
EEF position
EEF orientation error or rotation representation
gripper aperture
object/can pose features, if exposed in object_observation
lid-related features, if exposed in object_observation
relative EEF-to-object distance
relative EEF-to-lid distance, if derivable
contact proxies, if available
```

If explicit lid angle or contact variables are not cleanly exposed, they should be inferred only through observable dimensions of `object_observation`, not through future transitions.

### 4.2 Use PELT change-point detection on task-space variables

For each demonstration, run multivariate change-point detection, preferably **PELT** or a robust Bayesian alternative, on the selected task-space variables.

The objective is to segment trajectories into intervals where the task geometry/contact mode is approximately constant.

A typical offline segmentation model is:

\[
\min_{\tau} \sum_{j} \mathcal{C}(s_{\tau_j:\tau_{j+1}}) + \beta \cdot |\tau|
\]

where:

- \(s_t\) is the selected task-space signal,
- \(\mathcal{C}(\cdot)\) is a segment cost, e.g. Gaussian negative log-likelihood,
- \(\beta\) penalizes over-segmentation,
- \(|\tau|\) is the number of change points.

Use a minimum segment length to avoid noise-induced switching.

### 4.3 Cluster segment prototypes into a finite regime set

PELT produces variable-length segments. These segments should be converted into a finite set of regime prototypes.

Procedure:

1. Extract segment features:
   - mean task variables,
   - variance,
   - gripper state,
   - object state,
   - relative geometry,
   - action statistics, optional diagnostic only.
2. Cluster segment prototypes across all training demonstrations.
3. Select \(K\) using evidence, not by forcing \(K=6\).

Recommended selection criteria:

\[
K^\* = \arg\max_K 
\big[
\text{Observability}(K)
+ \text{ActionExplanation}(K)
+ \text{Stability}(K)
- \text{Complexity}(K)
\big]
\]

where:

- **Observability**: can a classifier predict the regime from \(x_t\) alone?
- **ActionExplanation**: do regimes reduce action variance/residual entropy?
- **Stability**: are regime durations and transition matrices consistent across seeds?
- **Complexity**: penalize excessive \(K\) or rare fragmented regimes.

### 4.4 Mandatory observability audit

Before any regime label is used for router supervision, it must pass an audit.

Train a simple classifier from the instantaneous state \(x_t\) to the discovered regime label \(r_t\), using a trajectory-aware split to prevent leakage.

Required checks:

| Check | Purpose |
|---|---|
| macro-F1 from \(x_t\) to regime | verifies instantaneous inferability |
| confusion matrix | identifies aliased regime pairs |
| per-regime duration distribution | detects pathological fragmentation |
| regime occupancy | detects dead regimes |
| action residual reduction | verifies regimes explain behavior |

If two regimes are strongly confused from \(x_t\), they must be merged or redefined. Do not ask the router to predict unobservable labels.

This is a critical correction. If a regime cannot be inferred from the current state, then a memoryless PhaseForge policy cannot be expected to route on it correctly.

---

## 5. Component 2: Contrastive regime-aligned representation

The current Stage 1 uses cross-entropy phase prediction. This is weaker than necessary because it only asks the latent to classify regimes; it does not explicitly shape the latent geometry.

The revised Stage 1 should use **Supervised Contrastive Learning** to make regimes form compact, separated clusters in latent space.

Let:

\[
z_t = E_\theta(x_t)
\]

with \(z_t\) L2-normalized.

For a batch of states with discovered regime labels \(r_i\), the SupCon loss is:

\[
\mathcal{L}_{\text{SupCon}}
=
\frac{1}{|B|}
\sum_{i \in B}
\frac{-1}{|P(i)|}
\sum_{p \in P(i)}
\log
\frac{
\exp(z_i \cdot z_p / \tau)
}{
\sum_{a \neq i}
\exp(z_i \cdot z_a / \tau)
}
\]

where:

- \(P(i)\) is the set of positive samples in the same regime as \(i\),
- \(\tau\) is a temperature hyperparameter.

The Stage 1 loss becomes:

\[
\mathcal{L}_{\text{Stage 1}}
=
\mathcal{L}_{\text{action}}
+
\lambda_{\text{sc}}
\mathcal{L}_{\text{SupCon}}
\]

The action loss may initially remain a direct action MSE for warm-starting:

\[
\mathcal{L}_{\text{action}}
=
\|g_\phi(z_t) - a_t\|^2
\]

However, for full consistency with the final architecture, Stage 1 may also use the impedance-style action adapter described below.

### 5.1 Why SupCon is necessary

SupCon directly shapes the latent space so that routing can become geometric rather than fragile. The desired result is:

```text
same regime     → nearby latent states
different regime → separated latent states
```

This makes prototype-based routing meaningful.

### 5.2 Acceptance criterion for Stage 1

Before moving to Stage 2, visualize and quantify the latent clustering:

- UMAP/t-SNE for inspection,
- k-NN regime classification accuracy in latent space,
- average intra-regime distance,
- average inter-regime distance,
- silhouette score.

If the latent does not separate regimes cleanly, router training will not fix the problem.

---

## 6. Component 3: Large-margin prototype router

The router should not be an unconstrained softmax MLP that can produce ambiguous 50/50 splits at regime boundaries.

The final router should be prototype-based and margin-trained.

### 6.1 Router scores

Maintain one prototype per regime:

\[
c_k \in \mathbb{R}^{d_z}
\]

For a latent state:

\[
z_t = E_\theta(x_t)
\]

compute distances:

\[
d_{tk} = \|z_t - c_k\|_2
\]

The selected expert at inference is:

\[
k_t^\* = \arg\min_k d_{tk}
\]

This is a memoryless nearest-prototype/Voronoi router.

During training, prototypes may be learned, EMA-updated, or initialized from regime centroids and then fine-tuned.

### 6.2 Large-margin loss

For true regime \(y_i\), enforce a distance margin:

\[
\mathcal{L}_{\text{margin}}
=
\frac{1}{|B|}
\sum_{i \in B}
\sum_{j \neq y_i}
\max\left(
0,
m - (d_{ij} - d_{iy_i})
\right)
\]

This encourages:

\[
d_{iy_i} + m \le d_{ij}
\]

for all incorrect regimes \(j\).

This is essential because it pushes decision boundaries away from dense regime clusters and reduces boundary chattering.

### 6.3 Inference routing policy

The default deployment policy should be:

> **Top-1 hard routing.**

Top-2 routing may be tested only as an ablation, and only if the expert action parameterization makes blending physically meaningful. But the primary final method should commit to one expert.

No oracle routing, no sticky post-processing, and no test-time label smoothing should be used for the main reported result.

---

## 7. Component 4: Impedance-parameterized expert heads

This is the central architectural correction for contact-rich manipulation.

Experts should not merely regress raw 7D actions. They should output parameters of a local feedback controller.

### 7.1 Task-state extraction

Define a task-state vector:

\[
y_t = \psi(x_t)
\]

For Can, a reasonable task-state vector is:

```text
EEF position              3
EEF rotation representation 6 or quaternion-derived error
gripper state             1 or 2
```

The exact dimension can vary, but the important point is that \(y_t\) must be the controllable/task-relevant part of the state.

Object information remains available to the encoder and expert through \(x_t\), but the feedback error is computed in task space.

### 7.2 Expert outputs

Each expert \(k\) outputs:

\[
T_k(z_t) = \text{target task state}
\]

and

\[
\kappa_k(z_t) = \text{positive feedback gains}
\]

where:

\[
\kappa_k > 0
\]

is enforced using softplus or sigmoid with bounds.

The target task state may include:

```text
target EEF position
target EEF orientation
target gripper state
```

For orientation, avoid directly subtracting quaternions. Use either:

- axis-angle rotation error,
- 6D rotation representation,
- quaternion error mapped to a 3D tangent vector.

### 7.3 Action adapter

Given expert target \(T_k\) and current task state \(y_t\), compute task error:

\[
e_k(t)
=
\text{TaskError}(T_k(z_t), y_t)
\]

For example:

\[
e_k =
\begin{bmatrix}
p^\* - p_t \\
\text{RotErr}(R^\*, R_t) \\
g^\* - g_t
\end{bmatrix}
\]

Then compute the raw command:

\[
u_k(t)
=
\kappa_k(z_t) \odot e_k(t)
\]

and map it to the robosuite action range:

\[
a_t = \tanh\left(\frac{u_k(t)}{s}\right)
\]

or clip/scale according to the environment’s action convention.

This preserves the existing 7D action contract while changing the meaning of the expert output from “absolute action” to “local feedback command.”

### 7.4 Why this fixes action averaging

If top-2 blending is ever used, the combined command is:

\[
u(t)
=
\sum_{i \in S}
w_i
\kappa_i
\left(
T_i - y_t
\right)
\]

With positive diagonal gains, this can be interpreted as an effective impedance command:

\[
u(t)
=
K_{\text{eff}}
\left(
T_{\text{eff}} - y_t
\right)
\]

where:

\[
K_{\text{eff}}
=
\sum_i w_i K_i
\]

and:

\[
T_{\text{eff}}
=
K_{\text{eff}}^{-1}
\sum_i w_i K_i T_i
\]

This is much more meaningful than averaging raw 7D action vectors.

Blending a stiff pull with a compliant hold yields a valid intermediate stiffness/target. Blending raw “pull” and “hold” actions can yield a useless intermediate command.

### 7.5 Baseline fairness requirement

If PhaseForge uses the impedance-style action adapter, then the baseline BC should also be trained with the same action adapter for the decisive comparison.

Otherwise, an improvement could be attributed to the action parameterization rather than to PhaseForge routing.

Therefore the final controlled comparison must include:

| Method | Action parameterization | Routing |
|---|---:|---|
| BC-direct | raw 7D action | none |
| BC-impedance | target/gain adapter | none |
| PhaseForge-direct | raw 7D action | MoE |
| PhaseForge-impedance | target/gain adapter | MoE |
| IS-PhaseForge full | target/gain adapter + SupCon + margin | MoE |

This prevents confounding architecture with action representation.

---

## 8. Component 5: Local contraction / stability regularization

A memoryless policy cannot rely on history to recover from drift. Therefore each expert should be biased toward locally stabilizing behavior.

### 8.1 Contraction intuition

For an impedance-style expert:

\[
u_k(t)
=
\kappa_k
\left(
T_k(x_t) - y_t
\right)
\]

the derivative with respect to task state is approximately:

\[
\frac{\partial u_k}{\partial y}
=
\kappa_k
\left(
\frac{\partial T_k}{\partial y}
-
I
\right)
\]

If the target map \(T_k\) changes slowly with respect to \(y\), then:

\[
\frac{\partial T_k}{\partial y}
\]

has norm less than one, and the expert behaves like a contractive controller.

### 8.2 Practical regularization

Use a Lipschitz penalty on the target map:

\[
\mathcal{L}_{\text{lip}}
=
\mathbb{E}_{(x_i,x_j)}
\left[
\max\left(
0,
\frac{
\|T_k(x_i)-T_k(x_j)\|_2
}{
\|y_i-y_j\|_2 + \epsilon
}
-
\rho
\right)^2
\right]
\]

where:

- pairs \((x_i,x_j)\) are sampled from the same regime or local neighborhood,
- \(\rho < 1\), e.g. \(0.7\) or \(0.8\).

This encourages:

\[
\left\|
\frac{\partial T_k}{\partial y}
\right\|
\le \rho < 1
\]

which makes the expert locally contractive under the simplified feedback model.

### 8.3 Important caveat

This is a **stability-inducing regularizer**, not a universal closed-loop stability proof under robosuite’s internal controller. A formal guarantee would require direct impedance/torque control or a validated model of the low-level controller.

The report should therefore state:

> The contraction term provides a local stabilizing bias and a useful diagnostic. It does not by itself certify global closed-loop stability in robosuite.

This avoids overclaiming.

---

## 9. Full Stage 2 objective

The final Stage 2 loss should be:

\[
\mathcal{L}_{\text{Stage 2}}
=
\mathcal{L}_{\text{action}}
+
\lambda_{\text{margin}}
\mathcal{L}_{\text{margin}}
+
\lambda_{\text{lip}}
\mathcal{L}_{\text{lip}}
+
\lambda_{\text{gain}}
\mathcal{L}_{\text{gain}}
+
\lambda_{\text{bal}}
\mathcal{L}_{\text{bal}}
\]

where:

### Action loss

\[
\mathcal{L}_{\text{action}}
=
\|a_{\text{pred}} - a_{\text{demo}}\|^2
\]

computed after the impedance action adapter.

### Margin loss

\[
\mathcal{L}_{\text{margin}}
=
\sum_{j \neq y}
\max(0, m - (d_j - d_y))
\]

### Lipschitz/contraction loss

\[
\mathcal{L}_{\text{lip}}
=
\max(0, \text{Lip}(T_k) - \rho)^2
\]

### Gain regularization

\[
\mathcal{L}_{\text{gain}}
=
\mathbb{E}
\left[
\|\kappa - \kappa_{\text{nominal}}\|^2
\right]
\]

or bound penalties to prevent pathological stiffness.

### Balance loss

Use a very small balance term only to prevent dead experts:

\[
\lambda_{\text{bal}} \ll 1
\]

It should not override the discovered regime structure.

---

## 10. Deployment contract

At every environment step:

```text
1. receive x_t
2. normalize x_t
3. compute z_t = E(x_t)
4. compute distances d_k = ||z_t - c_k||
5. select expert k* = argmin_k d_k
6. expert k* predicts target T_k* and gains κ_k*
7. compute task error e = TaskError(T_k*, y_t)
8. compute u = κ_k* ⊙ e
9. map u to environment action a_t
10. send a_t to environment
```

No hidden state is maintained.

No future information is used.

No oracle regime label is used.

No sticky post-processing is used.

---

## 11. Required rollout tracing

The current logs are insufficient. The revised architecture must produce per-step traces.

Minimum trace fields:

```text
episode_id
case_id
timestep
success/timeout/failure reason
raw observation summary
normalized state norm
task variables y_t
latent z_t norm
distance to each regime prototype
selected expert
top-2 expert
router margin
router entropy, if soft weights are logged
expert target T_k
expert gains κ_k
task error e
pre-clip command u
final clipped action a
nearest training-state distance / OOD score
expert disagreement, if top-2 is evaluated
Jacobian/Lipschitz diagnostic, sampled periodically
termination reason
```

### 11.1 Failure taxonomy

The traces should support classifying failures into:

| Failure type | Evidence |
|---|---|
| routing ambiguity | small router margin, high entropy |
| expert conflict | large top-2 action disagreement |
| OOD drift | large nearest-training-state distance |
| action saturation | command clipped persistently |
| gain collapse | expert gains near zero |
| target chasing | target continually moves away from state |
| reset geometry | same reset cases fail across methods |
| controller limit | commanded action valid but robot does not move as expected |

This is essential for causal interpretation.

---

## 12. Experiment sequence

The experiments should be incremental and controlled. Do not introduce all components at once.

### Stage A — Reproducibility and instrumentation

1. Run all baselines and PhaseForge variants from the same commit.
2. Freeze the evaluation bank, reset seed, horizon, and success metric.
3. Add full rollout tracing before architecture changes.
4. Preserve all discovered regime artifacts in the run directory.

### Stage B — Regime discovery and observability

Compare candidate regime definitions:

| Candidate | Source |
|---|---|
| current SLDS regimes | transition features |
| topological PELT regimes | task geometry/contact |
| merged regimes | observability-driven merging |
| K-sweeps | evidence-based K selection |

Metrics:

- predictability from \(x_t\),
- macro-F1,
- confusion matrix,
- regime occupancy,
- duration distribution,
- transition matrix,
- action residual reduction,
- stability across discovery seeds.

Stop criterion: if no regime definition is observable from \(x_t\), then memoryless dynamic routing is not a valid solution for this task under the current state space.

### Stage C — Minimal routing fix

Test whether routing improvements alone help, without changing the action head.

Methods:

1. BC-direct.
2. Current dynamic PhaseForge.
3. PhaseForge with topological regimes + cross-entropy.
4. PhaseForge with topological regimes + SupCon.
5. PhaseForge with topological regimes + SupCon + hard top-1 routing.
6. PhaseForge with topological regimes + SupCon + margin routing.

This isolates discovery and routing from action representation.

### Stage D — Impedance action parameterization

Introduce the impedance-style action adapter.

Methods:

1. BC-direct.
2. BC-impedance.
3. PhaseForge-direct + SupCon + margin.
4. PhaseForge-impedance + SupCon + margin.
5. Full IS-PhaseForge with contraction/lip regularization.

This isolates the effect of the expert action parameterization.

### Stage E — Ablations

Test one variable at a time:

| Ablation | Purpose |
|---|---|
| SupCon off / CE on | test contrastive alignment |
| margin off | test boundary chattering hypothesis |
| top-1 vs top-2 | test action blending |
| impedance off | test action parameterization |
| contraction off | test stability regularization |
| K sweep | test regime granularity |
| balance coefficient sweep | test utilization pressure |
| expert initialization | test warm-start dependence |

Use one seed for screening and three seeds for surviving configurations.

### Stage F — Confirmation

1. Select checkpoints using a validation reset bank.
2. Evaluate once on the final Can bank.
3. If Can improves decisively, confirm on Square.
4. Use Lift only as a regression check.

---

## 13. Expected interpretations

| Result | Interpretation |
|---|---|
| Topological regimes are predictable from \(x_t\), but rollout does not improve | Discovery is observable but not control-sufficient |
| SupCon + margin improves routing metrics but not success | Routing is no longer the bottleneck; action representation or controller dominates |
| Impedance experts improve over direct experts | raw action averaging was a real bottleneck |
| Top-1 improves over top-2 | destructive action blending was a real bottleneck |
| Margin loss reduces boundary chattering in traces | spatial hysteresis was missing |
| Contraction regularization improves recovery from perturbations | local stability bias helps compounding error |
| Same reset cases fail across all methods | reset geometry/controller/data coverage dominates |
| Regimes are not observable from \(x_t\) | memoryless dynamic routing is insufficient under the current state contract |
| IS-PhaseForge beats BC-impedance and PhaseForge-direct | the full architecture is justified |

---

## 14. Risks and mitigations

### Risk 1: PELT over-segments trajectories

**Mitigation:**
- increase penalty,
- enforce minimum segment length,
- merge segments with similar task prototypes,
- select \(K\) using observability and action explanation.

### Risk 2: Discovered regimes are still aliased

**Mitigation:**
- merge confused regimes,
- add instantaneous observable features if available,
- if impossible, report that memoryless routing is insufficient for this state space.

### Risk 3: Impedance adapter is incompatible with robosuite’s controller

**Mitigation:**
- implement the adapter as a command-shaping layer over the existing 7D action,
- use bounded gains,
- compare against BC-impedance to keep the test fair,
- if necessary, fall back to an affine/direct feedback action head.

### Risk 4: Hard routing creates discontinuous actions

**Mitigation:**
- train large spatial margins,
- place boundaries in low-density regions,
- optionally test top-2 impedance blending only in a narrow margin band,
- trace action jumps at switches.

### Risk 5: Contraction regularization conflicts with fast demonstrated motions

**Mitigation:**
- reduce \(\lambda_{\text{lip}}\),
- apply contraction only within regimes,
- use per-regime gain scales,
- monitor action loss versus contraction loss tradeoff.

### Risk 6: SupCon collapses useful within-regime variation

**Mitigation:**
- keep action loss active during Stage 1,
- monitor action MSE on validation,
- use moderate temperature,
- avoid excessive negative pressure,
- evaluate downstream action performance, not only clustering metrics.

---

## 15. Final answers to the original review questions

### 1. Should the main PhaseForge contribution be reformulated as a temporal dynamic MoE?

**No.** Not as the primary contribution.

The final contribution should be:

> **a memoryless, observation-consistent, switched impedance MoE for dynamic manipulation.**

Temporal models may be used only as external diagnostics if needed.

### 2. Is offline SLDS discovery acceptable?

Only as a diagnostic.

It is not acceptable as the primary router supervision signal unless the resulting regimes are demonstrably observable from \(x_t\). Otherwise, the router is being trained to predict inaccessible information.

The final method should use observation-consistent topological/contact regime discovery.

### 3. Should regimes be selected by held-out predictive evidence?

Yes.

The number of regimes should be selected by:

- observability,
- action explanation,
- stability,
- held-out predictive usefulness,
- simplicity.

It should not be fixed to six merely because the rule-based phase heuristic has six labels.

### 4. Should temporal BC be the first control baseline?

Not as the main direction, because temporal history is rejected for the proposed method.

However, a temporal BC baseline may be run separately as a diagnostic to determine whether the task itself requires memory under the given state space. If temporal BC succeeds while all memoryless methods fail, that is a valid negative result about memoryless routing.

### 5. Are Can-only diagnostics sufficient before Square?

Yes.

Can is the correct testbed because it contains contact transitions, re-approach, grasp/release structure, and compounding error. Square should be used only after Can diagnostics confirm the architecture.

---

## 16. Final recommendation

Proceed with the following final research direction:

1. **Keep the policy strictly memoryless.**
2. **Replace transition-only SLDS routing labels with observation-consistent topological/contact regimes.**
3. **Add an observability audit before router training.**
4. **Use SupCon to make regimes geometrically separable in latent space.**
5. **Use large-margin prototype routing with hard top-1 inference.**
6. **Replace direct action averaging with impedance/target-parameterized experts.**
7. **Add contraction/Lipschitz regularization as a local stability bias.**
8. **Instrument rollouts with full causal traces.**
9. **Run action-matched baselines, especially BC-impedance.**
10. **Do not expand to Square until Can diagnostics confirm the architecture.**

This is the cleanest non-temporal version of PhaseForge. It tests whether the real bottleneck is not the absence of history, but the current architecture’s failure to make regimes observable, routing stable, and expert actions physically composable.