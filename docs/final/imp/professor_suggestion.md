# Final Triple-Checked Review and Combined Solution  
**Subject:** PhaseForge Impedance Synthesis & Contraction Stability — full review, solution space, and final recommended path  

---

## 0. High-level verdict after careful review

Your report is directionally correct on several important points:

1. **Commit `c56275d` should not be reverted.**  
   It restored macroscopic physical execution. Reverting it would return the system to the Version 2 paralysis regime.

2. **The current 0% success is not primarily a routing failure.**  
   Routing NMI is high (`0.8941`), switch rate is low, and macroscopic phase completion is high. The router is no longer the main bottleneck.

3. **The immediate bottleneck is terminal precision.**  
   The robot reaches the correct macroscopic sequence but fails the narrow Robosuite success predicate at release/placement.

4. **The current action MSE plateau at ~0.123 is a real architectural/objective conflict.**  
   The combination of frozen SupCon encoder, randomly initialized impedance heads, action adapter scaling, and the current Lipschitz/contraction loss creates an optimization problem that cannot recover the baseline-level precision needed for the bin predicate.

However, the report still contains several likely mistakes or under-checked assumptions. The most important one is this:

> **The current Lipschitz/contraction loss is not merely badly tuned. It is mathematically mismatched to the operational-space behavioral cloning objective.**

Relaxing `rho` from `0.8` to `1.2` may help partially, but it does not fix the deeper issue. The current formulation penalizes the very structure that a tracking policy may need: a moving target equilibrium that can lead the current state. In operational-space delta-position control, the target may need to move faster than the current state during dynamic segments. A target-Lipschitz constraint with `rho < 1` can directly oppose accurate imitation.

The final solution must therefore combine several corrections, not one hyperparameter change.

---

# 1. Parameter-level audit: likely mistakes and required checks

Before proposing the final architecture, the following parameters and design choices must be audited. These are not optional. Any one of them can explain the precision collapse.

## 1.1 Action adapter scaling may still be dimensionally inconsistent

Your formula is:

\[
a = \tanh\left( \frac{\mathbf{s} \odot K \odot (\mathbf{x}_{\text{target}} - \mathbf{y}_{\text{eef}})}{\tau} \right)
\]

with:

\[
\mathbf{s} = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.04]
\]

In Robosuite/robomimic operational-space control, the policy action is usually a **normalized command** in `[-1, 1]`, and the low-level controller multiplies it by a physical limit. Therefore, if the expert produces a desired displacement in meters, the normalized action should usually be:

\[
a = \text{clip}\left(
\frac{\Delta x_{\text{desired}}}{\Delta x_{\max}},
-1,
1
\right)
\]

or, with soft clipping:

\[
a = \tanh\left(
\frac{\Delta x_{\text{desired}}}{\tau \Delta x_{\max}}
\right)
\]

That is, the physical limit should normally appear in the **denominator**, not as a multiplicative factor inside the numerator.

If the code truly multiplies by `s = 0.05`, then a desired displacement of `0.05 m` with unit gain gives approximately:

\[
\tanh(0.05) \approx 0.05
\]

which is only 5% of the normalized action range. That would cause sluggish motion unless the learned `K` becomes extremely large.

You report that Version 3 restored motion, so either:

1. the implemented code differs from the formula in the report,
2. the learned stiffnesses became very large,
3. the scaling was corrected in another way, or
4. the environment action interpretation is different from the usual Robosuite convention.

This must be verified with a unit test.

### Required adapter unit test

For each action dimension:

| Desired physical displacement | Expected normalized action |
|---:|---:|
| `0` | `0` |
| half maximum displacement | approximately `0.5` |
| full maximum displacement | approximately `1.0` |
| negative full maximum displacement | approximately `-1.0` |

If this mapping fails, terminal precision cannot be trusted.

---

## 1.2 The Lipschitz/contraction loss is likely formulated incorrectly

Your current loss is:

\[
\mathcal{L}_{\text{lip}}
=
\frac{1}{|P|}
\sum_{(i,j) \in P}
\max\left(
0,
\frac{
\|\mathbf{x}_{\text{target}}^{(i)}[0:3] - \mathbf{x}_{\text{target}}^{(j)}[0:3]\|_2
}{
\|\mathbf{y}^{(i)}[0:3] - \mathbf{y}^{(j)}[0:3]\|_2
}
-
\rho
\right)^2
\]

with:

\[
\rho = 0.8,
\quad
\lambda_{\text{lip}} = 0.1
\]

This is not a proper contraction loss for behavioral cloning of dynamic trajectories.

### Why it is problematic

In a reaching/impedance controller, the expert predicts a target equilibrium:

\[
\mathbf{x}_{\text{target}}
\]

and the command is roughly:

\[
\Delta x = K(\mathbf{x}_{\text{target}} - \mathbf{y})
\]

For dynamic tracking, the target may need to move faster than the current end-effector state. Therefore, for two nearby demonstration states, the targets can differ by more than the states themselves:

\[
\frac{
\|\Delta \mathbf{x}_{\text{target}}\|
}{
\|\Delta \mathbf{y}\|
}
> 1
\]

This is not necessarily instability. It may be exactly how the demonstration accelerates toward the bin.

Penalizing this ratio with `rho = 0.8` forces the target map to be overly smooth and slow. It directly fights the action MSE objective, especially in transport and placement segments where decisive motion is required.

### Additional numerical issue

The denominator:

\[
\|\mathbf{y}^{(i)}[0:3] - \mathbf{y}^{(j)}[0:3]\|_2
\]

can become extremely small. If pairs are not filtered by a minimum distance, the ratio can explode and produce large, noisy gradients.

Your reported `loss_lip ≈ 0.710` and gradient norm `≈ 16.8` strongly suggest that this term is dominant and harmful.

### Verdict

The current Lipschitz loss should be **disabled for recovery**.

It should not be merely relaxed. It should be replaced later, if needed, by a corrected stability or smoothness regularizer.

---

## 1.3 Freezing the SupCon encoder while training random impedance heads is a major mistake

The report says:

```text
freeze_encoder: true
```

and the Stage 2 impedance experts are initialized from scratch.

This is a severe optimization constraint.

The SupCon encoder is optimized to cluster phases. That is useful for routing, but it may discard fine-grained intra-phase geometric information needed for precise control. If the encoder is frozen and the impedance heads are random, the system must reconstruct precise operational-space commands from a latent space that was not trained to preserve them.

This explains why Stage 2 action MSE plateaus around `0.123`.

### Verdict

For precision recovery, the encoder must be at least partially unfrozen, or the experts must receive raw task-state information directly, not only the SupCon latent.

---

## 1.4 SupCon may have collapsed intra-phase geometric variance

Your Stage 1 results show:

- Stage 1 action validation MSE: `0.0365`
- baseline Stage 1 action validation MSE: `0.0301`
- SupCon loss: `4.88`
- routing NMI later: `0.8941`

The routing representation improved, but Stage 1 action performance degraded. This is a warning sign.

SupCon can make phases separable while reducing the continuous geometric detail needed for precise action prediction.

### Verdict

The final architecture should separate:

1. a **phase/routing representation**, and  
2. a **precision/control representation**.

A single frozen SupCon latent should not be forced to serve both purposes.

---

## 1.5 Random impedance expert initialization is inferior to drop-upcycling

The baseline used drop-upcycling from a working Stage 1 action head:

\[
W_e \leftarrow \text{Dropout}(W_{\text{stage1}}, p=0.5)
\]

This allowed Stage 2 to start near the baseline optimum.

The impedance model used random expert initialization. That means Stage 2 had to learn both:

1. how to map the latent to precise actions, and  
2. how to specialize experts,

from scratch, while the encoder was frozen and multiple regularizers were active.

This is too many simultaneous constraints.

### Verdict

The experts must be warm-started from the Stage 1 policy. For impedance experts, use inverse-consistent initialization so that the initial impedance policy reproduces the Stage 1 direct action.

---

## 1.6 The multi-objective Stage 2 loss is over-constrained at initialization

Your Stage 2 loss is:

\[
\mathcal{L}
=
\mathcal{L}_{\text{action}}
+
0.1 \mathcal{L}_{\text{lip}}
+
1.0 \mathcal{L}_{\text{margin}}
+
10^{-4} \mathcal{L}_{\text{gain}}
+
10^{-4} \mathcal{L}_{\text{balance}}
\]

With:

- `loss_lip ≈ 0.710`
- `loss_margin ≈ 0.088`
- `loss_action ≈ 0.123`

the effective contribution from the Lipschitz term is approximately:

\[
0.1 \times 0.710 = 0.071
\]

This is more than half the action loss magnitude, and its gradient norm is reported as much larger. The optimizer is therefore not primarily minimizing action error.

### Verdict

Stage 2 must be staged:

1. recover action imitation first,
2. then add routing/margin constraints,
3. then add stability regularizers only if needed.

Do not train all objectives at full strength from scratch.

---

## 1.7 Terminal phase granularity may be insufficient

Your six phases are:

```text
0 Approach
1 Pre-grasp
2 Grasp
3 Transport
4 Place
5 Retract
```

The success predicate is dominated by the final part of Phase 4 and the transition to Phase 5:

- object must be inside the bin,
- gripper must be released,
- end-effector must move away,
- object must not collide with bin rim,
- release height and timing matter.

A single `Place` regime may contain multiple distinct behaviors:

1. coarse bin approach,
2. fine object alignment,
3. lower/release preparation,
4. gripper release,
5. initial retract.

If one expert handles all of these, it may average incompatible terminal actions.

### Verdict

Split the terminal regime into finer observable subphases, or create a dedicated terminal precision expert.

---

## 1.8 The state must contain explicit instantaneous bin/object placement geometry

The policy is memoryless. Therefore, if precise placement is required, the current state must contain enough information to compute placement error.

At minimum, the policy should have instantaneous access to:

```text
object position
object orientation, if relevant
bin/target position
object-to-bin vector
EEF-to-object vector
EEF-to-bin vector
gripper aperture
gripper open/close state
contact or proximity proxies, if available
```

This is not temporal history. It is allowed under the memoryless contract.

If the bin target is randomized and not observable, no memoryless policy can reliably solve placement. If the bin is fixed, the policy can implicitly learn it, but explicit relative features are much safer.

### Verdict

Add instantaneous task-geometry features if they are available or can be derived from the environment observation.

---

## 1.9 Per-dimension action MSE has not been reported

The global action MSE of `0.1229` may hide a severe problem in one or two dimensions.

For example:

- translation MSE may be acceptable,
- rotation MSE may be high,
- gripper MSE may be very high,
- the gripper channel may dominate failure.

The gripper is especially important because success requires release. If the impedance formulation smooths the gripper command, the policy may fail even if the arm trajectory is good.

### Verdict

Report per-dimension MSE and phase-wise MSE before drawing final conclusions.

---

## 1.10 The success predicate must be traced directly

The report infers terminal failure from action MSE and approximate release displacement. This is plausible but not yet proven.

You should log:

```text
not_in_bin
objects_in_bins
r_reach
final object position
bin low/high bounds
object-bin x/y/z error
object position at gripper opening
object position 10 steps after release
EEF distance at release
whether the can contacts bin wall/rim
```

Without this, we cannot know whether the failure is:

1. release offset,
2. release height,
3. object tilt,
4. bin collision,
5. gripper not fully open,
6. premature release,
7. late release,
8. action scaling error.

### Verdict

Add success-predicate tracing before the next large training run.

---

# 2. Complete actionable solution space

Below is the full solution space I can identify from the evidence. I have grouped the solutions by failure mechanism. Each solution is analyzed before combining the best parts into the final recommendation.

Legend:

- **Use now**: include in the immediate recovery plan.
- **Conditional**: use after a diagnostic or after baseline recovery.
- **Diagnostic**: not a training solution, but required for interpretation.
- **Reject / deprioritize**: not recommended as the primary fix.

---

## 2.1 Solutions related to the contraction/Lipschitz loss

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| L1 | Disable current Lipschitz loss entirely | Removes the large conflicting gradient | Immediately allows action MSE recovery | Loses stability bias | **Use now** |
| L2 | Relax `rho` from `0.8` to `1.2` | Allows target maps that lead the state | Simple | Still uses a questionable target-ratio constraint | Deprioritize |
| L3 | Schedule `lambda_lip = 0` initially, anneal later | Lets action learning converge first | Safe | If the loss is wrong, annealing still hurts | **Use now if any lip loss is kept** |
| L4 | Apply contraction only to Phase 2/4/5 | Stability only where contact/precision matters | More physically motivated | Still needs correct formulation | Conditional |
| L5 | Replace target-ratio loss with action-field smoothness | Penalizes erratic action variation, not target lead | Safer for imitation | Does not guarantee stability | Conditional |
| L6 | Replace with discrete closed-loop contraction | Penalizes predicted next-state error reduction | More theoretically correct | Requires a forward model or offline next-state use | Conditional |
| L7 | Use spectral norm constraints on expert layers | Controls Lipschitz constant of expert mapping | Stable training | May still reduce precision if too strong | Conditional |
| L8 | Use phase-specific Lyapunov/energy loss | Closer to switched control theory | Principled | Hard to implement correctly in delta-position control | Later research |

### Analysis

The current Lipschitz loss is the most obvious optimization conflict. The fastest path to recovery is to disable it (`L1`). Relaxing `rho` (`L2`) is not sufficient because the loss itself is not properly aligned with the control objective.

The best replacement is not a target-ratio loss. It is either:

1. a mild action-field smoothness regularizer, or  
2. a phase-specific terminal precision/stability objective.

---

## 2.2 Solutions related to encoder freezing and representation

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| R1 | Fully unfreeze encoder in Stage 2 | Allows latent to adapt to impedance/action precision | High chance of MSE recovery | May damage routing clusters if LR too high | **Use now** |
| R2 | Partially unfreeze last encoder layers | Preserves early features, adapts late geometry | Safer | May still be too weak if bottleneck is early | **Use now** |
| R3 | Keep encoder frozen but add state reconstruction head | Forces latent to preserve geometry | Useful diagnostic | Does not help if Stage 2 cannot change encoder | Conditional |
| R4 | Dual-stream encoder: route latent + control latent | Separates clustering from precision | Strong architectural fix | More implementation complexity | **Use in final architecture** |
| R5 | Reduce SupCon weight in Stage 1 | Prevents phase clustering from destroying action info | Simple | May reduce routing NMI | **Use now if Stage1 MSE remains worse than baseline** |
| R6 | Add action-conditioned contrastive positives | Clusters states with similar actions, not only phase labels | Better alignment with control | More complex | Conditional |
| R7 | Pass raw task state directly to experts | Experts do not depend only on SupCon latent | Very important for precision | Needs careful normalization | **Use now** |

### Analysis

The frozen encoder is a structural bottleneck. The final architecture should not rely on a single SupCon latent for both routing and precision.

The best combination is:

- `R1/R2`: unfreeze or partially unfreeze the encoder,
- `R7`: give experts direct access to task-state variables,
- `R4`: in the final architecture, separate routing and control streams.

---

## 2.3 Solutions related to expert initialization

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| I1 | Drop-upcycling Stage 1 direct action head | Experts start from working policy | Recovers baseline quickly | Less physically structured | **Use now** |
| I2 | Initialize impedance target branch from Stage 1 action head | Impedance experts reproduce direct actions initially | Preserves impedance structure and precision | Requires inverse mapping | **Use now** |
| I3 | Initialize stiffness `K` from demonstration statistics | Gives physically meaningful gains | Reduces random search | Needs robust estimation | **Use now** |
| I4 | Residual impedance branch initialized to zero | Direct action path guarantees baseline behavior | Very safe | Slightly less pure impedance | **Use now** |
| I5 | Random expert initialization | No bias from Stage 1 | None relevant | Failed empirically | **Reject for recovery** |
| I6 | Teacher-student distillation from baseline direct MoE | Transfers working behavior | Useful if architecture changes | Extra training stage | Conditional |

### Analysis

Random expert initialization under a frozen encoder is one of the main reasons the impedance model cannot recover precision.

The safest solution is a **residual architecture**:

\[
a = a_{\text{direct}} + a_{\text{impedance residual}}
\]

with the impedance residual initialized to zero. This guarantees that the model can first recover the baseline behavior before adding structure.

---

## 2.4 Solutions related to action parameterization

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| A1 | Predict direct normalized action as primary output | Simplest exact imitation path | Highest chance of recovering MSE | Less physically structured | **Use now** |
| A2 | Predict target equilibrium and stiffness | Physically meaningful | Good for blending/contact | Harder to optimize precisely | Conditional |
| A3 | Predict relative target offset instead of absolute target | `x_target = y_eef + delta` | Easier to learn, avoids absolute coordinate regression | Still needs correct scaling | **Use if impedance is kept** |
| A4 | Use direct action + learned compliance offset | Combines precision and robustness | Strong final architecture | More components | **Use in final architecture** |
| A5 | Separate gripper head | Gripper is discrete/contact-critical | Prevents impedance smoothing of gripper | Slight architectural asymmetry | **Use now** |
| A6 | Per-dimension action scaling | Prevents mismatched action ranges | Necessary | Requires calibration | **Use now** |
| A7 | Per-dimension loss weighting | Focuses learning on critical dimensions | Helpful if gripper/translation dominate | Needs tuning | **Use now** |
| A8 | Replace tanh soft clip with linear clip in precision region | Avoids gradient suppression near saturation | Better precision | Less smooth | Conditional |
| A9 | Learn the action adapter from data | Adapter can correct controller nonlinearities | Flexible | Risk of overfitting | Later |

### Analysis

The pure impedance formulation is likely too constrained for precise delta-position imitation unless it is carefully initialized and scaled.

The best immediate path is:

1. recover direct action performance,
2. add impedance as a residual or auxiliary compliance term,
3. separate the gripper channel.

---

## 2.5 Solutions related to routing

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| G1 | Keep prototype router | Already gives high NMI | Stable | May not be action-optimal | Keep |
| G2 | Train router with action loss via straight-through Gumbel | Routing optimized for control, not only labels | Better alignment | More complex | Conditional |
| G3 | Split Place/Release into finer regimes | Dedicated terminal experts | High value for success predicate | More labels/experts | **Use now** |
| G4 | Add dedicated terminal precision expert | Explicitly handles bin alignment/release | Directly targets failure | Requires regime detection | **Use in final architecture** |
| G5 | Reduce margin loss weight during recovery | Prevents routing objective from harming action | Simple | May reduce phase separation temporarily | **Use now** |
| G6 | Use top-1 hard routing | Avoids destructive blending | Safe | Discontinuities possible | Keep |
| G7 | Use top-2 impedance blending | Smooth transitions | Physically valid only with impedance | Can mask routing ambiguity | Conditional |
| G8 | Add boundary spatial margin | Reduces chattering | Good | Should not dominate action loss | Conditional |

### Analysis

Routing is not the main current failure, but terminal routing granularity is important. The final architecture should include finer terminal regimes or a dedicated precision expert.

---

## 2.6 Solutions related to task state and terminal precision

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| T1 | Add object-to-bin relative features | Gives policy explicit placement error | Very high value | Requires observable bin pose | **Use now if available** |
| T2 | Add EEF-to-bin relative features | Helps terminal alignment | High value | Same as above | **Use now if available** |
| T3 | Add object-to-EEF relative features | Helps grasp/release geometry | Useful | May duplicate existing state | Conditional |
| T4 | Add bin bounding-box distance features | Directly encodes success geometry | Useful | May overfit predicate | Conditional |
| T5 | Object-centric action loss | Penalizes object trajectory error, not only robot action | Better aligned with success | Needs object prediction | Conditional |
| T6 | Release-event weighting | Increases loss near release steps | Directly targets failure | Needs release detection | **Use now** |
| T7 | Terminal-phase oversampling | Ensures enough terminal training signal | Simple | Could overfit terminal | **Use now** |
| T8 | Analytic placement primitive | Hard-coded final alignment/release | Can solve precision | Reduces learned contribution | Diagnostic / hybrid only |
| T9 | Learned release compensation | Predicts post-release object offset | Can fix bin-rim failure | Needs forward model | Later |

### Analysis

The success predicate is object-centric, but the action loss is robot-action-centric. This mismatch is likely critical.

The final architecture should include object/bin relative features and terminal-weighted learning.

---

## 2.7 Solutions related to training schedule and evaluation

| ID | Solution | Mechanism | Strength | Weakness / Risk | Verdict |
|---|---|---|---|---|---|
| S1 | Stage 2 action-only recovery phase | Recovers baseline MSE before regularization | Essential | None | **Use now** |
| S2 | Add regularizers only after MSE gate | Prevents multi-loss conflict | Essential | None | **Use now** |
| S3 | Use validation gates before rollouts | Avoids wasted compute | Essential | Needs good offline metrics | **Use now** |
| S4 | Use phase-wise validation MSE for checkpointing | Better alignment with terminal failure | Helpful | Needs phase labels | **Use now** |
| S5 | Use small dev rollout bank for checkpoint selection | Better than action MSE alone | Expensive but valuable | Must keep final eval separate | Conditional |
| S6 | Run per-dimension MSE diagnostics | Finds hidden gripper/rotation failure | Essential | None | **Use now** |
| S7 | Run success-predicate tracing | Finds true terminal failure mode | Essential | None | **Use now** |
| S8 | Use gradient surgery / PCGrad for multi-loss | Reduces gradient conflict | Advanced | May not fix wrong formulation | Later |

### Analysis

The next experiment should not be another full architecture change without gates. The system must first recover baseline offline precision.

---

# 3. Combined final solution

After analyzing the solution space, the best final path is not a single fix. It is a combined architecture and training protocol.

I recommend the following final architecture:

> **Precision-Residual Dual-Stream PhaseForge**  
> A memoryless PhaseForge variant with separate routing and control representations, warm-started direct action experts, optional residual impedance, terminal precision features, and disabled/corrected stability regularization during recovery.

This combines the best parts of the candidate solutions:

- direct action precision from baseline PhaseForge,
- impedance-style robustness from the IS formulation,
- routing stability from SupCon/prototype routing,
- terminal precision from object/bin-relative features and phase weighting,
- optimization safety from warm-starting and staged losses.

---

# 4. Final architecture: Precision-Residual Dual-Stream PhaseForge

## 4.1 Memoryless contract

The policy remains memoryless:

\[
a_t = \pi(x_t)
\]

No recurrent state. No observation history. No action chunking. No test-time oracle.

---

## 4.2 Dual-stream representation

Use two representation streams.

### Route stream

\[
z_r = E_r(x_t)
\]

Purpose:

- phase clustering,
- router stability,
- regime separation.

Training signals:

- SupCon,
- prototype margin,
- phase/regime labels.

This stream is used only for routing.

### Control stream

\[
z_c = E_c(x_t)
\]

Purpose:

- precise action prediction,
- geometric detail,
- object/bin alignment.

Training signals:

- action MSE,
- optional state/task-state reconstruction,
- optional object next-state prediction.

This stream is used by experts.

The experts should also receive raw task-state variables directly:

\[
y_{\text{task}} = \psi(x_t)
\]

where `y_task` includes:

```text
EEF position
EEF orientation representation
gripper state
object position/orientation features
object-to-bin vector
EEF-to-bin vector
object-to-EEF vector
```

The expert input is therefore:

\[
[z_c,\ y_{\text{task}},\ z_r \text{ optional}]
\]

This prevents the precision head from depending entirely on a phase-clustered latent.

---

## 4.3 Router

Keep the prototype router, but use it only for routing:

\[
d_k = \|z_r - c_k\|_2
\]

\[
k^\* = \arg\min_k d_k
\]

Use top-1 hard routing as the default.

The margin loss should be small during recovery:

\[
\lambda_{\text{margin}} \in [0.01, 0.1]
\]

Do not use `lambda_margin = 1.0` while trying to recover action precision.

If routing is already good, margin loss can be annealed later.

---

## 4.4 Expert formulation: direct action plus residual impedance

Do not force the expert to produce only `(x_target, K, D)` at first.

Instead, use a residual formulation.

Each expert outputs:

\[
a^{\text{base}}_k = h^{\text{base}}_k(z_c, y_{\text{task}})
\]

and optionally:

\[
\delta_k = h^{\delta}_k(z_c, y_{\text{task}})
\]

\[
\kappa_k = \text{softplus}(h^{\kappa}_k(z_c, y_{\text{task}}))
\]

The residual impedance command is:

\[
a^{\text{imp}}_k
=
\text{clip}
\left(
\frac{\kappa_k \odot \delta_k}{a_{\max}},
-1,
1
\right)
\]

The final action is:

\[
a_k
=
\text{clip}
\left(
a^{\text{base}}_k
+
\beta a^{\text{imp}}_k,
-1,
1
\right)
\]

where:

\[
\beta
\]

is initially zero or very small.

### Why this is better

1. The direct base action can recover baseline precision.
2. The impedance residual can later add contact robustness.
3. The model is not forced to solve precision and impedance simultaneously.
4. If the residual branch is initialized to zero, the system starts equivalent to the warm-started direct policy.

---

## 4.5 Relative target parameterization

If you keep an impedance branch, do not predict absolute target position as the only output.

Use a relative offset:

\[
\delta_k = x_{\text{target}} - y_{\text{eef}}
\]

Then:

\[
a^{\text{imp}}
=
\text{clip}
\left(
\frac{\kappa \odot \delta}{a_{\max}},
-1,
1
\right)
\]

This is easier to learn and easier to warm-start.

For initialization:

\[
\delta^{(0)}
=
a^{\text{demo}}
\odot
a_{\max}
\]

if:

\[
\kappa^{(0)} = 1
\]

Then the impedance branch initially reproduces the demo action.

---

## 4.6 Gripper should be treated separately

The gripper channel should not be forced through the same smooth impedance formulation as translational motion.

Use a direct gripper head:

\[
a_{\text{grip}}
=
\tanh(g_{\text{grip}}(z_c, y_{\text{task}}))
\]

or a binary-style BCE objective if the dataset uses discrete open/close commands.

This is important because release failure may be caused by smoothed gripper commands.

---

## 4.7 Terminal precision features

Add instantaneous task-geometry features if available.

Recommended features:

```text
object_pos - bin_center
EEF_pos - bin_center
object_pos - EEF_pos
object_pos - bin_low
bin_high - object_pos
gripper aperture
gripper open/close command
binary flag: object near bin
binary flag: object in bin x/y range
binary flag: object above bin floor
```

These features are memoryless and directly relevant to success.

If the bin pose is fixed, encode it as a constant. If the bin pose is randomized, it must be observable.

---

## 4.8 Split terminal phases

Replace the single `Place` regime with finer terminal regimes if the labels are observable.

Suggested terminal split:

```text
PlaceApproach: coarse approach to bin
PlaceAlign: object near bin, fine alignment
PlaceRelease: gripper opening / release
Retract: EEF withdrawal
```

If PELT or rule labels cannot support this cleanly, use a dedicated terminal precision expert triggered by:

```text
phase == Place
object-bin distance < threshold
gripper starting to open
```

This expert can focus exclusively on the narrow success predicate.

---

# 5. Final loss design

## 5.1 Recovery Stage 2 loss

For immediate recovery, use:

\[
\mathcal{L}
=
\mathcal{L}_{\text{action weighted}}
+
\lambda_{\text{margin}}
\mathcal{L}_{\text{margin}}
\]

with:

\[
\lambda_{\text{margin}} \le 0.1
\]

and:

\[
\mathcal{L}_{\text{lip}} = 0
\]

\[
\mathcal{L}_{\text{gain}} = 0
\]

\[
\mathcal{L}_{\text{balance}} \approx 0
\]

unless experts are dead.

The immediate goal is:

\[
\text{Stage 2 validation action MSE} \le 0.032
\]

or at least no worse than baseline by more than 5%.

Do not run full 50-episode rollouts until this gate is passed.

---

## 5.2 Weighted action loss

Use phase-wise and dimension-wise weights.

Let:

\[
\mathcal{L}_{\text{action weighted}}
=
\frac{1}{B}
\sum_i
w_{\text{phase},i}
\| W_{\text{dim}} \odot (a_i - a_i^\*) \|^2
\]

Recommended initial weights:

```text
Approach:     1.0
Pre-grasp:    1.0
Grasp:        1.5
Transport:    1.0
Place:        3.0
Release:      4.0
Retract:      1.0
```

If Release is not a separate phase, up-weight the last part of Place.

For dimension weights, start by inspecting per-dimension MSE. If gripper error is high, increase gripper weight. If translation error dominates terminal failure, increase translation weight.

---

## 5.3 Optional object-centric auxiliary loss

If object state is available, add an auxiliary loss that predicts object motion or terminal object error.

For example:

\[
\mathcal{L}_{\text{obj}}
=
\| \hat{p}_{\text{obj},t+1} - p_{\text{obj},t+1} \|^2
\]

or, for terminal states:

\[
\mathcal{L}_{\text{bin}}
=
\| p_{\text{obj}} - p_{\text{bin}} \|^2
\]

near release.

This auxiliary loss should be small:

\[
\lambda_{\text{obj}} \in [0.01, 0.1]
\]

It is not required for recovery, but it aligns training with the success predicate.

---

## 5.4 Corrected stability regularizer, if needed later

Do not use the current target-ratio Lipschitz loss.

If a stability regularizer is needed after precision recovery, use one of the following.

### Option A: action smoothness

For nearby same-phase states:

\[
\mathcal{L}_{\text{smooth}}
=
\mathbb{E}_{i,j}
\left[
\frac{
\|a(x_i) - a(x_j)\|^2
}{
\|x_i - x_j\|^2 + \epsilon
}
\right]
\]

Use a very small weight:

\[
\lambda_{\text{smooth}} \in [10^{-5}, 10^{-3}]
\]

### Option B: terminal error reduction

For Place/Release only, use offline next-state information as an auxiliary signal:

\[
\mathcal{L}_{\text{terminal}}
=
\max(
0,
\|e_{\text{bin},t+1}\|
-
\|e_{\text{bin},t}\|
+
m
)
\]

where:

\[
e_{\text{bin}} = p_{\text{obj}} - p_{\text{bin}}
\]

This encourages terminal actions to reduce object-bin error.

### Option C: spectral norm constraint

Apply spectral norm or weight decay to expert layers to prevent excessive action sensitivity.

This is less aggressive than the current target-ratio loss.

---

# 6. Final training protocol

## Phase 0 — Diagnostic-only pass, no new training

Run these diagnostics on the existing Version 3 checkpoint and baseline checkpoint.

### 6.0.1 Per-dimension action MSE

Report:

```text
MSE position x/y/z
MSE rotation x/y/z
MSE gripper
```

Separately for:

```text
all phases
Place only
last 50 steps before release
release step
```

### 6.0.2 Phase-wise action MSE

Report MSE for each phase.

If Place/Release MSE is much higher than average, the terminal regime is underfit.

### 6.0.3 Success predicate trace

For failed rollouts, log:

```text
object position at release
object-bin x/y/z error
not_in_bin
r_reach
release step
final object position
whether object contacts bin wall
gripper value at release
```

This confirms whether the failure is really release precision.

### 6.0.4 Action adapter unit test

Verify:

```text
zero error -> zero action
full physical displacement -> action near ±1
gripper open/close -> action near ±1
no dimension accidentally attenuated by controller limits
```

If this fails, fix the adapter before any training.

---

## Phase 1 — Recover direct-action baseline inside the new architecture

Train a simplified version:

```text
dual-stream or single control encoder allowed
experts warm-started from Stage 1 direct action head
no impedance residual or residual initialized zero
no Lipschitz loss
no gain loss
no balance loss unless dead experts
small margin loss only if routing degrades
encoder unfrozen or partially unfrozen
```

Target gates:

| Metric | Gate |
|---|---:|
| global validation action MSE | ≤ `0.032` or within 5% of baseline |
| Place-phase action MSE | ≤ baseline Place MSE × 1.1 |
| Release-phase action MSE | ≤ baseline Release MSE × 1.1 |
| gripper MSE | ≤ baseline gripper MSE × 1.2 |

Do not run full rollouts until these gates are passed.

---

## Phase 2 — Add residual impedance safely

After Phase 1 passes:

1. add the impedance residual branch,
2. initialize residual output to zero,
3. keep direct action branch active,
4. train with action weighted loss,
5. ensure validation MSE does not degrade by more than 3–5%.

Target:

\[
\text{MSE}_{\text{residual}} \le 1.05 \times \text{MSE}_{\text{direct recovery}}
\]

If it degrades, reduce the residual learning rate or keep `beta = 0` longer.

---

## Phase 3 — Add terminal precision mechanisms

Add, in this order:

1. object-to-bin and EEF-to-bin features,
2. terminal phase weighting,
3. finer Place/Release regimes or terminal precision expert,
4. optional object-centric auxiliary loss.

Target:

- Place/Release MSE improves,
- offline predicted release error decreases,
- small dev rollout bank shows non-zero success.

---

## Phase 4 — Optional corrected stability regularizer

Only after Phases 1–3 succeed:

1. add a mild smoothness or terminal error-reduction regularizer,
2. use a tiny weight,
3. verify that Place/Release MSE does not degrade,
4. verify that routing does not degrade.

Do not reintroduce the original target-ratio Lipschitz loss unless it is reformulated and empirically validated.

---

# 7. Direct answers to your four decision points

## Decision 1: Contraction ratio `rho` and weight `lambda_lip`

**Recommendation:** Disable the current Lipschitz/contraction loss for recovery.

Do not merely change:

\[
\rho = 0.8 \rightarrow 1.2
\]

The current loss is structurally mismatched to dynamic operational-space imitation.

If a stability regularizer is reintroduced later:

1. apply it only to terminal/contact phases,
2. use a corrected formulation,
3. start with `lambda = 0`,
4. anneal to a tiny value,
5. stop if action MSE or release precision degrades.

Final answer:

> **Current lip loss: off. Future stability: phase-specific, corrected, and tiny.**

---

## Decision 2: Backbone fine-tuning / `freeze_encoder`

**Recommendation:** Unfreeze the encoder, at least partially.

A frozen SupCon encoder with randomly initialized impedance heads is not viable for precision recovery.

Recommended settings:

```text
freeze_encoder: false
encoder LR: 1e-5 to 3e-5
expert LR: 1e-4 to 3e-4
router LR: 1e-4
warmup: 2–5 epochs
gradient clipping: 1.0
```

If routing degrades, use:

```text
partial unfreeze: last encoder blocks only
or dual-stream encoder: route stream frozen/low LR, control stream trainable
```

Final answer:

> **Do not keep the encoder fully frozen. Use low-LR fine-tuning or dual-stream separation.**

---

## Decision 3: Expert initialization

**Recommendation:** Warm-start all experts from the Stage 1 policy.

Use one of the following:

### Option A: Drop-upcycling direct action head

Copy the Stage 1 direct action head into each expert with small perturbation.

This is the fastest recovery path.

### Option B: Inverse-consistent impedance initialization

If using impedance:

1. choose nominal stiffness `K_nom`,
2. set relative target offset:

\[
\delta^{(0)} = a_{\text{stage1}} \odot a_{\max} / K_{\text{nom}}
\]

3. initialize the target/offset head to predict this offset,
4. initialize residual impedance output to zero if using residual architecture.

Final answer:

> **Random scratch experts are rejected. Use warm-started direct experts or inverse-initialized impedance experts.**

---

## Decision 4: Operational space formulation vs direct action imitation

**Recommendation:** Use direct action prediction as the primary path during recovery.

Then add impedance as a residual or auxiliary compliance term.

The immediate hierarchy should be:

```text
1. direct action base policy
2. residual impedance/compliance
3. optional pure impedance only after residual version matches direct precision
```

Final answer:

> **Predict direct normalized action first. Use impedance as a residual or auxiliary structure, not as the only precision path.**

---

# 8. Final recommended experiment sequence

## Experiment 0: Diagnostic audit

No full training.

Deliverables:

1. per-dimension MSE,
2. per-phase MSE,
3. terminal object-bin error histogram,
4. success predicate trace,
5. action adapter unit test,
6. router trace at Place/Release,
7. expert output range and gain statistics.

Decision gate:

- If terminal translation/gripper error is dominant, proceed with terminal precision fixes.
- If adapter scaling fails, fix adapter before anything else.
- If routing fails at Place/Release, split terminal phases.

---

## Experiment 1: Direct warm-start recovery

Configuration:

```text
encoder: unfrozen or partially unfrozen
experts: warm-started direct action heads
action adapter: direct action or residual initialized zero
loss: weighted action MSE only
lip: disabled
gain: disabled
balance: disabled or 1e-6
margin: 0.01–0.1 only if needed
```

Gate:

```text
global validation MSE <= 0.032
Place/Release MSE near baseline
```

Interpretation:

- If this recovers MSE, the original failure was optimization/regularization/initialization.
- If this does not recover MSE, the problem is deeper: state representation, action adapter, or data/action space mismatch.

---

## Experiment 2: Add terminal precision features

Configuration:

```text
same as Experiment 1
add object-to-bin and EEF-to-bin features
add terminal phase weighting
oversample Place/Release
```

Gate:

```text
terminal MSE improves or release error in open-loop replay improves
```

Interpretation:

- If terminal metrics improve, proceed to rollout.
- If not, inspect whether bin/target geometry is observable.

---

## Experiment 3: Add residual impedance

Configuration:

```text
direct base action retained
impedance residual branch initialized zero
beta initially 0, then slowly increased
no aggressive stability loss
```

Gate:

```text
MSE does not degrade by more than 5%
rollout macro-completion remains high
terminal release precision improves or remains stable
```

Interpretation:

- If residual impedance improves robustness without hurting precision, keep it.
- If it degrades precision, keep direct action as primary and use impedance only as a diagnostic.

---

## Experiment 4: Terminal expert / split Place regime

Configuration:

```text
split Place into PlaceAlign / PlaceRelease
or add dedicated terminal precision expert
terminal expert receives object-bin features
terminal expert may use higher gain / direct gripper head
```

Gate:

```text
small dev rollout bank success > 0
final release object-bin error reduced
```

This is the experiment most likely to convert macro-completion into actual success.

---

## Experiment 5: Corrected stability regularizer

Only after success is non-zero and stable.

Configuration:

```text
use action smoothness or terminal error reduction
tiny lambda
phase-specific
monitor Place/Release MSE
```

Gate:

```text
no meaningful degradation in terminal precision
improved robustness to perturbations or reset variation
```

---

# 9. Expected interpretations

| Result | Interpretation |
|---|---|
| Direct warm-start recovers MSE and success returns | The failure was optimization: frozen encoder, random heads, and lip loss |
| Direct warm-start recovers MSE but success remains 0 | Terminal object/release physics or success predicate is not captured by action MSE |
| Adding object/bin features improves terminal error | The policy lacked explicit placement geometry |
| Splitting Place/Release improves success | Terminal behaviors were averaged inside one expert/regime |
| Residual impedance improves robustness without hurting precision | The impedance idea is valid, but only as a residual/compliance term |
| Pure impedance still plateaus at high MSE | The pure target/gain parameterization is not suitable for this action space |
| Corrected stability regularizer helps | Stability can be added, but only after precision recovery |
| Corrected stability regularizer hurts precision | Do not use it for this task under current action interface |

---

# 10. Final recommended architecture summary

The final architecture should look like this:

```text
Input state x_t
    |
    +-----------------------------+
    |                             |
Route encoder E_r            Control encoder E_c
    |                             |
z_r = E_r(x_t)               z_c = E_c(x_t)
    |                             |
Prototype router               Expert k receives:
    |                          [z_c, y_task, optional z_r]
selected expert k                   |
    |                               |
    +--------------+----------------+
                   |
          Expert k outputs:
                   |
        +----------+----------+
        |                     |
   direct action          residual impedance
   a_base_k               delta_k, kappa_k
        |                     |
        +----------+----------+
                   |
          a_k = clip(a_base_k + beta * a_imp_k, -1, 1)
                   |
          separate gripper head recommended
                   |
              environment
```

Key properties:

1. memoryless,
2. warm-started,
3. direct precision path,
4. optional impedance residual,
5. terminal object/bin-aware,
6. no aggressive target-Lipschitz loss during recovery,
7. routing and precision representations are not forced to be identical.

---

# 11. Final concise recommendation

The immediate next step should not be another full impedance training run with the current loss configuration.

Do this instead:

1. **Do not revert `c56275d`.**
2. **Audit and unit-test the action adapter scaling.**
3. **Disable the current Lipschitz loss.**
4. **Unfreeze or partially unfreeze the encoder.**
5. **Warm-start experts from the Stage 1 direct action policy.**
6. **Use a direct-action or residual-impedance formulation.**
7. **Add per-dimension and per-phase MSE diagnostics.**
8. **Add object/bin relative features if available.**
9. **Split or specially handle Place/Release.**
10. **Only run full rollouts after offline precision gates are passed.**

The final architecture should be:

> **a memoryless, dual-stream, precision-residual PhaseForge policy, with direct-action warm-start, terminal object/bin awareness, and optional impedance residual, rather than a frozen-encoder pure impedance policy trained against a target-ratio contraction loss.**

This is the strongest combined solution given the current evidence.