# Square Failure Diagnostic Protocol

## Purpose

The current question is not “which method has the highest Square success
rate?” It is:

> Which component of the PhaseForge pipeline is associated with the Square
> regression, and which controlled intervention would distinguish the possible
> causes?

The candidate causes are deliberately separated:

1. state/representation mismatch or distribution shift;
2. topology-label and phase-representation mismatch;
3. router-boundary sensitivity;
4. incompatible expert actions at a route transition;
5. direct-action stiffness (`beta=0`);
6. an upstream data, training, or evaluation-contract issue.

No single rollout statistic identifies causality. The protocol therefore
requires a diagnostic signature and a matched intervention before any
mechanistic claim is made.

## What has already been implemented

### 1. State and phase suitability

`scripts/analysis/phase_router_diagnosis.py` measures:

- trajectory-grouped state-only observability of `phase` and `phase_topo`;
- held-out action-MSE reduction from phase-conditioned action means;
- phase/topology label agreement when cache alignment can be verified;
- Stage-1 metadata, including the effective phase-CE weight.

This is a data/representation diagnostic. It does not use a trained Stage-2
checkpoint and cannot prove that a router caused a rollout failure.

### 2. Router and action-transition audit

`scripts/analysis/square_failure_audit.py` measures, by task, method, and
seed:

- switch rate and selected-expert occupancy;
- router margin and entropy;
- action jumps at switches versus non-switch transitions;
- expert disagreement at switches;
- nearest-training-state distance;
- configuration/provenance fields and trace-field availability.

The script fails closed when summaries or traces are missing. It explicitly
reports that no force or measured velocity is available in the archived
schema.

### 3. Closed-loop motion-contract audit

`scripts/analysis/square_closed_loop_diagnostics.py` measures:

- the norm of the first three action components;
- observed EEF-position displacement between adjacent trace rows;
- the action-to-position response ratio;
- command/motion directional alignment;
- high-command/low-motion candidate stall transitions;
- success-conditioned and failure-conditioned summaries when the trace
  termination reason is available.

The EEF displacement is only a motion proxy. It is not contact force,
velocity, or impedance. The report makes that limitation explicit.

## Results from the archived focused ablation

The audits were run on the archived Can/Square focused output with 30 rollout
traces, 50 episodes per task/method/seed cell.

### Square success-conditioned pattern

Across all evaluated methods, failed Square episodes had higher median
`nearest_train_dist` than successful episodes and lower command-to-EEF-motion
alignment. This is consistent with closed-loop drift into states that are less
represented by the training data. It does **not** show whether the drift was
caused by the initial router, the representation, or the environment.

The topology arm did not have the highest Square switch rate. Instead, its
Square switch transitions had larger median action jumps than the phase and
random prototype arms in the archived traces. This is a more precise
candidate mechanism than “topology is unstable”:

- topology: approximately 0.134–0.160 median action jump at switches by seed;
- phase: approximately 0.078–0.089;
- random: approximately 0.084–0.108.

The ranges are descriptive seed-level summaries, not significance tests. The
softmax control also showed large transition jumps but achieved higher Square
success and lower switch frequency, so transition magnitude alone cannot be
the complete explanation.

No archived trace contains force, torque, contact-state, or measured EEF
velocity fields. Therefore the current archive cannot prove “contact jamming,”
“force spikes,” or a mechanical compliance failure.

## Diagnostic decision tree

### A. Representation/data hypothesis

**Prediction:** Square failures should show elevated nearest-training-state
distance before or during the failure, and the effect should be present in
multiple router variants.

**Current evidence:** supported as a candidate association. It is not
initialization-specific.

**Next test:** compare the same representation provider under random, phase,
and topology initialization with the same Stage-2 contract, and report
success-conditioned distance trajectories—not only aggregate medians.

### B. Phase/topology partition hypothesis

**Prediction:** topology and rule-phase labels should induce materially
different partitions, and the topology partition should be less aligned with
the action-relevant representation on Square.

**Current evidence:** Can has low phase/topology label agreement in the local
aligned cache; Square topology agreement is not available locally. This is a
data limitation, not evidence for Square.

**Next test:** rerun or restore aligned Square `phase` and `phase_topo` caches,
verify identical state/action/demo-key order, and compute NMI, ARI, state-only
macro-F1, action-MSE reduction, and within-label action variance.

### C. Router-boundary hypothesis

**Prediction:** topology should produce low margins near transitions, and
switching should be associated with unusually large changes in the selected
action.

**Current evidence:** partly supported descriptively. Topology has large
action jumps at its Square switches, but lower switch frequency than random.
That is compatible with rare but consequential transitions; it is not proof
of causality.

**Next test:** rerun with full checkpoints and log every expert’s action and
task-space output at each state. Compare the selected action with the
counterfactual action of the runner-up and measure the action discontinuity
across matched boundary states.

### D. Expert incompatibility hypothesis

**Prediction:** topology’s runner-up disagreement should be higher near Square
failures than in successful episodes or control arms.

**Current evidence:** the trace contains an expert-disagreement scalar, but
the current archives do not identify the individual expert actions or the
per-episode success state sequence in a reusable counterfactual format.

**Next test:** add per-expert direct-action/task-error logging to full traces,
then compare success and failure episodes using the same reset bank and seeds.

### E. Direct-action stiffness hypothesis

**Prediction:** changing only `beta` should change Square behavior while all
representation, router, expert initialization, and seed conditions remain
fixed.

**Current evidence:** not tested. Because `ResidualImpedanceExpert.forward`
returns the direct base action when `beta == 0`, the current focused results
are not evidence about residual impedance behavior.

**Next test:** a small matched Square experiment with `beta=0` and one
pre-registered positive beta condition. This is a separate architecture
question and must not be mixed into the router-initialization conclusion.

## Required cloud experiments, in order

### Experiment 1: CE-on representation check

Run Can and Square with:

- `train.supcon.zero_ce=false`;
- seeds 42, 43, 44;
- the existing PhaseForge topology, phase, and random router arms;
- unchanged Stage-2 settings, including `beta=0` and the disabled margin;
- the same reset banks and evaluation horizon.

Validate that `loss_phase` is nonzero and that the resolved configuration
records the override. This determines whether the executed representation
matches the intended phase-aware design.

### Experiment 2: matched beta test

On Square only, compare `beta=0` with one fixed, pre-registered `beta>0`
condition. Keep the router and representation fixed. Report expert target,
gain, task-error, action, and trace motion metrics. Do not call this a router
ablation.

### Experiment 3: expert-action boundary test

Rerun the selected topology and softmax controls with checkpoint retention and
per-expert trace logging. At states near a route boundary, record:

- selected and runner-up expert actions;
- action difference and task-error difference;
- router margin;
- EEF displacement over the next few steps;
- termination reason.

This is the minimum evidence needed to evaluate whether a route transition is
mechanistically harmful.

### Experiment 4: temporal counterfactual, only if the above remains ambiguous

Add a minimal sticky/hysteresis or short-history controller as a separate
diagnostic. This is not part of the current memoryless claim. It tests whether
history resolves ambiguous routing, but it should not be introduced until the
representation, beta, and expert-boundary diagnostics are complete.

## What not to conclude

- Do not conclude that topology initialization is the sole cause of Square
  failure from the current data.
- Do not conclude that SupCon collapses the contact manifold without a local
  geometry test and a matched CE-on control.
- Do not call EEF displacement a force measurement.
- Do not use UMAP or t-SNE as proof; they are visualization aids only.
- Do not compare a different router package and call the result a pure
  initialization effect.
- Do not tune beta, margin, loss weights, and router initialization together
  and then attribute the result to one factor.

## Current scientific position

The strongest current explanation is an interaction, not a single bug:

> On Square, failures are associated with closed-loop departure from the
> training-state distribution and poor action-to-motion alignment. Topology
> initialization may make some expert transitions more action-discontinuous,
> but the existing evidence does not show that this is the initiating cause.

This is a falsifiable working hypothesis. The next experiments should be
chosen to break that interaction apart, not to optimize the Square number
until it improves.

