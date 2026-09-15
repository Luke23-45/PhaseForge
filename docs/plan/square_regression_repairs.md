# Square regression repair study

This protocol is a diagnostic repair study, not a claim that any arm is
guaranteed to improve task success. It preserves the current experiment
family and changes one causal factor at a time around a same-seed topology
anchor.

## Mathematical motivation

The closed-loop error is not determined by the supervised action loss alone.
For a policy \(\pi\), behavior-cloning data provide samples from an expert
state distribution \(d_E\), while deployment visits the learner-induced
distribution \(d_\pi\). A useful target is therefore the learner-induced
loss \(\mathbb{E}_{s\sim d_\pi}[\ell(\pi(s),a_E(s))]\), not only the
demonstration loss under \(d_E\). Dataset aggregation (DAgger) addresses
this distribution mismatch by collecting labels on learner-visited states;
the current repository has no online expert-labeling interface, so this
protocol does not claim to implement DAgger.

For a hard prototype router, the deployed action is piecewise:

\[
  \pi(z)=a_{k^*(z)}(z),\qquad k^*(z)=\arg\min_k\|z-c_k\|_2.
\]

At a Voronoi boundary, an arbitrarily small latent change can replace one
expert by another. The top-2 arm instead uses

\[
  \pi_2(z)=w_1(z)a_{k_1}(z)+w_2(z)a_{k_2}(z),\quad
  w_i=\frac{\exp(-d_i)}{\exp(-d_1)+\exp(-d_2)},
\]

which tests whether reducing the action discontinuity at the nearest two
prototype boundary improves Square rollouts. In this repository the active
Square expert is `ResidualImpedanceExpert`; therefore this arm blends direct
actions, not impedance controller parameters.

The phase-CE arm tests whether the current SupCon configuration is starving
the phase classifier. `train.supcon.enabled=true` defaults to
`zero_ce=true`, so the phase CE contribution is zero unless the arm sets
`train.supcon.zero_ce=false`.

The beta arm is deliberately secondary. In the current implementation,
`ResidualImpedanceExpert.forward` computes a latent residual
\(\beta\,\kappa(z)\delta(z)\) and does not use the task-state feedback error.
Consequently, beta=0.1 is a residual-capacity probe, not evidence of proper
impedance control.

## Interpretation rule

Use the anchor and each repair arm with the same seed, reset bank, episode
count, and full trace level. A success-rate change alone is insufficient:
inspect action MSE, nearest-training-state distance, router margin, selected
expert changes, expert disagreement, and command/motion alignment together.
The phase-CE arm is evidence for representation failure only if phase/action
diagnostics improve with a corresponding rollout improvement. The top-2 arm
supports a boundary-discontinuity explanation only if transition action jumps
and/or disagreement-related failures decrease. The beta arm should not be
used to make an impedance-control claim.

## Run

The one-seed command is:

```text
uv run python -m phaseforge.runner \
  --manifest experiments/square_regression_repairs.json \
  --outputs outputs_square_regression_repairs_seed42 \
  --methods square_topology_top1_anchor square_phase_ce_on square_topology_top2 square_residual_beta_01 \
  --tasks Square \
  --seeds 42 \
  --with-dependencies \
  --expect-steps 10
```

The outputs namespace must be fresh. Do not merge these outputs with the
previous ablation namespace.

## What is still required for a coverage fix

If all four arms remain weak while offline metrics look acceptable, the next
mathematically appropriate experiment is learner-visited-state data
aggregation (DAgger or a safe/limited intervention variant), not more router
initialization tuning. That requires an expert or recovery-label interface;
it cannot be inferred from the existing demonstration cache.

## External references

The coverage argument follows Ross, Gordon, and Bagnell, “A Reduction of
Imitation Learning and Structured Prediction to No-Regret Online Learning”
([DAgger, PMLR 2011](https://proceedings.mlr.press/v15/ross11a.html)). The
lower-risk noisy-demonstration alternative is Laskey et al., “DART: Noise
Injection for Robust Imitation Learning”
([CoRL 2017, PMLR](https://proceedings.mlr.press/v78/laskey17a.html)). The
sparse weighted-expert formulation is consistent with Shazeer et al., “The
Sparsely-Gated Mixture-of-Experts Layer”
([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)); soft assignment is
discussed by Puigcerver et al., “From Sparse to Soft Mixtures of Experts”
([ICLR 2024 / OpenReview](https://openreview.net/pdf?id=jxpsAj7ltE)). The
impedance-control distinction is based on Hogan, “Impedance Control: An
Approach to Manipulation: Part I—Theory”
([ASME DOI](https://doi.org/10.1115/1.3140702)). These sources motivate the
interventions but do not establish that any arm will improve Square in this
repository.
