Yes. Your diagnosis is **directionally right**, but I would not yet conclude that checkpoint selection is *the* cause. The telemetry rules out two obvious failures—Stage-1 instability and router collapse—but several other bottlenecks remain.

The surgical approach should be:

## 1. First isolate checkpoint selection

This is the **highest-priority experiment**.

For each seed, evaluate:

[
\text{epoch }1,2,4,8,16,30,50,100,200
]

using the same 50 paired episodes.

Plot:

[
\text{val action loss}
\quad\text{vs}\quad
\text{rollout success}
]

and also:

[
\text{NMI, entropy, balance}
\quad\text{vs epoch}.
]

The critical question is:

> Does rollout SR peak near the selected checkpoint, or is `val/loss_action` simply a poor proxy for actual control success?

Your current facts already make this highly plausible: selected epochs 4/30/16 are very early, while Stage-2 loss later deteriorates. But we need the **SR-vs-checkpoint curve** to prove it.

### Decision

If SR is stable/high across a broad checkpoint range → checkpoint noise is not the main problem.

If SR is sharply peaked → checkpoint selection is a major bottleneck.

If lowest val MSE and highest SR occur at different epochs → **change the model-selection protocol**, not the architecture.

---

# 2. Test validation-set variance

Your 20-trajectory validation set is tiny relative to the stochasticity of the task.

Do this:

* keep training unchanged;
* create several fixed validation banks of equal size;
* calculate checkpoint ranking under each bank.

Then ask:

[
corr(\text{val-loss ranking},\text{rollout-SR ranking})
]

If this correlation is poor, the current model-selection criterion is simply noisy/misaligned.

**This is more important than EMA right now.**

---

# 3. Measure seed variance *before* Stage 2

You already know Stage-1 loss is stable.

Now compare the actual latent geometry across seeds:

[
h_{42}(x),h_{43}(x),h_{44}(x)
]

on the same samples.

Measure:

* cosine similarity between corresponding latents;
* phase separability;
* centroid-to-centroid geometry;
* bootstrap router logits.

You want to know whether:

> "same Stage-1 loss"

actually means

> "same representation."

Loss equality does **not** imply representation equality.

---

# 4. Test whether the router is actually learning useful specialization

Your NMI result is important:

[
0.40-0.46
]

and roughly unchanged across seeds.

So the router is not collapsing, but neither is it strongly phase-specialized.

Now compute the missing thing:

[
M_{z,e}=MSE(\text{expert }e\text{ on phase }z).
]

Then inspect:

### Case A

Each phase has a clearly best expert.

Good. The router is probably failing to exploit available specialization.

### Case B

All experts perform similarly on all phases.

Then the problem is deeper:

> **The experts have not actually differentiated.**

In that case, changing the router won't fix the fundamental bottleneck.

---

# 5. Run the most important counterfactual: frozen router

Take the trained experts and evaluate:

### learned router

vs.

### phase-centroid router

vs.

### oracle phase router

vs.

### uniform routing

vs.

### random routing.

This tells you where the performance is being lost.

For example:

[
Oracle \gg PhaseForge
]

means routing is the bottleneck.

But:

[
Oracle \approx PhaseForge \approx Uniform
]

means expert specialization itself isn't buying much.

---

# 6. Your teacher-forced result makes this especially important

You currently have:

[
TeacherForced=.527
<
PhaseForge=.640.
]

That is **not what we'd expect if "better phase alignment = better control"**.

Therefore do not try to increase NMI blindly.

We need to establish whether phase alignment is actually causally useful.

This may lead to an important discovery:

> The useful routing decomposition may not be the simulator's phase decomposition.

If so, PhaseForge is using phase as a **representation scaffold**, not as the final expert partition.

That is scientifically interesting.

---

# 7. Then isolate expert initialization

You already have the critical cells:

[
\text{centroid + warm}
]

vs

[
\text{centroid + random}.
]

And currently:

[
.660 > .640.
]

This is a major clue.

Run the four-way matrix cleanly:

| Router   | Experts |
| -------- | ------- |
| random   | random  |
| centroid | random  |
| random   | warm    |
| centroid | warm    |

Now you can determine whether warm-starting is:

* helping,
* neutral,
* or actively preventing specialization.

I would consider this a **core experiment**, not an ablation.

---

# 8. Test expert diversity directly at initialization

For the warm-started experts calculate:

[
D_{ij}^{(0)}
============

E[|\pi_i(x)-\pi_j(x)|].
]

Then after 1, 5, 20, 200 epochs.

You want to see:

[
D^{(0)}
\rightarrow
D^{(T)}.
]

If the experts remain almost identical, then your σ=0.02 jitter is not enough to produce meaningful specialization.

If they diverge substantially but performance doesn't improve, then specialization itself may not be useful.

---

# 9. Treat routing entropy differently from collapse

Your:

[
H\approx0.95
]

with:

[
H_{\max}=\ln 6\approx1.79
]

means the router is neither collapsed nor maximally uniform.

That's not inherently good or bad.

The important question is:

[
\boxed{\text{Is the routing distribution useful?}}
]

So cross-reference entropy with:

* expert MSE;
* phase MSE;
* rollout SR;
* routing contingency.

Don't optimize entropy itself.

Likewise, ~0.98 balance is not evidence that routing is good. It only says utilization isn't catastrophically skewed.

---

# 10. Investigate whether the balance loss is over-constraining specialization

This is a **very plausible bottleneck**.

Your balance term forces usage toward all six experts.

But what if the optimal control solution naturally uses:

[
2\text{–}3
]

experts much more than the others?

Then:

[
L_{balance}
]

could be fighting the control objective.

You should run:

[
balance_coeff
\in
{0,0.001,0.01,0.03}
]

with the rest fixed.

Do not interpret this as hyperparameter hunting; frame it as testing:

> **Does forced utilization prevent useful specialization?**

---

# 11. Investigate router noise separately

Same principle.

Your router has learned noisy gating.

Test:

[
\sigma_{router}
===============

0,;0.05,;0.1
]

or whatever the implementation's actual scale corresponds to.

The question is:

> Is exploration helping escape symmetry, or injecting unnecessary stochasticity once the phase prior is already informative?

Again, this is a mechanism test.

---

# 12. Check the action-loss → rollout mismatch

This is potentially huge.

You select based on:

[
MSE(a,\hat a)
]

but care about:

[
\text{task success}.
]

Those aren't necessarily monotonic.

Calculate across checkpoints:

[
corr(\text{val action MSE},\text{rollout SR}).
]

Also stratify failures.

You already know they're all `task_timeout`, which is useful.

Now determine:

* how many steps into the task they fail;
* whether failures cluster around particular phases;
* whether the policy makes recoverable versus irreversible errors.

This may reveal that the MSE objective isn't sufficiently aligned with the actual control objective.

---

# 13. Then analyze failure phase

For every failed episode, record:

[
z_{\text{failure}}
]

and ideally:

[
e_{\text{top1 at failure}}.
]

Then build:

[
Phase\times FailureRate.
]

If failures concentrate in, say, Phase 4:

> PhaseForge may be solving five regimes and failing in one.

That is much more actionable than looking at global SR.

---

# 14. Analyze routing on successful vs failed episodes

This is another high-value diagnostic.

Compare:

[
P(E|success)
]

against:

[
P(E|failure).
]

And:

[
H(E|success)
]

vs.

[
H(E|failure).
]

If failure episodes have diffuse/unstable routing, you've localized the problem to the router.

If successful and failed episodes have almost identical routing, routing probably isn't the bottleneck.

---

# 15. Only after that investigate representation quality

Do not assume:

> Stage-1 loss is stable → representation is fine.

Instead measure:

[
\text{phase separability}
]

[
\text{action-neighborhood consistency}
]

[
\text{centroid margin}
]

For each sample:

[
\Delta_i
========

## \cos(h_i,c_{true})

\max_{j\neq true}\cos(h_i,c_j).
]

If centroid margins are tiny, the bootstrap prior is intrinsically weak.

That would explain why NMI stays only around .4.

---

# 16. The complete surgical decision tree

I would run the investigation in this order:

```text
1. Checkpoint sweep
        ↓
Is SR highly checkpoint-sensitive?
        │
        ├── YES → fix selection/validation first
        │
        └── NO
             ↓
2. Representation geometry
             ↓
Are phase centroids actually separated?
             │
             ├── NO → Stage-1 representation is bottleneck
             │
             └── YES
                  ↓
3. Expert specialization matrix
                  ↓
Do experts actually differ/usefully specialize?
                  │
                  ├── NO → expert initialization/training is bottleneck
                  │
                  └── YES
                       ↓
4. Routing counterfactuals
                       ↓
Does learned routing exploit the experts?
                       │
                       ├── NO → router is bottleneck
                       │
                       └── YES
                            ↓
5. Balance/noise sensitivity
                            ↓
Is regularization fighting specialization?
                            │
                            └── optimize only after mechanism is known
```

---

# 17. What I would run next, specifically

**Wave A — absolutely first**

1. Checkpoint sweep on existing three seeds.
2. SR vs val-MSE correlation.
3. Multiple validation-bank/checkpoint-selection test.
4. `M_{phase,expert}` specialization matrix.
5. learned vs centroid vs oracle vs uniform routing.

**Wave B**

6. Four-way router/expert initialization.
7. expert-divergence trajectory.
8. balance coefficient ablation.
9. router-noise ablation.

**Wave C**

10. latent geometry / centroid-margin analysis.
11. failure-by-phase analysis.
12. success-vs-failure routing analysis.
13. only then K=3/6/12, corruption, FT, etc.

---

## The key point

Right now I would **not call PhaseForge "non-robust because of checkpoint selection."**

The defensible conclusion is:

> **Stage-1 representation and expert utilization are stable at the coarse telemetry level, while substantial rollout variance remains unexplained. The highest-priority hypotheses are checkpoint-selection/validation mismatch, weak expert specialization, and a mismatch between phase-aligned routing and actual control-optimal routing.**

The good news is that this is now a **diagnosable research problem**, not a reason to redesign the architecture. The next experiments should determine *where the causal bottleneck is* before we tune anything.
