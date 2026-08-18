Yes. I reviewed the uploaded implementation plan itself, rather than treating it as a blank proposal. Overall, the plan is **substantially better than the earlier version**, and I would preserve its central research direction. But I would **not execute it exactly as written yet**. There are a few experimental-design problems that could weaken the scientific conclusions, especially around the expert-count sweep, the initialization ablation, and how the statistics are interpreted.

The most important issue is not implementation. It is **causal identification**: making sure every result can actually support the claim you want to make.

I’ll go through the plan in that order.

---

# 1. My overall verdict

I would rate the current plan roughly:

| Area                       | Assessment               |
| -------------------------- | ------------------------ |
| Core research question     | **Strong**               |
| Literature positioning     | **Much improved**        |
| Existing factorial design  | **Strong**               |
| Code/config organization   | **Strong**               |
| Reproducibility discipline | **Very strong**          |
| Parameter fairness         | **Good, but incomplete** |
| Initialization ablations   | **Needs correction**     |
| Expert-count experiment    | **Currently confounded** |
| Specialization evidence    | **Needs strengthening**  |
| Statistical protocol       | **Needs tightening**     |
| Overall direction          | **Keep**                 |

The plan has one particularly good philosophy:

> **Add evidence, not architectural complexity.**

I strongly agree with that. The uploaded plan explicitly keeps the mechanism fixed and expands the evidence rather than adding attention, recurrence, contrastive losses, dynamic experts, etc. 

That is the right instinct for this project.

---

# 2. First, the central research question is now much better

The plan proposes:

> Can privileged regime information available during training be converted into useful latent geometry and transferred into an MoE routing prior, enabling specialized control without the privileged phase signal at inference?

That is significantly better than claiming novelty for "centroid initialization." 

And this is particularly important because the literature now contains direct precedents for generic latent/cluster-based routing initialization.

Royer et al. already use clustering-based MoE initialization. ([bmvc2022.mpi-inf.mpg.de][1])

More importantly, **Cluster-aware Upcycling**, accepted at CVPR 2026, explicitly uses activation-space clusters, cluster centroids for router initialization, and pretrained-model information for expert initialization. ([Open Access CVF][2])

And LAR-MoE is a recent robotics paper that explicitly connects latent structure and MoE routing, while avoiding explicit phase annotations. ([arXiv][3])

So the plan is correct to move the contribution toward **privileged regime geometry transfer** rather than "we invented centroid routing."

That part should stay.

---

# 3. But I would make the research hypothesis even sharper

Right now the plan says:

> phase supervision shapes latent geometry → phase prototypes transfer into router → phase labels disappear → autonomous specialization.

That's good. 

But the actual causal claim should be written internally as:

[
\boxed{
\text{Phase supervision}
\rightarrow
\text{phase-discriminative representation}
\rightarrow
\text{better routing initialization}
\rightarrow
\text{better specialization/control}
}
]

This gives you **four links** to test.

Your current experiments mostly test the first two links.

The final paper needs evidence for all four.

---

# 4. The most important problem: your K=3/6/12 experiment is currently confounded

This is the biggest thing I would fix.

Your plan currently says the bootstrap handles:

> E≠P, maps 1:1 for the first P, rest random.

and then proposes:

[
K\in{3,6,12}.
]

The plan itself records that implementation behavior. 

But think about what happens.

You have:

[
P=6
]

phases.

---

## K = 6

Fine:

```text
Phase 1 centroid → Expert 1
Phase 2 centroid → Expert 2
...
Phase 6 centroid → Expert 6
```

All experts have meaningful initialization.

---

## K = 12

Your current implementation would effectively do:

```text
Expert 1 ← Phase centroid 1
Expert 2 ← Phase centroid 2
...
Expert 6 ← Phase centroid 6

Expert 7 ← random
...
Expert 12 ← random
```

Then you're comparing:

> 12-expert PhaseForge with 6 informed experts + 6 random experts

against:

> 6-expert PhaseForge with 6 informed experts.

So if K=12 performs worse, you cannot tell whether:

* 12 experts are bad,
* random extra experts are bad,
* top-k dilution is bad,
* optimization is harder,
* or the method fundamentally prefers six experts.

Likewise, if K=12 wins, you cannot cleanly say that more experts helped.

---

# 5. K = 3 is even more problematic

You have six phase centroids but only three experts.

You can't simply map:

[
6\rightarrow3
]

with a one-to-one mapping.

The implementation statement "maps 1:1 for the first P" cannot logically produce six unique phase assignments when (E=3).

So this needs an explicit scientific definition.

There are at least three principled choices.

### Option A — Cluster the six phase prototypes

Take:

[
c_1,\ldots,c_6
]

and cluster those six prototypes into three groups.

Then:

[
{c_1,c_2,\ldots,c_6}
\rightarrow
3\text{ super-prototypes}.
]

This gives you a coherent:

[
6\ phases \rightarrow 3\ routing regimes
]

experiment.

### Option B — Learn expert prototypes by spherical K-means over all latent samples

For each K:

[
K=3,6,12
]

perform clustering directly over normalized Stage-1 latent vectors.

Then each K gets a consistent initialization.

This may actually be the cleanest **expert-count scaling experiment**, although it partially overlaps with your KMeans baseline.

### Option C — Preserve phase semantics explicitly

For K=3:

[
{phase_1,phase_2}\rightarrow expert_1
]

etc., but then the grouping must be predetermined from domain knowledge.

I don't recommend this unless the simulator itself gives you a principled phase adjacency.

---

# 6. For K=12, I would do something even more interesting

Don't introduce random experts.

Instead, split each phase into two latent sub-prototypes.

For example:

[
Phase_i
\rightarrow
{c_{i,1},c_{i,2}}.
]

This could be done using spherical K-means **within each phase**.

Then:

```text
Phase 1 → Expert 1a, Expert 1b
Phase 2 → Expert 2a, Expert 2b
...
Phase 6 → Expert 6a, Expert 6b
```

Now K=12 is a genuine test of:

> Can the phase scaffold support finer-grained specialization?

That is much more scientifically interesting than "six informed + six random."

And it aligns beautifully with your earlier hypothesis that the phase labels could provide a **coarse scaffold** from which Stage 2 discovers finer control regimes.

---

# 7. Therefore I would change the K experiment

Instead of:

```text
pf_k3
pf_k6
pf_k12
```

with ad hoc mapping, define:

[
K\in{3,6,12}
]

with a **specified prototype-generation rule for every K**.

Something like:

[
\text{normalized Stage-1 latents}
\rightarrow
\text{deterministic spherical K-means}(K)
\rightarrow
\text{router initialization}.
]

Then separately test:

[
\text{phase-centroid initialization}
]

as your primary PhaseForge mechanism.

This cleanly distinguishes:

> "phase prototypes"

from:

> "generic K-way latent clustering."

---

# 8. The four-way expert/router initialization ablation is excellent

This section of the plan is one of the strongest parts.

You propose:

| Router   | Expert    |
| -------- | --------- |
| random   | random    |
| centroid | random    |
| random   | warmstart |
| centroid | warmstart |

The plan correctly identifies that the current PhaseForge and random-router variants cover only two of those four cells. 

I strongly recommend this.

Because otherwise you cannot tell whether the gain comes from:

[
\text{router initialization}
]

or:

[
\text{expert warm-start}.
]

This is particularly important because dense-to-MoE warm-starting already has substantial prior literature. Sparse Upcycling explicitly establishes the value of initializing MoE models from dense checkpoints rather than from scratch. ([arXiv][4])

So your four-cell matrix should become a major piece of evidence.

---

# 9. But there is one subtle issue with the four-way ablation

The plan says:

> all four cells use the PhaseForge Stage-1 encoder.

That is correct if the question is:

> what does router/expert initialization contribute, holding representation fixed?

But you should be explicit that this is a **Stage-1 phase-supervised encoder**.

Otherwise somebody may interpret:

```text
random/random
```

as a normal scratch MoE.

It isn't.

It is:

[
\boxed{
\text{phase-supervised encoder}
+
\text{random router}
+
\text{random experts}
}
]

That's actually a very useful cell.

Keep the naming extremely explicit.

I'd call them something like:

```text
PS-Rand-Rand
PS-Centroid-Rand
PS-Rand-Warm
PS-Centroid-Warm
```

internally.

That eliminates ambiguity.

---

# 10. Your `scratch_moe` should remain separate

The plan correctly doesn't substitute `scratch_moe` for random/random in the four-way matrix. 

Keep it.

These answer different questions.

### Random/random

Tests initialization, holding the learned representation fixed.

### Scratch MoE

Tests the whole staged pretraining idea versus training an MoE without pretraining.

Excellent distinction.

---

# 11. The KMeans baseline is absolutely worth adding

I strongly agree with:

```text
pf_kmeans
```

because this is your strongest generic-clustering control.

Your question becomes:

[
\text{phase-supervised prototypes}
\quad vs \quad
\text{unsupervised latent clusters}.
]

This is especially important because recent literature already demonstrates generic activation clustering for MoE initialization. ([Open Access CVF][2])

Your method needs to show that **the phase signal adds something over merely clustering the latent space**.

That is one of the cleanest ways to distinguish PhaseForge from Cluster-aware Upcycling.

---

# 12. But `kmeans` alone is not enough

You need to specify whether `kmeans` is:

### ordinary Euclidean K-means

[
\min_c \sum_i|h_i-c_{z_i}|^2
]

or:

### spherical K-means

[
\max_c \sum_i
\cos(h_i,c_{z_i}).
]

Because your router uses cosine similarity.

The plan recognizes this with `spherical_centroid` and `spherical_kmeans`, but I would make the primary comparison more systematic. 

---

# 13. I strongly recommend spherical K-means as the generic clustering comparator

Why?

Because your actual router is directional:

[
g_i\propto \cos(h,c_i).
]

If you give generic K-means an advantage/disadvantage merely because of a mismatch between Euclidean and cosine geometry, the comparison becomes less clean.

Therefore I'd probably have:

### PhaseForge

[
\text{phase-conditioned spherical prototypes}
]

### Generic cluster baseline

[
\text{spherical K-means}
]

Same geometry.

Different source of supervision.

That's a much stronger causal comparison.

---

# 14. The phase-head initialization experiment is excellent—but interpret it correctly

Your plan includes:

```text
pf_phase_head
```

with router rows initialized from the linear phase classifier. 

I really like this experiment.

But remember:

[
W_{\text{phase head}}
]

is a **discriminative classifier**.

Whereas:

[
c_z
]

is a **prototype**.

They encode different things.

The comparison answers:

> Is routing better initialized from class prototypes or class-discriminative directions?

That is a legitimate and interesting question.

---

# 15. There is a subtle implementation issue here

Your phase head is:

[
logits_z = W_z h+b_z.
]

If you copy only:

[
W_z
]

into the router and normalize it, then you are intentionally throwing away:

[
b_z.
]

That's probably the right thing for a cosine router, but it must be intentional.

I would explicitly define:

[
W_{router,z}
============

\frac{W_{phase,z}}{|W_{phase,z}|}
]

and:

[
b_{router,z}=0.
]

Otherwise the experiment becomes ambiguous.

Don't copy the phase bias unless you have a specific theoretical reason.

---

# 16. Your fairness section is good, but I'd make one change

The plan correctly records:

* total parameters
* Stage-2 trainable parameters
* active parameters/sample

and notes:

[
PhaseForge=1.849\times BC
]

in total parameters, while Stage-2 trainable parameters are approximately equal to BC. 

This is good.

But **active parameters/sample ≠ inference compute**.

You should therefore treat:

[
Params
]

and:

[
FLOPs
]

as separate metrics.

A two-expert sparse policy may have fewer active parameters while still incurring router overhead, expert dispatch overhead, etc.

Your fairness table should therefore include:

[
\boxed{
\text{total params}
}
]

[
\boxed{
\text{trainable params}
}
]

[
\boxed{
\text{active params/sample}
}
]

[
\boxed{
\text{FLOPs/sample}
}
]

and ideally:

[
\boxed{
\text{wall-clock inference latency}
}
]

if feasible.

The plan already includes FLOPs, which is good. 

---

# 17. The BC-large baseline is mandatory

This is absolutely correct.

Current BC:

[
206,983
]

PhaseForge:

[
382,646.
]

That is nearly 1.85× total parameter count. 

So:

> PhaseForge > BC

would be insufficient.

Your proposed:

[
BC_{large}=385,855
]

is excellent because it is within 1%.

Keep this.

---

# 18. But BC-large needs one additional fairness condition

Parameter matching isn't enough.

Make sure BC-large also has comparable:

* optimizer
* LR schedule
* number of epochs
* batch size
* gradient clipping
* data
* random seeds
* evaluation
* hyperparameter-selection protocol

Otherwise someone could say:

> "You tuned PhaseForge but not the large dense baseline."

Your plan currently says "own Stage 1." 

I would make that explicit in the implementation protocol.

And ideally:

[
\text{BC-large}
]

should receive the same amount of total optimization budget that you're granting the relevant comparison.

---

# 19. Your phase-label corruption experiment is good, but the definition must be precise

The plan proposes:

[
10,25,50,75,100%.
]

Good idea.

But "corruption rate" must have an exact mathematical definition.

For example:

[
z'_i =
\begin{cases}
z_i & \text{with probability }1-p\
\text{Uniform}({1,\ldots,6}\setminus z_i)&\text{with probability }p
\end{cases}
]

That's much better than:

> randomly corrupt p% of labels.

Because with naive random replacement, some "corrupted" labels can accidentally remain identical.

I'd use a **permutation or forced-different replacement** and log the realized corruption rate.

---

# 20. More importantly: corruption should happen only to the Stage-1 supervision labels

The plan already recognizes this. 

That is exactly right.

You still need clean labels for:

* teacher-forced
* oracle evaluation
* final phase/expert analyses.

Do not corrupt those.

I would go further:

> Store both `phase_gt_clean` and `phase_stage1_train`.

Then the data object makes the distinction impossible to accidentally violate.

---

# 21. I would add one particularly important corruption control

At 100% corruption, don't just say:

> "the method fails."

You need a **phase-shuffled control**.

Because 100% corrupted phase supervision should approximate:

[
\text{no meaningful phase supervision}
]

but only if the corrupted labels retain roughly equal class frequencies.

Therefore:

1. preserve phase counts;
2. permute labels across training samples;
3. ensure no sample retains its original label if using 100% corruption.

That gives you a proper null control.

---

# 22. Your specialization metric is absolutely necessary

The plan's statement:

> NMI alone is never cited as specialization

is exactly right. 

NMI measures:

[
I(\text{phase};\text{expert assignment})
]

approximately.

It doesn't prove:

[
\text{expert policies differ}.
]

Your proposed:

[
M_{z,e}
=======

MSE(\pi_e(x_z),a_z)
]

is much stronger.

Keep that.

---

# 23. But I would add one more metric to the specialization matrix

Your pairwise expert-output divergence is good:

[
D(e_i,e_j)
==========

E[|\pi_i(x)-\pi_j(x)|].
]

But there is a potential problem:

Two experts can produce different outputs while both being wrong.

So I'd pair:

### Behavioral specialization

[
M_{z,e}
]

with:

### Expert divergence

[
D_{e_i,e_j}
]

and:

### Overall routed performance.

Then you can distinguish:

```text
experts are different
```

from:

```text
experts are differently useful.
```

That's important.

---

# 24. I would also measure the "best expert per phase"

From:

[
M_{z,e},
]

define:

[
e^*(z)
======

\arg\min_e M_{z,e}.
]

Then compare:

[
M_{z,e^*(z)}
]

against:

[
M_{z,\text{routed}}.
]

This tells you:

> How much performance is theoretically available from phase-conditioned expert selection, and how much does the learned router recover?

That directly connects to your oracle experiment.

---

# 25. Your teacher-forced/oracle section needs one clarification

The plan says:

> oracle − teacher_forced = phase-predictability gap
> teacher_forced − phaseforge = strategy gap.

This is potentially useful, but **I would not call those differences these things automatically**.

Why?

Because the models may differ in more than phase prediction.

You need to define exactly what teacher-forced means:

* same trained experts?
* same encoder?
* same expert weights?
* only routing replaced?
* phase head used at inference?
* top-1 or top-2?
* does GT phase select one expert, or determine top-2 somehow?

Without that exact definition, "phase-predictability gap" is too strong.

I would call them:

[
\text{oracle gap}
]

and

[
\text{routing gap}
]

until the implementation guarantees the causal interpretation.

---

# 26. This is particularly important because you have top-2 routing

Suppose:

```text
Oracle phase = Phase 3
```

What does oracle MoE do?

### Choice 1

Select Expert 3 only.

Then it is not directly comparable to the learned top-2 PhaseForge.

### Choice 2

Select Expert 3 as top-1 and another expert based on something else.

Then what determines the second expert?

### Choice 3

Use phase to assign probability entirely to Expert 3.

Again, that's top-1.

So you need a very explicit definition of the oracle.

This is one place I would resolve before execution.

---

# 27. Your Stage-2 freezing logic is scientifically good

I like:

```text
freeze encoder = true
encoder = eval()
```

as the primary protocol.

The reason is causal isolation.

You're testing whether:

[
\text{Stage-1 representation}
\rightarrow
\text{MoE specialization}
]

works.

If the encoder is allowed to move freely, then Stage 2 can simply learn a new representation and the bootstrap becomes harder to interpret.

Keep frozen PhaseForge as the main model.

---

# 28. PhaseForge-FT should be secondary

The plan proposes:

```text
pf_ft
freeze_encoder=false
encoder LR scale=0.1
```

Good.

But absolutely do **not** let this replace the frozen model as the main result.

Make:

[
PhaseForge
]

the primary mechanistic method.

Then:

[
PhaseForge\text{-FT}
]

an enhancement/upper variant.

Otherwise reviewers could say:

> "The final result depends on continuing representation learning."

and your original causal story becomes weaker.

---

# 29. The jitter sweep is useful—but I wouldn't prioritize it over the missing structural controls

You have:

[
\sigma \in {0,.005,.01,.02}
]

Good.

But this is lower priority than:

* random/random
* centroid/random
* KMeans
* BC-large
* specialization matrix
* oracle/teacher-forced.

The plan already places it in Wave 2, which is sensible. 

Keep it there.

---

# 30. One thing I would add to the jitter experiment

Don't only look at final return.

Look at:

[
\text{expert pairwise similarity at initialization}
]

and after:

[
1,5,10,\ldots
]

epochs.

The hypothesis is:

[
\sigma=0
\Rightarrow
\text{symmetry}
]

whereas:

[
\sigma>0
\Rightarrow
\text{symmetry breaking}.
]

Then the experiment actually tests the proposed mechanism.

Otherwise it is just hyperparameter tuning.

---

# 31. Your current literature audit is good, but the related-work table needs another correction

Your plan correctly identifies that the professor's:

> "Phase-conditioned IL, 2026"

row appears mislabeled and actually points to Jacobs et al. 1991. 

That should definitely be fixed.

Also, after checking the primary sources again:

* Royer explicitly reports clustering-based initialization. ([bmvc2022.mpi-inf.mpg.de][1])
* Cluster-aware Upcycling explicitly uses cluster centroids for router initialization. ([Open Access CVF][2])
* LAR-MoE explicitly frames latent structure as a routing signal in robotic imitation learning. ([arXiv][3])

So your related-work section should **not** imply that the generic idea of "latent structure → MoE routing" is new.

Your differentiator needs to be the **privileged, phase-supervised, control-specific transfer mechanism and its controlled factorial validation**.

---

# 32. I would add one more literature category

Your plan has:

* classical MoE
* clustering-based MoE initialization
* latent-aligned routing
* privileged information

I would add a distinct category:

[
\boxed{\text{phase / skill / subtask-conditioned imitation learning}}
]

Not because the architecture is the same.

Because you need to answer:

> Why should simulator phase be a meaningful latent variable for control?

This literature establishes whether phase/state decomposition has already been used as a useful control abstraction.

Then your argument becomes:

1. phases can represent behavioral regimes;
2. privileged phase information can be used during training;
3. latent geometry can guide MoE routing;
4. PhaseForge combines those ideas in a specific transfer mechanism.

That is much more defensible.

---

# 33. A concern about your statistical wording

The plan says the current result is:

> controlled null with directional PhaseForge advantage

and notes overlapping CIs. 

Be careful here.

**Overlapping confidence intervals do not mean "no significant difference."**

And "directional advantage" is descriptive, not inferential.

I would write internally:

> "The current estimate favors PhaseForge directionally, but the present experiment does not establish a statistically significant advantage."

That's much safer.

---

# 34. I would predefine one primary metric

The final paper should not choose the best metric after seeing results.

Pick:

[
\boxed{\text{primary control metric}}
]

before running the new matrix.

Then define secondary metrics:

* action MSE
* success/return
* NMI
* routing entropy
* balance
* collapse
* specialization
* FLOPs

Otherwise you'll eventually have many correlated metrics and many chances to find a favorable result.

This is especially important because you're running a large number of ablations.

---

# 35. You also need a multiple-comparison policy

Your plan is going from:

```text
3 seeds
× many cells
× several metrics.
```

That creates a lot of comparisons.

You don't need to make the paper statistically cumbersome, but you should distinguish:

### Primary confirmatory comparisons

For example:

[
PhaseForge > PhaseSupRandom
]

[
PhaseForge > PlainCentroid
]

[
PhaseForge > KMeans
]

[
PhaseForge > BC-large.
]

### Secondary exploratory comparisons

For example:

* jitter sweep
* K sweep
* corruption
* fine-tuning.

That separation will make the paper much more credible.

---

# 36. The biggest statistical limitation remains M=1 task

Your plan correctly notes:

> M=1-task caveat.

Keep that.

Three seeds × 50 episodes does **not** magically turn a one-task benchmark into a broadly generalizable robotics result.

So the claim should be:

> demonstrated on the Lift benchmark/task

rather than:

> generally establishes the superiority of PhaseForge.

The mechanism can still be strong. Just don't over-generalize.

---

# 37. Your 50 paired episodes are good, but preserve pairing all the way through analysis

Don't reduce everything to:

[
mean_{PhaseForge}
]

vs.

[
mean_{baseline}.
]

Because the episodes are paired, use the paired differences:

[
d_i
===

## R_i^{PhaseForge}

R_i^{baseline}.
]

Then analyze:

[
{d_1,\ldots,d_{50}}.
]

With seeds, retain the stratification.

Your existing stratified bootstrap is therefore a reasonable framework, but I'd inspect the implementation before treating it as final statistical methodology.

---

# 38. Your trajectory analysis is one of the strongest planned additions

You already have per-epoch:

* NMI
* entropy
* balance
* collapse
* action MSE.

The plan correctly proposes turning these into trajectory plots rather than merely reporting the endpoint. 

This is important because PhaseForge is specifically an **initialization** method.

If the hypothesis is correct, we should see the benefit **early**.

For example:

```text
epoch 0
PhaseForge:
high phase alignment
reasonable balance
↓
epochs 1–20
routing reorganizes
↓
later
expert specialization
```

Whereas random-router might start:

```text
low semantic alignment
unstable routing
↓
slow specialization
```

That is much more mechanistic evidence than final reward alone.

---

# 39. I would add an initialization-time measurement

At:

[
t=0
]

before Stage 2 optimization, record:

* routing NMI
* phase-expert contingency
* entropy
* balance
* expert output divergence
* action MSE

Then record:

[
t=1,5,10,\ldots
]

This tells you whether the bootstrap literally creates what you claim it creates.

That is very powerful.

For example:

[
PhaseForge:
NMI_{0}=0.45
]

vs.

[
Random:
NMI_{0}\approx0.
]

Then you can watch both converge.

---

# 40. One thing I'd change in the "best results" mindset

You said you want to think like a researcher and get the best results.

The correct objective is **not**:

> maximize PhaseForge's score.

It is:

> maximize the probability that the result is actually correct.

Those are different.

For example, suppose:

[
PhaseForge = 0.65
]

and:

[
BC-large = 0.66.
]

That's frustrating if your goal is performance.

But scientifically, it may tell you that the proposed specialization mechanism doesn't add enough beyond capacity.

That is useful.

Likewise, if:

[
PhaseForge=0.63
]

and:

[
KMeans=0.64,
]

we shouldn't tune PhaseForge until it wins.

We should conclude that generic latent clustering may be enough.

That is exactly the researcher mentality you asked for.

---

# 41. Therefore, I would impose a "no result-dependent retuning" rule

Your plan already has explicit gates.

I would make one more rule:

> **Once Wave 1 configurations are finalized, no hyperparameter may be altered based on their evaluation performance before the confirmatory comparison.**

Otherwise the ablations slowly become optimization loops.

For exploratory sweeps such as jitter:

* search on seed 42
* choose nothing based on reward
* ideally use a predetermined grid
* report the whole curve.

That is much cleaner.

---

# 42. I would separate "mechanism validation" from "optimization"

This is perhaps the biggest conceptual improvement I'd make to the plan.

### Mechanism validation

These should be fixed:

* PhaseForge
* PhaseSup + random
* Plain + centroid
* PhaseSup + KMeans
* random/random
* centroid/random
* random/warmstart
* centroid/warmstart
* BC-large
* oracle
* teacher-forced.

These answer scientific questions.

### Optimization/sensitivity

Then:

* K=3/6/12
* jitter
* corruption
* spherical centroid
* fine-tuning.

Those answer robustness/design questions.

Do not let the second category determine which first-category model gets reported.

---

# 43. I would modify the execution order

Your current plan is:

> Gate 0 → infrastructure → CPU → GPU Wave 1 → Wave 2 → Wave 3.

That is good.

But I would make one adjustment:

### Before writing all new infrastructure

Lock the experiment matrix on paper.

Specifically define:

[
\boxed{
\text{What question does each cell answer?}
}
]

and:

[
\boxed{
\text{What exact difference between cells gives the causal contrast?}
}
]

Then implement only what is necessary.

This reduces the chance of building an experiment and realizing afterward that it isn't interpretable.

---

# 44. My proposed core confirmatory matrix

I would make this the central matrix:

| Cell               | Encoder          | Router             | Experts       | Purpose                                    |
| ------------------ | ---------------- | ------------------ | ------------- | ------------------------------------------ |
| BC                 | plain            | —                  | —             | behavior floor                             |
| BC-large           | plain            | —                  | —             | capacity control                           |
| Scratch-MoE        | random           | random             | random        | no pretraining                             |
| PS-RR              | phase-supervised | random             | warm          | H1                                         |
| Plain-Centroid     | plain            | centroid           | warm          | H2                                         |
| **PhaseForge**     | phase-supervised | **phase centroid** | **warm**      | proposed                                   |
| PS-KMeans          | phase-supervised | spherical KMeans   | warm          | generic clustering control                 |
| PS-Head            | phase-supervised | phase-head         | warm          | prototype vs discriminative initialization |
| PS-Random-Random   | phase-supervised | random             | random        | expert-init control                        |
| PS-Centroid-Random | phase-supervised | centroid           | random        | router-init control                        |
| Teacher            | phase-supervised | GT phase           | corresponding | privileged routing                         |
| Oracle             | —                | GT routing         | —             | diagnostic ceiling                         |

That's the heart of the paper.

---

# 45. Then the secondary matrix

### Expert-count:

[
K\in{3,6,12}
]

but with a principled prototype generation mechanism.

### Jitter:

[
\sigma\in{0,.005,.01,.02}
]

### Label noise:

[
p\in{0,.1,.25,.5,.75,1}.
]

### Spherical:

ordinary vs spherical centroid.

### Fine tuning:

frozen vs small-LR encoder.

These are excellent supplementary analyses.

---

# 46. I would add one more baseline if computationally affordable

This is more ambitious, so I'd put it below the mandatory experiments.

A **phase-conditioned dense policy**:

[
\pi(a|x,z)
]

during training/inference with GT phase.

Then compare:

[
\text{phase-conditioned dense}
]

against:

[
\text{PhaseForge}.
]

Why?

Because it answers:

> Is the benefit of the MoE specifically about specialization, or does phase information simply help any conditional policy?

However, this is only worth doing if the architecture can be constructed cleanly without introducing another unfair parameterization. It is not mandatory for the first wave.

---

# 47. One more experiment that could be extremely informative: shuffled phase names

This is subtly different from corrupted labels.

Take the phase labels and permute **their semantic identities consistently**:

[
1\rightarrow4,\quad
2\rightarrow6,\ldots
]

Then train PhaseForge.

Nothing should change, because the labels are categorical and only their equivalence classes matter.

This is a sanity check rather than a research result.

More important is **cross-sample phase permutation**, which destroys state-phase association.

I mention it because debugging label-based pipelines can otherwise produce surprisingly misleading results.

---

# 48. What I think about the plan's existing gates

The four gates are good.

### Gate 0

Claim/research positioning.

Good.

### Gate 1

Code correctness and bit-identical defaults.

**Excellent.**

This is especially important because the plan explicitly says old protocol cells are not modified. 

### Gate 2

CPU sanity.

Good.

### Gate 3

GPU completion + accounting.

Good.

### Gate 4

Analysis.

Good.

I would add one fifth conceptual gate:

### Gate 4.5 — Claim audit

Before writing conclusions, create a table:

| Claim | Evidence | Supports? | Caveat |
| ----- | -------- | --------- | ------ |

This prevents the paper from becoming stronger in language than in evidence.

---

# 49. The "bit-identical default" requirement is particularly important

The plan says:

> defaults must remain bit-identical to the frozen protocol.

That is exactly what I would demand.

Especially because you've already had the λ-decay issue.

The old protocol remains:

[
\lambda=1
]

constant.

Don't change:

* optimizer
* LR
* loss
* schedule
* epoch count
* evaluation
* phase labels
* seed handling

for the established baseline.

New experiments should be additive.

The plan explicitly preserves that. 

Keep it.

---

# 50. One thing I would do before any GPU experiment

Make a single machine-readable table containing:

```text
experiment_id
encoder_source
router_init
expert_init
expert_count
top_k
phase_supervision
phase_corruption
encoder_frozen
stage1_steps
stage2_steps
seed
parameter_count
active_parameter_count
```

Then generate the YAML/manifest from it or validate the YAML against it.

The purpose is to make impossible combinations fail early.

You already have preflight infrastructure. 

Extend it aggressively.

---

# 51. I would specifically make preflight reject these

### Invalid expert count

[
K < top_k.
]

### Invalid oracle

No phase labels available.

### Corrupted phase labels

Don't allow corrupted labels into teacher/oracle evaluation.

### K ≠ phase count

Require explicit prototype strategy.

### `phase_head`

Require phase-supervised Stage 1.

### `phase_centroid`

Require phase labels in Stage 1.

### `random/random`

Require explicit encoder source.

### BC-large

Require parameter ratio constraint.

This turns experimental mistakes into configuration errors rather than paper problems.

---

# 52. One important issue with "phase centroid" and KMeans leakage

Be careful about **which data are used to compute centroids**.

The plan says:

> training set through frozen encoder, mean-aggregate latents.

Good. 

Keep that.

Do not calculate centroids using validation or test data.

For KMeans as well:

[
KMeans(train\ latent\ only)
]

then evaluate.

If you use all data, you create information leakage.

This should be explicit in code and metadata.

---

# 53. For the KMeans baseline, fit KMeans once—not continually

The intended experiment is:

```text
Stage 1 frozen
      ↓
training latents
      ↓
KMeans
      ↓
router initialization
      ↓
Stage 2
```

Do not recompute clusters during Stage 2 unless that becomes a completely different experiment.

That would turn initialization into dynamic routing supervision.

Your current plan appears to intend one-shot initialization, which is correct.

---

# 54. Another subtle issue: random initialization must be matched in distribution

For `router_init=random`, don't accidentally initialize with a different scale than the normal PyTorch initialization used elsewhere.

Similarly, for random experts:

[
\text{random expert initialization}
]

must use the same architecture and initialization distribution as the corresponding scratch MoE.

Otherwise the random/random comparison isn't clean.

Record the initialization seed and distribution in the metadata.

---

# 55. Your warm-start jitter should probably be additive and relative to the actual copied parameters

You're currently using:

[
W_e=W_{action}+\epsilon,
\quad
\epsilon\sim N(0,0.02).
]

That's fine.

But note that an absolute:

[
\sigma=0.02
]

means different relative perturbations for layers with different weight scales.

So your jitter sweep measures **absolute noise scale**, not a normalized symmetry-breaking magnitude.

I would keep 0.02 because it's your existing protocol.

But in supplementary analysis, you may eventually want:

[
\sigma_l
========

\alpha\cdot
std(W_l)
]

as a relative perturbation.

Not necessary now.

---

# 56. I would not add that relative-jitter experiment yet

This is an example of where restraint matters.

You already have enough experiments.

Don't let the paper explode into:

[
K
\times
\sigma
\times
corruption
\times
centroid
\times
KMeans
\times
fine-tune.
]

Get the central causal result first.

---

# 57. The current plan's cost accounting is good, but there is a possible accounting inconsistency

The plan says Wave 1 contains:

* 7 methods × 3 seeds
* 18 Stage-2 runs
* 6 Stage-1 runs

and notes oracle trains Stage 2. 

But elsewhere the plan describes oracle as an:

> "offline diagnostic only"

That is potentially inconsistent.

You need to settle exactly whether `oracle_moe`:

1. trains a policy with oracle routing, or
2. takes an already-trained expert set and merely evaluates oracle routing.

Those are scientifically very different.

I would prefer the second for a routing-ceiling diagnostic **if your goal is to measure routing quality**, because otherwise oracle gets its own training procedure and becomes less comparable.

But the correct choice depends on what the current implementation actually does.

The plan should not contain both descriptions.

---

# 58. This is important enough to make a Gate 1 requirement

Write one sentence:

> **Oracle MoE is an evaluation-time routing intervention applied to a fixed trained expert set.**

or:

> **Oracle MoE is independently trained under ground-truth phase routing.**

Then stick to it.

Don't leave this implicit.

---

# 59. My recommended final evidence hierarchy

The eventual paper should not present 20 experiments as equally important.

I would structure evidence like this:

## Claim 1

**Phase supervision produces useful latent phase geometry.**

Evidence:

* phase accuracy
* linear head
* angular separation
* within/between phase distances

## Claim 2

**That geometry provides a useful routing initialization.**

Evidence:

* PhaseForge vs PS-random
* initial NMI
* entropy
* early learning curve

## Claim 3

**The benefit is not merely generic clustering.**

Evidence:

* PhaseForge vs PS-spherical-KMeans

## Claim 4

**The benefit is not merely expert warm-starting.**

Evidence:

* four-way initialization matrix

## Claim 5

**The benefit is not merely parameter count.**

Evidence:

* BC-large

## Claim 6

**Experts actually specialize.**

Evidence:

* phase × expert MSE
* expert divergence
* routing contingency

## Claim 7

**The learned router recovers useful privileged routing without needing the phase at inference.**

Evidence:

* PhaseForge vs teacher/oracle gap.

That is an extremely coherent paper.

---

# 60. What I would change in your actual plan

I would make the following changes before execution.

### Change 1 — Fix K=3/12 initialization

Do **not** allow:

[
K=12
]

to mean six initialized experts + six random experts.

Do **not** allow:

[
K=3
]

to have an undefined phase-to-expert mapping.

Define an explicit prototype construction for arbitrary K.

This is the biggest correction.

---

### Change 2 — Make spherical KMeans the principal generic-clustering control

Because your router is cosine-based.

Then ordinary KMeans can be secondary if you care about the distinction.

---

### Change 3 — Clarify oracle semantics

Remove the ambiguity between "offline diagnostic" and "oracle trains Stage 2."

---

### Change 4 — Predefine primary comparisons

At minimum:

[
PF > PS\text{-Random}
]

[
PF > Plain\text{-Centroid}
]

[
PF > PS\text{-KMeans}
]

[
PF > BC\text{-Large}.
]

---

### Change 5 — Tighten statistical language

Replace:

> controlled null with directional advantage

with something like:

> current estimates favor PhaseForge, but the present sample does not establish a statistically reliable advantage.

Until the new matrix is actually analyzed.

---

### Change 6 — Treat specialization as a mechanistic result

Do not infer it from NMI.

Your plan already says this. Keep that commitment. 

---

### Change 7 — Add initialization-time diagnostics

Measure the routing state **before Stage 2 training**.

This is crucial for proving the bootstrap mechanism.

---

# 61. What I would *not* change

I would not:

* replace the shared encoder;
* add another neural module;
* add contrastive loss;
* add recurrence;
* add router attention;
* add dynamic expert creation;
* abandon the phase-supervised Stage 1;
* abandon the frozen Stage-2 encoder;
* abandon the top-2 router;
* change λ;
* change the existing V0 protocol.

The uploaded plan is right to explicitly avoid architectural additions. 

---

# 62. The most important mental model for the experiment

Think of the whole project as trying to establish this causal chain:

```text
                 PRIVILEGED TRAINING SIGNAL
                         phase z
                           │
                           ▼
                    Stage-1 supervision
                           │
                           ▼
                   latent geometry h
                           │
                           ▼
               phase prototypes / clusters
                           │
                           ▼
                  initial router W
                           │
                           ▼
                  Stage-2 specialization
                     /          \
                    /            \
                   ▼              ▼
             routing quality   expert quality
                    \            /
                     \          /
                       ▼      ▼
                     control performance
```

Every arrow should have an experiment attached to it.

That is how I'd approach the work as a researcher rather than trying to maximize one benchmark number.

---

# 63. Final assessment of the uploaded plan

The plan is **good enough to become the execution plan**, but I would make the corrections above first.

In particular, I would consider these three non-negotiable:

[
\boxed{\textbf{1. Fix the K=3/12 experiment design.}}
]

[
\boxed{\textbf{2. Define oracle/teacher-forced routing exactly.}}
]

[
\boxed{\textbf{3. Lock the primary causal comparisons and statistical interpretation before GPU runs.}}
]

Everything else is secondary.

The strongest thing you have going for you is that the plan already understands the central novelty problem: generic centroid/cluster-based MoE initialization now has clear prior art, including CVPR 2026, while LAR-MoE establishes a recent robotics connection between latent structure and expert routing. ([Open Access CVF][2]) Your research therefore needs to win on **the specific privileged-regime transfer mechanism and the quality of the causal experimental evidence**, not on claiming that "centroids initialize routers."

And I think that is a viable research direction.

The next thing I would do is turn this review into a **locked experimental specification**: a definitive table of every experiment, its exact hypothesis, controlled variables, required checkpoint source, initialization rule, metrics, statistical comparison, and whether it is **mandatory / supplementary / discardable**. That would eliminate ambiguity before any GPU run.

[1]: https://bmvc2022.mpi-inf.mpg.de/736/?utm_source=chatgpt.com "Revisiting single-gated Mixtures of Experts"
[2]: https://openaccess.thecvf.com/content/CVPR2026/html/Chu_Enhancing_Mixture-of-Experts_Specialization_via_Cluster-Aware_Upcycling_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
[3]: https://arxiv.org/abs/2603.08476?utm_source=chatgpt.com "LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning"
[4]: https://arxiv.org/abs/2212.05055?utm_source=chatgpt.com "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints"
