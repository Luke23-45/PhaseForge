I did the literature audit with a fairly high bar. My conclusion is **not** “change the whole idea.” The core research direction is good, but **the novelty claim needs to be sharpened substantially**, and there are several experiments I would add before treating the current protocol as final.

The most important finding is that parts of your current mechanism already exist independently in the literature. In particular, **centroid/cluster-based router initialization from a pretrained latent space is not, by itself, novel**. Royer et al. proposed clustering pretrained embeddings to initialize an MoE gate and initializing experts from a pretrained base model in 2022/2023. ([bmvc2022.mpi-inf.mpg.de][1]) And much more recently, CVPR 2026's Cluster-aware Upcycling explicitly initializes both experts and the router from clusters of a pretrained model's activation space, including using cluster centroids as router weights. ([CVPR][2])

That does **not** kill PhaseForge. It tells us where the actual contribution has to live.

---

# 1. Where PhaseForge sits in the literature

There are really four research traditions converging here.

### A. Classical Mixture-of-Experts

The basic idea goes back much further than modern sparse Transformer MoEs. Jordan & Jacobs and Jacobs et al. established the general idea of learning several local experts with a gating mechanism that partitions the input space. The original motivation was precisely that a complicated mapping can be decomposed into simpler local mappings; Jordan & Jacobs even demonstrated the approach in a robot-dynamics domain. ([MIT Press Direct][3])

So:

> "Different regions/regimes should use different policies"

is definitely not itself novel.

That is the foundation upon which PhaseForge sits.

---

# 2. Modern sparse/noisy MoE

Shazeer et al. introduced the influential noisy top-k gating formulation: a learned gate produces scores, Gaussian noise is added during training, only the top-k experts are activated, and the gate is encouraged toward balanced utilization. ([arXiv][4])

Switch Transformer later popularized a simpler load-balancing auxiliary objective based on the product of routing frequency and average routing probability. ([arXiv][5])

Therefore your:

[
\text{noisy top-2}
+
L_{\text{balance}}
]

is established machinery.

There is no problem with using it. It simply should not be presented as the novel part.

---

# 3. Dense → MoE warm-starting

This is another important lineage.

Sparse Upcycling showed that an already-trained dense model can be converted into an MoE by reusing the dense checkpoint rather than learning the MoE from scratch. The original motivation was both better efficiency and better performance than training the sparse model from scratch. ([arXiv][6])

That's directly relevant to your:

[
\text{Stage-1 action head}
\rightarrow
\text{expert initialization}
]

So:

> "We initialize experts from a pretrained generalist"

is also not new by itself.

---

# 4. The first major warning: centroid routing already exists

This is the first place where I would change how you describe PhaseForge.

Royer et al. proposed **per-sample clustering-based initialization**. Their procedure explicitly trains a base model, clusters its embeddings with K-means, uses the resulting centroids to define the initial gate, and initializes experts from the base model. ([arXiv][7])

That is surprisingly close to:

```text
Stage-1 encoder
      ↓
latent embeddings
      ↓
phase centroids
      ↓
router
```

So you cannot safely claim:

> "PhaseForge is the first method to initialize an MoE router from latent centroids."

That claim would be false.

---

# 5. It gets even more important in 2026

The CVPR 2026 paper **Cluster-aware Upcycling** makes the overlap even more explicit.

Their method:

1. takes a pretrained dense network,
2. clusters activation representations,
3. initializes experts from cluster-specific subspaces,
4. initializes router weights with the cluster centroids,
5. uses this structure to break expert symmetry and encourage early specialization. ([CVPR][2])

They even use **spherical K-means**, which is particularly relevant to your cosine router. ([ResearchGate][8])

So the general principle:

[
\boxed{
\text{pretrained latent structure}
\rightarrow
\text{clusters}
\rightarrow
\text{router initialization}
}
]

is now clearly established.

This is the single biggest thing I would fix before writing the paper.

---

# 6. But this is where PhaseForge becomes interesting

Your method is doing something different:

```text
                    privileged phase
                         │
                         ▼
State → Encoder → phase-structured latent space
                         │
                         ▼
                  phase centroids
                         │
                         ▼
                    MoE router
```

The crucial ingredient isn't merely "clusters."

It is:

> **privileged semantic/regime information is deliberately used to shape the representation, and that representation is then transferred into the routing mechanism.**

Your phase labels aren't just discovering arbitrary clusters.

You explicitly tell the encoder:

[
h=E(x)
]

must support both:

[
a \approx \hat a
]

and

[
z \approx \hat z.
]

That gives the latent space a specific semantic interpretation.

Then you transfer that structure to routing.

That is the much stronger research story.

---

# 7. And there is a very interesting 2026 paper you absolutely need to discuss

The March 2026 **LAR-MoE** paper is probably the closest current work I found in terms of the *high-level conceptual question*. It introduces latent-aligned routing for robotic imitation learning.

Their pipeline learns a joint latent representation of observations and future actions, then regularizes expert routing to follow that latent structure. Importantly, they don't require explicit phase annotations. They report strong simulation and hardware results and explicitly compare against a supervised MoE baseline. ([arXiv][9])

This actually gives us a very useful positioning opportunity.

Their question is approximately:

> Can the latent space discover useful behavioral structure without explicit phase labels?

Your question can be:

> **When privileged phase information is available during training, can we deliberately encode that structure into the latent geometry and use it to initialize a specialized control MoE?**

Those are related but different.

And this creates a potentially excellent scientific comparison:

[
\boxed{\text{unsupervised latent structure}}
]

versus

[
\boxed{\text{privileged phase-supervised latent structure}}
]

That is much stronger than pretending nobody has done latent-guided MoE routing.

---

# 8. I would therefore change the central claim

I would **not** make this the claim:

> "We introduce centroid-based MoE initialization."

Too weak and already substantially covered.

I would move toward something like:

> **PhaseForge investigates whether privileged regime information available during training can be converted into useful latent geometry and transferred into an MoE routing prior, enabling specialized control without requiring the privileged phase signal at inference.**

That is much harder to dismiss.

The conceptual object becomes:

[
\boxed{
\text{Privileged Regime Geometry Transfer}
}
]

rather than merely:

[
\boxed{
\text{Centroid Bootstrap}
}
]

The centroid bootstrap is then the mechanism implementing the idea.

---

# 9. The literature actually makes your two hypotheses more interesting

Your current H1/H2 are:

### H1

Does centroid initialization help?

### H2

Does phase supervision make centroid initialization better?

I think H2 is the more scientifically interesting question.

H1 is largely supported by the broader MoE literature already.

H2 asks something more specific:

[
\text{Does semantic structure learned with privileged information produce better routing priors?}
]

That is where I would spend the research effort.

---

# 10. There is an important missing baseline

Right now you have:

| Encoder          | Router initialization |
| ---------------- | --------------------- |
| phase-supervised | centroid              |
| phase-supervised | random                |
| BC               | centroid              |
| BC               | random                |

Good.

But you are missing:

[
\boxed{\text{phase-supervised encoder + unsupervised clustering router}}
]

For example:

```text
Phase-supervised encoder
        ↓
latent space
        ↓
K-means / spherical K-means
        ↓
centroids
        ↓
router
```

Why is this important?

Because right now someone can say:

> "Maybe all you need is a good representation. The ground-truth phase labels aren't actually necessary."

This experiment answers that.

---

# 11. I would make the factorial even stronger

I'd actually think about the design as:

|                          | Random | Unsupervised cluster | Phase centroid |
| ------------------------ | -----: | -------------------: | -------------: |
| BC encoder               |      ✓ |                    ✓ |              ✓ |
| Phase-supervised encoder |      ✓ |                    ✓ | **PhaseForge** |

Now you have much more information.

You can ask:

### Does good representation matter?

Compare:

[
BC + centroid
]

vs.

[
PhaseSup + centroid
]

### Does centroid routing matter?

Compare:

[
PhaseSup + random
]

vs.

[
PhaseSup + centroid
]

### Does privileged phase information matter specifically for routing?

Compare:

[
PhaseSup + unsupervised\ cluster
]

vs.

[
PhaseSup + phase\ centroid
]

That is an excellent ablation structure.

---

# 12. Another baseline I strongly recommend: phase-head initialization

You deliberately made the phase head linear.

That gives you something scientifically valuable that you aren't currently exploiting.

You have:

[
\text{phase head: } W_p h+b_p
]

Those six rows of (W_p) are themselves learned phase-discriminative directions.

So you should test:

### Centroid initialization

[
W_R^{(i)}
=========

\frac{c_i}{|c_i|}
]

versus:

### Phase-head initialization

[
W_R^{(i)}
=========

\frac{W_{p,i}}{|W_{p,i}|}
]

This is a beautiful experiment because it asks:

> Is the best routing prior a **generative/prototypical representation of each phase** or the **discriminative representation learned by the classifier**?

I would absolutely run this.

It also makes the linear phase head much more justified architecturally.

---

# 13. Another thing I would change/test: spherical centroids

This is a technical issue that matters.

Your router is based on cosine similarity.

You currently compute:

[
c_k = \frac1{N_k}\sum_i h_i
]

then normalize (c_k).

But cosine geometry cares about **directions**, not magnitudes.

Suppose:

[
h_1 = 10v
]

and another sample is:

[
h_2=v
]

The first contributes ten times as much to the raw centroid.

That may be undesirable.

Since your router effectively operates on the sphere, I'd test:

[
\tilde h_i=\frac{h_i}{|h_i|}
]

then:

[
c_k =
\frac1{N_k}\sum_{i:z_i=k}\tilde h_i
]

then normalize:

[
\tilde c_k=
\frac{c_k}{|c_k|}
]

This is essentially a spherical/prototype formulation and is more geometrically consistent with your cosine router. The fact that the 2026 Cluster-aware Upcycling work explicitly uses spherical K-means makes this worth investigating rather than ignoring. ([ResearchGate][8])

I wouldn't automatically change your main implementation yet.

I would run the ablation.

---

# 14. The biggest experimental issue I see: parameter fairness

This is potentially more serious than any architectural detail.

PhaseForge has:

```text
6 experts
```

while BC has:

```text
1 network
```

Therefore PhaseForge has substantially greater parameter capacity.

If PhaseForge beats BC, that does **not** establish that routing or phase structure caused the improvement.

You need a parameter-matched dense baseline.

For example:

```text
BC-small
BC-large
PhaseForge
```

where:

[
Params(BC\text{-large})
\approx
Params(PhaseForge)
]

Then you're asking:

> Does conditional specialization beat simply giving one dense network the same capacity?

That is a very important comparison.

---

# 15. I would also match computation/training budget

Your protocol has multiple stages.

PhaseForge gets:

[
100\text{ epochs Stage 1}
+
200\text{ epochs Stage 2}
]

A scratch MoE doesn't have that same amount of optimization history.

So someone could argue:

> "PhaseForge wins because it receives 100 epochs of useful pretraining."

That's not necessarily a problem—pretraining is part of your method—but you need to distinguish:

[
\text{benefit of pretraining}
]

from:

[
\text{benefit of phase-aware routing initialization}.
]

Your `warmstart_moe` helps, but I would make the accounting extremely explicit.

At minimum report:

* total optimizer steps
* total training examples seen
* parameter count
* approximate training FLOPs
* inference FLOPs
* number of active expert parameters per sample

That prevents a reviewer from finding an obvious fairness objection.

---

# 16. Expert specialization needs a more direct measurement

NMI is useful.

But NMI only tells you:

[
\text{phase assignment}
\leftrightarrow
\text{expert assignment}
]

It does **not** tell you whether experts actually became different policies.

This is important.

You should directly measure:

[
\boxed{\text{expert behavioral specialization}}
]

For every expert (e), evaluate it independently across every phase.

Construct:

[
M_{z,e}
=======

MSE(\pi_e(x),a)
]

where rows are phase and columns are experts.

You then get a 6×6 matrix:

|         | E1 | E2 | E3 | E4 | E5 | E6 |
| ------- | -: | -: | -: | -: | -: | -: |
| Phase 1 |  ↓ |    |    |    |    |    |
| Phase 2 |    |  ↓ |    |    |    |    |
| Phase 3 |    |    |  ↓ |    |    |    |
| ...     |    |    |    |    |    |    |

If the method genuinely specializes, you should see structured differences.

That's far stronger evidence than t-SNE plots or NMI alone.

---

# 17. I would add a counterfactual routing experiment

This could become one of the strongest figures in the paper.

For every state, compare:

[
\pi_{\text{learned routing}}
]

against:

[
\pi_{\text{oracle phase routing}}
]

against:

[
\pi_{\text{random routing}}
]

against:

[
\pi_{\text{uniform routing}}
]

This lets you quantify:

> How much of the achievable phase-conditioned specialization is recovered by the learned router?

Define, for example:

[
Gap =
\frac{
Performance_{learned}-Performance_{random}
}{
Performance_{oracle}-Performance_{random}
}
]

You don't have to use exactly this metric, but something along these lines would be extremely informative.

---

# 18. Your teacher-forced baseline can become much more useful

Right now teacher_forced sounds like a baseline.

I would treat it as a **diagnostic ceiling**.

You effectively have:

```text
Oracle phase
      ↓
expert
      ↓
action
```

and:

```text
learned router
      ↓
expert
      ↓
action
```

The gap between them tells you how good the router is at recovering the useful decomposition.

This gives you a very nice mechanistic story:

[
\text{Phase supervision}
\rightarrow
\text{latent geometry}
\rightarrow
\text{routing}
\rightarrow
\text{performance}
]

and you can observe each transition.

---

# 19. A particularly powerful experiment: corrupt the phase labels

I strongly recommend this.

Use:

```text
0% corruption
10%
25%
50%
75%
100%
```

or something similar.

Then see what happens.

If PhaseForge works because phase supervision supplies meaningful structure, you'd expect a degradation curve.

For example:

[
Performance(\text{phase quality})
]

This would test whether your method is:

* robust to imperfect privileged information,
* completely dependent on perfect simulator phases,
* or surprisingly tolerant to noisy phase labels.

That is scientifically much more interesting than another random seed.

It also connects naturally to the broader privileged-information literature, where the distinction between useful privileged signals and imperfect teachers is an active issue. ([ScienceDirect][10])

---

# 20. Another strong experiment: phase count ≠ expert count

Right now:

[
6\ phases = 6\ experts
]

That is convenient, but it creates a potential criticism:

> "Of course the method works—the architecture is explicitly given one expert per phase."

Even though your router is not frozen to phase assignments, a reviewer can still suspect this.

So I'd test:

[
K_{experts}\in{3,4,6,8,12}
]

while keeping six phase labels.

This gives very interesting possibilities.

### 3 experts

Multiple phases must share experts.

### 6 experts

One-to-one initialization.

### 12 experts

A phase can subdivide into multiple experts.

This could reveal whether PhaseForge is actually discovering **sub-phase structure** rather than merely reproducing the six phase labels.

That could make the research substantially stronger.

---

# 21. In fact, I think K > 6 could be particularly interesting

Imagine:

```text
6 simulator phases
12 experts
```

Then centroid initialization could duplicate or split phase prototypes.

During training, the router might discover:

```text
Phase 2
   ├── Expert 2a
   └── Expert 2b
```

That would support a more interesting interpretation:

> Phase supervision provides a coarse semantic scaffold, while the MoE discovers finer control-specialization structure.

That is much more interesting than:

> six phases → six experts.

I would definitely investigate this.

---

# 22. You should also test whether the learned phase representation is actually geometrically meaningful

Don't just report phase-head accuracy.

Measure:

[
\text{within-phase distance}
]

and:

[
\text{between-phase distance}.
]

For example:

[
D_{within}
==========

E[|h_i-h_j|,|,z_i=z_j]
]

and:

[
D_{between}
===========

E[|h_i-h_j|,|,z_i\neq z_j].
]

Better still, because your router is cosine-based, use angular distances.

Then compare:

```text
BC encoder
Phase-supervised encoder
```

If the phase-supervised encoder produces much clearer directional separation, you have direct evidence supporting your bootstrap premise.

---

# 23. I would be careful with t-SNE/UMAP

Use them as illustrations, not evidence.

A beautiful plot where six phases form six clusters can be visually persuasive but scientifically weak.

Your stronger evidence is:

[
\text{linear phase accuracy}
]

[
\text{silhouette / separation}
]

[
I(Z;E)
]

[
\text{phase-conditioned action error}
]

[
\text{expert specialization}
]

and actual control performance.

Use visualization to explain those results, not substitute for them.

---

# 24. There is another important competing philosophy: don't use phase labels at all

LAR-MoE is relevant precisely here.

It asks whether latent structure can arise without explicit phase annotations and then be aligned with expert routing. ([arXiv][9])

This creates a compelling research axis:

```text
                         expert routing
                              ↑
                              │
                   ┌──────────┴──────────┐
                   │                     │
             privileged              discovered
             structure                structure
                   │                     │
              PhaseForge              LAR-MoE-like
```

You don't necessarily need to reproduce their full architecture.

But **you absolutely need to discuss this line of work**.

Otherwise a reviewer familiar with recent robotics MoE literature will see the relationship immediately.

---

# 25. Your privileged-information story has strong precedent

Using information available in simulation during training but unavailable at deployment is established in robotics.

A prominent locomotion line uses privileged simulator/environment information during learning and trains policies that operate without it at deployment; surveys of learning-based locomotion describe this teacher/student and privileged-information paradigm explicitly. ([Sage Journals][11])

More recent work continues to formalize privileged-information learning and its limitations. ([ML Anthology][12])

So again:

> privileged information during training

is not novel.

Your potential novelty is **what you do with the privileged signal**:

[
\text{privileged phase}
\rightarrow
\text{latent geometry}
\rightarrow
\text{routing prior}.
]

That's the piece I'd defend.

---

# 26. I would not add random architectural complexity

This is where I disagree with the instinct of many research projects.

I would **not** immediately add:

* attention to the router,
* another transformer,
* recurrent router,
* complicated meta-learning,
* auxiliary contrastive losses,
* dynamic expert numbers,
* several more regularizers.

That would make the paper harder to understand.

Your mechanism has a nice causal structure already.

The goal should be:

[
\boxed{\text{simple mechanism + exceptionally strong evidence}}
]

rather than:

[
\text{complicated mechanism + weak ablations}.
]

---

# 27. I would keep the basic two-stage structure

I actually like this part.

Stage 1:

[
\boxed{
\text{learn useful latent geometry}
}
]

Stage 2:

[
\boxed{
\text{specialize experts / router}
}
]

Freezing the encoder in the **primary mechanistic experiment** is also defensible because it prevents Stage 2 from moving the representation and making causal attribution messy.

I'd keep that.

---

# 28. But I would add a fine-tuning variant separately

There is an obvious question:

> What happens if the encoder is allowed to adapt during Stage 2?

So make:

[
PhaseForge\text{-Frozen}
]

your mechanistic model.

Then optionally:

[
PhaseForge\text{-FT}
]

where the encoder has a small learning rate.

For example:

[
LR_{encoder}
<
LR_{router},LR_{experts}.
]

If FT gives better absolute performance, great.

But the paper's causal claims should be anchored in the frozen version.

---

# 29. Another potential issue: are phase labels actually the right decomposition?

This is a deeper scientific question.

You currently assume:

[
\text{simulator phase}
\approx
\text{useful policy regime}.
]

That may be true.

But it may also not be.

A simulator's phase partition could be physically meaningful while the control policy has a different natural decomposition.

That is why the post-training routing analysis matters.

Suppose:

[
NMI(Z,E)=0.3
]

but performance improves substantially.

That could mean your experts are finding control regimes that differ from simulator phases.

That is not necessarily failure.

It could be evidence that phase labels are a useful **initialization scaffold**, but not the final expert partition.

---

# 30. This suggests a better conceptual hypothesis

Instead of:

> experts should correspond to phases.

I'd say:

> **phase structure should provide a useful prior over the latent partition of the control problem.**

That is much safer.

Mathematically:

[
P(E|h)
\leftarrow
P(Z|h)
]

initially,

but later:

[
P(E|h)
]

is optimized independently for the action objective.

This avoids overclaiming one-to-one phase/expert equivalence.

---

# 31. The initialization itself could be studied as a trajectory

This is another figure I would want.

Plot across Stage 2:

[
\text{NMI}
]

[
\text{routing entropy}
]

[
\text{expert balance}
]

[
\text{action MSE}
]

against training epoch.

You want to see something like:

```text
Initialization
     │
     ├── high phase alignment
     ├── balanced experts
     └── reasonable action performance
             ↓
       specialization evolves
             ↓
       stable routing
             ↓
      lower action error
```

This would make your mechanism visible.

---

# 32. Another very important ablation: remove the jitter

You already understand why it should matter.

So prove it.

Compare:

[
\sigma=0
]

versus:

[
\sigma=0.02.
]

Possibly:

[
\sigma \in
{0,0.005,0.01,0.02,0.05}.
]

This gives you an actual empirical answer to the symmetry-breaking argument.

And importantly, check not only final MSE but:

[
\text{expert divergence}
]

and:

[
\text{routing entropy/NMI}.
]

---

# 33. Another ablation: router initialization only vs expert initialization only

This is crucial for causal interpretation.

You currently change two things at once at the Stage-1 → Stage-2 transition:

1. router gets centroids
2. experts get the Stage-1 action head

So create:

### A

Random router + random experts

### B

Centroid router + random experts

### C

Random router + warm-started experts

### D

Centroid router + warm-started experts

Then:

[
D-A
]

is total initialization benefit.

And:

[
B-A
]

isolates router initialization.

[
C-A
]

isolates expert initialization.

[
D-B
]

tells you whether expert warm-start adds value on top of the router bootstrap.

This is **much stronger** than only PhaseForge vs random-router.

---

# 34. I would call this a mandatory ablation

Because otherwise a reviewer can say:

> "You claim the centroid bootstrap is the core contribution, but maybe the pretrained experts are actually responsible for most of the improvement."

This four-way matrix answers it immediately.

---

# 35. There is also a very interesting relation to expert symmetry

Sparse Upcycling showed the benefit of reusing dense weights. Recent Cluster-aware Upcycling specifically identifies identical expert initialization plus random routing as a source of poor early specialization and proposes semantic cluster-aware initialization to break that symmetry. ([arXiv][6])

Your jitter idea is therefore reasonable, but it is **not enough to claim a novel symmetry-breaking mechanism**.

What may be novel is the specific source of the semantics:

[
\boxed{\text{privileged control phase}}
]

rather than generic activation clustering.

---

# 36. That gives us an excellent paper structure

I'd conceptually organize the contribution around three layers:

### Layer 1 — Privileged geometry

Phase supervision forces control latents to encode regime structure.

### Layer 2 — Geometry-to-router transfer

Phase prototypes initialize the router.

### Layer 3 — Autonomous specialization

Once Stage 2 begins, privileged phase labels disappear from routing and the MoE learns its own control decomposition.

That is cleaner than presenting "centroids + jitter + top-2 + balance loss" as one giant contribution.

---

# 37. What I think should stay exactly as it is

At this point, I would preserve these core choices:

[
\boxed{
\text{19-D state}
\rightarrow
\text{shared encoder}
}
]

[
\boxed{
L_{Stage1}=L_{action}+L_{phase}
}
]

[
\boxed{
\text{linear phase head}
}
]

[
\boxed{
\text{phase centroids}
}
]

[
\boxed{
\text{cosine router initialization}
}
]

[
\boxed{
\text{warm-start experts}
}
]

[
\boxed{
\text{frozen encoder in primary Stage 2}
}
]

[
\boxed{
\text{top-2 noisy routing}
}
]

[
\boxed{
\text{load balancing}
}
]

The *architecture* is already coherent.

I wouldn't redesign it yet.

---

# 38. What I would change before the final experimental run

My priority order would be:

### Highest priority

**1. Add parameter-matched dense baselines.**

This protects against the simplest capacity objection.

**2. Add the four-way initialization ablation.**

```text
random/random
centroid/random
random/warmstart
centroid/warmstart
```

This tells us exactly where the gain comes from.

**3. Add phase-supervised + unsupervised-clustering routing.**

This isolates the value of explicitly using phase labels.

**4. Compare phase centroid vs phase-head initialization.**

This tests whether the particular bootstrap is actually optimal.

**5. Measure expert specialization directly.**

Use phase × expert action-error matrices.

---

# 39. Second priority

Then I would add:

[
K={3,6,12}
]

for number of experts.

And:

[
\sigma={0,0.005,0.01,0.02}
]

for expert jitter.

And:

[
\text{phase-label corruption}
]

for robustness.

These don't all need to become headline experiments. Some can go into supplementary material.

---

# 40. Third priority

Then:

* spherical centroid vs ordinary centroid
* frozen vs lightly fine-tuned encoder
* router entropy trajectories
* learned-router vs oracle-router
* phase/expert confusion matrix
* expert pairwise behavioral divergence

These make the mechanism much more convincing.

---

# 41. One experiment I would consider exceptionally valuable

This one:

### Phase-supervised encoder → random router

versus

### Phase-supervised encoder → phase-centroid router

versus

### Phase-supervised encoder → spherical-KMeans router

versus

### phase-supervised encoder → phase-head-weight router.

Why?

Because then we're no longer merely demonstrating:

> "Our particular bootstrap works."

We are investigating:

> **What is the best way to convert privileged phase geometry into an MoE routing prior?**

That becomes a legitimate research question in its own right.

---

# 42. What I would *not* claim until these experiments are done

I would not yet claim:

> "PhaseForge discovers phase-specialized experts."

You haven't established that strongly enough from NMI.

I'd say:

> "PhaseForge initializes routing from phase-structured latent geometry and enables subsequent expert specialization."

Then prove specialization with the expert × phase behavioral matrix.

---

# 43. What I think the strongest eventual result would look like

The ideal experimental story is not simply:

[
PhaseForge > BC.
]

That's too easy.

The much stronger result would be:

[
PhaseForge

>

PhaseSup+RandomRouter
]

showing initialization matters,

and:

[
PhaseForge

>

PlainBC+Centroid
]

showing phase-supervised geometry matters,

and:

[
PhaseForge

>

PhaseSup+UnsupervisedCluster
]

showing semantic phase information matters beyond generic clustering,

and:

[
PhaseForge

>

ParameterMatchedDense
]

showing specialization matters beyond capacity,

while:

[
OracleMoe

>

PhaseForge
]

but only modestly, showing the learned router is recovering much of the available phase-conditioned benefit.

That would be a **very convincing mechanistic paper**.

---

# 44. And the really compelling version

Then suppose you additionally find:

[
K=12
]

outperforms:

[
K=6
]

and the experts subdivide phases into meaningful behavioral regions.

Then the story becomes even more interesting:

> Phase supervision does not simply assign six experts to six simulator phases. It provides a coarse structural prior from which the MoE discovers a finer-grained control decomposition.

That is substantially more scientifically interesting.

---

# 45. My current verdict

I would **not abandon the current PhaseForge architecture**.

But I would absolutely change our understanding of what the paper's novelty is.

### Weak claim

> Centroid-initialized MoE for control.

Already too close to existing work. ([arXiv][7])

### Better claim

> Phase-supervised latent representation + centroid routing.

Interesting, but needs careful positioning against LAR-MoE and other latent-routing work. ([arXiv][9])

### Strongest direction

> **A privileged-regime-to-routing mechanism: use training-only phase information to shape the geometry of a control representation, transfer that geometry into an MoE routing prior, and then remove the privileged signal so the MoE autonomously discovers its final specialization.**

That is the direction I would pursue.

And I would make the research question:

[
\boxed{
\textbf{Can privileged regime structure be transferred into a deployable expert-routing prior?}
}
]

rather than:

[
\boxed{
\text{Can we initialize an MoE with centroids?}
}
]

That distinction is extremely important.

---

## Literature that I would consider core to the eventual paper

| Work                                       | What it establishes                                                      | Relationship to PhaseForge                                 |
| ------------------------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------------------- |
| Jacobs et al., 1991                        | Adaptive mixtures of local experts                                       | Foundational expert specialization                         |
| Jordan & Jacobs, 1994                      | Hierarchical MoE, including robot dynamics                               | Classical regime decomposition                             |
| Shazeer et al., 2017                       | Noisy top-k sparse gating                                                | Basis for your router                                      |
| Switch Transformer, 2021                   | Load-balancing objective                                                 | Basis for your balance term                                |
| Royer et al., BMVC 2022                    | Cluster pretrained embeddings to initialize MoE gate; warm-start experts | **Direct prior on centroid bootstrap**                     |
| Sparse Upcycling, ICLR 2023                | Dense → MoE warm-start                                                   | Prior for expert warm-start                                |
| Cluster-aware Upcycling, CVPR 2026         | Cluster activations + centroid router + specialized experts              | **Very close generic initialization precedent**            |
| Privileged-information robotics literature | Training can use information unavailable at deployment                   | Supports your training-only phase signal                   |
| LAR-MoE, 2026                              | Latent structure can guide robotic MoE routing without phase labels      | **Closest recent robotics conceptual comparator**          |
| Phase-conditioned IL, 2026                 | Explicit phase-conditioned policies can improve robotic manipulation     | Shows phase is actively used in modern IL, but differently |

([ResearchGate][13])

---

## My recommendation

**Keep the architecture. Do not rush into changing it.**

But before we lock the experiment protocol, I would redesign the **experimental matrix** around the causal question above. The most important additions are:

[
\boxed{
\text{phase-supervision}
\times
\text{routing initialization}
\times
\text{expert initialization}
}
]

plus a **parameter-matched dense baseline**, **unsupervised clustering baseline**, **phase-head initialization**, and **direct expert specialization analysis**.

Those additions are much more valuable than making the network more complicated.

One final warning: because **CVPR 2026 Cluster-aware Upcycling** and **LAR-MoE (March 2026)** are both now part of the literature, I would **not freeze the paper's novelty statement yet**. The right next step is to do a second, much more exhaustive pass specifically on **phase/regime decomposition in control, hierarchical/modular policies, latent-space routing, privileged information, and recent 2025–2026 robotics MoE work**, then build a formal **“what has already been done vs. exactly what PhaseForge adds”** table. That is where I think we can turn this from a merely good architecture into a defensible research contribution.

[1]: https://bmvc2022.mpi-inf.mpg.de/736/ "https://bmvc2022.mpi-inf.mpg.de/736/"
[2]: https://openaccess.thecvf.com/content/CVPR2026/html/Chu_Enhancing_Mixture-of-Experts_Specialization_via_Cluster-Aware_Upcycling_CVPR_2026_paper.html "https://openaccess.thecvf.com/content/CVPR2026/html/Chu_Enhancing_Mixture-of-Experts_Specialization_via_Cluster-Aware_Upcycling_CVPR_2026_paper.html"
[3]: https://direct.mit.edu/neco/article-pdf/6/2/181/812708/neco.1994.6.2.181.pdf "https://direct.mit.edu/neco/article-pdf/6/2/181/812708/neco.1994.6.2.181.pdf"
[4]: https://arxiv.org/abs/1701.06538 "https://arxiv.org/abs/1701.06538"
[5]: https://arxiv.org/abs/2101.03961 "https://arxiv.org/abs/2101.03961"
[6]: https://arxiv.org/abs/2212.05055 "https://arxiv.org/abs/2212.05055"
[7]: https://arxiv.org/abs/2304.05497 "https://arxiv.org/abs/2304.05497"
[8]: https://www.researchgate.net/publication/403867013_Enhancing_Mixture-of-Experts_Specialization_via_Cluster-Aware_Upcycling "https://www.researchgate.net/publication/403867013_Enhancing_Mixture-of-Experts_Specialization_via_Cluster-Aware_Upcycling"
[9]: https://arxiv.org/abs/2603.08476 "https://arxiv.org/abs/2603.08476"
[10]: https://www.sciencedirect.com/science/article/pii/S0950705125003855 "https://www.sciencedirect.com/science/article/pii/S0950705125003855"
[11]: https://journals.sagepub.com/doi/10.1177/02783649241312698 "https://journals.sagepub.com/doi/10.1177/02783649241312698"
[12]: https://mlanthology.org/neurips/2024/cai2024neurips-provable/ "https://mlanthology.org/neurips/2024/cai2024neurips-provable/"
[13]: https://www.researchgate.net/publication/233806999_Adaptive_Mixtures_of_Local_Experts "https://www.researchgate.net/publication/233806999_Adaptive_Mixtures_of_Local_Experts"
