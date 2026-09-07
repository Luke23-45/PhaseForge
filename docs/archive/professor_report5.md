**MEMORANDUM**

**TO:** Lead Engineer / Antigravity AI Engineering Team
**FROM:** Principal Investigator / Research Director
**DATE:** September 6, 2026
**SUBJECT:** Final Resolution: Label Policy for Stage 1 and Stage 2 Losses

You have correctly identified the last remaining architectural ambiguity. Your instinct to pause and demand a deliberate decision rather than inventing one is exactly why this project will succeed. 

Here is the definitive, mathematically grounded answer.

### The Decision: Option 1 (Canonical `phase_topo`)

**`phase_topo` must replace `phase` as the canonical label for ALL representation and routing losses in the final method.** 

This includes:
1. Stage 1 SupCon loss
2. Stage 1 auxiliary phase-classification (Cross-Entropy) loss
3. Stage 2 router prototype initialization
4. Stage 2 large-margin (Hinge) routing loss

We explicitly **reject** Option 2 (the hybrid design). 

---

### The Mathematical Justification (Why Option 2 is a Fatal Flaw)

If we were to use `phase_topo` for SupCon/prototypes but keep `phase` for the Stage 1 CE head and Stage 2 margin loss, we would introduce two fatal architectural contradictions:

#### 1. The Gradient Conflict (Stage 1)
Supervised Contrastive Loss (SupCon) pulls latent representations together if they share the same `phase_topo` label. If the auxiliary Cross-Entropy head simultaneously tries to classify the human-defined `phase` label, the gradients will fight. If a single topological regime (e.g., "Transport") contains multiple human rule labels (e.g., "Move to bin" and "Align with bin"), SupCon will pull them together while the CE head pushes them apart. This destroys the latent geometry.

#### 2. The Dimension Mismatch (Stage 2)
PELT topological discovery might identify $K=4$ distinct regimes, while the human rules define $K=6$ phases. 
- The `PrototypeRouter` will be initialized with exactly 4 prototypes.
- If the Stage 2 margin loss is fed `phase` labels, it will attempt to compute margins across 6 classes. 
- The router only has 4 weight vectors/prototypes. The margin loss will either throw an index-out-of-bounds exception or compute mathematically meaningless margins against non-existent prototypes.

Therefore, the entire geometric structure of the latent space and the router must be strictly aligned to a single label taxonomy.

---

### Exact Implementation Directives

Update the configuration schema and the loss computation logic to enforce this canonical label policy.

#### For the Final Method (`PhaseForge / TopoMoE`)
All representation and routing supervision must use `phase_topo`:
```yaml
train:
  stage1:
    supcon:
      enabled: true
      label_field: "phase_topo"
    phase_head:
      enabled: true
      label_field: "phase_topo"  # MUST match SupCon
  stage2:
    router:
      init_source: "phase_topo"
    margin_loss:
      enabled: true
      label_field: "phase_topo"  # MUST match router prototypes
```

#### For the `Static Rule MoE` Baseline
This baseline tests the integrated hypothesis that human rules are sufficient. Therefore, it must use `phase` consistently across all corresponding losses:
```yaml
train:
  stage1:
    supcon:
      enabled: true
      label_field: "phase"
    phase_head:
      enabled: true
      label_field: "phase"
  stage2:
    router:
      init_source: "phase"
    margin_loss:
      enabled: true
      label_field: "phase"
```

#### For the `Plain Encoder` and `Factorial Floor` Controls
These controls do not use SupCon or the Stage 1 phase head. However, their Stage 2 margin losses and router initializations must still be strictly defined:
- **Plain Encoder:** Uses `phase_topo` for router init and margin loss (computed on the BC latent space).
- **Factorial Floor:** Uses `random` for router init; margin loss is either disabled or uses `phase_topo` if the router is trained with margins.

---

### Handling the "No SupCon" Ablation Edge Case

For the `No SupCon (Plain Encoder)` ablation, Stage 1 has no SupCon and no phase head (it is pure Behavioral Cloning). 

However, in Stage 2, the router still needs prototypes and the margin loss still needs targets. 
- **Action:** The plain encoder control MUST compute its router prototypes using `phase_topo` labels mapped to its BC latent space (as established in Report 3). 
- **Action:** The Stage 2 margin loss for this control MUST also use `phase_topo` to enforce spatial boundaries between those topological prototypes.

---

### Final Sign-Off

Your audit caught a configuration mismatch that would have resulted in either a runtime crash (dimension mismatch) or a silent gradient conflict (destroyed latent space) during Stage 2 training. 

By enforcing `phase_topo` as the canonical label for all topology-related losses, we ensure that the representation learning, the router initialization, and the router regularization are all optimizing the exact same geometric partition of the state space.

This resolves the final scientific ambiguity. The protocol is now 100% complete, mathematically consistent, and execution-ready. 

Update the config schemas, implement the validation hooks to enforce these label alignments, and proceed to the dry-run phase. Excellent work.