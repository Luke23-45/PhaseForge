**FINAL EXECUTION-READY PROTOCOL**
**Precision-Residual PhaseForge: Complete Baseline & Ablation Specification**

**TO:** Lead Engineer / Antigravity AI Engineering Team
**FROM:** Principal Investigator / Research Director
**DATE:** September 6, 2026
**STATUS:** Final. Supersedes all previous reports. Execution-ready.

---

## 0. Executive Summary

This document resolves every remaining ambiguity identified in your audit. It answers all four outstanding questions, specifies exact code changes (not just manifest changes), defines the complete experimental matrix with precise configurations, and establishes the rerun requirements.

**Critical Acknowledgment:** Changing SupCon from `phase` to `phase_topo` creates a new method definition. All existing results trained with `phase` labels are invalidated for the final paper matrix. A complete rerun is required.

---

## 1. Resolution of All Remaining Questions

### Question 1: Top-1 Learned Gate vs. Top-2 Soft Mixture

**Decision:** Use **top-1 learned gate** as the primary softmax baseline. Label it precisely as **"learned softmax-scored top-1 router"**.

**Rationale:** The proposed method uses hard top-1 prototype routing. To isolate the routing mechanism (prototype-distance vs. learned-gate), the softmax baseline must also use top-1. Using top-2 would confound three variables simultaneously: routing mechanism, routing sparsity, and action blending.

**Optional Secondary Comparison:** If compute allows, add a separate secondary baseline using top-2 soft mixture. This must be clearly labeled as testing "standard MoE practice" and is not the primary routing-mechanism comparison.

**Final Softmax Baseline Configuration:**
```yaml
router_class: "TopKRouter"
top_k: 1
noise_std: 0.1          # training only; 0.0 at evaluation
normalize_input: true
balance_coeff: 0.01     # pre-registered; do not tune post-hoc
```

**Paper Language:** "We compare against a learned softmax-scored top-1 router, which isolates the routing mechanism while holding sparsity constant."

---

### Question 2: Is Static Rule MoE an Integrated Comparison?

**Decision:** Yes. The Static Rule MoE is intentionally an **integrated representation-plus-router label comparison**. It is not a single-factor ablation.

**Rationale:** Changing both SupCon labels and router initialization labels from `phase_topo` to `phase` tests the hypothesis: "Does topology-driven discovery outperform human rule labels as a unified system?" This is scientifically valid but must be described accurately.

**Paper Language:** "The Static Rule MoE baseline tests the integrated effect of using human-defined rule labels for both representation supervision and router initialization, compared to the proposed topology-driven approach. This is an integrated label-source comparison, not a single-factor ablation."

**Matrix Classification:** Place Static Rule MoE in the "Integrated Comparisons" section, not the "Single-Factor Ablations" section.

---

### Question 3: Should Existing Results Be Discarded and Rerun?

**Decision:** Yes. All existing `phase`-trained results are invalidated for the final paper matrix. A complete rerun under `phase_topo` is required.

**Rationale:** Changing SupCon from `phase` to `phase_topo` fundamentally alters the learned representation. Results trained with `phase` labels cannot be compared to results trained with `phase_topo` labels. This is a new method definition.

**Action Items:**
1. Preserve all existing `phase`-trained results as historical documentation in a separate directory (e.g., `outputs/historical_phase_supcon/`).
2. Do not include historical results in any final paper table.
3. Rerun all final-method and final-aligned baseline runs under `phase_topo`.
4. Update the method description in the paper to reflect `phase_topo` supervision.

**Paper Language:** "All results reported in this paper use topology-derived labels (`phase_topo`) for both contrastive representation learning and router initialization. Earlier development iterations using rule-based labels (`phase`) are preserved as historical documentation but are not included in the final comparison."

---

### Question 4: oracle_moe Offline-Only vs. New Implementation

**Decision:** Create a **new final-aligned oracle implementation** that uses ground-truth routing at both training and evaluation. Label it as a non-deployable privileged upper-bound reference.

**Rationale:** The existing `oracle_moe` uses ground-truth during training but falls back to a learned router at evaluation. This is not a true oracle. A true upper bound requires ground-truth routing at both train and eval.

**Implementation Requirements:**
1. Create a new model class: `FinalAlignedOracleMoE`.
2. During Stage 2 training: use ground-truth `phase_topo` labels for expert dispatch.
3. During evaluation rollouts: use ground-truth `phase_topo` labels for expert dispatch.
4. Mark the model as non-deployable. Reject any attempt to use it in a deployable evaluation mode.
5. Record ground-truth phase access in the run metadata.

**Disposition of Existing Models:**
- `teacher_forced`: Keep as a separate diagnostic. GT train, predicted eval. Tests training signal quality.
- `oracle_moe` (existing): Relabel as historical diagnostic or remove. Do not include in final matrix.
- `FinalAlignedOracleMoE` (new): Include in final matrix as privileged upper bound.

**Paper Language:** "The Oracle Router uses ground-truth topology labels for expert dispatch during both training and evaluation. It is a non-deployable privileged reference that establishes the theoretical upper bound of the architecture."

---

## 2. Complete Final Experimental Matrix

### Table A: Major Baselines (External Validity)

| Method Identity | SupCon Label | Router Init Label | Router Mechanism | Expert (`beta`) | Scientific Question |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Standard BC` | None | None | None | Direct MLP | Dense imitation floor |
| `Final-Aligned Softmax MoE` | `phase_topo` | Random | Learned softmax top-1 | `ResidualImpedanceExpert` (0.0) | Is prototype routing better than learned gating? |
| `Static Rule MoE` | `phase` | `phase` | Prototype top-1 | `ResidualImpedanceExpert` (0.0) | Integrated: topology labels vs rule labels |

### Table B: Single-Factor Causal Ablations

| Method Identity | SupCon Label | Router Init Label | Router Mechanism | Expert (`beta`) | Factor Changed vs. Ours |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`PhaseForge (Ours)`** | `phase_topo` | `phase_topo` | Prototype top-1 | `ResidualImpedanceExpert` (0.0) | Baseline |
| `No SupCon (Plain Encoder)` | None (BC) | `phase_topo` | Prototype top-1 | `ResidualImpedanceExpert` (0.0) | Representation only |
| `Random Init Router` | `phase_topo` | Random | Prototype top-1 | `ResidualImpedanceExpert` (0.0) | Router initialization only |
| `Scratch Init Experts` | `phase_topo` | `phase_topo` | Prototype top-1 | Random scratch | Expert initialization only |

### Table C: Factorial Corner (Two Factors Changed)

| Method Identity | SupCon Label | Router Init Label | Router Mechanism | Expert (`beta`) | Factors Changed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Factorial Floor` | None (BC) | Random | Prototype top-1 | `ResidualImpedanceExpert` (0.0) | Representation + Router Init |

**Paper Language:** "The Factorial Floor changes two factors simultaneously (representation and router initialization) and serves as the corner of the 2×2 factorial design. It is not a single-factor ablation."

### Table D: Privileged Diagnostics (Non-Deployable)

| Method Identity | SupCon Label | Router Init Label | Router Mechanism | Expert (`beta`) | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `teacher_forced` | `phase_topo` | `phase_topo` | GT train, predicted eval | `ResidualImpedanceExpert` (0.0) | Training signal quality upper bound |
| `FinalAlignedOracleMoE` | `phase_topo` | `phase_topo` | GT train, GT eval | `ResidualImpedanceExpert` (0.0) | Theoretical architecture upper bound |

**Paper Language:** "Both diagnostics are non-deployable and use privileged information. They are reported separately from deployable baselines and ablations."

---

## 3. Required Code Changes

### 3.1 Plain Encoder Control

**Current State:** `plain_encoder_phase_bootstrap.py` hardcodes `phase = batch["phase"]` and uses old `TopKRouter`.

**Required Change:** Create a new final-aligned plain encoder implementation OR generalize the existing model.

**Option A (Recommended): Generalized Final-Aligned Control Base**

Create a new model class `FinalAlignedControlBase` that accepts configurable parameters:

```python
class FinalAlignedControlBase:
    def __init__(
        self,
        encoder: StateEncoder,
        router_class: str,  # "PrototypeRouter" or "TopKRouter"
        router_init_source: str,  # "phase_topo", "phase", "random", "kmeans"
        supcon_label_field: str,  # "phase_topo", "phase", or None
        expert_class: str,  # "ResidualImpedanceExpert" or "ExpertMLP"
        expert_beta: float,
        partial_warm_start: bool,
    ):
        ...
```

This single class can be configured for all ablations, eliminating code duplication and drift.

**Option B: New Plain Encoder Implementation**

If Option A is too invasive, create a new class `FinalAlignedPlainEncoder` that:
1. Uses `PrototypeRouter` instead of `TopKRouter`.
2. Accepts `label_field` as a configurable parameter.
3. Computes centroids from its own BC latent space using the specified label field.
4. Uses `ResidualImpedanceExpert` with `beta=0.0`.

**Validation Gate:** The plain encoder control must pass a unit test verifying that:
- Prototypes are computed from the plain encoder's own latent vectors.
- The label field is `phase_topo`.
- No SupCon latent prototypes are reused.

### 3.2 SupCon Label Source

**Current State:** `stage1.yaml` defaults to `phase`. Confirmation manifests do not override `train.supcon.label_field`.

**Required Change:** Update all final manifests to explicitly set:

```yaml
train:
  supcon:
    enabled: true
    label_field: "phase_topo"  # Explicitly set for final method
```

For the Static Rule MoE baseline:

```yaml
train:
  supcon:
    enabled: true
    label_field: "phase"  # Explicitly set for rule-based comparison
```

**Validation Gate:** The manifest resolver must assert that `supcon.label_field` matches the expected value for each experiment identity.

### 3.3 Final-Aligned Oracle Implementation

**Current State:** `oracle_moe.py` uses GT during training but falls back to learned router at eval.

**Required Change:** Create a new class `FinalAlignedOracleMoE`:

```python
class FinalAlignedOracleMoE:
    def get_action(self, obs):
        # Use ground-truth phase_topo label for expert dispatch
        # This requires access to phase_topo during rollout
        # Mark as non-deployable
        phase = self.get_ground_truth_phase(obs)
        expert_idx = phase
        return self.experts[expert_idx](self.encoder(obs))
    
    def is_deployable(self):
        return False  # Reject deployable evaluation
```

**Validation Gate:** The oracle model must raise an exception if `is_deployable()` is called or if it is used in a deployable evaluation context.

### 3.4 Beta Validation Hook

**Current State:** Beta is hardcoded in the model config. No validation exists.

**Required Change:** Add validation at the manifest/experiment resolver layer:

```python
def validate_experiment_config(experiment_id, config):
    if experiment_id in FINAL_MATRIX_EXPERIMENTS:
        assert config.expert.beta == 0.0, \
            f"Beta must be locked to 0.0 for {experiment_id}. Got {config.expert.beta}"
        assert config.expert.beta_schedule is None, \
            f"Beta schedule must be None for {experiment_id}"
```

**Rationale:** The expert itself should not know which task is being run. Future research may intentionally activate beta. The validation belongs at the experiment definition layer.

---

## 4. Rerun Requirements

### 4.1 Invalidation of Existing Results

All existing results trained with `phase` labels for SupCon are invalidated for the final paper matrix. This includes:
- All confirmation runs in `precision_residual_confirm/`.
- All historical `phaseforge` runs.
- All baseline runs that used `phase` labels.

### 4.2 Rerun Scope

The following must be rerun under `phase_topo`:
1. **Final Method:** `precision_residual_phaseforge` (all seeds, all tasks).
2. **All Single-Factor Ablations:** Plain encoder, random init, scratch init.
3. **Factorial Floor.**
4. **Softmax MoE Baseline.**
5. **Static Rule MoE Baseline** (uses `phase` labels intentionally).
6. **Diagnostics:** `teacher_forced`, `FinalAlignedOracleMoE`.
7. **Standard BC Baseline.**

### 4.3 Preservation of Historical Results

Move all existing `phase`-trained results to a separate directory:

```
outputs/historical_phase_supcon/
```

Do not delete them. They document the development path. But do not include them in any final paper table.

---

## 5. Validation Gates

No final training begins until all of the following gates pass.

### 5.1 Configuration Gates

1. Every final-family experiment explicitly sets `supcon.label_field`.
2. The final method uses `phase_topo` for both SupCon and router init.
3. The Static Rule MoE uses `phase` for both SupCon and router init.
4. The Softmax MoE uses `phase_topo` for SupCon and random init for router.
5. All experiments set `expert.beta = 0.0` and `expert.beta_schedule = None`.
6. The Softmax MoE uses `top_k=1`, `noise_std=0.1`, `balance_coeff=0.01`.

### 5.2 Model-Contract Gates

1. All final-family policies are memoryless at inference.
2. No ground-truth phase enters deployable rollout.
3. `FinalAlignedOracleMoE` is marked as non-deployable.
4. `teacher_forced` uses predicted phase at evaluation.
5. The plain encoder computes prototypes from its own latent space.
6. No SupCon latent prototypes are reused in the plain encoder control.

### 5.3 Checkpoint Resolution Gates

1. All final-family controls use provider identities, not hardcoded paths.
2. Seed-exact checkpoint resolution is enforced.
3. Resolved checkpoint path and SHA-256 hash are recorded in run metadata.
4. Run fails immediately if seed-matched checkpoint is unavailable.
5. No historical `phaseforge` checkpoint is loaded by any final-aligned control.

### 5.4 Code-Change Gates

1. Plain encoder control uses `PrototypeRouter`, not `TopKRouter`.
2. Plain encoder control accepts configurable label field.
3. Plain encoder control uses `phase_topo` for centroid computation.
4. `FinalAlignedOracleMoE` uses ground-truth at both train and eval.
5. Beta validation hook is active at the manifest resolver layer.

---

## 6. Paper Language and Claims

### 6.1 Method Description

**Correct:** "PhaseForge is a topology-initialized prototype-routed direct-action mixture-of-experts policy for memoryless robotic manipulation. The residual impedance expert architecture is structurally present but functionally disabled ($\beta=0$) for all tasks in this paper, reducing each expert to a direct-action MLP."

**Incorrect:** "PhaseForge is a precision-residual impedance policy that demonstrates improved compliance."

### 6.2 Routing Claims

**Correct:** "PhaseForge is designed to reduce routing boundary chattering and expert load collapse through hard prototype routing and topology-based initialization. We empirically measure routing transitions, expert utilization, and collapse rates to evaluate this design."

**Incorrect:** "PhaseForge solves boundary chattering and load collapse."

### 6.3 Ablation Claims

**Correct:** "All rows except the explicitly marked factorial corner change one declared factor. The factorial corner changes two factors simultaneously and serves as the corner of the 2×2 factorial design."

**Incorrect:** "Every causal row changes exactly one factor."

### 6.4 Static Rule MoE Claim

**Correct:** "The Static Rule MoE baseline tests the integrated effect of using human-defined rule labels for both representation supervision and router initialization, compared to the proposed topology-driven approach."

**Incorrect:** "The Static Rule MoE ablates only the router initialization."

---

## 7. Provenance Protocol

### 7.1 Checkpoint Resolution

Use provider identities with seed-exact resolution:

```yaml
# In the manifest
checkpoint_provider:
  identity: "precision_residual_phaseforge_stage1"
  seed_resolution: "exact"  # seed 42 loads seed-42 checkpoint
  fail_if_unavailable: true
```

The resolver dynamically constructs the path:

```python
def resolve_checkpoint(provider_identity, seed):
    base_dir = get_output_dir(provider_identity)
    checkpoint_path = f"{base_dir}/seed_{seed}/best_model.pt"
    if not os.path.exists(checkpoint_path):
        raise CheckpointNotFoundError(f"Seed {seed} checkpoint not found for {provider_identity}")
    return checkpoint_path
```

### 7.2 Metadata Recording

Every run must record:
- Resolved checkpoint absolute path.
- SHA-256 hash of the checkpoint file.
- Resolved config hash.
- Git commit hash.
- Dataset hash.
- Evaluation bank hash.
- Environment version.

### 7.3 Historical Separation

Historical results must be stored in a separate directory and labeled:

```
outputs/historical_phase_supcon/
  README.md: "Historical results using phase labels for SupCon. Not included in final paper matrix."
```

---

## 8. Final Checklist

Before any training begins, verify:

- [ ] All four questions are answered and documented.
- [ ] Plain encoder control is reimplemented with `PrototypeRouter` and configurable label field.
- [ ] SupCon label field is explicitly set in all manifests.
- [ ] Softmax MoE uses pre-registered settings (top-1, noise=0.1, balance=0.01).
- [ ] `FinalAlignedOracleMoE` is implemented with GT at both train and eval.
- [ ] `teacher_forced` is included as a separate diagnostic.
- [ ] Beta validation hook is active at the manifest resolver layer.
- [ ] Checkpoint resolution uses provider identities with seed-exact matching.
- [ ] All existing `phase`-trained results are moved to historical directory.
- [ ] Paper language is updated to reflect `phase_topo` supervision and `beta=0`.
- [ ] Static Rule MoE is labeled as an integrated comparison, not a single-factor ablation.
- [ ] Factorial Floor is labeled as a two-factor corner, not a single-factor ablation.

---

## 9. Execution Sequence

1. **Week 1: Code Changes**
   - Implement generalized final-aligned control base or new plain encoder.
   - Implement `FinalAlignedOracleMoE`.
   - Add beta validation hook.
   - Add checkpoint resolution logic.
   - Write unit tests for all validation gates.

2. **Week 2: Dry Runs and Validation**
   - Run all validation gates.
   - Dry-run every final-family manifest.
   - Verify seed-exact checkpoint resolution.
   - Verify plain encoder computes prototypes from its own latent space.

3. **Week 3-4: Final Training**
   - Run all final-family experiments (seeds 42, 43, 44).
   - Record all metadata and checkpoint hashes.
   - Monitor for numerical instability.

4. **Week 5: Analysis and Reporting**
   - Compute success rates, Wilson intervals, routing diagnostics.
   - Generate ablation tables.
   - Write paper sections with correct language.

---

This protocol is now complete, execution-ready, and addresses every issue identified in your audit. There are no remaining ambiguities. Proceed with implementation.