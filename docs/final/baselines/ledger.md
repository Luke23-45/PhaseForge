# Precision-Residual PhaseForge Final Baseline Program Ledger

**Status:** Active execution ledger. No final training is authorized until the required gates in this ledger and in the protocol pass.

**Authority:** [`PRECISION_RESIDUAL_BASELINE_PROTOCOL.md`](./PRECISION_RESIDUAL_BASELINE_PROTOCOL.md)

**Scope:** Final same-generation baseline and ablation program for `precision_residual_phaseforge` across Lift, Can, Square, ToolHang, and Transport.

**Repository root:** `C:\Users\Hellx\Documents\Programming\python\Project\Neryva\PhaseForge`

**Created:** 2026-09-08

## How to use this ledger

Update this file as work is completed. Every implementation, test, provider run, training run, evaluation run, and reporting item must have evidence before its status changes.

Allowed statuses:

- `MAPPED` — specified and mapped to code/configuration, but not verified as complete.
- `PENDING` — required work has not started or has no acceptable evidence.
- `IN_PROGRESS` — actively being implemented or verified.
- `VERIFIED` — acceptance evidence exists and is recorded below.
- `BLOCKED` — cannot proceed; record the exact blocker and the required decision.
- `HISTORICAL` — intentionally preserved but excluded from the final matrix.
- `N/A` — deliberately not applicable; record why.

Do not mark an item `VERIFIED` because code exists. Verification requires a test result, dry-run output, resolved configuration, artifact hash, or other concrete evidence.

## 1. Current state

| Item | Current state | Status |
|---|---|---|
| Final protocol | Written at `docs/final/baselines/PRECISION_RESIDUAL_BASELINE_PROTOCOL.md`; it is the execution authority. | `MAPPED` |
| Proposed identity | `precision_residual_phaseforge` is the sole proposed-method identity. | `MAPPED` |
| Final model config | `phaseforge/config/models/precision_residual_phaseforge.yaml` exists and resolves beta to `0.0`. | `MAPPED` |
| Final causal manifest | `experiments/final_causal_matrix.json` does not yet exist. | `PENDING` |
| Final-aligned baseline implementations | Required controls and diagnostics are specified but not yet verified as implemented. | `PENDING` |
| Final provider checkpoints | No final-generation provider set has been verified for all task/seed combinations. | `PENDING` |
| Final training | Not started and not authorized. | `PENDING` |
| Final evaluation | Not started and not authorized. | `PENDING` |
| Final paper tables | Must wait for the complete same-generation matrix. | `PENDING` |

## 2. Locked scientific definition

The final proposed method is a memoryless, hard top-1, topology-initialized prototype-routed direct-action MoE. The residual expert module is present in the graph, but `beta=0.0` makes its residual action path inactive. No paper claim may describe demonstrated residual-compliance improvement.

`phase_topo` is a train-only, topology-derived PELT label. It is not ground truth and must not enter deployable policy input. The locked topology protocol is `topo_pelt_k6` with six regimes. The topology artifact identity, hash, and label mapping must be recorded for every task/seed that consumes it.

`phase`/`phase_rule` is the existing human-defined or rule-derived phase vocabulary. Topology rows use `phase_topo` for Stage 1 phase CE, Stage 1 SupCon, Stage 2 prototype initialization, and Stage 2 margin targets. Static Rule MoE uses `phase` for those same declared functions. Plain and factorial controls have no Stage 1 phase/SupCon loss, but use `phase_topo` for Stage 2 prototype and margin targets.

The topology configuration is `phaseforge/config/topo/topo_pelt_k6.yaml`. A K-sweep or different regime count is outside this locked matrix and requires a new protocol, phase-head dimension, and method identities.

The beta-zero action contract is exact: `ResidualImpedanceExpert` returns the direct base action, with no residual pose-head contribution, no residual action-loss gradient, direct gripper channel, and no task-state feedback impedance controller. `bc_impedance` is only a separate secondary action-space reference and is not part of the core causal matrix.

### Repository audit anchors

| Concern | Current repository anchor | Ledger work |
|---|---|---|
| Final model contract | `phaseforge/config/models/precision_residual_phaseforge.yaml` | MODEL-01, MODEL-10 |
| Topology configuration | `phaseforge/config/topo/topo_pelt_k6.yaml` | DATA-01–DATA-04 |
| Stage 1 CE label access | `phaseforge/trains/loops/stage1_loop.py` | LABEL-01, LABEL-04 |
| Stage 1 SupCon label access | `phaseforge/trains/loops/stage1_loop.py` | LABEL-02, LABEL-04 |
| Stage 2 margin label access | `phaseforge/trains/loops/stage2_loop.py` | LABEL-03, LABEL-04, ROUTER-03 (`margin_loss_from_logits`) |
| Historical plain bootstrap | `phaseforge/models/baselines/plain_encoder_phase_bootstrap.py` | MODEL-02, MODEL-13 |
| Runner provider schema | `phaseforge/runner/protocol.py` | PROVIDER-01, MAN-07 |
| Seed-aware resolver | `phaseforge/runner/resolver.py` | PROVIDER-05–PROVIDER-09 |

All final residual-expert rows use `beta=0.0` and no beta schedule. All final-family Stage 2 rows that fine-tune the encoder use `encoder_lr_scale=0.1`.

Checkpoint selection is locked to minimum validation action MSE, `val/loss_action`, on the held-out validation split. Rollout success and final evaluation-bank results cannot select checkpoints.

## 3. Final method identities

These are the only identities permitted in the final causal matrix. Historical names must not be silently reused.

| Identity | Role | Evaluation mode | Stage 1 provider | Label policy | Status |
|---|---|---|---|---|---|
| `precision_residual_phaseforge` | Sole proposed method | Deployable rollout | `precision_residual_phaseforge_stage1` / self | `phase_topo` for phase CE, SupCon, prototype init, and margin | `PENDING` |
| `precision_residual_plain_encoder` | Representation control | Deployable rollout | `final_aligned_bc_stage1` | No Stage 1 phase/SupCon; `phase_topo` for Stage 2 prototype/margin | `PENDING` |
| `precision_residual_phase_random_router` | Router-initialization control | Deployable rollout | `precision_residual_phaseforge_stage1` | `phase_topo` | `PENDING` |
| `precision_residual_scratch_moe` | Expert-initialization control | Deployable rollout | `precision_residual_phaseforge_stage1` | `phase_topo` | `PENDING` |
| `precision_residual_factorial_floor` | Two-factor corner | Deployable rollout | `final_aligned_bc_stage1` | No Stage 1 phase/SupCon; `phase_topo` for Stage 2 prototype/margin | `PENDING` |
| `bc` | External imitation floor | Deployable rollout | Itself | None | `PENDING` |
| `final_aligned_softmax_top1` | Router-package comparison | Deployable rollout | `precision_residual_phaseforge_stage1` | `phase_topo` | `PENDING` |
| `final_aligned_static_rule` | Integrated rule-label comparison | Deployable rollout | `final_aligned_static_rule_stage1` / self | `phase` for all declared rule-label losses and initialization | `PENDING` |
| `precision_residual_teacher_forced` | Privileged training diagnostic | Separate diagnostic; not pooled with primary rollout table | `precision_residual_phaseforge_stage1` | `phase_topo` training; predicted phase-head routing at evaluation | `PENDING` |
| `precision_residual_oracle` | Privileged offline diagnostic | Held-out offline demonstration data only | `precision_residual_phaseforge_stage1` | Supplied `phase_topo` at training and offline evaluation | `PENDING` |

The old `warmstart_moe`, `plain_encoder_phase_bootstrap`, `phase_pretrain_random_router`, `scratch_moe`, `teacher_forced`, old `oracle_moe`, and other historical variants are not final rows. They remain `HISTORICAL` and must retain their original identity, commit, configuration hash, provider, and label policy.

## 4. Scientific questions and admissible conclusions

| Question | Required comparison | Permitted conclusion |
|---|---|---|
| Does the topology-supervised representation matter? | Proposed method vs. `precision_residual_plain_encoder` | Evidence about the representation package, with all declared differences reported. |
| Does topology-based prototype initialization matter? | Proposed method vs. `precision_residual_phase_random_router` | Evidence about router initialization. |
| Does partial expert warm-start matter? | Proposed method vs. `precision_residual_scratch_moe` | Evidence about expert initialization. |
| What is the two-factor floor? | `precision_residual_factorial_floor` | A 2×2 factorial corner, not a single-factor ablation. |
| Does prototype routing outperform learned softmax routing? | Proposed method vs. `final_aligned_softmax_top1` | A registered router-package comparison. Because balance coefficients differ (`0.0001` vs. `0.01`), do not claim a pure gate-only causal effect. |
| Do topology-derived labels outperform rule labels as a system? | Proposed method vs. `final_aligned_static_rule` | An integrated representation-plus-initialization label-source comparison. |
| How does the method compare with ordinary imitation? | Proposed method vs. `bc` | Same-protocol performance comparison against the imitation floor. |
| How much does privileged routing help offline? | Teacher-forced/oracle diagnostics | Separate mechanism diagnostics only; not deployable success-rate claims. |

## 5. Work ledger: governance and protocol control

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| GOV-01 | Keep the final protocol as the sole active authority. | No competing active protocol defines a different final matrix. | — | `MAPPED` |
| GOV-02 | Preserve archived professor reports as historical provenance. | `docs/archive/professor_report1.md` through `professor_report5.md` are present; no report is used as execution authority. | GOV-01 | `VERIFIED` |
| GOV-03 | Freeze the final protocol revision before implementation. | Git commit records the protocol revision; later conceptual changes require a new revision. | GOV-01, DATA-01, MODEL-01 | `PENDING` |
| GOV-04 | Use a fresh final output namespace. | Preflight proves the namespace is absent or empty and cannot overwrite historical outputs. | MAN-01 | `PENDING` |
| GOV-05 | Do not hard-delete historical configs or outputs during this program. | Cleanup is postponed until final provenance and publication archive are complete. | GOV-01 | `MAPPED` |

## 6. Work ledger: data and label contract

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| DATA-01 | Pin `topo_pelt_k6` for the final matrix. | Every topology row resolves the same topology config and six-class contract. | — | `PENDING` |
| DATA-02 | Generate or locate `phase_topo` for every task and required split. | Lift, Can, Square, ToolHang, and Transport train/validation data contain the field. | DATA-01 | `PENDING` |
| DATA-03 | Validate label type and range. | Labels are integer-valued, valid for six classes, and contiguous after the documented remapping. | DATA-02 | `PENDING` |
| DATA-04 | Lock topology artifact identity per task and training seed. | All methods consuming `phase_topo` for the same task/seed use the same artifact hash and label mapping. | DATA-02, MAN-01 | `PENDING` |
| DATA-05 | Keep topology labels out of deployable inputs. | Rollout model signatures and runner traces show state-only inference; no `phase_topo` enters `get_action`. | DATA-02, MODEL-06 | `PENDING` |
| DATA-06 | Keep Static Rule MoE on `phase`. | Resolved config and batch inspection show `phase` for its declared CE/SupCon/prototype/margin targets. | LABEL-01 | `PENDING` |
| DATA-07 | Document split and bank identity. | Dataset/cache hash, validation split identity, and evaluation reset-bank hash are recorded. | DATA-02, EVAL-01 | `PENDING` |

## 7. Work ledger: training label and loss implementation

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| LABEL-01 | Add resolved `phase_label_field` to Stage 1 and Stage 2. | Trainers read the resolved field instead of hardcoded `batch["phase"]`. | DATA-02 | `PENDING` |
| LABEL-02 | Make Stage 1 SupCon label field configurable. | `supcon.label_field` resolves to `phase_topo` or `phase` by method. | LABEL-01 | `PENDING` |
| LABEL-03 | Make Stage 2 margin label field configurable. | `margin.label_field` resolves to the declared method field. | LABEL-01 | `PENDING` |
| LABEL-04 | Apply the resolved field to phase CE, SupCon, prototype init, margin, and diagnostics. | Unit tests show each loss/target source uses the intended field. | LABEL-01, LABEL-02, LABEL-03 | `PENDING` |
| LABEL-05 | Fail closed on missing label fields. | Error includes method, task, split, and requested field before training begins. | LABEL-01 | `PENDING` |
| LABEL-06 | Validate cardinality and contiguity before training. | Invalid labels stop composition or preflight; no silent remapping beyond the documented mapping. | DATA-03, LABEL-01 | `PENDING` |
| LABEL-07 | Keep six phase-head outputs under `topo_pelt_k6`. | Resolved phase-head dimension is six for all rows retaining a phase head. | DATA-01 | `PENDING` |
| LABEL-08 | Preserve the repository's Hydra stage structure. | Use `train=stage1` and `train=stage2`; do not introduce an untracked nested `train.stage1`/`train.stage2` schema. | LABEL-01 | `PENDING` |
| LABEL-09 | Keep optional phase-specific losses disabled. | Release loss and other optional phase losses remain disabled; any future activation requires its own declared label-field contract and protocol revision. | MAN-04 | `PENDING` |

## 8. Work ledger: final-aligned model implementations

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| MODEL-01 | Confirm the proposed model contract. | Final config resolves normalized encoder, six experts, prototype top-1, topology prototypes, 50% partial warm-start, beta zero, and Stage 2 encoder scale `0.1`. | GOV-01 | `MAPPED` |
| MODEL-02 | Implement or generalize the final-aligned plain encoder. | It uses normalized BC representation, `PrototypeRouter`, residual expert beta zero, and its own latent prototypes. | LABEL-01, PROVIDER-02 | `PENDING` |
| MODEL-03 | Implement random router initialization control. | Same final family except registered random prototype initialization; initialization hash is recorded. | MODEL-01 | `PENDING` |
| MODEL-04 | Implement scratch-expert control. | Same final family except registered random expert initialization; no silent warm-start. | MODEL-01 | `PENDING` |
| MODEL-05 | Implement factorial floor. | Plain normalized BC representation plus random prototype initialization; metadata declares both factors. | MODEL-02, MODEL-03 | `PENDING` |
| MODEL-06 | Implement final-aligned softmax top-1 control. | `TopKRouter`, top_k=1, noise_std=0.1, normalize_input=true, balance_coeff=0.01, memoryless evaluation. | ROUTER-01, ROUTER-02 | `PENDING` |
| MODEL-07 | Implement final-aligned Static Rule MoE. | Final family with `phase` label policy and its own Stage 1 provider; not silently mapped to topology labels. | LABEL-06, PROVIDER-03 | `PENDING` |
| MODEL-08 | Generalize teacher-forced diagnostic. | Normalized final encoder, beta-zero expert, topology-labeled training, predicted phase-head routing at evaluation. | LABEL-04, MODEL-01 | `PENDING` |
| MODEL-09 | Implement privileged oracle evaluator. | Uses supplied recorded labels only in offline evaluator; refuses ordinary state-only rollout mode. | LABEL-04, MODEL-01 | `PENDING` |
| MODEL-10 | Enforce the beta-zero action contract. | Resolved beta is exactly `0.0`, beta schedule is null, residual heads do not affect action, and direct base action equality is tested. | MODEL-01 | `PENDING` |
| MODEL-11 | Preserve action dimensions and range. | Every final row satisfies task-specific action dimension and `[-1,1]` action contract. | MODEL-02–MODEL-10 | `PENDING` |
| MODEL-12 | Support only explicitly registered initialization modes. | Topology, rule, random, and any approved K-Means initialization mode have explicit identities, settings, and metadata; no implicit mode changes. | MODEL-01–MODEL-07 | `PENDING` |
| MODEL-13 | Share common final-family implementation. | Common behavior is implemented once where appropriate; copied models do not drift in defaults, dimensions, beta, labels, or metadata. | MODEL-02–MODEL-09 | `PENDING` |

## 9. Work ledger: router and loss mechanics

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| ROUTER-01 | Lock PrototypeRouter settings. | Hard top-1, margin `0.5`, `balance_coeff=0.0001`, memoryless behavior, topology/random init as declared. | MODEL-01 | `PENDING` |
| ROUTER-02 | Lock TopKRouter settings. | Top-1, noise `0.1` during training only, normalized input, balance coefficient `0.01`, deterministic evaluation. | MODEL-06 | `PENDING` |
| ROUTER-03 | Add common `margin_loss_from_logits` interface. | Both router classes produce the same multiclass margin formula and valid gradients. | ROUTER-01, ROUTER-02 | `PENDING` |
| ROUTER-04 | Expose deterministic pre-exploration logits for softmax margin. | Margin targets are not silently affected by training noise. | ROUTER-02, ROUTER-03 | `PENDING` |
| ROUTER-05 | Test prototype distance/logit equivalence. | `d_j-d_y` and the corresponding logit formula agree numerically. | ROUTER-03 | `PENDING` |
| ROUTER-06 | Record balance asymmetry honestly. | Final report calls softmax a router-package comparison, not a pure gate-only ablation. | ROUTER-01, ROUTER-02 | `PENDING` |
| ROUTER-07 | Record router diagnostics. | Utilization, entropy, margin, transition rate/count, and collapse are computed using the fixed definitions in Section 18.1 (protocol §12). | ROUTER-01–ROUTER-04 | `PENDING` |

The registered asymmetry is part of the router package: `PrototypeRouter.balance_coeff=0.0001` and `TopKRouter.balance_coeff=0.01`. It must be disclosed and cannot be tuned after results are observed.

### Required resolved label fields

| Stage/configuration | Required fields for topology rows | Required fields for Static Rule MoE |
|---|---|---|
| Stage 1 | `phase_label_field: phase_topo`; `supcon.enabled: true`; `supcon.label_field: phase_topo` | Corresponding fields resolve to `phase` |
| Stage 2 | `phase_label_field: phase_topo`; `margin.enabled: true`; `margin.label_field: phase_topo` | Corresponding fields resolve to `phase` |

The repository's existing Hydra selection remains `train=stage1` and `train=stage2`; the ledger does not authorize a new nested stage schema.

## 10. Work ledger: training hyperparameters and checkpointing

| ID | Work item | Required locked value / evidence | Depends on | Status |
|---|---|---|---|---|
| TRAIN-01 | Lock Stage 1 schedule. | 100 epochs; AdamW; lr `3e-4`; weight decay `1e-4`; betas `(0.9,0.999)`; epsilon `1e-8`; cosine `T_max=100`, `eta_min=1e-6`; clip `1.0`. | — | `MAPPED` |
| TRAIN-02 | Lock Stage 2 schedule. | 200 epochs; AdamW; lr `1e-4`; weight decay `1e-4`; betas `(0.9,0.999)`; epsilon `1e-8`; cosine `T_max=200`, `eta_min=1e-7`; clip `1.0`. | — | `MAPPED` |
| TRAIN-03 | Lock batch size. | Batch size `256` for all five task configs and both stages unless a new protocol revision is issued. | — | `MAPPED` |
| TRAIN-04 | Lock Stage 2 encoder scale. | Final-family encoder fine-tuning uses `encoder_lr_scale=0.1`; resolved config hash contains the value. | MODEL-01 | `PENDING` |
| TRAIN-05 | Lock checkpoint selection. | `checkpoint.monitor=val/loss_action`, `mode=min` for Stage 1 and Stage 2; selected epoch/value recorded. | TRAIN-01, TRAIN-02 | `PENDING` |
| TRAIN-06 | Disable early stopping for final runs. | Full-length runs complete; `early_stopping.enabled=false` in resolved configs. | TRAIN-05 | `PENDING` |
| TRAIN-07 | Prevent evaluation-bank selection leakage. | Final reset-bank success never selects a checkpoint or tunes a hyperparameter. | TRAIN-05, EVAL-01 | `PENDING` |

## 11. Work ledger: provider and provenance system

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| PROVIDER-01 | Add explicit provider identity support to the runner. | Protocol accepts final provider identities instead of only historical aliases. | — | `PENDING` |
| PROVIDER-02 | Create `final_aligned_bc_stage1`. | Explicit normalized-BC configuration/override produces a valid Stage 1 checkpoint for every task/seed. | MODEL-02, TRAIN-01 | `PENDING` |
| PROVIDER-03 | Create `final_aligned_static_rule_stage1`. | Rule-label Stage 1 checkpoint is produced independently and is not resolved through old `phaseforge`. | MODEL-07, LABEL-06 | `PENDING` |
| PROVIDER-04 | Create `precision_residual_phaseforge_stage1`. | Final topology/SupCon Stage 1 provider exists per task/seed with resolved metadata. | LABEL-04, TRAIN-01 | `PENDING` |
| PROVIDER-05 | Enforce exact task and seed matching. | Wrong-task or wrong-seed provider fails before subprocess launch. | PROVIDER-01 | `PENDING` |
| PROVIDER-06 | Enforce provider commit/config identity. | Expected commit and resolved-config hash are checked. | PROVIDER-01 | `PENDING` |
| PROVIDER-07 | Validate checkpoint artifact. | `checkpoint_best.pt` exists, is loadable, and its SHA-256 is recorded. | PROVIDER-04–PROVIDER-06 | `PENDING` |
| PROVIDER-08 | Record complete provenance. | Provider identity/path/hash, provider commit/config hash, consumer config hash, dataset/cache hash, topology hash, bank hash, and environment versions are stored. | PROVIDER-07 | `PENDING` |
| PROVIDER-09 | Prevent historical alias fallback. | Final rows cannot resolve `phaseforge`, `bc`, or old baseline aliases when a final provider is required. | PROVIDER-01 | `PENDING` |

The current runner schema limitation is tracked explicitly: `stage2_source` currently accepts only `self`, `bc`, and `phaseforge` in `phaseforge/runner/protocol.py`. The implementation must extend that schema and reuse the strict seed-aware resolver in `phaseforge/runner/resolver.py`, then pass the resolved checkpoint as `train.stage1_ckpt_path`.

## 12. Work ledger: final manifest

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| MAN-01 | Create `experiments/final_causal_matrix.json`. | Manifest contains exactly the locked final identities and no historical rows. | GOV-03, MODEL-01–MODEL-09 | `PENDING` |
| MAN-02 | Add all five tasks. | Lift, Can, Square, ToolHang, and Transport are present. | MAN-01 | `PENDING` |
| MAN-03 | Add seeds. | Training seeds 42, 43, and 44 are present for every applicable row. | MAN-01 | `PENDING` |
| MAN-04 | Declare labels explicitly. | `phase_topo` or `phase` is explicit for every loss/bootstrap field; no inherited ambiguity. | LABEL-04 | `PENDING` |
| MAN-05 | Declare beta and schedules. | Every residual row explicitly has `beta=0.0` and null beta schedule. | MODEL-10 | `PENDING` |
| MAN-06 | Declare topology pin. | Topology rows pin `topo_pelt_k6`; artifact mapping is recorded. | DATA-01, DATA-04 | `PENDING` |
| MAN-07 | Declare providers. | Stage 2 source/provider identity is explicit and seed-exact. | PROVIDER-01–PROVIDER-09 | `PENDING` |
| MAN-08 | Separate evaluation modes. | Deployable rows go to rollout evaluation; teacher-forced and oracle go to separate diagnostic paths. | MODEL-08, MODEL-09 | `PENDING` |
| MAN-09 | Use a fresh output namespace. | Historical outputs cannot be overwritten. | GOV-04 | `PENDING` |
| MAN-10 | Compose the entire manifest. | All rows compose successfully for all tasks before training. | MAN-01–MAN-09 | `PENDING` |
| MAN-11 | Preserve `experiments/five_task.json` as historical. | The final matrix is a new manifest; the old manifest is not edited in place or used as final evidence. | MAN-01 | `PENDING` |
| MAN-12 | Declare loss enablement explicitly. | Topology rows explicitly enable the intended SupCon/margin losses; plain/factorial and Static Rule rows declare their intended loss switches rather than inheriting defaults. | MAN-04, LABEL-04 | `PENDING` |

## 13. Work ledger: validation gates and tests

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| TEST-01 | Compose every final model for every task. | Hydra/config composition passes for five tasks. | MAN-10 | `PENDING` |
| TEST-02 | Validate representative batches. | Required labels and action/state dimensions are present. | DATA-02, LABEL-06 | `PENDING` |
| TEST-03 | Test plain latent isolation. | Plain prototypes differ from and do not reuse proposed-method latent/checkpoint artifacts. | MODEL-02 | `PENDING` |
| TEST-04 | Test initialization reproducibility. | Random and topology initialization hashes are deterministic under the declared seed policy. | MODEL-03, ROUTER-01 | `PENDING` |
| TEST-05 | Test partial warm-start reproducibility. | Drop mask and initialization metadata are seed-deterministic. | MODEL-01 | `PENDING` |
| TEST-06 | Test beta-zero equality. | Final residual action equals direct base action; residual heads have no action effect/gradient. | MODEL-10 | `PENDING` |
| TEST-07 | Test common margin loss. | Both routers match the registered formula and produce finite gradients. | ROUTER-03–ROUTER-05 | `PENDING` |
| TEST-08 | Test softmax noise policy. | Training noise is active only as registered; evaluation and margin logits are deterministic. | ROUTER-02, ROUTER-04 | `PENDING` |
| TEST-09 | Test label fail-closed behavior. | Missing, malformed, non-contiguous, or wrong-cardinality labels stop before training. | LABEL-05, LABEL-06 | `PENDING` |
| TEST-10 | Test teacher-forced routing. | Labels are used in training; predicted phase-head output is used in evaluation. | MODEL-08 | `PENDING` |
| TEST-11 | Test oracle restrictions. | Ordinary rollout mode is rejected; offline supplied-label path records privilege metadata. | MODEL-09 | `PENDING` |
| TEST-12 | Test provider failures. | Missing, wrong-seed, wrong-task, wrong-commit, or wrong-hash providers fail before subprocess launch. | PROVIDER-05–PROVIDER-09 | `PENDING` |
| TEST-13 | Test checkpoint selection. | Best checkpoint is minimum validation action MSE; no rollout metric can replace it. | TRAIN-05, TRAIN-07 | `PENDING` |
| TEST-14 | Run existing unit/integration/manifest suites. | Test command, commit, environment, and result are recorded. | TEST-01–TEST-13 | `PENDING` |

## 14. Work ledger: dry-run sequence

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| DRY-01 | Perform no-write inspection. | Repository commit, working tree, historical outputs, topology artifact identities/hashes, and output namespace are recorded. | GOV-04, DATA-04 | `PENDING` |
| DRY-02 | Build the final plan. | Runner expands the complete manifest without training. | MAN-10 | `PENDING` |
| DRY-03 | Verify provider dependency ordering. | Stage 1 providers precede Stage 2 consumers for every task/seed. | PROVIDER-04, MAN-07 | `PENDING` |
| DRY-04 | Verify resolved overrides. | Each command carries intended task, seed, model, provider, labels, beta, topology, and evaluation mode. | MAN-10 | `PENDING` |
| DRY-05 | Verify historical isolation. | No final row resolves to historical `phaseforge` or old baseline aliases. | PROVIDER-09, MAN-10 | `PENDING` |
| DRY-06 | Run an implementation-only pilot. | Small pilot checks execution only; pilot scores cannot select methods or tune final settings. | TEST-14, DRY-01–DRY-05 | `PENDING` |

## 15. Work ledger: final training and evaluation

### 15.1 Shared final-run contract

Every deployable row must use:

- one frozen commit;
- training seeds 42, 43, and 44;
- Lift, Can, Square, ToolHang, and Transport;
- identical task data within each task/seed;
- identical state/action conventions and action-range validation;
- six experts and the declared top-k for the relevant row;
- the beta-zero action contract for every residual-expert row;
- identical topology artifact and label mapping for topology-consuming rows within each task/seed;
- identical frozen reset bank per task;
- 50 reset-bank cases;
- horizons of 500 for Lift, Can, Square, and ToolHang, and 700 for Transport;
- fresh output namespace;
- complete resolved configuration, checkpoint, dataset, topology, bank, and environment metadata.

Teacher-forced is a separate diagnostic. Oracle is offline only. Neither is pooled into deployable success rates.

### 15.2 Run tracking matrix

Each row requires Stage 1/provider work, Stage 2 work where applicable, and the stated evaluation. Record run IDs, checkpoint hashes, and result paths in the cells as they complete. `PENDING` is the initial state.

| Method | Task | Seed 42 | Seed 43 | Seed 44 | Evaluation required | Status |
|---|---|---|---|---|---|---|
| `precision_residual_phaseforge` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phaseforge` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phaseforge` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phaseforge` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phaseforge` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_plain_encoder` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_plain_encoder` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_plain_encoder` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_plain_encoder` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_plain_encoder` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phase_random_router` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phase_random_router` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phase_random_router` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phase_random_router` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_phase_random_router` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_scratch_moe` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_scratch_moe` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_scratch_moe` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_scratch_moe` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_scratch_moe` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_factorial_floor` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_factorial_floor` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_factorial_floor` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_factorial_floor` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_factorial_floor` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `bc` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `bc` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `bc` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `bc` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `bc` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_softmax_top1` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_softmax_top1` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_softmax_top1` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_softmax_top1` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_softmax_top1` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_static_rule` | Lift | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_static_rule` | Can | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_static_rule` | Square | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_static_rule` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `final_aligned_static_rule` | Transport | `PENDING` | `PENDING` | `PENDING` | Rollout | `PENDING` |
| `precision_residual_teacher_forced` | Lift | `PENDING` | `PENDING` | `PENDING` | Separate diagnostic | `PENDING` |
| `precision_residual_teacher_forced` | Can | `PENDING` | `PENDING` | `PENDING` | Separate diagnostic | `PENDING` |
| `precision_residual_teacher_forced` | Square | `PENDING` | `PENDING` | `PENDING` | Separate diagnostic | `PENDING` |
| `precision_residual_teacher_forced` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Separate diagnostic | `PENDING` |
| `precision_residual_teacher_forced` | Transport | `PENDING` | `PENDING` | `PENDING` | Separate diagnostic | `PENDING` |
| `precision_residual_oracle` | Lift | `PENDING` | `PENDING` | `PENDING` | Offline only | `PENDING` |
| `precision_residual_oracle` | Can | `PENDING` | `PENDING` | `PENDING` | Offline only | `PENDING` |
| `precision_residual_oracle` | Square | `PENDING` | `PENDING` | `PENDING` | Offline only | `PENDING` |
| `precision_residual_oracle` | ToolHang | `PENDING` | `PENDING` | `PENDING` | Offline only | `PENDING` |
| `precision_residual_oracle` | Transport | `PENDING` | `PENDING` | `PENDING` | Offline only | `PENDING` |

## 16. Analysis and reporting ledger

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| REPORT-01 | Build external-baseline table. | BC and final-aligned softmax top-1 reported separately with per-task/per-seed results. | RUN-01, RUN-02 | `PENDING` |
| REPORT-02 | Build integrated Static Rule table. | Rule-label comparison clearly marked integrated, not single-factor. | RUN-01, RUN-02 | `PENDING` |
| REPORT-03 | Build causal-ablation table. | Plain, random-router, and scratch-expert controls have exact declared differences. | RUN-01, RUN-02 | `PENDING` |
| REPORT-04 | Build factorial table. | Factorial floor is explicitly labeled as two-factor corner. | RUN-01, RUN-02 | `PENDING` |
| REPORT-05 | Build privileged-diagnostic table. | Teacher-forced and oracle are separate and excluded from deployable success-rate pooling. | RUN-05 | `PENDING` |
| REPORT-06 | Report per-task and per-seed success. | Numerators, denominators, pooled rates, and Wilson intervals are present. | RUN-01–RUN-03 | `PENDING` |
| REPORT-07 | Report paired reset comparisons. | Applicable methods use identical reset cases and paired differences. | EVAL-01 | `PENDING` |
| REPORT-08 | Report action metrics. | Validation/held-out action MSE and relevant per-phase metrics are recorded. | RUN-01–RUN-05 | `PENDING` |
| REPORT-09 | Report routing diagnostics. | Utilization, normalized entropy, margin, transitions, and collapse use fixed formulas. | ROUTER-07, RUN-01–RUN-03 | `PENDING` |
| REPORT-10 | Report failure categories. | Infrastructure failures and policy failures are separated and denominators are documented. | EVAL-02 | `PENDING` |
| REPORT-11 | Report provenance. | Config, commit, checkpoint, data/cache, topology, reset-bank, and environment hashes are included. | PROVIDER-08, RUN-01–RUN-05 | `PENDING` |
| REPORT-12 | State claims conservatively. | No residual-compliance, solved-chattering, solved-collapse, or oracle-rollout claims are made. | REPORT-01–REPORT-11 | `PENDING` |
| REPORT-13 | Preserve historical appendix. | Historical phase-based/development results remain separated and labeled as excluded from final matrix. | GOV-02, REPORT-11 | `PENDING` |
| REPORT-14 | Report computational cost correctly. | Training cost is recorded; if inference latency is reported, hardware, batch size, warmup, and measurement window are recorded. | RUN-01–RUN-03 | `PENDING` |

## 17. Run and evaluation work IDs

The run matrix in Section 15 is the detailed per-method/per-task/per-seed tracker. These aggregate IDs control the required execution stages:

| ID | Aggregate work item | Exit condition | Depends on | Status |
|---|---|---|---|---|
| RUN-01 | Complete deployable proposed method runs. | Proposed method has valid Stage 1/Stage 2/evaluation artifacts for all five tasks and three seeds. | TEST-14, DRY-06 | `PENDING` |
| RUN-02 | Complete deployable final-aligned baseline runs. | Plain, random-router, scratch, factorial, BC, softmax, and Static Rule rows complete for all applicable task/seed cells. | RUN-01, DRY-06 | `PENDING` |
| RUN-03 | Complete privileged diagnostics. | Teacher-forced and oracle diagnostics complete through their allowed paths. | RUN-01, DRY-06 | `PENDING` |
| RUN-04 | Verify frozen evaluation banks. | Every rollout row uses the correct task bank hash and 50-case protocol. | EVAL-01 | `PENDING` |
| RUN-05 | Verify no accidental pooling. | Summaries and reports keep diagnostic/historical rows out of primary deployable success tables. | RUN-01–RUN-03 | `PENDING` |

## 18. Evaluation infrastructure ledger

| ID | Work item | Required evidence / exit condition | Depends on | Status |
|---|---|---|---|---|
| EVAL-01 | Freeze reset banks per task. | Bank generated/selected before final evaluation; hash recorded; auto-regeneration disabled during evaluation. | MAN-09 | `PENDING` |
| EVAL-02 | Lock rollout protocol. | Correct horizons, action validation, reset behavior, valid-episode rules, and failure accounting are recorded. | EVAL-01 | `PENDING` |
| EVAL-03 | Verify deployable state-only inference. | Rollout runner passes only normalized state to deployable `get_action`. | MODEL-11, TEST-06 | `PENDING` |
| EVAL-04 | Verify deterministic evaluation. | Router noise is off at evaluation; no history/sticky/oracle intervention is active for primary rows. | ROUTER-02, MODEL-06 | `PENDING` |
| EVAL-05 | Record routing traces. | Per-run diagnostics include the fields needed for fixed utilization/entropy/margin/transition/collapse calculations. | ROUTER-07 | `PENDING` |
| EVAL-06 | Validate offline oracle data path. | Held-out demonstration data and supplied labels are identified and hashed; no rollout labels are fabricated. | MODEL-09 | `PENDING` |

### 18.1 Fixed diagnostic definitions

These definitions must be used in implementation and reporting; a metric name alone is not sufficient evidence.

| Diagnostic | Locked definition |
|---|---|
| Utilization | Fraction of evaluation steps assigned to each expert by top-1 routing. If a future top-k row is run, report all-selected-expert fractions separately. |
| Entropy | Mean Shannon entropy of the full pre-top-k softmax distribution over six experts, normalized by `log(6)`. |
| Margin | Prototype router: `d_(2) - d_(1)` using the smallest and second-smallest prototype distances. Softmax router: pre-exploration `logit_(1) - logit_(2)`. |
| Transition count/rate | Adjacent in-trajectory pairs whose top-1 expert changes, requiring matching `trajectory_id` and consecutive `trajectory_position`; trajectory boundaries are excluded. |
| Collapse | Top-1 expert utilization `< 1/(5E)` using the final Stage 2 `expert_utilization.collapse_rate` definition; with `E=6`, threshold is `1/30`. The older initialization diagnostic threshold `<0.01/E` must remain separately labeled. |
| Oracle offline action loss | Action MSE on held-out demonstration states under supplied routing labels. |
| Oracle routing agreement | Agreement between supplied `phase_topo` labels and the learned phase head on held-out demonstration states. |
| Oracle per-phase error | Per-phase action error under label-directed routing on held-out demonstration states. |

The oracle has no ordinary rollout success-rate result in this matrix. A rollout success rate would require a separately validated privileged labeler for policy-generated states, which is out of scope.

## 19. Protocol-to-ledger traceability

Every section of the authority protocol has an explicit ledger destination.

| Protocol section | Covered by ledger |
|---|---|
| §1 Final decisions | Sections 2–4; GOV-01; MODEL-01, MODEL-10 |
| §2 Repository facts | LABEL-01–LABEL-09; ROUTER-01–ROUTER-05; PROVIDER-01; MODEL-01 |
| §3 Label/data contract | DATA-01–DATA-07; LABEL-01–LABEL-09 |
| §4 Action-path contract | Section 2; MODEL-10–MODEL-11; MAN-05 |
| §5 Experimental matrix | Section 3; Section 4; Section 15 run matrix |
| §6 Model implementations | MODEL-02–MODEL-13; TEST-03–TEST-11 |
| §7 Loss/label implementation | LABEL-01–LABEL-09; TEST-02, TEST-07, TEST-09 |
| §8 Checkpoint/runner provenance | PROVIDER-01–PROVIDER-09; TRAIN-05; TEST-12–TEST-13 |
| §9 Manifest structure | MAN-01–MAN-12 |
| §10 Validation gates | TEST-01–TEST-14; EVAL-03–EVAL-05; ROUTER-07 |
| §11 Dry-run/test sequence | DRY-01–DRY-06 |
| §12 Analysis/reporting | REPORT-01–REPORT-14; Section 18.1 |
| §13 Historical artifact policy | GOV-02, GOV-05; REPORT-13; completion gate item 13 |
| §14 Completion criterion | Section 20 final completion gate |

## 20. Final completion gate

The program is complete only when every item below is `VERIFIED`:

1. `precision_residual_phaseforge` is the sole proposed identity.
2. Final label fields and `topo_pelt_k6` are implemented and validated.
3. All final-aligned models exist and have exact declared differences.
4. Beta is zero and validated for every residual-expert row.
5. Router settings, margin loss, balance coefficients, and diagnostics are fixed.
6. Stage 1/Stage 2 hyperparameters and validation-MSE checkpoint selection are fixed.
7. Providers resolve exactly by task, seed, commit, config hash, and checkpoint hash.
8. The final manifest composes and dry-runs without historical alias fallback.
9. All unit, integration, composition, provider, and contract tests pass.
10. All deployable task/seed runs complete on identical frozen banks.
11. Teacher-forced and oracle diagnostics remain separate and correctly bounded.
12. Final reporting includes all required metrics, uncertainty, failure accounting, and provenance.
13. Historical results remain preserved and excluded from final claims.
14. Paper language matches the executed beta-zero direct-action method and measured evidence.

Until this gate passes, do not start publication tables, hard-delete historical artifacts, or relabel old results as final-generation evidence.

## 21. Decision and blocker log

Use this section for decisions that change scope or require a protocol revision. Do not silently edit a locked value.

| Date | Decision/blocker | Impact | Resolution / protocol revision | Owner | Status |
|---|---|---|---|---|---|
| 2026-09-08 | Ledger created from the finalized protocol. | Establishes execution tracking for the final baseline program. | No training authorized until gates pass. | Project | `VERIFIED` |
|  |  |  |  |  |  |

## 22. Evidence index

Record paths, commit IDs, hashes, and commands here as work is completed.

| Evidence ID | Description | Path / command | Commit | Hash / result | Date | Status |
|---|---|---|---|---|---|---|
| EVID-001 | Final protocol | `docs/final/baselines/PRECISION_RESIDUAL_BASELINE_PROTOCOL.md` |  |  | 2026-09-08 | `MAPPED` |
| EVID-001A | Archived professor reports | `docs/archive/professor_report1.md` through `professor_report5.md` |  |  | 2026-09-08 | `VERIFIED` |
| EVID-002 | Final causal manifest | `experiments/final_causal_matrix.json` |  |  |  | `PENDING` |
| EVID-003 | Test suite result |  |  |  |  | `PENDING` |
| EVID-004 | Provider registry/result index |  |  |  |  | `PENDING` |
| EVID-005 | Final evaluation bank index |  |  |  |  | `PENDING` |
| EVID-006 | Final result summary |  |  |  |  | `PENDING` |

## 23. Deferred legacy-baseline cleanup register

Legacy-baseline cleanup is a separate post-completion activity. It must not block implementation, and it must not begin merely because the final method is locked.

The cleanup rule is:

> Preserve legacy implementations, manifests, checkpoints, outputs, reports, and provenance until the final research record and provenance archive are complete. Prefer archiving over deletion. Any deletion must be a separate, reviewable change after the final completion gate passes.

The following are never permitted during final implementation or final training:

- deleting historical checkpoints or rollout outputs;
- deleting old manifests before their identities, commits, configuration hashes, and label policies are archived;
- editing old results in place to make them appear final-aligned;
- renaming an old method to `precision_residual_phaseforge`;
- deleting a file that is still required by a final provider, manifest, test, import, or historical reproduction path;
- deleting the archived professor reports or the final protocol.

| ID | Cleanup item | Required evidence / exit condition | Earliest timing | Status |
|---|---|---|---|---|
| CLEAN-01 | Inventory legacy baseline code, configs, manifests, checkpoints, outputs, reports, and references. | A path-level inventory identifies each item as active-final, historical-preserved, archive-candidate, or deletion-candidate. | After final matrix is complete | `PENDING` |
| CLEAN-02 | Protect final dependencies. | Search confirms no final manifest, provider, test, import, documentation link, or reproduction procedure depends on a proposed deletion target. | After CLEAN-01 | `PENDING` |
| CLEAN-03 | Freeze historical provenance. | Every retained legacy result has method identity, commit, resolved-config hash, provider, label policy, dataset/topology hash, reset-bank hash where applicable, and result path. | After final reporting | `PENDING` |
| CLEAN-04 | Create the immutable historical archive. | Archive contains the required legacy configs/results/reports and has an index with hashes and original paths. | After CLEAN-03 | `PENDING` |
| CLEAN-05 | Decide archive versus deletion per item. | Archive is the default. Deletion is allowed only for obsolete active files with no provenance or dependency role; the decision is recorded per path. | After CLEAN-04 | `PENDING` |
| CLEAN-06 | Obtain cleanup approval through the project review process. | The cleanup scope, exact paths, rationale, recovery status, and expected impact are reviewed before any deletion. | After CLEAN-05 | `PENDING` |
| CLEAN-07 | Execute cleanup in a dedicated change. | No final experiment or protocol change is mixed into the cleanup change; exact deleted/moved paths and commit are recorded. | After CLEAN-06 | `PENDING` |
| CLEAN-08 | Verify the repository after cleanup. | Final manifest composition, provider resolution, tests, historical archive checks, and documentation links still pass. | After CLEAN-07 | `PENDING` |
| CLEAN-09 | Record recoverability. | For every removed item, state whether it remains recoverable from the archive or Git history; material deletion is reported explicitly. | After CLEAN-07 | `PENDING` |

### 23.1 Initial cleanup classification

Until the inventory is completed, use these provisional classifications:

| Category | Treatment |
|---|---|
| `precision_residual_phaseforge` and its final providers/configuration | Active final; never delete as legacy. |
| `precision_residual_*` final-aligned controls and diagnostics | Active final until the matrix and reporting are complete; do not delete during implementation. |
| Old `warmstart_moe`, `plain_encoder_phase_bootstrap`, `phase_pretrain_random_router`, `scratch_moe`, `teacher_forced`, and `oracle_moe` | Historical or archive-candidate; preserve identity and results; do not relabel. |
| Old `phaseforge`, `phaseforge_dynamic`, `phaseforge_r50`, and other development variants | Historical or archive-candidate; preserve their provenance and exclusion status. |
| `experiments/five_task.json` and other historical manifests | Historical; do not edit in place for the final matrix. |
| Archived professor reports and `PRECISION_RESIDUAL_BASELINE_PROTOCOL.md` | Required provenance/authority; never delete. |

Cleanup is not a condition for claiming the final method. It is complete only when CLEAN-01 through CLEAN-09 have acceptable evidence and the final completion gate remains satisfied.
