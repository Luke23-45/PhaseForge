# PhaseForge — Research Data Provenance and Metrics Design (Review Draft)

**Status:** proposal awaiting review — no code has been changed for this document

**Related plans:** [final_evaluation_plan.md](final_evaluation_plan.md), [research_definition.md](research_definition.md)

**Scope of this document:** the exact set of training-time and evaluation-time
data that must be persisted per run so that the research paper can be written
from the recorded artifacts alone. It covers what is already implemented, what
is missing, and the proposed schemas and layout. It does not modify any code.

---

## 0. Purpose

The paper must present, for every model in the matrix and every training seed:

1. training curves (loss components, and for MoE methods the routing
   diagnostics) as a function of epoch;
2. a training-cost table (epochs, wall time, parameter counts);
3. final evaluation tables with confidence intervals (already covered by the
   existing eval aggregation);
4. model-type-specific diagnostics that only apply to certain cells
   (e.g. teacher-forced routing accuracy).

Everything below is designed so that **the paper can be assembled from the
persisted artifacts alone** — no dependence on wandb, console logs, or memory
state that is lost when a cloud session ends.

The current implementation is directionally correct (the eval side matches the
target layout) but the **training side does not yet persist the information the
paper needs**. Section 3 is the compliance gap; Sections 4–8 are the proposal.

---

## 1. Target outputs layout (the contract)

```text
outputs/
  _ledger/runs.jsonl + index.json        # every train AND eval run, one row   [implemented]
  _results/results.jsonl                 # global append-only EVAL rows (aggregation source) [implemented]
  _results/training_summary.jsonl        # global append-only TRAIN summary rows (PROPOSED)
  {model}/stage{N}/{ts}_{runid}/         # unchanged layout:
      resolved_config.yaml               #   [implemented]
      run_meta.json                      #   [implemented — MUST add kind]
      metadata/environment.json          #   [implemented]
      timings.json                       #   [implemented — total only; PROPOSE per-epoch]
      metrics/training_curves.jsonl      #   PROPOSED: one validated row per epoch
      metrics/summary.json               #   PROPOSED: per-run final scalars at on_train_end
      checkpoints/                       #   [implemented]
  eval/{model}/{ts}_{runid}/             # unchanged layout + [all implemented]:
      eval_results.json                  #   per-run snapshot
      metadata/environment.json, timings.json
  _summaries/                            # PRODUCED by summarize tooling:
      aggregates.csv                     #   eval side      [implemented]
      bootstrap_ci.csv                   #   eval side      [implemented]
      paired_wilcoxon.csv                #   eval side      [implemented]
      metrics.json                       #   eval side      [implemented]
      training_aggregates.csv            #   PROPOSED
      training_cost.csv                  #   PROPOSED
      training_curves.csv                #   PROPOSED
```

---

## 2. Design principles

**P1. Self-contained run directories.** A run directory must contain everything
needed to reproduce its tables: resolved config, environment fingerprint,
training curves, final summary, and checkpoints. Nothing the paper reports may
live only in wandb or in a process that has ended.

**P2. Schema-validated append-only ledgers.** Every row written to a shared
ledger is validated before it reaches disk, following the discipline already
implemented in `phaseforge/outputs_writer/schema.py`. Unknown or mistyped data
fails loudly at write time, never silently at analysis time.

**P3. One canonical key space.** All metrics use the existing `train/` and
`val/` prefix convention. A metric means the same thing in every file that
mentions it.

**P4. Model-type-aware schema.** A single row schema with *required core fields*
plus *optional stage/model-specific fields*. A `bc` row may omit every routing
field; a `phaseforge` Stage 2 row must carry them. Absence is honest (reported
as `null`/`n=0` in aggregates), never fabricated as zero.

**P5. Epoch resolution on disk, step resolution in memory.** Per-epoch rows are
small and sufficient for every loss/routing curve the paper plots. Per-step
values stay in the trainer (and wandb when enabled); they are not persisted, to
keep every run bounded in size.

**P6. Aggregation is derived, never hand-edited.** All aggregate CSVs are
regenerated from the ledgers by idempotent tooling (`summarize_eval.py` and its
proposed training counterpart), exactly as the eval side already works.

---

## 3. Current state vs. target — compliance matrix

| Artifact | Status | Note |
|---|---|---|
| `_ledger/runs.jsonl` + `index.json` | ✅ implemented | one row per train and eval run; `pending→completed/failed` lifecycle |
| `_results/results.jsonl` | ✅ implemented | schema-validated eval rows; the eval aggregation source |
| `run_meta.json` | ⚠️ partial | has `model_name`, `stage`, `seed`, `device`, git, `config_hash`, `tag`; **missing `kind`** |
| `metadata/environment.json` | ✅ implemented | deps, git, host, data-config hash, config hash |
| `timings.json` | ⚠️ partial | total start/end/wall seconds; **no per-epoch timing** |
| `checkpoints/` | ✅ implemented | full trainer state (model/opt/scheduler/RNG/callbacks) |
| eval run dir + `eval_results.json` | ✅ implemented | per-run snapshot; result row also carries `ckpt_path` for traceability |
| `_summaries/` eval artifacts | ✅ implemented | aggregates / bootstrap CI / paired Wilcoxon / metrics JSON |
| **Training curves (loss components vs epoch)** | ❌ **missing** | only in `MetricTrackerCallback` memory or wandb (disabled by default) — **lost after the run** |
| **MoE routing dynamics vs epoch (NMI, balance, collapse, entropy)** | ❌ **missing on disk** | computed every epoch in `Stage2Trainer._validate` but only logged to console |
| **Phase-classification accuracy** | ❌ **missing** | Stage 1 logs only `loss_phase`; the accuracy the plan requires (§8, Gate 3) is never computed |
| **Teacher-forced routing accuracy (predicted vs GT phase at inference)** | ❌ **missing** | the core diagnostic for the `teacher_forced` cell |
| **Per-epoch / per-step timing, steps-per-second** | ❌ **missing** | only total run wall time exists |
| **Trainable-parameter counts** | ⚠️ partial | logged at Stage 2 init (console only), never persisted |
| **Training-side aggregate tables** | ❌ **missing** | no `training_aggregates.csv` / cost table |

**Headline gap:** today, a completed training run leaves on disk its config,
checkpoints, and a wall-clock timestamp — but **none of its training curves**.
Every curve the paper would plot must be re-derived or is lost. This is the
single most important missing piece.

---

## 4. Data inventory by model and stage

The matrix (from `scripts/run_multi_seed_train.py`) has eight cells and two
training stages. The inventory is organized as: **core** (every run), **Stage 1
extras**, **Stage 2 extras**, and **cell-specific diagnostics**.

### 4.1 Core — every run, every stage

| Field | Source | Notes |
|---|---|---|
| `train/loss_total`, `train/loss_action` | `Stage1Trainer`/`Stage2Trainer._compute_loss` | already emitted per step; needs aggregation to epoch + persistence |
| `val/loss_total`, `val/loss_action` | `BaseTrainer._validate` (sample-weighted) | already computed |
| `train/lr` | `WandbLoggerCallback` reads `optimizer.param_groups[0]["lr"]` | move the read into the persisted row |
| `epoch` , `global_step` | trainer state | |
| `epoch_wall_seconds`, `steps_per_second` | **new**: time around `_train_epoch` + `_validate` | appendix cost table |
| `trainable_params`, `total_params` | **new**: count after freeze logic | persist, not console-only |

### 4.2 Stage 1 extras (phase-supervised pretraining)

Applies to `phaseforge` Stage 1 (phase-supervised). `bc` shares the Stage 1
loop but has no phase head, so its phase fields are schema-optional and are
omitted in `bc` rows (the loop would otherwise emit a constant `loss_phase` of
0).

| Field | Source | Notes |
|---|---|---|
| `train/loss_phase`, `val/loss_phase` | already emitted (Stage 1 only) | `bc` (no phase head) → schema-optional, omitted in `bc` rows |
| `train/phase_acc`, `val/phase_acc` | **new**: argmax of `phase_logits` vs `phase` label | required by plan §8 / Gate 3; currently never computed |
| `lambda_phase` | `cfg.train.lambda_phase` | part of resolved config; repeated in the row for table convenience |

### 4.3 Stage 2 extras (bootstrapped MoE)

Applies to every MoE cell: `phaseforge`, `scratch_moe`, `warmstart_moe`,
`oracle_moe`, `phase_pretrain_random_router`,
`plain_encoder_phase_bootstrap`, `teacher_forced`.

| Field | Source | Notes |
|---|---|---|
| `train/loss_balance` | already emitted | |
| `val/routing_entropy` | `Stage2Trainer._validate` | pre-top-k, normalized |
| `val/phase_expert_nmi` | same | emergent specialization (C3) |
| `val/topk_balance_score`, `val/top1_balance_score` | same | uses configured expert count; dead experts count |
| `val/topk_collapse_rate`, `val/top1_collapse_rate` | same | |
| `balance_coeff` | `models.router.balance_coeff` | resolved config; repeated for convenience |
| `freeze_encoder` | `cfg.train.freeze_encoder` | resolved config; repeated for convenience |

These are exactly the "balance-vs-NMI trajectory" diagnostics the plan calls out
(C3); today they exist only in console lines and are unrecoverable afterwards.

### 4.4 Cell-specific diagnostics

| Cell | Diagnostic | Why it is needed |
|---|---|---|
| `teacher_forced` | **routing accuracy**: fraction of inference steps where the learned phase predictor agrees with the GT phase used to partition experts | the plan requires it to be labeled privileged-training and to measure the predictor, not emergent specialization |
| `oracle_moe` | none beyond §4.3 — its routing is by GT labels | NMI is a sanity check (≈1.0), never presented as a learned result |
| `bc` | none beyond §4.1 | the action-only floor |
| `phaseforge` | none beyond §4.2/§4.3 | its mechanism claim is read from NMI + stability + success |

### 4.5 Eval-side inventory

Already complete: per-run `eval_results.json` (full metric set + definitions),
the schema-validated `results.jsonl` rows (with `ckpt_path`, `stage`, `seed`,
`config_hash`, device, git), and the four `_summaries/` artifacts. No new eval
fields are proposed.

---

## 5. Proposed artifacts and schemas

### 5.1 Per-run training curve file — `metrics/training_curves.jsonl`

One row per epoch, validated before write, stored inside the run directory.

```json
{
  "run_id": "a1b2c3d4",
  "epoch": 42,
  "global_step": 8400,
  "train/lr": 0.00021,
  "epoch_wall_seconds": 12.7,
  "steps_per_second": 1580.0,
  "train/loss_total": 0.031,
  "train/loss_action": 0.029,
  "val/loss_total": 0.028,
  "val/loss_action": 0.027,
  "train/loss_phase": 0.42,          // Stage 1 only (optional)
  "val/loss_phase": 0.40,            // Stage 1 only (optional)
  "train/phase_acc": 0.91,           // Stage 1 only (optional)
  "val/phase_acc": 0.89,             // Stage 1 only (optional)
  "train/loss_balance": 0.001,       // Stage 2 only (optional)
  "val/routing_entropy": 0.88,       // Stage 2 only (optional)
  "val/phase_expert_nmi": 0.46,      // Stage 2 only (optional)
  "val/topk_balance_score": 0.995,   // Stage 2 only (optional)
  "val/top1_balance_score": 0.982,   // Stage 2 only (optional)
  "val/topk_collapse_rate": 0.0,     // Stage 2 only (optional)
  "val/top1_collapse_rate": 0.0,     // Stage 2 only (optional)
  "val/routing_accuracy": 0.84       // teacher_forced only (optional)
}
```

The example shows the **union** of optional fields for illustration; a real row
carries only the fields for its stage and model — a Stage 1 `phaseforge` row has
core + phase fields, a Stage 2 `phaseforge` row has core + routing fields, a
`bc` row has core only.

Validation rules mirror `schema.py`: required core keys (`run_id`, `epoch`,
`global_step`, `train/lr`, the four loss fields, `epoch_wall_seconds`,
`steps_per_second`); stage/model fields are optional but type-checked
(finite-or-NaN numeric) when present; unknown top-level keys rejected. `bc`
rows carry only the core block.

### 5.2 Per-run final summary — `metrics/summary.json`

One small file per run, written at `on_train_end`, holding the scalars that feed
the training tables:

```json
{
  "run_id": "a1b2c3d4",
  "kind": "train",
  "model": "phaseforge",
  "stage": 2,
  "seed": 42,
  "config_hash": "defcde910a30a91f",
  "data_config_hash": "cd33f196b7778688",
  "git_sha": "c0e72de",
  "device": "cuda:0",
  "started_at": "...", "finished_at": "...", "wall_seconds": 2510.3,
  "epochs_run": 200,
  "trainable_params": 1843200, "total_params": 2345678,
  "best_epoch": 176, "best_val_monitor": 0.0241,
  "final_val": {"loss_total": 0.0243, "routing_entropy": 0.88, "phase_expert_nmi": 0.46, "...": "..."},
  "best_checkpoint": "checkpoints/checkpoint_best.pt",
  "extra": {}
}
```

### 5.3 Global training summary ledger — `_results/training_summary.jsonl`

One append-only, schema-validated row per completed training run (the exact
analog of `results.jsonl` for the eval side). The persistence callback writes
`metrics/summary.json` at `on_train_end`; the CLI then appends the
`training_summary.jsonl` row after `trainer.fit()` returns, best-effort, in the
same pattern as the eval result row. This is what the training aggregates are
computed from. The ledger keeps `results.jsonl` eval-only so the eval
aggregation contract is untouched.

### 5.4 Proposed summary artifacts — `_summaries/`

Extends the existing summarize tooling (e.g. `scripts/summarize_eval.py` gains a
training mode or a sibling `summarize_train.py`):

| File | Content |
|---|---|
| `training_aggregates.csv` | per `(model, stage)`: mean ± std over seeds of `final_val` scalars, `best_val_monitor`, `epochs_run`, `trainable_params` |
| `training_cost.csv` | per `(model, stage)`: `wall_seconds` (mean ± std), `epochs_run`, total `global_step`, params — the appendix cost table |
| `training_curves.csv` | per `(model, stage, epoch)`: mean ± std over seeds of every curve metric — the plot source |

### 5.5 Small consistency fix

`run_meta.json` gains `"kind": "train" | "eval"` (it already carries `stage`,
so this completes the identity block). No other run_meta change.

---

## 6. Storage and cost estimate

Training runs in the matrix: 9 runs per seed (bc=1, phaseforge=2, and six
Stage-2-only cells) × 3 seeds ≈ **27 training runs**. At ~100–200 epochs each
and ~25 fields per full Stage 2 row:

- `training_curves.jsonl`: ≈ 25–90 KB per run (a 9-field `bc` row is ~250 B;
  a 16-field Stage 2 row is ~450 B) → **≈ 1–2 MB total** across all runs;
- summary files + training ledger + aggregates: negligible (< 100 KB).

Eval side is unchanged. The proposal adds no material storage or compute
overhead; the only new compute is phase accuracy (one argmax over the already
computed `phase_logits` during validation) and per-epoch wall-clock timing.

---

## 7. Scope boundaries (deliberately not captured)

- **Per-step values are not persisted** on disk (bounded run size). Step-level
  detail remains available to wandb when enabled and to the in-memory
  `MetricTrackerCallback` during the run.
- **Tensors/weights beyond checkpoints** are not dumped per epoch. Checkpoints
  already capture full trainer state for any requested re-analysis.
- **No new evaluation metrics** are proposed; the eval schema and aggregation
  are considered complete for the current protocol.

---

## 8. Open questions for review

1. **Curve granularity** — is epoch-level persistence sufficient, or should a
   config flag allow step-level curve persistence for the main cells?
2. **Training ledger placement** — confirm the proposal to keep
   `results.jsonl` eval-only and add `training_summary.jsonl` alongside it
   (rather than merging both into one `runs.jsonl`).
3. **Phase accuracy definition** — confirm argmax over `phase_logits` vs the
   offline annotation label, computed per validation batch, as the Stage 1 /
   teacher-forced accuracy definition.
4. **Cost table scope** — whether `training_cost.csv` should also report peak
   GPU memory (requires a small `torch.cuda.max_memory_allocated()` read at
   epoch end; cheap on CUDA, empty on CPU).
5. **Checkpoint granularity** — whether a per-epoch record of the checkpoint
   monitor value (already in `checkpoint_best.pt` selection) should also appear
   in the curve row for plotting selection curves.

---

## 9. Implementation sketch (post-approval, not yet started)

1. Add per-epoch timing and trainable/total parameter counts in
   `BaseTrainer`/`Stage1Trainer`/`Stage2Trainer`; implement the persistence as a
   callback on the existing `on_epoch_end`/`on_train_end` hooks (defined in
   `phaseforge/trains/callbacks/base.py`); compute phase accuracy in the
   Stage 1 validation path and routing accuracy (predicted vs GT phase) in the
   teacher-forced validation path.
2. Add a `TrainingCurveWriter` (mirroring `RunWriter`) that appends validated
   epoch rows to `metrics/training_curves.jsonl` with `FileLock` + fsync, and a
   `metrics/summary.json` at `on_train_end`.
3. Wire the CLI to append the schema-validated `training_summary.jsonl` row
   after training completes (best-effort, exactly like the eval result row).
4. Extend the summarize tooling to emit `training_aggregates.csv`,
   `training_cost.csv`, `training_curves.csv`.
5. Add unit tests: schema validation for curve/summary rows, per-run writer
   behavior, ledger append/read, and aggregate correctness on synthetic
   multi-seed data. All CPU-only.

No step of this sketch has been executed; implementation waits on approval of
Sections 4–5 and the answers in Section 8.
