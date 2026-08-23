# Training hot-path performance review — Historical review (T5 reverted)

> **Current implementation status (2026-08-23):** The T5 `InMemoryBatchLoader`
> replacement was reverted because it changed the `RandomSampler` permutation
> stream without producing a verified wall-time improvement. The active
> training path uses the legacy `DataLoader`/collator path. The remaining
> behavior-preserving training changes from this review remain in the code;
> T5-specific measurements and claims below are historical only.

Date: 2026-08-22 (rev 3: implementation record added; rev 2 was the research
pass). Scope: the training path only (`phaseforge-train` stage 1 + stage 2,
all 10 methods × 5 tasks × 3 seeds = 315 runner steps + 165-step ablation
suite). Rollout was optimized separately (D12); this document is the training
analogue, built the same way: line-level code review + measurements on this
box, external claims verified against primary sources, nothing assumed.
**Rev 2 added a per-issue online research pass** (§0); **rev 3 records the
implementation** (§8): all patches T1, T1b, T2, T3, T4, T5, T6, T7a are
implemented and gated; T7b (stacked metric adds) was evaluated and NOT taken
(conditional metric key-sets make vectorization change grouping for no
measurable gain — the plan's "optional" framing); the `get_rng_state_all`
per-save sync stays by documented decision (resume fidelity).

**Historical status: implemented in commit `7baf0ce`, then partially reverted**.
The
verification gates in §8 passed: bit-identity for T1–T4, determinism for T5,
full suite 738 green, preflight 165+150, dry-runs exactly 315 and 165.

Environment of all measurements: Windows dev box, `.venv-rollout` (py3.11,
torch 2.13.0+cpu, 2 intra-op threads), Lift/ToolHang processed caches. The
final sweep runs on a cloud GPU (Linux), so every finding states both the
measured CPU fact and the cloud implication; GPU-specific numbers are marked
**estimate** (no GPU on this box).

---

## 0. Research pass (rev 2) — what the check against the community changed

Each finding's patch was re-examined against outside best practice. Outcomes:

| Issue | Community/official state | Effect on the plan |
|---|---|---|
| T1 version lookups | Repeated `importlib.metadata.version()` calls each rescan installed dists; the recommended robust form is ONE `distributions()` pass building a name→version dict (importlib_metadata#95; Python Discourse) | **T1 upgraded** to single-pass dict + import fallback |
| T1b wandb import | — | **New finding** from the second code sweep: `cli.py:485` imports wandb unconditionally (~2.2 s when installed, even with `mode=disabled`) — guard the import on the mode flag |
| T2 MoE syncs | Sparse top-k dispatch IS this code's design lineage (Shazeer 2017 noisy top-k gating, arXiv:1701.06538); dense compute-all-experts is a different mode, and would change every expert GEMM's batch size → float-path drift | **T2 kept** as the bit-identical sync removal; dense variant rejected |
| T3 loss guard | `torch._assert_async` enables a sync-free device-side assert — but on failure it corrupts the CUDA context, the error surfaces later at an unrelated line, the custom message is ignored (pytorch#131491), and the API is private/sparsely documented (Lei Mao's write-up; pytorch#36853) | **`_assert_async` REJECTED** (cheap-looking, not robust); epoch-end on-device flag kept |
| T4 validation syncs | On-device accumulation + single materialization is the standard recommendation (PyTorch tuning guide) | **T4 kept** unchanged |
| T5 loader | GPU/CPU-resident flat tensors + `randperm` + `index_select`, skipping the DataLoader for in-memory data, is the canonical pattern (PyTorch forums #27609/#79165, SO, Lightning #2361) | **T5 validated**; refined to always draw the permutation with a CPU generator (portable, avoids generator/device mismatch rules entirely) |
| Optimizer | Official AdamW docs: `foreach=None` (our default) already selects the foreach implementation on CUDA; `fused` would be fastest but changes float paths | **No optimizer change** (default already optimal; drift avoided) |
| torch.compile | Official caching tutorial + pytorch#114206/#113287/#96152: FX-graph cache reduces but does NOT eliminate per-process cold start (Dynamo tracing + AOTAutograd re-run every process; multiprocess reuse historically buggy) | **Rejection reaffirmed** with primary sources |

No research finding invalidated a proposed patch; two designs were upgraded
(T1 single-pass, T5 CPU-generator perm), one new issue was found (T1b), one
tempting alternative was rejected for robustness (`_assert_async`), and the
torch.compile rejection now rests on official sources instead of estimates
alone.

---

## 1. Where the time actually goes (measured)

3-epoch Lift stage-1 run: **72.4 s wall** of which the three epochs are
17.2 + 0.92 + 1.55 s. 3-epoch Lift stage-2: **35.4 s wall**, epochs ≈ 4.9 s.
Training compute is NOT the dominant term on small tasks; fixed per-process
cost is. Per-batch/step costs (clean, no profiler):

| Item | Measurement | Note |
|---|---|---|
| Python+torch imports | 3.4 s | `torch` alone ≈ 3.0 s |
| `collect_environment` | 8.3 s under profiler | **imports sklearn + wandb + scipy just for `__version__`** (§2.1) |
| Cache load (200 `.pt`) | 0.46 s | per-process, ×315 |
| Model+trainer init | 2.9 s | incl. first optimizer build |
| DataLoader worker spawn | 6.9 s (4 procs) | Windows spawn pickles the dataset ×2; ≈0 on Linux fork (**estimate**) |
| First epoch vs steady | 11.6 s vs 1.15 s | spawn + torch kernel warm-up |
| Run finalization | ≈3 s | checkpoint SHA-256 + artifact manifest |
| Batch fetch, workers=2 | 12.1 ms/batch steady | 197 ms/batch in epoch 1 (spawn) |
| Batch fetch, workers=0 | 50–54 ms/batch | in-process prep is 68 ms/batch (§2.5) |
| Step, stage-1 (B=256) | 17.4 ms | CPU fwd+loss+bwd+clip+step |
| Step, stage-2 (B=256) | 31.0 ms | +6 `Tensor.any()` calls/step measured |
| ToolHang epoch (337 batches) | 9.7 s steady | 29.8 s first; run 84.5 s / 3 epochs |

Dataset scale (train batches/epoch @ B=256): Lift 33, Can 81, Square 106,
ToolHang 337, Transport 328. A ToolHang stage-2 run is 337 × 200 = **67,400
steps** plus 67-batch validation × 200 epochs — the big two tasks dominate
sweep training time; per-step and per-epoch costs are what multiply there.

---

## 2. Findings and proposed patches

### 2.1 F1 — `collect_environment` imports sklearn/wandb/scipy per process (P0, measured)

`phaseforge/outputs_writer/metadata.py:21` `_safe_version` does
`__import__(modname)` for 9 packages. Measured on this box: importing
`sklearn` ≈ 2.5 s, `wandb` ≈ 2.2 s, `scipy` ≈ 0.4 s — **≈ 4.5–5 s per training
process** (stage-1 runs import none of them otherwise; `wandb.mode=disabled`
means the import is pure waste). ×315 cells ≈ **21–26 min of sweep time**.

`importlib.metadata.version("scikit-learn")` costs **2.6–10.8 ms** (measured,
all 9 packages ≈ 45 ms) because it reads dist metadata without importing.

**Patch T1 (proposed, research-upgraded):** replace the 9 import-based
lookups with ONE `importlib.metadata.distributions()` pass that builds a
name→version dict (robust form recommended by importlib_metadata#95 —
repeated `version()` calls each rescan installed dists), consulted through an
explicit module→dist-name map (`hydra`→`hydra-core`, `sklearn`→
`scikit-learn`), falling back to the current import-and-`__version__` only
for packages missing from metadata. Same `environment.json` keys, same
version strings for all 9 (verified above). Risk: negligible. Verification:
unit test asserting equality with the imported `__version__` for every
installed package.

**Patch T1b (proposed, new finding from the second code sweep):**
`cli.py:485` (`_train_body`) does `import wandb` unconditionally — with wandb
installed and `wandb.mode="disabled"` (the default) this still pays the ~2.2 s
import every training process (measured). Guard the import on the mode flag:
only `import wandb` when `mode != "disabled"`. Honest accounting note: stage-2
cells still import sklearn once (per-epoch NMI is protocol), so T1 removes the
cost fully only for stage-1-family cells; for stage-2 cells it moves from
bookkeeping to the first validation call.

### 2.2 F2 — MoE dispatch performs 6 host-device syncs per step (P0 on GPU)

`phaseforge/models/components/moe_layer.py:168` — the per-expert loop does
`if not match_mask.any(): continue`. Converting a CUDA tensor to bool
synchronizes the device (official PyTorch performance guide: bool conversion,
`.item()`, `.cpu()` transfers, and computed-index ops are implicit sync
points; same list in NVIDIA's sync-free guide — sources §7). Measured call
count: **exactly 6 `.any()` per forward** (one per expert) → 6 syncs/step ×
67,400 steps = **404k syncs per ToolHang stage-2 run**. Each sync flushes the
async queue; with this model's µs-scale kernels the stall dominates the step
(**estimate: 0.5–2 ms per sync on small kernels → plausibly 30–60% of
stage-2 step time on GPU**).

The `continue` saves nothing measurable: with B=256, top-2 over 6 experts and
a balance loss, an expert is almost never unselected, and a `(0, D)` expert
forward is a no-op. **Verified empirically on this box**: empty input forward,
empty `index_add_`, and empty weight-gather all succeed as exact no-ops.

**Patch T2 (proposed):** delete the `.any()` early-continue; always run the
per-expert gather/forward/accumulate. This is **bit-identical by
construction** (empty slices contribute nothing; non-empty slices run the
same ops). CPU also wins slightly (6 Python branches + 6 kernel launches
fewer). Verification: unit test forcing routing where 0–5 experts receive
items, asserting outputs identical to the current implementation under a
fixed seed.

*Research note (rev 2):* the dense alternative — run every expert on the full
batch and weight by sparse top-k gates — is a real, known mode but is NOT
what this code implements: the router's lineage is Shazeer 2017 sparse
noisy top-k gating (arXiv:1701.06538; the router docstring says exactly
this), and dense compute would change every expert GEMM's batch size
(variable-N → fixed-256), which can alter kernel selection/reduction order —
float-path drift for zero additional need at this model scale. Sparse
dispatch stays; only the sync is removed.

### 2.3 F3 — per-step `not torch.isfinite(loss)` host sync (P1)

`phaseforge/trains/loops/base.py:451` — `not torch.isfinite(loss)` converts a
device tensor to bool: **1 sync/step** (67.4k per ToolHang stage-2 run). When
`grad_clip_norm > 0` (both stage configs: 1.0) the guard is already redundant
per-step: `clip_grad_norm_(error_if_nonfinite=True)` (base.py:466) aborts on
non-finite *gradients*, which any non-finite loss produces. The separate
no-clip branch (base.py:480) is worse: a per-parameter `isfinite().all()`
loop = ~50 syncs/step, only on non-default configs.

**Patch T3 (proposed):** accumulate an on-device finite-flag
(`torch.isfinite(loss)` OR-reduced into a 0-dim tensor, no sync) per batch;
materialize and check it once at epoch end — still before the epoch-end
checkpoint save, so a corrupted run can never persist a checkpoint, and the
abort message still names the epoch. In the no-clip branch, replace the
per-parameter loop with one concatenated `isfinite` check (1 sync), keeping
per-parameter naming only in the (rare) failure path. Behavior change: abort
happens at epoch end instead of the exact batch — the guarantee that matters
(no checkpoint/ledger row from a corrupted run) is preserved. Verification:
unit test injecting NaN loss (clip on and off) asserting the run aborts
before any checkpoint write.

*Research note (rev 2) — `torch._assert_async` considered and REJECTED:* it
would enforce finiteness per-step with zero synchronization (device-side
assert, pytorch#36853), but on failure it corrupts the CUDA context (process
unusable), the error surfaces asynchronously at a later, unrelated line
(misleading traceback for the sweep logs), the custom `assert_msg` is ignored
(pytorch#131491), and the API is private with no stable documentation. A
failure-handling path that destroys diagnosability is a cheap trick, not a
robust patch — rejected in favor of the epoch-end flag, which fails loudly,
cleanly, and before any artifact write.

### 2.4 F4 — validation synchronizes per metric per batch (P1)

`base.py:591` and `stage2_loop.py:262` call
`self._metric_to_float(v) * n` per tensor metric per validation batch
(`.detach().cpu().item()` = sync). Stage-2 metrics: 5 device tensors →
**5 syncs × 67 val batches = 335 syncs/epoch → 67,000 per ToolHang stage-2
run** (stage-1: 3 × 67 × 100 = 20k). `stage2_loop.py:266-272` additionally
transfers `expert_indices`/`gate_logits`/phases to CPU per batch. The
codebase already has the right pattern (`_PhaseAccumulator`, the persistence
callback's on-device epoch accumulation) — validation just doesn't use it.

**Patch T4 (proposed):** accumulate loss metrics on device (same incremental
per-batch adds in the same order → **bit-identical epoch means**), appending
routing tensors to a device list and doing ONE `torch.cat` + one `.cpu()` at
epoch end (1 sync + 1 transfer per epoch instead of ~340). Verification:
unit test asserting epoch aggregates equal the current inline computation
exactly (same accumulation order), plus a NaN-propagation test.

### 2.5 F5 — per-item Python data path (biggest per-step lever, P1)

`dataset.py:161-193` `__getitem__` performs ~7 tensor ops + a dict +
3 scalar-tensor creations per sample — measured **225 µs/item = 58 ms per
256-batch** (+10.3 ms collate). With workers=2 the effective fetch is
12.1 ms/batch (prep parallelized across workers + IPC), so the loader is
already overlapped on CPU — but that 12 ms floor is **worker IPC
(serialize/queue/pin 8 stacked tensors)**, not prep. On the cloud GPU, where
the step itself shrinks to a few ms, the loader IPC becomes the likely
bottleneck (**estimate**). Windows-only: worker spawn costs 6.9 s/run
(2 loaders × 2 workers, dataset pickled per worker); Linux fork makes this
≈0.

**Patch T5 (proposed, recommended; research-validated): flat in-memory batch
iterator for the single-step configs.** All non-RNN cells have
`sequence_length=1, stride=1`: the whole split is a fixed set of rows. At
dataset build, concatenate all trajectories ONCE into flat
`state (N,S) / action (N,A) / phase (N,) / task_id / trajectory_id /
trajectory_position` tensors (the corruption path already produces
per-trajectory labels before this point, so it composes unchanged). Per
epoch: draw `perm = torch.randperm(N, generator=g)` with a CPU
`torch.Generator` seeded from `project.seed` (mirroring the existing
`_train_sampler_generator` contract), slice into `drop_last`-parity batches,
and `index_select` the 6 fields — **one gather per field per batch, zero
per-item Python**. Val iterates sequential slices. `num_workers` becomes
irrelevant on this path (no spawn, no IPC); the RNN cells (`*_rnn.yaml`,
`sequence_length=10`) keep the existing DataLoader+workers path, dispatched
by config. Callers (trainer loops, `bootstrap_moe`,
`compute_init_routing_diagnostics`) only iterate and index `batch["..."]` —
interface unchanged.

*Research note (rev 2):* resident-tensor + `randperm` + `index_select`
batching, skipping the DataLoader for in-memory data, is the canonical
community pattern for exactly this situation (PyTorch forums threads "How to
load all data into GPU for training" and "Load entire dataset on GPU",
Stack Overflow, Lightning issue #2361). One deliberate divergence: those
threads generate the permutation on the device; we keep a **CPU generator**
always (one small per-epoch index transfer in the optional GPU-resident
variant) because CPU-generator randperm is portable across devices and
machines and avoids generator/device-mismatch rules entirely —
determinism-robustness beats saving one ~2 MB transfer.

- Content: **bit-identical tensors** (same rows, same values; gather is exact).
- Order: the permutation sequence differs from `RandomSampler`'s generator
  consumption — same distribution, still 100% seed-reproducible. Disclosed as
  a protocol-neutral loader change (applies identically to every cell, so
  fairness is untouched; training hyperparameters unchanged).
- Cost: measured in-process prep 68 ms → expected ~1–2 ms/batch (6 gathers on
  ~35 KB); on-device-resident variant possible later (dataset is ~7–15 MB)
  but NOT proposed now — keep the change minimal and CPU/GPU-agnostic.

Conservative alternative (documented, not recommended): keep the DataLoader
and only flatten `__getitem__` to row-picks. Measured arithmetic says this
still leaves the ~12 ms worker-IPC floor, i.e. it does not beat the current
steady state — which is why the full iterator is the recommended form.

Verification: (i) content gate — fast-path epoch yields exactly the same
sample multiset as `StateOnlyDataset` (+ `drop_last` parity, batch count
equal); (ii) with an injected fixed permutation, batches bit-equal to
collated DataLoader batches; (iii) determinism — same seed twice → identical
batch sequence hash; (iv) RNN configs provably still take the legacy path.

### 2.6 F6 — repeated config hashing + git subprocesses (P2, minor)

Per run, `CacheManager.compute_hash(cfg.data)` runs 3× (bookkeeping,
`write_run_meta`, `_copy_data_provenance`) and each call re-runs
`git_commit()` (subprocess, measured 146 ms) + raw-file stats + `to_container`;
`config_hash(cfg)` likewise 3×. `utils/config.py:git_info` is already
`lru_cache`d — `cache_manager.git_commit` is not, and hashes are not reused.

**Patch T6 (proposed):** compute the data hash once in `cli.train` and pass
it down (the code already threads `data_config_hash` through most callers);
`functools.lru_cache` on `git_commit`. Saves ≈0.5–0.8 s × 315 ≈ 3–4 min.
Risk: none (same values).

### 2.7 F7 — micro items (optional, bundled for review)

- `stage2_loop.py:170,155` constructs `torch.tensor(0.0, device=...)`
  defaults every step (3 small device allocations × 67k steps) — cache one
  zero scalar per device, or reuse `action_pred.new_zeros(())`-style
  creation. Bit-identical.
- `persistence.on_train_batch` does ~7 separate on-device tensor adds per
  step; could stack metrics into one vector add. Bit-identical if summed in
  the same order. (~µs-scale on CPU, small launch saving on GPU.)
- `CheckpointCallback._build_state` calls `torch.cuda.get_rng_state_all()`
  per save (sync). Kept deliberately: resume fidelity outranks the saving;
  disclosed here so it is a decision, not an accident.
- `_batch_sample_count` `.item()` when `padding_mask` exists — RNN cells
  only; folded into T4's accumulator there if T5-legacy path needs it.

---

## 3. Rejected optimizations (with reasons — not slop, each measured, derived, or sourced)

- **`torch.compile`** *(rejection reaffirmed by research, rev 2)*: the official
  compile-time-caching tutorial and pytorch#114206/#113287/#96152 show that
  `TORCHINDUCTOR_FX_GRAPH_CACHE` + a persistent `TORCHINDUCTOR_CACHE_DIR`
  reduce but do NOT eliminate per-process cold start — Dynamo tracing and
  AOTAutograd re-run in every process, and multiprocess cache reuse has a
  buggy history. With 315 + 165 short-lived subprocesses, the remaining
  per-process warm-up plus cache-miss risk plus the MoE loop's
  data-dependent shapes plus numerics change = net negative for this sweep
  shape. Rejected with sources; revisit only if the sweep moves to
  long-lived processes.
- **`torch._assert_async` for the loss guard**: see §2.3 — sync-free but
  destroys diagnosability on failure (context corruption, unrelated-line
  traceback, ignored message, private API). Rejected for robustness.
- **AMP/bf16, TF32**: the model is ~0.4–1.2 M params; step time is
  launch/sync-bound, not matmul-bound (measured: 31 ms CPU step for ~3 MFLOP
  batches). Precision drift for ~0 gain. Rejected.
- **Optimizer changes (`fused=True`, custom foreach)**: official AdamW docs —
  `foreach=None` (the current default) already selects the foreach
  implementation on CUDA; `fused` would change float paths for a negligible
  gain at this parameter count. **No change** is the robust choice. Rejected.
- **Dense MoE dispatch (compute all 6 experts on the full batch)**: float-path
  drift (variable-N → fixed-256 GEMMs) with no need at this scale; the
  sparse dispatch is the code's Shazeer-lineage design. Rejected (§2.2).
- **Bigger batch / fewer epochs**: protocol changes. Rejected (protocol is
  frozen).
- **More DataLoader workers**: measured floor is IPC, not prep parallelism
  (2 workers already hide 68 ms prep at 12 ms effective). Wrong lever; T5
  removes the floor instead. Rejected.
- **In-process multi-cell runner** (amortize the ~10–14 s/process fixed cost
  across cells): breaks per-cell crash isolation and the runner's
  checkpointing model. Rejected; the fixed cost is quantified and reduced by
  T1/T1b/T6 instead.

---

## 4. Expected impact at sweep scale

Measured (CPU, this box) vs **estimate** (cloud GPU, marked):

| Lever | Per-process / per-step effect | Sweep total (315 + 165 steps) |
|---|---|---|
| T1 versions | −4.5–5 s/process (stage-1 family; stage-2 keeps one sklearn import by protocol) | ≈ −20 to −28 min (measured) |
| T1b wandb guard | −~2.2 s/process when wandb installed and disabled | ≈ −10 to −18 min (measured import) |
| T2 dispatch syncs | −6 syncs/step | ToolHang+Transport stage-2 ≈ −30–60% step time (**estimate**) |
| T3 loss guard | −1 sync/step | smaller but same order as T2 |
| T4 validation | −~340 syncs/epoch (big tasks) | minutes per big-task run (**estimate**) |
| T5 flat iterator | 12.1 ms → ~1–2 ms/batch fetch; no spawn | minutes per run; removes GPU loader bottleneck (**estimate**) |
| T6 hash memo | −0.5–0.8 s/process | ≈ −3 to −4 min (measured) |

Ordering if approved: **T1, T1b, T6** (trivial, zero-risk) → **T2, T3**
(bit-identical) → **T4** (bit-preserving aggregation) → **T5**
(order-changing, heaviest verification) → full suite + preflight + dry-run
315/165 → 3-epoch Lift s1+s2 smoke (T1–T4 build must reproduce current
curves bit-identically; T5 build must pass its determinism/content gates).
No training hyperparameter, seed contract, or fairness rule changes.

## 5. Standing facts this review establishes

- Training data loading is single-task per run by design; caches are fully
  in-memory after a 0.46 s load — no disk I/O on the epoch path.
- The existing trainer loop is already sync-disciplined in the *train* pass
  (on-device metric accumulation, non-blocking H2D, single-pass grad clip) —
  the defects found are in the MoE dispatch, the loss guard, validation, and
  process startup, not the main loop.
- Rollout determinism gates are unaffected by every proposed patch (training
  only; the rollout path is untouched).

## 6. Cloud (Linux/GPU) deltas

Worker spawn ≈0 (fork) — T5 makes the question moot; imports faster but T1
still saves seconds; the CUDA syncs (T2/T3/T4) are GPU-only costs and are the
dominant per-step lever there. First-epoch warm-up remains (~1 epoch).
Per-process fixed cost after T1+T1b+T6 ≈ 4–7 s (**estimate**) → ≈35–55 min
across 480 steps, unavoidable without changing runner isolation (rejected
above).

## 7. Sources

Synchronization facts (rev 1):
- PyTorch official Performance Tuning Guide — "Avoid unexpected
  synchronization" (`.item()`, bool conversion, `.cpu()`, computed indices):
  https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- NVIDIA "Writing Sync-Free Code" (host-device synchronization inventory):
  https://docs.nvidia.com/dl/cuda-graph/torch-cuda-graph/sync-free-code.html
- pytorch/pytorch#12461 (implicit sync points, historical reference):
  https://github.com/pytorch/pytorch/issues/12461

Research pass (rev 2):
- importlib.metadata performance, single-pass `distributions()`:
  https://github.com/python/importlib_metadata/issues/95 and
  https://discuss.python.org/t/please-make-package-version-go-away/58501
- Shazeer et al. 2017, sparsely-gated MoE / noisy top-k gating (T2 lineage):
  https://arxiv.org/abs/1701.06538 (explainer:
  https://huggingface.co/blog/moe)
- `torch._assert_async` semantics and caveats (T3 rejection):
  https://leimao.github.io/blog/PyTorch-Assert-Async/ ,
  https://github.com/pytorch/pytorch/issues/36853 ,
  https://github.com/pytorch/pytorch/issues/131491
- Resident-tensor + randperm + index_select batching (T5 validation):
  https://discuss.pytorch.org/t/how-to-load-all-data-into-gpu-for-training/27609 ,
  https://discuss.pytorch.org/t/load-entire-dataset-on-gpu/79165 ,
  https://stackoverflow.com/questions/62111599/load-data-into-gpu-directly-using-pytorch ,
  https://github.com/Lightning-AI/pytorch-lightning/issues/2361
- torch.compile cold start / caching limits (rejection):
  https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html ,
  https://github.com/pytorch/pytorch/issues/114206 ,
  https://github.com/pytorch/pytorch/issues/113287 ,
  https://github.com/pytorch/pytorch/issues/96152
- AdamW foreach/fused defaults (no-change decision):
  https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- All measurements: this repository, 2026-08-22, commands and raw numbers in
  the ledger Phase 8d entry.

---

## 8. Implementation record (2026-08-22, rev 3)

Implemented in the gated order T1/T1b/T6 → T2/T3/T7a → T4 → T5, each phase
verified before the next. Files: `metadata.py`, `cli.py`, `cache_manager.py`,
`moe_layer.py`, `router.py`, `loops/base.py`, `loops/stage2_loop.py`, new
`data/common/flat_iterator.py`, `state_machine.py` (dispatch), + 5 new/extended
test files.

### Verification gates (all passed)

| Gate | Result |
|---|---|
| Baseline determinism (pre-patch, 2× stage-1) | curves bit-IDENTICAL |
| T1/T1b/T6 → stage-1 + stage-2 vs baseline | curves bit-IDENTICAL (both stages) |
| T2/T3/T7a → stage-1 + stage-2 vs baseline | curves bit-IDENTICAL (both stages) |
| T4 → stage-1 + stage-2 vs baseline | curves bit-IDENTICAL (both stages) |
| T5 → stage-1 ×2 | bit-IDENTICAL (same seed) — deterministic |
| T5 → stage-2 ×2 (bootstrap + training) | bit-IDENTICAL (same seed) |
| T5 vs baseline curves | DIFFERENT — expected and disclosed (permutation stream differs from `RandomSampler`; seed-reproducible, applied to every cell) |
| Content parity (T5) | fast-path batches bit-equal to collator batches with the permutation pinned sequential; full-row multiset coverage; drop_last parity — 10 unit tests |
| Full suite | **738 passed** (713 + 25 new) |
| Preflight | 165 train + 150 eval cells OK |
| Dry-runs | five_task exactly **315** steps; lift_ablation exactly **165** |
| Ruff | all touched files clean; 3 pre-existing findings remain in untouched files (2×E501 baselines, 1×F401 runner test) — left alone deliberately |

### Measured effect (this box)

| Metric | Legacy path | Fast path / patched |
|---|---|---|
| Batch fetch, steady | 490–551 ms/epoch (workers=2, measured head-to-head) | **3.1–9.0 ms/epoch** (~130–150× fetch speedup; 6 gathers/epoch) |
| First-epoch fetch incl. spawn | 6.8 s (4 worker procs, Windows spawn) | none (no workers, no spawn) |
| Loader + compute epoch (isolated, 33 batches) | — | ~0.5–0.6 s (compute-bound, 15.7 ms/step measured) |
| Environment fingerprint | ~4.5–5 s (imports sklearn/wandb/scipy) | metadata-only (45 ms single pass; sys.modules-first preserves exact strings incl. torch's `+cpu` local tag) |

Note on wall-clock noise: CLI-run epoch timings on this dev box fluctuate with
background system load (the full test suite itself varied 88.9 s → 42.5 s
between runs); the head-to-head benchmark above interleaves both paths in one
process under identical conditions and is the authoritative comparison.

### Behavior notes (disclosed)

- **T3 ordering**: for a real (grad-carrying) NaN loss the abort now surfaces
  from the per-step gradient guard (clip's `error_if_nonfinite`, or the fused
  single-sync check when clipping is off); the epoch-end on-device loss flag
  is the backstop for grad-less non-finite losses. Either way the run aborts
  before any epoch-end artifact (regression-tested).
- **Pre-existing failure found and fixed (test-only)**:
  `tests/scripts/test_stratified_stats.py::test_mc_bootstrap_matches_exact_distribution`
  failed at HEAD, untouched by this work — the exact enumerator and the MC
  sampler compute the same means in different summation orders, so e.g. 0.56
  materializes as both `0.56` and `0.5599999999999999` and the MC mass splits
  across float neighbors. Fixed by quantizing values at 1e-9 in the TEST's
  comparison; the analysis module is unchanged.
- **T7b not taken** (stacked metric accumulation): conditional metric key
  sets (grad-cosine, teacher terms) make vectorization change accumulation
  grouping for no measurable gain; per-key adds stay.

### §8.1 External review adjudication (post-implementation, same day)

An external review of the implemented changes confirmed the core work clean
(dispatch bit-identity, loss guard, aggregation equality, fingerprinting,
caching; 738 tests) and raised two findings — both verified against source
and **both correct, both now patched**:

- **P1 (confirmed, performance-contract regression on CUDA)**: the flat
  loader yields freshly gathered PAGEABLE host tensors while the trainer
  requests `.to(device, non_blocking=True)` — non-blocking H2D copies only
  overlap from pinned memory, and the legacy DataLoader DID pin in exactly
  this configuration, so T5 silently dropped that contract. Honest
  magnitude: batches are ~30 KB, so a synchronous copy is ~tens of µs and
  cannot become the bottleneck at these sizes — but the contract is restored
  regardless. **Patch**: `InMemoryBatchLoader(pin_memory=...)` pins each
  gathered batch through a `_pin` helper, enabled by the pipeline under the
  SAME gating as the legacy branch (config flag AND cuda target AND CUDA
  available — `Tensor.pin_memory()` raises on CPU-only builds, verified);
  the gating is hoisted in `_build_dataloaders` and shared by both paths.
  Tests: wiring (default-off; every field routed, values bit-identical) +
  a real-pinning test that runs only where CUDA exists. The reviewer's
  remaining requirement — a real cloud-GPU benchmark before the final run —
  is recorded as a first-run gate in `final_run_plan.md` §3 (this box has
  no GPU; the claim is not considered demonstrated until that cell runs).
- **P2 (confirmed, cosmetic)**: stale local annotation
  `result: dict[str, DataLoader | None]` in `_build_dataloaders` while the
  function returns the union type. Fixed to
  `dict[str, DataLoader | InMemoryBatchLoader | None]`.

Post-patch gates: suite **740 passed + 1 skipped** (the CUDA-only pinning
test — it runs on the sweep machine), ruff clean on the touched files, and a
fresh same-seed determinism pair-run of the training path (curves identical,
pin branch off on CPU).
