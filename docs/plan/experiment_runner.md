# PhaseForge — Experiment Runner Design

**Status:** implemented (`phaseforge-sweep`, `python -m phaseforge.runner`)

**Related plans:** [data_provenance_design.md](data_provenance_design.md),
[final_evaluation_plan.md](final_evaluation_plan.md),
[research_definition.md](research_definition.md)

The runner executes the frozen experiment matrix from a single JSON protocol
manifest: for every selected method, task, and seed it runs each training
stage in order and then the configured evaluation of that method's
final-stage checkpoint,
honouring the protocol's Stage 1 source dependencies. It is the only entry
point for training: the legacy `scripts/run_multi_seed_train.py` was removed
because it did not cover the `bc_robot_only` cell and did not run
evaluations.

---

## 1. The protocol manifest (`experiments/five_task.json`)

The five-task manifest is the single source of truth for the final paper
matrix. The older Lift pilot remains available for debugging. The manifest
is frozen:
changing the method matrix is a deliberate, reviewed edit to this file, not a
code change.

Top level:

| key          | value                                  |
|--------------|----------------------------------------|
| `name`       | `lift_pilot`                           |
| `task`       | `Lift`                                 |
| `seeds`      | `[42, 43, 44]` (protocol order)        |
| `defaults`   | per-run hydra overrides, e.g. `["train.early_stopping.enabled=false"]` |
| `methods`    | the 9-cell matrix, indexed 1–9         |

### Method matrix (notebook methods 1–9)

| idx | name                       | model                          | data       | tag         | stages | stage2_source |
|----:|----------------------------|--------------------------------|------------|-------------|--------|---------------|
|   1 | `phaseforge`               | `phaseforge`                   | common     | —           | 1, 2   | `self`        |
|   2 | `bc`                       | `baselines/bc`                 | common     | —           | 1      | —             |
|   3 | `bc_robot_only`            | `baselines/bc`                 | robot_only | `robot_only`| 1      | —             |
|   4 | `scratch_moe`              | `baselines/scratch_moe`        | common     | —           | 2      | —             |
|   5 | `warmstart_moe`            | `baselines/warmstart_moe`      | common     | —           | 2      | `bc`          |
|   6 | `phase_pretrain_random_router` | `baselines/phase_pretrain_random_router` | common | — | 2 | `phaseforge` |
|   7 | `plain_encoder_phase_bootstrap` | `baselines/plain_encoder_phase_bootstrap` | common | — | 2 | `bc` |
|   8 | `teacher_forced`           | `baselines/teacher_forced`     | common     | —           | 2      | `phaseforge`  |
|   9 | `oracle_moe`               | `baselines/oracle_moe`         | common     | —           | 2      | —             |

Semantics:

- `stages: [1]` means the method **is** a Stage 1 provider (BC cells).
- `stages: [2]` means the method trains Stage 2 only, loading the Stage 1
  checkpoint of its provider.
- `stage2_source: "self"` — PhaseForge trains Stage 2 from **its own** Stage 1.
- `stage2_source: null` — random init, no Stage 1 needed (`scratch_moe`,
  `oracle_moe`).
- `data` + `tag` — `bc_robot_only` is the negative control: same
  `baselines/bc` output tree, but `data=robot_only` and every run recorded
  with `project.tag=robot_only` so it can never be confused with the default
  BC cell.
- `evaluate: true` everywhere — one row = a complete run (train → eval).

### Validation (enforced at load time)

- unique, ascending stage lists drawn from `{1, 2}`;
- unique method indices/names, unique seeds;
- `stage2_source` must name a real method that has a stage 1 **and is
  untagged** (a provider must always be the common-data cell — a tagged
  provider would point dependency injection at the wrong output tree);
- `stage2_source` set implies the method has a stage 2; `"self"` with only a
  stage 1 is rejected.

## 2. Plan construction

A *plan* is an ordered list of `Step`s. Ordering is deterministic:

1. methods sorted by manifest index (so a provider's stage 1 precedes its
   consumers in a full sweep);
2. within a method, seeds in protocol order;
3. within a seed, training stages ascending, then the method's eval.

`--with-dependencies` additionally prepends the Stage 1 (train only, marked
`dependency`) of a provider method that is not itself selected, so a partial
selection (`--methods teacher_forced`) still runs to completion. Stage
filtering (`--stage N`) and `--eval-only` disable dependency injection
(scope was explicitly narrowed; a missing prerequisite fails pre-flight
instead of silently running).

Every step knows its *required checkpoint*:

- stage-1 train → none;
- stage-2 train → the provider's stage-1 `checkpoint_best.pt`
  (`"self"` → this method's own stage 1);
- eval → the method's final-stage `checkpoint_best.pt`.

## 3. Commands

Each step is executed as a subprocess against the installed console entry
points. A missing entry point is a loud error, never a silent skip. Tool Hang
steps are routed to the interpreter supplied by `--toolhang-python`, the
`PHASEFORGE_TOOLHANG_PYTHON` environment variable, or the conventional
`.venv-toolhang` location. The runner preflights robosuite `1.5.0` and
MuJoCo `>=3.2.7` before starting any step. Lift, Can, Square, and Transport
use the current environment and their rollout parity pins are checked by the
evaluator.

- Train argv: `phaseforge-train models=<model> train=stageN project.seed=<s>
  project.output_dir=<abs> project.log_level=<INFO|WARNING> [data=<variant>]
  [project.tag=<tag>] [train.stage1_ckpt_path=<abs provider ckpt>] <defaults...>`
- Eval argv: `phaseforge-eval ... train.stage1_ckpt_path=<abs ckpt>
  eval=<rollout|metrics> eval.mode=<rollout|offline>` with the same
  `models`/`seed`/`output_dir`/`tag`/`log_level` plumbing.

  The `eval=<group>` selector is **required**: the eval group defines
  the schema the CLI's evaluator reads (rollout.yaml carries
  `bank`/`env`/`episodes`/`gates`; metrics.yaml does not). Setting
  `eval.mode=rollout` alone leaves the default `metrics` group in
  place and crashes on the first missing key. The runner maps
  `method.evaluate_mode` -> the group selector atomically; the
  `eval.mode=...` override is kept as an explicit assertion. The
  rollout entry points (`run_rollout_evaluation`, `run_all_gates`)
  additionally fail fast with an actionable `EnvParityError` if a
  hand-invoked caller ever lands in this state.

`project.output_dir` is always the **absolute** outputs base, so the runner is
location-independent. `data` is passed only when it differs from the protocol
default (the `bc_robot_only` cell); `project.tag` is passed only for tagged
cells. Logs stream to the console (inherited stdio); `--verbose` forwards
`project.log_level=INFO`, otherwise `WARNING`.

A stage-2 step that bootstraps from a provider always receives the provider's
exact Stage 1 checkpoint as `train.stage1_ckpt_path`, resolved by the strict
seed+tag scan (untagged provider, `.completed`-gated — §5). Passing it
explicitly is what makes the subprocess load the right artifact: without it the
`phaseforge-train` CLI falls back to `find_latest_checkpoint`, whose
`tag=None` means "no constraint" and can select a newer *tagged* sibling run
sharing the provider's output tree (e.g. `bc_robot_only` next to `bc`),
crashing the stage-2 load with an input-dimension mismatch.

## 4. State registry (resume + provenance)

`<outputs>/_runner/state.json` records, per `(method, seed, phase)`:

```json
{"runs": {
  "phaseforge": {"42": {"stage1": {"status": "completed",
                                    "run_dir": "phaseforge/stage1/seed42/<ts>_<id>",
                                    "ckpt": "phaseforge/stage1/seed42/<ts>_<id>/checkpoints/checkpoint_best.pt"},
                         "stage2": {...},
                         "eval":   {"status": "completed",
                                    "ckpt": "phaseforge/stage2/seed42/<ts>_<id>/checkpoints/checkpoint_best.pt",
                                    "run_dir": "eval/phaseforge/seed42/<ts>_<id>"}}}}
```

Writes are atomic (temp file + `os.replace`). A corrupt file raises
`RegistryError` — the runner refuses to silently overwrite provenance.

Why a registry and not a scan: evaluation must target the **exact** checkpoint
a stage produced, not "the newest one that happens to match". The registry
makes that deterministic even after later runs or re-runs are added. It is
read only after a step's subprocess returns 0, and the recorded checkpoint is
re-validated on disk before it is trusted.

### Re-run policy: "latest successful wins"

A re-run of a completed cell overwrites the registry entry for that cell (and
the run dirs themselves are immutable, timestamped artifacts that are never
modified). Evaluation therefore always targets the **newest successfully
completed** training run of the exact `(method, seed, tag)` cell. Older eval
rows stay in `results.jsonl`, and each row records the exact `ckpt_path` it
evaluated, so no result is ever silently lost or rewritten.

## 5. Resolution and the strict tag rule

Two sources of truth, in order:

1. the registry — the exact run dir / checkpoint a stage produced;
2. a strict seed+tag filesystem scan (`resolve_run_dir`), which covers runs
   launched manually, outside the runner.

Runs are organised under `{model}/stage{N}/seed{S}/{ts}_...` (eval under
`eval/{model}/seed{S}/`), so every seed is recognisable by path. Legacy runs
written before seeds became a directory dimension sit directly under
`stage{N}/` and remain resolvable — the scan descends into `seed{S}`
sub-directories when present, and filters by `run_meta.json` seed regardless
of layout.

The scan's tag semantics are **strict**: `tag=None` matches only runs whose
`run_meta.json` records no tag; `tag="robot_only"` matches only that tag. This
guarantees the default BC cell never resolves to the `robot_only` variant and
vice versa — the exact bug the tag exists to prevent. (This is intentionally
stricter than `phaseforge.utils.config.find_latest_checkpoint`, whose
`tag=None` means "no constraint"; the runner does not use that function.)

The scan is also gated on **completion**: a run directory is eligible only if
it carries the `<run_dir>.completed` sibling marker that
`RunWriter.mark_completed` writes at the very end of a successful run.
`run_meta.json` alone proves nothing — it is written when a run *starts*, so a
run killed after saving early checkpoints has correct seed/tag metadata but
never a `.completed` marker. Gating on `.completed` guarantees a crashed or
partial run can never be selected as an eval target.

The stage-2 prerequisite check uses the same strict lookup with `tag=None`,
because protocol validation guarantees providers are untagged.

## 6. CLI

```
phaseforge-sweep [--manifest PATH]
                 [--methods IDX,NAME,...]
                 [--seeds N,...]
                 [--stage 1|2] [--eval-only] [--skip-eval]
                 [--with-dependencies]
                 [--force] [--continue-on-error]
                 [--dry-run] [--list]
                 [--outputs DIR] [--verbose]
```

| flag                | effect |
|---------------------|--------|
| `--methods`         | subset by index or name; default all |
| `--seeds`           | subset of protocol seeds; invalid seeds rejected |
| `--stage 1\|2`      | train that stage only, no eval |
| `--eval-only`       | evaluations only (final checkpoints must exist) |
| `--skip-eval`       | all training, no evaluation |
| `--with-dependencies` | auto-run a provider's stage 1 for partial selections |
| `--force`           | re-run steps the registry marks completed |
| `--continue-on-error` | record failure, keep going; else fail fast |
| `--dry-run`         | print plan + exact commands, execute nothing |
| `--list`            | print the method matrix and exit |
| `--outputs`         | output base, resolved absolute from project root |

Dry-run reports `WOULD RUN` with the exact argv, `BLOCKED` when a prerequisite
(e.g. the eval target checkpoint) is missing, and `skip` for steps the
registry marks complete. Exit codes: `0` all steps ok/skipped, `1` one or more
failed, `2` usage/manifest/registry error, `130` interrupted.

## 7. Usage

```bash
phaseforge-sweep --list                               # show the matrix
phaseforge-sweep --methods 1,2 --seeds 42 --dry-run   # preview, don't run
phaseforge-sweep --methods 1 --seeds 42               # PhaseForge seed 42, full run
phaseforge-sweep --with-dependencies                  # everything incl. eval
phaseforge-sweep --stage 1                            # providers only (BC + PhaseForge stage 1)
phaseforge-sweep --continue-on-error                  # sweep that survives one bad cell
```

The full matrix is 9 methods × 3 seeds with 19 steps per seed
(`phaseforge` has 3: stage1, stage2, eval; the eight single-stage methods have
2 each) — 57 steps in total, `--with-dependencies` not required for a full
selection.

## 8. Testing

`tests/test_runner.py` (CPU-only) covers protocol validation (including every
rejection rule), plan order/count/dependency injection, exact command argv
construction, registry round-trip + corruption handling, strict resolver
semantics (seed + tag), and CLI orchestration (`--list`, `--dry-run`,
`--continue-on-error`, fail-fast). `tests/test_config.py` covers the
`run_meta` seed/tag reporting used by the scanner.

## 9. Gaps / notes

- The runner drives the CLI entry points; it does not validate hydra
  overrides itself. A bad `defaults` entry fails loudly in the subprocess.
- `--force` re-runs a step and then **overwrites** the registry entry with the
  new artifact path, per the "latest successful wins" policy. Older eval rows
  remain in `results.jsonl` (each tagged with its own `ckpt_path`), so a full
  audit trail survives in the results ledger even though the registry holds
  only the current pointer.
- Eval traceability: the CLI records the evaluated checkpoint path in each
  `results.jsonl` row (`ckpt_path`), and Stage 2 training records its resolved
  source (`source_stage1` in the training summary), so results and training
  provenance are self-traceable without the runner.
