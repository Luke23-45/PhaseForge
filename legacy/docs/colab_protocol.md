# Colab Execution Protocol (F3)

Ordered runbook for the full-length batch on free Google Colab. Every step is
fail-loud: a step that errors or prints a violation must be fixed before the
next step runs. Record the resolved versions and commit hashes in the run log
(lower table).

## 0. Environment

```bash
uv sync --extra rollout          # robosuite==1.4.0, mujoco==3.3.1, libero==0.1.1 (uv.lock)
uv run pytest -q                 # gate: 117/117
```

Mirror the raw LIBERO HDF5 cache (`{data_root}/raw/libero/{libero_90,libero_10}`)
with the one-off mirror script. The mirror is the single source of truth for
states, actions, and `env_kwargs`.

## 1. Census gate (B6) — run before ANY training or eval

```bash
uv run python -m phaseforge.data.scripts.build_object_index --suites libero_90 libero_10
```

Must complete with exit code 0 for every task. Any init-state mismatch aborts
the build (fail-loud, exit 1) — patch `object_state` decode or the index build
and re-run. This gate MUST pass before the P-Stage 1 re-ingest (E2 -> B6 -> E1
ordering).

## 2. Re-ingest (E1)

The object-state channel changed the state schema (23 -> 151 dim) and the cache
hash; re-ingest is triggered automatically. Verify the training HDF5 features
contain the 16 object slots (128 dim) + 23 proprio dims.

## 3. Physics sanity (F1) — one-time, before full training

```bash
uv run python scripts/benchmark_sim_forward.py --suite libero_90 --num-demos 3 --data-root /content/data
```

Decision: if replay error is comparable with and without `sim.forward()`, F1 is
ruled out (expected) and no physics patch is needed; if the extra-forward mode
is materially better, patch `libero_env.py` and re-run the census.

## 4. Full-length training (A4/C5)

Early stopping is disabled in the protocol runners (commit `643674a`); full
schedules run: stage 1 100 epochs, stage 2 200 epochs.

```bash
uv run python scripts/run_multi_seed_train.py
```

The runner takes NO arguments: the 8-cell matrix and seeds 42/43/44 are
hardcoded (`MODEL_STAGES`, `SEEDS`); edit the script if the matrix changes.
8 cells x 3 seeds = 24 runs (C2 sensitivity runs stay out of the headline set).

## 5. Rollout evaluation (A6, B1)

```bash
uv run python scripts/run_multi_seed_eval.py
```

The runner takes NO arguments: suites (`libero_90`, `libero_10`) and seeds
42/43/44 are hardcoded (`SUITES`, `SEEDS`). Protocol: `libero_90`
in-distribution (90 x 50 eps x 3 seeds, per-task breakdowns) + `libero_10`
labeled zero-shot (10 x 10 eps). Results JSONs must declare `eval/suite_roles`.
F3 estimate (issues register): ~110k episodes (8 cells x 13,800 eps) ~
2-4 weeks wall-clock on free T4 (2 workers) at ~13 s+/episode; training ~
3.5-4 days.

## 6. Diagnostics + results copy-back

Run the combined diagnostics sweep (C3) after eval. Copy back to the repo:

- `outputs/train/`, `outputs/eval/`, `outputs/object_index.json`, eval JSONs
- run log (table below) appended to this file

## Run log

| Step | Date | Version / commit | Status |
|------|------|------------------|--------|
| (fill per run) | | | |
