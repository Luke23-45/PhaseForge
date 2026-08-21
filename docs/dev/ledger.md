# PhaseForge Final Migration — Implementation Ledger

**Companion to:** `docs/dev/final_baselines_plan.md` (the approved direction)
**Purpose:** execute the final-baselines migration step by step, with no skipped
steps and no silent deviations. Every step is small, verifiable, and checked off
with evidence before the next begins.

---

## Ground rules

1. **No step is skipped or reordered.** Phases run top to bottom; steps inside a
   phase run in listed order unless a step explicitly says otherwise.
2. **A step is only "done" when its verification passes** and the checkbox is
   ticked with a one-line evidence note (command run + result, or commit hash).
3. **Anything unexpected → stop and record it** in the Progress Log before
   continuing. Do not improvise around a surprise mid-phase.
4. **The invariants in §Invariants are absolute.** If a step appears to require
   breaking one, that step is wrong — halt and re-plan.
5. Update this file in the same commit as the work it describes.

---

## Phase 0 — Approval and preconditions

- [ ] **S0.1 Professor approval** of `final_baselines_plan.md` recorded
      (date + decision in Progress Log). Nothing in Phases 2+ starts before this.
- [ ] **S0.2 Resolve pending decisions** (each needs an explicit written answer
      in Progress Log before the phase that depends on it — see §Pending
      decisions D1–D8).
- [ ] **S0.3 Baseline state captured:** working tree clean, all tests pass,
      `uv run python -m phaseforge.runner --manifest experiments/five_task.json --list`
      succeeds against the *current* (pre-migration) manifest.
      Record: commit hash, test summary, output of `--list`.

---

## Phase 1 — Canonical configuration migration

Goal: `phaseforge.yaml` becomes the R50 implementation under the name
`phaseforge`; both old identities disappear.

- [ ] **S1.1 Replace** `phaseforge/config/models/phaseforge.yaml` content with
      the complete contents of `phaseforge/config/models/phaseforge_r50.yaml`,
      changing only `name: "phaseforge_r50"` → `name: "phaseforge"`.
      The file must remain self-contained (no inheritance, no overrides).
- [ ] **S1.2 Delete** `phaseforge/config/models/phaseforge_r50.yaml`.
- [ ] **S1.3 Verify resolved contract** — a config-resolution test (or script)
      resolving `models=phaseforge` must confirm every row of the plan's §2.2
      table:

      | Check | Required value |
      |---|---|
      | `_target_` | `phaseforge.models.phase_moe.PhaseBootstrappedMoE` |
      | `router.num_experts` | 6 |
      | `router.top_k` | 2 |
      | `router.normalize_input` | true |
      | `router_init.type` | `centroid` |
      | `expert_init.type` | `partial_warm` |
      | `expert_init.drop_rate` | 0.5 |
      | `expert_init.seed` | `${project.seed}` (training seed) |
      | `expert_init.jitter_std` | 0.0 |
      | `soft_mapping.enabled` | false |
      | Stage 2 `freeze_encoder` | true |

- [ ] **S1.4 Confirm deletion:** `grep -rn "phaseforge_r50"` over
      `phaseforge/` returns only the two known sites to be cleaned in Phase 2
      (`utils/config.py` alias, `tests/models/test_model_baselines.py`) —
      nothing else.

---

## Phase 2 — Code and test cleanup (R50 identity removal)

- [ ] **S2.1 Remove the alias** `"phaseforge_r50": "phaseforge"` from
      `resolve_checkpoint_source` in `phaseforge/utils/config.py` (~line 266).
      All other aliases (controls → `phaseforge`, `warmstart_moe` → `bc`, etc.)
      stay — they now point at the canonical method, which is correct.
- [ ] **S2.2 Update** `tests/models/test_model_baselines.py` to reference the
      canonical `phaseforge` identity instead of `phaseforge_r50`.
- [ ] **S2.3 Mark the confirmation manifest archival:**
      `experiments/phaseforge_r50_confirmation.json` references a deleted model
      config. Either move it to an `experiments/archive/` folder or add a
      top-level note field marking it historical/non-runnable. Do not edit its
      recorded numbers.
- [ ] **S2.4 No-change verification (important):** the runner needs **no**
      provider-name changes — `protocol.py` (validator, cross-checks,
      `required_checkpoint`, `build_plan`) and `runner/cli.py`
      (`_auto_dependency_provider`) already hardcode `phaseforge`, which is now
      the canonical name. Verify by running the runner unit tests.
- [ ] **S2.5 Full test suite green** after S2.1–S2.3.

---

## Phase 3 — R50-matched mechanism controls (the H1–H4 cells)

Goal: the five controls use **50% partial expert warm-start**, matching the
canonical method. Until this phase completes, the plan's fallback applies:
controls are non-isolated behavioral comparisons and no clean H1–H4 causal
claim is permitted.

**Verified implementation facts (2026-08-22 code audit, commit `f1729c7`):**

- All five controls are already `num_experts: 6`, top-2 — capacity-matched to
  the canonical method.
- The shared utilities exist and are public in
  `phaseforge/models/components/expert.py`:
  `warm_start_experts_from_action_head` (L95, full copy + jitter) and
  `partial_reinit_experts_from_action_head` (L166, Drop-Upcycling-style,
  deterministic given seed, returns dropped indices for audit). The canonical
  R50 path calls the same function (`phase_moe.py` ~L610).
- `cli.py` ~L664 persists init metadata via generic
  `getattr(model, "_expert_init_info", None)` — any control class that sets
  `_expert_init_info` (as `PhaseBootstrappedMoE` does, `phase_moe.py` L675)
  is picked up with **no CLI changes**.

**Group A — config-only conversion (3 cells).** These already instantiate the
*same class* as the proposed method (`PhaseBootstrappedMoE`), whose
`bootstrap_moe` already supports their router types (`spherical_kmeans`,
`kmeans`, `phase_head`) and `partial_warm` expert init. Conversion = edit the
YAML `expert_init` block only; no Python changes.

- [ ] **S3.1 `pf_spherical_kmeans`** — in
      `config/models/baselines/pf_spherical_kmeans.yaml`, replace
      `expert_init: {type: warmstart, jitter_std: 0.02}` with
      `{type: partial_warm, drop_rate: 0.5, jitter_std: 0.0, seed: ${project.seed}}`.
- [ ] **S3.2 `pf_kmeans`** — same edit in `pf_kmeans.yaml`.
- [ ] **S3.3 `pf_phase_head`** — same edit in `pf_phase_head.yaml`.
- [ ] **S3.4 Group-A seed caveat:** the explicit `seed: ${project.seed}` line
      is mandatory — without it `bootstrap_moe` defaults `init_seed=42`
      (constant), breaking seed-dependent parity with the canonical method.

**Group B — small code change (2 cells).** These have their own bootstrap
implementations that hardcode the full warm-start call:

- [ ] **S3.5 `phase_pretrain_random_router`** — currently a pure subclass of
      `WarmStartMoEModel` inheriting its warm-start `bootstrap_moe`
      (`models/baselines/phase_pretrain_random_router.py`, entire class is the
      docstring + inheritance). Rework: parameterize `WarmStartMoEModel`'s
      expert init (config-driven) or override `bootstrap_moe` to call
      `partial_reinit_experts_from_action_head(experts, action_head,
      drop_rate=0.5, seed=training_seed)` and set `_expert_init_info`.
      Keep: provider = `phaseforge`, random router.
- [ ] **S3.6 `plain_encoder_phase_bootstrap`** — own `bootstrap_moe` ending in
      a hardcoded `warm_start_experts_from_action_head(...)` call
      (`models/baselines/plain_encoder_phase_bootstrap.py` ~L220). Swap that
      call for `partial_reinit_experts_from_action_head(...)` (drop 0.5,
      training seed) + set `_expert_init_info`. Keep: BC provider, centroid
      router over BC latents.
- [ ] **S3.7 Group-B training-seed plumbing:** both classes must receive the
      run's training seed (as `PhaseBootstrappedMoE.bootstrap_moe` does via
      `training_seed=`) so the partial-init draw is seed-dependent.
- [ ] **S3.8 Per-control run metadata** must persist (verify in test):
      router type, expert-init type, drop rate, dropped-neuron hash, training
      seed, resolved Stage 1 provider.
- [ ] **S3.9 Negative test:** a control configured with the old standard
      warm-start path fails or is loudly distinguishable in metadata — no
      silent inheritance of the old path.
- [ ] **S3.10 Deliberate non-change:** `warmstart_moe` and `scratch_moe` keep
      their current initialization (standard warm-start / random). They are
      behavioral baselines, not factorial controls. Do not "fix" them.
      Note: after S3.5, `phase_pretrain_random_router` is no longer
      "structurally identical to `warmstart_moe`" (its docstring says so
      today) — the 2×2 factorial becomes {encoder source} × {router init} at
      fixed partial-warm expert init, matched to the proposed method; update
      the class docstring accordingly.

---

## Phase 4 — New ablation cells

- [ ] **S4.1 Full-warm centroid cell** (canonical encoder + centroid router +
      standard full warm-start with the declared small jitter) — new config;
      this is the old method's behavior preserved as an ablation, clearly
      named (e.g. `pf_full_warm`), never as a proposed method.
- [ ] **S4.2 Drop-rate sweep cells:** 0%, 25%, 75%, 100% (50% is canonical).
      Same deterministic partial-init procedure, only `drop_rate` varies.
- [ ] **S4.3 Migrate existing ablation cells to the canonical provider:**
      `pf_random_random`, `pf_centroid_random`, `warmstart_r50`,
      `pf_one_warm_plus_random` (expert-init suite); `pf_ft`, `pf_corrupt_25`,
      `pf_corrupt_50`, `pf_shuffle_control` (representation suite);
      `pf_k3`, `pf_k12` (capacity suite). Each: point Stage 1 at canonical
      `phaseforge`, verify expert-count / router / seed parity (plan §6.2).
- [ ] **S4.4 Verify label-corruption semantics** of `pf_corrupt_*` /
      `pf_shuffle_control` before reuse (plan §6.3).
- [ ] **S4.5 Decide-and-document fate of unmentioned cells** (D3/D4 below):
      `pf_random_warm`, `phaseforge_e6`, `pf_jitter_00`, `pf_jitter_10`.

---

## Phase 5 — Manifest migration (`experiments/five_task.json`)

- [ ] **S5.1 Proposed-method rows:** exactly one per task. Full row shape
      (fields `index`, `data`, `task` are **required** — the plan's example
      block omits them; without them the row defaults to `data: "common"`,
      `task: null` and breaks five-task identity):

      ```json
      {"index": 1, "name": "phaseforge", "role": "proposed method",
       "model": "phaseforge", "data": "lift", "task": "Lift",
       "stages": [1, 2], "stage2_source": "self",
       "evaluate": true, "evaluate_mode": "rollout"}
      ```

      (analogously `can`/`Can`, `square`/`Square`, `tool_hang`/`ToolHang`,
      `transport`/`Transport`.)
- [ ] **S5.2 Consumer rows** (`phase_pretrain_random_router`, `teacher_forced`,
      and any clustering/phase-head controls in the matrix per D1):
      `stage2_source: "phaseforge"` — now resolving to the canonical method.
- [ ] **S5.3 Add promoted rows** per the plan §4.2–4.3 and D1:
      `bc_large` per task; `pf_spherical_kmeans` / `pf_kmeans` /
      `pf_phase_head` if D1 says five-task.
- [ ] **S5.4 Manifest hygiene:** no `phaseforge_r50` string anywhere; no
      duplicate rows; indices unique per task; `bc_rnn` rows unchanged.
- [ ] **S5.5 Validate with the runner parser:**
      `uv run python -m phaseforge.runner --manifest experiments/five_task.json --list`.
- [ ] **S5.6 Expected step count computed and recorded BEFORE the dry run**
      (do not reuse the old 315 number — the matrix changed). Formula: per
      method-row × seed: (#stages + 1 if evaluate). Then:

      `uv run python -m phaseforge.runner --manifest experiments/five_task.json --dry-run`

      Dry-run step count must equal the computed count; dependency graph
      contains no stale provider names (gate 7 of plan §11).

---

## Phase 6 — Checkpoint contract gating (new resolver logic)

Goal: make it *impossible* for a final run to consume a pre-final or
wrong-configuration `phaseforge` checkpoint. The current resolver
(`phaseforge/runner/resolver.py`, `_find_run` / `resolve_run_dir`) gates only
on seed + tag + `git_commit` — the config check below is **new logic**, not a
setting.

- [ ] **S6.1 Config-hash gating:** resolution accepts an expected config hash
      (or resolved-config contract) and rejects runs whose `run_meta.json`
      hash/contract does not match. `run_meta.json` already records
      `config_hash`.
- [ ] **S6.2 Contract check at load:** fail closed (loud, pre-training) if a
      selected checkpoint's state shows wrong expert count, router shape, or
      schema, or a task mismatch (plan §8.2.5).
- [ ] **S6.3 Commit check kept as auxiliary** — not the sole defense (a final
      sweep may legitimately span commits after a bugfix; the hash check is
      the robust one).
- [ ] **S6.4 Auto-detect fallback stays safe:** the CLI auto-detect path
      (`phaseforge/cli.py`, `find_latest_checkpoint` via
      `resolve_checkpoint_source`) must not be reachable for final runs
      without passing the same checks — the runner always passes an explicit
      `train.stage1_ckpt_path`; add a test that a final Stage 2 run with a
      missing explicit path does NOT silently fall back (plan §8.2.4).
- [ ] **S6.5 Fresh final namespace:** decide location (D5), create it empty,
      record its path here. All final runs and **all reporting commands**
      point at this base — never at the historical `outputs/` tree.
- [ ] **S6.6 Tests:** legacy 8-expert artifact (or fixture mimicking one) is
      rejected for every final step type (stage 1 resolve, stage 2 provider
      resolve, eval resolve).

---

## Phase 7 — Test and gate sweep

- [ ] **S7.1 Full unit suite green** (`uv run pytest`).
- [ ] **S7.2 Configuration-resolution tests** for canonical `phaseforge` and
      all five matched controls (contract table of S1.3 + per-control fields).
- [ ] **S7.3 Checkpoint-source tests** (Phase 6) green.
- [ ] **S7.4 Environment gates, checkpoint-free, per task:**

      ```bash
      uv run phaseforge-gates data=lift eval=rollout
      uv run phaseforge-gates data=can eval=rollout
      uv run phaseforge-gates data=square eval=rollout
      .venv-toolhang/<bin/python> phaseforge-gates data=tool_hang eval=rollout
      uv run phaseforge-gates data=transport eval=rollout
      ```

- [ ] **S7.5 Frozen reset bank verification:** 50 resets per task, seeds
      10000–10049, identical order for every method (registered protocol;
      Lift bank hash `a7d3953c0afcf560` for reference).
- [ ] **S7.6 Pinned versions verified:** robosuite 1.5.1 (main env) /
      1.5.0 + mujoco ≥ 3.2.7 (ToolHang env), dataset versions, state schema,
      action convention, horizon.
- [ ] **S7.7 Gates checklist of plan §11 (1–10)** each explicitly ticked in
      Progress Log. The sweep must not start if any fails.

---

## Phase 8 — Documentation sync

- [ ] **S8.1** `docs/plan/specs/research_definition.md` — EXP-101 row updated:
      proposed method is now 6-expert centroid + partial warm 0.5 (not
      "Warmstart (0.02)"); H1–H4 control descriptions matched to Phase 3
      implementations; note the lineage (R50 promoted and renamed).
- [ ] **S8.2** `docs/plan/design/final_evaluation_plan.md` — lineage note +
      (per D6) the declared primary comparison family and multiplicity
      correction.
- [ ] **S8.3** `docs/dev/final_run_plan.md` — rewritten to the migrated
      manifest, fresh namespace, and updated step count; stale
      `--baseline phaseforge` command examples reviewed (they are valid again
      post-rename, but the namespace/paths change).
- [ ] **S8.4** `docs/dev/baselines_methods.md` — matched-control
      implementations reflected.
- [ ] **S8.5** Add the **reproduction expectation note** to the final
      evaluation plan: final Lift results are *not* expected to reproduce the
      confirmation's 0.56 / 0.84 / 0.72 (mean 0.707) — the final method trains
      its own Stage 1 under the renamed config; a deviation is expected and is
      not a bug or regression (Stage 1 RNG differs from both the old 8-expert
      tree and the confirmation setup).
- [ ] **S8.6** Pre-final reports (`lift_rollout_eval_report.md`,
      `professor_decision_report.md`, fairness accounting) are **not edited**;
      they are pre-final records. Only add, if anything, a one-line pointer to
      the final plan.

---

## Phase 9 — Final sweep execution

- [ ] **S9.1 Smoke check** (optional but recommended): 10-episode smoke
      evaluation on one task first; never reported as final.
- [ ] **S9.2 Full sweep** against the fresh namespace:

      ```bash
      uv run python -m phaseforge.runner \
        --manifest experiments/five_task.json \
        --outputs <fresh-namespace> \
        --continue-on-error
      ```

- [ ] **S9.3 Post-sweep dry run** shows every step done; investigate every
      failed/pending cell before reporting; re-run failures.
- [ ] **S9.4 Reporting** (all against the fresh namespace only):

      ```bash
      uv run python scripts/analysis/summarize_train.py --outputs <fresh-namespace> --baseline phaseforge
      uv run python scripts/analysis/summarize_eval.py  --outputs <fresh-namespace> --baseline phaseforge
      uv run phaseforge-rollout-report <fresh-namespace>
      ```

- [ ] **S9.5 Report contents** per plan §9: per-task/per-seed success, Wilson
      intervals, paired PhaseForge-minus-baseline differences on identical
      resets, offline action metrics, routing diagnostics, parameter counts
      (re-verify `bc_large` match against the migrated MoE), capacity,
      training cost, configuration/provenance hashes. Three seeds reported as
      descriptive only.
- [ ] **S9.6 Ablation program** (Lift first) planned/tracked as a follow-up
      ledger section once the main matrix is secured.

---

## Pending decisions (answer in Progress Log before the dependent phase)

| # | Decision | Blocks | Default if undecided |
|---|---|---|---|
| D1 | Task scope of `pf_spherical_kmeans` / `pf_kmeans` / `pf_phase_head`: five-task matrix or Lift-only mechanism table? | S5.3 | Lift-only (cheaper; H3/H4 as secondary) |
| D2 | Multiplicity family + correction (e.g. Holm step-down over the 5 task baselines × 5 tasks; mechanism controls secondary family; teacher-forced/oracle excluded) | S8.2, S9.5 | Must be declared before S9.2 — no default |
| D3 | Fate of `pf_random_warm` and `phaseforge_e6` | S4.5 | Drop, with one-line rationale each |
| D4 | Fate of `pf_jitter_00` / `pf_jitter_10` (superseded by drop-rate sweep?) | S4.5 | Mark superseded by S4.2 sweep |
| D5 | Fresh final output namespace path | S6.5, S9 | e.g. `outputs_final/` at repo root |
| D6 | Rename `warmstart_r50` → e.g. `warmstart_partial`? | S4.3 | Rename (avoids stale "r50" token) |
| D7 | Seed budget: stay at 42/43/44 (descriptive) or extend? | S9 | Stay; revisit only via professor decision |
| D8 | Whether ablation suite runs interleave with or after the main sweep | S9.6 | After |

---

## Invariants — never do these

1. **Never** run, resolve, or report final artifacts from the historical
   `outputs/` tree; final work uses the fresh namespace only.
2. **Never** recreate the canonical config via Hydra overrides on another
   config — it is a self-contained file.
3. **Never** relabel pre-final results (old 8-expert or R50 confirmation) as
   final, pool them with final results, or edit recorded historical numbers.
4. **Never** present `teacher_forced` or oracle routing as baselines or as
   deployable-performance evidence; they are diagnostics.
5. **Never** let a final Stage 2 run auto-detect a Stage 1 checkpoint without
   the explicit path + contract checks.
6. **Never** declare superiority from a single seed or offline action MSE;
   closed-loop success on the frozen paired reset bank is the primary metric.
7. **Never** start the final sweep with any §11 gate unticked.

---

## Verified codebase facts (reference — do not re-derive)

| Fact | Where |
|---|---|
| Old config: 8 experts, `soft_mapping.enabled: true`, warmstart+jitter 0.02 | `phaseforge/config/models/phaseforge.yaml` (pre-migration) |
| R50 config: 6 experts, top-2, centroid, `partial_warm` 0.5, self-contained | `phaseforge/config/models/phaseforge_r50.yaml` |
| Alias `phaseforge_r50 → phaseforge` (auto-detect fallback) | `phaseforge/utils/config.py` ~L266; used at `phaseforge/cli.py` ~L591 |
| Runner hardcodes `phaseforge` as valid provider (no change needed post-rename) | `phaseforge/runner/protocol.py` L208/331/403/458; `runner/cli.py` ~L265 |
| Resolver gates on seed+tag+commit only (config gating is new work) | `phaseforge/runner/resolver.py` `_find_run` |
| Current matrix: 45 rows, proposed method = old `phaseforge` | `experiments/five_task.json` |
| Confirmation: Lift 0.56/0.84/0.72, mean 0.707, Stage 1 from old tree | `experiments/phaseforge_r50_confirmation.json` |
| Controls today use full warm-start (`warm_start_experts_from_action_head`) | e.g. `phaseforge/models/baselines/warmstart_moe.py` L111–121 |
| `bc_large` parameter-matched to ~382,646 (≈ 6-expert MoE) | `phaseforge/config/models/baselines/bc_large.yaml` |
| Registered eval: 50 resets/task/seed, bank seeds 10000–10049, Wilson 95% | `docs/plan/design/final_evaluation_plan.md` |
| Ablation manifest (27 cells incl. `pf_random_warm` #24, `phaseforge_e6` #25) | `experiments/lift_ablation.json` |

---

## Progress Log

| Date | Step | Evidence / note |
|---|---|---|
| 2026-08-22 | Ledger created | Derived from `final_baselines_plan.md` + code review of commit `f1729c7` state |
| 2026-08-22 | Phase 3 pre-implementation audit | Verified all 5 controls convertible: 3 config-only (same class as proposed method), 2 small code changes (shared `partial_reinit_experts_from_action_head` utility exists). No architectural blockers. Details in Phase 3 header. |
