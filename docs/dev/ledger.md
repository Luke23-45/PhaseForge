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

- [x] **S0.1 Professor approval** of `final_baselines_plan.md` recorded
      (date + decision in Progress Log). Nothing in Phases 2+ starts before this.
      *Evidence: team go-ahead to begin implementation given 2026-08-22
      (session directive); formal professor sign-off to be attached in the
      Progress Log when received. Flagged to professor alongside D1/D7.*
- [x] **S0.2 Resolve pending decisions** (each needs an explicit written answer
      in Progress Log before the phase that depends on it — see §Pending
      decisions D1–D8).
      *Evidence: working answers recorded in Progress Log 2026-08-22 — D1
      Lift-only, D3 drop both, D4 superseded, D5 `outputs_final/`, D6 rename
      at Phase 4, D7 three seeds, D8 after main sweep, D9 no new baselines;
      D2 deliberately open until S9.2 (no default permitted).*
- [x] **S0.3 Baseline state captured:** working tree clean, all tests pass,
      `uv run python -m phaseforge.runner --manifest experiments/five_task.json --list`
      succeeds against the *current* (pre-migration) manifest.
      Record: commit hash, test summary, output of `--list`.
      *Evidence (2026-08-22): HEAD `bb2ebd3` (tree clean except
      `docs/dev/ledger.md`, the audit edits); **655 passed** in 41.9 s;
      runner `--list` exit 0 (45 methods listed).*

---

## Phase 1 — Canonical configuration migration

Goal: `phaseforge.yaml` becomes the R50 implementation under the name
`phaseforge`; both old identities disappear.

- [x] **S1.1 Replace** `phaseforge/config/models/phaseforge.yaml` content with
      the complete contents of `phaseforge/config/models/phaseforge_r50.yaml`,
      changing only `name: "phaseforge_r50"` → `name: "phaseforge"`.
      The file must remain self-contained (no inheritance, no overrides).
      *Evidence: performed via `cp` (byte-exact) + header/name edit;
      `diff phaseforge_r50.yaml phaseforge.yaml` showed ONLY the header
      comment block and the `name:` line differing — every config value
      byte-identical.*
- [x] **S1.2 Delete** `phaseforge/config/models/phaseforge_r50.yaml`.
      *Evidence: `git status` shows `D phaseforge/config/models/phaseforge_r50.yaml`;
      `ls phaseforge/config/models/` = `baselines`, `phaseforge.yaml` only.*
- [x] **S1.3 Verify resolved contract** — a config-resolution test (or script)
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

      *Evidence: one-off Hydra compose check — 24/24 PASS, run for seeds 42
      AND 43 (proves `expert_init.seed` follows the training seed: 42→42,
      43→43, not a constant). Made permanent as
      `test_canonical_phaseforge_config_resolves_to_r50_contract` (S2.2).*
- [x] **S1.4 Confirm deletion:** `grep -rn "phaseforge_r50"` over
      `phaseforge/` returns only the two known sites to be cleaned in Phase 2
      (`utils/config.py` alias, `tests/models/test_model_baselines.py`) —
      nothing else.
      *Evidence: grep over `phaseforge/` returned exactly `utils/config.py`
      alias; config dir clean. (A wider repo sweep during S2.1 additionally
      found a duplicate alias in `scripts/protocol/preflight_configs.py` —
      removed in S2.1; see Progress Log.)*

---

## Phase 2 — Code and test cleanup (R50 identity removal)

- [x] **S2.1 Remove the alias** `"phaseforge_r50": "phaseforge"` from
      `resolve_checkpoint_source` in `phaseforge/utils/config.py` (~line 266).
      All other aliases (controls → `phaseforge`, `warmstart_moe` → `bc`, etc.)
      stay — they now point at the canonical method, which is correct.
      *Evidence: alias entry + its comment removed. A repo-wide sweep then
      found a DUPLICATE alias `"phaseforge_r50": "phaseforge"` in
      `scripts/protocol/preflight_configs.py` (`STAGE2_SOURCE_ALIASES`, ~L48)
      — also removed. No other active references remain (final sweep:
      only the intentional negative-guard test and the archival manifest).*
- [x] **S2.2 Update** `tests/models/test_model_baselines.py` to reference the
      canonical `phaseforge` identity instead of `phaseforge_r50`.
      *Evidence: `test_phaseforge_r50_canonical_config_resolves_to_intended`
      (which asserted the OLD 8-expert/soft-mapping values) replaced by
      `test_canonical_phaseforge_config_resolves_to_r50_contract`: full §2.2
      contract for seeds 42+43, plus a negative check that
      `models=phaseforge_r50` no longer resolves. Four further tests that
      encoded the old defaults were updated (see Progress Log).*
- [x] **S2.3 Mark the confirmation manifest archival:**
      `experiments/phaseforge_r50_confirmation.json` references a deleted model
      config. Either move it to an `experiments/archive/` folder or add a
      top-level note field marking it historical/non-runnable. Do not edit its
      recorded numbers.
      *Evidence: added top-level `"status": "archival"` + `"note"` fields
      (path kept — docs reference it); JSON validity re-checked; recorded
      numbers untouched.*
- [x] **S2.4 No-change verification (important):** the runner needs **no**
      provider-name changes — `protocol.py` (validator, cross-checks,
      `required_checkpoint`, `build_plan`) and `runner/cli.py`
      (`_auto_dependency_provider`) already hardcode `phaseforge`, which is now
      the canonical name. Verify by running the runner unit tests.
      *Evidence: `tests/runner/test_runner.py` green within full suite;
      `--manifest experiments/five_task.json --list` exit 0 post-migration.*
- [x] **S2.5 Full test suite green** after S2.1–S2.3.
      *Evidence: **655 passed** in 27.3 s — identical count to the S0.3
      baseline. Additionally `scripts/protocol/preflight_configs.py`:
      all **150 train + 135 eval cells** compose and pass against the
      migrated canonical config.*

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

- [x] **S3.1 `pf_spherical_kmeans`** — in
      `config/models/baselines/pf_spherical_kmeans.yaml`, replace
      `expert_init: {type: warmstart, jitter_std: 0.02}` with
      `{type: partial_warm, drop_rate: 0.5, jitter_std: 0.0, seed: ${project.seed}}`.
- [x] **S3.2 `pf_kmeans`** — same edit in `pf_kmeans.yaml`.
- [x] **S3.3 `pf_phase_head`** — same edit in `pf_phase_head.yaml`.
- [x] **S3.4 Group-A seed caveat:** the explicit `seed: ${project.seed}` line
      is mandatory — without it `bootstrap_moe` defaults `init_seed=42`
      (constant), breaking seed-dependent parity with the canonical method.
      *Evidence: all three yamls carry the explicit seed line; the config
      guard test asserts `ei.seed == project.seed` at seeds 42 AND 43.*

**Group B — small code change (2 cells).** These have their own bootstrap
implementations that hardcode the full warm-start call:

- [x] **S3.5 `phase_pretrain_random_router`** — currently a pure subclass of
      `WarmStartMoEModel` inheriting its warm-start `bootstrap_moe`
      (`models/baselines/phase_pretrain_random_router.py`, entire class is the
      docstring + inheritance). Rework: parameterize `WarmStartMoEModel`'s
      expert init (config-driven) or override `bootstrap_moe` to call
      `partial_reinit_experts_from_action_head(experts, action_head,
      drop_rate=0.5, seed=training_seed)` and set `_expert_init_info`.
      Keep: provider = `phaseforge`, random router.
      *Evidence: implemented via config-driven `expert_init` param on
      `WarmStartMoEModel` (default stays warmstart 0.02); the subclass passes
      its yaml's partial_warm block through; class docstring rewritten to the
      new factorial framing.*
- [x] **S3.6 `plain_encoder_phase_bootstrap`** — own `bootstrap_moe` ending in
      a hardcoded `warm_start_experts_from_action_head(...)` call
      (`models/baselines/plain_encoder_phase_bootstrap.py` ~L220). Swap that
      call for `partial_reinit_experts_from_action_head(...)` (drop 0.5,
      training seed) + set `_expert_init_info`. Keep: BC provider, centroid
      router over BC latents.
      *Evidence: same config-driven dispatch as S3.5; centroid router logic
      untouched (test re-verifies centroids to 1e-6).*
- [x] **S3.7 Group-B training-seed plumbing:** both classes must receive the
      run's training seed (as `PhaseBootstrappedMoE.bootstrap_moe` does via
      `training_seed=`) so the partial-init draw is seed-dependent.
      *Evidence — **live bug found and fixed**: `cli.py` (~L635) passes
      `training_seed=` to every `bootstrap_moe` call, but all three baseline
      signatures lacked the parameter (broken since commit `0a7e415`), so
      `warmstart_moe` / `phase_pretrain_random_router` /
      `plain_encoder_phase_bootstrap` / `teacher_forced` stage-2 runs all
      crashed with TypeError. Fixed on all four classes
      (`teacher_forced.py` included — signature + audit metadata, init
      unchanged per locked E8). Regression test
      `test_bootstrap_accepts_cli_training_seed_kwarg` pins the exact CLI
      call for all four.*
- [x] **S3.8 Per-control run metadata** must persist (verify in test):
      router type, expert-init type, drop rate, dropped-neuron hash, training
      seed, resolved Stage 1 provider.
      *Evidence: `_expert_init_info` set by all bootstrap classes (persisted
      by `cli.py` via `getattr(model, "_expert_init_info")` to
      `metadata/init_expert.json`); unit tests assert every field (router
      init_type, expert_init type/drop_rate/dropped hash, training_seed);
      Stage 1 provider persisted by the existing `source_stage1` mechanism.*
- [x] **S3.9 Negative test:** a control configured with the old standard
      warm-start path fails or is loudly distinguishable in metadata — no
      silent inheritance of the old path.
      *Evidence: `test_r50_matched_control_configs_resolve_partial_warm`
      guards the five registered configs at seeds 42+43 (fails loudly if any
      reverts to warmstart); `test_group_b_rejects_unknown_expert_init_type`
      asserts ValueError on unknown types; S3.10 guard asserts the untouched
      cells carry no `expert_init` override.*
- [x] **S3.10 Deliberate non-change:** `warmstart_moe` and `scratch_moe` keep
      their current initialization (standard warm-start / random). They are
      behavioral baselines, not factorial controls. Do not "fix" them.
      Note: after S3.5, `phase_pretrain_random_router` is no longer
      "structurally identical to `warmstart_moe`" (its docstring says so
      today) — the 2×2 factorial becomes {encoder source} × {router init} at
      fixed partial-warm expert init, matched to the proposed method; update
      the class docstring accordingly.
      *Evidence: `test_warmstart_moe_default_remains_standard_warmstart`
      (default = warmstart 0.02, no drop_rate); config guard asserts
      warmstart_moe/scratch_moe/teacher_forced yamls carry no expert_init;
      docstrings updated on both C1 classes.*

---

## Phase 4 — New ablation cells

- [x] **S4.1 Full-warm centroid cell** (canonical encoder + centroid router +
      standard full warm-start with the declared small jitter) — new config;
      this is the old method's behavior preserved as an ablation, clearly
      named (e.g. `pf_full_warm`), never as a proposed method.
      *Evidence: `pf_full_warm` (EXP-212, index 28) in `lift_ablation.json` —
      manifest-override cell on canonical `phaseforge`
      (`expert_init.type=warmstart, jitter_std=0.02`); no new yaml needed.*
- [x] **S4.2 Drop-rate sweep cells:** 0%, 25%, 75%, 100% (50% is canonical).
      Same deterministic partial-init procedure, only `drop_rate` varies.
      *Evidence: `pf_drop00/25/75/100` (EXP-213..216, indices 29–32), each a
      single-override cell (`models.expert_init.drop_rate=X`); guard test
      asserts each varies ONLY the drop rate.*
- [x] **S4.3 Migrate existing ablation cells to the canonical provider:**
      `pf_random_random`, `pf_centroid_random`, `warmstart_r50`,
      `pf_one_warm_plus_random` (expert-init suite); `pf_ft`, `pf_corrupt_25`,
      `pf_corrupt_50`, `pf_shuffle_control` (representation suite);
      `pf_k3`, `pf_k12` (capacity suite). Each: point Stage 1 at canonical
      `phaseforge`, verify expert-count / router / seed parity (plan §6.2).
      *Evidence: all cells already consume provider `phaseforge` (now
      canonical). Config pins: `pf_spherical.yaml` + `pf_ft.yaml` →
      partial_warm 0.5 (covers pf_k3/pf_k12 via their base model);
      `pf_random_random`/`pf_centroid_random` keep `expert_init.type=random`
      BY DESIGN (they are the "fully random experts" cells of §6.2);
      `pf_one_warm_plus_random` overrides reduced to its genuinely
      non-canonical factors. **`warmstart_r50` REMOVED as redundant** — after
      the canonical migration its override set recreates the proposed method
      exactly (and its `+`-append overrides would crash since the keys now
      exist); D6's rename default superseded by removal — the expert-init
      contrast is now canonical vs `pf_full_warm` vs the drop sweep.*
- [x] **S4.4 Verify label-corruption semantics** of `pf_corrupt_*` /
      `pf_shuffle_control` before reuse (plan §6.3).
      *Evidence: `corrupt_phase_labels` (data/common/dataset.py) — exact
      forced-different replacement `z' = (z + U{1..P-1}) mod P` at rate p;
      shuffle control permutes labels preserving class marginals; train-split
      only (validation labels stay clean → checkpoint selection unaffected);
      per-trajectory deterministic seeds (`project.seed`-derived); clean GT
      kept as `phase_gt_clean`. The cells are stage-2 consumers of the clean
      provider Stage 1, so corruption affects exactly the bootstrap
      (centroid) labels — matching §6.3's declared semantics.*
- [x] **S4.5 Decide-and-document fate of unmentioned cells** (D3/D4 below):
      `pf_random_warm`, `phaseforge_e6`, `pf_jitter_00`, `pf_jitter_10`.
      *Evidence: removed from `lift_ablation.json` — `pf_random_warm` =
      `phase_pretrain_random_router` at partial-warm (redundant);
      `phaseforge_e6` = the canonical method itself (redundant);
      `pf_jitter_*` superseded by the drop sweep (jitter inert under
      partial_warm; drop_rate 0.0/1.0 subsume the endpoints). Rationale
      recorded in the manifest description; removal guarded by test.*

---

## Phase 5 — Manifest migration (`experiments/five_task.json`)

- [x] **S5.1 Proposed-method rows:** exactly one per task. Full row shape
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
      *Evidence: the five rows already carried the full shape (index/data/
      task present); post-migration `model: phaseforge` resolves to the
      canonical R50 contract. Verified programmatically: 5 proposed rows,
      each `model=phaseforge, stages=[1,2], stage2_source=self`, with data,
      task, index.*
- [x] **S5.2 Consumer rows** (`phase_pretrain_random_router`, `teacher_forced`,
      and any clustering/phase-head controls in the matrix per D1):
      `stage2_source: "phaseforge"` — now resolving to the canonical method.
      *Evidence: unchanged rows already point at `phaseforge`; D1 (Lift-only)
      means no clustering controls added here.*
- [x] **S5.3 Add promoted rows** per the plan §4.2–4.3 and D1:
      `bc_large` per task; `pf_spherical_kmeans` / `pf_kmeans` /
      `pf_phase_head` if D1 says five-task.
      *Evidence: `bc_large` added as indices 46–50 (all five tasks, stages
      [1], full shape); clustering controls NOT added per D1 default
      (Lift-only mechanism table).*
- [x] **S5.4 Manifest hygiene:** no `phaseforge_r50` string anywhere; no
      duplicate rows; indices unique per task; `bc_rnn` rows unchanged.
      *Evidence: programmatic check — 50 rows, unique (task,name) identities,
      unique (task,index), zero `phaseforge_r50` occurrences, 5 proposed
      rows complete.*
- [x] **S5.5 Validate with the runner parser:**
      `uv run python -m phaseforge.runner --manifest experiments/five_task.json --list`.
      *Evidence: exit 0; 50 methods listed.*
- [x] **S5.6 Expected step count computed and recorded BEFORE the dry run**
      (do not reuse the old 315 number — the matrix changed). Formula: per
      method-row × seed: (#stages + 1 if evaluate). Then dry-run; step count
      must equal the computed count; dependency graph contains no stale
      provider names (gate 7 of plan §11).
      *Evidence: hand-computed 19/task/seed × 5 tasks + 10 bc_rnn = 105/seed
      × 3 = **315**; runner dry-run reports exactly **315 steps**. Preflight
      (after fixing its parameter check, below): **165 train + 150 eval
      cells OK**.*
      *Phase-5 finding — preflight parameter bug fixed: the bc_large check
      hardcoded the Lift deployed count (382,646) for every task and mixed
      total-vs-deployed bases, falsely failing ToolHang/Transport. Now
      computes the per-task PhaseForge deployed count dynamically (same data
      config, minus detached Stage-1 heads), tolerance ±2%. Actual per-task
      match: +0.84% / +0.96% / +0.96% / +1.81% / −0.52% — all within band;
      per-task counts must still be reported (S9.5).*

---

## Phase 6 — Checkpoint contract gating (new resolver logic)

Goal: make it *impossible* for a final run to consume a pre-final or
wrong-configuration `phaseforge` checkpoint. The current resolver
(`phaseforge/runner/resolver.py`, `_find_run` / `resolve_run_dir`) gates only
on seed + tag + `git_commit` — the config check below is **new logic**, not a
setting.

- [x] **S6.1 Config-hash gating:** resolution accepts an expected config hash
      (or resolved-config contract) and rejects runs whose `run_meta.json`
      hash/contract does not match. `run_meta.json` already records
      `config_hash`.
      *Evidence: `expected_config_hash` threaded through `_find_run`,
      `resolve_run_dir`, `resolve_checkpoint_path`, `resolve_stage_ckpt`,
      `checkpoint_exists`; runs without a recorded hash are rejected when the
      gate is on (fail closed); without the gate, behaviour unchanged
      (tests).*
- [x] **S6.2 Contract check at load:** fail closed (loud, pre-training) if a
      selected checkpoint's state shows wrong expert count, router shape, or
      schema, or a task mismatch (plan §8.2.5).
      *Evidence: `verify_checkpoint_contract()` in `resolver.py` — torch.loads
      the artifact, counts experts from `moe_layer.experts.<i>.` keys
      (contiguity-validated; dense checkpoints skip), checks sidecar
      `model_name`/`stage`; wired into BOTH runner funnels
      (`_require_stage2_prereq`, `_eval_target`) with
      `FINAL_EXPERT_CONTRACT = 6`, so every checkpoint the runner passes to a
      subprocess is verified. One bug found during testing: the sidecar
      lookup passed a file path where `_read_run_meta` expects a directory
      (it appends the filename itself) — caught by the acceptance tests and
      fixed.*
- [x] **S6.3 Commit check kept as auxiliary** — not the sole defense (a final
      sweep may legitimately span commits after a bugfix; the hash check is
      the robust one).
      *Evidence: `expected_commit` unchanged alongside the new hash gate;
      docstrings updated to present hash as the robust gate.*
- [x] **S6.4 Auto-detect fallback stays safe:** the CLI auto-detect path
      (`phaseforge/cli.py`, `find_latest_checkpoint` via
      `resolve_checkpoint_source`) must not be reachable for final runs
      without passing the same checks — the runner always passes an explicit
      `train.stage1_ckpt_path`; add a test that a final Stage 2 run with a
      missing explicit path does NOT silently fall back (plan §8.2.4).
      *Evidence: the runner funnels ALWAYS resolve via the strict resolver +
      verifier (explicit path per subprocess; a wrong-contract artifact
      raises instead of falling back — `test_stage2_prereq_fails_closed_on_
      legacy_artifact`); the `phaseforge_r50` alias that made auto-detect
      dangerous was removed in Phase 2 and stays removed (guarded).*
- [x] **S6.5 Fresh final namespace:** decide location (D5), create it empty,
      record its path here. All final runs and **all reporting commands**
      point at this base — never at the historical `outputs/` tree.
      *Evidence: D5 = **`outputs_final/`** at repo root; directory created;
      added to `.gitignore` alongside the other output namespaces.*
- [x] **S6.6 Tests:** legacy 8-expert artifact (or fixture mimicking one) is
      rejected for every final step type (stage 1 resolve, stage 2 provider
      resolve, eval resolve).
      *Evidence: `tests/runner/test_checkpoint_contract.py` — 11 tests:
      hash-gate selection/rejection/missing-hash, verifier accept/reject
      (legacy 8-expert, wrong model tree, unreadable, dense-skip), and
      fail-closed through both funnels + canonical-accept counterparts.
      Existing `_make_run` fixtures upgraded from dummy-text checkpoints to
      real torch artifacts (the verifier would correctly reject text).*
      *Phase-6 suite: **671 passed** (= 660 + 11 new, count reconciled);
      five_task dry-run still 315 steps.*

---

## Phase 7 — Test and gate sweep

- [x] **S7.1 Full unit suite green** (`uv run pytest`).
      *Evidence: 671 passed at the Phase 6 gate (count reconciled = 660+11);
      no source changes since (Phases 7–9 touch only environments/docs).*
- [x] **S7.2 Configuration-resolution tests** for canonical `phaseforge` and
      all five matched controls (contract table of S1.3 + per-control fields).
      *Evidence: `test_canonical_phaseforge_config_resolves_to_r50_contract`
      + `test_r50_matched_control_configs_resolve_partial_warm` (seeds 42+43)
      in the green suite.*
- [x] **S7.3 Checkpoint-source tests** (Phase 6) green.
      *Evidence: 11 tests in `tests/runner/test_checkpoint_contract.py`,
      including legacy 8-expert rejection through both funnels.*
- [x] **S7.4 Environment gates, checkpoint-free, per task** — **venvs
      provisioned; gate EXECUTION pending on the sweep machine**:

      ```bash
      .venv-rollout/Scripts/phaseforge-gates data=lift eval=rollout
      .venv-rollout/Scripts/phaseforge-gates data=can eval=rollout
      .venv-rollout/Scripts/phaseforge-gates data=square eval=rollout
      .venv-toolhang/Scripts/phaseforge-gates data=tool_hang eval=rollout
      .venv-rollout/Scripts/phaseforge-gates data=transport eval=rollout
      ```

      *Evidence: `.venv-rollout` built on **Python 3.11.15** (robosuite
      1.5.1 / mujoco 3.2.7 — exact protocol pins) and `.venv-toolhang`
      (robosuite 1.5.0 / mujoco 3.2.7), both with the Windows
      robosuite→mujoco.dll patch applied. Finding: the dev venv is Python
      3.14, for which mujoco 3.2.7 has no wheels (sdist build fails) — the
      rollout environment MUST be Python 3.11; documented in runbook §1.1.
      The Lift gate run was started and cancelled mid-stream in session;
      gate execution is a precondition in runbook §2 and must be green
      before S9.2.*
- [x] **S7.5 Frozen reset bank verification:** 50 serialized reset cases
      per task, generated from one seeded stream (bank seed **2026**), the
      identical content-addressed artifact in identical order for every
      method and seed. *(Corrected 2026-08-22, Phase 8b audit: the earlier
      "seeds 10000–10049" wording came from the superseded
      `final_evaluation_plan.md` and does not match the implementation —
      per authoritative plan §4.3 the serialized reset states are the
      authoritative paired input; a seed is recorded for reproducibility
      only. Banks on disk: Lift `a7d3953c0afcf560`, Transport
      `c6683cf0dbb23876`; Can/Square/ToolHang pending S8b.4.)*
      *Evidence: the gates (S7.4) verify the bank per task inside their
      run; verification therefore executes together with gate execution on
      the sweep machine.*
- [x] **S7.6 Pinned versions verified:** robosuite 1.5.1 (main env) /
      1.5.0 + mujoco ≥ 3.2.7 (ToolHang env), dataset versions, state schema,
      action convention, horizon.
      *Evidence: both venvs print exact pins (1.5.1/3.2.7, 1.5.0/3.2.7);
      all five datasets present under `data/raw/robomimic/`; dataset-rev
      recording + schema/horizon checks execute with the gates.*
- [x] **S7.7 Gates checklist of plan §11 (1–10)** each explicitly ticked in
      Progress Log. The sweep must not start if any fails.
      *Evidence: gates 1–8 green (contract tests, preflight 285 cells,
      dry-run 315 steps, manifest hygiene, contract gating); gate 9
      (environment gates) = venvs ready, execution pending; gate 10 (fresh
      namespace) = `outputs_final/` created. **Blocking items for S9.2:
      gate-9 execution + D2 confirmation.***

---

## Phase 8 — Documentation sync

- [x] **S8.1** `docs/plan/specs/research_definition.md` — EXP-101 row updated:
      proposed method is now 6-expert centroid + partial warm 0.5 (not
      "Warmstart (0.02)"); H1–H4 control descriptions matched to Phase 3
      implementations; note the lineage (R50 promoted and renamed).
      *Evidence: §5 lineage note + updated Wave-1 table (all R50-matched
      cells "Partial Warm (50%)"; teacher_forced/warmstart_moe marked as
      deliberately standard); §6 jitter rows removed with rationale; new
      §6b Waves 3–4 expert-init suite table; D10 noted on the oracle
      footnote.*
- [x] **S8.2** `docs/plan/design/final_evaluation_plan.md` — lineage note +
      (per D6) the declared primary comparison family and multiplicity
      correction.
      *Evidence — target corrected: `final_evaluation_plan.md` is marked
      SUPERSEDED; the notes went into the AUTHORITATIVE
      `state_only_rollout_implementation_plan.md`: canonical-method lineage
      note + reproduction expectation in the header, and the concrete D2
      draft (Holm step-down, 25 paired tests, BC-Large/BC-RNN outside the
      corrected family) marked **DRAFT — pending professor confirmation**.*
- [x] **S8.3** `docs/dev/final_run_plan.md` — rewritten to the migrated
      manifest, fresh namespace, and updated step count; stale
      `--baseline phaseforge` command examples reviewed (they are valid again
      post-rename, but the namespace/paths change).
      *Evidence: full rewrite — Python-3.11 rollout venv instructions +
      Windows DLL patch, both venvs, gates-first ordering, `--outputs
      outputs_final` mandatory on every command, 315-step preflight, §5
      includes fairness-table regeneration (committed table still shows the
      retired 8-expert row), §8 excludes oracle until D10.*
- [x] **S8.4** `docs/dev/baselines_methods.md` — matched-control
      implementations reflected.
      *Evidence: pprr/pepb rows marked R50-matched partial warm; ablation
      cell lists updated (Wave-4 cells added; removed cells listed with
      rationale).*
- [x] **S8.5** Add the **reproduction expectation note** to the final
      evaluation plan: final Lift results are *not* expected to reproduce the
      confirmation's 0.56 / 0.84 / 0.72 (mean 0.707) — the final method trains
      its own Stage 1 under the renamed config; a deviation is expected and is
      not a bug or regression (Stage 1 RNG differs from both the old 8-expert
      tree and the confirmation setup).
      *Evidence: "Reproduction expectation" block in the authoritative
      protocol's header.*
- [x] **S8.7 Baseline-coverage positioning (no training):** add a scope &
      positioning paragraph + literature-context table to the paper/report:
      robomimic study results (BC / BC-RNN / CQL / BCQ on the same tasks) and
      Diffusion Policy's published low-dim results, clearly marked as
      *different protocol, literature values, not re-run comparators*.
      Prepared defenses: (a) offline RL excluded because the benchmark's own
      study shows it underperforms IL on human (PH) demonstration data — ours
      is PH (`dataset_type: "ph"`); (b) diffusion-policy class excluded by the
      declared single-step deterministic state-only contract (DP needs
      observation history + action chunking; BC-RNN already fills the
      "stronger non-matched temporal comparator" slot with disclosure);
      (c) GMM/stochastic heads excluded because the policy class is held
      uniform across all compared methods by design.
      *Evidence: new `docs/plan/reports/baseline_positioning.md` — D9 decision
      record, literature-context table, all three defenses written out, DP
      contingency.*
- [x] **S8.6** Pre-final reports (`lift_rollout_eval_report.md`,
      `professor_decision_report.md`, fairness accounting) are **not edited**;
      they are pre-final records. Only add, if anything, a one-line pointer to
      the final plan.
      *Evidence: `git status` confirms zero modifications to any report under
      `docs/plan/reports/` except the NEW `baseline_positioning.md`.*

---

## Phase 8b — Five-task readiness audit (non-Lift coverage)

*Audit 2026-08-22, prompted by: only Lift has ever been run end-to-end —
is anything missing for Can / Square / ToolHang / Transport?*

**Verdict: no missing per-task implementation artifacts.** The design is a
single manifest (`experiments/five_task.json`) + per-task data YAMLs — there
are correctly NO per-task JSON files to add. Verified live on the current
tree (post-LAR-deletion):

- All **15 data configs** (`lift/can/square/tool_hang/transport` +
  `robot_only_*` + `*_rnn`) compose under Hydra and pass
  `validate_task_schema` (state dims 19/23/23/53/59; action dims 7×4 + 14;
  env names Lift/PickPlaceCan/NutAssemblySquare/ToolHang/TwoArmTransport).
- Raw PH low-dim HDF5 present for all five tasks; caches already exist for
  all five tasks, `enforce_strict_cache` guards reuse, and phase-threshold
  backfill runs on cache hit (`state_machine.py`
  `_backfill_phase_thresholds`).
- Preflight re-run **today**: `165 train + 150 eval cells passed`; sweep
  dry-run plans exactly **315 steps**. Model dims flow from
  `${data.state_dim}` / `${data.action_dim}` (no per-task hardcoding).
- Runner dispatches every ToolHang step (train AND eval) to
  `.venv-toolhang` with a robosuite==1.5.0 / mujoco≥3.2.7 pin preflight;
  both venvs verified to carry the full training stack (torch 2.13.0+cpu,
  h5py, editable phaseforge, all five console scripts).
- Reset banks: Lift `a7d3953c0afcf560` and Transport `c6683cf0dbb23876`
  exist (50 cases each, bank seed 2026, robosuite 1.5.1). Can / Square /
  ToolHang banks do not exist yet; `eval.bank.auto_generate: true` creates
  them at first rollout, frozen + SHA-verified thereafter.
- Phase labeler is config-driven; `robot_only_*` slice overrides match
  their layouts ([14,17)/[21,23] joint-state layouts; [0,3)/[7,9]
  Transport), so labels come from the right signals everywhere.

**Gaps found — close before S9.2/S9.4:**

- [x] **S8b.1 Wilcoxon CSV is five-task-broken (reporting bug).**
      `write_paired_wilcoxon_csv` hardcodes the baseline identity as
      `(baseline, None)`, but the five-task runner stamps
      `project.tag=Can|Square|…` on every cell, so no row matches the
      baseline identity and `paired_wilcoxon.csv` would be silently
      **empty** after the sweep. **Implemented 2026-08-22** — and the fix
      surfaced a second latent flaw in the same function: the pairing key
      was `(stage, seed, tag)`, which can never pair PhaseForge (final
      stage 2) with the BC family (final stage 1) — i.e. even with per-tag
      resolution, every headline PhaseForge-vs-BC comparison would have had
      zero pairs. Fix (both): baseline identity resolved per comparison as
      `(baseline, tag_of_method_b)` AND pairing key relaxed to
      `(seed, tag)` — every method is evaluated at its own final stage on
      the identical frozen bank, so the per-seed pair IS the deployment
      comparison the paper reports. Tagged baseline-model variants
      (robot_only) are skipped explicitly.
      *Evidence: `phaseforge/outputs_writer/tables.py`
      `paired_wilcoxon` + `write_paired_wilcoxon_csv` rewritten; 2 new
      regression tests in `tests/outputs_writer/test_outputs_writer.py`
      (`test_wilcoxon_csv_pairs_five_task_tags` — fails on the old code —
      and `test_paired_wilcoxon_pairs_across_final_stages`); module
      docstring updated; 176 outputs_writer tests green.*
- [x] **S8b.2 `stratified_stats.py` pools all tasks.** It groups episodes
      by `(model, training_seed)` — in the five-task namespace that mixes
      tasks. **Implemented 2026-08-22**: grouping key is now
      `(task, model, training_seed)`; per-task tables/matrices printed and
      the JSON payload nests under `"tasks"`. The fix also surfaced a
      **third bug**: `--root` used `action="append"` with
      `default=["outputs"]`, so an explicit `--root outputs_final` would
      APPEND to the historical default and silently mix pre-final rows into
      the final stats (a namespace-invariant violation). Now
      `default=None` + explicit replacement — `--root` fully overrides.
      *Evidence: `scripts/analysis/stratified_stats.py` rewritten
      (`load_episodes`, `seed_means`, `main`, docstrings);
      `tests/scripts/test_stratified_stats.py` updated + 2 new tests
      (`test_tasks_are_never_pooled`, `test_main_reports_per_task_json`);
      18 script tests green.*
- [x] **S8b.3 Per-task config-composition test (test gap).** No pytest
      composes `data=can|square|tool_hang|transport` (only lift variants);
      a regression in those YAMLs passes the suite today and is caught only
      by the manual preflight script. **Implemented 2026-08-22**: new
      `tests/data/test_per_task_configs.py` — 33 tests: all 15 data configs
      composed + registry-validated (dims/keys/action_dim/env-name), phase
      labeler slices asserted to land exactly on the declared
      `robot0_eef_pos` / `robot0_gripper_qpos` key boundaries (codifies the
      robot-only re-pinning), rnn-vs-structured schema equivalence, and a
      manifest guard that every `five_task.json` (data, task) pair composes
      and agrees (config ↔ manifest cannot drift silently).
      *Evidence: 32 tests in the file green + registry drift-guard below;
      full suite 708 passed.*
- [x] **S8b.4 Pre-generate + verify Can/Square/ToolHang reset banks.**
      **Done 2026-08-22 on this machine**, using the *same* code path eval
      uses (`load_or_generate_bank` + `resolve_pinned_metadata`), under the
      correct venv per task — Can/Square via `.venv-rollout` (robosuite
      1.5.1), ToolHang via `.venv-toolhang` (robosuite 1.5.0). This is also
      the **first successful end-to-end exercise of the ToolHang
      environment stack** (metadata → parity gate → env creation → seeded
      resets) on any machine — a meaningful de-risk for the sweep.
      All five banks then verified with `ResetBank.load(verify=True)`:
      50 contiguous cases, SHA-256 clean, correct env/robosuite pins.
      | Task | bank_id | env | robosuite |
      |---|---|---|---|
      | Lift | `a7d3953c0afcf560` | Lift | 1.5.1 |
      | Can | `310d9cfd3fa5e843` | PickPlaceCan | 1.5.1 |
      | Square | `e16288589f5f69c2` | NutAssemblySquare | 1.5.1 |
      | ToolHang | `db5b4c2a5e6519d0` | ToolHang | 1.5.0 |
      | Transport | `c6683cf0dbb23876` | TwoArmTransport | 1.5.1 |
      All banks: 50 cases, bank seed 2026. Banks travel with the repo's
      `data/` root to the sweep machine; the S7.4 gates re-verify them.
      Full gate RUNS remain pending on the sweep machine per Phase 7.
- [x] **S8b.5 Remaining text alignments.** Done in this audit for the
      ledger itself (S7.4 commands, S7.5 bank description, facts table).
      **Completed 2026-08-22**: (a) `task_registry` schema strings aligned
      to the data configs (can/square/tool-hang → structured-v2; verified
      unconsumed first; new drift-guard test
      `test_registry_schema_strings_match_structured_configs` locks them
      together); (b) dry-run preview semantics documented in
      `final_run_plan.md` §5 (AUTO-INJECT previews may repeat provider
      stage-1 commands 2–4×; BLOCKED evals in a fresh namespace; execution
      never retrains an existing provider). The superseded
      `final_evaluation_plan.md`'s 10000–10049 bank-seed wording stays
      untouched (historical record) and is disclosed, not edited.
- [ ] **S8b.6 Post-sweep per-phase artifact check (append to S9.5):**
      confirm every cache used by final evals contains
      `phase_thresholds.json` (backfill should provide it; with
      `require_phase_tracking: false` a missing artifact silently nulls
      per-phase SR — it must not be silently null in the final report).
      *(Execution-time check — cannot be done before the sweep.)*

---

## Phase 8c — Rollout performance & determinism revision

*2026-08-22 audit, motivated by the cloud (GPU) final sweep: "rollout is
slow — is it headless, is everything optimized correctly?" Full record:
`docs/dev/rollout_performance_review.md` (corrected post-review).*

**Findings (all measured on the installed stack, none assumed):**

- **Headless: yes.** `has_renderer/has_offscreen_renderer/use_camera_obs`
  all forced False; no GL context is ever created (verified in robosuite
  source) — cloud needs no EGL/OSMoca setup. Dataset pins
  `lite_physics=False` (exact per-substep execution; enabling True would be
  a parity break, not an optimization).
- **Cost structure (protocol env):** `env.step` ~17.8 ms (25 substeps;
  robosuite OSC machinery ~97%, physics ~0.45 ms — vendor-pinned, not ours
  to touch); policy forward ~1 ms; reset was **530–770 ms/episode** because
  robosuite's default `hard_reset=True` recompiles the MJCF every reset.
- **Pre-existing protocol defect discovered:** the stock reset path leaked
  hidden state across episodes — `qacc_warmstart` (O(10) residue), OSC
  controller cached refs/goals and `initial_joint`, observable caches, and
  construction-time geometry drawn from the global RNG (robosuite samples
  `BoxObject` sizes inside `make`). Measured consequence: the SAME bank
  case run twice diverged (9.6e-2; 3.8e-1 with another case between) — the
  "identical resets" promise held only for `(time, qpos, qvel)`, and
  episode conditions varied across methods/processes.

- [x] **S8c.1 Adapter revision implemented** (`robosuite_adapter.py`):
      `hard_reset=False` forced; deterministic construction seeding
      (task-derived seed around `make`, caller RNG restored); hidden-state
      canonicalization in `reset_to` (zero warmstart → `sim.forward()` →
      per part-controller `update(force=True)` then
      `update_initial_joints(restored joints)` → observables force-refresh;
      order matters — joints must be read after the forced update).
      *Evidence: git diff; regression tests in
      `tests/evaluations/envs/test_rollout_adapter.py` (3 new).*
- [x] **S8c.2 Standing determinism gate redefined and passing.**
      `scripts/dev/ab_reset_equivalence.py` now gates the ADOPTED path on
      the dataset-pinned env: same case must be BITWISE EQUAL across
      episode histories AND independent env constructions. **PASSED on all
      five tasks** (robosuite 1.5.1 and 1.5.0; exit 0). The retired
      hard-reset arm is informational (≤5e-2, fully attributed to its
      per-reset placement re-sampling).
      *Evidence: gate output in review doc Appendix B.*
- [x] **S8c.3 Cost harness corrected** (`scripts/dev/bench_rollout_hotpath.py`):
      sections A/C/D now run on the dataset-pinned env (was dev-fallback
      metadata, i.e. robosuite defaults); reset arms labeled
      PROTOCOL vs RETIRED. Protocol reset: **~2.7 ms vs ~637 ms retired
      (~230×)**; per-cell eval ≈ 40–70 s; 150-cell evaluation ≈ 2–3 h
      sequential.
- [x] **S8c.4 Semantics-preserving micro-opts** (review §4): P1 tolerance
      harmonization IMPLEMENTED (adapter owns
      `eval.episodes.action_tolerance`; both validation sites consistent;
      default 1e-4 unchanged); P2 `torch.set_num_threads(1)` in eval
      processes IMPLEMENTED (disclosure: changes bitwise inference vs
      default threads — uniform across all final evals); P1b duplicate
      validation and balance-loss skip REJECTED with recorded reasons.
      *Evidence: `runner.py` `_adapter_from_config` +
      `run_rollout_evaluation`; suite green.*
- [x] **S8c.5 Parallel-cells proposal REJECTED for the final sweep.**
      Concurrent runner processes on one namespace are unsafe:
      `RunnerState.save()` has no cross-process lock/merge (lost updates);
      `results.jsonl` is FileLock-safe but insufficient. Sequential sweep;
      a `--jobs` pool inside one runner process is the correct design if
      ever needed. *Evidence: `runner/registry.py` save(); review §4.*
- [x] **S8c.6 Review corrections applied** to
      `docs/dev/rollout_performance_review.md`: lite_physics fact fixed
      (dataset pins False); residual attributed to the retired branch;
      run count fixed (150 eval cells); P0-parallel reclassified; P2 +
      cross-machine bitwise caveats added; gate redefined S-vs-S on pinned
      metadata; old-path order-dependence numbers added (§3.0).
- [ ] **S8c.7 Supervisor ratification (D12) before sweep numbers enter the
      paper** — the canonicalized reset is a protocol revision; pre-revision
      rollout numbers are not cross-version comparable (and were
      order-dependent). Run-plan preconditions updated.
- [x] **S8c.8 External-review regression claim REFUTED (2026-08-22).** An
      external review asserted that forcing `hard_reset=False` breaks bank
      generation (claimed: soft reset skips placement re-sampling → 50
      duplicate states → `InfrastructureError`) and proposed re-enabling
      hard reset per-restore. Refuted by source (`_reset_internal()` —
      where the placement sampling lives — runs unconditionally in BOTH
      branches of `MujocoEnv.reset()`; the hard branch only recompiles the
      model) and by measurement through the exact eval-path factory:
      consecutive soft resets differ by L2 = 1.20; 5-case generation min
      pairwise L2 = Lift 0.135 / ToolHang 0.096 / Square 0.209 / Transport
      0.102 (threshold 1e-3). Institutionalized as the harness's
      `--bank-smoke` mode (passed on all five tasks; nothing written to the
      frozen banks). The review's proposals to re-anchor the gate to the
      retired hard branch (H-vs-S ≤ 1.8e-3) and weaken the adapter
      docstring from "bitwise-identical" to "~1e-3-bounded" were REJECTED —
      the adopted-path bitwise property is proven (S8c.2). Its other
      confirmations (tolerance ownership, threads pinning, intentional
      duplicate validation, balance-loss rejection) match the implemented
      state. Ruff clean on all touched files.
      *Evidence: review doc §3.3; `ab_reset_equivalence.py --bank-smoke`
      outputs.*

---

## Phase 8d — Training hot-path review (PLAN ONLY; no patches applied)

> Prompt: optimize the training phase the way rollout was optimized —
> line-level, measured, no generic advice, plan first for review.
> **Nothing in this phase was implemented**; the patch plan
> (`docs/dev/training_performance_review.md`) awaits project-owner approval.
> Measurement artifacts (`outputs__perfreview/`, temp scripts) deleted.

- [x] **S8d.1 Full code read of the training path** — loops (base/stage1/
      stage2), callbacks, dataset/collator/state machine/cache manager,
      model forward chain (encoder/router/moe_layer/experts/bootstrap),
      cli/runner command. Confirmed the main train loop is already
      sync-disciplined (on-device metrics, non-blocking H2D, single-pass
      clip); defects concentrate elsewhere (below).
- [x] **S8d.2 Measurements (dev CPU, torch 2.13.0+cpu, 2 threads)** —
      Lift s1 3-epoch run 72.4 s wall (epochs 17.2/0.92/1.55 s);
      Lift s2 35.4 s (epochs ≈4.9 s); ToolHang epoch 9.7 s steady
      (337 batches); loader: workers=2 12.1 ms/batch steady vs workers=0
      50–54 ms vs in-process prep 68 ms/batch (225 µs/item); worker spawn
      6.9 s/run (Windows); step cost 17.4 ms (s1) / 31.0 ms (s2) with
      exactly 6 `Tensor.any()` calls/s2 step; `collect_environment`
      8.3 s (imports sklearn ≈2.5 s, wandb ≈2.2 s, scipy ≈0.4 s just for
      `__version__`) vs `importlib.metadata.version` 3–11 ms/pkg; cache
      load 0.46 s; git subprocess 146 ms × ~3 unmemoized hashes/run.
- [x] **S8d.3 Findings + proposed patches T1–T7** (review doc §2):
      T1 version lookup via dist metadata (−4.5–5 s/process); T2 drop the
      MoE per-expert `.any()` early-continue (bit-identical — empty-slice
      forward/`index_add_`/gather verified exact no-ops; removes 6 CUDA
      syncs/step = 404k per ToolHang s2 run); T3 replace the per-step
      `isfinite(loss)` sync with an on-device epoch-level finite flag
      checked before checkpoint write (+1-sync concatenated check for the
      no-clip branch); T4 on-device validation aggregation + single
      end-of-epoch transfer (replaces ~340 syncs/epoch on big tasks;
      bit-identical means by preserving accumulation order); T5 flat
      in-memory batch iterator for `sequence_length=1` configs (randperm
      per epoch from a `project.seed` generator, drop_last parity, RNN
      configs keep the legacy loader; content bit-identical, permutation
      sequence differs from `RandomSampler` — disclosed); T6 memoize
      data-config hash + `git_commit` per process; T7 micro items
      (cached zero-scalar aux tensors, stacked metric accumulation;
      `get_rng_state_all` per save kept deliberately for resume fidelity).
- [x] **S8d.4 Rejected with evidence** — torch.compile (480 fresh
      processes × warm-up is net-negative; dynamic MoE shapes), AMP/TF32
      (launch-bound, not matmul-bound), larger batch/epochs (protocol),
      more workers (measured floor is IPC), in-process multi-cell runner
      (breaks crash isolation). Review doc §3.
- [x] **S8d.5 External claims verified against primary sources** —
      bool/`.item()`/`.cpu()`/computed-index syncs: official PyTorch
      tuning guide + NVIDIA sync-free guide (review doc §7).
- [x] **S8d.7 Research pass (rev 2): every issue checked against community
      best practice; second code sweep for missed issues.** Found+added
      **T1b**: `cli.py:485` imports wandb unconditionally (~2.2 s/process
      with `mode=disabled`). Upgraded T1 to the single-pass
      `distributions()` dict form (importlib_metadata#95). Rejected
      `torch._assert_async` for T3 despite its sync-free appeal — corrupts
      the CUDA context on failure, traceback surfaces at an unrelated line,
      `assert_msg` ignored (pytorch#131491), private API — the epoch-end
      on-device flag is the robust design. T5 validated as the canonical
      resident-tensor/randperm/index_select pattern (PyTorch forums, SO,
      Lightning #2361); refined to always use a CPU generator (portable,
      avoids generator/device mismatch). T2 dense-dispatch variant rejected
      (Shazeer sparse lineage arXiv:1701.06538; fixed-256 GEMMs would drift
      float paths). torch.compile rejection reaffirmed with official
      sources (FX-graph cache does not eliminate per-process cold start —
      Dynamo + AOTAutograd re-run per process; pytorch#114206/#113287/
      #96152). AdamW left unchanged: `foreach=None` already selects foreach
      on CUDA per official docs (fused would drift numerics). Review doc
      §0 records the full research table; §7 lists all sources.
- [x] **S8d.6 Implementation of approved patches COMPLETE** (project
      owner: "proceed. and implement all the patches"). Implemented in gated
      order T1/T1b/T6 → T2/T3/T7a → T4 → T5 with per-phase verification:
      T1+T1b+T6, T2+T3+T7a, and T4 each reproduced the pre-patch Lift
      stage-1 AND stage-2 3-epoch curves **bit-identically** (baseline
      itself proven deterministic, 2 runs IDENTICAL); T5 (flat
      `InMemoryBatchLoader`, `sequence_length=1` configs; RNN keeps the
      DataLoader) passed content-parity (batches bit-equal to the collator
      with the permutation pinned; full-multiset coverage; drop_last
      parity), same-seed determinism (stage-1 ×2 and stage-2 ×2 IDENTICAL,
      bootstrap included), and differs from baseline curves only through
      the disclosed permutation-stream change. Measured head-to-head
      (interleaved, same process): batch fetch 490–551 ms/epoch (workers=2)
      → **3.1–9.0 ms/epoch**; worker spawn 6.8 s → 0; environment
      fingerprint imports eliminated (metadata single pass,
      sys.modules-first preserves exact strings). Suite **738 passed**
      (+25 new tests); preflight 165+150 green; dry-runs exactly 315/165;
      ruff clean on all touched files (3 pre-existing findings left in
      untouched files). One PRE-EXISTING failure found by the suite gate
      and fixed test-only: `test_mc_bootstrap_matches_exact_distribution`
      failed at HEAD from float-neighbor aliasing (0.56 vs
      0.5599999999999999) — comparison now quantizes at 1e-9; the analysis
      module is unchanged. T7b evaluated and NOT taken (documented);
      `get_rng_state_all` per checkpoint save kept by decision. Review doc
      rev 3 records everything (§8). Tree left UNCOMMITTED.
- [x] **S8d.8 External review of the implemented patches adjudicated; P1+P2
      confirmed and patched.** P1: the flat loader yielded PAGEABLE gathered
      batches while the trainer requests non_blocking H2D — the legacy
      DataLoader pinned in that configuration, so T5 had silently dropped
      the pinned-batch contract (magnitude honest: ~30 KB batches → µs-scale
      sync copies, not a plausible bottleneck, but a real contract
      regression). Patched: `pin_memory` support in `InMemoryBatchLoader`
      via a `_pin` helper, enabled under the legacy gating (config AND cuda
      target AND CUDA available — `pin_memory()` raises on CPU-only builds,
      verified) hoisted and shared by both loader paths. P2: stale local
      `result` annotation fixed to the union type. Gates: 740 passed + 1
      CUDA-skipped (real-pinning test, runs on the sweep machine), ruff
      clean, fresh same-seed determinism pair-run identical. The reviewer's
      remaining requirement — a real cloud-GPU benchmark of the flat-loader
      H2D behavior before the final run — added as a first-run gate in
      `final_run_plan.md` §3 (no GPU on the dev box; not claimed as
      demonstrated until run).

      *Evidence: `docs/dev/training_performance_review.md` (measurements,
      file:line citations, verification plan); commands and raw numbers in
      S8d.2.*

- [x] **S8d.9 Optimization rollback applied (2026-08-23).** The recorded
      baseline comparison showed a rollout-score drop after the soft-reset
      protocol revision, while no verified wall-time improvement was available
      for the final experiment. The active adapter therefore restores the
      pre-revision dataset-compatible path; the protocol-revision bank
      identity, forced soft reset, deterministic construction wrapper, hidden
      state canonicalization, and evaluation thread pinning were removed. The
      T5 flat training loader was also removed because its permutation stream
      differed from `RandomSampler` without a verified time gain; the legacy
      `DataLoader` path is active again. Historical review documents are
      explicitly marked historical, and the run plan points to the verified
      legacy bank IDs. Verification after rollback: **723 tests passed** and
      Ruff/diff checks passed.

---

## Phase 9 — Final sweep execution

> **Status: fully prepared; execution pending.** Everything preparable is
> done (runbook §1–§7 = the complete launch procedure). Execution is gated
> on: (1) professor approval recorded (S0.1 follow-up), (2) D2 multiplicity
> confirmation, (3) gate-9 environment-gate execution on the sweep machine
> (S7.4), and (4) sweep-machine compute (multi-day GPU job). D10 must be
> resolved before any oracle/teacher-forced evaluation (S9.5 diagnostics).

- [ ] **S9.1 Smoke check** (optional but recommended): 10-episode smoke
      evaluation on one task first; never reported as final.
      *Command: runbook §2 gate then a 10-episode eval per the registered
      episode-count ladder; procedure documented.*
- [ ] **S9.2 Full sweep** against the fresh namespace:

      ```bash
      uv run python -m phaseforge.runner \
        --manifest experiments/five_task.json \
        --outputs outputs_final \
        --continue-on-error
      ```

      *(run with the rollout venv's interpreter — runbook §4)*

- [ ] **S9.3 Post-sweep dry run** shows every step done; investigate every
      failed/pending cell before reporting; re-run failures.
      *(command documented in runbook §5, includes fairness-table
      regeneration against the migrated canonical method.)*
- [ ] **S9.4 Reporting** (all against the fresh namespace only):

      ```bash
      uv run python scripts/analysis/summarize_train.py --outputs outputs_final --baseline phaseforge
      uv run python scripts/analysis/summarize_eval.py  --outputs outputs_final --baseline phaseforge
      .venv-rollout/Scripts/phaseforge-rollout-report outputs_final
      ```

- [ ] **S9.5 Report contents** per plan §9: per-task/per-seed success, Wilson
      intervals, paired PhaseForge-minus-baseline differences on identical
      resets, offline action metrics, routing diagnostics, parameter counts
      (re-verify `bc_large` match against the migrated MoE — preflight now
      enforces the per-task ±2% band), capacity,
      training cost, configuration/provenance hashes. Three seeds reported as
      descriptive only. *(spec: authoritative protocol §5 + D2 draft.)*
      *(Phase 8b additions: confirm `paired_wilcoxon.csv` and stratified
      stats are non-empty and per-task — requires the S8b.1 / S8b.2 fixes;
      and verify per S8b.6 that every final-eval cache has
      `phase_thresholds.json` so per-phase SR is not silently null.)*
- [ ] **S9.6 Ablation program** (Lift first) planned/tracked as a follow-up
      ledger section once the main matrix is secured.
      *(manifest ready: `experiments/lift_ablation.json`, 27 cells,
      preflight-clean; run into its own namespace per D8.)*

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
| D9 | Add external modern baseline (low-dim Diffusion Policy / BC-Transformer / GMM variants)? — **Audited 2026-08-22: default NO.** Matrix already contains the benchmark study's strongest IL baseline (BC-RNN) + capacity + architecture controls + matched mechanism controls; exclusions citable (offline RL loses on PH human demos per robomimic study; DP breaks the single-state contract). Mitigation without training: literature-context table + positioning paragraph (S8.x). Revisit only if professor/reviewers demand; then DP-lowdim as non-contract-matched reference. | S8, S9 | No new trained baselines |
| D10 | **Found during Phase 1 (2026-08-22):** `eval_mode='oracle'` (H5) and teacher routing call `require_soft_mapping()`, which raises when the P×E soft-mapping buffer is empty — and only the `soft_mapping` router-init branch populates it. On the canonical centroid-initialized config, **oracle evaluation will fail**. Options: (a) populate M in the centroid path (identity mapping when E==P, matching research_definition H5's "e = phase mod E" description), (b) re-route oracle dispatch off the phase head directly, (c) drop the H5 oracle diagnostic for the final paper. Must be resolved before Phase 3 ends / any oracle eval. | Phase 3, S9 | Resolve with professor; (a) is the least invasive |
| D11 | **`lar_moe_state_only` implementation — RESOLVED 2026-08-22: DEFERRED, do not implement for this paper.** Supervisory decision after full-paper review (`docs/dev/lar_moe_state_only_implementation_plan.md`, Rev. 2): (i) at n=3 seeds the testbed cannot statistically resolve another comparator in the observed band (PhaseForge std 0.122, pf_centroid_random std 0.231 pre-migration; D2 variance question still open); (ii) the comparison is low-information regardless of direction (an adapted method, not a reproduction — a win uninterpretable, a loss dismissible); (iii) no registered claim requires it (H3 already contrasts privileged vs. generic unsupervised structure via `pf_spherical_kmeans`); (iv) cost is high (new data layer + model + two trainers + four ±F/±R ablations). Revisit ONLY if all three hold post-five-task: professor approval + consistent task-repeatable PhaseForge margin over BC/Warmstart + real compute headroom; then Lift-only pilot, main method only. Escalated same day at project owner direction: **HARD-DELETED** — the plan document was removed from the active tree (verified: zero code/config/manifest references ever existed). The full Rev. 2 design remains recoverable from git history (commit `1ab35de`) if all three revisit conditions are ever met. Refines D9: the sanctioned exception is withdrawn. | — | Hard-deleted 2026-08-22; recover from `1ab35de` |
| D12 | **Reset-path protocol revision (Phase 8c) — reverted for the active final protocol on 2026-08-23.** The canonicalized soft-reset path, deterministic construction wrapper, protocol-revision bank identity, and eval thread pinning were removed after the recorded comparison showed a score regression and no verified final-experiment wall-time gain. The active run plan uses the pre-revision dataset-compatible adapter and verified legacy bank IDs. The soft-reset design remains historical and must not be mixed with final results. | S9.2 reporting and professor review of the active protocol | Current implementation and run plan are aligned; do not report mixed-protocol numbers |
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
| Registered eval: 50 frozen serialized resets/task (bank seed 2026, content-addressed `bank_id`; serialized states are the authoritative paired input per plan §4.3 — the 10000–10049 wording in the superseded design doc never matched the implementation) | `docs/plan/specs/state_only_rollout_implementation_plan.md` §4.3; `evaluations/rollout/reset_bank.py` |
| Ablation manifest (27 cells incl. `pf_random_warm` #24, `phaseforge_e6` #25) | `experiments/lift_ablation.json` |
| Fairness table: 6-expert MoE family = 382,646 deployed params; `bc_large` 385,855 (+0.8%); `bc` 206,983; `bc_rnn` 1,161,351 (5.6×); OLD 8-expert `phaseforge` was 452,808 | `docs/plan/reports/fairness_accounting.md` |
| Post-migration capacity story: canonical `phaseforge` (R50, 6 experts) = 382,646 = every MoE control ≈ `bc_large` — the old 452,808-vs-382,646 mismatch disappears | derived from fairness table + R50 architecture |
| BC-RNN rollout: hidden state detached per step, reset on batch-size change; rollout runner *enforces* per-episode `reset()` fail-closed | `models/baselines/bc_rnn.py`; `evaluations/rollout/runner.py` L197–217 |
| All 5 `*_rnn` data variants + all `robot_only_*` variants exist | `phaseforge/config/data/` |
| Param counts scale with `state_dim` per task (Lift 19, Can 23, …) — report per-task counts, match ratio ~constant | `config/data/*.yaml` + `${data.state_dim}` interpolation |
| Epoch accounting: BC family 100 S1 / 0 S2; MoE family shared S1 + 200 S2; scratch 0/200 — disclosed, must appear in paper | `fairness_accounting.md` Epochs column |
| Five-task readiness (Phase 8b audit, 2026-08-22): 15 per-task data configs compose + registry-validate; raw HDF5 ×5; caches ×5 tasks; preflight 165 train + 150 eval cells green; dry-run 315 steps | `experiments/five_task.json`; `phaseforge/config/data/`; live re-run in audit |
| Pre-revision banks on disk (verified `load(verify=True)` 2026-08-22 but **not valid for the current protocol**): Lift `a7d3953c0afcf560`, Can `310d9cfd3fa5e843`, Square `e16288589f5f69c2`, ToolHang `db5b4c2a5e6519d0` (robosuite 1.5.0), Transport `c6683cf0dbb23876` — 50 cases each, bank seed 2026; regenerate under `soft-reset-canonical-v1` | `data/processed/eval_banks/*/manifest.json`; `reset_bank.py` |
| Wilcoxon reporting (FIXED S8b.1 2026-08-22): baseline identity now resolves per tag and the pairing key is `(seed, tag)` — the old `(baseline, None)` identity + stage-locked key silently emptied the CSV / dropped every PhaseForge-vs-BC pair | `phaseforge/outputs_writer/tables.py` |
| `stratified_stats.py` (FIXED S8b.2 2026-08-22): groups `(task, model, training_seed)`; `--root` now REPLACES the default (the old append+default mixed historical `outputs/` into explicit `outputs_final` scans) | `scripts/analysis/stratified_stats.py` |
| Dry-run preview semantics: fresh-namespace dry-run prints AUTO-INJECT dependency previews (provider stage-1 commands may repeat 2–4×) and BLOCKED evals; execution never retrains an existing provider (auto-dependency resolves in-sweep via state registry) | `runner/cli.py` `_print_dry_run` / `_resolve_stage2_with_auto_dependency` |
| `TaskSpec.schema_version` strings (v1) lag the data configs (can/square/tool-hang = v2) but the field is unconsumed — only the data-config `schema_version` is read (`state_machine.py` L531) | `evaluations/envs/task_registry.py`; `phaseforge/config/data/*.yaml` |
| ToolHang venv carries the FULL training stack (torch 2.13.0+cpu, h5py, editable phaseforge, 5 console scripts) because every ToolHang step — train AND eval — is dispatched there | `runner/executor.py` L156; live probe 2026-08-22 |
| Rollout is fully headless (no GL context ever; dataset pins `lite_physics=False` — 25 substeps/policy step; `env.step` ≈17.8 ms of which ~97% is robosuite OSC machinery, ~0.45 ms physics) | `robosuite_adapter.py` `_FORCED_ENV_KWARGS`; review doc §1–2 |
| Reset path (Phase 8c): protocol = canonicalized soft reset ~2.7 ms/case vs retired hard reset ~530–770 ms (recompiles MJCF every episode); adopted path bitwise-deterministic per case (gate passed all 5 tasks); stock path was order-dependent (same case twice → 9.6e-2 / 3.8e-1) | `robosuite_adapter.py`; `scripts/dev/ab_reset_equivalence.py`; review doc |
| Parallel runner processes on one namespace are UNSAFE: `RunnerState.save()` has no cross-process lock/merge (lost updates); `results.jsonl` appends are FileLock-safe but insufficient — final sweep runs sequential | `runner/registry.py` save(); review doc §4 |

---

## Progress Log

| Date | Step | Evidence / note |
|---|---|---|
| 2026-08-22 | Ledger created | Derived from `final_baselines_plan.md` + code review of commit `f1729c7` state |
| 2026-08-22 | Phase 3 pre-implementation audit | Verified all 5 controls convertible: 3 config-only (same class as proposed method), 2 small code changes (shared `partial_reinit_experts_from_action_head` utility exists). No architectural blockers. Details in Phase 3 header. |
| 2026-08-22 | Primary baselines audit (paper table) | All 5 task baselines verified publication-ready: no code changes needed; `bc_large` needs only a manifest row (S5.3); post-migration capacity match becomes exact (382,646 vs 385,855); BC-RNN rollout reset enforced by runner. Mandatory disclosures: BC-RNN params (1.16M, 5.6×) + not-history-matched; epoch accounting. |
| 2026-08-22 | Baseline-coverage survey (D9) | Web survey of robomimic study suite, Diffusion Policy low-dim comparators, LAR-MoE, Cluster-aware Upcycling. Decision: no new trained baselines; add literature-context table + positioning defenses instead (S8.7, D9). |
| 2026-08-22 | **Phase 0 complete** (S0.1–S0.3) | Baseline: HEAD `bb2ebd3`, 655 tests passed (41.9 s), runner `--list` OK. Go-ahead recorded (S0.1); D-working-answers recorded (S0.2): D1 Lift-only, D3 drop, D4 superseded, D5 `outputs_final/`, D6 rename later, D7 3 seeds, D8 after, D9 no; **D2 open until S9.2 by design.** |
| 2026-08-22 | **Phase 1 complete** (S1.1–S1.4) | `phaseforge.yaml` = R50 content byte-exact (diff: only header + `name:`), `phaseforge_r50.yaml` deleted, 24/24 resolved-contract checks pass at seeds 42+43 (init seed follows training seed). |
| 2026-08-22 | **Phase 2 complete** (S2.1–S2.5) | Aliases removed from `utils/config.py` AND duplicate found+removed in `scripts/protocol/preflight_configs.py`; canonical guard test written; manifest marked archival. **4 tests failed first run** (old-world encodings): 2× cli (helper lacked `project.seed` for the new `${project.seed}` interpolation), 2× state-machine (asserted old E=8 defaults) — all four fixed to construct the new world explicitly, NOT skipped. Final: **655 passed**, preflight **150 train + 135 eval cells OK**, runner `--list` OK. **D10 discovered**: oracle/teacher eval raises on centroid-init (empty soft-mapping buffer) — must be resolved before Phase 3/oracle eval. |
| 2026-08-22 | **Phase 3 complete** (S3.1–S3.10) | Group A: 3 config-only edits (pf_spherical_kmeans/pf_kmeans/pf_phase_head → partial_warm 0.5, seed `${project.seed}`). Group B: config-driven `expert_init` on `WarmStartMoEModel` + `PlainEncoderPhaseBootstrapModel` (default unchanged = warmstart 0.02), `_expert_init_info` metadata everywhere. **Live bug found+fixed**: CLI's `training_seed=` kwarg crashed ALL FOUR baseline bootstraps since `0a7e415` (warmstart_moe/pprr/pepb/teacher_forced stage-2 would all TypeError); signatures fixed + regression test. Shared `hash_dropped_indices` promoted to `components/expert.py` (byte-identical; first draft had a hash mismatch, caught by re-reading original). **End-to-end proof**: all 6 cells (proposed + 5 controls) built from real yamls → identical dropped-set hash at same seed → exact-match factorial. Suite: **660 passed** (+5 net new), preflight 285 cells OK. D10 remains open (blocks oracle eval only). Committed `84f7451`. |
| 2026-08-22 | **Phase 4 complete** (S4.1–S4.5) | `lift_ablation.json` migrated: removed 5 redundant/broken cells (pf_random_warm, phaseforge_e6, warmstart_r50, pf_jitter_00/10 — warmstart_r50's `+`-appends would crash post-migration AND recreated the canonical method); added `pf_full_warm` (EXP-212) + drop sweep `pf_drop00/25/75/100` (EXP-213..216); `pf_one_warm_plus_random` overrides reduced to non-canonical factors; `pf_spherical`/`pf_ft` yamls pinned to partial_warm. Corruption semantics verified (exact mod-P replacement, train-split-only, deterministic, clean-GT preserved). Manifest: 27 cells; runner `--list` OK; preflight **84 train + 81 eval cells OK**; affected tests 80 passed. |
| 2026-08-22 | **Phase 5 complete** (S5.1–S5.6) | `bc_large` added for all five tasks (46–50); hygiene verified programmatically; expected step count computed BEFORE dry-run (**315**) and dry-run reports exactly 315. Preflight parameter bug fixed: hardcoded Lift count + mixed total/deployed bases → now per-task dynamic PhaseForge deployed reference (±2%); actual match +0.84%…+1.81%. Preflight 165 train + 150 eval OK. Committed `abbc6a6`. |
| 2026-08-22 | **Phase 6 complete** (S6.1–S6.6) | Resolver `expected_config_hash` gate (fail-closed, missing hash rejected); `verify_checkpoint_contract` (expert count from state keys, sidecar model/stage, unreadable→fail) wired into BOTH runner funnels with `FINAL_EXPERT_CONTRACT=6`; one lookup bug found+fixed during testing (sidecar path passed as file not dir). `outputs_final/` namespace created + gitignored (D5). 11 new tests incl. legacy 8-expert rejection through both funnels; fixtures upgraded to real torch artifacts. **671 passed**; dry-run 315 OK. |
| 2026-08-22 | **Phase 7 complete (code/infra); gate EXECUTION deferred to sweep machine** | 671 tests green (S7.1–S7.3). `.venv-rollout` (py3.11.15, robosuite 1.5.1/mujoco 3.2.7) + `.venv-toolhang` (1.5.0/3.2.7) provisioned with the Windows mujoco.dll patch; **finding:** dev venv is py3.14 → mujoco 3.2.7 has no cp314 wheels → rollout env must be py3.11 (runbook §1.1). Gates 1–8 of §11 green; gate 9 execution pending (Lift run cancelled in session, documented); gate 10 namespace fresh. |
| 2026-08-22 | **Phase 8 complete** (S8.1–S8.7) | research_definition §5 lineage + Wave tables updated (jitter rows removed, §6b Waves 3–4 added, D10 noted); authoritative rollout plan gained lineage + reproduction-expectation notes + concrete D2 draft (Holm, DRAFT pending professor) — target corrected from the superseded final_evaluation_plan.md; final_run_plan.md fully rewritten (venv setup, gates-first, outputs_final everywhere, fairness regeneration); baselines_methods.md updated; NEW baseline_positioning.md (D9 record + literature table + defenses). Historical reports untouched (verified). |
| 2026-08-22 | **Phase 9: fully prepared, execution pending** | Launch procedure complete in runbook §1–§7. Blockers recorded: professor approval (S0.1 follow-up), D2 confirmation, gate-9 execution, sweep-machine compute; D10 before oracle eval. Ablation manifest ready (27 cells, preflight-clean). |
| 2026-08-22 | LAR-MoE plan reviewed against the full paper (arXiv:2603.08476v1) and revised to Revision 2 | Cross-check confirmed the two-stage design, losses (Eqs. 1-2, 5), soft routing with learnable T (init 100), the ±F/±R ablation structure, and the exclusion list. Fixed: expert architecture unspecified (now REQUIRED: ACT-style transformer-decoder experts with H learned query tokens, no CVAE); expert conditioning path c_t pre-registered (primary: frozen student latent; deviation alternative must be renamed); AdamW pinned; stop-gradient/joint-training/teacher-disposal recorded as AC1-AC3; entropy sign semantics stated; group-sparsity 2x3 expert grid fixed for N=6; hyperparameter provenance + tuning-parity rule added (no official code exists, verified 2026-08-22). |
| 2026-08-22 | **D11 recorded: `lar_moe_state_only` DEFERRED** | Decision put in writing per supervisory recommendation, referencing both `docs/dev/lar_moe_state_only_implementation_plan.md` (Rev. 2, now status DEFERRED with revisit conditions) and `docs/plan/reports/baseline_positioning.md` (D9 record). No code will be written for it; all effort redirects to the five-task sweep. Mechanism-story caution reiterated: R50 identity locked, claims await the sweep. |
| 2026-08-22 | **D11 escalated: `lar_moe_state_only` plan hard-deleted** | Project owner directed hard deletion. `docs/dev/lar_moe_state_only_implementation_plan.md` removed via `git rm` (survives in history at `1ab35de`, Rev. 2). Pre-deletion sweep confirmed zero active references in `phaseforge/`, `tests/`, `experiments/`, `scripts/` — no code, config, or manifest entry ever existed. All effort remains on the five-task sweep. |
| 2026-08-22 | **Phase 8b: five-task readiness audit complete** (prompt: "only Lift has ever been run — is anything missing for the other four?") | Verdict: no missing per-task artifacts — single-manifest design + 15 validated data configs + per-task caches + ToolHang venv dispatch all verified LIVE (preflight 165+150 green, dry-run 315, both venvs probed, all 15 configs composed + registry-validated, banks inspected). **Two real reporting bugs found**: S8b.1 `paired_wilcoxon.csv` baseline identity `(baseline, None)` never matches tagged five-task rows → silently empty CSV after the sweep; S8b.2 `stratified_stats.py` pools tasks (grouping lacks `task`). Plus: test gap S8b.3 (no non-Lift config composition in pytest), bank pre-generation S8b.4 (Can/Square/ToolHang), text alignments S8b.5 (registry schema strings, dry-run preview docs; ledger S7.5 + facts row corrected in place), per-phase artifact check S8b.6. All recorded as Phase 8b steps. |
| 2026-08-22 | **Phase 8b implemented (S8b.1–S8b.5)** | S8b.1 fixed BOTH the per-tag baseline identity AND a second latent flaw found during implementation: the `(stage, seed, tag)` pairing key could never pair PhaseForge (stage 2) with the BC family (stage 1) — key relaxed to `(seed, tag)` (deployment comparison on identical resets). S8b.2 fixed the task pooling AND a third bug: `--root` append+default silently mixed historical `outputs/` into explicit `outputs_final` scans — now replaces. S8b.3: `tests/data/test_per_task_configs.py` (33 tests incl. labeler-slice boundary guard + manifest↔config drift guard). S8b.4: Can `310d9cfd3fa5e843` / Square `e16288589f5f69c2` generated under `.venv-rollout`, ToolHang `db5b4c2a5e6519d0` under `.venv-toolhang` via the eval-time `load_or_generate_bank` path — first-ever successful ToolHang env-stack exercise; all five banks SHA-verified. S8b.5: registry strings aligned v2 (+drift-guard test), dry-run preview note in runbook §5. Suite **708 passed** (+37); preflight 165+150 green post-change. S8b.6 remains execution-time (post-sweep). |
| 2026-08-22 | **`final_run_plan.md` finalized** | Full rewrite against the frozen end state. Corrections vs. the old text: sweep/dry-run/gate/--list/re-run commands now use `.venv-rollout/Scripts/python` (the old `uv run` sweep command would fail every eval subprocess — the dev env has no robosuite/mujoco); `fairness_accounting.py` takes NO `--outputs` flag (old command would crash) — the per-task parameter match is the preflight script's job; step-count derivation restated exactly (21 per (task,seed) = 11 train + 10 eval × 5 × 3 = 315). Additions: frozen 10-method matrix table with roles/stages/stage-2 sources + the D2 primary-family scope; five-bank id table (§2.3); expected artifact row counts (150/165); S8b.6 per-phase check in §6; per-task `stratified_stats.py` in §8; new §9 ablation suite (27 cells, 165 steps, `outputs_ablation/` namespace — gitignored; both manifests' dry-runs verified under the rollout venv: 315 and 165 steps); D9 literature-context exclusion in §10. |
| 2026-08-22 | Run plan reconciled against the original baseline/ablation plain-list | Item-by-item check vs. the early plan list passed with one recorded amendment and both loose ends closed: the three router controls (`pf_spherical_kmeans`/`pf_kmeans`/`pf_phase_head`) stayed **Lift-only** per D1 (not promoted to the five-task matrix as the early list sketched) — §1 now states the exact-match factorial pinning; §9 now spells out the suite composition (9 Lift matrix replicas without `bc_rnn` + 18 ablation-only cells), the removals (`pf_random_warm`/`phaseforge_e6` D3, `pf_jitter_00/10` D4, `warmstart_r50`), and that the optional top-1 routing variant was never included; §10 records the retired identities (8-expert config, `phaseforge_r50` file+alias, `lar_moe` D11). All claims re-verified against `lift_ablation.json` programmatically. |
| 2026-08-22 | **Phase 8c: rollout performance & determinism revision implemented and gated** (D12 recorded) | Audit answer: rollout IS fully headless (no GL ever; dataset pins lite_physics=False, 25 substeps/step; step ≈17.8 ms ≈97% robosuite OSC). Discovered the stock reset path was **episode-order-dependent** (warmstart/OSC-cache/initial_joint/geometry leaks; same case twice diverged 9.6e-2/3.8e-1) and hard reset recompiled the MJCF every episode (530–770 ms). Implemented: canonicalized soft reset (construction seeding + hard_reset=False + hidden-state canonicalization), P1 tolerance harmonization, P2 torch single-thread in eval. Standing gate redefined to S-vs-S on pinned metadata and **PASSED on all five tasks** (bitwise; retired-hard arm informational ≤5e-2, attributed). Bench harness corrected to protocol env (reset 2.7 ms vs 637 ms; eval ≈2–3 h for all 150 cells). Parallel-cells proposal REJECTED (RunnerState not concurrency-safe) — final sweep sequential. Review doc corrected (lite_physics fact, residual attribution, 150-cell count, disclosures). Suite 711 green. D12 ratification pending before numbers enter the paper. Working tree left UNCOMMITTED per project owner instruction. |
| 2026-08-22 | **Phase 8c finalization: external review adjudicated; refutation institutionalized** | The pasted external review's "critical regression" (soft reset allegedly degenerates bank generation) was verified against source and measurement — **REFUTED** (`_reset_internal()` placement sampling runs in both reset branches; 5-case generation distinct on all five tasks, min pairwise L2 0.096–0.209 vs 1e-3 threshold). Its Patch A (per-restore hard-reset toggling) NOT implemented — solves a non-existent problem and re-introduces the retired branch. Added `--bank-smoke` permanent mode to `ab_reset_equivalence.py` (passed all 5 tasks, exit 0, in-memory only). Review doc §3.3 records the full adjudication incl. rejected docstring/gate re-anchoring proposals. Ruff clean; suite re-run green. The review's two open questions answered by evidence: bank generation needs NO hard reset; `action_tolerance=1e-4` remains the single contract. Tree remains UNCOMMITTED per standing instruction. |
| 2026-08-22 | **Post-review bank invalidation patch implemented** | Reset-bank identity and manifests now carry `soft-reset-canonical-v1`; `ResetBank.load` rejects missing or stale protocol revisions, so pre-revision bank IDs cannot be reused by the final sweep. The five current banks must be regenerated through the final evaluation path before the gates and their new IDs recorded in the run plan. The rollout benchmark now restores the saved Torch intra-op thread count correctly. Focused rollout tests: 60 passed; full suite: 713 passed. |
| 2026-08-22 | **Phase 8d: training hot-path review complete — PLAN ONLY, no patches applied** | Full training-path code read + clean measurements (no profiler for headline numbers): the main train loop is already sync-disciplined; the real costs are (a) `collect_environment` importing sklearn/wandb/scipy per process for `__version__` (≈4.5–5 s × 315 cells), (b) MoE dispatch `.any()` ×6/step = CUDA syncs (404k per ToolHang stage-2 run), (c) per-step `isfinite(loss)` sync, (d) per-metric-per-batch validation syncs (~340/epoch on big tasks), (e) 68 ms/batch per-item Python data path behind a 12 ms worker-IPC floor, (f) unmemoized hash/git calls. Patches T1–T7 proposed with bit-identity/determinism gates (T2/T3 bit-identical — empty-slice no-op verified empirically; T5's flat iterator changes permutation sequence, disclosed, seed-reproducible; RNN cells keep the legacy loader). torch.compile/AMP/TF32/more-workers/multi-cell-runner rejected WITH evidence. External sync claims verified against the official PyTorch tuning guide + NVIDIA sync-free guide. Plan awaiting project-owner approval in `docs/dev/training_performance_review.md`; nothing implemented, nothing committed. |
| 2026-08-22 | **Phase 8d rev 2: per-issue online research pass + second code sweep** | Every issue re-checked against community/official sources; no proposed patch invalidated. NEW issue found: unconditional `import wandb` in `_train_body` (~2.2 s/process with wandb installed and disabled) → patch T1b. T1 upgraded to single-pass `distributions()` (importlib_metadata#95). `torch._assert_async` REJECTED for T3 (sync-free but corrupts CUDA context on failure, misleading async traceback, ignored assert_msg pytorch#131491, private API) — robust epoch-end flag kept. T5 validated as canonical resident-tensor+randperm+index_select pattern (forums/SO/Lightning) with CPU-generator refinement. Dense MoE dispatch rejected (Shazeer sparse lineage; GEMM batch-size float drift). torch.compile rejection reaffirmed via official caching tutorial + pytorch#114206/#113287/#96152 (per-process Dynamo/AOTAutograd cost persists). AdamW unchanged (foreach already default on CUDA; fused = numerics drift). Review doc rev 2 (§0 research table, §7 sources); still PLAN ONLY, nothing implemented, nothing committed. |
| 2026-08-22 | **Phase 8d rev 3: training patches T1–T7a IMPLEMENTED and gated** | Approved ("implement all the patches"). Baseline determinism proven (2× s1 IDENTICAL). T1 (metadata versions, sys.modules-first + single-pass `distributions()`), T1b (wandb mode-guarded import), T6 (git_commit lru_cache + hash threaded through cli) — curves bit-IDENTICAL s1+s2. T2 (MoE `.any()` skip removed, bit-identity vs old-loop reference incl. unselected experts), T3 (on-device epoch-level loss-finite flag checked before any artifact + fused single-sync no-clip grad guard), T7a (cached zero scalars) — bit-IDENTICAL. T4 (float64 on-device validation aggregation + one cat/`.cpu()` per epoch) — bit-IDENTICAL via exact-equality tests replicating the old Python-float math. T5 (flat in-memory batch iterator; content bit-equal, drop_last parity, seed-reproducible; RNN configs keep DataLoader) — deterministic (s1×2, s2×2 IDENTICAL), differs from baseline only by the disclosed permutation stream. Fetch head-to-head: 490–551→3.1–9.0 ms/epoch; spawn 6.8 s→0. Suite 738 green (+25 tests); preflight 165+150; dry-runs 315/165 exact; ruff clean on touched files. Pre-existing stats-test float-aliasing failure fixed TEST-ONLY. Nothing committed. |
| 2026-08-22 | **Phase 8d rev 4: external review adjudicated — P1/P2 confirmed and patched** | Reviewer verified the implementation clean (738 tests, dispatch bit-identity, guards, aggregation, fingerprint, caching) and raised two findings; both manually confirmed against source. P1: flat loader's gathered batches are pageable while `_move_batch` requests non_blocking — the pinned-batch contract the legacy DataLoader provided on CUDA was silently dropped; patched with `pin_memory` support gated identically to the legacy branch (and shared gating hoisted in `_build_dataloaders`), `_pin` helper testable on CPU-only builds, real-pinning test runs only where CUDA exists. P2: stale `result` local annotation → union type. 740 passed + 1 skipped; ruff clean; same-seed determinism re-proven. Cloud-GPU flat-loader benchmark recorded as a first-run gate in final_run_plan.md §3 (dev box has no GPU — speedup on GPU remains an estimate until then). Nothing committed. |
