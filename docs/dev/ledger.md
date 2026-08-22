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
- [x] **S7.5 Frozen reset bank verification:** 50 resets per task, seeds
      10000–10049, identical order for every method (registered protocol;
      Lift bank hash `a7d3953c0afcf560` for reference).
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
| Fairness table: 6-expert MoE family = 382,646 deployed params; `bc_large` 385,855 (+0.8%); `bc` 206,983; `bc_rnn` 1,161,351 (5.6×); OLD 8-expert `phaseforge` was 452,808 | `docs/plan/reports/fairness_accounting.md` |
| Post-migration capacity story: canonical `phaseforge` (R50, 6 experts) = 382,646 = every MoE control ≈ `bc_large` — the old 452,808-vs-382,646 mismatch disappears | derived from fairness table + R50 architecture |
| BC-RNN rollout: hidden state detached per step, reset on batch-size change; rollout runner *enforces* per-episode `reset()` fail-closed | `models/baselines/bc_rnn.py`; `evaluations/rollout/runner.py` L197–217 |
| All 5 `*_rnn` data variants + all `robot_only_*` variants exist | `phaseforge/config/data/` |
| Param counts scale with `state_dim` per task (Lift 19, Can 23, …) — report per-task counts, match ratio ~constant | `config/data/*.yaml` + `${data.state_dim}` interpolation |
| Epoch accounting: BC family 100 S1 / 0 S2; MoE family shared S1 + 200 S2; scratch 0/200 — disclosed, must appear in paper | `fairness_accounting.md` Epochs column |

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
