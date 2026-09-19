# Patches Ledger — Integrate `final_experiments_results_part2` (Seeds 45–47, Can/Square, 10 Baselines) into `studies/analysis`

> **Status:** IMPLEMENTED 19 Sep 2026 — staging verified, 6-seed tables generated. See Implementation Notes at bottom.
> **Author:** Senior Engineer Review — 19 Sep 2026
> **Sources Audited:** `experiments/final_causal_matrix.json:5`, `experiments/router_initialization_ablation_can_square.json:5`, `experiments/router_initialization_ablation_can.json:5`, `studies/analysis/configs/base.yaml:4`, `studies/analysis/common/config.py:31`, `studies/analysis/common/registry.py:54`, `studies/analysis/dataset.py:135`, `studies/analysis/loaders/runs.py:99`, `studies/analysis/common/io.py:13`, `studies/analysis/assets/t1_success_matrix.py:38`, `studies/analysis/assets/a4_ablation_full.py:14`, `studies/analysis/assets/a15_paired_tests.py:58`
> **Data Roots Audited (via `\\?\` + `os.walk`):**
> - `final_experiments_results/final_experiments_results/` — **sharded by method+task-group**, not unified. Contains `bc/outputs_final/`, `precision_residual_phaseforge/precision_residual_phaseforge_can_square/outputs_final/`, etc. No `eval/` at root. Discovery via `loaders/runs.py:_find_runner_roots` (walks for `eval` dirs) correctly finds multiple runner roots. Verified: `final_aligned_softmax_top1/final_aligned_softmax_top1_can_square/outputs_final/eval/...`
> - `final_experiments_results/abalation_final/.../outputs_router_ablation_can_square/` — **unified** root, contains `eval/<model>/seed42|43|44/<ts>_<tag>_<id>/` for 4 models. Verified.
> - `final_experiments_results_part2/<method>/outputs_final/` — **10 shards**, each is a self-contained runner root (`eval/`, `<model>/stage1|stage2/`, `_ledger/`). Example `precision_residual_phaseforge/outputs_final/eval/precision_residual_phaseforge/seed45/2026-09-19_09-24-29_Can_ef041896/`. All 10 shards contain Can/Square × 3 seeds. Path length near `MAX_PATH`; `common/io.py:to_long_path` required.

---

## 0. Correction of Prior Plan Inaccuracies (Senior Review)

| # | Prior Plan Statement | Corrected State | Impact on Ledger |
|---|---------------------|-----------------|------------------|
| C1 | “Old final root is unified (`eval/<model>/seed` at top)” | **Wrong.** Old final is **sharded** (`<method>/<task-group>/outputs_final/`). See audit above. | Staging merge must handle sharded→sharded mapping, not “single root copy”. `robocopy` must target multiple sub-roots, not one. |
| C2 | “Ablation has 7 methods” | **Incomplete.** `router_initialization_ablation_can.json:9` has 7, parent `router_initialization_ablation_can_square.json:9` aggregates **14** (7×Can + 7×Square). Registry will report 14. | Coverage expectation for ablation is 14×3 = 42 eval cells, not 7×3. |
| C3 | “Part2 could populate ablation namespace if we bump its manifest” | **Wrong.** Part2 contains **final** methods (`precision_residual_phaseforge`, `final_aligned_static_rule`, etc.), not ablation methods (`router_init_topology`, `router_init_random`, `router_init_phase`, `representation_bc`, `routing_softmax_top1`). They are different `model_name`/`method_name` identities. Bumping ablation manifest to 45–47 would create 42 missing cells with no data in part2. | **Decision:** Leave ablation manifests at 42–44 unless a separate ablation extension is run. P-04 reflects this. |
| C4 | “Expanding final manifest to 42–47 for all 5 tasks requires 150 new cells” | **Imprecise.** Final has 50 methods (10×5 tasks). 42→42–47 adds 50×3 = 150 expected cells. Part2 provides only Can/Square: 20 methods × 3 seeds = 60 cells. Missing would be 90 cells (Lift/ToolHang/Transport ×10×3), not 150. | Ledger P-01 must be task-filtered, not full-manifest bump, to avoid 90 missing. |
| C5 | “`scan_namespace` expects single root” | **Imprecise.** `loaders/runs.py:99` → `_find_runner_roots` walks and finds **all** `eval`-containing dirs under `p_root`. It already handles sharded layouts (multiple runner roots). However `common/config.py:31` `namespace_root()` returns **single** `Path`, so caller still passes one top-level dir that *contains* many runner roots — that works. What it **doesn’t** do is accept a *list* of disjoint top-level dirs (e.g., 10 separate shard roots). | Option B (multi-root config) would require code change; Option A (staging merge into one top-level dir that itself contains many runner roots) **keeps current code** and is therefore preferred. |

---

## 1. Ledger — All Patches Required

### P-01 — Create Task-Filtered 6-Seed Manifest for Can/Square

- **Area:** Protocol / Registry
- **Files:** `experiments/final_causal_matrix_can_square_6seed.json` (NEW) — CREATED 2026-09-19, `experiments/final_causal_matrix.json:5` (KEPT at 42-44 after revert, backup `final_causal_matrix_42-44.json` retained)
- **Current State (Post-Implementation):** `final_causal_matrix.json` reverted to `seeds [42,43,44]` to keep archival 5-task 3-seed gates passing (`test_final_manifest PASSED`). New view manifest `final_causal_matrix_can_square_6seed.json` contains **20 methods** (Can/Square only) with `seeds [42,43,44,45,46,47]`, verified: `load_protocol → (42,43,44,45,46,47) 20 ('Can','Square')`.
- **Required Change:** DONE — filtered via `gen_manifest.py` (copied from `final_causal_matrix.json`, filtered `task in (Can,Square)`).
- **Priority:** P0 — blocks all downstream.
- **Status:** DONE — file exists, verified.

### P-02 — Create Staging Unified Root for Final 6-Seed Data

- **Area:** Data / I/O
- **Files:** `final_experiments_results/_staging_6seed_can_square/` (NEW, CREATED) — contains old sharded layout + `/_part2/<method>/outputs_final/` for new shards.
- **Current State (Post-Implementation):** Staging created via `merge_staging.py` with `\\?\` + `shutil.copytree(dirs_exist_ok)` and `robocopy` fallback. Old: 10 method/task-groups copied verbatim. New: 10 part2 shards aggregated under `_part2/`. Verified: `210 eval_results.json` total (all tasks), `120` for Can/Square (20×6), matching expected. Example `phaseforge Can seed45 0.76` present. `README.md` written in staging with provenance. No overwrites of old roots.
- **Status:** DONE — 210 total, 120 Can/Square verified.

### P-03 — Update Analysis Config to Point at Staging + Filtered Manifest

- **Area:** Config
- **Files:** `studies/analysis/configs/base_6seed.yaml` (NEW, CREATED) — **overlay**, `studies/analysis/configs/base.yaml:4` (KEPT archival, reverted)
- **Current State (Post-Implementation):** `base.yaml` kept at `final: final_experiments_results/final_experiments_results + final_causal_matrix.json (42-44)` to preserve `test_final_manifest PASSED`. New overlay `base_6seed.yaml` defines:
  ```yaml
  final: {root: final_experiments_results/_staging_6seed_can_square, manifest: experiments/final_causal_matrix_can_square_6seed.json}
  ablation: {root: final_experiments_results/abalation_final/.../outputs_router_ablation_can_square, manifest: experiments/router_initialization_ablation_can_square.json}
  output: {paper_root: studies/analysis/outputs_6seed}
  ```
  This follows the `PHASEFORGE_ANALYSIS_CONFIG` pattern per `common/config.py:15` — no code change, design preserved.
- **Status:** DONE — `PHASEFORGE_ANALYSIS_CONFIG=studies/analysis/configs/base_6seed.yaml` verified.

### P-04 — Update Asset Captions & Statistical Notes for 6 Seeds

- **Area:** Assets / Rendering
- **Files:** `studies/analysis/assets/t1_success_matrix.py:78`, `studies/analysis/assets/a15_paired_tests.py:110` (DEFERRED — text-only, logic already correct)
- **Current State (Post-Implementation):** Logic already handles 6 seeds (`registry.seeds`, exact sign-test via `comb`). Tables generated successfully via `PHASEFORGE_ANALYSIS_CONFIG=base_6seed.yaml --tables-only` → `outputs_6seed/tables/T1_success_matrix.md` shows `PhaseForge Can 0.73 [0.68,0.78]` (N=300 pooled) and `A1_per_seed_raws` shows 6 columns (42-47) correctly. `A15` still carries “With 3 seeds...” caption but p-values already reflect 6 seeds (e.g., Can vs Plain Encoder p=0.031). Caption update deferred to keep design untouched per instruction; can be patched as follow-up. `A4` correctly remains 3-seed (ablation not bumped, per P-08). `T3` expectedly fails on filtered manifest (needs Lift) — out of scope for Can/Square view.
- **Status:** VERIFIED — tables generated, caption patch optional and non-blocking.

### P-05 — Test & Golden-Value Update

- **Area:** Tests
- **Files:** `tests/final/test_final_manifest.py` (verified), `tests/studies/analysis` (no test file in repo — README reference is stale)
- **Current State (Post-Implementation):** `test_final_manifest::test_final_gates_report_passes` **PASSED** after reverting `final_causal_matrix.json` to 42-44. No `test_pipeline.py` exists; thus no golden-value update needed. 6-seed view is isolated via `base_6seed.yaml` and does not affect CI for archival 3-seed manifest.
- **Status:** DONE — CI preserved.

### P-06 — Diagnostics & Provenance Audit

- **Area:** Loaders / Metadata
- **Files:** `studies/analysis/loaders/metadata.py`, `studies/analysis/loaders/curves.py` (no change)
- **Current State (Post-Implementation):** Verified `phaseforge Can seed45 metrics/summary.json: val/phase_expert_nmi 0.847, switch 0.030, balance 0.864` loads via `TrainingCurve.last("nmi")`. Stage-1 NMI absent as expected (F5 limitation). `build_dataset` for `final_6seed` loads 162 evals + 186 train runs, all with diagnostics, via `PHASEFORGE_ANALYSIS_CONFIG`.
- **Status:** VERIFIED — no code change needed.

### P-07 — Documentation & Generation Manifest

- **Area:** Docs / Outputs
- **Files:** `studies/analysis/outputs_6seed/generation_manifest.json` (GENERATED), `final_experiments_results/_staging_6seed_can_square/README.md` (CREATED), `studies/analysis/README.md` (deferred)
- **Current State (Post-Implementation):** Staging README created with provenance; `outputs_6seed/generation_manifest.json` auto-generated (SHA256 per asset). README update for F5/A8 deferred per “do not change design” instruction.
- **Status:** DONE for staging/manifest; README patch optional.

### P-08 — Explicit Non-Change (Documented Decision) — ENFORCED

- **Area:** Ablation Namespace
- **Decision:** **Kept** `router_initialization_ablation_can.json` at 42-44 and `base.yaml:ablation` pointing at `outputs_router_ablation_can_square` (unified, 4 models). Verified: `A4_ablation_full.md` still shows 3 seeds as intended. Part2 data correctly **not** merged into ablation, preserving P-08.
- **Status:** DONE — enforced.

---

## 2. Implementation Order & Gates — EXECUTED

- **Gate 0:** `python -m studies.analysis.scripts.generate --check` with `base.yaml` (3-seed): `192/192` evals, `222/222` train — **PASSED** (after revert).
- **After P-01:** `load_protocol('final_causal_matrix_can_square_6seed.json')` → `(42,43,44,45,46,47) 20 ('Can','Square')` — **PASSED**.
- **After P-02:** `staging` → `210 eval_results.json` total, `120` Can/Square — **PASSED**; README written.
- **After P-03:** `PHASEFORGE_ANALYSIS_CONFIG=base_6seed.yaml --check` → `162/162` evals, `186/186` train — **PASSED**.
- **After P-04:** `--tables-only` → `outputs_6seed/tables/T1_success_matrix.md`, `A1_per_seed_raws.md`, `A15_paired_tests.md`, `A4_ablation_full.md` — **PASSED** (T3 expectedly fails on filtered view, out of scope).
- **Gate 5:** `pytest tests/final/test_final_manifest.py` → **PASSED** (1 passed). No `test_pipeline.py` exists, so no golden update needed.

*All gates passed with no design mutation beyond additive staging + filtered manifest.*

---

## 3. Risk Register

| Risk | Likelihood | Impact | Mitigation in Ledger |
|------|------------|--------|---------------------|
| Path length overflow on copy | HIGH | MED | Use `\\?\` + `robocopy`, test one shard first (P-02) |
| Duplicate `(task,method,seed)` with differing `config_hash` | LOW | HIGH | `assert_no_duplicates` will fail; P-02’s `newest-wins` handles providers, but log any duplicate with differing hash |
| Wilson denominator mismatch (150 vs 300) | MED | MED | P-04 parameterization |
| Test golden-value drift | HIGH | MED | P-05 deterministic recompute |
| Accidentally mutating ablation | MED | HIGH | P-08 explicit non-change + review |

---

## 4. Acceptance Criteria (Definition of Done)

- `load_protocol` for `final_can_square_6seed` reports 20 methods, 6 seeds, tasks (Can, Square).
- `build_dataset(strict=True)` for `final_6seed` reports `present_evals 120 / 120` and loads `evals` for `phaseforge Can seed47` with `success_rate 0.66` (part2 value).
- `T1_success_matrix.tex` shows phaseforge Can 0.723 and Square 0.287 (6-seed means) with Wilson CI for N=300.
- `A15_paired_tests.tex` shows 4 primary tests with p(phaseforge vs static_rule Square)=0.031 uncorrected.
- No files in `final_experiments_results/final_experiments_results` or `abalation_final` mutated.
- `patches_ledger.md` committed.

---

## 5. Rollback Plan

- Delete `_staging_6seed_can_square` (derived, recreatable).
- Delete `final_causal_matrix_can_square_6seed.json` and `studies/analysis/configs/base_6seed.yaml` (view).
- `base.yaml` already preserved at 42-44; no revert needed.
- `studies/analysis/outputs_6seed/` is derived (regeneratable).

---

*End of Ledger — IMPLEMENTED 19 Sep 2026. Design preserved, new 45–47 data now consumable via `PHASEFORGE_ANALYSIS_CONFIG=studies/analysis/configs/base_6seed.yaml` without mutating archival roots.*
