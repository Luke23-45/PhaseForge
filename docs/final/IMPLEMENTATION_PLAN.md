# IS-PhaseForge Implementation Plan — Impedance-Switched, Memoryless, Observation-Consistent MoE

**Source of Truth:** `docs/final/final_phaseforge.md` (2026-09-03, Final Revised Report — Memoryless Impedance-Switched PhaseForge)
**Repo Root:** `C:\Users\Hellx\Documents\Programming\python\Project\Neryva\PhaseForge`
**Status:** Planning only — no implementation code in this document. All changes modular, CPU-verifiable, GPU-free for local checks.
**Recheck:** 2026-09-03 (final) — fully re-read from disk a second time with no memory use (`final_phaseforge.md` §§1–16; `phase_moe.py:forward/get_action/bootstrap_moe`, `router.py:TopKRouter`, `expert.py:ExpertMLP/warm_start`, `moe_layer.py`, `encoder.py`, `action_head.py`, `base.py`, `stage1_loop.py:_compute_loss`, `stage2_loop.py:_compute_loss`, `features.py:phi_t`, `phase_labeler.py`, `dataset.py`, `collator.py`, `switching_linear_k6.yaml`, `rollout.yaml`, `task_registry.py`, `runner.py:_policy_action/_episode`, `training_summary.jsonl` 6 rows, 3× `eval_results.json`/`rollout_summary.json`, baseline report, `tests/`, `scripts/dev/`, `experiments/`, `runner/cli.py --dry-run`). All paths/shapes/numbers/math re-verified; phase grouping made disjoint (R02→Phase 4, R03→Phase 5).
**Deployment contract (locked):** `x_t → normalize → E(x_t)=z_t → router → expert(s) → impedance/target params → action adapter → env`. No hidden state, no history, no chunking, no oracle/sticky test-time routing for the primary result.

---

## 0. Verified current-state baseline (code-inspected, not speculated)

### 0.1 Models

| Component | File | Current behavior |
|---|---|---|
| `BaseManipulationModel` / `ModelOutput` | `phaseforge/models/base.py` | `forward(batch)->ModelOutput(action_pred (B,A), phase_logits (B,P)\|None, routing_weights (B,K), expert_indices (B,K), gate_logits (B,E), aux_losses{"balance","sticky"} in Stage2; base docstring lists {"balance","phase"})`; `get_action(state (B,S))->(B,A)` |
| `StateEncoder` | `phaseforge/models/components/encoder.py` | `input_dim=S (Can 23, Lift 19) → hidden [256,256,256] GELU+Dropout0.1 → output_proj → latent_dim=128 + residual (res_proj Linear if S≠128)`; output `z (B,128)` unnormalized |
| `ActionHead` | `phaseforge/models/components/action_head.py` | `trunk Linear(128→256)+GELU → mean_head Linear(256→A) → tanh` → `(B,A)` in `(-1,1)`; `sample()` for gaussian variant |
| `PhaseClassificationHead` | `phaseforge/models/components/phase_head.py` | Single `Linear(128→P)`, P=6; linear-bottleneck forces phase structure into `z` |
| `TopKRouter` | `phaseforge/models/components/router.py` | `gate_linear Linear(128→E)`, E=6, `top_k=2`, `noise_std=0.1` (+`noise_linear`), `balance_coeff=0.01`, `normalize_input=True` (cosine when weights unit-norm), optional `use_history` + `history_embedding(E+1→128)`/`history_proj`, `sticky_beta=0.9` EMA only for `sticky` eval; `forward(latent (B,128), traj_id, traj_pos)->RouterOutput(weights (B,K), indices (B,K), gate_logits (B,E), balance_loss scalar, sticky_loss scalar)`; `uniform_selection`, `sticky_selection`, `oracle_selection`, `_resolve_previous_top1`, `_compute_balance_loss` (Switch-style `E*sum(f_i*p_i)` on top-1) |
| `ExpertMLP` | `phaseforge/models/components/expert.py` | `input 128 → hidden [256] GELU → output_proj → tanh` → `(B,A)`; `warm_start_experts_from_action_head`, `partial_reinit_experts_from_action_head` (Drop-Upcycling-style shared index set, Kaiming), `one_warm_experts_from_action_head`, `reset_parameters` |
| `MoELayer` | `phaseforge/models/components/moe_layer.py` | `router + ModuleList[E]`; dispatch loop over experts, `combined_output (B,A)=Σ w_i*expert_i(z)`; supports `router_override ∈ {sticky,uniform,oracle}` (eval-only); rejects `latent.ndim!=2` (no `sequence_length>1`) |
| `PhaseBootstrappedMoE` | `phaseforge/models/phase_moe.py` | Stage1: `encoder→action_head+phase_head`; Stage2: `encoder→moe_layer (+phase_head only if teacher_routing.enabled)`; `EVAL_ROUTER_MODES={learned,sticky,uniform,oracle}`; `bootstrap_moe(dataloader, device, router_init, expert_init, training_seed)` handles `centroid/phase_centroid/spherical_centroid/spherical_kmeans/kmeans/phase_head/soft_mapping/random` × `warmstart/partial_warm/one_warm/random`, `prototype_source` accepting `rule/phase/phase_rule→rule` and `dynamic/phase_dynamic→dynamic` (WP1 adds `topo`); `soft_mapping` buffer `(P,E)` zero-init + `require_soft_mapping()` fail-closed; `freeze_encoder/unfreeze_encoder`, `train(mode)` keeps frozen encoder in eval; `get_action` inference path |
| `BehaviorCloningModel` | `phaseforge/models/baselines/bc.py` | `encoder→action_head`, direct 7D tanh |
| Other baselines | `phaseforge/models/baselines/{scratch_moe,warmstart_moe,teacher_forced,oracle_moe,plain_encoder_phase_bootstrap,phase_pretrain_random_router}.py` | Same encoder/expert dims; differ only in bootstrap/routing supervision |

Current routing math (to be replaced/augmented): `logits=gate_linear([norm(z)]) (+noise train) (+history_bias opt)`, `topk_logits,topk_idx=topk(logits,K=2)`, `w=softmax(topk_logits)`, `â=Σ w_e f_e(z)`, `f_e=tanh(MLP(z))` direct 7D.

### 0.2 Training

| Stage | File | Current loss |
|---|---|---|
| Stage1 | `phaseforge/trains/loops/stage1_loop.py:Stage1Trainer._compute_loss` | `L= MSE(â,a) + λ_phase*CE(phase_logits,phase)`; `λ_phase=1.0` (`train.lambda_schedule constant`), optional `phase_weights` (balanced/cui), `soft_target_eps` label smoothing (default 0), `grad_cosine` diagnostic only |
| Stage2 | `phaseforge/trains/loops/stage2_loop.py:Stage2Trainer._compute_loss` | `L= MSE + balance + sticky_coeff*sticky + teacher_lambda*KL(M^T softmax(phase_logits)‖softmax(gate_logits))`; `teacher_lambda` = λ0 first half → linear anneal to 0; `freeze_encoder=true` default, `encoder_lr_scale` opt; `_validate()` logs `phase_expert_nmi, topk/top1 balance/collapse, routing_entropy, routing_switch_rate, routing_accuracy` |
| Callbacks | `phaseforge/trains/callbacks/*` | `MetricTracker, MetricPersistence (summary.json), Checkpointing (monitor val/loss_action), EarlyStopping (disabled by protocol), WandbLogger` |
| Registry | `phaseforge/utils/registry.py` | `build_model` (strips `name/freeze_encoder/encoder_lr_scale`), `build_data_pipeline→DataPipelineStateMachine`, `build_trainer` stage switch |
| CLI | `phaseforge/cli.py` | `train()` / `evaluate()`; `_load_state_dict_checked` fail-closed; `bootstrap_moe` + `init_routing.json` + `init_expert.json` persistence; device fallback cuda→cpu with warning |

Configs: `phaseforge/config/main.yaml` (defaults `data:common, models:phaseforge, train:stage1, eval:metrics`); `config/models/phaseforge.yaml` (canonical R50: E=6,K=2,centroid,partial_warm d=0.5) and `config/models/phaseforge_dynamic.yaml` (same + `prototype_source:dynamic`); `config/data/can.yaml` (S=23: 3+4+2+14, A=7, `RuleBasedPhaseLabeler` P=6, `RobomimicHDF5Ingester`, trajectory split 0.1, `phase_corruption_rate=0`); `square.yaml` identical S=23; `common.yaml/lift.yaml` S=19; `config/train/stage1.yaml` (100ep, AdamW 3e-4, Cosine T=100, monitor val/loss_action) / `stage2.yaml` (200ep, AdamW 1e-4, Cosine T=200, `sticky_coeff=0`, `teacher_routing.lambda0=0.5` inert unless `models.teacher_routing.enabled`); `config/dynamics/switching_linear_k6.yaml` (enabled, K=6, kappa 20, alpha 1, ridge 1e-4, min_var 1e-4, EM 40, tol 1e-3, min_duration 3, seed=${project.seed}, `train_label_field=phase_dynamic`, gates `min_occ 0.005, max_single 0.5, max_switch 0.6, min_nll 0.0, residual_ratio 1.0, enforce=true`); `config/eval/rollout.yaml` (`bank 50/seed2026/auto_generate`, `episodes horizon null→dataset, action_tolerance 1e-4, router_mode learned, require_phase_tracking false`).

### 0.3 Data / discovery

* `phaseforge/data/robomimic/ingester.py:RobomimicHDF5Ingester` — reads `state_keys` low-dim only, validates `action_dim` + `[-1,1]`, `phase=RuleBasedPhaseLabeler.label(traj)`, `phase_thresholds=calibrate(traj)`.
* `phaseforge/data/robomimic/phase_labeler.py:RuleBasedPhaseLabeler` — 6-phase (0 approach,1 pre-grasp,2 grasp,3 transport,4 place,5 retract) from `eef_pos[0:3]` + `gripper max|q|[7:9]` hysteresis (per-demo 5/95 pct calibrate, mirror-aware) + `eef_velocity_threshold 0.01/min_duration 5/median 7`; `CausalPhaseStepLabeler` replicates causally at rollout.
* `phaseforge/data/dynamics/features.py` — `phi_t=[x_t,a_t,Δx_t]`, `TransitionBatch`; `switching_linear.py:StickySLDS` (K=6, EM forward-backward, Viterbi + `min_duration` smoothing, `SingleDynamicsModel` baseline); `diagnostics.py:evaluate_discovery_quality` (occupancy, single-regime frac, switch rate, held-out NLL vs single, residual var); `artifacts.py` versioned `dynamics_artifact/` (`model_parameters.pt, decoded_labels.pt, discovery_manifest.json` v2.0.0, sha256).
* `phaseforge/data/ingestion/state_machine.py:DataPipelineStateMachine` + `cache_manager.py:CacheManager` — fail-closed cache hash, trajectory split, zscore `RunningStatNormalizer→FrozenNormalizer` (`phaseforge/data/common/normalizer.py`), train-only SLDS fit → `phase_dynamic (T,)` + `phase_rule` duplicate, `train_label_field` selects primary `phase`.
* `phaseforge/data/common/dataset.py:StateOnlyDataset` (`sequence_length=1,stride=1`, `phase_field`, corruption opt) + `collator.py:PhaseAwareCollator` (stacks `state (B,S), action (B,A), phase (B,), task_id, trajectory_id, trajectory_position, phase_dynamic/phase_rule/phase_gt_clean`).

### 0.4 Rollout / eval

* `phaseforge/evaluations/envs/task_registry.py` — Can: `PickPlaceCan, S=23, A=7, horizon 500`; Square: `NutAssemblySquare, S=23, A=7, 500`.
* `phaseforge/evaluations/envs/robosuite_adapter.py:RobosuiteStateAdapter` — direct `env.step` (no rescale), `reset_to(set_ep_meta→reset_from_xml→set_state_from_flattened→forward)`, `extract_state` by `StateSpec`, `validate_action` contract `[-1,1]+tol`, `check_success=env._check_success`.
* `phaseforge/evaluations/rollout/runner.py:RolloutEvaluator` — `_policy_action`: `(x-mean)/std → model.get_action → validate_action`; `_run_episode` horizon loop; `_episode` row `{run_id,model,checkpoint_sha256,task,training_seed,reset_seed,episode_index,valid_episode,steps,success,timed_out,termination_reason,failure_category,exception,extra{max_phase}}`; `_summarize` + `rollout_summary.json` + `eval_results.json` (`success_rate, valid_episodes, wilson CI, per_phase_sr`). Only `max_phase` via causal labeler today — **not** the §11 full trace.
* Metrics: `phase_alignment, expert_utilization, routing_stability, specialization, task_metrics, init_diagnostics` (all present in `phaseforge/evaluations/metrics/`); outputs: `outputs_writer/{writer,episodes,results,training_summary,schema,provenance,tables,ledger,metadata}` + `evaluations/rollout/{gates_cli,report_cli}`.

### 0.5 Latest evidence (Can `phaseforge_dynamic`, same bank `310d9cfd`, seed 2026, horizon 500)

`debug_run/phaseforge_598c077`: Stage1 `val/phase_acc 0.875–0.911, balanced 0.75–0.81, loss_action ~0.030` (Can S=23, total_params 419779; Lift baseline 418K differs only by input dim S=19 vs 23); Stage2 `NMI 0.46/0.55/0.63, entropy 0.93–0.95, switch_rate 0.046–0.061, balance ~0.99, collapse 0`; rollout `seed42 0.48 (24/50, 26 timeouts), seed43 0.44 (22/50, 28 timeouts), seed44 0.58 (29/50, 21 timeouts)` mean 0.50, all failures `task_timeout`, `policy_failures 0`. Lift baseline reference: `docs/dev/DEBUG_RUN_BASELINE_COMPARISON_REPORT.md` (PhaseForge 0.64 Group A; see `final_ouput/docs/00_OVERVIEW.md,01_BASELINES.md,02_ABLATIONS.md,03_ARTIFACT_SCHEMA.md,04_COOKBOOK.md,05_ANALYSIS_CHECKLIST.md`).

**What this proves for the plan:** SLDS-dynamic routing trains (NMI>0.4, balanced, stable) yet Can rollout ≈0.5 with pure timeouts — consistent with Professor §2 (unobservable labels + unsafe averaging + no margin + no stability bias). Do not re-tune SLDS kappa first; implement §4–§8 in order.

---

## 1. Global engineering rules (apply to every work package)

1. **Memoryless invariant.** No `torch hidden`, no `history_embedding` in the primary path, no `trajectory_id/position` consumed by model `forward`/`get_action` except legacy eval ablations. New modules take `(z_t (B,Dz), y_t (B,Dy))` only. `reset()` must remain a no-op for IS path (keep `reset_sticky_ema` only for legacy `sticky` eval).
2. **CPU-only local verification.** Every WP ships `pytest tests/...` + `python -c` shape asserts on CPU (`torch.device("cpu")`, `project.device=cpu`, `data.num_workers=0`). No `cuda` import guards removed; heavy sweeps stay in cloud manifests (`experiments/*.json`). Batch sizes for local dry runs ≤32, traj ≤50 steps.
3. **No regressions by default.** All new behavior behind config flags defaulting to legacy (`discovery.method=slds`, `train.supcon.enabled=false`, `models.router.type=topk`, `models.expert.type=direct`, `train.lipschitz.enabled=false`, `eval.episodes.trace_level=minimal`). IS path opt-in via `models=is_phaseforge` or explicit overrides until Stage F.
4. **Shape contracts (Can).** `S=23` (`eef_pos 0:3, eef_quat 3:7, gripper 7:9, object 9:23`), `A=7`, `Z: Dz=128`, `K=E` (default 6, swept), `prototypes C (K,Dz)`, `gate_logits/dists (B,K)`, `weights (B,K) top-1 one-hot in primary`, `task-state y (B,Dy)` with `Dy=7` for 3D-tangent path (`3 pos + 3 rot-tangent + 1 gripper`) up to `Dy=10` for 6D-rotation path (`3 + 6 + 1`); default `Dy=7` (see WP5), `target T (B,Dy)`, `gains κ (B,Dy)>0`, `error e (B,Dy)`, `cmd u (B,Dy)→ adapter → a (B,7)∈[-1,1]`.
5. **Fail-closed.** Missing `dynamics_artifact`, all-zero `soft_mapping`/prototypes, `NaN` gains, unknown `prototype_source`, shape mismatch → `RuntimeError`, never silent fallback. Preserve `_load_state_dict_checked` semantics; extend allowed prefixes explicitly per WP.
6. **Provenance.** Every new artifact (`topo_artifact/`, `observability_report.json`, `supcon_metrics.json`, `prototype_ckpt`, `trace.jsonl`) gets sha256 manifest entry via `write_artifact_manifest` + `metadata/data_provenance.json` copy. Do not break `outputs_writer/schema.py`.

---

## 1.5 Phase grouping — implement one phase at a time (robust order)

Do not implement WPs in parallel. Finish a phase, meet its exit gate, get review, then move on. Each phase is independently testable on CPU.

| Phase | Name | WPs | Ledger Reqs | Goal (one sentence) | Exit gate before next phase |
|---|---|---|---|---|---|
| Phase 1 | Foundation & Instrumentation | WP0 + WP8-infra | R01, R08, R09, R34-contract, R35-infra, R36-infra, R37, R56, R57 | Frozen repro + `trace_level` scaffolding + artifact preservation, no architecture change | `pytest tests/runner tests/cli tests/evaluations/rollout -q` green; `phaseforge-sweep --dry-run` parses; Can 3-seed rerun reproduces §0.5 within Wilson overlap |
| Phase 2 | Observable Regimes | WP1 + WP2 | R04, R10–R16, R38, R44, R45, R51, R52 | PELT-topo discovery + mandatory observability audit; SLDS demoted to diagnostic | `observability_report.json` for `phase_topo` vs `phase_dynamic` exists; either audit passes (proceed) or negative-result stop triggers (do not proceed to Phase 3) |
| Phase 3 | Representation & Routing | WP3 + WP4 | R06, R17–R23, R39, R47-routing, R49 | SupCon encoder + Voronoi margin top-1 router; routing gains isolated with direct head fixed | `supcon_metrics.json` gate (`silhouette>0.2 (proposed), kNN>0.7`); `test_prototype_router` + `test_stage1_supcon` green; Stage C manifest dry-runs |
| Phase 4 | Action & Stability | WP5 + WP6 + WP7 | R02 (canonical config), R05, R07, R24–R33, R40, R46, R48 | Impedance `T/κ` adapter + `BC-impedance` control + Lipschitz/gain + full Stage 2 loss; `is_phaseforge.yaml` completed | `test_impedance` + `test_stage2_is_loss` green; shape smoke `Dy=7 → a (8,7)∈[-1,1]`; Stage D manifest dry-runs |
| Phase 5 | Evidence & Confirmation | WP8-full + WP9 | R34-full, R35-full, R36-full, R41–R43, R50, R53-deferred, R54–R55 | Full 22-field traces + taxonomy + A–F sequence, ablations, Can→Square→Lift gating | `test_trace` green on dummy env; all `is_stage*.json` dry-run; F-gate blocks Square pre-Can |

WP→Phase index: WP0→Phase 1; WP1–WP2→Phase 2; WP3–WP4→Phase 3; WP5–WP7→Phase 4; WP8–WP9→Phase 5 (WP8 infra starts in Phase 1, full tracing lands in Phase 5).

---

## 2. Work packages (ordered, modular, independently verifiable)

### WP0 — Reproducibility freeze + instrumentation baseline (Professor Stage A) [Phase 1]

**Goal:** Same-commit, same-bank, same-config comparisons + full-trace scaffolding before architecture changes.

* Files: `phaseforge/runner/{cli,executor,protocol}.py`, `experiments/five_task*.json`, `phaseforge/evaluations/rollout/{runner,reset_bank}.py`, `phaseforge/config/eval/rollout.yaml`, `phaseforge/cli.py`.
* Tasks:
  1. Pin `git_sha` + `config_hash` + `data_config_hash` in every run (already in `run_meta.json`/`environment.json`; verify for Can/Square dynamic runs).
  2. Freeze bank: `eval.bank={num_cases:50, seed:2026, auto_generate:false}` for confirmation runs; keep `auto_generate:true` only for first creation. Record `bank_id 310d9cfd...` for Can.
  3. Add `eval.episodes.trace_level: minimal|full` (default `minimal` = today). Implement `full` writer hook (detail WP6) but enable only in diagnostics runs to avoid disk blowup.
  4. Preserve regime artifacts: ensure `dynamics_artifact/` + `metadata/{init_routing,init_expert}.json` + `phase_thresholds.json` copied per run (extend `write_artifact_manifest` map in `cli.py:_finalize_training_run/_finalize_eval_run`).
  5. Add `scripts/dev/freeze_check.py` (CPU): asserts resolved configs equal except intended `method/seed` diff.
* Verify (CPU): `pytest tests/runner tests/cli tests/evaluations/rollout -q`; `phaseforge-sweep --dry-run`; `python -c "from phaseforge.outputs_writer.writer import parse_run_dir; ..."`; check `outputs_final/_ledger/runs.jsonl` pending→completed transitions.
* Exit: Can dynamic 3-seed rerun from same commit reproduces §0.5 NMI/entropy/SR within Wilson overlap.

### WP1 — Observation-consistent topological regime discovery (Professor §4) [Phase 2]

**Replaces** `phi_t=[x,a,Δx]` SLDS as primary router supervision. Keeps SLDS as diagnostic only.

* New files:
  * `phaseforge/data/topo/task_vars.py` — `extract_task_vars(state (B,S), spec: StateSpec) -> dict{s_eef_pos (B,3), eef_quat (B,4), gripper (B,2), obj (B,14), rel_ee_obj (B,3), gripper_aperture (B,1)}`. Slices from `data.state_keys` (never hardcode 0:3/7:9 except via `StateSpec.index_of`). Relative vectors computed in raw (unnormalized) space then z-scored with train stats. If `object` layout lacks lid/contact channels, derive proxies only from `object` dims + document which dims used per task in `metadata/topo_vars.json`.
  * `phaseforge/data/topo/pelt.py` — `run_pelt(s_t (T,Ds), penalty_beta, min_segment_len, cost='l2'|'rbf') -> boundaries (M+1,)`. Implement PELT DP `min_τ Σ C(s_{τj:τj+1})+β|τ|` with Gaussian NLL cost; pure numpy/scipy, CPU. Wrap `ruptures` if available else vendored DP (no new GPU dep). Config `beta` swept; `min_len≥5` (match `min_phase_duration`).
  * `phaseforge/data/topo/cluster.py` — `segment_features(segments)->(Nseg,Df)` (mean/var/gripper/object/relative geometry; action stats optional flag `include_action_stats=false` default for routing labels), `cluster_segments(feats)->labels (Nseg,)`, `select_K(metrics)->K*`. Support `kmeans/spherical_kmeans/agglomerative`; K sweep 3–10.
  * `phaseforge/data/topo/artifacts.py` — `save_topo_artifact(topo_artifact/: params.pt, seg_labels.pt, topo_manifest.json v1.0.0 + sha256)` mirroring `data/dynamics/artifacts.py`.
* Modified:
  * `phaseforge/data/ingestion/state_machine.py` — add `_run_topo_discovery(persist)` parallel to `_run_dynamics_discovery`; attach `phase_topo (T,) long` + `phase_topo_confidence`; keep `phase_dynamic` + `phase_rule`; new `data.topo={enabled,penalty_beta,min_segment_len,cost,num_regimes|auto,seed}` merged like `data.dynamics`. `CacheManager.compute_hash` must include `data.topo`.
  * `phaseforge/config/data/{can,square}.yaml` + `phaseforge/config/topo/*.yaml` (new group `topo_pelt_k_auto`, `topo_pelt_k6`): `topo:{enabled:false default, penalty_beta, min_len, cost, train_label_field: phase_topo}`.
  * `phaseforge/data/common/dataset.py` + `collator.py` — expose `phase_topo/phase_topo_conf` alongside `phase_dynamic/phase_rule` (extend `extras` tuple + collate).
  * `phaseforge/models/phase_moe.py:bootstrap_moe` — extend `prototype_source ∈ {rule,dynamic,topo}` (preserving existing aliases `phase/phase_rule→rule`, `phase_dynamic→dynamic`); `topo` reads `batch["phase_topo"]`; fail-closed if missing with hint (mirror dynamic branch L472–493).
* Math to implement exactly: PELT objective §4.2; K-selection `K*=argmax[Observability+ActionExplanation+Stability−Complexity]` — each term a computed scalar (see WP2 for Observability; ActionExplanation = fractional action-residual reduction vs single-head; Stability = ARI across seeds + transition-matrix L1; Complexity = BIC-like `0.5*K*Df*log N /N`).
* Verify (CPU): `pytest tests/data/test_topo_*` (new): synthetic `s_t` with known changepoints → boundaries ±2 steps; `min_len` enforced; K sweep returns finite scores; `bootstrap_moe` with `prototype_source=topo` on 200-sample CPU loader yields `(E,Dz)` prototypes finite + unit-norm. `python scripts/dev/topo_smoke.py --task can --num-traj 4` (new).

### WP2 — Mandatory observability audit (Professor §4.4) [Phase 2]

**Gate:** no `phase_*` label trains the router until it passes this audit from `x_t` alone.

* New: `phaseforge/data/topo/observability.py` — `audit_regimes(states (N,S) normalized, labels (N,), traj_ids, K) -> ObservabilityReport{macro_f1, confusion (K,K), duration_hist, occupancy, action_residual_reduction, stability_ARI}`. Internals: `LogisticRegression`/`LinearProbe (S→K)` + trajectory-aware GroupKFold (group=traj), never shuffle-split; confusion merge suggestion `pairs with F1<0.5 → merge candidate`.
* New: `scripts/analysis/observability_audit.py` (new) — CLI `python -m scripts.analysis.observability_audit --cache <hash> --labels phase_topo|phase_dynamic|phase_rule --out observability_report.json`; exit code 1 if `macro_f1 < threshold (proposed default 0.6, not spec-fixed)` or `min_occupancy<0.01 (proposed)` or `confused pair` without `--allow-aliased`.
* Modified: `state_machine.py:_run_*_discovery` — call audit, persist `observability_report.json` into `topo_artifact/` and `dynamics_artifact/`; if `enforce_observability=true` (default) and audit fails → raise (fail-closed) unless `allow_aliased` merge applied. `diagnostics.py` — add `audit_*` fields to `DiscoveryQualityReport` (backward-compat defaults).
* Verify (CPU): unit test with aliased synthetic labels (two regimes identical `x`) → audit fails + suggests merge; separable synthetic → passes. Check Can `phase_dynamic` audit reproduces Professor §2.1 concern (expect low F1 on contact-aliased pairs) vs `phase_topo` higher.

### WP3 — Contrastive regime-aligned encoder Stage 1 (Professor §5) [Phase 3]

**Replaces** pure `CE` shaping with `L_Stage1 = L_action + λ_sc*L_SupCon`.

* Modified: `phaseforge/models/components/encoder.py` — add `normalize_output: bool=false` flag; when SupCon enabled, `forward` returns L2-normalized `z/‖z‖` (keep unnormalized path for legacy). Do not change `latent_dim=128`.
* New: `phaseforge/trains/losses/supcon.py` — `supcon_loss(z (B,Dz) normalized, labels (B,), tau=0.07) = 1/|B| Σ_i -1/|P(i)| Σ_{p∈P(i)} log[exp(z_i·z_p/τ)/Σ_{a≠i} exp(z_i·z_a/τ)]`. Masked, handles singleton `P(i)` (skip, avoid div0), `tau` from `train.supcon.temperature`.
* Modified: `phaseforge/trains/loops/stage1_loop.py` — read `train.supcon={enabled:false, lambda_sc:1.0, temperature:0.07, label_field:phase_topo, warm_action_mse:true}`; compute `L_action=MSE(g_φ(z),a)` (keep direct head for warm-start; impedance adapter variant gated by WP4 flag `action_adapter=direct|impedance`), `L_total=L_action+λ_sc*L_SupCon`; log `loss_supcon, intra_dist, inter_dist`; keep `_PhaseAccumulator` for CE baseline when `lambda_phase>0` (allow `lambda_phase=0 + supcon` pure-contrastive runs).
* Modified: `phaseforge/config/train/stage1.yaml` — add `supcon` block (defaults disabled). New model config `phaseforge/config/models/is_phaseforge.yaml` sets `encoder.normalize_output=true`, `train.supcon.enabled=true`.
* Acceptance (CPU + cloud val): UMAP/t-SNE script `scripts/analysis/latent_cluster.py` (CPU on cached latents); metrics `kNN acc, intra/inter dist, silhouette` in `supcon_metrics.json`; gate `silhouette>0.2 + kNN>0.7` before Stage2 (warn, not hard fail, except `--strict`).
* Verify (CPU): `pytest tests/trains/test_stage1_supcon.py` (new): random `(B=16,Dz=8)` with 2 regimes → loss finite, gradient flows to encoder, `tau` scaling correct, singleton batch doesn't NaN; `test_lambda_schedule` still passes.

### WP4 — Large-margin prototype router + top-1 hard routing (Professor §6) [Phase 3]

**Replaces** unconstrained `softmax MLP TopKRouter` with `Voronoi` prototype router, margin-trained, `top-1` deployment.

* New: `phaseforge/models/components/prototype_router.py:PrototypeRouter(nn.Module)` —
  * Params: `prototypes C (K,Dz)` (`nn.Parameter`, init from regime centroids §WP1, EMA or learned), `margin m (proposed default 0.5; spec gives no numeric value)`, `top_k=1` primary (allow `top_k=2` only for ablation flag).
  * `forward(z (B,Dz) normalized, labels (B,) opt) -> {dists (B,K)=‖z−c_k‖2, k_star (B,)=argmin, weights (B,1)=ones, gate_logits (B,K)=−dists, margin_loss scalar}`. Keep `balance_loss` (tiny, `λ_bal≪1`, default 1e-4) + `router_margin` diagnostic. No noise, no history, no `noise_linear`.
  * Loss exactly: `L_margin=1/|B| Σ_i Σ_{j≠y_i} max(0, m−(d_ij−d_iy_i))`.
  * Implement `ema_update(z,labels,decay=0.99)` alternative to gradient (config `prototypes.update: learned|ema|frozen_centroid`).
* Modified: `phaseforge/models/components/moe_layer.py` — accept `router: TopKRouter|PrototypeRouter` (duck-type `RouterOutput`); hard dispatch `combined=expert_{k*}(z)` (no averaging) when `K_sel=1`; preserve `uniform/sticky/oracle` overrides only for legacy `TopKRouter` (raise if requested on prototype path except `uniform` ablation).
* Modified: `phaseforge/models/phase_moe.py` — add `router_type` switch; `bootstrap_moe` for prototype: `C←compute_hierarchical_phase_prototypes(latents,phase_topo,K,K)` (reuse `clustering.py`), copy into `router.prototypes`, zero grad option; persist `metadata/init_prototypes.json` (+sha).
* Config: `config/models/is_phaseforge.yaml` — `router:{_target_:...PrototypeRouter, num_experts:6, top_k:1, margin:0.5 (proposed), prototypes_update:learned, balance_coeff:0.0001 (proposed, spec λ_bal≪1)}`; legacy `phaseforge.yaml` untouched. `train.stage2.yaml` adds `margin:{lambda_margin:1.0 (proposed), margin:0.5 (proposed)}` (inert unless prototype router present).
* Verify (CPU): `pytest tests/models/test_prototype_router.py`: `dists` shape, `argmin` correctness, margin 0 when `d_y+m≤d_j`, positive otherwise; boundary test: two close `z` on opposite sides don't flap when margin-trained (check `router margin` metric); `MoELayer` top-1 output equals selected expert output bit-exact.

### WP5 — Impedance-parameterized expert heads + action adapter (Professor §7, central fix) [Phase 4]

**Replaces** `tanh(MLP(z))` direct 7D averaging with per-expert `target+gains → task-error feedback → tanh(u/s)`.

* New: `phaseforge/models/components/task_state.py` — `psi(x (B,S), spec) -> y (B,Dy)`. Can default `y=[eef_pos 3, rot-tangent 3, gripper 1]` with `Dy=7`; 6D-rotation variant `Dy=10` (`3 + 6 + 1`); 2-gripper-dim variant `Dy=8/11`. Implement `RotErr` as 6D-rotation continuation then 3D tangent: store `eef_quat`, convert target 6D→matrix→quat-error→axis-angle (avoid raw quat subtraction). Expose `TASK_STATE_DIMS` per task in `task_registry.py` (add `task_state_keys/dims`). Object `x` still feeds encoder/expert via `z`; feedback error only in task space.
* New: `phaseforge/models/components/impedance_expert.py:ImpedanceExpert(nn.Module)` — `shared trunk Linear(Dz→256)+GELU → target_head Linear(256→Dy) + gain_head Linear(256→Dy)→softplus+clamp[κ_min,κ_max] (defaults 0.1–5.0, config)`. Forward `(z (B,Dz), y (B,Dy)) -> (T (B,Dy), κ (B,Dy))`.
* New: `phaseforge/models/components/action_adapter.py` — `task_error(T,y)->e (B,Dy)` (pos `T−y`, rot tangent, gripper `T−y`), `u=κ⊙e (B,Dy)`, `a=tanh(u/s)` with `scale s (default 1.0, per-dim `action_scale (Dy,)` config) + `gripper_map` to 7D `robosuite_environment_action` (pos 3 + rot 3 + gripper 1; pad/route per `StateSpec`+`action_dim`). Effective blend derivation for top-2 ablation: `K_eff=Σ w_i K_i, T_eff=K_eff^{−1} Σ w_i K_i T_i` — implement `blend_impedance(Ts,Ks,w)` helper (used only when `top_k=2` ablation).
* Modified: `MoELayer` + `PhaseBootstrappedMoE.forward/get_action` — new signature `forward(batch)` extracts `y=psi(state)` (raw or normalized? use raw for geometry + document; normalize only `x` for encoder), routes `z→k*`, calls `expert_{k*}(z,y)→a via adapter`; return `aux {target,gains,task_error,pre_clip_u}` for tracing. Keep direct path when `expert.type=direct`.
* Modified: `BehaviorCloningModel` — add `BCImpedance` variant (`models/baselines/bc_impedance.py:BCImpedanceModel`) sharing same `psi+adapter` + single `ImpedanceExpert(K=1)` so `BC-impedance` control is action-matched (Professor §7.5). Register in `config/models/bc_impedance.yaml`.
* Config matrix (Professor §7.5 table): `bc_direct (existing config/models/baselines/bc.yaml), bc_impedance (new), phaseforge_direct (existing), phaseforge_impedance (is_phaseforge with supcon+margin off), is_full (all on)`. Each a Hydra `models=` + `train=` override; document in `experiments/is_phaseforge_matrix.json` (new).
* Verify (CPU): `pytest tests/models/test_impedance.py`: shapes `T,κ,e,u (B,Dy)`, `κ>0`, `a∈[-1,1]`, `tanh` saturation guarded; `blend_impedance` equals direct weighted `u` formulation; `BCImpedance` vs `BC` same encoder different head; adapter with extreme `e` clips without NaN.

### WP6 — Local contraction / Lipschitz regularization (Professor §8) [Phase 4]

**Bias** (not proof): `‖∂T_k/∂y‖≤ρ<1`.

* New: `phaseforge/trains/losses/lipschitz.py` — `lip_penalty(T_k(x_i),T_k(x_j), y_i,y_j, rho=0.8 (spec e.g. 0.7/0.8), eps=1e-6) = mean max(0, ‖ΔT‖/(‖Δy‖+eps) − ρ)^2`. Pair sampling: same-regime or kNN neighborhood (`num_pairs=256` proposed, CPU kNN via torch.cdist).
* Modified: `stage2_loop.py` — add `train.lipschitz={enabled:false, lambda_lip:0.1 (proposed), rho:0.8, num_pairs:256, same_regime_only:true}` + `train.gain_reg={enabled:true when impedance, lambda_gain:1e-4 (proposed), kappa_nominal:1.0}` (`L_gain=E‖κ−κ_nom‖²` or bound penalty). Total per §9: `L=L_action+λ_margin L_margin+λ_lip L_lip+λ_gain L_gain+λ_bal L_bal` with `λ_bal≪1` (spec; proposed IS default 1e-4 vs legacy 0.01). Log each term; monitor `action vs lip` tradeoff curve.
* Caveat text (required in code docstring + final report): “contraction term provides local stabilizing bias + diagnostic; does not certify global closed-loop stability in robosuite.”
* Verify (CPU): synthetic `T_k` linear with slope 0.5 vs 1.5 → penalty 0 vs >0; gradient check; `stage2` 2-epoch CPU dry run loss decreases without NaN.

### WP7 — Full Stage 2 objective wiring (Professor §9) [Phase 4]

Single place: `phaseforge/trains/loops/stage2_loop.py:_compute_loss` (+ `config/train/stage2.yaml`).

```text
L_action  = ‖a_pred − a_demo‖²            (after adapter when impedance)
L_margin  = Σ_{j≠y} max(0, m−(d_j−d_y))   (prototype router only; 0 legacy)
L_lip     = max(0, Lip(T_k)−ρ)²           (impedance only; 0 direct)
L_gain    = E‖κ−κ_nom‖²                   (impedance only)
L_bal     = E*Σ f_i p_i                   (λ_bal≪1, default 1e-4 IS / 0.01 legacy)
L_total   = L_action + λ_margin L_margin + λ_lip L_lip + λ_gain L_gain + λ_bal L_bal
```

* Keep `sticky/teacher_kl` terms inert (coeff 0) on IS path; do not delete (legacy ablations need them). Add `train.is_weights={lambda_margin,lambda_lip,lambda_gain,lambda_bal}` with validation `≥0`, `lambda_bal<1e-3` warn if larger on IS.
* Verify: `pytest tests/trains/test_stage2_is_loss.py` asserts each λ=0 disables term bit-exact; ok.

### WP8 — Deployment contract + rollout tracing (Professor §10–§11) [Phase 1 infra → Phase 5 full]

* Deployment (`phaseforge/models/phase_moe.py:get_action`, `evaluations/rollout/runner.py:_policy_action`): enforce 10-step §10 pipeline `receive x → normalize → z=E(x) → d_k=‖z−c_k‖ → k*=argmin → (T,κ) → e → u=κ⊙e → a=tanh(u/s) → send`. Assert no `trajectory_id`, no `soft_mapping`, no `sticky_ema`, no `phase` input. Add `model.deployment_contract() -> dict` returning `{memoryless:true, top_k, router_type, expert_type}` checked by runner (fail if `router_mode!=learned` on primary runs).
* Tracing (`runner.py:RolloutEvaluator` + `outputs_writer/episodes.py` schema):
  * Add `eval.episodes.trace_level=minimal|full`, `trace_every_n_steps` (default 1 for full, subsample Jacobian every 10).
  * `full` per-step record (22 fields per spec §11): `{episode_id,case_id,timestep, success/timeout/failure_reason, raw_obs_summary{norm, eef_pos}, normalized_state_norm, y_t (Dy,), z_norm, dists (K,), selected_expert, top2_expert, router_margin (d_2nd−d_1st), router_entropy (if soft logged), T_k (Dy,), κ (Dy,), e (Dy,), pre_clip_u (Dy,), final_a (7,), nearest_train_dist/OOD_score, expert_disagreement (‖a_top1−a_top2‖, top-2 eval only), jacobian_diag/lip_sample, termination_reason}`.
  * Implement `TraceWriter` (`phaseforge/evaluations/rollout/trace.py` new, `trace.jsonl` per run, schema-validated, sha in manifest). `minimal` keeps today’s `episodes.jsonl` shape (backward-compat).
  * Failure taxonomy classifier `scripts/analysis/classify_failures.py`: `routing_ambiguity (margin<τ + entropy high), expert_conflict (disagreement large), OOD_drift (nearest_train_dist>τ), action_saturation (clip frac>0.8), gain_collapse (κ≈0), target_chasing (‖T−y‖ growing), reset_geometry (same case fails across methods), controller_limit (valid a but Δx≈0)`.
* Verify (CPU): `pytest tests/evaluations/rollout/test_trace.py` with dummy env (no robosuite): trace schema validates, `minimal` unchanged; `python scripts/analysis/classify_failures.py --trace <cpu fixture>` outputs taxonomy counts.

### WP9 — Experiment sequence + baselines (Professor §12, §7.5, §15) [Phase 5; Stage C used in Phase 3, Stage D in Phase 4]

Implement as Hydra sweep manifests (cloud) + CPU smoke equivalents. Do not run heavy sweeps locally.

| Stage | Manifest / command | Cells |
|---|---|---|
| A Repro | `experiments/is_stageA_repro.json` | Same-commit rerun `bc_direct, phaseforge_dynamic` Can 3-seed, frozen bank |
| B Discovery | `experiments/is_stageB_discovery.json` + `scripts/analysis/observability_audit.py` | `slds_dynamic vs pelt_topo vs merged vs K{3,4,6,8}`; metrics macro-F1, confusion, occupancy, duration, transition matrix, residual reduction, ARI |
| C Routing-only | `experiments/is_stageC_routing.json` | `BC-direct; PF-current; PF-topo+CE; PF-topo+SupCon; +hard top-1; +margin` (direct head fixed) |
| D Impedance | `experiments/is_stageD_impedance.json` | `BC-direct; BC-impedance; PF-direct+SupCon+margin; PF-impedance+SupCon+margin; full IS+lip` |
| E Ablations | `experiments/is_stageE_ablations.json` | One-var-off: supcon, margin, top1vs2, impedance, lip, K, balance coeff, expert init; 1 seed screen → 3 seeds survivors |
| F Confirm | `experiments/is_stageF_confirm.json` | Val-bank checkpoint select → final Can bank once → Square if Can decisive → Lift regression |

* Stop rule B: if no `phase_*` passes observability audit → report negative result “memoryless routing insufficient under current state contract,” do not proceed to C.
* Temporal BC diagnostic (Professor Q4): DEFERRED by design — no temporal
  baseline model exists in-repo (verified: `models/baselines/` holds 8
  files, none temporal), so shipping `is_temporal_probe.json` would produce
  an unrunnable manifest that fails closed at sweep time. When a temporal
  baseline lands, the probe is one single-cell Can manifest (temporal BC,
  3 seeds, rollout), run once as a negative-result diagnostic and never in
  the primary table.
* Verify (CPU): `phaseforge-sweep --dry-run` parses every manifest; `pytest tests/runner/test_runner.py -q`.

---

## 3. File-by-file change index (planned)

| # | File | Change |
|---|---|---|
| 1 | `phaseforge/data/topo/task_vars.py` (new) | `psi` + task-var extraction |
| 2 | `phaseforge/data/topo/pelt.py` (new) | PELT segmentation |
| 3 | `phaseforge/data/topo/cluster.py` (new) | Segment features, clustering, K-select |
| 4 | `phaseforge/data/topo/artifacts.py` (new) | `topo_artifact/` versioned persistence |
| 5 | `phaseforge/data/topo/observability.py` (new) | Audit probe + merge suggestions |
| 6 | `phaseforge/data/dynamics/diagnostics.py` | Add audit fields (compat) |
| 7 | `phaseforge/data/ingestion/state_machine.py` | `_run_topo_discovery`, `data.topo` group, hash include |
| 8 | `phaseforge/data/ingestion/cache_manager.py` | Persist `topo_artifact/` + `observability_report.json` |
| 9 | `phaseforge/data/common/dataset.py` | `phase_topo` extras |
| 10 | `phaseforge/data/common/collator.py` | Collate `phase_topo` |
| 11 | `phaseforge/trains/losses/supcon.py` (new) | SupCon loss |
| 12 | `phaseforge/trains/loops/stage1_loop.py` | `L_action+λ_sc L_SupCon`, normalized z |
| 13 | `phaseforge/models/components/encoder.py` | `normalize_output` flag |
| 14 | `phaseforge/models/components/prototype_router.py` (new) | Voronoi + margin loss |
| 15 | `phaseforge/models/components/moe_layer.py` | Hard top-1 dispatch, impedance branch |
| 16 | `phaseforge/models/phase_moe.py` | `router_type`, `prototype_source=topo`, `deployment_contract()` |
| 17 | `phaseforge/models/components/task_state.py` (new) | `psi(x)` per task |
| 18 | `phaseforge/models/components/impedance_expert.py` (new) | `T,κ` heads |
| 19 | `phaseforge/models/components/action_adapter.py` (new) | `e→u→tanh→a`, `blend_impedance` |
| 20 | `phaseforge/models/baselines/bc_impedance.py` (new) | Action-matched BC control |
| 21 | `phaseforge/trains/losses/lipschitz.py` (new) | `L_lip` |
| 22 | `phaseforge/trains/loops/stage2_loop.py` | Full §9 objective |
| 23 | `phaseforge/evaluations/rollout/trace.py` (new) | `TraceWriter` full schema |
| 24 | `phaseforge/evaluations/rollout/runner.py` | `trace_level`, contract check |
| 25 | `phaseforge/outputs_writer/episodes.py` | Unchanged (`episodes.jsonl` schema frozen; trace schema lives in row 23) |
| 26 | `phaseforge/evaluations/envs/task_registry.py` | Unchanged (task-state dims live in `task_state.py`, validated at runtime) |
| 27 | `phaseforge/config/topo/*.yaml` (new) | PELT configs |
| 28 | `phaseforge/config/models/is_phaseforge.yaml` (new) | IS canonical (supcon+margin+impedance+lip, top-1) |
| 29 | `phaseforge/config/models/bc_impedance.yaml` (new) | BC-impedance control |
| 29b | `phaseforge/config/models/phaseforge_prototype.yaml` (new) + `normalize_output`/`prototype_source` defaults in `phaseforge.yaml`, `phaseforge_dynamic.yaml` | Prototype-router + direct-head control; additive defaults keep legacy paths bit-identical |
| 30 | `phaseforge/config/train/stage1.yaml` | `supcon` block |
| 31 | `phaseforge/config/train/stage2.yaml` | Flat `margin`/`lipschitz`/`gain_reg` blocks (kept flat for Phase 3 key compat) |
| 32 | `phaseforge/config/eval/rollout.yaml` | `trace_level` |
| 33 | `phaseforge/config/data/can.yaml, square.yaml` | Untouched (absent `topo` block = disabled; group files carry everything) |
| 34 | `experiments/is_stage{A,B,C,D,E,F}.json` + `experiments/is_phaseforge_matrix.json` (all new, 7 manifests; temporal probe deferred, see WP9) | Sweep manifests |
| 35 | `scripts/analysis/{observability_audit,latent_cluster,classify_failures,confirm_gate}.py` (new) | Diagnostics CLIs |
| 36 | `tests/**` | Per-WP CPU tests (see §4) |

No change (diagnostic-only, frozen): `StickySLDS`, `RuleBasedPhaseLabeler`, legacy `TopKRouter`, `soft_mapping`, `teacher_routing`, `use_history/sticky` eval modes (kept for ablations).

---

## 4. CPU verification matrix (local, no GPU)

| Check | Command | Gate |
|---|---|---|
| Lint+types | `ruff check phaseforge tests; mypy phaseforge/models/components/prototype_router.py phaseforge/models/components/action_adapter.py` | 0 errors |
| Unit all | `pytest tests/ -q` | green; new tests: `test_topo_{pelt,cluster,observability}`, `test_stage1_supcon`, `test_prototype_router`, `test_impedance`, `test_stage2_is_loss`, `test_trace` |
| Shape smoke | `python scripts/dev/is_shape_smoke.py --device cpu` (new, B=8,S=23,A=7,Dz=128,K=6,Dy=7): asserts `z (8,128)‖=1`, `dists (8,6)`, `k* (8,)`, `T,κ,e,u (8,7)`, `a (8,7)∈[-1,1]` | asserts pass |
| Data smoke | `python scripts/dev/topo_smoke.py --task can --num-traj 4 --device cpu` (new) | PELT+cluster+audit JSON written |
| Train dry | `phaseforge-train data=can models=is_phaseforge train=stage1 project.device=cpu train.epochs=2 data.batch_size=32` then `train=stage2` 2ep | loss finite, no CUDA call |
| Trace dry | `pytest tests/evaluations/rollout/test_trace.py -q` + dummy-env rollout 5 steps | schema validates |
| Freeze | `python scripts/dev/freeze_check.py` (new) | same-bank/hash asserts |

Cloud-only (never local): full 100/200-epoch runs, 50-episode robosuite rollouts, K-sweeps, 3-seed confirmations.

---

## 5. Risks → active mitigations (map to code)

* PELT over-segments → `penalty_beta↑`, `min_len`, prototype merge, K-penalty (WP1 `cluster.py:select_K`).
* Aliased regimes → audit merge, add observable `object` features, else negative-result report (WP2 exit).
* Adapter vs robosuite OSC mismatch → command-shaping over 7D (not torque), bounded `κ`, `BC-impedance` control, fallback `affine` head flag `expert.fallback=affine` (WP5).
* Hard-route discontinuity → margin training, boundaries in low-density (check `dists` histogram), narrow-margin top-2 impedance blend ablation only, trace `‖Δa‖` at switches (WP4+WP8).
* Lip vs fast motions → `λ_lip↓`, per-regime `ρ_k`, monitor `L_action vs L_lip` (WP6).
* SupCon collapse → keep `L_action` on, `tau≈0.07–0.2`, monitor val MSE + downstream success not only silhouette (WP3).

---

## 6. Definition of done (per phase — review each gate before proceeding)

1. Phase 1: `IMPLEMENTATION_PLAN.md` + `ledger.md` reviewed; WP0 infra green + dry-run artifacts.
2. Phase 2: topo observability pass (or documented negative-result stop); no routing/action changes yet.
3. Phase 3: SupCon silhouette/kNN gate + margin reduces chattering in traces (Stage C evidence).
4. Phase 4: impedance beats direct under action-matched controls; top-1 vs top-2 recorded; full Stage 2 loss wired (Stage D evidence).
5. Phase 5: val-bank-selected checkpoint evaluated once on final Can bank; Square only if Can decisive; Lift regression. Report states contraction caveat verbatim; no oracle/sticky/history in primary numbers.

**Explicit non-goals:** temporal MoE as primary, SLDS re-tuning as fix, Square/Transport expansion before Can, GPU-dependent local scripts, changing `[-1,1]` action contract.
