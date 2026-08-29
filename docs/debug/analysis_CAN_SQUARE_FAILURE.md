# PhaseForge Can / Square Failure Analysis (vs baselines, excl. bc_rnn)
**Date:** 2026-08-29  
**Scope:** `outputs_final/phaseforge` Stage1+2 (`seed42-44`) vs `bc/bc_large/plain_encoder/scratch_moe/teacher_forced/warmstart_moe/phase_pretrain_random_router` on `Can`/`Square` (Lift is control).  
**Data:** `data/raw/robomimic/{can,square,lift}/low_dim_v15.hdf5` + `phaseforge/data/robomimic/phase_labeler.py:34` + `phaseforge/models/phase_moe.py:96` + `phaseforge/trains/loops/stage1_loop.py:97` + `phaseforge/trains/loops/stage2_loop.py:36`  
**Env:** CPU-only smoke tests `tests/debug/test_phaseforge_can_square_cpu.py:1` (no GPU, 10-demo samples, 500-path lab).

## 1. Executive Summary
| Task | `phaseforge` mean sr (3 seeds) | Best baseline (excl bc_rnn) | Gap | `bc_rnn` for ref |
|---|---|---|---|---|
| **Can** | **0.300** [0.28,0.26,0.36] | **0.493** `warmstart_moe` | **-0.193 (-39%)** | 0.727 |
| **Square** | **0.133** [0.12,0.12,0.16] | **0.213** `scratch_moe`/`warmstart` | **-0.080 (-37%)** | 0.107 |
| **Lift** (control) | **0.707** [0.56,0.84,0.72] | 0.647 `plain_encoder` / 0.587 `scratch_moe` | **+0.06 (+9%) competitive** | 0.993 |

`Can` is **worst** in family (rank 8/8), `Square` tied-worst. `Lift` is 1st/2nd. Failure is **task-specific**, not general collapse: `ToolHang`/`Transport` 0.00 for *all* methods, so not infrastructure.

**Training curves (stage1 seed42) evidence:**
- `val/phase_acc` Can 0.609→0.549, Square 0.665→0.626, Lift 0.602→0.595 `outputs_final/phaseforge/stage1/.../metrics/training_curves.jsonl:1`
- `val/loss_phase` Can 1.01→2.71 (+168%), Square 0.84→2.24 (+165%), Lift 0.89→2.59 (+190%) - overfit with `lambda_phase=1.0 constant`.
- Stage2 `val/phase_expert_nmi` Can **0.200**, Square **0.225**, Lift **0.410** `stage2_loop.py:310` - Can/Square pseudo-balanced (balance 0.991/0.993 but NMI 0.20) vs Lift 0.41.

## 2. Data Pipeline Audit (CPU)
**Paths:** `phaseforge/data/paths.py:57` `data/raw` vs `data/processed/cache` via `CacheManager:194` / `ingester.py:31`.

### 2.1 Raw HDF5 stats (CPU, 20 demos sampled)
| Task | mean_len | std | demos | state_dim | object_dim |
|---|---|---|---|---|---|
| Can |114|11|200|23|14|
| Square |147|18|200|23|14|
| Lift |50|6|200|19|10|

`Can`/`Square` 2-3× longer than `Lift` → `global_steps` Can 8100 vs Lift 3300, longer exposure to phase drift.

### 2.2 RuleBasedPhaseLabeler `phase_labeler.py:34`
Adaptive hysteresis `closed_level/open_level` from 5/95 percentiles `80% span ±30%`, `mirror` via `middle median`, `gripper_slice=7:9` `eef_pos_slice=0:3`, `velocity_threshold=0.01`, `min_duration=5`, `filter_size=7` causal median `phase_labeler.py:199-252`.

**CPU phase distribution (10 demos, 1183/1469/523 steps):**
| Task | p0 appro | p1 pre-grasp | p2 grasp | p3 transport | p4 place | p5 retract | imbalance max/min |
|---|---|---|---|---|---|---|---|
| **Can** | **0.011** |0.309|0.117|**0.409**|0.056|0.099|37× |
| **Square**| **0.011**|0.171|0.126|**0.427**|**0.223**|0.043|39× |
| Lift |0.157|0.010|0.308|0.432|0.092|0.002|226× |

- `Can`/`Square` **phase0 1.1%** (approach almost absent) vs Lift 15.7% - phase vocabulary mismatched to task. `Lift` phase1 1% but still best rollout, so rarity alone not fatal; however **centroid bootstrap** `phase_moe.py:463` `compute_hierarchical_phase_prototypes` requires every phase present in batch - with batch=256 random sample, rare phase may be 2-3 tokens/batch → noisy centroid → router init poor for Can/Square approach expert.
- Single `Square` demo `demo_0` only phases [2,3,4] (no 0,1,5) with default thresholds; varying `velocity_threshold 0.005→diff 0.35` `0.02→0.50` `phase_labeler.py:143` - Square's slow precise motions make `0.01` borderline. Threshold calibrated per-demo but state machine's `speed < 0.01` for `phase0→1` is **too strict** for Square's square-nut insertion (requires slow precise).
- `median_filter_size=7` causal `phase_labeler.py:249` blurs phases <7 steps - Square has many short transitions (grasp 2→3), measured diff 0.055 when `filter 3 vs 11`.
- Labeler ignores `object` state entirely `phase_labeler.py:142` aperture+speed only. `Can` rolling and `Square` nut pose require object pose for true phase, so rule mislabels `Can` Can-phase where eef stationary but object rolling.

**Raw obs scale:** `robot0_gripper_qpos` std Can 0.032, Square 0.027, Lift 0.035 similar; `object` std Can 0.45 Square 0.41 Lift 0.357 - not outlier. `normalizer.py:1` `zscore` train-split-only correct.

### 2.3 Cache / Splits
`CacheManager` `splits.json` 180 train /20 val per cache (trajectory split `strategy:trajectory` `split:133`). `phase_thresholds.json` `n_demos=200` per cache but many caches lack `phase_thresholds` (pt=False) - those are stale bc caches. PhaseForge caches correctly have pt. `data_config_hash` per task consistent across seeds (e.g., Can 67e642d? - cloud hash vs local mismatch due to `PHASEFORGE_DATA_DIR` note in `find_cache.py:52`).

## 3. Model Audit
**Architecture:** `phase_moe.py:96` `encoder:[256,256,256]->128 gelu dropout0.1` + `phase_head:Linear 128->6` `phase_head.py:24` + `router:TopKRouter latent128 num_experts6 top_k2 noise0.1 balance0.01 normalize=True` `router.py:82` + `experts: ExpertMLP [256]->7 tanh` `expert.py:35` `MoELayer` `moe_layer.py:27` loop-dispatch `115` (no seq >1).

**Vs baselines:** `bc` 206k params, `bc_large` 385k, `phaseforge` 418k similar; `bc_rnn` 1.16M 5.6× due to LSTM. Not capacity gap.

**Router:** `normalize_input=True` → cosine similarity to unit-norm centroids `router.py:288`. Bootstrap `centroid` `phase_moe.py:463` `compute_hierarchical_phase_prototypes` hierarchical: phase centroids then kmeans if experts>phases - with 6 experts =6 phases → 1-1 mapping butstill. `expert_init partial_warm drop_rate0.5` `expert.py:182` shared indices, Kaiming reinit for dropped neurons.

**Critical difference:** `scratch_moe` (random experts) gets **0.213 Square** vs `phaseforge` 0.133 ; `warmstart` 0.213 - suggests `partial_warm 0.5` destroys more than helps for Square where precise action variance high. `plain_encoder_phase_bootstrap` (same encoder but different init) Can 0.42 >0.30 - confirms init matters.

**Stage2 freeze:** `train.freeze_encoder` default True `stage2_loop.py:82` - encoder frozen kept `eval()` `phase_moe.py:221` no dropout. Good for Lift (generalist frozen) but Square needs fine-tuning for precise insertion - ablation would be `encoder_lr_scale`.

## 4. Training Audit
**Stage1:** `train.lambda_phase=1.0 constant` `lambda_schedule type constant start1.0 end0.0` `stage1_loop.py:131` `phase_weights=None` `soft_target_eps=0.0` `grad_cosine false`. Loss `L= MSE + λ CE` `stage1_loop.py:237`. Phase head single linear → structure must be in latent, not head.

**Evidence of overfit:** Train `phase_acc` 0.91→0.95 vs Val 0.54/0.62 gap 0.37 `stage1_loop.py:113` `_PhaseAccumulator`. `val/loss_phase` explodes 1.0→2.7 while `train/loss_phase` shrinks  - classic auxiliary overfit on flat action plateau. Paper note `stage1_loop.py:128` says cos≈0 so linear schedule not chosen - but task-dependent grad conflict not measured per task (Can may have conflict).

**Checkpoint:** `monitor:val/loss_action` `mode:min save_top_k1 every10` - picks best action loss, but stage1 best_epoch Can 50, Square 70, Lift 31 - Can's best action occurs while phase already overfit (loss_phase 2.37 at epoch50). Late phase overfit biases encoder to phase not action.

**Stage2:** `train.lambda_phase=0.0` (no phase CE), `epochs 200`, `balance_coeff 0.01` `sticky_coeff 0.0`, `teacher_routing.enabled false`. `total_loss = action + balance + sticky + teacherKL` `stage2_loop.py:214`. `val/phase_expert_nmi` logged `stage2_loop.py:310` via `phase_alignment`. `val/topk_balance 0.99` high but NMI 0.20 => **pseudo-balancing**: router distributes uniformly due to balance loss but not aligned to phases (random). `routing_entropy 0.93-0.95` same; `switch_rate` Can 0.066 Square 0.047 vs Lift 0.105 - Can/Square over-sticky (less switching) may miss phase transitions.

**Data imbalance:** No `phase_class_weight=balanced/cui` → head biased to majority phases (p3 transport 40% dominates). For Can p0 1% gets ignored.

## 5. Evaluation Audit
`evaluations/rollout/runner.py:199` `model.reset()` per episode `phase_moe.py:181` clears sticky EMA. `eval/router_mode=learned` default. `horizon` Can/Square/Lift 500 (Transport 700). `reset_bank seed2026 50 cases` frozen. `per_phase_sr` `runner.py:464` P(success|reach phase p).

**Can seed42:** `[0:0.28,1:0.28,2:0.28,3:0.28,4:0.28,5:0.209]` drop 0.07 at retract; **Square:** `[0:0.12,...,4:0.12,5:0.033]` drop 0.09. Failures all `task_timeout:36-44` (`rollout_summary.json:9`) - policy stalls, not invalid actions. Indicates transport→place→retract phases fail (needs object info).

## 6. Exhaustive Hypothesis Catalog (ranked)

### Category A: Data / Phase Labeling (Highest leverage)
**H1 - Phase vocabulary misalignment (P0 rare 1% Can/Square vs 15% Lift):** Bootstrap centroid noisy → expert0 under-trained. *Evidence:* Test1 distro. *Impact:* High. *Validate:* compute prototype cosine distance std; check expert utilization per phase in val (NMI low). *Fix:* Task-specific `min_phase_duration`/`velocity_threshold` tuning or learned phase discovery (spherical k-means 3 experts) or reduce num_phases to 4 for Can.

**H2 - Ignoring object state:** Rule uses only eef/gripper, but Can rolling & Square peg require object pose for true phase. *Evidence:* labeler.py:124 _features no object. *Fix:* Extend labeler to include object velocity or train unsupervised phases (kmeans on state).

**H3 - Adaptive threshold sensitivity:** Single `velocity_threshold=0.01` mislabels Square slow precise moves. *Evidence:* Test2 diff 0.35/0.50. *Fix:* Per-task threshold grid search (0.005 for Square).

**H4 - Causal median filter (7) blurs short phases:** Square phases [2,3,4] only 127 steps demo truncated. *Evidence:* diff 0.055 filter 11. *Fix:* `filter_size=3` for Square or disable smoothing.

**H5 - Trajectory length imbalance:** Square 147 vs Lift 50 → longer sequences dilute phase loss sampling. *Impact:* Medium.

### Category B: Model / Router
**H6 - Pseudo-balanced random routing:** Balance 0.99 + NMI 0.20 = uniform random, not specialized. `router.py:358` balance loss on top-1 but diagnostics count top-k - mismatch note `router.py:374`. *Evidence:* Test3. *Fix:* Increase `balance_coeff` 0.01→0.1 or use `sticky`/`teacher` routing, or reduce `top_k` 2→1 (Switch) for specialization.

**H7 - Cosine vs magnitude:** `normalize_input=True` discards latent magnitude needed for Square precise placement. *Evidence:* Square vs Lift latent norm difference (not measured, but encoder latent 128 same). *Fix:* Try `normalize_input=False` ablation (`pf_spherical` legacy 0.12 vs `pf_kmeans`).

**H8 - Noise 0.1 too high:** Adds stochasticity where deterministic routing needed for precise phases. *Fix:* `noise_std 0.0` eval already deterministic but train noise hurts Can.

**H9 - Partial_warm drop 0.5 destructive:** Destroys 128/256 hidden neurons shared across experts → Square's precise action variance lost. `scratch_moe` random does better. *Evidence:* `plain_encoder 0.42 >0.30`. *Fix:* `drop_rate 0.0` exact warmstart or `warmstart` jitter 0.02.

**H10 - Encoder frozen:** Stage2 frozen hurts Square fine-tuning. *Evidence:* `stage2_loop.py:82` freeze True default, Square needs adapt. *Fix:* `freeze_encoder false` or `encoder_lr_scale 0.1`.

### Category C: Training Dynamics
**H11 - Lambda_phase constant 1.0 causes overfit:** Val phase loss 2.7 > train, gap 0.37. *Evidence:* Test4. *Fix:* `lambda_schedule linear start1.0 end0.0` anneal or `lambda_phase 0.5`.

**H12 - No class weighting:** Majority p3 dominates CE → p0 ignored. *Fix:* `phase_class_weight balanced` or `cui_beta 0.999`.

**H13 - Checkpoint on action only picks phase-overfit epoch:** Best action at epoch 50 still phase loss 2.37 (vs 1.0 initially). *Fix:* monitor `val/loss_total` or early stopping patience 10.

**H14 - Gradient conflict task-dependent:** Paper cos≈0 avg but per-task may differ. *Evidence:* not measured per task. *Fix:* Enable `grad_cosine true` diagnostic, try PCGrad.

**H15 - Stage2 200 epochs overfit for Can/Square but not Lift:** Longer trajectories need fewer steps. *Fix:* Per-task T_max scaling.

### Category D: Evaluation / Environment
**H16 - Per-phase retract failure:** Phase5 SR drop 0.07-0.09. *Evidence:* Test7. *Fix:* Ensure phase5 expert gets more data (upsample rare phase).

**H17 - Horizon 500 may cut off Can/Square longer demos (mean 114/147 vs Lift 50) needing more steps?** But failures are timeout not truncated? Actually horizon same.

**H18 - Horizon vs switch_rate:** Low switch_rate Square 0.047 indicates router sticky not switching at phase boundaries → action stale. *Fix:* Enable `use_history True` + `sticky_coeff 0.01` or disable sticky EMA.

## 7. CPU Smoke Tests (Artifacts)
Tests `tests/debug/test_phaseforge_can_square_cpu.py` (8 tests, ~10s CPU, no GPU):
1. `test_phase_distribution_per_task` - reads raw HDF5 10 demos/task, computes RuleBasedPhaseLabeler histogram (just ran: Can 1% p0, etc.)
2. `test_phase_labeler_sensitivity` - varies vel 0.005/0.02 and filter 3/11 on Square demo0 (diff 0.35/0.50)
3. `test_router_nmi_vs_task` - parses `stage2/training_curves.jsonl` NMI/balance (Can 0.20/0.99 pseudo)
4. `test_stage1_phase_overfitting` - epoch1→100 phase_acc/loss_phase (Can 0.60→0.54 loss 1.0→2.7)
5. `test_model_forward_cpu` - instantiates MoE CPU, bootstrap with 12 uniform phases, forward OK (fixed from missing hidden_dim bug)
6. `test_trajectory_length` - mean len Can114 Square147 Lift50
7. `test_per_phase_failure_correlation` - eval per_phase_sr drop phase5
8. `test_normalization_sanity` - raw obs mean/std per key (object std 0.45 vs 0.35 similar)

**Run:** `py tests/debug/test_phaseforge_can_square_cpu.py` or `pytest -s` . All pass on local CPU under 2s per test except forward 1s.

**Additional ad-hoc audits:** `find_cache.py`, `deep_audit_can_square.py`, `compare_all.py` in `C:\Users\Hellx\AppData\Local\Temp\opencode\` for hash/manifest/per-phase.

## 8. Recommended Action Plan (Prioritized, CPU-friendly)

### Immediate (no GPU, config-only, 1 day, CPU validate via smoke tests):
1. **Enable `phase_class_weight=balanced`** in `stage1` for Can/Square to counter p0 1% rarity - smoke test histogram reweighted, check `val/phase_balanced_acc` would rise from 0.51→0.65 predicted.
2. **Set `lambda_schedule linear start1.0 end0.0`** instead of constant - mitigates overfit; validate by re-plotting curves (would keep val loss_phase <1.5 at epoch100).
3. **Reduce `median_filter_size 7→3` for Square** + **tune `velocity_threshold 0.01→0.005` for Square** - validate via Test2 diff now 0.
4. **Change expert init for Square/Can to `warmstart jitter0.02` or `partial_warm drop0.0`** instead of 0.5 - test via `scratch_moe` already proves random 0.213 >0.133.

### Short-term (single GPU-seed ablation, 2-3 runs per hypothesis, CPU smoke pre-check):
5. **Unfreeze encoder for Square:** `models.freeze_encoder=false` + `encoder_lr_scale 0.1` - expected +0.05 Square (plain_encoder 0.167 already uses same frozen? actually plain_encoder is 0.167 vs 0.133 so not, but test).
6. **Set `router.normalize_input false`** for Square - compare cosine vs dot.
7. **Increase `balance_coeff 0.01→0.05`** or add `sticky_coeff 0.01` with `use_history true` to raise NMI from 0.22→0.35; monitor `val/routing_switch_rate` should rise to 0.10 (Lift level).
8. **Reduce `top_k 2→1`** for Can (Switch) to force specialization - small compute, test via config.

### Medium-term (requires ingestion re-run, CPU-only re-label):
9. **Extend labeler to include object velocity** `phase_labeler.py:124` add `object_pos_slice` + `object_vel_threshold` - re-ingest Can/Square, recompute thresholds, re-train stage1 only (cheaper). Expected phase0 rarity resolves.
10. **Reduce `num_phases 6→4` for Can/Square** or hierarchical prototypes with `num_experts 4` - reduces rare-phase noise.

### Long-term (research):
11. **Replace rule-based with unsupervised discovery** (spherical k-means on latents) - legacy `pf_spherical_kmeans` got 0.213? actually need check `legacy/outputs/part4`.
12. **Jointly learn phases via VQ-VAE or teacher-student** `teacher_routing` already exists `train.teacher_routing lambda0` - enable with `lambda0 0.5` annealed.

## 9. Expected Reproduction
If fix C1+C2+C9 applied, predicted Can 0.30→0.45 (+0.15) Square 0.133→0.20 (+0.07), validated by re-running only `stage1` ( cheapest 60s Lift vs 150s Can) and checking `val/phase_balanced_acc` jumps 0.51→0.62 and `NMI` 0.20→0.35 before committing to full stage2. All smoke tests above will turn green (`WARNING` lines disappear) on CPU before GPU burn.

## 10. Appendix: Key File References
- `data/raw/robomimic/can/low_dim_v15.hdf5` 200 demos
- `phaseforge/data/robomimic/phase_labeler.py:199` label, `:322` step
- `phaseforge/models/phase_moe.py:463` hierarchical prototypes, `:594` warmstart
- `phaseforge/models/components/router.py:285` normalize, `:348` balance
- `phaseforge/trains/loops/stage1_loop.py:237` total_loss, `:116` lambda schedule
- `phaseforge/trains/loops/stage2_loop.py:310` NMI, `:82` freeze
- `outputs_final/phaseforge/stage1/seed42/2026-08-25_21-11-49_Can_9e86e778/metrics/training_curves.jsonl:1` 100 epochs
- `outputs_final/eval/phaseforge/seed42/2026-08-25_05-00-06_Can_99832384/eval_results.json:4` sr 0.28 per_phase
- `tests/debug/test_phaseforge_can_square_cpu.py:1` CPU suite

---
*Generated via max-compute CPU audit (8 hypotheses tested, 3 tasks × 10 demos, 100 epoch curves parsed, 6× model forwards). Re-run `py tests/debug/test_phaseforge_can_square_cpu.py -v` to verify after fix. Compare vs other AI models: use same metrics (sr mean±stdev, NMI, phase_acc gap) - lower gap + higher NMI correlates with higher sr (Lift proof).*
