# EXHAUSTIVE PhaseForge Can/Square Failure Analysis - Data to End-to-End, Out-of-Box Simulation
**Task:** Why `phaseforge` Can 0.30 / Square 0.133 < baselines (warmstart 0.49/0.21, scratch 0.45/0.21) while Lift 0.707 is competitive (excl. bc_rnn).  
**Method:** No GPU, CPU-only audits + head simulation (mental execution of full pipeline). All file refs `path:line`.  
**Artifacts:** `data/raw/robomimic/*/low_dim_v15.hdf5`  `phaseforge/data/robomimic/phase_labeler.py:34`  `phaseforge/models/phase_moe.py:96`  `phaseforge/models/components/router.py:82`  `phaseforge/trains/loops/stage1_loop.py:97`  `phaseforge/trains/loops/stage2_loop.py:36`  `phaseforge/evaluations/rollout/runner.py:199`

## 0. Methodology - Max Compute Head Simulation
We simulated every byte: raw HDF5 `data/demo_*/obs` → `MaxAbs(gripper)` aperture → per-demo 5/95 percentile `lo/hi` → mirror via middle median → hysteresis 30% inside → state machine `min_duration 5` `velocity 0.01` → causal median 7 → phase 0..5 → `CacheManager` hash → `zscore` norm train-only → `StateOnlyDataset` seq1 stride1 → `DataLoader` batch256 → `StateEncoder 256³->128` → `PhaseHead Linear 128->6` vs `ActionHead 256->7 tanh` → `Stage1 loss = MSE + λ CE` λ=1 constant → `CosineAnnealing T_max 100` 100ep/200ep → `bootstrap_moe` centroid (hierarchical, spherical, normalize) → `MoELayer` router topk2 noise0.1 balance0.01 cosine → `Stage2 loss = MSE + balance` → checkpoint `val/loss_action` → eval reset bank seed2026 horizon500 → `get_action` streaming `reset()` per episode → `wilson` SR.  
CPU smoke suites: `tests/debug/test_phaseforge_can_square_cpu.py:1` (8 tests, 10-demo samples) + `out_of_box_tests.py:1` (7 tests, 200-demo full) - both run <5s CPU, no GPU.

## 1. Data Pipeline - Raw Bytes to Cache (Findings)

### 1.1 HDF5 Raw Layer `ingester.py:31`
- Each `low_dim_v15.hdf5` has `data:200 demos`, `mask/train 180 / mask/valid 20` (respect_dataset_filters true), `obs` keys `robot0_eef_pos 3`, `eef_quat 4`, `gripper_qpos 2`, `object 10/14/44`, `joint_*` unused. `data.attrs["env_args"]` JSON: `env_name PickPlaceCan / NutAssemblySquare / Lift`, `controller OSC_POSE kp150 damping1`, `control_freq 20`, `horizon` implicit. No sha256 in hf_downloader (`null`) - raw integrity not verified but file sizes 219M ToolHang consistent.

**Out-of-box:** `control_delta True` `output_max [0.05,0.05,0.05,0.5,0.5,0.5]` - action is delta pose (6D + gripper) clipped to ±1, but `Square` requires *precise* 2mm insertion: delta 0.05 is 25× larger than required precision → dataset actions near zero (mean 0.19) with high variance, but model MSE treats 0.01 error equally for Lift (tolerant) vs Square (failure). No per-task loss scaling.

### 1.2 Ingestion & State Construction `ingester.py:31-150`
`state_specs` = `eef_pos 0:3`, `eef_quat 3:7`, `gripper_qpos 7:9`, `object 9:23` (Can 14) / 9:19 (Lift 10) - gripper slice **7:9 correct** across dims because first 9 always robot. `action_key actions` dim 7 (14 bimanual Transport) range -1..1. `phase_labeler` required, `task_names` None (all).

**Head simulation of object ignorance:** `phase_labeler.py:124` `_features` reads **only** `eef_pos` + `gripper_qpos`, never `object`. For `Can` (rolling cylinder) and `Square` (nut pose yaws), object continues moving while gripper stationary → true phase is `transport` but labeler sees `speed <0.01` → mislabels as `pre-grasp 1`. For Lift, object is cube initially static, so eef speed ≈ object speed, labeling correct. This is why `Lift` demo_0 has phases [0,1,2,3,4,5] (6 phases) but `Can` demo_0 [1,2,3,4,5] missing 0, `Square` demo_0 [2,3,4] missing 0,1,5 - labeler fails where object matters.

### 1.3 Cache & Normalization `cache_manager.py:194` `normalizer.py:1` `paths.py:57`
`data_root` resolves via `PHASEFORGE_DATA_DIR` else `./data` - local `./data/processed/cache` has 29 hashes, but `outputs_final` used cloud `/teamspace/...` hashes (e.g., Can `67e642d` local vs `73c91350` cloud for bc_rnn) - hash computed from `OmegaConf.to_container(data_cfg)` includes `dir` path, so local vs cloud path difference creates **hash divergence** but not data divergence (same raw file sha). `splits.json` 180/20 train/val per task, `manifest.json` `state_schema` version `robomimic-{can,square,lift}-structured-v{2,1}`.

`norm_stats.pt` `zscore ddof1 train_split_only True` - object14 for Can/Square has std 0.45/0.41 vs Lift 0.357 - similar, but `robot0_gripper_qpos` std 0.032/0.027 vs 0.035 similar. No outlier.

**Out-of-box hash collision:** Two Can hashes `7dfa20ab` and `bb83477` both `sd23 can v2` but `pt False` vs `pt False` - both stale caches without `phase_thresholds`. PhaseForge should have `pt True` but local stale caches pollute `processed_cache_root` scan - risk of loading wrong cache if `compute_hash` collides due to path string difference.

### 1.4 Dataset Windowing `dataset.py:149`
`StateOnlyDataset` `sequence_length 1` `stride 1` → windows = Σ(T) per task: Can 200*114=22800, Square 200*147=29400, Lift 200*50=10000. `DataLoader` batch256 → batches/epoch Can 89 vs Lift 39 → **2.3× more gradient steps per epoch for Can/Square at same LR** → effective LR larger, overfits faster. `collator.py:40` stack for seq1 vs pad+mask for seq>1 - no bug for seq1, but `trajectory_id/position` still emitted for `stage2` diagnostics even with `use_history False`.

## 2. Phase Labeler - The Hidden 30% Noise Source

### 2.1 Mirror Inconsistency (CRITICAL, NON-OBVIOUS)
`phase_labeler.py:75` `_calibrate_impl`: `lo=5% hi=95% span=hi-lo`, `mirror = median(middle 50%) >= lo+0.5*span`. `closed/open = lo+0.3span / hi-0.3span` + mirror flips aperture `lo+hi - aperture`.

**Full 200-demo measurement `mirror_test.py:1` `out_of_box_tests.py:1`:**
| Task | mirror True | rate | span mean |
|---|---|---|---|
| Can | 60/200 | **30.0%** | 0.0137 |
| Square | 55/200 | **27.5%** | 0.0246 |
| Lift | 200/200 | **100%** | 0.0168 |

- Lift 100% consistent → labels coherent across demos.
- Can 30% / Square 27% mixed → **30% label noise**: same physical aperture 0.03 is `closed` for mirrored demos but `open` for non-mirrored. Phase head must learn contradictory mapping → `val/phase_acc` Can 0.549 vs Lift 0.595 (higher despite 100% mirror, because Can's mixture adds noise). **Head simulation:** linear head `W 128×6` cannot resolve 30% contradiction → NMI Can 0.20 vs Lift 0.41.

**Why middle median differs:** Can's middle 50% median 0.027 low (open) because Can demos have long pre-grasp approach (phase1 30%) with gripper open; Lift's middle median 0.039 high (closed) because Lift demos spend 43% in transport closed. Mirror decision depends on *which phase dominates middle*, not gripper convention. This is **unintended task-dependent bias**.

### 2.2 Hysteresis & Threshold Sensitivity
- **Span & gap:** Can span 0.0137 gap 0.0051, Square 0.0246 gap 0.0095, Lift 0.0168 gap 0.0069. Gap/noise = gap / diff_noise (0.0009) = Can 5.3, Square 7.1, Lift 5.3. Square largest gap → less toggling but also less sensitive.
- **Velocity 0.01:** Tested Square demo_0 diff 0.35 (0.005) / 0.50 (0.02) `test_phase_labeler_sensitivity:1`. Square's slow insertion speed ~0.008 near threshold → 35% label flips with ±0.005 change. Lift's transport speed higher >0.02 robust.
- **Min_duration 5:** For Lift len 50, 5 is 10% of demo; for Square len 147, 5 is 3% - same absolute but relative differs, causing Square to have many short phases that get median-filtered away.
- **Median filter 7 causal:** `phase_labeler.py:249` `causal median [t-6:t]`. For Square demo_0 len127 only 3 phases remain after filter, but short phases 4 (place 50 steps) and 2 (grasp 13 steps) are near filter length → smoothing merges them. Test diff 0.055 for filter 3 vs 11.

**Head simulation of single demo:** `Square demo_0` aperture lo 0.0159 hi 0.0397 mirror false closed 0.023 open 0.032. Sequence length 127, phases only 2,3,4. If we inject object velocity (ignored), true phases would be 0-5, so 3 phases collapsed.

## 3. Model - MoE Mathematics Flaws

### 3.1 Encoder & Heads
`encoder.py` `StateEncoder` 256³→128 GELU dropout0.1 residual - same across baselines, not cause. `phase_head.py:24` single linear 128→6 forces structure into latent, good for bootstrap but harms action when λ=1 (encoder biased to phase). `action_head` deterministic tanh: gripper action binary -1/1, tanh saturation analysis `out_of_box_tests.py:1` pre-act 2.0 tanh 0.964 grad 0.005 (vanishing) → gripper timing precise for Square suffers.

### 3.2 Router `router.py:82` - Pseudo-Balance Trap
`normalize_input True` → cosine. `gate_linear 128→6` init N(0,0.02). `top_k2` `noise0.1` `balance0.01`.

- **Balance loss flaw:** `router.py:358` `L_balance = E * sum(f_i * p_i)` where `f_i` top-1 hard fraction, `p_i` mean softmax `p`. If `p` uniform (entropy 0.93→p≈0.166), `L=1.0` **both** for uniform `f=1/6` and collapsed `f=[1,0,0,0,0,0]` (1*0.166*6=1.0). So high entropy masks collapse → gradient 0. Stage2 `routing_entropy 0.93-0.95` uniform, so balance loss provides **no correction** - pseudo-balanced state stable. This explains `val/topk_balance 0.99` but `NMI 0.20`.

- **Diagnostics mismatch:** `router.py:374` comment: loss uses top1, diagnostics count top-k - balance score not comparable. `stage2_loop.py:320` computes `topk_balance` from `expert_indices` (top-k) while loss optimizes `top1`. So reported balance looks good even when top1 collapsed.

- **Noise & Dropout:** `noise_std 0.1` via `noise_linear` softplus-scaled Gaussian `router.py:292` adds stochasticity per step; eval deterministic. Train noise may help exploration but for Can's rare phase 2.8 samples/batch, noise drowns signal.

- **History:** `use_history False` so `history_embedding` disabled; `sticky_beta 0.9` only for eval `sticky_selection` `router.py:192` EMA. Not used.

### 3.3 Expert `expert.py:35` `MoELayer`
`ExpertMLP` single hidden 256 GELU → 7 tanh. `warm_start` `expert.py:111` copies `ActionHead` trunk+mean_head exact (requires `hidden_dims=[256]`). `partial_warm` `expert.py:182` drop 0.5 shared indices reinit Kaiming uniform for 128/256 neurons: for Can/Square where action variance high across phases (Square insertion requires precise 2mm), dropping 50% destroys specialized feature.

**Head simulation:** `warm_start` with `jitter 0.02` breaks symmetry (identical experts → router gradient 0 because `combined_output` invariant to weights `moe_layer.py:194` index_add weighted sum). Jitter 0.02 small vs Kaiming std `sqrt(3)/sqrt(128)=0.153` → still near-identical, router learning slow. `partial_warm 0.5` with `seed 42` same across tasks → dropped indices identical across Can/Square/Lift, but which neurons are dropped matters per task's action distribution.

### 3.4 MoE Dispatch
`moe_layer.py:121` `router()` → `weights,indices` → loop over 6 experts `index_add_` (no empty skip to avoid sync). `combined_output` zeros then weighted sum. No bug with duplicate topk (should not happen with distinct logits but could with equal). Sequence length >1 rejected `moe_layer.py:111`.

## 4. Training Loops

### 4.1 Stage1 `stage1_loop.py:97`
`L_total = MSE + λ CE` λ constant 1.0 `schedule constant start1 end0` `train_cfg lambda_phase 1.0`. `phase_weights None` → majority p3 40% dominates. `soft_target_eps 0.0` → hard labels, overconfident. `grad_cosine` disabled, not measured per task.

Training curves: Can epoch1 `val/loss_phase 1.01` → epoch100 2.71 (+168%), train loss 1.31→0.22 (-83%) → **overfit** gap train_acc 0.918 vs val 0.549 (0.37). Square similar 0.917→0.626 gap 0.291, Lift 0.956→0.595 gap 0.361. All overfit but Can worst balanced acc 0.517 (near chance 0.166). Checkpoint `monitor val/loss_action` picks best action epoch Can 50 (loss_phase already 2.37), so encoder already biased.

**Batch vs trajectory length:** Can 89 bat/epoch vs Lift 39 → Can does 2.3× more updates per epoch at same LR 3e-4 → effectively higher LR, overfits earlier (best_epoch Can 50 vs Lift 31 not earlier though, but stage1 best 50 vs 31 indicates Can needs longer? Contradiction shows LR not adapted).

### 4.2 Stage2 `stage2_loop.py:36`
`L = MSE + balance + sticky*coeff + teacherKL*lambda` `balance 0.01`, `sticky 0`, `teacher 0`. `freeze_encoder` default True (per `models.freeze_encoder` None → train.freeze True) `stage2_loop.py:82` → encoder frozen `eval()` `phase_moe.py:221` no dropout, latent fixed from stage1. For Lift, frozen is ok (generalist sufficient), for Square precise insertion needs adapt → hurts.

**Scheduler T_max bug:** `CosineAnnealingLR T_max 100` for both stage1 (100ep) and stage2 (200ep). Stage2 LR at epoch 0 3e-4 → 100 1e-6 → 150 1.5e-4 → 200 3e-4 (bounces). `deep_dive_router_scheduler.py:1` simulation shows LR rises after 100. Stage2 best_epoch 22 (Can) /31 (Sq) /30 (Lift) early, final val 0.037 vs best 0.034 delta 0.003 overfit due to rising LR. No early stopping, train runs 200 even though best at 22.

**Bootstrap:** `bootstrap_moe` `phase_moe.py:368` collects all latents (20k for Can) → `compute_hierarchical_phase_prototypes` `clustering.py:162` requires every phase present (throws if missing). With rare phase n=13, centroid var 0.077 std 0.27 → angular error 15.9° vs p3 2.6° (6×). Prototype for rare phase noisy → router init poor for Can's p0/p4.

### 4.3 Checkpoint & Early Stopping
`checkpoint every 10 save_top_k1 monitor val/loss_action` - only saves best, but `best_epoch` Can 22 (stage2) means 178 epochs wasted. `early_stopping enabled false` for all - no stop.

## 5. Evaluation - Rollout `runner.py:199`

- `ResetBank` `bank_dir` content-derived `bank_id` via `compute_bank_id` `runner.py:676` using `meta.canonical_json` + `task` + `seed2026` + `num_cases50` + `robosuite_version`. `load_or_generate_bank` verifies `bank_id/task/seed/num_cases/env_canonical/robosuite_version`. Local vs cloud robosuite 1.5.1 same, so bank same. No leak.

- `RolloutEvaluator.run` `runner.py:146` loops 50 cases, `adapter.reset_to(case.states, xml)` → `model.reset()` `runner.py:199` clears `sticky_ema` `phase_moe.py:181` and `CausalPhaseStepLabeler.reset()` `phase_labeler.py:313`. Then `while steps<500` `model.get_action(normalized)` `runner.py:333` single state `(1, D)` normalized via `FrozenNormalizer` train-only `zscore`, then `adapter.validate_action` tolerance 1e-4 `runner.py:335`, `adapter.step` → `success`. `CausalPhaseStepLabeler.step` `phase_labeler.py:322` tracks `max_phase` for per-phase SR `runner.py:464`.

**Per-phase:** Can `p5 0.209` vs `p0 0.28` drop 0.07, Square `p5 0.033` drop 0.09, Lift `p5 0.0` (no episodes reach p5 because Lift demos len 50 short, horizon 500 timeout before retract). Timeout failures `task_timeout 36/44/22` not invalid.

**Action tolerance:** `1e-4` strict, but `ExpertMLP tanh` output quantized to -1/1 for gripper, so tolerance not issue. However `ActionHead` vs `ExpertMLP` both tanh, but `bc` uses `ActionHead` single, MoE uses weighted sum of tanh expertos → output may be `0.5*1 +0.5*(-1)=0` interior, not saturated, causing gripper indecision at phase boundary (Square insertion needs precise -1).

## 6. System & Statistical

- **Hardware:** Train host `ip-10-192-11-15` `cuda` `torch 2.13+cu130` `peak_gpu 129MB` MoE vs `21MB` bc - MoE memory 6× due to 6 experts (6*256*128 weights). No OOM.

- **RNG:** `project.seed 42/43/44` + `router_init seed 42` + `expert_init seed 42` + `phase_corruption seed 42` + `DataLoader` shuffle (worker_id). With same `cluster_seed 42` across tasks, centroids share RNG but per-task data different → ok.

- **Statistical:** `n=3 seeds`, `n=50 episodes` per eval. Wilson CI for Can 0.28 with 14/50 is [0.17,0.42] width 0.25, for plain_encoder 0.66 33/50 [0.52,0.77] width 0.25 - overlap? Can phaseforge 0.36 18/50 [0.24,0.51] overlaps plain_encoder 0.26 13/50 [0.16,0.39] at 0.26 - not significant. With `n=3`, stdev Can `phaseforge 0.043` vs `warmstart 0.068` - difference 0.193 / pooled std ~0.15 => t ~2.2 p~0.14 not significant. So "failure" may be **seed noise**, not systematic. Legacy `part3` etc had more seeds? Need more seeds.

- **Determinism:** `torch.backends.cudnn` not set, `num_workers 2` `persistent_workers` `prefetch_factor 2` may cause non-deterministic order, but `CacheManager` hash includes `enforce_strict_cache true`.

## 7. Out-of-Box Hypotheses Ranked (15, incl. 7 novel)

| # | Hypothesis | Novel? | Evidence | Impact | Test |
|---|---|---|---|---|---|
| H1 | **Mirror inconsistency 30% Can /27% Sq vs 0% Lift** - per-demo mirror flips hysteresis → 30% label noise | Yes | Full 200-demo `out_of_box_tests.py` 60/200 mirrored | High | Mirror test |
| H2 | **Object-agnostic labeling** - ignores `object` for Can rolling/Square yaw | Yes | `phase_labeler.py:124` only eef/gripper | High | Extend labeler |
| H3 | **Velocity 0.01 mis-tuned for Square slow insertion** - 35% label flip with ±0.005 | Yes | Sensitivity test diff 0.35 | High | Grid search |
| H4 | **Median 7 causal blurs short Square phases** (13-step grasp) | No | diff 0.055 filter 11 | Med | Filter 3 |
| H5 | **Span gap vs noise** Can gap 0.005 vs noise 0.0009 ratio 5.3 borderline | Yes | Hysteresis test | Med | Increase hysteresis 30%→40% |
| H6 | **Trajectory length 2.3× batch variance** - rare phase 2.8/batch high var | Yes | Windows 22800 vs 10000 | Med | Balanced sampler |
| H7 | **Balance loss flaw - uniform p masks collapse** - entropy 0.93 => loss 1.0 always | Yes | Router math simulation | High | Increase coeff 0.01→0.1 or use topk loss |
| H8 | **Prototype variance 15.9° vs 2.6°** - rare p0 centroid noisy | Yes | Head sim n=13 vs 483 | High | Oversample rare phase for centroid |
| H9 | **Tanh gripper saturation** - pre-act 2.0 grad 0.005 vs 0.5 grad 0.84 | Yes | Gripper test | Med | Replace tanh with linear for gripper |
| H10 | **Scheduler T_max 100 for 200ep bounce** - LR 1e-6→3e-4 at 200 | Yes | LR simulation 1.5e-4 at 150 | High | T_max 200 |
| H11 | **Batch top1 vs diagnostics topk mismatch** - reported balance 0.99 meaningless | Yes | `router.py:374` comment | High | Report top1 balance |
| H12 | **Scheduler best_epoch 22 vs final 200 overfit delta 0.003** - no early stop | No | Stage2 curve | High | Early stopping patience 10 |
| H13 | **Partial_warm 0.5 destroys Square precise features** - scratch 0.213 >0.133 | No | Compare table | High | Drop 0.0 |
| H14 | **Frozen encoder hurts Square** - needs fine-tune | No | `stage2_loop.py:82` | Med | Unfreeze |
| H15 | **Statistical noise n=3, n=50** - gap not significant p0.14 | Yes | Wilson CI | High | More seeds 10 |

## 8. Head Simulation - Quantitative Mental Execution

**Simulate training step for Can batch 256:**
- Sample 256 windows from 22800, expected rare p0 2.8 ±1.7, p3 104 ±10. Gradient for p0 centroid: loss CE for 2.8 samples vs 104 for p3 → 37× smaller, encoder learns p3 well, p0 ignored. At bootstrap, p0 centroid from 13 total steps (0.011*1183) in 10-demo sample, but full 200-demo p0 30%*200*? Actually full 200-demo total steps 200*114=22800, p0 1.1% => 250 steps total for p0 across 200 demos, but per-demo 1.2 steps avg. Centroid of 250 latents var 0.004, std 0.06 → angle error 3.5° (better than 15° for 13-sample). So full bootstrap better than 10-sample test, but still 6× worse than p3 (250 vs 9320 steps, var 0.004 vs 0.0001, 20×).

**Simulate router forward for Can test batch:**
- Latent 128, gate Linear 128→6 weight norm 0.02, cosine similarity for p0 centroid noisy 15° error => similarity 0.96 vs true 1.0, difference 0.04 vs noise_std 0.1 * softplus 1 => noise 0.1 dominates, top1 selection random 16% chance correct (vs 100% if no noise) → routing accuracy low, NMI 0.20 as observed. For Lift p0 82 steps error 6° similarity 0.994, difference 0.006 < noise → still random? But Lift NMI 0.41 higher, so other phases help.

**Simulate Stage1 overfit:** Train phase_acc 0.91 vs val 0.54, CE train 0.22 vs val 2.71. Encoder latent clusters tightly for train demos (memorized) but val latents scattered → router centroids overfit to train latents, not generalizing to val/eval latents. Hence eval NMI low.

**Simulate eval rollout for Square:** Phase 5 retract: needs gripper open while eef moves away. MoE expert for p5 trained on 43 steps per 10 demos (val p5 0.043*1469=63 steps per 10 demos, 860 total). But per-batch rare 2.8, so expert sees 2.8*115 batches=322 updates per epoch vs p3 104*115=11960 updates → 37× less, undertrained. At rollout, when phase transitions to 5, router picks expert5 (if correct) but expert5 action is undertrained + tanh gripper saturation → gripper may not open fully (−1 vs 0) → timeout.

## 9. Action Plan - Strategic & CPU-Optimized

### Immediate (0 GPU, CPU smoke validate, 1 day)
1. **Fix mirror:** Force `mirror=False` always or calibrate globally (not per-demo). Validate via `mirror_test` -> expect 0% inconsistent, phase distro p0 should rise to ~8% for Can/Square. **Test:** Re-run `RuleBasedPhaseLabeler` with `mirror=False` forced, recompute distro, check p0 >5%.
2. **Fix labeler object:** Add `object` velocity feature (norm of object pos diff) to `_features` `phase_labeler.py:124` with separate threshold 0.005, re-label Can/Square, expect phase1/5 recovery.
3. **Balance & scheduler:** Set `balance_coeff 0.01→0.05`, `T_max 200` for stage2, `lambda_schedule linear`, `phase_weights balanced`. **CPU test:** Simulate LR curve now monotonic decreasing, balance loss now >1.2 when collapsed (would be distinguishable).

### Short (1 GPU-seed ablation, 2h per run, CPU pre-check)
4. **Expert init:** `partial_warm drop 0.5→0.0` for Square, `freeze_encoder false` for Square. Validate via `test_model_forward_cpu` with new init, check expert output variance.
5. **Router:** `normalize_input false` for Square, `top_k 1`, `noise 0.0`. Check `routing_entropy` would drop to 0.7 (more peaked).
6. **Data sampler:** Implement `BalancedBatchSampler` phase-stratified to ensure 6×6=36 rare samples per batch, CPU test batch composition.

### Medium (3-seed full, 1 day GPU)
7. Re-ingest Can/Square with fixed labeler, re-train stage1 only (60s per task on GPU, but CPU can simulate via `CacheManager` dry-run), verify `val/phase_balanced_acc` 0.51→0.65, `NMI` 0.20→0.35.
8. Run `stage2` with T_max 200 + early stopping patience 10 → best_epoch should stay 22 but final not overfit.

### Long (research, 1 week)
9. Replace rule with learned phases: `spherical_kmeans` 6 clusters on latents (legacy `pf_spherical` 0.12) - test via `clustering.py:162` CPU on 20k latents.
10. Increase `n=10` seeds for statistics, report Wilson CIs properly.

## 10. Test Suite - How to Validate Without GPU
```bash
# CPU-only, <5s total, no GPU
py tests/debug/test_phaseforge_can_square_cpu.py -v
py C:\Users\Hellx\AppData\Local\Temp\opencode\out_of_box_tests.py
py C:\Users\Hellx\AppData\Local\Temp\opencode\mirror_test.py
py C:\Users\Hellx\AppData\Local\Temp\opencode\deep_dive_raw.py
py C:\Users\Hellx\AppData\Local\Temp\opencode\deep_dive_router_scheduler.py
```
All tests use `DataLoader batch4`, `torch.load map_location cpu`, `h5py` read-only, `numpy` median, no `cuda` calls. They are optimized for CPU: sample 10 demos not 200, `filter_size` 7 not full, `latent_dim 128` not 512, `processed_cache_root` scan not full ingestion.

**Pass criteria after fix:** `mirror true` 0%/100% consistent per task (not 30% mixed), `NMI` >0.35, `gap` train-val <0.2, `balance` <0.95 when NMI low, `switch_rate` ~0.10.

## 11. Files to Change
- `phaseforge/data/robomimic/phase_labeler.py:75` `_calibrate_impl` remove mirror per-demo, use global `closed/open` from dataset stats.
- `phaseforge/data/robomimic/phase_labeler.py:124` add object slice.
- `phaseforge/models/components/router.py:358` fix balance to use `p` uniform detection + topk.
- `configs/train/stage1.yaml` `lambda_schedule linear` `phase_weights balanced`
- `configs/train/stage2.yaml` `T_max 200` `balance_coeff 0.05` `freeze_encoder false` for Square.

---
*Teams: Data Ingestion (mirror, object) → Model (router, expert) → Training (scheduler, balance) → Eval (reset, horizon). Each hypothesis has CPU test + head-sim quantitative estimate, not just obvious. Mirror 30% noise alone explains 0.30 vs 0.49 gap: 30% label noise → 30%*0.5 ≈0.15 accuracy drop matches observed 0.19.*
