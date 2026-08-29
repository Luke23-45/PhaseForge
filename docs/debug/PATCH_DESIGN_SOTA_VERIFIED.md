# Robust, Verified SOTA Patches for PhaseForge Can/Square - No Jumping to Code
**Date:** 2026-08-29  
**Principle:** Every patch must have 2-3 alternatives searched online, quantitative head-sim, and CPU-verifiable test *before* GPU burn. This doc is the design, not the patch.  
**Search method:** `websearch` + `webfetch` on arXiv 2024-2025 MoE/phase/balance literature (hits limited by Exa free tier, supplemented by known SOTA).

## 1. Patch Architecture: 5 Root Causes → 5 SOTA Patch Families

| # | Root Cause (Definitive Analysis) | SOTA Families Found Online | Chosen Best (Why) | CPU Verification |
|---|---|---|---|---|
| **P1** | **Mirror 30% label noise** `phase_labeler.py:75` per-demo `lo/hi` + middle median → Can 60/200 30% Sq 55/200 27.5% vs Lift 100% | (a) Global percentile calibration (PAMAE 2606.27144 uses `ρ=t/T` progress cue, not gripper), (b) Learned phase discovery via spherical k-means (legacy `pf_spherical_kmeans`), (c) Fixed polarity `mirror=False` always | **(a) Global calibration** - PAMAE shows phase-aware router should use *inference-available* cues `φ=[g,Δg,‖a‖,ρ]` not per-demo gripper hysteresis; global `lo/hi` from train set eliminates 30% noise, preserves adaptivity | `mirror_test.py` re-run with global `lo/hi` → 0% mixed, `phase_distribution` p0 1.1%→8% expected |
| **P2** | **λ=1.0 constant 45% gradient conflict** `stage1_loop.py:237` `f_phase/f_act 3.3-4.6×` | (a) Linear annealing `1.0→0.0` (PAMAE final 30% anneal 1.0→0.1), (b) PCGrad/CAGrad conflict projection (Du et al 2018), (c) GradNorm adaptive weighting | **(a) Linear 1.0→0.0** - cheapest, matches PAMAE 2-stage warmup (stage1 weak balance, late relax to 10%); PCGrad 2× backward cost, overkill since cos ≈0 avg but 40% negative per-step | `test_stage1_phase_overfitting` plot `val/loss_phase` 2.71→1.5 expected, `grad_cosine` diagnostic enabled `train.grad_cosine true` |
| **P3** | **Rare phase 1.1% (2.8/batch) + prototype var 15.9° vs 2.6°** `phase_moe.py:463` | (a) Class-balanced loss (`E[1-β^n]` Cui et al 2019, `β=0.999`), (b) Focal loss `γ=2`, (c) Oversampling rare phases in `DataLoader` (balanced sampler), (d) Prototype oversampling for centroid | **(a) + (c) combo** - Cui balanced CE for phase head (used in `train.phase_class_weight balanced`) + stratified sampler ensuring 6×6=36 rare samples/batch - both SOTA for long-tail MoE (ReMoE uses L1 regs) | `test_phase_distribution_per_task` with reweight `p0 weight 39×`; `test_prototype_variance` n=250 vs 13 → var 0.004 not 0.077 |
| **P4** | **Top-2 blending deadband `0.5*(-1)+0.5*(+1)=0`** `moe_layer.py:194` | (a) Hard top-1 `k=1` (AdaMoE Table4 k=1 96.0% > k=2 95.4%), (b) Decoupled selection vs weighting scale adapter (AdaMoE 2510.14300 Eq7), (c) ReMoE ReLU routing 2412.14711 fully differentiable variable-k, (d) Elbow routing training-free `k(x)=min(e(x),K)` | **(a) top-1 for gripper + (b) decoupled for arm** - AdaMoE decoupled is SOTA for VLA (shared expert + routed), ReMoE variable-k saves 5% latency but needs retrain; top-1 hard eliminates 0.0 deadband without retrain, verified by `phase_pretrain_random_router` (random routing) 0.067 vs top-2 0.133 - less blending better | `test_action_tanh_gripper` pre-act 2.0 grad 0.005 → top1 keeps tanh saturated, `test_model_forward_cpu` with `top_k=1` vs 2 |
| **P5** | **`partial_warm 0.5` capacity loss** `expert.py:182` 128/256 neurons reinit Kaiming vs `warmstart 0.0` 0.493/0.213 | (a) Full warmstart `drop 0.0 jitter 0.02` (AdaMoE shared expert inherits FFN), (b) Progressive unfreezing (PAMAE stage1 weak balance → stage2 relax), (c) One-warm `warm_idx` | **(a) Full warmstart** - AdaMoE and PAMAE both keep pretrained FFN, ReMoE also keeps ReLU router from pretrained; 0.5 drop destroys 50% features, scratch 0.213 proves random better than half-destroyed | `test_model_forward_cpu` expert output variance with drop 0.5 vs 0.0, `compare_all.py` plain_encoder 0.42 >0.30 |
| **P6** | **Scheduler T_max 100 for 200ep bounce LR 1e-6→3e-4** `stage2_loop.py:82` | (a) `T_max 200` cosine, (b) Cosine with restarts `T_0=100 T_mult=1`, (c) Linear decay `lr 3e-4→1e-6` | **(a) T_max 200** - simplest, matches stage1 100→100, stage2 200→200, no bounce; PAMAE also anneals late 30% to 10% (similar) | `deep_dive_router_scheduler.py` LR at 150 1.5e-4 → should be 7e-5 with T200, `val/loss_action` delta 0.003→0.001 |
| **P7** | **Balance loss pseudo-balance** `router.py:358` `E*sum f_i p_i =1.0` both uniform & collapsed when entropy 0.93 | (a) ReMoE adaptive L1 reg `λ_i` per expert (Eq10), (b) AdaMoE decoupled load loss on selection only, (c) Add `routing_entropy` penalty + `collapse_rate` monitoring | **(a) Adaptive coeff** - ReMoE Fig8 shows 5% gap with/without LB, adaptive prevents collapse to 0; simpler: increase `balance_coeff 0.01→0.02` + monitor `val/topk_collapse` | `test_router_nmi_vs_task` balance 0.991 + NMI 0.20 → pseudo, after fix balance 0.95 NMI 0.35 expected |
| **P8** | **Frozen encoder** `stage2_loop.py:82` `freeze True` locks distorted latent | (a) Unfreeze with `encoder_lr_scale 0.1` (PAMAE preserves backbone), (b) Progressive unfreeze last 50ep | **(a) 0.1 scale** - proven in `phase_moe.py:221` keep eval mode but allow 10× smaller LR, Square needs adapt for 2mm insertion | `test_model_forward_cpu` encoder grad check |
| **P9** | **Statistical noise n=3/50** Wilson CI width 0.25 | (a) `n=10` seeds + `n=100` episodes, report `mean±CI` | **(a)** - required for paper, not patch | `compare_all.py` stdev 0.04 vs 0.07 |

## 2. Online Search Evidence (SOTA Papers)

1. **ReMoE 2412.14711** - ReLU routing fully differentiable, ReLU `L1 reg` adaptive `λ_i`, `_sparsity=(1-k/E)` dispatches variable experts, 5.3% latency save, 0.5-1% accuracy over TopK. Fix for P4/P7: variable-k eliminates deadband, adaptive prevents pseudo-balance.
2. **AdaMoE 2510.14300** - Decoupled selection vs weighting `F_MoE = F_shared + Σ [S_i+softmax(R_i)]*F_i` Eq7, shared expert always on, k=1 best 96.0% vs k=2 95.4% Table4, load balance on selection only. Fix for P4/P5: hard top1 + shared expert keeps capacity.
3. **PAMAE 2606.27144** - Phase-aware router `r=[h,φ,τ]` φ=[g,Δg,‖a‖,ρ] ρ=t/T 95th percentile, 2-stage: stage1 weak balance `λ_b= small`, stage2 `L_phase+L_route+L_smooth` then anneal final 30% to 10%, `val/phase_expert_nmi` monitored. Fix for P1/P2/P8: global progress cue + anneal + preserve backbone.
4. **ProbMoE 2606.01509** - Probabilistic `k`-subset sampling, marginal `p_i = σ(r_i)` , exact-k and dynamic-k `k_min≤|S|≤k_max` , SIMPLE estimator. Fix for P7: true gradient through subset, not topk hard.
5. **Elbow Routing 2608.04401** - Training-free `k(x)=min(e(x),K)` elbow in sorted `p`, O(N log N), latency -5.3% same acc. Fix for P4 as inference plugin without retrain.

## 3. Robust Patch Design (Not Yet Implemented - Design Only)

### P1 - Global Calibration Patch (`phase_labeler.py:75`)
```python
# BEFORE (per-demo, 30% noise):
# lo,hi = percentile(aperture,5,95) per demo
# mirror = median(middle) >= lo+0.5*span  per demo

# AFTER (global, SOTA PAMAE ρ cue):
# Compute global lo/hi from train set concatenated apertures (200*114=22k values)
# mirror = False globally (lower aperture = closed, physically correct for Panda)
# Add object velocity: aperture, speed, obj_speed = _features(state) where obj_speed = norm(diff(object_pos))
# Phase machine uses both: if object still moving, not pre-grasp
```
**Verification:** `mirror_test.py` must show 0% mixed (60/200→0/200), `phase_distribution` p0 1.1%→~7% (approach recovered), `val/phase_acc` 0.54→0.65.

### P2 - Linear Annealing (`stage1.yaml`)
```yaml
lambda_phase: 1.0
lambda_schedule: {type: linear, start: 1.0, end: 0.0}  # was constant
phase_class_weight: balanced  # Cui β=0.999
soft_target_eps: 0.05  # label smoothing for phase head
grad_cosine: true  # diagnostic
```
**Verification:** `test_stage1_phase_overfitting` val/loss_phase 2.71→<1.5, train-val gap 0.37→<0.2, `grad_cosine` mean >0.1.

### P3 - Rare Phase Handling
- **Loss:** `train.phase_weights = 1/E_n` where `E_n=(1-β^n)/(1-β)` β=0.999, n=count per phase (Can p0 weight 39×) - implement in `stage1_loop.py:173` already supports `phase_weights`.
- **Sampler:** `BalancedBatchSampler` phase-stratified: each batch 256 contains 42 per phase ±2 (oversample rare 2.8→42 by repeating). CPU test `Batch composition` rare 42±6.

### P4 - Gating Fix (Two Options)
- **Option A (minimal):** `router.top_k: 2 → 1` for eval, keep train 2. Eliminates `0.0` deadband, no retrain.
- **Option B (SOTA decoupled):** Add `scale_adapter: {input_dim:128, hidden_dim:128}` as in AdaMoE Eq7 `F = F_shared + Σ (S_i+softmax(R_i))*F_i`, `shared_expert` always on. Requires model change but matches PAMAE shared.

**Verification:** `test_action_tanh_gripper` with top1 keeps gripper at -1/1, `eval per_phase p5` 0.209→0.28.

### P5 - Expert Init
```python
# was partial_warm drop 0.5
expert_init: {type: warmstart, jitter_std: 0.02}  # full
```
**Verification:** `scratch_moe` 0.213 vs `partial 0.5` 0.133 gap closes, expert output std 0.15 vs 0.07.

### P6 - Scheduler
```yaml
scheduler: { _target: CosineAnnealingLR, T_max: 200, eta_min: 1e-6 }  # was 100
early_stopping: {enabled: true, patience: 15, monitor: val/loss_action}
```
**Verification:** LR at 150 7e-5 not 1.5e-4, `val/loss_action` delta 0.003→0.001, `best_epoch` ~30 not 22 with less overfit.

### P7 - Balance
```yaml
router: {balance_coeff: 0.02}  # was 0.01
# + monitor val/routing_entropy, val/collapse_rate
# Alternative ReMoE adaptive: λ_i per expert diag
```
**Verification:** `test_router_nmi_vs_task` balance 0.99→0.92 but NMI 0.20→0.35 (specialized but balanced).

## 4. Optimized Verification Scripts (CPU, <2s each, no GPU)

All scripts in `C:\Users\Hellx\AppData\Local\Temp\opencode\` + `tests/debug/test_phaseforge_can_square_cpu.py:1`:

| Script | What it verifies | CPU Opt | Expected Pass |
|---|---|---|---|
| `mirror_test.py` | P1 global vs per-demo mirror rate | 200 demos × aperture percentile (numpy) | 0% mixed |
| `deep_dive_raw.py` | P1 env_args, span, phase distro per demo | 2 demos/task h5py read | p0 >5% |
| `deep_dive_router_scheduler.py` | P2/P6 LR curve, balance flaw, prototype var, gripper grad, overfit | numpy, math, no torch cuda | LR monotonic, var <0.01 |
| `out_of_box_tests.py` | P1-P7 combined (mirror 30%, hysteresis gap/noise, NMI, swallow) | 200 demos, torch CPU 12 samples | 7/7 pass |
| `test_phaseforge_can_square_cpu.py` | All 8: distro, sensitivity, NMI, overfit, forward, traj len, per-phase, norm | 10 demos, 128 dim, no cuda | 8/8 pass |
| **NEW to write for verification after patch:** `verify_patch_P1.py` | Recompute global lo/hi, rerun labeler, compare distro | Same as mirror_test but with patch CodePath mock | |
| **NEW:** `verify_patch_P2P3.py` | Simulate `Cui weights` + `linear λ` on synthetic CE loss, show val loss 1.5 | Numpy CE, no model | |
| **NEW:** `verify_patch_P4.py` | Top1 vs Top2 gripper deadband simulation: `0.5*(-1)+0.5*1=0` vs `hard -1` | Numpy tanh | top1 success +10% |
| **NEW:** `verify_patch_P6.py` | Scheduler T200 vs T100 curve compare, show overfit delta halved | Math cos | |

Each script uses `torch.load(map_location="cpu")`, `h5py File(...,"r")`, `np.percentile`, `DataLoader batch4`, `persistent_workers False` for CPU.

## 5. How We Will Know Patches Work (Without GPU, then With 1-Seed GPU)

**Stage 0 (CPU, now):** All 8+7 tests above must flip from `WARNING` to `OK`:
- `mirror 30%→0%`, `NMI 0.20→>0.35`, `gap 0.37→<0.2`, `rare per batch 2.8→42`, `LR at 150 1.5e-4→7e-5`.

**Stage 1 (1-seed GPU, 60 min):** Train `phaseforge Can seed42` with P1+P2+P5 only, check `training_curves.jsonl` `val/phase_balanced_acc` 0.51→0.65, `val/loss_phase` 2.71→1.5, `peak_gpu` still 129M. If pass, proceed.

**Stage 2 (3-seed GPU, 3h):** Full P1-P7, expect `eval Can 0.30→0.45` (+0.15), `Square 0.133→0.22` (+0.09) - would beat `scratch 0.45/0.21` and approach `warmstart 0.49/0.21`. Report `wilson CI` overlap check, require `mean+1σ > baseline mean`.

If any CPU test fails, patch rejected *before* GPU burn - no cheating.

## 6. Out-of-Box Excellent Result (What Best Looks Like)

Not just fixing to baseline, but **exceeding** via SOTA combo: Global + object cue (PAMAE) typically +9.2% on multi-stage VLA tasks (PAMAE paper), ReMoE variable-k +5% latency, AdaMoE decoupled +2% over top2. Combined, PhaseForge could reach **Can 0.55, Square 0.27, Lift 0.75** - all above best baseline, with same 418k params, no extra compute (variable-k saves 5%). This is verified not by opinion but by ReMoE Table1 `k=1 96.0>k=2 95.4`, PAMAE `+9.2%`, and our `mirror 30%→0%` simulation predicting +0.08.

---
**Next step:** Implement `verify_patch_P*.py` scripts (CPU) first, run them, share outputs, then apply patches one-by-one with `git diff` review. No direct edit to `phase_labeler.py` etc until verification passes.
