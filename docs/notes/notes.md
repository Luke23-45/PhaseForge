

## 1. Full Master Runs with `uv`

### Main 5-Task Benchmark (Proposed + All 9 Baselines across 5 Tasks, 3 Seeds = 315 Steps)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner `
  --manifest experiments/five_task.json `
  --outputs outputs_final `
  --expect-steps 315 `
  --continue-on-error
```

### Lift Ablation Suite (All 27 Ablation Cells, 3 Seeds = 165 Steps)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner `
  --manifest experiments/lift_ablation.json `
  --outputs outputs_ablation `
  --expect-steps 165 `
  --continue-on-error
```

---

## 2. All Baseline Runs (Run by Method via `uv`)

You can run individual baselines across all 5 tasks (`Lift`, `Can`, `Square`, `ToolHang`, `Transport`) and seeds (`42, 43, 44`) using `uv run --python .venv-rollout`:

### 1. Proposed Method (`phaseforge` — Stages 1 & 2)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods phaseforge --with-dependencies
```

### 2. Standard Behavior Cloning Floor (`bc` — Stage 1)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc
```

### 3. Parameter-Matched Dense Baseline (`bc_large` — Stage 1)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc_large
```

### 4. Temporal History Comparator (`bc_rnn` — 10-step history, Stage 1)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc_rnn
```

### 5. Robot-Only Negative Control (`bc_robot_only` — Stage 1)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods bc_robot_only
```

### 6. Scratch MoE Baseline (`scratch_moe` — Random init, Stage 2)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods scratch_moe
```

### 7. Warm-Start MoE Baseline (`warmstart_moe` — Plain BC encoder × Random router, Stage 2)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods warmstart_moe --with-dependencies
```

### 8. Phase Pretraining / Random Router (`phase_pretrain_random_router` — H1 control, Stage 2)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods phase_pretrain_random_router --with-dependencies
```

### 9. Plain Encoder / Centroid Router (`plain_encoder_phase_bootstrap` — H2 control, Stage 2)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods plain_encoder_phase_bootstrap --with-dependencies
```

### 10. Privileged Training Diagnostic (`teacher_forced` — GT phase routing, Stage 2)
```powershell
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/five_task.json --outputs outputs_final --methods teacher_forced --with-dependencies
```

---

## 3. All Ablation Runs (Lift Ablation Suite via `uv`)

All ablation variants are executed under the `outputs_ablation` namespace using `experiments/lift_ablation.json`:

### Group A: Router Initialization Controls
```powershell
# Spherical K-Means Router (EXP-109)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_spherical_kmeans --with-dependencies

# Euclidean K-Means Router (EXP-110)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_kmeans --with-dependencies

# Discriminative Phase Head Classifier (EXP-111)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_phase_head --with-dependencies

# PS-Rand-Rand 4-Way Init Cell A (EXP-112)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_random_random --with-dependencies

# PS-Centroid-Rand 4-Way Init Cell B (EXP-113)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_centroid_random --with-dependencies
```

### Group B: Expert Initialization & Warm-Start Drop Sweep
```powershell
# Full standard warm-start (EXP-212)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_full_warm --with-dependencies

# 0% drop / exact full copy (EXP-213)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_drop00 --with-dependencies

# 25% drop rate (EXP-214)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_drop25 --with-dependencies

# 75% drop rate (EXP-215)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_drop75 --with-dependencies

# 100% drop rate / full re-init (EXP-216)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_drop100 --with-dependencies

# One warmstart generalist + random experts (EXP-211)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_one_warm_plus_random --with-dependencies
```

### Group C: Representation, Fine-Tuning & Phase Noise
```powershell
# Spherical averaging vs Euclidean centroid (EXP-114)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_spherical --with-dependencies

# PhaseForge-FT with fine-tuned encoder (EXP-115)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_ft --with-dependencies

# 25% phase label corruption (EXP-205)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_corrupt_25 --with-dependencies

# 50% phase label corruption (EXP-206)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_corrupt_50 --with-dependencies

# 100% permutation shuffle control (EXP-207)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_shuffle_control --with-dependencies
```

### Group D: Capacity & Expert Scaling (K Sweep)
```powershell
# Super-prototype reduction: E=3 (EXP-201)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_k3 --with-dependencies

# Sub-prototype scaling: E=12 (EXP-202)
uv run --python .venv-rollout python -m phaseforge.runner --manifest experiments/lift_ablation.json --outputs outputs_ablation --methods pf_k12 --with-dependencies
```

---

## 4. Direct Individual Model Training & Evaluation with `uv`

If you want to train and evaluate any model directly without the runner:

```powershell
# 1. Train Stage 1 (BC or Phase-supervised encoder)
uv run --python .venv-rollout phaseforge-train models=phaseforge data=lift train=stage1 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/bc data=lift train=stage1 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/bc_large data=lift train=stage1 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/bc_rnn data=lift_rnn train=stage1 project.seed=42

# 2. Train Stage 2 MoE models
uv run --python .venv-rollout phaseforge-train models=phaseforge data=lift train=stage2 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/scratch_moe data=lift train=stage2 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/warmstart_moe data=lift train=stage2 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/phase_pretrain_random_router data=lift train=stage2 project.seed=42
uv run --python .venv-rollout phaseforge-train models=baselines/plain_encoder_phase_bootstrap data=lift train=stage2 project.seed=42

# 3. Rollout Evaluation
uv run --python .venv-rollout phaseforge-eval models=phaseforge data=lift eval=rollout project.seed=42
uv run --python .venv-rollout phaseforge-eval models=baselines/bc data=lift eval=rollout project.seed=42
```

*(For ToolHang, replace `--python .venv-rollout` with `--python .venv-toolhang` and `data=lift` with `data=tool_hang`)*.