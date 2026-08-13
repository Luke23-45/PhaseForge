# PhaseForge — Experiment Runbook

## Dependency Overview

```
TERMINAL 1          TERMINAL 2          TERMINAL 3          TERMINAL 4
─────────────────   ─────────────────   ─────────────────   ─────────────────
phaseforge Stage1   BC Stage1           scratch_moe         oracle_moe
        │                  │
        ▼                  ▼
phaseforge Stage2    warmstart_moe
(auto BC ckpt if     (needs BC ckpt)
 not provided)

                         ▼  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  ▼
                         EVALUATE ALL MODELS (after all training)
```

**Key facts:**
- **PhaseForge Stage 1** and **BC Stage 1** can run in parallel (no dependencies)
- **PhaseForge Stage 2** auto-detects the latest Stage 1 checkpoint from `outputs/phaseforge/stage1/`
- **WarmStartMoE** auto-detects the latest BC checkpoint from `outputs/bc/stage1/`
- **ScratchMoE** and **OracleMoE** have zero dependencies — they train from scratch immediately
- **Evaluation** runs after all training completes

---

## Quick Command Reference

| # | Command | Depends On | Output Dir |
|---|---------|-----------|------------|
| 1 | `phaseforge-train models=phaseforge train=stage1` | — | `outputs/phaseforge/stage1/<ts>_<id>/` |
| 2 | `phaseforge-train models=baselines/bc train=stage1` | — | `outputs/bc/stage1/<ts>_<id>/` |
| 3 | `phaseforge-train models=baselines/scratch_moe train=stage2` | — | `outputs/scratch_moe/stage2/<ts>_<id>/` |
| 4 | `phaseforge-train models=baselines/oracle_moe train=stage2` | — | `outputs/oracle_moe/stage2/<ts>_<id>/` |
| 5 | `phaseforge-train models=phaseforge train=stage2` | #1 done | `outputs/phaseforge/stage2/<ts>_<id>/` |
| 6 | `phaseforge-train models=baselines/warmstart_moe train=stage2` | #2 done | `outputs/warmstart_moe/stage2/<ts>_<id>/` |
| 7 | `phaseforge-eval models=phaseforge train.stage1_ckpt_path=<ckpt>` | #5 done | `outputs/eval/phaseforge/<ts>_<id>/` |
| 8 | `phaseforge-eval models=baselines/bc train.stage1_ckpt_path=<ckpt>` | #2 done | `outputs/eval/bc/<ts>_<id>/` |
| 9 | `phaseforge-eval models=baselines/scratch_moe train.stage1_ckpt_path=<ckpt>` | #3 done | `outputs/eval/scratch_moe/<ts>_<id>/` |
| 10 | `phaseforge-eval models=baselines/oracle_moe train.stage1_ckpt_path=<ckpt>` | #4 done | `outputs/eval/oracle_moe/<ts>_<id>/` |
| 11 | `phaseforge-eval models=baselines/warmstart_moe train.stage1_ckpt_path=<ckpt>` | #6 done | `outputs/eval/warmstart_moe/<ts>_<id>/` |

---

## Phase 1: Launch All Independent Runs

Open **4 separate terminals** and run these in parallel:

### Terminal 1 — PhaseForge Stage 1 (the main method)
```bash
phaseforge-train models=phaseforge train=stage1
```
- This trains encoder + action_head + phase_head with phase-supervised loss
- **Expected duration:** longest run (~100 epochs)
- **Checkpoint saved to:** `outputs/phaseforge/stage1/<timestamp>_<runid>/checkpoints/checkpoint_best.pt`
- When it finishes → proceed to Phase 2 Terminal 1

### Terminal 2 — BC Baseline Stage 1
```bash
phaseforge-train models=baselines/bc train=stage1
```
- Standard behavioral cloning (MSE loss), no phase supervision
- **Checkpoint saved to:** `outputs/bc/stage1/<timestamp>_<runid>/checkpoints/checkpoint_best.pt`
- This checkpoint will be auto-detected by WarmStartMoE in Phase 2
- When it finishes → proceed to Phase 2 Terminal 2

### Terminal 3 — Scratch MoE (no dependencies)
```bash
phaseforge-train models=baselines/scratch_moe train=stage2
```
- MoE trained entirely from scratch, no pretraining
- Runs immediately, no checkpoint needed
- When it finishes → note the checkpoint path for evaluation

### Terminal 4 — Oracle MoE (no dependencies)
```bash
phaseforge-train models=baselines/oracle_moe train=stage2
```
- MoE with ground-truth phase routing (upper-bound performance)
- Runs immediately, no checkpoint needed
- When it finishes → note the checkpoint path for evaluation

---

## Phase 2: Stage 2 Training (after Phase 1 deps complete)

Run these after their respective Stage 1 finishes:

### Terminal 1 (after PhaseForge Stage 1 completes)
```bash
phaseforge-train models=phaseforge train=stage2
```
- Auto-detects the latest checkpoint from `outputs/phaseforge/stage1/`
- Runs bootstrapping algorithm (centroids → router init → expert copy)
- Then trains the full MoE with load-balancing loss

### Terminal 2 (after BC Stage 1 completes)
```bash
phaseforge-train models=baselines/warmstart_moe train=stage2
```
- Auto-detects the latest BC checkpoint from `outputs/bc/stage1/`
- Router stays random (no phase centroids), experts initialized from BC action_head
- Tests impact of phase supervision vs. standard warm-start

---

## Phase 3: Evaluate All Models

After all training completes, run evaluation for each model.

You can run these in parallel across terminals. Each needs the path to its **best checkpoint**.

### Quick way to find checkpoints (using the discovery script)

The project includes a dedicated CLI tool for checkpoint discovery. It
uses the **same** logic as ``phaseforge-train``'s auto-detect, so the
results are always consistent.

```bash
# Print path to latest PhaseForge Stage 1 checkpoint (for use in commands):
python scripts/find_checkpoint.py latest --model phaseforge --stage 1

# List all checkpoints for a model+stage with metadata:
python scripts/find_checkpoint.py list --model phaseforge --stage 1

# List ALL checkpoints across every model (overview of experiment):
python scripts/find_checkpoint.py list-all

# Validate a checkpoint is loadable:
python scripts/find_checkpoint.py validate outputs/phaseforge/stage1/*/checkpoints/checkpoint_best.pt

# Use in a command substitution (PowerShell):
phaseforge-eval models=phaseforge `
    train.stage1_ckpt_path=$(python scripts/find_checkpoint.py latest --model phaseforge --stage 1)
```

### Evaluate commands

```bash
# Terminal A — PhaseForge
phaseforge-eval models=phaseforge `
    train.stage1_ckpt_path=outputs/phaseforge/stage2/<ts>_<id>/checkpoints/checkpoint_best.pt

# Terminal B — BC
phaseforge-eval models=baselines/bc `
    train.stage1_ckpt_path=outputs/bc/stage1/<ts>_<id>/checkpoints/checkpoint_best.pt

# Terminal C — Scratch MoE
phaseforge-eval models=baselines/scratch_moe `
    train.stage1_ckpt_path=outputs/scratch_moe/stage2/<ts>_<id>/checkpoints/checkpoint_best.pt

# Terminal D — Oracle MoE
phaseforge-eval models=baselines/oracle_moe `
    train.stage1_ckpt_path=outputs/oracle_moe/stage2/<ts>_<id>/checkpoints/checkpoint_best.pt

# Terminal E — WarmStart MoE
phaseforge-eval models=baselines/warmstart_moe `
    train.stage1_ckpt_path=outputs/warmstart_moe/stage2/<ts>_<id>/checkpoints/checkpoint_best.pt
```

Each evaluation writes results to:
```
outputs/eval/<model_name>/<ts>_<id>/
└── eval_results.json
```

---

## Expected Output Structure (after full experiment)

```
outputs/
├── phaseforge/
│   ├── stage1/
│   │   └── 2026-07-17_12-00-00_a1b2c3d4/
│   │       ├── resolved_config.yaml
│   │       ├── run_meta.json
│   │       └── checkpoints/
│   │           ├── checkpoint_best.pt
│   │           └── checkpoint_epoch_0010.pt
│   └── stage2/
│       └── 2026-07-17_16-00-00_e5f6g7h8/
│           ├── resolved_config.yaml
│           ├── run_meta.json
│           └── checkpoints/
│               ├── checkpoint_best.pt
│               └── checkpoint_epoch_0020.pt
├── bc/
│   └── stage1/
│       └── 2026-07-17_12-00-01_i9j0k1l2/
│           ├── resolved_config.yaml
│           ├── run_meta.json
│           └── checkpoints/
│               ├── checkpoint_best.pt
│               └── checkpoint_epoch_0010.pt
├── scratch_moe/
│   └── stage2/
│       └── 2026-07-17_12-00-02_m3n4o5p6/
│           └── ...
├── oracle_moe/
│   └── stage2/
│       └── 2026-07-17_12-00-03_q7r8s9t0/
│           └── ...
├── warmstart_moe/
│   └── stage2/
│       └── 2026-07-17_16-30-00_u1v2w3x4/
│           └── ...
└── eval/
    ├── phaseforge/
    │   └── 2026-07-17_18-00-00_y5z6a7b8/
    │       └── eval_results.json
    ├── bc/
    │   └── ...
    ├── scratch_moe/
    │   └── ...
    ├── oracle_moe/
    │   └── ...
    └── warmstart_moe/
        └── ...
```

---

## Verifying Runs

### During training
```bash
# Check training logs (printed to console)
# Check output dir was created:
Get-ChildItem outputs/<model_name>/stage<N>/

# Check run_meta.json for quick info:
Get-Content outputs/phaseforge/stage1/*/run_meta.json | ConvertFrom-Json
```

### After training
```bash
# Quick overview of all checkpoints:
python scripts/find_checkpoint.py list-all

# Validate all checkpoints are loadable:
Get-ChildItem outputs/*/stage*/*/checkpoints/checkpoint_best.pt | ForEach-Object {
    python scripts/find_checkpoint.py validate $_
}

# Compare eval results across models:
Get-ChildItem outputs/eval/*/*/eval_results.json | ForEach-Object {
    Write-Host "--- $_ ---"
    Get-Content $_ | ConvertFrom-Json
}
```

---

## Tips

- **Tags:** Append `project.tag=<label>` to any command to label the run directory, e.g.:
  ```bash
  phaseforge-train models=phaseforge train=stage1 project.tag=main
  # Creates: outputs/phaseforge/stage1/2026-07-17_12-00-00_main_a1b2c3d4/
  ```
- **W&B:** Set `project.wandb.mode=online` to enable W&B logging
- **Manual checkpoint:** If auto-detect picks the wrong checkpoint, specify it explicitly:
  ```bash
  # Use the discovery script to get the exact path:
  $ckpt = python scripts/find_checkpoint.py latest --model phaseforge --stage 1
  phaseforge-train models=phaseforge train=stage2 train.stage1_ckpt_path=$ckpt

  # Or type the path manually:
  phaseforge-train models=phaseforge train=stage2 `
      train.stage1_ckpt_path=outputs/phaseforge/stage1/<ts>_<id>/checkpoints/checkpoint_best.pt
  ```
- **Resume from crash:** The auto-detect always picks the latest run, so just re-run the same command
- **Checkpoint discovery script:** Use ``python scripts/find_checkpoint.py --help`` for all available commands
