# PhaseForge

A PyTorch-based state-only robotic manipulation framework testing the **Phase-Bootstrapped Mixture-of-Experts** hypothesis for long-horizon tasks.

## Setup

```bash
uv sync
uv run phaseforge-train --help
```

## Training

```bash
# Stage 1: Phase-supervised generalist pretraining
uv run phaseforge-train models=phaseforge train=stage1

# Stage 2: MoE bootstrapping (auto-detects latest Stage 1 checkpoint)
uv run phaseforge-train models=phaseforge train=stage2

# Or specify a specific Stage 1 checkpoint explicitly
uv run phaseforge-train models=phaseforge train=stage2 \
    train.stage1_ckpt_path=outputs/phaseforge/stage1/2026-07-17_12-00-00_a1b2/checkpoints/checkpoint_best.pt

# Add a custom tag to label the run (optional)
uv run phaseforge-train models=phaseforge train=stage1 project.tag=lr3e-4

# Baselines
uv run phaseforge-train models=baselines/bc train=stage1
uv run phaseforge-train models=baselines/scratch_moe train=stage2
```

## Evaluation

```bash
uv run phaseforge-eval models=phaseforge
```

## Output Structure

```
outputs/
├── <model_name>/             # phaseforge, bc, scratch_moe, …
│   └── stage<N>/             # 1 or 2
│       └── <timestamp>[_<tag>]_<run_id>/   # unique per run
│           ├── checkpoints/
│           │   ├── checkpoint_best.pt
│           │   └── checkpoint_epoch_XXXX.pt
│           ├── resolved_config.yaml
│           ├── run_meta.json               # quick inspection
│           └── wandb/
```
