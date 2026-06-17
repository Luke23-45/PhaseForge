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

# Stage 2: MoE bootstrapping (requires Stage 1 checkpoint)
uv run phaseforge-train models=phaseforge train=stage2 train.stage1_ckpt_path=outputs/.../checkpoint_best.pt

# Baselines
uv run phaseforge-train models=baselines/bc train=stage1
uv run phaseforge-train models=baselines/scratch_moe train=stage2
```

## Evaluation

```bash
uv run phaseforge-eval models=phaseforge train.stage1_ckpt_path=outputs/.../checkpoint_best.pt
```
