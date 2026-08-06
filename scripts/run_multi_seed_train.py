"""Run full multi-seed training for all PhaseForge model variants.

Trains every model for every seed so that downstream evaluation can report
mean +/- std over training seeds. Stage 2 runs auto-detect the SAME seed's
Stage 1 checkpoint via seed-matched lookup
(``phaseforge.utils.config.find_latest_checkpoint``), so each seed trains an
independent, consistent pipeline.

Usage:
    uv run python scripts/run_multi_seed_train.py

Requires:
    - Real LIBERO-90 data cache (config hash a4c74be17f117a4b) built by the
      data pipeline before the first training run.
    - CUDA machine (run on the GPU box, not a CPU laptop).
"""

from __future__ import annotations

import subprocess
import sys

# Model config -> list of training stages to run (BC is Stage 1 only).
MODEL_STAGES: dict[str, tuple[str, list[int]]] = {
    "bc": ("baselines/bc", [1]),
    "phaseforge": ("phaseforge", [1, 2]),
    "scratch_moe": ("baselines/scratch_moe", [2]),
    "warmstart_moe": ("baselines/warmstart_moe", [2]),
    "oracle_moe": ("baselines/oracle_moe", [2]),
}

SEEDS = [42, 43, 44]


def run_train(model_cfg: str, stage: int, seed: int) -> None:
    cmd = [
        "phaseforge-train",
        f"models={model_cfg}",
        f"train=stage{stage}",
        f"project.seed={seed}",
    ]
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! FAILED (rc={result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode or 1)


def main() -> None:
    for seed in SEEDS:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        for model_name, (model_cfg, stages) in MODEL_STAGES.items():
            print(f"\n--- Model: {model_name} ---")
            for stage in stages:
                run_train(model_cfg, stage, seed)


if __name__ == "__main__":
    main()
