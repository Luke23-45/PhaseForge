"""Run full multi-seed training for all PhaseForge model variants.

Trains every model for every seed so that downstream evaluation can report
mean +/- std over training seeds. Stage 2 runs auto-detect the SAME seed's
Stage 1 checkpoint via seed-matched lookup
(``phaseforge.utils.config.find_latest_checkpoint``), so each seed trains an
independent, consistent pipeline.

Cell inventory (issues register C1/C7):
    * Stage 1 only:  bc (plain), phaseforge (phase-supervised).
    * Stage 2 only:  scratch_moe, oracle_moe, warmstart_moe (plain encoder
      x random router), phase_pretrain_random_router (phase encoder x
      random router), plain_encoder_phase_bootstrap (plain encoder x
      centroid router), teacher_forced (E8: GT-partitioned experts,
      predicted-phase routing at inference).

Usage:
    uv run python scripts/run_multi_seed_train.py

Requires:
    - Real robomimic low-dimensional data cache built by the data pipeline before the first
      training run (the pipeline prints the config hash; stage-2 lookups
      are seed-matched automatically).
    - CUDA machine (run on the GPU box, not a CPU laptop).
"""

from __future__ import annotations

import subprocess
import sys

# Model config -> list of training stages to run (BC is Stage 1 only).
#
# The 2x2 factorial cells (C1) and the teacher-forced cell (E8) run Stage 2
# only: their Stage 1 checkpoints are shared (resolve_checkpoint_source)
# with ``phaseforge`` (phase-supervised) or ``bc`` (plain) — which must be
# trained first, hence the dict ordering below.
MODEL_STAGES: dict[str, tuple[str, list[int]]] = {
    "bc": ("baselines/bc", [1]),
    "phaseforge": ("phaseforge", [1, 2]),
    "scratch_moe": ("baselines/scratch_moe", [2]),
    "warmstart_moe": ("baselines/warmstart_moe", [2]),
    "oracle_moe": ("baselines/oracle_moe", [2]),
    "phase_pretrain_random_router": ("baselines/phase_pretrain_random_router", [2]),
    "plain_encoder_phase_bootstrap": ("baselines/plain_encoder_phase_bootstrap", [2]),
    "teacher_forced": ("baselines/teacher_forced", [2]),
}

SEEDS = [42, 43, 44]


def run_train(model_cfg: str, stage: int, seed: int) -> None:
    cmd = [
        "phaseforge-train",
        f"models={model_cfg}",
        f"train=stage{stage}",
        f"project.seed={seed}",
        # Locked protocol (novelty_claim E2): full-length schedules with NO
        # truncating early stop. The stage yamls default early_stopping to
        # enabled (patience 10), which would silently truncate the protocol
        # runs — disable it explicitly.
        "train.early_stopping.enabled=false",
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
