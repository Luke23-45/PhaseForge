"""Run the balance-weight sweep on ``phaseforge`` (issues register C3).

Tests the pseudo-balancing mechanism: balance weight beta in {0, 0.01, 0.1}
on the proposed method, logging the balance-vs-NMI trajectory per epoch
(Stage2Trainer now logs ``val/phase_expert_nmi`` versus explicitly named
``val/topk_balance_score`` and ``val/top1_balance_score``
every epoch). If balance kills NMI at ALL weights, the bootstrap claim needs
the orthogonal-basis direction (SMP) or decoupling (AdaMoE).

NOTE on the effective knob: the router applies its own
``models.router.balance_coeff`` internally (``TopKRouter.forward``), so
that config key — NOT ``train.balance_coeff`` (dead config) — controls the
balance loss weight.

Usage:
    uv run python scripts/run_balance_sweep.py

Requires:
    - ``phaseforge`` Stage 1 checkpoints (seed-matched auto-detected).
    - CUDA machine (run on the GPU box, not a CPU laptop).
"""

from __future__ import annotations

import subprocess
import sys

BALANCE_WEIGHTS = [0.0, 0.01, 0.1]

SEEDS = [42, 43, 44]


def run_sweep(balance_coeff: float, seed: int) -> None:
    cmd = [
        "phaseforge-train",
        "models=phaseforge",
        "train=stage2",
        f"project.seed={seed}",
        f"models.router.balance_coeff={balance_coeff}",
        f"project.tag=balance_{balance_coeff}",
        # Full-length runs only: the sweep compares final-epoch routing
        # states, so early stopping must not truncate any cell.
        "train.early_stopping.enabled=false",
    ]
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!!! FAILED (rc={result.returncode}): {' '.join(cmd)}")
        sys.exit(result.returncode or 1)


def main() -> None:
    for balance_coeff in BALANCE_WEIGHTS:
        print(f"\n{'='*70}\nBALANCE WEIGHT {balance_coeff}\n{'='*70}")
        for seed in SEEDS:
            run_sweep(balance_coeff, seed)


if __name__ == "__main__":
    main()
