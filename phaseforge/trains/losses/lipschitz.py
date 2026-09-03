"""Local contraction (Lipschitz) regularization (WP6, Professor §8).

Biases each impedance expert toward locally stabilizing behavior by
penalizing target maps that move faster than the task state::

    L_lip = E_{(i,j)} [ max(0, ‖T(x_i) − T(x_j)‖₂ / (‖y_i − y_j‖₂ + ε) − ρ)² ]

Pairs ``(i, j)`` share a predicted expert (same-regime pairs); the target
is ``‖∂T/∂y‖ ≤ ρ < 1``.

> The contraction term provides a local stabilizing bias and a useful
> diagnostic. It does not by itself certify global closed-loop stability
> in robosuite.
"""

from __future__ import annotations

import torch
from torch import Tensor


def lip_penalty(
    targets: Tensor,
    task_states: Tensor,
    expert_index: Tensor,
    rho: float = 0.8,
    eps: float = 1e-6,
    num_pairs: int = 256,
) -> Tensor:
    """Lipschitz penalty over deterministic within-expert pairs.

    Args:
        targets: Per-sample expert targets ``(B, Dy)``.
        task_states: Per-sample task states ``(B, Dy)``.
        expert_index: Per-sample predicted expert ``(B,)`` long.
        rho: Contraction target (must satisfy ``0 < rho < 1``... validated
            as ``0 <= rho < 1`` so ``rho=0`` stays expressible in sweeps).
        eps: Floor avoiding division by zero.
        num_pairs: Cap on evaluated pairs (first-N deterministic order).

    Pairs are consecutive samples sharing an expert (batch order), so the
    result is exactly reproducible with no RNG. Fewer than one pair (or no
    finite ratio) returns exactly ``0.0``.
    """
    if float(rho) < 0.0 or float(rho) >= 1.0:
        raise ValueError(f"rho must satisfy 0 <= rho < 1, got {rho}.")
    if targets.ndim != 2 or task_states.ndim != 2:
        raise ValueError("targets and task_states must be (B, D) tensors.")
    if targets.shape != task_states.shape:
        raise ValueError("targets and task_states must share their shape.")
    flat_experts = expert_index.reshape(-1).long()
    if flat_experts.numel() != targets.size(0):
        raise ValueError("expert_index must have one entry per sample.")
    cap = max(0, int(num_pairs))
    ratios: list[Tensor] = []
    for regime in torch.unique(flat_experts).tolist():
        positions = (flat_experts == int(regime)).nonzero().flatten()
        for first, second in zip(positions[:-1:2], positions[1::2]):
            delta_target = (targets[first] - targets[second]).norm()
            delta_state = (task_states[first] - task_states[second]).norm()
            ratios.append(delta_target / (delta_state + float(eps)))
            if len(ratios) >= cap > 0:
                break
        if len(ratios) >= cap > 0:
            break
    if not ratios or cap == 0:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)
    stacked = torch.stack(ratios)
    if not torch.isfinite(stacked).all():
        return torch.zeros((), device=targets.device, dtype=targets.dtype)
    excess = torch.clamp(stacked - float(rho), min=0.0)
    return (excess * excess).mean()


def gain_penalty(
    gains: Tensor,
    kappa_nominal: float = 1.0,
) -> Tensor:
    """Gain regularization ``E‖κ − κ_nom‖²`` against pathological stiffness."""
    nominal = float(kappa_nominal)
    if nominal <= 0.0:
        raise ValueError(f"kappa_nominal must be > 0.0, got {nominal}.")
    if not torch.isfinite(gains).all():
        raise ValueError("Non-finite gains in gain regularization.")
    return ((gains - nominal) ** 2).mean()


__all__ = ["gain_penalty", "lip_penalty"]
