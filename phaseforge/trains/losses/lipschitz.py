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
    trajectory_ids: Tensor | None = None,
    rho: float = 0.8,
    eps: float = 1e-6,
    num_pairs: int = 256,
    max_ratio: float = 10.0,
    min_delta_state: float = 1e-4,
) -> Tensor:
    """Lipschitz penalty over deterministic within-expert pairs.

    Args:
        targets: Per-sample expert targets ``(B, Dy)``.
        task_states: Per-sample task states ``(B, Dy)``.
        expert_index: Per-sample predicted expert ``(B,)`` long.
        trajectory_ids: Optional per-sample trajectory index ``(B,)``. When provided,
            pairs are formed strictly within the same episode to avoid evaluating
            finite differences across diverging environmental contexts.
        rho: Contraction target (must satisfy ``0 <= rho < 1``).
        eps: Floor avoiding division by zero.
        num_pairs: Cap on evaluated pairs.
        max_ratio: Maximum clamped ratio to prevent gradient explosions.
        min_delta_state: Minimum task-state displacement to avoid singular division.
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
    if cap == 0:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)

    flat_trajs: Tensor | None = None
    if trajectory_ids is not None:
        flat_trajs = trajectory_ids.reshape(-1).long()

    ratios: list[Tensor] = []
    for regime in torch.unique(flat_experts).tolist():
        expert_mask = flat_experts == int(regime)
        if flat_trajs is not None:
            for tid in torch.unique(flat_trajs[expert_mask]).tolist():
                positions = (expert_mask & (flat_trajs == int(tid))).nonzero().flatten()
                for first, second in zip(positions[:-1:2], positions[1::2]):
                    delta_state = (task_states[first] - task_states[second]).norm()
                    if delta_state < float(min_delta_state):
                        continue
                    delta_target = (targets[first] - targets[second]).norm()
                    ratio = torch.clamp(delta_target / (delta_state + float(eps)), max=float(max_ratio))
                    ratios.append(ratio)
                    if len(ratios) >= cap:
                        break
                if len(ratios) >= cap:
                    break
        else:
            positions = expert_mask.nonzero().flatten()
            for first, second in zip(positions[:-1:2], positions[1::2]):
                delta_state = (task_states[first] - task_states[second]).norm()
                if delta_state < float(min_delta_state):
                    continue
                delta_target = (targets[first] - targets[second]).norm()
                ratio = torch.clamp(delta_target / (delta_state + float(eps)), max=float(max_ratio))
                ratios.append(ratio)
                if len(ratios) >= cap:
                    break
        if len(ratios) >= cap:
            break

    if not ratios:
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
