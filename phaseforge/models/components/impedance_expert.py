"""ImpedanceExpert: per-expert local feedback-controller parameters (WP5).

Each expert maps the shared latent to a target task state and positive
feedback gains (Professor §7.2)::

    T_k(z_t) = target task state      (Dy = 8: pos 3, quat 4, gripper 1)
    κ_k(z_t) = positive feedback gains (De = 7: pos 3, rot 3, gripper 1)

Positivity (``κ > 0``) is enforced by softplus plus ``[kappa_min,
kappa_max]`` clamping; a non-positive gain would flip the feedback sign
and is rejected downstream as well (see ``impedance_action``).

Unlike :class:`ExpertMLP` (which regresses absolute 7D actions), this
head cannot be warm-started from the Stage 1 ``ActionHead`` — the output
structure differs — so IS-PhaseForge uses random expert init. Stage 1
keeps its direct action head for warm-starting the shared encoder only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from phaseforge.models.components.action_adapter import impedance_action
from phaseforge.models.components.task_state import ACTION_DIM, TASK_ERROR_DIM, TASK_STATE_DIM


class ImpedanceExpert(nn.Module):
    """One impedance-parameterized expert.

    Args:
        input_dim: Dimension of the latent vector from the encoder.
        hidden_dim: Width of the shared trunk hidden layer.
        task_state_dim: Target task-state width (default 8).
        error_dim: Gain/error width (default 7).
        action_dim: Environment action width (default 7).
        action_scale: Command scale ``s`` in ``a = tanh(u / s)`` (> 0).
        kappa_min: Lower gain clamp (must be > 0).
        kappa_max: Upper gain clamp (>= kappa_min).
    """

    IS_IMPEDANCE = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        task_state_dim: int = TASK_STATE_DIM,
        error_dim: int = TASK_ERROR_DIM,
        action_dim: int = ACTION_DIM,
        action_scale: float = 1.0,
        kappa_min: float = 0.1,
        kappa_max: float = 5.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if task_state_dim < 1 or error_dim < 1 or action_dim < 1:
            raise ValueError("task/error/action dims must be positive.")
        if float(action_scale) <= 0.0:
            raise ValueError(f"action_scale must be > 0.0, got {action_scale}.")
        kappa_lo, kappa_hi = float(kappa_min), float(kappa_max)
        if kappa_lo <= 0.0 or kappa_hi < kappa_lo:
            raise ValueError(f"Require 0 < kappa_min <= kappa_max, got {kappa_lo, kappa_hi}.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.task_state_dim = int(task_state_dim)
        self.error_dim = int(error_dim)
        self.output_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.kappa_min = kappa_lo
        self.kappa_max = kappa_hi

        self.trunk = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.GELU())
        self.target_head = nn.Linear(self.hidden_dim, self.task_state_dim)
        self.gain_head = nn.Linear(self.hidden_dim, self.error_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
                nn.init.zeros_(module.bias)

    def reset_parameters(self) -> None:
        """Re-initialize all weights (symmetry-breaking, MoE template clones)."""
        self._init_weights()

    def params(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """Predict ``(target (..., Dy), gains (..., De))`` with ``κ > 0``."""
        hidden = self.trunk(latent)
        target = self.target_head(hidden)
        gains = torch.nn.functional.softplus(self.gain_head(hidden))
        gains = gains.clamp(min=self.kappa_min, max=self.kappa_max)
        return target, gains

    def forward(self, latent: Tensor, task_state: Tensor | None = None) -> Tensor:
        """Predict a clipped action from ``(latent, task_state)``.

        ``task_state`` is required (the feedback error needs ``y_t``);
        a missing value fails closed instead of silently reusing zeros.
        """
        if task_state is None:
            raise ValueError(
                "ImpedanceExpert.forward needs task_state=y_t; pass the task "
                "state alongside the latent (MoELayer threads it through)."
            )
        target, gains = self.params(latent)
        action, _info = impedance_action(target, gains, task_state, self.action_scale)
        return action


__all__ = ["ImpedanceExpert"]
