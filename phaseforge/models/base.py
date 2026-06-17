"""Abstract base class for all PhaseForge manipulation models.

Every model (BC, ScratchMoE, WarmStartMoE, OracleMoE, PhaseBootstrappedMoE)
MUST inherit from BaseManipulationModel and implement every abstract method.
The trainer interacts only through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ModelOutput:
    """Standardized output returned by every model's forward pass."""

    action_pred: Tensor
    """(B, A) or (B, T, A) — predicted action(s)."""

    phase_logits: Optional[Tensor] = None
    """(B, P) — raw phase classification logits. None for non-phase models."""

    routing_weights: Optional[Tensor] = None
    """(B, K) — top-k normalized gating weights. None for non-MoE models."""

    expert_indices: Optional[Tensor] = None
    """(B, K) — top-k expert indices. None for non-MoE models."""

    gate_logits: Optional[Tensor] = None
    """(B, E) — raw gate logits over all experts (for metric logging)."""

    aux_losses: dict[str, Tensor] = field(default_factory=dict)
    """Auxiliary losses: keys may include ``"balance"``, ``"phase"``."""


class BaseManipulationModel(nn.Module, ABC):
    """Shared interface for all PhaseForge model variants.

    Subclasses must implement:
        - :meth:`forward`
        - :meth:`get_action`
        - :meth:`num_parameters`
    """

    @abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        """Training forward pass.

        Args:
            batch: Dict with keys:
                - ``"state"``:   (B, state_dim)
                - ``"action"``:  (B, action_dim) — ground truth
                - ``"phase"``:   (B,) int — phase labels (may be ignored)
                - ``"task_id"``: (B,) int
        """
        ...

    @abstractmethod
    def get_action(self, state: Tensor) -> Tensor:
        """Inference-only path.

        No phase labels, no auxiliary losses, no gradients required.

        Args:
            state: (B, state_dim) or (1, state_dim)

        Returns:
            action: (B, action_dim)
        """
        ...

    @abstractmethod
    def num_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        ...

    def freeze_encoder(self) -> None:
        """Freeze the encoder sub-module. Default: no-op.

        Override in models that have a distinct encoder attribute.
        """

    def get_routing_info(self) -> Optional[dict[str, Tensor]]:
        """Return the most recent routing state for metric logging.

        Returns:
            Dict with ``"gate_logits": Tensor(B, E)`` or None if not an MoE model.
        """
        return None
