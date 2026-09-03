"""Impedance-parameterized Behavior Cloning baseline (WP5, Professor §7.5).

Action-matched control for the decisive comparison: shares the exact
``ψ → (T, κ) → adapter`` action parameterization with IS-PhaseForge but
has no routing (a single impedance expert). Any remaining gap to the MoE
is therefore attributable to routing, never to the action representation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_adapter import impedance_action
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.impedance_expert import ImpedanceExpert
from phaseforge.models.components.task_state import extract_task_state


class BCImpedanceModel(BaseManipulationModel):
    """Single-expert impedance BC: encoder → ImpedanceExpert → adapter."""

    def __init__(
        self,
        encoder: StateEncoder,
        expert: ImpedanceExpert,
    ) -> None:
        super().__init__()
        if int(expert.output_dim) != 7:
            raise ValueError(
                "BCImpedanceModel needs a 7D impedance expert "
                f"(single-arm action contract), got {expert.output_dim}."
            )
        self.encoder = encoder
        self.expert = expert

    def _act(
        self, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        latent = self.encoder(state)
        task_state = extract_task_state(state)
        target, gains = self.expert.params(latent)
        action, parts = impedance_action(target, gains, task_state, self.expert.action_scale)
        info = {
            "target": target,
            "gains": gains,
            "task_error": parts["task_error"],
            "pre_clip_u": parts["pre_clip_u"],
            "task_state": task_state,
            "expert_index": torch.zeros(
                state.shape[0], dtype=torch.long, device=state.device
            ),
        }
        return action, latent, target, info

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        action_pred, latent, _target, info = self._act(batch["state"])
        return ModelOutput(
            action_pred=action_pred,
            phase_logits=None,
            routing_weights=None,
            expert_indices=None,
            gate_logits=None,
            latent=latent,
            info=info,
        )

    def get_action(self, state: Tensor) -> Tensor:
        action, _latent, _target, _info = self._act(state)
        return action

    @torch.no_grad()
    def describe_step(self, state: Tensor) -> dict[str, Any]:
        """Snapshot one inference step for full rollout tracing (WP8-full).

        Single-expert counterpart of ``PhaseBootstrappedMoE.describe_step``:
        no routing fields (all ``None``), impedance ``(T, κ, e, u)`` filled.
        """
        latent = self.encoder(state)
        task_state = extract_task_state(state)
        target, gains = self.expert.params(latent)
        _action, parts = impedance_action(target, gains, task_state, self.expert.action_scale)
        return {
            "latent_norm": latent.norm(dim=-1),
            "dists": None,
            "selected_expert": torch.zeros(state.shape[0], dtype=torch.long),
            "top2_expert": None,
            "router_margin": None,
            "router_entropy": None,
            "task_vars": task_state,
            "expert_target": target,
            "expert_gains": gains,
            "task_error": parts["task_error"],
            "pre_clip_u": parts["pre_clip_u"],
            "expert_disagreement": None,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def deployment_contract(self) -> dict[str, Any]:
        """Memoryless single-expert impedance contract."""
        return {"memoryless": True, "router_type": "none", "expert_type": "impedance"}


__all__ = ["BCImpedanceModel"]
