"""Scratch MoE baseline."""

from __future__ import annotations

from torch import Tensor

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.router import TopKRouter


class ScratchMoEModel(BaseManipulationModel):
    """MoE trained entirely from scratch (Stage 2 only).
    
    No Stage 1 pretraining. Random initialization for everything.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        router: TopKRouter,
        expert: ExpertMLP,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.moe_layer = MoELayer(router=router, experts=expert)
        self._last_gate_logits: Tensor | None = None

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        latent = self.encoder(state)
        
        moe_out = self.moe_layer(latent)
        self._last_gate_logits = moe_out.gate_logits.detach()
        
        return ModelOutput(
            action_pred=moe_out.combined_output,
            phase_logits=None,
            routing_weights=moe_out.routing_weights,
            expert_indices=moe_out.expert_indices,
            gate_logits=moe_out.gate_logits,
            aux_losses={"balance": moe_out.balance_loss},
        )

    def get_action(self, state: Tensor) -> Tensor:
        latent = self.encoder(state)
        moe_out = self.moe_layer(latent)
        return moe_out.combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}
