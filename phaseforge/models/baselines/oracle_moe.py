"""Oracle Phase MoE baseline."""

from __future__ import annotations

import torch
from torch import Tensor

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.router import TopKRouter


class OraclePhaseMoEModel(BaseManipulationModel):
    """MoE trained with Oracle routing (ground truth phases).
    
    During training, the router is bypassed, and the ground truth phase
    labels are used to perfectly select the corresponding expert.
    Provides an upper bound on performance.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        router: TopKRouter,
        expert: ExpertMLP,
        num_phases: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        
        # We enforce deterministic routing, Top-1
        router.top_k = 1
        router.noise_std = 0.0
        
        self.moe_layer = MoELayer(router=router, experts=expert)
        self.num_phases = num_phases
        self._last_gate_logits: Tensor | None = None

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        phase = batch.get("phase")
        
        if phase is None or not self.training:
            # During eval (or if no phases provided), we must use the router
            return self._standard_forward(state)

        # ORACLE ROUTING (Training)
        latent = self.encoder(state)
        B = latent.size(0)
        
        # Flatten time dim if sequence
        if latent.ndim == 3:
            latent = latent.view(-1, latent.size(-1))
            phase = phase.view(-1)
            B = latent.size(0)

        # Ensure E >= P for oracle mapping
        E = self.moe_layer.router.num_experts
        
        # Clamp phases to available experts (fallback)
        expert_indices = torch.clamp(phase, max=E-1).unsqueeze(-1)  # (B, 1)
        
        # Oracle weights are 1.0 (perfect certainty)
        routing_weights = torch.ones((B, 1), device=latent.device)  # (B, 1)
        
        # Generate dummy logits for metric compatibility
        gate_logits = torch.zeros((B, E), device=latent.device)
        gate_logits.scatter_(1, expert_indices, 100.0) # Highly peaked
        self._last_gate_logits = gate_logits.detach()

        # Gather output
        out_dim = self.moe_layer.experts[0].output_dim
        combined_output = torch.zeros((B, out_dim), device=latent.device)

        for expert_idx, expert_net in enumerate(self.moe_layer.experts):
            match_mask = (expert_indices == expert_idx).squeeze(-1)
            if not match_mask.any():
                continue
            
            batch_idx = torch.where(match_mask)[0]
            expert_inputs = latent[batch_idx]
            expert_outputs = expert_net(expert_inputs)
            
            combined_output.index_copy_(0, batch_idx, expert_outputs)

        return ModelOutput(
            action_pred=combined_output,
            phase_logits=None,
            routing_weights=routing_weights,
            expert_indices=expert_indices,
            gate_logits=gate_logits,
            # No balance loss needed for oracle routing
            aux_losses={"balance": torch.tensor(0.0, device=latent.device)},
        )

    def _standard_forward(self, state: Tensor) -> ModelOutput:
        """Used during evaluation when oracle labels are not guaranteed."""
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
        # Standard inference (no oracle)
        latent = self.encoder(state)
        moe_out = self.moe_layer(latent)
        return moe_out.combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}
