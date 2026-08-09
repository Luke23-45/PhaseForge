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
        if num_phases < 1:
            raise ValueError(f"num_phases must be positive, got {num_phases}")
        if num_phases > router.num_experts:
            raise ValueError(
                f"Oracle routing requires at least one expert per phase: "
                f"num_phases={num_phases}, num_experts={router.num_experts}."
            )
        self._last_gate_logits: Tensor | None = None
        # Stage 2-only baseline (no Stage 1 exists). Kept as a plain
        # attribute so checkpointing/eval metadata records the right stage.
        self.stage = 2

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        phase = batch.get("phase")

        if phase is None:
            raise RuntimeError(
                "OraclePhaseMoEModel requires ground-truth 'phase' labels in "
                "every forward pass and never falls back to the (untrained) "
                "router: routing by it would silently corrupt the oracle "
                "upper bound. The oracle is a routing-signature reference "
                "only and is not deployable without phase labels."
            )

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
        if phase.numel() and (phase.min() < 0 or phase.max() >= self.num_phases):
            raise ValueError(
                f"Oracle phase labels must be in [0, {self.num_phases - 1}], "
                f"got range [{int(phase.min())}, {int(phase.max())}]."
            )
        expert_indices = phase.unsqueeze(-1)  # (B, 1)
        
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

    def get_action(self, state: Tensor) -> Tensor:
        """Label-free inference path (rollouts).

        Routing falls back to the router's gate, which was never trained for
        the oracle (routing is by GT phase during training). Rollout scores
        from this path are therefore NOT a policy-deployable signal — the
        oracle is a routing-signature reference only.
        """
        latent = self.encoder(state)
        moe_out = self.moe_layer(latent)
        return moe_out.combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}
