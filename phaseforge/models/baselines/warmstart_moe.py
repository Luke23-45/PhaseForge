"""Warm-Start MoE baseline."""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.router import TopKRouter
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.moe_layer import MoELayer

logger = logging.getLogger(__name__)


class WarmStartMoEModel(BaseManipulationModel):
    """MoE trained with a Warm-Start approach.
    
    Stage 1: Pretrain encoder + action_head (λ_phase = 0).
    Stage 2: Bootstrap MoE, but with random router init (no phase centroids).
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        router: TopKRouter,
        expert: ExpertMLP,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head
        self.moe_layer = MoELayer(router=router, experts=expert)
        self._stage = 1
        self._last_gate_logits: Tensor | None = None

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        self._stage = value

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        state = batch["state"]
        latent = self.encoder(state)

        if self._stage == 1:
            action_pred = self.action_head(latent)
            return ModelOutput(
                action_pred=action_pred,
                phase_logits=None,
                routing_weights=None,
                expert_indices=None,
                gate_logits=None,
            )
        elif self._stage == 2:
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
        else:
            raise RuntimeError("Invalid stage")

    def get_action(self, state: Tensor) -> Tensor:
        latent = self.encoder(state)
        if self._stage == 1:
            return self.action_head(latent)
        else:
            return self.moe_layer(latent).combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._stage == 1 or self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}

    @torch.no_grad()
    def bootstrap_moe(self, dataloader: DataLoader, device: torch.device | str = "cuda") -> None:
        """Standard warm-start: Initialize experts from ActionHead, but leave router random."""
        self.to(device)
        
        # 1. Router remains randomly initialized (standard MoE initialization)
        logger.info("WarmStartMoE: Leaving router randomly initialized.")

        # 2. Initialize Experts with ActionHead weights
        action_head_state_dict = self.action_head.state_dict()
        for i, expert in enumerate(self.moe_layer.experts):
            expert_dict = expert.state_dict()
            mapping = {
                "trunk.0.weight": "hidden.0.weight",
                "trunk.0.bias": "hidden.0.bias",
                "mean_head.weight": "output_proj.weight",
                "mean_head.bias": "output_proj.bias",
            }
            new_dict = {}
            for src_k, dst_k in mapping.items():
                if src_k in action_head_state_dict and dst_k in expert_dict:
                    new_dict[dst_k] = action_head_state_dict[src_k].clone()
            expert.load_state_dict(new_dict, strict=False)
            
        logger.info("WarmStartMoE: Initialized all experts with Stage 1 ActionHead weights.")
        self.stage = 2
