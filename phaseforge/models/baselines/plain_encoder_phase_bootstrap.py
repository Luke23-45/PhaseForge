"""Plain-encoder MoE with a centroid-bootstrapped router (C1 cell).

Completes the 2x2 factorial (issues register C1):

    encoder x router     centroid (bootstrap)   random init
    phase-supervised     phaseforge             phase_pretrain_random_router
    plain (BC)           plain_encoder_phase_bootstrap   warmstart_moe

This cell shares the plain BC Stage 1 checkpoint with ``warmstart_moe``
(``resolve_checkpoint_source`` maps it to ``bc``) but runs the *centroid*
bootstrap: phase centroids computed over the plain encoder's latent space
initialize the router. Comparing it against ``warmstart_moe`` isolates the
effect of the centroid router init without phase supervision; comparing it
against ``phaseforge`` isolates the effect of phase supervision in the
pretraining encoder given the same centroid init.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import ExpertMLP
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.router import TopKRouter

logger = logging.getLogger(__name__)


class PlainEncoderPhaseBootstrapModel(BaseManipulationModel):
    """MoE bootstrapped from a plain (BC) encoder using phase centroids.

    Structure mirrors :class:`WarmStartMoEModel` (no phase head — the BC
    checkpoint has none), but ``bootstrap_moe`` initializes the router's
    gate weights with phase latent centroids exactly like
    :class:`PhaseBootstrappedMoE`. The phase structure is therefore induced
    into the plain encoder's latent space via the router init alone.

    Args:
        encoder: The StateEncoder instance (loaded from the BC Stage 1 ckpt).
        action_head: The ActionHead used in Stage 1.
        router: The TopKRouter for Stage 2.
        expert: A single ExpertMLP template to be cloned for Stage 2.
        num_phases: Number of phases used by the phase labeler (drives the
            centroid computation; there is no phase head in this model).
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        router: TopKRouter,
        expert: ExpertMLP,
        num_phases: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head
        self.moe_layer = MoELayer(router=router, experts=expert)
        self.num_phases = num_phases
        self._stage = 1
        self._last_gate_logits: Tensor | None = None

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError(f"Stage must be 1 or 2, got {value}")
        self._stage = value
        logger.info(f"PlainEncoderPhaseBootstrapModel transitioned to Stage {value}.")

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen.")

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
            raise RuntimeError(f"Invalid stage {self._stage}")

    def get_action(self, state: Tensor) -> Tensor:
        latent = self.encoder(state)
        if self._stage == 1:
            return self.action_head(latent)
        return self.moe_layer(latent).combined_output

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        if self._stage == 2 and self._last_gate_logits is None:
            logger.warning("get_routing_info called before any forward passes.")
        if self._stage == 1 or self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}

    @torch.no_grad()
    def bootstrap_moe(self, dataloader: DataLoader, device: torch.device | str = "cuda") -> None:
        """Centroid-bootstrap a MoE on top of the plain (BC) encoder.

        Identical algorithm to :class:`PhaseBootstrappedMoE.bootstrap_moe`:
        1. Compute phase centroids in the (plain) encoder's latent space.
        2. Initialize the router gate weights with the normalized centroids.
        3. Warm-start the experts from the Stage 1 ActionHead weights.

        Args:
            dataloader: Training dataloader to compute centroids over.
            device: Compute device.
        """
        logger.info("Starting centroid bootstrap over the plain (BC) encoder...")
        self.to(device)
        self.eval()

        num_experts = self.moe_layer.router.num_experts
        latent_dim = self.encoder.latent_dim

        if self.num_phases != num_experts:
            logger.warning(
                f"Number of phases ({self.num_phases}) != number of experts "
                f"({num_experts}). Phase-bootstrapping works best when E >= P. "
                "Centroids will be mapped 1:1 for the first P experts, the rest "
                "remain random."
            )

        # 1. Compute latent centroids
        phase_sums = torch.zeros((self.num_phases, latent_dim), device=device)
        phase_counts = torch.zeros((self.num_phases,), device=device)

        for batch in dataloader:
            state = batch["state"].to(device)
            phase = batch["phase"].to(device)

            if state.ndim == 3:
                state = state.view(-1, state.size(-1))
                phase = phase.view(-1)

            latent = self.encoder(state)
            phase_expanded = phase.unsqueeze(1).expand_as(latent)
            phase_sums.scatter_add_(0, phase_expanded, latent)
            counts = torch.bincount(phase, minlength=self.num_phases).float()
            phase_counts += counts

        phase_counts = torch.clamp(phase_counts, min=1.0)
        centroids = phase_sums / phase_counts.unsqueeze(1)  # (P, D)

        logger.info(f"Computed latent centroids for {self.num_phases} phases.")

        # 2. Initialize Router with the (normalized) centroids
        centroids_normalized = torch.nn.functional.normalize(centroids, p=2, dim=-1)

        router_weight = self.moe_layer.router.gate_linear.weight.data
        router_bias = self.moe_layer.router.gate_linear.bias.data

        limit = min(self.num_phases, num_experts)
        router_weight[:limit] = centroids_normalized[:limit]
        router_bias.zero_()

        logger.info(f"Initialized router weights with {limit} phase centroids.")

        # 3. Initialize Experts with ActionHead weights
        action_head_state_dict = self.action_head.state_dict()
        mapping = {
            "trunk.0.weight": "hidden.0.weight",
            "trunk.0.bias": "hidden.0.bias",
            "mean_head.weight": "output_proj.weight",
            "mean_head.bias": "output_proj.bias",
        }

        for i, expert in enumerate(self.moe_layer.experts):
            expert_dict = expert.state_dict()
            new_dict = {}
            for src_k, dst_k in mapping.items():
                if src_k in action_head_state_dict and dst_k in expert_dict:
                    new_dict[dst_k] = action_head_state_dict[src_k].clone()
            expert.load_state_dict(new_dict, strict=False)

        logger.info("Initialized all experts with Stage 1 ActionHead weights.")

        # Automatically transition to Stage 2
        self.stage = 2
        logger.info("Centroid bootstrap complete. Ready for Stage 2.")
