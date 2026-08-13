"""PhaseBootstrappedMoE: The core proposed model architecture."""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import (
    ExpertMLP,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.phase_head import PhaseClassificationHead
from phaseforge.models.components.router import TopKRouter

logger = logging.getLogger(__name__)


class PhaseBootstrappedMoE(BaseManipulationModel):
    """Phase-Bootstrapped Mixture-of-Experts.

    This model operates in two distinct stages:
    Stage 1: Generalist pretraining with auxiliary phase supervision.
             Forward pass uses encoder → action_head + phase_head.
    Stage 2: Bootstrapped MoE specialization.
             Forward pass uses encoder → moe_layer.

    The transition between stages is mediated by the `bootstrap_moe()` method,
    which computes latent centroids for each phase and initializes the router
    weights accordingly.

    Args:
        encoder: The StateEncoder instance.
        action_head: The ActionHead used in Stage 1.
        phase_head: The PhaseClassificationHead used in Stage 1.
        router: The TopKRouter for Stage 2.
        expert: A single ExpertMLP template to be cloned for Stage 2.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        phase_head: PhaseClassificationHead,
        router: TopKRouter,
        expert: ExpertMLP,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        
        # Stage 1 components
        self.action_head = action_head
        self.phase_head = phase_head
        
        # Stage 2 components
        self.moe_layer = MoELayer(router=router, experts=expert)
        
        # Internal state to track which stage the model is currently configured for
        self._stage = 1
        self._encoder_frozen = False
        
        # Storage for the most recent routing information for metrics tracking
        self._last_gate_logits: Tensor | None = None

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError(f"Stage must be 1 or 2, got {value}")
        self._stage = value
        logger.info(f"PhaseBootstrappedMoE transitioned to Stage {value}.")

    def freeze_encoder(self) -> None:
        """Freeze the encoder for Stage 2 training.

        Disables gradients AND keeps the encoder in eval mode: a frozen
        encoder must not apply dropout, or its latent representation stays
        stochastic during training despite being frozen.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()
        logger.info("Encoder weights frozen; encoder kept in eval mode (no dropout).")

    def train(self, mode: bool = True) -> PhaseBootstrappedMoE:
        """Override so a frozen encoder stays deterministic during Stage 2.

        The training loop calls ``model.train()`` every epoch, which would
        otherwise re-enable the encoder's dropout.
        """
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        """Forward pass depends on the active stage.

        Args:
            batch: Dictionary containing "state", "action", "phase", etc.

        Returns:
            ModelOutput containing predictions, logits, and auxiliary losses.
        """
        state = batch["state"]
        latent = self.encoder(state)

        if self._stage == 1:
            # Stage 1: Generalist action prediction + phase classification
            action_pred = self.action_head(latent)
            phase_logits = self.phase_head(latent)
            
            return ModelOutput(
                action_pred=action_pred,
                phase_logits=phase_logits,
                # MoE fields are empty in Stage 1
                routing_weights=None,
                expert_indices=None,
                gate_logits=None,
            )
            
        elif self._stage == 2:
            # Stage 2: MoE routing
            moe_out = self.moe_layer(latent)
            
            # Store gate logits for metric callbacks
            self._last_gate_logits = moe_out.gate_logits.detach()
            
            return ModelOutput(
                action_pred=moe_out.combined_output,
                phase_logits=None,  # Phase head is ignored in Stage 2
                routing_weights=moe_out.routing_weights,
                expert_indices=moe_out.expert_indices,
                gate_logits=moe_out.gate_logits,
                aux_losses={"balance": moe_out.balance_loss},
            )
        else:
            raise RuntimeError(f"Invalid stage {self._stage}")

    def get_action(self, state: Tensor) -> Tensor:
        """Inference path without auxiliary outputs or gradients."""
        latent = self.encoder(state)
        
        if self._stage == 1:
            return self.action_head(latent)
        else:
            moe_out = self.moe_layer(latent)
            return moe_out.combined_output

    def num_parameters(self) -> int:
        """Count all trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_routing_info(self) -> dict[str, Tensor] | None:
        """Expose raw routing logits for evaluation metrics."""
        if self._stage == 2 and self._last_gate_logits is None:
            logger.warning("get_routing_info called before any forward passes.")
        if self._stage == 1 or self._last_gate_logits is None:
            return None
        return {"gate_logits": self._last_gate_logits}

    @torch.no_grad()
    def bootstrap_moe(self, dataloader: DataLoader, device: torch.device | str = "cuda") -> None:
        """The core contribution: Bootstrapping the MoE from Stage 1 knowledge.

        This algorithm:
        1. Computes the centroid in latent space for every phase using the training data.
        2. Assigns each expert to a phase and initializes its router weights with that centroid.
        3. Copies the pre-trained ActionHead weights into each ExpertMLP to jumpstart them.

        Args:
            dataloader: Training dataloader to compute centroids over.
            device: Compute device.
        """
        logger.info("Starting MoE bootstrapping process...")
        self.to(device)
        self.eval()

        # We assume number of experts == number of phases for the default PhaseForge
        num_phases = self.phase_head.num_phases
        num_experts = self.moe_layer.router.num_experts
        latent_dim = self.encoder.latent_dim

        if num_phases != num_experts:
            logger.warning(
                f"Number of phases ({num_phases}) != number of experts ({num_experts}). "
                "Phase-bootstrapping works best when E >= P. Centroids will be mapped 1:1 "
                "for the first P experts, the rest remain random."
            )

        # 1. Compute latent centroids
        phase_sums = torch.zeros((num_phases, latent_dim), device=device)
        phase_counts = torch.zeros((num_phases,), device=device)
        non_blocking = torch.device(device).type == "cuda"

        for batch in dataloader:
            state = batch["state"].to(device, non_blocking=non_blocking)
            phase = batch["phase"].to(device, non_blocking=non_blocking)
            
            # Handle sequence length dimension if present
            if state.ndim == 3:
                state = state.view(-1, state.size(-1))
                phase = phase.view(-1)

            latent = self.encoder(state)

            # Scatter add the latents to the correct phase sum
            # Expand phase to match latent dims: (B, D)
            phase_expanded = phase.unsqueeze(1).expand_as(latent)
            phase_sums.scatter_add_(0, phase_expanded, latent)
            
            # Count occurrences of each phase
            counts = torch.bincount(phase, minlength=num_phases).float()
            phase_counts += counts

        # Compute mean
        # An absent phase would produce a zero centroid reported as if it had
        # one sample — that silently corrupts the router init, so it is a
        # hard failure instead.
        absent = phase_counts == 0
        if absent.any():
            raise ValueError(
                "Phase centroid bootstrap: "
                f"{int(absent.sum().item())} phase(s) have zero samples in the "
                f"bootstrap dataloader (missing: "
                f"{[int(i) for i in absent.nonzero().flatten().tolist()]}). "
                "Refusing to bootstrap with zero centroids; every phase must "
                "be present in the training data."
            )
        centroids = phase_sums / phase_counts.unsqueeze(1)  # (P, D)

        logger.info(f"Computed latent centroids for {num_phases} phases.")
        for p in range(num_phases):
            logger.debug(f"  Phase {p} count: {phase_counts[p].item()}")

        # 2. Initialize Router
        # The router's gate computes logits: L = x @ W^T + b. The centroids
        # are unit-normalized and the router is configured with
        # normalize_input=True, so L is a true cosine similarity between the
        # (normalized) latent and each (unit-norm) centroid — not a raw dot
        # product with an unnormalized latent.
        
        # We normalize centroids to ensure stable initial logits
        centroids_normalized = torch.nn.functional.normalize(centroids, p=2, dim=-1)
        
        router_weight = self.moe_layer.router.gate_linear.weight.data
        router_bias = self.moe_layer.router.gate_linear.bias.data
        
        # Assign centroids to the first min(P, E) experts
        limit = min(num_phases, num_experts)
        router_weight[:limit] = centroids_normalized[:limit]
        
        # Zero the bias to let dot product dominate initially
        router_bias.zero_()
        
        logger.info(f"Initialized router weights with {limit} phase centroids.")

        # 3. Initialize Experts with ActionHead weights
        # This provides the "warm start" for the experts, so they don't destroy
        # the performance achieved in Stage 1. The copy is exact and strict
        # (architecturally compatible ExpertMLP required) plus a small
        # symmetry-breaking jitter so the router can learn to differentiate
        # experts from the action loss.
        warm_start_experts_from_action_head(self.moe_layer.experts, self.action_head)
        
        # Stage 1 heads are not part of the Stage 2 computation graph; exclude
        # them from the optimizer so trainable-parameter counts and optimizer
        # state reflect the Stage 2 architecture.
        for param in self.action_head.parameters():
            param.requires_grad = False
        for param in self.phase_head.parameters():
            param.requires_grad = False
            
        logger.info("Initialized all experts with Stage 1 ActionHead weights.")
        
        # Automatically transition to Stage 2
        self.stage = 2
        logger.info("MoE Bootstrapping complete. Ready for Stage 2.")
