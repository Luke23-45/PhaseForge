from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.clustering import (
    compute_hierarchical_phase_prototypes,
    compute_phase_head_router_weights,
    spherical_kmeans,
)
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
             Forward pass uses encoder -> action_head + phase_head.
    Stage 2: Bootstrapped MoE specialization.
             Forward pass uses encoder -> moe_layer.

    The transition between stages is mediated by the `bootstrap_moe()` method,
    which transfers privileged regime structure (or generic clustering / phase head
    weights) into the MoE router and warm-starts or resets experts according to
    configuration.

    Args:
        encoder: The StateEncoder instance.
        action_head: The ActionHead used in Stage 1.
        phase_head: The PhaseClassificationHead used in Stage 1.
        router: The TopKRouter for Stage 2.
        expert: A single ExpertMLP template to be cloned for Stage 2.
        router_init: Optional configuration dictionary for router initialization.
            Keys: 'type' ('centroid' | 'spherical_centroid' | 'spherical_kmeans' |
                  'kmeans' | 'phase_head' | 'random'), 'seed' (int).
        expert_init: Optional configuration dictionary for expert initialization.
            Keys: 'type' ('warmstart' | 'random'), 'jitter_std' (float).
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        phase_head: PhaseClassificationHead,
        router: TopKRouter,
        expert: ExpertMLP,
        router_init: dict[str, Any] | None = None,
        expert_init: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder

        # Stage 1 components
        self.action_head = action_head
        self.phase_head = phase_head

        # Stage 2 components
        self.moe_layer = MoELayer(router=router, experts=expert)

        # Initialization strategy configurations
        self.router_init_cfg = (
            dict(router_init) if router_init is not None else {"type": "centroid"}
        )
        self.expert_init_cfg = (
            dict(expert_init)
            if expert_init is not None
            else {"type": "warmstart", "jitter_std": 0.02}
        )

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

    def unfreeze_encoder(self) -> None:
        """Unfreeze the encoder for fine-tuning in Stage 2 (PhaseForge-FT)."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen = False
        logger.info("Encoder weights unfrozen for Stage 2 fine-tuning.")

    def train(self, mode: bool = True) -> PhaseBootstrappedMoE:
        """Override so a frozen encoder stays deterministic during Stage 2.

        The training loop calls ``model.train()`` every epoch, which would
        otherwise re-enable the encoder's dropout when frozen.
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
    def bootstrap_moe(
        self,
        dataloader: DataLoader,
        device: torch.device | str = "cuda",
        router_init: dict[str, Any] | None = None,
        expert_init: dict[str, Any] | None = None,
    ) -> None:
        """Bootstrapping the MoE from Stage 1 knowledge.

        Supports the full factorial and unconfounded prototype generation:
        - Router initialization:
            * 'centroid' / 'phase_centroid': unconfounded hierarchical phase prototypes.
            * 'spherical_centroid': phase centroids with spherical pre-normalization.
            * 'spherical_kmeans': unsupervised spherical K-means on all training latents.
            * 'kmeans': Euclidean K-means on all training latents.
            * 'phase_head': normalized rows of linear phase head classifier.
            * 'random': keeps random router initialization.
        - Expert initialization:
            * 'warmstart': ActionHead weights copied with configurable jitter_std.
            * 'random': independent Kaiming draws.

        Args:
            dataloader: Training dataloader to compute centroids/clusters over.
            device: Compute device.
            router_init: Optional override for router initialization config.
            expert_init: Optional override for expert initialization config.
        """
        r_cfg = router_init or self.router_init_cfg
        e_cfg = expert_init or self.expert_init_cfg

        r_type = str(r_cfg.get("type", "centroid")).lower()
        e_type = str(e_cfg.get("type", "warmstart")).lower()
        jitter_std = float(e_cfg.get("jitter_std", 0.02))
        cluster_seed = int(r_cfg.get("seed", 42))

        logger.info(
            f"Starting MoE bootstrapping process (router_init='{r_type}', "
            f"expert_init='{e_type}', jitter_std={jitter_std})..."
        )
        self.to(device)
        self.eval()

        num_phases = self.phase_head.num_phases
        num_experts = self.moe_layer.router.num_experts
        latent_dim = self.encoder.latent_dim
        non_blocking = torch.device(device).type == "cuda"

        # 1. Collect training latents and phase labels if needed for data-driven router inits
        needs_latents = r_type in (
            "centroid", "phase_centroid", "spherical_centroid", "spherical_kmeans", "kmeans"
        )
        latents_list: list[Tensor] = []
        phases_list: list[Tensor] = []

        if needs_latents:
            logger.info("Computing latent representations across training dataloader...")
            for batch in dataloader:
                state = batch["state"].to(device, non_blocking=non_blocking)
                phase = batch["phase"].to(device, non_blocking=non_blocking)

                if state.ndim == 3:
                    state = state.view(-1, state.size(-1))
                    phase = phase.view(-1)

                latent = self.encoder(state)
                latents_list.append(latent)
                phases_list.append(phase)

            if not latents_list:
                raise ValueError("Bootstrap dataloader is empty. Cannot compute prototypes.")

            all_latents = torch.cat(latents_list, dim=0)
            all_phases = torch.cat(phases_list, dim=0)
            logger.info(
                f"Collected {all_latents.size(0)} training latent vectors (dim={latent_dim})."
            )

        # 2. Router Initialization
        if r_type in ("centroid", "phase_centroid"):
            prototypes = compute_hierarchical_phase_prototypes(
                all_latents,
                all_phases,
                num_phases,
                num_experts,
                seed=cluster_seed,
                spherical=False,
            )
            self.moe_layer.router.gate_linear.weight.data.copy_(prototypes)
            self.moe_layer.router.gate_linear.bias.data.zero_()
            logger.info(
                f"Initialized router weights with {num_experts} hierarchical phase prototypes."
            )

        elif r_type == "spherical_centroid":
            prototypes = compute_hierarchical_phase_prototypes(
                all_latents,
                all_phases,
                num_phases,
                num_experts,
                seed=cluster_seed,
                spherical=True,
            )
            self.moe_layer.router.gate_linear.weight.data.copy_(prototypes)
            self.moe_layer.router.gate_linear.bias.data.zero_()
            logger.info(
                f"Initialized router weights with {num_experts} spherical phase prototypes."
            )

        elif r_type == "spherical_kmeans":
            centroids, _ = spherical_kmeans(all_latents, k=num_experts, seed=cluster_seed)
            self.moe_layer.router.gate_linear.weight.data.copy_(centroids)
            self.moe_layer.router.gate_linear.bias.data.zero_()
            logger.info(
                f"Initialized router weights with {num_experts} Spherical K-Means centroids."
            )

        elif r_type == "kmeans":
            from sklearn.cluster import KMeans

            latents_np = all_latents.cpu().numpy()
            km = KMeans(
                n_clusters=num_experts, random_state=cluster_seed, n_init=10
            ).fit(latents_np)
            centroids = torch.from_numpy(km.cluster_centers_).to(
                device=device, dtype=all_latents.dtype
            )
            centroids_normalized = F.normalize(centroids, p=2, dim=-1)
            self.moe_layer.router.gate_linear.weight.data.copy_(centroids_normalized)
            self.moe_layer.router.gate_linear.bias.data.zero_()
            logger.info(
                f"Initialized router weights with {num_experts} Euclidean KMeans centroids."
            )

        elif r_type == "phase_head":
            phase_weight = self.phase_head.head.weight.data
            weights = compute_phase_head_router_weights(phase_weight, num_experts)
            self.moe_layer.router.gate_linear.weight.data.copy_(weights)
            self.moe_layer.router.gate_linear.bias.data.zero_()
            logger.info(
                "Initialized router weights with normalized phase-head directions "
                f"({num_experts} rows)."
            )

        elif r_type == "random":
            logger.info("Router initialization: keeping random router weights (standard init).")
        else:
            raise ValueError(
                f"Unknown router_init type '{r_type}'. Supported: "
                "centroid, spherical_centroid, spherical_kmeans, kmeans, phase_head, random."
            )

        # 3. Expert Initialization
        if e_type == "warmstart":
            warm_start_experts_from_action_head(
                self.moe_layer.experts, self.action_head, jitter_std=jitter_std
            )
            logger.info(
                f"Warm-started all {num_experts} experts from ActionHead (jitter_std={jitter_std})."
            )
        elif e_type == "random":
            for expert in self.moe_layer.experts:
                expert.reset_parameters()
            logger.info(
                f"Reset all {num_experts} experts to independent random draws "
                "(scratch distribution)."
            )
        else:
            raise ValueError(f"Unknown expert_init type '{e_type}'. Supported: warmstart, random.")

        # Exclude Stage 1 heads from the optimizer in Stage 2
        for param in self.action_head.parameters():
            param.requires_grad = False
        for param in self.phase_head.parameters():
            param.requires_grad = False

        # Transition to Stage 2
        self.stage = 2
        logger.info("MoE Bootstrapping complete. Ready for Stage 2.")

