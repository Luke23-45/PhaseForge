"""Warm-Start MoE baseline."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from phaseforge.models.base import BaseManipulationModel, ModelOutput
from phaseforge.models.components.action_head import ActionHead
from phaseforge.models.components.encoder import StateEncoder
from phaseforge.models.components.expert import (
    ExpertMLP,
    hash_dropped_indices,
    partial_reinit_experts_from_action_head,
    warm_start_experts_from_action_head,
)
from phaseforge.models.components.moe_layer import MoELayer
from phaseforge.models.components.router import TopKRouter

logger = logging.getLogger(__name__)


class WarmStartMoEModel(BaseManipulationModel):
    """MoE trained with a Warm-Start approach.

    Stage 1: Pretrain encoder + action_head (λ_phase = 0).
    Stage 2: Bootstrap MoE, but with random router init (no phase centroids).

    Expert initialization is config-driven via ``expert_init`` (same schema
    subset as :class:`PhaseBootstrappedMoE`: ``warmstart`` with
    ``jitter_std`` or ``partial_warm`` with ``drop_rate``/``seed``). The
    default — and the ``warmstart_moe`` cell's registered configuration — is
    the standard full warm start; the R50-matched control
    (:class:`PhasePretrainRandomRouterModel`) overrides it to a 50% partial
    warm start through its own model config.
    """

    def __init__(
        self,
        encoder: StateEncoder,
        action_head: ActionHead,
        router: TopKRouter,
        expert: ExpertMLP,
        expert_init: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.action_head = action_head
        self.moe_layer = MoELayer(router=router, experts=expert)
        self.expert_init_cfg: dict[str, Any] = (
            dict(expert_init) if expert_init is not None else {"type": "warmstart", "jitter_std": 0.02}
        )
        self._expert_init_info: dict[str, Any] | None = None
        self._stage = 1
        self._encoder_frozen = False
        self._last_gate_logits: Tensor | None = None

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        self._stage = value

    def freeze_encoder(self) -> None:
        """Freeze the encoder for Stage 2 (weights + eval mode, no dropout)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()

    def train(self, mode: bool = True) -> WarmStartMoEModel:
        """Override so a frozen encoder stays deterministic during Stage 2."""
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
        return self

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
    def bootstrap_moe(
        self,
        dataloader: DataLoader,
        device: torch.device | str = "cuda",
        training_seed: int | None = None,
    ) -> None:
        """Initialize experts from the ActionHead; the router stays random.

        The CLI passes ``training_seed`` (the run's ``project.seed``) to every
        bootstrap call; it is recorded in the audit metadata and is the
        fallback for the partial-warm init seed when the config omits one.

        Args:
            dataloader: Training dataloader (signature parity with
                :class:`PhaseBootstrappedMoE`; not iterated — no data-driven
                initialization happens with a random router).
            device: Compute device.
            training_seed: The run's training seed (audit + partial-warm
                fallback).
        """
        self.to(device)

        e_cfg = self.expert_init_cfg
        e_type = str(e_cfg.get("type", "warmstart")).lower()
        jitter_std = float(e_cfg.get("jitter_std", 0.02))

        # 1. Router remains randomly initialized (standard MoE initialization)
        logger.info("WarmStartMoE: Leaving router randomly initialized.")

        # 2. Initialize Experts per the config-driven init type.
        num_experts = len(self.moe_layer.experts)
        expert_init_info: dict[str, Any] = {"type": e_type, "jitter_std": jitter_std}

        if e_type == "warmstart":
            # Exact, strict copy + small symmetry-breaking jitter so the
            # action loss can shape routing.
            warm_start_experts_from_action_head(
                self.moe_layer.experts, self.action_head, jitter_std=jitter_std
            )
            logger.info(
                f"WarmStartMoE: warm-started all {num_experts} experts from "
                f"ActionHead (jitter_std={jitter_std})."
            )
        elif e_type == "partial_warm":
            init_seed = int(e_cfg.get("seed", training_seed if training_seed is not None else 42))
            drop_rate = float(e_cfg.get("drop_rate", 0.5))
            first_typed = cast(ExpertMLP, self.moe_layer.experts[0]) if num_experts > 0 else None
            hidden_dim = (
                int(first_typed.hidden[0].weight.size(0))  # type: ignore[union-attr,operator]
                if first_typed is not None
                else 0
            )
            dropped_indices = partial_reinit_experts_from_action_head(
                self.moe_layer.experts,
                self.action_head,
                drop_rate=drop_rate,
                seed=init_seed,
            )
            expert_init_info.update(
                {
                    "drop_rate": drop_rate,
                    "init_seed": init_seed,
                    "hidden_dim": hidden_dim,
                    "num_dropped_neurons": len(dropped_indices),
                    "dropped_neuron_indices": dropped_indices,
                    "dropped_indices_sha256": hash_dropped_indices(dropped_indices),
                }
            )
            logger.info(
                f"WarmStartMoE: partial-warm-started all {num_experts} experts "
                f"from ActionHead (drop_rate={drop_rate}, seed={init_seed}, "
                f"dropped={len(dropped_indices)}/{hidden_dim}, jitter_std=0.0)."
            )
        else:
            raise ValueError(
                f"Unknown expert_init type '{e_type}'. Supported: "
                "warmstart, partial_warm."
            )

        self._expert_init_info = {
            "expert_init": expert_init_info,
            "router": {
                "num_experts": int(num_experts),
                "top_k": int(self.moe_layer.router.top_k),
                "init_type": "random",
            },
            "training_seed": int(training_seed) if training_seed is not None else None,
        }

        # The Stage 1 action head is unused in Stage 2; exclude it from the
        # optimizer so trainable-parameter counts are correct.
        for param in self.action_head.parameters():
            param.requires_grad = False

        self.stage = 2
