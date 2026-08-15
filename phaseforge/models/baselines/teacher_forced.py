"""Teacher-forced MoE: GT-partitioned experts, predicted-phase routing (E8).

The decomposable-oracle cell (issues register C7 / novelty claim E8):

    Training:  experts are partitioned by the GROUND-TRUTH phase label
               (top-1 hard partition, exclusive per phase) — the same
               privileged signal the oracle uses.
    Inference: routing uses a LEARNED phase predictor — ``argmax`` of the
               Stage 1 phase head — so the policy is deployable without
               labels.

Locked implementation decisions (2026-08-07):
    (i)  The phase predictor is the Stage 1 phase head of the SAME
         phase-supervised checkpoint that ``phaseforge`` uses
         (``resolve_checkpoint_source`` maps it there). Shared pretraining,
         so only the Stage 2 supervision regime differs. The phase head is
         part of the frozen Stage 1 bundle: only the experts train in
         Stage 2.
    (ii) Top-k asymmetry is footnoted, not hidden: this cell routes top-1
         (exclusive GT phase partition) vs ``phaseforge``'s top-2 (method
         hyperparameter).
    (iii) Natural sampling for parity; starvation of a phase is a reported
         diagnostic, not a silent failure.

This turns the oracle into a decomposable instrument:
    oracle (GT routing) - teacher_forced (predicted routing)
        = phase-predictability loss (Gap 1)
    teacher_forced (predicted) - phaseforge
        = strategy loss (Gap 2)
"""

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


class TeacherForcedMoEModel(BaseManipulationModel):
    """MoE with ground-truth-partitioned experts and predicted-phase routing.

    Structurally mirrors :class:`PhaseBootstrappedMoE` (encoder + action_head
    + phase_head + MoE layer) so the SAME phaseforge Stage 1 checkpoint
    loads cleanly. The router is kept for structural parity but is never
    consulted: routing is by GT phase in training and by the (frozen) phase
    head at inference.

    Args:
        encoder: The StateEncoder instance.
        action_head: The ActionHead used in Stage 1.
        phase_head: The PhaseClassificationHead — the learned phase
            predictor used for label-free routing at inference.
        router: The TopKRouter for Stage 2 (structural parity only).
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

        self._stage = 1
        self._encoder_frozen = False
        self._last_gate_logits: Tensor | None = None

    @property
    def stage(self) -> int:
        return self._stage

    @stage.setter
    def stage(self, value: int) -> None:
        if value not in (1, 2):
            raise ValueError(f"Stage must be 1 or 2, got {value}")
        self._stage = value
        logger.info(f"TeacherForcedMoEModel transitioned to Stage {value}.")

    def freeze_encoder(self) -> None:
        """Freeze the Stage 1 bundle (encoder + phase predictor) for Stage 2.

        Only the experts train in Stage 2 — the supervision regime is the
        ONLY thing that differs from ``phaseforge``. Frozen modules are also
        kept in eval mode so dropout stays off during training.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.phase_head.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        self.encoder.eval()
        self.phase_head.eval()
        logger.info("Encoder and phase predictor frozen (shared Stage 1 bundle).")

    def train(self, mode: bool = True) -> TeacherForcedMoEModel:
        """Override so the frozen Stage 1 bundle stays deterministic."""
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
            self.phase_head.eval()
        return self

    def forward(self, batch: dict[str, Tensor]) -> ModelOutput:
        """Forward pass depends on the active stage and the mode.

        Stage 1: encoder -> action_head + phase_head (checkpoint-compatible).
        Stage 2 training: experts partitioned by the GT phase label.
        Stage 2 eval: experts selected by ``argmax`` of the phase head.
        """
        state = batch["state"]
        latent = self.encoder(state)

        if self._stage == 1:
            action_pred = self.action_head(latent)
            phase_logits = self.phase_head(latent)
            return ModelOutput(
                action_pred=action_pred,
                phase_logits=phase_logits,
                routing_weights=None,
                expert_indices=None,
                gate_logits=None,
            )
        elif self._stage == 2:
            if self.training:
                phase = batch.get("phase")
                if phase is None:
                    raise RuntimeError(
                        "TeacherForcedMoEModel requires ground-truth 'phase' "
                        "labels during Stage 2 training."
                    )
                if latent.ndim == 3:
                    latent = latent.view(-1, latent.size(-1))
                    phase = phase.view(-1)
                return self._dispatch(latent, phase)
            else:
                if latent.ndim == 3:
                    latent = latent.view(-1, latent.size(-1))
                phase_logits = self.phase_head(latent)
                return self._dispatch(
                    latent, phase_logits.argmax(dim=-1), phase_logits=phase_logits
                )
        else:
            raise RuntimeError(f"Invalid stage {self._stage}")

    def _dispatch(
        self,
        latent: Tensor,
        expert_indices: Tensor,
        phase_logits: Tensor | None = None,
    ) -> ModelOutput:
        """Top-1 hard expert dispatch with certainty-one routing weights.

        Mirrors the oracle's dispatch (weights 1.0, one-hot gate logits) so
        the two cells differ ONLY in the routing signal (GT vs predicted).
        ``phase_logits`` is surfaced from the label-free eval path so the
        trainer can compute the routing accuracy diagnostic (spec §4.4); the
        GT-routed training path leaves it ``None``.
        """
        B = latent.size(0)
        E = self.moe_layer.router.num_experts

        if expert_indices.numel() and (expert_indices.min() < 0 or expert_indices.max() >= E):
            raise ValueError(
                f"Teacher-forced phase labels must map to experts in [0, {E - 1}], "
                f"got range [{int(expert_indices.min())}, {int(expert_indices.max())}]."
            )
        expert_indices = expert_indices.long()
        if expert_indices.ndim == 1:
            expert_indices = expert_indices.unsqueeze(-1)  # (B, 1)

        routing_weights = torch.ones((B, 1), dtype=latent.dtype, device=latent.device)

        # Dummy logits for metric compatibility (highly peaked one-hot)
        gate_logits = torch.zeros((B, E), dtype=latent.dtype, device=latent.device)
        gate_logits.scatter_(1, expert_indices, 100.0)
        self._last_gate_logits = gate_logits.detach()

        out_dim = self.moe_layer.experts[0].output_dim
        combined_output = torch.zeros((B, out_dim), dtype=latent.dtype, device=latent.device)

        for expert_idx, expert_net in enumerate(self.moe_layer.experts):
            match_mask = (expert_indices == expert_idx).squeeze(-1)
            if not match_mask.any():
                continue
            batch_idx = torch.where(match_mask)[0]
            combined_output.index_copy_(0, batch_idx, expert_net(latent[batch_idx]))

        return ModelOutput(
            action_pred=combined_output,
            phase_logits=phase_logits,
            routing_weights=routing_weights,
            expert_indices=expert_indices,
            gate_logits=gate_logits,
            # No balance loss: routing is deterministic by construction
            aux_losses={"balance": torch.tensor(0.0, device=latent.device)},
        )

    def get_action(self, state: Tensor) -> Tensor:
        """Label-free inference path: route by the predicted phase."""
        latent = self.encoder(state)
        if self._stage == 1:
            return self.action_head(latent)
        if latent.ndim == 3:
            latent = latent.view(-1, latent.size(-1))
        phase_pred = self.phase_head(latent).argmax(dim=-1)
        return self._dispatch(latent, phase_pred).action_pred

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
        """Transition to Stage 2: warm-start experts, keep the frozen phase head.

        No centroid computation is needed — routing is by the phase head, not
        the router's gate. The router stays at its random init for structural
        parity with the other cells; it is never consulted.

        Args:
            dataloader: Training dataloader (kept for CLI signature parity;
                not iterated — no centroids are computed).
            device: Compute device.
        """
        self.to(device)

        # 1. Router remains randomly initialized (never consulted — structural
        #    parity with phaseforge / oracle cells).
        logger.info("TeacherForcedMoE: router kept at init (routing by phase head).")

        # 2. Initialize Experts with ActionHead weights (identical warm start
        #    to every other cell: exact strict copy + symmetry-breaking jitter).
        warm_start_experts_from_action_head(self.moe_layer.experts, self.action_head)

        # The Stage 1 action head is unused in Stage 2 (and the phase head is
        # frozen with the encoder bundle); exclude both from the optimizer.
        for param in self.action_head.parameters():
            param.requires_grad = False
        for param in self.phase_head.parameters():
            param.requires_grad = False

        logger.info("TeacherForcedMoE: initialized all experts with Stage 1 ActionHead weights.")

        # Automatically transition to Stage 2
        self.stage = 2
        logger.info("TeacherForcedMoE bootstrap complete. Ready for Stage 2.")
