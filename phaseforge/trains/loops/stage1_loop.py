"""Stage 1 Trainer: Phase-supervised generalist pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phaseforge.models.base import ModelOutput
from phaseforge.trains.loops.base import (
    BaseTrainer,
    MetricValue,
    _PhaseAccumulator,
)


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage 1.

    Computes: L_total = L_action + λ_phase * L_phase
    Action loss is MSE. Phase loss is CrossEntropy.

    Also persists the phase-classification accuracy the protocol requires
    (final specification §4.2): micro ``phase_acc`` and macro/balanced
    ``phase_balanced_acc`` (mean per-class recall), computed in the
    training loop and in validation. ``bc`` has no phase head, so its
    ``phase_logits`` are ``None`` and no phase fields are emitted — absence
    is honest, never a fabricated zero.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._train_phase_acc = _PhaseAccumulator(device=self.device)
        self._val_phase_acc = _PhaseAccumulator(device=self.device)

    def _compute_loss(
        self, batch: dict[str, torch.Tensor], out: ModelOutput | None = None
    ) -> tuple[torch.Tensor, dict[str, MetricValue]]:
        # Forward pass
        if out is None:
            out = self.model(batch)
        
        # Ground truths
        target_action = batch["action"]  # (B, A) or (B, T, A)
        target_phase = batch["phase"]    # (B,) or (B, T)
        mask = batch.get("padding_mask") # (B, T) boolean or None
        
        lambda_phase = self.train_cfg.lambda_phase
        
        # Action Loss (MSE)
        if mask is not None:
            # Masked MSE for variable length
            action_loss = F.mse_loss(out.action_pred, target_action, reduction="none")
            # Average only over valid steps
            action_loss = action_loss[mask].mean()
        else:
            action_loss = F.mse_loss(out.action_pred, target_action)
            
        # Phase Loss (Cross Entropy)
        phase_loss = torch.tensor(0.0, device=self.device)
        if out.phase_logits is not None and lambda_phase > 0.0:
            logits = out.phase_logits
            if mask is not None:
                # Reshape for CE: (B*T, num_classes) and (B*T,)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = target_phase.view(-1)
                mask_flat = mask.view(-1)
                
                # Filter by mask
                logits_valid = logits_flat[mask_flat]
                targets_valid = targets_flat[mask_flat]
                
                if len(targets_valid) > 0:
                    phase_loss = F.cross_entropy(logits_valid, targets_valid)
            else:
                phase_loss = F.cross_entropy(logits, target_phase)
                
        # Total Loss
        total_loss = action_loss + lambda_phase * phase_loss
        
        metrics = {
            # Keep scalar metrics on-device until the trainer actually logs or
            # aggregates them. Calling .item() for every metric on every CUDA
            # batch would force a host/device synchronization each step.
            "loss_total": total_loss.detach(),
            "loss_action": action_loss.detach(),
            "loss_phase": phase_loss.detach() if lambda_phase > 0.0 else 0.0,
        }
        
        return total_loss, metrics

    def _on_train_batch_processed(
        self,
        batch: dict[str, torch.Tensor],
        out: ModelOutput,
        metrics: dict[str, MetricValue],
        n: int,
    ) -> None:
        if out.phase_logits is not None:
            self._train_phase_acc.update(
                out.phase_logits, batch["phase"], mask=batch.get("padding_mask")
            )

    def epoch_train_metrics(self) -> dict[str, float]:
        if not self._train_phase_acc.has_data:
            return {}
        acc, balanced = self._train_phase_acc.compute()
        return {
            "train/phase_acc": acc,
            "train/phase_balanced_acc": balanced,
        }

    def _reset_train_pool(self) -> None:
        self._train_phase_acc.reset()

    def _reset_validation_pool(self) -> None:
        self._val_phase_acc.reset()

    def _collect_validation_pool(
        self,
        batch: dict[str, torch.Tensor],
        out: ModelOutput,
        metrics: dict[str, MetricValue],
    ) -> None:
        if out.phase_logits is not None:
            self._val_phase_acc.update(
                out.phase_logits, batch["phase"], mask=batch.get("padding_mask")
            )

    def _finalize_validation_pool(self, agg_metrics: dict[str, float]) -> None:
        if self._val_phase_acc.has_data:
            acc, balanced = self._val_phase_acc.compute()
            agg_metrics["val/phase_acc"] = acc
            agg_metrics["val/phase_balanced_acc"] = balanced
