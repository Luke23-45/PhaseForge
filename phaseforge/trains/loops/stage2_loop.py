"""Stage 2 Trainer: Bootstrapped MoE specialization."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2.

    Computes: L_total = L_action + β_balance * L_balance
    Action loss is MSE. Balance loss is auxiliary.
    Optionally freezes the encoder.
    """

    def fit(self) -> None:
        """Override fit to handle encoder freezing before the loop starts."""
        if self.train_cfg.freeze_encoder:
            self.model.freeze_encoder()
            
        # Re-initialize optimizer because freezing might have changed requires_grad
        from hydra.utils import instantiate
        
        active_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = instantiate(self.train_cfg.optimizer, params=active_params)
        self.scheduler = instantiate(self.train_cfg.scheduler, optimizer=self.optimizer)
        
        logger.info(f"Stage 2 initialized. Trainable parameters: {sum(p.numel() for p in active_params)}")
        
        super().fit()

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        # Forward pass
        out = self.model(batch)
        
        # Ground truths
        target_action = batch["action"]  # (B, A) or (B, T, A)
        mask = batch.get("padding_mask") # (B, T) boolean or None
        
        # Action Loss (MSE)
        if mask is not None:
            action_loss = F.mse_loss(out.action_pred, target_action, reduction="none")
            action_loss = action_loss[mask].mean()
        else:
            action_loss = F.mse_loss(out.action_pred, target_action)
            
        # Balance Loss
        balance_loss = out.aux_losses.get("balance", torch.tensor(0.0, device=self.device))
        
        # Total Loss
        total_loss = action_loss + balance_loss
        
        metrics = {
            "loss_total": total_loss.item(),
            "loss_action": action_loss.item(),
            "loss_balance": balance_loss.item(),
        }
        
        return total_loss, metrics
