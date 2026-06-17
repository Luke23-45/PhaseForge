"""Stage 1 Trainer: Phase-supervised generalist pretraining."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phaseforge.trains.loops.base import BaseTrainer


class Stage1Trainer(BaseTrainer):
    """Trainer for Stage 1.

    Computes: L_total = L_action + λ_phase * L_phase
    Action loss is MSE. Phase loss is CrossEntropy.
    """

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        # Forward pass
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
            "loss_total": total_loss.item(),
            "loss_action": action_loss.item(),
            "loss_phase": phase_loss.item() if lambda_phase > 0.0 else 0.0,
        }
        
        return total_loss, metrics
