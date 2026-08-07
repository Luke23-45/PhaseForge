"""Stage 2 Trainer: Bootstrapped MoE specialization."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from phaseforge.evaluations.metrics import expert_utilization, phase_alignment, routing_stability
from phaseforge.trains.loops.base import BaseTrainer

logger = logging.getLogger(__name__)


class Stage2Trainer(BaseTrainer):
    """Trainer for Stage 2.

    Computes: L_total = L_action + β_balance * L_balance
    Action loss is MSE. Balance loss is auxiliary.
    Optionally freezes the encoder.

    Every epoch also logs the routing diagnostics (C3: balance-vs-NMI
    trajectory) over the validation set — ``val/phase_expert_nmi``,
    ``val/balance_score``, ``val/collapse_rate``, ``val/routing_entropy`` —
    so pseudo-balancing (balance high while NMI collapses to 0) is visible
    at training time instead of only in offline evaluation.
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
        
        n_params = sum(p.numel() for p in active_params)
        logger.info(f"Stage 2 initialized. Trainable parameters: {n_params}")
        
        super().fit()

    def _compute_loss(
        self, batch: dict[str, torch.Tensor], out=None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Forward pass (reuse an existing output when provided, e.g. from the
        # validation loop, to avoid a double forward).
        if out is None:
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

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        """Validation plus per-epoch routing diagnostics (C3).

        Reuses the forward outputs for both the loss metrics and the routing
        metrics so validation stays a single forward pass per batch.
        """
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        agg_metrics: dict[str, float] = {}
        num_batches = 0

        expert_indices_all: list[torch.Tensor] = []
        phases_all: list[torch.Tensor] = []
        gate_logits_all: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            out = self.model(batch)
            _, metrics = self._compute_loss(batch, out=out)

            for k, v in metrics.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + v
            num_batches += 1

            if out.expert_indices is not None:
                expert_indices_all.append(out.expert_indices.detach().cpu())
                phases_all.append(batch["phase"].cpu())
            if out.gate_logits is not None:
                gate_logits_all.append(out.gate_logits.detach().cpu())

        if num_batches == 0:
            return {}
        agg_metrics = {k: v / num_batches for k, v in agg_metrics.items()}

        # Routing diagnostics: balance-vs-specialization trajectory (C3).
        if expert_indices_all:
            expert_indices = torch.cat(expert_indices_all, dim=0)
            phases = torch.cat(phases_all, dim=0)

            agg_metrics["val/phase_expert_nmi"] = phase_alignment.phase_expert_nmi(
                phases, expert_indices
            )

            num_experts = int(expert_indices.max().item()) + 1
            fractions = expert_utilization.expert_utilization(expert_indices, num_experts)
            agg_metrics["val/balance_score"] = (
                expert_utilization.expert_utilization_balance(fractions)
            )
            agg_metrics["val/collapse_rate"] = expert_utilization.collapse_rate(fractions)

        if gate_logits_all:
            gate_logits = torch.cat(gate_logits_all, dim=0)
            agg_metrics["val/routing_entropy"] = (
                routing_stability.routing_entropy(gate_logits, normalize=True).item()
            )

        logger.info(
            "Epoch %d routing diagnostics: NMI=%.4f balance=%.4f collapse=%.4f entropy=%.4f",
            self.current_epoch,
            agg_metrics.get("val/phase_expert_nmi", float("nan")),
            agg_metrics.get("val/balance_score", float("nan")),
            agg_metrics.get("val/collapse_rate", float("nan")),
            agg_metrics.get("val/routing_entropy", float("nan")),
        )

        return agg_metrics
